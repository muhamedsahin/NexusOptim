/**
 * @file hybrid_update.cpp
 * @brief CPU SIMD ve CUDA parametre guncellemesini ust uste kosar.
 *
 * Cihaz tamponlari adimlar arasinda tutulur. Ayni host isaretcileriyle gelen
 * sonraki adim parametre ve momentleri yeniden yukleme: onlar bir onceki
 * adimin cihaz kopyasidir. Her adimda gradyan gider, guncellenmis parametre
 * ve momentler hosta doner. CUDA yoksa tum vektor CPU'da kalir.
 */

#include "nexus_optim/device/hybrid_update.hpp"

#include "nexus_optim/simd/simd_dispatch.hpp"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#if defined(NEXUS_OPTIM_WITH_CUDA)
#include "nexus_optim/cuda/cuda_launch.hpp"

#include <cuda_runtime.h>
#endif

namespace nexus_optim {
namespace {

class PersistentCpu {
 public:
    PersistentCpu() : thread_([this] { loop(); }) {}
    ~PersistentCpu() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_one();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void start(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = std::move(job);
            ready_ = true;
        }
        wake_.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock, [&] { return !ready_; });
    }

 private:
    void loop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return ready_ || stop_; });
                if (stop_) {
                    return;
                }
                job = std::move(job_);
            }
            job();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ready_ = false;
            }
            finished_.notify_one();
        }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable finished_;
    std::function<void()> job_;
    bool ready_ = false;
    bool stop_ = false;
    std::thread thread_;
};

PersistentCpu& cpu_worker() {
    static PersistentCpu worker;
    return worker;
}



std::size_t align_down(std::size_t value) { return value & ~std::size_t{63}; }

HybridCounts plan(std::size_t n) {
  HybridCounts counts;
#if defined(NEXUS_OPTIM_WITH_CUDA)
  int devices = 0;
  if (n >= 4096 && cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0) {
    counts.gpu = align_down(static_cast<std::size_t>(static_cast<double>(n) * 0.75));
    if (counts.gpu < 64) {
      counts.gpu = 0;
    }
    if (counts.gpu > n) {
      counts.gpu = align_down(n);
    }
  }
#else
  (void)n;
#endif
  counts.cpu = n - counts.gpu;
  return counts;
}

#if defined(NEXUS_OPTIM_WITH_CUDA)

struct AdamDevice {
  float* params = nullptr;
  float* grads = nullptr;
  float* moment = nullptr;
  float* second = nullptr;
  float* maximum = nullptr;
  std::size_t capacity = 0;
  float* host_params = nullptr;
  float* host_moment = nullptr;
  float* host_second = nullptr;
  float* host_maximum = nullptr;
  std::size_t bound = 0;
  bool resident = false;

  void release() noexcept {
    cuda::device_free(params);
    cuda::device_free(grads);
    cuda::device_free(moment);
    cuda::device_free(second);
    cuda::device_free(maximum);
    params = grads = moment = second = maximum = nullptr;
    capacity = 0;
    resident = false;
  }

  void ensure(std::size_t count, bool want_max) {
    if (capacity >= count && (maximum != nullptr) == want_max) {
      return;
    }
    release();
    params = static_cast<float*>(cuda::device_alloc_bytes(count * sizeof(float)));
    grads = static_cast<float*>(cuda::device_alloc_bytes(count * sizeof(float)));
    moment = static_cast<float*>(cuda::device_alloc_bytes(count * sizeof(float)));
    second = static_cast<float*>(cuda::device_alloc_bytes(count * sizeof(float)));
    if (want_max) {
      maximum = static_cast<float*>(cuda::device_alloc_bytes(count * sizeof(float)));
    }
    capacity = count;
  }

  ~AdamDevice() { release(); }
};

struct SgdDevice {
  float* params = nullptr;
  float* grads = nullptr;
  float* velocity = nullptr;
  std::size_t capacity = 0;
  float* host_params = nullptr;
  float* host_velocity = nullptr;
  std::size_t bound = 0;
  bool resident = false;

  void release() noexcept {
    cuda::device_free(params);
    cuda::device_free(grads);
    cuda::device_free(velocity);
    params = grads = velocity = nullptr;
    capacity = 0;
    resident = false;
  }

  void ensure(std::size_t count, bool want_velocity) {
    if (capacity >= count && (velocity != nullptr) == want_velocity) {
      return;
    }
    release();
    params = static_cast<float*>(cuda::device_alloc_bytes(count * sizeof(float)));
    grads = static_cast<float*>(cuda::device_alloc_bytes(count * sizeof(float)));
    if (want_velocity) {
      velocity = static_cast<float*>(cuda::device_alloc_bytes(count * sizeof(float)));
    }
    capacity = count;
  }

  ~SgdDevice() { release(); }
};

AdamDevice& adam_device() {
  static AdamDevice slot;
  return slot;
}

SgdDevice& sgd_device() {
  static SgdDevice slot;
  return slot;
}

#endif

}  // namespace

HybridCounts adam_update_hybrid(float* params, const float* grads, float* exp_avg, float* exp_avg_sq,
                                float* max_exp_avg_sq, std::size_t n, AdamCoeffs coeffs) {
  const HybridCounts counts = plan(n);
  if (counts.gpu == 0) {
    adam_update(params, grads, exp_avg, exp_avg_sq, max_exp_avg_sq, n, coeffs);
    return counts;
  }
#if defined(NEXUS_OPTIM_WITH_CUDA)
  const std::size_t cpu_n = counts.cpu;
  const std::size_t gpu_n = counts.gpu;
  const bool want_max = max_exp_avg_sq != nullptr;
  AdamDevice& slot = adam_device();
  slot.ensure(gpu_n, want_max);
  float* host_p = params + cpu_n;
  float* host_m = exp_avg + cpu_n;
  float* host_v = exp_avg_sq + cpu_n;
  float* host_x = want_max ? max_exp_avg_sq + cpu_n : nullptr;
  const bool reuse = slot.resident && slot.bound == gpu_n && slot.host_params == host_p && slot.host_moment == host_m &&
                     slot.host_second == host_v && slot.host_maximum == host_x;
  if (!reuse) {
    cuda::copy_host_to_device(slot.params, host_p, gpu_n * sizeof(float));
    cuda::copy_host_to_device(slot.moment, host_m, gpu_n * sizeof(float));
    cuda::copy_host_to_device(slot.second, host_v, gpu_n * sizeof(float));
    if (want_max) {
      cuda::copy_host_to_device(slot.maximum, host_x, gpu_n * sizeof(float));
    }
    slot.host_params = host_p;
    slot.host_moment = host_m;
    slot.host_second = host_v;
    slot.host_maximum = host_x;
    slot.bound = gpu_n;
    slot.resident = true;
  }
  cpu_worker().start([&] { adam_update(params, grads, exp_avg, exp_avg_sq, max_exp_avg_sq, cpu_n, coeffs); });
  cuda::copy_host_to_device(slot.grads, grads + cpu_n, gpu_n * sizeof(float));
  cuda::adam_update(slot.params, slot.grads, slot.moment, slot.second, want_max ? slot.maximum : nullptr, gpu_n, coeffs);
  cpu_worker().wait();
  cuda::copy_device_to_host(host_p, slot.params, gpu_n * sizeof(float));
  cuda::copy_device_to_host(host_m, slot.moment, gpu_n * sizeof(float));
  cuda::copy_device_to_host(host_v, slot.second, gpu_n * sizeof(float));
  if (want_max) {
    cuda::copy_device_to_host(host_x, slot.maximum, gpu_n * sizeof(float));
  }
#else
  adam_update(params, grads, exp_avg, exp_avg_sq, max_exp_avg_sq, n, coeffs);
#endif
  return counts;
}

HybridCounts sgd_update_hybrid(float* params, const float* grads, float* velocity, std::size_t n, SgdCoeffs coeffs) {
  const HybridCounts counts = plan(n);
  if (counts.gpu == 0) {
    sgd_update(params, grads, velocity, n, coeffs);
    return counts;
  }
#if defined(NEXUS_OPTIM_WITH_CUDA)
  const std::size_t cpu_n = counts.cpu;
  const std::size_t gpu_n = counts.gpu;
  const bool momentum = velocity != nullptr && coeffs.momentum != 0.f;
  SgdDevice& slot = sgd_device();
  slot.ensure(gpu_n, momentum);
  float* host_p = params + cpu_n;
  float* host_v = momentum ? velocity + cpu_n : nullptr;
  const bool reuse = slot.resident && slot.bound == gpu_n && slot.host_params == host_p && slot.host_velocity == host_v;
  if (!reuse) {
    cuda::copy_host_to_device(slot.params, host_p, gpu_n * sizeof(float));
    if (momentum) {
      cuda::copy_host_to_device(slot.velocity, host_v, gpu_n * sizeof(float));
    }
    slot.host_params = host_p;
    slot.host_velocity = host_v;
    slot.bound = gpu_n;
    slot.resident = true;
  }
  cpu_worker().start([&] { sgd_update(params, grads, velocity, cpu_n, coeffs); });
  cuda::copy_host_to_device(slot.grads, grads + cpu_n, gpu_n * sizeof(float));
  cuda::sgd_update(slot.params, slot.grads, momentum ? slot.velocity : nullptr, gpu_n, coeffs);
  cpu_worker().wait();
  cuda::copy_device_to_host(host_p, slot.params, gpu_n * sizeof(float));
  if (momentum) {
    cuda::copy_device_to_host(host_v, slot.velocity, gpu_n * sizeof(float));
  }
#else
  sgd_update(params, grads, velocity, n, coeffs);
#endif
  return counts;
}

}  // namespace nexus_optim
