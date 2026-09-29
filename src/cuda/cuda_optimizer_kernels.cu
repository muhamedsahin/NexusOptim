#include "nexus_optim/cuda/cuda_launch.hpp"

#include "nexus_optim/cuda/cuda_optimizer_kernels.cuh"
#include "nexus_optim/cuda/cuda_stream_manager.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace nexus_optim {
namespace cuda {
namespace {

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(status));
  }
}

cudaStream_t work_stream() { return StreamManager::instance().stream_for(0); }

void finish(const char* what) {
  check(cudaGetLastError(), what);
  check(cudaStreamSynchronize(work_stream()), "synchronize");
}

bool aligned16(const void* ptr) {
  return ptr == nullptr || (reinterpret_cast<std::uintptr_t>(ptr) % 16u) == 0;
}

double* norm_scratch() {
  static double* ptr = nullptr;
  if (ptr == nullptr) {
    check(cudaMalloc(&ptr, sizeof(double) * 2), "norm scratch");
    check(cudaMemset(ptr, 0, sizeof(double) * 2), "norm scratch zero");
  }
  return ptr;
}

}  // namespace

bool enabled() noexcept { return true; }
int stream_count() noexcept { return 4; }

void* device_alloc_bytes(std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  void* ptr = nullptr;
  check(cudaMalloc(&ptr, bytes), "malloc");
  check(cudaMemset(ptr, 0, bytes), "calloc");
  return ptr;
}

void device_free(void* ptr) noexcept {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

void copy_device_to_host(void* host, const void* device, std::size_t bytes) {
  check(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost), "D2H");
}

void copy_host_to_device(void* device, const void* host, std::size_t bytes) {
  check(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice), "H2D");
}

void zero(void* ptr, std::size_t bytes) {
  if (ptr == nullptr || bytes == 0) {
    return;
  }
  check(cudaMemsetAsync(ptr, 0, bytes, work_stream()), "memset");
  finish("zero");
}

void sgd_update(float* params, const float* grads, float* velocity, std::size_t n, SgdCoeffs c) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  const float damp = 1.f - c.dampening;
  const bool vec = n >= 4 && aligned16(params) && aligned16(grads) && aligned16(velocity);
  auto stream = work_stream();
  if (vec) {
    const std::size_t n4 = n / 4;
    const int blocks = launch_blocks(n4, threads);
    if (c.momentum != 0.f && c.nesterov) {
      sgd_vec4_kernel<true><<<blocks, threads, 0, stream>>>(params, grads, velocity, n4, c.lr,
                                                            c.weight_decay, c.momentum, damp);
    } else {
      sgd_vec4_kernel<false><<<blocks, threads, 0, stream>>>(params, grads, velocity, n4, c.lr,
                                                             c.weight_decay, c.momentum, damp);
    }
    const std::size_t tail = n4 * 4;
    if (tail < n) {
      sgd_scalar_kernel<<<1, 32, 0, stream>>>(params + tail, grads + tail,
                                              velocity != nullptr ? velocity + tail : nullptr, n - tail,
                                              c.lr, c.weight_decay, c.momentum, damp, c.nesterov);
    }
  } else {
    const int blocks = launch_blocks(n, threads);
    sgd_scalar_kernel<<<blocks, threads, 0, stream>>>(params, grads, velocity, n, c.lr, c.weight_decay,
                                                      c.momentum, damp, c.nesterov);
  }
  finish("sgd");
}

void adam_update(float* params, const float* grads, float* exp_avg, float* exp_avg_sq,
                 float* max_exp_avg_sq, std::size_t n, AdamCoeffs c) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  const bool ams = c.amsgrad && max_exp_avg_sq != nullptr;
  const bool vec = n >= 4 && aligned16(params) && aligned16(grads) && aligned16(exp_avg) &&
                   aligned16(exp_avg_sq) && aligned16(max_exp_avg_sq);
  auto stream = work_stream();
  if (vec) {
    const std::size_t n4 = n / 4;
    const int blocks = launch_blocks(n4, threads);
    if (c.decoupled && ams) {
      adam_vec4_kernel<true, true><<<blocks, threads, 0, stream>>>(
          params, grads, exp_avg, exp_avg_sq, max_exp_avg_sq, n4, c.lr, c.beta1, c.one_minus_beta1,
          c.beta2, c.one_minus_beta2, c.eps, c.weight_decay, c.inv_bc1, c.inv_bc2);
    } else if (c.decoupled) {
      adam_vec4_kernel<true, false><<<blocks, threads, 0, stream>>>(
          params, grads, exp_avg, exp_avg_sq, max_exp_avg_sq, n4, c.lr, c.beta1, c.one_minus_beta1,
          c.beta2, c.one_minus_beta2, c.eps, c.weight_decay, c.inv_bc1, c.inv_bc2);
    } else if (ams) {
      adam_vec4_kernel<false, true><<<blocks, threads, 0, stream>>>(
          params, grads, exp_avg, exp_avg_sq, max_exp_avg_sq, n4, c.lr, c.beta1, c.one_minus_beta1,
          c.beta2, c.one_minus_beta2, c.eps, c.weight_decay, c.inv_bc1, c.inv_bc2);
    } else {
      adam_vec4_kernel<false, false><<<blocks, threads, 0, stream>>>(
          params, grads, exp_avg, exp_avg_sq, max_exp_avg_sq, n4, c.lr, c.beta1, c.one_minus_beta1,
          c.beta2, c.one_minus_beta2, c.eps, c.weight_decay, c.inv_bc1, c.inv_bc2);
    }
    const std::size_t tail = n4 * 4;
    if (tail < n) {
      adam_scalar_kernel<<<1, 32, 0, stream>>>(
          params + tail, grads + tail, exp_avg + tail, exp_avg_sq + tail,
          max_exp_avg_sq != nullptr ? max_exp_avg_sq + tail : nullptr, c.lr, c.beta1, c.beta2, c.eps,
          c.weight_decay, c.inv_bc1, c.inv_bc2, c.decoupled, ams ? 1 : 0, n - tail);
    }
  } else {
    const int blocks = launch_blocks(n, threads);
    adam_scalar_kernel<<<blocks, threads, 0, stream>>>(
        params, grads, exp_avg, exp_avg_sq, max_exp_avg_sq, c.lr, c.beta1, c.beta2, c.eps,
        c.weight_decay, c.inv_bc1, c.inv_bc2, c.decoupled, ams ? 1 : 0, n);
  }
  finish("adam");
}

void adagrad_update(float* params, const float* grads, float* state, std::size_t n, AdagradCoeffs c) {
  if (n == 0) {
    return;
  }
  const float lr = c.lr / (1.f + (c.step - 1.f) * c.lr_decay);
  constexpr int threads = 256;
  adagrad_kernel<<<launch_blocks(n, threads), threads, 0, work_stream()>>>(params, grads, state, n, lr,
                                                                           c.eps, c.weight_decay);
  finish("adagrad");
}

void rmsprop_update(float* params, const float* grads, float* square, float* buf, float* grad_avg,
                    std::size_t n, RmspropCoeffs c) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  rmsprop_kernel<<<launch_blocks(n, threads), threads, 0, work_stream()>>>(
      params, grads, square, buf, grad_avg, n, c.lr, c.alpha, c.one_minus_alpha, c.eps, c.weight_decay,
      c.momentum, c.centered);
  finish("rmsprop");
}

void adadelta_update(float* params, const float* grads, float* square_avg, float* acc_delta,
                     std::size_t n, AdadeltaCoeffs c) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  adadelta_kernel<<<launch_blocks(n, threads), threads, 0, work_stream()>>>(
      params, grads, square_avg, acc_delta, n, c.lr, c.rho, c.one_minus_rho, c.eps, c.weight_decay);
  finish("adadelta");
}

void nadam_update(float* params, const float* grads, float* exp_avg, float* exp_avg_sq, std::size_t n,
                  NadamCoeffs c) {
  if (n == 0) {
    return;
  }
  const float mu_product_next = c.mu_product * c.mu_next;
  const float step_g = -c.lr * (1.f - c.mu) / (1.f - c.mu_product);
  const float step_m = -c.lr * c.mu_next / (1.f - mu_product_next);
  const float bc2 = c.bias_correction2 > 0.f ? c.bias_correction2 : 1.f;
  constexpr int threads = 256;
  nadam_kernel<<<launch_blocks(n, threads), threads, 0, work_stream()>>>(
      params, grads, exp_avg, exp_avg_sq, n, c.lr, c.beta1, c.one_minus_beta1, c.beta2,
      c.one_minus_beta2, c.eps, c.weight_decay, c.mu, step_g, step_m, 1.f / bc2, c.decoupled);
  finish("nadam");
}

void radam_update(float* params, const float* grads, float* exp_avg, float* exp_avg_sq, std::size_t n,
                  RadamCoeffs c) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  radam_kernel<<<launch_blocks(n, threads), threads, 0, work_stream()>>>(
      params, grads, exp_avg, exp_avg_sq, n, c.lr, c.beta1, c.one_minus_beta1, c.beta2,
      c.one_minus_beta2, c.eps, c.weight_decay, c.lr * c.inv_bc1, c.adaptive_step, c.decoupled,
      c.rectified);
  finish("radam");
}

void adabelief_update(float* params, const float* grads, float* exp_avg, float* exp_avg_var,
                      std::size_t n, AdaBeliefCoeffs c) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  adabelief_kernel<<<launch_blocks(n, threads), threads, 0, work_stream()>>>(
      params, grads, exp_avg, exp_avg_var, n, c.lr, c.beta1, c.one_minus_beta1, c.beta2,
      c.one_minus_beta2, c.eps, c.weight_decay, c.inv_bc1, c.inv_bc2, c.decoupled);
  finish("adabelief");
}

void lamb_update(float* params, const float* grads, float* exp_avg, float* exp_avg_sq, float* scratch,
                 std::size_t n, LambCoeffs c) {
  if (n == 0) {
    return;
  }
  double* norms = norm_scratch();
  auto stream = work_stream();
  check(cudaMemsetAsync(norms, 0, sizeof(double) * 2, stream), "lamb zero");
  constexpr int threads = 256;
  lamb_moments_kernel<<<launch_blocks(n, threads), threads, 0, stream>>>(
      params, grads, exp_avg, exp_avg_sq, scratch, n, c.beta1, c.one_minus_beta1, c.beta2,
      c.one_minus_beta2, c.eps, c.weight_decay, c.inv_bc1, c.inv_bc2, norms, norms + 1);
  check(cudaGetLastError(), "lamb moments");
  lamb_apply_kernel<<<launch_blocks(n, threads), threads, 0, stream>>>(params, scratch, n, c.lr, norms,
                                                                       norms + 1);
  finish("lamb");
}

void lars_update(float* params, const float* grads, float* velocity, std::size_t n, LarsCoeffs c) {
  if (n == 0) {
    return;
  }
  double* norms = norm_scratch();
  auto stream = work_stream();
  check(cudaMemsetAsync(norms, 0, sizeof(double) * 2, stream), "lars zero");
  constexpr int threads = 256;
  lars_norm_kernel<<<launch_blocks(n, threads), threads, 0, stream>>>(params, grads, n, norms,
                                                                     norms + 1);
  check(cudaGetLastError(), "lars norm");
  lars_apply_kernel<<<launch_blocks(n, threads), threads, 0, stream>>>(
      params, grads, velocity, n, c.lr, c.weight_decay, c.momentum, c.eta, c.eps, norms, norms + 1);
  finish("lars");
}

void lion_update(float* params, const float* grads, float* exp_avg, std::size_t n, LionCoeffs c) {
  if (n == 0) {
    return;
  }
  const float decay = 1.f - c.lr * c.weight_decay;
  const int apply = c.weight_decay != 0.f ? 1 : 0;
  constexpr int threads = 256;
  auto stream = work_stream();
  const bool vec = n >= 4 && aligned16(params) && aligned16(grads) && aligned16(exp_avg);
  if (vec) {
    const std::size_t n4 = n / 4;
    lion_vec4_kernel<<<launch_blocks(n4, threads), threads, 0, stream>>>(
        params, grads, exp_avg, n4, c.lr, c.beta1, c.one_minus_beta1, c.beta2, c.one_minus_beta2, decay,
        apply);
    const std::size_t tail = n4 * 4;
    if (tail < n) {
      lion_scalar_kernel<<<1, 32, 0, stream>>>(params + tail, grads + tail, exp_avg + tail, n - tail,
                                               c.lr, c.beta1, c.one_minus_beta1, c.beta2,
                                               c.one_minus_beta2, decay, apply);
    }
  } else {
    lion_scalar_kernel<<<launch_blocks(n, threads), threads, 0, stream>>>(
        params, grads, exp_avg, n, c.lr, c.beta1, c.one_minus_beta1, c.beta2, c.one_minus_beta2, decay,
        apply);
  }
  finish("lion");
}

struct StreamManager::Impl {
  cudaStream_t streams[4]{};
  Impl() {
    for (auto& stream : streams) {
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    }
  }
  ~Impl() {
    for (auto stream : streams) {
      cudaStreamDestroy(stream);
    }
  }
};

StreamManager& StreamManager::instance() {
  static StreamManager manager;
  return manager;
}

StreamManager::StreamManager() : impl_(new Impl) {}
StreamManager::~StreamManager() { delete impl_; }

cudaStream_t StreamManager::stream_for(std::size_t group_index) const {
  return impl_->streams[group_index % 4];
}

void StreamManager::synchronize() const {
  for (auto stream : impl_->streams) {
    check(cudaStreamSynchronize(stream), "stream sync");
  }
}

}  // namespace cuda
}  // namespace nexus_optim
