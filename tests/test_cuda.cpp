#include "nexus_optim/nexus_optim.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {

class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::vector<float>& host) : count_(host.size()) {
    if (count_ == 0) {
      return;
    }
    if (cudaMalloc(&ptr_, count_ * sizeof(float)) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
    if (cudaMemcpy(ptr_, host.data(), count_ * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy H2D failed");
    }
  }
  ~DeviceBuffer() { cudaFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  float* get() noexcept { return ptr_; }
  std::vector<float> download() const {
    std::vector<float> host(count_);
    EXPECT_EQ(cudaMemcpy(host.data(), ptr_, count_ * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return host;
  }

 private:
  float* ptr_ = nullptr;
  std::size_t count_ = 0;
};

void expect_close(const std::vector<float>& got, const std::vector<float>& ref, float tol) {
  ASSERT_EQ(got.size(), ref.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float scale = std::max(1.f, std::abs(ref[i]));
    EXPECT_NEAR(got[i], ref[i], tol * scale) << "index " << i;
  }
}

std::vector<float> pattern(int n, float offset) {
  std::vector<float> values(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    values[static_cast<std::size_t>(i)] = offset + 0.001f * static_cast<float>((i % 97) - 40);
  }
  return values;
}

template <typename RunCpu, typename RunGpu>
void compare(int n, RunCpu cpu, RunGpu gpu) {
  auto params = pattern(n, 0.25f);
  auto grads = pattern(n, 0.01f);
  auto params_cpu = params;
  auto grads_cpu = grads;
  cpu(params_cpu, grads_cpu);

  DeviceBuffer dp(params);
  DeviceBuffer dg(grads);
  nexus_optim::ParamGroup<float> group;
  group.params = {dp.get()};
  group.grads = {dg.get()};
  group.numels = {static_cast<std::size_t>(n)};
  group.devices = {nexus_optim::Device::CUDA};
  group.options.learning_rate = 1e-3;
  gpu(group);
  expect_close(dp.download(), params_cpu, 2e-4f);
}

}  // namespace

TEST(Cuda, MatchesCpuSgdAdamLionLamb) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "no CUDA device";
  }
  for (int n : {7, 8, 17, 1024, 10007}) {
    compare(
        n,
        [](std::vector<float>& p, std::vector<float>& g) {
          nexus_optim::ParamGroupOptions options;
          options.learning_rate = 1e-3;
          nexus_optim::SGD<float>::Options sgd;
          sgd.weight_decay = 0.01;
          nexus_optim::SGD<float> opt({nexus_optim::make_group(p.data(), g.data(), p.size(), options)},
                                      sgd);
          opt.step();
        },
        [](nexus_optim::ParamGroup<float>& group) {
          group.options.learning_rate = 1e-3;
          nexus_optim::SGD<float>::Options sgd;
          sgd.weight_decay = 0.01;
          nexus_optim::SGD<float> opt({group}, sgd);
          opt.step();
        });

    compare(
        n,
        [](std::vector<float>& p, std::vector<float>& g) {
          nexus_optim::ParamGroupOptions options;
          options.learning_rate = 1e-3;
          nexus_optim::AdamW<float>::Options adam;
          adam.lr = 1e-3;
          adam.weight_decay = 0.01;
          nexus_optim::AdamW<float> opt(
              {nexus_optim::make_group(p.data(), g.data(), p.size(), options)}, adam);
          opt.step();
          std::fill(g.begin(), g.end(), 0.02f);
          opt.step();
        },
        [](nexus_optim::ParamGroup<float>& group) {
          group.options.learning_rate = 1e-3;
          nexus_optim::AdamW<float>::Options adam;
          adam.lr = 1e-3;
          adam.weight_decay = 0.01;
          nexus_optim::AdamW<float> opt({group}, adam);
          opt.step();
          std::vector<float> g2(group.numels[0], 0.02f);
          ASSERT_EQ(cudaMemcpy(group.grads[0], g2.data(), g2.size() * sizeof(float),
                               cudaMemcpyHostToDevice),
                    cudaSuccess);
          opt.step();
        });

    compare(
        n,
        [](std::vector<float>& p, std::vector<float>& g) {
          nexus_optim::ParamGroupOptions options;
          options.learning_rate = 1e-4;
          nexus_optim::Lion<float>::Options lion;
          lion.lr = 1e-4;
          lion.weight_decay = 0.01;
          nexus_optim::Lion<float> opt({nexus_optim::make_group(p.data(), g.data(), p.size(), options)},
                                       lion);
          opt.step();
        },
        [](nexus_optim::ParamGroup<float>& group) {
          group.options.learning_rate = 1e-4;
          nexus_optim::Lion<float>::Options lion;
          lion.lr = 1e-4;
          lion.weight_decay = 0.01;
          nexus_optim::Lion<float> opt({group}, lion);
          opt.step();
        });

    compare(
        n,
        [](std::vector<float>& p, std::vector<float>& g) {
          nexus_optim::ParamGroupOptions options;
          options.learning_rate = 1e-3;
          nexus_optim::LAMB<float>::Options lamb;
          lamb.lr = 1e-3;
          lamb.weight_decay = 0.01;
          nexus_optim::LAMB<float> opt({nexus_optim::make_group(p.data(), g.data(), p.size(), options)},
                                       lamb);
          opt.step();
        },
        [](nexus_optim::ParamGroup<float>& group) {
          group.options.learning_rate = 1e-3;
          nexus_optim::LAMB<float>::Options lamb;
          lamb.lr = 1e-3;
          lamb.weight_decay = 0.01;
          nexus_optim::LAMB<float> opt({group}, lamb);
          opt.step();
        });
  }
}
