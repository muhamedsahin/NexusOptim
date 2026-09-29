#include "nexus_optim/nexus_optim.hpp"

#include <chrono>
#include <iostream>
#include <vector>

#if defined(NEXUS_OPTIM_WITH_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

void time_adamw(std::size_t n, int iters) {
  std::vector<float> params(n, 0.1f);
  std::vector<float> grads(n, 0.01f);
  nexus_optim::ParamGroupOptions group;
  group.learning_rate = 1e-3;
  nexus_optim::AdamW<float>::Options options;
  options.lr = 1e-3;
  options.weight_decay = 0.01;
  nexus_optim::AdamW<float> cpu(
      {nexus_optim::make_group(params.data(), grads.data(), n, group)}, options);
  cpu.step();
  const auto cpu_start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    cpu.step();
  }
  const double cpu_step =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - cpu_start).count() / iters;
  std::cout << "n=" << n << "  cpu " << cpu_step * 1e6 << " us\n";

#if defined(NEXUS_OPTIM_WITH_CUDA)
  float* dp = nullptr;
  float* dg = nullptr;
  cudaMalloc(&dp, n * sizeof(float));
  cudaMalloc(&dg, n * sizeof(float));
  cudaMemcpy(dp, params.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dg, grads.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  nexus_optim::ParamGroup<float> device_group;
  device_group.params = {dp};
  device_group.grads = {dg};
  device_group.numels = {n};
  device_group.devices = {nexus_optim::Device::CUDA};
  device_group.options.learning_rate = 1e-3;
  nexus_optim::AdamW<float> gpu({device_group}, options);
  gpu.step();
  const auto gpu_start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    gpu.step();
  }
  const double gpu_step =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - gpu_start).count() / iters;
  const double gbps = (static_cast<double>(n) * 28.0) / gpu_step / 1e9;
  std::cout << "n=" << n << "  gpu " << gpu_step * 1e6 << " us  " << gbps << " GB/s  cpu/gpu "
            << (cpu_step / gpu_step) << "x\n";
  cudaFree(dp);
  cudaFree(dg);
#endif
}

}  // namespace

int main() {
  time_adamw(1 << 20, 40);
  return 0;
}
