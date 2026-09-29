#include "nexus_optim/nexus_optim.hpp"

#include <chrono>
#include <iostream>
#include <vector>

int main() {
  constexpr std::size_t n = 1'000'000;
  std::vector<float> params(n, 1.f);
  std::vector<float> grads(n, 0.01f);
  nexus_optim::ParamGroupOptions group;
  group.learning_rate = 0.01;
  nexus_optim::SGD<float>::Options options;
  options.lr = 0.01;
  options.weight_decay = 1e-4;
  nexus_optim::SGD<float> optimizer(
      {nexus_optim::make_group(params.data(), grads.data(), n, group)}, options);
  optimizer.step();
  constexpr int iters = 100;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) optimizer.step();
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const double usec = seconds * 1e6 / iters;
  const double gbps = (iters / seconds) * static_cast<double>(n) * 12.0 / 1e9;
  std::cout << "sgd " << usec << " us/step  " << gbps << " GB/s  level="
            << nexus_optim::simd_level_name(nexus_optim::detected_simd_level()) << '\n';
  return 0;
}
