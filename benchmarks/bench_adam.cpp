#include "nexus_optim/nexus_optim.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

double bench_adam(std::size_t n, int iters, nexus_optim::SimdLevel level) {
  std::vector<float> params(n, 0.1f);
  std::vector<float> grads(n, 0.01f);
  nexus_optim::ParamGroupOptions group;
  group.learning_rate = 1e-3;
  nexus_optim::AdamW<float>::Options options;
  options.lr = 1e-3;
  options.weight_decay = 0.01;
  nexus_optim::AdamW<float> optimizer(
      {nexus_optim::make_group(params.data(), grads.data(), n, group)}, options);
  if (!nexus_optim::force_simd_level(level)) {
    return -1.0;
  }
  optimizer.step();
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    optimizer.step();
  }
  const auto stop = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(stop - start).count();
  return static_cast<double>(iters) / seconds;
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t n = 1'000'000;
  int iters = 50;
  if (argc > 1) n = static_cast<std::size_t>(std::stoull(argv[1]));
  if (argc > 2) iters = std::stoi(argv[2]);

  std::cout << "detected " << nexus_optim::simd_level_name(nexus_optim::detected_simd_level())
            << "  helpers=" << nexus_optim::ThreadPool::instance().helper_count()
            << "  n=" << n << "\n";
  const double scalar = bench_adam(n, iters, nexus_optim::SimdLevel::Scalar);
  const double avx2 = bench_adam(n, iters, nexus_optim::SimdLevel::Avx2);
  const double avx512 = bench_adam(n, iters, nexus_optim::SimdLevel::Avx512);
  // AdamW touches parameter, gradient, m, and v: 16 bytes read/write per element roughly
  // p read+write, g read, m read+write, v read+write = 4+4+4+4+4+4 = 28 bytes of traffic.
  constexpr double bytes_per_element = 28.0;
  auto report = [&](const char* name, double steps_per_sec) {
    if (steps_per_sec < 0.0) {
      std::cout << name << " unavailable\n";
      return;
    }
    const double gbps = steps_per_sec * static_cast<double>(n) * bytes_per_element / 1e9;
    const double usec = 1e6 / steps_per_sec;
    std::cout << name << "  " << usec << " us/step  " << gbps << " GB/s\n";
  };
  report("scalar", scalar);
  report("avx2  ", avx2);
  report("avx512", avx512);
  if (scalar > 0.0 && avx2 > 0.0) {
    std::cout << "avx2/scalar " << (avx2 / scalar) << "x\n";
  }
  if (scalar > 0.0 && avx512 > 0.0) {
    std::cout << "avx512/scalar " << (avx512 / scalar) << "x\n";
  }
  nexus_optim::force_simd_level(std::nullopt);
  return 0;
}
