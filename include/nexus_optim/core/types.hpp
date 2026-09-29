#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

/// @file types.hpp
/// Precision tags, device identifiers, and library-wide constants.

namespace nexus_optim {

/// Where a parameter buffer lives. Elementwise updates never move data across devices.
enum class Device : std::uint8_t {
  CPU = 0,
  CUDA = 1,
};

/// Storage precision. Moments for narrow types are kept in FP32 (master state).
enum class DType : std::uint8_t {
  F32 = 0,
  F64 = 1,
  F16 = 2,
  BF16 = 3,
};

/// Runtime-selected SIMD path. Chosen once at process start via CPUID, never inside a tight loop.
enum class SimdLevel : std::uint8_t {
  Scalar = 0,
  Neon = 1,
  Avx2 = 2,
  Avx512 = 3,
};

/// Parameter tensors shorter than this are updated on the calling thread.
/// Below the threshold, thread wake-up costs more than the bandwidth saved.
inline constexpr std::size_t kParallelGrain = 50000;

/// Optimizer state buffers and vectorized tails are aligned to an AVX-512 cache line.
inline constexpr std::size_t kAlignment = 64;

inline constexpr const char* version() noexcept { return "1.0.0"; }

inline const char* device_name(Device d) noexcept {
  return d == Device::CUDA ? "cuda" : "cpu";
}

inline const char* simd_level_name(SimdLevel level) noexcept {
  switch (level) {
    case SimdLevel::Avx512:
      return "avx512";
    case SimdLevel::Avx2:
      return "avx2";
    case SimdLevel::Neon:
      return "neon";
    case SimdLevel::Scalar:
      return "scalar";
  }
  return "scalar";
}

}  // namespace nexus_optim
