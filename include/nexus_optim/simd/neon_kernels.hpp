#pragma once

// ARM NEON path. On x86 this header is intentionally empty of intrinsics;
// the dispatch layer only selects NEON when the translation unit was built
// for AArch64 and the CPU reports AdvSIMD.

#include "nexus_optim/simd/kernel_config.hpp"

#include <cstddef>

namespace nexus_optim {
namespace neon {

bool compiled() noexcept;

void sgd_update(float* p, const float* g, float* velocity, std::size_t n, SgdCoeffs c);
void adam_update(float* p, const float* g, float* m, float* v, float* v_max, std::size_t n,
                 AdamCoeffs c);

}  // namespace neon
}  // namespace nexus_optim
