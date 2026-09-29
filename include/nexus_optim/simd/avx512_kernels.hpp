#pragma once

#include "nexus_optim/simd/kernel_config.hpp"

#include <cstddef>

#if !defined(NEXUS_OPTIM_ENABLE_AVX512)
#include "nexus_optim/simd/avx2_kernels.hpp"
#endif

namespace nexus_optim {
namespace avx512 {

#if defined(NEXUS_OPTIM_ENABLE_AVX512)

void sgd_update(float* p, const float* g, float* velocity, std::size_t n, SgdCoeffs c);
void adam_update(float* p, const float* g, float* m, float* v, float* v_max, std::size_t n,
                 AdamCoeffs c);
double l2_sq(const float* x, std::size_t n);
void scale_inplace(float* x, std::size_t n, float s);

#else

inline void sgd_update(float* p, const float* g, float* velocity, std::size_t n, SgdCoeffs c) {
  avx2::sgd_update(p, g, velocity, n, c);
}
inline void adam_update(float* p, const float* g, float* m, float* v, float* v_max, std::size_t n,
                        AdamCoeffs c) {
  avx2::adam_update(p, g, m, v, v_max, n, c);
}
inline double l2_sq(const float* x, std::size_t n) { return avx2::l2_sq(x, n); }
inline void scale_inplace(float* x, std::size_t n, float s) { avx2::scale_inplace(x, n, s); }

#endif

}  // namespace avx512
}  // namespace nexus_optim
