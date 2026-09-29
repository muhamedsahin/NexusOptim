#pragma once

#include "nexus_optim/core/types.hpp"
#include "nexus_optim/simd/avx2_kernels.hpp"
#include "nexus_optim/simd/avx512_kernels.hpp"
#include "nexus_optim/simd/kernel_config.hpp"
#include "nexus_optim/simd/neon_kernels.hpp"
#include "nexus_optim/simd/scalar_kernels.hpp"

#include <cstddef>
#include <optional>

namespace nexus_optim {

/// Detected once at startup. `force_simd_level` overrides it for benchmarks.
SimdLevel detected_simd_level() noexcept;
SimdLevel cpu_simd_level() noexcept;
bool force_simd_level(std::optional<SimdLevel> level) noexcept;

void sgd_update(float* p, const float* g, float* velocity, std::size_t n, SgdCoeffs c);
void sgd_update(double* p, const double* g, double* velocity, std::size_t n, SgdCoeffs c);
void adam_update(float* p, const float* g, float* m, float* v, float* v_max, std::size_t n,
                 AdamCoeffs c);
void adam_update(double* p, const double* g, double* m, double* v, double* v_max, std::size_t n,
                 AdamCoeffs c);
void adagrad_update(float* p, const float* g, float* state, std::size_t n, AdagradCoeffs c);
void rmsprop_update(float* p, const float* g, float* square, float* buf, float* grad_avg,
                    std::size_t n, RmspropCoeffs c);
void adadelta_update(float* p, const float* g, float* square_avg, float* acc_delta, std::size_t n,
                     AdadeltaCoeffs c);
void nadam_update(float* p, const float* g, float* m, float* v, std::size_t n, NadamCoeffs c);
void radam_update(float* p, const float* g, float* m, float* v, std::size_t n, RadamCoeffs c);
void adabelief_update(float* p, const float* g, float* m, float* s, std::size_t n,
                      AdaBeliefCoeffs c);
void lamb_update(float* p, const float* g, float* m, float* v, float* scratch, std::size_t n,
                 LambCoeffs c);
void lars_update(float* p, const float* g, float* velocity, std::size_t n, LarsCoeffs c);
void lion_update(float* p, const float* g, float* m, std::size_t n, LionCoeffs c);

double l2_sq(const float* x, std::size_t n);
double l2_sq(const double* x, std::size_t n);
double linf(const float* x, std::size_t n);
void scale_inplace(float* x, std::size_t n, float scale);
void clip_abs(float* x, std::size_t n, float clip);
void lookahead_blend(float* fast, float* slow, std::size_t n, float alpha);
void centralize(float* grad, std::size_t channels, std::size_t spatial);

}  // namespace nexus_optim
