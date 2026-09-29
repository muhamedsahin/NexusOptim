#include "nexus_optim/simd/neon_kernels.hpp"

#include "nexus_optim/simd/scalar_kernels.hpp"

#include <cstddef>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define NEXUS_OPTIM_NEON_IMPL 1
#else
#define NEXUS_OPTIM_NEON_IMPL 0
#endif

namespace nexus_optim {
namespace neon {

bool compiled() noexcept { return NEXUS_OPTIM_NEON_IMPL != 0; }

#if NEXUS_OPTIM_NEON_IMPL

void sgd_update(float* p, const float* g, float* velocity, std::size_t n, SgdCoeffs c) {
  if (c.momentum != 0.f) {
    scalar::sgd_update(p, g, velocity, n, c);
    return;
  }
  const float32x4_t lr = vdupq_n_f32(c.lr);
  const float32x4_t wd = vdupq_n_f32(c.weight_decay);
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t param = vld1q_f32(p + i);
    float32x4_t grad = vmlaq_f32(vld1q_f32(g + i), wd, param);
    const float32x4_t step = vmulq_f32(lr, grad);
    vst1q_f32(p + i, vsubq_f32(param, step));
  }
  if (i < n) {
    scalar::sgd_update(p + i, g + i, nullptr, n - i, c);
  }
}

void adam_update(float* p, const float* g, float* m, float* v, float* v_max, std::size_t n,
                 AdamCoeffs c) {
  const float32x4_t lr = vdupq_n_f32(c.lr);
  const float32x4_t b1 = vdupq_n_f32(c.beta1);
  const float32x4_t ob1 = vdupq_n_f32(c.one_minus_beta1);
  const float32x4_t b2 = vdupq_n_f32(c.beta2);
  const float32x4_t ob2 = vdupq_n_f32(c.one_minus_beta2);
  const float32x4_t eps = vdupq_n_f32(c.eps);
  const float32x4_t wd = vdupq_n_f32(c.weight_decay);
  const float32x4_t inv1 = vdupq_n_f32(c.inv_bc1);
  const float32x4_t inv2 = vdupq_n_f32(c.inv_bc2);
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t param = vld1q_f32(p + i);
    float32x4_t grad = vld1q_f32(g + i);
    if (c.decoupled) {
      param = vmlsq_f32(param, vmulq_f32(lr, wd), param);
    } else {
      grad = vmlaq_f32(grad, wd, param);
    }
    float32x4_t mt = vmlaq_f32(vmulq_f32(ob1, grad), b1, vld1q_f32(m + i));
    float32x4_t vt =
        vmlaq_f32(vmulq_f32(ob2, vmulq_f32(grad, grad)), b2, vld1q_f32(v + i));
    vst1q_f32(m + i, mt);
    vst1q_f32(v + i, vt);
    float32x4_t src = vt;
    if (c.amsgrad && v_max != nullptr) {
      src = vmaxq_f32(vld1q_f32(v_max + i), vt);
      vst1q_f32(v_max + i, src);
    }
#if defined(__aarch64__)
    const float32x4_t denom = vaddq_f32(vsqrtq_f32(vmulq_f32(src, inv2)), eps);
    const float32x4_t upd = vdivq_f32(vmulq_f32(mt, inv1), denom);
    param = vmlsq_f32(param, lr, upd);
#else
    alignas(16) float ps[4], ms[4], ss[4];
    vst1q_f32(ps, param);
    vst1q_f32(ms, mt);
    vst1q_f32(ss, src);
    for (int k = 0; k < 4; ++k) {
      ps[k] -= c.lr * (ms[k] * c.inv_bc1) / (std::sqrt(ss[k] * c.inv_bc2) + c.eps);
    }
    param = vld1q_f32(ps);
#endif
    vst1q_f32(p + i, param);
  }
  if (i < n) {
    scalar::adam_update(p + i, g + i, m + i, v + i, v_max != nullptr ? v_max + i : nullptr, n - i,
                        c);
  }
}

#else

void sgd_update(float* p, const float* g, float* velocity, std::size_t n, SgdCoeffs c) {
  scalar::sgd_update(p, g, velocity, n, c);
}

void adam_update(float* p, const float* g, float* m, float* v, float* v_max, std::size_t n,
                 AdamCoeffs c) {
  scalar::adam_update(p, g, m, v, v_max, n, c);
}

#endif

}  // namespace neon
}  // namespace nexus_optim
