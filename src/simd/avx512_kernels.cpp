#include "nexus_optim/simd/avx512_kernels.hpp"

#include "nexus_optim/simd/avx2_kernels.hpp"

#include <cstddef>
#include <immintrin.h>

namespace nexus_optim {
namespace avx512 {
namespace {

template <bool Nesterov>
void sgd_mom(float* p, const float* g, float* velocity, std::size_t n, SgdCoeffs c) {
  const __m512 lr = _mm512_set1_ps(c.lr);
  const __m512 wd = _mm512_set1_ps(c.weight_decay);
  const __m512 mu = _mm512_set1_ps(c.momentum);
  const __m512 damp = _mm512_set1_ps(1.f - c.dampening);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 param = _mm512_loadu_ps(p + i);
    __m512 grad = _mm512_fmadd_ps(wd, param, _mm512_loadu_ps(g + i));
    __m512 vel = _mm512_fmadd_ps(mu, _mm512_loadu_ps(velocity + i), _mm512_mul_ps(damp, grad));
    _mm512_storeu_ps(velocity + i, vel);
    if constexpr (Nesterov) {
      grad = _mm512_fmadd_ps(mu, vel, grad);
    } else {
      grad = vel;
    }
    _mm512_storeu_ps(p + i, _mm512_fnmadd_ps(lr, grad, param));
  }
  if (i < n) {
    avx2::sgd_update(p + i, g + i, velocity + i, n - i, c);
  }
}

template <bool Decoupled, bool Amsgrad>
void adam_f32(float* p, const float* g, float* m, float* v, float* v_max, std::size_t n,
              AdamCoeffs c) {
  const __m512 lr = _mm512_set1_ps(c.lr);
  const __m512 b1 = _mm512_set1_ps(c.beta1);
  const __m512 ob1 = _mm512_set1_ps(c.one_minus_beta1);
  const __m512 b2 = _mm512_set1_ps(c.beta2);
  const __m512 ob2 = _mm512_set1_ps(c.one_minus_beta2);
  const __m512 eps = _mm512_set1_ps(c.eps);
  const __m512 wd = _mm512_set1_ps(c.weight_decay);
  const __m512 lr_wd = _mm512_set1_ps(c.lr * c.weight_decay);
  const __m512 inv1 = _mm512_set1_ps(c.inv_bc1);
  const __m512 inv2 = _mm512_set1_ps(c.inv_bc2);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 param = _mm512_loadu_ps(p + i);
    __m512 grad = _mm512_loadu_ps(g + i);
    __m512 mt = _mm512_loadu_ps(m + i);
    __m512 vt = _mm512_loadu_ps(v + i);
    if constexpr (Decoupled) {
      param = _mm512_fnmadd_ps(lr_wd, param, param);
    } else {
      grad = _mm512_fmadd_ps(wd, param, grad);
    }
    mt = _mm512_fmadd_ps(b1, mt, _mm512_mul_ps(ob1, grad));
    vt = _mm512_fmadd_ps(b2, vt, _mm512_mul_ps(ob2, _mm512_mul_ps(grad, grad)));
    _mm512_storeu_ps(m + i, mt);
    _mm512_storeu_ps(v + i, vt);
    __m512 src = vt;
    if constexpr (Amsgrad) {
      __m512 vm = _mm512_loadu_ps(v_max + i);
      src = _mm512_max_ps(vm, vt);
      _mm512_storeu_ps(v_max + i, src);
    }
    const __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(_mm512_mul_ps(src, inv2)), eps);
    param = _mm512_fnmadd_ps(lr, _mm512_div_ps(_mm512_mul_ps(mt, inv1), denom), param);
    _mm512_storeu_ps(p + i, param);
  }
  if (i < n) {
    avx2::adam_update(p + i, g + i, m + i, v + i, v_max != nullptr ? v_max + i : nullptr, n - i, c);
  }
}

}  // namespace

void sgd_update(float* p, const float* g, float* velocity, std::size_t n, SgdCoeffs c) {
  if (c.momentum == 0.f || velocity == nullptr) {
    const __m512 lr = _mm512_set1_ps(c.lr);
    const __m512 wd = _mm512_set1_ps(c.weight_decay);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
      const __m512 param = _mm512_loadu_ps(p + i);
      const __m512 grad = _mm512_fmadd_ps(wd, param, _mm512_loadu_ps(g + i));
      _mm512_storeu_ps(p + i, _mm512_fnmadd_ps(lr, grad, param));
    }
    if (i < n) {
      avx2::sgd_update(p + i, g + i, nullptr, n - i, c);
    }
    return;
  }
  if (c.nesterov) {
    sgd_mom<true>(p, g, velocity, n, c);
  } else {
    sgd_mom<false>(p, g, velocity, n, c);
  }
}

void adam_update(float* p, const float* g, float* m, float* v, float* v_max, std::size_t n,
                 AdamCoeffs c) {
  const bool ams = c.amsgrad && v_max != nullptr;
  if (c.decoupled) {
    if (ams) {
      adam_f32<true, true>(p, g, m, v, v_max, n, c);
    } else {
      adam_f32<true, false>(p, g, m, v, v_max, n, c);
    }
  } else if (ams) {
    adam_f32<false, true>(p, g, m, v, v_max, n, c);
  } else {
    adam_f32<false, false>(p, g, m, v, v_max, n, c);
  }
}

double l2_sq(const float* x, std::size_t n) {
  double sum = 0.0;
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512 v = _mm512_loadu_ps(x + i);
    const __m512 sq = _mm512_mul_ps(v, v);
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, sq);
    for (float t : tmp) {
      const double d = static_cast<double>(t);
      sum += d;
    }
  }
  if (i < n) {
    sum += avx2::l2_sq(x + i, n - i);
  }
  return sum;
}

void scale_inplace(float* x, std::size_t n, float s) {
  const __m512 scale = _mm512_set1_ps(s);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), scale));
  }
  if (i < n) {
    avx2::scale_inplace(x + i, n - i, s);
  }
}

}  // namespace avx512
}  // namespace nexus_optim
