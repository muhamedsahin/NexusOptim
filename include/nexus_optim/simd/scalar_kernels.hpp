#pragma once

#include "nexus_optim/simd/kernel_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#if defined(_MSC_VER)
#define NEXUS_SCALAR_LOOP __pragma(loop(no_vector))
#elif defined(__clang__)
#define NEXUS_SCALAR_LOOP _Pragma("clang loop vectorize(disable) interleave(disable)")
#elif defined(__GNUC__)
#define NEXUS_SCALAR_LOOP _Pragma("GCC novector")
#else
#define NEXUS_SCALAR_LOOP
#endif

namespace nexus_optim {
namespace scalar {

template <typename T>
inline T sign_of(T x) noexcept {
  return x > T(0) ? T(1) : (x < T(0) ? T(-1) : T(0));
}

template <typename T>
inline void sgd_update(T* __restrict p, const T* __restrict g, T* velocity, std::size_t n,
                       SgdCoeffs c) {
  const T lr = static_cast<T>(c.lr);
  const T wd = static_cast<T>(c.weight_decay);
  const T mu = static_cast<T>(c.momentum);
  const T scale = static_cast<T>(1.0 - static_cast<double>(c.dampening));
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    T grad = g[i];
    if (c.weight_decay != 0.f) {
      grad += wd * p[i];
    }
    if (c.momentum != 0.f && velocity != nullptr) {
      velocity[i] = mu * velocity[i] + scale * grad;
      if (c.nesterov) {
        grad += mu * velocity[i];
      } else {
        grad = velocity[i];
      }
    }
    p[i] -= lr * grad;
  }
}

template <typename T>
inline void adam_update(T* __restrict p, const T* __restrict g, T* __restrict m, T* __restrict v,
                        T* v_max, std::size_t n, AdamCoeffs c) {
  const T lr = static_cast<T>(c.lr);
  const T b1 = static_cast<T>(c.beta1);
  const T ob1 = static_cast<T>(c.one_minus_beta1);
  const T b2 = static_cast<T>(c.beta2);
  const T ob2 = static_cast<T>(c.one_minus_beta2);
  const T eps = static_cast<T>(c.eps);
  const T wd = static_cast<T>(c.weight_decay);
  const T inv1 = static_cast<T>(c.inv_bc1);
  const T inv2 = static_cast<T>(c.inv_bc2);
  const T lr_wd = lr * wd;
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    T param = p[i];
    T grad = g[i];
    if (c.weight_decay != 0.f) {
      if (c.decoupled) {
        param -= lr_wd * param;
      } else {
        grad += wd * param;
      }
    }
    const T mt = b1 * m[i] + ob1 * grad;
    const T vt = b2 * v[i] + ob2 * grad * grad;
    m[i] = mt;
    v[i] = vt;
    T src = vt;
    if (c.amsgrad && v_max != nullptr) {
      src = std::max(v_max[i], vt);
      v_max[i] = src;
    }
    p[i] = param - lr * (mt * inv1) / (std::sqrt(src * inv2) + eps);
  }
}

template <typename T>
inline void adagrad_update(T* __restrict p, const T* __restrict g, T* __restrict state,
                           std::size_t n, AdagradCoeffs c) {
  const T clr =
      static_cast<T>(c.lr / (1.f + (c.step - 1.f) * c.lr_decay));
  const T eps = static_cast<T>(c.eps);
  const T wd = static_cast<T>(c.weight_decay);
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    const T grad = g[i] + wd * p[i];
    const T sum = state[i] + grad * grad;
    state[i] = sum;
    p[i] -= clr * grad / (std::sqrt(sum) + eps);
  }
}

template <typename T>
inline void rmsprop_update(T* __restrict p, const T* __restrict g, T* __restrict square, T* buf,
                           T* grad_avg, std::size_t n, RmspropCoeffs c) {
  const T lr = static_cast<T>(c.lr);
  const T alpha = static_cast<T>(c.alpha);
  const T oma = static_cast<T>(c.one_minus_alpha);
  const T eps = static_cast<T>(c.eps);
  const T wd = static_cast<T>(c.weight_decay);
  const T mu = static_cast<T>(c.momentum);
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    const T grad = g[i] + wd * p[i];
    const T sq = alpha * square[i] + oma * grad * grad;
    square[i] = sq;
    T avg;
    if (c.centered && grad_avg != nullptr) {
      const T ga = alpha * grad_avg[i] + oma * grad;
      grad_avg[i] = ga;
      const T centered = sq - ga * ga;
      avg = std::sqrt(centered > T(0) ? centered : T(0)) + eps;
    } else {
      avg = std::sqrt(sq) + eps;
    }
    if (c.momentum != 0.f && buf != nullptr) {
      buf[i] = mu * buf[i] + grad / avg;
      p[i] -= lr * buf[i];
    } else {
      p[i] -= lr * grad / avg;
    }
  }
}

template <typename T>
inline void adadelta_update(T* __restrict p, const T* __restrict g, T* __restrict square_avg,
                            T* __restrict acc_delta, std::size_t n, AdadeltaCoeffs c) {
  const T lr = static_cast<T>(c.lr);
  const T rho = static_cast<T>(c.rho);
  const T omr = static_cast<T>(c.one_minus_rho);
  const T eps = static_cast<T>(c.eps);
  const T wd = static_cast<T>(c.weight_decay);
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    const T grad = g[i] + wd * p[i];
    const T sq = rho * square_avg[i] + omr * grad * grad;
    square_avg[i] = sq;
    const T std = std::sqrt(sq + eps);
    const T delta = grad * std::sqrt(acc_delta[i] + eps) / std;
    acc_delta[i] = rho * acc_delta[i] + omr * delta * delta;
    p[i] -= lr * delta;
  }
}

template <typename T>
inline void nadam_update(T* __restrict p, const T* __restrict g, T* __restrict m, T* __restrict v,
                         std::size_t n, NadamCoeffs c) {
  const T lr = static_cast<T>(c.lr);
  const T b1 = static_cast<T>(c.beta1);
  const T ob1 = static_cast<T>(c.one_minus_beta1);
  const T b2 = static_cast<T>(c.beta2);
  const T ob2 = static_cast<T>(c.one_minus_beta2);
  const T eps = static_cast<T>(c.eps);
  const T wd = static_cast<T>(c.weight_decay);
  const T decay = static_cast<T>(1.0 - static_cast<double>(c.lr) * c.weight_decay);
  const double mu_product_next = static_cast<double>(c.mu_product) * c.mu_next;
  const T step_g =
      static_cast<T>(-static_cast<double>(c.lr) * (1.0 - c.mu) / (1.0 - c.mu_product));
  const T step_m = static_cast<T>(-static_cast<double>(c.lr) * c.mu_next / (1.0 - mu_product_next));
  const T bc2 = static_cast<T>(c.bias_correction2 > 0.f ? c.bias_correction2 : 1.f);
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    T param = p[i];
    T grad = g[i];
    if (c.weight_decay != 0.f) {
      if (c.decoupled) {
        param *= decay;
      } else {
        grad += wd * param;
      }
    }
    const T mt = b1 * m[i] + ob1 * grad;
    const T vt = b2 * v[i] + ob2 * grad * grad;
    m[i] = mt;
    v[i] = vt;
    const T denom = std::sqrt(vt / bc2) + eps;
    param += (step_g * grad + step_m * mt) / denom;
    p[i] = param;
  }
  (void)lr;
}

template <typename T>
inline void radam_update(T* __restrict p, const T* __restrict g, T* __restrict m, T* __restrict v,
                         std::size_t n, RadamCoeffs c) {
  const T lr = static_cast<T>(c.lr);
  const T b1 = static_cast<T>(c.beta1);
  const T ob1 = static_cast<T>(c.one_minus_beta1);
  const T b2 = static_cast<T>(c.beta2);
  const T ob2 = static_cast<T>(c.one_minus_beta2);
  const T eps = static_cast<T>(c.eps);
  const T wd = static_cast<T>(c.weight_decay);
  const T decay = static_cast<T>(1.0 - static_cast<double>(c.lr) * c.weight_decay);
  const T adapt = static_cast<T>(c.adaptive_step);
  const T plain = static_cast<T>(static_cast<double>(c.lr) * c.inv_bc1);
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    T param = p[i];
    T grad = g[i];
    if (c.weight_decay != 0.f) {
      if (c.decoupled) {
        param *= decay;
      } else {
        grad += wd * param;
      }
    }
    const T mt = b1 * m[i] + ob1 * grad;
    const T vt = b2 * v[i] + ob2 * grad * grad;
    m[i] = mt;
    v[i] = vt;
    if (c.rectified) {
      param -= adapt * mt / (std::sqrt(vt) + eps);
    } else {
      param -= plain * mt;
    }
    p[i] = param;
  }
  (void)lr;
}

template <typename T>
inline void adabelief_update(T* __restrict p, const T* __restrict g, T* __restrict m, T* __restrict s,
                             std::size_t n, AdaBeliefCoeffs c) {
  const T lr = static_cast<T>(c.lr);
  const T b1 = static_cast<T>(c.beta1);
  const T ob1 = static_cast<T>(c.one_minus_beta1);
  const T b2 = static_cast<T>(c.beta2);
  const T ob2 = static_cast<T>(c.one_minus_beta2);
  const T eps = static_cast<T>(c.eps);
  const T wd = static_cast<T>(c.weight_decay);
  const T decay = static_cast<T>(1.0 - static_cast<double>(c.lr) * c.weight_decay);
  const T inv1 = static_cast<T>(c.inv_bc1);
  const T inv2 = static_cast<T>(c.inv_bc2);
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    T param = p[i];
    T grad = g[i];
    if (c.weight_decay != 0.f) {
      if (c.decoupled) {
        param *= decay;
      } else {
        grad += wd * param;
      }
    }
    const T mt = b1 * m[i] + ob1 * grad;
    const T diff = grad - mt;
    const T st = b2 * s[i] + ob2 * diff * diff + eps;
    m[i] = mt;
    s[i] = st;
    p[i] = param - lr * (mt * inv1) / (std::sqrt(st * inv2) + eps);
  }
}

template <typename T>
inline void lamb_moments(T* __restrict p, const T* __restrict g, T* __restrict m, T* __restrict v,
                         T* __restrict u, std::size_t n, LambCoeffs c, double& param_sq,
                         double& update_sq) {
  const T b1 = static_cast<T>(c.beta1);
  const T ob1 = static_cast<T>(c.one_minus_beta1);
  const T b2 = static_cast<T>(c.beta2);
  const T ob2 = static_cast<T>(c.one_minus_beta2);
  const T eps = static_cast<T>(c.eps);
  const T wd = static_cast<T>(c.weight_decay);
  const T inv1 = static_cast<T>(c.inv_bc1);
  const T inv2 = static_cast<T>(c.inv_bc2);
  double p2 = 0.0;
  double u2 = 0.0;
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    const T grad = g[i];
    const T mt = b1 * m[i] + ob1 * grad;
    const T vt = b2 * v[i] + ob2 * grad * grad;
    m[i] = mt;
    v[i] = vt;
    const T adam = (mt * inv1) / (std::sqrt(vt * inv2) + eps);
    const T upd = adam + wd * p[i];
    u[i] = upd;
    const double pd = static_cast<double>(p[i]);
    const double ud = static_cast<double>(upd);
    p2 += pd * pd;
    u2 += ud * ud;
  }
  param_sq = p2;
  update_sq = u2;
}

template <typename T>
inline void lamb_apply(T* __restrict p, const T* __restrict u, std::size_t n, float lr, float trust) {
  const T scale = static_cast<T>(static_cast<double>(lr) * trust);
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    p[i] -= scale * u[i];
  }
}

template <typename T>
inline void lars_update(T* __restrict p, const T* __restrict g, T* velocity, std::size_t n,
                        LarsCoeffs c) {
  double p2 = 0.0;
  double g2 = 0.0;
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    const double pd = static_cast<double>(p[i]);
    const double gd = static_cast<double>(g[i]);
    p2 += pd * pd;
    g2 += gd * gd;
  }
  const double p_norm = std::sqrt(p2);
  const double g_norm = std::sqrt(g2);
  double trust = 1.0;
  if (p_norm > 0.0 && g_norm > 0.0) {
    trust = static_cast<double>(c.eta) * p_norm /
            (g_norm + static_cast<double>(c.weight_decay) * p_norm + c.eps);
  }
  const T local = static_cast<T>(static_cast<double>(c.lr) * trust);
  const T wd = static_cast<T>(c.weight_decay);
  const T mu = static_cast<T>(c.momentum);
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    const T scaled = g[i] + wd * p[i];
    if (c.momentum != 0.f && velocity != nullptr) {
      velocity[i] = mu * velocity[i] + scaled;
      p[i] -= local * velocity[i];
    } else {
      p[i] -= local * scaled;
    }
  }
}

template <typename T>
inline void lion_update(T* __restrict p, const T* __restrict g, T* __restrict m, std::size_t n,
                        LionCoeffs c) {
  const T lr = static_cast<T>(c.lr);
  const T b1 = static_cast<T>(c.beta1);
  const T ob1 = static_cast<T>(c.one_minus_beta1);
  const T b2 = static_cast<T>(c.beta2);
  const T ob2 = static_cast<T>(c.one_minus_beta2);
  const T decay = static_cast<T>(1.0 - static_cast<double>(c.lr) * c.weight_decay);
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    const T grad = g[i];
    const T blended = b1 * m[i] + ob1 * grad;
    T param = p[i];
    if (c.weight_decay != 0.f) {
      param *= decay;
    }
    p[i] = param - lr * sign_of(blended);
    m[i] = b2 * m[i] + ob2 * grad;
  }
}

template <typename T>
inline double l2_sq(const T* x, std::size_t n) {
  double sum = 0.0;
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    const double v = static_cast<double>(x[i]);
    sum += v * v;
  }
  return sum;
}

template <typename T>
inline double linf(const T* x, std::size_t n) {
  double m = 0.0;
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(static_cast<double>(x[i])));
  }
  return m;
}

template <typename T>
inline void scale_inplace(T* x, std::size_t n, T s) {
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    x[i] *= s;
  }
}

template <typename T>
inline void clip_abs(T* x, std::size_t n, T clip) {
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    const T v = x[i];
    x[i] = v > clip ? clip : (v < -clip ? -clip : v);
  }
}

template <typename T>
inline void lookahead_blend(T* __restrict fast, T* __restrict slow, std::size_t n, float alpha) {
  const T a = static_cast<T>(alpha);
  const T one_minus = static_cast<T>(1.0 - static_cast<double>(alpha));
  NEXUS_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    slow[i] = one_minus * slow[i] + a * fast[i];
    fast[i] = slow[i];
  }
}

template <typename T>
inline void centralize(T* g, std::size_t channels, std::size_t spatial) {
  if (spatial <= 1 || channels == 0) {
    return;
  }
  const double inv = 1.0 / static_cast<double>(spatial);
  for (std::size_t c = 0; c < channels; ++c) {
    T* row = g + c * spatial;
    double sum = 0.0;
    for (std::size_t i = 0; i < spatial; ++i) {
      sum += static_cast<double>(row[i]);
    }
    const T mean = static_cast<T>(sum * inv);
    for (std::size_t i = 0; i < spatial; ++i) {
      row[i] -= mean;
    }
  }
}

}  // namespace scalar
}  // namespace nexus_optim
