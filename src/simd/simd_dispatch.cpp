#include "nexus_optim/simd/simd_dispatch.hpp"

#include "nexus_optim/core/thread_pool.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

namespace nexus_optim {
namespace {

std::atomic<int> g_override{-1};

#if defined(_MSC_VER) || defined(__i386__) || defined(__x86_64__)

void read_cpuid(int out[4], int leaf) {
#if defined(_MSC_VER)
  __cpuid(out, leaf);
#else
  __cpuid(leaf, out[0], out[1], out[2], out[3]);
#endif
}

bool os_supports_avx() {
  int cpu[4] = {};
  read_cpuid(cpu, 1);
  const bool osxsave = (cpu[2] & (1 << 27)) != 0;
  const bool avx = (cpu[2] & (1 << 28)) != 0;
  if (!osxsave || !avx) {
    return false;
  }
#if defined(_MSC_VER)
  const unsigned long long xcr = _xgetbv(0);
#else
  unsigned int eax = 0;
  unsigned int edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  const unsigned long long xcr = (static_cast<unsigned long long>(edx) << 32) | eax;
#endif
  return (xcr & 0x6ULL) == 0x6ULL;
}

bool os_supports_avx512() {
  if (!os_supports_avx()) {
    return false;
  }
#if defined(_MSC_VER)
  const unsigned long long xcr = _xgetbv(0);
#else
  unsigned int eax = 0;
  unsigned int edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  const unsigned long long xcr = (static_cast<unsigned long long>(edx) << 32) | eax;
#endif
  // XMM, YMM, Opmask, ZMM.
  return (xcr & 0xE6ULL) == 0xE6ULL;
}

SimdLevel detect() noexcept {
  int cpu[4] = {};
  read_cpuid(cpu, 0);
  const int max_leaf = cpu[0];
  read_cpuid(cpu, 1);
  const bool fma = (cpu[2] & (1 << 12)) != 0;
  bool avx2 = false;
  bool avx512 = false;
  if (max_leaf >= 7) {
    read_cpuid(cpu, 7);
    avx2 = (cpu[1] & (1 << 5)) != 0;
    avx512 = (cpu[1] & (1 << 16)) != 0;
  }
  if (avx512 && fma && os_supports_avx512()) {
#if defined(NEXUS_OPTIM_ENABLE_AVX512)
    return SimdLevel::Avx512;
#else
    (void)avx512;
#endif
  }
  if (avx2 && fma && os_supports_avx()) {
    return SimdLevel::Avx2;
  }
  return SimdLevel::Scalar;
}

#else

SimdLevel detect() noexcept { return neon::compiled() ? SimdLevel::Neon : SimdLevel::Scalar; }

#endif

const SimdLevel kDetected = detect();

void atomic_add(std::atomic<double>& slot, double value) {
  double current = slot.load(std::memory_order_relaxed);
  while (!slot.compare_exchange_weak(current, current + value, std::memory_order_relaxed)) {
  }
}

template <typename Fn>
void split(std::size_t n, Fn&& fn) {
  parallel_for(n, std::forward<Fn>(fn));
}

}  // namespace

SimdLevel detected_simd_level() noexcept { return kDetected; }

SimdLevel cpu_simd_level() noexcept {
  const int forced = g_override.load(std::memory_order_relaxed);
  if (forced >= 0) {
    return static_cast<SimdLevel>(forced);
  }
  return kDetected;
}

bool force_simd_level(std::optional<SimdLevel> level) noexcept {
  if (!level.has_value()) {
    g_override.store(-1, std::memory_order_relaxed);
    return true;
  }
  if (*level == SimdLevel::Scalar) {
    g_override.store(static_cast<int>(SimdLevel::Scalar), std::memory_order_relaxed);
    return true;
  }
  if (*level == SimdLevel::Avx2 &&
      (kDetected == SimdLevel::Avx2 || kDetected == SimdLevel::Avx512)) {
    g_override.store(static_cast<int>(SimdLevel::Avx2), std::memory_order_relaxed);
    return true;
  }
  if (*level == SimdLevel::Avx512 && kDetected == SimdLevel::Avx512) {
    g_override.store(static_cast<int>(SimdLevel::Avx512), std::memory_order_relaxed);
    return true;
  }
  if (*level == SimdLevel::Neon && kDetected == SimdLevel::Neon) {
    g_override.store(static_cast<int>(SimdLevel::Neon), std::memory_order_relaxed);
    return true;
  }
  return false;
}

void sgd_update(float* p, const float* g, float* velocity, std::size_t n, SgdCoeffs c) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    float* vel = velocity != nullptr ? velocity + b : nullptr;
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx512) {
      avx512::sgd_update(p + b, g + b, vel, len, c);
    } else if (level == SimdLevel::Avx2) {
      avx2::sgd_update(p + b, g + b, vel, len, c);
    } else if (level == SimdLevel::Neon) {
      neon::sgd_update(p + b, g + b, vel, len, c);
    } else {
      scalar::sgd_update(p + b, g + b, vel, len, c);
    }
  });
}

void sgd_update(double* p, const double* g, double* velocity, std::size_t n, SgdCoeffs c) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    double* vel = velocity != nullptr ? velocity + b : nullptr;
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
      avx2::sgd_update(p + b, g + b, vel, len, c);
    } else {
      scalar::sgd_update(p + b, g + b, vel, len, c);
    }
  });
}

void adam_update(float* p, const float* g, float* m, float* v, float* v_max, std::size_t n,
                 AdamCoeffs c) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    float* vm = v_max != nullptr ? v_max + b : nullptr;
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx512) {
      avx512::adam_update(p + b, g + b, m + b, v + b, vm, len, c);
    } else if (level == SimdLevel::Avx2) {
      avx2::adam_update(p + b, g + b, m + b, v + b, vm, len, c);
    } else if (level == SimdLevel::Neon) {
      neon::adam_update(p + b, g + b, m + b, v + b, vm, len, c);
    } else {
      scalar::adam_update(p + b, g + b, m + b, v + b, vm, len, c);
    }
  });
}

void adam_update(double* p, const double* g, double* m, double* v, double* v_max, std::size_t n,
                 AdamCoeffs c) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    double* vm = v_max != nullptr ? v_max + b : nullptr;
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
      avx2::adam_update(p + b, g + b, m + b, v + b, vm, len, c);
    } else {
      scalar::adam_update(p + b, g + b, m + b, v + b, vm, len, c);
    }
  });
}

void adagrad_update(float* p, const float* g, float* state, std::size_t n, AdagradCoeffs c) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
      avx2::adagrad_update(p + b, g + b, state + b, len, c);
    } else {
      scalar::adagrad_update(p + b, g + b, state + b, len, c);
    }
  });
}

void rmsprop_update(float* p, const float* g, float* square, float* buf, float* grad_avg,
                    std::size_t n, RmspropCoeffs c) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
      avx2::rmsprop_update(p + b, g + b, square + b, buf != nullptr ? buf + b : nullptr,
                           grad_avg != nullptr ? grad_avg + b : nullptr, len, c);
    } else {
      scalar::rmsprop_update(p + b, g + b, square + b, buf != nullptr ? buf + b : nullptr,
                             grad_avg != nullptr ? grad_avg + b : nullptr, len, c);
    }
  });
}

void adadelta_update(float* p, const float* g, float* square_avg, float* acc_delta, std::size_t n,
                     AdadeltaCoeffs c) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
      avx2::adadelta_update(p + b, g + b, square_avg + b, acc_delta + b, len, c);
    } else {
      scalar::adadelta_update(p + b, g + b, square_avg + b, acc_delta + b, len, c);
    }
  });
}

void nadam_update(float* p, const float* g, float* m, float* v, std::size_t n, NadamCoeffs c) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
      avx2::nadam_update(p + b, g + b, m + b, v + b, len, c);
    } else {
      scalar::nadam_update(p + b, g + b, m + b, v + b, len, c);
    }
  });
}

void radam_update(float* p, const float* g, float* m, float* v, std::size_t n, RadamCoeffs c) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
      avx2::radam_update(p + b, g + b, m + b, v + b, len, c);
    } else {
      scalar::radam_update(p + b, g + b, m + b, v + b, len, c);
    }
  });
}

void adabelief_update(float* p, const float* g, float* m, float* s, std::size_t n,
                      AdaBeliefCoeffs c) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
      avx2::adabelief_update(p + b, g + b, m + b, s + b, len, c);
    } else {
      scalar::adabelief_update(p + b, g + b, m + b, s + b, len, c);
    }
  });
}

void lamb_update(float* p, const float* g, float* m, float* v, float* scratch, std::size_t n,
                 LambCoeffs c) {
  const SimdLevel level = cpu_simd_level();
  std::atomic<double> p2{0.0};
  std::atomic<double> u2{0.0};
  split(n, [&](std::size_t b, std::size_t e) {
    double lp = 0.0;
    double lu = 0.0;
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
      avx2::lamb_moments(p + b, g + b, m + b, v + b, scratch + b, len, c, lp, lu);
    } else {
      scalar::lamb_moments(p + b, g + b, m + b, v + b, scratch + b, len, c, lp, lu);
    }
    atomic_add(p2, lp);
    atomic_add(u2, lu);
  });
  const double pn = std::sqrt(p2.load(std::memory_order_relaxed));
  const double un = std::sqrt(u2.load(std::memory_order_relaxed));
  float trust = 1.f;
  if (pn > 0.0 && un > 0.0) {
    trust = static_cast<float>(pn / un);
  }
  split(n, [&](std::size_t b, std::size_t e) {
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
      avx2::lamb_apply(p + b, scratch + b, len, c.lr, trust);
    } else {
      scalar::lamb_apply(p + b, scratch + b, len, c.lr, trust);
    }
  });
}

void lars_update(float* p, const float* g, float* velocity, std::size_t n, LarsCoeffs c) {
  const SimdLevel level = cpu_simd_level();
  if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
    avx2::lars_update(p, g, velocity, n, c);
  } else {
    scalar::lars_update(p, g, velocity, n, c);
  }
}

void lion_update(float* p, const float* g, float* m, std::size_t n, LionCoeffs c) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
      avx2::lion_update(p + b, g + b, m + b, len, c);
    } else {
      scalar::lion_update(p + b, g + b, m + b, len, c);
    }
  });
}

double l2_sq(const float* x, std::size_t n) {
  const SimdLevel level = cpu_simd_level();
  if (n < kParallelGrain) {
    if (level == SimdLevel::Avx512) {
      return avx512::l2_sq(x, n);
    }
    if (level == SimdLevel::Avx2) {
      return avx2::l2_sq(x, n);
    }
    return scalar::l2_sq(x, n);
  }
  std::atomic<double> sum{0.0};
  split(n, [&](std::size_t b, std::size_t e) {
    double part = 0.0;
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx512) {
      part = avx512::l2_sq(x + b, len);
    } else if (level == SimdLevel::Avx2) {
      part = avx2::l2_sq(x + b, len);
    } else {
      part = scalar::l2_sq(x + b, len);
    }
    atomic_add(sum, part);
  });
  return sum.load(std::memory_order_relaxed);
}

double l2_sq(const double* x, std::size_t n) { return scalar::l2_sq(x, n); }

double linf(const float* x, std::size_t n) {
  const SimdLevel level = cpu_simd_level();
  if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
    return avx2::linf(x, n);
  }
  return scalar::linf(x, n);
}

void scale_inplace(float* x, std::size_t n, float scale) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx512) {
      avx512::scale_inplace(x + b, len, scale);
    } else if (level == SimdLevel::Avx2) {
      avx2::scale_inplace(x + b, len, scale);
    } else {
      scalar::scale_inplace(x + b, len, scale);
    }
  });
}

void clip_abs(float* x, std::size_t n, float clip) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
      avx2::clip_abs(x + b, len, clip);
    } else {
      scalar::clip_abs(x + b, len, clip);
    }
  });
}

void lookahead_blend(float* fast, float* slow, std::size_t n, float alpha) {
  const SimdLevel level = cpu_simd_level();
  split(n, [&](std::size_t b, std::size_t e) {
    const std::size_t len = e - b;
    if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
      avx2::lookahead_blend(fast + b, slow + b, len, alpha);
    } else {
      scalar::lookahead_blend(fast + b, slow + b, len, alpha);
    }
  });
}

void centralize(float* grad, std::size_t channels, std::size_t spatial) {
  const SimdLevel level = cpu_simd_level();
  if (level == SimdLevel::Avx2 || level == SimdLevel::Avx512) {
    avx2::centralize(grad, channels, spatial);
  } else {
    scalar::centralize(grad, channels, spatial);
  }
}

}  // namespace nexus_optim
