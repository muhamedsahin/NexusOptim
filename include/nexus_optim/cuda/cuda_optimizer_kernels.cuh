#pragma once

#include "nexus_optim/simd/kernel_config.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace nexus_optim {
namespace cuda {

__device__ __forceinline__ float device_sign(float x) {
  return x > 0.f ? 1.f : (x < 0.f ? -1.f : 0.f);
}
__device__ __forceinline__ float4 ld4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ void st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }

__device__ void reduce_atomic(double* dst, double value) {
  constexpr int kWarp = 32;
  __shared__ double warp_sums[32];
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x >> 5;
  const unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    const int warp_count = (blockDim.x + kWarp - 1) >> 5;
    double block = (lane < warp_count) ? warp_sums[lane] : 0.0;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      block += __shfl_down_sync(mask, block, offset);
    }
    if (lane == 0) {
      atomicAdd(dst, block);
    }
  }
}

template <bool Nesterov>
__global__ void sgd_vec4_kernel(float* __restrict__ params, const float* __restrict__ grads,
                                float* __restrict__ velocity, std::size_t n4, float lr, float wd,
                                float momentum, float damp) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const float4 lr4 = make_float4(lr, lr, lr, lr);
  const float4 wd4 = make_float4(wd, wd, wd, wd);
  const float4 mu4 = make_float4(momentum, momentum, momentum, momentum);
  const float4 d4 = make_float4(damp, damp, damp, damp);
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
       i += stride) {
    const std::size_t o = i * 4;
    float4 p = ld4(params + o);
    float4 g = ld4(grads + o);
    g.x += wd4.x * p.x;
    g.y += wd4.y * p.y;
    g.z += wd4.z * p.z;
    g.w += wd4.w * p.w;
    if (velocity != nullptr && momentum != 0.f) {
      float4 v = ld4(velocity + o);
      v.x = mu4.x * v.x + d4.x * g.x;
      v.y = mu4.y * v.y + d4.y * g.y;
      v.z = mu4.z * v.z + d4.z * g.z;
      v.w = mu4.w * v.w + d4.w * g.w;
      st4(velocity + o, v);
      if constexpr (Nesterov) {
        g.x += mu4.x * v.x;
        g.y += mu4.y * v.y;
        g.z += mu4.z * v.z;
        g.w += mu4.w * v.w;
      } else {
        g = v;
      }
    }
    p.x -= lr4.x * g.x;
    p.y -= lr4.y * g.y;
    p.z -= lr4.z * g.z;
    p.w -= lr4.w * g.w;
    st4(params + o, p);
  }
}

__global__ void sgd_scalar_kernel(float* __restrict__ params, const float* __restrict__ grads,
                                  float* __restrict__ velocity, std::size_t n, float lr, float wd,
                                  float momentum, float damp, int nesterov) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    float grad = grads[i] + wd * params[i];
    if (momentum != 0.f && velocity != nullptr) {
      velocity[i] = momentum * velocity[i] + damp * grad;
      grad = nesterov ? (grad + momentum * velocity[i]) : velocity[i];
    }
    params[i] -= lr * grad;
  }
}

template <bool Decoupled, bool Amsgrad>
__global__ void adam_vec4_kernel(float* __restrict__ params, const float* __restrict__ grads,
                                 float* __restrict__ m, float* __restrict__ v, float* __restrict__ vmax,
                                 std::size_t n4, float lr, float b1, float ob1, float b2, float ob2,
                                 float eps, float wd, float inv1, float inv2) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
       i += stride) {
    const std::size_t o = i * 4;
    float4 p = ld4(params + o);
    float4 g = ld4(grads + o);
    float4 mt = ld4(m + o);
    float4 vt = ld4(v + o);
    if constexpr (Decoupled) {
      const float decay = lr * wd;
      p.x -= decay * p.x;
      p.y -= decay * p.y;
      p.z -= decay * p.z;
      p.w -= decay * p.w;
    } else {
      g.x += wd * p.x;
      g.y += wd * p.y;
      g.z += wd * p.z;
      g.w += wd * p.w;
    }
    mt.x = b1 * mt.x + ob1 * g.x;
    mt.y = b1 * mt.y + ob1 * g.y;
    mt.z = b1 * mt.z + ob1 * g.z;
    mt.w = b1 * mt.w + ob1 * g.w;
    vt.x = b2 * vt.x + ob2 * g.x * g.x;
    vt.y = b2 * vt.y + ob2 * g.y * g.y;
    vt.z = b2 * vt.z + ob2 * g.z * g.z;
    vt.w = b2 * vt.w + ob2 * g.w * g.w;
    st4(m + o, mt);
    st4(v + o, vt);
    float4 src = vt;
    if constexpr (Amsgrad) {
      float4 vm = ld4(vmax + o);
      src.x = src.x > vm.x ? src.x : vm.x;
      src.y = src.y > vm.y ? src.y : vm.y;
      src.z = src.z > vm.z ? src.z : vm.z;
      src.w = src.w > vm.w ? src.w : vm.w;
      st4(vmax + o, src);
    }
    p.x -= lr * (mt.x * inv1) / (sqrtf(src.x * inv2) + eps);
    p.y -= lr * (mt.y * inv1) / (sqrtf(src.y * inv2) + eps);
    p.z -= lr * (mt.z * inv1) / (sqrtf(src.z * inv2) + eps);
    p.w -= lr * (mt.w * inv1) / (sqrtf(src.w * inv2) + eps);
    st4(params + o, p);
  }
}

__global__ void adam_scalar_kernel(float* __restrict__ params, const float* __restrict__ grads,
                                   float* __restrict__ exp_avg, float* __restrict__ exp_avg_sq,
                                   float* __restrict__ max_exp_avg_sq, float lr, float beta1,
                                   float beta2, float eps, float weight_decay, float inv_bc1,
                                   float inv_bc2, int decoupled, int amsgrad, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    float param = params[i];
    float grad = grads[i];
    if (weight_decay != 0.f) {
      if (decoupled) {
        param -= lr * weight_decay * param;
      } else {
        grad += weight_decay * param;
      }
    }
    const float mt = beta1 * exp_avg[i] + (1.f - beta1) * grad;
    const float vt = beta2 * exp_avg_sq[i] + (1.f - beta2) * grad * grad;
    exp_avg[i] = mt;
    exp_avg_sq[i] = vt;
    float src = vt;
    if (amsgrad && max_exp_avg_sq != nullptr) {
      src = src > max_exp_avg_sq[i] ? src : max_exp_avg_sq[i];
      max_exp_avg_sq[i] = src;
    }
    params[i] = param - lr * (mt * inv_bc1) / (sqrtf(src * inv_bc2) + eps);
  }
}

__global__ void adagrad_kernel(float* __restrict__ p, const float* __restrict__ g,
                               float* __restrict__ state, std::size_t n, float lr, float eps,
                               float wd) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const float grad = g[i] + wd * p[i];
    const float sum = state[i] + grad * grad;
    state[i] = sum;
    p[i] -= lr * grad / (sqrtf(sum) + eps);
  }
}

__global__ void rmsprop_kernel(float* __restrict__ p, const float* __restrict__ g,
                               float* __restrict__ square, float* __restrict__ buf,
                               float* __restrict__ grad_avg, std::size_t n, float lr, float alpha,
                               float one_minus, float eps, float wd, float momentum, int centered) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const float grad = g[i] + wd * p[i];
    const float sq = alpha * square[i] + one_minus * grad * grad;
    square[i] = sq;
    float avg;
    if (centered && grad_avg != nullptr) {
      const float ga = alpha * grad_avg[i] + one_minus * grad;
      grad_avg[i] = ga;
      const float c = sq - ga * ga;
      avg = sqrtf(c > 0.f ? c : 0.f) + eps;
    } else {
      avg = sqrtf(sq) + eps;
    }
    if (momentum != 0.f && buf != nullptr) {
      buf[i] = momentum * buf[i] + grad / avg;
      p[i] -= lr * buf[i];
    } else {
      p[i] -= lr * grad / avg;
    }
  }
}

__global__ void adadelta_kernel(float* __restrict__ p, const float* __restrict__ g,
                                float* __restrict__ square, float* __restrict__ acc, std::size_t n,
                                float lr, float rho, float one_minus, float eps, float wd) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const float grad = g[i] + wd * p[i];
    const float sq = rho * square[i] + one_minus * grad * grad;
    square[i] = sq;
    const float std = sqrtf(sq + eps);
    const float delta = grad * sqrtf(acc[i] + eps) / std;
    acc[i] = rho * acc[i] + one_minus * delta * delta;
    p[i] -= lr * delta;
  }
}

__global__ void nadam_kernel(float* __restrict__ p, const float* __restrict__ g,
                             float* __restrict__ m, float* __restrict__ v, std::size_t n, float lr,
                             float b1, float ob1, float b2, float ob2, float eps, float wd,
                             float mu, float step_g, float step_m, float inv_bc2, int decoupled) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const float decay = 1.f - lr * wd;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    float param = p[i];
    float grad = g[i];
    if (wd != 0.f) {
      if (decoupled) {
        param *= decay;
      } else {
        grad += wd * param;
      }
    }
    const float mt = b1 * m[i] + ob1 * grad;
    const float vt = b2 * v[i] + ob2 * grad * grad;
    m[i] = mt;
    v[i] = vt;
    const float denom = sqrtf(vt * inv_bc2) + eps;
    p[i] = param + (step_g * grad + step_m * mt) / denom;
  }
}

__global__ void radam_kernel(float* __restrict__ p, const float* __restrict__ g,
                             float* __restrict__ m, float* __restrict__ v, std::size_t n, float lr,
                             float b1, float ob1, float b2, float ob2, float eps, float wd,
                             float plain, float adapt, int decoupled, int rectified) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const float decay = 1.f - lr * wd;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    float param = p[i];
    float grad = g[i];
    if (wd != 0.f) {
      if (decoupled) {
        param *= decay;
      } else {
        grad += wd * param;
      }
    }
    const float mt = b1 * m[i] + ob1 * grad;
    const float vt = b2 * v[i] + ob2 * grad * grad;
    m[i] = mt;
    v[i] = vt;
    if (rectified) {
      param -= adapt * mt / (sqrtf(vt) + eps);
    } else {
      param -= plain * mt;
    }
    p[i] = param;
  }
}

__global__ void adabelief_kernel(float* __restrict__ p, const float* __restrict__ g,
                                 float* __restrict__ m, float* __restrict__ s, std::size_t n,
                                 float lr, float b1, float ob1, float b2, float ob2, float eps,
                                 float wd, float inv1, float inv2, int decoupled) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const float decay = 1.f - lr * wd;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    float param = p[i];
    float grad = g[i];
    if (wd != 0.f) {
      if (decoupled) {
        param *= decay;
      } else {
        grad += wd * param;
      }
    }
    const float mt = b1 * m[i] + ob1 * grad;
    const float diff = grad - mt;
    const float st = b2 * s[i] + ob2 * diff * diff + eps;
    m[i] = mt;
    s[i] = st;
    p[i] = param - lr * (mt * inv1) / (sqrtf(st * inv2) + eps);
  }
}

__global__ void lamb_moments_kernel(float* __restrict__ p, const float* __restrict__ g,
                                    float* __restrict__ m, float* __restrict__ v,
                                    float* __restrict__ u, std::size_t n, float b1, float ob1,
                                    float b2, float ob2, float eps, float wd, float inv1, float inv2,
                                    double* __restrict__ param_sq, double* __restrict__ update_sq) {
  double p2 = 0.0;
  double u2 = 0.0;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const float grad = g[i];
    const float mt = b1 * m[i] + ob1 * grad;
    const float vt = b2 * v[i] + ob2 * grad * grad;
    m[i] = mt;
    v[i] = vt;
    const float adam = (mt * inv1) / (sqrtf(vt * inv2) + eps);
    const float upd = adam + wd * p[i];
    u[i] = upd;
    p2 += static_cast<double>(p[i]) * static_cast<double>(p[i]);
    u2 += static_cast<double>(upd) * static_cast<double>(upd);
  }
  reduce_atomic(param_sq, p2);
  reduce_atomic(update_sq, u2);
}

__global__ void lamb_apply_kernel(float* __restrict__ p, const float* __restrict__ u, std::size_t n,
                                  float lr, const double* __restrict__ param_sq,
                                  const double* __restrict__ update_sq) {
  const double pn = sqrt(*param_sq);
  const double un = sqrt(*update_sq);
  float trust = 1.f;
  if (pn > 0.0 && un > 0.0) {
    trust = static_cast<float>(pn / un);
  }
  const float scale = lr * trust;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    p[i] -= scale * u[i];
  }
}

__global__ void lars_norm_kernel(const float* __restrict__ p, const float* __restrict__ g,
                                 std::size_t n, double* __restrict__ param_sq,
                                 double* __restrict__ grad_sq) {
  double p2 = 0.0;
  double g2 = 0.0;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const double pd = static_cast<double>(p[i]);
    const double gd = static_cast<double>(g[i]);
    p2 += pd * pd;
    g2 += gd * gd;
  }
  reduce_atomic(param_sq, p2);
  reduce_atomic(grad_sq, g2);
}

__global__ void lars_apply_kernel(float* __restrict__ p, const float* __restrict__ g,
                                  float* __restrict__ velocity, std::size_t n, float lr, float wd,
                                  float momentum, float eta, float eps, const double* __restrict__ p2,
                                  const double* __restrict__ g2) {
  const float p_norm = sqrtf(static_cast<float>(*p2));
  const float g_norm = sqrtf(static_cast<float>(*g2));
  float trust = 1.f;
  if (p_norm > 0.f && g_norm > 0.f) {
    trust = eta * p_norm / (g_norm + wd * p_norm + eps);
  }
  const float local = lr * trust;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    float scaled = g[i] + wd * p[i];
    if (momentum != 0.f && velocity != nullptr) {
      velocity[i] = momentum * velocity[i] + scaled;
      scaled = velocity[i];
    }
    p[i] -= local * scaled;
  }
}

__global__ void lion_vec4_kernel(float* __restrict__ p, const float* __restrict__ g,
                                 float* __restrict__ m, std::size_t n4, float lr, float b1,
                                 float ob1, float b2, float ob2, float decay, int apply_decay) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
       i += stride) {
    const std::size_t o = i * 4;
    float4 grad = ld4(g + o);
    float4 prev = ld4(m + o);
    float4 blended;
    blended.x = b1 * prev.x + ob1 * grad.x;
    blended.y = b1 * prev.y + ob1 * grad.y;
    blended.z = b1 * prev.z + ob1 * grad.z;
    blended.w = b1 * prev.w + ob1 * grad.w;
    float4 param = ld4(p + o);
    if (apply_decay) {
      param.x *= decay;
      param.y *= decay;
      param.z *= decay;
      param.w *= decay;
    }
    param.x -= lr * device_sign(blended.x);
    param.y -= lr * device_sign(blended.y);
    param.z -= lr * device_sign(blended.z);
    param.w -= lr * device_sign(blended.w);
    st4(p + o, param);
    prev.x = b2 * prev.x + ob2 * grad.x;
    prev.y = b2 * prev.y + ob2 * grad.y;
    prev.z = b2 * prev.z + ob2 * grad.z;
    prev.w = b2 * prev.w + ob2 * grad.w;
    st4(m + o, prev);
  }
}

__global__ void lion_scalar_kernel(float* __restrict__ p, const float* __restrict__ g,
                                   float* __restrict__ m, std::size_t n, float lr, float b1,
                                   float ob1, float b2, float ob2, float decay, int apply_decay) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const float grad = g[i];
    const float blended = b1 * m[i] + ob1 * grad;
    float param = p[i];
    if (apply_decay) {
      param *= decay;
    }
    const float sign = blended > 0.f ? 1.f : (blended < 0.f ? -1.f : 0.f);
    p[i] = param - lr * sign;
    m[i] = b2 * m[i] + ob2 * grad;
  }
}

inline int launch_blocks(std::size_t work, int threads) {
  if (work == 0) {
    return 1;
  }
  int sm = 1;
  cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
  if (sm < 1) {
    sm = 1;
  }
  const std::size_t need = (work + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads);
  const std::size_t cap = static_cast<std::size_t>(sm) * 32u;
  const std::size_t blocks = need < cap ? need : cap;
  return static_cast<int>(blocks < 1 ? 1 : blocks);
}

}  // namespace cuda
}  // namespace nexus_optim
