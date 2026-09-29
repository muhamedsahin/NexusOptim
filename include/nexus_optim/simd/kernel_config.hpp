#pragma once

/// Scalar coefficients computed once per `step()` (bias corrections, trust extras).
/// The inner kernels only see plain floats so they stay branch-light and allocation-free.

namespace nexus_optim {

struct SgdCoeffs {
  float lr = 0.f;
  float weight_decay = 0.f;
  float momentum = 0.f;
  float dampening = 0.f;
  int nesterov = 0;
};

struct AdamCoeffs {
  float lr = 0.f;
  float beta1 = 0.9f;
  float one_minus_beta1 = 0.1f;
  float beta2 = 0.999f;
  float one_minus_beta2 = 0.001f;
  float eps = 1e-8f;
  float weight_decay = 0.f;
  float inv_bc1 = 1.f;
  float inv_bc2 = 1.f;
  int decoupled = 0;
  int amsgrad = 0;
};

struct AdagradCoeffs {
  float lr = 0.f;
  float eps = 1e-10f;
  float weight_decay = 0.f;
  float lr_decay = 0.f;
  float step = 1.f;
};

struct RmspropCoeffs {
  float lr = 0.f;
  float alpha = 0.99f;
  float one_minus_alpha = 0.01f;
  float eps = 1e-8f;
  float weight_decay = 0.f;
  float momentum = 0.f;
  int centered = 0;
};

struct AdadeltaCoeffs {
  float lr = 1.f;
  float rho = 0.9f;
  float one_minus_rho = 0.1f;
  float eps = 1e-6f;
  float weight_decay = 0.f;
};

struct NadamCoeffs {
  float lr = 0.f;
  float beta1 = 0.9f;
  float one_minus_beta1 = 0.1f;
  float beta2 = 0.999f;
  float one_minus_beta2 = 0.001f;
  float eps = 1e-8f;
  float weight_decay = 0.f;
  float mu = 0.f;
  float mu_next = 0.f;
  float mu_product = 1.f;
  float bias_correction2 = 1.f;
  int decoupled = 0;
};

struct RadamCoeffs {
  float lr = 0.f;
  float beta1 = 0.9f;
  float one_minus_beta1 = 0.1f;
  float beta2 = 0.999f;
  float one_minus_beta2 = 0.001f;
  float eps = 1e-8f;
  float weight_decay = 0.f;
  float inv_bc1 = 1.f;
  /// When @c rectified is 0 the variance is still unusable and the update is SGD-like.
  float adaptive_step = 0.f;
  int decoupled = 0;
  int rectified = 0;
};

struct AdaBeliefCoeffs {
  float lr = 0.f;
  float beta1 = 0.9f;
  float one_minus_beta1 = 0.1f;
  float beta2 = 0.999f;
  float one_minus_beta2 = 0.001f;
  float eps = 1e-8f;
  float weight_decay = 0.f;
  float inv_bc1 = 1.f;
  float inv_bc2 = 1.f;
  int decoupled = 0;
};

struct LambCoeffs {
  float lr = 0.f;
  float beta1 = 0.9f;
  float one_minus_beta1 = 0.1f;
  float beta2 = 0.999f;
  float one_minus_beta2 = 0.001f;
  float eps = 1e-6f;
  float weight_decay = 0.f;
  float inv_bc1 = 1.f;
  float inv_bc2 = 1.f;
};

struct LarsCoeffs {
  float lr = 0.f;
  float momentum = 0.f;
  float weight_decay = 0.f;
  float eta = 1e-3f;
  float eps = 1e-8f;
};

struct LionCoeffs {
  float lr = 0.f;
  float beta1 = 0.9f;
  float one_minus_beta1 = 0.1f;
  float beta2 = 0.99f;
  float one_minus_beta2 = 0.01f;
  float weight_decay = 0.f;
};

}  // namespace nexus_optim
