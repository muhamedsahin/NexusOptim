#pragma once

#include "nexus_optim/simd/kernel_config.hpp"

#include <cstddef>

namespace nexus_optim {
namespace cuda {

bool enabled() noexcept;

void* device_alloc_bytes(std::size_t bytes);
void device_free(void* ptr) noexcept;
void copy_device_to_host(void* host, const void* device, std::size_t bytes);
void copy_host_to_device(void* device, const void* host, std::size_t bytes);
void zero(void* ptr, std::size_t bytes);

void sgd_update(float* params, const float* grads, float* velocity, std::size_t n, SgdCoeffs c);
void adam_update(float* params, const float* grads, float* exp_avg, float* exp_avg_sq,
                 float* max_exp_avg_sq, std::size_t n, AdamCoeffs c);
void adagrad_update(float* params, const float* grads, float* state, std::size_t n, AdagradCoeffs c);
void rmsprop_update(float* params, const float* grads, float* square, float* buf, float* grad_avg,
                    std::size_t n, RmspropCoeffs c);
void adadelta_update(float* params, const float* grads, float* square_avg, float* acc_delta,
                     std::size_t n, AdadeltaCoeffs c);
void nadam_update(float* params, const float* grads, float* exp_avg, float* exp_avg_sq, std::size_t n,
                  NadamCoeffs c);
void radam_update(float* params, const float* grads, float* exp_avg, float* exp_avg_sq, std::size_t n,
                  RadamCoeffs c);
void adabelief_update(float* params, const float* grads, float* exp_avg, float* exp_avg_var,
                      std::size_t n, AdaBeliefCoeffs c);
void lamb_update(float* params, const float* grads, float* exp_avg, float* exp_avg_sq, float* scratch,
                 std::size_t n, LambCoeffs c);
void lars_update(float* params, const float* grads, float* velocity, std::size_t n, LarsCoeffs c);
void lion_update(float* params, const float* grads, float* exp_avg, std::size_t n, LionCoeffs c);

int stream_count() noexcept;

}  // namespace cuda
}  // namespace nexus_optim
