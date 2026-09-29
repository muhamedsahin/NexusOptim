#pragma once

/// @file radam.hpp
/// Rectified Adam (Liu et al.). The adaptive step is used only once the SMA
/// estimate ρ_t exceeds 5; before that the update is bias-corrected SGD.
///   ρ_inf = 2/(1-β2) - 1
///   ρ_t = ρ_inf - 2 t β2^t / (1-β2^t)
///   r_t = sqrt( (ρ_t-4)(ρ_t-2)ρ_inf / ((ρ_inf-4)(ρ_inf-2)ρ_t) )

#include "nexus_optim/core/optimizer_base.hpp"
#include "nexus_optim/core/state_store.hpp"
#include "nexus_optim/simd/simd_dispatch.hpp"

#if defined(NEXUS_OPTIM_WITH_CUDA)
#include "nexus_optim/cuda/cuda_launch.hpp"
#endif

#include <cmath>
#include <type_traits>
#include <vector>

namespace nexus_optim {

template <typename ScalarT = float>
class RAdam : public OptimizerBase<RAdam<ScalarT>, ScalarT> {
 public:
  struct Options {
    double lr = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 0.0;
    bool decoupled_weight_decay = false;
  };

  RAdam(std::vector<ParamGroup<ScalarT>> groups, Options opts)
      : OptimizerBase<RAdam<ScalarT>, ScalarT>(std::move(groups)), opts_(opts) {
    exp_avg_ = StateStore<ScalarT>::from_groups(this->groups_);
    exp_avg_sq_ = StateStore<ScalarT>::from_groups(this->groups_);
  }

  void step_impl() {
    ++this->step_;
    const double t = static_cast<double>(this->step_);
    const double bc1 = 1.0 - std::pow(opts_.beta1, t);
    const double beta2_t = std::pow(opts_.beta2, t);
    const double bc2 = 1.0 - beta2_t;
    const double rho_inf = 2.0 / (1.0 - opts_.beta2) - 1.0;
    const double rho_t = rho_inf - 2.0 * t * beta2_t / bc2;
    RadamCoeffs coeffs;
    coeffs.beta1 = static_cast<float>(opts_.beta1);
    coeffs.one_minus_beta1 = static_cast<float>(1.0 - opts_.beta1);
    coeffs.beta2 = static_cast<float>(opts_.beta2);
    coeffs.one_minus_beta2 = static_cast<float>(1.0 - opts_.beta2);
    coeffs.eps = static_cast<float>(opts_.eps);
    coeffs.inv_bc1 = static_cast<float>(1.0 / bc1);
    coeffs.decoupled = opts_.decoupled_weight_decay ? 1 : 0;
    if (rho_t > 5.0) {
      const double rect = std::sqrt(((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) /
                                    ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t));
      coeffs.rectified = 1;
      coeffs.adaptive_step = static_cast<float>(opts_.lr * rect * std::sqrt(bc2) / bc1);
    } else {
      coeffs.rectified = 0;
      coeffs.adaptive_step = 0.f;
    }
    const double rect =
        rho_t > 5.0 ? std::sqrt(((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) /
                                ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                    : 0.0;
    std::size_t slot = 0;
    for (auto& group : this->groups_) {
      coeffs.lr = static_cast<float>(group.options.learning_rate);
      coeffs.weight_decay =
          static_cast<float>(this->group_weight_decay(group, opts_.weight_decay));
      if (rho_t > 5.0) {
        coeffs.adaptive_step =
            static_cast<float>(group.options.learning_rate * rect * std::sqrt(bc2) / bc1);
      }
      for (std::size_t i = 0; i < group.params.size(); ++i, ++slot) {
        if (group.numels[i] == 0 || group.params[i] == nullptr) continue;
        if (group.devices[i] == Device::CUDA) {
#if defined(NEXUS_OPTIM_WITH_CUDA)
          if constexpr (std::is_same_v<ScalarT, float>) {
            cuda::radam_update(group.params[i], group.grads[i], exp_avg_.ptr(slot),
                               exp_avg_sq_.ptr(slot), group.numels[i], coeffs);
            continue;
          } else {
            throw std::runtime_error("RAdam CUDA kernel requires FP32 storage");
          }
#else
          throw std::runtime_error("RAdam CUDA kernel is not enabled; pass host tensors");
#endif
        }
        if constexpr (std::is_same_v<ScalarT, float>) {
          radam_update(group.params[i], group.grads[i], exp_avg_.ptr(slot),
                       exp_avg_sq_.ptr(slot), group.numels[i], coeffs);
        } else {
          scalar::radam_update(group.params[i], group.grads[i], exp_avg_.ptr(slot),
                               exp_avg_sq_.ptr(slot), group.numels[i], coeffs);
        }
      }
    }
  }

  OptimizerState state_dict_impl() const {
    OptimizerState state;
    state.algorithm = "radam";
    state.step = this->step_;
    this->capture_group_lrs(state);
    exp_avg_.save(state);
    exp_avg_sq_.save(state);
    return state;
  }

  void load_state_dict_impl(const OptimizerState& state) {
    this->step_ = state.step;
    this->restore_group_lrs(state);
    std::size_t cursor = 0;
    exp_avg_.load(state, cursor);
    exp_avg_sq_.load(state, cursor);
  }

 private:
  Options opts_;
  StateStore<ScalarT> exp_avg_;
  StateStore<ScalarT> exp_avg_sq_;
};

}  // namespace nexus_optim
