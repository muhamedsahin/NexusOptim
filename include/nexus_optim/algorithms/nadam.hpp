#pragma once

/// @file nadam.hpp
/// NAdam as implemented by PyTorch 2.4 (Dozat). μ_t = β1 (1 - 0.5 * 0.96^(t ψ)),
/// the first moment is a Nesterov combination of m_t and g_t, and the second
/// moment is bias-corrected inside the square root.

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
class NAdam : public OptimizerBase<NAdam<ScalarT>, ScalarT> {
 public:
  struct Options {
    double lr = 2e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 0.0;
    double momentum_decay = 4e-3;
    bool decoupled_weight_decay = false;
  };

  NAdam(std::vector<ParamGroup<ScalarT>> groups, Options opts)
      : OptimizerBase<NAdam<ScalarT>, ScalarT>(std::move(groups)), opts_(opts) {
    exp_avg_ = StateStore<ScalarT>::from_groups(this->groups_);
    exp_avg_sq_ = StateStore<ScalarT>::from_groups(this->groups_);
  }

  void step_impl() {
    ++this->step_;
    const double t = static_cast<double>(this->step_);
    const double mu = opts_.beta1 * (1.0 - 0.5 * std::pow(0.96, t * opts_.momentum_decay));
    const double mu_next =
        opts_.beta1 * (1.0 - 0.5 * std::pow(0.96, (t + 1.0) * opts_.momentum_decay));
    mu_product_ *= mu;
    NadamCoeffs coeffs;
    coeffs.beta1 = static_cast<float>(opts_.beta1);
    coeffs.one_minus_beta1 = static_cast<float>(1.0 - opts_.beta1);
    coeffs.beta2 = static_cast<float>(opts_.beta2);
    coeffs.one_minus_beta2 = static_cast<float>(1.0 - opts_.beta2);
    coeffs.eps = static_cast<float>(opts_.eps);
    coeffs.mu = static_cast<float>(mu);
    coeffs.mu_next = static_cast<float>(mu_next);
    coeffs.mu_product = static_cast<float>(mu_product_);
    coeffs.bias_correction2 = static_cast<float>(1.0 - std::pow(opts_.beta2, t));
    coeffs.decoupled = opts_.decoupled_weight_decay ? 1 : 0;
    std::size_t slot = 0;
    for (auto& group : this->groups_) {
      coeffs.lr = static_cast<float>(group.options.learning_rate);
      coeffs.weight_decay =
          static_cast<float>(this->group_weight_decay(group, opts_.weight_decay));
      for (std::size_t i = 0; i < group.params.size(); ++i, ++slot) {
        if (group.numels[i] == 0 || group.params[i] == nullptr) continue;
        if (group.devices[i] == Device::CUDA) {
#if defined(NEXUS_OPTIM_WITH_CUDA)
          if constexpr (std::is_same_v<ScalarT, float>) {
            cuda::nadam_update(group.params[i], group.grads[i], exp_avg_.ptr(slot),
                               exp_avg_sq_.ptr(slot), group.numels[i], coeffs);
            continue;
          } else {
            throw std::runtime_error("NAdam CUDA kernel requires FP32 storage");
          }
#else
          throw std::runtime_error("NAdam CUDA kernel is not enabled; pass host tensors");
#endif
        }
        if constexpr (std::is_same_v<ScalarT, float>) {
          nadam_update(group.params[i], group.grads[i], exp_avg_.ptr(slot),
                       exp_avg_sq_.ptr(slot), group.numels[i], coeffs);
        } else {
          scalar::nadam_update(group.params[i], group.grads[i], exp_avg_.ptr(slot),
                               exp_avg_sq_.ptr(slot), group.numels[i], coeffs);
        }
      }
    }
  }

  OptimizerState state_dict_impl() const {
    OptimizerState state;
    state.algorithm = "nadam";
    state.step = this->step_;
    state.scalars = {mu_product_};
    this->capture_group_lrs(state);
    exp_avg_.save(state);
    exp_avg_sq_.save(state);
    return state;
  }

  void load_state_dict_impl(const OptimizerState& state) {
    this->step_ = state.step;
    if (!state.scalars.empty()) mu_product_ = state.scalars[0];
    this->restore_group_lrs(state);
    std::size_t cursor = 0;
    exp_avg_.load(state, cursor);
    exp_avg_sq_.load(state, cursor);
  }

 private:
  Options opts_;
  double mu_product_ = 1.0;
  StateStore<ScalarT> exp_avg_;
  StateStore<ScalarT> exp_avg_sq_;
};

}  // namespace nexus_optim
