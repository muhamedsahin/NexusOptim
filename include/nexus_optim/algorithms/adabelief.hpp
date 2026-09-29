#pragma once

/// @file adabelief.hpp
/// AdaBelief (Zhuang et al.): the second moment tracks the deviation of the
/// gradient from its predicted mean, plus ε inside the accumulator.
///   m <- β1 m + (1-β1) g
///   s <- β2 s + (1-β2) (g - m)² + ε
///   p <- p - γ m̂ / (sqrt(ŝ) + ε)

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
class AdaBelief : public OptimizerBase<AdaBelief<ScalarT>, ScalarT> {
 public:
  struct Options {
    double lr = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 0.0;
    bool decoupled_weight_decay = true;
  };

  AdaBelief(std::vector<ParamGroup<ScalarT>> groups, Options opts)
      : OptimizerBase<AdaBelief<ScalarT>, ScalarT>(std::move(groups)), opts_(opts) {
    exp_avg_ = StateStore<ScalarT>::from_groups(this->groups_);
    exp_avg_var_ = StateStore<ScalarT>::from_groups(this->groups_);
  }

  void step_impl() {
    ++this->step_;
    const double bc1 = 1.0 - std::pow(opts_.beta1, static_cast<double>(this->step_));
    const double bc2 = 1.0 - std::pow(opts_.beta2, static_cast<double>(this->step_));
    AdaBeliefCoeffs coeffs;
    coeffs.beta1 = static_cast<float>(opts_.beta1);
    coeffs.one_minus_beta1 = static_cast<float>(1.0 - opts_.beta1);
    coeffs.beta2 = static_cast<float>(opts_.beta2);
    coeffs.one_minus_beta2 = static_cast<float>(1.0 - opts_.beta2);
    coeffs.eps = static_cast<float>(opts_.eps);
    coeffs.inv_bc1 = static_cast<float>(1.0 / bc1);
    coeffs.inv_bc2 = static_cast<float>(1.0 / bc2);
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
            cuda::adabelief_update(group.params[i], group.grads[i], exp_avg_.ptr(slot),
                                   exp_avg_var_.ptr(slot), group.numels[i], coeffs);
            continue;
          } else {
            throw std::runtime_error("AdaBelief CUDA kernel requires FP32 storage");
          }
#else
          throw std::runtime_error("AdaBelief CUDA kernel is not enabled; pass host tensors");
#endif
        }
        if constexpr (std::is_same_v<ScalarT, float>) {
          adabelief_update(group.params[i], group.grads[i], exp_avg_.ptr(slot),
                           exp_avg_var_.ptr(slot), group.numels[i], coeffs);
        } else {
          scalar::adabelief_update(group.params[i], group.grads[i], exp_avg_.ptr(slot),
                                   exp_avg_var_.ptr(slot), group.numels[i], coeffs);
        }
      }
    }
  }

  OptimizerState state_dict_impl() const {
    OptimizerState state;
    state.algorithm = "adabelief";
    state.step = this->step_;
    this->capture_group_lrs(state);
    exp_avg_.save(state);
    exp_avg_var_.save(state);
    return state;
  }

  void load_state_dict_impl(const OptimizerState& state) {
    this->step_ = state.step;
    this->restore_group_lrs(state);
    std::size_t cursor = 0;
    exp_avg_.load(state, cursor);
    exp_avg_var_.load(state, cursor);
  }

 private:
  Options opts_;
  StateStore<ScalarT> exp_avg_;
  StateStore<ScalarT> exp_avg_var_;
};

}  // namespace nexus_optim
