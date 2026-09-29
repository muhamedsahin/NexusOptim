#pragma once

/// @file lamb.hpp
/// LAMB (You et al.) for large-batch training. The layer-wise trust ratio is
/// ||p|| / ||u|| where u is the Adam step plus decoupled weight decay.
/// Moments and the parameter write are two passes; the scratch buffer is
/// allocated in the constructor.

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
class LAMB : public OptimizerBase<LAMB<ScalarT>, ScalarT> {
 public:
  struct Options {
    double lr = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-6;
    double weight_decay = 0.0;
  };

  LAMB(std::vector<ParamGroup<ScalarT>> groups, Options opts)
      : OptimizerBase<LAMB<ScalarT>, ScalarT>(std::move(groups)), opts_(opts) {
    exp_avg_ = StateStore<ScalarT>::from_groups(this->groups_);
    exp_avg_sq_ = StateStore<ScalarT>::from_groups(this->groups_);
    scratch_ = StateStore<ScalarT>::from_groups(this->groups_);
  }

  void step_impl() {
    ++this->step_;
    const double bc1 = 1.0 - std::pow(opts_.beta1, static_cast<double>(this->step_));
    const double bc2 = 1.0 - std::pow(opts_.beta2, static_cast<double>(this->step_));
    LambCoeffs coeffs;
    coeffs.beta1 = static_cast<float>(opts_.beta1);
    coeffs.one_minus_beta1 = static_cast<float>(1.0 - opts_.beta1);
    coeffs.beta2 = static_cast<float>(opts_.beta2);
    coeffs.one_minus_beta2 = static_cast<float>(1.0 - opts_.beta2);
    coeffs.eps = static_cast<float>(opts_.eps);
    coeffs.inv_bc1 = static_cast<float>(1.0 / bc1);
    coeffs.inv_bc2 = static_cast<float>(1.0 / bc2);
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
            cuda::lamb_update(group.params[i], group.grads[i], exp_avg_.ptr(slot),
                              exp_avg_sq_.ptr(slot), scratch_.ptr(slot), group.numels[i], coeffs);
            continue;
          } else {
            throw std::runtime_error("LAMB CUDA kernel requires FP32 storage");
          }
#else
          throw std::runtime_error("LAMB CUDA kernel is not enabled; pass host tensors");
#endif
        }
        if constexpr (std::is_same_v<ScalarT, float>) {
          lamb_update(group.params[i], group.grads[i], exp_avg_.ptr(slot),
                      exp_avg_sq_.ptr(slot), scratch_.ptr(slot), group.numels[i], coeffs);
        } else {
          double p2 = 0.0;
          double u2 = 0.0;
          scalar::lamb_moments(group.params[i], group.grads[i], exp_avg_.ptr(slot),
                               exp_avg_sq_.ptr(slot), scratch_.ptr(slot), group.numels[i],
                               coeffs, p2, u2);
          const double pn = std::sqrt(p2);
          const double un = std::sqrt(u2);
          const float trust = (pn > 0.0 && un > 0.0) ? static_cast<float>(pn / un) : 1.f;
          scalar::lamb_apply(group.params[i], scratch_.ptr(slot), group.numels[i], coeffs.lr,
                             trust);
        }
      }
    }
  }

  OptimizerState state_dict_impl() const {
    OptimizerState state;
    state.algorithm = "lamb";
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
  StateStore<ScalarT> scratch_;
};

}  // namespace nexus_optim
