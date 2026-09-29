#pragma once

/// @file lion.hpp
/// Lion (Chen et al.): sign momentum, one buffer instead of Adam's two.
///   c <- β1 m + (1-β1) g
///   p <- (1 - γ λ) p - γ sign(c)
///   m <- β2 m + (1-β2) g

#include "nexus_optim/core/optimizer_base.hpp"
#include "nexus_optim/core/state_store.hpp"
#include "nexus_optim/simd/simd_dispatch.hpp"

#if defined(NEXUS_OPTIM_WITH_CUDA)
#include "nexus_optim/cuda/cuda_launch.hpp"
#endif

#include <type_traits>
#include <vector>

namespace nexus_optim {

template <typename ScalarT = float>
class Lion : public OptimizerBase<Lion<ScalarT>, ScalarT> {
 public:
  struct Options {
    double lr = 1e-4;
    double beta1 = 0.9;
    double beta2 = 0.99;
    double weight_decay = 0.0;
  };

  Lion(std::vector<ParamGroup<ScalarT>> groups, Options opts)
      : OptimizerBase<Lion<ScalarT>, ScalarT>(std::move(groups)), opts_(opts) {
    exp_avg_ = StateStore<ScalarT>::from_groups(this->groups_);
  }

  void step_impl() {
    ++this->step_;
    LionCoeffs coeffs;
    coeffs.beta1 = static_cast<float>(opts_.beta1);
    coeffs.one_minus_beta1 = static_cast<float>(1.0 - opts_.beta1);
    coeffs.beta2 = static_cast<float>(opts_.beta2);
    coeffs.one_minus_beta2 = static_cast<float>(1.0 - opts_.beta2);
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
            cuda::lion_update(group.params[i], group.grads[i], exp_avg_.ptr(slot), group.numels[i],
                              coeffs);
            continue;
          } else {
            throw std::runtime_error("Lion CUDA kernel requires FP32 storage");
          }
#else
          throw std::runtime_error("Lion CUDA kernel is not enabled; pass host tensors");
#endif
        }
        if constexpr (std::is_same_v<ScalarT, float>) {
          lion_update(group.params[i], group.grads[i], exp_avg_.ptr(slot), group.numels[i],
                      coeffs);
        } else {
          scalar::lion_update(group.params[i], group.grads[i], exp_avg_.ptr(slot),
                              group.numels[i], coeffs);
        }
      }
    }
  }

  OptimizerState state_dict_impl() const {
    OptimizerState state;
    state.algorithm = "lion";
    state.step = this->step_;
    this->capture_group_lrs(state);
    exp_avg_.save(state);
    return state;
  }

  void load_state_dict_impl(const OptimizerState& state) {
    this->step_ = state.step;
    this->restore_group_lrs(state);
    std::size_t cursor = 0;
    exp_avg_.load(state, cursor);
  }

 private:
  Options opts_;
  StateStore<ScalarT> exp_avg_;
};

}  // namespace nexus_optim
