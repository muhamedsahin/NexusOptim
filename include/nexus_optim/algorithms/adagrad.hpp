#pragma once

/// @file adagrad.hpp
/// Adagrad: s <- s + g²,  p <- p - γ_t g / (sqrt(s) + ε)
/// with optional learning-rate decay γ_t = γ / (1 + (t-1) lr_decay).

#include "nexus_optim/core/optimizer_base.hpp"
#include "nexus_optim/core/state_store.hpp"
#include "nexus_optim/simd/simd_dispatch.hpp"

#if defined(NEXUS_OPTIM_WITH_CUDA)
#include "nexus_optim/cuda/cuda_launch.hpp"
#endif

#include <stdexcept>
#include <type_traits>
#include <vector>

namespace nexus_optim {

template <typename ScalarT = float>
class Adagrad : public OptimizerBase<Adagrad<ScalarT>, ScalarT> {
 public:
  struct Options {
    double lr = 1e-2;
    double lr_decay = 0.0;
    double weight_decay = 0.0;
    double eps = 1e-10;
  };

  Adagrad(std::vector<ParamGroup<ScalarT>> groups, Options opts)
      : OptimizerBase<Adagrad<ScalarT>, ScalarT>(std::move(groups)), opts_(opts) {
    state_ = StateStore<ScalarT>::from_groups(this->groups_);
  }

  void step_impl() {
    ++this->step_;
    AdagradCoeffs coeffs;
    coeffs.lr = static_cast<float>(opts_.lr);
    coeffs.eps = static_cast<float>(opts_.eps);
    coeffs.lr_decay = static_cast<float>(opts_.lr_decay);
    coeffs.step = static_cast<float>(this->step_);
    std::size_t slot = 0;
    for (auto& group : this->groups_) {
      coeffs.lr = static_cast<float>(group.options.learning_rate);
      coeffs.weight_decay =
          static_cast<float>(this->group_weight_decay(group, opts_.weight_decay));
      for (std::size_t i = 0; i < group.params.size(); ++i, ++slot) {
        if (group.numels[i] == 0 || group.params[i] == nullptr || group.grads[i] == nullptr) {
          continue;
        }
        if (group.devices[i] == Device::CUDA) {
#if defined(NEXUS_OPTIM_WITH_CUDA)
          if constexpr (std::is_same_v<ScalarT, float>) {
            cuda::adagrad_update(group.params[i], group.grads[i], state_.ptr(slot), group.numels[i],
                                 coeffs);
            continue;
          } else {
            throw std::runtime_error("Adagrad CUDA kernel requires FP32 storage");
          }
#else
          throw std::runtime_error("Adagrad CUDA kernel is not enabled; pass host tensors");
#endif
        }
        if constexpr (std::is_same_v<ScalarT, float>) {
          adagrad_update(group.params[i], group.grads[i], state_.ptr(slot), group.numels[i], coeffs);
        } else {
          scalar::adagrad_update(group.params[i], group.grads[i], state_.ptr(slot), group.numels[i],
                                 coeffs);
        }
      }
    }
  }

  OptimizerState state_dict_impl() const {
    OptimizerState state;
    state.algorithm = "adagrad";
    state.step = this->step_;
    this->capture_group_lrs(state);
    state_.save(state);
    return state;
  }

  void load_state_dict_impl(const OptimizerState& state) {
    this->step_ = state.step;
    this->restore_group_lrs(state);
    std::size_t cursor = 0;
    state_.load(state, cursor);
  }

 private:
  Options opts_;
  StateStore<ScalarT> state_;
};

}  // namespace nexus_optim
