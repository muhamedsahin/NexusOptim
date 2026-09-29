#pragma once

/// @file rmsprop.hpp
/// RMSProp: v <- α v + (1-α) g²,  p <- p - γ g / (sqrt(v) + ε)
/// Optional centered second moment and momentum buffer, matching PyTorch.

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
class RMSProp : public OptimizerBase<RMSProp<ScalarT>, ScalarT> {
 public:
  struct Options {
    double lr = 1e-2;
    double alpha = 0.99;
    double eps = 1e-8;
    double weight_decay = 0.0;
    double momentum = 0.0;
    bool centered = false;
  };

  RMSProp(std::vector<ParamGroup<ScalarT>> groups, Options opts)
      : OptimizerBase<RMSProp<ScalarT>, ScalarT>(std::move(groups)), opts_(opts) {
    square_ = StateStore<ScalarT>::from_groups(this->groups_);
    if (opts_.momentum != 0.0) buf_ = StateStore<ScalarT>::from_groups(this->groups_);
    if (opts_.centered) grad_avg_ = StateStore<ScalarT>::from_groups(this->groups_);
  }

  void step_impl() {
    ++this->step_;
    RmspropCoeffs coeffs;
    coeffs.alpha = static_cast<float>(opts_.alpha);
    coeffs.one_minus_alpha = static_cast<float>(1.0 - opts_.alpha);
    coeffs.eps = static_cast<float>(opts_.eps);
    coeffs.momentum = static_cast<float>(opts_.momentum);
    coeffs.centered = opts_.centered ? 1 : 0;
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
            cuda::rmsprop_update(group.params[i], group.grads[i], square_.ptr(slot),
                                 buf_.empty() ? nullptr : buf_.ptr(slot),
                                 grad_avg_.empty() ? nullptr : grad_avg_.ptr(slot), group.numels[i],
                                 coeffs);
            continue;
          } else {
            throw std::runtime_error("RMSProp CUDA kernel requires FP32 storage");
          }
#else
          throw std::runtime_error("RMSProp CUDA kernel is not enabled; pass host tensors");
#endif
        }
        ScalarT* buf = buf_.empty() ? nullptr : buf_.ptr(slot);
        ScalarT* ga = grad_avg_.empty() ? nullptr : grad_avg_.ptr(slot);
        if constexpr (std::is_same_v<ScalarT, float>) {
          rmsprop_update(group.params[i], group.grads[i], square_.ptr(slot), buf, ga,
                         group.numels[i], coeffs);
        } else {
          scalar::rmsprop_update(group.params[i], group.grads[i], square_.ptr(slot), buf, ga,
                                 group.numels[i], coeffs);
        }
      }
    }
  }

  OptimizerState state_dict_impl() const {
    OptimizerState state;
    state.algorithm = "rmsprop";
    state.step = this->step_;
    this->capture_group_lrs(state);
    square_.save(state);
    buf_.save(state);
    grad_avg_.save(state);
    return state;
  }

  void load_state_dict_impl(const OptimizerState& state) {
    this->step_ = state.step;
    this->restore_group_lrs(state);
    std::size_t cursor = 0;
    square_.load(state, cursor);
    buf_.load(state, cursor);
    grad_avg_.load(state, cursor);
  }

 private:
  Options opts_;
  StateStore<ScalarT> square_;
  StateStore<ScalarT> buf_;
  StateStore<ScalarT> grad_avg_;
};

}  // namespace nexus_optim
