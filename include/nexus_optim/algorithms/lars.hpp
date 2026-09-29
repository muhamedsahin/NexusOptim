#pragma once

/// @file lars.hpp
/// LARS layer-wise adaptive rate (You, Gitman, Ginsburg):
///   trust = η ||p|| / (||g|| + λ ||p|| + ε)
///   p <- p - γ * trust * (μ v + g + λ p)

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
class LARS : public OptimizerBase<LARS<ScalarT>, ScalarT> {
 public:
  struct Options {
    double lr = 1.0;
    double momentum = 0.9;
    double weight_decay = 0.0;
    double eta = 1e-3;
    double eps = 1e-8;
  };

  LARS(std::vector<ParamGroup<ScalarT>> groups, Options opts)
      : OptimizerBase<LARS<ScalarT>, ScalarT>(std::move(groups)), opts_(opts) {
    if (opts_.momentum != 0.0) velocity_ = StateStore<ScalarT>::from_groups(this->groups_);
  }

  void step_impl() {
    ++this->step_;
    LarsCoeffs coeffs;
    coeffs.momentum = static_cast<float>(opts_.momentum);
    coeffs.eta = static_cast<float>(opts_.eta);
    coeffs.eps = static_cast<float>(opts_.eps);
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
            cuda::lars_update(group.params[i], group.grads[i],
                              velocity_.empty() ? nullptr : velocity_.ptr(slot), group.numels[i],
                              coeffs);
            continue;
          } else {
            throw std::runtime_error("LARS CUDA kernel requires FP32 storage");
          }
#else
          throw std::runtime_error("LARS CUDA kernel is not enabled; pass host tensors");
#endif
        }
        ScalarT* velocity = velocity_.empty() ? nullptr : velocity_.ptr(slot);
        if constexpr (std::is_same_v<ScalarT, float>) {
          lars_update(group.params[i], group.grads[i], velocity, group.numels[i], coeffs);
        } else {
          scalar::lars_update(group.params[i], group.grads[i], velocity, group.numels[i], coeffs);
        }
      }
    }
  }

  OptimizerState state_dict_impl() const {
    OptimizerState state;
    state.algorithm = "lars";
    state.step = this->step_;
    this->capture_group_lrs(state);
    velocity_.save(state);
    return state;
  }

  void load_state_dict_impl(const OptimizerState& state) {
    this->step_ = state.step;
    this->restore_group_lrs(state);
    std::size_t cursor = 0;
    velocity_.load(state, cursor);
  }

 private:
  Options opts_;
  StateStore<ScalarT> velocity_;
};

}  // namespace nexus_optim
