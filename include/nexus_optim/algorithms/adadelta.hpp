#pragma once

/// @file adadelta.hpp
/// Adadelta (Zeiler), with PyTorch's learning-rate scale (default γ = 1):
///   E[g²] <- ρ E[g²] + (1-ρ) g²
///   Δ <- g * sqrt(E[Δ²] + ε) / sqrt(E[g²] + ε)
///   E[Δ²] <- ρ E[Δ²] + (1-ρ) Δ²
///   p <- p - γ Δ

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
class Adadelta : public OptimizerBase<Adadelta<ScalarT>, ScalarT> {
 public:
  struct Options {
    double lr = 1.0;
    double rho = 0.9;
    double eps = 1e-6;
    double weight_decay = 0.0;
  };

  Adadelta(std::vector<ParamGroup<ScalarT>> groups, Options opts)
      : OptimizerBase<Adadelta<ScalarT>, ScalarT>(std::move(groups)), opts_(opts) {
    square_ = StateStore<ScalarT>::from_groups(this->groups_);
    acc_ = StateStore<ScalarT>::from_groups(this->groups_);
  }

  void step_impl() {
    ++this->step_;
    AdadeltaCoeffs coeffs;
    coeffs.rho = static_cast<float>(opts_.rho);
    coeffs.one_minus_rho = static_cast<float>(1.0 - opts_.rho);
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
            cuda::adadelta_update(group.params[i], group.grads[i], square_.ptr(slot), acc_.ptr(slot),
                                  group.numels[i], coeffs);
            continue;
          } else {
            throw std::runtime_error("Adadelta CUDA kernel requires FP32 storage");
          }
#else
          throw std::runtime_error("Adadelta CUDA kernel is not enabled; pass host tensors");
#endif
        }
        if constexpr (std::is_same_v<ScalarT, float>) {
          adadelta_update(group.params[i], group.grads[i], square_.ptr(slot), acc_.ptr(slot),
                          group.numels[i], coeffs);
        } else {
          scalar::adadelta_update(group.params[i], group.grads[i], square_.ptr(slot), acc_.ptr(slot),
                                  group.numels[i], coeffs);
        }
      }
    }
  }

  OptimizerState state_dict_impl() const {
    OptimizerState state;
    state.algorithm = "adadelta";
    state.step = this->step_;
    this->capture_group_lrs(state);
    square_.save(state);
    acc_.save(state);
    return state;
  }

  void load_state_dict_impl(const OptimizerState& state) {
    this->step_ = state.step;
    this->restore_group_lrs(state);
    std::size_t cursor = 0;
    square_.load(state, cursor);
    acc_.load(state, cursor);
  }

 private:
  Options opts_;
  StateStore<ScalarT> square_;
  StateStore<ScalarT> acc_;
};

}  // namespace nexus_optim
