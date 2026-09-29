#pragma once

/// @file sgd.hpp
/// Plain SGD, momentum SGD, and Nesterov SGD.
///
/// PyTorch-compatible update (dampening defaults to 0):
///   g <- g + λ p                         if λ ≠ 0
///   v <- μ v + (1 - τ) g                 if μ ≠ 0
///   g <- g + μ v                         if Nesterov, else g <- v
///   p <- p - γ g

#include "nexus_optim/core/half.hpp"
#include "nexus_optim/core/optimizer_base.hpp"
#include "nexus_optim/core/state_store.hpp"
#include "nexus_optim/simd/simd_dispatch.hpp"

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(NEXUS_OPTIM_WITH_CUDA)
#include "nexus_optim/cuda/cuda_launch.hpp"
#endif

namespace nexus_optim {

enum class SgdVariant { Plain, Momentum, Nesterov };

template <typename ScalarT, SgdVariant Variant>
class SgdImpl : public OptimizerBase<SgdImpl<ScalarT, Variant>, ScalarT> {
 public:
  struct Options {
    double lr = 1e-3;
    double momentum = Variant == SgdVariant::Plain ? 0.0 : 0.9;
    double weight_decay = 0.0;
    double dampening = 0.0;
    bool nesterov = Variant == SgdVariant::Nesterov;
  };

  SgdImpl(std::vector<ParamGroup<ScalarT>> groups, Options opts)
      : OptimizerBase<SgdImpl<ScalarT, Variant>, ScalarT>(std::move(groups)), opts_(opts) {
    if constexpr (Variant != SgdVariant::Plain) {
      velocity_ = StateStore<compute_type>::from_groups(this->groups_);
    }
    init_master();
  }

  void step_impl() {
    ++this->step_;
    SgdCoeffs coeffs;
    coeffs.momentum = static_cast<float>(opts_.momentum);
    coeffs.dampening = static_cast<float>(opts_.dampening);
    coeffs.nesterov = opts_.nesterov ? 1 : 0;
    std::size_t slot = 0;
    for (auto& group : this->groups_) {
      coeffs.lr = static_cast<float>(group.options.learning_rate);
      coeffs.weight_decay =
          static_cast<float>(this->group_weight_decay(group, opts_.weight_decay));
      for (std::size_t i = 0; i < group.params.size(); ++i, ++slot) {
        if (group.numels[i] == 0 || group.params[i] == nullptr || group.grads[i] == nullptr) {
          continue;
        }
        compute_type* velocity = velocity_.empty() ? nullptr : velocity_.ptr(slot);
        launch(group.params[i], group.grads[i], velocity, group.numels[i], group.devices[i], coeffs,
               slot);
      }
    }
  }

  OptimizerState state_dict_impl() const {
    OptimizerState state;
    state.algorithm = "sgd";
    state.step = this->step_;
    state.scalars = {opts_.momentum, opts_.weight_decay};
    this->capture_group_lrs(state);
    velocity_.save(state);
    for (const auto& buffer : master_) {
      state_push_buffer(state, buffer);
    }
    return state;
  }

  void load_state_dict_impl(const OptimizerState& state) {
    this->step_ = state.step;
    this->restore_group_lrs(state);
    std::size_t cursor = 0;
    velocity_.load(state, cursor);
    for (auto& buffer : master_) {
      state_load_buffer(state, cursor, buffer);
    }
  }

 private:
  using compute_type = std::conditional_t<is_narrow_storage<ScalarT>::value, float, ScalarT>;

  void init_master() {
    if constexpr (is_narrow_storage<ScalarT>::value) {
      master_ = this->template allocate_slots<float>();
      grad_cast_ = this->template allocate_slots<float>();
      std::size_t slot = 0;
      for (const auto& group : this->groups_) {
        for (std::size_t i = 0; i < group.params.size(); ++i, ++slot) {
          float* dst = master_[slot].data();
          for (std::size_t k = 0; k < group.numels[i]; ++k) {
            dst[k] = group.params[i][k].to_float();
          }
        }
      }
    }
  }

  void launch(ScalarT* params, const ScalarT* grads, compute_type* velocity, std::size_t n,
              Device device, SgdCoeffs coeffs, std::size_t slot) {
    if (device == Device::CUDA) {
#if defined(NEXUS_OPTIM_WITH_CUDA)
      if constexpr (std::is_same_v<ScalarT, float>) {
        cuda::sgd_update(params, grads, velocity, n, coeffs);
        return;
      } else {
        throw std::runtime_error("CUDA SGD currently updates FP32 parameter storage");
      }
#else
      throw std::runtime_error("parameter is on CUDA but NexusOptim was built without CUDA");
#endif
    }
    if constexpr (std::is_same_v<ScalarT, float>) {
      sgd_update(params, grads, velocity, n, coeffs);
    } else if constexpr (std::is_same_v<ScalarT, double>) {
      sgd_update(params, grads, velocity, n, coeffs);
    } else {
      float* cast = grad_cast_[slot].data();
      for (std::size_t k = 0; k < n; ++k) {
        cast[k] = grads[k].to_float();
      }
      sgd_update(master_[slot].data(), cast, velocity, n, coeffs);
      for (std::size_t k = 0; k < n; ++k) {
        params[k] = ScalarT::from_float(master_[slot].data()[k]);
      }
    }
  }

  Options opts_;
  StateStore<compute_type> velocity_;
  std::vector<AlignedBuffer<float>> master_;
  std::vector<AlignedBuffer<float>> grad_cast_;
};

template <typename ScalarT = float>
using SGD = SgdImpl<ScalarT, SgdVariant::Plain>;

template <typename ScalarT = float>
using MomentumSGD = SgdImpl<ScalarT, SgdVariant::Momentum>;

template <typename ScalarT = float>
using NesterovSGD = SgdImpl<ScalarT, SgdVariant::Nesterov>;

}  // namespace nexus_optim
