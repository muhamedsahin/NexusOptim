#pragma once

/// @file adam.hpp
/// Adam and AdamW. AdamW is not a copy of Adam: both call the same fused kernel
/// with `decoupled` selecting where weight decay is applied.
///
/// Adam (Kingma & Ba):
///   g <- g + λ p                         (coupled weight decay)
///   m <- β1 m + (1-β1) g
///   v <- β2 v + (1-β2) g²
///   p <- p - γ m̂ / (sqrt(v̂) + ε)
///   m̂ = m / (1-β1^t),  v̂ = v / (1-β2^t)
///
/// AdamW (Loshchilov & Hutter):
///   p <- p - γ λ p                       (decoupled, moments see raw g)
///   then the same moment update as Adam.
/// AMSGrad tracks the max of v (Reddi et al.).

#include "nexus_optim/core/half.hpp"
#include "nexus_optim/core/optimizer_base.hpp"
#include "nexus_optim/core/state_store.hpp"
#include "nexus_optim/simd/simd_dispatch.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(NEXUS_OPTIM_WITH_CUDA)
#include "nexus_optim/cuda/cuda_launch.hpp"
#endif

namespace nexus_optim {

template <typename ScalarT, bool Decoupled>
class AdamImpl : public OptimizerBase<AdamImpl<ScalarT, Decoupled>, ScalarT> {
 public:
  struct Options {
    double lr = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 0.0;
    bool amsgrad = false;
  };

  AdamImpl(std::vector<ParamGroup<ScalarT>> groups, Options opts)
      : OptimizerBase<AdamImpl<ScalarT, Decoupled>, ScalarT>(std::move(groups)), opts_(opts) {
    exp_avg_ = StateStore<compute_type>::from_groups(this->groups_);
    exp_avg_sq_ = StateStore<compute_type>::from_groups(this->groups_);
    if (opts_.amsgrad) {
      max_exp_avg_sq_ = StateStore<compute_type>::from_groups(this->groups_);
    }
    init_master();
  }

  void step_impl() {
    ++this->step_;
    const double bc1 = 1.0 - std::pow(opts_.beta1, static_cast<double>(this->step_));
    const double bc2 = 1.0 - std::pow(opts_.beta2, static_cast<double>(this->step_));
    AdamCoeffs coeffs;
    coeffs.beta1 = static_cast<float>(opts_.beta1);
    coeffs.one_minus_beta1 = static_cast<float>(1.0 - opts_.beta1);
    coeffs.beta2 = static_cast<float>(opts_.beta2);
    coeffs.one_minus_beta2 = static_cast<float>(1.0 - opts_.beta2);
    coeffs.eps = static_cast<float>(opts_.eps);
    coeffs.inv_bc1 = static_cast<float>(1.0 / bc1);
    coeffs.inv_bc2 = static_cast<float>(1.0 / bc2);
    coeffs.decoupled = Decoupled ? 1 : 0;
    coeffs.amsgrad = opts_.amsgrad ? 1 : 0;
    std::size_t slot = 0;
    for (auto& group : this->groups_) {
      coeffs.lr = static_cast<float>(group.options.learning_rate);
      coeffs.weight_decay =
          static_cast<float>(this->group_weight_decay(group, opts_.weight_decay));
      for (std::size_t i = 0; i < group.params.size(); ++i, ++slot) {
        if (group.numels[i] == 0 || group.params[i] == nullptr || group.grads[i] == nullptr) {
          continue;
        }
        compute_type* vmax = opts_.amsgrad ? max_exp_avg_sq_.ptr(slot) : nullptr;
        launch(group.params[i], group.grads[i], exp_avg_.ptr(slot), exp_avg_sq_.ptr(slot), vmax, group.numels[i], group.devices[i], coeffs, slot);
      }
    }
  }

  OptimizerState state_dict_impl() const {
    OptimizerState state;
    state.algorithm = Decoupled ? "adamw" : "adam";
    state.step = this->step_;
    this->capture_group_lrs(state);
    exp_avg_.save(state);
    exp_avg_sq_.save(state);
    max_exp_avg_sq_.save(state);
    for (const auto& buffer : master_) state_push_buffer(state, buffer);
    return state;
  }

  void load_state_dict_impl(const OptimizerState& state) {
    if ((state.algorithm == "adamw") != Decoupled && !state.algorithm.empty()) {
      throw std::runtime_error("optimizer state algorithm does not match Adam/AdamW");
    }
    this->step_ = state.step;
    this->restore_group_lrs(state);
    std::size_t cursor = 0;
    exp_avg_.load(state, cursor);
    exp_avg_sq_.load(state, cursor);
    max_exp_avg_sq_.load(state, cursor);
    for (auto& buffer : master_) state_load_buffer(state, cursor, buffer);
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
          for (std::size_t k = 0; k < group.numels[i]; ++k) {
            master_[slot].data()[k] = group.params[i][k].to_float();
          }
        }
      }
    }
  }

  void launch(ScalarT* params, const ScalarT* grads, compute_type* m, compute_type* v,
              compute_type* vmax, std::size_t n, Device device, AdamCoeffs coeffs,
              std::size_t slot) {
    if (device == Device::CUDA) {
#if defined(NEXUS_OPTIM_WITH_CUDA)
      if constexpr (std::is_same_v<ScalarT, float>) {
        cuda::adam_update(params, grads, m, v, vmax, n, coeffs);
        return;
      } else {
        throw std::runtime_error("CUDA Adam requires FP32 storage");
      }
#else
      throw std::runtime_error("CUDA Adam requires -DNEXUS_OPTIM_WITH_CUDA=ON");
#endif
    }
    if constexpr (std::is_same_v<ScalarT, float> || std::is_same_v<ScalarT, double>) {
      adam_update(params, grads, m, v, vmax, n, coeffs);
    } else {
      float* cast = grad_cast_[slot].data();
      for (std::size_t k = 0; k < n; ++k) {
        cast[k] = grads[k].to_float();
      }
      adam_update(master_[slot].data(), cast, m, v, vmax, n, coeffs);
      for (std::size_t k = 0; k < n; ++k) {
        params[k] = ScalarT::from_float(master_[slot].data()[k]);
      }
    }
  }

  Options opts_;
  StateStore<compute_type> exp_avg_;
  StateStore<compute_type> exp_avg_sq_;
  StateStore<compute_type> max_exp_avg_sq_;
  std::vector<AlignedBuffer<float>> master_;
  std::vector<AlignedBuffer<float>> grad_cast_;
};

template <typename ScalarT = float>
using Adam = AdamImpl<ScalarT, false>;

template <typename ScalarT = float>
using AdamW = AdamImpl<ScalarT, true>;

}  // namespace nexus_optim
