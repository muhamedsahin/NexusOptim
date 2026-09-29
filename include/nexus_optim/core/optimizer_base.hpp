#pragma once

#include "nexus_optim/core/aligned_buffer.hpp"
#include "nexus_optim/core/optimizer_state.hpp"
#include "nexus_optim/core/param_group.hpp"
#include "nexus_optim/core/thread_pool.hpp"

#if defined(NEXUS_OPTIM_WITH_CUDA)
#include "nexus_optim/cuda/cuda_launch.hpp"
#endif

#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nexus_optim {

/// CRTP base. `step()` is a non-virtual inline that jumps straight into `Derived::step_impl`.
/// The only virtuals in the library live on learning-rate schedulers, which run once per step
/// rather than once per element.
template <typename Derived, typename ScalarT>
class OptimizerBase {
 public:
  using scalar_type = ScalarT;

  explicit OptimizerBase(std::vector<ParamGroup<ScalarT>> groups) : groups_(std::move(groups)) {
    if (groups_.empty()) {
      throw std::invalid_argument("optimizer requires at least one parameter group");
    }
    std::size_t slots = 0;
    for (auto& group : groups_) {
      group.validate();
      slots += group.params.size();
    }
    // Touching the pool here allocates workers once, off the hot path.
    (void)ThreadPool::instance();
    slot_count_ = slots;
  }

  OptimizerBase(const OptimizerBase&) = delete;
  OptimizerBase& operator=(const OptimizerBase&) = delete;
  OptimizerBase(OptimizerBase&&) noexcept = default;
  OptimizerBase& operator=(OptimizerBase&&) noexcept = default;
  ~OptimizerBase() = default;

  /// Hot path. Must not allocate and must not virtual-dispatch.
  inline void step() { static_cast<Derived*>(this)->step_impl(); }

  /// Zero borrowed gradient storage in place.
  ///
  /// PyTorch can drop the gradient tensor when `set_to_none` is true. These pointers are
  /// borrowed from the caller, so both modes write zeros: the caller still owns the buffer
  /// and the next backward pass accumulates into it. The flag is accepted for call-site
  /// compatibility.
  void zero_grad(bool set_to_none = true) {
    (void)set_to_none;
    for (auto& group : groups_) {
      for (std::size_t i = 0; i < group.grads.size(); ++i) {
      if (group.grads[i] == nullptr || group.numels[i] == 0) {
        continue;
      }
      if (group.devices[i] == Device::CPU) {
        std::memset(group.grads[i], 0, group.numels[i] * sizeof(ScalarT));
      } else {
#if defined(NEXUS_OPTIM_WITH_CUDA)
        cuda::zero(group.grads[i], group.numels[i] * sizeof(ScalarT));
#else
        throw std::runtime_error("zero_grad on a CUDA tensor requires -DNEXUS_OPTIM_WITH_CUDA=ON");
#endif
      }
      }
    }
  }

  OptimizerState state_dict() const { return static_cast<const Derived*>(this)->state_dict_impl(); }

  void load_state_dict(const OptimizerState& state) {
    static_cast<Derived*>(this)->load_state_dict_impl(state);
  }

  std::vector<ParamGroup<ScalarT>>& param_groups() noexcept { return groups_; }
  const std::vector<ParamGroup<ScalarT>>& param_groups() const noexcept { return groups_; }

  std::uint64_t step_count() const noexcept { return step_; }

 protected:
  template <typename Fn>
  void for_each_param(Fn&& fn) {
    std::size_t slot = 0;
    for (auto& group : groups_) {
      const std::size_t count = group.params.size();
      for (std::size_t i = 0; i < count; ++i, ++slot) {
        if (group.numels[i] == 0 || group.params[i] == nullptr || group.grads[i] == nullptr) {
          continue;
        }
        fn(group, i, slot);
      }
    }
  }

  template <typename Fn>
  void for_each_param(Fn&& fn) const {
    std::size_t slot = 0;
    for (const auto& group : groups_) {
      const std::size_t count = group.params.size();
      for (std::size_t i = 0; i < count; ++i, ++slot) {
        if (group.numels[i] == 0 || group.params[i] == nullptr || group.grads[i] == nullptr) {
          continue;
        }
        fn(group, i, slot);
      }
    }
  }

  template <typename StateT>
  std::vector<AlignedBuffer<StateT>> allocate_slots() const {
    std::vector<AlignedBuffer<StateT>> buffers;
    buffers.reserve(slot_count_);
    for (const auto& group : groups_) {
      for (std::size_t numel : group.numels) {
        buffers.emplace_back(numel);
      }
    }
    return buffers;
  }

  double group_weight_decay(const ParamGroup<ScalarT>& group, double optimizer_weight_decay) const {
    return group.options.weight_decay_override ? group.options.weight_decay
                                                : optimizer_weight_decay;
  }

  void restore_group_lrs(const OptimizerState& state) {
    if (!state.group_learning_rates.empty() &&
        state.group_learning_rates.size() != groups_.size()) {
      throw std::runtime_error("optimizer state group count does not match");
    }
    for (std::size_t i = 0; i < state.group_learning_rates.size(); ++i) {
      groups_[i].options.learning_rate = state.group_learning_rates[i];
    }
  }

  void capture_group_lrs(OptimizerState& state) const {
    state.group_learning_rates.resize(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
      state.group_learning_rates[i] = groups_[i].options.learning_rate;
    }
  }

  std::vector<ParamGroup<ScalarT>> groups_;
  std::uint64_t step_ = 0;
  std::size_t slot_count_ = 0;
};

}  // namespace nexus_optim
