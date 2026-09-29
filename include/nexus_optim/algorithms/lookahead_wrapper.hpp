#pragma once

/// @file lookahead_wrapper.hpp
/// Zhang et al. Lookahead. Every `k` inner steps the slow weights move toward
/// the fast weights: slow <- slow + α (fast - slow), fast <- slow.

#include "nexus_optim/core/aligned_buffer.hpp"
#include "nexus_optim/simd/simd_dispatch.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace nexus_optim {

template <typename Inner>
class Lookahead {
 public:
  using scalar_type = typename Inner::scalar_type;

  Lookahead(Inner inner, int k, float alpha)
      : inner_(std::move(inner)), k_(k), alpha_(alpha) {
    if (k_ < 1) {
      throw std::invalid_argument("Lookahead k must be >= 1");
    }
    if (!(alpha_ > 0.f && alpha_ <= 1.f)) {
      throw std::invalid_argument("Lookahead alpha must be in (0, 1]");
    }
    for (const auto& group : inner_.param_groups()) {
      for (std::size_t i = 0; i < group.params.size(); ++i) {
        slow_.emplace_back(group.numels[i]);
        scalar_type* dst = slow_.back().data();
        const scalar_type* src = group.params[i];
        for (std::size_t n = 0; n < group.numels[i]; ++n) {
          dst[n] = src[n];
        }
      }
    }
  }

  void step() {
    inner_.step();
    ++count_;
    if (count_ % k_ != 0) {
      return;
    }
    std::size_t slot = 0;
    for (auto& group : inner_.param_groups()) {
      for (std::size_t i = 0; i < group.params.size(); ++i, ++slot) {
        if (group.numels[i] == 0 || group.params[i] == nullptr) continue;
        if constexpr (std::is_same_v<scalar_type, float>) {
          lookahead_blend(group.params[i], slow_[slot].data(), group.numels[i], alpha_);
        } else {
          scalar::lookahead_blend(group.params[i], slow_[slot].data(), group.numels[i], alpha_);
        }
      }
    }
  }

  void zero_grad(bool set_to_none = true) { inner_.zero_grad(set_to_none); }

  Inner& base() noexcept { return inner_; }
  const Inner& base() const noexcept { return inner_; }

  auto& param_groups() noexcept { return inner_.param_groups(); }
  const auto& param_groups() const noexcept { return inner_.param_groups(); }

 private:
  Inner inner_;
  int k_ = 5;
  float alpha_ = 0.5f;
  int count_ = 0;
  std::vector<AlignedBuffer<scalar_type>> slow_;
};

}  // namespace nexus_optim
