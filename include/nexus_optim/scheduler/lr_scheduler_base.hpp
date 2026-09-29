#pragma once

#include "nexus_optim/core/param_group.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace nexus_optim {

template <typename Optim>
std::vector<double*> bind_learning_rates(Optim& optimizer) {
  std::vector<double*> out;
  auto& groups = optimizer.param_groups();
  out.reserve(groups.size());
  for (auto& group : groups) {
    out.push_back(&group.options.learning_rate);
  }
  return out;
}

/// Cold-path base. Virtual dispatch is acceptable: a scheduler runs once per step,
/// not once per parameter.
class LRSchedulerBase {
 public:
  virtual ~LRSchedulerBase() = default;
  virtual void step(std::optional<double> metric = std::nullopt) = 0;
  virtual double get_last_lr() const { return last_.empty() ? 0.0 : last_.front(); }
  const std::vector<double>& get_last_lrs() const noexcept { return last_; }

 protected:
  explicit LRSchedulerBase(std::vector<double*> group_lrs) : lrs_(std::move(group_lrs)) {
    if (lrs_.empty()) {
      throw std::invalid_argument("scheduler requires at least one parameter group");
    }
    base_.reserve(lrs_.size());
    for (double* lr : lrs_) {
      base_.push_back(*lr);
    }
    last_ = base_;
  }

  void apply(const std::vector<double>& values) {
    last_ = values;
    for (std::size_t i = 0; i < lrs_.size(); ++i) {
      *lrs_[i] = values[i];
    }
  }

  std::vector<double*> lrs_;
  std::vector<double> base_;
  std::vector<double> last_;
  std::int64_t last_epoch_ = -1;
};

}  // namespace nexus_optim
