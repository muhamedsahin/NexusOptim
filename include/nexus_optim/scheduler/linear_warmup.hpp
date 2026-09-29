#pragma once

#include "nexus_optim/scheduler/lr_scheduler_base.hpp"

#include <algorithm>

namespace nexus_optim {

/// Linear warmup: after epoch e the rate is base * min(1, (e+1) / warmup_steps).
class LinearWarmup : public LRSchedulerBase {
 public:
  template <typename Optim>
  explicit LinearWarmup(Optim& optimizer, std::int64_t warmup_steps)
      : LRSchedulerBase(bind_learning_rates(optimizer)), warmup_steps_(warmup_steps) {
    if (warmup_steps_ < 1) {
      throw std::invalid_argument("LinearWarmup warmup_steps must be >= 1");
    }
  }

  void step(std::optional<double> metric = std::nullopt) override {
    (void)metric;
    ++last_epoch_;
    const double scale = std::min(
        1.0, static_cast<double>(last_epoch_ + 1) / static_cast<double>(warmup_steps_));
    std::vector<double> next(base_.size());
    for (std::size_t i = 0; i < base_.size(); ++i) next[i] = base_[i] * scale;
    apply(next);
  }

 private:
  std::int64_t warmup_steps_;
};

}  // namespace nexus_optim
