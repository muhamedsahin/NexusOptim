#pragma once

#include "nexus_optim/scheduler/lr_scheduler_base.hpp"

#include <cmath>

namespace nexus_optim {

/// γ <- γ0 * factor^(epoch / step_size). Epoch 0 keeps the base learning rate.
class StepLR : public LRSchedulerBase {
 public:
  template <typename Optim>
  StepLR(Optim& optimizer, int step_size, double gamma = 0.1)
      : LRSchedulerBase(bind_learning_rates(optimizer)),
        step_size_(step_size),
        gamma_(gamma) {
    if (step_size_ < 1) {
      throw std::invalid_argument("StepLR step_size must be >= 1");
    }
  }

  void step(std::optional<double> metric = std::nullopt) override {
    (void)metric;
    ++last_epoch_;
    const double scale = std::pow(gamma_, static_cast<double>(last_epoch_ / step_size_));
    std::vector<double> next(base_.size());
    for (std::size_t i = 0; i < base_.size(); ++i) next[i] = base_[i] * scale;
    apply(next);
  }

 private:
  int step_size_;
  double gamma_;
};

}  // namespace nexus_optim
