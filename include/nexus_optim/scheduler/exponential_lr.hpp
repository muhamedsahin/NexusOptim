#pragma once

#include "nexus_optim/scheduler/lr_scheduler_base.hpp"

#include <cmath>

namespace nexus_optim {

class ExponentialLR : public LRSchedulerBase {
 public:
  template <typename Optim>
  ExponentialLR(Optim& optimizer, double gamma)
      : LRSchedulerBase(bind_learning_rates(optimizer)), gamma_(gamma) {}

  void step(std::optional<double> metric = std::nullopt) override {
    (void)metric;
    ++last_epoch_;
    const double scale = std::pow(gamma_, static_cast<double>(last_epoch_));
    std::vector<double> next(base_.size());
    for (std::size_t i = 0; i < base_.size(); ++i) next[i] = base_[i] * scale;
    apply(next);
  }

 private:
  double gamma_;
};

}  // namespace nexus_optim
