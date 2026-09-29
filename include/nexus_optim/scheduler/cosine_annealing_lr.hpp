#pragma once

#include "nexus_optim/scheduler/lr_scheduler_base.hpp"

#include <algorithm>
#include <cmath>

namespace nexus_optim {

class CosineAnnealingLR : public LRSchedulerBase {
 public:
  template <typename Optim>
  CosineAnnealingLR(Optim& optimizer, std::int64_t t_max, double eta_min = 0.0)
      : LRSchedulerBase(bind_learning_rates(optimizer)), t_max_(t_max), eta_min_(eta_min) {
    if (t_max_ < 1) {
      throw std::invalid_argument("CosineAnnealingLR T_max must be >= 1");
    }
  }

  void step(std::optional<double> metric = std::nullopt) override {
    (void)metric;
    ++last_epoch_;
    const double t = static_cast<double>(std::min(last_epoch_, t_max_));
    const double cosv = std::cos(3.14159265358979323846 * t / static_cast<double>(t_max_));
    std::vector<double> next(base_.size());
    for (std::size_t i = 0; i < base_.size(); ++i) {
      next[i] = eta_min_ + (base_[i] - eta_min_) * (1.0 + cosv) * 0.5;
    }
    apply(next);
  }

 private:
  std::int64_t t_max_;
  double eta_min_;
};

}  // namespace nexus_optim
