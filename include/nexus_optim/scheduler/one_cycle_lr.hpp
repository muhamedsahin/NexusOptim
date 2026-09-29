#pragma once

#include "nexus_optim/scheduler/lr_scheduler_base.hpp"

#include <algorithm>
#include <cmath>

namespace nexus_optim {

/// One-cycle schedule (Smith). `max_lr` is the peak; the cycle starts at
/// max_lr / div_factor, climbs linearly for `pct_start` of the cycle, then
/// cosine-anneals to max_lr / (div_factor * final_div_factor).
class OneCycleLR : public LRSchedulerBase {
 public:
  template <typename Optim>
  OneCycleLR(Optim& optimizer, double max_lr, std::int64_t total_steps, double pct_start = 0.3,
             double div_factor = 25.0, double final_div_factor = 1e4)
      : LRSchedulerBase(bind_learning_rates(optimizer)),
        max_lr_(max_lr),
        total_steps_(total_steps),
        pct_start_(pct_start),
        initial_(max_lr / div_factor),
        final_(max_lr / (div_factor * final_div_factor)) {
    if (total_steps_ < 1) {
      throw std::invalid_argument("OneCycleLR total_steps must be >= 1");
    }
  }

  void step(std::optional<double> metric = std::nullopt) override {
    (void)metric;
    const double progress = std::min(1.0, static_cast<double>(++cycle_) / static_cast<double>(total_steps_));
    const double value = progress < pct_start_
                             ? lerp(initial_, max_lr_, progress / pct_start_)
                             : cosine(max_lr_, final_, (progress - pct_start_) / (1.0 - pct_start_));
    // Preserve relative group scales against the first group's base.
    const double ref = base_.front() == 0.0 ? 1.0 : base_.front();
    std::vector<double> next(base_.size());
    for (std::size_t i = 0; i < base_.size(); ++i) {
      next[i] = value * (base_[i] / ref);
    }
    apply(next);
    ++last_epoch_;
  }

 private:
  static double lerp(double a, double b, double t) { return a + (b - a) * t; }
  static double cosine(double a, double b, double t) {
    return b + (a - b) * (1.0 + std::cos(3.14159265358979323846 * t)) * 0.5;
  }

  double max_lr_;
  std::int64_t total_steps_;
  double pct_start_;
  double initial_;
  double final_;
  std::int64_t cycle_ = 0;
};

}  // namespace nexus_optim
