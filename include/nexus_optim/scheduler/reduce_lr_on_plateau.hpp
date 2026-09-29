#pragma once

#include "nexus_optim/scheduler/lr_scheduler_base.hpp"

#include <cmath>
#include <optional>

namespace nexus_optim {

/// Drops the learning rate by `factor` when `metric` stops improving for
/// `patience` observations. `mode` is "min" (lower is better).
class ReduceLROnPlateau : public LRSchedulerBase {
 public:
  template <typename Optim>
  ReduceLROnPlateau(Optim& optimizer, double factor = 0.1, int patience = 10,
                    double threshold = 1e-4, int cooldown = 0, double min_lr = 0.0)
      : LRSchedulerBase(bind_learning_rates(optimizer)),
        factor_(factor),
        patience_(patience),
        threshold_(threshold),
        cooldown_(cooldown),
        min_lr_(min_lr) {
    if (!(factor_ > 0.0 && factor_ < 1.0)) {
      throw std::invalid_argument("ReduceLROnPlateau factor must be in (0, 1)");
    }
  }

  void step(std::optional<double> metric = std::nullopt) override {
    if (!metric.has_value()) {
      throw std::invalid_argument("ReduceLROnPlateau::step requires a metric");
    }
    ++last_epoch_;
    if (cooldown_left_ > 0) {
      --cooldown_left_;
      return;
    }
    const double value = *metric;
    const bool improved = !has_best_ || value < best_ * (1.0 - threshold_);
    if (improved) {
      best_ = value;
      has_best_ = true;
      bad_epochs_ = 0;
      return;
    }
    ++bad_epochs_;
    if (bad_epochs_ > patience_) {
      std::vector<double> next(last_.size());
      for (std::size_t i = 0; i < last_.size(); ++i) {
        next[i] = std::max(min_lr_, last_[i] * factor_);
      }
      apply(next);
      cooldown_left_ = cooldown_;
      bad_epochs_ = 0;
    }
  }

 private:
  double factor_;
  int patience_;
  double threshold_;
  int cooldown_;
  double min_lr_;
  double best_ = 0.0;
  bool has_best_ = false;
  int bad_epochs_ = 0;
  int cooldown_left_ = 0;
};

}  // namespace nexus_optim
