#pragma once

#include "nexus_optim/scheduler/lr_scheduler_base.hpp"

#include <cmath>

namespace nexus_optim {

/// SGDR (Loshchilov & Hutter). T_cur resets when it reaches T_i, then T_i *= T_mult.
class CosineAnnealingWarmRestarts : public LRSchedulerBase {
 public:
  template <typename Optim>
  CosineAnnealingWarmRestarts(Optim& optimizer, std::int64_t t_0, std::int64_t t_mult = 1,
                              double eta_min = 0.0)
      : LRSchedulerBase(bind_learning_rates(optimizer)),
        t_0_(t_0),
        t_i_(t_0),
        t_mult_(t_mult),
        eta_min_(eta_min) {
    if (t_0_ < 1 || t_mult_ < 1) {
      throw std::invalid_argument("Warm restarts require T_0 >= 1 and T_mult >= 1");
    }
  }

  void step(std::optional<double> metric = std::nullopt) override {
    (void)metric;
    if (t_cur_ >= t_i_) {
      t_cur_ = 0;
      t_i_ *= t_mult_;
    }
    const double cosv =
        std::cos(3.14159265358979323846 * static_cast<double>(t_cur_) / static_cast<double>(t_i_));
    std::vector<double> next(base_.size());
    for (std::size_t i = 0; i < base_.size(); ++i) {
      next[i] = eta_min_ + (base_[i] - eta_min_) * (1.0 + cosv) * 0.5;
    }
    apply(next);
    ++t_cur_;
    ++last_epoch_;
  }

 private:
  std::int64_t t_0_;
  std::int64_t t_i_;
  std::int64_t t_mult_;
  std::int64_t t_cur_ = 0;
  double eta_min_;
};

}  // namespace nexus_optim
