#include "nexus_optim/nexus_optim.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace {

struct Fixture {
  std::vector<float> p{1.f, 1.f};
  std::vector<float> g{0.f, 0.f};
  nexus_optim::SGD<float> opt;
  Fixture() : opt({group()}, {}) {}
  nexus_optim::ParamGroup<float> group() const {
    nexus_optim::ParamGroupOptions options;
    options.learning_rate = 0.8;
    return nexus_optim::make_group(const_cast<float*>(p.data()), const_cast<float*>(g.data()),
                                   p.size(), options);
  }
};

}  // namespace

TEST(StepLR, DecaysOnInterval) {
  std::vector<float> p{1.f};
  std::vector<float> g{0.f};
  nexus_optim::ParamGroupOptions options;
  options.learning_rate = 0.8;
  nexus_optim::SGD<float> opt({nexus_optim::make_group(p.data(), g.data(), 1, options)}, {});
  nexus_optim::StepLR sched(opt, 2, 0.1);
  sched.step();
  EXPECT_NEAR(sched.get_last_lr(), 0.8, 1e-12);
  sched.step();
  EXPECT_NEAR(opt.param_groups()[0].options.learning_rate, 0.8, 1e-12);
  sched.step();
  EXPECT_NEAR(opt.param_groups()[0].options.learning_rate, 0.08, 1e-12);
}

TEST(ExponentialLR, Geometric) {
  std::vector<float> p{1.f};
  std::vector<float> g{0.f};
  nexus_optim::ParamGroupOptions options;
  options.learning_rate = 1.0;
  nexus_optim::SGD<float> opt({nexus_optim::make_group(p.data(), g.data(), 1, options)}, {});
  nexus_optim::ExponentialLR sched(opt, 0.5);
  sched.step();
  EXPECT_NEAR(sched.get_last_lr(), 1.0, 1e-12);
  sched.step();
  EXPECT_NEAR(sched.get_last_lr(), 0.5, 1e-12);
}

TEST(CosineAnnealing, EndsAtEtaMin) {
  std::vector<float> p{1.f};
  std::vector<float> g{0.f};
  nexus_optim::ParamGroupOptions options;
  options.learning_rate = 1.0;
  nexus_optim::SGD<float> opt({nexus_optim::make_group(p.data(), g.data(), 1, options)}, {});
  nexus_optim::CosineAnnealingLR sched(opt, 2, 0.0);
  sched.step();
  EXPECT_NEAR(sched.get_last_lr(), 1.0, 1e-6);
  sched.step();
  EXPECT_NEAR(sched.get_last_lr(), 0.5, 1e-5);
  sched.step();
  EXPECT_NEAR(sched.get_last_lr(), 0.0, 1e-5);
}

TEST(WarmRestarts, ResetsToBase) {
  std::vector<float> p{1.f};
  std::vector<float> g{0.f};
  nexus_optim::ParamGroupOptions options;
  options.learning_rate = 1.0;
  nexus_optim::SGD<float> opt({nexus_optim::make_group(p.data(), g.data(), 1, options)}, {});
  nexus_optim::CosineAnnealingWarmRestarts sched(opt, 2, 1, 0.0);
  sched.step();
  const double first = sched.get_last_lr();
  sched.step();
  sched.step();
  EXPECT_NEAR(sched.get_last_lr(), first, 1e-5);
}

TEST(LinearWarmup, ReachesBase) {
  std::vector<float> p{1.f};
  std::vector<float> g{0.f};
  nexus_optim::ParamGroupOptions options;
  options.learning_rate = 0.4;
  nexus_optim::SGD<float> opt({nexus_optim::make_group(p.data(), g.data(), 1, options)}, {});
  nexus_optim::LinearWarmup sched(opt, 4);
  sched.step();
  EXPECT_NEAR(sched.get_last_lr(), 0.1, 1e-12);
  sched.step();
  sched.step();
  sched.step();
  EXPECT_NEAR(sched.get_last_lr(), 0.4, 1e-12);
}

TEST(OneCycle, ClimbsThenFalls) {
  std::vector<float> p{1.f};
  std::vector<float> g{0.f};
  nexus_optim::ParamGroupOptions options;
  options.learning_rate = 1.0;
  nexus_optim::SGD<float> opt({nexus_optim::make_group(p.data(), g.data(), 1, options)}, {});
  nexus_optim::OneCycleLR sched(opt, 1.0, 10, 0.3);
  sched.step();
  const double early = sched.get_last_lr();
  for (int i = 0; i < 3; ++i) sched.step();
  const double mid = sched.get_last_lr();
  for (int i = 0; i < 6; ++i) sched.step();
  EXPECT_GT(mid, early);
  EXPECT_LT(sched.get_last_lr(), mid);
}

TEST(ReduceLROnPlateau, DropsWhenStuck) {
  std::vector<float> p{1.f};
  std::vector<float> g{0.f};
  nexus_optim::ParamGroupOptions options;
  options.learning_rate = 0.5;
  nexus_optim::SGD<float> opt({nexus_optim::make_group(p.data(), g.data(), 1, options)}, {});
  nexus_optim::ReduceLROnPlateau sched(opt, 0.1, /*patience=*/1);
  sched.step(1.0);
  sched.step(1.0);
  sched.step(1.0);
  EXPECT_NEAR(opt.param_groups()[0].options.learning_rate, 0.05, 1e-12);
}
