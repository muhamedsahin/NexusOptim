#include "nexus_optim/nexus_optim.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace {

nexus_optim::ParamGroup<float> grp(std::vector<float>& p, std::vector<float>& g, double lr) {
  nexus_optim::ParamGroupOptions options;
  options.learning_rate = lr;
  return nexus_optim::make_group(p.data(), g.data(), p.size(), options);
}

}  // namespace

TEST(Adagrad, AccumulatesSquaredGradient) {
  std::vector<float> p{0.f};
  std::vector<float> g{1.f};
  nexus_optim::Adagrad<float>::Options options;
  options.lr = 0.1;
  options.eps = 1e-8;
  nexus_optim::Adagrad<float> opt({grp(p, g, 0.1)}, options);
  opt.step();
  EXPECT_NEAR(p[0], -0.1f / (1.f + 1e-8f), 1e-5f);
}

TEST(RmsProp, MovesAgainstGradient) {
  std::vector<float> p{1.f};
  std::vector<float> g{0.2f};
  nexus_optim::RMSProp<float>::Options options;
  options.lr = 1e-2;
  options.alpha = 0.99;
  nexus_optim::RMSProp<float> opt({grp(p, g, 1e-2)}, options);
  opt.step();
  EXPECT_LT(p[0], 1.f);
  EXPECT_TRUE(std::isfinite(p[0]));
}

TEST(Lion, SignUpdate) {
  std::vector<float> p{0.5f};
  std::vector<float> g{1.f};
  nexus_optim::Lion<float>::Options options;
  options.lr = 1e-3;
  options.weight_decay = 0.0;
  nexus_optim::Lion<float> opt({grp(p, g, 1e-3)}, options);
  opt.step();
  EXPECT_NEAR(p[0], 0.499f, 1e-5f);
}

TEST(Lamb, TrustRatioDoesNotExplode) {
  std::vector<float> p{0.2f, -0.2f, 0.1f, 0.0f};
  std::vector<float> g{0.01f, 0.02f, -0.01f, 0.03f};
  nexus_optim::LAMB<float>::Options options;
  options.lr = 1e-3;
  options.weight_decay = 0.01;
  nexus_optim::LAMB<float> opt({grp(p, g, 1e-3)}, options);
  opt.step();
  for (float v : p) EXPECT_TRUE(std::isfinite(v));
}

TEST(Lars, StaysFinite) {
  std::vector<float> p{1.f, 1.f, 1.f, 1.f};
  std::vector<float> g{0.1f, -0.1f, 0.2f, 0.0f};
  nexus_optim::LARS<float>::Options options;
  options.lr = 0.1;
  options.momentum = 0.9;
  options.weight_decay = 1e-4;
  nexus_optim::LARS<float> opt({grp(p, g, 0.1)}, options);
  opt.step();
  for (float v : p) EXPECT_TRUE(std::isfinite(v));
}

TEST(RAdam, EarlyStepsAreFinite) {
  std::vector<float> p{0.3f, -0.1f};
  std::vector<float> g{0.05f, 0.02f};
  nexus_optim::RAdam<float>::Options options;
  options.lr = 1e-3;
  nexus_optim::RAdam<float> opt({grp(p, g, 1e-3)}, options);
  for (int i = 0; i < 5; ++i) {
    g[0] = 0.05f;
    g[1] = 0.02f;
    opt.step();
  }
  EXPECT_TRUE(std::isfinite(p[0]));
  EXPECT_TRUE(std::isfinite(p[1]));
}

TEST(Lookahead, PullsFastWeightsTowardSlow) {
  std::vector<float> p{1.f};
  std::vector<float> g{1.f};
  nexus_optim::SGD<float>::Options options;
  options.lr = 0.1;
  nexus_optim::SGD<float> inner({grp(p, g, 0.1)}, options);
  nexus_optim::Lookahead<nexus_optim::SGD<float>> opt(std::move(inner), /*k=*/1, 0.5f);
  opt.step();
  // fast becomes 0.9, slow starts at 1, blend = 0.5*1 + 0.5*0.9 = 0.95
  EXPECT_NEAR(p[0], 0.95f, 1e-5f);
}

TEST(Nadam, FirstStepIsFinite) {
  std::vector<float> p{0.2f};
  std::vector<float> g{0.1f};
  nexus_optim::NAdam<float>::Options options;
  options.lr = 2e-3;
  nexus_optim::NAdam<float> opt({grp(p, g, 2e-3)}, options);
  opt.step();
  EXPECT_TRUE(std::isfinite(p[0]));
  EXPECT_LT(p[0], 0.2f);
}

TEST(AdaBelief, MovesDown) {
  std::vector<float> p{0.4f};
  std::vector<float> g{0.2f};
  nexus_optim::AdaBelief<float>::Options options;
  options.lr = 1e-3;
  options.weight_decay = 0.0;
  options.decoupled_weight_decay = false;
  nexus_optim::AdaBelief<float> opt({grp(p, g, 1e-3)}, options);
  opt.step();
  EXPECT_LT(p[0], 0.4f);
}
