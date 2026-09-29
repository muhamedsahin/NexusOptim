#include "nexus_optim/nexus_optim.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

TEST(AdamW, DecoupledWeightDecay) {
  std::vector<float> p{1.f};
  std::vector<float> g{0.1f};
  nexus_optim::AdamW<float>::Options options;
  options.lr = 1e-3;
  options.beta1 = 0.9;
  options.beta2 = 0.999;
  options.eps = 1e-8;
  options.weight_decay = 0.01;
  nexus_optim::ParamGroupOptions group;
  group.learning_rate = 1e-3;
  nexus_optim::AdamW<float> opt({nexus_optim::make_group(p.data(), g.data(), 1, group)}, options);
  opt.step();
  // p <- 1 - lr*wd*p, then the same Adam step as the no-decay case (0.001).
  EXPECT_NEAR(p[0], 0.99899f, 2e-5f);
}

TEST(AdamW, GroupCanDisableDecay) {
  std::vector<float> decayed{1.f};
  std::vector<float> plain{1.f};
  std::vector<float> gd{0.1f};
  std::vector<float> gp{0.1f};
  nexus_optim::AdamW<float>::Options options;
  options.lr = 1e-3;
  options.weight_decay = 0.01;
  nexus_optim::ParamGroupOptions with_decay;
  with_decay.learning_rate = 1e-3;
  nexus_optim::ParamGroupOptions no_decay;
  no_decay.learning_rate = 1e-3;
  no_decay.weight_decay = 0.0;
  no_decay.weight_decay_override = true;
  nexus_optim::AdamW<float> opt(
      {nexus_optim::make_group(decayed.data(), gd.data(), 1, with_decay),
       nexus_optim::make_group(plain.data(), gp.data(), 1, no_decay)},
      options);
  opt.step();
  EXPECT_GT(plain[0], decayed[0]);
  EXPECT_NEAR(plain[0], 0.999f, 2e-5f);
}

TEST(AdamW, HalfMasterWeightsStayFinite) {
  std::vector<nexus_optim::float16> p(8);
  std::vector<nexus_optim::float16> g(8);
  for (int i = 0; i < 8; ++i) {
    p[static_cast<std::size_t>(i)] = nexus_optim::float16::from_float(0.5f);
    g[static_cast<std::size_t>(i)] = nexus_optim::float16::from_float(0.01f);
  }
  nexus_optim::AdamW<nexus_optim::float16>::Options options;
  options.lr = 1e-3;
  options.weight_decay = 0.01;
  nexus_optim::ParamGroupOptions group;
  group.learning_rate = 1e-3;
  nexus_optim::AdamW<nexus_optim::float16> opt(
      {nexus_optim::make_group(p.data(), g.data(), p.size(), group)}, options);
  opt.step();
  const float updated = p[0].to_float();
  EXPECT_TRUE(std::isfinite(updated));
  EXPECT_LT(updated, 0.5f);
}
