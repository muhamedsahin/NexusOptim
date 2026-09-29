#include "nexus_optim/nexus_optim.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

TEST(ClipGradNorm, ScalesGlobalL2) {
  std::vector<float> p{0.f, 0.f, 0.f};
  std::vector<float> g{3.f, 4.f, 0.f};
  auto group = nexus_optim::make_group(p.data(), g.data(), 3);
  std::vector<nexus_optim::ParamGroup<float>> groups{group};
  const double total = nexus_optim::clip_grad_norm_(groups, 1.0);
  EXPECT_NEAR(total, 5.0, 1e-5);
  EXPECT_NEAR(g[0], 0.6f, 1e-5f);
  EXPECT_NEAR(g[1], 0.8f, 1e-5f);
}

TEST(ClipGradValue, ClampsElements) {
  std::vector<float> p{0.f, 0.f, 0.f};
  std::vector<float> g{-3.f, 0.2f, 4.f};
  auto group = nexus_optim::make_group(p.data(), g.data(), 3);
  std::vector<nexus_optim::ParamGroup<float>> groups{group};
  nexus_optim::clip_grad_value_(groups, 1.f);
  EXPECT_EQ(g[0], -1.f);
  EXPECT_FLOAT_EQ(g[1], 0.2f);
  EXPECT_EQ(g[2], 1.f);
}

TEST(Centralization, RemovesMean) {
  std::vector<float> p{0.f, 0.f, 0.f, 0.f};
  std::vector<float> g{1.f, 3.f, 0.f, 4.f};
  auto group = nexus_optim::make_group(p.data(), g.data(), 4);
  std::vector<nexus_optim::ParamGroup<float>> groups{group};
  nexus_optim::gradient_centralization_(groups, 2);
  EXPECT_NEAR(g[0], -1.f, 1e-5f);
  EXPECT_NEAR(g[1], 1.f, 1e-5f);
  EXPECT_NEAR(g[2], -2.f, 1e-5f);
  EXPECT_NEAR(g[3], 2.f, 1e-5f);
}

TEST(Accumulation, ScalesBeforeStep) {
  std::vector<float> p{1.f};
  std::vector<float> g{2.f};
  auto group = nexus_optim::make_group(p.data(), g.data(), 1, { .learning_rate = 0.1 });
  std::vector<nexus_optim::ParamGroup<float>> groups{group};
  nexus_optim::scale_gradients_(groups, 0.5);
  EXPECT_NEAR(g[0], 1.f, 1e-6f);
  nexus_optim::SGD<float> opt(groups, {});
  opt.step();
  EXPECT_NEAR(p[0], 0.9f, 1e-5f);
}
