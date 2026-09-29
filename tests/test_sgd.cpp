#include "nexus_optim/nexus_optim.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace {

nexus_optim::ParamGroup<float> host_group(std::vector<float>& params, std::vector<float>& grads,
                                          double lr, double wd = 0.0, bool override_wd = false) {
  nexus_optim::ParamGroupOptions options;
  options.learning_rate = lr;
  options.weight_decay = wd;
  options.weight_decay_override = override_wd;
  return nexus_optim::make_group(params.data(), grads.data(), params.size(), options);
}

}  // namespace

TEST(Sgd, SubtractsScaledGradient) {
  std::vector<float> p{1.f, 2.f, 3.f};
  std::vector<float> g{0.1f, -0.2f, 0.5f};
  nexus_optim::SGD<float> opt({host_group(p, g, 0.1)}, {});
  opt.step();
  EXPECT_NEAR(p[0], 0.99f, 1e-6f);
  EXPECT_NEAR(p[1], 2.02f, 1e-6f);
  EXPECT_NEAR(p[2], 2.95f, 1e-6f);
}

TEST(Sgd, WeightDecayAddsL2Term) {
  std::vector<float> p{1.f};
  std::vector<float> g{0.f};
  auto group = host_group(p, g, 0.1, 0.5, true);
  nexus_optim::SGD<float>::Options options;
  options.lr = 0.1;
  nexus_optim::SGD<float> opt({group}, options);
  opt.step();
  // g_eff = 0 + 0.5 * 1, p -= 0.1 * 0.5
  EXPECT_NEAR(p[0], 0.95f, 1e-6f);
}

TEST(Momentum, MatchesPytorchBuffer) {
  std::vector<float> p{1.f};
  std::vector<float> g{0.5f};
  nexus_optim::MomentumSGD<float>::Options options;
  options.momentum = 0.9;
  options.lr = 0.1;
  auto group = host_group(p, g, 0.1);
  nexus_optim::MomentumSGD<float> opt({group}, options);
  opt.step();
  EXPECT_NEAR(p[0], 0.95f, 1e-5f);
  g[0] = 0.5f;
  opt.step();
  // v = 0.9*0.5 + 0.5 = 0.95, p = 0.95 - 0.1*0.95
  EXPECT_NEAR(p[0], 0.855f, 1e-5f);
}

TEST(Nesterov, AppliesLookaheadGradient) {
  std::vector<float> p{1.f};
  std::vector<float> g{0.5f};
  nexus_optim::NesterovSGD<float>::Options options;
  options.lr = 0.1;
  options.momentum = 0.9;
  nexus_optim::NesterovSGD<float> opt({host_group(p, g, 0.1)}, options);
  opt.step();
  // v = 0.5, update = 0.5 + 0.9*0.5 = 0.95, p = 1 - 0.1*0.95
  EXPECT_NEAR(p[0], 0.905f, 1e-5f);
}

TEST(Sgd, ZeroGradAndEmptyTensor) {
  std::vector<float> p{1.f, 2.f};
  std::vector<float> g{3.f, 4.f};
  std::vector<float> empty;
  std::vector<float> empty_g;
  nexus_optim::SGD<float> opt(
      {host_group(p, g, 0.1), host_group(empty, empty_g, 0.1)}, {});
  opt.zero_grad();
  EXPECT_EQ(g[0], 0.f);
  EXPECT_EQ(g[1], 0.f);
  opt.step();
  EXPECT_EQ(p[0], 1.f);
  EXPECT_EQ(p[1], 2.f);
}

TEST(Sgd, RejectsEmptyOptimizer) {
  EXPECT_THROW(nexus_optim::SGD<float>({}, {}), std::invalid_argument);
}

TEST(Sgd, SingleElementAndLargeGradient) {
  std::vector<float> p{0.f};
  std::vector<float> g{1e20f};
  nexus_optim::SGD<float> opt({host_group(p, g, 1e-20)}, {});
  opt.step();
  EXPECT_TRUE(std::isfinite(p[0]));
  EXPECT_NEAR(p[0], -1.f, 1e-4f);
}

TEST(Sgd, StateRoundTrip) {
  std::vector<float> p{1.f};
  std::vector<float> g{0.25f};
  nexus_optim::MomentumSGD<float>::Options options;
  options.momentum = 0.8;
  options.lr = 0.1;
  nexus_optim::MomentumSGD<float> opt({host_group(p, g, 0.1)}, options);
  opt.step();
  const auto state = opt.state_dict();
  const float checkpoint = p[0];

  std::vector<float> p2{checkpoint};
  std::vector<float> g2{0.25f};
  nexus_optim::MomentumSGD<float> restored({host_group(p2, g2, 0.1)}, options);
  restored.load_state_dict(state);
  g[0] = 0.25f;
  opt.step();
  restored.step();
  EXPECT_NEAR(p2[0], p[0], 1e-5f);
}

TEST(Sgd, ParameterGroupsUseDifferentRates) {
  std::vector<float> a{1.f};
  std::vector<float> ag{1.f};
  std::vector<float> b{1.f};
  std::vector<float> bg{1.f};
  nexus_optim::SGD<float> opt({host_group(a, ag, 0.1), host_group(b, bg, 0.01)}, {});
  opt.step();
  EXPECT_NEAR(a[0], 0.9f, 1e-6f);
  EXPECT_NEAR(b[0], 0.99f, 1e-6f);
}
