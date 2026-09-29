#include "nexus_optim/nexus_optim.hpp"
#include "nexus_optim/simd/scalar_kernels.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace {

void expect_close(const std::vector<float>& got, const std::vector<float>& ref, float tol) {
  ASSERT_EQ(got.size(), ref.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float scale = std::max(1.f, std::abs(ref[i]));
    EXPECT_NEAR(got[i], ref[i], tol * scale) << "index " << i;
  }
}

}  // namespace

TEST(Adam, FirstStepMatchesClosedForm) {
  // m = 0.01, v = 1e-5, mhat = 0.1, vhat = 0.01, step = 0.001 * 0.1 / 0.1
  std::vector<float> p{1.f};
  std::vector<float> g{0.1f};
  nexus_optim::Adam<float>::Options options;
  options.lr = 1e-3;
  options.beta1 = 0.9;
  options.beta2 = 0.999;
  options.eps = 1e-8;
  nexus_optim::ParamGroupOptions group;
  group.learning_rate = 1e-3;
  nexus_optim::Adam<float> opt({nexus_optim::make_group(p.data(), g.data(), 1, group)}, options);
  opt.step();
  EXPECT_NEAR(p[0], 0.999f, 1e-5f);
}

TEST(Adam, AmsgradMatchesScalarReference) {
  constexpr int n = 17;
  std::vector<float> p(n), g(n), p_ref(n), g_ref(n);
  for (int i = 0; i < n; ++i) {
    p[i] = p_ref[i] = 0.01f * static_cast<float>(i - 8);
    g[i] = g_ref[i] = 0.05f * static_cast<float>((i % 5) - 2);
  }
  nexus_optim::Adam<float>::Options options;
  options.lr = 1e-3;
  options.amsgrad = true;
  nexus_optim::ParamGroupOptions group_options;
  group_options.learning_rate = 1e-3;
  nexus_optim::force_simd_level(nexus_optim::SimdLevel::Scalar);
  std::vector<float> m(n, 0.f), v(n, 0.f), vmax(n, 0.f);
  nexus_optim::AdamCoeffs coeffs;
  coeffs.lr = 1e-3f;
  coeffs.beta1 = 0.9f;
  coeffs.one_minus_beta1 = 0.1f;
  coeffs.beta2 = 0.999f;
  coeffs.one_minus_beta2 = 0.001f;
  coeffs.eps = 1e-8f;
  coeffs.inv_bc1 = 1.f / 0.1f;
  coeffs.inv_bc2 = 1.f / 0.001f;
  coeffs.amsgrad = 1;
  nexus_optim::scalar::adam_update(p_ref.data(), g_ref.data(), m.data(), v.data(), vmax.data(), n,
                                   coeffs);
  nexus_optim::force_simd_level(std::nullopt);
  nexus_optim::Adam<float> opt(
      {nexus_optim::make_group(p.data(), g.data(), static_cast<std::size_t>(n), group_options)},
      options);
  opt.step();
  expect_close(p, p_ref, 1e-5f);
}

TEST(Adam, SimdMatchesScalarOnUnevenLengths) {
  if (!nexus_optim::force_simd_level(nexus_optim::SimdLevel::Avx2) &&
      !nexus_optim::force_simd_level(nexus_optim::SimdLevel::Avx512)) {
    GTEST_SKIP() << "CPU has no AVX2";
  }
  for (int n : {1, 7, 8, 9, 31, 32, 1000}) {
    std::vector<float> fast(n), slow(n), g(n), gs(n);
    for (int i = 0; i < n; ++i) {
      fast[i] = slow[i] = std::sin(0.1f * static_cast<float>(i));
      g[i] = gs[i] = std::cos(0.07f * static_cast<float>(i));
    }
    nexus_optim::Adam<float>::Options options;
    options.lr = 1e-3;
    options.beta1 = 0.9;
    options.beta2 = 0.999;
    nexus_optim::ParamGroupOptions group_options;
    group_options.learning_rate = 1e-3;
    nexus_optim::force_simd_level(nexus_optim::SimdLevel::Scalar);
    nexus_optim::Adam<float> scalar_opt(
        {nexus_optim::make_group(slow.data(), gs.data(), static_cast<std::size_t>(n), group_options)},
        options);
    scalar_opt.step();
    scalar_opt.step();
    const auto fast_level = nexus_optim::detected_simd_level();
    nexus_optim::force_simd_level(fast_level);
    nexus_optim::Adam<float> fast_opt(
        {nexus_optim::make_group(fast.data(), g.data(), static_cast<std::size_t>(n), group_options)},
        options);
    fast_opt.step();
    fast_opt.step();
    expect_close(fast, slow, 2e-5f);
  }
  nexus_optim::force_simd_level(std::nullopt);
}

TEST(Adam, ParallelRangeMatchesSerialScalar) {
  constexpr int n = 200000;
  std::vector<float> fast(n), slow(n), g(n), gs(n), m(n, 0.f), v(n, 0.f);
  for (int i = 0; i < n; ++i) {
    fast[i] = slow[i] = 0.001f * static_cast<float>(i % 97);
    g[i] = gs[i] = 0.002f * static_cast<float>((i % 13) - 6);
  }
  nexus_optim::AdamCoeffs coeffs{};
  coeffs.lr = 1e-3f;
  coeffs.beta1 = 0.9f;
  coeffs.one_minus_beta1 = 0.1f;
  coeffs.beta2 = 0.999f;
  coeffs.one_minus_beta2 = 0.001f;
  coeffs.eps = 1e-8f;
  coeffs.inv_bc1 = 10.f;
  coeffs.inv_bc2 = 1000.f;
  nexus_optim::scalar::adam_update(slow.data(), gs.data(), m.data(), v.data(),
                                  static_cast<float*>(nullptr), n, coeffs);

  nexus_optim::Adam<float>::Options options;
  options.lr = 1e-3;
  nexus_optim::ParamGroupOptions group;
  group.learning_rate = 1e-3;
  nexus_optim::Adam<float> opt(
      {nexus_optim::make_group(fast.data(), g.data(), static_cast<std::size_t>(n), group)}, options);
  opt.step();
  expect_close(fast, slow, 2e-5f);
}

TEST(Adam, NonFiniteGradientPropagates) {
  std::vector<float> p{1.f, 1.f};
  std::vector<float> g{std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::infinity()};
  nexus_optim::ParamGroupOptions group;
  group.learning_rate = 1e-3;
  nexus_optim::Adam<float> opt({nexus_optim::make_group(p.data(), g.data(), 2, group)}, {});
  opt.step();
  EXPECT_TRUE(std::isnan(p[0]));
  EXPECT_TRUE(std::isinf(p[1]) || std::isnan(p[1]));
}
