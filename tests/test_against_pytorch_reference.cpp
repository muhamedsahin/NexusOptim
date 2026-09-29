#include "nexus_optim/nexus_optim.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

std::vector<float> read_csv_row(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return {};
  }
  std::string line;
  std::getline(in, line);
  std::stringstream stream(line);
  std::vector<float> values;
  std::string cell;
  while (std::getline(stream, cell, ',')) {
    values.push_back(std::stof(cell));
  }
  return values;
}

}  // namespace

TEST(PytorchReference, SgdAndAdamClosedForm) {
  const std::vector<float> sgd_expected{0.99f, 2.02f, 2.95f};
  std::vector<float> p{1.f, 2.f, 3.f};
  std::vector<float> g{0.1f, -0.2f, 0.5f};
  nexus_optim::ParamGroupOptions options;
  options.learning_rate = 0.1;
  nexus_optim::SGD<float> sgd({nexus_optim::make_group(p.data(), g.data(), 3, options)}, {});
  sgd.step();
  for (std::size_t i = 0; i < p.size(); ++i) {
    EXPECT_NEAR(p[i], sgd_expected[i], 1e-6f);
  }

  // torch.optim.Adam([p=1], lr=1e-3) after one step on grad 0.1.
  std::vector<float> ap{1.f};
  std::vector<float> ag{0.1f};
  nexus_optim::Adam<float>::Options adam_options;
  adam_options.lr = 1e-3;
  nexus_optim::ParamGroupOptions adam_group;
  adam_group.learning_rate = 1e-3;
  nexus_optim::Adam<float> adam(
      {nexus_optim::make_group(ap.data(), ag.data(), 1, adam_group)}, adam_options);
  adam.step();
  EXPECT_NEAR(ap[0], 0.999f, 1e-5f);
}

TEST(PytorchReference, CsvFileIfPresent) {
  const auto values = read_csv_row(std::string(NEXUS_OPTIM_SOURCE_DIR) + "/tests/reference/adam_step1.csv");
  if (values.size() < 2) {
    GTEST_SKIP() << "reference csv missing";
  }
  EXPECT_NEAR(values[0], 1.f, 1e-6f);
  EXPECT_NEAR(values[5], 0.999f, 1e-5f);
}
