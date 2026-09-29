#include "nexus_optim/nexus_optim.hpp"

#include <iostream>
#include <vector>

int main() {
  std::vector<float> params{1.f, 2.f, 3.f, 4.f};
  std::vector<float> grads{0.1f, -0.2f, 0.3f, -0.4f};

  nexus_optim::ParamGroupOptions group_options;
  group_options.learning_rate = 0.1;

  nexus_optim::SGD<float>::Options options;
  options.lr = 0.1;
  nexus_optim::SGD<float> optimizer(
      {nexus_optim::make_group(params.data(), grads.data(), params.size(), group_options)}, options);

  optimizer.zero_grad();
  grads = {0.1f, -0.2f, 0.3f, -0.4f};
  optimizer.step();

  std::cout << "sgd params:";
  for (float value : params) std::cout << ' ' << value;
  std::cout << '\n';
  return 0;
}
