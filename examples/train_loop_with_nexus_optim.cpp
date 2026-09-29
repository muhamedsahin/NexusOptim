#include "nexus_optim/nexus_optim.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Tiny stand-in for NexusTrain: fit y = 3x + 1 with plain SGD.
// NexusModel would own the parameters; NexusLoss would produce the gradient.
// NexusOptim only applies that gradient.

int main() {
  float weight = 0.f;
  float bias = 0.f;
  float weight_grad = 0.f;
  float bias_grad = 0.f;

  nexus_optim::ParamGroupOptions options;
  options.learning_rate = 0.05;
  nexus_optim::SGD<float> optimizer(
      {nexus_optim::make_group(&weight, &weight_grad, 1, options),
       nexus_optim::make_group(&bias, &bias_grad, 1, options)},
      {});
  nexus_optim::LinearWarmup warmup(optimizer, 10);

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> inputs(-1.f, 1.f);
  float last_loss = 0.f;
  for (int step = 0; step < 200; ++step) {
    optimizer.zero_grad();
    float loss = 0.f;
    constexpr int batch = 32;
    for (int i = 0; i < batch; ++i) {
      const float x = inputs(rng);
      const float y = 3.f * x + 1.f;
      const float pred = weight * x + bias;
      const float err = pred - y;
      loss += err * err;
      weight_grad += err * x;
      bias_grad += err;
    }
    loss /= static_cast<float>(batch);
    weight_grad /= static_cast<float>(batch);
    bias_grad /= static_cast<float>(batch);
    optimizer.step();
    warmup.step();
    last_loss = loss;
  }
  std::cout << "weight=" << weight << " bias=" << bias << " loss=" << last_loss << '\n';
  if (!(std::abs(weight - 3.f) < 0.15f && std::abs(bias - 1.f) < 0.15f)) {
    std::cerr << "linear fit did not converge\n";
    return 1;
  }
  return 0;
}
