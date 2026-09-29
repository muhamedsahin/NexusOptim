#pragma once

#include "nexus_optim/core/param_group.hpp"
#include "nexus_optim/simd/simd_dispatch.hpp"

#include <stdexcept>
#include <type_traits>
#include <vector>

namespace nexus_optim {

/// Subtract the per-channel mean of the gradient (Yong et al.).
/// `channels` is the leading dimension; the remaining `numel / channels` elements
/// form the spatial/fan-in axis. Biases (spatial <= 1) are left untouched.
/// Pass `channels == 0` to treat the whole tensor as a single vector.
template <typename ScalarT>
void gradient_centralization_(std::vector<ParamGroup<ScalarT>>& groups,
                              std::size_t channels = 0) {
  for (auto& group : groups) {
    for (std::size_t i = 0; i < group.grads.size(); ++i) {
      if (group.grads[i] == nullptr || group.numels[i] == 0 || group.devices[i] != Device::CPU) {
        continue;
      }
      const std::size_t c = channels == 0 ? 1 : channels;
      if (group.numels[i] % c != 0) {
        throw std::invalid_argument("gradient_centralization_: numel is not divisible by channels");
      }
      const std::size_t spatial = group.numels[i] / c;
      if constexpr (std::is_same_v<ScalarT, float>) {
        centralize(group.grads[i], c, spatial);
      } else {
        scalar::centralize(group.grads[i], c, spatial);
      }
    }
  }
}

}  // namespace nexus_optim
