#pragma once

/// Gradient accumulation is a training-loop policy: do not call `zero_grad` between
/// micro-batches, then scale once before `step()`. This helper applies that scale.

#include "nexus_optim/core/param_group.hpp"
#include "nexus_optim/simd/simd_dispatch.hpp"

#include <type_traits>
#include <vector>

namespace nexus_optim {

template <typename ScalarT>
void scale_gradients_(std::vector<ParamGroup<ScalarT>>& groups, double scale) {
  const auto s = static_cast<ScalarT>(scale);
  for (auto& group : groups) {
    for (std::size_t i = 0; i < group.grads.size(); ++i) {
      if (group.grads[i] == nullptr || group.devices[i] != Device::CPU) continue;
      if constexpr (std::is_same_v<ScalarT, float>) {
        scale_inplace(group.grads[i], group.numels[i], s);
      } else {
        scalar::scale_inplace(group.grads[i], group.numels[i], s);
      }
    }
  }
}

}  // namespace nexus_optim
