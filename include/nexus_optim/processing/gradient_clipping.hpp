#pragma once

#include "nexus_optim/core/param_group.hpp"
#include "nexus_optim/simd/simd_dispatch.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace nexus_optim {

/// Global p-norm clip. Returns the total norm before scaling.
/// Matches PyTorch: scale = min(1, max_norm / (total + 1e-6)).
template <typename ScalarT>
double clip_grad_norm_(std::vector<ParamGroup<ScalarT>>& groups, double max_norm,
                       double norm_type = 2.0) {
  if (max_norm < 0.0) {
    throw std::invalid_argument("max_norm must be non-negative");
  }
  double total = 0.0;
  const bool is_inf = norm_type == std::numeric_limits<double>::infinity();
  for (const auto& group : groups) {
    for (std::size_t i = 0; i < group.grads.size(); ++i) {
      if (group.grads[i] == nullptr || group.numels[i] == 0 || group.devices[i] != Device::CPU) {
        continue;
      }
      if (is_inf) {
        if constexpr (std::is_same_v<ScalarT, float>) {
          total = std::max(total, linf(group.grads[i], group.numels[i]));
        } else {
          total = std::max(total, scalar::linf(group.grads[i], group.numels[i]));
        }
      } else if (norm_type == 2.0) {
        double sum = 0.0;
        if constexpr (std::is_same_v<ScalarT, float>) {
          sum = l2_sq(group.grads[i], group.numels[i]);
        } else {
          sum = scalar::l2_sq(group.grads[i], group.numels[i]);
        }
        total += sum;
      } else {
        for (std::size_t k = 0; k < group.numels[i]; ++k) {
          total += std::pow(std::abs(static_cast<double>(group.grads[i][k])), norm_type);
        }
      }
    }
  }
  if (!is_inf && norm_type == 2.0) {
    total = std::sqrt(total);
  } else if (!is_inf) {
    total = std::pow(total, 1.0 / norm_type);
  }
  const double clip = max_norm / (total + 1e-6);
  if (clip < 1.0) {
    const auto scale = static_cast<ScalarT>(clip);
    for (auto& group : groups) {
      for (std::size_t i = 0; i < group.grads.size(); ++i) {
        if (group.grads[i] == nullptr || group.devices[i] != Device::CPU) continue;
        if constexpr (std::is_same_v<ScalarT, float>) {
          scale_inplace(group.grads[i], group.numels[i], scale);
        } else {
          scalar::scale_inplace(group.grads[i], group.numels[i], scale);
        }
      }
    }
  }
  return total;
}

template <typename ScalarT>
void clip_grad_value_(std::vector<ParamGroup<ScalarT>>& groups, ScalarT clip_value) {
  const ScalarT limit = clip_value < ScalarT(0) ? -clip_value : clip_value;
  for (auto& group : groups) {
    for (std::size_t i = 0; i < group.grads.size(); ++i) {
      if (group.grads[i] == nullptr || group.devices[i] != Device::CPU) continue;
      if constexpr (std::is_same_v<ScalarT, float>) {
        clip_abs(group.grads[i], group.numels[i], limit);
      } else {
        scalar::clip_abs(group.grads[i], group.numels[i], limit);
      }
    }
  }
}

}  // namespace nexus_optim
