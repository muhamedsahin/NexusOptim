#pragma once

#include "nexus_optim/core/types.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nexus_optim {

/// Per-group hyperparameters. Learning-rate schedulers write @c learning_rate in place.
///
/// Weight decay resolution:
/// - if @c weight_decay_override is false, the optimizer-level option is used;
/// - if true, @c weight_decay on this group wins (bias / LayerNorm groups pass 0).
struct ParamGroupOptions {
  double learning_rate = 1e-3;
  double weight_decay = 0.0;
  bool weight_decay_override = false;
};

/// One homogeneous group of borrowed, contiguous parameter tensors.
///
/// NexusOptim does not own the tensors and never allocates them. The caller
/// (NexusModel / MatrixFlash) keeps the storage alive for the optimizer's lifetime.
/// Layout is structure-of-arrays: parameter pointer, gradient pointer, length, device.
template <typename ScalarT>
struct ParamGroup {
  std::vector<ScalarT*> params;
  std::vector<ScalarT*> grads;
  std::vector<std::size_t> numels;
  std::vector<Device> devices;
  ParamGroupOptions options{};

  void validate() const {
    const std::size_t n = params.size();
    if (grads.size() != n || numels.size() != n || devices.size() != n) {
      throw std::invalid_argument(
          "ParamGroup: params, grads, numels, and devices must have the same length");
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (numels[i] > 0 && (params[i] == nullptr || grads[i] == nullptr)) {
        throw std::invalid_argument("ParamGroup: non-empty tensor has a null data pointer");
      }
    }
  }
};

/// Build a CPU group from parallel host pointers.
template <typename ScalarT>
ParamGroup<ScalarT> make_group(std::vector<ScalarT*> params, std::vector<ScalarT*> grads,
                               std::vector<std::size_t> numels, ParamGroupOptions options = {},
                               Device device = Device::CPU) {
  ParamGroup<ScalarT> group;
  group.params = std::move(params);
  group.grads = std::move(grads);
  group.numels = std::move(numels);
  group.devices.assign(group.params.size(), device);
  group.options = options;
  group.validate();
  return group;
}

/// Single contiguous CPU tensor convenience overload.
template <typename ScalarT>
ParamGroup<ScalarT> make_group(ScalarT* params, ScalarT* grads, std::size_t numel,
                               ParamGroupOptions options = {}) {
  return make_group<ScalarT>({params}, {grads}, {numel}, options, Device::CPU);
}

}  // namespace nexus_optim
