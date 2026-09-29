#pragma once

#include "nexus_optim/core/param_group.hpp"

#include "matrix_pro/core/tensor.hpp"

#include <stdexcept>
#include <vector>

namespace nexus_optim {

/// Borrow the host mirror. Does not call `download()`; the caller keeps the mirror current.
inline ParamGroup<float> group_from_host(const std::vector<matrix_pro::Tensor*>& params,
                                         const std::vector<matrix_pro::Tensor*>& grads,
                                         ParamGroupOptions options = {}) {
  if (params.size() != grads.size()) {
    throw std::invalid_argument("host parameter and gradient tensor counts differ");
  }
  ParamGroup<float> group;
  group.options = options;
  group.params.reserve(params.size());
  group.grads.reserve(params.size());
  group.numels.reserve(params.size());
  group.devices.reserve(params.size());
  for (std::size_t i = 0; i < params.size(); ++i) {
    if (params[i] == nullptr || grads[i] == nullptr) {
      throw std::invalid_argument("null MatrixFlash tensor");
    }
    if (params[i]->size() != grads[i]->size()) {
      throw std::invalid_argument("parameter and gradient sizes differ");
    }
    group.params.push_back(params[i]->host_ptr());
    group.grads.push_back(grads[i]->host_ptr());
    group.numels.push_back(params[i]->size());
    group.devices.push_back(Device::CPU);
  }
  return group;
}

/// Borrow `device_data()`. NexusOptim writes the device buffer and marks the host mirror stale.
inline ParamGroup<float> group_from_device(const std::vector<matrix_pro::Tensor*>& params,
                                           const std::vector<matrix_pro::Tensor*>& grads,
                                           ParamGroupOptions options = {}) {
  if (params.size() != grads.size()) {
    throw std::invalid_argument("device parameter and gradient tensor counts differ");
  }
  ParamGroup<float> group;
  group.options = options;
  group.params.reserve(params.size());
  group.grads.reserve(params.size());
  group.numels.reserve(params.size());
  group.devices.reserve(params.size());
  for (std::size_t i = 0; i < params.size(); ++i) {
    if (params[i] == nullptr || grads[i] == nullptr) {
      throw std::invalid_argument("null MatrixFlash tensor");
    }
    if (params[i]->device_data() == nullptr || grads[i]->device_data() == nullptr) {
      throw std::invalid_argument("tensor has no device allocation");
    }
    group.params.push_back(params[i]->device_data());
    group.grads.push_back(grads[i]->device_data());
    group.numels.push_back(params[i]->size());
    group.devices.push_back(Device::CUDA);
  }
  return group;
}

inline void mark_host_stale(const std::vector<matrix_pro::Tensor*>& tensors) {
  for (matrix_pro::Tensor* tensor : tensors) {
    if (tensor != nullptr) {
      tensor->mark_host_stale();
    }
  }
}

}  // namespace nexus_optim
