#pragma once

#include "nexus_optim/core/aligned_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nexus_optim {

/// Checkpoint blob for a single optimizer. NexusTrain owns the file format;
/// NexusOptim only packs and restores its own moments, step counter, and learning rates.
struct OptimizerState {
  std::string algorithm;
  std::uint64_t step = 0;
  std::vector<double> group_learning_rates;
  /// Raw little-endian copies of moment buffers, in slot-major order.
  std::vector<std::vector<std::uint8_t>> blobs;
  /// Scalar side state (bias products, mu_product, lookahead counters).
  std::vector<double> scalars;
};

template <typename T>
inline void state_push_buffer(OptimizerState& state, const AlignedBuffer<T>& buffer) {
  std::vector<std::uint8_t> bytes(buffer.size() * sizeof(T));
  if (!bytes.empty() && buffer.data() != nullptr) {
    std::memcpy(bytes.data(), buffer.data(), bytes.size());
  }
  state.blobs.push_back(std::move(bytes));
}

template <typename T>
inline void state_load_buffer(const OptimizerState& state, std::size_t& cursor,
                              AlignedBuffer<T>& buffer) {
  if (cursor >= state.blobs.size()) {
    throw std::runtime_error("optimizer state is missing a moment buffer");
  }
  const auto& bytes = state.blobs[cursor++];
  if (bytes.size() != buffer.size() * sizeof(T)) {
    throw std::runtime_error("optimizer state buffer size does not match the parameter layout");
  }
  if (!bytes.empty()) {
    std::memcpy(buffer.data(), bytes.data(), bytes.size());
  }
}

}  // namespace nexus_optim
