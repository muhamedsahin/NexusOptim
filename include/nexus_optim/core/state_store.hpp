#pragma once

#include "nexus_optim/core/aligned_buffer.hpp"
#include "nexus_optim/core/optimizer_state.hpp"
#include "nexus_optim/core/param_group.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(NEXUS_OPTIM_WITH_CUDA)
#include "nexus_optim/cuda/cuda_launch.hpp"
#endif

namespace nexus_optim {

/// Moment storage. CPU slots live in aligned host memory. CUDA slots are
/// allocated once here, so `step()` never calls the device allocator.
template <typename T>
class StateStore {
 public:
  struct Slot {
    AlignedBuffer<T> host;
    T* device = nullptr;
    std::size_t count = 0;
    bool on_device = false;

    Slot() = default;
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    Slot(Slot&& other) noexcept
        : host(std::move(other.host)),
          device(std::exchange(other.device, nullptr)),
          count(other.count),
          on_device(other.on_device) {}
    Slot& operator=(Slot&& other) noexcept {
      if (this != &other) {
        release();
        host = std::move(other.host);
        device = std::exchange(other.device, nullptr);
        count = other.count;
        on_device = other.on_device;
      }
      return *this;
    }
    ~Slot() { release(); }

    void release() noexcept {
#if defined(NEXUS_OPTIM_WITH_CUDA)
      if (device != nullptr) {
        cuda::device_free(device);
        device = nullptr;
      }
#else
      device = nullptr;
#endif
    }
  };

  StateStore() = default;

  template <typename Group>
  static StateStore from_groups(const std::vector<Group>& groups) {
    StateStore store;
    for (const auto& group : groups) {
      for (std::size_t i = 0; i < group.numels.size(); ++i) {
        store.add(group.numels[i], group.devices[i]);
      }
    }
    return store;
  }

  void add(std::size_t count, Device device) {
    Slot slot;
    slot.count = count;
    slot.on_device = device == Device::CUDA;
    if (slot.on_device) {
#if defined(NEXUS_OPTIM_WITH_CUDA)
      if (count > 0) {
        slot.device = static_cast<T*>(cuda::device_alloc_bytes(count * sizeof(T)));
      }
#else
      throw std::runtime_error("CUDA moment buffer requested but NexusOptim was built without CUDA");
#endif
    } else if (count > 0) {
      slot.host = AlignedBuffer<T>(count);
    }
    slots_.push_back(std::move(slot));
  }

  bool empty() const noexcept { return slots_.empty(); }
  std::size_t size() const noexcept { return slots_.size(); }

  T* ptr(std::size_t index) noexcept {
    Slot& slot = slots_[index];
    return slot.on_device ? slot.device : slot.host.data();
  }
  const T* ptr(std::size_t index) const noexcept {
    const Slot& slot = slots_[index];
    return slot.on_device ? slot.device : slot.host.data();
  }

  void save(OptimizerState& state) const {
    for (const Slot& slot : slots_) {
      std::vector<std::uint8_t> bytes(slot.count * sizeof(T));
      if (!bytes.empty()) {
        if (slot.on_device) {
#if defined(NEXUS_OPTIM_WITH_CUDA)
          cuda::copy_device_to_host(bytes.data(), slot.device, bytes.size());
#else
          throw std::runtime_error("device state without CUDA");
#endif
        } else if (slot.host.data() != nullptr) {
          std::memcpy(bytes.data(), slot.host.data(), bytes.size());
        }
      }
      state.blobs.push_back(std::move(bytes));
    }
  }

  void load(const OptimizerState& state, std::size_t& cursor) {
    for (Slot& slot : slots_) {
      if (cursor >= state.blobs.size()) {
        throw std::runtime_error("optimizer state is missing a moment buffer");
      }
      const auto& bytes = state.blobs[cursor++];
      if (bytes.size() != slot.count * sizeof(T)) {
        throw std::runtime_error("optimizer state buffer size does not match the parameter layout");
      }
      if (bytes.empty()) {
        continue;
      }
      if (slot.on_device) {
#if defined(NEXUS_OPTIM_WITH_CUDA)
        cuda::copy_host_to_device(slot.device, bytes.data(), bytes.size());
#else
        throw std::runtime_error("device state without CUDA");
#endif
      } else if (slot.host.data() != nullptr) {
        std::memcpy(slot.host.data(), bytes.data(), bytes.size());
      }
    }
  }

 private:
  std::vector<Slot> slots_;
};

}  // namespace nexus_optim
