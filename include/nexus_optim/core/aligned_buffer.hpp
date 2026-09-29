#pragma once

#include "nexus_optim/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace nexus_optim {

namespace detail {

inline void* aligned_alloc_bytes(std::size_t bytes, std::size_t alignment) {
  if (bytes == 0) {
    return nullptr;
  }
  // aligned_alloc requires the size to be a multiple of the alignment.
  const std::size_t rounded = (bytes + alignment - 1) & ~(alignment - 1);
#if defined(_MSC_VER)
  void* ptr = _aligned_malloc(rounded, alignment);
#else
  void* ptr = std::aligned_alloc(alignment, rounded);
#endif
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

inline void aligned_free_bytes(void* ptr) noexcept {
  if (ptr == nullptr) {
    return;
  }
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

}  // namespace detail

/// RAII buffer aligned to @p Align bytes. All optimizer moments are allocated here,
/// once, in the constructor — `step()` never touches the allocator.
template <typename T, std::size_t Align = kAlignment>
class AlignedBuffer {
 public:
  AlignedBuffer() = default;

  explicit AlignedBuffer(std::size_t count) : size_(count) {
    if (count == 0) {
      return;
    }
    data_ = static_cast<T*>(detail::aligned_alloc_bytes(count * sizeof(T), Align));
    std::memset(data_, 0, count * sizeof(T));
  }

  ~AlignedBuffer() { detail::aligned_free_bytes(data_); }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  AlignedBuffer(AlignedBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      detail::aligned_free_bytes(data_);
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }

  void fill(T value) {
    for (std::size_t i = 0; i < size_; ++i) {
      data_[i] = value;
    }
  }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace nexus_optim
