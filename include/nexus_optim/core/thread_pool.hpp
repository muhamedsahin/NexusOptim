#pragma once

#include "nexus_optim/core/types.hpp"

#include <cstddef>
#include <memory>

namespace nexus_optim {

/// Process-wide pool, warmed from the optimizer constructor so `step()` does not allocate
/// and does not spawn threads. Idle workers sleep on a condition variable.
class ThreadPool {
 public:
  static ThreadPool& instance();

  int helper_count() const noexcept;

  /// Splits `[0, n)` across the pool plus the calling thread. @p grain below kParallelGrain
  /// (or a single hardware thread) runs entirely on the caller.
  void parallel(std::size_t n, void (*fn)(void*, std::size_t, std::size_t), void* ctx);

 private:
  ThreadPool();
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

template <typename Fn>
inline void parallel_for(std::size_t n, Fn&& fn) {
  if (n == 0) {
    return;
  }
  if (n < kParallelGrain || ThreadPool::instance().helper_count() == 0) {
    fn(static_cast<std::size_t>(0), n);
    return;
  }
  auto* obj = &fn;
  ThreadPool::instance().parallel(
      n,
      [](void* ctx, std::size_t begin, std::size_t end) {
        (*static_cast<Fn*>(ctx))(begin, end);
      },
      obj);
}

}  // namespace nexus_optim
