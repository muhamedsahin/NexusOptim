#pragma once

#include "nexus_optim/core/types.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

namespace nexus_optim {

/// Process-wide pool, warmed from the optimizer constructor so `step()` does not allocate
/// and does not spawn threads. Idle workers sleep on a condition variable.
class ThreadPool {
 public:
  static ThreadPool& instance();

  /// Python kabugu, havuz kurulmadan once bunu cagirir. Yardimci is parcacigi
  /// acilmaz; pyd bellekten kalkarken katilim erisim ihlali yapmaz.
  static void use_caller_thread() noexcept;

  /// Yorumlayici kalici is parcacigi tutmaz. Buyuk adim, donmeden once katilir.
  [[nodiscard]] static bool join_per_call() noexcept;

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
  if (ThreadPool::join_per_call()) {
    if (n < kParallelGrain) {
      fn(static_cast<std::size_t>(0), n);
      return;
    }
    const unsigned hardware = std::thread::hardware_concurrency();
    const int parts = hardware < 2 ? 1 : static_cast<int>(std::min<unsigned>(hardware, 8u));
    if (parts <= 1) {
      fn(static_cast<std::size_t>(0), n);
      return;
    }
    const std::size_t chunk = (n + static_cast<std::size_t>(parts) - 1) / static_cast<std::size_t>(parts);
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(parts - 1));
    for (int part = 1; part < parts; ++part) {
      const std::size_t begin = static_cast<std::size_t>(part) * chunk;
      if (begin >= n) {
        break;
      }
      const std::size_t end = std::min(n, begin + chunk);
      threads.emplace_back([&fn, begin, end] { fn(begin, end); });
    }
    fn(static_cast<std::size_t>(0), std::min(n, chunk));
    for (std::thread& thread : threads) {
      thread.join();
    }
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
