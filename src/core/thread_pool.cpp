#include "nexus_optim/core/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <mutex>
#include <thread>
#include <vector>

namespace nexus_optim {

struct ThreadPool::Impl {
  struct Slot {
    std::size_t begin = 0;
    std::size_t end = 0;
  };

  Impl() {
    unsigned hc = std::thread::hardware_concurrency();
    if (hc < 2) {
      helpers_ = 0;
      return;
    }
    helpers_ = static_cast<int>(std::min<unsigned>(hc - 1, 63));
    slots_.resize(static_cast<std::size_t>(helpers_) + 1);
    workers_.reserve(static_cast<std::size_t>(helpers_));
    for (int i = 0; i < helpers_; ++i) {
      workers_.emplace_back([this, i] { worker_loop(i); });
    }
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_.store(true, std::memory_order_release);
      epoch_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  void worker_loop(int index) {
    std::uint64_t seen = 0;
    while (true) {
      std::uint64_t published = epoch_.load(std::memory_order_acquire);
      int spins = 0;
      while (published == seen && !stop_.load(std::memory_order_acquire)) {
        if (spins < 8000) {
          ++spins;
#if defined(_M_X64) || defined(__x86_64__)
          _mm_pause();
#else
          std::this_thread::yield();
#endif
          published = epoch_.load(std::memory_order_acquire);
        } else {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_.wait(lock, [&] {
            return stop_.load(std::memory_order_acquire) ||
                   epoch_.load(std::memory_order_acquire) != seen;
          });
          published = epoch_.load(std::memory_order_acquire);
          break;
        }
      }
      if (stop_.load(std::memory_order_acquire)) {
        return;
      }
      seen = published;
      const std::size_t begin = slots_[static_cast<std::size_t>(index)].begin;
      const std::size_t end = slots_[static_cast<std::size_t>(index)].end;
      void (*fn)(void*, std::size_t, std::size_t) = fn_;
      void* ctx = ctx_;
      if (fn != nullptr && end > begin) {
        fn(ctx, begin, end);
      }
      remaining_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  void parallel(std::size_t n, void (*fn)(void*, std::size_t, std::size_t), void* ctx) {
    if (helpers_ == 0 || n < kParallelGrain) {
      fn(ctx, 0, n);
      return;
    }
    const int parts = helpers_ + 1;
    std::size_t chunk = (n + static_cast<std::size_t>(parts) - 1) / static_cast<std::size_t>(parts);
    chunk = (chunk + 63) & ~std::size_t{63};
    if (chunk == 0) {
      chunk = 64;
    }

    fn_ = fn;
    ctx_ = ctx;
    std::size_t cursor = 0;
    for (int i = 0; i < parts; ++i) {
      slots_[static_cast<std::size_t>(i)].begin = cursor;
      const std::size_t next = std::min(n, cursor + chunk);
      slots_[static_cast<std::size_t>(i)].end = next;
      cursor = next;
    }
    remaining_.store(helpers_, std::memory_order_relaxed);
    epoch_.fetch_add(1, std::memory_order_release);
    cv_.notify_all();

    const auto& mine = slots_[static_cast<std::size_t>(helpers_)];
    if (mine.end > mine.begin) {
      fn(ctx, mine.begin, mine.end);
    }
    while (remaining_.load(std::memory_order_acquire) != 0) {
#if defined(_M_X64) || defined(__x86_64__)
      _mm_pause();
#else
      std::this_thread::yield();
#endif
    }
  }

  std::vector<std::thread> workers_;
  std::vector<Slot> slots_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable cv_done_;
  void (*fn_)(void*, std::size_t, std::size_t) = nullptr;
  void* ctx_ = nullptr;
  std::atomic<std::uint64_t> epoch_{0};
  std::atomic<int> remaining_{0};
  int helpers_ = 0;
  std::atomic<bool> stop_{false};
};

ThreadPool& ThreadPool::instance() {
  static ThreadPool pool;
  return pool;
}

ThreadPool::ThreadPool() : impl_(std::make_unique<Impl>()) {}

ThreadPool::~ThreadPool() = default;

int ThreadPool::helper_count() const noexcept { return impl_->helpers_; }

void ThreadPool::parallel(std::size_t n, void (*fn)(void*, std::size_t, std::size_t), void* ctx) {
  impl_->parallel(n, fn, ctx);
}

}  // namespace nexus_optim
