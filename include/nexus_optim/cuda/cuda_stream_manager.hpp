#pragma once

#include <cstddef>

#if defined(NEXUS_OPTIM_WITH_CUDA)
#include <cuda_runtime.h>
#endif

namespace nexus_optim {
namespace cuda {

/// Small fixed pool of non-blocking streams. Created on first use (not inside a
/// parameter element loop). Elementwise kernels are deterministic: no atomics.
class StreamManager {
 public:
  static StreamManager& instance();
  void synchronize() const;

#if defined(NEXUS_OPTIM_WITH_CUDA)
  cudaStream_t stream_for(std::size_t group_index) const;
#endif

 private:
  StreamManager();
  ~StreamManager();
  StreamManager(const StreamManager&) = delete;
  StreamManager& operator=(const StreamManager&) = delete;
  struct Impl;
  Impl* impl_;
};

}  // namespace cuda
}  // namespace nexus_optim
