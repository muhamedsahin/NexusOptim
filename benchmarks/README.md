# Benchmarks

Release build, host SIMD. Times below are from one Windows machine whose CPU advertises AVX2 and does not advertise AVX-512. The AVX-512 objects are still compiled; dispatch will not call them here.

| Workload | Scalar | AVX2 | AVX-512 |
|---|---|---|---|
| AdamW, 1,000,000 parameters | 1712 µs/step, 16.4 GB/s | 899 µs/step, 31.1 GB/s | CPU does not support it |
| AdamW, 8,000,000 parameters | 12.4 ms/step, 18.1 GB/s | 9.3 ms/step, 24.1 GB/s | CPU does not support it |
| SGD, 1,000,000 parameters | not timed separately | 176 µs/step, 68.2 GB/s | CPU does not support it |

AdamW traffic model is 28 bytes per element (parameter read/write, gradient read, first moment read/write, second moment read/write). SGD model is 12 bytes per element.

Profiles with `perf`, VTune, or Nsight were not collected in this tree. The elementwise fused loops are the hot path; for large tensors the two CPU paths converge because both are limited by memory traffic and by `sqrt`/`div` latency, not by the scheduler.

CUDA kernels (SGD and fused Adam/AdamW) are built only with `-DNEXUS_OPTIM_WITH_CUDA=ON`. `bench_cpu_vs_gpu` prints whether that flag was on.
