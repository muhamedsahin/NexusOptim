#pragma once

/// Mixed precision contract.
///
/// MatrixFlash stores compute in FP32. When NexusOptim is asked to update FP16 or
/// BF16 parameter storage it keeps two FP32 masters, both allocated in the optimizer
/// constructor:
///   - the parameter master (Apex / DeepSpeed style; the rounded storage is only a view)
///   - the moment master (m and v are never stored in the narrow type)
/// `step()` writes the rounded value back to the caller's buffer and does not allocate.

#include "nexus_optim/core/half.hpp"

namespace nexus_optim {

template <typename Storage>
inline float promote(Storage value) {
  return load_as_float(value);
}

template <typename Storage>
inline Storage demote(float value) {
  return store_from_float<Storage>(value);
}

}  // namespace nexus_optim
