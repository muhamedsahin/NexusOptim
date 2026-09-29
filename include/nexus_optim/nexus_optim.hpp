#pragma once

/// Umbrella header. Pulls in every CPU algorithm, scheduler, and gradient helper.
/// The MatrixFlash adapter is opt-in (`adapters/matrixflash.hpp`) because it needs
/// that library's include path. CUDA entry points are compiled only with
/// `NEXUS_OPTIM_WITH_CUDA`.

#include "nexus_optim/algorithms/adabelief.hpp"
#include "nexus_optim/algorithms/adadelta.hpp"
#include "nexus_optim/algorithms/adagrad.hpp"
#include "nexus_optim/algorithms/adam.hpp"
#include "nexus_optim/algorithms/adamw.hpp"
#include "nexus_optim/algorithms/lamb.hpp"
#include "nexus_optim/algorithms/lars.hpp"
#include "nexus_optim/algorithms/lion.hpp"
#include "nexus_optim/algorithms/lookahead_wrapper.hpp"
#include "nexus_optim/algorithms/momentum_sgd.hpp"
#include "nexus_optim/algorithms/nadam.hpp"
#include "nexus_optim/algorithms/nesterov_sgd.hpp"
#include "nexus_optim/algorithms/radam.hpp"
#include "nexus_optim/algorithms/rmsprop.hpp"
#include "nexus_optim/algorithms/sgd.hpp"
#include "nexus_optim/cuda/mixed_precision.cuh"
#include "nexus_optim/processing/gradient_accumulation.hpp"
#include "nexus_optim/processing/gradient_centralization.hpp"
#include "nexus_optim/processing/gradient_clipping.hpp"
#include "nexus_optim/scheduler/cosine_annealing_lr.hpp"
#include "nexus_optim/scheduler/cosine_warm_restarts.hpp"
#include "nexus_optim/scheduler/exponential_lr.hpp"
#include "nexus_optim/scheduler/linear_warmup.hpp"
#include "nexus_optim/scheduler/one_cycle_lr.hpp"
#include "nexus_optim/scheduler/reduce_lr_on_plateau.hpp"
#include "nexus_optim/scheduler/step_lr.hpp"

namespace nexus_optim {
using OptimVersion = decltype(version());
}
