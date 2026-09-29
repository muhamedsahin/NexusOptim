#include "nexus_optim/algorithms/adabelief.hpp"
#include "nexus_optim/algorithms/adadelta.hpp"
#include "nexus_optim/algorithms/adagrad.hpp"
#include "nexus_optim/algorithms/adam.hpp"
#include "nexus_optim/algorithms/lamb.hpp"
#include "nexus_optim/algorithms/lars.hpp"
#include "nexus_optim/algorithms/lion.hpp"
#include "nexus_optim/algorithms/nadam.hpp"
#include "nexus_optim/algorithms/radam.hpp"
#include "nexus_optim/algorithms/rmsprop.hpp"
#include "nexus_optim/algorithms/sgd.hpp"
#include "nexus_optim/core/half.hpp"

namespace nexus_optim {

template class SgdImpl<float, SgdVariant::Plain>;
template class SgdImpl<float, SgdVariant::Momentum>;
template class SgdImpl<float, SgdVariant::Nesterov>;
template class SgdImpl<double, SgdVariant::Plain>;
template class SgdImpl<float16, SgdVariant::Plain>;

template class AdamImpl<float, false>;
template class AdamImpl<float, true>;
template class AdamImpl<double, true>;
template class AdamImpl<float16, true>;
template class AdamImpl<bfloat16, false>;

template class Adagrad<float>;
template class Adagrad<double>;
template class RMSProp<float>;
template class Adadelta<float>;
template class NAdam<float>;
template class RAdam<float>;
template class AdaBelief<float>;
template class LAMB<float>;
template class LARS<float>;
template class Lion<float>;
template class Lion<double>;

}  // namespace nexus_optim
