#include "glog/logging.h"
#include "paddle/phi/extension.h"
#include <iostream>
namespace musa {

// Abs forward kernel
template <typename T, typename Context>
void AbsKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
    std::cout<<"hello world"<<std::endl;
}


}  // namespace supa

PD_REGISTER_PLUGIN_KERNEL(abs, musa, ALL_LAYOUT, musa::AbsKernel, float) {}

