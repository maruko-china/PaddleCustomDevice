#include "paddle/phi/core/enforce.h"
#include "paddle/phi/extension.h"
#include "runtime/mudnn/Handle.h"
#include "runtime/utils.h"
namespace custom_kernel{

using SOFTMAX_MODE = ::musa::dnn::Softmax::Mode;


template <typename T, typename Context>
void SoftmaxKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   phi::DenseTensor* out) {
    dev_ctx.template Alloc<T>(out);
    ::musa::muHandle& h = ::musa::GetMudnnHandle(dev_ctx);
    ::musa::dnn::Softmax softmax;
    auto input_m = CreateMUTensor(x);
    auto output_m = CreateMUTensor(*out);
    CHECK_MUDNN_STATUS(softmax.SetMode(SOFTMAX_MODE::SOFTMAX), "SetMode");
    CHECK_MUDNN_STATUS(softmax.SetDim(static_cast<int>(axis)), "SetDim");
    CHECK_MUDNN_STATUS(
        softmax.SetAlgorithm(::musa::dnn::Softmax::Algorithm::ACCURATE),
        "SetAlgorithm");
    CHECK_MUDNN_STATUS(softmax.Run(h, output_m, input_m), "Run");
}


}

PD_REGISTER_PLUGIN_KERNEL(softmax,
                          musa,
                          ALL_LAYOUT,
                          custom_kernel::SoftmaxKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}