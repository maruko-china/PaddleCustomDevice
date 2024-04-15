#include "runtime/utils.h"
#include "runtime/mudnn/Handle.h"
#include "paddle/phi/extension.h"
#include "kernels/musa/contiguous_kernel.h"

namespace custom_kernel {
using namespace musa;
template <typename T, typename Context>
void BmmKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out)  {

    auto out_data = dev_ctx.template Alloc<T>(out);
    phi::DenseTensor contiguous_x;
    phi::DenseTensor contiguous_y;
    muHandle& h = GetMudnnHandle(dev_ctx);
    if(!x.meta().is_contiguous()){
        custom_kernel::ContiguousKernel<T,Context>(dev_ctx,x,&contiguous_x);
    }else{
        contiguous_x = x;
    }
    if(!y.meta().is_contiguous()){
        custom_kernel::ContiguousKernel<T,Context>(dev_ctx,y,&contiguous_y);
    }else{
        contiguous_y = y;
    }
    auto lmt = CreateMUTensor(contiguous_x);
    auto rmt = CreateMUTensor(contiguous_y);

    auto rst = CreateMUTensor(*out);
    ::musa::dnn::BatchMatMul b_mm;
    CHECK_MUDNN_STATUS(
        b_mm.SetComputeMode(::musa::dnn::Convolution::ComputeMode::SCALAR),
        "SetComputeMode");
    CHECK_MUDNN_STATUS(b_mm.SetTranspose(false, false), "SetTranspose");
    CHECK_MUDNN_STATUS(b_mm.Run(h, rst, lmt, rmt, InternalMemAlloc), "Run");

}


}// namespace musa

PD_REGISTER_PLUGIN_KERNEL(bmm,
                          musa,
                          ALL_LAYOUT,
                          custom_kernel::BmmKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16 ) {}