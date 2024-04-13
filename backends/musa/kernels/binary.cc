#include "runtime/utils.h"
#include "runtime/mudnn/Handle.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {
  using BINARY_MODE = ::musa::dnn::Binary::Mode;
  using UNARY_MODE = ::musa::dnn::Unary::Mode;

  template <typename T, typename Context>
  void Binary_kernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::DenseTensor& y,
                phi::DenseTensor* out,
                const BINARY_MODE & m){
      auto out_data = dev_ctx.template Alloc<T>(out);
      musa::muHandle& h = musa::GetMudnnHandle();
      muTensor musa_self = CreateMUTensor(x);
      muTensor musa_other = CreateMUTensor(y);
      muTensor musa_out = CreateMUTensor(*out);
      ::musa::dnn::Binary bop;
      CHECK_MUDNN_STATUS(bop.SetMode(m),"SetMode");
      CHECK_MUDNN_STATUS(bop.Run(h, musa_out, musa_self, musa_other),"Run");
  }

  template <typename T, typename Context>
    void AddKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::DenseTensor& y,
                phi::DenseTensor* out) {
    std::cout<<"hello"<<std::endl;
    Binary_kernel<T>(dev_ctx,x,y,out,BINARY_MODE::ADD);
  }

}

PD_REGISTER_PLUGIN_KERNEL(add,
                          musa,
                          ALL_LAYOUT,
                          custom_kernel::AddKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}