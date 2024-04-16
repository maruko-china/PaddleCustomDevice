#include "runtime/utils.h"
#include "runtime/mudnn/Handle.h"
#include "paddle/phi/extension.h"

#define BINARY_OP(op_name,mode)                                     \
namespace custom_kernel{                                            \
    using BINARY_MODE = ::musa::dnn::Binary::Mode;                  \
    using UNARY_MODE = ::musa::dnn::Unary::Mode;                    \
    template <typename T, typename Context>                         \
    void op_name##Kernel(const Context& dev_ctx,                    \
                const phi::DenseTensor& x,                          \
                const phi::DenseTensor& y,                          \
                phi::DenseTensor* out) {                            \
    Binary_kernel<T>(dev_ctx,x,y,out,mode);                         \
  }                                                                 \
}                                                                   \
PD_REGISTER_PLUGIN_KERNEL(op_name,                                  \
                        musa,                                       \
                        ALL_LAYOUT,                                 \
                        custom_kernel::op_name##Kernel,             \
                        int,                                        \
                        int64_t,                                    \
                        float,                                      \
                        phi::dtype::float16,                        \
                        phi::dtype::bfloat16,                       \
                        double) {}

#define BINARY_OP_BOOL(op_name,mode)                                \
namespace custom_kernel{                                            \
    using BINARY_MODE = ::musa::dnn::Binary::Mode;                  \
    using UNARY_MODE = ::musa::dnn::Unary::Mode;                    \
    template <typename T, typename Context>                         \
    void op_name##Kernel(const Context& dev_ctx,                    \
                const phi::DenseTensor& x,                          \
                const phi::DenseTensor& y,                          \
                phi::DenseTensor* out) {                            \
    Binary_kernel<bool>(dev_ctx,x,y,out,mode);                      \
  }                                                                 \
}                                                                   \
PD_REGISTER_PLUGIN_KERNEL(op_name,                                  \
                        musa,                                       \
                        ALL_LAYOUT,                                 \
                        custom_kernel::op_name##Kernel,             \
                        int,                                        \
                        int64_t,                                    \
                        float,                                      \
                        phi::dtype::float16,                        \
                        phi::dtype::bfloat16,                       \
                        double) {}

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
      musa::muHandle& h = musa::GetMudnnHandle<Context>(dev_ctx);
      muTensor musa_self = CreateMUTensor(x);
      muTensor musa_other = CreateMUTensor(y);
      muTensor musa_out = CreateMUTensor(*out);
      ::musa::dnn::Binary bop;
      CHECK_MUDNN_STATUS(bop.SetMode(m),"SetMode");
      CHECK_MUDNN_STATUS(bop.Run(h, musa_out, musa_self, musa_other),"Run");
  }

}

BINARY_OP(add,BINARY_MODE::ADD)
BINARY_OP(multiply,BINARY_MODE::MUL)
BINARY_OP(divide,BINARY_MODE::DIV)
BINARY_OP(elementwise_pow,BINARY_MODE::POW)
BINARY_OP(maximum,BINARY_MODE::MAX)
BINARY_OP(minimum,BINARY_MODE::MIN)

BINARY_OP_BOOL(equal,BINARY_MODE::EQ)
BINARY_OP_BOOL(not_equal,BINARY_MODE::NE)
BINARY_OP_BOOL(greater_than,BINARY_MODE::GT)
BINARY_OP_BOOL(greater_equal,BINARY_MODE::GE)
BINARY_OP_BOOL(less_than,BINARY_MODE::LT)
BINARY_OP_BOOL(less_equal,BINARY_MODE::LE)