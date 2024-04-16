#include "runtime/utils.h"
#include "runtime/mudnn/Handle.h"
#include "paddle/phi/extension.h"
#include "kernels/musa/contiguous_kernel.h"



#define ACTIVATION_OP(op_name,mode)                                                 \
namespace custom_kernel {                                                           \
template <typename T, typename Context>                                             \
void op_name##Kernel(const Context& dev_ctx,                                        \
               const phi::DenseTensor& x,                                           \
               phi::DenseTensor* out_tensor) {                                      \
        auto out_data = dev_ctx.template Alloc<T>(out_tensor);                      \
        auto in = CreateMUTensor(x);                                                \
        auto out = CreateMUTensor(*out_tensor);                                     \
        ::musa::dnn::Unary op;                                                      \
        CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");                            \
        CHECK_MUDNN_STATUS(op.SetAlpha(static_cast<int64_t>(0)),"SetAlpha");        \
        CHECK_MUDNN_STATUS(op.SetBeta(static_cast<int64_t>(0)),"SetBeta");          \
        ::musa::muHandle& h = ::musa::GetMudnnHandle(dev_ctx);                      \
        CHECK_MUDNN_STATUS(op.Run(h, out, in), std::string("Run") + __func__);      \
    }                                                                               \
}                                                                                   \
PD_REGISTER_PLUGIN_KERNEL(op_name,                                                  \
                          musa,                                                     \
                          ALL_LAYOUT,                                               \
                          custom_kernel::op_name##Kernel,                           \
                          float,                                                    \
                          double,                                                   \
                          phi::dtype::float16,                                      \
                          phi::dtype::bfloat16) {}


ACTIVATION_OP(relu, ::musa::dnn::Unary::Mode::RELU)
ACTIVATION_OP(silu, ::musa::dnn::Unary::Mode::SILU)
ACTIVATION_OP(sqrt, ::musa::dnn::Unary::Mode::SQRT)
ACTIVATION_OP(round, ::musa::dnn::Unary::Mode::ROUND)
ACTIVATION_OP(rsqrt, ::musa::dnn::Unary::Mode::RSQRT)
ACTIVATION_OP(hardSwish, ::musa::dnn::Unary::Mode::HARDSWISH)
ACTIVATION_OP(tanh, ::musa::dnn::Unary::Mode::TANH)
ACTIVATION_OP(sigmoid, ::musa::dnn::Unary::Mode::SIGMOID)
ACTIVATION_OP(exp, ::musa::dnn::Unary::Mode::EXP)
ACTIVATION_OP(sin, ::musa::dnn::Unary::Mode::SIN)
ACTIVATION_OP(cos, ::musa::dnn::Unary::Mode::COS)
ACTIVATION_OP(abs, ::musa::dnn::Unary::Mode::ABS)
ACTIVATION_OP(atan, ::musa::dnn::Unary::Mode::ATAN)
ACTIVATION_OP(log, ::musa::dnn::Unary::Mode::LOG)
ACTIVATION_OP(log10, ::musa::dnn::Unary::Mode::LOG10)
ACTIVATION_OP(log2, ::musa::dnn::Unary::Mode::LOG2)
ACTIVATION_OP(floor, ::musa::dnn::Unary::Mode::FLOOR)