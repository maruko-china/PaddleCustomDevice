#include <string>
#include <vector>
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/extension.h"
#include "runtime/utils.h"
#include "kernels/musa/contiguous_kernel.h"
#include "runtime/mudnn/Handle.h"

namespace custom_kernel{
template <typename T = int>
inline void UpdatePadding(std::vector<T>* paddings,
                          const bool global_pooling,
                          const bool adaptive,
                          const std::string padding_algorithm,
                          const phi::DDim data_dims,
                          const std::vector<T>& strides,
                          const std::vector<T>& kernel_size) {
  // set padding size == data_dims.size() * 2
  auto data_shape = phi::vectorize<T>(data_dims);

  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(data_dims.size() * 2,
                      paddings->size(),
                      phi::errors::InvalidArgument(
                          "Paddings size %d should be the same or twice as the "
                          "pooling size %d.",
                          paddings->size(),
                          data_dims.size() * 2));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + kernel_size[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }

  // if global_pooling == true or adaptive == true, padding will be ignore
  if (global_pooling || adaptive) {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}


template <typename T, typename Context>
void AdaptiveAvgPool2dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& in_x,
                  phi::DenseTensor* out_tensor,
                  const std::string& data_format){
    PADDLE_ENFORCE(data_format == "NCHW",phi::errors::PreconditionNotMet("Paddle musa now only support NCHW/NCDHW"));
    auto out_data = dev_ctx.template Alloc<T>(out_tensor);
    phi::DenseTensor contiguous_in_x;
    if(!in_x.meta().is_contiguous()){
        custom_kernel::ContiguousKernel<T,Context>(dev_ctx,in_x,&contiguous_in_x);
    }else{
      contiguous_in_x = in_x;
    }
    auto out = CreateMUTensor(*out_tensor);
    auto in = CreateMUTensor(contiguous_in_x);
    muTensor inds;
    ::musa::muHandle& h = ::musa::GetMudnnHandle(dev_ctx);
    ::musa::dnn::Pooling pool;
    CHECK_MUDNN_STATUS(pool.SetMode(::musa::dnn::Pooling::Mode::ADAPTIVE_AVGPOOL), "SetMode");
    CHECK_MUDNN_STATUS(
        pool.SetNdInfo(
            {0, 0},
            {0, 0},
            {0, 0},
            {0, 0}),
        "SetNdInfo");
    CHECK_MUDNN_STATUS(pool.Run(h, out, in, inds), "Run");
}


template <typename T, typename Context>
void Pool2dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& in_x,
                  const phi::IntArray& kernel_size,
                  const std::vector<int>& strides_t,
                  const std::vector<int>& paddings_t,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  phi::DenseTensor* out_tensor){
    PADDLE_ENFORCE((ceil_mode == false) && (exclusive == true) && (global_pooling == false) && (padding_algorithm == "EXPLICIT"),
    phi::errors::PreconditionNotMet("Paddle musa now does not support it for pool2d"));
    
    if(adaptive && pooling_type== "avg"){
        AdaptiveAvgPool2dKernel<T,Context>(dev_ctx,in_x,out_tensor,data_format);
        return;
    }
    if(adaptive && pooling_type== "max"){
        PADDLE_ENFORCE(false,phi::errors::PreconditionNotMet("musa does not support adaptive max!"));
        return;
    }
    phi::DenseTensor contiguous_in_x;
    if(!in_x.meta().is_contiguous()){
        custom_kernel::ContiguousKernel<T,Context>(dev_ctx,in_x,&contiguous_in_x);
    }else{
        contiguous_in_x = in_x;
    }
    PADDLE_ENFORCE(data_format == "NCHW" && contiguous_in_x.layout() == common::DataLayout::NCHW,phi::errors::PreconditionNotMet("Paddle musa now only support NCHW/NCDHW"));
    std::vector<int> ksize(kernel_size.GetData().begin(),
                            kernel_size.GetData().end());
    auto strides = strides_t;
    auto paddings = paddings_t;

    auto in_x_dims = in_x.dims();
    auto out_dims = out_tensor->dims();
    phi::DDim in_x_hw_dim;
    phi::DDim out_hw_dim;

    in_x_hw_dim = phi::slice_ddim(in_x_dims, 2, in_x_dims.size());
    out_hw_dim = phi::slice_ddim(out_dims, 2, out_dims.size());

    if (in_x_hw_dim[0] == 1 && in_x_hw_dim[1] == 1) {
        TensorCopy(dev_ctx, in_x, false, out_tensor);
        return;
    }
    auto out_data = dev_ctx.template Alloc<T>(out_tensor);
    auto out = CreateMUTensor(*out_tensor);
    auto in = CreateMUTensor(contiguous_in_x);
    muTensor inds;
    ::musa::muHandle& h = ::musa::GetMudnnHandle(dev_ctx);
    ::musa::dnn::Pooling pool;
    if(!adaptive && pooling_type == "avg"){
        CHECK_MUDNN_STATUS(pool.SetMode(::musa::dnn::Pooling::Mode::AVGPOOL_COUNT_WITHOUT_PAD), "SetMode");
    }else if(!adaptive && pooling_type == "max"){
        CHECK_MUDNN_STATUS(pool.SetMode(::musa::dnn::Pooling::Mode::MAXPOOL), "SetMode");
    }else{
        PADDLE_ENFORCE(false,phi::errors::PreconditionNotMet("unknown mode for paddle musa pool2d"));
    }
    CHECK_MUDNN_STATUS(
        pool.SetNdInfo(
            {ksize[0], ksize[1]},
            {paddings[0], paddings[1]},
            {strides[0],strides[1]},
            {1,1}),
        "SetNdInfo");
    CHECK_MUDNN_STATUS(pool.Run(h, out, in, inds), "Run");
    return;
    
}
}//custom_kernel

PD_REGISTER_PLUGIN_KERNEL(pool2d,
                          musa,
                          ALL_LAYOUT,
                          custom_kernel::Pool2dKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}