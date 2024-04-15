#include "runtime/utils.h"
#include "runtime/mudnn/Handle.h"
#include "paddle/phi/extension.h"
#include "kernels/musa/contiguous_kernel.h"
namespace custom_kernel{  


void ConfigConv(
    ::musa::dnn::Convolution& c,
    const std::vector<int>& str,
    const std::vector<int>& pad,
    const std::vector<int>& dil,
    int64_t groups) {
  auto sz = str.size();
  if (sz == 2) {
    CHECK_MUDNN_STATUS(
        c.SetNdInfo(
            {static_cast<int>(pad[0]), static_cast<int>(pad[1])},
            {static_cast<int>(str[0]), static_cast<int>(str[1])},
            {static_cast<int>(dil[0]), static_cast<int>(dil[1])}),
        "SetNdInfo");
  } else {
    // conv3d
    CHECK_MUDNN_STATUS(
        c.SetNdInfo(
            {static_cast<int>(pad[0]),
             static_cast<int>(pad[1]),
             static_cast<int>(pad[2])},
            {static_cast<int>(str[0]),
             static_cast<int>(str[1]),
             static_cast<int>(str[2])},
            {static_cast<int>(dil[0]),
             static_cast<int>(dil[1]),
             static_cast<int>(dil[2])}),
        "SetNdInfo");
  }
  CHECK_MUDNN_STATUS(
      c.SetComputeMode(::musa::dnn::Convolution::ComputeMode::SCALAR),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(c.SetGroups(groups), "SetGroups");
}




template <typename T = int>
inline void UpdatePaddingAndDilation(std::vector<T>* paddings,
                                     std::vector<T>* dilation,
                                     const std::string padding_algorithm,
                                     const phi::DDim data_dims,
                                     const std::vector<T>& strides,
                                     const std::vector<T>& ksize) {
  // set padding size == data_dims.size() * 2
  auto data_shape = phi::vectorize<T>(data_dims);
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        data_dims.size() * 2,
        paddings->size(),
        phi::errors::InvalidArgument(
            "Attribute padding's size should be the same or twice as the "
            "input's dimension. "
            "But received: padding's size is %d, padding is [%s]; input's "
            "dimension is %d, input's shape is [%s].",
            paddings->size(),
            phi::make_ddim(*paddings),
            data_dims.size(),
            data_dims));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + ksize[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;

      // dilation
      *(dilation->begin() + i) = 1;
    }

  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}




template <typename T, typename Context>
void Conv2dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& filter,
                  const std::vector<int>& strides_t,
                  const std::vector<int>& paddings_t,
                  const std::string& padding_algorithm,
                  const std::vector<int>& dilations_t,
                  int groups,
                  const std::string& data_format,
                  phi::DenseTensor* output){
    auto strides = strides_t;
    auto paddings = paddings_t;
    auto dilations = dilations_t;

    const bool channel_last = data_format == "NHWC";
    auto in_dims = input.dims();
    auto filter_dims = filter.dims();
    phi::DDim in_data_dims;
    phi::DDim filter_data_dims;

    if (channel_last) {
        in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
        in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
    }
    filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    custom_kernel::UpdatePaddingAndDilation(
        &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

    auto weight_memory_format = filter.layout();
    PADDLE_ENFORCE(
       (filter.layout() == phi::DataLayout::NCHW)|| (filter.layout() == phi::DataLayout::NCDHW) || channel_last,
        phi::errors::PreconditionNotMet("Paddle musa now only support NCHW/NCDHW"));
                
    auto out_data = dev_ctx.template Alloc<T>(output);
    phi::DenseTensor contiguous_input;
    phi::DenseTensor contiguous_filter;

    if(!input.meta().is_contiguous()){
        custom_kernel::ContiguousKernel<T,Context>(dev_ctx,input,&contiguous_input);
    }else{
      contiguous_input = input;
    }
    if(!filter.meta().is_contiguous()){
        custom_kernel::ContiguousKernel<T,Context>(dev_ctx,filter,&contiguous_filter);
    }else{
      contiguous_filter = filter;
    }
    auto in = CreateMUTensor(contiguous_input);
    auto out = CreateMUTensor(*output);
    auto ke = CreateMUTensor(contiguous_filter);
    musa::muHandle& h = musa::GetMudnnHandle(dev_ctx);
    ::musa::dnn::Convolution c;
    ConfigConv(c, strides, paddings, dilations, groups);
    
    ::musa::dnn::Convolution::FusedActivationDesc act;
    act.SetMode(::musa::dnn::Convolution::FusedActivationDesc::Mode::IDENTITY);

    ::musa::dnn::Convolution::Algorithm algo =
    static_cast<::musa::dnn::Convolution::Algorithm>(0);
    c.GetRecommendForwardAlgorithm(h, algo, out, in, ke);

    muTensor bias= muTensor();
    muTensor add = muTensor();
    CHECK_MUDNN_STATUS(
      c.RunFusion(h, out, in, ke, bias, add, act, algo, InternalMemAlloc),
      "RunFusion");
}

}      


PD_REGISTER_PLUGIN_KERNEL(conv2d,
                          musa,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
