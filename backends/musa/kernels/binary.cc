// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/phi/capi/all.h"
#include "runtime/utils.h"
#include "runtime/mudnn/Handle.h"
namespace custom_kernel {
  using BINARY_MODE = ::musa::dnn::Binary::Mode;
  using UNARY_MODE = ::musa::dnn::Unary::Mode;

  template <typename T>
  void AddKernel(const phi::Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::DenseTensor& y,
                phi::DenseTensor* out) {
      auto out_data = dev_ctx.template Alloc<T>(out);

      musa::muHandle& h = musa::GetMudnnHandle();
      muTensor musa_self = CreateMUTensor(x);
      muTensor musa_other = CreateMUTensor(y);
      muTensor musa_out = CreateMUTensor(*out);
      BINARY_MODE m = BINARY_MODE::ADD;
      ::musa::dnn::Binary bop;
      CHECK_MUDNN_STATUS(bop.SetMode(m),"SetMode");
      CHECK_MUDNN_STATUS(bop.Run(h, musa_out, musa_self, musa_other),"Run");
  }

  // template <typename T>
  // void MultiplyKernel(const phi::Context& dev_ctx,
  //                     const phi::DenseTensor& x,
  //                     const phi::DenseTensor& y,
  //                     phi::DenseTensor* out) {
  //   auto out_data = dev_ctx.template Alloc<T>(out);
  //   musa::muHandle& h = musa::GetMudnnHandle();
  //   muTensor musa_self = CreateMUTensor(x);
  //   muTensor musa_other = CreateMUTensor(y);
  //   muTensor musa_out = CreateMUTensor(*out);
  //   BINARY_MODE m = BINARY_MODE::MUL;
  //   ::musa::dnn::Binary bop;
  //   CHECK_MUDNN_STATUS(bop.SetMode(m),"SetMode");
  //   CHECK_MUDNN_STATUS(bop.Run(h, musa_out, musa_self, musa_other),"Run");
// }

}  // namespace custom_kernel



PD_BUILD_PHI_KERNEL(add,
                    musa,
                    ALL_LAYOUT,
                    custom_kernel::AddKernel,
                    int32_t,
                    int64_t,
                    float,
                    double) {}