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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/phi/common/type_traits.h"

namespace custom_kernel {


template <typename T, typename Context>
void AbsKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  std::cout<<"hello world"<<std::endl;
}


}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(abs,
                          musa,
                          ALL_LAYOUT,
                          custom_kernel::AbsKernel,
                          float,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          double,
                          int64_t) {
  // kernel->InputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
