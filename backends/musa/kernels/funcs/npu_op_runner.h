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

#pragma once

#include <dlfcn.h>

#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_op_prepare.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/extension.h"
#include "paddle/utils/blank.h"
#include "paddle/utils/variant.h"
