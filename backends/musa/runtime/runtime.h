// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

// Copyright©2020-2023 Shanghai Biren Technology Co., Ltd. All rights reserved.

#pragma once

#include <cstdio>

#include "paddle/phi/extension.h"

C_Status memcpy_h2d(const C_Device device,
                    void *dst,
                    const void *src,
                    size_t size);

C_Status memcpy_d2d(const C_Device device,
                    void *dst,
                    const void *src,
                    size_t size);

C_Status memcpy_d2h(const C_Device device,
                    void *dst,
                    const void *src,
                    size_t size);

C_Status async_memcpy_h2d(const C_Device device,
                          C_Stream stream,
                          void *dst,
                          const void *src,
                          size_t size);

C_Status async_memcpy_d2d(const C_Device device,
                          C_Stream stream,
                          void *dst,
                          const void *src,
                          size_t size);

C_Status async_memcpy_d2h(const C_Device device,
                          C_Stream stream,
                          void *dst,
                          const void *src,
                          size_t size);
