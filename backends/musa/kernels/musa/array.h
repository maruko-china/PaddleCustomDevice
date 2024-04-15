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

#pragma once

#include <cstdint>

#include "paddle/common/enforce.h"
#include "kernels/musa//unroll_array_ops.h"

namespace musa_common {

template <typename T, size_t N>
class Array {
 public:
  static constexpr size_t kSize = N;

  __host__ __device__ inline Array() {}

  template <typename... Args>
  __host__ __device__ inline explicit Array(const T &val, Args... args) {
    static_assert(N == sizeof...(Args) + 1, "Invalid argument");
    UnrollVarArgsAssign<T>::Run(data_, val, args...);
  }

  __host__ __device__ inline void Fill(const T &val) {
    UnrollFillConstant<N>::Run(data_, val);
  }

  __host__ __device__ inline const T *Get() const { return data_; }

  __host__ __device__ inline T *GetMutable() { return data_; }

  __host__ __device__ inline T &operator[](size_t i) { return *advance(data_, i); }

  // Writing "return data_[i]" would cause compilation warning/error:
  // "array subscript is above array bound" in Python 35 CI.
  // It seems that it is a false warning of GCC if we do not check the bounds
  // of array index. But for better performance, we do not check in operator[]
  // like what is in STL. If users want to check the bounds, use at() instead
  __host__ __device__ inline const T &operator[](size_t i) const {
    return *advance(data_, i);
  }

  __host__ __device__ inline T &at(size_t i) {
    return (*this)[i];
  }

  __host__ __device__ inline const T &at(size_t i) const {
    return (*this)[i];
  }

  __host__ __device__ constexpr size_t size() const { return N; }

  __host__ __device__ inline bool operator==(const Array<T, N> &other) const {
    return UnrollCompare<N>::Run(data_, other.data_);
  }

  __host__ __device__ inline bool operator!=(const Array<T, N> &other) const {
    return !(*this == other);
  }

 private:
  template <typename U>
  __host__ __device__ static inline U *advance(U *ptr, size_t i) {
    return ptr + i;
  }

  T data_[N] = {};
};

template <typename T>
class Array<T, 0> {
 public:
  static constexpr size_t kSize = 0;

  __host__ __device__ inline Array() {}

  __host__ __device__ inline void Fill(const T &val) {}

  __host__ __device__ inline constexpr T *Get() const { return nullptr; }

  // Add constexpr to GetMutable() cause warning in MAC
  __host__ __device__ inline T *GetMutable() { return nullptr; }

  __host__ __device__ inline T &operator[](size_t) {
    static T obj{};
    return obj;
  }

  __host__ __device__ inline const T &operator[](size_t) const {
    static const T obj{};
    return obj;
  }

  __host__ __device__ inline T &at(size_t i) { return (*this)[i]; }

  __host__ __device__ inline const T &at(size_t i) const { return (*this)[i]; }

  __host__ __device__ constexpr size_t size() const { return 0; }

  __host__ __device__ constexpr bool operator==(const Array<T, 0> &other) const {
    return true;
  }

  __host__ __device__ constexpr bool operator!=(const Array<T, 0> &other) const {
    return false;
  }
};

}  // namespace common

namespace musa {
template <typename T, size_t N>
using Array = musa_common::Array<T, N>;
}  // namespace phi
