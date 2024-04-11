#ifndef PADDLE_MUSA_CSRC_ATEN_MUDNN_HANDLE_H
#define PADDLE_MUSA_CSRC_ATEN_MUDNN_HANDLE_H
#include <array>
#include "mudnn.h"

namespace musa {

using mudnnHandle_t = ::musa::dnn::Handle*;
using muHandle = ::musa::dnn::Handle;
::musa::dnn::Handle& GetMudnnHandle();

} // namespace at

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_HANDLE_H
