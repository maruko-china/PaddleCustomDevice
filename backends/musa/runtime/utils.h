#include <array>
#include <mudnn.h>
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/extension.h"

using muTensor = ::musa::dnn::Tensor;
#define CHECK_MUDNN_STATUS(rst, msg)       \
  PADDLE_ENFORCE_EQ(                             \
      rst , ::musa::dnn::Status::SUCCESS, \
      std::string("MUDNN failed in: ") + std::string(__FUNCTION__) + ": " + std::string(msg));

inline void SetTensorTypeAndAddr(const phi::DenseTensor& t, muTensor& m_t);

void ConfigFormat(
    const phi::DenseTensor& t,
    muTensor& mt,
    bool permute_if_not_contiguous);



muTensor CreateMUTensor(const phi::DenseTensor& t, bool permute_if_not_contiguous=true);

void MemFree(void* ptr);

::musa::dnn::MemoryHandler InternalMemAlloc(size_t s);
