#include "paddle/phi/core/enforce.h"
#include "runtime/utils.h"
#include <array>
#include <mudnn.h>
using muTensor = ::musa::dnn::Tensor;
using namespace phi;
inline void SetTensorTypeAndAddr(const DenseTensor& t, muTensor& m_t) {
  DataType t_type = t.dtype();
  switch (t_type) {
    case phi::DataType::FLOAT16:
      m_t.SetType(muTensor::Type::HALF);
      break;
    case phi::DataType::FLOAT32:
      m_t.SetType(muTensor::Type::FLOAT);
      break;
    case phi::DataType::INT16:
      m_t.SetType(muTensor::Type::INT16);
      break;
    case phi::DataType::INT32:
      m_t.SetType(muTensor::Type::INT32);
      break;
    case phi::DataType::INT64:
      m_t.SetType(muTensor::Type::INT64);
      break;
    case phi::DataType::FLOAT64:
      m_t.SetType(muTensor::Type::DOUBLE);
      break;
    case phi::DataType::BOOL:
      m_t.SetType(muTensor::Type::BOOL);
      break;
    case phi::DataType::INT8:
      m_t.SetType(muTensor::Type::INT8);
      break;
    case phi::DataType::UINT8:
      m_t.SetType(muTensor::Type::UINT8);
      break;
    case phi::DataType::BFLOAT16:
      m_t.SetType(muTensor::Type::BFLOAT16);
      break;
    default:
      PADDLE_THROW(phi::errors::Unimplemented("Not support for dtype"));
      throw;
  }
  m_t.SetAddr(t.data());
}


void ConfigFormat(
    const DenseTensor& t,
    muTensor& mt,
    bool permute_if_not_contiguous) {
  const auto memory_format = t.layout();
  muTensor::Format mudnn_format = muTensor::Format::NCHW;
  DenseTensor mu_t = t;

  if(memory_format == phi::DataLayout::NCHW){
    mudnn_format = muTensor::Format::NCHW;
  }else if(memory_format == phi::DataLayout::NCDHW){
    mudnn_format = muTensor::Format::NCDHW;
  }else{
    PADDLE_THROW(phi::errors::Unimplemented("muTensor now only support NCHW/NCDHW"));
  }

  mt.SetFormat(mudnn_format);
  mt.SetNdInfo(static_cast<int64_t>(t.dims().size()), t.dims().Get(), t.strides().Get());
}

muTensor CreateMUTensor(const DenseTensor& t, bool permute_if_not_contiguous) {
  muTensor rst;
  SetTensorTypeAndAddr(t, rst);
  ConfigFormat(t, rst, permute_if_not_contiguous);
  return rst;
}
