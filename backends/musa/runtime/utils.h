#include <array>
#include <mudnn.h>
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/extension.h"
#include "runtime/runtime.h"
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



template <typename Context>
inline void TensorCopy(const Context& dev_ctx,
                       const phi::DenseTensor& src,
                       bool blocking,
                       phi::DenseTensor* dst,
                       const phi::Place& dst_place = phi::CustomPlace()) {
  auto* src_ptr = src.data();
  const auto& src_place = src.place();
  if (src_ptr == nullptr) {
    PADDLE_ENFORCE(false, phi::errors::InvalidArgument("src not be nullptr in TensorCopy"));
  }
  auto dst_place_ = dst_place;

  if (&src == dst) {
    PADDLE_ENFORCE(false, phi::errors::InvalidArgument("src and dst should not be the same in TensorCopy"));
    return;
  }

  VLOG(3) << "TensorCopy " << src.dims() << " from " << src_place << " to "
          << dst_place_;

  //allocate memory for dst tensor
  dst->Resize(src.dims());
  void* dst_ptr = nullptr;
  if (dst_place_.GetType() != phi::AllocationType::CPU) {
    dst_ptr = dev_ctx.Alloc(dst, src.dtype());
  } else {
    dst_ptr = dev_ctx.HostAlloc(dst, src.dtype());
  }

  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;

  C_Stream stream = static_cast<C_Stream>(dev_ctx.stream());

  auto size =
      (src.dims().size() != 0 ? src.numel() : 1) * phi::SizeOf(src.dtype());
  if (UNLIKELY(size) == 0) {
    return;
  }

  if (src_place.GetType() == phi::AllocationType::CPU &&
      dst_place_.GetType() == phi::AllocationType::CUSTOM) {
    C_Device_st device{dst_place_.GetDeviceId()};
    AsyncMemCpyh2d(&device, stream, dst_ptr, src_ptr, size);
    if (blocking) {
      dev_ctx.Wait();
    }
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place_.GetType() == phi::AllocationType::CPU) {
    C_Device_st device{src_place.GetDeviceId()};
    AsyncMemCpyd2h(&device, stream, dst_ptr, src_ptr, size);
    if (blocking) {
      dev_ctx.Wait();
    }
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place_.GetType() == phi::AllocationType::CUSTOM) {
    if (src_place.GetDeviceType() == dst_place_.GetDeviceType()) {
      if (src_place.GetDeviceId() == dst_place_.GetDeviceId()) {
        C_Device_st device{src_place.GetDeviceId()};
        AsyncMemCpyd2d(&device, stream, dst_ptr, src_ptr, size);
        if (blocking) {
          dev_ctx.Wait();
        }
      } else {
        C_Device_st src_device{src_place.GetDeviceId()};
        C_Device_st dst_device{dst_place_.GetDeviceId()};
        AsyncMemCpyP2P(&dst_device,&src_device, stream, dst_ptr, src_ptr, size);
        if (blocking) {
          dev_ctx.Wait();
        }
      }
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("TensorCopy is not supported."));
    }
  } else if (src_place.GetType() == phi::AllocationType::CPU &&
             dst_place_.GetType() == phi::AllocationType::CPU) {
    std::memcpy(dst_ptr, src_ptr, size);
  }
}