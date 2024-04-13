#include "runtime/mudnn/Handle.h"
#include "runtime/mudnn/DeviceThreadHandles.h"
#include "runtime/runtime.h"
#include "paddle/phi/core/enforce.h"


namespace musa {
namespace {

void CreateMuDNNHandle(mudnnHandle_t* handle) {
    PADDLE_ENFORCE_NOT_NULL(
        handle, ::common::errors::InvalidType("Handle pointer is no-nullptr"));
    int device;
    PADDLE_ENFORCE_EQ(musaGetDevice(&device),musaSuccess, phi::errors::Fatal("Failed to get device!"));
    PADDLE_ENFORCE_GE(device , 0, phi::errors::Fatal("Device number should not be zero!"));
    *handle = new musa::muHandle(device);
}

void DestroyMuDNNHandle(mudnnHandle_t /*handle*/) {
  // this is because of something dumb in the ordering of
  // destruction. Sometimes atexit, the musa context (or something)
  // would already be destroyed by the time this gets destroyed. It
  // happens in fbcode setting. Not destroy the handle as a workaround.
}

using MudnnPoolType = ::musa::DeviceThreadHandlePool<
    mudnnHandle_t,
    CreateMuDNNHandle,
    DestroyMuDNNHandle>;

} // namespace

::musa::dnn::Handle& GetMudnnHandle() {
  int device;
  PADDLE_ENFORCE_EQ(musaGetDevice(&device),musaSuccess, phi::errors::Fatal("Failed to get device!"));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<MudnnPoolType>();
  thread_local std::unique_ptr<MudnnPoolType::PoolWindow> myPoolWindow(
      pool->NewPoolWindow());

  mudnnHandle_t handle = myPoolWindow->reserve(device);
  //handle->SetStream(c10::musa::getCurrentMUSAStream());
  return *handle;
}

} // namespace at