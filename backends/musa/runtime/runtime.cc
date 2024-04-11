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

#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <musa_runtime.h>
#include "paddle/phi/backends/device_ext.h"
#include "runtime/runtime.h"

struct C_CCLComm_st {
  size_t rank;
  size_t nranks;
  sem_t *sig;
  sem_t *sig_2;
  std::string sig_name;
  std::string sig_2_name;
};

C_Status SetDevice(const C_Device device) {
  CHECK_MUSA(musaSetDevice(device->id));
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  int device_id = 0;
  CHECK_MUSA(musaGetDevice(&device_id));
  device->id = device_id;
  return C_SUCCESS;
}

C_Status DestroyDevice(const C_Device device) { return C_SUCCESS; }

C_Status Finalize() { return C_SUCCESS; }

C_Status GetDevicesCount(size_t *count) {
  int temp = 0;
  CHECK_MUSA(musaGetDeviceCount(&temp));
  *count = static_cast<size_t>(temp);
  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *devices) {
  size_t device_count=0;
  if(GetDevicesCount(&device_count)==C_SUCCESS){
    for(int i=0;i<device_count;i++){
      devices[i]=i;
    }
    return C_SUCCESS;
  }
  return C_FAILED;
}

C_Status MemCpyh2d(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaMemcpy(dst, src, size, musaMemcpyHostToDevice));
  return C_SUCCESS;
}

C_Status MemCpyd2d(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaMemcpy(dst, src, size, musaMemcpyDeviceToDevice));
  return C_SUCCESS;
}

C_Status MemCpyd2h(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaMemcpy(dst, src, size, musaMemcpyDeviceToHost));
  return C_SUCCESS;
}

C_Status AsyncMemCpy(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size) {
  musaError_t musaStatus;
  musaStatus = musaSetDevice(device->id);
  if (musaStatus != musaSuccess) {
        return C_FAILED;
  }

  return C_SUCCESS;
}

C_Status MemCpyP2P(const C_Device dst_device,
                   const C_Device src_device,
                   void *dst,
                   const void *src,
                   size_t size) {
    musaError_t musaStatus;
    int canAccessPeer = 0;

    musaStatus = musaDeviceCanAccessPeer(&canAccessPeer, dst_device->id, src_device->id);
    if (musaStatus != musaSuccess || canAccessPeer == 0) {
        return C_FAILED;
    }

    musaStatus = musaDeviceEnablePeerAccess(src_device->id, 0);
    if (musaStatus != musaSuccess) {
        return C_FAILED;
    }

    // 执行P2P内存复制
    musaStatus = musaMemcpyPeer(dst, dst_device->id, src, src_device->id, size);
    if (musaStatus != musaSuccess) {
        musaDeviceDisablePeerAccess(src_device->id);
        return C_FAILED;
    }

    // 禁用P2P访问
    musaStatus = musaDeviceDisablePeerAccess(src_device->id);
    if (musaStatus != musaSuccess) {
        return C_FAILED;
    }

    return C_SUCCESS;
}

C_Status AsyncMemCpyP2P(const C_Device dst_device,
                        const C_Device src_device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  musaError_t musaStatus;
  int canAccessPeer = 0;

  // 检查两个设备之间是否支持P2P访问
  musaStatus = musaDeviceCanAccessPeer(&canAccessPeer, dst_device->id, src_device->id);
  if (musaStatus != musaSuccess || canAccessPeer == 0) {
      return C_FAILED;
  }

  // 启用P2P访问
  musaStatus = musaDeviceEnablePeerAccess(src_device->id, 0);
  if (musaStatus != musaSuccess) {
      return C_FAILED;
  }

  // 执行异步P2P内存复制
  musaStatus = musaMemcpyPeerAsync(dst, dst_device->id, src, src_device->id, size, reinterpret_cast<musaStream_t>(stream));
  if (musaStatus != musaSuccess) {
      musaDeviceDisablePeerAccess(src_device->id);
      return C_FAILED;
  }

  CHECK_MUSA(musaDeviceDisablePeerAccess(src_device->id));
  return C_SUCCESS;
}

C_Status Device_Allocate(const C_Device device, void **ptr, size_t size) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaMalloc(ptr, size));
  return C_SUCCESS;
}

C_Status Host_Allocate(const C_Device device, void **ptr, size_t size) {
  auto data = malloc(size);
  if (data) {
    *ptr = data;
    return C_SUCCESS;
  } else {
    *ptr = nullptr;
  }
  return C_FAILED;
}

C_Status Host_Deallocate(const C_Device device, void *ptr, size_t size) {
  free(ptr); 
  return C_SUCCESS; 
}

C_Status Device_Deallocate(const C_Device device, void *ptr, size_t size) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaFree(ptr));
  return C_SUCCESS;
}

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaStreamCreate(reinterpret_cast<musaStream_t*>(stream)));
  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaStreamDestroy((reinterpret_cast<musaStream_t>(stream))));
  return C_SUCCESS;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaEventCreate(reinterpret_cast<musaEvent_t*>(event)));
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaEventRecord(reinterpret_cast<musaEvent_t>(event),reinterpret_cast<musaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaEventDestroy((reinterpret_cast<musaEvent_t>(event))));
  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaDeviceSynchronize());
  return C_SUCCESS; 
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaStreamSynchronize(reinterpret_cast<musaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaEventSynchronize(reinterpret_cast<musaEvent_t>(event)));
  return C_SUCCESS;  
  }

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaStreamWaitEvent(reinterpret_cast<musaStream_t>(stream),reinterpret_cast<musaEvent_t>(event)));
  return C_SUCCESS;  
}

C_Status VisibleDevices(size_t *devices) { return C_SUCCESS; }

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory) {
  CHECK_MUSA(musaSetDevice(device->id));
  CHECK_MUSA(musaMemGetInfo(free_memory,total_memory));
  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  *size = 512;
  return C_SUCCESS;
}


// for unittest
C_Status XcclGetUniqueIdSize(size_t *sz) {
  *sz = sizeof(size_t);
  return C_SUCCESS;
}

C_Status XcclGetUniqueId(C_CCLRootId *unique_id) {
  auto ptr = reinterpret_cast<int8_t *>(unique_id->data);
  for (auto i = 0; i < unique_id->sz - 1; ++i) {
    ptr[i] = static_cast<int8_t>(std::rand() % ('z' - 'a') + 'a');
  }
  ptr[unique_id->sz - 1] = '\0';
  return C_SUCCESS;
}

C_Status XcclCommInitRank(size_t ranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  auto sig = sem_open(static_cast<char *>(unique_id->data), O_CREAT, 0644, 0);
  auto sig_2 =
      sem_open(static_cast<char *>(unique_id->data) + 1, O_CREAT, 0644, 0);
  *comm =
      new C_CCLComm_st({rank,
                        ranks,
                        sig,
                        sig_2,
                        std::string(static_cast<char *>(unique_id->data)),
                        std::string(static_cast<char *>(unique_id->data) + 1)});
  return C_SUCCESS;
}

C_Status XcclDestroyComm(C_CCLComm comm) {
  if (comm) {
    sem_unlink(comm->sig_name.c_str());
    sem_unlink(comm->sig_2_name.c_str());
    delete comm;
  }
  return C_SUCCESS;
}

C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  sem_post(comm->sig);

  if (comm->rank == 0) {
    for (auto i = 0; i < comm->nranks; ++i) {
      sem_wait(comm->sig);
    }

    for (auto i = 0; i < comm->nranks; ++i) {
      sem_post(comm->sig_2);
    }
  }

  sem_wait(comm->sig_2);
  return C_SUCCESS;
}

C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  sem_post(comm->sig);

  if (comm->rank == 0) {
    for (auto i = 0; i < comm->nranks; ++i) {
      sem_wait(comm->sig);
    }

    for (auto i = 0; i < comm->nranks; ++i) {
      sem_post(comm->sig_2);
    }
  }

  sem_wait(comm->sig_2);
  return C_SUCCESS;
}

C_Status ProfilerInitialize(C_Profiler prof, void **user_data) {
  return C_SUCCESS;
}

C_Status ProfilerFinalize(C_Profiler prof, void *user_data) {
  return C_SUCCESS;
}

C_Status ProfilerPrepare(C_Profiler prof, void *user_data) { return C_SUCCESS; }

C_Status ProfilerStart(C_Profiler prof, void *user_data) { return C_SUCCESS; }

C_Status ProfilerStop(C_Profiler prof, void *user_data) { return C_SUCCESS; }

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t start_ns,
                             void *user_data) {
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = "musa";
  musaDeviceProp properties;
  musaGetDeviceProperties(&properties, 0);
  params->sub_device_type = properties.name;

  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  params->interface->finalize = Finalize;

  params->interface->set_device = SetDevice;
  params->interface->get_device = GetDevice;
  params->interface->deinit_device = DestroyDevice;

  params->interface->create_stream = CreateStream;
  params->interface->destroy_stream = DestroyStream;

  params->interface->create_event = CreateEvent;
  params->interface->destroy_event = DestroyEvent;
  params->interface->record_event = RecordEvent;

  params->interface->synchronize_device = SyncDevice;
  params->interface->synchronize_stream = SyncStream;
  params->interface->synchronize_event = SyncEvent;
  params->interface->stream_wait_event = StreamWaitEvent;

  params->interface->memory_copy_h2d = MemCpyh2d;
  params->interface->memory_copy_d2d = MemCpyd2d;
  params->interface->memory_copy_d2h = MemCpyd2h;
  params->interface->memory_copy_p2p = MemCpyP2P;
  params->interface->async_memory_copy_h2d = AsyncMemCpy;
  params->interface->async_memory_copy_d2d = AsyncMemCpy;
  params->interface->async_memory_copy_d2h = AsyncMemCpy;
  params->interface->async_memory_copy_p2p = AsyncMemCpyP2P;
  params->interface->device_memory_allocate = Device_Allocate;
  params->interface->host_memory_allocate = Host_Allocate;
  params->interface->unified_memory_allocate = Host_Allocate;
  params->interface->device_memory_deallocate = Device_Deallocate;
  params->interface->host_memory_deallocate = Host_Deallocate;
  params->interface->unified_memory_deallocate = Host_Deallocate;

  params->interface->get_device_count = GetDevicesCount;
  params->interface->get_device_list = GetDevicesList;
  params->interface->device_memory_stats = DeviceMemStats;
  params->interface->device_min_chunk_size = DeviceMinChunkSize;

  params->interface->xccl_get_unique_id_size = XcclGetUniqueIdSize;
  params->interface->xccl_get_unique_id = XcclGetUniqueId;
  params->interface->xccl_comm_init_rank = XcclCommInitRank;
  params->interface->xccl_destroy_comm = XcclDestroyComm;
  params->interface->xccl_all_reduce = XcclAllReduce;
  params->interface->xccl_broadcast = XcclBroadcast;

  params->interface->profiler_collect_trace_data = ProfilerCollectData;
  params->interface->profiler_initialize = ProfilerInitialize;
  params->interface->profiler_finalize = ProfilerFinalize;
  params->interface->profiler_start_tracing = ProfilerStart;
  params->interface->profiler_stop_tracing = ProfilerStop;
  params->interface->profiler_prepare_tracing = ProfilerPrepare;
}
