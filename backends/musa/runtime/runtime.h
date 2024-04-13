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
#include <musa_runtime.h>

#define CHECK_MUSA(func)											                    \
{																                                  \
	musaError_t status = (func);									                  \
	if (status != musaSuccess) {									                  \
		fprintf(stderr, "MUSA error at %s:%d, code=%d (%s)\n",	      \
				__FILE__, __LINE__, status, musaGetErrorString(status));  \
		return C_FAILED;										                          \
	}															                                  \
}

#define MEMORY_FRACTION 0.5f

C_Status SetDevice(const C_Device device);

C_Status GetDevice(const C_Device device);

C_Status DestroyDevice(const C_Device device);

C_Status Finalize();

C_Status GetDevicesCount(size_t *count);

C_Status GetDevicesList(size_t *devices);
C_Status MemCpyh2d(const C_Device device,
                void *dst,
                const void *src,
                size_t size);

C_Status MemCpyd2d(const C_Device device,
                void *dst,
                const void *src,
                size_t size);

C_Status MemCpyd2h(const C_Device device,
                void *dst,
                const void *src,
                size_t size);

C_Status AsyncMemCpy(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size);

C_Status MemCpyP2P(const C_Device dst_device,
                   const C_Device src_device,
                   void *dst,
                   const void *src,
                   size_t size);

C_Status AsyncMemCpyP2P(const C_Device dst_device,
                        const C_Device src_device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);

C_Status Device_Allocate(const C_Device device, void **ptr, size_t size);

C_Status Host_Allocate(const C_Device device, void **ptr, size_t size);

C_Status Host_Deallocate(const C_Device device, void *ptr, size_t size);

C_Status Device_Deallocate(const C_Device device, void *ptr, size_t size);

C_Status CreateStream(const C_Device device, C_Stream *stream);

C_Status DestroyStream(const C_Device device, C_Stream stream);

C_Status CreateEvent(const C_Device device, C_Event *event);

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event);

C_Status DestroyEvent(const C_Device device, C_Event event);
C_Status SyncDevice(const C_Device device);

C_Status SyncStream(const C_Device device, C_Stream stream);

C_Status SyncEvent(const C_Device device, C_Event event);

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event);

C_Status VisibleDevices(size_t *devices);

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory);

C_Status DeviceMinChunkSize(const C_Device device, size_t *size);



// for unittest
C_Status XcclGetUniqueIdSize(size_t *sz);
C_Status XcclGetUniqueId(C_CCLRootId *unique_id);

C_Status XcclCommInitRank(size_t ranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm);
C_Status XcclDestroyComm(C_CCLComm comm);
C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream);

C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream);

C_Status ProfilerInitialize(C_Profiler prof, void **user_data) ;

C_Status ProfilerFinalize(C_Profiler prof, void *user_data);

C_Status ProfilerPrepare(C_Profiler prof, void *user_data);

C_Status ProfilerStart(C_Profiler prof, void *user_data);

C_Status ProfilerStop(C_Profiler prof, void *user_data);

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t start_ns,
                             void *user_data);