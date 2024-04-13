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

#include "runtime/runtime.h"

#include <vector>

#include "glog/logging.h"
// #include "sucl/br_cl.h"

constexpr std::size_t br_compute_capability = 75;


C_Status set_device(const C_Device device) {
  return C_SUCCESS;
}

C_Status get_device(const C_Device device) {
  return C_SUCCESS;
}

C_Status get_device_count(size_t *count) {

  return C_SUCCESS;
}

C_Status get_device_list(size_t *device) {

  return C_SUCCESS;
}

C_Status get_compute_capability(size_t *compute_capability) {
  return C_SUCCESS;
}

C_Status get_runtime_version(size_t *version) {
  return C_SUCCESS;
}

C_Status get_driver_version(size_t *version) {
  return C_SUCCESS;
}

C_Status memcpy_h2d(const C_Device device,
                    void *dst,
                    const void *src,
                    size_t size) {
  return C_SUCCESS;
}

C_Status memcpy_d2d(const C_Device device,
                    void *dst,
                    const void *src,
                    size_t size) {
  return C_SUCCESS;
}

C_Status memcpy_d2h(const C_Device device,
                    void *dst,
                    const void *src,
                    size_t size) {

  return C_SUCCESS;
}

C_Status async_memcpy_h2d(const C_Device device,
                          C_Stream stream,
                          void *dst,
                          const void *src,
                          size_t size) {
  return C_SUCCESS;
}

C_Status async_memcpy_d2d(const C_Device device,
                          C_Stream stream,
                          void *dst,
                          const void *src,
                          size_t size) {
  return C_SUCCESS;
}

C_Status async_memcpy_d2h(const C_Device device,
                          C_Stream stream,
                          void *dst,
                          const void *src,
                          size_t size) {
  return C_SUCCESS;
}

C_Status allocate(const C_Device device, void **ptr, size_t size) {

  return C_SUCCESS;
}

C_Status deallocate(const C_Device device, void *ptr, size_t size) {
  return C_SUCCESS;
}

C_Status create_stream(const C_Device device, C_Stream *stream) {
  return C_SUCCESS;
}

C_Status destroy_stream(const C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

C_Status create_event(const C_Device device, C_Event *event) {
  return C_SUCCESS;
}

C_Status record_event(const C_Device device, C_Stream stream, C_Event event) {

  return C_SUCCESS;
}

C_Status destroy_event(const C_Device device, C_Event event) {
  return C_SUCCESS;
}

C_Status sync_device(const C_Device device) {
  return C_SUCCESS;
}

C_Status sync_stream(const C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

C_Status sync_event(const C_Device device, C_Event event) {
  return C_SUCCESS;
}

C_Status stream_wait_event(const C_Device device,
                           C_Stream stream,
                           C_Event event) {
  return C_SUCCESS;
}

C_Status memstats(const C_Device device,
                  size_t *total_memory,
                  size_t *free_memory) {
  return C_SUCCESS;
}

C_Status get_min_chunk_size(const C_Device device, size_t *size) {

  return C_SUCCESS;
}

C_Status get_max_chunk_size(const C_Device device, size_t *size) {

  return C_SUCCESS;
}

C_Status init() { return C_SUCCESS; }

C_Status deinit() { return C_SUCCESS; }

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = const_cast<char *>("musa");
  params->sub_device_type = const_cast<char *>("S4000");

  params->interface->set_device = set_device;
  params->interface->get_device = get_device;
  params->interface->create_stream = create_stream;
  params->interface->destroy_stream = destroy_stream;
  params->interface->create_event = create_event;
  params->interface->destroy_event = destroy_event;
  params->interface->record_event = record_event;
  params->interface->synchronize_device = sync_device;
  params->interface->synchronize_stream = sync_stream;
  params->interface->synchronize_event = sync_event;
  params->interface->stream_wait_event = stream_wait_event;
  params->interface->memory_copy_h2d = memcpy_h2d;
  params->interface->memory_copy_d2d = memcpy_d2d;
  params->interface->memory_copy_d2h = memcpy_d2h;
  params->interface->async_memory_copy_h2d = async_memcpy_h2d;
  params->interface->async_memory_copy_d2d = async_memcpy_d2d;
  params->interface->async_memory_copy_d2h = async_memcpy_d2h;
  params->interface->device_memory_allocate = allocate;
  params->interface->device_memory_deallocate = deallocate;
  params->interface->get_device_count = get_device_count;
  params->interface->get_device_list = get_device_list;
  params->interface->device_memory_stats = memstats;
  params->interface->device_min_chunk_size = get_min_chunk_size;
  params->interface->device_max_chunk_size = get_max_chunk_size;
  params->interface->get_compute_capability = get_compute_capability;
  params->interface->get_runtime_version = get_runtime_version;
  params->interface->get_driver_version = get_driver_version;

  params->interface->initialize = init;
  params->interface->finalize = deinit;
}
