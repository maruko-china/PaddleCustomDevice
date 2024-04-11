# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License

cmake_minimum_required(VERSION 3.10)

project(paddle-musa CXX C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

option(WITH_TESTING "compile with unit testing" ON)
option(ON_INFER "compile with inference c++ lib" OFF)

set(PLUGIN_NAME "paddle-musa")
set(PLUGIN_VERSION "0.0.1")

if(NOT WITH_TESTING)
  set(NO_PADDLE_SUBMODULE ON)
endif()
include(paddle)

include_directories(${PADDLE_INC_DIR} ${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/kernels ${CMAKE_SOURCE_DIR}/runtime)
link_directories(${PADDLE_LIB_DIR})

add_definitions(-std=c++14)
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)

file(
  GLOB_RECURSE PLUGIN_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  kernels/*.cc
  runtime/*.cc
  runtime/mudnn/*.cc)

# build shared library
add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SRCS})
if(ON_INFER)
  target_link_directories(${PLUGIN_NAME} PRIVATE ${PADDLE_INFERENCE_LIB_DIR})
  target_link_libraries(${PLUGIN_NAME} PRIVATE paddle_inference)
else()
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${PADDLE_CORE_LIB})
endif()

# packing wheel package
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(
  TARGET ${PLUGIN_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMENT "Creating plugin directories------>>>")

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${PLUGIN_NAME}
  COMMENT "Packing whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)

if(WITH_TESTING)
  set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Paddle")
  enable_testing()
  add_subdirectory(tests)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp
    COMMAND cp -r ${CMAKE_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(python_tests ALL
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp)
endif()



find_package(MUDNN)

if(MUDNN_FOUND)
  list(APPEND DEPENDENT_INCLUDE_DIRS ${MUDNN_INCLUDE_DIRS})
  list(APPEND DEPENDENT_LIBRARIES ${MUDNN_LIBRARIES})
else()
  # set default mudnn library path
  message(
    WARNING
      " The environment variable MUSA_HOME may be not specified. Using default MUDNN PATH: /usr/local/musa"
  )
  list(APPEND DEPENDENT_INCLUDE_DIRS "/usr/local/musa/include")
  list(APPEND DEPENDENT_LIBRARIES "/usr/local/musa/lib/libmudnn.so")
  set(MUDNN_PATH "/usr/local/musa")
  set(MUDNN_LIBRARIES "/usr/local/musa/lib/libmudnn.so")
endif()

if(USE_MCCL)
  find_package(MCCL)

  if(MCCL_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${MCCL_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${MCCL_LIBRARIES})
  else()
    message(WARNING " NO MCCL FOUND?")
    list(APPEND DEPENDENT_INCLUDE_DIRS "/usr/local/musa/include")
    list(APPEND DEPENDENT_LIBRARIES "/usr/local/musa/lib/libmccl.so")
    set(MCCL_PATH "/usr/local/musa")
    set(MCCL_LIBRARIES "/usr/local/musa/lib/libmccl.so")
  endif()

  add_definitions(-DUSE_MCCL)
endif()

find_package(MUSAToolkits)

if(MUSAToolkits_FOUND)
  list(APPEND DEPENDENT_INCLUDE_DIRS ${MUSAToolkits_INCLUDE_DIRS})
  list(APPEND DEPENDENT_LIBRARIES ${MUSAToolkits_LIBRARIES})
else()
  # set default musa_toolkits path
  message(
    WARNING
      " The environment variable MUSA_HOME may be not specified. Using default MUSATOOLKITS PATH: /usr/local/musa"
  )
  list(APPEND DEPENDENT_INCLUDE_DIRS "/usr/local/musa/include/")
  list(APPEND DEPENDENT_LIBRARIES "/usr/local/musa/lib/libmusart.so")
  set(ENV{MUSA_HOME} "/usr/local/musa")
  set(MUSATOOLKITS_PATH "/usr/local/musa")
  set(MUSAToolkits_LIBRARIES "/usr/local/musa/lib/")
endif()


include_directories(${DEPENDENT_INCLUDE_DIRS})
target_link_libraries(${PLUGIN_NAME} PUBLIC ${DEPENDENT_LIBRARIES})


