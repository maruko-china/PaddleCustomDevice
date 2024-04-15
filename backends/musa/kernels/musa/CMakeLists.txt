cmake_minimum_required(VERSION 3.10)

project(musa_kernel CXX C)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(MUSA_KERNEL "musa_kernel")

list(APPEND CMAKE_MODULE_PATH $ENV{MUSA_HOME}/cmake)
list(APPEND MUSA_MCC_FLAGS --cuda-gpu-arch=mp_22)
find_package(MUSA REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)

# compile *.cu kernels
file(
  GLOB_RECURSE
  MU_SRCS
  ${CMAKE_SOURCE_DIR}/kernels/musa/*.mu
)

message(STATUS "MKLDNN_INC_DIR:${MKLDNN_INC_DIR}")
musa_include_directories(${MKLDNN_INC_DIR})
musa_add_library(musa_kernel SHARED ${MU_SRCS})

# link paddle shared library
add_dependencies(${MUSA_KERNEL} third_party)
target_link_libraries(${MUSA_KERNEL} ${PADDLE_CORE_LIB})
target_link_libraries(${MUSA_KERNEL} pybind)
target_link_libraries(${MUSA_KERNEL} ${MUSA_DEPENDENT_LIBRARIES})
