cmake_minimum_required(VERSION 3.10)

project(paddle-custom-musa CXX C)

set(PLUGIN_NAME "paddle-custom-musa")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

include(paddle)
include(version)
include(generic)
include(external/musa)
include_directories(${CMAKE_SOURCE_DIR})

option(WITH_KERNELS "compile with custom kernels" ON)
option(WITH_TESTING "compile with unit testing" OFF)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "AR tools: ${CMAKE_AR}")

add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)

# custom kernels
if(WITH_KERNELS)
  file(
    GLOB_RECURSE CUSTOM_KERNEL_SRCS
    RELATIVE ${CMAKE_SOURCE_DIR}
    kernels/*.cc)
  list(APPEND CUSTOM_MUSA_SRCS ${CUSTOM_KERNEL_SRCS})
endif()

# custom runtime
file(
  GLOB_RECURSE CUSTOM_RUNTIME_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  runtime/*.cc)
list(APPEND CUSTOM_MUSA_SRCS ${CUSTOM_RUNTIME_SRCS})

# build shared library
add_library(${PLUGIN_NAME} SHARED ${CUSTOM_MUSA_SRCS})

# link paddle shared library
include(third_party)
add_dependencies(${PLUGIN_NAME} third_party)
target_link_libraries(${PLUGIN_NAME} PRIVATE ${PADDLE_CORE_LIB})
target_link_libraries(${PLUGIN_NAME} PRIVATE pybind)
target_link_libraries(${PLUGIN_NAME} PRIVATE ${MUSA_DEPENDENT_LIBRARIES})

# testing
if(WITH_TESTING)
  set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../python")
  add_subdirectory(tests)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp
    COMMAND cp -r ${CMAKE_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(python_tests ALL
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp)
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

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${PLUGIN_NAME}
  COMMENT "Packing whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
