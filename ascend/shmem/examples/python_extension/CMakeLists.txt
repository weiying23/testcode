# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.18)
project(shmem_kernel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成位置无关代码
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 获取CANN相关环境变量
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

option(USE_UNIT_TEST "USE_UNIT_TEST" OFF)
message(STATUS "USE_UNIT_TEST:${USE_UNIT_TEST}")

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

add_compile_options(
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    # avoid ascendc interference
    -DL2_CACHE_HINT
    -DTILING_KEY_VAR
)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${ASCEND_HOME_PATH}/compiler/tikcpp
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof
)

link_directories(
    ${ASCEND_HOME_PATH}/lib64
    ${CMAKE_CURRENT_SOURCE_DIR}/../../install/shmem/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared_lib/output/lib
)

link_libraries(runtime stdc++ ascendcl m tiling_api platform c_sec dl nnopbase)


if(NOT DEFINED Python3_EXECUTABLE)
    message(WARNING "Python3_EXECUTABLE is not defined, using default python3")
else()
    message("Using python: ${Python3_EXECUTABLE}")
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}")
# set(Python3_SITELIB /usr/local/lib64/python3.9/site-packages)
set(Torch_DIR ${Python3_SITELIB}/torch/share/cmake/Torch)
find_package(Torch REQUIRED)

include_directories(
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_SITELIB}/torch_npu/include
    ${SHMEM_SHARED_LIB_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/shared_lib/include
)

link_directories(
    ${CMAKE_BINARY_DIR}
    ${Python3_SITELIB}/torch_npu/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/shared_lib/obj
)

link_libraries(shmem_example_kernel "${TORCH_LIBRARIES}" torch_npu tiling_api shmem)

if(BUILD_TORCH_LIB)
    set(TORCH_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/torch_bindings.cpp)
    add_library(shmem_torch
        SHARED ${TORCH_BINDINGS_SRC})
    target_link_libraries(shmem_torch PRIVATE shmem_example_kernel)
    install(TARGETS shmem_torch DESTINATION lib COMPONENT shmem_torch)
endif()