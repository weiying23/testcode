# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# This software is provided on an "AS IS" basis, WITHOUT WARRANTIES OF ANY KIND, either express or implied,
# including but not limited to non-infringement, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.19)

# Comm Benchmark 项目
project(CommBenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 获取CANN环境
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

# 编译选项
set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -Xhost-start -ftrapv -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
)

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

# 头文件路径
include_directories(
    ${ASCEND_HOME_PATH}/compiler/tikcpp
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/../include
    ${PROJECT_SOURCE_DIR}/../install/memfabric_hybrid/include/smem/host
    ${PROJECT_SOURCE_DIR}/../install/memfabric_hybrid/include/smem/device
)

# 库路径
link_directories(
    ${ASCEND_HOME_PATH}/lib64
    ${PROJECT_SOURCE_DIR}/../install/memfabric_hybrid/lib
)

# HCCL库路径 (华为集合通信库)
find_library(HCCL_LIB hccl PATHS ${ASCEND_HOME_PATH}/lib64 NO_DEFAULT_PATH)
if(NOT HCCL_LIB)
    message(WARNING "HCCL library not found, HCCL tests will be disabled")
endif()

# ========== Kernel库 ==========
# RDMA PingPong Kernel
add_library(rdma_pingpong_kernel SHARED kernels/rdma_pingpong_kernel.cpp)
target_compile_options(rdma_pingpong_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220)
target_include_directories(rdma_pingpong_kernel PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/../include)
target_link_options(rdma_pingpong_kernel PRIVATE --cce-fatobj-link)

# RDMA Bandwidth Kernel
add_library(rdma_bw_kernel SHARED kernels/rdma_bw_kernel.cpp)
target_compile_options(rdma_bw_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220)
target_include_directories(rdma_bw_kernel PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/../include)
target_link_options(rdma_bw_kernel PRIVATE --cce-fatobj-link)

# MTE PingPong Kernel
add_library(mte_pingpong_kernel SHARED kernels/mte_pingpong_kernel.cpp)
target_compile_options(mte_pingpong_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220)
target_include_directories(mte_pingpong_kernel PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/../include)
target_link_options(mte_pingpong_kernel PRIVATE --cce-fatobj-link)

# MTE Bandwidth Kernel
add_library(mte_bw_kernel SHARED kernels/mte_bw_kernel.cpp)
target_compile_options(mte_bw_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220)
target_include_directories(mte_bw_kernel PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/../include)
target_link_options(mte_bw_kernel PRIVATE --cce-fatobj-link)

# Hidden Communication Kernel (非阻塞 + 计算)
add_library(hidden_comm_kernel SHARED kernels/hidden_comm_kernel.cpp)
target_compile_options(hidden_comm_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220)
target_include_directories(hidden_comm_kernel PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/../include)
target_link_options(hidden_comm_kernel PRIVATE --cce-fatobj-link)

# MatMul Compute Kernel (用于隐藏测试的计算负载)
add_library(matmul_compute_kernel SHARED kernels/matmul_compute_kernel.cpp)
target_compile_options(matmul_compute_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220)
target_include_directories(matmul_compute_kernel PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/../include)
target_link_options(matmul_compute_kernel PRIVATE --cce-fatobj-link)

# ========== 主程序 ==========
add_executable(comm_benchmark main.cpp)
target_compile_options(comm_benchmark PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(comm_benchmark PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/host
    ${PROJECT_SOURCE_DIR}/../include
    ${PROJECT_SOURCE_DIR}/../install/memfabric_hybrid/include/smem/host
)
target_link_libraries(comm_benchmark PRIVATE
    shmem
    rdma_pingpong_kernel
    rdma_bw_kernel
    mte_pingpong_kernel
    mte_bw_kernel
    hidden_comm_kernel
    matmul_compute_kernel
    ${PROJECT_SOURCE_DIR}/../install/memfabric_hybrid/lib/libmf_smem.so
    ascendcl
    runtime
    hccl
    stdc++
)