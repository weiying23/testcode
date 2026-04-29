# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file be in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Comm Benchmark - 独立构建配置
# 支持无MPI环境编译运行

# 编译kernel库
add_library(comm_benchmark_kernel SHARED comm_benchmark_kernel.cpp)
target_compile_options(comm_benchmark_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    ${CCE_AICORE_ARCH_VEC}
    ${SANITIZER_FLAGS}
)
target_include_directories(comm_benchmark_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
    ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
    ${PROJECT_SOURCE_DIR}/examples/comm_benchmark
    ${PROJECT_SOURCE_DIR}/examples/utils
    ${PROJECT_SOURCE_DIR}/src/device/
    ${PROJECT_SOURCE_DIR}/src/device_simt/
    ${PROJECT_SOURCE_DIR}/src/
    $ENV{ASCEND_HOME_PATH}/tools/mssanitizer/include
)
target_link_options(comm_benchmark_kernel PRIVATE
    --cce-fatobj-link
    ${CCE_AICORE_ARCH_VEC}
    ${SANITIZER_FLAGS}
)

# 仅当USE_EXAMPLES开启时编译可执行文件
if(USE_EXAMPLES)
    add_executable(comm_benchmark main.cpp)
    target_compile_options(comm_benchmark PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS} ${MPI_CXX_COMPILE_FLAGS})
    target_include_directories(comm_benchmark PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
        ${PROJECT_SOURCE_DIR}/examples/comm_benchmark
        ${PROJECT_SOURCE_DIR}/examples/templates/include
        ${PROJECT_SOURCE_DIR}/examples/utils
        ${PROJECT_SOURCE_DIR}/src/device/
        ${PROJECT_SOURCE_DIR}/src/device_simt/
        ${PROJECT_SOURCE_DIR}/src/host
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/include
        ${PROJECT_SOURCE_DIR}/src/host/mem/heap
        ${MPI_INCLUDE_PATH}
    )
    target_link_libraries(comm_benchmark PRIVATE shmem comm_benchmark_kernel)
    if(DEFINED ENABLE_HCCL AND ENABLE_HCCL)
        target_link_libraries(comm_benchmark PRIVATE hccl)
    endif()
endif()
