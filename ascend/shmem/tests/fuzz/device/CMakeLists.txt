# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)

file(GLOB_RECURSE FUZZ_KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(shmem_fuzztest_device SHARED ${FUZZ_KERNEL_FILES} $<TARGET_OBJECTS:shmem_device>)
target_link_options(shmem_fuzztest_device PRIVATE --cce-fatobj-link)
target_compile_options(shmem_fuzztest_device PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220)
target_include_directories(shmem_fuzztest_device PRIVATE
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/install/memfabric_hybrid/include/smem/device
    ${PROJECT_SOURCE_DIR}/tests/fuzz/include
    ${PROJECT_SOURCE_DIR}/tests/fuzz/include/utils
)
