# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

file(GLOB_RECURSE SHMEM_HOST_FILES ${PROJECT_SOURCE_DIR}/src/host/*.cpp)
list(FILTER SHMEM_HOST_FILES EXCLUDE REGEX "python_wrapper")

file(GLOB_RECURSE FUZZ_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(shmem_fuzztest_host OBJECT ${SHMEM_HOST_FILES} ${FUZZ_SOURCE_FILES})
target_compile_options(shmem_fuzztest_host PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_compile_options(shmem_fuzztest_host PRIVATE
    -g -fprofile-arcs -ftest-coverage
    -fsanitize-coverage=trace-pc
)
target_include_directories(shmem_fuzztest_host PRIVATE
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/install/memfabric_hybrid/include/smem/host
    ${PROJECT_SOURCE_DIR}/src/host
    ${PROJECT_SOURCE_DIR}/src/device
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include
    ${PROJECT_SOURCE_DIR}/3rdparty/secodefuzz/include
    ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
    ${PROJECT_SOURCE_DIR}/tests/fuzz/include
)
