# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
add_subdirectory(device)
add_subdirectory(host)
add_subdirectory(include)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

add_executable(shmem_unittest $<TARGET_OBJECTS:shmem_unittest_host>)
target_link_directories(shmem_unittest PRIVATE
    ${PROJECT_SOURCE_DIR}/install/memfabric_hybrid/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib
)
target_link_libraries(shmem_unittest PRIVATE shmem_unittest_device gtest gcov mf_smem shmem_unittest_include)
