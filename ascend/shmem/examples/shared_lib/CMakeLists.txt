# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.19)
project(shmem_kernel)

# 设置C++标准
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
)

link_libraries(runtime stdc++ ascendcl m tiling_api platform c_sec dl nnopbase)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/obj)

function(shmem_add_share_lib NAME)
    add_library(${NAME}_kernel OBJECT src/kernel/${NAME}_kernel.cpp)
    set_target_properties(${NAME}_kernel PROPERTIES
        OBJECT_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/obj"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/obj"  # 兼容性保留
    )

    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-vec)
    target_include_directories(${NAME}_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/memfabric_hybrid/src/smem/include/device/
        ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/catlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/catlass/examples/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/${NAME}
    )

    add_library(${NAME}_shared SHARED src/host/${NAME}.cpp)

    set_target_properties(${NAME}_shared PROPERTIES
        OBJECT_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/obj"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/obj"
    )

    target_compile_options(${NAME}_shared PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS} -D_GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(${NAME}_shared PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/memfabric_hybrid/src/smem/include/host/
        ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/catlass/examples/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/${NAME}
    )
    target_link_libraries(${NAME}_shared shmem ${NAME}_kernel)

    set(GLOBAL_LIBS ${GLOBAL_LIBS} ${NAME}_kernel ${NAME}_shared PARENT_SCOPE)
endfunction()

foreach(EXAMPLE_KERNELS
    kv_shuffle
)
    shmem_add_share_lib(${EXAMPLE_KERNELS})
endforeach()

add_library(shmem_example_kernel SHARED src/shmem_kernel.cpp)
target_link_options(shmem_example_kernel PRIVATE --cce-fatobj-link)
foreach(kernel_lib IN LISTS GLOBAL_LIBS)
    target_sources(shmem_example_kernel PRIVATE $<TARGET_OBJECTS:${kernel_lib}>)
endforeach()

# set_target_properties(shmem_example_kernel PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/obj")
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    install(FILES ${CMAKE_BINARY_DIR}/libshmem_example_kernel.so DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/shmem_kernel.h DESTINATION include)
endif()
