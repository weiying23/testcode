# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: BSD-3-Clause-Clear

cmake_minimum_required (VERSION 3.18.1)

project(SME2_MATMUL LANGUAGES C ASM)

# Set path for custom modules, and load modules.
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
)
include(SME2_MATMUL)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_C_FLAGS_RELEASE "-Wall -O2 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_DEBUG "-Wall -g -DDEBUG" CACHE STRING "" FORCE)
set(CMAKE_ASM_FLAGS_RELEASE "-Wall -O2" CACHE STRING "" FORCE)
set(CMAKE_ASM_FLAGS_DEBUG "-Wall -g" CACHE STRING "" FORCE)

if (NOT DEFINED SME2_MATMUL_BAREMETAL)
  set(SME2_MATMUL_BAREMETAL 0)
endif()
if (NOT DEFINED BASELINE_MARCH)
  set(BASELINE_MARCH "-march=armv8-a")
endif()
if (NOT DEFINED SME2_MARCH)
  set(SME2_MARCH "-march=native+sme2")
endif()
if (NOT DEFINED SME2_ASM_MARCH)
  set(SME2_ASM_MARCH "-march=native+sve+sme2")
endif()

add_sme2_executable(hello
  BASELINE_SOURCES hello.c
)

add_sme2_executable(sme2_check
  SME2_SOURCES sme2_check.c misc.c
  COMPILE_DEFINITIONS "BAREMETAL=${SME2_MATMUL_BAREMETAL}"
)

add_sme2_executable(sme2_matmul_asm
  SME2_SOURCES main.c matmul_asm.c misc.c
  ASSEMBLY_SOURCES matmul_asm_impl.S preprocess_l_asm.S
  BASELINE_SOURCES matmul_vanilla.c preprocess_vanilla.c
  COMPILE_DEFINITIONS "IMPL=asm" "BAREMETAL=${SME2_MATMUL_BAREMETAL}"
)

add_sme2_executable(sme2_matmul_intr
  SME2_SOURCES main.c matmul_intr.c misc.c
  BASELINE_SOURCES matmul_vanilla.c preprocess_vanilla.c
  COMPILE_DEFINITIONS "IMPL=intr" "BAREMETAL=${SME2_MATMUL_BAREMETAL}"
)
