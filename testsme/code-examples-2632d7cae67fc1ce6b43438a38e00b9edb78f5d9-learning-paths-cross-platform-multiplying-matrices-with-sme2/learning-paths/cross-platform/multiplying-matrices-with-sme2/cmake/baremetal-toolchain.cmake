# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: BSD-3-Clause-Clear

set (ATFE_VERSION "21.1.1")

if (DEFINED ENV{SME2_MATMUL_DOCKER})
  # We are executing in the SME2 Matmul docker container, the toolchain path is known.
  # SME2_MATMUL_DOCKER is an environment variable defined in the LP docker container.
  set(ATFE_ROOT "/tools/ATfE-${ATFE_VERSION}-Linux-AArch64")
  set(ATFE_PREFIX "${ATFE_ROOT}/bin/")
  message(STATUS "Using ATfE from: ${ATFE_ROOT}")
else()
  # We are executing on the host machine. If the user has provided us with a
  # ATFE_ROOT, then we are all set. If not, then all bets are off as ATfE might
  # have been installed anywhere. Check the user provided ATFE_ROOT and give a
  # try at a few platform specific locations and bail out if none of them works.
  if (NOT DEFINED ATFE_ROOT)
    if (APPLE)
      set(ATFE_ROOTS "/Applications/ATfE-${ATFE_VERSION}-Darwin-universal/")
    elseif (LINUX)
      set(ATFE_ROOTS "/opt/ATfE-${ATFE_VERSION}-Linux-AArch64/;/opt/ATfE-${ATFE_VERSION}-Linux-x86_64/")
    else()
      message(FATAL_ERROR "ATFE_ROOT was not set and host platform not yet supported by this toolchain file !")
    endif()
  endif()
  foreach (D IN ITEMS ${ATFE_ROOT} ${ATFE_ROOTS})
    if (EXISTS "${D}")
      set(ATFE_ROOT "${D}")
      set(ATFE_PREFIX "${D}/bin/")
      message(STATUS "Using ATfE from: ${ATFE_ROOT}")
      break()
    endif()
  endforeach()
endif()

if (NOT DEFINED ATFE_ROOT)
  message(FATAL_ERROR "No ATFE_ROOT discovered by this toolchain file. Clean your build directory and reinvoke with -DATFE_ROOT:STRING=... to point to the ATfE toolchain ")
else()
  set(ATFE_ROOT "${ATFE_ROOT}" CACHE PATH "Path to Arm Toolchain for Embedded" FORCE)
  list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES ATFE_ROOT)
endif()

# The name of the target operating system
set(CMAKE_SYSTEM_NAME Generic)

# Which C and C++ compiler to use.
# Note that ATFE_PREFIX is expected to end with the directory separator.
set(CMAKE_C_COMPILER ${ATFE_PREFIX}clang)
set(CMAKE_CXX_COMPILER ${ATFE_PREFIX}clang++)
set(CMAKE_OBJDUMP ${ATFE_PREFIX}llvm-objdump)

set(SME2_MATMUL_BAREMETAL 1)
set(BASELINE_MARCH "")
set(SME2_MARCH "-march=armv9.4-a+sme2")
set(SME2_ASM_MARCH "-march=armv9.4-a+sme2")

set(CMAKE_C_FLAGS_INIT "--target=aarch64-none-elf -fno-exceptions -fno-rtti -mno-unaligned-access")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-nostartfiles -lcrt0-semihost -lsemihost -Wl,--defsym=__boot_flash=0x80000000 -Wl,--defsym=__flash=0x80001000 -Wl,--defsym=__ram=0x81000000 -T picolibc.ld")

# Location of the target environment
set(CMAKE_FIND_ROOT_PATH ${ATFE_ROOT})

# Adjust the default behavior of the FIND_XXX() commands:
#  - search for headers and libraries in the target environment,
#  - search for programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
