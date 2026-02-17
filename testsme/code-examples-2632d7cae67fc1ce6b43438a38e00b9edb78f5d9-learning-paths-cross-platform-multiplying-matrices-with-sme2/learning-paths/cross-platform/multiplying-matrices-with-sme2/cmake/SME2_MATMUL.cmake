# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: BSD-3-Clause-Clear

function(add_sme2_executable target)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs "SME2_SOURCES;BASELINE_SOURCES;ASSEMBLY_SOURCES;COMPILE_DEFINITIONS")
  cmake_parse_arguments(ARG
    "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable(${target} ${ARG_SME2_SOURCES} ${ARG_BASELINE_SOURCES} ${ARG_ASSEMBLY_SOURCES})

  if(ARG_COMPILE_DEFINITIONS)
    target_compile_definitions(${target} PRIVATE ${ARG_COMPILE_DEFINITIONS})
  endif()

  if (ARG_BASELINE_SOURCES)
    set_source_files_properties(${ARG_BASELINE_SOURCES} PROPERTIES COMPILE_OPTIONS "${BASELINE_MARCH}")
  endif()

  if (ARG_SME2_SOURCES)
    set_source_files_properties(${ARG_SME2_SOURCES} PROPERTIES COMPILE_OPTIONS "${SME2_MARCH}")
  endif()

  if (ARG_ASSEMBLY_SOURCES)
    set_source_files_properties(${ARG_ASSEMBLY_SOURCES} PROPERTIES COMPILE_OPTIONS "${SME2_ASM_MARCH}")
  endif()

  if (CMAKE_OBJDUMP)
    # Generate an .lst file next to the binary after linking
    add_custom_command(
      TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_OBJDUMP}
              --demangle -d
              $<TARGET_FILE:${target}>
              > $<TARGET_FILE_DIR:${target}>/$<TARGET_FILE_BASE_NAME:${target}>.lst
      VERBATIM
    )
  else()
    message(WARNING "No llvm-objdump found; skipping assembly listing for target ${target}")
  endif()

endfunction()
