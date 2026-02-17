/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#ifndef MATMUL_H
#define MATMUL_H

#include <stdint.h>

// ========================================================================================
// Vanilla matrix multiplication using the by-the-book definition.
void matmul(uint64_t M, uint64_t K, uint64_t N, const float *restrict matLeft,
            const float *restrict matRight, float *restrict matResult);

// A reference implementation in pure C of what the preprocessing actually does.
void preprocess_l(uint64_t M, uint64_t K, uint64_t SVL, const float *restrict a,
                  float *restrict a_mod);

#ifdef __ARM_FEATURE_SME2

#include <arm_sme.h>

// ========================================================================================
// SME2 Matrix multiplication handwritten in assembly code. This is split in 2
// functions that have to be invoked one after the other, with a top level
// binding.

// The top level matrix multiplication.
__arm_new("za") __arm_locally_streaming void matmul_asm(
    uint64_t M, uint64_t K, uint64_t N, const float *restrict matLeft,
    const float *restrict matRight, float *restrict matLeft_mod,
    float *restrict matResult);

// Matrix preprocessing, in assembly.
void preprocess_l_asm(uint64_t M, uint64_t K, const float *restrict a,
                      float *restrict a_mod) __arm_streaming __arm_inout("za");

// Matrix multiplication (with the *transposed* RHS), in assembly.
void matmul_asm_impl(
    uint64_t M, uint64_t K, uint64_t N, const float *restrict matLeft_mod,
    const float *restrict matRight,
    float *restrict matResult) __arm_streaming __arm_inout("za");

// ========================================================================================
// SME2 Matrix multiplication implemented with compiler intrinsics.

// The top level Matrix multiplication.
__arm_new("za") __arm_locally_streaming void matmul_intr(
    uint64_t M, uint64_t K, uint64_t N, const float *restrict matLeft,
    const float *restrict matRight, float *restrict matLeft_mod,
    float *restrict matResult);

// Matrix preprocessing, implemented with intrinsics.
void preprocess_l_intr(
    uint64_t M, uint64_t K, uint64_t SVL, const float *restrict matLeft,
    float *restrict matLeft_mod) __arm_streaming __arm_inout("za");

// Matrix multiplication (with the *transposed* RHS), implemented with
// intrinsics
void matmul_intr_impl(
    uint64_t M, uint64_t K, uint64_t N, uint64_t SVL,
    const float *restrict matLeft_mod, const float *restrict matRight,
    float *restrict matResult) __arm_streaming __arm_inout("za");

#endif

#endif // MATMUL_H
