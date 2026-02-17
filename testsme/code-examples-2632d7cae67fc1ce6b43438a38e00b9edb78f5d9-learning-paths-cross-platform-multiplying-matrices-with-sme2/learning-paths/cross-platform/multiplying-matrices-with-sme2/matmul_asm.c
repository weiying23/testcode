/*
 * SPDX-FileCopyrightText: Copyright 2024,2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#include "matmul.h"

__arm_new("za") __arm_locally_streaming void matmul_asm(
    uint64_t M, uint64_t K, uint64_t N, const float *restrict matLeft,
    const float *restrict matRight, float *restrict matLeft_mod,
    float *restrict matResult) {

    preprocess_l_asm(M, K, matLeft, matLeft_mod);
    matmul_asm_impl(M, K, N, matLeft_mod, matRight, matResult);
}
