/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates
 * <open-source-office@arm.com> SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#include "matmul.h"
#include <stdint.h>

void preprocess_l_intr(
    uint64_t M, uint64_t K, uint64_t SVL, const float *restrict a,
    float *restrict a_mod) __arm_streaming __arm_inout("za") {
    const uint64_t M_mod = SVL * (M / SVL + (M % SVL != 0 ? 1 : 0));

    // The outer loop, iterating over rows (M dimension)
    for (uint64_t row = 0; row < M; row += SVL) {

        svbool_t pMDim = svwhilelt_b32(row, M);

        // The inner loop, iterating on columns (K dimension).
        for (uint64_t col = 0; col < K; col += 2 * SVL) {

            svcount_t pKDim = svwhilelt_c32(col, K, 2);

            // Load-as-rows
            for (uint64_t trow = 0; trow < SVL; trow += 4) {
                svcount_t p0 = svpsel_lane_c32(pKDim, pMDim, trow + 0);
                svcount_t p1 = svpsel_lane_c32(pKDim, pMDim, trow + 1);
                svcount_t p2 = svpsel_lane_c32(pKDim, pMDim, trow + 2);
                svcount_t p3 = svpsel_lane_c32(pKDim, pMDim, trow + 3);

                const uint64_t tile_UL_corner = (row + trow) * K + col;
                svfloat32x2_t zp0 = svld1_x2(p0, &a[tile_UL_corner + 0 * K]);
                svfloat32x2_t zp1 = svld1_x2(p1, &a[tile_UL_corner + 1 * K]);
                svfloat32x2_t zp2 = svld1_x2(p2, &a[tile_UL_corner + 2 * K]);
                svfloat32x2_t zp3 = svld1_x2(p3, &a[tile_UL_corner + 3 * K]);

                svfloat32x4_t zq0 = svcreate4(svget2(zp0, 0), svget2(zp1, 0),
                                              svget2(zp2, 0), svget2(zp3, 0));
                svfloat32x4_t zq1 = svcreate4(svget2(zp0, 1), svget2(zp1, 1),
                                              svget2(zp2, 1), svget2(zp3, 1));
                svwrite_hor_za32_f32_vg4(
                    /* tile: */ 0, /* slice: */ trow, zq0);
                svwrite_hor_za32_f32_vg4(
                    /* tile: */ 1, /* slice: */ trow, zq1);
            }

            // Read-as-columns and store
            const uint64_t dest_0 = row * K + col * SVL;
            const uint64_t dest_1 = dest_0 + SVL * SVL;
            for (uint64_t tcol = 0; tcol < SVL; tcol += 4) {
                svcount_t p0 = svwhilelt_c32(dest_0 + tcol * SVL, K * M_mod, 4);
                svcount_t p1 = svwhilelt_c32(dest_1 + tcol * SVL, K * M_mod, 4);
                svfloat32x4_t zq0 =
                    svread_ver_za32_f32_vg4(/* tile: */ 0, /* slice: */ tcol);
                svfloat32x4_t zq1 =
                    svread_ver_za32_f32_vg4(/* tile: */ 1, /* slice: */ tcol);
                svst1(p0, &a_mod[dest_0 + tcol * SVL], zq0);
                svst1(p1, &a_mod[dest_1 + tcol * SVL], zq1);
            }
        }
    }
}

void matmul_intr_impl(
    uint64_t M, uint64_t K, uint64_t N, uint64_t SVL,
    const float *restrict matLeft_mod, const float *restrict matRight,
    float *restrict matResult) __arm_streaming __arm_inout("za") {

    // Build the result matrix tile by tile.
    for (uint64_t row = 0; row < M; row += SVL) {

        svbool_t pMDim = svwhilelt_b32(row, M);

        for (uint64_t col = 0; col < N; col += SVL) {

            svbool_t pNDim = svwhilelt_b32(col, N);

            // Outer product + accumulation
            svzero_za();
            const uint64_t matLeft_pos = row * K;
            const uint64_t matRight_UL_corner = col;
            for (uint64_t k = 0; k < K; k++) {
                svfloat32_t zL =
                    svld1(pMDim, &matLeft_mod[matLeft_pos + k * SVL]);
                svfloat32_t zR =
                    svld1(pNDim, &matRight[matRight_UL_corner + k * N]);
                svmopa_za32_m(0, pMDim, pNDim, zL, zR);
            }

            // Store ZA to matResult.
            const uint64_t result_tile_UL_corner = row * N + col;
            for (uint64_t trow = 0; trow < SVL && row + trow < M; trow += 4) {
                svbool_t p0 = svpsel_lane_b32(pNDim, pMDim, row + trow + 0);
                svbool_t p1 = svpsel_lane_b32(pNDim, pMDim, row + trow + 1);
                svbool_t p2 = svpsel_lane_b32(pNDim, pMDim, row + trow + 2);
                svbool_t p3 = svpsel_lane_b32(pNDim, pMDim, row + trow + 3);

                svst1_hor_za32(
                    /* tile: */ 0, /* slice: */ trow + 0, p0,
                    &matResult[result_tile_UL_corner + (trow + 0) * N]);
                svst1_hor_za32(
                    /* tile: */ 0, /* slice: */ trow + 1, p1,
                    &matResult[result_tile_UL_corner + (trow + 1) * N]);
                svst1_hor_za32(
                    /* tile: */ 0, /* slice: */ trow + 2, p2,
                    &matResult[result_tile_UL_corner + (trow + 2) * N]);
                svst1_hor_za32(
                    /* tile: */ 0, /* slice: */ trow + 3, p3,
                    &matResult[result_tile_UL_corner + (trow + 3) * N]);
            }
        }
    }
}

__arm_new("za") __arm_locally_streaming void matmul_intr(
    uint64_t M, uint64_t K, uint64_t N, const float *restrict matLeft,
    const float *restrict matRight, float *restrict matLeft_mod,
    float *restrict matResult) {
    uint64_t SVL = svcntsw();
    preprocess_l_intr(M, K, SVL, matLeft, matLeft_mod);
    matmul_intr_impl(M, K, N, SVL, matLeft_mod, matRight, matResult);
}
