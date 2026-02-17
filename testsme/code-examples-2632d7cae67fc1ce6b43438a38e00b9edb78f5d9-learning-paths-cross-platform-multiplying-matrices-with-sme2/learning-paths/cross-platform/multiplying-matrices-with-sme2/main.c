/*
 * SPDX-FileCopyrightText: Copyright 2024,2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#ifndef __ARM_FEATURE_SME2
#error __ARM_FEATURE_SME2 is not defined
#endif

#ifndef IMPL
#error matmul implementation selection macro IMPL is not defined
#endif

#include "matmul.h"
#include "misc.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define STRINGIFY_(I) #I
#define STRINGIFY(I) STRINGIFY_(I)
#define FN(M, I) M##I
#define MATMUL(I, M, K, N, mL, mR, mM, m) FN(matmul_, I)(M, K, N, mL, mR, mM, m)

void usage(const char *prog_name) {
#if BAREMETAL == 1
    printf("Usage: %s <M> <K> <N>\n", prog_name);
    printf("  M: number of rows in matLeft (default: 125)\n");
    printf("  K: number of columns in matLeft and matRight (default: 35)\n");
    printf("  N: number of columns in matRight (default: 70)\n");
    printf("Example: matmul 125 35 70\n");
#else
    printf("Depending on the number of arguments, the program can be invoked "
           "in two modes:\n");
    printf(" - verification mode. The program will run the assembly or "
           "intrinsics implementatation of the matrix multiplication and "
           "compare the results with a reference implementation.\n");
    printf(" - benchmarking mode. The program will run the assembly or "
           "intrinsics implementation of the matrix multiplication a number of "
           "times and print the time taken to perform the operation.\n");

    printf("\n");
    printf("Verification mode:\n");
    printf(" %s\n", prog_name);
    printf(" %s <M> <K> <N>\n", prog_name);
    printf("with:\n");
    printf("  - M: number of rows in matLeft (default: 125)\n");
    printf("  - K: number of columns in matLeft and number of rows in matRight "
           "(default: 35). Must be > 2 for assembly version of matmul.\n");
    printf("  - N: number of columns in matRight (default: 70)\n");
    printf("Example: %s 67 18 23\n", prog_name);

    printf("\n");
    printf("Benchmarking mode:\n");
    printf(" %s <I>\n", prog_name);
    printf(" %s <I> <M> <K> <N>\n", prog_name);
    printf("with:\n");
    printf("  - I: number of iterations to perform. Must be > 0.\n");
    printf("  - M: number of rows in matLeft (default: 125)\n");
    printf("  - K: number of columns in matLeft and number of rows in matRight "
           "(default: 35). Must be > 2 for assembly version of matmul.\n");
    printf("  - N: number of columns in matRight (default: 70)\n");
    printf("Example: %s 1000 67 18 23\n", prog_name);
#endif
}

int main(int argc, char **argv) {

    /* Matrices size parameters, defaults to 125x35x70.
       Assumptions (for assembly handwritten matmul) are:
         - number of rows in matLeft (M): any
         - number of columns in matLeft and number of rows in matRight (K): any K > 2
         - number of columns in matRight (N): any
    */
    uint64_t I = 0; // Number of iterations to perform for benchmarking.
    uint64_t M = 125; // Number of rows in matLeft.
    uint64_t N = 35;  // Number of columns in matRight.
    uint64_t K = 70;  // Number of columns (resp. rows) in matLeft (resp. matRight).

    switch (argc) {
    case 1:
        // Verification mode, with default matrix sizes.
        break;
#if BAREMETAL == 0
    case 2:
        // Benchmarking mode, with default matrix sizes.
        I = strtoull(argv[1], NULL, 0);
        if (I == 0) {
            printf("Error, in benchmarking mode, I must be > 0.\n");
            return EXIT_FAILURE;
        }
        break;
#endif
    case 4:
        // Verification mode, with user-defined matrix sizes.
        M = strtoul(argv[1], NULL, 0);
        K = strtoul(argv[2], NULL, 0);
        N = strtoul(argv[3], NULL, 0);
        break;
#if BAREMETAL == 0
    case 5:
        // Benchmarking mode, with user-defined matrix sizes.
        I = strtoull(argv[1], NULL, 0);
        if (I == 0) {
            printf("Error, in benchmarking mode, I must be > 0.\n");
            return EXIT_FAILURE;
        }
        M = strtoul(argv[2], NULL, 0);
        K = strtoul(argv[3], NULL, 0);
        N = strtoul(argv[4], NULL, 0);
        break;
#endif
    default:
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Check assumptions hold.
    if (strcmp(STRINGIFY(IMPL), "asm")==0 && K <= 2) {
        printf("Error, for assembly implementation of matmul, K must be > 2.\n");
        return EXIT_FAILURE;
    }

    // Describe the operation that will be performed.
    printf("SME2 Matrix Multiply fp32 *%s* ", STRINGIFY(IMPL));
    if (I != 0)
        printf("[benchmarking mode, %" PRIu64 " iterations] ", I);
    else
        printf("[verification mode] ");
    printf("with M=%" PRIu64 ", K=%" PRIu64 ", N=%" PRIu64 "\n", M, K, N);

#if BAREMETAL == 1
    setup_sme_baremetal();
#endif

    const uint64_t SVL = svcntsw();

    // Calculate M of transformed matLeft.
    const uint64_t M_mod = SVL * (M / SVL + (M % SVL != 0 ? 1 : 0));

    // Allocate memory for all matrices.
    float *matRight = (float *)malloc(K * N * sizeof(float));

    float *matLeft = (float *)malloc(M * K * sizeof(float));
    float *matLeft_mod = (float *)malloc(M_mod * K * sizeof(float));
    float *matLeft_mod_ref = (float *)malloc(M_mod * K * sizeof(float));

    float *matResult = (float *)malloc(M * N * sizeof(float));
    float *matResult_ref = (float *)malloc(M * N * sizeof(float));

    // Initialize matrices. Input matrices are initialized with random values in
    // non debug mode. In debug mode, all matrices are initialized with linear
    // or known values values for easier debugging.
#ifdef DEBUG
    initialize_matrix(matLeft, M * K, LINEAR_INIT);
    initialize_matrix(matRight, K * N, LINEAR_INIT);
    initialize_matrix(matLeft_mod, M_mod * K, DEAD_INIT);
    initialize_matrix(matResult, M * N, DEAD_INIT);

    print_matrix(M, K, matLeft, "matLeft");
    print_matrix(K, N, matRight, "matRight");
#else
    initialize_matrix(matLeft, M * K, RANDOM_INIT);
    initialize_matrix(matRight, K * N, RANDOM_INIT);
#endif

    unsigned error = 0;
    if (I == 0) {
        // Verification mode.
        MATMUL(IMPL, M, K, N, matLeft, matRight, matLeft_mod, matResult);

        // Compute the reference values with the vanilla implementations.
        preprocess_l(M, K, SVL, matLeft, matLeft_mod_ref);
        matmul(M, K, N, matLeft, matRight, matResult_ref);

        error = compare_matrices(K, M_mod, matLeft_mod_ref, matLeft_mod,
                                 "Matrix preprocessing");
        if (!error)
            error = compare_matrices(M, N, matResult_ref, matResult,
                                     "Matrix multiplication");
    } else {
#if BAREMETAL == 0
        // Benchmarking mode.
        uint64_t min_time = UINT64_MAX;
        uint64_t max_time = 0;
        double sum = 0.0;

        // Warm-up runs to ensure the CPU is ready for benchmarking.
        for (uint64_t i = 0; i < 10; i++)
            matmul(M, K, N, matLeft, matRight, matResult_ref);

        // Measure the time taken by the matrix multiplication.
        for (uint64_t i = 0; i < I; i++) {
            const uint64_t start_time = get_time_microseconds();
            matmul(M, K, N, matLeft, matRight, matResult_ref);
            const uint64_t elapsed_time = get_time_microseconds() - start_time;

            if (elapsed_time < min_time)
                min_time = elapsed_time;
            if (elapsed_time > max_time)
                max_time = elapsed_time;
            sum += elapsed_time;
        }
        printf("Reference implementation: min time = %" PRIu64 " us, "
               "max time = %" PRIu64 " us, avg time = %.2f us\n",
               min_time, max_time, sum / I);

        // Benchmarking mode (SME2 implementation).
        min_time = UINT64_MAX;
        max_time = 0;
        sum = 0.0;

        // Warm-up runs to ensure the CPU is ready for benchmarking.
        for (uint64_t i = 0; i < 10; i++)
            MATMUL(IMPL, M, K, N, matLeft, matRight, matLeft_mod, matResult);

        // Measure the time taken by the SME2 matrix multiplication.
        for (uint64_t i = 0; i < I; i++) {
            const uint64_t start_time = get_time_microseconds();
            MATMUL(IMPL, M, K, N, matLeft, matRight, matLeft_mod, matResult);
            const uint64_t elapsed_time = get_time_microseconds() - start_time;

            if (elapsed_time < min_time)
                min_time = elapsed_time;
            if (elapsed_time > max_time)
                max_time = elapsed_time;
            sum += elapsed_time;
        }
        printf("SME2 implementation *%s*: min time = %" PRIu64 " us, "
               "max time = %" PRIu64 " us, avg time = %.2f us\n",
               STRINGIFY(IMPL), min_time, max_time, sum / I);
#else
        printf("Error, can not run in benchmarking mode in baremetal.\n");
        return EXIT_FAILURE;
#endif
    }

    // Free allocated memory.
    free(matRight);

    free(matLeft);
    free(matLeft_mod);
    free(matLeft_mod_ref);

    free(matResult);
    free(matResult_ref);

    return error ? EXIT_FAILURE : EXIT_SUCCESS;
}
