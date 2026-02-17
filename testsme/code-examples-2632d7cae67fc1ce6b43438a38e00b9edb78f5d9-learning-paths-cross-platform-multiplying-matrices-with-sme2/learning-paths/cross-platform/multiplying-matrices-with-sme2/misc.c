/*
 * SPDX-FileCopyrightText: Copyright 2024,2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#include "misc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if BAREMETAL == 0
#include <time.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#ifdef __ANDROID__
#include <sys/auxv.h>
#endif
#endif

#ifdef __ARM_FEATURE_SME2
#include <arm_sme.h>
#else
#error __ARM_FEATURE_SME2 is not defined
#endif

unsigned __aarch64_sme_accessible() { return 1; }

char SME_save_blk[16] = {0};

#if BAREMETAL == 1

void setup_sme_baremetal() {
    // Disable all SME, SVE, and SIMD traps in CPTR_EL3 by setting ESM bit 12 and EZ
    // bit 8, and clearing TFP bit 10
    __asm__ volatile("mrs x0, CPTR_EL3\n"
                     "orr x0, x0, #(1<<12)\n"
                     "bic x0, x0, #(1<<10)\n"
                     "orr x0, x0, #(1<<8)\n"
                     "msr CPTR_EL3, x0\n"
                     "isb\n");

    // Refer to
    // https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst to
    // understand the usage and format of TPIDR_EL0.
    __asm__ volatile("msr  TPIDR2_EL0, %0" : : "r"((void *)SME_save_blk));
}

int display_cpu_features() {
#define get_cpu_ftr(regId, feat, msb, lsb)                                     \
    ({                                                                         \
        unsigned long __val;                                                   \
        __asm__("mrs %0, " #regId : "=r"(__val));                              \
        printf("%-20s: 0x%016lx\n", #regId, __val);                            \
        printf("  - %-10s: 0x%08lx\n", #feat,                                  \
               (__val >> lsb) & ((1 << (msb - lsb)) - 1));                     \
    })
    get_cpu_ftr(ID_AA64PFR0_EL1, SVE, 35, 32);
    get_cpu_ftr(ID_AA64PFR1_EL1, SME, 27, 24);
    printf("Checking has_sme: %d\n", __arm_has_sme());
    return __arm_has_sme() ? 1 : 0;
#undef get_cpu_ftr
}

#else // BAREMETAL == 0

uint64_t get_time_microseconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec) * 1000000 + (ts.tv_nsec / 1000);
}

#ifdef __APPLE__
int display_cpu_features() {
    int oldp = 0;
    size_t size = sizeof(oldp);

    printf("HAS_SVE: 0\n"); // No SVE support ?

    if (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &size, NULL, 0) == 0) {
        printf("HAS_SME: %d\n", oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &oldp, &size, NULL, 0) == 0) {
        printf("HAS_SME2: %d\n", oldp);
    }
    return oldp;
}
#endif // End of Apple specific code

#ifdef __ANDROID__
int display_cpu_features() {
    unsigned long hwcaps = getauxval(AT_HWCAP);
    printf("HAS_SVE: %d\n", hwcaps & HWCAP_SVE ? 1 : 0);

    hwcaps = getauxval(AT_HWCAP2);
    printf("HAS_SME: %d\n", hwcaps & HWCAP2_SME ? 1 : 0);
    int has_sme2 = hwcaps & HWCAP2_SME2 ? 1 : 0;
    printf("HAS_SME2: %d\n", has_sme2);

    return has_sme2;
}
#endif

#endif // BAREMETAL

void initialize_matrix(float *mat, size_t num_elements, enum InitKind kind) {
    for (size_t i = 0; i < num_elements; i++)
        switch (kind) {
        case RANDOM_INIT:
            mat[i] = (((float)(rand() % 10000) / 100.0f) - 30.0);
            break;
        case LINEAR_INIT:
            mat[i] = i+1;
            break;
        case DEAD_INIT:
            mat[i] = nan("");
            break;
        }
}

void print_matrix(size_t nbr, size_t nbc, const float *mat, const char *name) {
    printf("%s(%lu,%lu) = [", name, nbr, nbc);
    for (size_t y = 0; y < nbr; y++) {
        printf("\n  ");
        for (size_t x = 0; x < nbc; x++)
            printf("%9.2f, ", mat[y * nbc + x]);
    }
    printf("\n];\n");
}

unsigned compare_matrices(size_t nbr, size_t nbc, const float *reference,
                          const float *result, const char *str) {
    unsigned error = 0;

    for (size_t y = 0; y < nbr; y++) {
        for (size_t x = 0; x < nbc; x++) {
            if (fabsf(reference[y * nbc + x] - result[y * nbc + x]) >
                fabsf(0.0002f * reference[y * nbc + x])) {
                error = 1;
#ifdef DEBUG
                printf("%lu (%lu,%lu): %f <> %f\n", y * nbc + x, x, y,
                       reference[y * nbc + x], result[y * nbc + x]);
#endif
            }
        }
    }
#ifdef DEBUG
    if (error) {
        print_matrix(nbr, nbc, reference, "reference");
        print_matrix(nbr, nbc, result, "result");
    }
#endif

    printf("%s: %s !\n", str, error ? "FAILED" : "PASS");

    return error;
}
