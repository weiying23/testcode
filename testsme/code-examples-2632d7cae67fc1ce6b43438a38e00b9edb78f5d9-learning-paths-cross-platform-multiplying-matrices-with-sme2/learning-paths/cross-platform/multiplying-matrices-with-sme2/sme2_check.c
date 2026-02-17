/*
 * SPDX-FileCopyrightText: Copyright 2024,2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#include "misc.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __ARM_FEATURE_SME2
#include <arm_sme.h>
#else
#error __ARM_FEATURE_SME2 is not defined
#endif

__arm_locally_streaming void function_in_streaming_mode() {
    printf("In streaming_mode: %d, SVL: %" PRIu64 " bits\n",
           __arm_in_streaming_mode(), svcntb() * 8);
}

int main(int argc, char *argv[]) {

#if BAREMETAL == 1
    setup_sme_baremetal();
#endif

    if (!display_cpu_features()) {
        printf("SME2 is not supported on this CPU.\n");
        exit(EXIT_FAILURE);
    }

    printf("Checking initial in_streaming_mode: %d\n",
           __arm_in_streaming_mode());

    printf("Switching to streaming mode...\n");

    function_in_streaming_mode();

    printf("Switching back from streaming mode...\n");

    printf("Checking in_streaming_mode: %d\n", __arm_in_streaming_mode());

    return EXIT_SUCCESS;
}
