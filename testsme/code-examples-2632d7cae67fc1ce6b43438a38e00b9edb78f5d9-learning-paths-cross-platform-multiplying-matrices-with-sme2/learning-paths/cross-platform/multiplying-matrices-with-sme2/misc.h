/*
 * SPDX-FileCopyrightText: Copyright 2024,2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */
#ifndef MISC_H
#define MISC_H

#include <stdint.h>
#include <string.h>

// Print on stdout the available CPU features. Returns a non zero value if the
// CPU does not have SME2 support.
int display_cpu_features();

#if BAREMETAL == 0
// Get the current time in microseconds.
uint64_t get_time_microseconds();
#endif // BAREMETAL == 0

#if BAREMETAL == 1
// Perform some setup to be able to use SME.
void setup_sme_baremetal();
#endif // BAREMETAL == 1

// ========================================================================================
// Generally useful helper routines.

// Initialize an array of float.
enum InitKind { RANDOM_INIT, LINEAR_INIT, DEAD_INIT };
void initialize_matrix(float *mat, size_t num_elements, enum InitKind kind);

// Compare 2 matrices for equality.
unsigned compare_matrices(size_t nbr, size_t nbc, const float *reference,
                          const float *result, const char *str);

// Pretty print a matrix.
void print_matrix(size_t nbr, size_t nbc, const float *mat, const char *name);

#endif // MISC_H
