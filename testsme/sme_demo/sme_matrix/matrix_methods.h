#include <arm_sme.h>
#ifndef MATRIX_METHODS_H
#define MATRIX_METHODS_H
void gemmkernel(double*, double*, double*, int, int, int, double) __arm_streaming;

void test_kernel() __arm_streaming;
#endif