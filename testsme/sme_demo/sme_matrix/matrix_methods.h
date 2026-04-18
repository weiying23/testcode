/*
 * matrix_methods.h
 *
 * SME GEMM kernel 的函数声明。
 *
 * 注意：声明中不使用 __arm_streaming 属性。
 * 原因：该属性会让编译器在调用方插入 cntd 指令（保存 VG 用于异常展开），
 * 而 macOS 用户态不允许执行非流式 SVE 系统指令，导致 SIGILL。
 * 流式模式切换由函数体内的内联汇编 smstart/smstop 自行管理。
 */

#include <arm_sme.h>
#ifndef MATRIX_METHODS_H
#define MATRIX_METHODS_H

/* 双精度矩阵乘法：C = A(M×K) × B(K×N)，使用 SME fmopa 外积指令 */
void gemmkernel(double*, double*, double*, int, int, int, double);

/* SME 流式模式可用性验证（进入/退出 streaming mode 并打印结果） */
void test_kernel();

/* SME 硬件状态全面验证（sysctl + SVCR + RDSVL + 结果正确性） */
void verify_sme();

#endif
