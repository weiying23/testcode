/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MatMul计算Kernel - 用于隐藏测试的计算负载
 */

#include "kernel_operator.h"
#include "acl/acl.h"

extern "C" __global__ __aicore__ void matmul_compute_kernel(
    GM_ADDR A,          // 输入矩阵A (M x K)
    GM_ADDR B,          // 输入矩阵B (K x N)
    GM_ADDR C,          // 输出矩阵C (M x N)
    int64_t M,
    int64_t K,
    int64_t N,
    GM_ADDR result_buffer) {

    if (AscendC::GetSubBlockIdx() != 0) return;

    int64_t start_cycle = AscendC::GetSystemCycle();

    // 简化的MatMul计算 (实际项目应使用完整的MatMul kernel)
    // 这里用累加操作模拟计算负载
    __gm__ float* a_ptr = (__gm__ float*)A;
    __gm__ float* b_ptr = (__gm__ float*)B;
    __gm__ float* c_ptr = (__gm__ float*)C;

    // 计算量: M * K * N * 2 FLOPS
    // 简化实现: 每个元素累加
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0;
            for (int64_t k = 0; k < K; k++) {
                sum += a_ptr[i * K + k] * b_ptr[k * N + j];
            }
            c_ptr[i * N + j] = sum;
        }
    }

    int64_t end_cycle = AscendC::GetSystemCycle();
    *(__gm__ int64_t*)(result_buffer) = end_cycle - start_cycle;
}

void launch_matmul_compute(uint32_t block_dim, void* stream,
                            uint8_t* A, uint8_t* B, uint8_t* C,
                            int64_t M, int64_t K, int64_t N,
                            uint8_t* result_buffer) {
    matmul_compute_kernel<<<1, nullptr, stream>>>(
        A, B, C, M, K, N, result_buffer);
}