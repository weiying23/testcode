/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * 通信隐藏测试Kernel - 非阻塞通信 + 计算重叠
 */

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem_api.h"

extern "C" __global__ __aicore__ void hidden_comm_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    GM_ADDR matmul_A,       // MatMul输入A
    GM_ADDR matmul_B,       // MatMul输入B
    GM_ADDR matmul_C,       // MatMul输出C
    int64_t matmul_M,       // MatMul维度M
    int64_t matmul_K,       // MatMul维度K
    int64_t matmul_N,       // MatMul维度N
    GM_ADDR result_buffer) {

    shmemx_set_ffts_config(ffts_config);
    if (AscendC::GetSubBlockIdx() != 0) return;

    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
    uint64_t copy_ub = device_state->mte_config.shmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;

    int64_t rank = smem_shm_get_global_rank();
    uint32_t peer = (rank == 0) ? 1 : 0;

    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    if (rank == 0) {
        for (int64_t i = 0; i < iterations; i++) {
            int64_t iter_start = AscendC::GetSystemCycle();

            // 1. 发起非阻塞通信
            shmem_mte_put_mem_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);

            // 2. 同时进行MatMul计算 (简化版，实际需要调用MatMul kernel)
            // 这里用简单的循环模拟计算负载
            volatile float sum = 0;
            for (int64_t j = 0; j < matmul_M * matmul_K * matmul_N / 1000; j++) {
                sum += j * 0.001f;
            }

            // 3. 等待通信完成
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            int64_t iter_end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = iter_end - iter_start;
        }
    }
}

void launch_hidden_comm(uint32_t block_dim, void* stream,
                         uint64_t ffts_config, uint8_t* gva,
                         int64_t msg_size, int64_t iterations,
                         uint8_t* matmul_A, uint8_t* matmul_B, uint8_t* matmul_C,
                         int64_t M, int64_t K, int64_t N,
                         uint8_t* result_buffer) {
    hidden_comm_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations,
        matmul_A, matmul_B, matmul_C, M, K, N,
        result_buffer);
}