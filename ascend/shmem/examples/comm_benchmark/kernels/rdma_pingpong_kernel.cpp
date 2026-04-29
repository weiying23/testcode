/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * RDMA PingPong延迟测试Kernel
 */

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem_api.h"
#include "benchmark_config.h"

constexpr uint32_t MAGIC_VAL = 12345;

extern "C" __global__ __aicore__ void rdma_pingpong_latency_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t warmup,
    GM_ADDR result_buffer) {

    shmemx_set_ffts_config(ffts_config);
    if (AscendC::GetSubBlockIdx() != 0) return;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE);
    AscendC::LocalTensor<uint32_t> ubLocal = buf.GetWithOffset<uint32_t>(UB_ALIGN_SIZE / sizeof(uint32_t), 0);

    int64_t rank = smem_shm_get_global_rank();
    int64_t rank_size = smem_shm_get_global_rank_size();
    uint32_t peer = (rank == 0) ? 1 : 0;

    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    // Warmup阶段
    for (int64_t i = 0; i < warmup; i++) {
        if (rank == 0) {
            shmemi_roce_write((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr, peer, 0, msg_size, ubLocal, ubLocal);
            while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != peer + MAGIC_VAL + i) {
                cacheWriteThrough(gva + msg_size * 2 - 8, 8);
                AscendC::GetSystemCycle();
            }
        } else {
            while (*(__gm__ uint32_t*)(gva + msg_size * 1 - 8) != peer + MAGIC_VAL + i) {
                cacheWriteThrough(gva + msg_size * 1 - 8, 8);
                AscendC::GetSystemCycle();
            }
            shmemi_roce_write((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr, peer, 0, msg_size, ubLocal, ubLocal);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 正式测试阶段，记录每次迭代的时间
    if (rank == 0) {
        int64_t start_cycle = AscendC::GetSystemCycle();

        for (int64_t i = 0; i < iterations; i++) {
            int64_t iter_start = AscendC::GetSystemCycle();

            shmemi_roce_write((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr, peer, 0, msg_size, ubLocal, ubLocal);
            while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != peer + MAGIC_VAL + warmup + i) {
                cacheWriteThrough(gva + msg_size * 2 - 8, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            int64_t iter_end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = iter_end - iter_start;
        }

        int64_t end_cycle = AscendC::GetSystemCycle();
        *(__gm__ int64_t*)(result_addr + iterations * sizeof(int64_t)) = end_cycle - start_cycle;
    } else {
        for (int64_t i = 0; i < iterations; i++) {
            while (*(__gm__ uint32_t*)(gva + msg_size * 1 - 8) != peer + MAGIC_VAL + warmup + i) {
                cacheWriteThrough(gva + msg_size * 1 - 8, 8);
                AscendC::GetSystemCycle();
            }
            shmemi_roce_write((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr, peer, 0, msg_size, ubLocal, ubLocal);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

// Host端调用接口
void launch_rdma_pingpong_latency(uint32_t block_dim, void* stream,
                                   uint64_t ffts_config, uint8_t* gva,
                                   int64_t msg_size, int64_t iterations,
                                   int64_t warmup, uint8_t* result_buffer) {
    rdma_pingpong_latency_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, warmup, result_buffer);
}