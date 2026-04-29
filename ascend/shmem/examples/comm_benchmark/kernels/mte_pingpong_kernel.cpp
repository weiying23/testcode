/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MTE PingPong延迟测试Kernel (同节点最优)
 */

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem_api.h"

constexpr uint32_t MAGIC_VAL = 12345;

extern "C" __global__ __aicore__ void mte_pingpong_latency_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t warmup,
    GM_ADDR result_buffer) {

    shmemx_set_ffts_config(ffts_config);
    if (AscendC::GetSubBlockIdx() != 0) return;

    // 获取MTE配置
    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
    uint64_t copy_ub = device_state->mte_config.shmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;

    int64_t rank = smem_shm_get_global_rank();
    uint32_t peer = (rank == 0) ? 1 : 0;

    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    // Warmup阶段
    for (int64_t i = 0; i < warmup; i++) {
        if (rank == 0) {
            shmem_mte_put_mem_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != peer + MAGIC_VAL + i) {
                cacheWriteThrough(gva + msg_size * 2 - 8, 8);
                AscendC::GetSystemCycle();
            }
        } else {
            while (*(__gm__ uint32_t*)(gva + msg_size * 1 - 8) != peer + MAGIC_VAL + i) {
                cacheWriteThrough(gva + msg_size * 1 - 8, 8);
                AscendC::GetSystemCycle();
            }
            shmem_mte_put_mem_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 正式测试
    if (rank == 0) {
        for (int64_t i = 0; i < iterations; i++) {
            int64_t iter_start = AscendC::GetSystemCycle();

            shmem_mte_put_mem_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != peer + MAGIC_VAL + warmup + i) {
                cacheWriteThrough(gva + msg_size * 2 - 8, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            int64_t iter_end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = iter_end - iter_start;
        }
    } else {
        for (int64_t i = 0; i < iterations; i++) {
            while (*(__gm__ uint32_t*)(gva + msg_size * 1 - 8) != peer + MAGIC_VAL + warmup + i) {
                cacheWriteThrough(gva + msg_size * 1 - 8, 8);
                AscendC::GetSystemCycle();
            }
            shmem_mte_put_mem_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

void launch_mte_pingpong_latency(uint32_t block_dim, void* stream,
                                  uint64_t ffts_config, uint8_t* gva,
                                  int64_t msg_size, int64_t iterations,
                                  int64_t warmup, uint8_t* result_buffer) {
    mte_pingpong_latency_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, warmup, result_buffer);
}