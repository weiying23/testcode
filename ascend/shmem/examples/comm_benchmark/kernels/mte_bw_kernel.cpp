/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MTE带宽测试Kernel
 */

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem_api.h"

extern "C" __global__ __aicore__ void mte_bandwidth_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
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
        int64_t start_cycle = AscendC::GetSystemCycle();

        // 连续发送iterations次
        for (int64_t i = 0; i < iterations; i++) {
            shmem_mte_put_mem_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
        }

        // 等待完成
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

        int64_t end_cycle = AscendC::GetSystemCycle();
        *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;
    }
}

void launch_mte_bandwidth(uint32_t block_dim, void* stream,
                           uint64_t ffts_config, uint8_t* gva,
                           int64_t msg_size, int64_t iterations,
                           uint8_t* result_buffer) {
    mte_bandwidth_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, result_buffer);
}