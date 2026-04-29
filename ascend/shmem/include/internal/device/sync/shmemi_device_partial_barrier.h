/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHEMEI_PARTIAL_BARRIER_H
#define SHEMEI_PARTIAL_BARRIER_H

#include "kernel_operator.h"
#include "shmemi_device_barrier.h"

SHMEM_DEVICE
__gm__ shmemi_sync_bit *shmemi_get_team_partial_barrier_slot(shmem_team_t team_idx, uint32_t slot)
{
    uint64_t addr = (uint64_t)shmemi_get_state()->partial_barrier_pool;
    addr += (uint64_t)team_idx * SHMEM_PARTIAL_BARRIER_PER_TEAM_SIZE;
    addr += (uint64_t)slot * SHMEMI_SYNCBIT_SIZE;
    return (__gm__ shmemi_sync_bit *)addr;
}

SHMEM_DEVICE uint32_t* get_partial_barrier_idx()
{
    static uint32_t g_partial_barrier_idx = 0;
    return &g_partial_barrier_idx;
}

SHMEM_DEVICE bool in_pes(__gm__ uint32_t *pes, int my_pe, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        if ((int32_t)pes[i] == my_pe) {
            return true;
        }
    }
    return false;
}

SHMEM_DEVICE void shmemi_partial_barrier_npu_v3(shmemi_team_t *team,
                                                __gm__ int32_t *slot_base,
                                                __gm__ uint32_t *pes,
                                                uint32_t count)
{
    if (count == 0 || pes == nullptr) {
        return;
    }

    int vec_id = AscendC::GetBlockIdx();
    int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    int my_pe = shmemi_get_state()->team_pools[SHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int my_pe_in_team = (my_pe - start) / stride;

    int k = SHMEM_BARRIER_TG_DISSEM_KVAL;
    k = k < (int)count ? k : (int)count;
    k = k < vec_size ? k : vec_size;
    if (k <= 0) {
        k = 1;
    }

    for (uint32_t i = (uint32_t)vec_id; i < count; i += (uint32_t)k) {
        uint32_t remote_pe = pes[i];
        if ((int)remote_pe == my_pe_in_team) {
            shmemi_signal_set(slot_base, 1);
        } else {
            shmemi_signal_wait_until_eq_for_barrier(
                (__gm__ int32_t *)shmemi_ptr(slot_base, (int)(remote_pe * stride + start)), 1);
        }
    }
}

template<bool is_aiv_only = true>
SHMEM_DEVICE void shmemi_partial_barrier(shmem_team_t tid, __gm__ uint32_t *pes, uint32_t count)
{
    if (pes == nullptr || count == 0) {
        return;
    }

    auto state = shmemi_get_state();
    shmemi_team_t *team = state->team_pools[tid];

    int my_pe = state->team_pools[SHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int my_pe_in_team = (my_pe - start) / stride;

    if ((my_pe - start) % stride != 0) {
        // not in this team
        return;
    }
    uint32_t *g_partial_barrier_idx = get_partial_barrier_idx();
    uint32_t idx_snapshot = *g_partial_barrier_idx;
    uint32_t slot = idx_snapshot % SHMEM_PARTIAL_BARRIER_MAX_SLOTS;

    auto slot_sync = shmemi_get_team_partial_barrier_slot(team->team_idx, slot);
    auto slot_base = (__gm__ int32_t *)slot_sync;

    int vec_id   = AscendC::GetBlockIdx();
    int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    
    shmemi_barrier_core<is_aiv_only>();
    if (vec_id == 0) {
        uint32_t new_idx = *g_partial_barrier_idx + 1;
        if (new_idx >= SHMEM_PARTIAL_BARRIER_MAX_SLOTS) {
            auto s_sync = shmemi_get_team_partial_barrier_slot(team->team_idx, 0);
            auto s_base = (__gm__ int32_t *)s_sync;
            constexpr uint32_t clear_count = SHMEM_PARTIAL_BARRIER_PER_TEAM_SIZE / sizeof(int32_t);
            AscendC::LocalTensor<int32_t> ub_tensor_32;
            ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
            ub_tensor_32.address_.bufferAddr = SHMEM_INTERNAL_UB_BUF_START_ADDR;
            ub_tensor_32.address_.dataLen = clear_count;
            AscendC::Duplicate(ub_tensor_32, 0, clear_count);
            AscendC::GlobalTensor<int32_t> s_gm;
            s_gm.SetGlobalBuffer(s_base, clear_count * sizeof(int32_t));
            AscendC::DataCopy(s_gm, ub_tensor_32, clear_count);
            new_idx = 0;
        }
        *g_partial_barrier_idx = new_idx;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(reinterpret_cast<uint64_t>(g_partial_barrier_idx)));
    }

    shmemi_barrier_core<is_aiv_only>();
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(reinterpret_cast<uint64_t>(g_partial_barrier_idx)));
    if (*g_partial_barrier_idx == 0) {
        shmemi_barrier<is_aiv_only>(tid);
    }
    
    if (!in_pes(pes, my_pe_in_team, count)) {
        return;
    }

    shmemi_barrier_core<is_aiv_only>();

    if ASCEND_IS_AIV {
        shmemi_partial_barrier_npu_v3(team, slot_base, pes, count);
    }
    shmemi_barrier_core<is_aiv_only>();
}

#endif