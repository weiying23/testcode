/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHEMEI_DEVICE_HANDLE_H
#define SHEMEI_DEVICE_HANDLE_H

#include "shmemi_device_barrier.h"

#include "kernel_operator.h"

SHMEM_DEVICE void shmemi_barrier_cross_host(shmemi_team_t *team)
{
    if (AscendC::GetBlockIdx() != 0)
        return;

    int my_pe = shmemi_get_state()->team_pools[SHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    auto sync_array = shmemi_get_team_sync_array(team->team_idx);
    auto sync_counter = shmemi_get_team_sync_counter(team->team_idx);

    int shift = 1;
    int my_pe_in_team = (my_pe - start) / stride;
    int32_t count = shmemi_load((__gm__ int32_t *)sync_counter) + 1;
    shmemi_store((__gm__ int32_t *)sync_counter, count);
    dcci_cacheline((__gm__ uint8_t *)sync_counter);
    while (shift < size) {
        int pre_pe_in_team = (my_pe_in_team - shift + size) % size;
        int next_pe_in_team = (my_pe_in_team + shift) % size;

        int pre_pe = start + pre_pe_in_team * stride;
        int next_pe = start + next_pe_in_team * stride;

        // signal next pe
        shmemi_highlevel_signal_set((__gm__ int32_t *)(sync_array + my_pe), (__gm__ int32_t *)sync_counter, next_pe);

        // wait pre pe
        shmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t *)(sync_array + pre_pe), count);

        shift *= SHIFT_MULTIPLIER;
    }
}

SHMEM_DEVICE void shmemi_handle(shmem_team_t tid)
{
    shmemi_team_t *team = shmemi_get_state()->team_pools[tid];

    int mype = shmemi_get_state()->team_pools[SHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;

    if ((mype - start) % stride != 0) {
        // not in this team
        return;
    }

    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR
                                                                        + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;

    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == mype) {
            continue;
        }
        shmemi_roce_quiet(peer, 0, ub_tensor_64, ub_tensor_32);
    }

    if ASCEND_IS_AIV {
        shmemi_barrier_cross_host(team);
    }
}

#endif