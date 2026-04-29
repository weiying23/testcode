/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "device/shmem_device_def.h"
#include "shmem_api.h"
#include "team_allgather_kernel.h"

extern "C" __global__ __aicore__ void device_team_all_gather_test(uint64_t config, GM_ADDR gva, int team_id)
{
    shmemx_set_ffts_config(config);
    int64_t team_rank = shmem_team_my_pe(team_id);
    int64_t team_size = shmem_team_n_pes(team_id);
    __gm__ int32_t* gva_gm = (__gm__ int32_t *)gva;
    AscendC::PipeBarrier<PIPE_ALL>();
    // All Gather
    for (int i = 0; i < team_size - 1; i++) {
        int64_t dst_rank = shmem_team_translate_pe(team_id, (team_rank + 1 + i) % team_size, SHMEM_TEAM_WORLD);
        shmem_put_int32_mem_nbi(gva_gm + 16 * team_rank, gva_gm + 16 * team_rank, 16, dst_rank);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    shmemi_barrier(team_id);
}

void team_allgather(uint32_t block_dim, void* stream, uint64_t config, uint8_t* gva, shmem_team_t team_id)
{
    device_team_all_gather_test<<<block_dim, nullptr, stream>>>(config, gva, (int)team_id);
}