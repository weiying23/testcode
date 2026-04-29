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
#include "shmem_api.h"
#include "team_kernel.h"

class kernel_state_test {
public:
    __aicore__ inline kernel_state_test() {}
    __aicore__ inline void Init(GM_ADDR gva, shmem_team_t team_id)
    {
        gva_gm = (__gm__ int *)gva;
        team_idx= team_id;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();
    }
    __aicore__ inline void Process()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        shmem_int32_p(gva_gm, shmem_n_pes(), rank);
        shmem_int32_p(gva_gm + 1U, shmem_my_pe(), rank);
        shmem_int32_p(gva_gm + 2U, shmem_team_my_pe(team_idx), rank);
        shmem_int32_p(gva_gm + 3U, shmem_team_n_pes(team_idx), rank);
        shmem_int32_p(gva_gm + 4U, shmem_team_translate_pe(team_idx,
            shmem_team_n_pes(team_idx) - 1, SHMEM_TEAM_WORLD), rank);
    }
private:
    __gm__ int *gva_gm;
    shmem_team_t team_idx;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void device_state_test(GM_ADDR gva, int team_id)
{
    kernel_state_test op;
    op.Init(gva, (shmem_team_t)team_id);
    op.Process();
}

void get_device_state(uint32_t block_dim, void* stream, uint8_t* gva, shmem_team_t team_id)
{
    device_state_test<<<block_dim, nullptr, stream>>>(gva, (int)team_id);
}