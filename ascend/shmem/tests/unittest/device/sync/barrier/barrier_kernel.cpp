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

extern "C" SHMEM_GLOBAL void increase(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    shmemx_set_ffts_config(config);

#ifdef __DAV_C220_CUBE__
    // scalar unit of cube core is not affected by barrier
    shmem_barrier_all();
    shmem_barrier_all();
#endif

#ifdef __DAV_C220_VEC__
    uint64_t val = shmemi_load((__gm__ uint64_t *)addr);

    shmem_barrier_all();
    GM_ADDR remote = shmemi_ptr(addr, (rank_id + 1) % rank_size);
    shmemi_store((__gm__ uint64_t *)remote, val + 1);
    shmem_barrier_all();
#endif
}

extern "C" SHMEM_GLOBAL void increase_vec(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    shmemx_set_ffts_config(config);

#ifdef __DAV_C220_VEC__
    uint64_t val = shmemi_load((__gm__ uint64_t *)addr);

    shmemx_barrier_all_vec();
    GM_ADDR remote = shmemi_ptr(addr, (rank_id + 1) % rank_size);
    shmemi_store((__gm__ uint64_t *)remote, val + 1);
    shmemx_barrier_all_vec();
#endif
}

extern "C" SHMEM_GLOBAL void increase_odd_team(uint64_t config, GM_ADDR addr, int rank_id,
    int rank_size, shmem_team_t team_id) {
    shmemx_set_ffts_config(config);

#ifdef __DAV_C220_CUBE__
    // scalar unit of cube core is not affected by barrier
    shmem_barrier_all();
    shmem_barrier_all();
#endif

#ifdef __DAV_C220_VEC__
    uint64_t val = shmemi_load((__gm__ uint64_t *)addr);

    shmem_barrier(team_id);
    if (rank_id & 1) {
        GM_ADDR remote = shmemi_ptr(addr, (rank_id + 2) % rank_size);
        shmemi_store((__gm__ uint64_t *)remote, val + 1);
    }
    shmem_barrier(team_id);
#endif
}

extern "C" SHMEM_GLOBAL void increase_vec_odd_team(uint64_t config, GM_ADDR addr, int rank_id,
    int rank_size, shmem_team_t team_id) {
    shmemx_set_ffts_config(config);

#ifdef __DAV_C220_VEC__
    uint64_t val = shmemi_load((__gm__ uint64_t *)addr);

    shmemx_barrier_vec(team_id);
    if (rank_id & 1) {
        GM_ADDR remote = shmemi_ptr(addr, (rank_id + 2) % rank_size);
        shmemi_store((__gm__ uint64_t *)remote, val + 1);
    }
    shmemx_barrier_vec(team_id);
#endif
}

extern "C" SHMEM_GLOBAL void partial_increase(uint64_t config,
    GM_ADDR addr, GM_ADDR pes_addr, uint32_t count, int rank_id, int rank_size, shmem_team_t team_id)
{
    shmemx_set_ffts_config(config);

#ifdef __DAV_C220_CUBE__
    // scalar unit of cube core is not affected by barrier
    __gm__ uint32_t *pes = (__gm__ uint32_t *)pes_addr;

    shmemx_partial_barrier(team_id, pes, count);
    shmemx_partial_barrier(team_id, pes, count);
#endif

#ifdef __DAV_C220_VEC__
    uint64_t val = shmemi_load((__gm__ uint64_t *)addr);
    __gm__ uint32_t *pes = (__gm__ uint32_t *)pes_addr;
    auto state = shmemi_get_state();
    shmemi_team_t *team = state->team_pools[team_id];
    int start = team->start;
    int stride = team->stride;
    shmemx_partial_barrier(team_id, pes, count);
    int team_pe = shmem_team_my_pe(team_id);
    int peer = (team_pe + 1) % count;
    if (team_pe < count) {
        GM_ADDR remote = shmemi_ptr(addr, peer * stride + start);
        shmemi_store((__gm__ uint64_t *)remote, val + 1);
    }
    shmemx_partial_barrier(team_id, pes, count);
#endif
}

extern "C" SHMEM_GLOBAL void partial_increase_vec(uint64_t config,
    GM_ADDR addr, GM_ADDR pes_addr, uint32_t count, int rank_id, int rank_size, shmem_team_t team_id)
{
    shmemx_set_ffts_config(config);

#ifdef __DAV_C220_VEC__
    uint64_t val = shmemi_load((__gm__ uint64_t *)addr);
    __gm__ uint32_t *pes = (__gm__ uint32_t *)pes_addr;
    auto state = shmemi_get_state();
    shmemi_team_t *team = state->team_pools[team_id];
    int start = team->start;
    int stride = team->stride;
    shmemx_partial_barrier_vec(team_id, pes, count);
    int team_pe = shmem_team_my_pe(team_id);
    int peer = (team_pe + 1) % count;
    if (team_pe < count) {
        GM_ADDR remote = shmemi_ptr(addr, peer * stride + start);
        shmemi_store((__gm__ uint64_t *)remote, val + 1);
    }
    shmemx_partial_barrier_vec(team_id, pes, count);
#endif
}

void increase_do(void* stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size)
{
    increase<<<16, nullptr, stream>>>(config, addr, rank_id, rank_size);
}

void increase_vec_do(void* stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size)
{
    increase_vec<<<16, nullptr, stream>>>(config, addr, rank_id, rank_size);
}

void increase_do_odd_team(void* stream, uint64_t config, uint8_t *addr, int rank_id,
    int rank_size, shmem_team_t team_id)
{
    increase_odd_team<<<16, nullptr, stream>>>(config, addr, rank_id, rank_size, team_id);
}

void increase_vec_do_odd_team(void* stream, uint64_t config, uint8_t *addr, int rank_id,
    int rank_size, shmem_team_t team_id)
{
    increase_vec_odd_team<<<16, nullptr, stream>>>(config, addr, rank_id, rank_size, team_id);
}

void partial_increase_do(void *stream, uint64_t config,
    uint8_t *addr, uint8_t *pes_addr, uint32_t count, int rank_id, int rank_size, shmem_team_t team_id)
{
    partial_increase<<<16, nullptr, stream>>>(config, addr, pes_addr, count, rank_id, rank_size, team_id);
}

void partial_increase_vec_do(void *stream, uint64_t config,
    uint8_t *addr, uint8_t *pes_addr, uint32_t count, int rank_id, int rank_size, shmem_team_t team_id)
{
    partial_increase_vec<<<16, nullptr, stream>>>(config, addr, pes_addr, count, rank_id, rank_size, team_id);
}