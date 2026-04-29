/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_TEAM_H
#define SHMEM_DEVICE_TEAM_H

#include "host_device/shmem_types.h"
#include "internal/host_device/shmemi_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns the PE number of the local PE
 *
 * @return Integer between 0 and npes - 1
 */
SHMEM_DEVICE int shmem_my_pe(void)
{
    return shmemi_get_state()->team_pools[SHMEM_TEAM_WORLD]->mype;
}

/**
 * @brief Returns the number of PEs running in the program.
 *
 * @return Number of PEs in the program.
 */
SHMEM_DEVICE int shmem_n_pes(void)
{
    return shmemi_get_state()->team_pools[SHMEM_TEAM_WORLD]->size;
}

/**
 * @brief Returns the number of the calling PE in the specified team.
 *
 * @param team              [in] A team handle.
 *
 * @return The number of the calling PE within the specified team.
 *         If the team handle is SHMEM_TEAM_INVALID, returns -1.
 */
SHMEM_DEVICE int shmem_team_my_pe(shmem_team_t team)
{
    if (team == SHMEM_TEAM_INVALID) {
        return -1;
    } else {
        shmemi_team_t *src_team_ptr = shmemi_get_state()->team_pools[team];
        return (src_team_ptr != nullptr ? src_team_ptr->mype : -1);
    }
}

/**
 * @brief Returns the number of PEs in the specified team.
 *
 * @param team              [in] A team handle.
 *
 * @return The number of PEs in the specified team.
 *         If the team handle is SHMEM_TEAM_INVALID, returns -1.
 */
SHMEM_DEVICE int shmem_team_n_pes(shmem_team_t team)
{
    if (team == SHMEM_TEAM_INVALID) {
        return -1;
    } else {
        shmemi_team_t *src_team_ptr = shmemi_get_state()->team_pools[team];
        return (src_team_ptr != nullptr ? src_team_ptr->size : -1);
    }
}

/**
 * @brief Translate a given PE number in one team into the corresponding PE number in another team.
 *
 * @param src_team           [in] A SHMEM team handle.
 * @param src_pe             [in] The PE number in src_team.
 * @param dest_team          [in] A SHMEM team handle.
 *
 * @return The number of PEs in the specified team.
 *         If the team handle is SHMEM_TEAM_INVALID, returns -1.
 */
SHMEM_DEVICE int shmem_team_translate_pe(shmem_team_t src_team, int src_pe, shmem_team_t dest_team)
{
    if (src_team == SHMEM_TEAM_INVALID || dest_team == SHMEM_TEAM_INVALID) {
        return -1;
    }
    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();

    shmemi_team_t *src_team_ptr = device_state->team_pools[src_team];
    shmemi_team_t *dest_team_ptr = device_state->team_pools[dest_team];

    if (src_pe > src_team_ptr->size) {
        return -1;
    }

    int global_pe = src_team_ptr->start + src_pe * src_team_ptr->stride;
    int pe_start = dest_team_ptr->start;
    int pe_stride = dest_team_ptr->stride;
    int pe_size = dest_team_ptr->size;

    int n = (global_pe - pe_start) / pe_stride;
    if (global_pe < pe_start || (global_pe - pe_start) % pe_stride || n >= pe_size) {
        return -1;
    }

    return n;
}

#ifdef __cplusplus
}
#endif

#endif