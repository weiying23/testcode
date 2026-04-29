/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <cmath>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "shmemi_device_intf.h"
using namespace std;

namespace shm {
uint64_t g_team_mask = 0;
shmemi_team_t *g_shmem_team_pool = nullptr;

inline std::string team_config2string(shmemi_team_t *config)
{
    std::ostringstream oss;
    oss << "[team:" << config->team_idx;
    oss << ",npes:" << config->size;
    oss << ",mype:" << config->mype;
    oss << ",start:" << config->start;
    oss << ",stride:" << config->stride;
    oss << "]";
    return oss.str();
}

inline bool is_valid_team(shmem_team_t &team)
{
    return (g_state.is_shmem_initialized && g_shmem_team_pool != nullptr && team >= 0 && team < SHMEM_MAX_TEAMS &&
            ((g_team_mask >> team) & 1));
}

inline void device_team_destroy(int32_t team_idx)
{
    // device_ptr Free
    shmemi_team_t *device_team_ptr = g_state.team_pools[team_idx];
    if (device_team_ptr != nullptr) {
        if (aclrtFree((void *)device_team_ptr) != ACL_SUCCESS) {
            SHM_LOG_ERROR("aclrtFree for device_team_ptr failed for team " << team_idx);
        }
        g_state.team_pools[team_idx] = nullptr;
    }
}

inline int32_t device_team_update(int team_idx, shmemi_team_t *host_team_ptr)
{
    // device_ptr Malloc
    void *team_ptr = nullptr;
    SHMEM_CHECK_RET(aclrtMalloc(&team_ptr, sizeof(shmemi_team_t), ACL_MEM_MALLOC_NORMAL_ONLY), aclrtMalloc);
    auto ret = aclrtMemcpy((shmemi_team_t *)team_ptr, sizeof(shmemi_team_t), host_team_ptr, sizeof(shmemi_team_t),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        SHM_LOG_ERROR("memcpy device team info failed, ret: " << ret);
        SHMEM_CHECK_RET(aclrtFree(team_ptr), aclrtFree);
        return SHMEM_INNER_ERROR;
    }
    g_state.team_pools[team_idx] = (shmemi_team_t *)team_ptr;
    return SHMEM_SUCCESS;
}

int32_t shmemi_team_init_sync_pool()
{
    g_state.sync_pool = (uint64_t)shmem_malloc(SYNC_POOL_SIZE);
    if (g_state.sync_pool == 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("malloc sync pool failed.");
        return SHMEM_INNER_ERROR;
    }
    auto ret = aclrtMemset((void *)g_state.sync_pool, SYNC_POOL_SIZE, 0, SYNC_POOL_SIZE);
    if (ret != 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("memset sync pool failed, ret=" << ret);
        return SHMEM_INNER_ERROR;
    }
    return SHMEM_SUCCESS;
}

int32_t shmemi_team_init_sync_counter()
{
    g_state.sync_counter = (uint64_t)shmem_malloc(SYNC_COUNTERS_SIZE);
    if (g_state.sync_counter == 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("malloc sync counter failed.");
        return SHMEM_INNER_ERROR;
    }
    auto ret = aclrtMemset((void *)g_state.sync_counter, SYNC_COUNTERS_SIZE, 0, SYNC_COUNTERS_SIZE);
    if (ret != 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("memset sync counter failed.");
        return SHMEM_INNER_ERROR;
    }
    return SHMEM_SUCCESS;
}

int32_t shmemi_team_init_core_sync_pool()
{
    auto ret = aclrtMalloc((void **)&(g_state.core_sync_pool), SHMEM_CORE_SYNC_POOL_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0 || g_state.core_sync_pool == 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("malloc core sync pool failed.");
        return SHMEM_INNER_ERROR;
    }
    ret = aclrtMemset((void *)g_state.core_sync_pool, SHMEM_CORE_SYNC_POOL_SIZE, 0, SHMEM_CORE_SYNC_POOL_SIZE);
    if (ret != 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("memset core sync pool failed.");
        return SHMEM_INNER_ERROR;
    }
    return SHMEM_SUCCESS;
}

int32_t shmemi_team_init_core_sync_counter()
{
    auto ret = aclrtMalloc((void **)&(g_state.core_sync_counter), SHMEM_CORE_SYNC_COUNTER_SIZE,
        ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0 || g_state.core_sync_counter == 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("malloc core sync counter failed.");
        return SHMEM_INNER_ERROR;
    }
    ret = aclrtMemset((void *)g_state.core_sync_counter, SHMEM_CORE_SYNC_COUNTER_SIZE, 0, SHMEM_CORE_SYNC_COUNTER_SIZE);
    if (ret != 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("memset core sync counter failed.");
        return SHMEM_INNER_ERROR;
    }
    return SHMEM_SUCCESS;
}

int32_t shmemi_team_init_partial_barrier_pool()
{
    g_state.partial_barrier_pool = (uint64_t)shmem_malloc(SHMEM_PARTIAL_BARRIER_POOL_SIZE);
    if (g_state.partial_barrier_pool == 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("malloc partial barrier pool failed.");
        return SHMEM_INNER_ERROR;
    }

    auto ret = aclrtMemset((void *)g_state.partial_barrier_pool, SHMEM_PARTIAL_BARRIER_POOL_SIZE,
        0, SHMEM_PARTIAL_BARRIER_POOL_SIZE);
    if (ret != 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("memset partial barrier pool failed, ret: " << ret);
        return SHMEM_INNER_ERROR;
    }

    return SHMEM_SUCCESS;
}

int32_t shmemi_team_init(int32_t rank, int32_t size)
{
    /* Initialize SHMEM_TEAM_WORLD */
    g_shmem_team_pool = (shmemi_team_t *)calloc(SHMEM_MAX_TEAMS, sizeof(shmemi_team_t));
    if (g_shmem_team_pool == nullptr) {
        SHM_LOG_ERROR("malloc host shmem team pool failed.");
        return SHMEM_INNER_ERROR;
    }
    for (int i = 0; i < SHMEM_MAX_TEAMS; i++) {
        g_shmem_team_pool[i] = shmemi_team_t{-1, -1, -1, -1, -1};
    }

    shmemi_team_t &shmem_team_world = g_shmem_team_pool[SHMEM_TEAM_WORLD];
    shmem_team_world.team_idx = SHMEM_TEAM_WORLD;
    shmem_team_world.start = 0;
    shmem_team_world.stride = 1;
    shmem_team_world.size = size;
    shmem_team_world.mype = rank;
    g_team_mask |= 1ULL << SHMEM_TEAM_WORLD;
    SHMEM_CHECK_RET(device_team_update(SHMEM_TEAM_WORLD, &shmem_team_world));

    /* Initialize TEAM SYNC */
    auto ret = shmemi_team_init_sync_pool();
    if (ret != 0) {
        return ret;
    }

    ret = shmemi_team_init_sync_counter();
    if (ret != 0) {
        return ret;
    }

    ret = shmemi_team_init_core_sync_pool();
    if (ret != 0) {
        return ret;
    }

    ret = shmemi_team_init_core_sync_counter();
    if (ret != 0) {
        return ret;
    }

    return shmemi_team_init_partial_barrier_pool();
}

int32_t first_free_idx_fetch()
{
    int32_t shmem_max_teams = SHMEM_MAX_TEAMS;
    for (int32_t i = 0; i < shmem_max_teams; i++) {
        if (!((g_team_mask >> i) & 1)) {
            g_team_mask |= 1ULL << i;
            return i;
        }
    }
    return -1;
}

int32_t shmemi_team_finalize()
{
    /* Destroy all undestroyed teams */
    int32_t shmem_max_teams = SHMEM_MAX_TEAMS;
    for (int32_t i = 0; i < shmem_max_teams; i++) {
        shmem_team_t team = i;
        if (is_valid_team(team)) {
            shmem_team_destroy(team);
        }
    }

    if (g_state.sync_counter != 0) {
        shmem_free(reinterpret_cast<void *>(g_state.sync_counter));
        g_state.sync_counter = 0;
    }
    if (g_state.sync_pool != 0) {
        shmem_free(reinterpret_cast<void *>(g_state.sync_pool));
        g_state.sync_pool = 0;
    }
    if (g_state.partial_barrier_pool != 0) {
        shmem_free(reinterpret_cast<void *>(g_state.partial_barrier_pool));
        g_state.partial_barrier_pool = 0;
    }
    if (g_state.core_sync_counter != 0) {
        SHMEM_CHECK_RET(aclrtFree(reinterpret_cast<void *>(g_state.core_sync_counter)), aclrtFree);
        g_state.core_sync_counter = 0;
    }
    if (g_state.core_sync_pool != 0) {
        SHMEM_CHECK_RET(aclrtFree(reinterpret_cast<void *>(g_state.core_sync_pool)), aclrtFree);
        g_state.core_sync_pool = 0;
    }
    if (g_shmem_team_pool != nullptr) {
        free(g_shmem_team_pool);
        g_shmem_team_pool = nullptr;
    }
    return 0;
}

}  // namespace shm

int32_t shmem_team_split_strided_precheck(shmem_team_t parent_team, int32_t pe_start, int32_t pe_stride,
                                          int32_t pe_size, shmem_team_t *&new_team)
{
    if (new_team == nullptr) {
        SHM_LOG_ERROR("output team is null.");
        return SHMEM_INVALID_PARAM;
    }

    *new_team = SHMEM_TEAM_INVALID;
    if (!shm::is_valid_team(parent_team)) {
        SHM_LOG_ERROR("input parent team is invalid!, team: " << parent_team);
        return SHMEM_INVALID_PARAM;
    }

    shmemi_team_t *src_team = &shm::g_shmem_team_pool[parent_team];
    if (pe_start >= SHMEM_MAX_RANKS || pe_stride >= SHMEM_MAX_RANKS || pe_size > SHMEM_MAX_RANKS) {
        SHM_LOG_ERROR("create team failed, input invalid, pe_start:" << pe_start << " pe_size:" << pe_size
                                                                     << " pe_stride:" << pe_stride << " parent:"
                                                                     << shm::team_config2string(src_team));
        return SHMEM_INVALID_PARAM;
    }
    return SHMEM_SUCCESS;
}

int32_t shmem_team_split_strided(shmem_team_t parent_team, int32_t pe_start, int32_t pe_stride, int32_t pe_size,
                                 shmem_team_t *new_team)
{
    auto ret = shmem_team_split_strided_precheck(parent_team, pe_start, pe_stride, pe_size, new_team);
    if (ret != 0) {
        return ret;
    }

    shmemi_team_t *src_team = &shm::g_shmem_team_pool[parent_team];
    int32_t global_pe = src_team->mype;
    int32_t global_pe_start = src_team->start + pe_start * src_team->stride;
    int32_t global_pe_stride = src_team->stride * pe_stride;
    int32_t global_pe_end = global_pe_start + global_pe_stride * (pe_size - 1);

    if (pe_start < 0 || pe_start >= src_team->size || pe_size <= 0 || pe_size > src_team->size || pe_stride < 1) {
        SHM_LOG_ERROR("create team failed, input invalid:" << pe_start << ":" << pe_size << ":" << pe_stride << ":"
            << shm::team_config2string(src_team));
        return SHMEM_INVALID_PARAM;
    }

    if (global_pe_start >= shmem_n_pes() || global_pe_end >= shmem_n_pes()) {
        SHM_LOG_ERROR("create team failed, large than world size:" << pe_start << ":" << pe_size << ":" << pe_stride
            << ":" << shmem_n_pes() << ":" << shm::team_config2string(src_team));
        return SHMEM_INVALID_PARAM;
    }

    shmemi_team_t my_team;
    my_team.mype = (global_pe - global_pe_start) / global_pe_stride;

    if (global_pe < global_pe_start || (global_pe - global_pe_start) % global_pe_stride || my_team.mype >= pe_size) {
        SHM_LOG_INFO("This PE is not a member of the new team.");
        return 0;
    }

    my_team.start = global_pe_start;
    my_team.stride = global_pe_stride;
    my_team.size = pe_size;

    my_team.team_idx = shm::first_free_idx_fetch();
    if (my_team.team_idx == -1) {
        SHM_LOG_ERROR("create team failed, team num is full!");
        return SHMEM_INNER_ERROR;
    }

    shm::g_shmem_team_pool[my_team.team_idx] = my_team;
    if (shm::device_team_update(my_team.team_idx, &shm::g_shmem_team_pool[my_team.team_idx]) != 0) {
        shmem_team_destroy(my_team.team_idx);
        SHM_LOG_ERROR("create team failed, malloc device state failed!");
        return SHMEM_INNER_ERROR;
    }
    if (shm::update_device_state() != 0) {
        shmem_team_destroy(my_team.team_idx);
        SHM_LOG_ERROR("create team failed, update state failed!");
        return SHMEM_INNER_ERROR;
    }
    SHM_LOG_INFO("create team success:" << shm::team_config2string(&my_team));
    *new_team = my_team.team_idx;
    return 0;
}

int shmemi_team_split_2d_precheck(shmem_team_t p_team, int x_range, shmem_team_t *&x_team, shmem_team_t *&y_team)
{
    if (x_team == nullptr || y_team == nullptr) {
        SHM_LOG_ERROR("output team is null.");
        return SHMEM_INVALID_PARAM;
    }

    if (x_range <= 0) {
        SHM_LOG_ERROR("input x range must be larger than 0.");
        return SHMEM_INVALID_PARAM;
    }

    *x_team = SHMEM_TEAM_INVALID;
    *y_team = SHMEM_TEAM_INVALID;
    if (!shm::is_valid_team(p_team)) {
        SHM_LOG_ERROR("input parent team is invalid!, team: " << p_team);
        return SHMEM_INVALID_PARAM;
    }

    return SHMEM_SUCCESS;
}

int shmemi_team_split_2d_x(shmem_team_t &parent_team, int32_t &x_team_counts, int32_t &src_size,
                           int &x_range, shmem_team_t *&x_team)
{
    int start = 0;
    int errorCode = 0;
    shmemi_team_t *src_team = &shm::g_shmem_team_pool[parent_team];

    for (int i = 0; i < x_team_counts; ++i) {
        shmem_team_t my_xteam;
        int x_size = (i == x_team_counts - 1 && src_size % x_range) ? src_size % x_range : x_range;
        errorCode = shmem_team_split_strided(parent_team, start, 1, x_size, &my_xteam);
        if (errorCode != 0) {
            SHM_LOG_WARN("create x-axis team " << (i + 1) << " of " << x_team_counts << " failed");
        }

        start += x_range;

        if (my_xteam != SHMEM_TEAM_INVALID) {
            if (*x_team == SHMEM_TEAM_INVALID) {
                *x_team = my_xteam;
                SHM_LOG_INFO("Current pe is " << src_team->mype << " , split x-axis succeed for x- " << i);
            } else {
                return SHMEM_INNER_ERROR;
            }
        }
    }
    return SHMEM_SUCCESS;
}

int shmemi_team_split_2d_y(shmem_team_t &parent_team, int32_t &y_team_counts, int32_t &src_size,
                           int &x_range, shmem_team_t *&y_team)
{
    int start = 0;
    int errorCode = 0;
    shmemi_team_t *src_team = &shm::g_shmem_team_pool[parent_team];

    for (int i = 0; i < y_team_counts; ++i) {
        shmem_team_t my_yteam;
        int remainder = src_size % x_range;
        int y_range = src_size / x_range;
        int y_size = (remainder && i < remainder) ? y_range + 1 : y_range;

        errorCode = shmem_team_split_strided(parent_team, start, x_range, y_size, &my_yteam);
        if (errorCode != 0) {
            SHM_LOG_WARN("create y-axis team " << (i + 1) << " of " << y_team_counts << " failed");
        }

        start += 1;
        if (my_yteam != SHMEM_TEAM_INVALID) {
            if (*y_team == SHMEM_TEAM_INVALID) {
                *y_team = my_yteam;
                SHM_LOG_INFO("Current pe is " << src_team->mype << " , split y-axis succeed for y- " << i);
            } else {
                return SHMEM_INNER_ERROR;
            }
        }
    }
    return SHMEM_SUCCESS;
}

int shmem_team_split_2d(shmem_team_t parent_team, int x_range, shmem_team_t *x_team, shmem_team_t *y_team)
{
    auto ret = shmemi_team_split_2d_precheck(parent_team, x_range, x_team, y_team);
    if (ret != 0) {
        return ret;
    }

    shmemi_team_t *src_team = &shm::g_shmem_team_pool[parent_team];
    int32_t src_start = src_team->start;
    int32_t src_stride = src_team->stride;
    int32_t src_size = src_team->size;
    double x_team_counts_double = std::ceil(src_size / static_cast<double>(x_range));
    int32_t x_team_counts = static_cast<int32_t>(x_team_counts_double);
    int32_t y_team_counts = x_range;

    if (x_range > src_size) {
        x_range = src_size;
    }

    ret = shmemi_team_split_2d_x(parent_team, x_team_counts, src_size, x_range, x_team);
    if (ret != 0) {
        return ret;
    }

    return shmemi_team_split_2d_y(parent_team, y_team_counts, src_size, x_range, y_team);
}

int32_t shmem_team_translate_pe(shmem_team_t src_team, int32_t src_pe, shmem_team_t dest_team)
{
    if (!shm::is_valid_team(src_team) || !shm::is_valid_team(dest_team)) {
        return -1;
    }

    shmemi_team_t *src_team_ptr = &shm::g_shmem_team_pool[src_team];
    shmemi_team_t *dest_team_ptr = &shm::g_shmem_team_pool[dest_team];

    if (src_pe > src_team_ptr->size) {
        return -1;
    }

    int32_t global_pe = src_team_ptr->start + src_pe * src_team_ptr->stride;
    int32_t pe_start = dest_team_ptr->start;
    int32_t pe_stride = dest_team_ptr->stride;
    int32_t pe_size = dest_team_ptr->size;

    int32_t n = (global_pe - pe_start) / pe_stride;
    if (global_pe < pe_start || (global_pe - pe_start) % pe_stride || n >= pe_size) {
        return -1;
    }

    return n;
}

void shmem_team_destroy(shmem_team_t team)
{
    if (!shm::is_valid_team(team)) {
        SHM_LOG_WARN("input team is invalid!, team: " << team);
        return;
    }

    shm::device_team_destroy(team);
    shm::g_team_mask ^= 1ULL << team;
    if (shm::update_device_state() != SHMEM_SUCCESS) {
        SHM_LOG_WARN("update state failed when destroy team!");
    }
}

int32_t shmem_my_pe(void)
{
    return shm::g_state.mype;
}

int32_t shmem_n_pes(void)
{
    return shm::g_state.npes;
}

int32_t shmem_team_my_pe(shmem_team_t team)
{
    if (shm::is_valid_team(team)) {
        return shm::g_shmem_team_pool[team].mype;
    } else {
        return -1;
    }
}

int32_t shmem_team_n_pes(shmem_team_t team)
{
    if (shm::is_valid_team(team)) {
        return shm::g_shmem_team_pool[team].size;
    } else {
        return -1;
    }
}

int shmem_team_get_config(shmem_team_t team, shmem_team_config_t *config)
{
    SHMEM_CHECK_RET(config == nullptr);
    if (shm::is_valid_team(team)) {
        config->num_contexts = 0;
        return 0;
    } else {
        return SHMEM_INVALID_PARAM;
    }
}
