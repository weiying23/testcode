/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <secodefuzz/secodeFuzz.h>
#include <sys/types.h>

#include "acl/acl_rt.h"
#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_team.h"
#include "host_device/shmem_types.h"
#include "shmem_fuzz.h"

extern void get_device_state(uint32_t block_dim, void *stream, uint8_t *gva, shmem_team_t team_id);

class ShmemTeamHostFuzz : public testing::Test {
public:
    void SetUp()
    {
        DT_Enable_Leak_Check(0, 0);
        DT_Set_Running_Time_Second(SHMEM_FUZZ_RUNNING_SECONDS);
    }

    void TearDown()
    {
    }
};

static void test_shmem_team_strided(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
    ASSERT_EQ(shmem_init_status(), SHMEM_STATUS_IS_INITIALIZED);

    shmem_team_t team;
    int start = rank_id & 1;
    int stride = 2;
    int team_size = n_ranks / stride;
    ASSERT_EQ(shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, team_size, &team), SHMEM_SUCCESS);
    ASSERT_EQ(shmem_n_pes(), n_ranks);
    ASSERT_EQ(shmem_my_pe(), rank_id);

    int team_rank_id = shmem_team_my_pe(team);
    int team_n_ranks = shmem_team_n_pes(team);
    ASSERT_EQ(team_n_ranks, team_size);
    ASSERT_EQ(team_rank_id, rank_id / stride);

    ASSERT_EQ(shmem_team_translate_pe(team, team_rank_id, SHMEM_TEAM_WORLD), rank_id);
    ASSERT_EQ(shmem_team_translate_pe(SHMEM_TEAM_WORLD, rank_id, team), team_rank_id);

    ASSERT_EQ(shmem_team_translate_pe(team, team_rank_id, SHMEM_TEAM_INVALID), -1);
    ASSERT_EQ(shmem_team_translate_pe(team, team_size, SHMEM_TEAM_WORLD), -1);

    shmem_team_destroy(team);
}

TEST_F(ShmemTeamHostFuzz, shmem_team_host_team_strided_success)
{
    char fuzzName[] = "shmem_team_host_team_strided_success";
    uint64_t seed = 0;
    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        uint64_t local_mem_size = fuzz_get_ranged_number(FUZZ_VALUE_0_ID, 1, 1, 512) * 2 * MiB;
        shmem_fuzz_multi_task(test_shmem_team_strided, local_mem_size, shmem_fuzz_gnpu_num());
    }
    DT_FUZZ_END()
}

static void test_shmem_team_2d(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
    ASSERT_EQ(shmem_init_status(), SHMEM_STATUS_IS_INITIALIZED);

    shmem_team_t team_x;
    shmem_team_t team_y;
    int x_range = 2;
    int y_range = n_ranks / 2;
    ASSERT_EQ(shmem_team_split_2d(SHMEM_TEAM_WORLD, x_range, &team_x, &team_y), SHMEM_SUCCESS);

    ASSERT_EQ(shmem_team_n_pes(team_x), x_range);
    ASSERT_EQ(shmem_team_n_pes(team_y), y_range);
    ASSERT_EQ(shmem_n_pes(), n_ranks);
    ASSERT_EQ(shmem_my_pe(), rank_id);

    int rank_id_x = shmem_team_my_pe(team_x);
    int rank_id_y = shmem_team_my_pe(team_y);

    ASSERT_EQ(shmem_team_translate_pe(team_x, rank_id_x, SHMEM_TEAM_WORLD), rank_id);
    ASSERT_EQ(shmem_team_translate_pe(team_y, rank_id_y, SHMEM_TEAM_WORLD), rank_id);

    ASSERT_EQ(shmem_team_translate_pe(SHMEM_TEAM_WORLD, rank_id, team_x), rank_id_x);
    ASSERT_EQ(shmem_team_translate_pe(SHMEM_TEAM_WORLD, rank_id, team_y), rank_id_y);

    ASSERT_EQ(shmem_team_translate_pe(team_x, rank_id_x, team_y), rank_id_y);
    ASSERT_EQ(shmem_team_translate_pe(team_y, rank_id_y, team_x), rank_id_x);

    shmem_team_destroy(team_x);
    shmem_team_destroy(team_y);
}

TEST_F(ShmemTeamHostFuzz, shmem_team_host_team_2d_success)
{
    char fuzzName[] = "shmem_team_host_team_2d_success";
    uint64_t seed = 0;
    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        uint64_t local_mem_size = fuzz_get_ranged_number(FUZZ_VALUE_0_ID, 1, 1, 512) * 2 * MiB;
        shmem_fuzz_multi_task(test_shmem_team_2d, local_mem_size, shmem_fuzz_gnpu_num());
    }
    DT_FUZZ_END()
}

static void test_shmem_team_device(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
    ASSERT_EQ(shmem_init_status(), SHMEM_STATUS_IS_INITIALIZED);

    shmem_team_t team;
    int start = rank_id & 1;
    int stride = 2;
    int team_size = n_ranks / stride;
    ASSERT_EQ(shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, team_size, &team), SHMEM_SUCCESS);

    int *y_host;
    size_t data_size = 5 * sizeof(int);
    ASSERT_EQ(aclrtMallocHost((void **)(&y_host), data_size), ACL_SUCCESS);

    uint32_t block_dim = 1;
    void *ptr = shmem_malloc(data_size);
    int32_t device_id;
    ASSERT_EQ(aclrtGetDevice(&device_id), ACL_SUCCESS);
    get_device_state(block_dim, scope.stream, (uint8_t *)ptr, team);
    ASSERT_EQ(aclrtSynchronizeStream(scope.stream), ACL_SUCCESS);
    sleep(1);

    ASSERT_EQ(aclrtMemcpy(y_host, data_size, ptr, data_size, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);

    if (rank_id & 1) {
        int idx = 0;
        ASSERT_EQ(y_host[idx++], n_ranks);
        ASSERT_EQ(y_host[idx++], rank_id);
        ASSERT_EQ(y_host[idx++], rank_id / stride);
        ASSERT_EQ(y_host[idx++], n_ranks / stride);
        ASSERT_EQ(y_host[idx++], (n_ranks / stride - 1) * stride + rank_id % stride);
    }

    EXPECT_EQ(aclrtFreeHost(y_host), 0);
}

TEST_F(ShmemTeamHostFuzz, shmem_team_host_team_device_success)
{
    char fuzzName[] = "shmem_team_host_team_device_success";
    uint64_t seed = 0;
    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        uint64_t local_mem_size = fuzz_get_ranged_number(FUZZ_VALUE_0_ID, 1, 1, 512) * 2 * MiB;
        shmem_fuzz_multi_task(test_shmem_team_device, local_mem_size, shmem_fuzz_gnpu_num());
    }
    DT_FUZZ_END()
}
