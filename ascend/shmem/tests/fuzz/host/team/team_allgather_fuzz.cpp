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
#include <iostream>
#include <ostream>
#include <secodefuzz/secodeFuzz.h>
#include <sstream>

#include "acl/acl_rt.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_sync.h"
#include "host/shmem_host_team.h"
#include "host_device/shmem_types.h"
#include "shmem_fuzz.h"

extern void team_allgather(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, shmem_team_t team_id,
                           uint32_t trans_count);

class ShmemTeamAllGatherFuzz : public testing::Test {
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

template <class T>
std::ostream &print_array(std::ostream &os, T &arr, size_t size, size_t start = 0, size_t stride = 1)
{
    os << "[ ";
    for (int j = start; j < size; j += stride) {
        os << arr[j] << " ";
    }
    os << "]";
    return os;
}

void team_all_gather_test(int rank_id, int n_ranks, aclrtStream stream, uint32_t trans_count)
{
    shmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = n_ranks / stride;

    uint32_t trans_size = trans_count * sizeof(int32_t);
    size_t total_size = team_size * trans_size;
    void *ptr = shmem_malloc(total_size);
    ASSERT_NE(ptr, nullptr);

    if (rank_id & 1) {
        // Team split
        shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

        // Initialize data
        constexpr uint32_t RANK_ID_OFFSET = 10;
        std::vector<int32_t> input(trans_count, (rank_id + RANK_ID_OFFSET) * (rank_id + RANK_ID_OFFSET));
        std::stringstream input_str;
        print_array(input_str, input, trans_count);
        std::printf("[rank#%d] input = %s;\n", rank_id, input_str.str().c_str());

        void *dst = (void *)((uint64_t)ptr + shmem_team_my_pe(team_odd) * trans_size);
        ASSERT_EQ(aclrtMemcpy(dst, trans_size, input.data(), trans_size, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

        // Execute AllGather
        team_allgather(1, stream, shmemx_get_ffts_config(), (uint8_t *)ptr, team_odd, trans_count);
        EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

        // Check results
        int32_t *y_host;
        EXPECT_EQ(aclrtMallocHost((void **)(&y_host), total_size), ACL_SUCCESS);
        EXPECT_EQ(aclrtMemcpy(y_host, total_size, ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);

        for (int team_id = 0; team_id < team_size; team_id++) {
            int32_t *team_data = y_host + trans_count * team_id;
            std::stringstream output_str;
            print_array(output_str, team_data, trans_count);
            std::printf("[rank#%d] data[team = %d] = %s;\n", rank_id, team_id, output_str.str().c_str());
            int32_t expect_value = (11 + team_id * 2) * (11 + team_id * 2);
            for (int i = 0; i < trans_count; i++) {
                EXPECT_EQ(team_data[i], expect_value);
            }
        }

        EXPECT_EQ(aclrtFreeHost(y_host), 0);
        shmem_team_destroy(team_odd);
    }
    shmem_free(ptr);
}

TEST_F(ShmemTeamAllGatherFuzz, shmem_team_all_gather_success)
{
    char fuzzName[] = "shmem_team_all_gather_success";
    uint64_t seed = 0;
    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        uint32_t trans_count = fuzz_get_ranged_number(FUZZ_VALUE_0_ID, 16, 16, 32);
        shmem_fuzz_multi_task(
            [&](int rank_id, int n_ranks, uint64_t local_mem_size) {
                shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
                team_all_gather_test(rank_id, n_ranks, scope.stream, trans_count);
            },
            1 * GiB, shmem_fuzz_gnpu_num());
    }
    DT_FUZZ_END()
}
