/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <secodefuzz/secodeFuzz.h>

#include "host/shmem_host_heap.h"
#include "shmem_fuzz.h"

class ShmemHostHeapFuzz : public testing::Test {
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

TEST_F(ShmemHostHeapFuzz, shmem_malloc_free_success)
{
    char fuzzName[] = "shmem_malloc_free_success";
    uint64_t seed = 0;
    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        const int process_count = shmem_fuzz_gnpu_num();
        uint64_t local_mem_size = 1 * GiB;
        uint64_t alloc_size_0 = fuzz_get_ranged_number(FUZZ_VALUE_0_ID, 10 * MiB, 1 * MiB, local_mem_size / 2);
        uint64_t alloc_size_1 = fuzz_get_ranged_number(FUZZ_VALUE_1_ID, 100 * MiB, 1 * MiB, local_mem_size / 2);
        shmem_fuzz_multi_task(
            [&](int rank_id, int n_ranks, uint64_t local_mem_size) {
                shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
                auto ptr0 = shmem_malloc(alloc_size_0);
                auto ptr1 = shmem_malloc(alloc_size_1);
                ASSERT_NE(ptr0, nullptr);
                ASSERT_NE(ptr1, nullptr);
                shmem_free(ptr0);
                shmem_free(ptr1);
            },
            local_mem_size, process_count);
    }
    DT_FUZZ_END()
}

TEST_F(ShmemHostHeapFuzz, shmem_calloc_free_success)
{
    char fuzzName[] = "shmem_calloc_free_success";
    uint64_t seed = 0;
    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        const int process_count = shmem_fuzz_gnpu_num();
        uint64_t local_mem_size = 1 * GiB;
        uint64_t alloc_size_0 = fuzz_get_ranged_number(FUZZ_VALUE_0_ID, 10 * MiB, 1 * MiB, local_mem_size / 2);
        uint64_t alloc_size_1 = fuzz_get_ranged_number(FUZZ_VALUE_1_ID, 100 * MiB, 1 * MiB, local_mem_size / 2);
        shmem_fuzz_multi_task(
            [&](int rank_id, int n_ranks, uint64_t local_mem_size) {
                shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
                auto ptr0 = shmem_calloc(sizeof(uint64_t), alloc_size_0 / sizeof(uint64_t));
                auto ptr1 = shmem_calloc(sizeof(uint64_t), alloc_size_1 / sizeof(uint64_t));
                ASSERT_NE(ptr0, nullptr);
                ASSERT_NE(ptr1, nullptr);
                shmem_free(ptr0);
                shmem_free(ptr1);
            },
            local_mem_size, process_count);
    }
    DT_FUZZ_END()
}
