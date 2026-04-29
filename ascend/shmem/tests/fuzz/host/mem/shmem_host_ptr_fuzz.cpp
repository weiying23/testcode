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

#include "acl/acl_rt.h"
#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_rma.h"
#include "shmem_fuzz.h"

extern void get_device_ptr(uint32_t block_dim, void *stream, uint8_t *gva);

class ShmemHostPtrFuzz : public testing::Test {
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

static int32_t test_get_device_ptr(aclrtStream stream, uint8_t *ptr, size_t byte_size)
{
    int *y_host;
    EXPECT_EQ(aclrtMallocHost((void **)(&y_host), byte_size), ACL_SUCCESS);

    uint32_t block_dim = 1;
    int32_t device_id;
    EXPECT_EQ(aclrtGetDevice(&device_id), ACL_SUCCESS);

    get_device_ptr(block_dim, stream, ptr);
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    sleep(1);

    EXPECT_EQ(aclrtMemcpy(y_host, byte_size, ptr, byte_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    EXPECT_EQ(y_host[0], 1);  // @see get_device_ptr
    EXPECT_EQ(y_host[1], 1);  // @see get_device_ptr

    EXPECT_EQ(aclrtFreeHost(y_host), ACL_SUCCESS);

    if (testing::Test::HasFailure()) {
        return 1;
    }
    return 0;
}

TEST_F(ShmemHostPtrFuzz, shmem_get_ptr_success)
{
    char fuzzName[] = "shmem_get_ptr_success";
    uint64_t seed = 0;
    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        const int process_count = shmem_fuzz_gnpu_num();
        uint64_t local_mem_size = 1 * GiB;
        uint64_t int_count = fuzz_get_ranged_number(FUZZ_VALUE_0_ID, 2, 2, 1024);
        shmem_fuzz_multi_task(
            [&](int rank_id, int n_ranks, uint64_t local_mem_size) {
                shmem_init_scope scope(rank_id, n_ranks, local_mem_size);

                int *ptr = (int *)shmem_malloc(int_count * sizeof(int));
                ASSERT_NE(ptr, nullptr);

                int *host_self = (int *)shmem_ptr(ptr, rank_id);
                ASSERT_NE(host_self, nullptr);
                ASSERT_EQ(host_self, ptr);

                int peer = (rank_id + 1) % n_ranks;
                int *host_remote = (int *)shmem_ptr(ptr, peer);
                int *next_remote = (int *)shmem_ptr(ptr + 1, peer);
                ASSERT_NE(host_remote, nullptr);
                ASSERT_EQ(next_remote - host_remote, 1);

                auto status = test_get_device_ptr(scope.stream, (uint8_t *)ptr, int_count * sizeof(int));
                ASSERT_EQ(status, 0);

                shmem_free(ptr);
            },
            local_mem_size, process_count);
    }
    DT_FUZZ_END()
}

TEST_F(ShmemHostPtrFuzz, shmem_mte_set_ub_params_success)
{
    char fuzzName[] = "shmem_mte_set_ub_params_success";
    uint64_t seed = 0;
    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        const int process_count = shmem_fuzz_gnpu_num();
        uint64_t local_mem_size = 1 * GiB;
        uint64_t offset = 1;
        uint32_t ub_size = fuzz_get_ranged_number(FUZZ_VALUE_0_ID, 256, 256, 512);
        uint32_t event_id = 0;
        shmem_fuzz_multi_task(
            [&](int rank_id, int n_ranks, uint64_t local_mem_size) {
                shmem_init_scope scope(rank_id, n_ranks, local_mem_size);

                ASSERT_EQ(shmem_mte_set_ub_params(offset, ub_size, event_id), SHMEM_SUCCESS);
            },
            local_mem_size, process_count);
    }
    DT_FUZZ_END()
}
