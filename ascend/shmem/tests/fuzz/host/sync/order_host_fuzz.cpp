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
#include <functional>
#include <gtest/gtest.h>
#include <secodefuzz/secodeFuzz.h>
#include <sys/types.h>

#include "acl/acl_rt.h"
#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_sync.h"
#include "shmem_fuzz.h"
#include "utils/func_type.h"

extern void quiet_order_do(void *stream, uint64_t config, uint8_t *addr, int32_t rank_id, int32_t n_ranks);
extern void fence_order_do(void *stream, uint64_t config, uint8_t *addr, int32_t rank_id, int32_t n_ranks);

class ShmemSyncOrderFuzz : public testing::Test {
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

TEST_F(ShmemSyncOrderFuzz, shmem_quiet_order_success)
{
    char fuzzName[] = "shmem_quiet_order_success";
    uint64_t seed = 0;

    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        shmem_fuzz_multi_task(
            [](int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size) {
                shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
                ASSERT_EQ(shmem_init_status(), SHMEM_STATUS_IS_INITIALIZED);

                size_t length = 64;
                size_t byte_size = length * sizeof(uint64_t);
                uint64_t *dev_ptr = (uint64_t *)shmem_malloc(byte_size);
                ASSERT_NE(dev_ptr, nullptr);
                ASSERT_EQ(aclrtMemset(dev_ptr, byte_size, 0, byte_size), ACL_SUCCESS);

                std::vector<uint64_t> host_buf(length, 0);

                std::cout << "[TEST] quiet order test rank " << rank_id << std::endl;
                quiet_order_do(scope.stream, shmemx_get_ffts_config(), (uint8_t *)dev_ptr, rank_id, n_ranks);

                ASSERT_EQ(aclrtSynchronizeStream(scope.stream), ACL_SUCCESS);
                ASSERT_EQ(aclrtMemcpy(host_buf.data(), byte_size, dev_ptr, byte_size, ACL_MEMCPY_DEVICE_TO_HOST),
                          ACL_SUCCESS);

                if (rank_id == 1) {
                    ASSERT_EQ(host_buf[33U], 0xBBu);
                    ASSERT_EQ(host_buf[34U], 0xAAu);
                }

                shmem_free(dev_ptr);
            },
            1 * GiB, shmem_fuzz_gnpu_num());
    }
    DT_FUZZ_END()
}

TEST_F(ShmemSyncOrderFuzz, shmem_fence_order_success)
{
    char fuzzName[] = "shmem_fence_order_success";
    uint64_t seed = 0;

    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        shmem_fuzz_multi_task(
            [](int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size) {
                shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
                ASSERT_EQ(shmem_init_status(), SHMEM_STATUS_IS_INITIALIZED);

                size_t length = 64;
                size_t byte_size = length * sizeof(uint64_t);
                uint64_t *addr_dev = (uint64_t *)shmem_malloc(byte_size);
                ASSERT_NE(addr_dev, nullptr);
                ASSERT_EQ(aclrtMemset(addr_dev, byte_size, 0, byte_size), ACL_SUCCESS);

                std::vector<uint64_t> addr_host(length, 0);

                std::cout << "[TEST] fence order test rank " << rank_id << std::endl;
                fence_order_do(scope.stream, shmemx_get_ffts_config(), (uint8_t *)addr_dev, rank_id, n_ranks);

                ASSERT_EQ(aclrtSynchronizeStream(scope.stream), ACL_SUCCESS);
                ASSERT_EQ(aclrtMemcpy(addr_host.data(), byte_size, addr_dev, byte_size, ACL_MEMCPY_DEVICE_TO_HOST),
                          ACL_SUCCESS);

                if (rank_id == 1) {
                    ASSERT_EQ(addr_host[17U], 84u);
                    ASSERT_EQ(addr_host[18U], 42u);
                }
                shmem_free(addr_dev);
            },
            1 * GiB, shmem_fuzz_gnpu_num());
    }
    DT_FUZZ_END()
}
