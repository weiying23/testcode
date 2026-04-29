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

extern void p2p_chain_do(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size);

class ShmemSyncP2pFuzz : public testing::Test {
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

TEST_F(ShmemSyncP2pFuzz, shmem_sync_p2p_success)
{
    char fuzzName[] = "shmem_sync_p2p_success";
    uint64_t seed = 0;

    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        shmem_fuzz_multi_task(
            [](int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size) {
                shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
                ASSERT_EQ(shmem_init_status(), SHMEM_STATUS_IS_INITIALIZED);

                size_t size = sizeof(int32_t);
                int32_t *addr_dev = (int32_t *)shmem_malloc(size);
                ASSERT_NE(addr_dev, nullptr);
                ASSERT_EQ(aclrtMemset(addr_dev, size, 0, size), ACL_SUCCESS);
                p2p_chain_do(scope.stream, shmemx_get_ffts_config(), (uint8_t *)addr_dev, rank_id, n_ranks);
                ASSERT_EQ(aclrtSynchronizeStream(scope.stream), ACL_SUCCESS);
                shmem_free(addr_dev);
            },
            1 * GiB, shmem_fuzz_gnpu_num());
    }
    DT_FUZZ_END()
}
