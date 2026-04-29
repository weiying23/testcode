/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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

#include "host/shmem_host_def.h"
#include "host/shmem_host_init.h"
#include "shmem_fuzz.h"

class ShmemHostFuzz : public testing::Test {
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

TEST_F(ShmemHostFuzz, shmem_init_success)
{
    char fuzzName[] = "shmem_init_success";
    uint64_t seed = 0;
    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        const int process_count = shmem_fuzz_gnpu_num();
        uint64_t local_mem_size = fuzz_get_ranged_number(FUZZ_VALUE_0_ID, 1, 1, 512) * 2 * MiB;
        shmem_fuzz_multi_task(
            [](int rank_id, int n_ranks, uint64_t local_mem_size) {
                aclrtStream stream = nullptr;
                shmem_fuzz_test_init(rank_id, n_ranks, local_mem_size, &stream);
                ASSERT_EQ(shmem_init_status(), SHMEM_STATUS_IS_INITIALIZED);
                shmem_fuzz_test_deinit(stream, shmem_fuzz_device_id(rank_id));
            },
            local_mem_size, process_count);
    }
    DT_FUZZ_END()
}

TEST_F(ShmemHostFuzz, shmem_init_custom_attr_success)
{
    char fuzzName[] = "shmem_init_custom_attr_success";
    uint64_t seed = 0;
    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        const int process_count = shmem_fuzz_gnpu_num();
        uint64_t local_mem_size = fuzz_get_ranged_number(FUZZ_VALUE_0_ID, 1, 1, 512) * 2 * MiB;
        const int timeout = 50;
        shmem_fuzz_multi_task(
            [](int rank_id, int n_ranks, uint64_t local_mem_size) {
                shmem_init_attr_t *attributes;
                aclrtStream stream = nullptr;
                shmem_fuzz_test_set_attr(rank_id, n_ranks, local_mem_size, &attributes);
                shmem_set_data_op_engine_type(attributes, SHMEM_DATA_OP_MTE);
                shmem_set_timeout(attributes, timeout);
                shmem_fuzz_test_init_attr(attributes, &stream);
                ASSERT_EQ(shmem_init_status(), SHMEM_STATUS_IS_INITIALIZED);
                shmem_fuzz_test_deinit(stream, shmem_fuzz_device_id(rank_id));
            },
            local_mem_size, process_count);
    }
    DT_FUZZ_END()
}
