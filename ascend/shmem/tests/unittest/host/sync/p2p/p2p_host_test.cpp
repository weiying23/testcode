/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <string>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem_api.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "p2p_kernel.h"

static void test_p2p(int rank_id, int rank_size, uint64_t local_mem_size)
{
    aclrtStream stream;
    test_init(rank_id, rank_size, local_mem_size, &stream);

    int32_t *addr_dev = static_cast<int32_t *>(shmem_malloc(sizeof(int32_t)));
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(int32_t), 0, sizeof(int32_t)), 0);
    p2p_chain_do(stream, shmemx_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    shmem_free(addr_dev);

    int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;
    test_finalize(stream, dev_id);
    if (::testing::Test::HasFailure()) {
        exit(1);
    }
}

TEST(TEST_SYNC_API, test_p2p)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_p2p, local_mem_size, process_count);
}