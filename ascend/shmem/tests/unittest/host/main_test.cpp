/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include <algorithm>
#include "acl/acl.h"
#include "shmem_api.h"
#include "unittest_main_test.h"

int test_global_ranks;
int test_gnpu_num;
char test_global_ipport[SHMEM_MAX_IP_PORT_LEN];
int test_first_rank;
int test_first_npu;

void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st)
{
    *st = nullptr;
    int status = 0;
    if (n_ranks != (n_ranks & (~(n_ranks - 1)))) {
        std::cout << "[TEST] input rank_size: "<< n_ranks << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);

    EXPECT_EQ(status = shmem_set_conf_store_tls(false, nullptr, 0), 0);

    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, 0);
    *st = stream;
}

int32_t test_rdma_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st)
{
    *st = nullptr;
    int status = 0;
    if (n_ranks != (n_ranks & (~(n_ranks - 1)))) {
        std::cout << "[TEST] input rank_size: "<< n_ranks << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);

    EXPECT_EQ(status = shmem_set_conf_store_tls(false, nullptr, 0), 0);

    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
    status = shmem_init_attr(attributes);
    *st = stream;
    return status;
}

void test_finalize(aclrtStream stream, int device_id)
{
    int status = shmem_finalize();
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclrtDestroyStream(stream), 0);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count)
{
    pid_t pids[process_count];
    int status[process_count];
    for (int i = 0; i < process_count; ++i) {
        pids[i] = fork();
        if (pids[i] < 0) {
            std::cout << "fork failed ! " << pids[i] << std::endl;
        } else if (pids[i] == 0) {
            func(i + test_first_rank, test_global_ranks, local_mem_size);
            exit(0);
        }
    }
    for (int i = 0; i < process_count; ++i) {
        waitpid(pids[i], &status[i], 0);
        if (WIFEXITED(status[i]) && WEXITSTATUS(status[i]) != 0) {
            FAIL();
        }
    }
}

int main(int argc, char** argv)
{
    int idx = 1;
    test_global_ranks = std::atoi(argv[idx++]);
    std::copy_n(argv[idx++], SHMEM_MAX_IP_PORT_LEN - 1, test_global_ipport);
    test_global_ipport[SHMEM_MAX_IP_PORT_LEN - 1] = '\0';
    test_gnpu_num = std::atoi(argv[idx++]);
    test_first_rank = std::atoi(argv[idx++]);
    test_first_npu = std::atoi(argv[idx++]);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}