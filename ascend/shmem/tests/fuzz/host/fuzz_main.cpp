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
#include <functional>
#include <gtest/gtest.h>
#include <secodefuzz/secodeFuzz.h>

#include "acl/acl.h"
#include "host/shmem_host_def.h"
#include "host/shmem_host_init.h"
#include "shmem_fuzz.h"

namespace {
int fuzz_global_ranks;
int fuzz_gnpu_num;
const char *fuzz_global_ip_port;
int fuzz_first_rank;
int fuzz_first_npu;
}  // namespace

int shmem_fuzz_gnpu_num(void)
{
    return fuzz_gnpu_num;
}

int32_t shmem_fuzz_device_id(int rank_id)
{
    return rank_id % fuzz_gnpu_num + fuzz_first_npu;
}

void shmem_fuzz_test_set_attr(int rank_id, int n_ranks, uint64_t local_mem_size, shmem_init_attr_t **attributes)
{
    *attributes = nullptr;

    ASSERT_EQ(n_ranks, (n_ranks & (~(n_ranks - 1)))) << "n_ranks is not the power of 2";
    shmem_init_attr_t *tmp_attributes;
    ASSERT_EQ(shmem_set_attr(rank_id, n_ranks, local_mem_size, fuzz_global_ip_port, &tmp_attributes), SHMEM_SUCCESS);

    *attributes = tmp_attributes;
}

void shmem_fuzz_test_init_attr(shmem_init_attr_t *attributes, aclrtStream *stream)
{
    *stream = nullptr;

    aclrtStream tmp_stream = nullptr;
    ASSERT_NE(attributes, nullptr);
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    ASSERT_EQ(aclrtSetDevice(shmem_fuzz_device_id(attributes->my_rank)), ACL_SUCCESS);
    ASSERT_EQ(aclrtCreateStream(&tmp_stream), ACL_SUCCESS);
    ASSERT_EQ(shmem_set_conf_store_tls(false, nullptr, 0), 0);
    ASSERT_EQ(shmem_init_attr(attributes), SHMEM_SUCCESS);

    *stream = tmp_stream;
}

void shmem_fuzz_test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *stream)
{
    *stream = nullptr;

    aclrtStream tmp_stream = nullptr;
    shmem_init_attr_t *attributes;
    shmem_fuzz_test_set_attr(rank_id, n_ranks, local_mem_size, &attributes);
    shmem_fuzz_test_init_attr(attributes, &tmp_stream);

    *stream = tmp_stream;
}

void shmem_fuzz_test_deinit(aclrtStream stream, int device_id)
{
    ASSERT_EQ(shmem_finalize(), SHMEM_SUCCESS);
    ASSERT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtResetDevice(device_id), ACL_SUCCESS);
    ASSERT_EQ(aclFinalize(), ACL_SUCCESS);
}

void shmem_fuzz_multi_task(std::function<void(int, int, uint64_t)> task, uint64_t local_mem_size, int process_count)
{
    pid_t pids[process_count];
    int status[process_count];
    for (int i = 0; i < process_count; ++i) {
        pids[i] = fork();
        if (pids[i] < 0) {
            std::cout << "fork failed: " << pids[i] << std::endl;
        } else if (pids[i] == 0) {
            task(fuzz_first_rank + i, fuzz_global_ranks, local_mem_size);
            if (testing::Test::HasFailure()) {
                std::cout << "[shmem_fuzz_multi_task] fork: task " << i << " failed" << std::endl;
                exit(1);
            } else {
                std::cout << "[shmem_fuzz_multi_task] fork: task " << i << " succeeded" << std::endl;
                exit(0);
            }
        }
    }
    for (int i = 0; i < process_count; ++i) {
        waitpid(pids[i], &status[i], 0);
    }
    int success_count = 0;
    for (int i = 0; i < process_count; ++i) {
        if (WIFEXITED(status[i]) && (WEXITSTATUS(status[i]) == 0)) {
            std::cout << "[shmem_fuzz_multi_task] summary: task " << i << " succeeded" << std::endl;
            success_count++;
        } else {
            std::cout << "[shmem_fuzz_multi_task] summary: task " << i << " failed" << std::endl;
        }
    }
    ASSERT_EQ(success_count, process_count);
}

int main(int argc, char *argv[])
{
    int idx = 1;
    fuzz_global_ranks = std::atoi(argv[idx++]);
    fuzz_global_ip_port = argv[idx++];
    fuzz_gnpu_num = std::atoi(argv[idx++]);
    fuzz_first_rank = std::atoi(argv[idx++]);
    fuzz_first_npu = std::atoi(argv[idx++]);

    testing::InitGoogleTest(&argc, argv);

    char fuzzReportPath[] = "./shmem_fuzz";
    DT_Set_Report_Path(fuzzReportPath);

    int ret = RUN_ALL_TESTS();
    return ret;
}
