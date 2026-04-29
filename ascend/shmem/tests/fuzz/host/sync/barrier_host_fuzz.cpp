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
#include "host/shmem_host_team.h"
#include "init/shmemi_init.h"
#include "shmem_fuzz.h"
#include "utils/func_type.h"

extern void increase_do(void *stream, uint64_t config, uint8_t *addr, int32_t rankId, int32_t rankSize);
extern void increase_vec_do(void *stream, uint64_t config, uint8_t *addr, int32_t rankId, int32_t rankSize);
extern void increase_do_odd_team(void *stream, uint64_t config, uint8_t *addr, int32_t rankId, int32_t rankSize,
                                 shmem_team_t team_id);
extern void increase_vec_do_odd_team(void *stream, uint64_t config, uint8_t *addr, int32_t rankId, int32_t rankSize,
                                     shmem_team_t team_id);

constexpr int32_t SHMEM_BARRIER_TEST_NUM = 3;

class ShmemSyncBarrierFuzz : public testing::Test {
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

TEST_F(ShmemSyncBarrierFuzz, shmem_barrier_black_box_success)
{
    char fuzzName[] = "shmem_barrier_black_box_success";
    uint64_t seed = 0;

    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        shmem_fuzz_multi_task(
            [](int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size) {
                shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
                ASSERT_EQ(shmem_init_status(), SHMEM_STATUS_IS_INITIALIZED);

                size_t size = sizeof(uint64_t);
                uint64_t *addr_dev = (uint64_t *)shmem_malloc(size);
                ASSERT_NE(addr_dev, nullptr);
                ASSERT_EQ(aclrtMemset(addr_dev, size, 0, size), ACL_SUCCESS);
                uint64_t *addr_host;
                ASSERT_EQ(aclrtMallocHost((void **)&addr_host, size), ACL_SUCCESS);

                for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
                    std::cout << "[TEST] barriers test blackbox rank_id: " << rank_id << " time: " << i << std::endl;
                    increase_do(scope.stream, shmemx_get_ffts_config(), (uint8_t *)addr_dev, rank_id, n_ranks);
                    ASSERT_EQ(aclrtSynchronizeStream(scope.stream), ACL_SUCCESS);
                    ASSERT_EQ(aclrtMemcpy(addr_host, size, addr_dev, size, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
                    ASSERT_EQ((*addr_host), i);
                    shm::shmemi_control_barrier_all();
                }

                ASSERT_EQ(aclrtFreeHost(addr_host), ACL_SUCCESS);
                shmem_free(addr_dev);
            },
            1 * GiB, shmem_fuzz_gnpu_num());
    }
    DT_FUZZ_END()
}

TEST_F(ShmemSyncBarrierFuzz, shmem_vec_barrier_black_box_success)
{
    char fuzzName[] = "shmem_vec_barrier_black_box_success";
    uint64_t seed = 0;

    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        shmem_fuzz_multi_task(
            [](int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size) {
                shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
                ASSERT_EQ(shmem_init_status(), SHMEM_STATUS_IS_INITIALIZED);

                size_t size = sizeof(uint64_t);
                uint64_t *addr_dev_vec = (uint64_t *)shmem_malloc(size);
                ASSERT_NE(addr_dev_vec, nullptr);
                ASSERT_EQ(aclrtMemset(addr_dev_vec, size, 0, size), ACL_SUCCESS);
                uint64_t *addr_host_vec;
                ASSERT_EQ(aclrtMallocHost((void **)&addr_host_vec, size), ACL_SUCCESS);

                for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
                    std::cout << "[TEST] vec barriers test blackbox rank_id: " << rank_id << " time: " << i
                              << std::endl;
                    increase_vec_do(scope.stream, shmemx_get_ffts_config(), (uint8_t *)addr_dev_vec, rank_id, n_ranks);
                    ASSERT_EQ(aclrtSynchronizeStream(scope.stream), ACL_SUCCESS);
                    ASSERT_EQ(aclrtMemcpy(addr_host_vec, size, addr_dev_vec, size, ACL_MEMCPY_DEVICE_TO_HOST),
                              ACL_SUCCESS);
                    ASSERT_EQ((*addr_host_vec), i);
                    shm::shmemi_control_barrier_all();
                }

                ASSERT_EQ(aclrtFreeHost(addr_host_vec), ACL_SUCCESS);
                shmem_free(addr_dev_vec);
            },
            1 * GiB, shmem_fuzz_gnpu_num());
    }
    DT_FUZZ_END()
}

TEST_F(ShmemSyncBarrierFuzz, shmem_barrier_black_box_odd_team_success)
{
    char fuzzName[] = "shmem_barrier_black_box_odd_team_success";
    uint64_t seed = 0;

    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        shmem_fuzz_multi_task(
            [](int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size) {
                shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
                ASSERT_EQ(shmem_init_status(), SHMEM_STATUS_IS_INITIALIZED);

                shmem_team_t team_odd;
                int start = 1;
                int stride = 2;
                int team_size = n_ranks / 2;
                shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);
                size_t size = sizeof(uint64_t);

                uint64_t *addr_dev = (uint64_t *)shmem_malloc(size);
                ASSERT_NE(addr_dev, nullptr);
                ASSERT_EQ(aclrtMemset(addr_dev, size, 0, size), ACL_SUCCESS);
                uint64_t *addr_host;
                ASSERT_EQ(aclrtMallocHost((void **)&addr_host, size), ACL_SUCCESS);

                if (rank_id & 1) {
                    for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
                        std::cout << "[TEST] barriers test blackbox rank_id: " << rank_id << " time: " << i
                                  << std::endl;
                        increase_do_odd_team(scope.stream, shmemx_get_ffts_config(), (uint8_t *)addr_dev, rank_id,
                                             n_ranks, team_odd);
                        ASSERT_EQ(aclrtSynchronizeStream(scope.stream), ACL_SUCCESS);
                        ASSERT_EQ(aclrtMemcpy(addr_host, size, addr_dev, size, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
                        ASSERT_EQ((*addr_host), i);
                        shm::shmemi_control_barrier_all();
                    }
                }

                ASSERT_EQ(aclrtFreeHost(addr_host), ACL_SUCCESS);
                shmem_free(addr_dev);

                shmem_team_destroy(team_odd);
            },
            1 * GiB, shmem_fuzz_gnpu_num());
    }
    DT_FUZZ_END()
}

TEST_F(ShmemSyncBarrierFuzz, shmem_vec_barrier_black_box_odd_team_success)
{
    char fuzzName[] = "shmem_vec_barrier_black_box_odd_team_success";
    uint64_t seed = 0;

    DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)
    {
        shmem_fuzz_multi_task(
            [](int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size) {
                shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
                ASSERT_EQ(shmem_init_status(), SHMEM_STATUS_IS_INITIALIZED);

                shmem_team_t team_odd;
                int start = 1;
                int stride = 2;
                int team_size = n_ranks / 2;
                shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);
                size_t size = sizeof(uint64_t);

                uint64_t *addr_dev_vec = (uint64_t *)shmem_malloc(size);
                ASSERT_NE(addr_dev_vec, nullptr);
                ASSERT_EQ(aclrtMemset(addr_dev_vec, size, 0, size), ACL_SUCCESS);
                uint64_t *addr_host_vec;
                ASSERT_EQ(aclrtMallocHost((void **)&addr_host_vec, size), ACL_SUCCESS);

                if (rank_id & 1) {
                    for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
                        std::cout << "[TEST] vec barriers test blackbox rank_id: " << rank_id << " time: " << i
                                  << std::endl;
                        increase_vec_do_odd_team(scope.stream, shmemx_get_ffts_config(), (uint8_t *)addr_dev_vec,
                                                 rank_id, n_ranks, team_odd);
                        ASSERT_EQ(aclrtSynchronizeStream(scope.stream), ACL_SUCCESS);
                        ASSERT_EQ(aclrtMemcpy(addr_host_vec, size, addr_dev_vec, size, ACL_MEMCPY_DEVICE_TO_HOST),
                                  ACL_SUCCESS);
                        ASSERT_EQ((*addr_host_vec), i);
                        shm::shmemi_control_barrier_all();
                    }
                }

                ASSERT_EQ(aclrtFreeHost(addr_host_vec), ACL_SUCCESS);
                shmem_free(addr_dev_vec);

                shmem_team_destroy(team_odd);
            },
            1 * GiB, shmem_fuzz_gnpu_num());
    }
    DT_FUZZ_END()
}
