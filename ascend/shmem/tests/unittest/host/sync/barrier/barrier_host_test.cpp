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
#include "barrier_kernel.h"

constexpr int32_t SHMEM_BARRIER_TEST_NUM = 100;

static void test_barrier_black_box(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    uint64_t *addr_dev = static_cast<uint64_t *>(shmem_malloc(sizeof(uint64_t)));
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    uint64_t *addr_host;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&addr_host), sizeof(uint64_t)), 0);

    for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
        std::cout << "[TEST] barriers test blackbox rank_id: " << rank_id << " time: " << i << std::endl;
        increase_do(stream, shmemx_get_ffts_config(), (uint8_t *)addr_dev, rank_id, n_ranks);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        ASSERT_EQ(aclrtMemcpy(addr_host, sizeof(uint64_t), addr_dev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
        ASSERT_EQ((*addr_host), i);
        shm::shmemi_control_barrier_all();
    }

    uint64_t *addr_dev_vec = static_cast<uint64_t *>(shmem_malloc(sizeof(uint64_t)));
    ASSERT_EQ(aclrtMemset(addr_dev_vec, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    uint64_t *addr_host_vec;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&addr_host_vec), sizeof(uint64_t)), 0);

    for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
        std::cout << "[TEST] vec barriers test blackbox rank_id: " << rank_id << " time: " << i << std::endl;
        increase_vec_do(stream, shmemx_get_ffts_config(), (uint8_t *)addr_dev_vec, rank_id, n_ranks);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        ASSERT_EQ(
            aclrtMemcpy(addr_host_vec, sizeof(uint64_t), addr_dev_vec, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
        ASSERT_EQ((*addr_host_vec), i);
        shm::shmemi_control_barrier_all();
    }

    ASSERT_EQ(aclrtFreeHost(addr_host), 0);
    shmem_free(addr_dev);
    ASSERT_EQ(aclrtFreeHost(addr_host_vec), 0);
    shmem_free(addr_dev_vec);

    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()) {
        exit(1);
    }
}

static void test_barrier_black_box_odd_team(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    shmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = n_ranks / 2;
    shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

    uint64_t *addr_dev = static_cast<uint64_t *>(shmem_malloc(sizeof(uint64_t)));
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    uint64_t *addr_host;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&addr_host), sizeof(uint64_t)), 0);

    uint64_t *addr_dev_vec = static_cast<uint64_t *>(shmem_malloc(sizeof(uint64_t)));
    ASSERT_EQ(aclrtMemset(addr_dev_vec, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    uint64_t *addr_host_vec;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&addr_host_vec), sizeof(uint64_t)), 0);

    if (rank_id & 1) {
        for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
            std::cout << "[TEST] barriers test blackbox rank_id: " << rank_id << " time: " << i << std::endl;
            increase_do_odd_team(stream, shmemx_get_ffts_config(), (uint8_t *)addr_dev, rank_id, n_ranks, team_odd);
            ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
            ASSERT_EQ(aclrtMemcpy(addr_host, sizeof(uint64_t), addr_dev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST),
                      0);
            ASSERT_EQ((*addr_host), i);
            shm::shmemi_control_barrier_all();
        }

        for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
            std::cout << "[TEST] vec barriers test blackbox rank_id: " << rank_id << " time: " << i << std::endl;
            increase_vec_do_odd_team(stream, shmemx_get_ffts_config(), (uint8_t *)addr_dev_vec, rank_id, n_ranks,
                                     team_odd);
            ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
            ASSERT_EQ(
                aclrtMemcpy(addr_host_vec, sizeof(uint64_t), addr_dev_vec, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST),
                0);
            ASSERT_EQ((*addr_host_vec), i);
            shm::shmemi_control_barrier_all();
        }
    }

    ASSERT_EQ(aclrtFreeHost(addr_host), 0);
    shmem_free(addr_dev);
    ASSERT_EQ(aclrtFreeHost(addr_host_vec), 0);
    shmem_free(addr_dev_vec);

    shmem_team_destroy(team_odd);

    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()) {
        exit(1);
    }
}

void test_partial_barrier_black_box(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    shmem_team_t team;
    int start = 1;
    int stride = 1;
    int team_size = n_ranks - 1;
    shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, team_size, &team);

    int pes_size = team_size - 1 >= 1 ? team_size - 1 : 1;
    std::vector<uint32_t> pes_host;
    for (int i = 0; i < pes_size; ++i) {
        pes_host.push_back(static_cast<uint32_t>(i));
    }
    uint32_t count = static_cast<uint32_t>(pes_host.size());

    uint32_t *pes_dev = static_cast<uint32_t *>(shmem_malloc(sizeof(uint32_t) * count));
    ASSERT_NE(pes_dev, nullptr);
    ASSERT_EQ(aclrtMemcpy(pes_dev, sizeof(uint32_t) * count,
                          pes_host.data(), sizeof(uint32_t) * count,
                          ACL_MEMCPY_HOST_TO_DEVICE),
              0);

    uint64_t *addr_dev = static_cast<uint64_t *>(shmem_malloc(sizeof(uint64_t)));
    ASSERT_NE(addr_dev, nullptr);
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);

    uint64_t *addr_host = nullptr;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&addr_host), sizeof(uint64_t)), 0);

    for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
        if (rank_id > 0) {
            std::cout << "[TEST] partial barriers (cube) blackbox rank_id: "
                    << rank_id << " time: " << i << std::endl;
            std::cout << "count " << count << std::endl;
            partial_increase_do(stream, shmemx_get_ffts_config(),
                                reinterpret_cast<uint8_t *>(addr_dev),
                                reinterpret_cast<uint8_t *>(pes_dev),
                                count, rank_id, n_ranks, team);

            ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
            ASSERT_EQ(aclrtMemcpy(addr_host, sizeof(uint64_t),
                                  addr_dev, sizeof(uint64_t),
                                  ACL_MEMCPY_DEVICE_TO_HOST),
                      0);

            if ((rank_id - start) / stride < pes_size) {
                std::cout << "compare rank" << rank_id << std::endl;
                ASSERT_EQ(*addr_host, static_cast<uint64_t>(i));
            }
        }

        shm::shmemi_control_barrier_all();
    }

    uint64_t *addr_dev_vec = static_cast<uint64_t *>(shmem_malloc(sizeof(uint64_t)));
    ASSERT_NE(addr_dev_vec, nullptr);
    ASSERT_EQ(aclrtMemset(addr_dev_vec, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);

    uint64_t *addr_host_vec = nullptr;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&addr_host_vec), sizeof(uint64_t)), 0);

    for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
        if (rank_id > 0) {
            std::cout << "[TEST] partial barriers (vec) blackbox rank_id: "
                    << rank_id << " time: " << i << std::endl;

            partial_increase_vec_do(stream, shmemx_get_ffts_config(),
                                    reinterpret_cast<uint8_t *>(addr_dev_vec),
                                    reinterpret_cast<uint8_t *>(pes_dev),
                                    count, rank_id, n_ranks, team);

            ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
            ASSERT_EQ(aclrtMemcpy(addr_host_vec, sizeof(uint64_t),
                                  addr_dev_vec, sizeof(uint64_t),
                                  ACL_MEMCPY_DEVICE_TO_HOST),
                      0);
        
            if ((rank_id - start) / stride < pes_size) {
                ASSERT_EQ(*addr_host_vec, static_cast<uint64_t>(i));
            }
        }

        shm::shmemi_control_barrier_all();
    }

    ASSERT_EQ(aclrtFreeHost(addr_host), 0);
    ASSERT_EQ(aclrtFreeHost(addr_host_vec), 0);
    shmem_free(addr_dev);
    shmem_free(addr_dev_vec);
    shmem_free(pes_dev);

    test_finalize(stream, device_id);
}

TEST(TEST_SYNC_API, test_barrier_black_box)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_barrier_black_box, local_mem_size, process_count);
}

TEST(TEST_SYNC_API, test_barrier_black_box_odd_team)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_barrier_black_box_odd_team, local_mem_size, process_count);
}

TEST(TEST_SYNC_API, test_partial_barrier_black_box)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_partial_barrier_black_box, local_mem_size, process_count);
}