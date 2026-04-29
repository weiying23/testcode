/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include "bfloat16.h"
#include "fp16_t.h"
#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_sync.h"
#include "shmem_fuzz.h"
#include "utils/func_type.h"

static constexpr size_t input_repeat = 32;
static constexpr size_t input_length = 16;

class ShmemGmNonContiguousFuzz : public testing::Test {
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

#define TEST_FUNC(NAME, TYPE)                                                                                     \
    extern void test_##NAME##_non_contiguous_put(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, \
                                                 uint8_t *dev_ptr, int repeat, int length);                       \
    extern void test_##NAME##_non_contiguous_get(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, \
                                                 uint8_t *dev_ptr, int repeat, int length)

SHMEM_FUNC_TYPE_HOST(TEST_FUNC);

using device_func = std::function<void(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva,
                                       uint8_t *dev_ptr, int repeat, int length)>;

template <class Tp>
static void test_shmem_gm_non_contiguous(int rank_id, int n_ranks, uint64_t local_mem_size, device_func put_func,
                                         device_func get_func)
{
    shmem_init_scope scope(rank_id, n_ranks, local_mem_size);
    ASSERT_EQ(shmem_init_status(), SHMEM_STATUS_IS_INITIALIZED);

    int rank_flag = rank_id * 10;
    size_t total_size = input_repeat * input_length;
    size_t input_size = total_size * sizeof(Tp);

    std::vector<Tp> input(total_size, 0);
    for (size_t i = 0; i < input_repeat; i++) {
        for (size_t j = 0; j < input_length; j++) {
            input[i * input_length + j] = static_cast<Tp>(rank_flag) + static_cast<Tp>(i);
        }
    }

    void *dev_ptr;
    ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    void *ptr = shmem_malloc(input_size);
    ASSERT_NE(ptr, nullptr);

    uint32_t block_dim = 1;
    put_func(block_dim, scope.stream, shmemx_get_ffts_config(), (uint8_t *)ptr, (uint8_t *)dev_ptr, input_repeat,
             input_length);
    ASSERT_EQ(aclrtSynchronizeStream(scope.stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(input.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);

    get_func(block_dim, scope.stream, shmemx_get_ffts_config(), (uint8_t *)ptr, (uint8_t *)dev_ptr, input_repeat / 2U,
             input_length);
    ASSERT_EQ(aclrtSynchronizeStream(scope.stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(input.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);

    for (size_t i = 0; i < input_repeat / 4U; i++) {
        for (size_t j = 0; j < input_length; j++) {
            input[i * input_length + j] = static_cast<Tp>(rank_flag) + static_cast<Tp>(i * 4U);
        }
    }

    shmem_free(ptr);
    ASSERT_EQ(aclrtFree(dev_ptr), ACL_SUCCESS);
}

#define TEST_CASE(NAME, TYPE)                                                                                 \
    TEST_F(ShmemGmNonContiguousFuzz, shmem_gm_non_contiguous_##NAME##_success)                                \
    {                                                                                                         \
        char fuzzName[] = "shmem_gm_non_contiguous_##NAME##_success";                                         \
        uint64_t seed = 0;                                                                                    \
        DT_FUZZ_START(seed, SHMEM_FUZZ_COUNT, fuzzName, 0)                                                    \
        {                                                                                                     \
            shmem_fuzz_multi_task(                                                                            \
                [](int rank_id, int n_ranks, uint64_t local_mem_size) {                                       \
                    device_func put_func = test_##NAME##_non_contiguous_put;                                  \
                    device_func get_func = test_##NAME##_non_contiguous_get;                                  \
                    test_shmem_gm_non_contiguous<TYPE>(rank_id, n_ranks, local_mem_size, put_func, get_func); \
                },                                                                                            \
                1 * GiB, shmem_fuzz_gnpu_num());                                                              \
        }                                                                                                     \
        DT_FUZZ_END()                                                                                         \
    }

SHMEM_FUNC_TYPE_HOST(TEST_CASE);
