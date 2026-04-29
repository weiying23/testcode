/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"

#include "shmem_api.h"
#include "unittest/utils/func_type.h"
constexpr uint64_t MESSAGE_SIZE = 64;

#define ATOMIC_ADD_TEST_KERNEL(NAME, TYPE)                                                              \
    extern "C" __global__ __aicore__ void test_atomic_add_##NAME##_kernel(GM_ADDR gva, uint64_t config) \
    {                                                                                                   \
        shmemx_set_ffts_config(config);                                                                 \
        int64_t rank = smem_shm_get_global_rank();                                                      \
        int64_t rank_size = smem_shm_get_global_rank_size();                                            \
        GM_ADDR dst_addr;                                                                               \
                                                                                                        \
        for (int64_t peer = 0; peer < rank_size; peer++) {                                              \
            if (peer == rank) {                                                                         \
                continue;                                                                               \
            }                                                                                           \
            dst_addr = gva + rank * MESSAGE_SIZE;                                                       \
            if (AscendC::GetSubBlockIdx() == 0) {                                                       \
                shmem_##NAME##_atomic_add((__gm__ TYPE *)dst_addr, rank + 1, peer);                     \
            }                                                                                           \
        }                                                                                               \
        shmem_barrier_all();                                                                            \
    }
SHMEM_ATOMIC_ADD_FUNC_TYPE_KERNEL(ATOMIC_ADD_TEST_KERNEL);

#define ATOMIC_ADD_TEST(NAME, TYPE)                                                                   \
    void test_atomic_add_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config) \
    {                                                                                                 \
        test_atomic_add_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                 \
    }
SHMEM_ATOMIC_ADD_FUNC_TYPE_KERNEL(ATOMIC_ADD_TEST);