/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_COMMON_H
#define SHMEMI_DEVICE_COMMON_H

#include "shmemi_device_arch.h"
#include "shmemi_device_def.h"

#include "smem_shm_aicore_base_api.h"

constexpr int ub_limit = 192 * 1024;

SHMEM_DEVICE __gm__ shmemi_device_host_state_t *shmemi_get_state()
{
    return reinterpret_cast<__gm__ shmemi_device_host_state_t *>(smem_shm_get_extra_context_addr());
}

SHMEM_DEVICE int shmemi_get_my_pe()
{
    return shmemi_get_state()->mype;
}

SHMEM_DEVICE int shmemi_get_total_pe()
{
    return shmemi_get_state()->npes;
}

SHMEM_DEVICE uint64_t shmemi_get_heap_size()
{
    return shmemi_get_state()->heap_size;
}

template<typename T>
SHMEM_DEVICE void shmemi_store(__gm__ T *addr, T val)
{
    *((__gm__ T *)addr) = val;
}

template<typename T>
SHMEM_DEVICE T shmemi_load(__gm__ T *cache)
{
    return *((__gm__ T *)cache);
}

template<typename T>
SHMEM_DEVICE __gm__ T *shmemi_ptr(__gm__ T *local, int pe)
{
    uint64_t shm_size = shmemi_get_heap_size();
    int my_pe = shmemi_get_my_pe();

    uint64_t remote = reinterpret_cast<uint64_t>(local) + shm_size * (pe - my_pe);
    return reinterpret_cast<__gm__ T*>(remote);
}
#endif
