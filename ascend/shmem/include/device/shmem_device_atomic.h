/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_ATOMIC_H
#define SHMEM_DEVICE_ATOMIC_H

#include "kernel_operator.h"
#include "internal/device/shmemi_device_common.h"
#include "low_level/shmem_device_low_level_rma.h"
#include "shmem_device_team.h"

/**
 * @brief Standard Atomic Add Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |half       | half      |
 * |float      | float     |
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 */
#define SHMEM_TYPE_FUNC_ATOMIC_INT(FUNC) \
    FUNC(int8, int8_t, ATOMIC_S8);       \
    FUNC(int16, int16_t, ATOMIC_S16);    \
    FUNC(int32, int32_t, ATOMIC_S32)

#define SHMEM_TYPE_FUNC_ATOMIC_FLOAT(FUNC) \
    FUNC(half, half, ATOMIC_F16);          \
    FUNC(float, float, ATOMIC_F32)

#define SHMEM_ATOMIC_ADD_TYPENAME(NAME, TYPE, ATOMIC_TYPE)                                                         \
    /**                                                                                                            \
     * @brief Asynchronous interface. Perform contiguous data atomic add opeartion                                 \
              on symmetric memory from the specified PE to address on the local PE.                                \
              We use scalar instructions to implement single element atomic add. Therefore, both                   \
     *        vector and cube cores can execute atomic add operation.                                              \
     *                                                                                                             \
     * @param dst               [in] Pointer on local device of the destination data.                              \
     * @param value             [in] Value atomic add to destination.                                              \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    SHMEM_DEVICE void shmem_##NAME##_atomic_add(__gm__ TYPE *dst, TYPE value, int32_t pe)                          \
    {                                                                                                              \
        /* ROCE */                                                                                                 \
        /* RDMA */                                                                                                 \
        /* MTE  */                                                                                                 \
        dcci_atomic();                                                                                             \
        dsb_all();                                                                                                 \
        set_st_atomic_cfg(ATOMIC_TYPE, ATOMIC_SUM);                                                                \
        st_atomic<TYPE>(value, (__gm__ TYPE *)shmemi_ptr(dst, pe));                                                \
        dcci_atomic();                                                                                             \
    }

SHMEM_TYPE_FUNC_ATOMIC_INT(SHMEM_ATOMIC_ADD_TYPENAME);

#define SHMEM_ATOMIC_ADD_TYPENAME_FLOAT(NAME, TYPE, ATOMIC_TYPE)                                                   \
    /**                                                                                                            \
     * @brief Asynchronous interface. Perform contiguous data atomic add opeartion on                              \
              symmetric memory from the specified PE to address on the local PE.                                   \
     *                                                                                                             \
     * @param dst               [in] Pointer on local device of the destination data.                              \
     * @param value             [in] Value atomic add to destination.                                              \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    SHMEM_DEVICE void shmem_##NAME##_atomic_add(__gm__ TYPE *dst, TYPE value, int32_t pe)                          \
    {                                                                                                              \
        /* ROCE */                                                                                                 \
        /* RDMA */                                                                                                 \
        /* MTE  */                                                                                                 \
        dcci_atomic();                                                                                             \
        dsb_all();                                                                                                 \
        set_st_atomic_cfg(ATOMIC_TYPE, ATOMIC_SUM);                                                                \
        st_atomic(value, (__gm__ TYPE *)shmemi_ptr(dst, pe));                                                      \
        dcci_atomic();                                                                                             \
    }

SHMEM_TYPE_FUNC_ATOMIC_FLOAT(SHMEM_ATOMIC_ADD_TYPENAME_FLOAT);
#endif
