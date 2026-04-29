/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMX_DEVICE_RMA_H
#define SHMEMX_DEVICE_RMA_H

#include "kernel_operator.h"
#include "internal/device/shmemi_device_common.h"
#include "low_level/shmemx_device_low_level_rma.h"

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address
 *        on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param enable_L2         [in] whether to enable L2 cache
 */
SHMEM_DEVICE void shmemx_mte_get_mem_nbi(__gm__ int8_t* dst, __gm__ int8_t* src, uint32_t elem_size,
    int32_t pe, bool enable_L2)
{
    /* Global State Get */
    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.shmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
    shmemx_mte_get_mem_nbi_low_level(dst, src, reinterpret_cast<__ubuf__ int8_t*>(copy_ub), copy_ub_size,
        elem_size, pe, copy_event_id, enable_L2);
}

/**
 * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local UB of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param enable_L2         [in] whether to enable L2 cache
 */
SHMEM_DEVICE void shmemx_mte_put_mem_nbi(__gm__ int8_t* dst, __gm__ int8_t* src, uint32_t elem_size, int32_t pe,
    bool enable_L2)
{
        /* Global State Get */
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
        /* CopyUB Config Set */
        uint64_t copy_ub = device_state->mte_config.shmem_ub;
        uint32_t copy_ub_size = device_state->mte_config.ub_size;
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
        shmemx_mte_put_mem_nbi_low_level(dst, src, reinterpret_cast<__ubuf__ int8_t*>(copy_ub), copy_ub_size,
            elem_size, pe, copy_event_id, enable_L2);
}
#endif
