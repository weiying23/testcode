/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMX_DEVICE_LOW_LEVEL_RMA_H
#define SHMEMX_DEVICE_LOW_LEVEL_RMA_H

#include "kernel_operator.h"
#include "internal/device/shmemi_device_common.h"


/**
 * @brief Async interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 * @param enable_L2         [in] whether to enable L2 cache.
 */
SHMEM_DEVICE void shmemx_mte_get_mem_nbi_low_level(__gm__ int8_t* dst, __gm__ int8_t* src, __ubuf__ int8_t* buf,
    uint32_t ub_size, uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID, bool enable_L2)
{
    auto ptr = shmem_ptr(src, pe);
    __gm__ int8_t* remote_ptr = reinterpret_cast<__gm__ int8_t*>(ptr);

    // block_size: dataMove Unit
    uint64_t block_size = ub_size;
    uint64_t remain = (elem_size) % block_size;

    uint64_t repeat_times = (elem_size) / block_size;
    uint64_t repeat_elem = block_size;
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, remote_ptr + i * repeat_elem, block_size, enable_L2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst + i * repeat_elem, buf, block_size, enable_L2);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, remote_ptr + repeat_times * repeat_elem, remain, enable_L2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst + repeat_times * repeat_elem, buf, remain, enable_L2);
    }
}

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 * @param enable_L2         [in] whether to enable L2 cache.
 */
SHMEM_DEVICE void shmemx_mte_put_mem_nbi_low_level(__gm__ int8_t* dst, __gm__ int8_t* src, __ubuf__ int8_t* buf,
    uint32_t ub_size, uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID, bool enable_L2)
{
    auto ptr = shmem_ptr(dst, pe);
    __gm__ int8_t* remote_ptr = reinterpret_cast<__gm__ int8_t*>(ptr);

    // block_size: dataMove Unit
    uint64_t block_size = ub_size;
    uint64_t remain = (elem_size) % block_size;

    uint64_t repeat_times = (elem_size) / block_size;
    uint64_t repeat_elem = block_size;
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, src + i * repeat_elem, block_size, enable_L2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remote_ptr + i * repeat_elem, buf, block_size, enable_L2);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, src + repeat_times * repeat_elem, remain, enable_L2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remote_ptr + repeat_times * repeat_elem, buf, remain, enable_L2);
    }
}

#endif