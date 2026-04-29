/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_ARCH_H
#define SHMEMI_DEVICE_ARCH_H

#include "device/shmem_device_def.h"
constexpr uint64_t SHMEM_DATA_CACHE_LINE_SIZE = 64;

SHMEM_DEVICE void dcci_cacheline(__gm__ uint8_t * addr)
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(addr);

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

SHMEM_DEVICE void dcci_cachelines(__gm__ uint8_t* addr, uint64_t length)
{
    __gm__ uint8_t* start = (__gm__ uint8_t*)((uint64_t)addr / SHMEM_DATA_CACHE_LINE_SIZE
        * SHMEM_DATA_CACHE_LINE_SIZE);
    __gm__ uint8_t* end = (__gm__ uint8_t*)(((uint64_t)addr + length) / SHMEM_DATA_CACHE_LINE_SIZE
        * SHMEM_DATA_CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint64_t i = 0; i <= end - start; i+= SHMEM_DATA_CACHE_LINE_SIZE) {
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<uint8_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global[i]);
        __asm__ __volatile__("");
    }
}

SHMEM_DEVICE void dcci_entire_cache()
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    
    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

SHMEM_DEVICE void dcci_atomic()
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;

    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_ATOMIC>(global);
    __asm__ __volatile__("");
}

SHMEM_DEVICE void dsb_all()
{
    using namespace AscendC;
    
    DataSyncBarrier<MemDsbT::ALL>();
}

#endif