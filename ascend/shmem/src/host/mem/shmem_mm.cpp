/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "shmemi_mm_heap.h"

namespace shm {
namespace {
std::shared_ptr<memory_heap> shm_memory_heap;
}

int32_t memory_manager_initialize(void *base, uint64_t size)
{
    shm_memory_heap = std::make_shared<memory_heap>(base, size);
    if (shm_memory_heap == nullptr) {
        SHM_LOG_ERROR("Failed to initialize shared memory heap");
        return SHMEM_INNER_ERROR;
    }
    return SHMEM_SUCCESS;
}

void memory_manager_destroy()
{
    shm_memory_heap.reset();
}
}  // namespace shm

void *shmem_malloc(size_t size)
{
    if (shm::shm_memory_heap == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }

    void *ptr = shm::shm_memory_heap->allocate(size);
    SHM_LOG_DEBUG("shmem_malloc(" << size << ")");
    auto ret = shm::shmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("malloc mem barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            shm::shm_memory_heap->release(ptr);
            ptr = nullptr;
        }
    }
    return ptr;
}

void *shmem_calloc(size_t nmemb, size_t size)
{
    if (shm::shm_memory_heap == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }
    SHM_ASSERT_MULTIPLY_OVERFLOW(nmemb, size, shm::g_state.heap_size, nullptr);

    auto total_size = nmemb * size;
    auto ptr = shm::shm_memory_heap->allocate(total_size);
    if (ptr != nullptr) {
        auto ret = aclrtMemset(ptr, size, 0, size);
        if (ret != 0) {
            SHM_LOG_ERROR("shmem_calloc(" << nmemb << ", " << size << ") memset failed: " << ret);
            shm::shm_memory_heap->release(ptr);
            ptr = nullptr;
        }
    }

    auto ret = shm::shmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("calloc mem barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            shm::shm_memory_heap->release(ptr);
            ptr = nullptr;
        }
    }

    SHM_LOG_DEBUG("shmem_calloc(" << nmemb << ", " << size << ")");
    return ptr;
}

void *shmem_align(size_t alignment, size_t size)
{
    if (shm::shm_memory_heap == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }

    auto ptr = shm::shm_memory_heap->aligned_allocate(alignment, size);
    auto ret = shm::shmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("shmem_align barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            shm::shm_memory_heap->release(ptr);
            ptr = nullptr;
        }
    }
    SHM_LOG_DEBUG("shmem_align(" << alignment << ", " << size << ")");
    return ptr;
}

void shmem_free(void *ptr)
{
    if (shm::shm_memory_heap == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return;
    }
    if (ptr == nullptr) {
        return;
    }

    auto ret = shm::shm_memory_heap->release(ptr);
    if (ret != 0) {
        SHM_LOG_ERROR("release failed: " << ret);
    }

    SHM_LOG_DEBUG("shmem_free " << ret);
}
