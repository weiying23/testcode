/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_HOST_HEAP_H
#define SHMEM_HOST_HEAP_H

#include "shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief allocate <i>size</i> bytes and returns a pointer to the allocated memory. The memory is not initialized.
 *        If <i>size</i> is 0, then <b>shmem_malloc()</b> returns NULL.
 *
 * @param size             [in] bytes to be allocated
 * @return pointer to the allocated memory.
 */
SHMEM_HOST_API void *shmem_malloc(size_t size);

/**
 * @brief allocate memory for an array of <i>nmemb</i> elements of <i>size</i> bytes each and returns a pointer to the
 *        allocated memory. The memory is set to zero. If <i>nmemb</i> or <i>size</i> is 0, then <b>shmem_calloc()</b>
 *        returns either NULL.
 *
 * @param nmemb            [in] elements count
 * @param size             [in] each element size in bytes
 * @return pointer to the allocated memory.
 */
SHMEM_HOST_API void *shmem_calloc(size_t nmemb, size_t size);

/**
 * @brief allocate <i>size</i> bytes and returns a pointer to the allocated memory. The memory address will be a
 *        multiple of <i>alignment</i>, which must be a power of two.
 *
 * @param alignment        [in] memory address alignment
 * @param size             [in] bytes allocated
 * @return pointer to the allocated memory.
 */
SHMEM_HOST_API void *shmem_align(size_t alignment, size_t size);

/**
 * @brief Free the memory space pointed to by <i>ptr</i>, which must have been returned by a previous call to
 *       <b>shmem_malloc()</b>, <b>shmem_calloc()</b> or <b>shmem_align()</b>. If <i>ptr</i> is NULL,
 *       no operation is performed.
 * @param ptr              [in] point to memory block to be free.
 */
SHMEM_HOST_API void shmem_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif