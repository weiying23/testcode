/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_HOST_RMA_H
#define SHMEM_HOST_RMA_H

#include "acl/acl.h"
#include "shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    NO_NBI = 0,
    NBI,
};

/**
 * @brief Translate an local symmetric address to remote symmetric address on the specified PE.
 *        Firstly, check whether the input address is legal on local PE. Then translate it into remote address
 *        on specified PE. Otherwise, returns a null pointer.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] The number of the remote PE.
 * @return If the input address is legal, returns a remote symmetric address on the specified PE that can be
 *         accessed using memory loads and stores. Otherwise, a null pointer is returned.
 */
SHMEM_HOST_API void* shmem_ptr(void *ptr, int pe);

/**
 * @brief Set necessary parameters for put or get.
 *
 * @param offset                [in] The start address on UB.
 * @param ub_size               [in] The Size of Temp UB Buffer.
 * @param event_id              [in] Sync ID for put or get.
 * @return Returns 0 on success or an error code on failure.
 */
SHMEM_HOST_API int shmem_mte_set_ub_params(uint64_t offset, uint32_t ub_size, uint32_t event_id);

#define SHMEM_TYPE_PUT(NAME, TYPE)                                                                                    \
    /**                                                                                                               \
    * @brief Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.      \
    *                                                                                                                 \
    * @param dest               [in] Pointer on Symmetric memory of the destination data.                             \
    * @param source             [in] Pointer on local device of the source data.                                      \
    * @param nelems             [in] Number of elements in the destination and source arrays.                         \
    * @param pe                 [in] PE number of the remote PE.                                                      \
    */                                                                                                                \
    SHMEM_HOST_API void shmem_put_##NAME##_mem(TYPE *dest, TYPE *source, size_t nelems, int pe);

SHMEM_TYPE_FUNC(SHMEM_TYPE_PUT)
#undef SHMEM_TYPE_PUT

#define SHMEM_TYPE_PUT_NBI(NAME, TYPE)                                                                                \
    /**                                                                                                               \
    * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.     \
    *                                                                                                                 \
    * @param dest               [in] Pointer on Symmetric memory of the destination data.                             \
    * @param source             [in] Pointer on local device of the source data.                                      \
    * @param nelems             [in] Number of elements in the destination and source arrays.                         \
    * @param pe                 [in] PE number of the remote PE.                                                      \
    */                                                                                                                \
    SHMEM_HOST_API void shmem_put_##NAME##_mem_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe);

SHMEM_TYPE_FUNC(SHMEM_TYPE_PUT_NBI)
#undef SHMEM_TYPE_PUT_NBI

#define SHMEM_TYPE_GET(NAME, TYPE)                                                                                    \
    /**                                                                                                               \
    * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the  \
    *   local PE.                                                                                                     \
    *                                                                                                                 \
    * @param dest               [in] Pointer on local device of the destination data.                                 \
    * @param source             [in] Pointer on Symmetric memory of the source data.                                  \
    * @param nelems             [in] Number of elements in the destination and source arrays.                         \
    * @param pe                 [in] PE number of the remote PE.                                                      \
    */                                                                                                                \
    SHMEM_HOST_API void shmem_get_##NAME##_mem(TYPE *dest, TYPE *source, size_t nelems, int pe);

SHMEM_TYPE_FUNC(SHMEM_TYPE_GET)
#undef SHMEM_TYPE_GET

#define SHMEM_TYPE_GET_NBI(NAME, TYPE)                                                                                \
    /**                                                                                                               \
    * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the \
    * local PE.                                                                                                       \
    *                                                                                                                 \
    * @param dest               [in] Pointer on local device of the destination data.                                 \
    * @param source             [in] Pointer on Symmetric memory of the source data.                                  \
    * @param nelems             [in] Number of elements in the destination and source arrays.                         \
    * @param pe                 [in] PE number of the remote PE.                                                      \
    */                                                                                                                \
    SHMEM_HOST_API void shmem_get_##NAME##_mem_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe);

SHMEM_TYPE_FUNC(SHMEM_TYPE_GET_NBI)
#undef SHMEM_TYPE_GET_NBI

#define SHMEM_PUT_TYPENAME_MEM_SIGNAL(NAME, TYPE)                                                                     \
    /**                                                                                                               \
    * @brief Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.      \
    *                                                                                                                 \
    * @param dst               [in] Pointer on local device of the destination data.                                  \
    * @param src               [in] Pointer on Symmetric memory of the source data.                                   \
    * @param elem_size         [in] Number of elements in the dest and source arrays.                                 \
    * @param sig_addr          [in] Symmetric address of the signal word to be updated.                               \
    * @param signal            [in] The value used to update sig_addr.                                                \
    * @param sig_op            [in] Operation used to update sig_addr with signal.                                    \
    *                               Supported operations: SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD                           \
    * @param pe                [in] PE number of the remote PE.                                                       \
    */                                                                                                                \
    SHMEM_HOST_API void shmem_put_##NAME##_mem_signal(TYPE* dst, TYPE* src, size_t elem_size,                         \
                                                        uint8_t *sig_addr, int32_t signal, int sig_op, int pe);

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_SIGNAL)
#undef SHMEM_PUT_TYPENAME_MEM_SIGNAL

#define SHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI(NAME, TYPE)                                                                 \
    /**                                                                                                               \
    * @brief Asynchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.     \
    *                                                                                                                 \
    * @param dst               [in] Pointer on local device of the destination data.                                  \
    * @param src               [in] Pointer on Symmetric memory of the source data.                                   \
    * @param elem_size         [in] Number of elements in the dest and source arrays.                                 \
    * @param sig_addr          [in] Symmetric address of the signal word to be updated.                               \
    * @param signal            [in] The value used to update sig_addr.                                                \
    * @param sig_op            [in] Operation used to update sig_addr with signal.                                    \
    *                               Supported operations: SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD                           \
    * @param pe                [in] PE number of the remote PE.                                                       \
    */                                                                                                                \
    SHMEM_HOST_API void shmem_put_##NAME##_mem_signal_nbi(TYPE* dst, TYPE* src, size_t elem_size,                     \
                                                        uint8_t *sig_addr, int32_t signal, int sig_op, int pe);

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI)
#undef SHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI

#define SHMEM_TYPENAME_P(NAME, TYPE)                                                        \
    /**                                                                                     \
    * @brief Provide a low latency put capability for single element of most basic types.   \
    *                                                                                       \
    * @param dst               [in] Symmetric address of the destination data on local PE.  \
    * @param value             [in] The element to be put.                                  \
    * @param pe                [in] The number of the remote PE.                            \
    */                                                                                      \
    SHMEM_HOST_API void shmem_##NAME##_p(TYPE* dst, const TYPE value, int pe);

SHMEM_TYPE_FUNC(SHMEM_TYPENAME_P)
#undef SHMEM_TYPENAME_P

#define SHMEM_TYPENAME_G(NAME, TYPE)                                                        \
    /**                                                                                     \
    * @brief Provide a low latency get single element of most basic types.                  \
    *                                                                                       \
    * @param src               [in] Symmetric address of the destination data on local PE.  \
    * @param pe                [in] The number of the remote PE.                            \
    * @return A single element of type specified in the input pointer.                      \
    */                                                                                      \
    SHMEM_HOST_API TYPE shmem_##NAME##_g(TYPE* src, int32_t pe);

SHMEM_TYPE_FUNC(SHMEM_TYPENAME_G)
#undef SHMEM_TYPENAME_G

/**
* @brief Synchronous interface. Copy contiguous data on symmetric memory from local PE to address on the specified PE.
*
* @param dst                [in] Pointer on Symmetric addr of local PE.
* @param src                [in] Pointer on local memory of the source data.
* @param elem_size          [in] size of elements in the destination and source addr.
* @param pe                 [in] PE number of the remote PE.
*/
SHMEM_HOST_API void shmem_putmem(void* dst, void* src, size_t elem_size, int32_t pe);

/**
* @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to
*        address on the local PE.
*
* @param dst                [in] Pointer on local device of the destination data.
* @param src                [in] Pointer on Symmetric memory of the source data.
* @param elem_size          [in] size of elements in the destination and source addr.
* @param pe                 [in] PE number of the remote PE.
*/
SHMEM_HOST_API void shmem_getmem(void* dst, void* src, size_t elem_size, int32_t pe);

/**
* @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to
*        address on the local PE.
*
* @param dst                [in] Pointer on Symmetric addr of local PE.
* @param src                [in] Pointer on local memory of the source data.
* @param elem_size          [in] size of elements in the destination and source addr.
* @param pe                 [in] PE number of the remote PE.
*/
SHMEM_HOST_API void shmem_putmem_nbi(void* dst, void* src, size_t elem_size, int32_t pe);

/**
* @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to
*        address on the local PE.
*
* @param dst                [in] Pointer on local device of the destination data.
* @param src                [in] Pointer on Symmetric memory of the source data.
* @param elem_size          [in] size of elements in the destination and source addr.
* @param pe                 [in] PE number of the remote PE.
*/
SHMEM_HOST_API void shmem_getmem_nbi(void* dst, void* src, size_t elem_size, int32_t pe);

/**
    * @brief Asynchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.
    *
    * @param dst               [in] Pointer on local device of the destination data.
    * @param src               [in] Pointer on Symmetric memory of the source data.
    * @param elem_size         [in] Number of elements in the dest and source arrays.
    * @param sig_addr          [in] Symmetric address of the signal word to be updated.
    * @param signal            [in] The value used to update sig_addr.
    * @param sig_op            [in] Operation used to update sig_addr with signal.
    *                               Supported operations: SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD
    * @param pe                [in] PE number of the remote PE.
 */
SHMEM_HOST_API void shmem_putmem_signal_nbi(void* dst, void* src, size_t elem_size,
                                            void* sig_addr, int32_t signal, int sig_op, int pe);

/**
    * @brief Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.
    *
    * @param dst               [in] Pointer on local device of the destination data.
    * @param src               [in] Pointer on Symmetric memory of the source data.
    * @param elem_size         [in] Number of elements in the dest and source arrays.
    * @param sig_addr          [in] Symmetric address of the signal word to be updated.
    * @param signal            [in] The value used to update sig_addr.
    * @param sig_op            [in] Operation used to update sig_addr with signal.
    *                               Supported operations: SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD
    * @param pe                [in] PE number of the remote PE.
 */
SHMEM_HOST_API void shmem_putmem_signal(void* dst, void* src, size_t elem_size,
                                        void* sig_addr, int32_t signal, int sig_op, int pe);

/**
    * @brief Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
    *
    * @param dst               [in] Pointer on local device of the destination data.
    * @param src               [in] Pointer on Symmetric memory of the source data.
    * @param elem_size         [in] Number of elements in the dest and source arrays.
    * @param pe                [in] PE number of the remote PE.
    * @param stream            [in] copy used stream(use default stream if stream == NULL).
 */
SHMEM_HOST_API void shmemx_getmem_on_stream(void* dst, void* src, size_t elem_size,
                                            int32_t pe, aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif