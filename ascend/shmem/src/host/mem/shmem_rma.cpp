/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "host/shmem_host_rma.h"
#include "shmemi_device_rma.h"
#include "host_device/shmem_types.h"

using namespace std;
void *shmem_ptr(void *ptr, int32_t pe)
{
    if (pe < 0 || pe >= shmem_n_pes()) {
        SHM_LOG_ERROR("shmem_ptr Failed. PE: " << shmem_my_pe() << " Got Ilegal PE !!");
        return nullptr;
    }
    uint64_t lower_bound = (uint64_t)shm::g_state.heap_base;
    uint64_t upper_bound = lower_bound + shm::g_state.heap_size;
    if (uint64_t(ptr) < lower_bound || uint64_t(ptr) >= upper_bound) {
        SHM_LOG_ERROR("shmem_ptr Failed. PE: " << shmem_my_pe() << " Got Ilegal Address !!");
        return nullptr;
    }

    uint64_t offset = (uint64_t)ptr - (uint64_t)shm::g_state.heap_base;
    void *symm_ptr = shm::g_state.p2p_heap_host_base[pe];
    if (symm_ptr != nullptr) {
        symm_ptr = reinterpret_cast<void*>(reinterpret_cast<uint64_t>(symm_ptr) + offset);
        return symm_ptr;
    }
    SHM_LOG_ERROR("shmem_ptr Failed. PE: " << shmem_my_pe()
                  << " g_state.p2p_heap_host_base contains nullptr, Please Check Init Status!!");
    return nullptr;
}

// Set Memcpy Interfaces necessary UB Buffer.
int32_t shmem_mte_set_ub_params(uint64_t offset, uint32_t ub_size, uint32_t event_id)
{
    shm::g_state.mte_config.shmem_ub = static_cast<int64_t>(offset);
    shm::g_state.mte_config.ub_size = ub_size;
    shm::g_state.mte_config.event_id = event_id;
    SHMEM_CHECK_RET(shm::update_device_state(), update_device_state);
    return SHMEM_SUCCESS;
}

#define SHMEM_TYPE_PUT(NAME, TYPE)                                                                                    \
    /**                                                                                                               \
     * @brief Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.     \
     *                                                                                                                \
     * @param dest               [in] Pointer on Symmetric memory of the destination data.                            \
     * @param source             [in] Pointer on local device of the source data.                                     \
     * @param nelems             [in] Number of elements in the destination and source arrays.                        \
     * @param pe                 [in] PE number of the remote PE.                                                     \
     */                                                                                                               \
    SHMEM_HOST_API void shmem_put_##NAME##_mem(TYPE *dest, TYPE *source, size_t nelems, int pe)                       \
    {                                                                                                                 \
        int ret = shmemi_prepare_and_post_rma("shmem_put_" #NAME "_mem", SHMEMI_OP_PUT, NO_NBI, (uint8_t *)dest,      \
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, 1, 1,       \
                                              shm::g_state_host.default_stream, shm::g_state_host.default_block_num); \
        if (ret < 0) {                                                                                                \
            SHM_LOG_ERROR("device calling transfer failed");                                                          \
        }                                                                                                             \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPE_PUT)
#undef SHMEM_TYPE_PUT

#define SHMEM_TYPE_PUT_NBI(NAME, TYPE)                                                                                \
    /**                                                                                                               \
     * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.    \
     *                                                                                                                \
     * @param dest               [in] Pointer on Symmetric memory of the destination data.                            \
     * @param source             [in] Pointer on local device of the source data.                                     \
     * @param nelems             [in] Number of elements in the destination and source arrays.                        \
     * @param pe                 [in] PE number of the remote PE.                                                     \
     */                                                                                                               \
    SHMEM_HOST_API void shmem_put_##NAME##_mem_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe)                   \
    {                                                                                                                 \
        int ret = shmemi_prepare_and_post_rma("shmem_put_" #NAME "_mem_nbi", SHMEMI_OP_PUT, NBI, (uint8_t *)dest,     \
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, 1, 1,       \
                                              shm::g_state_host.default_stream, shm::g_state_host.default_block_num); \
        if (ret < 0) {                                                                                                \
            SHM_LOG_ERROR("device calling transfer failed");                                                          \
        }                                                                                                             \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPE_PUT_NBI)
#undef SHMEM_TYPE_PUT_NBI

#define SHMEM_TYPE_GET(NAME, TYPE)                                                                                    \
    /**                                                                                                               \
     * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the \
     * local PE.                                                                                                      \
     *                                                                                                                \
     * @param dest               [in] Pointer on local device of the destination data.                                \
     * @param source             [in] Pointer on Symmetric memory of the source data.                                 \
     * @param nelems             [in] Number of elements in the destination and source arrays.                        \
     * @param pe                 [in] PE number of the remote PE.                                                     \
     */                                                                                                               \
    SHMEM_HOST_API void shmem_get_##NAME##_mem(TYPE *dest, TYPE *source, size_t nelems, int pe)                       \
    {                                                                                                                 \
        int ret = shmemi_prepare_and_post_rma("shmem_get_" #NAME "_mem", SHMEMI_OP_GET, NO_NBI, (uint8_t *)dest,      \
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, 1, 1,       \
                                              shm::g_state_host.default_stream, shm::g_state_host.default_block_num); \
        if (ret < 0) {                                                                                                \
            SHM_LOG_ERROR("device calling transfer failed");                                                          \
        }                                                                                                             \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPE_GET)
#undef SHMEM_TYPE_GET

#define SHMEM_TYPE_GET_NBI(NAME, TYPE)                                                                                 \
    /**                                                                                                                \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the \
     * local PE.                                                                                                       \
     *                                                                                                                 \
     * @param dest               [in] Pointer on local device of the destination data.                                 \
     * @param source             [in] Pointer on Symmetric memory of the source data.                                  \
     * @param nelems             [in] Number of elements in the destination and source arrays.                         \
     * @param pe                 [in] PE number of the remote PE.                                                      \
     */                                                                                                                \
    SHMEM_HOST_API void shmem_get_##NAME##_mem_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe)                    \
    {                                                                                                                  \
        int ret = shmemi_prepare_and_post_rma("shmem_get_" #NAME "_mem_nbi", SHMEMI_OP_GET, NBI, (uint8_t *)dest,      \
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, 1, 1,        \
                                              shm::g_state_host.default_stream, shm::g_state_host.default_block_num);  \
        if (ret < 0) {                                                                                                 \
            SHM_LOG_ERROR("device calling transfer failed");                                                           \
        }                                                                                                              \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPE_GET_NBI)
#undef SHMEM_TYPE_GET_NBI

#define SHMEM_PUT_TYPENAME_MEM_SIGNAL(NAME, TYPE)                                                                    \
    /**                                                                                                              \
     * @brief Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.    \
     *                                                                                                               \
     * @param dst               [in] Pointer on local device of the destination data.                                \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                 \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                               \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                             \
     * @param signal            [in] The value used to update sig_addr.                                              \
     * @param sig_op            [in] Operation used to update sig_addr with signal.                                  \
     *                          Supported operations: SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD                              \
     * @param pe                [in] PE number of the remote PE.                                                     \
     */                                                                                                              \
    SHMEM_HOST_API void shmem_put_##NAME##_mem_signal(TYPE *dst, TYPE *src, size_t elem_size, uint8_t *sig_addr,     \
                                                      int32_t signal, int sig_op, int pe)                            \
    {                                                                                                                \
        int ret = shmemi_prepare_and_post_rma("shmem_put_" #NAME "_mem_signal", SHMEMI_OP_PUT_SIGNAL, NO_NBI,        \
                                              (uint8_t *)dst, (uint8_t *)src, elem_size, sizeof(TYPE), pe, sig_addr, \
                                              signal, sig_op, 1, 1, shm::g_state_host.default_stream,                \
                                              shm::g_state_host.default_block_num);                                  \
        if (ret < 0) {                                                                                               \
            SHM_LOG_ERROR("device calling transfer failed");                                                         \
        }                                                                                                            \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_SIGNAL)
#undef SHMEM_PUT_TYPENAME_MEM_SIGNAL

#define SHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI(NAME, TYPE)                                                                \
    /**                                                                                                              \
     * @brief Asynchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.   \
     *                                                                                                               \
     * @param dst               [in] Pointer on local device of the destination data.                                \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                 \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                               \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                             \
     * @param signal            [in] The value used to update sig_addr.                                              \
     * @param sig_op            [in] Operation used to update sig_addr with signal.                                  \
     *                          Supported operations: SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD                              \
     * @param pe                [in] PE number of the remote PE.                                                     \
     */                                                                                                              \
    SHMEM_HOST_API void shmem_put_##NAME##_mem_signal_nbi(TYPE *dst, TYPE *src, size_t elem_size, uint8_t *sig_addr, \
                                                          int32_t signal, int sig_op, int pe)                        \
    {                                                                                                                \
        int ret = shmemi_prepare_and_post_rma("shmem_put_" #NAME "_mem_signal_nbi", SHMEMI_OP_PUT_SIGNAL, NBI,       \
                                              (uint8_t *)dst, (uint8_t *)src, elem_size, sizeof(TYPE), pe, sig_addr, \
                                              signal, sig_op, 1, 1, shm::g_state_host.default_stream,                \
                                              shm::g_state_host.default_block_num);                                  \
        if (ret < 0) {                                                                                               \
            SHM_LOG_ERROR("device calling transfer failed");                                                         \
        }                                                                                                            \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI)
#undef SHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI

#define SHMEM_TYPENAME_P(NAME, TYPE)                                                                                   \
    /**                                                                                                                \
     * @brief Provide a low latency put capability for single element of most basic types.                             \
     *                                                                                                                 \
     * @param dst               [in] Symmetric address of the destination data on local PE.                            \
     * @param value             [in] The element to be put.                                                            \
     * @param pe                [in] The number of the remote PE.                                                      \
     */                                                                                                                \
    SHMEM_HOST_API void shmem_##NAME##_p(TYPE *dst, const TYPE value, int pe)                                          \
    {                                                                                                                  \
        shmemi_prepare_and_post_rma_##NAME##_p("shmem_" #NAME "_p", (uint8_t *)dst, value, pe,                         \
                                               shm::g_state_host.default_stream, shm::g_state_host.default_block_num); \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPENAME_P)
#undef SHMEM_TYPENAME_P

#define SHMEM_TYPENAME_G(NAME, TYPE)                                                                                   \
    /**                                                                                                                \
     * @brief Provide a low latency get single element of most basic types.                                            \
     *                                                                                                                 \
     * @param src               [in] Symmetric address of the destination data on local PE.                            \
     * @param pe                [in] The number of the remote PE.                                                      \
     * @return A single element of type specified in the input pointer.                                                \
     */                                                                                                                \
    SHMEM_HOST_API TYPE shmem_##NAME##_g(TYPE *src, int32_t pe)                                                        \
    {                                                                                                                  \
        TYPE value {};                                                                                                 \
        auto ptr = shmem_ptr(src, pe);                                                                                 \
        if (ptr == nullptr) {                                                                                          \
            SHM_LOG_ERROR("shmem_g failed");                                                                           \
            return value;                                                                                              \
        }                                                                                                              \
        int ret =                                                                                                      \
            aclrtMemcpy(&value, sizeof(TYPE), reinterpret_cast<void *>(ptr), sizeof(TYPE), ACL_MEMCPY_DEVICE_TO_HOST); \
        if (ret != 0) {                                                                                                \
            SHM_LOG_ERROR("shmem_g failed");                                                                           \
        }                                                                                                              \
        return value;                                                                                                  \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPENAME_G)
#undef SHMEM_TYPENAME_G

void shmem_putmem(void *dst, void *src, size_t elem_size, int32_t pe)
{
    int ret = shmemi_prepare_and_post_rma("shmem putmem", SHMEMI_OP_PUT, NO_NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, shm::g_state_host.default_stream,
                                          shm::g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("shmem_putmem failed");
    }
}

void shmem_getmem(void *dst, void *src, size_t elem_size, int32_t pe)
{
    int ret = shmemi_prepare_and_post_rma("shmem getmem", SHMEMI_OP_GET, NO_NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, shm::g_state_host.default_stream,
                                          shm::g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("shmem_getmem failed");
    }
}

void shmem_putmem_nbi(void *dst, void *src, size_t elem_size, int32_t pe)
{
    int ret = shmemi_prepare_and_post_rma("shmem_putmem_nbi", SHMEMI_OP_PUT, NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, shm::g_state_host.default_stream,
                                          shm::g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("shmem_putmem_nbi failed");
    }
}

void shmem_getmem_nbi(void *dst, void *src, size_t elem_size, int32_t pe)
{
    int ret = shmemi_prepare_and_post_rma("shmem_getmem_nbi", SHMEMI_OP_GET, NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, shm::g_state_host.default_stream,
                                          shm::g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("shmem_getmem_nbi failed");
    }
}

void shmem_putmem_signal_nbi(void *dst, void *src, size_t elem_size, void *sig_addr, int32_t signal, int sig_op, int pe)
{
    int ret = shmemi_prepare_and_post_rma("shmem_putmem_signal_nbi", SHMEMI_OP_PUT_SIGNAL, NBI, (uint8_t *)dst,
                                          (uint8_t *)src, elem_size, 1, pe, (uint8_t *)sig_addr, signal, sig_op, 1, 1,
                                          shm::g_state_host.default_stream, shm::g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("device calling transfer failed");
    }
}

void shmem_putmem_signal(void *dst, void *src, size_t elem_size, void *sig_addr, int32_t signal, int sig_op, int pe)
{
    int ret = shmemi_prepare_and_post_rma("shmem_putmem_signal", SHMEMI_OP_PUT_SIGNAL, NO_NBI, (uint8_t *)dst,
                                          (uint8_t *)src, elem_size, 1, pe, (uint8_t *)sig_addr, signal, sig_op, 1, 1,
                                          shm::g_state_host.default_stream, shm::g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("device calling transfer failed");
    }
}

void shmemx_getmem_on_stream(void* dst, void* src, size_t elem_size, int32_t pe, aclrtStream stream)
{
    int ret = shmemi_getmem_on_stream((uint8_t *)dst, (uint8_t *)src, elem_size, pe, stream);
    if (ret < 0) {
        SHM_LOG_ERROR("shmemi_getmem_on_stream failed");
    }
}
