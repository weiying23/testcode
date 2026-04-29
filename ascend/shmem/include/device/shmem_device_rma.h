/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_RMA_H
#define SHMEM_DEVICE_RMA_H

#include "kernel_operator.h"
#include "internal/device/shmemi_device_common.h"
#include "low_level/shmem_device_low_level_rma.h"
#include "low_level/shmem_device_low_level_roce.h"
#include "shmem_device_team.h"
#include "internal/device/sync/shmemi_device_p2p.h"
#include "shmem_device_sync.h"
#include "host/shmem_host_def.h"

/**
 * @brief Standard RMA Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |half       | half      |
 * |float      | float     |
 * |double     | double    |
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 * |int64      | int64     |
 * |uint8      | uint8     |
 * |uint16     | uint16    |
 * |uint32     | uint32    |
 * |uint64     | uint64    |
 * |char       | char      |
 * |bfloat16   | bfloat16  |
 */
#define SHMEM_TYPE_FUNC(FUNC) \
    FUNC(half, half);         \
    FUNC(float, float);       \
    FUNC(double, double);     \
    FUNC(int8, int8_t);       \
    FUNC(int16, int16_t);     \
    FUNC(int32, int32_t);     \
    FUNC(int64, int64_t);     \
    FUNC(uint8, uint8_t);     \
    FUNC(uint16, uint16_t);   \
    FUNC(uint32, uint32_t);   \
    FUNC(uint64, uint64_t);   \
    FUNC(char, char);         \
    FUNC(bfloat16, bfloat16_t)

#define SHMEM_TEST_TYPE_FUNC(FUNC) \
    FUNC(float, float);            \
    FUNC(int8, int8_t);            \
    FUNC(int16, int16_t);          \
    FUNC(int32, int32_t);          \
    FUNC(int64, int64_t);          \
    FUNC(uint8, uint8_t);          \
    FUNC(uint16, uint16_t);        \
    FUNC(uint32, uint32_t);        \
    FUNC(uint64, uint64_t);        \
    FUNC(char, char);

#define SHMEM_TYPENAME_P_AICORE(NAME, TYPE)                                                 \
    /**                                                                                     \
     * @brief Provide a low latency put capability for single element of most basic types.  \
     *                                                                                      \
     * @param dst               [in] Symmetric address of the destination data on local PE. \
     * @param value             [in] The element to be put.                                 \
     * @param pe                [in] The number of the remote PE.                           \
     */                                                                                     \
    SHMEM_DEVICE void shmem_##NAME##_p(__gm__ TYPE *dst, const TYPE value, int pe)          \
    {                                                                                       \
        auto ptr = shmem_ptr(dst, pe);                                                      \
        __gm__ TYPE *addr_gm = reinterpret_cast<__gm__ TYPE *>(ptr);                        \
                                                                                            \
        *addr_gm = value;                                                                   \
        dcci_cacheline((__gm__ uint8_t *)addr_gm);                                          \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPENAME_P_AICORE);

#define SHMEM_TYPENAME_G_AICORE(NAME, TYPE)                                                 \
    /**                                                                                     \
     * @brief Provide a low latency get capability for single element of most basic types.  \
     *                                                                                      \
     * @param src               [in] Symmetric address of the destination data on local PE. \
     * @param pe                [in] The number of the remote PE.                           \
     * @return A single element of type specified in the input pointer.                     \
     */                                                                                     \
    SHMEM_DEVICE TYPE shmem_##NAME##_g(__gm__ TYPE *src, int32_t pe)                        \
    {                                                                                       \
        auto ptr = shmem_ptr(src, pe);                                                      \
        __gm__ TYPE *addr_gm = reinterpret_cast<__gm__ TYPE *>(ptr);                        \
                                                                                            \
        dcci_cacheline((__gm__ uint8_t *)addr_gm);                                          \
        return *addr_gm;                                                                    \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPENAME_G_AICORE);

/**
 * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *                       address on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
SHMEM_DEVICE void shmem_getmem(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)
{
    /* ROCE */
    /* RDMA */
    /* MTE  */
    /* Global State Get */
    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.shmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
    shmem_mte_get_mem_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
    shmem_quiet();
}

#define SHMEM_GET_TYPENAME_MEM(NAME, TYPE)                                                                       \
    /**                                                                                                          \
     * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to           \
     *                               address on the local PE.                                                    \
     *                                                                                                           \
     * @param dst               [in] Pointer on local device of the destination data.                            \
     * @param src               [in] Pointer on Symmetric memory of the source data.                             \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                           \
     * @param pe                [in] PE number of the remote PE.                                                 \
     */                                                                                                          \
    SHMEM_DEVICE void shmem_get_##NAME##_mem(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe) \
    {                                                                                                            \
        /* ROCE */                                                                                               \
        /* RDMA */                                                                                               \
        /* MTE  */                                                                                               \
        /* Global State Get */                                                                                   \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                    \
        /* CopyUB Config Set */                                                                                  \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                    \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                  \
        shmem_mte_get_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe, \
                              copy_event_id);                                                                    \
        shmem_quiet();                                                                                           \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM);

/**
 * @brief Synchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
SHMEM_DEVICE void shmem_putmem(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)
{
    /* ROCE */
    /* RDMA */
    /* MTE  */
    /* Global State Get */
    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.shmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
    shmem_mte_put_mem_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
    shmem_quiet();
}

#define SHMEM_PUT_TYPENAME_MEM(NAME, TYPE)                                                                        \
    /**                                                                                                           \
     * @brief Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE. \
     *                                                                                                            \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                         \
     * @param src               [in] Pointer on local device of the source data.                                  \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                     \
     * @param pe                [in] PE number of the remote PE.                                                  \
     */                                                                                                           \
    SHMEM_DEVICE void shmem_put_##NAME##_mem(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)  \
    {                                                                                                             \
        /* ROCE */                                                                                                \
        /* RDMA */                                                                                                \
        /* MTE  */                                                                                                \
        /* Global State Get */                                                                                    \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                     \
        /* CopyUB Config Set */                                                                                   \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                     \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                 \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                   \
        shmem_mte_put_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,  \
                              copy_event_id);                                                                     \
        shmem_quiet();                                                                                            \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM);

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *                                              address on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
SHMEM_DEVICE void shmem_getmem_nbi(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)
{
    /* ROCE */
    /* RDMA */
    /* MTE  */
    /* Global State Get */
    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.shmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
    shmem_mte_get_mem_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
}

#define SHMEM_GET_TYPENAME_MEM_NBI(NAME, TYPE)                                                                       \
    /**                                                                                                              \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to              \
     *                                                   address on the local PE.                                    \
     *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the             \
     *                 same PE are not supported.                                                                    \
     *                                                                                                               \
     * @param dst               [in] Pointer on local device of the destination data.                                \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                 \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                               \
     * @param pe                [in] PE number of the remote PE.                                                     \
     */                                                                                                              \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe) \
    {                                                                                                                \
        /* Global State Get */                                                                                       \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                        \
        if (device_state->topo_list[pe] & SHMEM_TRANSPORT_MTE) {                                                     \
            /* MTE  */                                                                                               \
            /* CopyUB Config Set */                                                                                  \
            uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                    \
            uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                \
            AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                  \
            shmem_mte_get_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe, \
                                    copy_event_id);                                                                  \
        } else if (device_state->topo_list[pe] & SHMEM_TRANSPORT_ROCE) {                                             \
            /* RoCE */                                                                                               \
            auto ptr = shmem_ptr(src, pe);                                                                           \
            if (ptr == nullptr) return;                                                                              \
            /* Create LocalTensor */                                                                                 \
            AscendC::LocalTensor<uint32_t> ub_tensor_32;                                                             \
            ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                       \
            ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR);         \
            ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;                                                           \
            AscendC::LocalTensor<uint64_t> ub_tensor_64;                                                             \
            ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                       \
            ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR           \
                                                                             + UB_ALIGN_SIZE);                       \
            ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;                                                           \
            shmemi_roce_read((__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(TYPE),            \
                                ub_tensor_64, ub_tensor_32);                                                         \
        }                                                                                                            \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_NBI);

#define SHMEM_GET_TYPENAME_MEM_DETAILED_NBI(NAME, TYPE)                                                            \
    /**                                                                                                            \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                   \
     *        on symmetric memory from the specified PE to address on the local device.                            \
     *                                                                                                             \
     * @param dst               [in] Pointer on local device of the destination data.                              \
     * @param src               [in] Pointer on Symmetric memory of the source data.                               \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.         \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                               \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)         \
    {                                                                                                              \
        /* ROCE */                                                                                                 \
        /* RDMA */                                                                                                 \
        /* MTE  */                                                                                                 \
        /* Global State Get */                                                                                     \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                      \
        /* CopyUB Config Set */                                                                                    \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                      \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                  \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                    \
        shmem_mte_get_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, copy_params, pe, \
                              copy_event_id);                                                                      \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_DETAILED_NBI);

#define SHMEM_GET_TYPENAME_MEM_TENSOR_NBI(NAME, TYPE)                                                              \
    /**                                                                                                            \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to            \
     *                           address on the local PE.                                                          \
     *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the           \
     *                 same PE are not supported.                                                                  \
     *                                                                                                             \
     * @param dst               [in] GlobalTensor on local device of the destination data.                         \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                          \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                             \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                 uint32_t elem_size, int pe)                                       \
    {                                                                                                              \
        /* Global State Get */                                                                                     \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                      \
        if (device_state->topo_list[pe] & SHMEM_TRANSPORT_MTE) {                                                   \
            /* MTE  */                                                                                             \
            /* CopyUB Config Set */                                                                                \
            uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                  \
            /* Create LocalTensor */                                                                               \
            AscendC::LocalTensor<TYPE> ub_tensor;                                                                  \
            ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                         \
            ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                   \
            ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                         \
            AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                \
            shmem_mte_get_mem_nbi(dst, src, ub_tensor, elem_size, pe, copy_event_id);                              \
        } else if (device_state->topo_list[pe] & SHMEM_TRANSPORT_ROCE) {                                           \
            /* RoCE */                                                                                             \
            auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);                                             \
            if (ptr == nullptr) return;                                                                            \
            /* Create LocalTensor */                                                                               \
            AscendC::LocalTensor<uint32_t> ub_tensor_32;                                                           \
            ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                     \
            ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR);       \
            ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;                                                         \
            AscendC::LocalTensor<uint64_t> ub_tensor_64;                                                           \
            ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                     \
            ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR         \
                                                                            + UB_ALIGN_SIZE);                      \
            ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;                                                         \
            shmemi_roce_read((__gm__ uint8_t*)(dst.GetPhyAddr()), (__gm__ uint8_t*)ptr, pe, 0,                     \
                                elem_size * sizeof(TYPE), ub_tensor_64, ub_tensor_32);                             \
        }                                                                                                          \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_TENSOR_NBI);

#define SHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED_NBI(NAME, TYPE)                                                     \
    /**                                                                                                            \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                   \
     *        on symmetric memory from the specified PE to address on the local device.                            \
     *                                                                                                             \
     * @param dst               [in] GlobalTensor on local device of the destination data.                         \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                          \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.         \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                 const non_contiguous_copy_param &copy_params, int pe)             \
    {                                                                                                              \
        /* ROCE */                                                                                                 \
        /* RDMA */                                                                                                 \
        /* MTE  */                                                                                                 \
        /* Global State Get */                                                                                     \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                      \
        /* CopyUB Config Set */                                                                                    \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                      \
        /* Create LocalTensor */                                                                                   \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                      \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                             \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                       \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                             \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                    \
        shmem_mte_get_mem_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED_NBI);

#define SHMEM_PUT_TYPENAME_MEM_NBI(NAME, TYPE)                                                                       \
    /**                                                                                                              \
     * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.     \
     *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the             \
     *                 same PE are not supported.                                                                    \
     *                                                                                                               \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                            \
     * @param src               [in] Pointer on local device of the source data.                                     \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                        \
     * @param pe                [in] PE number of the remote PE.                                                     \
     */                                                                                                              \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe) \
    {                                                                                                                \
        /* Global State Get */                                                                                       \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                        \
        if (device_state->topo_list[pe] & SHMEM_TRANSPORT_MTE) {                                                     \
            /* MTE  */                                                                                               \
            /* CopyUB Config Set */                                                                                  \
            uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                    \
            uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                \
            AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                  \
            shmem_mte_put_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe, \
                                    copy_event_id);                                                                  \
        } else if (device_state->topo_list[pe] & SHMEM_TRANSPORT_ROCE) {                                             \
            /* RoCE */                                                                                               \
            auto ptr = shmem_ptr(dst, pe);                                                                           \
            if (ptr == nullptr) return;                                                                              \
            /* Create LocalTensor */                                                                                 \
            AscendC::LocalTensor<uint32_t> ub_tensor_32;                                                             \
            ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                       \
            ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR);         \
            ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;                                                           \
            AscendC::LocalTensor<uint64_t> ub_tensor_64;                                                             \
            ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                       \
            ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR           \
                                                                            + UB_ALIGN_SIZE);                        \
            ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;                                                           \
            shmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(TYPE),           \
                                    ub_tensor_64, ub_tensor_32);                                                     \
        }                                                                                                            \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_NBI);

#define SHMEM_PUT_TYPENAME_MEM_DETAILED_NBI(NAME, TYPE)                                                            \
    /**                                                                                                            \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                   \
     *        on local PE to symmetric address on the specified PE.                                                \
     *                                                                                                             \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                          \
     * @param src               [in] Pointer on local device of the source data.                                   \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.         \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                               \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)         \
    {                                                                                                              \
        /* ROCE */                                                                                                 \
        /* RDMA */                                                                                                 \
        /* MTE  */                                                                                                 \
        /* Global State Get */                                                                                     \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                      \
        /* CopyUB Config Set */                                                                                    \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                      \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                  \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                    \
        shmem_mte_put_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, copy_params, pe, \
                              copy_event_id);                                                                      \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_DETAILED_NBI);

#define SHMEM_PUT_TYPENAME_MEM_TENSOR_NBI(NAME, TYPE)                                                              \
    /**                                                                                                            \
     * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.   \
     *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the           \
     *                 same PE are not supported.                                                                  \
     *                                                                                                             \
     * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                     \
     * @param src               [in] GlobalTensor on local device of the source data.                              \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                      \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                 uint32_t elem_size, int pe)                                       \
    {                                                                                                              \
        /* Global State Get */                                                                                     \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                      \
        if (device_state->topo_list[pe] & SHMEM_TRANSPORT_MTE) {                                                   \
            /* MTE  */                                                                                             \
            /* CopyUB Config Set */                                                                                \
            uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                  \
            /* Create LocalTensor */                                                                               \
            AscendC::LocalTensor<TYPE> ub_tensor;                                                                  \
            ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                         \
            ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                   \
            ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                         \
            AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                \
            shmem_mte_put_mem_nbi(dst, src, ub_tensor, elem_size, pe, copy_event_id);                              \
        } else if (device_state->topo_list[pe] & SHMEM_TRANSPORT_ROCE) {                                           \
            /* RoCE */                                                                                             \
            auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);                                             \
            if (ptr == nullptr) return;                                                                            \
            /* Create LocalTensor */                                                                               \
            AscendC::LocalTensor<uint32_t> ub_tensor_32;                                                           \
            ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                     \
            ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR);       \
            ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;                                                         \
            AscendC::LocalTensor<uint64_t> ub_tensor_64;                                                           \
            ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                     \
            ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR         \
                                                                            + UB_ALIGN_SIZE);                      \
            ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;                                                         \
            shmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)(src.GetPhyAddr()), pe, 0,                    \
                                                    elem_size * sizeof(TYPE), ub_tensor_64, ub_tensor_32);         \
        }                                                                                                          \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_TENSOR_NBI);

#define SHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED_NBI(NAME, TYPE)                                                     \
    /**                                                                                                            \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                   \
     *        on local PE to symmetric address on the specified PE.                                                \
     *                                                                                                             \
     * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                     \
     * @param src               [in] GlobalTensor on local device of the source data.                              \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.         \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                 const non_contiguous_copy_param &copy_params, int pe)             \
    {                                                                                                              \
        /* ROCE */                                                                                                 \
        /* RDMA */                                                                                                 \
        /* MTE  */                                                                                                 \
        /* Global State Get */                                                                                     \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                      \
        /* CopyUB Config Set */                                                                                    \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                      \
        /* Create LocalTensor */                                                                                   \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                      \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                             \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                       \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                             \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                    \
        shmem_mte_put_mem_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED_NBI);

#define SHMEM_GET_TYPENAME_MEM_UB_NBI(NAME, TYPE)                                                                  \
    /**                                                                                                            \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the                            \
     *          specified PE to address on the local UB.                                                           \
     *                                                                                                             \
     * @param dst               [in] Pointer on local UB of the destination data.                                  \
     * @param src               [in] Pointer on Symmetric memory of the source data.                               \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                      \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int pe) \
    {                                                                                                              \
        /* MTE  */                                                                                                 \
        /* Global State Get */                                                                                     \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                      \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                    \
        shmem_mte_get_mem_nbi(dst, src, elem_size, pe, copy_event_id);                                             \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_UB_NBI);

#define SHMEM_GET_TYPENAME_MEM_UB_TENSOR_NBI(NAME, TYPE)                                                          \
    /**                                                                                                           \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to           \
     *          address on the local UB.                                                                          \
     *                                                                                                            \
     * @param dst               [in] LocalTensor on local UB of the destination data.                             \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                         \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                     \
     * @param pe                [in] PE number of the remote PE.                                                  \
     */                                                                                                           \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::LocalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                 uint32_t elem_size, int pe)                                      \
    {                                                                                                             \
        /* MTE  */                                                                                                \
        /* Global State Get */                                                                                    \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                     \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                   \
        shmem_mte_get_mem_nbi(dst, src, elem_size, pe, copy_event_id);                                            \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_UB_TENSOR_NBI);

#define SHMEM_GET_TYPENAME_MEM_UB_DETAILED_NBI(NAME, TYPE)                                                 \
    /**                                                                                                    \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data           \
     *        on symmetric memory from the specified PE to address on the local UB.                        \
     *                                                                                                     \
     * @param dst               [in] Pointer on local UB of the destination data.                          \
     * @param src               [in] Pointer on Symmetric memory of the source data.                       \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst. \
     * @param pe                [in] PE number of the remote PE.                                           \
     */                                                                                                    \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src,                     \
                                                 const non_contiguous_copy_param &copy_params, int pe)     \
    {                                                                                                      \
        /* MTE  */                                                                                         \
        /* Global State Get */                                                                             \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                              \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;            \
        shmem_mte_get_mem_nbi(dst, src, copy_params, pe, copy_event_id);                                   \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_UB_DETAILED_NBI);

#define SHMEM_GET_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI(NAME, TYPE)                                                 \
    /**                                                                                                           \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                  \
     *        on symmetric memory from the specified PE to address on the local UB.                               \
     *                                                                                                            \
     * @param dst               [in] LocalTensor on local UB of the destination data.                             \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                         \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.        \
     * @param pe                [in] PE number of the remote PE.                                                  \
     */                                                                                                           \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::LocalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                 const non_contiguous_copy_param &copy_params, int pe)            \
    {                                                                                                             \
        /* MTE  */                                                                                                \
        /* Global State Get */                                                                                    \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                     \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                   \
        shmem_mte_get_mem_nbi(dst, src, copy_params, pe, copy_event_id);                                          \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
SHMEM_DEVICE void shmem_putmem_nbi(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)
{
    /* MTE  */
    /* Global State Get */
    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.shmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
    shmem_mte_put_mem_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
}

#define SHMEM_PUT_TYPENAME_MEM_UB_NBI(NAME, TYPE)                                                                      \
    /**                                                                                                                \
     * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.       \
     *                                                                                                                 \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                              \
     * @param src               [in] Pointer on local UB of the source data.                                           \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                          \
     * @param pe                [in] PE number of the remote PE.                                                       \
     */                                                                                                                \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src, uint32_t elem_size, int32_t pe) \
    {                                                                                                                  \
        /* MTE  */                                                                                                     \
        /* Global State Get */                                                                                         \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                          \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                        \
        shmem_mte_put_mem_nbi(dst, src, elem_size, pe, copy_event_id);                                                 \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_UB_NBI);

#define SHMEM_PUT_TYPENAME_MEM_UB_TENSOR_NBI(NAME, TYPE)                                                          \
    /**                                                                                                           \
     * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.  \
     *                                                                                                            \
     * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                    \
     * @param src               [in] LocalTensor on local UB of the source data.                                  \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                     \
     * @param pe                [in] PE number of the remote PE.                                                  \
     */                                                                                                           \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src, \
                                                 uint32_t elem_size, int32_t pe)                                  \
    {                                                                                                             \
        /* MTE  */                                                                                                \
        /* Global State Get */                                                                                    \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                     \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                   \
        shmem_mte_put_mem_nbi(dst, src, elem_size, pe, copy_event_id);                                            \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_UB_TENSOR_NBI);

#define SHMEM_PUT_TYPENAME_MEM_UB_DETAILED_NBI(NAME, TYPE)                                                   \
    /**                                                                                                      \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data             \
     *        on local UB to symmetric address on the specified PE.                                          \
     *                                                                                                       \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                    \
     * @param src               [in] Pointer on local UB of the source data.                                 \
     * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst. \
     * @param pe                [in] PE number of the remote PE.                                             \
     */                                                                                                      \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src,                       \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)   \
    {                                                                                                        \
        /* MTE  */                                                                                           \
        /* Global State Get */                                                                               \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;              \
        shmem_mte_put_mem_nbi(dst, src, copy_params, pe, copy_event_id);                                     \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_UB_DETAILED_NBI);

#define SHMEM_PUT_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI(NAME, TYPE)                                                 \
    /**                                                                                                           \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                  \
     *        on local UB to symmetric address on the specified PE.                                               \
     *                                                                                                            \
     * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                    \
     * @param src               [in] LocalTensor on local UB of the source data.                                  \
     * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.      \
     * @param pe                [in] PE number of the remote PE.                                                  \
     */                                                                                                           \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src, \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)        \
    {                                                                                                             \
        /* MTE  */                                                                                                \
        /* Global State Get */                                                                                    \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                     \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                   \
        shmem_mte_put_mem_nbi(dst, src, copy_params, pe, copy_event_id);                                          \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI);

/**
 * @brief Synchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE
 *       then update sig_addr
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param sig_addr          [in] Symmetric address of the signal word to be updated.
 * @param signal            [in] The value used to update sig_addr.
 * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:
 *                               SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD
 * @param pe                [in] PE number of the remote PE.
 */
SHMEM_DEVICE void shmem_putmem_signal(__gm__ void *dst, __gm__ void *src, size_t elem_size, __gm__ int32_t *sig_addr,
                                      int32_t signal, int sig_op, int pe)
{
    /* ROCE */
    /* RDMA */
    /* MTE  */
    /* Global State Set */
    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.shmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
    shmem_mte_put_mem_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
    shmem_quiet();
    shmemix_signal_op(sig_addr, signal, sig_op, pe);
}

#define SHMEM_PUT_TYPENAME_MEM_SIGNAL(NAME, TYPE)                                                                 \
    /**                                                                                                           \
     * @brief Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE. \
     *                                                                                                            \
     * @param dst               [in] Pointer on local device of the destination data.                             \
     * @param src               [in] Pointer on Symmetric memory of the source data.                              \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                            \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                          \
     * @param signal            [in] The value used to update sig_addr.                                           \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:         \
     *                               SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD                                            \
     * @param pe                [in] PE number of the remote PE.                                                  \
     */                                                                                                           \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_signal(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size,         \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe) \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                     \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                     \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                   \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                     \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                 \
        shmem_mte_put_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,  \
                              copy_event_id);                                                                     \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                            \
        shmem_quiet();                                                                                            \
        shmemix_signal_op(sig_addr, signal, sig_op, pe);                                                          \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_SIGNAL);

#define SHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR(NAME, TYPE)                                                              \
    /**                                                                                                               \
     * @brief Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.     \
     *                                                                                                                \
     * @param dst               [in] Pointer on local device of the destination data.                                 \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                  \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                                \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                              \
     * @param signal            [in] The value used to update sig_addr.                                               \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:             \
     *                               SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD                                                \
     * @param pe                [in] PE number of the remote PE.                                                      \
     */                                                                                                               \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_signal(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                    size_t elem_size, __gm__ int32_t *sig_addr, int32_t signal,       \
                                                    int sig_op, int pe)                                               \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                         \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                         \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                       \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                         \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                         \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                          \
        ub_tensor.address_.logicPos = device_state->mte_config.ub_size;                                               \
        shmem_mte_put_mem_nbi(dst, src, ub_tensor, elem_size, pe, copy_event_id);                                     \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                                \
        shmem_quiet();                                                                                                \
        shmemix_signal_op(sig_addr, signal, sig_op, pe);                                                              \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR);

#define SHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED(NAME, TYPE)                                                         \
    /**                                                                                                            \
     * @brief Synchronous interface. Provide a high-performance way to copy non-contiguous data                    \
     *        on local UB to symmetric address on the specified PE then update sig_addr                            \
     *                                                                                                             \
     * @param dst               [in] Pointer on local device of the destination data.                              \
     * @param src               [in] Pointer on Symmetric memory of the source data.                               \
     * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.       \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                           \
     * @param signal            [in] The value used to update sig_addr.                                            \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:          \
     *                               SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD                                             \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_signal(__gm__ TYPE *dst, __gm__ TYPE *src,                            \
                                                    const non_contiguous_copy_param &copy_params,                  \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)  \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                      \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                      \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                    \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                      \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                  \
        shmem_mte_put_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, copy_params, pe, \
                              copy_event_id);                                                                      \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                             \
        shmem_quiet();                                                                                             \
        shmemix_signal_op(sig_addr, signal, sig_op, pe);                                                           \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED);

#define SHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED(NAME, TYPE)                                                     \
    /**                                                                                                               \
     * @brief Synchronous interface. Provide a high-performance way to copy non-contiguous data                       \
     *        on local UB to symmetric address on the specified PE.                                                   \
     *                                                                                                                \
     * @param dst               [in] Pointer on local device of the destination data.                                 \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                  \
     * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.          \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                              \
     * @param signal            [in] The value used to update sig_addr.                                               \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:             \
     *                               SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD                                                \
     * @param pe                [in] PE number of the remote PE.                                                      \
     */                                                                                                               \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_signal(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                    const non_contiguous_copy_param &copy_params,                     \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)     \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                         \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                         \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                       \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                         \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                         \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                          \
        ub_tensor.address_.logicPos = device_state->mte_config.ub_size;                                               \
        shmem_mte_put_mem_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                   \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                                \
        shmem_quiet();                                                                                                \
        shmemix_signal_op(sig_addr, signal, sig_op, pe);                                                              \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE then update
 * sig_addr
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param sig_addr          [in] Symmetric address of the signal word to be updated.
 * @param signal            [in] The value used to update sig_addr.
 * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:
 *                               SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD
 * @param pe                [in] PE number of the remote PE.
 */
SHMEM_DEVICE void shmem_putmem_signal_nbi(__gm__ void *dst, __gm__ void *src, size_t elem_size,
                                          __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
{
    /* ROCE */
    /* RDMA */
    /* MTE  */
    /* Global State Set */
    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.shmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
    shmem_mte_put_mem_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
    shmem_fence();
    shmemix_signal_op(sig_addr, signal, sig_op, pe);
}

#define SHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI(NAME, TYPE)                                                                 \
    /**                                                                                                               \
     * @brief Asynchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.    \
     *                                                                                                                \
     * @param dst               [in] Pointer on local device of the destination data.                                 \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                  \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                                \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                              \
     * @param signal            [in] The value used to update sig_addr.                                               \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:             \
     *                               SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD                                                \
     * @param pe                [in] PE number of the remote PE.                                                      \
     */                                                                                                               \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_signal_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size,         \
                                                        __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe) \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                         \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                         \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                       \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                         \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                     \
        shmem_mte_put_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,      \
                              copy_event_id);                                                                         \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                                \
        shmem_fence();                                                                                                \
        shmemix_signal_op(sig_addr, signal, sig_op, pe);                                                              \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI);

#define SHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_NBI(NAME, TYPE)                                                          \
    /**                                                                                                               \
     * @brief Asynchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.    \
     *                                                                                                                \
     * @param dst               [in] Pointer on local device of the destination data.                                 \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                  \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                                \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                              \
     * @param signal            [in] The value used to update sig_addr.                                               \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:             \
     *                               SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD                                                \
     * @param pe                [in] PE number of the remote PE.                                                      \
     */                                                                                                               \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_signal_nbi(AscendC::GlobalTensor<TYPE> dst,                              \
                                                        AscendC::GlobalTensor<TYPE> src, size_t elem_size,            \
                                                        __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe) \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                         \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                         \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                       \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                         \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                         \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                          \
        ub_tensor.address_.logicPos = device_state->mte_config.ub_size;                                               \
        shmem_mte_put_mem_nbi(dst, src, ub_tensor, elem_size, pe, copy_event_id);                                     \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                                \
        shmem_fence();                                                                                                \
        shmemix_signal_op(sig_addr, signal, sig_op, pe);                                                              \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_NBI);

#define SHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED_NBI(NAME, TYPE)                                                        \
    /**                                                                                                               \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                      \
     *        on local UB to symmetric address on the specified PE then update sig_addr                               \
     *                                                                                                                \
     * @param dst               [in] Pointer on local device of the destination data.                                 \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                  \
     * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.          \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                              \
     * @param signal            [in] The value used to update sig_addr.                                               \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:             \
     *                               SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD                                                \
     * @param pe                [in] PE number of the remote PE.                                                      \
     */                                                                                                               \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_signal_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                           \
                                                        const non_contiguous_copy_param &copy_params,                 \
                                                        __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe) \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                         \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                         \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                       \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                         \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                     \
        shmem_mte_put_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, copy_params, pe,    \
                              copy_event_id);                                                                         \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                                \
        shmem_fence();                                                                                                \
        shmemix_signal_op(sig_addr, signal, sig_op, pe);                                                              \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED_NBI);

#define SHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED_NBI(NAME, TYPE)                                               \
    /**                                                                                                             \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                    \
     *        on local UB to symmetric address on the specified PE.                                                 \
     *                                                                                                              \
     * @param dst               [in] Pointer on local device of the destination data.                               \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                \
     * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.        \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                            \
     * @param signal            [in] The value used to update sig_addr.                                             \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:           \
     *                               SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD                                              \
     * @param pe                [in] PE number of the remote PE.                                                    \
     */                                                                                                             \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_signal_nbi(                                                            \
        AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,                                           \
        const non_contiguous_copy_param &copy_params, __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe) \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                       \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                       \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                     \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                       \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                       \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                              \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                        \
        ub_tensor.address_.logicPos = device_state->mte_config.ub_size;                                             \
        shmem_mte_put_mem_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                 \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                              \
        shmem_fence();                                                                                              \
        shmemix_signal_op(sig_addr, signal, sig_op, pe);                                                            \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED_NBI);

#define SHMEM_TEST(NAME, TYPE)                                                                                \
    /**                                                                                                       \
     * @brief Synchronous interface. Provide a high-performance way to compare data                           \
     *        on local UB to symmetric address on the specified PE.                                           \
     *                                                                                                        \
     * @param ivar               [in] Pointer on local device of the destination data.                        \
     * @param cmp                [in] Pointer on Symmetric memory of the source data.                         \
     * @param cmp_value          [in] Params to describe how non-contiguous data is organized in src and dst. \
     */                                                                                                       \
    SHMEM_DEVICE int shmem_##NAME##_test(__gm__ TYPE *ivar, int cmp, TYPE cmp_value)                          \
    {                                                                                                         \
        shmem_fence();                                                                                        \
        return shmemi_test(ivar, cmp, cmp_value);                                                             \
    }

SHMEM_TEST_TYPE_FUNC(SHMEM_TEST);
#endif
