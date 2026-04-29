/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_LOW_LEVEL_RMA_H
#define SHMEM_DEVICE_LOW_LEVEL_RMA_H

#include "kernel_operator.h"
#include "internal/device/shmemi_device_common.h"
#include "device/shmem_device_team.h"
#include "shmem_device_low_level_roce.h"

constexpr uint64_t SHMEM_INTERNAL_UB_BUF_START_ADDR = 188 * 1024;
constexpr uint32_t UB_ALIGN_SIZE = 32;

/**
 * @brief Translate an local symmetric address to remote symmetric address on the specified PE.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] The number of the remote PE.
 * @return A remote symmetric address on the specified PE that can be accessed using memory loads and stores.
 */
SHMEM_DEVICE __gm__ void *shmem_ptr(__gm__ void *ptr, int pe)
{
    // Get Global State
    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();

    // Back to root address
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(device_state->heap_base);

    // Address translate
    uint64_t remote_ptr = reinterpret_cast<uint64_t>(device_state->p2p_heap_device_base[pe]) + offset;

    return reinterpret_cast<__gm__ void *>(remote_ptr);
}

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified
 *        PE to address on the local device.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                        uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(src, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    // block_size: dataMove Unit
    uint64_t block_size = ub_size / sizeof(T) * sizeof(T);
    uint64_t remain = (elem_size * sizeof(T)) % block_size;

    uint64_t repeat_times = (elem_size * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, remote_ptr + i * repeat_elem, block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst + i * repeat_elem, buf, block_size);
        if (i != loop_times - 1) {  // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, remote_ptr + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst + repeat_times * repeat_elem, buf, remain);
    }
}

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified
 * PE to address on the local device.
 * WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations
 * to the same PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB, available space larger than 64 Bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
SHMEM_DEVICE void shmem_roce_get_mem_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe)
{
    auto ptr = shmem_ptr(src, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    shmemi_roce_read((__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T),
        ub_tensor_64, ub_tensor_32);
}


/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on symmetric memory from the specified PE to address on the local device.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                        const non_contiguous_copy_param &copy_params, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(src, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    AscendC::GlobalTensor<T> src_tensor;
    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> dst_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(remote_ptr));
    dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst));

    uint64_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint64_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    smem_shm_copy_gm2ub(ub_tensor, src_tensor, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    smem_shm_copy_ub2gm(dst_tensor, ub_tensor, data_copy_params_ub2gm);
}

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified
 *        PE to address on the local PE.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    // block_size: dataMove Unit
    uint64_t block_size = buf.GetSize() * sizeof(T);
    uint64_t remain = (elem_size * sizeof(T)) % block_size;

    uint64_t repeat_times = (elem_size * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, remote_buff[i * repeat_elem], block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst[i * repeat_elem], buf, block_size);
        if (i != loop_times - 1) {  // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, remote_buff[repeat_times * repeat_elem], remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst[repeat_times * repeat_elem], buf, remain);
    }
}

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified
 * PE to address on the local PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations
 * to the same PE are not supported.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param buf               [in] LocalTensor on local UB, available space larger than 64 Bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
SHMEM_DEVICE void shmem_roce_get_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
    AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe)
{
    auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    shmemi_roce_read((__gm__ uint8_t*)dst.GetPhyAddr(), (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T),
        ub_tensor_64, ub_tensor_32);
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on symmetric memory from the specified PE to address on the local device.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        AscendC::LocalTensor<T> buf, const non_contiguous_copy_param &copy_params,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    uint64_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint64_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    smem_shm_copy_gm2ub(buf, remote_buff, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    smem_shm_copy_ub2gm(dst, buf, data_copy_params_ub2gm);
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
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                        uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(dst, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    // block_size: dataMove Unit
    uint64_t block_size = ub_size / sizeof(T) * sizeof(T);
    uint64_t remain = (elem_size * sizeof(T)) % block_size;

    uint64_t repeat_times = (elem_size * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, src + i * repeat_elem, block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remote_ptr + i * repeat_elem, buf, block_size);
        if (i != loop_times - 1) {  // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, src + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remote_ptr + repeat_times * repeat_elem, buf, remain);
    }
}

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB, available space larger than 64 Bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
SHMEM_DEVICE void shmem_roce_put_mem_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe)
{
    auto ptr = shmem_ptr(dst, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    shmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(T),
        ub_tensor_64, ub_tensor_32);
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                        const non_contiguous_copy_param &copy_params, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(dst, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    AscendC::GlobalTensor<T> src_tensor;
    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> dst_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src));
    dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(remote_ptr));

    uint64_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint64_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    smem_shm_copy_gm2ub(ub_tensor, src_tensor, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    smem_shm_copy_ub2gm(dst_tensor, ub_tensor, data_copy_params_ub2gm);
}

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    // block_size: dataMove Unit
    uint64_t block_size = buf.GetSize() * sizeof(T);
    uint64_t remain = (elem_size * sizeof(T)) % block_size;

    uint64_t repeat_times = (elem_size * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, src[i * repeat_elem], block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remote_buff[i * repeat_elem], buf, block_size);
        if (i != loop_times - 1) {  // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, src[repeat_times * repeat_elem], remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remote_buff[repeat_times * repeat_elem], buf, remain);
    }
}

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        PE are not supported.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] Pointer on local UB, available space larger than 64 Bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
SHMEM_DEVICE void shmem_roce_put_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
    AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    shmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)(src.GetPhyAddr()), pe, 0,
        elem_size * sizeof(T), ub_tensor_64, ub_tensor_32);
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        AscendC::LocalTensor<T> buf, const non_contiguous_copy_param &copy_params,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    uint64_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint64_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    smem_shm_copy_gm2ub(buf, src, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    smem_shm_copy_ub2gm(remote_buff, buf, data_copy_params_ub2gm);
}

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE
 *        to address on the local UB.
 *
 * @param dst               [in] Pointer on local UB of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(__ubuf__ T *dst, __gm__ T *src, uint32_t elem_size, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(src, pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(dst);
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit)
        return;

    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    smem_shm_copy_gm2ub(dst, remote_ptr, elem_size * sizeof(T));
}

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *        address on the local UB.
 *
 * @param dst               [in] LocalTensor on local UB of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src, uint32_t elem_size,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    smem_shm_copy_gm2ub(dst, remote_buff, elem_size * sizeof(T));
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on symmetric memory from the specified PE to address on the local UB.
 *
 * @param dst               [in] Pointer on local UB of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(__ubuf__ T *dst, __gm__ T *src, const non_contiguous_copy_param &copy_params,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(src, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(dst);
    uint64_t process_num = (copy_params.repeat - 1) * copy_params.dst_ld + copy_params.length;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> src_tensor;
    AscendC::LocalTensor<T> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(dst);
    src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(remote_ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (copy_params.dst_ld - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    smem_shm_copy_gm2ub(ub_tensor, src_tensor, data_copy_params_gm2ub);
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on symmetric memory from the specified PE to address on the local UB.
 *
 * @param dst               [in] LocalTensor on local UB of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        const non_contiguous_copy_param &copy_params, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
    uint64_t process_num = (copy_params.repeat - 1) * copy_params.dst_ld + copy_params.length;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (copy_params.dst_ld - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    smem_shm_copy_gm2ub(dst, remote_buff, data_copy_params_gm2ub);
}

/**
 * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local UB of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(__gm__ T *dst, __ubuf__ T *src, uint32_t elem_size, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(dst, pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src);
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit)
        return;

    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    smem_shm_copy_ub2gm(remote_ptr, src, elem_size * sizeof(T));
}

/**
 * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] LocalTensor on local UB of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src, uint32_t elem_size,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src.GetPhyAddr());
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    smem_shm_copy_ub2gm(remote_buff, src, elem_size * sizeof(T));
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local UB of the source data.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(__gm__ T *dst, __ubuf__ T *src, const non_contiguous_copy_param &copy_params,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(dst, pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src);
    uint64_t process_num = (copy_params.repeat - 1) * copy_params.src_ld + copy_params.length;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> dst_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(src);
    dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(remote_ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    smem_shm_copy_ub2gm(dst_tensor, ub_tensor, data_copy_params_ub2gm);
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] LocalTensor on local UB of the source data.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src,
                                        const non_contiguous_copy_param &copy_params, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src.GetPhyAddr());
    uint64_t process_num = (copy_params.repeat - 1) * copy_params.src_ld + copy_params.length;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    smem_shm_copy_ub2gm(remote_buff, src, data_copy_params_ub2gm);
}

#endif