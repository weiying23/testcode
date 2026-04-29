/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_H
#define ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_H

#include "device/shmem_def.h"
#include "gm2gm/engine/shmemi_device_rdma.h"

/*
 *  =====================================================================================================
 *  SHMEM RDMA Architecture Structure —— Base Header
 *  =====================================================================================================
 *  We provide two categories of APIs:
 *  1. Fine-grained primitives (ibverbs-style)
 *      - aclshmemi_roce_fill_wqe
 *      - aclshmemi_roce_ring_sq_doorbell
 *      - aclshmemi_roce_post_send
 *      - aclshmemi_roce_ring_cq_doorbell
 *      - aclshmemi_roce_poll_cq
 * 
 *  2. Coarse-grained operations
 *      - aclshmemi_roce_write
 *      - aclshmemi_roce_read
 * =====================================================================================================
 *  Internal call chain (base -> backend-specialized):
 *      aclshmemi_roce_post_send<B, OP_CODE>()                  // compile-time dispatched entry
 *          -> aclshmemi_roce_post_send_read_write()            // shared post-send helper for READ/WRITE
 *              -> aclshmemi_roce_fill_wqe<B, OP_CODE>()        // compile-time dispatched fill_wqe
 *                  -> aclshmemi_rdma_fill_wqe_write_read()     // shared fill_wqe helper for READ/WRITE
 *          -> aclshmemi_roce_ring_sq_doorbell<B>()             // backend-dispatched doorbell 
 * =====================================================================================================
 */

/**
 * @brief Fill WQE for RDMA operation
 *
 * @tparam B                     RDMA Backend type
 * @tparam OP_CODE               rdma opcode in aclshmemi_rdma_opcode_t enum class
 * @param wr                     [in] Work request describing the RDMA operation, details in aclshmemi_rdma_send_wr
 * @param sq_context             [in] Current QP's Send queue Context
 * @param wqe_addr               [in] WQE address to fill
 * @param cur_head               [in] Current head of WQE queue
 * @return uint32_t              [out] total size of WQE in bytes
 */
template<aclshmemi_rdma_backend_t B, aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_fill_wqe(aclshmemi_rdma_send_wr &wr,
                                                 __gm__ aclshmemi_rdma_sq_ctx*& sq_context,
                                                 __gm__ uint8_t* wqe_addr, uint32_t cur_head);

/**
 * @brief Ring SQ DB for RDMA operation
 *
 * @tparam B                     RDMA Backend type
 * @param sq_context             [in] Current QP's Send queue Context
 * @param cur_head               [in] Current head of SQ WQE
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template<aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_sq_doorbell(__gm__ aclshmemi_rdma_sq_ctx*& sq_context, uint32_t cur_head,
                                                     AscendC::LocalTensor<uint64_t>& ub_local64,
                                                     AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/**
 * @brief AIV direct RDMA helper function for post send, prepare WQE and ring doorbell.
 * Directly calls the underlying implementation through template parameters
 * without using switch, resulting in higher performance.
 *
 * @tparam B                     RDMA Backend type
 * @tparam OP_CODE               rdma opcode in aclshmemi_rdma_opcode_t enum class
 * @param wr                     [in] Work request describing the RDMA operation, details in aclshmemi_rdma_send_wr
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template<aclshmemi_rdma_backend_t B, aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send(aclshmemi_rdma_send_wr &wr,
                                              uint32_t pe, uint32_t qp_idx,
                                              AscendC::LocalTensor<uint64_t>& ub_local64,
                                              AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/**
 * @brief Ring CQ DB for RDMA operation
 *
 * @tparam B                     RDMA Backend type
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param cur_tail               [in] Current tail of CQ WQE
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template<aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_cq_doorbell(uint32_t pe, uint32_t qp_idx, uint32_t cur_tail, 
                                                     AscendC::LocalTensor<uint64_t>& ub_local64,
                                                     AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/**
 * @brief RDMA Poll Completion Queue (CQ) function. Return status: 0 means success, non-zero means error.
 *
 * @tparam B                     RDMA Backend type
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param target_idx             [in] expect completion queue consumer index after polling
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template<aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_poll_cq(uint32_t pe, uint32_t qp_idx, uint32_t target_idx,
                                                AscendC::LocalTensor<uint64_t>& ub_local64,
                                                AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/**
 * @brief Asynchronous RDMA Write function.
 *
 * @tparam B                     RDMA Backend type
 * @param dst                    [in] destination address in remote HBM
 * @param src                    [in] source address in local HBM
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param message_len            [in] message length in Bytes
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template<typename T, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_write(__gm__ T* dst, __gm__ T* src, 
                                          uint32_t pe, uint32_t qp_idx, uint64_t message_len,
                                          AscendC::LocalTensor<uint64_t>& ub_local64,
                                          AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    aclshmemi_rdma_send_wr wr = {};
    wr.remote_addr   = (__gm__ uint8_t*)dst;
    wr.local_addr    = (__gm__ uint8_t*)src;
    wr.message_len   = message_len;

    aclshmemi_roce_post_send<B, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(wr, pe, qp_idx, ub_local64, ub_local32, sync_id);
}

/**
 * @brief Asynchronous RDMA READ function.
 *
 * @tparam B                     RDMA Backend type
 * @param dst                    [in] destination address in local HBM
 * @param src                    [in] source address in remote HBM
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param message_len            [in] message length in Bytes
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template<typename T, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_read(__gm__ T* dst, __gm__ T* src,
                                         uint32_t pe, uint32_t qp_idx, uint64_t message_len,
                                         AscendC::LocalTensor<uint64_t>& ub_local64,
                                         AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    // Attention: Read need src to be wr.remote_addr.
    aclshmemi_rdma_send_wr wr = {};
    wr.remote_addr   = (__gm__ uint8_t*)src;
    wr.local_addr    = (__gm__ uint8_t*)dst;
    wr.message_len   = message_len;

    aclshmemi_roce_post_send<B, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(wr, pe, qp_idx, ub_local64, ub_local32, sync_id);
}

#endif  // ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_H