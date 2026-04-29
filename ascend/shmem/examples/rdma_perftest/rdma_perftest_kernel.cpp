/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem_api.h"

constexpr uint32_t MAGIC_VAL = 10;
constexpr uint32_t WARMUP_MSG_LEN = 32;

extern "C" __global__ __aicore__ void rdma_highlevel_put_pingpong_latency(uint64_t cfg, GM_ADDR gva, int msg_len) {
    shmemx_set_ffts_config(cfg);
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE);
    AscendC::LocalTensor<uint32_t> ubLocalRead = buf.GetWithOffset<uint32_t>(UB_ALIGN_SIZE / sizeof(uint32_t), 0);

    int64_t rank = smem_shm_get_global_rank();
    int64_t rank_size = smem_shm_get_global_rank_size();
    uint32_t peer;

    // Warm up
    GM_ADDR warm_addr = gva + rank_size * msg_len + WARMUP_MSG_LEN * (rank + 1);
    if (rank == 0) {
        peer = 1;
        shmem_put_uint8_mem_nbi(warm_addr, warm_addr, WARMUP_MSG_LEN, peer);
        while (*(__gm__ uint32_t*)(gva + rank_size * msg_len + WARMUP_MSG_LEN * (peer + 1)) != peer + MAGIC_VAL) {
            cacheWriteThrough(gva + rank_size * msg_len + WARMUP_MSG_LEN * (peer + 1), sizeof(uint32_t));
            AscendC::GetSystemCycle();
        }
    } else {
        peer = 0;
        while (*(__gm__ uint32_t*)(gva + rank_size * msg_len + WARMUP_MSG_LEN * (peer + 1)) != peer + MAGIC_VAL) {
            cacheWriteThrough(gva + rank_size * msg_len + WARMUP_MSG_LEN * (peer + 1), sizeof(uint32_t));
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        shmem_put_uint8_mem_nbi(warm_addr, warm_addr, WARMUP_MSG_LEN, peer);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    // Actual test
    GM_ADDR src_addr = gva + rank * msg_len;
    if (rank == 0) {
        peer = 1;
        int64_t start = AscendC::GetSystemCycle();
        shmem_put_uint8_mem_nbi(src_addr, src_addr, msg_len, peer);
        while (*(__gm__ uint32_t*)(gva + msg_len * 2 - 8) != peer + MAGIC_VAL) {
            cacheWriteThrough(gva + msg_len * 2 - 8, 8);
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end = AscendC::GetSystemCycle();
        *(__gm__ int64_t*)(gva + msg_len * 2) = end - start;
    } else {
        peer = 0;
        while (*(__gm__ uint32_t*)(gva + msg_len * 1 - 8) != peer + MAGIC_VAL) {
            cacheWriteThrough(gva + msg_len * 1 - 8, 8);
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        shmem_put_uint8_mem_nbi(src_addr, src_addr, msg_len, peer);
    }
}

void rdma_highlevel_put_pingpong_latency_do(uint32_t block_dim, void* stream, uint64_t cfg, uint8_t* gva, int len)
{
    rdma_highlevel_put_pingpong_latency<<<1, nullptr, stream>>>(cfg, gva, len);
}

extern "C" __global__ __aicore__ void rdma_postsend_cost(uint64_t fftsConfig, GM_ADDR gva, int message_length) {
    shmemx_set_ffts_config(fftsConfig);
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint32_t> ubLocal32 = buf.GetWithOffset<uint32_t>(UB_ALIGN_SIZE / sizeof(uint32_t), 0);
    AscendC::LocalTensor<uint64_t> ubLocal64 =
        buf.GetWithOffset<uint64_t>(UB_ALIGN_SIZE / sizeof(uint64_t), UB_ALIGN_SIZE);

    int64_t rank = smem_shm_get_global_rank();
    int64_t rank_size = smem_shm_get_global_rank_size();
    uint32_t peer;

    // Actual test
    GM_ADDR src_addr = gva + rank * message_length;

    if (rank == 0) {
        peer = 1;
        GM_ADDR dest_addr = (GM_ADDR)(shmem_ptr(src_addr, peer));
        int64_t start = AscendC::GetSystemCycle();
        for (uint32_t i = 0; i < 500; i++) {
            shmemi_roce_write(dest_addr, src_addr, peer, 0, message_length, ubLocal64, ubLocal32);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end = AscendC::GetSystemCycle();
        *(__gm__ int64_t*)(gva + message_length * 2) = end - start;
    }
}

void rdma_postsend_cost_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length)
{
    rdma_postsend_cost<<<1, nullptr, stream>>>(fftsConfig, gva, message_length);
}

extern "C" __global__ __aicore__ void rdma_highlevel_put_bw(uint64_t fftsConfig, GM_ADDR gva, int message_length) {
    shmemx_set_ffts_config(fftsConfig);
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint32_t> ubLocal32 = buf.GetWithOffset<uint32_t>(UB_ALIGN_SIZE / sizeof(uint32_t), 0);
    AscendC::LocalTensor<uint64_t> ubLocal64 =
        buf.GetWithOffset<uint64_t>(UB_ALIGN_SIZE / sizeof(uint64_t), UB_ALIGN_SIZE);

    int64_t rank = smem_shm_get_global_rank();
    int64_t rank_size = smem_shm_get_global_rank_size();
    uint32_t peer;

    // Actual test
    GM_ADDR src_addr = gva + rank * message_length;
    if (rank == 0) {
        peer = 1;
        int64_t start = AscendC::GetSystemCycle();
        for (int i = 0; i < 10000; i++) {
            shmem_put_uint8_mem_nbi(src_addr, src_addr, message_length, peer);
        }
        shmemi_roce_quiet(peer, 0, ubLocal64, ubLocal32);
        shmem_put_uint8_mem_nbi(gva + rank_size * message_length + 8, src_addr, sizeof(uint32_t), peer);
        while (*(__gm__ uint32_t*)(gva + message_length * rank_size + 16) != peer + MAGIC_VAL) {
            cacheWriteThrough(gva + message_length * rank_size + 16, 8);
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end = AscendC::GetSystemCycle();
        *(__gm__ int64_t*)(gva + message_length * rank_size) = end - start;
    } else {
        peer = 0;
        while (*(__gm__ uint32_t*)(gva + rank_size * message_length + 8) != peer + MAGIC_VAL) {
            cacheWriteThrough(gva + rank_size * message_length + 8, 8);
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        shmem_put_uint8_mem_nbi(gva + message_length * rank_size + 16, src_addr, sizeof(uint32_t), peer);
    }
}

void rdma_highlevel_put_bw_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length)
{
    rdma_highlevel_put_bw<<<1, nullptr, stream>>>(fftsConfig, gva, message_length);
}

extern "C" __global__ __aicore__ void rdma_mte_put_bw(uint64_t cfg, GM_ADDR gva, int message_length, int64_t iter) {
    shmemx_set_ffts_config(cfg);
    AscendC::LocalTensor<uint32_t> ubLocal32;
    ubLocal32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ubLocal32.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR);
    ubLocal32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ubLocal64;
    ubLocal64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ubLocal64.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR + UB_ALIGN_SIZE);
    ubLocal64.address_.dataLen = UB_ALIGN_SIZE;

    int64_t rank = smem_shm_get_global_rank();
    int64_t rank_size = smem_shm_get_global_rank_size();
    uint32_t peer;

    // Core 0, RDMA
    if (AscendC::GetBlockIdx() == 0) {
        GM_ADDR src_addr = gva + rank * message_length;
        if (rank == 0) {
            peer = 1;
            int64_t start = AscendC::GetSystemCycle();
            for (int i = 0; i < 10000; i++) {
                shmemi_roce_write((GM_ADDR)shmem_ptr(src_addr, peer), src_addr, peer, 0,
                    message_length, ubLocal64, ubLocal32);
            }
            shmemi_roce_quiet(peer, 0, ubLocal64, ubLocal32);
            shmemi_roce_write((GM_ADDR)shmem_ptr(gva + rank_size * message_length * 2 + 8, peer),
                src_addr, peer, 0, sizeof(int64_t), ubLocal64, ubLocal32);
            while (*(__gm__ int64_t*)(gva + message_length * rank_size * 2 + 16) != peer + MAGIC_VAL + iter) {
                cacheWriteThrough(gva + message_length * rank_size * 2 + 16, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            int64_t end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(gva + message_length * rank_size * 2) = end - start;
        } else {
            peer = 0;
            while (*(__gm__ int64_t*)(gva + rank_size * message_length * 2 + 8) != peer + MAGIC_VAL + iter) {
                cacheWriteThrough(gva + rank_size * message_length * 2 + 8, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            shmemi_roce_write((GM_ADDR)shmem_ptr(gva + rank_size * message_length * 2 + 16, peer),
                src_addr, peer, 0, sizeof(int64_t), ubLocal64, ubLocal32);
        }
    } else { // core 1, MTE
        GM_ADDR src_addr = gva + (rank + rank_size) * message_length;
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
        /* CopyUB Config Set */
        uint64_t copy_ub = device_state->mte_config.shmem_ub;
        uint32_t copy_ub_size = device_state->mte_config.ub_size;
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
        if (rank == 0) {
            peer = 1;
            int64_t start = AscendC::GetSystemCycle();
            for (int i = 0; i < 10000; i++) {
                shmem_mte_put_mem_nbi(src_addr, src_addr, reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                    copy_ub_size, message_length, peer, copy_event_id);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            shmem_mte_put_mem_nbi(gva + rank_size * message_length * 2 + 24, src_addr,
                reinterpret_cast<__ubuf__ uint8_t*>(copy_ub), copy_ub_size, sizeof(uint32_t), peer, copy_event_id);
            while (*(__gm__ uint32_t*)(gva + message_length * rank_size * 2 + 32) != peer + MAGIC_VAL + iter) {
                cacheWriteThrough(gva + message_length * rank_size * 2 + 32, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            int64_t end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(gva + message_length * rank_size * 2 + 48) = end - start;
        } else {
            peer = 0;
            while (*(__gm__ uint32_t*)(gva + rank_size * message_length * 2 + 24) != peer + MAGIC_VAL + iter) {
                cacheWriteThrough(gva + rank_size * message_length * 2 + 24, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            shmem_mte_put_mem_nbi(gva + rank_size * message_length * 2 + 32, src_addr,
                reinterpret_cast<__ubuf__ uint8_t*>(copy_ub), copy_ub_size, sizeof(uint32_t), peer, copy_event_id);
        }
    }
}

void rdma_mte_put_bw_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int len, int64_t iter)
{
    rdma_mte_put_bw<<<2, nullptr, stream>>>(fftsConfig, gva, len, iter);
}