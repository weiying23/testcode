/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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
#include "kv_shuffle_kernel.h"

#undef inline
#include "fp16_t.h"
#include "bfloat16.h"
#define inline inline attribute((always_inline))
using namespace AscendC;
using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;
namespace ShmemKernel {

constexpr int64_t SYNC_FLAG_INTERVAL = 16;

extern "C" __global__ __aicore__ void ShmemKVShuffle(
    uint64_t fftsAddr,
    GM_ADDR k_cache,
    GM_ADDR v_cache,
    GM_ADDR global_shuffle_table,
    GM_ADDR src_block_table,
    GM_ADDR dst_block_table,
    GM_ADDR sync_ptr,
    int64_t block_num,
    int64_t kv_head_num, int64_t page_size, int64_t head_dim, int32_t count)
{
#ifdef __DAV_C220_VEC__
    shmemx_set_ffts_config(fftsAddr);

    int64_t n_pes = shmem_n_pes();
    int64_t local_rank = shmem_my_pe();
    int64_t pair_rank = *((__gm__ int64_t*)global_shuffle_table + 2 * local_rank);
    int64_t operation = *((__gm__ int64_t*)global_shuffle_table + 2 * local_rank + 1);
    int64_t pair_operation = *((__gm__ int64_t*)global_shuffle_table + 2 * pair_rank + 1);
    
    __gm__ int32_t* gva_sync_gm = (__gm__ int32_t *)sync_ptr;

    const int64_t aiv_num = AscendC::GetBlockNum();
    const int64_t aiv_idx = AscendC::GetBlockIdx();
    
    const int64_t local_rank_offset = (local_rank * aiv_num + aiv_idx) * SYNC_FLAG_INTERVAL;
    const int64_t pair_rank_offset = (pair_rank * aiv_num + aiv_idx) * SYNC_FLAG_INTERVAL;
    
    uint64_t copy_ping_k_ub = 0;
    uint64_t copy_pong_k_ub = 32 * 1024;
    uint64_t copy_ping_v_ub = 64 * 1024;
    uint64_t copy_pong_v_ub = 96 * 1024;
    uint32_t copy_ub_size = 32 * 1024;

    // 0 means send
    if (operation == 0) {
        shmem_signal_wait_until(gva_sync_gm + pair_rank_offset, SHMEM_CMP_EQ, count);
        // block data num
        int64_t block_size = kv_head_num * page_size * head_dim;
        // core average move
        int64_t core_size = block_size / (aiv_num / 2);

        int ping_pong_flag = 0;
        for (int block_idx = 0; block_idx < block_num; ++block_idx) {
            // Get dst&&src Block ID
            int src_block_id = *((__gm__ int64_t*)src_block_table + block_idx);
            int dst_block_id = *((__gm__ int64_t*)dst_block_table + block_idx);

            // Cal dst&&src Data Offset
            int64_t src_offset = src_block_id * block_size + (aiv_idx % 8) * core_size;
            int64_t dst_offset = dst_block_id * block_size + (aiv_idx % 8) * core_size;

            // Ping Pong Prepare
            uint64_t k_cache_copy_ub = ping_pong_flag == 0 ? copy_ping_k_ub : copy_pong_k_ub;
            uint64_t v_cache_copy_ub = ping_pong_flag == 0 ? copy_ping_v_ub : copy_pong_v_ub;
            AscendC::TEventID copy_event_k = ping_pong_flag == 0 ? EVENT_ID0 : EVENT_ID1;
            AscendC::TEventID copy_event_v = ping_pong_flag == 0 ? EVENT_ID2 : EVENT_ID3;

            __gm__ int8_t* local_ptr;
            __gm__ int8_t* remote_ptr;
            // K Cache Remote Copy
            if (aiv_idx < 8) {
                local_ptr = (__gm__ int8_t*)k_cache + src_offset;
                remote_ptr = (__gm__ int8_t*)k_cache + dst_offset;
                shmem_mte_put_mem_nbi(
                    remote_ptr,
                    local_ptr,
                    reinterpret_cast<__ubuf__ int8_t*>(k_cache_copy_ub),
                    copy_ub_size,
                    core_size,
                    pair_rank,
                    copy_event_k);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copy_event_k);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copy_event_k);
            }
            // V Cache Remote Copy
            if (aiv_idx >= 8) {
                local_ptr = (__gm__ int8_t*)v_cache + src_offset;
                remote_ptr = (__gm__ int8_t*)v_cache + dst_offset;
                shmem_mte_put_mem_nbi(
                    remote_ptr,
                    local_ptr,
                    reinterpret_cast<__ubuf__ int8_t*>(v_cache_copy_ub),
                    copy_ub_size,
                    core_size,
                    pair_rank,
                    copy_event_v);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copy_event_v);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copy_event_v);
            }

            ping_pong_flag = 1 - ping_pong_flag;
        }
        shmemx_signal_op(gva_sync_gm + local_rank_offset, count, SHMEM_SIGNAL_SET, pair_rank);
    } else if (pair_operation == 0) {
        shmemx_signal_op(gva_sync_gm + local_rank_offset, count, SHMEM_SIGNAL_SET, pair_rank);
        shmem_signal_wait_until(gva_sync_gm + pair_rank_offset, SHMEM_CMP_EQ, count);
    }
#endif
}

void kv_shuffle(
    uint32_t block_dim, void* stream, uint64_t fftsAddr,
    uint8_t* k_cache,
    uint8_t* v_cache,
    uint8_t* global_shuffle_table,
    uint8_t* src_block_table,
    uint8_t* dst_block_table,
    uint8_t* sync_ptr,
    int64_t block_num,
    int64_t kv_head_num, int64_t page_size, int64_t head_dim, int32_t sync_count)
{
    ShmemKVShuffle<<<block_dim, nullptr, stream>>>(
        fftsAddr,
        k_cache,
        v_cache,
        global_shuffle_table,
        src_block_table,
        dst_block_table,
        sync_ptr,
        block_num,
        kv_head_num, page_size, head_dim, sync_count);
}
}