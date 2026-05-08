/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGINE_BENCHMARK_KERNEL_H
#define ENGINE_BENCHMARK_KERNEL_H

#include <cstring>
#include "kernel_operator.h"
#include "shmem.h"
#include "utils/prof/shmemi_prof.h"
#include "benchmark_config.h"

using namespace engine_bench;

// ========== MTE PUT Kernel实现 ==========
template <typename T>
__aicore__ inline void mte_bench_put_impl(GM_ADDR dst_gva, GM_ADDR src_gva,
                                           int elements, int peer_pe,
                                           int ub_size_kb, int loop_count, int warmup)
{
    int64_t pe = aclshmem_my_pe();

    __gm__ T *dst_gm = (__gm__ T *)dst_gva;
    __gm__ T *src_gm = (__gm__ T *)src_gva;

    int32_t block_elements = (int32_t)(elements / AscendC::GetBlockNum());
    int32_t current_block_index = AscendC::GetBlockIdx();

    int32_t offset;
    if (block_elements * sizeof(T) < 512) {
        offset = 512 / sizeof(T) * current_block_index;
    } else {
        offset = block_elements * current_block_index;
    }

    int ub_size_bytes = ub_size_kb * 1024;

    // 使用UB地址0作为临时缓冲区
    __ubuf__ T *ub_buf = reinterpret_cast<__ubuf__ T *>(0);

    AscendC::PipeBarrier<PIPE_ALL>();

    // 性能测试循环
    for (int i = 0; i < (warmup + loop_count); ++i) {
        if (i >= warmup) {
            SHMEMI_PROF_START(0);
        }

        // 同步开始
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);

        // MTE Put: 发送数据到peer_pe
        aclshmemx_mte_put_nbi(dst_gm + offset, src_gm + offset,
                              ub_buf, ub_size_bytes, block_elements, peer_pe, 0);

        // 等待完成
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);

        if (i >= warmup) {
            SHMEMI_PROF_END(0);
        }
    }

    aclshmemx_barrier_all_vec();
}

// ========== MTE GET Kernel实现 ==========
template <typename T>
__aicore__ inline void mte_bench_get_impl(GM_ADDR dst_gva, GM_ADDR src_gva,
                                           int elements, int peer_pe,
                                           int ub_size_kb, int loop_count, int warmup)
{
    int64_t pe = aclshmem_my_pe();

    __gm__ T *dst_gm = (__gm__ T *)dst_gva;
    __gm__ T *src_gm = (__gm__ T *)src_gva;

    int32_t block_elements = (int32_t)(elements / AscendC::GetBlockNum());
    int32_t current_block_index = AscendC::GetBlockIdx();

    int32_t offset;
    if (block_elements * sizeof(T) < 512) {
        offset = 512 / sizeof(T) * current_block_index;
    } else {
        offset = block_elements * current_block_index;
    }

    int ub_size_bytes = ub_size_kb * 1024;
    __ubuf__ T *ub_buf = reinterpret_cast<__ubuf__ T *>(0);

    AscendC::PipeBarrier<PIPE_ALL>();

    for (int i = 0; i < (warmup + loop_count); ++i) {
        if (i >= warmup) {
            SHMEMI_PROF_START(0);
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);

        // MTE Get: 从peer_pe拉取数据
        aclshmemx_mte_get_nbi(dst_gm + offset, src_gm + offset,
                              ub_buf, ub_size_bytes, block_elements, peer_pe, 0);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);

        if (i >= warmup) {
            SHMEMI_PROF_END(0);
        }
    }

    aclshmemx_barrier_all_vec();
}

// ========== SDMA PUT Kernel实现 ==========
template <typename T>
__aicore__ inline void sdma_bench_put_impl(GM_ADDR dst_gva, GM_ADDR src_gva,
                                            int elements, int peer_pe,
                                            int ub_size_kb, int loop_count, int warmup)
{
    int64_t pe = aclshmem_my_pe();
    int64_t n_pes = aclshmem_n_pes();

    // SDMA使用字节指针，统一模板参数为uint8_t
    __gm__ uint8_t *dst_gm = (__gm__ uint8_t *)dst_gva;
    __gm__ uint8_t *src_gm = (__gm__ uint8_t *)src_gva;

    // SDMA数据分布（按字节计算）
    const auto cur_block_idx = AscendC::GetBlockIdx();
    const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    uint64_t total_bytes = elements * sizeof(T);
    uint64_t base_per_core = total_bytes / comm_block_dim;
    uint64_t extra_bytes = total_bytes % comm_block_dim;
    uint64_t data_offset = 0;

    if (cur_block_idx < extra_bytes) {
        data_offset = cur_block_idx * (base_per_core + 1);
        base_per_core += 1;
    } else {
        data_offset = extra_bytes * (base_per_core + 1) +
                      (cur_block_idx - extra_bytes) * base_per_core;
    }

    if (base_per_core == 0) {
        return;
    }

    // SDMA需要的UB缓冲区（使用uint8_t字节指针）
    constexpr uint32_t ub_offset = 1024;
    __ubuf__ uint8_t *tmp_buff = reinterpret_cast<__ubuf__ uint8_t *>(ub_offset);

    AscendC::PipeBarrier<PIPE_ALL>();

    for (int i = 0; i < (warmup + loop_count); ++i) {
        if (i >= warmup) {
            SHMEMI_PROF_START(0);
        }

        // SDMA Put: 直接发送数据到目标PE（所有参数使用uint8_t类型）
        aclshmemx_sdma_put_nbi(dst_gm + data_offset, src_gm + data_offset,
                               tmp_buff, ub_size_kb * 1024,
                               base_per_core, peer_pe, EVENT_ID0);

        if (i >= warmup) {
            SHMEMI_PROF_END(0);
        }
    }

    // 等待所有SDMA操作完成
    aclshmemx_sdma_quiet(tmp_buff, ub_size_kb * 1024, EVENT_ID0);

    aclshmemx_barrier_all_vec();
}

// ========== SDMA GET Kernel实现 ==========
template <typename T>
__aicore__ inline void sdma_bench_get_impl(GM_ADDR dst_gva, GM_ADDR src_gva,
                                            int elements, int peer_pe,
                                            int ub_size_kb, int loop_count, int warmup)
{
    int64_t pe = aclshmem_my_pe();
    int64_t n_pes = aclshmem_n_pes();

    // SDMA使用字节指针，统一模板参数为uint8_t
    __gm__ uint8_t *dst_gm = (__gm__ uint8_t *)dst_gva;
    __gm__ uint8_t *src_gm = (__gm__ uint8_t *)src_gva;

    // SDMA数据分布（按字节计算）
    const auto cur_block_idx = AscendC::GetBlockIdx();
    const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    uint64_t total_bytes = elements * sizeof(T);
    uint64_t base_per_core = total_bytes / comm_block_dim;
    uint64_t extra_bytes = total_bytes % comm_block_dim;
    uint64_t data_offset = 0;

    if (cur_block_idx < extra_bytes) {
        data_offset = cur_block_idx * (base_per_core + 1);
        base_per_core += 1;
    } else {
        data_offset = extra_bytes * (base_per_core + 1) +
                      (cur_block_idx - extra_bytes) * base_per_core;
    }

    if (base_per_core == 0) {
        return;
    }

    constexpr uint32_t ub_offset = 1024;
    __ubuf__ uint8_t *tmp_buff = reinterpret_cast<__ubuf__ uint8_t *>(ub_offset);

    AscendC::PipeBarrier<PIPE_ALL>();

    for (int i = 0; i < (warmup + loop_count); ++i) {
        if (i >= warmup) {
            SHMEMI_PROF_START(0);
        }

        // SDMA Get: 从目标PE拉取数据（所有参数使用uint8_t类型）
        aclshmemx_sdma_get_nbi(dst_gm + data_offset, src_gm + data_offset,
                               tmp_buff, ub_size_kb * 1024,
                               base_per_core, peer_pe, EVENT_ID0);

        if (i >= warmup) {
            SHMEMI_PROF_END(0);
        }
    }

    aclshmemx_sdma_quiet(tmp_buff, ub_size_kb * 1024, EVENT_ID0);

    aclshmemx_barrier_all_vec();
}

// ========== Kernel定义宏 ==========
#define DEFINE_MTE_KERNEL_FOR_TYPE(type_name, cpp_type) \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_bench_##type_name##_put( \
    GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int peer_pe, int ub_size_kb, int loop_count, int warmup) \
{ \
    mte_bench_put_impl<cpp_type>(dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup); \
} \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_bench_##type_name##_get( \
    GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int peer_pe, int ub_size_kb, int loop_count, int warmup) \
{ \
    mte_bench_get_impl<cpp_type>(dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup); \
}

#define DEFINE_SDMA_KERNEL_FOR_TYPE(type_name, cpp_type) \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void sdma_bench_##type_name##_put( \
    GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int peer_pe, int ub_size_kb, int loop_count, int warmup) \
{ \
    sdma_bench_put_impl<cpp_type>(dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup); \
} \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void sdma_bench_##type_name##_get( \
    GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int peer_pe, int ub_size_kb, int loop_count, int warmup) \
{ \
    sdma_bench_get_impl<cpp_type>(dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup); \
}

// 定义所有类型的Kernel
DEFINE_MTE_KERNEL_FOR_TYPE(float, float)
DEFINE_MTE_KERNEL_FOR_TYPE(int32, int32_t)
DEFINE_MTE_KERNEL_FOR_TYPE(int64, int64_t)

DEFINE_SDMA_KERNEL_FOR_TYPE(float, float)
DEFINE_SDMA_KERNEL_FOR_TYPE(int32, int32_t)
DEFINE_SDMA_KERNEL_FOR_TYPE(int64, int64_t)

// ========== Host端Kernel启动函数 ==========
#define DISPATCH_KERNEL(engine, type_name, mode, block_dim, stream, dst, src, elems, peer, ub_kb, loops, warmup) \
    engine##_bench_##type_name##_##mode<<<block_dim, nullptr, stream>>>(dst, src, elems, peer, ub_kb, loops, warmup)

extern "C" void launch_mte_bench_kernel(uint32_t block_dim, void *stream,
                                        uint8_t *dst_gva, uint8_t *src_gva,
                                        int elements, int peer_pe, int ub_size_kb,
                                        int loop_count, int warmup, int mode, int dtype)
{
    TestMode test_mode = static_cast<TestMode>(mode);
    DataType data_type = static_cast<DataType>(dtype);

    switch (data_type) {
        case DataType::FLOAT:
            if (test_mode == TestMode::PUT || test_mode == TestMode::BI_PUT) {
                DISPATCH_KERNEL(mte, float, put, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            } else {
                DISPATCH_KERNEL(mte, float, get, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            }
            break;
        case DataType::INT32:
            if (test_mode == TestMode::PUT || test_mode == TestMode::BI_PUT) {
                DISPATCH_KERNEL(mte, int32, put, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            } else {
                DISPATCH_KERNEL(mte, int32, get, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            }
            break;
        case DataType::INT64:
            if (test_mode == TestMode::PUT || test_mode == TestMode::BI_PUT) {
                DISPATCH_KERNEL(mte, int64, put, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            } else {
                DISPATCH_KERNEL(mte, int64, get, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            }
            break;
        default:
            DISPATCH_KERNEL(mte, float, put, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            break;
    }
}

extern "C" void launch_sdma_bench_kernel(uint32_t block_dim, void *stream,
                                         uint8_t *dst_gva, uint8_t *src_gva,
                                         int elements, int peer_pe, int ub_size_kb,
                                         int loop_count, int warmup, int mode, int dtype)
{
    TestMode test_mode = static_cast<TestMode>(mode);
    DataType data_type = static_cast<DataType>(dtype);

    switch (data_type) {
        case DataType::FLOAT:
            if (test_mode == TestMode::PUT || test_mode == TestMode::BI_PUT) {
                DISPATCH_KERNEL(sdma, float, put, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            } else {
                DISPATCH_KERNEL(sdma, float, get, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            }
            break;
        case DataType::INT32:
            if (test_mode == TestMode::PUT || test_mode == TestMode::BI_PUT) {
                DISPATCH_KERNEL(sdma, int32, put, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            } else {
                DISPATCH_KERNEL(sdma, int32, get, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            }
            break;
        case DataType::INT64:
            if (test_mode == TestMode::PUT || test_mode == TestMode::BI_PUT) {
                DISPATCH_KERNEL(sdma, int64, put, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            } else {
                DISPATCH_KERNEL(sdma, int64, get, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            }
            break;
        default:
            DISPATCH_KERNEL(sdma, float, put, block_dim, stream, dst_gva, src_gva, elements, peer_pe, ub_size_kb, loop_count, warmup);
            break;
    }
}

#endif // ENGINE_BENCHMARK_KERNEL_H