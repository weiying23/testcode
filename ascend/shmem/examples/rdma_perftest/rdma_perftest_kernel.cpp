/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

constexpr uint32_t MAGIC_VAL = 10;
constexpr uint32_t WARMUP_MESSAGE_LENGTH = 32;

/**
 * @brief RDMA High-level Put Pingpong延迟测试kernel
 *
 * WHY测试Pingpong延迟：
 * - Pingpong测试测量RDMA通信的往返延迟
 * - 延迟是分布式训练同步操作的关键性能指标
 * - 用于评估跨节点通信的时间开销
 *
 * Pingpong流程：
 * 1. PE 0发送数据到PE 1
 * 2. PE 1接收后发送回PE 0
 * 3. PE 0测量往返时间
 *
 * @param fftsConfig FFTS配置地址（硬件同步配置）
 * @param gva 对称内存指针（GVA格式）
 * @param message_length 消息长度（字节）
 */
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_highlevel_put_pingpong_latency(uint64_t fftsConfig, GM_ADDR gva, int message_length) {
    util_set_ffts_config(fftsConfig);

    // WHY检查GetSubBlockIdx：仅主核执行测试，副核退出
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;

    // WHY需要UB缓冲区：
    // - 用于读取同步标志值
    // - RDMA需要轮询检查数据到达状态
    pipe.InitBuffer(buf, UB_ALIGN_SIZE);
    AscendC::LocalTensor<uint32_t> ubLocalRead = buf.GetWithOffset<uint32_t>(UB_ALIGN_SIZE / sizeof(uint32_t), 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    uint32_t peer;

    // ========== Warmup阶段 ==========
    // WHY需要Warmup：
    // - 预热通信链路，消除冷启动影响
    // - 确保RDMA连接已建立并稳定
    // - 避免首次通信的额外开销影响测试结果
    GM_ADDR warm_addr = gva + rank_size * message_length + WARMUP_MESSAGE_LENGTH * (rank + 1);
    if (rank == 0) {
        peer = 1;
        // aclshmem_uint8_put_nbi: RDMA高层Put接口
        // WHY使用highlevel接口：封装了RDMA底层细节，更易使用
        // 参数：目标地址、源地址、长度、目标PE
        aclshmem_uint8_put_nbi(warm_addr, warm_addr, WARMUP_MESSAGE_LENGTH, peer);

        // WHY轮询等待：RDMA是非阻塞操作，需要轮询检查数据到达
        // dcci_cachelines: 刷新缓存，确保读取最新数据
        // MAGIC_VAL: 魔法值，用于判断数据是否正确接收
        while (*(__gm__ uint32_t*)(gva + rank_size * message_length + WARMUP_MESSAGE_LENGTH * (peer + 1)) != peer + MAGIC_VAL) {
            dcci_cachelines(gva + rank_size * message_length + WARMUP_MESSAGE_LENGTH * (peer + 1), sizeof(uint32_t));
            AscendC::GetSystemCycle();  // WHY调用GetSystemCycle：触发缓存刷新
        }
    } else {
        peer = 0;
        // PE 1等待PE 0的数据到达
        while (*(__gm__ uint32_t*)(gva + rank_size * message_length + WARMUP_MESSAGE_LENGTH * (peer + 1)) != peer + MAGIC_VAL) {
            dcci_cachelines(gva + rank_size * message_length + WARMUP_MESSAGE_LENGTH * (peer + 1), sizeof(uint32_t));
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        // PE 1回复数据给PE 0
        aclshmem_uint8_put_nbi(warm_addr, warm_addr, WARMUP_MESSAGE_LENGTH, peer);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    // ========== 实际测试阶段 ==========
    GM_ADDR src_addr = gva + rank * message_length;
    if (rank == 0) {
        peer = 1;
        // WHY记录start时间：测量发送到接收完成的往返时间
        int64_t start = AscendC::GetSystemCycle();
        aclshmem_uint8_put_nbi(src_addr, src_addr, message_length, peer);

        // WHY轮询等待回复：Pingpong需要等待对端返回数据
        while (*(__gm__ uint32_t*)(gva + message_length * 2 - 8) != peer + MAGIC_VAL) {
            dcci_cachelines(gva + message_length * 2 - 8, 8);
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // WHY记录end时间：计算往返延迟
        int64_t end = AscendC::GetSystemCycle();
        // WHY存储到gva：Host端需要读取延迟结果
        *(__gm__ int64_t*)(gva + message_length * 2) = end - start;
    } else {
        peer = 0;
        // PE 1等待PE 0的数据
        while (*(__gm__ uint32_t*)(gva + message_length * 1 - 8) != peer + MAGIC_VAL) {
            dcci_cachelines(gva + message_length * 1 - 8, 8);
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        // PE 1回复数据给PE 0
        aclshmem_uint8_put_nbi(src_addr, src_addr, message_length, peer);
    }
}

void rdma_highlevel_put_pingpong_latency_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length) {
    rdma_highlevel_put_pingpong_latency<<<1, nullptr, stream>>>(fftsConfig, gva, message_length);
}

/**
 * @brief RDMA Postsend开销测试kernel
 *
 * WHY测试Postsend开销：
 * - Postsend开销衡量RDMA发送操作本身的时间
 * - 用于评估单次RDMA Put的硬件开销
 * - 不包括数据传输时间，仅测量任务下发开销
 *
 * @param fftsConfig FFTS配置地址
 * @param gva 对称内存指针
 * @param message_length 消息长度
 */
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_postsend_cost(uint64_t fftsConfig, GM_ADDR gva, int message_length) {
    util_set_ffts_config(fftsConfig);
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;

    // WHY需要两个UB缓冲区：
    // - ubLocal32: 用于32位数据操作
    // - ubLocal64: 用于64位数据操作（SQE地址等）
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint32_t> ubLocal32 = buf.GetWithOffset<uint32_t>(UB_ALIGN_SIZE / sizeof(uint32_t), 0);
    AscendC::LocalTensor<uint64_t> ubLocal64 = buf.GetWithOffset<uint64_t>(UB_ALIGN_SIZE / sizeof(uint64_t), UB_ALIGN_SIZE);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    uint32_t peer;

    GM_ADDR src_addr = gva + rank * message_length;

    if (rank == 0) {
        peer = 1;

        // aclshmem_roce_ptr: 将本地GVA地址转换为远程PE的可访问地址
        // WHY需要转换：RDMA需要知道远程PE的物理地址
        GM_ADDR dest_addr = (GM_ADDR)(aclshmem_roce_ptr(src_addr, peer));

        // WHY循环500次：测量平均Postsend开销
        int64_t start = AscendC::GetSystemCycle();
        for (uint32_t i = 0; i < 500; i++) {
            // aclshmemi_roce_write: RDMA底层Put接口
            // WHY使用底层接口：直接测量Postsend开销，排除高层封装的影响
            // 参数：目标地址、源地址、目标PE、长度、UB workspace
            aclshmemi_roce_write(dest_addr, src_addr, peer, 0, message_length, ubLocal64, ubLocal32, 0);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        int64_t end = AscendC::GetSystemCycle();
        // WHY存储到gva + message_length * 2：Host端读取结果
        *(__gm__ int64_t*)(gva + message_length * 2) = end - start;
    }
}

void rdma_postsend_cost_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length) {
    rdma_postsend_cost<<<1, nullptr, stream>>>(fftsConfig, gva, message_length);
}

/**
 * @brief RDMA High-level Put带宽测试kernel
 *
 * WHY测试RDMA带宽：
 * - 带宽衡量RDMA引擎的数据传输吞吐量
 * - 用于评估跨节点通信的性能上限
 * - 优化分布式训练的数据传输效率
 *
 * 测试方法：
 * - 连续发送10000次Put操作
 * - 使用quiet等待所有操作完成
 * - 计算总时间得到带宽
 *
 * @param fftsConfig FFTS配置地址
 * @param gva 对称内存指针
 * @param message_length 消息长度
 */
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_highlevel_put_bw(uint64_t fftsConfig, GM_ADDR gva, int message_length) {
    util_set_ffts_config(fftsConfig);
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;

    // WHY需要UB缓冲区：RDMA quiet操作需要workspace
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    uint32_t peer;

    GM_ADDR src_addr = gva + rank * message_length;
    if (rank == 0) {
        peer = 1;
        int64_t start = AscendC::GetSystemCycle();

        // WHY循环10000次：连续发送大量数据，测量带宽
        for (int i = 0; i < 10000; i++) {
            aclshmem_uint8_put_nbi(src_addr, src_addr, message_length, peer);
        }

        // aclshmemx_roce_quiet: RDMA quiet操作
        // WHY需要quiet：等待所有RDMA Put操作完成
        // 参数：目标PE、UB缓冲区、sync_id
        // 执行效果：阻塞直到所有之前发起的RDMA操作完成
        aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);

        // WHY发送完成标志：通知对端测试结束
        aclshmem_uint8_put_nbi(gva + rank_size * message_length + 8, src_addr, sizeof(uint32_t), peer);

        // WHY轮询等待对端回复：确保测试双方同步完成
        while (*(__gm__ uint32_t*)(gva + message_length * rank_size + 16) != peer + MAGIC_VAL) {
            dcci_cachelines(gva + message_length * rank_size + 16, 8);
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        int64_t end = AscendC::GetSystemCycle();
        *(__gm__ int64_t*)(gva + message_length * rank_size) = end - start;
    } else {
        peer = 0;
        // PE 1等待PE 0的完成标志
        while (*(__gm__ uint32_t*)(gva + rank_size * message_length + 8) != peer + MAGIC_VAL) {
            dcci_cachelines(gva + rank_size * message_length + 8, 8);
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        // PE 1回复完成标志
        aclshmem_uint8_put_nbi(gva + message_length * rank_size + 16, src_addr, sizeof(uint32_t), peer);
    }
}

void rdma_highlevel_put_bw_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length) {
    rdma_highlevel_put_bw<<<1, nullptr, stream>>>(fftsConfig, gva, message_length);
}

/**
 * @brief RDMA和MTE带宽对比测试kernel
 *
 * WHY对比RDMA和MTE带宽：
 * - RDMA：跨节点通信引擎（RoCE网络）
 * - MTE：节点内通信引擎（片上互联）
 * - 对比两种引擎的性能差异，帮助选择合适的通信方式
 *
 * 测试方法：
 * - Core 0测试RDMA带宽
 * - Core 1测试MTE带宽
 * - 同时运行，对比结果
 *
 * @param fftsConfig FFTS配置地址
 * @param gva 对称内存指针
 * @param message_length 消息长度
 * @param iter 迭代编号（用于区分不同轮次的测试）
 */
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_mte_put_bw(uint64_t fftsConfig, GM_ADDR gva, int message_length, int64_t iter) {
    util_set_ffts_config(fftsConfig);

    // WHY手动设置UB地址：MTE需要特定的UB缓冲区配置
    AscendC::LocalTensor<uint32_t> ubLocal;
    ubLocal.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ubLocal.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);
    ubLocal.address_.dataLen = UB_ALIGN_SIZE_64;

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    uint32_t peer;

    // ========== Core 0: RDMA测试 ==========
    // WHY区分Core：同时测试RDMA和MTE，需要两个核并行执行
    if (AscendC::GetBlockIdx() == 0) {
        GM_ADDR src_addr = gva + rank * message_length;
        if (rank == 0) {
            peer = 1;
            int64_t start = AscendC::GetSystemCycle();

            // WHY循环10000次：测量RDMA带宽
            for (int i = 0; i < 10000; i++) {
                // aclshmemx_roce_put_nbi: RDMA底层Put接口
                // WHY使用底层接口：精确测量RDMA性能
                aclshmemx_roce_put_nbi(src_addr, src_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), message_length, peer, 0);
            }

            // WHY需要quiet：等待所有RDMA操作完成
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);

            // WHY发送完成标志：通知对端RDMA测试结束
            aclshmemx_roce_put_nbi(gva + rank_size * message_length * 2 + 8, src_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), sizeof(int64_t), peer, 0);

            // WHY轮询等待：确保对端完成MTE测试
            while (*(__gm__ int64_t*)(gva + message_length * rank_size * 2 + 16) != peer + MAGIC_VAL + iter) {
                dcci_cachelines(gva + message_length * rank_size * 2 + 16, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            int64_t end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(gva + message_length * rank_size * 2) = end - start;
        } else {
            peer = 0;
            // PE 1等待PE 0的RDMA完成标志
            while (*(__gm__ int64_t*)(gva + rank_size * message_length * 2 + 8) != peer + MAGIC_VAL + iter) {
                dcci_cachelines(gva + rank_size * message_length * 2 + 8, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            // PE 1回复完成标志
            aclshmemx_roce_put_nbi(gva + rank_size * message_length * 2 + 16, src_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), sizeof(int64_t), peer, 0);
        }
    } else {
        // ========== Core 1: MTE测试 ==========
        // WHY使用MTE：对比节点内通信引擎的性能
        GM_ADDR src_addr = gva + (rank + rank_size) * message_length;

        // aclshmemi_get_state: 获取device state结构体
        // WHY需要device_state：读取MTE配置（UB地址、大小、sync_id）
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
        uint32_t copy_ub_size = device_state->mte_config.ub_size;
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;

        if (rank == 0) {
            peer = 1;
            int64_t start = AscendC::GetSystemCycle();

            // WHY循环10000次：测量MTE带宽
            for (int i = 0; i < 10000; i++) {
                // aclshmemx_mte_put_nbi: MTE Put接口
                // WHY使用MTE接口：节点内通信引擎
                // 参数：目标地址、源地址、UB缓冲区、UB大小、长度、目标PE、sync_id
                aclshmemx_mte_put_nbi(src_addr, src_addr, reinterpret_cast<__ubuf__ uint8_t*>(copy_ub), copy_ub_size, message_length, peer, copy_event_id);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // WHY发送MTE完成标志：通知对端MTE测试结束
            aclshmemx_mte_put_nbi(gva + rank_size * message_length * 2 + 24, src_addr, reinterpret_cast<__ubuf__ uint8_t*>(copy_ub), copy_ub_size, sizeof(uint32_t), peer, copy_event_id);

            // WHY轮询等待：确保对端完成MTE回复
            while (*(__gm__ uint32_t*)(gva + message_length * rank_size * 2 + 32) != peer + MAGIC_VAL + iter) {
                dcci_cachelines(gva + message_length * rank_size * 2 + 32, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            int64_t end = AscendC::GetSystemCycle();
            // WHY存储到不同位置：与RDMA结果分开存储
            *(__gm__ int64_t*)(gva + message_length * rank_size * 2 + 48) = end - start;
        } else {
            peer = 0;
            // PE 1等待PE 0的MTE完成标志
            while (*(__gm__ uint32_t*)(gva + rank_size * message_length * 2 + 24) != peer + MAGIC_VAL + iter) {
                dcci_cachelines(gva + rank_size * message_length * 2 + 24, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            // PE 1回复MTE完成标志
            aclshmemx_mte_put_nbi(gva + rank_size * message_length * 2 + 32, src_addr, reinterpret_cast<__ubuf__ uint8_t*>(copy_ub), copy_ub_size, sizeof(uint32_t), peer, copy_event_id);
        }
    }
}

void rdma_mte_put_bw_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length, int64_t iter) {
    // WHY启动2个核：Core 0测试RDMA，Core 1测试MTE
    rdma_mte_put_bw<<<2, nullptr, stream>>>(fftsConfig, gva, message_length, iter);
}