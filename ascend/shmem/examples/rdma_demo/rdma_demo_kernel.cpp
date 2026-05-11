/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RDMA_DEMO_KERNEL_
#define _RDMA_DEMO_KERNEL_

#include "kernel_operator.h"
#include "shmem.h"

/**
 * @brief RDMA AllGather kernel实现
 *
 * 此kernel使用RDMA引擎（RoCE网络）实现跨节点AllGather操作
 *
 * WHY使用RDMA引擎：
 * - RDMA支持跨节点NPU间通信（通过RoCE网络）
 * - 利用RDMA硬件实现零拷贝远程直接内存访问
 * - 相比MTE/SDMA（仅支持节点内），RDMA适合超节点分布式训练场景
 *
 * AllGather语义：
 * - 每个PE将自己的数据广播给所有其他PE
 * - 最终每个PE都拥有所有PE的数据
 * - 数据布局：PE i的数据位于offset = i * message_length
 */

extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void device_all_gather_test(GM_ADDR gva, int message_length)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;

    // WHY需要UB缓冲区：
    // - RDMA任务下发需要一个长度大于等于64字节的LocalTensor作为workspace
    // - 用于存储RDMA SQE（Send Queue Entry）任务描述符
    // - UB_ALIGN_SIZE * 2确保足够的缓冲区空间
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE * 2, 0);

    // aclshmem_my_pe(): 获取当前PE编号（进程ID）
    // WHY需要my_rank：用于确定本PE的数据位置和通信目标
    // 返回值：当前进程在通信组中的编号，范围[0, n_pes-1]
    int64_t my_rank = aclshmem_my_pe();

    // aclshmem_n_pes(): 获取通信组中的总PE数量
    // WHY需要pe_size：用于计算AllGather循环范围
    int64_t pe_size = aclshmem_n_pes();

    // WHY需要PipeBarrier：
    // - 确保所有AIV核同步启动，避免竞态条件
    // - AllGather开始前需要所有核准备好
    AscendC::PipeBarrier<PIPE_ALL>();

    // AllGather核心循环：向所有其他PE发送本PE数据
    // WHY循环范围是pe_size：需要向每个PE发送数据（跳过自己）
    for (int i = 0; i < pe_size; i++) {
        if (i == my_rank) {
            continue;  // WHY跳过自己：不需要向自己发送数据
        }

        // aclshmemx_roce_put_nbi: RDMA非阻塞Put操作
        // WHY使用RDMA Put：跨节点数据传输，利用RoCE网络
        // 参数详解：
        // - gva + message_length * my_rank: 目标地址（远程PE的接收地址）
        //   GVA = Global Virtual Address，所有PE看到的相同虚拟地址
        //   WHY使用GVA：RDMA可以直接通过GVA访问远程节点数据
        // - gva + message_length * my_rank: 源地址（本PE的发送地址）
        //   WHY源和目标地址相同：AllGather中，本PE数据在所有PE的位置相同
        // - (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(): UB缓冲区地址
        //   WHY需要UB：RDMA任务下发需要workspace
        // - message_length: 数据长度（字节）
        // - i: 目标PE编号
        // - 0: sync_id（事件ID）
        // 非阻塞特性：函数立即返回，不等待传输完成
        aclshmemx_roce_put_nbi(gva + message_length * my_rank, gva + message_length * my_rank,
                                (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), message_length, i, 0);
    }
}

/**
 * @brief AllGather demo启动函数
 *
 * WHY需要此函数：
 * - 封装kernel启动参数
 * - Host端通过此函数调用Device kernel
 *
 * @param block_dim 核数配置
 * @param stream ACL流
 * @param gva 对称内存指针（GVA格式）
 * @param elements 消息长度（字节）
 */
void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva, int elements)
{
    device_all_gather_test<<<block_dim, nullptr, stream>>>(gva, elements);
}

#endif  // _RDMA_DEMO_KERNEL_