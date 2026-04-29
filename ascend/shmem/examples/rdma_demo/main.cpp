/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "utils.h"

int g_npus = 8;
const char *ipport;
int f_pe = 0;
int f_npu = 0;
extern void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva, int message_length);

aclshmemx_uniqueid_t default_flag_uid;

int test_aclshmem_team_all_gather(int pe_id, int n_pes, uint64_t local_mem_size)
{
    // 初始化ACL和ACLSHMEM
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    const int num10 = 10;
    aclrtStream stream = nullptr;

    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);

    // aclshmemx_init_attr_t: shmem初始化属性结构体
    // 包含以下关键字段：
    // - my_pe: 当前PE编号（进程ID），范围[0, n_pes-1]
    // - n_pes: 总PE数量（进程总数）
    // - ip_port: rendezvous地址（TCP socket地址）
    // - local_mem_size: 对称内存大小（字节）
    // - option_attr: 可选属性
    //   .data_op_engine_type: 数据传输引擎类型
    //   .timeout: 各阶段超时设置
    // - instance_id: 多实例模式下的实例编号
    // - comm_args: 通信参数指针
    aclshmemx_init_attr_t attributes;

    // test_set_attr: 辅助函数，填充shmem初始化属性结构体
    // 参数详解:
    // - pe_id: 当前PE编号
    // - n_pes: 总PE数量
    // - local_mem_size: 对称内存大小
    // - ipport: rendezvous地址字符串，如"tcp://127.0.0.1:8998"
    // - default_flag_uid: uniqueid结构体
    // - &attributes: 属性结构体指针（输出参数）
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // ACLSHMEM_DATA_OP_ROCE: 设置数据传输引擎为RDMA（RoCE协议）
    // RDMA引擎特点：
    // - 用于跨节点NPU间通信
    // - 通过RoCE网络进行远程直接内存访问
    // - 支持跨节点的高速低延迟数据传输
    // - 利用RDMA硬件实现零拷贝传输
    // 与其他引擎对比:
    // - ACLSHMEM_DATA_OP_MTE: MTE引擎（片上互联，仅节点内）
    // - ACLSHMEM_DATA_OP_SDMA: SDMA引擎（片上SDMA单元，仅节点内）
    // - ACLSHMEM_DATA_OP_UDMA: UDMA引擎（高性能互联）
    // RDMA适合跨节点分布式训练场景
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;

    // aclshmemx_init_attr: 初始化shmem运行时（默认socket模式）
    // 参数详解:
    // - ACLSHMEMX_INIT_WITH_DEFAULT: 初始化模式标志
    //   使用TCP socket进行进程间rendezvous
    // - &attributes: 初始化属性结构体指针
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 执行后完成:
    // 1. 建立进程间通信通道
    // 2. 分配对称内存堆
    // 3. 初始化RDMA通信引擎（RoCE网络）
    // 4. 设置PE编号和通信组信息
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    // aclshmem_malloc: 分配对称内存
    // 用于存放AllGather操作的数据
    // 参数详解:
    // - 1024: 对称内存大小（字节）
    // 返回值: 对称内存指针（GVA格式）
    // 对称内存核心特点：
    // 1. 所有PE在同一虚拟地址上拥有相同大小的内存块
    // 2. PE i可以直接通过GVA地址访问PE j的数据（跨节点也可访问）
    // 3. RDMA引擎可以直接访问远程节点的对称内存
    // 注意：
    // - 必须通过aclshmem_free释放
    // - 跨节点场景中对称内存地址必须一致
    uint8_t *ptr = static_cast<uint8_t*>(aclshmem_malloc(1024));

    // 初始化数据
    uint32_t trans_size = 16;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = (pe_id + num10);
    }

    // aclshmem_my_pe(): 获取当前PE编号
    // 用于计算数据在对称内存中的偏移位置
    // 返回值: 当前PE编号，范围[0, n_pes-1]
    // 对称内存布局：PE i的数据位于offset = i * trans_size * sizeof(int32_t)
    status |= aclrtMemcpy(ptr + aclshmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t),
        input.data(), trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // AllGather
    allgather_demo(1, stream, (uint8_t *)ptr, trans_size * sizeof(int32_t));

    // aclshmem_handle_t: 操作句柄结构体，用于等待特定操作完成
    // 包含以下字段：
    // - team_id: 通信组ID
    // - handle: 操作句柄值
    // 用途：异步操作的同步等待
    aclshmem_handle_t handle;

    // ACLSHMEM_TEAM_WORLD: 全局通信组ID，包含所有PE
    // 预定义的通信组常量，代表所有PE组成的通信组
    // 其他可用通信组：
    // - ACLSHMEM_TEAM_SHARED: 共享内存通信组（节点内）
    // - 自定义子组（通过shmem_team_split创建）
    handle.team_id = ACLSHMEM_TEAM_WORLD;

    // aclshmemx_handle_wait: 等待handle指定的操作完成
    // 参数详解:
    // - handle: 操作句柄（包含team_id和handle值）
    // - stream: ACL流
    // 执行效果:
    // - 阻塞直到handle指定的操作完成
    // - 用于异步操作的同步等待
    // - 与shmem_quiet相比，可以等待特定操作而非所有操作
    aclshmemx_handle_wait(handle, stream);
    status |= aclrtSynchronizeStream(stream);

    // aclshmemi_control_barrier_all: 内部屏障同步函数
    // 确保所有PE的通信操作完成
    // 执行流程:
    // 1. 当前PE到达屏障，标记自己已完成
    // 2. 等待所有其他PE也到达屏障
    // 3. 所有PE都到达后，一起释放继续执行
    // 注意：这是内部函数，推荐使用aclshmem_barrier_all
    aclshmemi_control_barrier_all();

    // 结果校验打印
    int32_t *y_host;
    size_t input_size = n_pes * trans_size * sizeof(int32_t);
    status |= aclrtMallocHost(reinterpret_cast<void**>(&y_host), input_size);
    status |= aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST);

    const int block_size = 16;
    for (int i = 0; i < n_pes; i++) {
        for (int j = 0; j < block_size; j++) {
            if (y_host[trans_size * i + trans_size / block_size * j] != num10 + i) {
                std::cout << y_host[trans_size * i + trans_size / block_size * j] << " != " << num10 + i << std::endl;
                // std::exit(EXIT_FAILURE);
                return -1;
            }
        }
    }
    std::cout << "check transport result success, relative pe=" << pe_id << std::endl;
    // 去初始化
    status |= aclrtFreeHost(y_host);

    // aclshmem_free: 释放对称内存
    // 参数: aclshmem_malloc返回的对称内存指针（ptr）
    // 必须与aclshmem_malloc配对使用
    // 执行效果:
    // - 将对称内存归还到Symmetric Heap
    // - 其他shmem操作可以重新分配此内存
    // - 释放后该地址不再可用于通信
    // 重要提示：
    // 1. 不能使用aclrtFree释放对称内存，必须使用aclshmem_free
    // 2. 所有PE应同时释放对称内存，避免内存碎片
    // 3. 释放前确保所有RDMA操作已完成
    aclshmem_free(ptr);

    // aclshmem_finalize: 终止shmem运行时
    // 功能详解：
    // - 释放对称内存堆
    // - 关闭进程间通信通道（TCP socket和RoCE连接）
    // - 清理RDMA通信引擎状态
    // - 释放内部同步机制资源
    // 执行流程：
    // 1. 等待所有pending的RDMA操作完成
    // 2. 通知其他PE本PE即将退出
    // 3. 释放所有对称内存资源
    // 4. 关闭RoCE网络连接
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 注意：
    // 1. 每个PE必须调用此函数后才能退出程序
    // 2. 跨节点场景中所有节点应同时退出
    status |= aclshmem_finalize();
    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();
    return 0;
}

int main(int argc, char *argv[])
{
    int argIdx = 1;
    int status = 0;
    int n_pes = atoi(argv[argIdx++]);
    int pe_id = atoi(argv[argIdx++]);
    ipport = argv[argIdx++];
    g_npus = atoi(argv[argIdx++]);
    f_pe = atoi(argv[argIdx++]);
    f_npu = atoi(argv[argIdx++]);
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    status = test_aclshmem_team_all_gather(pe_id, n_pes, local_mem_size);

    std::cout << "[SUCCESS] demo run success in relative pe " << pe_id << std::endl;
    return 0;
}