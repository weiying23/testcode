/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file main.cpp
 * @brief RDMA Demo - 跨节点AllGather示例程序
 *
 * 此程序演示如何使用RDMA引擎（RoCE网络）进行跨节点NPU间通信
 *
 * WHY使用RDMA引擎：
 * - RDMA支持跨节点NPU间通信（通过RoCE网络）
 * - 相比MTE/SDMA（仅支持节点内），RDMA适合超节点分布式训练场景
 * - 利用RDMA硬件实现零拷贝远程直接内存访问
 *
 * AllGather语义：
 * - 每个PE将自己的数据广播给所有其他PE
 * - 最终每个PE都拥有所有PE的数据
 * - 数据布局：PE i的数据位于offset = i * trans_size * sizeof(T)
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

/**
 * @brief RDMA AllGather测试函数
 *
 * WHY测试AllGather：
 * - AllGather是分布式训练中常用的集合通信操作
 * - 验证RDMA引擎的跨节点通信能力
 * - 确保数据正确传输到所有PE
 *
 * @param pe_id 当前PE编号（进程ID）
 * @param n_pes 总PE数量（进程总数）
 * @param local_mem_size 对称内存大小（字节）
 * @return 0表示成功，-1表示失败
 */
int test_aclshmem_team_all_gather(int pe_id, int n_pes, uint64_t local_mem_size)
{
    // ========== ACL初始化阶段 ==========
    // WHY需要ACL初始化：必须在调用任何ACL API之前执行

    // 计算物理设备ID：pe_id % g_npus + f_npu
    // WHY这样计算：
    // - pe_id % g_npus：将逻辑PE编号映射到节点内NPU编号
    // - f_npu：物理设备编号偏移量（节点内NPU的起始编号）
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    const int num10 = 10;
    aclrtStream stream = nullptr;

    // aclInit: 初始化ACL运行时环境
    // WHY参数nullptr：使用默认配置
    // 执行效果：
    // - 初始化CANN软件栈
    // - 加载驱动
    // - 准备NPU资源
    status |= aclInit(nullptr);

    // aclrtSetDevice: 设置当前进程使用的NPU设备
    // WHY需要设置设备：将进程绑定到指定NPU，后续所有ACL操作在该设备上执行
    status |= aclrtSetDevice(device_id);

    // aclrtCreateStream: 创建ACL流
    // WHY需要流：用于异步操作队列，管理kernel执行顺序
    status |= aclrtCreateStream(&stream);

    // ========== Shmem初始化阶段 ==========
    // aclshmemx_init_attr_t: shmem初始化属性结构体
    // WHY需要此结构体：
    // - 配置PE编号、进程数、内存大小等关键参数
    // - 设置数据传输引擎类型（RDMA/MTE/SDMA）
    // - 配置通信超时和同步机制
    aclshmemx_init_attr_t attributes;

    // test_set_attr: 辅助函数，填充shmem初始化属性结构体
    // WHY需要辅助函数：简化初始化流程，避免手动设置复杂的参数
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // ACLSHMEM_DATA_OP_ROCE: 设置数据传输引擎为RDMA（RoCE协议）
    // WHY使用RDMA引擎：
    // - RDMA支持跨节点NPU间通信
    // - 通过RoCE网络进行远程直接内存访问
    // - 相比MTE（节点内），RDMA适合超节点分布式训练
    // 其他可选引擎：
    // - ACLSHMEM_DATA_OP_MTE: MTE引擎（片上互联，仅节点内）
    // - ACLSHMEM_DATA_OP_SDMA: SDMA引擎（片上SDMA单元，仅节点内）
    // - ACLSHMEM_DATA_OP_UDMA: UDMA引擎（高性能互联）
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;

    // aclshmemx_init_attr: 初始化shmem运行时（默认socket模式）
    // WHY使用ACLSHMEMX_INIT_WITH_DEFAULT：
    // - 使用TCP socket进行进程间rendezvous
    // - 不需要MPI，简化部署
    // 执行后完成：
    // 1. 建立进程间通信通道（TCP socket和RoCE连接）
    // 2. 分配对称内存堆（Symmetric Heap）
    // 3. 初始化RDMA通信引擎
    // 4. 设置PE编号和通信组信息
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    // ========== 对称内存分配阶段 ==========
    // aclshmem_malloc: 分配对称内存（用于AllGather数据存储）
    // WHY使用对称内存：
    // - 所有PE在同一虚拟地址上拥有相同大小的内存块
    // - PE i可以直接通过GVA地址访问PE j的数据（跨节点也可）
    // - RDMA引擎可以直接访问远程节点的对称内存
    // 参数详解：
    // - 1024: 对称内存大小（字节）
    // 返回值：对称内存指针（GVA格式）
    uint8_t *ptr = static_cast<uint8_t*>(aclshmem_malloc(1024));

    // ========== 数据初始化阶段 ==========
    // WHY初始化数据：为AllGather操作准备源数据
    uint32_t trans_size = 16;
    std::vector<int32_t> input(trans_size, 0);

    // WHY每个PE的数据值为pe_id + num10：
    // - 用于验证数据传输的正确性
    // - 每个PE的数据有唯一标识，便于检查
    for (int i = 0; i < trans_size; i++) {
        input[i] = (pe_id + num10);
    }

    // aclshmem_my_pe(): 获取当前PE编号
    // WHY需要my_pe：计算数据在对称内存中的偏移位置
    // 数据布局：PE i的数据位于offset = i * trans_size * sizeof(int32_t)
    status |= aclrtMemcpy(ptr + aclshmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t),
        input.data(), trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // ========== AllGather执行阶段 ==========
    // allgather_demo: 执行RDMA AllGather kernel
    // WHY参数1：使用单个核执行AllGather
    // WHY参数stream：通过ACL流管理kernel执行
    allgather_demo(1, stream, (uint8_t *)ptr, trans_size * sizeof(int32_t));

    // aclshmem_handle_t: 操作句柄结构体，用于等待特定操作完成
    // WHY需要handle：
    // - 异步操作的同步等待
    // - 与shmem_quiet相比，可以等待特定操作而非所有操作
    aclshmem_handle_t handle;

    // ACLSHMEM_TEAM_WORLD: 全局通信组ID，包含所有PE
    // WHY使用全局通信组：AllGather需要所有PE参与
    handle.team_id = ACLSHMEM_TEAM_WORLD;

    // aclshmemx_handle_wait: 等待handle指定的操作完成
    // WHY需要等待：确保AllGather操作完成后再进行结果校验
    aclshmemx_handle_wait(handle, stream);
    status |= aclrtSynchronizeStream(stream);

    // aclshmemi_control_barrier_all: 内部屏障同步函数
    // WHY需要屏障：确保所有PE的通信操作完成
    // 执行流程：
    // 1. 当前PE到达屏障，标记自己已完成
    // 2. 等待所有其他PE也到达屏障
    // 3. 所有PE都到达后，一起释放继续执行
    aclshmemi_control_barrier_all();

    // ========== 结果校验阶段 ==========
    // WHY需要结果校验：验证AllGather操作的正确性
    int32_t *y_host;
    size_t input_size = n_pes * trans_size * sizeof(int32_t);

    // aclrtMallocHost: 在Host端分配内存用于读取Device端数据
    status |= aclrtMallocHost(reinterpret_cast<void**>(&y_host), input_size);

    // aclrtMemcpy: 将Device端数据拷贝到Host端
    status |= aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST);

    const int block_size = 16;
    for (int i = 0; i < n_pes; i++) {
        for (int j = 0; j < block_size; j++) {
            // WHY检查y_host[trans_size * i + ...] == num10 + i：
            // - 验证PE i的数据是否正确传输到本PE
            // - AllGather后，每个PE应该拥有所有PE的数据
            if (y_host[trans_size * i + trans_size / block_size * j] != num10 + i) {
                std::cout << y_host[trans_size * i + trans_size / block_size * j] << " != " << num10 + i << std::endl;
                return -1;
            }
        }
    }
    std::cout << "check transport result success, relative pe=" << pe_id << std::endl;

    // ========== 资源释放阶段 ==========
    // WHY需要资源释放：避免内存泄漏和资源占用
    status |= aclrtFreeHost(y_host);

    // aclshmem_free: 释放对称内存
    // WHY必须使用aclshmem_free：
    // - 不能使用aclrtFree释放对称内存
    // - 对称内存由shmem运行时管理
    // - 必须与aclshmem_malloc配对使用
    aclshmem_free(ptr);

    // aclshmem_finalize: 终止shmem运行时
    // WHY需要终止：
    // - 释放对称内存堆
    // - 关闭进程间通信通道（TCP socket和RoCE连接）
    // - 清理RDMA通信引擎状态
    status |= aclshmem_finalize();
    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();
    return 0;
}

/**
 * @brief 主函数
 *
 * 参数详解：
 * - argv[1]: n_pes - 总PE数量（进程总数）
 * - argv[2]: pe_id - 当前PE编号（进程ID）
 * - argv[3]: ipport - rendezvous地址（TCP socket地址，如"tcp://127.0.0.1:8998"）
 * - argv[4]: g_npus - 节点内NPU总数
 * - argv[5]: f_pe - PE编号偏移量（未使用）
 * - argv[6]: f_npu - NPU编号偏移量（物理设备ID的起点）
 *
 * WHY参数设计：
 * - 支持跨节点部署（通过ipport进行rendezvous）
 * - 支持多节点、多NPU配置
 * - 简化启动脚本的参数传递
 */
int main(int argc, char *argv[])
{
    int argIdx = 1;
    int status = 0;

    // 解析命令行参数
    int n_pes = atoi(argv[argIdx++]);  // 总PE数量
    int pe_id = atoi(argv[argIdx++]);  // 当前PE编号
    ipport = argv[argIdx++];           // rendezvous地址
    g_npus = atoi(argv[argIdx++]);     // 节点内NPU总数
    f_pe = atoi(argv[argIdx++]);       // PE编号偏移量
    f_npu = atoi(argv[argIdx++]);      // NPU编号偏移量

    // 对称内存大小：1GB
    // WHY设置1GB：
    // - 提供足够的对称内存空间
    // - 支持大规模数据传输
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;

    status = test_aclshmem_team_all_gather(pe_id, n_pes, local_mem_size);

    std::cout << "[SUCCESS] demo run success in relative pe " << pe_id << std::endl;
    return 0;
}