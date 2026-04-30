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
void copy_demo(uint32_t block_dim, void* stream, uint8_t* src, uint8_t* dst, int elements);

aclshmemx_uniqueid_t default_flag_uid;

int test_aclshmem_team_all_gather(int pe_id, int n_pes, uint64_t local_mem_size)
{
    // ========== ACL && ACLSHMEM 初始化 ==========
    // 计算物理设备ID：pe_id % g_npus + f_npu
    // pe_id: 当前进程的逻辑编号（Process ID）
    // g_npus: 节点内NPU总数
    // f_npu: NPU编号偏移量（物理设备ID的起点）
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    const int num10 = 10;
    const uint32_t mem_size = 1024UL * 1024UL;
    const uint32_t half_mem_size = 512UL * 1024UL;

    aclrtStream stream = nullptr;

    // aclInit: 初始化ACL（Ascend Computing Language）运行时环境
    // 参数: nullptr表示使用默认配置
    // 必须在调用任何ACL API之前执行
    status = aclInit(nullptr);
    // aclrtSetDevice: 设置当前进程使用的NPU设备
    // 将进程绑定到指定NPU，后续所有ACL操作在该设备上执行
    status = aclrtSetDevice(device_id);
    // aclrtCreateStream: 创建ACL流（用于异步操作队列）
    status = aclrtCreateStream(&stream);

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
    // - local_mem_size: 对称内存大小（1GB）
    // - ipport: rendezvous地址字符串
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
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;

    // aclshmemx_init_attr: 初始化shmem运行时（默认socket模式）
    // 参数详解:
    // - ACLSHMEMX_INIT_WITH_DEFAULT: 初始化模式标志
    //   使用TCP socket进行进程间rendezvous
    // - &attributes: 初始化属性结构体指针
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 执行后完成:
    // 1. 建立进程间通信通道（TCP socket和RoCE连接）
    // 2. 分配对称内存堆
    // 3. 初始化RDMA通信引擎
    // 4. 设置PE编号和通信组信息
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    // aclshmem_malloc: 分配对称内存（用于AllGather通信数据）
    // 参数详解:
    // - mem_size: 对称内存大小（1MB）
    // 返回值: 对称内存指针（GVA格式）
    // 对称内存核心特点：
    // 1. 所有PE在同一虚拟地址上拥有相同大小的内存块
    // 2. PE i可以直接通过GVA地址访问PE j的数据
    // 3. 用于存放通信数据和同步标志
    // 注意：
    // - 必须通过aclshmem_free释放
    // - 分配大小不能超过初始化时设置的local_mem_size
    uint8_t *ptr = static_cast<uint8_t*>(aclshmem_malloc(mem_size));
    uint8_t *ptr_A = ptr + half_mem_size;

    // ========== 数据初始化 ==========
    uint32_t trans_size = 32UL * 1024UL;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = (pe_id + num10);
    }

    // aclshmem_my_pe(): 获取当前PE编号（在Host端调用）
    // 返回当前进程在通信组中的编号，范围[0, n_pes-1]
    // 用于计算数据在对称内存中的偏移位置
    // 对称内存布局：PE i的数据位于offset = i * trans_size * sizeof(int32_t)
    status = aclrtMemcpy(ptr + aclshmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t),
                         input.data(), trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // ========== AllGather操作 ==========
    // AllGather: 每个PE将自己的数据分片发送给所有其他PE
    // 执行后每个PE都拥有所有PE的数据总和
    allgather_demo(1, stream, (uint8_t *)ptr, trans_size * sizeof(int32_t));

    // ========== 关键差异：不使用handle_wait ==========
    // 本测试与use_handlewait的关键区别：
    // - use_handlewait版本调用aclshmemx_handle_wait等待AllGather完成
    // - 本版本直接执行copy_demo，不等待AllGather完成
    // 这可能导致数据不一致：
    // - copy_demo可能在AllGather完成之前执行
    // - ptr_A中的数据可能不完整或不正确
    // - 用于对比验证handle_wait的必要性
    copy_demo(1, stream, ptr, ptr_A, n_pes * trans_size * sizeof(int32_t));

    status = aclrtSynchronizeStream(stream);

    // ========== 结果校验 ==========
    // 校验ptr_A中的AllGather结果是否正确
    // 由于不使用handle_wait，数据可能不完整或不正确
    // 用于对比验证handle_wait的必要性
    if (pe_id <= n_pes) {
        int32_t *y_host;
        size_t input_size = n_pes * trans_size * sizeof(int32_t);

        // aclrtMallocHost: 在Host端分配内存用于接收Device数据
        // 必须与aclrtFreeHost配对使用
        status = aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size);
        // aclrtMemcpy: 将Device端的结果拷贝到Host端进行校验
        status = aclrtMemcpy(y_host, input_size, ptr_A, input_size, ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "Relative pe " << pe_id << " AllGather result in ptr_A without handle_wait:" << std::endl;

        // 校验每个PE的数据：期望值 = num10 + pe_index
        // 由于没有handle_wait，可能会有unexpected值
        int unexpected_count = 0;
        for (int i = 0; i < n_pes; i++) {
            for (int j = 0; j < trans_size; j++) {
                if (y_host[trans_size * i + j] != num10 + i) {
                    unexpected_count++;
                }
            }
        }
        std::cout << "Relative pe " << pe_id << " has " << unexpected_count << " unexpected values." << std::endl;
        // aclrtFreeHost: 释放Host端内存
        status = aclrtFreeHost(y_host);
    }

    // ========== 资源释放 ==========
    // aclshmem_free: 释放对称内存
    // 参数: aclshmem_malloc返回的对称内存指针（ptr）
    // 必须与aclshmem_malloc配对使用
    // 执行效果:
    // - 将对称内存归还到Symmetric Heap
    // - 其他shmem操作可以重新分配此内存
    // - 释放后该地址不再可用于通信
    // 重要提示：
    // 1. 不能使用aclrtFree释放对称内存，必须使用aclshmem_free
    // 2. 所有PE应同时释放对称内存
    // 3. 释放前确保RDMA操作已完成
    aclshmem_free(ptr);

    // aclshmem_finalize: 终止shmem运行时，释放所有shmem资源
    // 功能详解：
    // - 释放对称内存堆（Symmetric Heap）
    // - 关闭进程间通信通道（TCP socket和RoCE连接）
    // - 清理RDMA通信引擎状态
    // - 释放内部同步机制资源（barrier、quiet等）
    // 执行流程：
    // 1. 等待所有pending的RDMA操作完成
    // 2. 通知其他PE本PE即将退出
    // 3. 释放所有对称内存资源
    // 4. 关闭bootstrap通信通道
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 注意：
    // 1. 每个PE必须调用此函数后才能退出程序
    // 2. 所有PE应同时调用此函数
    // 3. 调用后不能再执行任何shmem操作
    status = aclshmem_finalize();
    status = aclrtDestroyStream(stream);
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
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