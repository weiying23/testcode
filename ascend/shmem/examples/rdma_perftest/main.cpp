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
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "utils.h"

int g_npus = 8;
const char *ipport;
int f_pe = 0;
int f_npu = 0;
const char *test_type;

extern void rdma_highlevel_put_pingpong_latency_do(uint32_t block_dim, void* st, uint64_t cfg, uint8_t* gva, int len);
extern void rdma_postsend_cost_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int len);
extern void rdma_highlevel_put_bw_do(uint32_t block_dim, void* stream, uint64_t cfg, uint8_t* gva, int len);
extern void rdma_mte_put_bw_do(uint32_t block_dim, void* stream, uint64_t cfg, uint8_t* gva, int len, int64_t iter);

aclshmemx_uniqueid_t default_flag_uid;

int test_aclshmem_rdma_highlevel_put_pingpong_latency(int pe_id, int n_pes, uint64_t local_mem_size, int message_length)
{
    uint32_t iteration = 1;
    // 计算物理设备ID：pe_id % g_npus + f_npu
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    const double ration50 = 50.0;
    const int times32 = 32;
    const int iterRange = 10;
    const int size6M = 6 * 1024 * 1024;

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
    // 包含PE编号、进程数、内存大小、引擎类型等配置
    aclshmemx_init_attr_t attributes;

    // test_set_attr: 辅助函数，填充shmem初始化属性结构体
    // 参数详解:
    // - pe_id: 当前PE编号
    // - n_pes: 总PE数量
    // - local_mem_size: 对称内存大小
    // - ipport: rendezvous地址字符串
    // - default_flag_uid: uniqueid结构体
    // - &attributes: 属性结构体指针（输出参数）
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // ACLSHMEM_DATA_OP_ROCE: 设置数据传输引擎为RDMA（RoCE协议）
    // RDMA引擎特点：
    // - 用于跨节点NPU间通信
    // - 通过RoCE网络进行远程直接内存访问
    // - 支持跨节点的高速低延迟数据传输
    // - Pingpong latency测试使用RDMA引擎测量跨节点往返延迟
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;

    // aclshmemx_set_conf_store_tls: 设置配置存储模式
    // 参数: false表示不使用TLS配置存储
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

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

    // util_get_ffts_config: 获取FFTS配置地址
    // FFTS = Fast Fourier Transform Scheduler，用于硬件同步
    uint64_t fftsConfig = util_get_ffts_config();

    // aclshmem_malloc: 分配对称内存（用于RDMA Pingpong测试数据）
    // 参数详解:
    // - size6M: 对称内存大小（6MB）
    // 返回值: 对称内存指针（GVA格式）
    // 对称内存用途：
    // - 存存Pingpong测试的源数据和目标数据
    // - RDMA引擎通过GVA地址直接访问远程PE的数据
    // 对称内存核心特点：
    // 1. 所有PE在同一虚拟地址上拥有相同大小的内存块
    // 2. PE i可以直接通过GVA地址访问PE j的数据
    // 3. 用于存放通信数据和同步标志
    uint8_t *gva = static_cast<uint8_t*>(aclshmem_malloc(size6M));

    int64_t *xHost;
    size_t totalSize = message_length * n_pes;

    // aclrtMallocHost: 在Host端分配内存用于初始化数据
    aclrtMallocHost(reinterpret_cast<void **>(&xHost), totalSize);
    // 初始化测试数据：每个PE的数据值为 pe_id + iterRange
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = pe_id + iterRange;
    }
    // aclrtMemcpy: 将Host端数据拷贝到Device端对称内存
    // 数据布局：PE i的数据位于 gva + i * message_length
    aclrtMemcpy(gva + pe_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);
    // 拷贝同步标志数据到对称内存
    aclrtMemcpy(gva + n_pes * message_length + times32 * (pe_id + 1), times32,
        xHost, times32, ACL_MEMCPY_HOST_TO_DEVICE);

    // ========== RDMA Pingpong Latency测试 ==========
    // rdma_highlevel_put_pingpong_latency_do: 执行RDMA Pingpong延迟测试
    // Pingpong测试流程：
    // 1. PE 0发送数据到PE 1
    // 2. PE 1接收数据后发送回PE 0
    // 3. 测量往返延迟时间
    // 参数详解:
    // - 1: block_dim（核数）
    // - stream: ACL流
    // - fftsConfig: FFTS配置地址
    // - gva: 对称内存指针（GVA格式）
    // - message_length: 消息长度（字节）
    for (uint32_t i = 0; i < iteration; i++) {
        rdma_highlevel_put_pingpong_latency_do(1, stream, fftsConfig, gva, message_length);
    }
    aclrtSynchronizeStream(stream);

    // ========== 结果输出 ==========
    // PE 0输出Pingpong延迟结果
    if (pe_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_pes,
            sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        // 延迟计算：xHost[0] / ration50 us
        // ration50 = 50.0，将硬件周期转换为微秒
        std::cout << "RDMA highlevel put pingpong latency test. Message length = " << message_length
            << " Byte; latency = " << xHost[0] / ration50 << " us." << std::endl;
    }

    // ========== 资源释放 ==========
    aclrtFreeHost(xHost);
    // aclshmem_finalize: 终止shmem运行时，释放所有shmem资源
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}

int test_aclshmem_rdma_postsend_cost(int pe_id, int n_pes, uint64_t local_mem_size, int message_length)
{
    uint32_t iteration = 1;
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    const double ration2500 = 50.0 * 500;
    const int iterRange = 10;
    const int size6M = 6 * 1024 * 1024;

    // ========== ACL初始化 ==========
    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    // aclshmemx_init_attr_t: shmem初始化属性结构体
    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // ACLSHMEM_DATA_OP_ROCE: 设置数据传输引擎为RDMA
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    uint64_t fftsConfig = util_get_ffts_config();
    // aclshmem_malloc: 分配对称内存（用于Postsend cost测试）
    uint8_t *gva = static_cast<uint8_t*>(aclshmem_malloc(size6M));

    int64_t *xHost;
    size_t totalSize = message_length * n_pes;

    aclrtMallocHost(reinterpret_cast<void **>(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = pe_id + iterRange;
    }
    aclrtMemcpy(gva + pe_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);

    // ========== RDMA Postsend Cost测试 ==========
    // rdma_postsend_cost_do: 执行RDMA Postsend开销测试
    // Postsend开销：测量RDMA发送操作的开销时间
    // 用于评估RDMA通信的单次发送延迟
    for (uint32_t i = 0; i < iteration; i++) {
        rdma_postsend_cost_do(1, stream, fftsConfig, gva, message_length);
    }
    aclrtSynchronizeStream(stream);
    if (pe_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_pes,
            sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "RDMA postsend cost test. Message length = " << message_length
            << " Byte; postsend cost = " << xHost[0] / ration2500 << " us." << std::endl;
    }

    aclrtFreeHost(xHost);
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}

int test_aclshmem_rdma_highlevel_put_bw(int pe_id, int n_pes, uint64_t local_mem_size, int message_length)
{
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    const double ration50 = 50.0;
    const int iterRange = 10;
    const int size6M = 6 * 1024 * 1024;

    // ========== ACL初始化 ==========
    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    // aclshmemx_init_attr_t: shmem初始化属性结构体
    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // ACLSHMEM_DATA_OP_ROCE: 设置数据传输引擎为RDMA
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    uint64_t fftsConfig = util_get_ffts_config();
    // aclshmem_malloc: 分配对称内存（用于带宽测试）
    uint8_t *gva = static_cast<uint8_t*>(aclshmem_malloc(size6M));

    int64_t *xHost;
    size_t totalSize = message_length * n_pes;

    aclrtMallocHost(reinterpret_cast<void **>(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = pe_id + iterRange;
    }
    aclrtMemcpy(gva + pe_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);

    // ========== RDMA带宽测试 ==========
    // rdma_highlevel_put_bw_do: 执行RDMA High-level Put带宽测试
    // 测量RDMA Put操作的吞吐量
    // 用于评估RDMA引擎的数据传输带宽
    rdma_highlevel_put_bw_do(1, stream, fftsConfig, gva, message_length);
    aclrtSynchronizeStream(stream);
    if (pe_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_pes, sizeof(int64_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "RDMA high level put bandwidth test. Message length = " << message_length
            << " Byte; time = " << xHost[0] / ration50 << " us." << std::endl;
    }

    aclrtFreeHost(xHost);
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}

int test_aclshmem_rdma_mte_put_bw(int pe_id, int n_pes, uint64_t local_mem_size, int message_length)
{
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    const int size32M = 32 * 1024 * 1024;
    const int size128K = 128 * 1024;

    // ========== ACL初始化 ==========
    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    // aclshmemx_init_attr_t: shmem初始化属性结构体
    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // ACLSHMEM_DATA_OP_ROCE: 设置数据传输引擎为RDMA
    // 此测试对比RDMA和MTE两种引擎的带宽性能
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    // util_get_ffts_config: 获取FFTS配置地址
    uint64_t fftsConfig = util_get_ffts_config();
    // aclshmem_malloc: 分配对称内存（用于RDMA/MTE对比带宽测试）
    // 参数详解:
    // - size32M: 对称内存大小（32MB）
    // 返回值: 对称内存指针（GVA格式）
    uint8_t *gva = static_cast<uint8_t*>(aclshmem_malloc(size32M));
    int64_t *inHost;
    int64_t *outHost;
    size_t totalSize = message_length * n_pes * 3;
    aclrtMallocHost(reinterpret_cast<void **>(&inHost), totalSize);
    aclrtMallocHost(reinterpret_cast<void **>(&outHost), totalSize);
    bzero(inHost, totalSize);
    double rdmaTotalTime = 0.0;
    double mteTotalTime = 0.0;
    const int mteIdx = 6;
    const double ratio10 = 10.0;
    const double ration50 = 50.0;
    const int dstMax = 64;
    const int iterRange = 10;
    const int maxIter = 20;
    const int peTimes = 2;

    // ========== RDMA/MTE带宽对比测试 ==========
    // 此测试对比RDMA引擎和MTE引擎的带宽性能
    // RDMA: 跨节点通信引擎（RoCE网络）
    // MTE: 节点内通信引擎（片上互联）
    for (int iter = 0; iter < maxIter; iter++) {
        for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
            inHost[i + pe_id * message_length / sizeof(int64_t)] = pe_id + iterRange + iter;
        }
        for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
            inHost[i + (pe_id + n_pes) * message_length / sizeof(int64_t)] = pe_id + iterRange + iter;
        }
        aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE);

        // aclshmemi_control_barrier_all: 内部屏障同步，确保所有PE数据初始化完成
        // 执行流程:
        // 1. 当前PE到达屏障，标记自己已完成初始化
        // 2. 等待所有其他PE也到达屏障
        // 3. 所有PE都到达后，一起释放继续执行
        aclshmemi_control_barrier_all();

        // rdma_mte_put_bw_do: 执行RDMA和MTE带宽对比测试
        // 参数详解:
        // - 1: block_dim（核数）
        // - stream: ACL流
        // - fftsConfig: FFTS配置地址
        // - gva: 对称内存指针（GVA格式）
        // - message_length: 消息长度（字节）
        // - iter: 当前迭代编号
        rdma_mte_put_bw_do(1, stream, fftsConfig, gva, message_length, iter);
        aclrtSynchronizeStream(stream);

        // PE 0收集并输出测试结果（跳过前iterRange次迭代的预热）
        if (pe_id == 0 && iter >= iterRange) {
            aclrtMemcpy(outHost, dstMax, gva + message_length * n_pes * peTimes, dstMax, ACL_MEMCPY_DEVICE_TO_HOST);
            rdmaTotalTime += outHost[0] / ration50;
            mteTotalTime += outHost[mteIdx] / ration50;
        }
    }
    if (pe_id == 0) {
        std::cout << "RDMA rdma mte test. Message length = " << message_length << " Byte; average RDMA time = "
            << rdmaTotalTime / ratio10 << " us." << std::endl;
        std::cout << "RDMA rdma mte test. Message length = " << message_length << " Byte; average MTE time = "
            << mteTotalTime / ratio10 << " us." << std::endl;
    }

    // ========== 资源释放 ==========
    aclrtFreeHost(inHost);
    aclrtFreeHost(outHost);
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}

int main(int argc, char *argv[])
{
    const int expected_argc = 9;
    if (argc != expected_argc) {
        std::cout << "[ERROR] Paramater number mismatch." << std::endl;
        std::cout << "[USAGE] ./rdma_perftest <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> "
            << "<test_type> <msg_len>. See README for more details." << std::endl;
    }
    int sub = 1;
    int status = 0;
    int n_pes = atoi(argv[sub++]);
    const int pe_max = 2;
    if (n_pes != pe_max) {
        std::cout << "[ERROR] Error number of pes! Only support 2 pes!" << std::endl;
    }
    int pe_id = atoi(argv[sub++]);
    if (pe_id >= pe_max) {
        std::cout << "[ERROR] Error pe ID! Only support 2 pes!" << std::endl;
    }
    ipport = argv[sub++];
    g_npus = atoi(argv[sub++]);
    f_pe = atoi(argv[sub++]);
    f_npu = atoi(argv[sub++]);
    test_type = argv[sub++];
    int msg_len = atoi(argv[sub++]);
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    if (std::string(test_type) == "highlevel_put_pingpong_latency") {
        test_aclshmem_rdma_highlevel_put_pingpong_latency(pe_id, n_pes, local_mem_size, msg_len);
    } else if (std::string(test_type) == "postsend_cost") {
        test_aclshmem_rdma_postsend_cost(pe_id, n_pes, local_mem_size, msg_len);
    } else if (std::string(test_type) == "highlevel_put_bw") {
        test_aclshmem_rdma_highlevel_put_bw(pe_id, n_pes, local_mem_size, msg_len);
    } else if (std::string(test_type) == "rdma_mte_bw") {
        test_aclshmem_rdma_mte_put_bw(pe_id, n_pes, local_mem_size, msg_len);
    }

    std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    return 0;
}