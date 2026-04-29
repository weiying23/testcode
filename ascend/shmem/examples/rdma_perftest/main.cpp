/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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

#include "acl/acl.h"
#include "shmem_api.h"
#include "shmemi_host_common.h"

int g_npus = 8;
const char *ipport;
int f_rank = 0;
int f_npu = 0;
const char *test_type;

extern void rdma_highlevel_put_pingpong_latency_do(uint32_t block_dim, void* st, uint64_t cfg, uint8_t* gva, int len);
extern void rdma_postsend_cost_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int len);
extern void rdma_highlevel_put_bw_do(uint32_t block_dim, void* stream, uint64_t cfg, uint8_t* gva, int len);
extern void rdma_mte_put_bw_do(uint32_t block_dim, void* stream, uint64_t cfg, uint8_t* gva, int len, int64_t iter);

int test_shmem_rdma_highlevel_put_pingpong_latency(int rank_id, int n_ranks, uint64_t mem_size, int message_length)
{
    uint32_t iteration = 1;
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    const double ration50 = 50.0;
    const int times32 = 32;
    const int iterRange = 10;
    const int size6M = 6 * 1024 * 1024;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    shmem_init_attr_t *attributes;
    // shmem_set_attr(): 设置shmem初始化属性参数
    // 参数1 rank_id: 当前进程的rank编号（进程在通信组中的唯一标识）
    // 参数2 n_ranks: 通信组中总进程数量（所有参与分布式计算的rank总数）
    // 参数3 mem_size: 每个rank分配的对称内存空间大小
    // 参数4 ipport: 网络通信的IP地址和端口字符串，用于rank间RDMA网络连接建立
    // 参数5 &attributes: 输出参数，返回配置好的初始化属性结构体指针
    status = shmem_set_attr(rank_id, n_ranks, mem_size, ipport, &attributes);
    // attributes->option_attr.data_op_engine_type: 设置数据操作引擎类型
    // SHMEM_DATA_OP_ROCE表示使用RoCE(RDMA over Converged Ethernet)引擎进行数据传输
    // RoCE提供高性能的RDMA通信，适合大规模分布式计算场景
    attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
    // shmem_set_conf_store_tls(): 禁用TLS(Thread Local Storage)存储配置方式
    // 参数: false表示禁用TLS，nullptr和0表示不使用默认配置文件路径和长度
    shmem_set_conf_store_tls(false, nullptr, 0);
    // shmem_init_attr(): 根据attributes中的配置参数初始化shmem运行环境
    status = shmem_init_attr(attributes);

    // shmemx_get_ffts_config(): 获取FFTS(Fast Flag Task Sync)硬件同步配置地址
    // FFTS是NPU核间快速同步机制，用于在kernel执行时实现核间的轻量级同步操作
    // 返回: FFTS配置寄存器的物理地址，传递给kernel用于设置同步基址
    uint64_t fftsConfig = shmemx_get_ffts_config();
    // shmem_malloc(): 从对称共享内存堆(Symmetric Heap)中分配指定大小的内存空间
    // 对称内存是指所有rank在相同偏移位置都能访问的共享内存区域，用于跨rank RDMA通信
    // 参数: size6M = 6 * 1024 * 1024 (6MB)
    // 返回: 对称内存指针，所有rank都可以通过相同偏移访问该内存区域
    uint8_t *gva = static_cast<uint8_t*>(shmem_malloc(size6M));

    int64_t *xHost;
    size_t totalSize = message_length * n_ranks;

    aclrtMallocHost(reinterpret_cast<void **>(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = rank_id + iterRange;
    }
    aclrtMemcpy(gva + rank_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(gva + n_ranks * message_length + times32 * (rank_id + 1), times32,
        xHost, times32, ACL_MEMCPY_HOST_TO_DEVICE);

    for (uint32_t i = 0; i < iteration; i++) {
        rdma_highlevel_put_pingpong_latency_do(1, stream, fftsConfig, gva, message_length);
    }
    aclrtSynchronizeStream(stream);
    if (rank_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_ranks,
            sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "RDMA highlevel put pingpong latency test. Message length = " << message_length
            << " Byte; latency = " << xHost[0] / ration50 << " us." << std::endl;
    }

    aclrtFreeHost(xHost);
    // shmem_finalize(): 结束并清理shmem运行环境，释放所有shmem相关资源
    // 此函数会执行以下操作:
    // 1. 释放所有未释放的对称内存资源
    // 2. 关闭rank间的网络通信连接
    // 3. 清理通信通道和同步资源
    // 4. 重置shmem运行状态
    shmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}

int test_shmem_rdma_postsend_cost(int rank_id, int n_ranks, uint64_t local_mem_size, int message_length)
{
    uint32_t iteration = 1;
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    const double ration2500 = 50.0 * 500;
    const int iterRange = 10;
    const int size6M = 6 * 1024 * 1024;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    shmem_init_attr_t *attributes;
    // shmem_set_attr(): 设置shmem初始化属性参数
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    // attributes->option_attr.data_op_engine_type: 设置数据操作引擎类型为RoCE
    attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
    // shmem_set_conf_store_tls(): 禁用TLS存储配置方式
    shmem_set_conf_store_tls(false, nullptr, 0);
    // shmem_init_attr(): 根据attributes中的配置参数初始化shmem运行环境
    status = shmem_init_attr(attributes);

    // shmemx_get_ffts_config(): 获取FFTS硬件同步配置地址
    uint64_t fftsConfig = shmemx_get_ffts_config();
    // shmem_malloc(): 从对称共享内存堆中分配指定大小的内存空间
    uint8_t *gva = static_cast<uint8_t*>(shmem_malloc(size6M));

    int64_t *xHost;
    size_t totalSize = message_length * n_ranks;

    aclrtMallocHost(reinterpret_cast<void **>(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = rank_id + iterRange;
    }
    aclrtMemcpy(gva + rank_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);

    for (uint32_t i = 0; i < iteration; i++) {
        rdma_postsend_cost_do(1, stream, fftsConfig, gva, message_length);
    }
    aclrtSynchronizeStream(stream);
    if (rank_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_ranks,
            sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "RDMA postsend cost test. Message length = " << message_length
            << " Byte; postsend cost = " << xHost[0] / ration2500 << " us." << std::endl;
    }

    aclrtFreeHost(xHost);
    // shmem_finalize(): 结束并清理shmem运行环境，释放所有shmem相关资源
    // 此函数会执行以下操作:
    // 1. 释放所有未释放的对称内存资源
    // 2. 关闭rank间的网络通信连接
    // 3. 清理通信通道和同步资源
    // 4. 重置shmem运行状态
    shmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}

int test_shmem_rdma_highlevel_put_bw(int rank_id, int n_ranks, uint64_t local_mem_size, int message_length)
{
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    const double ration50 = 50.0;
    const int iterRange = 10;
    const int size6M = 6 * 1024 * 1024;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    shmem_init_attr_t *attributes;
    // shmem_set_attr(): 设置shmem初始化属性参数
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    // attributes->option_attr.data_op_engine_type: 设置数据操作引擎类型为RoCE
    attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
    // shmem_set_conf_store_tls(): 禁用TLS存储配置方式
    shmem_set_conf_store_tls(false, nullptr, 0);
    // shmem_init_attr(): 根据attributes中的配置参数初始化shmem运行环境
    status = shmem_init_attr(attributes);

    // shmemx_get_ffts_config(): 获取FFTS硬件同步配置地址
    uint64_t fftsConfig = shmemx_get_ffts_config();
    // shmem_malloc(): 从对称共享内存堆中分配指定大小的内存空间
    uint8_t *gva = static_cast<uint8_t*>(shmem_malloc(size6M));

    int64_t *xHost;
    size_t totalSize = message_length * n_ranks;

    aclrtMallocHost(reinterpret_cast<void **>(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = rank_id + iterRange;
    }
    aclrtMemcpy(gva + rank_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);

    rdma_highlevel_put_bw_do(1, stream, fftsConfig, gva, message_length);
    aclrtSynchronizeStream(stream);
    if (rank_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_ranks, sizeof(int64_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "RDMA high level put bandwidth test. Message length = " << message_length
            << " Byte; time = " << xHost[0] / ration50 << " us." << std::endl;
    }

    aclrtFreeHost(xHost);
    // shmem_finalize(): 结束并清理shmem运行环境，释放所有shmem相关资源
    // 此函数会执行以下操作:
    // 1. 释放所有未释放的对称内存资源
    // 2. 关闭rank间的网络通信连接
    // 3. 清理通信通道和同步资源
    // 4. 重置shmem运行状态
    shmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}

int test_shmem_rdma_mte_put_bw(int rank_id, int n_ranks, uint64_t local_mem_size, int message_length)
{
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    const int size32M = 32 * 1024 * 1024;
    const int size128K = 128 * 1024;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    shmem_init_attr_t *attributes;
    // shmem_set_attr(): 设置shmem初始化属性参数
    // 参数1 rank_id: 当前进程的rank编号
    // 参数2 n_ranks: 通信组中总进程数量
    // 参数3 local_mem_size: 每个rank分配的对称内存空间大小
    // 参数4 ipport: 网络通信的IP地址和端口字符串
    // 参数5 &attributes: 输出参数，返回配置好的初始化属性结构体指针
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    // attributes->option_attr.data_op_engine_type: 设置数据操作引擎类型为RoCE
    // SHMEM_DATA_OP_ROCE表示使用RoCE(RDMA over Converged Ethernet)引擎进行数据传输
    attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
    // shmem_set_conf_store_tls(): 禁用TLS(Thread Local Storage)存储配置方式
    // 参数: false表示禁用TLS，nullptr和0表示不使用默认配置文件路径和长度
    shmem_set_conf_store_tls(false, nullptr, 0);
    // shmem_init_attr(): 根据attributes中的配置参数初始化shmem运行环境
    status = shmem_init_attr(attributes);
    // shmem_mte_set_ub_params(): 设置MTE（Memory Transfer Engine）UB（Unified Buffer）参数
    // 参数1: UB索引（0表示第一个UB）
    // 参数2: UB大小（size128K = 128 * 1024 = 128KB）
    // 参数3: 其他参数（0表示默认配置）
    // MTE是NPU的内存传输引擎，用于高性能数据传输
    // UB参数配置影响MTE的数据传输效率和缓冲区管理
    shmem_mte_set_ub_params(0, size128K, 0);

    // shmemx_get_ffts_config(): 获取FFTS(Fast Flag Task Sync)硬件同步配置地址
    // FFTS是NPU核间快速同步机制，用于在kernel执行时实现核间的轻量级同步操作
    uint64_t fftsConfig = shmemx_get_ffts_config();
    // shmem_malloc(): 从对称共享内存堆中分配指定大小的内存空间
    // 参数: size32M = 32 * 1024 * 1024 (32MB)
    uint8_t *gva = static_cast<uint8_t*>(shmem_malloc(size32M));
    int64_t *inHost;
    int64_t *outHost;
    size_t totalSize = message_length * n_ranks * 3;
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
    const int rankTimes = 2;

    for (int iter = 0; iter < maxIter; iter++) {
        for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
            inHost[i + rank_id * message_length / sizeof(int64_t)] = rank_id + iterRange + iter;
        }
        for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
            inHost[i + (rank_id + n_ranks) * message_length / sizeof(int64_t)] = rank_id + iterRange + iter;
        }
        aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE);
        shm::shmemi_control_barrier_all();
        rdma_mte_put_bw_do(1, stream, fftsConfig, gva, message_length, iter);
        aclrtSynchronizeStream(stream);
        if (rank_id == 0 && iter >= iterRange) {
            aclrtMemcpy(outHost, dstMax, gva + message_length * n_ranks * rankTimes, dstMax, ACL_MEMCPY_DEVICE_TO_HOST);
            rdmaTotalTime += outHost[0] / ration50;
            mteTotalTime += outHost[mteIdx] / ration50;
        }
    }
    if (rank_id == 0) {
        std::cout << "RDMA rdma mte test. Message length = " << message_length << " Byte; average RDMA time = "
            << rdmaTotalTime / ratio10 << " us." << std::endl;
        std::cout << "RDMA rdma mte test. Message length = " << message_length << " Byte; average MTE time = "
            << mteTotalTime / ratio10 << " us." << std::endl;
    }

    aclrtFreeHost(inHost);
    aclrtFreeHost(outHost);
    // 结束shmem环境，释放所有资源
    shmem_finalize();
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
        std::cout << "[USAGE] ./rdma_perftest <n_ranks> <rank_id> <ipport> <g_npus> <f_rank> <f_npu> "
            << "<test_type> <msg_len>. See README for more details." << std::endl;
    }
    int sub = 1;
    int status = 0;
    int n_ranks = atoi(argv[sub++]);
    const int rank_max = 2;
    if (n_ranks != rank_max) {
        std::cout << "[ERROR] Error number of ranks! Only support 2 ranks!" << std::endl;
    }
    int rank_id = atoi(argv[sub++]);
    if (rank_id >= rank_max) {
        std::cout << "[ERROR] Error rank ID! Only support 2 ranks!" << std::endl;
    }
    ipport = argv[sub++];
    g_npus = atoi(argv[sub++]);
    f_rank = atoi(argv[sub++]);
    f_npu = atoi(argv[sub++]);
    test_type = argv[sub++];
    int msg_len = atoi(argv[sub++]);
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    if (std::string(test_type) == "highlevel_put_pingpong_latency") {
        test_shmem_rdma_highlevel_put_pingpong_latency(rank_id, n_ranks, local_mem_size, msg_len);
    } else if (std::string(test_type) == "postsend_cost") {
        test_shmem_rdma_postsend_cost(rank_id, n_ranks, local_mem_size, msg_len);
    } else if (std::string(test_type) == "highlevel_put_bw") {
        test_shmem_rdma_highlevel_put_bw(rank_id, n_ranks, local_mem_size, msg_len);
    } else if (std::string(test_type) == "rdma_mte_bw") {
        test_shmem_rdma_mte_put_bw(rank_id, n_ranks, local_mem_size, msg_len);
    }

    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;

    return 0;
}