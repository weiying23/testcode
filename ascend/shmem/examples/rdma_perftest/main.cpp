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
 * @brief RDMA Performance Test - 跨节点RDMA性能测试程序
 *
 * 此程序测试RDMA引擎（RoCE网络）的性能指标：
 * 1. Pingpong延迟：测量RDMA通信的往返延迟
 * 2. Postsend开销：测量RDMA发送操作本身的开销
 * 3. High-level Put带宽：测量RDMA Put操作的吞吐量
 * 4. RDMA/MTE带宽对比：对比RDMA（跨节点）和MTE（节点内）的性能
 *
 * WHY测试这些性能指标：
 * - 延迟是分布式训练同步操作的关键性能指标
 * - 带宽衡量数据传输吞吐量，影响大规模模型训练效率
 * - 对比RDMA和MTE帮助选择合适的通信引擎
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

/**
 * @brief RDMA Pingpong延迟测试
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
 * @param pe_id 当前PE编号
 * @param n_pes 总PE数量
 * @param local_mem_size 对称内存大小
 * @param message_length 消息长度（字节）
 * @return 0表示成功
 */
int test_aclshmem_rdma_highlevel_put_pingpong_latency(int pe_id, int n_pes, uint64_t local_mem_size, int message_length)
{
    uint32_t iteration = 1;

    // 计算物理设备ID：pe_id % g_npus + f_npu
    // WHY这样计算：
    // - pe_id % g_npus：将逻辑PE编号映射到节点内NPU编号
    // - f_npu：物理设备编号偏移量
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    const double ration50 = 50.0;  // WHY ration50 = 50.0：将硬件周期转换为微秒
    const int times32 = 32;
    const int iterRange = 10;
    const int size6M = 6 * 1024 * 1024;

    // ========== ACL初始化 ==========
    // aclInit: 初始化ACL运行时环境
    // WHY必须首先执行：后续所有ACL API依赖此初始化
    status = aclInit(nullptr);

    // aclrtSetDevice: 设置当前进程使用的NPU设备
    // WHY需要设置设备：绑定进程到指定NPU
    status = aclrtSetDevice(device_id);

    // aclrtCreateStream: 创建ACL流
    // WHY需要流：管理kernel执行顺序
    status = aclrtCreateStream(&stream);

    // ========== Shmem初始化 ==========
    // aclshmemx_init_attr_t: shmem初始化属性结构体
    // WHY需要此结构体：配置PE编号、引擎类型等参数
    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // ACLSHMEM_DATA_OP_ROCE: 设置数据传输引擎为RDMA
    // WHY使用RDMA：
    // - Pingpong latency测试使用RDMA引擎测量跨节点往返延迟
    // - RDMA支持跨节点NPU间通信
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;

    // aclshmemx_set_conf_store_tls: 设置配置存储模式
    // WHY参数false：不使用TLS配置存储
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    // aclshmemx_init_attr: 初始化shmem运行时
    // WHY使用ACLSHMEMX_INIT_WITH_DEFAULT：TCP socket模式，不需要MPI
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    // util_get_ffts_config: 获取FFTS配置地址
    // WHY需要FFTS：硬件同步配置，用于kernel内部同步
    uint64_t fftsConfig = util_get_ffts_config();

    // ========== 对称内存分配 ==========
    // aclshmem_malloc: 分配对称内存（用于Pingpong测试数据）
    // WHY参数size6M = 6MB：
    // - 存储源数据、目标数据和同步标志
    // - 提供足够的缓冲区空间
    uint8_t *gva = static_cast<uint8_t*>(aclshmem_malloc(size6M));

    int64_t *xHost;
    size_t totalSize = message_length * n_pes;

    // aclrtMallocHost: 在Host端分配内存用于初始化数据
    aclrtMallocHost(reinterpret_cast<void **>(&xHost), totalSize);

    // WHY初始化数据值为pe_id + iterRange：
    // - 用于验证数据传输的正确性
    // - 每个PE的数据有唯一标识
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = pe_id + iterRange;
    }

    // aclrtMemcpy: 将Host端数据拷贝到Device端对称内存
    // 数据布局：PE i的数据位于 gva + i * message_length
    aclrtMemcpy(gva + pe_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);

    // WHY拷贝同步标志数据：用于Pingpong轮询等待
    aclrtMemcpy(gva + n_pes * message_length + times32 * (pe_id + 1), times32,
        xHost, times32, ACL_MEMCPY_HOST_TO_DEVICE);

    // ========== RDMA Pingpong Latency测试执行 ==========
    // rdma_highlevel_put_pingpong_latency_do: 执行RDMA Pingpong延迟测试kernel
    // WHY执行Pingpong测试：
    // - 测量RDMA通信的往返延迟
    // - 用于评估跨节点通信性能
    for (uint32_t i = 0; i < iteration; i++) {
        rdma_highlevel_put_pingpong_latency_do(1, stream, fftsConfig, gva, message_length);
    }
    aclrtSynchronizeStream(stream);

    // ========== 结果输出 ==========
    // WHY仅PE 0输出结果：PE 0测量并存储了延迟数据
    if (pe_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_pes,
            sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

        // 延迟计算：xHost[0] / ration50 us
        // WHY ration50 = 50.0：硬件周期转换为微秒（1us = 50 cycles）
        std::cout << "RDMA highlevel put pingpong latency test. Message length = " << message_length
            << " Byte; latency = " << xHost[0] / ration50 << " us." << std::endl;
    }

    // ========== 资源释放 ==========
    aclrtFreeHost(xHost);
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}

/**
 * @brief RDMA Postsend开销测试
 *
 * WHY测试Postsend开销：
 * - Postsend开销衡量RDMA发送操作本身的时间
 * - 用于评估单次RDMA Put的硬件开销
 * - 不包括数据传输时间，仅测量任务下发开销
 *
 * @param pe_id 当前PE编号
 * @param n_pes 总PE数量
 * @param local_mem_size 对称内存大小
 * @param message_length 消息长度
 * @return 0表示成功
 */
int test_aclshmem_rdma_postsend_cost(int pe_id, int n_pes, uint64_t local_mem_size, int message_length)
{
    uint32_t iteration = 1;
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;

    // ration2500 = 50.0 * 500：用于将500次循环的总周期转换为平均微秒
    // WHY乘以500：测试循环500次，需要计算平均值
    const double ration2500 = 50.0 * 500;
    const int iterRange = 10;
    const int size6M = 6 * 1024 * 1024;

    // ========== ACL初始化 ==========
    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    // ========== Shmem初始化 ==========
    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // ACLSHMEM_DATA_OP_ROCE: 设置数据传输引擎为RDMA
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    uint64_t fftsConfig = util_get_ffts_config();
    uint8_t *gva = static_cast<uint8_t*>(aclshmem_malloc(size6M));

    int64_t *xHost;
    size_t totalSize = message_length * n_pes;

    aclrtMallocHost(reinterpret_cast<void **>(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = pe_id + iterRange;
    }
    aclrtMemcpy(gva + pe_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);

    // ========== RDMA Postsend Cost测试执行 ==========
    // rdma_postsend_cost_do: 执行RDMA Postsend开销测试kernel
    // WHY执行Postsend测试：
    // - 测量RDMA发送操作本身的开销
    // - 用于评估单次RDMA Put的硬件开销
    for (uint32_t i = 0; i < iteration; i++) {
        rdma_postsend_cost_do(1, stream, fftsConfig, gva, message_length);
    }
    aclrtSynchronizeStream(stream);

    // WHY仅PE 0输出结果：PE 0测量并存储了Postsend开销数据
    if (pe_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_pes,
            sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

        // WHY除以ration2500：500次循环的平均Postsend开销
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

/**
 * @brief RDMA带宽测试
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
 * @param pe_id 当前PE编号
 * @param n_pes 总PE数量
 * @param local_mem_size 对称内存大小
 * @param message_length 消息长度
 * @return 0表示成功
 */
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

    // ========== Shmem初始化 ==========
    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // ACLSHMEM_DATA_OP_ROCE: 设置数据传输引擎为RDMA
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    uint64_t fftsConfig = util_get_ffts_config();
    uint8_t *gva = static_cast<uint8_t*>(aclshmem_malloc(size6M));

    int64_t *xHost;
    size_t totalSize = message_length * n_pes;

    aclrtMallocHost(reinterpret_cast<void **>(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = pe_id + iterRange;
    }
    aclrtMemcpy(gva + pe_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);

    // ========== RDMA带宽测试执行 ==========
    // rdma_highlevel_put_bw_do: 执行RDMA带宽测试kernel
    // WHY执行带宽测试：测量RDMA Put操作的吞吐量
    rdma_highlevel_put_bw_do(1, stream, fftsConfig, gva, message_length);
    aclrtSynchronizeStream(stream);

    if (pe_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_pes, sizeof(int64_t),
            ACL_MEMCPY_DEVICE_TO_HOST);

        // WHY输出时间而非带宽：用户可根据message_length和time自行计算带宽
        // 带宽 = (10000 * message_length) / time
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

/**
 * @brief RDMA和MTE带宽对比测试
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
 * @param pe_id 当前PE编号
 * @param n_pes 总PE数量
 * @param local_mem_size 对称内存大小
 * @param message_length 消息长度
 * @return 0表示成功
 */
int test_aclshmem_rdma_mte_put_bw(int pe_id, int n_pes, uint64_t local_mem_size, int message_length)
{
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;

    // WHY参数size32M：RDMA和MTE需要更大的缓冲区
    const int size32M = 32 * 1024 * 1024;
    const int size128K = 128 * 1024;

    // ========== ACL初始化 ==========
    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    // ========== Shmem初始化 ==========
    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // ACLSHMEM_DATA_OP_ROCE: 设置数据传输引擎为RDMA
    // WHY使用RDMA：此测试对比RDMA和MTE两种引擎的带宽性能
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    uint64_t fftsConfig = util_get_ffts_config();

    // aclshmem_malloc: 分配对称内存（用于RDMA/MTE对比带宽测试）
    // WHY参数size32M = 32MB：
    // - 存储RDMA测试数据、MTE测试数据和同步标志
    // - 提供足够的缓冲区空间进行10000次迭代
    uint8_t *gva = static_cast<uint8_t*>(aclshmem_malloc(size32M));

    int64_t *inHost;
    int64_t *outHost;
    size_t totalSize = message_length * n_pes * 3;  // WHY *3：RDMA数据 + MTE数据 + 结果存储

    aclrtMallocHost(reinterpret_cast<void **>(&inHost), totalSize);
    aclrtMallocHost(reinterpret_cast<void **>(&outHost), totalSize);
    bzero(inHost, totalSize);

    double rdmaTotalTime = 0.0;
    double mteTotalTime = 0.0;
    const int mteIdx = 6;         // WHY mteIdx = 6：MTE结果存储在offset 48字节处
    const double ratio10 = 10.0;  // WHY ratio10 = 10.0：跳过前10次迭代预热，计算10次平均值
    const double ration50 = 50.0;
    const int dstMax = 64;
    const int iterRange = 10;     // WHY iterRange = 10：预热迭代次数
    const int maxIter = 20;       // WHY maxIter = 20：总迭代次数
    const int peTimes = 2;

    // ========== RDMA/MTE带宽对比测试执行 ==========
    // WHY循环maxIter = 20次：
    // - 前10次预热，消除冷启动影响
    // - 后10次用于计算平均性能
    for (int iter = 0; iter < maxIter; iter++) {
        // WHY每轮迭代更新数据值：避免数据相同导致缓存命中
        for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
            inHost[i + pe_id * message_length / sizeof(int64_t)] = pe_id + iterRange + iter;
        }
        for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
            inHost[i + (pe_id + n_pes) * message_length / sizeof(int64_t)] = pe_id + iterRange + iter;
        }
        aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE);

        // aclshmemi_control_barrier_all: 内部屏障同步
        // WHY需要屏障：确保所有PE数据初始化完成后再开始测试
        aclshmemi_control_barrier_all();

        // rdma_mte_put_bw_do: 执行RDMA和MTE带宽对比测试kernel
        // WHY参数iter：用于区分不同轮次的测试，避免同步标志冲突
        rdma_mte_put_bw_do(1, stream, fftsConfig, gva, message_length, iter);
        aclrtSynchronizeStream(stream);

        // WHY仅PE 0且iter >= iterRange收集结果：
        // - PE 0存储了RDMA和MTE的时间数据
        // - 跳过前iterRange次迭代的预热
        if (pe_id == 0 && iter >= iterRange) {
            aclrtMemcpy(outHost, dstMax, gva + message_length * n_pes * peTimes, dstMax, ACL_MEMCPY_DEVICE_TO_HOST);
            rdmaTotalTime += outHost[0] / ration50;       // RDMA时间
            mteTotalTime += outHost[mteIdx] / ration50;   // MTE时间
        }
    }

    // ========== 结果输出 ==========
    if (pe_id == 0) {
        // WHY输出平均时间：后10次迭代的平均值
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

/**
 * @brief 主函数
 *
 * 参数详解：
 * - argv[1]: n_pes - 总PE数量（必须为2）
 * - argv[2]: pe_id - 当前PE编号（必须为0或1）
 * - argv[3]: ipport - rendezvous地址
 * - argv[4]: g_npus - 节点内NPU总数
 * - argv[5]: f_pe - PE编号偏移量
 * - argv[6]: f_npu - NPU编号偏移量
 * - argv[7]: test_type - 测试类型
 *   - "highlevel_put_pingpong_latency": Pingpong延迟测试
 *   - "postsend_cost": Postsend开销测试
 *   - "highlevel_put_bw": 带宽测试
 *   - "rdma_mte_bw": RDMA/MTE对比测试
 * - argv[8]: msg_len - 消息长度（字节）
 *
 * WHY仅支持2个PE：
 * - Pingpong和对比测试需要两个PE参与
 * - 简化测试逻辑，专注于性能测量
 */
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

    // WHY仅支持2个PE：Pingpong和对比测试需要两个PE参与
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

    // 对称内存大小：64MB
    // WHY设置64MB：提供足够的对称内存空间用于性能测试
    uint64_t local_mem_size = 1024UL * 1024UL * 64;

    // 根据test_type选择测试类型
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