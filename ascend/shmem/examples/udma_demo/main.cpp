/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

int g_npus = 8;
const char* ipport;
int f_pe = 0;
int f_npu = 0;
constexpr uint64_t DEBUG_DUMP_SIZE = 200 * 1024 * 1024;
extern void launch_udma_all_gather(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dump, int message_length);
extern void launch_udma_put_signal(
    uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr, uint8_t* dump_addr, int elements,
    uint64_t signal);

aclshmemx_uniqueid_t default_flag_uid;

// Common initialization function
int init_acl_shmem(
    int pe_id, int n_pes, uint64_t local_mem_size, int32_t& device_id, aclrtStream& stream, uint8_t*& ptr)
{
    int status = 0;
    device_id = pe_id % g_npus + f_npu;

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
    // - ipport: rendezvous地址字符串
    // - default_flag_uid: uniqueid结构体
    // - &attributes: 属性结构体指针（输出参数）
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // ACLSHMEM_DATA_OP_UDMA: 设置数据传输引擎为UDMA
    // UDMA引擎特点：
    // - 高性能片上互联通信引擎
    // - 支持AllGather等集合操作
    // - 高带宽、低延迟
    // - 适合大规模数据传输
    // 与其他引擎对比:
    // - ACLSHMEM_DATA_OP_MTE: MTE引擎（片上互联，节点内）
    // - ACLSHMEM_DATA_OP_SDMA: SDMA引擎（片上SDMA单元，节点内）
    // - ACLSHMEM_DATA_OP_ROCE: RDMA引擎（RoCE网络，跨节点）
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;

    // aclshmemx_init_attr: 初始化shmem运行时（默认socket模式）
    // 参数详解:
    // - ACLSHMEMX_INIT_WITH_DEFAULT: 初始化模式标志
    //   使用TCP socket进行进程间rendezvous
    // - &attributes: 初始化属性结构体指针
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 执行后完成:
    // 1. 建立进程间通信通道
    // 2. 分配对称内存堆
    // 3. 初始化UDMA通信引擎
    // 4. 设置PE编号和通信组信息
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    // aclshmem_malloc: 分配对称内存，用于存放通信数据
    // 参数详解:
    // - 1024: 对称内存大小（字节）
    // 返回值: 对称内存指针（GVA格式）
    // 对称内存核心特点：
    // 1. 所有PE在同一虚拟地址上拥有相同大小的内存块
    // 2. PE i可以直接通过GVA地址访问PE j的数据
    // 3. 用于存放通信数据和同步标志
    // 注意：
    // - 必须通过aclshmem_free释放
    // - 分配大小不能超过初始化时设置的local_mem_size
    ptr = static_cast<uint8_t*>(aclshmem_malloc(1024));
    return status;
}

// Common data initialization function
void init_data(int pe_id, uint8_t* ptr, uint32_t trans_size)
{
    const int num10 = 10;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = (pe_id + num10);
    }

    aclrtMemcpy(
        ptr + aclshmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t), input.data(),
        trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
}

// Common result validation function
bool validate_result(uint8_t* ptr, int n_pes, uint32_t trans_size)
{
    const int num10 = 10;
    int32_t* y_host;
    size_t input_size = n_pes * trans_size * sizeof(int32_t);
    aclrtMallocHost(reinterpret_cast<void**>(&y_host), input_size);
    aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST);

    const int block_size = 16;
    bool success = true;
    for (int i = 0; i < n_pes; i++) {
        for (int j = 0; j < block_size; j++) {
            if (y_host[trans_size * i + trans_size / block_size * j] != num10 + i) {
                std::cout << y_host[trans_size * i + trans_size / block_size * j] << " != " << num10 + i << std::endl;
                success = false;
            }
        }
    }

    aclrtFreeHost(y_host);
    return success;
}

// Common cleanup function
int cleanup_resources(aclrtStream stream, int32_t device_id, uint8_t* ptr, uint8_t* extra_ptr = nullptr)
{
    int status = 0;
    // aclshmem_free: 释放对称内存
    // 参数: aclshmem_malloc返回的对称内存指针
    // 必须与aclshmem_malloc配对使用
    // 执行效果:
    // - 将对称内存归还到Symmetric Heap
    // - 其他shmem操作可以重新分配此内存
    // - 释放后该地址不再可用于通信
    // 重要提示：
    // 1. 不能使用aclrtFree释放对称内存
    // 2. 所有PE应同时释放对称内存
    aclshmem_free(ptr);
    if (extra_ptr != nullptr) {
        // aclshmem_free: 释放额外的对称内存
        // 用于释放信号地址等额外分配的对称内存
        aclshmem_free(extra_ptr);
    }
    // aclshmem_finalize: 终止shmem运行时
    // 功能详解：
    // - 释放对称内存堆
    // - 关闭进程间通信通道
    // - 清理UDMA通信引擎状态
    // - 释放内部同步机制资源
    // 返回值: ACLSHMEM_SUCCESS表示成功
    status |= aclshmem_finalize();
    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();
    return status;
}

int test_aclshmem_team_all_gather(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int status = 0;
    int32_t device_id;
    aclrtStream stream = nullptr;
    uint8_t* ptr = nullptr;

    status = init_acl_shmem(pe_id, n_pes, local_mem_size, device_id, stream, ptr);
    if (status != 0) {
        return status;
    }

    uint32_t trans_size = 16;
    init_data(pe_id, ptr, trans_size);

    uint8_t* dump = nullptr;
#if defined(ENABLE_ASCENDC_DUMP)
    (void)aclrtMalloc((void**)&dump, DEBUG_DUMP_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (dump == nullptr) {
        std::cout << "dump workspace is nullptr" << std::endl;
        return -1;
    }
#endif
    // Launch the all-gather kernel.
    launch_udma_all_gather(1, stream, (uint8_t*)ptr, dump, trans_size * sizeof(int32_t));
    status |= aclrtSynchronizeStream(stream);
#if defined(ENABLE_ASCENDC_DUMP)
    Adx::AdumpPrintWorkSpace(dump, DEBUG_DUMP_SIZE, stream, "udma_demo");
#endif

    if (validate_result(ptr, n_pes, trans_size)) {
        std::cout << "check transport result success, pe=" << pe_id << std::endl;
    } else {
        std::cout << "check transport result failed, pe=" << pe_id << std::endl;
        cleanup_resources(stream, device_id, ptr);
        return -1;
    }

    return cleanup_resources(stream, device_id, ptr);
}

int test_aclshmem_udma_put_signal(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int status = 0;
    int32_t device_id;
    aclrtStream stream = nullptr;
    uint8_t* ptr = nullptr;

    status = init_acl_shmem(pe_id, n_pes, local_mem_size, device_id, stream, ptr);
    if (status != 0) {
        return status;
    }

    // aclshmem_malloc: 分配用于存放信号的对称内存
    // 每个PE需要n_pes * sizeof(uint64_t)的空间来存储所有PE的信号值
    // 参数详解:
    // - n_pes * sizeof(uint64_t): 对称内存大小
    //   为每个PE预留一个uint64_t空间存储信号值
    // 返回值: 对称内存指针（GVA格式）
    // 信号地址用途：
    // - 存存PutSignal操作的信号值
    // - 用于异步操作的同步通知
    // - 每个PE可以从其他PE的信号地址读取信号值
    uint8_t* sig_addr = static_cast<uint8_t*>(aclshmem_malloc(n_pes * sizeof(uint64_t)));
    // Initialize signal addresses to 0 to avoid dirty data
    std::vector<uint64_t> init_signals(n_pes, 0);
    aclrtMemcpy(
        sig_addr, n_pes * sizeof(uint64_t), init_signals.data(), n_pes * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // Allocate dump workspace
    uint8_t* dump = nullptr;
#if defined(ENABLE_ASCENDC_DUMP)
    (void)aclrtMalloc((void**)&dump, DEBUG_DUMP_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (dump == nullptr) {
        std::cout << "dump workspace is nullptr" << std::endl;
        return -1;
    }
#endif

    uint32_t trans_size = 16;
    init_data(pe_id, ptr, trans_size);

    // Launch the put signal kernel.
    uint64_t signal = 1000;
    launch_udma_put_signal(1, stream, (uint8_t*)ptr, sig_addr, dump, trans_size * sizeof(int32_t), signal);
    status |= aclrtSynchronizeStream(stream);

    if (validate_result(ptr, n_pes, trans_size)) {
        std::cout << "check udma put signal result success, pe=" << pe_id << std::endl;
    } else {
        std::cout << "check udma put signal result failed, pe=" << pe_id << std::endl;
        cleanup_resources(stream, device_id, ptr, sig_addr);
        return -1;
    }

    // Read and validate all signals
    std::vector<uint64_t> signal_values(n_pes, 0);
    status |= aclrtMemcpy(
        signal_values.data(), n_pes * sizeof(uint64_t), sig_addr, n_pes * sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "Signal values received on pe=" << pe_id << ":" << std::endl;
    bool all_signals_set = true;
    for (int i = 0; i < n_pes; i++) {
        std::cout << "  PE " << i << ": " << signal_values[i] << std::endl;
        if (i == pe_id) {
            continue;
        }
        if (signal_values[i] != signal) {
            all_signals_set = false;
        }
    }

    // Check if all signals are set
    if (!all_signals_set) {
        std::cout << "[ERROR]: Some signals not equal " << signal << ", may not have been set properly" << std::endl;
        return -1;
    } else {
        std::cout << "All signals are set successfully" << std::endl;
    }

    // Free dump workspace
#if defined(ENABLE_ASCENDC_DUMP)
    if (dump != nullptr) {
        aclrtFree(dump);
    }
#endif
    return cleanup_resources(stream, device_id, ptr, sig_addr);
}

int main(int argc, char* argv[])
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

    // Default to run all-gather test if no test type specified
    int test_type = 0;
    if (argIdx < argc) {
        test_type = atoi(argv[argIdx++]);
    }

    if (test_type == 1) {
        status = test_aclshmem_udma_put_signal(pe_id, n_pes, local_mem_size);
    } else {
        status = test_aclshmem_team_all_gather(pe_id, n_pes, local_mem_size);
    }

    std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    return 0;
}
