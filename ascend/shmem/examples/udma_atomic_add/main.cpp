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
const char *ipport;
int f_pe = 0;
int f_npu = 0;
constexpr uint64_t DEBUG_DUMP_SIZE = 200 * 1024 * 1024;
extern void launch_udma_atomic_add(uint32_t block_dim, void *stream, uint8_t *gva, uint8_t *dump, int message_length);

aclshmemx_uniqueid_t default_flag_uid;

int test_aclshmem_udma_atomic_add(int pe_id, int n_pes, uint64_t local_mem_size)
{
    // Initialize ACL and ACLSHMEM.
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    const int num10 = 10;
    aclrtStream stream = nullptr;

    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);

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

    // ACLSHMEM_DATA_OP_UDMA: 设置数据传输引擎为UDMA
    // UDMA引擎特点：
    // - 高性能片上互联通信引擎
    // - 支持Atomic操作（Atomic Add等）
    // - 高带宽、低延迟
    // - 适合原子操作场景
    // Atomic Add执行流程：
    // 1. 每个PE向目标地址发送原子加法操作
    // 2. 目标地址的值被原子性地增加
    // 3. 所有PE完成后，目标地址值为所有PE贡献之和
    // 其他可选引擎类型:
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
    // UDMA Atomic Add场景中的作用：
    // 1. 建立所有PE之间的通信通道
    // 2. 分配对称内存堆，用于Atomic操作数据
    // 3. 初始化UDMA通信引擎（支持Atomic操作）
    // 4. 设置pe_id和n_pes信息
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    // aclshmem_malloc: 分配对称内存，用于Atomic Add操作的目标地址
    // 参数详解:
    // - 1024: 对称内存大小（字节）
    // 返回值: 对称内存指针（GVA格式）
    // Atomic Add场景中对称内存的用途：
    // - 存存Atomic Add操作的目标值
    // - 所有PE向同一个地址进行原子加法操作
    // - 结果为所有PE的贡献之和
    uint8_t *ptr = static_cast<uint8_t *>(aclshmem_malloc(1024));

    // Initialize input data.
    uint32_t trans_size = 1;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = pe_id + num10;
    }

    status |= aclrtMemcpy(ptr, trans_size * sizeof(int32_t), input.data(), trans_size * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    uint8_t *dump = nullptr;
#if defined(ENABLE_ASCENDC_DUMP)
    (void)aclrtMalloc((void **)&dump, DEBUG_DUMP_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (dump == nullptr) {
        std::cout << "dump workspace is nullptr" << std::endl;
        return -1;
    }
#endif
    // Launch the udma atomic add kernel.
    launch_udma_atomic_add(1, stream, (uint8_t *)ptr, dump, trans_size * sizeof(int32_t));
    status |= aclrtSynchronizeStream(stream);

    // aclshmem_barrier_all: 全局屏同步，确保所有PE的Atomic Add操作完成
    // 功能详解：
    // - 所有PE都调用此函数后才能继续执行
    // - 确保所有PE的Atomic Add操作都已完成
    // - 用于结果校验前的同步
    // Atomic Add场景中的作用：
    // - 确保所有PE的原子加法操作都已完成
    // - 保证最终结果的正确性
    // - 防止数据竞争和结果不一致
    aclshmem_barrier_all();
#if defined(ENABLE_ASCENDC_DUMP)
    Adx::AdumpPrintWorkSpace(dump, DEBUG_DUMP_SIZE, stream, "udma_atomic_add");
#endif
    // Copy back and validate the result.
    int32_t *y_host;
    size_t input_size = trans_size * sizeof(int32_t);
    status |= aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size);
    status |= aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST);

    if (y_host[0] != (pe_id + num10 * 2)) {
        std::cout << "pe" << pe_id << ": " << y_host[0] << " != " << (pe_id + num10 * 2) << std::endl;
        status |= -1;
    } else {
        std::cout << "check transport result success, pe=" << pe_id << std::endl;
    }

    // Release resources.
    status |= aclrtFreeHost(y_host);

    // aclshmem_free: 释放对称内存
    // 参数: aclshmem_malloc返回的对称内存指针（ptr）
    // 必须与aclshmem_malloc配对使用
    // 执行效果:
    // - 将对称内存归还到Symmetric Heap
    // - 其他shmem操作可以重新分配此内存
    // - 释放后该地址不再可用于通信
    // 重要提示：
    // 1. 不能使用aclrtFree释放对称内存
    // 2. 释放前确保Atomic Add操作已完成
    aclshmem_free(ptr);

    // aclshmem_finalize: 终止shmem运行时，释放所有shmem资源
    // 功能详解：
    // - 释放对称内存堆
    // - 关闭进程间通信通道
    // - 清理UDMA通信引擎状态
    // - 释放Atomic操作相关资源
    // 返回值: ACLSHMEM_SUCCESS表示成功
    status |= aclshmem_finalize();
    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();
    return status;
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
    status = test_aclshmem_udma_atomic_add(pe_id, n_pes, local_mem_size);
    if (status == 0) {
        std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl; 
    }
    return status;
}
