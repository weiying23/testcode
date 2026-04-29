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
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <iomanip>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <algorithm>

#include "opdev/fp16_t.h"
#include "opdev/bfloat16.h"
#include "utils.h"
#include "param.h"

using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;

#include "acl/acl.h"
#include "shmem.h"
#include "kv_shuffle_kernel.h"
int g_npus = 8;
const char *ipport;
int f_pe = 0;
int f_npu = 0;
const char *data_type;

constexpr int64_t MAX_SEQLEN = 1024;
constexpr int64_t MAX_BATCH = 10;
constexpr int64_t page_size = 128;
constexpr int64_t max_block_nums = MAX_SEQLEN * MAX_BATCH / page_size;
constexpr int64_t kv_head_num = 8;
constexpr int64_t head_dim = 128;

int test_aclshmem_kv_shuffle(int pe_id, int n_pes)
{
    // ACLStream init
    int status = 0;
    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);

    uint32_t BLOCK_NUM = 16;

    int64_t kv_cache_size = max_block_nums * kv_head_num * page_size * head_dim * sizeof(int8_t);

    std::string inputFile;
    // k_cache input
    uint8_t *k_cache_host;
    aclrtMallocHost(reinterpret_cast<void **>(&k_cache_host), kv_cache_size);
    inputFile = "../../examples/kv_shuffle/scripts/output/k_cache_input_pe_" + std::to_string(pe_id) + ".bin";
    ReadFile(inputFile, k_cache_host, kv_cache_size);
    // aclshmem_malloc: 分配对称内存（用于KV Cache数据通信）
    // KV Shuffle需要在对称内存中存储KV Cache以便跨PE传输
    // 参数详解:
    // - kv_cache_size: KV Cache数据大小
    //   max_block_nums * kv_head_num * page_size * head_dim * sizeof(int8_t)
    // 返回值: 对称内存指针（GVA格式）
    // KV Shuffle场景中对称内存的用途：
    // - 存存KV Cache数据（Key和Value）
    // - 用于跨PE的KV Cache传输和重组
    // - KV Shuffle是LLM推理中的关键操作，用于处理动态batch和sequence
    // KV Shuffle执行流程：
    // 1. 根据global_shuffle_table确定KV Cache的传输目标
    // 2. 将本地KV Cache发送到目标PE
    // 3. 从其他PE接收KV Cache
    // 4. 根据block_table重组KV Cache
    // 对称内存核心特点：
    // 1. 所有PE在同一虚拟地址上拥有相同大小的内存块
    // 2. PE i可以直接通过GVA地址访问PE j的KV Cache
    // 3. 用于存放通信数据和同步标志
    // 注意：
    // - 必须通过aclshmem_free释放
    // - KV Cache大小可能很大，需要合理规划内存
    void *k_cache_ptr = aclshmem_malloc(kv_cache_size);
    aclrtMemcpy(k_cache_ptr, kv_cache_size, k_cache_host, kv_cache_size, ACL_MEMCPY_HOST_TO_DEVICE);

    // v_cache input
    uint8_t *v_cache_host;
    aclrtMallocHost(reinterpret_cast<void **>(&v_cache_host), kv_cache_size);
    inputFile = "../../examples/kv_shuffle/scripts/output/v_cache_input_pe_" + std::to_string(pe_id) + ".bin";
    ReadFile(inputFile, v_cache_host, kv_cache_size);
    // aclshmem_malloc: 分配对称内存（用于V Cache数据）
    // 参数详解:
    // - kv_cache_size: V Cache数据大小（与K Cache相同大小）
    // 返回值: 对称内存指针（GVA格式）
    // V Cache用途：与K Cache类似，用于存储Value向量
    void *v_cache_ptr = aclshmem_malloc(kv_cache_size);
    aclrtMemcpy(v_cache_ptr, kv_cache_size, v_cache_host, kv_cache_size, ACL_MEMCPY_HOST_TO_DEVICE);

    // global_shuffle_table input
    uint8_t *global_shuffle_table_host;
    constexpr uint32_t PAIR_PER_PE = 2;
    aclrtMallocHost(reinterpret_cast<void **>(&global_shuffle_table_host), n_pes * PAIR_PER_PE * sizeof(int64_t));
    inputFile = "../../examples/kv_shuffle/scripts/output/pair_list.bin";
    ReadFile(inputFile, global_shuffle_table_host, n_pes * PAIR_PER_PE * sizeof(int64_t));
    void *global_shuffle_table_ptr;
    aclrtMalloc(&global_shuffle_table_ptr, n_pes * PAIR_PER_PE * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(global_shuffle_table_ptr, n_pes * PAIR_PER_PE * sizeof(int64_t),
                global_shuffle_table_host, n_pes * PAIR_PER_PE * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // global_block_num input
    uint8_t *global_block_num_host;
    aclrtMallocHost(reinterpret_cast<void **>(&global_block_num_host), sizeof(int64_t));
    inputFile = "../../examples/kv_shuffle/scripts/output/block_num_pe_" + std::to_string(pe_id) + ".bin";
    ReadFile(inputFile, global_block_num_host, sizeof(int64_t));
    void *global_block_num_ptr;
    aclrtMalloc(&global_block_num_ptr, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(global_block_num_ptr, sizeof(int64_t), global_block_num_host,
                sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);

    const int64_t block_nums = *reinterpret_cast<int64_t *>(global_block_num_host);

    // src_block_table input
    uint8_t *src_block_table_host;
    void *src_block_table_ptr;
    if (block_nums != 0) {
        aclrtMallocHost(reinterpret_cast<void **>(&src_block_table_host), block_nums * sizeof(int64_t));
        inputFile = "../../examples/kv_shuffle/scripts/output/src_block_table_pe_" + std::to_string(pe_id) + ".bin";

        aclrtMalloc(&src_block_table_ptr, block_nums * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
        ReadFile(inputFile, src_block_table_host, block_nums * sizeof(int64_t));
        aclrtMemcpy(src_block_table_ptr, block_nums * sizeof(int64_t),
                    src_block_table_host, block_nums * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        std::cout << "Relative pe " << pe_id << " block_nums = 0, Skip src_block_table input" << std::endl;
    }

    // dst_block_table input
    uint8_t *dst_block_table_host;
    void *dst_block_table_ptr;
    if (block_nums != 0) {
        aclrtMallocHost(reinterpret_cast<void **>(&dst_block_table_host), block_nums * sizeof(int64_t));
        inputFile = "../../examples/kv_shuffle/scripts/output/dst_block_table_pe_" + std::to_string(pe_id) + ".bin";

        aclrtMalloc(&dst_block_table_ptr, block_nums * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
        ReadFile(inputFile, dst_block_table_host, block_nums * sizeof(int64_t));
        aclrtMemcpy(dst_block_table_ptr, block_nums * sizeof(int64_t),
                    dst_block_table_host, block_nums * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        std::cout << "Relative pe " << pe_id << " block_nums = 0, Skip dst_block_table input" << std::endl;
    }

    // KVShuffle
    KVShuffleOps ops(BLOCK_NUM, stream);
    int PERF_TIMES = 10;
    for (int zz = 0; zz < PERF_TIMES; zz++) {
        ops.compute(
            (uint8_t *)k_cache_ptr,
            (uint8_t *)v_cache_ptr,
            (uint8_t *)global_shuffle_table_ptr,
            (uint8_t *)src_block_table_ptr,
            (uint8_t *)dst_block_table_ptr,
            block_nums,
            kv_head_num, page_size, head_dim);
    }
    status = aclrtSynchronizeStream(stream);

    // Result Check
    std::string outputFile;
    int8_t *k_output_host;
    status = aclrtMallocHost(reinterpret_cast<void**>(&k_output_host), kv_cache_size);
    status = aclrtMemcpy(k_output_host, kv_cache_size, k_cache_ptr, kv_cache_size, ACL_MEMCPY_DEVICE_TO_HOST);
    outputFile = "../../examples/kv_shuffle/scripts/output/k_cache_output_pe_" + std::to_string(pe_id) + ".bin";
    WriteFile(outputFile, k_output_host, kv_cache_size);

    int8_t *v_output_host;
    status = aclrtMallocHost(reinterpret_cast<void**>(&v_output_host), kv_cache_size);
    status = aclrtMemcpy(v_output_host, kv_cache_size, v_cache_ptr, kv_cache_size, ACL_MEMCPY_DEVICE_TO_HOST);
    outputFile = "../../examples/kv_shuffle/scripts/output/v_cache_output_pe_" + std::to_string(pe_id) + ".bin";
    WriteFile(outputFile, v_output_host, kv_cache_size);

    // aclshmem_free: 释放对称内存（KV Cache）
    // 参数: aclshmem_malloc返回的对称内存指针
    // 必须与aclshmem_malloc配对使用
    // 执行效果:
    // - 将KV Cache对称内存归还到Symmetric Heap
    // - 其他shmem操作可以重新分配此内存
    // - 释放后该地址不再可用于通信
    // 重要提示：
    // 1. 不能使用aclrtFree释放对称内存
    // 2. KV Cache可能占用大量内存，释放后及时回收
    // 3. 释放前确保KV Shuffle操作已完成
    aclshmem_free(k_cache_ptr);
    aclshmem_free(v_cache_ptr);
    aclrtFree(global_shuffle_table_ptr);
    aclrtFree(global_block_num_ptr);
    if (block_nums > 0) {
        aclrtFree(src_block_table_ptr);
        aclrtFree(dst_block_table_ptr);
    }

    status = aclrtFreeHost(k_cache_host);
    status = aclrtFreeHost(v_cache_host);
    status = aclrtFreeHost(global_shuffle_table_host);
    status = aclrtFreeHost(global_block_num_host);
    if (block_nums > 0) {
        status = aclrtFreeHost(src_block_table_host);
        status = aclrtFreeHost(dst_block_table_host);
    }
    status = aclrtFreeHost(k_output_host);
    status = aclrtFreeHost(v_output_host);

    status = aclrtDestroyStream(stream);
    return status;
}

aclshmemx_uniqueid_t default_flag_uid;

int main(int argc, char *argv[])
{
    int status = 0;
    int n_pes = atoi(argv[INDEX1]);
    int pe_id = atoi(argv[INDEX2]);
    ipport = argv[INDEX3];
    // int32_t ret = aclshmemx_set_conf_store_tls(false, nullptr, 0);

    // Acl && Shmem init
    int32_t device_id = pe_id % g_npus + f_npu;
    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
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

    // aclshmemx_init_attr: 初始化shmem运行时（默认socket模式）
    // 参数详解:
    // - ACLSHMEMX_INIT_WITH_DEFAULT: 初始化模式标志
    //   使用TCP socket进行进程间rendezvous
    // - &attributes: 初始化属性结构体指针
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // KV Shuffle场景中的作用：
    // 1. 建立所有PE之间的通信通道
    // 2. 分配对称内存堆，用于KV Cache数据传输
    // 3. 初始化通信引擎，支持KV Shuffle操作
    // 4. 设置pe_id和n_pes信息
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    status = test_aclshmem_kv_shuffle(pe_id, n_pes);

    // aclshmem_finalize: 终止shmem运行时，释放所有shmem资源
    // 功能详解：
    // - 释放对称内存堆
    // - 关闭进程间通信通道
    // - 清理通信引擎状态
    // - 释放内部同步机制资源
    // 执行流程：
    // 1. 等待所有pending的KV Shuffle操作完成
    // 2. 通知其他PE本PE即将退出
    // 3. 释放所有对称内存资源
    // 4. 关闭bootstrap通信通道
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 注意：
    // 1. 每个PE必须调用此函数后才能退出程序
    // 2. 所有PE应同时调用此函数
    // 3. 调用后不能再执行任何shmem操作
    status = aclshmem_finalize();
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    return 0;
}
