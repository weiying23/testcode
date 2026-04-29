/**
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

#include "fp16_t.h"
#include "bfloat16.h"
#include "utils.h"
#include "param.h"

using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;

#include "acl/acl.h"
#include "shmem_api.h"
#include "kv_shuffle_kernel.h"
int g_npus = 8;
const char *ipport;
int f_rank = 0;
int f_npu = 0;
const char *data_type;

constexpr int64_t MAX_SEQLEN = 1024;
constexpr int64_t MAX_BATCH = 10;
constexpr int64_t page_size = 128;
constexpr int64_t max_block_nums = MAX_SEQLEN * MAX_BATCH / page_size;
constexpr int64_t kv_head_num = 8;
constexpr int64_t head_dim = 128;

int test_shmem_kv_shuffle(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    // 初始化ACL和SHMEM
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    shmem_init_attr_t *attributes;
    // shmem_set_attr(): 设置shmem初始化属性参数
    // 参数1 rank_id: 当前进程的rank编号（进程在通信组中的唯一标识）
    // 参数2 n_ranks: 通信组中总进程数量（所有参与分布式计算的rank总数）
    // 参数3 local_mem_size: 每个rank分配的对称内存空间大小（1GB）
    // 参数4 ipport: 网络通信的IP地址和端口字符串，用于rank间RDMA网络连接建立
    // 参数5 &attributes: 输出参数，返回配置好的初始化属性结构体指针
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    // shmem_init_attr(): 根据attributes中的配置参数初始化shmem运行环境
    // 此函数会执行: 建立rank间RDMA网络连接、分配对称内存堆、初始化通信通道和同步资源等
    status = shmem_init_attr(attributes);

    uint32_t BLOCK_NUM = 16;

    int64_t kv_cache_size = max_block_nums * kv_head_num * page_size * head_dim * sizeof(int8_t);

    std::string inputFile;
    // k_cache input
    uint8_t *k_cache_host;
    aclrtMallocHost(reinterpret_cast<void **>(&k_cache_host), kv_cache_size);
    inputFile = "../../examples/kv_shuffle/scripts/output/k_cache_input_rank_" + std::to_string(rank_id) + ".bin";
    ReadFile(inputFile, k_cache_host, kv_cache_size);
    // shmem_malloc(): 从对称共享内存堆(Symmetric Heap)中分配指定大小的内存空间
    // 对称内存是指所有rank在相同偏移位置都能访问的共享内存区域，用于跨rank RDMA通信
    // 参数: kv_cache_size - KV cache数据的大小
    // 返回: 对称内存指针，所有rank都可以通过相同偏移访问该内存区域
    // 该内存用于存储KV Shuffle操作中的k_cache数据，实现跨rank的KV cache重排
    void *k_cache_ptr = shmem_malloc(kv_cache_size);
    aclrtMemcpy(k_cache_ptr, kv_cache_size, k_cache_host, kv_cache_size, ACL_MEMCPY_HOST_TO_DEVICE);

    // v_cache input
    uint8_t *v_cache_host;
    aclrtMallocHost(reinterpret_cast<void **>(&v_cache_host), kv_cache_size);
    inputFile = "../../examples/kv_shuffle/scripts/output/v_cache_input_rank_" + std::to_string(rank_id) + ".bin";
    ReadFile(inputFile, v_cache_host, kv_cache_size);
    // shmem_malloc(): 从对称共享内存堆中分配内存空间用于存储v_cache数据
    // 该内存用于存储KV Shuffle操作中的v_cache数据，实现跨rank的KV cache重排
    void *v_cache_ptr = shmem_malloc(kv_cache_size);
    aclrtMemcpy(v_cache_ptr, kv_cache_size, v_cache_host, kv_cache_size, ACL_MEMCPY_HOST_TO_DEVICE);

    // global_shuffle_table input
    uint8_t *global_shuffle_table_host;
    constexpr uint32_t PAIR_PER_RANK = 2;
    aclrtMallocHost(reinterpret_cast<void **>(&global_shuffle_table_host), n_ranks * PAIR_PER_RANK * sizeof(int64_t));
    inputFile = "../../examples/kv_shuffle/scripts/output/pair_list.bin";
    ReadFile(inputFile, global_shuffle_table_host, n_ranks * PAIR_PER_RANK * sizeof(int64_t));
    void *global_shuffle_table_ptr;
    aclrtMalloc(&global_shuffle_table_ptr, n_ranks * PAIR_PER_RANK * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(global_shuffle_table_ptr, n_ranks * PAIR_PER_RANK * sizeof(int64_t),
                global_shuffle_table_host, n_ranks * PAIR_PER_RANK * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // global_block_num input
    uint8_t *global_block_num_host;
    aclrtMallocHost(reinterpret_cast<void **>(&global_block_num_host), sizeof(int64_t));
    inputFile = "../../examples/kv_shuffle/scripts/output/block_num_rank_" + std::to_string(rank_id) + ".bin";
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
        inputFile = "../../examples/kv_shuffle/scripts/output/src_block_table_rank_" + std::to_string(rank_id) + ".bin";

        aclrtMalloc(&src_block_table_ptr, block_nums * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
        ReadFile(inputFile, src_block_table_host, block_nums * sizeof(int64_t));
        aclrtMemcpy(src_block_table_ptr, block_nums * sizeof(int64_t),
                    src_block_table_host, block_nums * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        std::cout << "Rank " << rank_id << " block_nums = 0, Skip src_block_table input" << std::endl;
    }

    // dst_block_table input
    uint8_t *dst_block_table_host;
    void *dst_block_table_ptr;
    if (block_nums != 0) {
        aclrtMallocHost(reinterpret_cast<void **>(&dst_block_table_host), block_nums * sizeof(int64_t));
        inputFile = "../../examples/kv_shuffle/scripts/output/dst_block_table_rank_" + std::to_string(rank_id) + ".bin";

        aclrtMalloc(&dst_block_table_ptr, block_nums * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
        ReadFile(inputFile, dst_block_table_host, block_nums * sizeof(int64_t));
        aclrtMemcpy(dst_block_table_ptr, block_nums * sizeof(int64_t),
                    dst_block_table_host, block_nums * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        std::cout << "Rank " << rank_id << " block_nums = 0, Skip dst_block_table input" << std::endl;
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
    outputFile = "../../examples/kv_shuffle/scripts/output/k_cache_output_rank_" + std::to_string(rank_id) + ".bin";
    WriteFile(outputFile, k_output_host, kv_cache_size);

    int8_t *v_output_host;
    status = aclrtMallocHost(reinterpret_cast<void**>(&v_output_host), kv_cache_size);
    status = aclrtMemcpy(v_output_host, kv_cache_size, v_cache_ptr, kv_cache_size, ACL_MEMCPY_DEVICE_TO_HOST);
    outputFile = "../../examples/kv_shuffle/scripts/output/v_cache_output_rank_" + std::to_string(rank_id) + ".bin";
    WriteFile(outputFile, v_output_host, kv_cache_size);

    // shmem_free(): 释放之前通过shmem_malloc()分配的对称共享内存空间
    // 参数: k_cache_ptr - 要释放的k_cache对称内存指针
    // 此函数会将内存归还到对称内存堆，供后续shmem_malloc调用重新使用
    shmem_free(k_cache_ptr);
    // shmem_free(): 释放v_cache对称内存空间
    shmem_free(v_cache_ptr);
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

    // shmem_finalize(): 结束并清理shmem运行环境，释放所有shmem相关资源
    // 此函数会执行以下操作:
    // 1. 释放所有未释放的对称内存资源（如果还有未释放的会自动释放）
    // 2. 关闭rank间的RDMA网络通信连接
    // 3. 清理通信通道和同步资源（FFTS、信号量等）
    // 4. 重置shmem运行状态，使后续shmem API调用无效
    // 调用此函数后，所有shmem API都不应再被调用，直到重新初始化
    // 返回: SHMEM_SUCCESS表示成功清理，否则表示清理过程中出现错误
    status = shmem_finalize();
    status = aclrtDestroyStream(stream);
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    return 0;
}

int main(int argc, char *argv[])
{
    int status = 0;
    int n_ranks = atoi(argv[INDEX1]);
    int rank_id = atoi(argv[INDEX2]);
    ipport = argv[INDEX3];
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    // shmem_set_conf_store_tls(): 禁用TLS(Thread Local Storage)存储配置方式
    // 参数: false表示禁用TLS，nullptr和0表示不使用默认配置文件路径和长度
    // 设置为false后使用shmem_set_attr/shmem_init_attr自定义配置方式初始化shmem环境
    int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);

    status = test_shmem_kv_shuffle(rank_id, n_ranks, local_mem_size);

    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;

    return 0;
}
