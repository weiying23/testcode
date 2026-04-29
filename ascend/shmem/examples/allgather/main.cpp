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
#include "allgather_kernel.h"

int g_npus = 8;
const char *ipport;
int f_rank = 0;
int f_npu = 0;
const char *data_type;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr uint32_t MAGIC_MULTIPLIER = 1024;
constexpr uint32_t DATA_SIZE_THRESHOLD = 2097152;
constexpr uint32_t BLOCK_NUM_SMALL_DATA = 8;
constexpr uint32_t BLOCK_NUM_LARGE_DATA = 16;

template<class T>
int test_shmem_all_gather(int rank_id, int n_ranks, uint64_t local_mem_size)
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

    // shmemx_get_ffts_config(): 获取FFTS(Fast Flag Task Sync)硬件同步配置地址
    // FFTS是NPU核间快速同步机制，用于在kernel执行时实现核间的轻量级同步操作
    // 返回: FFTS配置寄存器的物理地址，传递给kernel用于设置同步基址
    // kernel内部会使用此地址进行AllGather通信操作的核间同步，确保数据收集顺序正确
    uint64_t fftsAddr = shmemx_get_ffts_config();

    int PERF_TIMES = 50;

    int case_num = 24;
    std::vector<uint32_t> test_cases = {};
    for (int i = 0; i < case_num; i++) {
        int data_len = 16 * (1 << i);
        test_cases.push_back(data_len);
    }

    uint32_t BLOCK_NUM = 8;

    std::ofstream outFile("./results.csv");
    if (!outFile.is_open()) {
        std::cerr << "错误：无法创建文件！" << std::endl;
        return 1;
    }
    outFile << "M,N,Time(us)\n";

    // magic is used to sync.
    int magic = 1;

    for (int i = 0; i < test_cases.size(); i++) {
        if (rank_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Started." << std::endl;
        }
        uint32_t trans_size = test_cases[i];

        //  Small data kernel needs 8 AIV core, Big data kernel needs 16 AIV.
        if (trans_size * sizeof(T) < DATA_SIZE_THRESHOLD) {
            BLOCK_NUM = BLOCK_NUM_SMALL_DATA;
        } else {
            BLOCK_NUM = BLOCK_NUM_LARGE_DATA;
        }

        void *input_ptr;
        aclrtMalloc(&input_ptr, trans_size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        uint8_t *input_host;
        aclrtMallocHost(reinterpret_cast<void**>(&input_host), trans_size * sizeof(T));
        std::string inputFile = "../../examples/allgather/golden/allgather_" + std::to_string(trans_size) + "_" +
                                std::to_string(n_ranks) + "/input_gm_" + std::to_string(rank_id) + ".bin";
        ReadFile(inputFile, input_host, trans_size * sizeof(T));
        aclrtMemcpy(input_ptr, trans_size * sizeof(T), input_host, trans_size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

        void *output_ptr;
        aclrtMalloc(&output_ptr, trans_size * n_ranks * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

        // shmem_malloc(): 从对称共享内存堆(Symmetric Heap)中分配指定大小的内存空间
        // 对称内存是指所有rank在相同偏移位置都能访问的共享内存区域，用于跨rank RDMA通信
        // 参数: aiv_num * SYNC_FLAG_INTERVAL * sizeof(T) + GVA_BUFF_MAX_SIZE / sizeof(T)
        // 返回: 对称内存指针，所有rank都可以通过相同偏移访问该内存区域
        // 该内存用于存储AllGather操作的同步标志和通信数据缓冲
        // SYNC_FLAG_INTERVAL用于核间同步，GVA_BUFF_MAX_SIZE用于存储通信中间结果
        int aiv_num = BLOCK_NUM;
        void *ptr = shmem_malloc(aiv_num * SYNC_FLAG_INTERVAL * sizeof(T) + GVA_BUFF_MAX_SIZE / sizeof(T));

        // AllGather
        for (int zz = 0; zz < PERF_TIMES; zz++) {
            magic++;
            allgather_demo<T>(BLOCK_NUM, stream, fftsAddr, (uint8_t *)input_ptr,
                              (uint8_t *)output_ptr, (uint8_t *)ptr, trans_size, magic * MAGIC_MULTIPLIER);
        }
        status = aclrtSynchronizeStream(stream);

        // Result Check
        T *output_host;
        size_t output_size = n_ranks * trans_size * sizeof(T);
        status = aclrtMallocHost(reinterpret_cast<void**>(&output_host), output_size);
        status = aclrtMemcpy(output_host, output_size, output_ptr, output_size, ACL_MEMCPY_DEVICE_TO_HOST);

        T *golden_host;
        status = aclrtMallocHost(reinterpret_cast<void**>(&golden_host), output_size);
        std::string goldenFile = "../../examples/allgather/golden/allgather_" +
            std::to_string(trans_size) + "_" + std::to_string(n_ranks) + "/golden.bin";
        ReadFile(goldenFile, golden_host, n_ranks * trans_size * sizeof(T));
        for (int zz = 0; zz < n_ranks * trans_size; zz++) {
            if (static_cast<float>(output_host[zz]) != static_cast<float>(golden_host[zz])) {
                std::cout << static_cast<float>(output_host[zz]) << " != " << static_cast<float>(golden_host[zz])
                          << ", trans_size is : " << trans_size << ", idx is: " << zz
                          << ", rank_id is: "<< rank_id << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // 去初始化
        status = aclrtFreeHost(input_host);
        status = aclrtFreeHost(output_host);
        status = aclrtFreeHost(golden_host);

        // shmem_free(): 释放之前通过shmem_malloc()分配的对称共享内存空间
        // 参数: ptr - 要释放的对称内存指针
        // 此函数会将内存归还到对称内存堆，供后续shmem_malloc调用重新使用
        // 注意：释放后不应再访问该内存区域，否则会导致未定义行为或数据损坏
        shmem_free(ptr);
        aclrtFree(input_ptr);
        aclrtFree(output_ptr);

        outFile << 1 << "," << trans_size << "," << " " << "\n";

        if (rank_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Finised !! Result Correct !!" << std::endl;
        }
    }

    outFile.close();

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
    g_npus = atoi(argv[INDEX4]);
    f_rank = atoi(argv[INDEX5]);
    f_npu = atoi(argv[INDEX6]);
    data_type = argv[INDEX7];
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    // shmem_set_conf_store_tls(): 禁用TLS(Thread Local Storage)存储配置方式
    // 参数: false表示禁用TLS，nullptr和0表示不使用默认配置文件路径和长度
    // 设置为false后使用shmem_set_attr/shmem_init_attr自定义配置方式初始化shmem环境
    int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);
    std::cout << "init shmem tls result:" << ret << std::endl;
    if (std::string(data_type) == "int") {
        status = test_shmem_all_gather<int>(rank_id, n_ranks, local_mem_size);
    } else if (std::string(data_type) == "int32_t") {
        status = test_shmem_all_gather<int32_t>(rank_id, n_ranks, local_mem_size);
    } else if (std::string(data_type) == "float16_t") {
        status = test_shmem_all_gather<fp16_t>(rank_id, n_ranks, local_mem_size);
    } else if (std::string(data_type) == "bfloat16_t") {
        status = test_shmem_all_gather<bfloat16>(rank_id, n_ranks, local_mem_size);
    }
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;

    return 0;
}