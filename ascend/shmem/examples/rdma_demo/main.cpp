/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "acl/acl.h"
#include "shmem_api.h"
#include "shmemi_host_common.h"

int g_npus = 8;
const char *ipport;
int f_rank = 0;
int f_npu = 0;
extern void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva, int message_length);

int test_shmem_team_all_gather(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    // 初始化ACL和SHMEM
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    const int num10 = 10;
    aclrtStream stream = nullptr;

    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);

    shmem_init_attr_t *attributes;
    // shmem_set_attr(): 设置shmem初始化属性参数
    // 参数1 rank_id: 当前进程的rank编号（进程在通信组中的唯一标识）
    // 参数2 n_ranks: 通信组中总进程数量（所有参与分布式计算的rank总数）
    // 参数3 local_mem_size: 每个rank分配的对称内存空间大小（1GB）
    // 参数4 ipport: 网络通信的IP地址和端口字符串，用于rank间RDMA网络连接建立
    // 参数5 &attributes: 输出参数，返回配置好的初始化属性结构体指针
    status |= shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    // attributes->option_attr.data_op_engine_type: 设置数据操作引擎类型
    // SHMEM_DATA_OP_ROCE表示使用RoCE(RDMA over Converged Ethernet)引擎进行数据传输
    // RoCE提供高性能的RDMA通信，适合大规模分布式计算场景
    attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
    // shmem_set_conf_store_tls(): 禁用TLS(Thread Local Storage)存储配置方式
    // 参数: false表示禁用TLS，nullptr和0表示不使用默认配置文件路径和长度
    shmem_set_conf_store_tls(false, nullptr, 0);
    // shmem_init_attr(): 根据attributes中的配置参数初始化shmem运行环境
    // 此函数会执行: 建立rank间RDMA网络连接、分配对称内存堆、初始化通信通道和同步资源等
    status |= shmem_init_attr(attributes);

    // shmem_malloc(): 从对称共享内存堆(Symmetric Heap)中分配指定大小的内存空间
    // 对称内存是指所有rank在相同偏移位置都能访问的共享内存区域，用于跨rank RDMA通信
    // 参数: 1024 - 分配1024字节的对称内存空间
    // 返回: 对称内存指针，所有rank都可以通过相同偏移访问该内存区域
    uint8_t *ptr = static_cast<uint8_t*>(shmem_malloc(1024));

    // 初始化数据
    uint32_t trans_size = 16;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = (rank_id + num10);
    }

    status |= aclrtMemcpy(ptr + shmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t),
        input.data(), trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // AllGather
    allgather_demo(1, stream, (uint8_t *)ptr, trans_size * sizeof(int32_t));
    // shmem_handle_t: shmem handle结构体，用于跟踪和等待异步通信操作的完成
    // handle包含通信操作的元信息，如team_id标识通信组
    shmem_handle_t handle;
    // handle.team_id: 设置通信组ID为SHMEM_TEAM_WORLD，表示使用全局通信组
    // SHMEM_TEAM_WORLD是预定义的全局通信组，包含所有rank
    handle.team_id = SHMEM_TEAM_WORLD;
    // shmem_handle_wait(): 等待handle对应的通信操作完成
    // 参数: handle - 要等待的通信handle, stream - ACL流用于同步
    // 此函数会阻塞直到handle对应的通信操作完成，确保数据传输完成后再继续执行
    // 在本场景中，等待AllGather通信完成后再进行结果校验
    shmem_handle_wait(handle, stream);
    status |= aclrtSynchronizeStream(stream);

    // 结果校验打印
    int32_t *y_host;
    size_t input_size = n_ranks * trans_size * sizeof(int32_t);
    status |= aclrtMallocHost(reinterpret_cast<void**>(&y_host), input_size);
    status |= aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST);

    const int block_size = 16;
    for (int i = 0; i < n_ranks; i++) {
        for (int j = 0; j < block_size; j++) {
            if (y_host[trans_size * i + trans_size / block_size * j] != num10 + i) {
                std::cout << y_host[trans_size * i + trans_size / block_size * j] << " != " << num10 + i << std::endl;
                // std::exit(EXIT_FAILURE);
                return -1;
            }
        }
    }
    std::cout << "check transport result success, rank=" << rank_id << std::endl;
    // 去初始化
    status |= aclrtFreeHost(y_host);
    // shmem_free(): 释放之前通过shmem_malloc()分配的对称共享内存空间
    // 参数: ptr - 要释放的对称内存指针
    // 此函数会将内存归还到对称内存堆，供后续shmem_malloc调用重新使用
    shmem_free(ptr);
    // shmem_finalize(): 结束并清理shmem运行环境，释放所有shmem相关资源
    // 此函数会执行以下操作:
    // 1. 释放所有未释放的对称内存资源（如果还有未释放的会自动释放）
    // 2. 关闭rank间的RDMA网络通信连接
    // 3. 清理通信通道和同步资源（FFTS、信号量等）
    // 4. 重置shmem运行状态，使后续shmem API调用无效
    // 调用此函数后，所有shmem API都不应再被调用，直到重新初始化
    status |= shmem_finalize();
    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();
    return 0;
}

int main(int argc, char *argv[])
{
    int argIdx = 1;
    int status = 0;
    int n_ranks = atoi(argv[argIdx++]);
    int rank_id = atoi(argv[argIdx++]);
    ipport = argv[argIdx++];
    g_npus = atoi(argv[argIdx++]);
    f_rank = atoi(argv[argIdx++]);
    f_npu = atoi(argv[argIdx++]);
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    status = test_shmem_team_all_gather(rank_id, n_ranks, local_mem_size);
    std::cout << "demo run finished in rank " << rank_id << " with status " << status << std::endl;

    return 0;
}