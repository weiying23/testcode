/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <cstring>
#include <cerrno>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "shmem.h"
#include "../utils/utils.h"


const char *ipport = "tcp://127.0.0.1:8998";


aclshmemx_uniqueid_t default_flag_uid;

extern "C" __global__ __aicore__ void kernel_test(__gm__ int* input, __gm__ int* output)
{
    if ASCEND_IS_AIC {
        return;
    }

    auto coreId = AscendC::GetBlockIdx();
    if (coreId > 0) {
        return;
    }

    // aclshmem_my_pe(): 获取当前PE编号（在Kernel内调用）
    // 返回当前进程在通信组中的编号，范围 [0, n_pes-1]
    // 用于确定本PE的数据位置和通信目标
    int mype = aclshmem_my_pe();
    // aclshmem_n_pes(): 获取通信组中的总PE数量
    // 返回参与通信的进程总数，用于计算数据分布和循环范围
    int npes = aclshmem_n_pes();
    // 计算下一个PE编号（环形拓扑）
    int peer = (mype + 1) % npes;

    for (int iii = 0; iii < 10; iii++) {
        // aclshmem_int32_p: 对int32类型数据的Put操作（发送到目标PE）
        // 参数详解:
        // - input: 目标地址（目标PE的对称内存地址，GVA格式）
        //   所有PE看到的相同虚拟地址
        // - peer: 要发送的值（本PE的peer值）
        // - peer: 目标PE编号（接收数据的PE）
        // 执行流程:
        // 1. 将peer值从本PE发送到peer PE的input地址
        // 2. 数据通过RDMA引擎传输（跨节点）或MTE引擎传输（节点内）
        // 3. 非阻塞操作，需要配合quiet等待完成
        aclshmem_int32_p(input, peer, peer);
        // aclshmem_quiet: 等待所有shmem操作完成
        // 执行效果:
        // - 阻塞直到所有之前发起的Put/Get操作完成
        // - 确保数据已完全传输到目标地址
        // - 相当于同步屏障，保证数据一致性
        // 必须在Put操作后调用，否则数据可能未完全传输
        aclshmem_quiet();
        // aclshmem_int32_g: 对int32类型数据的Get操作（从目标PE获取数据）
        // 参数详解:
        // - input: 目标地址（目标PE的对称内存地址，GVA格式）
        // - peer: 目标PE编号（数据来源PE）
        // 返回值: 从peer PE获取的int32值
        // 执行流程:
        // 1. 本PE向peer PE发起读取请求
        // 2. peer PE通过通信引擎发送数据
        // 3. 本PE接收数据并返回值
        auto get_num = aclshmem_int32_g(input, peer);
        // aclshmem_quiet: 等待Get操作完成
        // 确保数据已完全接收
        aclshmem_quiet();
        *(output ) = get_num;
    }
}

void run_demo_scalar(uint32_t block_dim, void* stream, int* input, int* output)
{
    kernel_test<<<block_dim, nullptr, stream>>>(input, output);
}

int test_aclshmem_rma_scalar_8p(int my_pe, int n_pes)
{
    // 初始化ACL和ACLSHMEM
    aclrtStream stream = nullptr;

    ACL_CHECK_WITH_RET(aclInit(nullptr), ERROR_LOG("aclInit failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtSetDevice(my_pe), ERROR_LOG("aclrtSetDevice failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtCreateStream(&stream), ERROR_LOG("aclrtCreateStream failed"), return -1);

    int32_t *input_host;
    int32_t *output_host;
    ACL_CHECK_WITH_RET(aclrtMallocHost(reinterpret_cast<void**>(&input_host), sizeof(int)),
        ERROR_LOG("aclrtMallocHost failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMallocHost(reinterpret_cast<void**>(&output_host), sizeof(int)),
        ERROR_LOG("aclrtMallocHost failed"), return -1);
    *input_host = 0;
    *output_host = my_pe;

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    // aclshmemx_init_attr_t: shmem初始化属性结构体
    // 包含以下关键字段：
    // - my_pe: 当前PE编号（进程ID），范围[0, n_pes-1]
    // - n_pes: 总PE数量（进程总数）
    // - ip_port: rendezvous地址（TCP socket地址）
    // - local_mem_size: 对称内存大小（字节）
    // - option_attr: 可选属性（引擎类型、超时等）
    // - instance_id: 多实例模式下的实例编号
    // - comm_args: 通信参数指针
    aclshmemx_init_attr_t attributes;

    // test_set_attr: 辅助函数，填充shmem初始化属性结构体
    // 参数详解:
    // - my_pe: 当前PE编号
    // - n_pes: 总PE数量
    // - local_mem_size: 对称内存大小（1GB）
    // - ipport: rendezvous地址字符串，如"tcp://127.0.0.1:8998"
    // - default_flag_uid: uniqueid结构体
    // - &attributes: 属性结构体指针（输出参数）
    test_set_attr(my_pe, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // aclshmemx_init_attr: 初始化shmem运行时（默认socket模式）
    // 参数详解:
    // - ACLSHMEMX_INIT_WITH_DEFAULT: 初始化模式标志
    //   使用TCP socket进行进程间rendezvous，不需要MPI
    // - &attributes: 初始化属性结构体指针
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 执行后完成:
    // 1. 建立进程间通信通道
    // 2. 分配对称内存堆
    // 3. 初始化通信引擎
    // 4. 设置PE编号和通信组信息
    auto status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("aclshmemx_init_attr failed"), return -1);

    // aclshmemx_malloc: 分配对称内存（带HOST_SIDE参数，从Host端分配）
    // 参数详解:
    // - 2*1024*1024: 对称内存大小（2MB）
    // - HOST_SIDE: 分配端标识
    //   HOST_SIDE表示在Host端分配对称内存
    //   DEVICE_SIDE表示在Device端分配对称内存
    // 返回值: 对称内存指针（GVA格式）
    // HOST_SIDE分配特点：
    // - 内存位于Host端，可通过PCIe访问
    // - 适合Host端发起的RMA操作
    // - 与aclshmem_malloc的Device端分配不同
    // 注意：必须通过aclshmemx_free释放（带相同SIDE参数）
    uint8_t *input = (uint8_t*)aclshmemx_malloc(2*1024*1024, HOST_SIDE);
    uint8_t *output = nullptr;
    ACL_CHECK_WITH_RET(aclrtMalloc((void **)&output, sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST),
        ERROR_LOG("aclrtMalloc failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMemcpy(input, sizeof(int), input_host, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("aclrtMemcpy failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMemcpy(output, sizeof(int), output_host, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("aclrtMemcpy failed"), return -1);

    // aclshmem_barrier_all: 全局屏障同步，确保所有PE数据初始化完成
    // 功能详解：
    // - 所有PE都调用此函数后才能继续执行
    // - 执行流程：
    //   1. 当前PE到达屏障，标记自己已完成初始化
    //   2. 等待所有其他PE也到达屏障
    //   3. 所有PE都到达后，一起释放继续执行
    // - 用途：
    //   * 确保所有PE的数据都已初始化
    //   * 用于Kernel执行前的同步
    //   * 保证数据一致性
    // - 注意：必须所有PE都调用此函数，否则会造成死锁
    aclshmem_barrier_all();
    run_demo_scalar(1, stream, (int*)input, (int*)output);

    ACL_CHECK_WITH_RET(aclrtSynchronizeStream(stream), ERROR_LOG("aclrtSynchronizeStream failed"), return -1);
    // aclshmem_barrier_all: 全局屏障同步，确保所有PE的RMA操作完成
    // 确保所有PE都完成了Put和Get操作后再进行结果验证
    // 在结果拷贝前执行，保证数据一致性
    aclshmem_barrier_all();

    ACL_CHECK_WITH_RET(aclrtMemcpy(input_host, sizeof(int), input, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("aclrtMemcpy failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMemcpy(output_host, sizeof(int), output, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("aclrtMemcpy failed"), return -1);

    printf("%d: received message %d %d\n", my_pe, *input_host, *output_host);
    if ( *output_host == ((my_pe + 1) % n_pes)) {
        printf("[SUCCESS] run success in pe %d\n", my_pe);
    } else {
        printf("[ERROR] run result incorrect in pe %d\n", my_pe);  // 期望input变为前卡, output变为后卡
    }

    // aclshmemx_free: 释放对称内存（带HOST_SIDE参数）
    // 参数详解:
    // - input: aclshmemx_malloc返回的对称内存指针
    // - HOST_SIDE: 分配端标识（必须与分配时的SIDE参数一致）
    // 必须与aclshmemx_malloc配对使用，SIDE参数必须匹配
    // 执行效果:
    // - 将Host端对称内存归还到Symmetric Heap
    // - 其他shmem操作可以重新分配此内存
    // - 释放后该地址不再可用于通信
    // 重要提示：
    // 1. 不能使用aclrtFree释放对称内存
    // 2. 必须使用与分配时相同的SIDE参数
    // 3. 所有PE应同时释放对称内存
    aclshmemx_free(input, HOST_SIDE);

    // aclshmem_finalize: 终止shmem运行时，释放所有shmem资源
    // 功能详解：
    // - 释放对称内存堆（Symmetric Heap）
    // - 关闭进程间通信通道（TCP socket连接）
    // - 清理通信引擎状态
    // - 释放内部同步机制资源
    // 执行流程：
    // 1. 等待所有pending的RMA操作完成
    // 2. 通知其他PE本PE即将退出
    // 3. 释放所有对称内存资源
    // 4. 关闭bootstrap通信通道
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 注意：
    // 1. 每个PE必须调用此函数后才能退出程序
    // 2. 所有PE应同时调用此函数
    // 3. 调用后不能再执行任何shmem操作
    aclshmem_finalize();
    ACL_CHECK_WITH_RET(aclrtFreeHost(input_host), ERROR_LOG("aclrtFreeHost failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtFreeHost(output_host), ERROR_LOG("aclrtFreeHost failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtDestroyStream(stream), ERROR_LOG("aclrtDestroyStream failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtResetDevice(my_pe), ERROR_LOG("aclrtResetDevice failed"), return -1);
    ACL_CHECK_WITH_RET(aclFinalize(), ERROR_LOG("aclFinalize failed"), return -1);
    return 0;
}

int main(int argc, char *argv[])
{
    int argIdx = 1;
    int n_pes = atoi(argv[argIdx++]);
    int my_pe = atoi(argv[argIdx++]);

    (void)test_aclshmem_rma_scalar_8p(my_pe, n_pes);
    INFO_LOG("[INFO] demo run end in pe %d.", my_pe);
    return 0;
}
