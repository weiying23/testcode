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
 * @brief A3 超节点 RMA D2H Demo - Host 端内存的 Put & Get 示例
 *
 * 此程序演示在 A3 超节点环境下，使用 SHMEM RMA 接口访问 Host 端对称内存
 *
 * WHY 使用 Host 端对称内存：
 * - A3 超节点支持 Host 端统一内存编址
 * - Host 内存可通过 PCIe 和超节点互联网络被远程 PE 直接访问
 * - 相比 Device 端内存，Host 端内存容量更大，适合大规模数据传输
 * - 跨节点场景中，Host 端 RMA 可以绕过 Device 内存限制
 *
 * WHY A3 超节点特殊配置：
 * - Server ID：每个超节点服务器需要配置正确的 Server ID
 * - 内存地址段：A3 超节点使用特定的统一内存地址段（如 0x29580000000）
 * - 内存大小：需要使用 check_support.py 检查可用超节点内存
 *
 * 支持的产品型号：
 * - Atlas A3 训练系列产品
 * - Atlas A3 推理系列产品
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

// rendezvous 地址（TCP socket）
// WHY 使用 TCP socket：进程间 rendezvous，建立通信连接
const char *ipport = "tcp://127.0.0.1:8998";

// uniqueid 结构体（用于 DEFAULT 初始化模式）
aclshmemx_uniqueid_t default_flag_uid;

// ============================================================================
// Kernel 实现：RMA Scalar 数据传输
// ============================================================================
/**
 * @brief RMA Scalar 测试 Kernel
 *
 * 此 kernel 使用 aclshmem_int32_p 和 aclshmem_int32_g 接口进行 scalar 数据传输
 *
 * WHY 在 Kernel 中使用 RMA：
 * - A3 超节点支持 Device 直接发起 RMA 操作访问 Host 内存
 * - 避免数据从 Device 到 Host 的额外拷贝
 * - 实现 Device-to-Host（D2H）的直接数据传输
 *
 * 测试流程：
 * 1. 每个 PE 向下一个 PE 发送自己的 peer 值
 * 2. 每个 PE 从下一个 PE 获取数据
 * 3. 验证接收的数据是否正确
 *
 * @param input 目标地址（Host 端对称内存，GVA 格式）
 * @param output 输出地址（Device 端内存）
 */
extern "C" __global__ __aicore__ void kernel_test(__gm__ int* input, __gm__ int* output)
{
    // WHY 仅 AIV 核执行：RMA 操作在 AIV 核上执行
    if ASCEND_IS_AIC {
        return;
    }

    // WHY 仅 core 0 执行：scalar 传输只需单核
    auto coreId = AscendC::GetBlockIdx();
    if (coreId > 0) {
        return;
    }

    // aclshmem_my_pe(): 获取当前 PE 编号（在 Kernel 内调用）
    // WHY 需要 my_pe：确定本 PE 的身份和通信目标
    // 返回值：当前进程在通信组中的编号，范围 [0, n_pes-1]
    int mype = aclshmem_my_pe();

    // aclshmem_n_pes(): 获取通信组中的总 PE 数量
    // WHY 需要 npes：计算环形拓扑中的下一个 PE
    int npes = aclshmem_n_pes();

    // 计算下一个 PE 编号（环形拓扑）
    // WHY 环形拓扑：每个 PE 向下一个 PE 发送，形成环形通信链
    // 例如：PE 0 → PE 1 → PE 2 → PE 3 → PE 0
    int peer = (mype + 1) % npes;

    // ========== RMA Scalar 测试循环 ==========
    // WHY 循环 10 次：测试 RMA 操作的稳定性和正确性
    for (int iii = 0; iii < 10; iii++) {
        // ========== Put 操作：发送数据到目标 PE ==========
        // aclshmem_int32_p: 对 int32 类型数据的 Put 操作（发送到目标 PE）
        //
        // WHY 使用 aclshmem_int32_p：
        // - 直接从 Device 发送 int32 数据到远程 Host 内存
        // - A3 超节点支持 Device-to-Host 的 RMA 操作
        // - 避免 Device → Host → Network → Host 的多步拷贝
        //
        // 参数详解：
        // - input: 目标地址（目标 PE 的 Host 端对称内存地址）
        //   GVA = Global Virtual Address，超节点统一内存地址
        //   WHY Host 端地址：数据写入远程 PE 的 Host 内存
        // - peer: 要发送的值（本 PE 的 peer 值）
        // - peer: 目标 PE 编号（接收数据的 PE）
        //
        // 执行流程：
        // 1. 本 PE 的 AIV 核发起 RMA Put 请求
        // 2. 数据通过超节点互联网络传输（节点内用 RDMA/MTE，跨节点用 RDMA）
        // 3. 数据写入目标 PE 的 Host 端 input 地址
        // 4. 非阻塞操作，需要配合 quiet 等待完成
        //
        // 超节点数据路径：
        // - 节点内：Device → PCIe → Host → 超节点互联 → 远程 Host
        // - 跨节点：Device → PCIe → Host → RoCE 网络 → 远程 Host
        aclshmem_int32_p(input, peer, peer);

        // aclshmem_quiet: 等待所有 shmem 操作完成
        //
        // WHY 需要 quiet：
        // - aclshmem_int32_p 是非阻塞操作
        // - quiet 确保 Put 操作的数据已完全传输到目标地址
        // - 避免 Get 操作读取到未完全传输的数据
        //
        // 执行效果：
        // - 阻塞直到所有之前发起的 Put/Get 操作完成
        // - 确保数据已写入远程 Host 内存
        // - 相当于同步屏障，保证数据一致性
        //
        // 超节点场景中的 quiet：
        // - 跨节点通信时，quiet 会等待 RoCE 网络传输完成
        // - 超节点互联网络的延迟较高，quiet 时间可能较长
        aclshmem_quiet();

        // ========== Get 操作：从目标 PE 获取数据 ==========
        // aclshmem_int32_g: 对 int32 类型数据的 Get 操作（从目标 PE 获取数据）
        //
        // WHY 使用 aclshmem_int32_g：
        // - 直接从远程 Host 内存读取 int32 数据到 Device
        // - A3 超节点支持 Host-to-Device 的 RMA 操作
        // - 实现 Host 端数据的直接读取
        //
        // 参数详解：
        // - input: 目标地址（目标 PE 的 Host 端对称内存地址）
        //   WHY 相同地址：读取刚才 Put 写入的数据
        // - peer: 目标 PE 编号（数据来源 PE）
        //
        // 返回值：从 peer PE 获取的 int32 值
        //
        // 执行流程：
        // 1. 本 PE 的 AIV 核发起 RMA Get 请求
        // 2. 超节点互联网络从远程 Host 内存读取数据
        // 3. 数据传输到本 PE 的 AIV 核
        // 4. 返回读取的值
        //
        // 超节点数据路径：
        // - 节点内：远程 Host → 超节点互联 → 本地 Host → PCIe → Device
        // - 跨节点：远程 Host → RoCE 网络 → 本地 Host → PCIe → Device
        auto get_num = aclshmem_int32_g(input, peer);

        // aclshmem_quiet: 等待 Get 操作完成
        // WHY 需要 quiet：确保数据已完全接收
        aclshmem_quiet();

        // 将获取的数据写入 output
        // WHY 写入 output：用于 Host 端验证结果
        *(output) = get_num;
    }
}

/**
 * @brief 启动 RMA Scalar 测试 Kernel
 *
 * WHY 需要此函数：
 * - 封装 kernel 启动参数
 * - Host 端通过此函数调用 Device kernel
 *
 * @param block_dim 核数配置（scalar 测试只需 1 核）
 * @param stream ACL 流
 * @param input Host 端对称内存指针（GVA 格式）
 * @param output Device 端输出内存指针
 */
void run_demo_scalar(uint32_t block_dim, void* stream, int* input, int* output)
{
    kernel_test<<<block_dim, nullptr, stream>>>(input, output);
}

// ============================================================================
// Host 端测试函数
// ============================================================================
/**
 * @brief A3 超节点 RMA D2H 测试函数
 *
 * WHY 测试 RMA D2H：
 * - 验证 A3 超节点的 Host 端对称内存功能
 * - 测试 Device 直接访问 Host 内存的能力
 * - 验证跨节点 RMA 操作的正确性
 *
 * @param my_pe 当前 PE 编号
 * @param n_pes 总 PE 数量
 * @return 0 表示成功，-1 表示失败
 */
int test_aclshmem_rma_scalar_8p(int my_pe, int n_pes)
{
    // ========== ACL 初始化阶段 ==========
    aclrtStream stream = nullptr;

    // aclInit: 初始化 ACL 运行时环境
    // WHY 必须首先执行：后续所有 ACL API 依赖此初始化
    ACL_CHECK_WITH_RET(aclInit(nullptr), ERROR_LOG("aclInit failed"), return -1);

    // aclrtSetDevice: 设置当前进程使用的 NPU 设备
    // WHY 直接使用 my_pe 作为 device_id：本示例假设 PE 编号 = 设备编号
    // 实际部署中可能需要根据超节点拓扑计算正确的 device_id
    ACL_CHECK_WITH_RET(aclrtSetDevice(my_pe), ERROR_LOG("aclrtSetDevice failed"), return -1);

    // aclrtCreateStream: 创建 ACL 流
    // WHY 需要流：管理 kernel 执行顺序
    ACL_CHECK_WITH_RET(aclrtCreateStream(&stream), ERROR_LOG("aclrtCreateStream failed"), return -1);

    // ========== Host 端内存准备 ==========
    int32_t *input_host;
    int32_t *output_host;

    // aclrtMallocHost: 在 Host 端分配内存
    // WHY 需要 Host 端内存：
    // - 用于初始化 Host 端对称内存的数据
    // - 用于接收 Device 端的结果进行验证
    ACL_CHECK_WITH_RET(aclrtMallocHost(reinterpret_cast<void**>(&input_host), sizeof(int)),
        ERROR_LOG("aclrtMallocHost failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMallocHost(reinterpret_cast<void**>(&output_host), sizeof(int)),
        ERROR_LOG("aclrtMallocHost failed"), return -1);

    // 初始化数据
    // WHY input_host 初始化为 0：等待其他 PE 通过 Put 写入数据
    // WHY output_host 初始化为 my_pe：本 PE 的初始输出值
    *input_host = 0;
    *output_host = my_pe;

    // ========== SHMEM 初始化阶段 ==========
    // 对称内存大小：1GB
    // WHY 1GB：A3 超节点需要足够的 Host 端对称内存
    // 实际大小应根据 check_support.py 的扫描结果设置
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;

    // aclshmemx_init_attr_t: shmem 初始化属性结构体
    //
    // A3 超节点关键配置：
    // - my_pe: 当前 PE 编号（超节点内的全局 PE 编号）
    // - n_pes: 总 PE 数量（超节点内的总 NPU 数量）
    // - ip_port: rendezvous 地址（超节点服务器间的通信地址）
    // - local_mem_size: Host 端对称内存大小
    //   WHY Host 端：A3 超节点支持 Host 端统一内存编址
    //
    // 超节点注意事项：
    // 1. Server ID 必须正确配置（使用 npu-smi info -t spod-info 查询）
    // 2. 内存地址必须在超节点支持的地址段内（如 0x29580000000）
    // 3. 跨节点通信需要 RoCE 网络正常配置
    aclshmemx_init_attr_t attributes;

    // test_set_attr: 辅助函数，填充 shmem 初始化属性结构体
    test_set_attr(my_pe, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // aclshmemx_init_attr: 初始化 shmem 运行时（默认 socket 模式）
    //
    // WHY 使用 ACLSHMEMX_INIT_WITH_DEFAULT：
    // - TCP socket 进行进程间 rendezvous
    // - 不需要 MPI，简化超节点部署
    //
    // A3 超节点执行后完成：
    // 1. 建立超节点服务器间的通信通道
    // 2. 分配 Host 端对称内存堆（Symmetric Heap）
    //    WHY Host 端：支持 Device-to-Host RMA 操作
    // 3. 初始化超节点互联通信引擎
    //    - 节点内：使用 RDMA/MTE 引擎
    //    - 跨节点：使用 RDMA/RoCE 引擎
    // 4. 设置超节点 PE 编号和通信组信息
    auto status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("aclshmemx_init_attr failed"), return -1);

    // ========== Host 端对称内存分配 ==========
    // aclshmemx_malloc: 分配对称内存（带 HOST_SIDE 参数）
    //
    // WHY 使用 HOST_SIDE 参数：
    // - A3 超节点支持 Host 端对称内存分配
    // - Host 端内存可通过超节点互联被远程 PE 直接访问
    // - 相比 DEVICE_SIDE，Host 端内存容量更大
    // - Device 可以通过 RMA 直接访问 Host 内存（D2H）
    //
    // 参数详解：
    // - 2*1024*1024: 对称内存大小（2MB）
    //   WHY 2MB：满足测试需求，实际应根据超节点可用内存设置
    // - HOST_SIDE: 分配端标识
    //   HOST_SIDE：在 Host 端分配对称内存
    //   DEVICE_SIDE：在 Device 端分配对称内存（aclshmem_malloc 默认）
    //
    // 返回值：对称内存指针（GVA 格式）
    //
    // A3 超节点 Host 端对称内存特点：
    // 1. 内存位于 Host 端，通过 PCIe 与 Device 连接
    // 2. 超节点互联网络可直接访问 Host 内存
    // 3. 所有 PE 看到相同的 GVA 地址
    // 4. 支持 Device-to-Host 和 Host-to-Device 的 RMA 操作
    //
    // 超节点内存地址段（check_support.py 中定义）：
    // - 0x29580000000 - 0x2958xxxxxxxx（682GB）
    // - 0xa9580000000 - 0xa958xxxxxxxx（682GB）
    // - 0x129580000000 - 0x129580xxxxxxxx（682GB）
    // - 0x1a9580000000 - 0x1a958xxxxxxxx（682GB）
    uint8_t *input = (uint8_t*)aclshmemx_malloc(2*1024*1024, HOST_SIDE);

    // Device 端 output 内存（用于存储结果）
    uint8_t *output = nullptr;
    ACL_CHECK_WITH_RET(aclrtMalloc((void **)&output, sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST),
        ERROR_LOG("aclrtMalloc failed"), return -1);

    // 初始化 Host 端对称内存数据
    // WHY 拷贝到 input：Host 端对称内存需要初始化
    ACL_CHECK_WITH_RET(aclrtMemcpy(input, sizeof(int), input_host, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("aclrtMemcpy failed"), return -1);

    // 初始化 Device 端 output 数据
    ACL_CHECK_WITH_RET(aclrtMemcpy(output, sizeof(int), output_host, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("aclrtMemcpy failed"), return -1);

    // ========== 同步和 Kernel 执行 ==========
    // aclshmem_barrier_all: 全局屏障同步
    //
    // WHY 需要屏障：
    // - 确保所有 PE 都完成 Host 端对称内存的初始化
    // - 避免 PE 读取到未初始化的数据
    // - 超节点场景中，跨节点屏障需要等待所有节点
    //
    // 超节点屏障特点：
    // - 节点内屏障：通过超节点互联快速完成
    // - 跨节点屏障：通过 RoCE 网络同步
    aclshmem_barrier_all();

    // 启动 RMA Scalar 测试 Kernel
    run_demo_scalar(1, stream, (int*)input, (int*)output);

    // aclrtSynchronizeStream: 同步流，等待 kernel 完成
    ACL_CHECK_WITH_RET(aclrtSynchronizeStream(stream), ERROR_LOG("aclrtSynchronizeStream failed"), return -1);

    // aclshmem_barrier_all: 全局屏障同步
    // WHY 需要屏障：
    // - 确保所有 PE 都完成了 RMA Put 和 Get 操作
    // - 在结果拷贝前执行，保证数据一致性
    // - 超节点场景中，需要等待跨节点传输完成
    aclshmem_barrier_all();

    // ========== 结果验证 ==========
    // 从 Host 端对称内存拷贝结果到 Host
    ACL_CHECK_WITH_RET(aclrtMemcpy(input_host, sizeof(int), input, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("aclrtMemcpy failed"), return -1);

    // 从 Device 端内存拷贝结果到 Host
    ACL_CHECK_WITH_RET(aclrtMemcpy(output_host, sizeof(int), output, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("aclrtMemcpy failed"), return -1);

    // 打印结果
    printf("%d: received message %d %d\n", my_pe, *input_host, *output_host);

    // 验证结果正确性
    // WHY 期望 output_host == (my_pe + 1) % n_pes：
    // - 每个 PE 从下一个 PE 获取数据
    // - 下一个 PE 的初始 output 值是其 PE 编号
    // - 环形拓扑：PE i 应收到 PE (i+1) % n_pes 的值
    if ( *output_host == ((my_pe + 1) % n_pes)) {
        printf("[SUCCESS] run success in pe %d\n", my_pe);
    } else {
        printf("[ERROR] run result incorrect in pe %d\n", my_pe);
    }

    // ========== 资源释放阶段 ==========
    // aclshmemx_free: 释放 Host 端对称内存
    //
    // WHY 必须使用 aclshmemx_free：
    // - 不能使用 aclrtFree 释放对称内存
    // - 必须使用与分配时相同的 SIDE 参数（HOST_SIDE）
    // - Host 端对称内存必须归还到 Symmetric Heap
    //
    // 参数详解：
    // - input: aclshmemx_malloc 返回的对称内存指针
    // - HOST_SIDE: 分配端标识（必须与分配时的 SIDE 参数一致）
    //
    // 超节点注意事项：
    // - 所有 PE 应同时释放对称内存
    // - 释放后该地址不再可用于 RMA 操作
    // - Host 端对称内存释放可能涉及超节点互联网络的清理
    aclshmemx_free(input, HOST_SIDE);

    // aclshmem_finalize: 终止 shmem 运行时
    //
    // WHY 需要终止：
    // - 释放 Host 端对称内存堆
    // - 关闭超节点服务器间的通信通道
    // - 清理超节点互联通信引擎状态
    //
    // A3 超节点 finalize 特点：
    // 1. 等待所有 pending 的 RMA 操作完成
    // 2. 通知超节点其他 PE 本 PE 即将退出
    // 3. 释放所有 Host 端对称内存资源
    // 4. 关闭 RoCE 网络连接和超节点互联
    //
    // 注意：
    // 1. 每个超节点 PE 必须调用此函数
    // 2. 所有 PE 应同时调用此函数
    // 3. 调用后不能再执行任何 RMA 操作
    aclshmem_finalize();

    // 释放 Host 端内存
    ACL_CHECK_WITH_RET(aclrtFreeHost(input_host), ERROR_LOG("aclrtFreeHost failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtFreeHost(output_host), ERROR_LOG("aclrtFreeHost failed"), return -1);

    // 释放 Device 端内存
    ACL_CHECK_WITH_RET(aclrtDestroyStream(stream), ERROR_LOG("aclrtDestroyStream failed"), return -1);

    // aclrtResetDevice: 重置设备
    ACL_CHECK_WITH_RET(aclrtResetDevice(my_pe), ERROR_LOG("aclrtResetDevice failed"), return -1);

    // aclFinalize: 终止 ACL 运行时
    ACL_CHECK_WITH_RET(aclFinalize(), ERROR_LOG("aclFinalize failed"), return -1);

    return 0;
}

// ============================================================================
// 主函数
// ============================================================================
/**
 * @brief 主函数入口
 *
 * A3 超节点运行要求：
 * 1. Server ID 配置正确（使用 npu-smi info -t spod-info 检查）
 * 2. 超节点内存足够（使用 check_support.py 检查）
 * 3. RoCE 网络配置正确（跨节点通信需要）
 *
 * 启动方式：
 * bash run.sh
 * 或手动启动：
 * ./rma_d2h_demo <n_pes> <my_pe>
 *
 * 例如：8 PE 测试
 * PE 0: ./rma_d2h_demo 8 0
 * PE 1: ./rma_d2h_demo 8 1
 * ...
 * PE 7: ./rma_d2h_demo 8 7
 *
 * @param argc 参数数量
 * @param argv 参数列表
 * @return 0 表示成功
 */
int main(int argc, char *argv[])
{
    int argIdx = 1;

    // n_pes: 总 PE 数量（超节点内的总 NPU 数量）
    // WHY 需要知道 n_pes：确定通信组大小和环形拓扑
    // A3 超节点典型配置：384 NPU（Super Pod Size）
    int n_pes = atoi(argv[argIdx++]);

    // my_pe: 当前 PE 编号（超节点内的全局 PE 编号）
    // WHY 需要知道 my_pe：确定本 PE 的身份和通信目标
    // A3 超节点 PE 编号：范围 [0, Super Pod Size - 1]
    int my_pe = atoi(argv[argIdx++]);

    // 执行 RMA D2H 测试
    (void)test_aclshmem_rma_scalar_8p(my_pe, n_pes);

    INFO_LOG("[INFO] demo run end in pe %d.", my_pe);
    return 0;
}