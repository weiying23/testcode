/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Comm Benchmark主程序 - NPU通信性能对比测试
 *
 * 问题诊断与修复状态：
 * 1. Pingpong latency定值：[已修复]
 *    - 根因：Kernel未写入magic value到数据末尾
 *    - 修复：发送方在发送前写入 magic value
 *    - 位置：comm_benchmark_kernel.cpp 的 pingpong latency kernels
 *
 * 2. 带宽测量过大：[已修复 - 参考最佳实现]
 *    - 根因：原实现只测量指令下发时间
 *    - 最佳实现参考：rdma_perftest/rdma_perftest_kernel.cpp
 *    - 新同步机制：
 *      a) 发送方批量发送 → quiet（保证数据到达）
 *      b) 发送方发送通知消息（内容来自 src_addr 前4字节）
 *      c) 接收方轮询通知 → 发送确认（不调用 quiet）
 *      d) 发送方轮询确认 → 记录结束时间
 *    - 关键改进：
 *      * 使用单独通知消息（不破坏数据）
 *      * 接收方不调用 quiet（无意义）
 *      * 数据初始化：Host端初始化 src_addr = pe_id + MAGIC_VAL
 *
 * 3. RDMA/MTE带宽相同：[已诊断] 测试配置问题（非bug）
 *    - MTE用于节点内，RDMA用于跨节点
 *    - 单节点测试时两者可能相近
 *
 * 4. Segmentation fault：[已修复]
 *    - init_environment内部创建stream但未返回
 *    - 修复：删除内部stream创建
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "utils/utils.h"
#include "benchmark_config.h"
#include "benchmark_utils.h"

// HCCL头文件（仅在启用HCCL时包含）
#ifdef ENABLE_HCCL
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#endif

// Kernel函数声明
extern void launch_rdma_pingpong_latency(uint32_t block_dim, void* stream,
                                          uint64_t ffts_config, uint8_t* gva,
                                          int64_t msg_size, int64_t iterations,
                                          int64_t warmup, uint8_t* result_buffer);

extern void launch_rdma_bandwidth(uint32_t block_dim, void* stream,
                                   uint64_t ffts_config, uint8_t* gva,
                                   int64_t msg_size, int64_t iterations,
                                   uint8_t* result_buffer);

extern void launch_mte_pingpong_latency(uint32_t block_dim, void* stream,
                                         uint64_t ffts_config, uint8_t* gva,
                                         int64_t msg_size, int64_t iterations,
                                         int64_t warmup, uint8_t* result_buffer);

extern void launch_mte_bandwidth(uint32_t block_dim, void* stream,
                                  uint64_t ffts_config, uint8_t* gva,
                                  int64_t msg_size, int64_t iterations,
                                  uint8_t* result_buffer);

extern void launch_hidden_comm(uint32_t block_dim, void* stream,
                                uint64_t ffts_config, uint8_t* gva,
                                int64_t msg_size, int64_t iterations,
                                uint8_t* matmul_A, uint8_t* matmul_B, uint8_t* matmul_C,
                                int64_t M, int64_t K, int64_t N,
                                uint8_t* result_buffer);

extern void launch_matmul_compute(uint32_t block_dim, void* stream,
                                   uint8_t* A, uint8_t* B, uint8_t* C,
                                   int64_t M, int64_t K, int64_t N,
                                   uint8_t* result_buffer);

using namespace benchmark;

// 全局配置变量
int g_npus = 8;                 // 节点内NPU总数，用于计算物理设备ID
const char* ipport;             // rendezvous地址（TCP socket地址）
int f_rank = 0;                 // rank编号偏移量（用于多节点场景）
int f_npu = 0;                  // NPU编号偏移量（物理设备ID的起点）

aclshmemx_uniqueid_t default_flag_uid;  // uniqueid结构体（DEFAULT模式下使用）

// NPU频率 (MHz) - 用于将cycles转换为时间
// BUG警告：这个值需要确认，如果NPU实际频率不是1000MHz，
// 所有延迟和带宽计算都会出错！应该使用正确的频率值
const double NPU_FREQ_MHZ = 1000.0;

/**
 * 初始化ACL和SHMEM环境
 *
 * 注意：此函数不再内部创建stream，调用者需要自己创建stream
 * 修复说明：原bug是函数内部创建stream但不返回，导致调用者重复创建stream
 */
int init_environment(int rank, int world_size, uint64_t mem_size, EngineType engine) {
    // 计算物理设备ID：rank % g_npus + f_npu
    // rank: 当前进程的逻辑编号
    // g_npus: 节点内NPU总数
    // f_npu: NPU编号偏移量
    int32_t device_id = rank % g_npus + f_npu;
    int status = 0;

    // aclInit: 初始化ACL运行时环境
    // 参数: nullptr表示使用默认配置
    // 必须在调用任何ACL API之前执行
    // 返回值: ACL_SUCCESS表示成功
    status = aclInit(nullptr);

    // aclrtSetDevice: 设置当前进程使用的NPU设备
    // 参数: device_id - 物理NPU设备编号
    // 将进程绑定到指定的NPU，后续所有ACL操作在该设备上执行
    status = aclrtSetDevice(device_id);

    // 注意：不再在此函数内创建stream
    // stream由调用者创建，避免重复创建导致的资源冲突
    // 调用者应在调用此函数后执行：
    // aclrtStream stream = nullptr;
    // aclrtCreateStream(&stream);

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
    // - rank: 当前PE编号（进程ID）
    // - world_size: 总PE数量（进程总数）
    // - mem_size: 对称内存大小（字节）
    // - ipport: rendezvous地址字符串，如"tcp://127.0.0.1:8998"
    //   PE 0监听此地址，其他PE连接到此地址进行握手
    // - default_flag_uid: uniqueid结构体（DEFAULT模式下使用）
    // - &attributes: 属性结构体指针（输出参数）
    test_set_attr(rank, world_size, mem_size, ipport, default_flag_uid, &attributes);

    // 根据引擎类型设置数据传输引擎:
    // ACLSHMEM_DATA_OP_ROCE: RDMA引擎（跨节点通信）
    //   - 使用RoCE网络进行远程直接内存访问
    //   - 支持跨节点的高速低延迟数据传输
    // ACLSHMEM_DATA_OP_MTE: MTE引擎（节点内通信）
    //   - 使用片上MTE单元进行数据传输
    //   - 仅支持节点内NPU间通信
    //   - 高带宽、低延迟
    // ACLSHMEM_DATA_OP_SDMA: SDMA引擎（节点内通信）
    //   - 使用片上SDMA单元进行数据传输
    //   - 仅支持节点内NPU间通信
    switch (engine) {
        case EngineType::RDMA:
            // ACLSHMEM_DATA_OP_ROCE: 设置数据传输引擎为RDMA（RoCE协议）
            // RDMA引擎用于跨节点NPU间通信
            // 通过RoCE网络进行远程直接内存访问
            attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
            break;
        case EngineType::MTE:
            // ACLSHMEM_DATA_OP_MTE: 设置数据传输引擎为MTE
            // MTE引擎用于节点内NPU间通信
            // 使用片上MTE单元进行数据传输
            attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
            break;
        case EngineType::SDMA:
            // ACLSHMEM_DATA_OP_SDMA: 设置数据传输引擎为SDMA
            // SDMA引擎用于节点内NPU间通信
            // 使用片上SDMA单元进行数据传输
            attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;
            break;
        default:
            // 默认使用RDMA引擎
            attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    }

    // aclshmemx_set_conf_store_tls: 设置配置存储TLS（可选）
    // 参数详解:
    // - false: 是否启用TLS存储
    // - nullptr: 配置数据指针
    // - 0: 配置数据大小
    // 执行效果：设置shmem内部配置存储方式
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    // aclshmemx_init_attr: 初始化shmem运行时
    // 参数详解:
    // - BENCHMARK_INIT_FLAG: 初始化模式标志（根据是否启用MPI选择）
    //   可选模式:
    //   * ACLSHMEMX_INIT_WITH_DEFAULT: TCP socket模式（推荐）
    //   * ACLSHMEMX_INIT_WITH_MPI: 使用MPI进行初始化
    // - &attributes: 初始化属性结构体指针
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 执行后完成:
    // 1. 建立进程间通信通道
    // 2. 分配对称内存堆（Symmetric Heap）
    // 3. 初始化通信引擎（根据engine类型）
    // 4. 设置PE编号和通信组信息
    status = aclshmemx_init_attr(BENCHMARK_INIT_FLAG, &attributes);

    // BUG!!! 这里只返回status，丢失了stream指针
    // 调用者无法获取函数内部创建的stream，会导致资源管理混乱
    return status;
}

/**
 * 终止环境 - 清理ACL和SHMEM资源
 */
void finalize_environment(int rank) {
    // 计算物理设备ID
    int32_t device_id = rank % g_npus + f_npu;

    // aclshmem_finalize: 终止shmem运行时，释放所有shmem资源
    // 功能详解：
    // - 释放对称内存堆
    // - 关闭进程间通信通道
    // - 清理通信引擎状态
    // - 释放内部同步机制资源
    // 返回值: ACLSHMEM_SUCCESS表示成功
    aclshmem_finalize();

    // aclrtResetDevice: 重置NPU设备状态
    // 参数: device_id - 要重置的NPU设备编号
    // 执行效果：
    // - 清除设备上下文
    // - 释放设备上的计算资源
    aclrtResetDevice(device_id);

    // aclFinalize: 终止ACL运行时环境
    // 执行效果：
    // - 释放ACL内部资源
    // - 关闭驱动连接
    // - 清理CANN软件栈状态
    // 返回值: ACL_SUCCESS表示成功
    // 注意：调用后不能再执行任何ACL操作
    aclFinalize();
}

/**
 * ========== RDMA PingPong延迟测试 ==========
 *
 * 修复说明：Kernel中已添加magic value写入逻辑
 * - 发送方在发送前将magic value写入数据末尾
 * - 参见comm_benchmark_kernel.cpp中的rdma_pingpong_latency_kernel
 */
StatsResult test_rdma_pingpong_latency(aclrtStream stream, uint64_t ffts_config,
                                         uint8_t* gva, size_t msg_size,
                                         int iterations, int warmup) {
    // 分配结果buffer：存储每次迭代的cycles值
    // 大小：iterations * sizeof(int64_t) + sizeof(int64_t)（额外一个可能用于统计）
    uint8_t* result_buffer;
    size_t result_size = iterations * sizeof(int64_t) + sizeof(int64_t);
    // aclrtMalloc: 分配NPU设备内存
    // 参数: &result_buffer - 输出指针
    //       result_size - 内存大小
    //       ACL_MEM_MALLOC_HUGE_FIRST - 分配策略（优先使用大页内存）
    aclrtMalloc((void**)&result_buffer, result_size, ACL_MEM_MALLOC_HUGE_FIRST);

    // launch_rdma_pingpong_latency: 启动RDMA PingPong延迟测试Kernel
    // 参数详解:
    // - 1: block_dim（block数量，通常为1）
    // - stream: ACL流
    // - ffts_config: FFTS配置地址（用于硬件同步）
    // - gva: 对称内存地址（GVA格式）
    //   用于存放pingpong测试的数据
    // - msg_size: 消息大小（字节）
    // - iterations: 正式迭代次数
    // - warmup: warmup迭代次数（不计入统计）
    // - result_buffer: 结果buffer地址（存储每次迭代的cycles）
    //
    // Kernel内部实现（参见comm_benchmark_kernel.cpp）：
    // - Warmup阶段：预热数据传输路径
    // - 正式测试：记录每次迭代的cycles差值
    // - Magic value同步：发送方在数据末尾写入magic value，接收方检测
    launch_rdma_pingpong_latency(1, stream, ffts_config, gva,
                                  msg_size, iterations, warmup, result_buffer);

    // aclrtSynchronizeStream: 同步stream，等待Kernel执行完成
    aclrtSynchronizeStream(stream);

    // 分配Host端内存，用于接收结果数据
    int64_t* host_result;
    // aclrtMallocHost: 在Host端分配内存
    // 参数: &host_result - 输出指针
    //       result_size - 内存大小
    aclrtMallocHost((void**)&host_result, result_size);

    // aclrtMemcpy: 将结果从Device拷贝到Host
    // 参数详解:
    // - host_result: 目标地址（Host端）
    // - result_size: 数据大小
    // - result_buffer: 源地址（Device端）
    // - result_size: 数据大小
    // - ACL_MEMCPY_DEVICE_TO_HOST: 拷贝方向（Device到Host）
    aclrtMemcpy(host_result, result_size, result_buffer, result_size, ACL_MEMCPY_DEVICE_TO_HOST);

    // 将cycles转换为延迟时间（us）
    std::vector<double> latencies;
    for (int i = 0; i < iterations; i++) {
        // cycles_to_us: 将cycles转换为微秒
        // 参数: host_result[i] - cycles值
        //       NPU_FREQ_MHZ - NPU频率（MHz）
        // BUG警告：如果NPU_FREQ_MHZ不准确，所有延迟计算都会错误！
        double latency_us = cycles_to_us(host_result[i], NPU_FREQ_MHZ);
        latencies.push_back(latency_us);
    }

    // aclrtFreeHost: 释放Host端内存
    // 参数: aclrtMallocHost分配的内存指针
    aclrtFreeHost(host_result);

    // aclrtFree: 释放NPU设备内存
    // 参数: aclrtMalloc分配的设备内存指针
    aclrtFree(result_buffer);

    // compute_stats: 计算统计结果（平均值、标准差、最小值、最大值、中位数）
    return compute_stats(latencies);
}

/**
 * ========== RDMA带宽测试（改进版）==========
 *
 * 改进内容：
 * 1. 多轮测试取平均（提高稳定性）
 * 2. 正确的带宽计算（单向带宽，不乘2）
 *
 * 带宽计算公式：
 * - bandwidth = iterations * msg_size / total_time
 * - 注意：不乘2，因为这是单向带宽测试（发送方测量发送时间）
 */
StatsResult test_rdma_bandwidth(aclrtStream stream, uint64_t ffts_config,
                                 uint8_t* gva, size_t msg_size, int iterations,
                                 int warmup_rounds, int test_rounds) {
    // 分配结果buffer
    uint8_t* result_buffer;
    aclrtMalloc((void**)&result_buffer, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    std::vector<double> bandwidths;

    // 多轮测试取平均
    for (int round = 0; round < warmup_rounds + test_rounds; round++) {
        // launch_rdma_bandwidth: 启动RDMA带宽测试Kernel
        // block_dim = 1（单核测试）
        launch_rdma_bandwidth(1, stream, ffts_config, gva, msg_size, iterations, result_buffer);

        aclrtSynchronizeStream(stream);

        int64_t* host_result;
        aclrtMallocHost((void**)&host_result, sizeof(int64_t));
        aclrtMemcpy(host_result, sizeof(int64_t), result_buffer, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

        double total_time_us = cycles_to_us(host_result[0], NPU_FREQ_MHZ);

        // 带宽计算（单向带宽）
        // 总数据量 = iterations * msg_size
        double bw_gb_s = compute_bandwidth(iterations * msg_size, total_time_us);

        // 跳过 warmup 轮次
        if (round >= warmup_rounds) {
            bandwidths.push_back(bw_gb_s);
        }

        aclrtFreeHost(host_result);
    }

    aclrtFree(result_buffer);

    // 计算统计结果（平均带宽、标准差等）
    return compute_stats(bandwidths);
}

/**
 * ========== MTE PingPong测试 ==========
 *
 * 修复说明：Kernel中已添加magic value写入逻辑
 * - 发送方在发送前将magic value写入数据末尾
 * - 参见comm_benchmark_kernel.cpp中的mte_pingpong_latency_kernel
 * - 与RDMA pingpong使用相同的同步机制
 */
StatsResult test_mte_pingpong_latency(aclrtStream stream, uint64_t ffts_config,
                                       uint8_t* gva, size_t msg_size,
                                       int iterations, int warmup) {
    // 分配结果buffer
    uint8_t* result_buffer;
    size_t result_size = iterations * sizeof(int64_t) + sizeof(int64_t);
    aclrtMalloc((void**)&result_buffer, result_size, ACL_MEM_MALLOC_HUGE_FIRST);

    // launch_mte_pingpong_latency: 启动MTE PingPong延迟测试Kernel
    // 参数详解:
    // - 1: block_dim
    // - stream: ACL流
    // - ffts_config: FFTS配置地址
    // - gva: 对称内存地址
    // - msg_size: 消息大小（字节）
    // - iterations: 正式迭代次数
    // - warmup: warmup迭代次数
    // - result_buffer: 结果buffer
    //
    // Kernel内部实现（参见comm_benchmark_kernel.cpp）：
    // - 使用aclshmemx_mte_put_nbi发送数据（MTE引擎）
    // - 使用MTE3_S事件同步等待传输完成
    // - Magic value同步：发送方在数据末尾写入magic value
    launch_mte_pingpong_latency(1, stream, ffts_config, gva,
                                 msg_size, iterations, warmup, result_buffer);

    // aclrtSynchronizeStream: 同步stream
    aclrtSynchronizeStream(stream);

    // 分配Host端内存
    int64_t* host_result;
    aclrtMallocHost((void**)&host_result, result_size);

    // aclrtMemcpy: 拷贝结果到Host
    aclrtMemcpy(host_result, result_size, result_buffer, result_size, ACL_MEMCPY_DEVICE_TO_HOST);

    // 将cycles转换为延迟时间
    std::vector<double> latencies;
    for (int i = 0; i < iterations; i++) {
        double latency_us = cycles_to_us(host_result[i], NPU_FREQ_MHZ);
        latencies.push_back(latency_us);
    }

    // 释放Host端内存
    aclrtFreeHost(host_result);

    // 释放Device端内存
    aclrtFree(result_buffer);

    // 计算统计结果
    return compute_stats(latencies);
}

/**
 * ========== MTE带宽测试（改进版）==========
 *
 * 改进内容：
 * 1. 多轮测试取平均（提高稳定性）
 * 2. 正确的带宽计算（单向带宽，不乘2）
 *
 * 关于RDMA和MTE带宽相同的说明：
 * - 这不是bug，而是测试配置问题
 * - MTE用于节点内通信（片上互联）
 * - RDMA用于跨节点通信（RoCE网络）
 * - 如果测试在单节点内运行，RDMA可能使用节点内传输路径
 * - 建议：跨节点测试时使用RDMA，节点内测试时使用MTE，以获得准确性能对比
 */
StatsResult test_mte_bandwidth(aclrtStream stream, uint64_t ffts_config,
                                uint8_t* gva, size_t msg_size, int iterations,
                                int warmup_rounds, int test_rounds) {
    // 分配结果buffer
    uint8_t* result_buffer;
    aclrtMalloc((void**)&result_buffer, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    std::vector<double> bandwidths;

    // 多轮测试取平均
    for (int round = 0; round < warmup_rounds + test_rounds; round++) {
        // launch_mte_bandwidth: 启动MTE带宽测试Kernel
        // block_dim = 1（单核测试）
        //
        // Kernel 内部流程：
        // 发送方：
        //   1. 批量发送 iterations 次数据（aclshmemx_mte_put_nbi）
        //   2. WaitFlag 等待 MTE3_S 事件（发送完成）
        //   3. 发送通知消息到 notify_addr
        //   4. 轮询 ack_addr 等待确认
        // 接收方：
        //   1. 轮询 notify_addr 等待通知
        //   2. 发送确认到 ack_addr
        launch_mte_bandwidth(1, stream, ffts_config, gva, msg_size, iterations, result_buffer);

        aclrtSynchronizeStream(stream);

        int64_t* host_result;
        aclrtMallocHost((void**)&host_result, sizeof(int64_t));
        aclrtMemcpy(host_result, sizeof(int64_t), result_buffer, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

        double total_time_us = cycles_to_us(host_result[0], NPU_FREQ_MHZ);

        // 带宽计算（单向带宽）
        // 总数据量 = iterations * msg_size
        double bw_gb_s = compute_bandwidth(iterations * msg_size, total_time_us);

        // 跳过 warmup 轮次
        if (round >= warmup_rounds) {
            bandwidths.push_back(bw_gb_s);
        }

        aclrtFreeHost(host_result);
    }

    aclrtFree(result_buffer);

    // 计算统计结果（平均带宽、标准差等）
    return compute_stats(bandwidths);
}

/**
 * ========== CPU中转测试 ==========
 *
 * 测试CPU Host端作为中转的传输延迟（Device->Host->Device）
 * 用于对比NPU直连通信的性能优势
 */
StatsResult test_cpu_transfer(aclrtStream stream, size_t msg_size, int iterations, int warmup) {
    // 延迟结果数组
    std::vector<double> latencies;

    // 分配Device端内存
    void* device_buf;
    // aclrtMalloc: 在NPU设备上分配内存
    // 参数: &device_buf - 输出指针
    //       msg_size - 内存大小
    //       ACL_MEM_MALLOC_HUGE_FIRST - 分配策略
    aclrtMalloc(&device_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);

    // 分配Host端内存
    void* host_buf;
    // aclrtMallocHost: 在Host端分配内存
    aclrtMallocHost(&host_buf, msg_size);

    // memset: 初始化Host端数据为0xAA
    // 用于测试数据传输
    memset(host_buf, 0xAA, msg_size);

    // warmup迭代：预热数据传输路径
    // 不计入统计结果
    for (int i = 0; i < warmup; i++) {
        // aclrtMemcpy: Device -> Host拷贝
        // 参数详解:
        // - host_buf: 目标地址（Host端）
        // - msg_size: 数据大小
        // - device_buf: 源地址（Device端）
        // - msg_size: 数据大小
        // - ACL_MEMCPY_DEVICE_TO_HOST: 拷贝方向
        aclrtMemcpy(host_buf, msg_size, device_buf, msg_size, ACL_MEMCPY_DEVICE_TO_HOST);

        // aclrtMemcpy: Host -> Device拷贝
        // 参数详解:
        // - device_buf: 目标地址（Device端）
        // - msg_size: 数据大小
        // - host_buf: 源地址（Host端）
        // - msg_size: 数据大小
        // - ACL_MEMCPY_HOST_TO_DEVICE: 拷贝方向
        aclrtMemcpy(device_buf, msg_size, host_buf, msg_size, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    // 正式迭代：测量传输延迟
    for (int i = 0; i < iterations; i++) {
        // std::chrono::high_resolution_clock::now(): 获取高精度时间戳
        // 用于测量传输延迟
        auto start = std::chrono::high_resolution_clock::now();

        // aclrtMemcpy: Device -> Host拷贝
        aclrtMemcpy(host_buf, msg_size, device_buf, msg_size, ACL_MEMCPY_DEVICE_TO_HOST);

        // aclrtMemcpy: Host -> Device拷贝
        aclrtMemcpy(device_buf, msg_size, host_buf, msg_size, ACL_MEMCPY_HOST_TO_DEVICE);

        // 获取结束时间戳
        auto end = std::chrono::high_resolution_clock::now();

        // std::chrono::duration: 计算时间差（微秒）
        // 将start到end的时间差转换为微秒
        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(latency_us);
    }

    // aclrtFree: 释放Device端内存
    aclrtFree(device_buf);

    // aclrtFreeHost: 释放Host端内存
    aclrtFreeHost(host_buf);

    // 计算统计结果
    return compute_stats(latencies);
}

/**
 * ========== 通信隐藏测试 ==========
 *
 * 测试通信与计算重叠的性能
 * 用于评估通信隐藏（Communication-Computation Overlap）的效果
 */
double test_hidden_comm(aclrtStream stream, uint64_t ffts_config,
                         uint8_t* gva, size_t msg_size, int iterations,
                         ComputeConfig compute_cfg) {
    // 分配结果buffer：存储每次迭代的重叠时间
    uint8_t* result_buffer;
    aclrtMalloc((void**)&result_buffer, iterations * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    // 分配矩阵乘法所需的矩阵A、B、C
    uint8_t* matmul_A, *matmul_B, *matmul_C;

    // 计算矩阵大小
    size_t A_size = compute_cfg.M * compute_cfg.K * sizeof(float);
    size_t B_size = compute_cfg.K * compute_cfg.N * sizeof(float);
    size_t C_size = compute_cfg.M * compute_cfg.N * sizeof(float);

    // aclrtMalloc: 分配矩阵A的Device端内存
    aclrtMalloc(reinterpret_cast<void**>(&matmul_A), A_size, ACL_MEM_MALLOC_HUGE_FIRST);

    // BUG!!! 这里使用了&而不是&matmul_B的正确类型
    // 应该是: aclrtMalloc(reinterpret_cast<void**>(&matmul_B), ...)
    // 错误写法可能导致编译警告或运行时问题
    aclrtMalloc(reinterpret_cast<void**>(&matmul_B), B_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&matmul_C), C_size, ACL_MEM_MALLOC_HUGE_FIRST);

    // launch_hidden_comm: 启动通信隐藏测试Kernel
    // 参数详解:
    // - 1: block_dim
    // - stream: ACL流
    // - ffts_config: FFTS配置地址
    // - gva: 对称内存地址
    // - msg_size: 通信消息大小
    // - iterations: 迭代次数
    // - matmul_A, matmul_B, matmul_C: 矩阵乘法输入输出地址
    // - compute_cfg.M, K, N: 矩阵乘法维度
    // - result_buffer: 结果buffer
    launch_hidden_comm(1, stream, ffts_config, gva, msg_size, iterations,
                       matmul_A, matmul_B, matmul_C,
                       compute_cfg.M, compute_cfg.K, compute_cfg.N,
                       result_buffer);

    // aclrtSynchronizeStream: 同步stream
    aclrtSynchronizeStream(stream);

    // 分配Host端内存，接收结果
    int64_t* host_result;
    aclrtMallocHost((void**)&host_result, iterations * sizeof(int64_t));

    // aclrtMemcpy: 拷贝结果到Host
    aclrtMemcpy(host_result, iterations * sizeof(int64_t), result_buffer,
                iterations * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

    // 计算平均重叠时间
    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        total_time += cycles_to_us(host_result[i], NPU_FREQ_MHZ);
    }
    double avg_overlap_time = total_time / iterations;

    // 释放Host端内存
    aclrtFreeHost(host_result);

    // 释放Device端内存
    aclrtFree(result_buffer);
    aclrtFree(matmul_A);
    aclrtFree(matmul_B);
    aclrtFree(matmul_C);

    // 返回平均重叠时间
    return avg_overlap_time;
}

/**
 * ========== 主测试流程 ==========
 */
int run_benchmark(int rank, int world_size) {
    // 定义对称内存大小：256MB
    uint64_t mem_size = 256UL * 1024UL * 1024UL;

    // make_dir: 创建结果目录
    make_dir("results");

    // CSVWriter: 创建CSV文件写入器
    // 用于记录测试结果
    CSVWriter latency_csv("results/latency_results.csv");
    CSVWriter bandwidth_csv("results/bandwidth_results.csv");
    CSVWriter hidden_csv("results/hidden_results.csv");

    // 定义要测试的引擎类型列表
    std::vector<EngineType> engines = {
        EngineType::RDMA,  // RDMA引擎（跨节点）
        EngineType::MTE,   // MTE引擎（节点内）
    };

    // 打印测试信息
    std::cout << "\n==================== Comm Benchmark ====================\n";
    std::cout << "Mode: " << BENCHMARK_MODE_NAME << "\n";
    std::cout << "HCCL: " << BENCHMARK_HCCL_MODE_NAME << "\n";
    std::cout << "Rank: " << rank << ", WorldSize: " << world_size << "\n";
    print_separator();

    // ========== RDMA/MTE测试 ==========
    for (EngineType engine : engines) {
        std::cout << "\n---------- Testing Engine: " << engine_name(engine) << " ----------\n";

        // init_environment: 初始化ACL和SHMEM环境
        // 注意：此函数已修复，不再内部创建stream
        init_environment(rank, world_size, mem_size, engine);

        // 调用者创建stream（修复后不再重复创建，避免资源冲突）
        aclrtStream stream = nullptr;
        aclrtCreateStream(&stream);

        // util_get_ffts_config: 获取FFTS配置地址
        // FFTS = Fast Fabric Task Scheduler，用于硬件同步
        uint64_t ffts_config = util_get_ffts_config();

        // aclshmem_malloc: 分配对称内存（用于通信数据缓冲区）
        // 参数详解:
        // - mem_size: 请求的内存大小（256MB）
        // 返回: 对称内存指针（GVA格式）
        // 对称内存用途：
        // - 存放pingpong测试的数据
        // - 存放带宽测试的数据
        // - 用于跨PE的数据传输
        uint8_t* gva = (uint8_t*)aclshmem_malloc(mem_size);

        // PingPong延迟测试
        std::cout << "\n[PingPong Latency Test]\n";
        for (size_t msg_size : MSG_SIZES) {
            // get_iterations: 根据消息大小获取迭代次数
            int iterations = get_iterations(msg_size);

            // get_warmup_iterations: 根据消息大小获取warmup次数
            int warmup = get_warmup_iterations(msg_size);

            // print_test_header: 打印测试头信息
            print_test_header({rank, world_size, engine, TestType::PINGPONG_LATENCY,
                               msg_size, iterations, warmup, ipport});

            StatsResult result;
            if (engine == EngineType::RDMA) {
                // test_rdma_pingpong_latency: 执行RDMA pingpong延迟测试
                result = test_rdma_pingpong_latency(stream, ffts_config, gva,
                                                     msg_size, iterations, warmup);
            } else if (engine == EngineType::MTE) {
                // test_mte_pingpong_latency: 执行MTE pingpong延迟测试
                // 问题：如果latency是定值，需要检查Kernel实现
                result = test_mte_pingpong_latency(stream, ffts_config, gva,
                                                    msg_size, iterations, warmup);
            }

            // print_result: 打印测试结果
            print_result(result);

            // latency_csv.write_row: 将结果写入CSV文件
            latency_csv.write_row(engine_name(engine), "pingpong_latency",
                                   msg_size, iterations, result);
        }

        // 带宽测试
        std::cout << "\n[Bandwidth Test]\n";
        for (size_t msg_size : MSG_SIZES) {
            // 跳过小消息的带宽测试（小于64KB）
            if (msg_size < 64 * 1024) continue;

            // 根据消息大小设置迭代次数
            int iterations = (msg_size <= 8 * 1024 * 1024) ? 1000 : 100;

            // warmup和正式测试轮次
            int warmup_rounds = 3;
            int test_rounds = 10;

            // ========== 数据初始化 ==========
            // 初始化 src_addr 的数据，使得通知消息内容是 pe_id + MAGIC_VAL
            constexpr uint32_t BW_MAGIC_VAL = 10;
            uint32_t* bw_init_data;
            aclrtMallocHost((void**)&bw_init_data, msg_size);
            for (size_t i = 0; i < msg_size / sizeof(uint32_t); i++) {
                bw_init_data[i] = rank + BW_MAGIC_VAL;
            }
            aclrtMemcpy(gva + rank * msg_size, msg_size, bw_init_data, msg_size, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtFreeHost(bw_init_data);

            // 打印测试头信息
            print_test_header({rank, world_size, engine, TestType::BANDWIDTH,
                               msg_size, iterations, 0, ipport});
            std::cout << "WarmupRounds: " << warmup_rounds << ", TestRounds: " << test_rounds << "\n";

            StatsResult result;
            if (engine == EngineType::RDMA) {
                // test_rdma_bandwidth: 执行RDMA带宽测试
                result = test_rdma_bandwidth(stream, ffts_config, gva, msg_size,
                                              iterations, warmup_rounds, test_rounds);
            } else if (engine == EngineType::MTE) {
                // test_mte_bandwidth: 执行MTE带宽测试
                result = test_mte_bandwidth(stream, ffts_config, gva, msg_size,
                                             iterations, warmup_rounds, test_rounds);
            }

            std::cout << "Bandwidth: " << result.mean << " +/- " << result.std
                      << " GB/s (min=" << result.min << ", max=" << result.max << ")\n";

            // bandwidth_csv.write_row: 将结果写入CSV文件
            bandwidth_csv.write_row(engine_name(engine), "bandwidth",
                                      msg_size, iterations, result);
        }

        // 通信隐藏测试
        std::cout << "\n[Hidden Communication Test]\n";
        if (rank == 0) {
            for (size_t msg_size : MSG_SIZES) {
                // 跳过小消息的隐藏测试
                if (msg_size < 256 * 1024) continue;

                // 设置迭代次数
                int iterations = (msg_size <= 8 * 1024 * 1024) ? 100 : 20;

                // match_compute: 根据通信消息大小匹配计算量
                ComputeConfig compute_cfg = match_compute(msg_size);

                std::cout << "MsgSize: " << msg_size << " bytes"
                          << ", Compute: " << compute_cfg.M << "x" << compute_cfg.K << "x" << compute_cfg.N
                          << "\n";

                // 先测量纯通信时间
                StatsResult comm_result;
                if (engine == EngineType::RDMA) {
                    comm_result = test_rdma_pingpong_latency(stream, ffts_config, gva,
                                                              msg_size, iterations, 10);
                } else {
                    comm_result = test_mte_pingpong_latency(stream, ffts_config, gva,
                                                             msg_size, iterations, 10);
                }
                double comm_time = comm_result.mean;

                // test_hidden_comm: 测试通信与计算重叠的时间
                double overlap_time = test_hidden_comm(stream, ffts_config, gva,
                                                        msg_size, iterations, compute_cfg);

                // 计算通信隐藏率
                // 公式: hidden_rate = 100% * (1 - overlap_time / (comm_time * 2))
                // comm_time * 2: 双向通信时间
                double hidden_rate = 100.0 * (1.0 - overlap_time / (comm_time * 2));

                std::cout << "CommTime: " << comm_time << " us"
                          << ", OverlapTime: " << overlap_time << " us"
                          << ", HiddenRate: " << hidden_rate << " %\n";

                // hidden_csv.write_hidden_result: 写入隐藏测试结果
                hidden_csv.write_hidden_result(engine_name(engine), msg_size,
                                            comm_time, 0, overlap_time, hidden_rate);
            }
        }

        // aclshmem_free: 释放对称内存
        // 参数: aclshmem_malloc返回的对称内存指针
        // 必须与aclshmem_malloc配对使用
        aclshmem_free(gva);

        // aclrtDestroyStream: 销毁ACL流
        // 参数: aclrtCreateStream创建的流指针
        aclrtDestroyStream(stream);

        // finalize_environment: 终止ACL和SHMEM环境
        // 注意：调用后ACL环境被销毁，不能再执行任何ACL操作
        finalize_environment(rank);
    }

    // ========== CPU中转测试 ==========
    // BUG!!! 这里是导致segmentation fault的关键位置
    // 在finalize_environment之后，ACL环境已被销毁
    // 再次调用init_environment时，需要确保正确重新初始化
    std::cout << "\n---------- Testing Engine: CPU_D2H_H2D ----------\n";

    // init_environment: 再次初始化ACL和SHMEM环境
    // 注意：此函数已修复，不再内部创建stream
    init_environment(rank, world_size, mem_size, EngineType::RDMA);

    // 调用者创建stream（修复后正常工作）
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    std::cout << "\n[CPU Transfer Latency Test]\n";
    for (size_t msg_size : MSG_SIZES) {
        // 获取迭代次数
        int iterations = get_iterations(msg_size);

        // 获取warmup次数
        int warmup = get_warmup_iterations(msg_size);

        // test_cpu_transfer: 执行CPU中转延迟测试
        // 如果这里出现segmentation fault，可能是：
        // 1. stream资源冲突（init_environment和外部创建的两个stream）
        // 2. ACL环境初始化不正确
        // 3. aclrtMalloc或aclrtMemcpy使用已销毁的资源
        StatsResult result = test_cpu_transfer(stream, msg_size, iterations, warmup);

        // latency_csv.write_row: 写入结果
        latency_csv.write_row("CPU_D2H_H2D", "pingpong_latency",
                               msg_size, iterations, result);
    }

    // aclrtDestroyStream: 销毁ACL流
    aclrtDestroyStream(stream);

    // finalize_environment: 终止ACL和SHMEM环境
    finalize_environment(rank);

// ========== HCCL测试（条件编译） ==========
#ifdef ENABLE_HCCL
    std::cout << "\n---------- Testing Engine: HCCL ----------\n";
    std::cout << "[HCCL] Huawei Collective Communication Library Test\n";

    // 计算物理设备ID
    int32_t device_id = rank % g_npus + f_npu;

    // aclrtSetDevice: 设置当前进程使用的NPU设备
    aclrtSetDevice(device_id);

    // aclrtCreateStream: 创建ACL流
    aclrtStream hccl_stream = nullptr;
    aclrtCreateStream(&hccl_stream);

    // HcclComm: HCCL通信组句柄
    HcclComm hccl_comm = nullptr;

    // HcclRootInfo: HCCL根信息结构体
    // 用于进程间通信组初始化
    HcclRootInfo root_info;

    // HcclGetRootInfo: 获取根信息
    // 注意：多进程场景下，root_info需要由rank0广播给其他进程
    HcclGetRootInfo(&root_info);

    // HcclCommInitRootInfo: 使用根信息初始化通信组
    // 参数详解:
    // - world_size: 通信组大小（进程总数）
    // - &root_info: 根信息指针
    // - rank: 当前进程编号
    // - &hccl_comm: 通信组句柄（输出参数）
    HcclCommInitRootInfo(world_size, &root_info, rank, &hccl_comm);

    std::cout << "[HCCL] Rank " << rank
              << " of " << world_size << " initialized\n";

    // HCCL PingPong延迟测试
    std::cout << "\n[HCCL PingPong Latency Test]\n";

    // hccl_peer: pingpong通信的目标PE
    uint32_t hccl_peer = (rank == 0) ? 1 : 0;

    for (size_t msg_size : MSG_SIZES) {
        int iterations = get_iterations(msg_size);
        int warmup = get_warmup_iterations(msg_size);

        std::cout << "MsgSize: " << msg_size << " bytes, Iterations: " << iterations << "\n";

        // 分配发送和接收buffer
        void* send_buf;
        void* recv_buf;
        aclrtMalloc(&send_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&recv_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);

        std::vector<double> latencies;

        // warmup迭代
        for (int i = 0; i < warmup; i++) {
            if (rank == 0) {
                // HcclSend: 发送数据到目标PE
                // 参数详解:
                // - send_buf: 发送数据地址
                // - msg_size: 数据大小
                // - HcclDataType::HCCL_DATA_TYPE_UINT8: 数据类型
                // - hccl_peer: 目标PE编号
                // - hccl_comm: 通信组句柄
                // - hccl_stream: ACL流
                HcclSend(send_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);

                // HcclRecv: 从目标PE接收数据
                HcclRecv(recv_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
            } else {
                // rank != 0: 先接收，再发送（pingpong响应）
                HcclRecv(recv_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
                HcclSend(send_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
            }
            // aclrtSynchronizeStream: 同步stream
            aclrtSynchronizeStream(hccl_stream);
        }

        // 正式迭代：测量延迟
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();

            if (rank == 0) {
                HcclSend(send_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
                HcclRecv(recv_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
            } else {
                HcclRecv(recv_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
                HcclSend(send_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
            }
            aclrtSynchronizeStream(hccl_stream);

            auto end = std::chrono::high_resolution_clock::now();
            double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
            latencies.push_back(latency_us);
        }

        // 计算统计结果
        StatsResult result = compute_stats(latencies);
        print_result(result);

        // latency_csv.write_row: 写入结果
        latency_csv.write_row("HCCL", "pingpong_latency", msg_size, iterations, result);

        // aclrtFree: 释放Device端内存
        aclrtFree(send_buf);
        aclrtFree(recv_buf);
    }

    // HCCL AllReduce带宽测试
    std::cout << "\n[HCCL AllReduce Bandwidth Test]\n";
    for (size_t msg_size : MSG_SIZES) {
        if (msg_size < 64 * 1024) continue;

        int iterations = (msg_size <= 8 * 1024 * 1024) ? 1000 : 100;

        void* buf;
        aclrtMalloc(&buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);

        // warmup
        for (int i = 0; i < 10; i++) {
            // HcclAllReduce: 执行AllReduce操作
            // 参数详解:
            // - buf: 输入和输出buffer（原地操作）
            // - msg_size / sizeof(float): 数据元素数量
            // - HcclDataType::HCCL_DATA_TYPE_FLOAT: 数据类型
            // - HcclReduceOp::HCCL_REDUCE_SUM: Reduce操作类型（求和）
            // - hccl_comm: 通信组句柄
            // - hccl_stream: ACL流
            HcclAllReduce(buf, buf, msg_size / sizeof(float), HcclDataType::HCCL_DATA_TYPE_FLOAT,
                         HcclReduceOp::HCCL_REDUCE_SUM, hccl_comm, hccl_stream);
        }
        aclrtSynchronizeStream(hccl_stream);

        // 正式测试：测量带宽
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            HcclAllReduce(buf, buf, msg_size / sizeof(float), HcclDataType::HCCL_DATA_TYPE_FLOAT,
                         HcclReduceOp::HCCL_REDUCE_SUM, hccl_comm, hccl_stream);
        }
        aclrtSynchronizeStream(hccl_stream);
        auto end = std::chrono::high_resolution_clock::now();

        // 计算总时间和带宽
        double total_time_us = std::chrono::duration<double, std::micro>(end - start).count();

        // compute_bandwidth: 计算带宽
        // msg_size * iterations * 2: AllReduce的数据量（发送+接收各一次）
        double bw_gb_s = compute_bandwidth(msg_size * iterations * 2, total_time_us);

        std::cout << "Bandwidth: " << bw_gb_s << " GB/s\n";

        // bandwidth_csv.write_row: 写入结果
        bandwidth_csv.write_row("HCCL", "allreduce_bandwidth", msg_size, iterations,
                                {bw_gb_s, 0, bw_gb_s, bw_gb_s, bw_gb_s});

        aclrtFree(buf);
    }

    // HCCL AllGather测试
    std::cout << "\n[HCCL AllGather Bandwidth Test]\n";
    for (size_t msg_size : MSG_SIZES) {
        if (msg_size < 64 * 1024) continue;

        int iterations = (msg_size <= 8 * 1024 * 1024) ? 100 : 10;

        void* send_buf;
        void* recv_buf;

        // recv_size: 接收buffer大小（每个PE发送msg_size，总共world_size份）
        size_t recv_size = msg_size * world_size;

        aclrtMalloc(&send_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&recv_buf, recv_size, ACL_MEM_MALLOC_HUGE_FIRST);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            // HcclAllGather: 执行AllGather操作
            // 参数详解:
            // - send_buf: 发送数据地址
            // - recv_buf: 接收数据地址
            // - msg_size / sizeof(float): 每个PE发送的元素数量
            // - HcclDataType::HCCL_DATA_TYPE_FLOAT: 数据类型
            // - hccl_comm: 通信组句柄
            // - hccl_stream: ACL流
            HcclAllGather(send_buf, recv_buf, msg_size / sizeof(float),
                         HcclDataType::HCCL_DATA_TYPE_FLOAT, hccl_comm, hccl_stream);
        }
        aclrtSynchronizeStream(hccl_stream);
        auto end = std::chrono::high_resolution_clock::now();

        double total_time_us = std::chrono::duration<double, std::micro>(end - start).count();

        // compute_bandwidth: 计算带宽
        // recv_size * iterations: AllGather的总数据量
        double bw_gb_s = compute_bandwidth(recv_size * iterations, total_time_us);

        std::cout << "MsgSize: " << msg_size << " bytes, AllGather BW: " << bw_gb_s << " GB/s\n";
        bandwidth_csv.write_row("HCCL", "allgather_bandwidth", msg_size, iterations,
                                {bw_gb_s, 0, bw_gb_s, bw_gb_s, bw_gb_s});

        aclrtFree(send_buf);
        aclrtFree(recv_buf);
    }

    // HCCL ReduceScatter测试
    std::cout << "\n[HCCL ReduceScatter Bandwidth Test]\n";
    for (size_t msg_size : MSG_SIZES) {
        if (msg_size < 64 * 1024) continue;

        int iterations = (msg_size <= 8 * 1024 * 1024) ? 100 : 10;

        void* send_buf;
        void* recv_buf;

        // send_size: 发送buffer大小（每个PE接收msg_size，总共world_size份）
        size_t send_size = msg_size * world_size;

        aclrtMalloc(&send_buf, send_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&recv_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            // HcclReduceScatter: 执行ReduceScatter操作
            // 参数详解:
            // - send_buf: 发送数据地址
            // - recv_buf: 接收数据地址
            // - msg_size / sizeof(float): 每个PE接收的元素数量
            // - HcclDataType::HCCL_DATA_TYPE_FLOAT: 数据类型
            // - HcclReduceOp::HCCL_REDUCE_SUM: Reduce操作类型
            // - hccl_comm: 通信组句柄
            // - hccl_stream: ACL流
            HcclReduceScatter(send_buf, recv_buf, msg_size / sizeof(float),
                             HcclDataType::HCCL_DATA_TYPE_FLOAT,
                             HcclReduceOp::HCCL_REDUCE_SUM, hccl_comm, hccl_stream);
        }
        aclrtSynchronizeStream(hccl_stream);
        auto end = std::chrono::high_resolution_clock::now();

        double total_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        double bw_gb_s = compute_bandwidth(send_size * iterations, total_time_us);

        std::cout << "MsgSize: " << msg_size << " bytes, ReduceScatter BW: " << bw_gb_s << " GB/s\n";
        bandwidth_csv.write_row("HCCL", "reducescatter_bandwidth", msg_size, iterations,
                                {bw_gb_s, 0, bw_gb_s, bw_gb_s, bw_gb_s});

        aclrtFree(send_buf);
        aclrtFree(recv_buf);
    }

    // HcclCommDestroy: 销毁HCCL通信组
    HcclCommDestroy(hccl_comm);

    // aclrtDestroyStream: 销毁ACL流
    aclrtDestroyStream(hccl_stream);

    // aclrtResetDevice: 重置NPU设备
    aclrtResetDevice(device_id);
#else
    std::cout << "\n---------- HCCL Test Skipped ----------\n";
    std::cout << "[INFO] HCCL not enabled. Define ENABLE_HCCL in benchmark_config.h to enable.\n";
#endif

    std::cout << "\n==================== Benchmark Complete ====================\n";
    std::cout << "Results saved to results/ directory\n";

    return 0;
}

/**
 * 主函数：解析命令行参数并运行测试
 */
int main(int argc, char* argv[]) {
    // 检查命令行参数数量
    if (argc < 7) {
        std::cout << "Usage: ./comm_benchmark <n_ranks> <rank_id> <ipport> <g_npus> <f_rank> <f_npu>\n";
        std::cout << "Example: ./comm_benchmark 2 0 tcp://127.0.0.1:8765 8 0 0\n";
        return -1;
    }

    // 解析命令行参数
    int argIdx = 1;

    // n_ranks: 总进程数量（通信组大小）
    int n_ranks = atoi(argv[argIdx++]);

    // rank_id: 当前进程编号（PE编号）
    int rank_id = atoi(argv[argIdx++]);

    // ipport: rendezvous地址（TCP socket地址）
    // 格式: "tcp://IP:PORT"
    // PE 0监听此地址，其他PE连接到此地址
    ipport = argv[argIdx++];

    // g_npus: 节点内NPU总数
    g_npus = atoi(argv[argIdx++]);

    // f_rank: rank编号偏移量（用于多节点场景）
    f_rank = atoi(argv[argIdx++]);

    // f_npu: NPU编号偏移量（物理设备ID的起点）
    f_npu = atoi(argv[argIdx++]);

    // check_env: 检查环境变量和配置
    if (!check_env()) {
        return -1;
    }

    // 打印benchmark模式信息
    std::cout << "\n[Benchmark Mode] " << BENCHMARK_MODE_NAME << "\n";
    std::cout << "[HCCL Status] " << BENCHMARK_HCCL_MODE_NAME << "\n";

    // run_benchmark: 执行benchmark测试
    int status = run_benchmark(rank_id, n_ranks);

    std::cout << "[SUCCESS] Benchmark completed for rank " << rank_id << "\n";
    return status;
}