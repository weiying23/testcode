/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Comm Benchmark主程序 - NPU通信性能对比测试
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>

#include "acl/acl.h"
#include "shmem_api.h"
#include "shmemi_host_common.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "benchmark_config.h"
#include "benchmark_utils.h"

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

// 全局配置
int g_npus = 8;
const char* ipport;
int f_rank = 0;
int f_npu = 0;
const char* test_type_str;

// NPU频率 (MHz)，用于转换cycles到时间
const double NPU_FREQ_MHZ = 1000.0;

// 初始化ACL和SHMEM
int init_environment(int rank, int world_size, uint64_t mem_size, EngineType engine) {
    int32_t device_id = rank % g_npus + f_npu;
    int status = 0;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);

    shmem_init_attr_t* attributes;
    status = shmem_set_attr(rank, world_size, mem_size, ipport, &attributes);

    // 根据engine类型设置通信引擎
    switch (engine) {
        case EngineType::RDMA:
            attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
            break;
        case EngineType::MTE:
            attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_MTE;
            break;
        case EngineType::SDMA:
            attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_SDMA;
            break;
        default:
            attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
    }

    shmem_set_conf_store_tls(false, nullptr, 0);
    status = shmem_init_attr(attributes);

    return status;
}

// 清理环境
void finalize_environment(int rank) {
    int32_t device_id = rank % g_npus + f_npu;
    shmem_finalize();
    aclrtResetDevice(device_id);
    aclFinalize();
}

// ========== RDMA PingPong测试 ==========<
StatsResult test_rdma_pingpong_latency(aclrtStream stream, uint64_t ffts_config,
                                         uint8_t* gva, size_t msg_size,
                                         int iterations, int warmup) {
    // 分配结果缓冲区
    uint8_t* result_buffer;
    size_t result_size = iterations * sizeof(int64_t) + sizeof(int64_t);
    aclrtMalloc(&result_buffer, result_size, ACL_MEM_MALLOC_HUGE_FIRST);

    // 调用kernel
    launch_rdma_pingpong_latency(1, stream, ffts_config, gva,
                                  msg_size, iterations, warmup, result_buffer);
    aclrtSynchronizeStream(stream);

    // 读取结果
    int64_t* host_result;
    aclrtMallocHost((void**)&host_result, result_size);
    aclrtMemcpy(host_result, result_size, result_buffer, result_size, ACL_MEMCPY_DEVICE_TO_HOST);

    // 转换cycles到微秒，并计算统计结果
    std::vector<double> latencies;
    for (int i = 0; i < iterations; i++) {
        double latency_us = cycles_to_us(host_result[i], NPU_FREQ_MHZ);
        latencies.push_back(latency_us);
    }

    aclrtFreeHost(host_result);
    aclrtFree(result_buffer);

    return compute_stats(latencies);
}

// ========== RDMA带宽测试 ==========<
StatsResult test_rdma_bandwidth(aclrtStream stream, uint64_t ffts_config,
                                 uint8_t* gva, size_t msg_size, int iterations) {
    uint8_t* result_buffer;
    aclrtMalloc(&result_buffer, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    launch_rdma_bandwidth(1, stream, ffts_config, gva, msg_size, iterations, result_buffer);
    aclrtSynchronizeStream(stream);

    int64_t* host_result;
    aclrtMallocHost((void**)&host_result, sizeof(int64_t));
    aclrtMemcpy(host_result, sizeof(int64_t), result_buffer, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

    // 计算带宽
    double total_time_us = cycles_to_us(host_result[0], NPU_FREQ_MHZ);
    double bw_gb_s = compute_bandwidth(msg_size * iterations, total_time_us);

    aclrtFreeHost(host_result);
    aclrtFree(result_buffer);

    // 带宽测试返回单值，没有std等
    return {bw_gb_s, 0, bw_gb_s, bw_gb_s, bw_gb_s};
}

// ========== MTE PingPong测试 ==========<
StatsResult test_mte_pingpong_latency(aclrtStream stream, uint64_t ffts_config,
                                       uint8_t* gva, size_t msg_size,
                                       int iterations, int warmup) {
    uint8_t* result_buffer;
    size_t result_size = iterations * sizeof(int64_t) + sizeof(int64_t);
    aclrtMalloc(&result_buffer, result_size, ACL_MEM_MALLOC_HUGE_FIRST);

    launch_mte_pingpong_latency(1, stream, ffts_config, gva,
                                 msg_size, iterations, warmup, result_buffer);
    aclrtSynchronizeStream(stream);

    int64_t* host_result;
    aclrtMallocHost((void**)&host_result, result_size);
    aclrtMemcpy(host_result, result_size, result_buffer, result_size, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<double> latencies;
    for (int i = 0; i < iterations; i++) {
        double latency_us = cycles_to_us(host_result[i], NPU_FREQ_MHZ);
        latencies.push_back(latency_us);
    }

    aclrtFreeHost(host_result);
    aclrtFree(result_buffer);

    return compute_stats(latencies);
}

// ========== MTE带宽测试 ==========<
StatsResult test_mte_bandwidth(aclrtStream stream, uint64_t ffts_config,
                                uint8_t* gva, size_t msg_size, int iterations) {
    uint8_t* result_buffer;
    aclrtMalloc(&result_buffer, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    launch_mte_bandwidth(1, stream, ffts_config, gva, msg_size, iterations, result_buffer);
    aclrtSynchronizeStream(stream);

    int64_t* host_result;
    aclrtMallocHost((void**)&host_result, sizeof(int64_t));
    aclrtMemcpy(host_result, sizeof(int64_t), result_buffer, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

    double total_time_us = cycles_to_us(host_result[0], NPU_FREQ_MHZ);
    double bw_gb_s = compute_bandwidth(msg_size * iterations, total_time_us);

    aclrtFreeHost(host_result);
    aclrtFree(result_buffer);

    return {bw_gb_s, 0, bw_gb_s, bw_gb_s, bw_gb_s};
}

// ========== CPU中转测试 ==========<
StatsResult test_cpu_transfer(aclrtStream stream, size_t msg_size, int iterations, int warmup) {
    std::vector<double> latencies;

    void* device_buf;
    void* host_buf;
    aclrtMalloc(&device_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost(&host_buf, msg_size);

    // 初始化数据
    memset(host_buf, 0xAA, msg_size);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        aclrtMemcpy(host_buf, msg_size, device_buf, msg_size, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(device_buf, msg_size, host_buf, msg_size, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    // 正式测试: D2H + H2D往返时间
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        // D2H
        aclrtMemcpy(host_buf, msg_size, device_buf, msg_size, ACL_MEMCPY_DEVICE_TO_HOST);
        // H2D
        aclrtMemcpy(device_buf, msg_size, host_buf, msg_size, ACL_MEMCPY_HOST_TO_DEVICE);

        auto end = std::chrono::high_resolution_clock::now();
        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(latency_us);
    }

    aclrtFree(device_buf);
    aclrtFreeHost(host_buf);

    return compute_stats(latencies);
}

// ========== 通信隐藏测试 ==========<
double test_hidden_comm(aclrtStream stream, uint64_t ffts_config,
                         uint8_t* gva, size_t msg_size, int iterations,
                         ComputeConfig compute_cfg) {
    uint8_t* result_buffer;
    aclrtMalloc(&result_buffer, iterations * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    // 分配MatMul缓冲区
    uint8_t* matmul_A, *matmul_B, *matmul_C;
    size_t A_size = compute_cfg.M * compute_cfg.K * sizeof(float);
    size_t B_size = compute_cfg.K * compute_cfg.N * sizeof(float);
    size_t C_size = compute_cfg.M * compute_cfg.N * sizeof(float);
    aclrtMalloc(&matmul_A, A_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&matmul_B, B_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&matmul_C, C_size, ACL_MEM_MALLOC_HUGE_FIRST);

    launch_hidden_comm(1, stream, ffts_config, gva, msg_size, iterations,
                       matmul_A, matmul_B, matmul_C,
                       compute_cfg.M, compute_cfg.K, compute_cfg.N,
                       result_buffer);
    aclrtSynchronizeStream(stream);

    int64_t* host_result;
    aclrtMallocHost((void**)&host_result, iterations * sizeof(int64_t));
    aclrtMemcpy(host_result, iterations * sizeof(int64_t), result_buffer,
                iterations * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        total_time += cycles_to_us(host_result[i], NPU_FREQ_MHZ);
    }
    double avg_overlap_time = total_time / iterations;

    aclrtFreeHost(host_result);
    aclrtFree(result_buffer);
    aclrtFree(matmul_A);
    aclrtFree(matmul_B);
    aclrtFree(matmul_C);

    return avg_overlap_time;
}

// ========== 主测试流程 ==========<
int run_benchmark(int rank, int world_size) {
    uint64_t mem_size = 256UL * 1024UL * 1024UL;  // 256MB对称内存

    // 创建CSV输出文件
    CSVWriter latency_csv("results/latency_results.csv");
    CSVWriter bandwidth_csv("results/bandwidth_results.csv");
    CSVWriter hidden_csv("results/hidden_results.csv");

    // 测试的引擎类型
    std::vector<EngineType> engines = {
        EngineType::RDMA,
        EngineType::MTE,
        // EngineType::SDMA,  // SDMA可选
    };

    std::cout << "\n==================== Comm Benchmark ====================\n";
    std::cout << "Rank: " << rank << ", WorldSize: " << world_size << "\n";
    print_separator();

    for (EngineType engine : engines) {
        std::cout << "\n---------- Testing Engine: " << engine_name(engine) << " ----------\n";

        // 初始化环境
        init_environment(rank, world_size, mem_size, engine);
        aclrtStream stream = nullptr;
        aclrtCreateStream(&stream);

        uint64_t ffts_config = shmemx_get_ffts_config();

        // 分配对称内存
        uint8_t* gva = (uint8_t*)shmem_malloc(mem_size);

        // ========== 延迟测试 ==========<
        std::cout << "\n[PingPong Latency Test]\n";
        for (size_t msg_size : MSG_SIZES) {
            int iterations = get_iterations(msg_size);
            int warmup = get_warmup_iterations(msg_size);

            print_test_header({rank, world_size, engine, TestType::PINGPONG_LATENCY,
                               msg_size, iterations, warmup, ipport});

            StatsResult result;
            if (engine == EngineType::RDMA) {
                result = test_rdma_pingpong_latency(stream, ffts_config, gva,
                                                     msg_size, iterations, warmup);
            } else if (engine == EngineType::MTE) {
                result = test_mte_pingpong_latency(stream, ffts_config, gva,
                                                    msg_size, iterations, warmup);
            }

            print_result(result);
            latency_csv.write_row(engine_name(engine), "pingpong_latency",
                                   msg_size, iterations, result);
        }

        // ========== 带宽测试 ==========<
        std::cout << "\n[Bandwidth Test]\n";
        for (size_t msg_size : MSG_SIZES) {
            if (msg_size < 64 * 1024) continue;  // 小消息不做带宽测试

            int iterations = (msg_size <= 8 * 1024 * 1024) ? 1000 : 100;

            print_test_header({rank, world_size, engine, TestType::BANDWIDTH,
                               msg_size, iterations, 0, ipport});

            StatsResult result;
            if (engine == EngineType::RDMA) {
                result = test_rdma_bandwidth(stream, ffts_config, gva, msg_size, iterations);
            } else if (engine == EngineType::MTE) {
                result = test_mte_bandwidth(stream, ffts_config, gva, msg_size, iterations);
            }

            std::cout << "Bandwidth: " << result.mean << " GB/s\n";
            bandwidth_csv.write_row(engine_name(engine), "bandwidth",
                                      msg_size, iterations, result);
        }

        // ========== 通信隐藏测试 ==========<
        std::cout << "\n[Hidden Communication Test]\n";
        if (rank == 0) {  // 只在rank0测试
            for (size_t msg_size : MSG_SIZES) {
                if (msg_size < 256 * 1024) continue;  // 中大消息才测隐藏效果

                int iterations = (msg_size <= 8 * 1024 * 1024) ? 100 : 20;
                ComputeConfig compute_cfg = match_compute(msg_size);

                std::cout << "MsgSize: " << msg_size << " bytes"
                          << ", Compute: " << compute_cfg.M << "x" << compute_cfg.K << "x" << compute_cfg.N
                          << "\n";

                // 测试纯通信时间
                StatsResult comm_result;
                if (engine == EngineType::RDMA) {
                    comm_result = test_rdma_pingpong_latency(stream, ffts_config, gva,
                                                              msg_size, iterations, 10);
                } else {
                    comm_result = test_mte_pingpong_latency(stream, ffts_config, gva,
                                                             msg_size, iterations, 10);
                }
                double comm_time = comm_result.mean;

                // 测试通信+计算重叠时间
                double overlap_time = test_hidden_comm(stream, ffts_config, gva,
                                                        msg_size, iterations, compute_cfg);

                // 计算隐藏率
                // hidden_rate = (comm_time - (overlap_time - compute_time)) / comm_time
                // 简化: 隐藏率 = 1 - overlap_time/(comm_time+compute_time)
                double hidden_rate = 100.0 * (1.0 - overlap_time / (comm_time * 2));

                std::cout << "CommTime: " << comm_time << " us"
                          << ", OverlapTime: " << overlap_time << " us"
                          << ", HiddenRate: " << hidden_rate << " %\n";

                hidden_csv.write_hidden_result(engine_name(engine), msg_size,
                                            comm_time, 0, overlap_time, hidden_rate);
            }
        }

        shmem_free(gva);
        aclrtDestroyStream(stream);
        finalize_environment(rank);
    }

    // ========== CPU中转测试 ==========<
    std::cout << "\n---------- Testing Engine: CPU_D2H_H2D ----------\n";
    init_environment(rank, world_size, mem_size, EngineType::RDMA);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    std::cout << "\n[CPU Transfer Latency Test]\n";
    for (size_t msg_size : MSG_SIZES) {
        int iterations = get_iterations(msg_size);
        int warmup = get_warmup_iterations(msg_size);

        StatsResult result = test_cpu_transfer(stream, msg_size, iterations, warmup);
        latency_csv.write_row("CPU_D2H_H2D", "pingpong_latency",
                               msg_size, iterations, result);
    }

    aclrtDestroyStream(stream);
    finalize_environment(rank);

    // ========== HCCL测试 ==========<
    std::cout << "\n---------- Testing Engine: HCCL ----------\n";
    std::cout << "[HCCL] Huawei Collective Communication Library Test\n";

    // 初始化HCCL环境
    int32_t device_id = rank % g_npus + f_npu;
    aclrtSetDevice(device_id);
    aclrtStream hccl_stream = nullptr;
    aclrtCreateStream(&hccl_stream);

    // 初始化HCCL通信域
    HcclComm hccl_comm = nullptr;
    HcclRootInfo root_info;

    if (rank == 0) {
        HcclGetRootInfo(&root_info);
    }

    // 实际场景需要广播root_info，这里简化处理
    // 创建HCCL通信域
    HcclCommInitClusterInfo(rank, world_size, &root_info, &hccl_comm);

    std::cout << "[HCCL] Rank " << HcclGetRankId(hccl_comm)
              << " of " << HcclGetRankSize(hccl_comm) << " initialized\n";

    // ========== HCCL PingPong延迟测试 ==========<
    std::cout << "\n[HCCL PingPong Latency Test]\n";
    uint32_t hccl_peer = (rank == 0) ? 1 : 0;

    for (size_t msg_size : MSG_SIZES) {
        int iterations = get_iterations(msg_size);
        int warmup = get_warmup_iterations(msg_size);

        std::cout << "MsgSize: " << msg_size << " bytes, Iterations: " << iterations << "\n";

        // 分配Device内存
        void* send_buf;
        void* recv_buf;
        aclrtMalloc(&send_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&recv_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);

        std::vector<double> latencies;

        // Warmup
        for (int i = 0; i < warmup; i++) {
            if (rank == 0) {
                HcclSend(send_buf, msg_size / sizeof(uint8_t), HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
                HcclRecv(recv_buf, msg_size / sizeof(uint8_t), HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
            } else {
                HcclRecv(recv_buf, msg_size / sizeof(uint8_t), HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
                HcclSend(send_buf, msg_size / sizeof(uint8_t), HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
            }
            aclrtSynchronizeStream(hccl_stream);
        }

        // 正式测试
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();

            if (rank == 0) {
                HcclSend(send_buf, msg_size / sizeof(uint8_t), HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
                HcclRecv(recv_buf, msg_size / sizeof(uint8_t), HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
            } else {
                HcclRecv(recv_buf, msg_size / sizeof(uint8_t), HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
                HcclSend(send_buf, msg_size / sizeof(uint8_t), HcclDataType::HCCL_DATA_TYPE_UINT8,
                         hccl_peer, hccl_comm, hccl_stream);
            }
            aclrtSynchronizeStream(hccl_stream);

            auto end = std::chrono::high_resolution_clock::now();
            double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
            latencies.push_back(latency_us);
        }

        StatsResult result = compute_stats(latencies);
        print_result(result);
        latency_csv.write_row("HCCL", "pingpong_latency", msg_size, iterations, result);

        aclrtFree(send_buf);
        aclrtFree(recv_buf);
    }

    // ========== HCCL带宽测试 (AllReduce) ==========<
    std::cout << "\n[HCCL AllReduce Bandwidth Test]\n";
    for (size_t msg_size : MSG_SIZES) {
        if (msg_size < 64 * 1024) continue;  // 小消息不做带宽测试

        int iterations = (msg_size <= 8 * 1024 * 1024) ? 1000 : 100;

        std::cout << "MsgSize: " << msg_size << " bytes, Iterations: " << iterations << "\n";

        void* buf;
        aclrtMalloc(&buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);

        // Warmup
        for (int i = 0; i < 10; i++) {
            HcclAllReduce(buf, buf, msg_size / sizeof(float), HcclDataType::HCCL_DATA_TYPE_FLOAT,
                         HcclReduceOp::HCCL_REDUCE_SUM, hccl_comm, hccl_stream);
        }
        aclrtSynchronizeStream(hccl_stream);

        // 正式测试
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            HcclAllReduce(buf, buf, msg_size / sizeof(float), HcclDataType::HCCL_DATA_TYPE_FLOAT,
                         HcclReduceOp::HCCL_REDUCE_SUM, hccl_comm, hccl_stream);
        }
        aclrtSynchronizeStream(hccl_stream);
        auto end = std::chrono::high_resolution_clock::now();

        double total_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        // AllReduce带宽 = msg_size * iterations * 2 / time
        double bw_gb_s = compute_bandwidth(msg_size * iterations * 2, total_time_us);

        std::cout << "Bandwidth: " << bw_gb_s << " GB/s\n";
        bandwidth_csv.write_row("HCCL", "allreduce_bandwidth", msg_size, iterations,
                                {bw_gb_s, 0, bw_gb_s, bw_gb_s, bw_gb_s});

        aclrtFree(buf);
    }

    // ========== HCCL AllGather测试 ==========<
    std::cout << "\n[HCCL AllGather Bandwidth Test]\n";
    for (size_t msg_size : MSG_SIZES) {
        if (msg_size < 64 * 1024) continue;

        int iterations = (msg_size <= 8 * 1024 * 1024) ? 100 : 10;

        void* send_buf;
        void* recv_buf;
        size_t recv_size = msg_size * world_size;
        aclrtMalloc(&send_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&recv_buf, recv_size, ACL_MEM_MALLOC_HUGE_FIRST);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            HcclAllGather(send_buf, recv_buf, msg_size / sizeof(float),
                         HcclDataType::HCCL_DATA_TYPE_FLOAT, hccl_comm, hccl_stream);
        }
        aclrtSynchronizeStream(hccl_stream);
        auto end = std::chrono::high_resolution_clock::now();

        double total_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        double bw_gb_s = compute_bandwidth(recv_size * iterations, total_time_us);

        std::cout << "MsgSize: " << msg_size << " bytes, AllGather BW: " << bw_gb_s << " GB/s\n";
        bandwidth_csv.write_row("HCCL", "allgather_bandwidth", msg_size, iterations,
                                {bw_gb_s, 0, bw_gb_s, bw_gb_s, bw_gb_s});

        aclrtFree(send_buf);
        aclrtFree(recv_buf);
    }

    // ========== HCCL ReduceScatter测试 ==========<
    std::cout << "\n[HCCL ReduceScatter Bandwidth Test]\n";
    for (size_t msg_size : MSG_SIZES) {
        if (msg_size < 64 * 1024) continue;

        int iterations = (msg_size <= 8 * 1024 * 1024) ? 100 : 10;

        void* send_buf;
        void* recv_buf;
        size_t send_size = msg_size * world_size;
        aclrtMalloc(&send_buf, send_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&recv_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
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

    // 清理HCCL
    HcclCommDestroy(hccl_comm);
    aclrtDestroyStream(hccl_stream);
    aclrtResetDevice(device_id);

    std::cout << "\n==================== Benchmark Complete ====================\n";
    std::cout << "Results saved to results/ directory\n";

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cout << "Usage: ./comm_benchmark <n_ranks> <rank_id> <ipport> <g_npus> <f_rank> <f_npu>\n";
        std::cout << "Example: ./comm_benchmark 2 0 tcp://127.0.0.1:8765 8 0 0\n";
        return -1;
    }

    int argIdx = 1;
    int n_ranks = atoi(argv[argIdx++]);
    int rank_id = atoi(argv[argIdx++]);
    ipport = argv[argIdx++];
    g_npus = atoi(argv[argIdx++]);
    f_rank = atoi(argv[argIdx++]);
    f_npu = atoi(argv[argIdx++]);

    if (!check_env()) {
        return -1;
    }

    int status = run_benchmark(rank_id, n_ranks);

    std::cout << "[SUCCESS] Benchmark completed for rank " << rank_id << "\n";
    return status;
}