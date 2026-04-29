/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Comm Benchmark配置文件
 */

#ifndef BENCHMARK_CONFIG_H
#define BENCHMARK_CONFIG_H

#include <vector>
#include <cstdint>
#include <string>

namespace benchmark {

// ========== MPI开关配置 ==========
// 定义 ENABLE_MPI 宏来启用MPI模式
// 默认关闭MPI，使用socket模式进行进程间通信
// 取消下面注释以启用MPI:
// #define ENABLE_MPI

#ifdef ENABLE_MPI
#define BENCHMARK_INIT_FLAG ACLSHMEMX_INIT_WITH_MPI
#define BENCHMARK_MODE_NAME "MPI Mode"
#else
#define BENCHMARK_INIT_FLAG ACLSHMEMX_INIT_WITH_DEFAULT
#define BENCHMARK_MODE_NAME "Socket Mode (No MPI)"
#endif

// ========== 消息大小配置 ==========
inline std::vector<size_t> get_msg_sizes() {
    return {
        1 * 1024,          // 1KB
        4 * 1024,          // 4KB
        16 * 1024,         // 16KB
        64 * 1024,         // 64KB
        256 * 1024,        // 256KB
        1 * 1024 * 1024,   // 1MB
        2 * 1024 * 1024,   // 2MB
        4 * 1024 * 1024,   // 4MB
        8 * 1024 * 1024,   // 8MB
        16 * 1024 * 1024,  // 16MB
        32 * 1024 * 1024,  // 32MB
        64 * 1024 * 1024,  // 64MB
        128 * 1024 * 1024  // 128MB
    };
}

const std::vector<size_t> MSG_SIZES = get_msg_sizes();

// ========== 迭代次数配置 ==========
inline int get_iterations(size_t msg_size) {
    if (msg_size <= 256 * 1024) {
        return 10000;
    } else if (msg_size <= 8 * 1024 * 1024) {
        return 1000;
    } else {
        return 100;
    }
}

inline int get_warmup_iterations(size_t msg_size) {
    int total = get_iterations(msg_size);
    if (total >= 10000) return 100;
    if (total >= 1000) return 10;
    return 5;
}

// ========== 通信引擎类型 ==========
enum class EngineType {
    RDMA,       // RoCE RDMA通信
    MTE,        // MTE引擎 (同节点最优)
    SDMA,       // 系统DMA
    CPU_D2H_H2D // CPU中转 (Host拷贝)
};

inline std::string engine_name(EngineType type) {
    switch (type) {
        case EngineType::RDMA: return "RDMA";
        case EngineType::MTE: return "MTE";
        case EngineType::SDMA: return "SDMA";
        case EngineType::CPU_D2H_H2D: return "CPU_D2H_H2D";
        default: return "UNKNOWN";
    }
}

// ========== 测试类型 ==========
enum class TestType {
    PINGPONG_LATENCY,
    ONE_WAY_LATENCY,
    BANDWIDTH,
    HIDDEN_COMM
};

inline std::string test_name(TestType type) {
    switch (type) {
        case TestType::PINGPONG_LATENCY: return "pingpong_latency";
        case TestType::ONE_WAY_LATENCY: return "one_way_latency";
        case TestType::BANDWIDTH: return "bandwidth";
        case TestType::HIDDEN_COMM: return "hidden_comm";
        default: return "UNKNOWN";
    }
}

// ========== 计算负载配置 ==========
struct ComputeConfig {
    int M, K, N;
    long long flops() const { return 2LL * M * K * N; }
    long long data_size() const { return (M * K + K * N + M * N) * sizeof(float); }
};

inline ComputeConfig match_compute(size_t msg_size) {
    if (msg_size <= 64 * 1024) return {512, 512, 512};
    else if (msg_size <= 1 * 1024 * 1024) return {1024, 1024, 1024};
    else if (msg_size <= 8 * 1024 * 1024) return {2048, 2048, 2048};
    else return {4096, 4096, 4096};
}

// ========== 统计结果结构 ==========
struct StatsResult {
    double mean, std, min, max, median;
    std::string to_string() const {
        return "mean=" + std::to_string(mean) +
               ", std=" + std::to_string(std) +
               ", min=" + std::to_string(min) +
               ", max=" + std::to_string(max) +
               ", median=" + std::to_string(median);
    }
};

// ========== Benchmark配置汇总 ==========
struct BenchmarkConfig {
    int rank;
    int world_size;
    EngineType engine;
    TestType test;
    size_t msg_size;
    int iterations;
    int warmup;
    std::string ipport;
    void print() const;
};

} // namespace benchmark

#endif // BENCHMARK_CONFIG_H