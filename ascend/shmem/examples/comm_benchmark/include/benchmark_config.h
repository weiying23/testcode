/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except be in compliance with the License.
 * This software is provided on an "AS IS" basis, WITHOUT WARRANTIES OF ANY KIND, either express or implied,
 * including but not limited to non-infringement, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BENCHMARK_CONFIG_H
#define BENCHMARK_CONFIG_H

#include <vector>
#include <cstdint>
#include <string>

namespace benchmark {

// ========== 消息大小配置 ==========
// 小消息: 1KB - 256KB (高迭代次数)
// 中消息: 1MB - 8MB (中迭代次数)
// 大消息: 16MB - 128MB (低迭代次数)
constexpr std::vector<size_t> MSG_SIZES = {
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

// ========== 迭代次数配置 ==========
// 根据消息大小自动调整迭代次数
inline int get_iterations(size_t msg_size) {
    if (msg_size <= 256 * 1024) {          // <= 256KB: 小消息
        return 10000;
    } else if (msg_size <= 8 * 1024 * 1024) {  // <= 8MB: 中消息
        return 1000;
    } else {                               // > 8MB: 大消息
        return 100;
    }
}

// warmup比例: 丢弃前1-5%的迭代
inline int get_warmup_iterations(size_t msg_size) {
    int total = get_iterations(msg_size);
    if (total >= 10000) return 100;        // 小消息: 丢弃100次
    if (total >= 1000) return 10;          // 中消息: 丢弃10次
    return 5;                              // 大消息: 丢弃5次
}

// ========== 通信引擎类型 ==========
enum class EngineType {
    RDMA,       // RoCE RDMA通信
    MTE,        // MTE引擎 (同节点最优)
    SDMA,       // 系统DMA
    CPU_D2H_H2D // CPU中转 (Host拷贝)
};

// 获取引擎类型名称
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
    PINGPONG_LATENCY,     // PingPong延迟测试
    ONE_WAY_LATENCY,      // 单向延迟测试
    BANDWIDTH,            // 带宽测试
    HIDDEN_COMM           // 通信隐藏测试
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

// ========== 计算负载配置 (用于隐藏测试) ==========
struct ComputeConfig {
    int M;      // 矩阵行数
    int K;      // 矩阵中间维度
    int N;      // 矩阵列数

    // 计算FLOPS数量
    long long flops() const {
        return 2LL * M * K * N;
    }

    // 计算数据量 (bytes)
    long long data_size() const {
        return (M * K + K * N + M * N) * sizeof(float);
    }
};

// 根据消息大小匹配计算负载
inline ComputeConfig match_compute(size_t msg_size) {
    if (msg_size <= 64 * 1024) {
        return {512, 512, 512};          // 轻量计算
    } else if (msg_size <= 1 * 1024 * 1024) {
        return {1024, 1024, 1024};       // 中等计算
    } else if (msg_size <= 8 * 1024 * 1024) {
        return {2048, 2048, 2048};       // 重量计算
    } else {
        return {4096, 4096, 4096};       // 超重计算
    }
}

// ========== 统计结果结构 ==========
struct StatsResult {
    double mean;      // 平均值
    double std;       // 标准差
    double min;       // 最小值
    double max;       // 最大值
    double median;    // 中位数

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
    int rank;                 // 当前rank编号
    int world_size;           // 总rank数
    EngineType engine;        // 通信引擎
    TestType test;            // 测试类型
    size_t msg_size;          // 消息大小
    int iterations;           // 迭代次数
    int warmup;               // warmup次数
    std::string ipport;       // IP端口

    // 打印配置
    void print() const;
};

} // namespace benchmark

#endif // BENCHMARK_CONFIG_H