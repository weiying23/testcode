/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGINE_BENCHMARK_CONFIG_H
#define ENGINE_BENCHMARK_CONFIG_H

#include <vector>
#include <cstdint>
#include <string>
#include <cmath>
#include "acl/acl.h"  // For aclrtGetSocName

namespace engine_bench {

// ========== 通信引擎类型 ==========
enum class EngineType {
    MTE_INTRA_CARD,     // MTE 卡内通信（同一NPU内）
    MTE_INTER_CARD,     // MTE 卡间通信（同节点不同NPU）
    SDMA_INTER_CARD     // SDMA 卡间通信（同节点不同NPU）
};

inline std::string engine_name(EngineType type) {
    switch (type) {
        case EngineType::MTE_INTRA_CARD: return "MTE_INTRA";
        case EngineType::MTE_INTER_CARD: return "MTE_INTER";
        case EngineType::SDMA_INTER_CARD: return "SDMA_INTER";
        default: return "UNKNOWN";
    }
}

inline std::string engine_desc(EngineType type) {
    switch (type) {
        case EngineType::MTE_INTRA_CARD: return "MTE Card-Internal (same NPU)";
        case EngineType::MTE_INTER_CARD: return "MTE Card-Inter (same node, different NPU)";
        case EngineType::SDMA_INTER_CARD: return "SDMA Card-Inter (same node, different NPU)";
        default: return "UNKNOWN";
    }
}

// ========== 测试模式 ==========
enum class TestMode {
    PUT,        // 单向发送
    GET,        // 单向接收
    BI_PUT,     // 双向发送
    BI_GET      // 双向接收
};

inline std::string mode_name(TestMode mode) {
    switch (mode) {
        case TestMode::PUT: return "put";
        case TestMode::GET: return "get";
        case TestMode::BI_PUT: return "bi_put";
        case TestMode::BI_GET: return "bi_get";
        default: return "UNKNOWN";
    }
}

// ========== 数据类型 ==========
enum class DataType {
    FLOAT,
    INT32,
    INT64
};

inline std::string type_name(DataType type) {
    switch (type) {
        case DataType::FLOAT: return "float";
        case DataType::INT32: return "int32";
        case DataType::INT64: return "int64";
        default: return "UNKNOWN";
    }
}

inline size_t type_size(DataType type) {
    switch (type) {
        case DataType::FLOAT: return sizeof(float);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::INT64: return sizeof(int64_t);
        default: return sizeof(float);
    }
}

// ========== 消息大小配置 ==========
inline std::vector<size_t> get_msg_sizes() {
    return {
        64,                // 64B
        256,               // 256B
        512,               // 512B
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
    };
}

// ========== 迭代次数配置 ==========
inline int get_iterations(size_t msg_size) {
    if (msg_size <= 256) return 10000;
    else if (msg_size <= 64 * 1024) return 1000;
    else if (msg_size <= 4 * 1024 * 1024) return 500;
    else return 100;
}

inline int get_warmup_iterations(size_t msg_size) {
    int total = get_iterations(msg_size);
    if (total >= 10000) return 100;
    if (total >= 1000) return 50;
    return 10;
}

// ========== 测试配置结构 ==========
struct TestConfig {
    int pe_id;
    int n_pes;
    int g_npus;
    int f_npu;
    int device_id;          // 直接指定NPU ID，-1表示自动计算
    EngineType engine;
    TestMode mode;
    DataType dtype;
    size_t msg_size;
    int block_size;
    int ub_size_kb;
    int iterations;
    int warmup;
    std::string ipport;
};

// ========== 统计结果结构 ==========
struct PerfResult {
    size_t msg_size;
    double bandwidth_gbs;   // 带宽 GB/s
    double latency_us;      // 延迟 us
    double time_us;         // 总时间 us
    int iterations;

    std::string to_csv_row() const {
        return std::to_string(msg_size) + "," +
               std::to_string(bandwidth_gbs) + "," +
               std::to_string(latency_us) + "," +
               std::to_string(time_us) + "," +
               std::to_string(iterations);
    }
};

// ========== 硬件周期到微秒转换 ==========
// Note: This function requires ACL to be initialized before calling
inline int64_t get_cycle_to_us_ratio() {
    const char *soc_name = aclrtGetSocName();
    if (soc_name != nullptr && std::string(soc_name).find("Ascend950") != std::string::npos) {
        return 1000;  // Ascend950
    }
    return 50;  // Ascend910
}

// Default cycle to us ratio (for Ascend910)
const int64_t DEFAULT_CYCLE_TO_US = 50;

} // namespace engine_bench

#endif // ENGINE_BENCHMARK_CONFIG_H