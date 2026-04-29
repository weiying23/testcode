/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file be in compliance with the License.
 * This software is provided on an "AS IS" basis, WITHOUT WARRANTIES OF ANY KIND, either express or implied,
 * including but not limited to non-infringement, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include "benchmark_config.h"

namespace benchmark {

// ========== 统计计算函数 ==========
inline StatsResult compute_stats(std::vector<double>& data) {
    if (data.empty()) {
        return {0, 0, 0, 0, 0};
    }

    // 排序用于计算中位数
    std::sort(data.begin(), data.end());

    // 平均值
    double sum = 0;
    for (double v : data) {
        sum += v;
    }
    double mean = sum / data.size();

    // 标准差
    double variance = 0;
    for (double v : data) {
        variance += (v - mean) * (v - mean);
    }
    double std_dev = sqrt(variance / data.size());

    // 最小/最大值
    double min_val = data.front();
    double max_val = data.back();

    // 中位数
    double median;
    size_t n = data.size();
    if (n % 2 == 0) {
        median = (data[n/2 - 1] + data[n/2]) / 2;
    } else {
        median = data[n/2];
    }

    return {mean, std_dev, min_val, max_val, median};
}

// ========== CSV结果输出 ==========
class CSVWriter {
public:
    CSVWriter(const std::string& filename) : file_(filename) {
        if (file_.is_open()) {
            // 写入表头
            file_ << "engine,test,msg_size_bytes,iterations,mean_us,std_us,min_us,max_us,median_us\n";
        }
    }

    ~CSVWriter() {
        if (file_.is_open()) {
            file_.close();
        }
    }

    void write_row(const std::string& engine, const std::string& test,
                   size_t msg_size, int iterations, const StatsResult& stats) {
        if (file_.is_open()) {
            file_ << engine << ","
                  << test << ","
                  << msg_size << ","
                  << iterations << ","
                  << stats.mean << ","
                  << stats.std << ","
                  << stats.min << ","
                  << stats.max << ","
                  << stats.median << "\n";
        }
    }

    void write_hidden_result(const std::string& engine, size_t msg_size,
                              double comm_time, double compute_time, double overlap_time,
                              double hidden_rate) {
        if (file_.is_open()) {
            file_ << engine << ","
                  << "hidden_comm" << ","
                  << msg_size << ","
                  << comm_time << ","
                  << compute_time << ","
                  << overlap_time << ","
                  << hidden_rate << "\n";
        }
    }

    bool is_open() const { return file_.is_open(); }

private:
    std::ofstream file_;
};

// ========== 带宽计算 ==========
inline double compute_bandwidth(size_t msg_size_bytes, double latency_us) {
    // bandwidth = msg_size / latency
    // 单位: GB/s
    if (latency_us <= 0) return 0;
    double latency_s = latency_us / 1e6;  // 微秒转秒
    double size_gb = msg_size_bytes / 1e9;  // bytes转GB
    return size_gb / latency_s;
}

// ========== 延迟转换单位 ==========
inline double us_to_ms(double us) {
    return us / 1000.0;
}

inline double cycles_to_us(int64_t cycles, double freq_mhz = 1000.0) {
    // NPU cycle频率约1GHz (1000MHz)
    return cycles / freq_mhz;
}

// ========== 打印辅助 ==========
inline void print_separator() {
    std::cout << "============================================================\n";
}

inline void print_test_header(const BenchmarkConfig& config) {
    print_separator();
    std::cout << "Test: " << test_name(config.test)
              << " | Engine: " << engine_name(config.engine)
              << " | MsgSize: " << config.msg_size << " bytes"
              << " | Iterations: " << config.iterations << "\n";
    print_separator();
}

inline void print_result(const StatsResult& stats, const std::string& unit = "us") {
    std::cout << "Result (" << unit << "): " << stats.to_string() << "\n";
}

// ========== 环境检查 ==========
inline bool check_env() {
    // 检查ASCEND_HOME_PATH
    const char* ascend_home = getenv("ASCEND_HOME_PATH");
    if (!ascend_home) {
        std::cerr << "Error: ASCEND_HOME_PATH not set\n";
        return false;
    }
    std::cout << "ASCEND_HOME_PATH: " << ascend_home << "\n";
    return true;
}

} // namespace benchmark

#endif // BENCHMARK_UTILS_H