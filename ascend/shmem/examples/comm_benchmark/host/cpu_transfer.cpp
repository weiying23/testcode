/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * CPU中转通信实现 - D2H + H2D方式
 */

#include <iostream>
#include <vector>
#include <chrono>
#include "acl/acl.h"
#include "benchmark_config.h"
#include "benchmark_utils.h"

namespace benchmark {

class CPUTransferBenchmark {
public:
    CPUTransferBenchmark(int rank, int world_size)
        : rank_(rank), world_size_(world_size) {}

    // PingPong延迟测试 (CPU中转)
    StatsResult pingpong_latency(size_t msg_size, int iterations, int warmup) {
        std::vector<double> latencies;

        // 分配Host和Device内存
        void* device_buf;
        void* host_buf;
        aclrtMalloc(&device_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMallocHost(&host_buf, msg_size);

        int peer = (rank_ == 0) ? 1 : 0;

        // Warmup
        for (int i = 0; i < warmup; i++) {
            if (rank_ == 0) {
                // D2H: NPU → Host
                aclrtMemcpy(host_buf, msg_size, device_buf, msg_size, ACL_MEMCPY_DEVICE_TO_HOST);
                // 这里需要通过某种方式发送host_buf到peer (如MPI或共享内存)
                // 简化实现: 直接等待peer完成

                // H2D: Host → NPU (接收peer数据)
                aclrtMemcpy(device_buf, msg_size, host_buf, msg_size, ACL_MEMCPY_HOST_TO_DEVICE);
            }
        }

        // 正式测试
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();

            if (rank_ == 0) {
                // 发送端: D2H → Host → ... → H2D (peer)
                aclrtMemcpy(host_buf, msg_size, device_buf, msg_size, ACL_MEMCPY_DEVICE_TO_HOST);
                // 等待peer响应 (实际需要IPC通信)
                aclrtMemcpy(device_buf, msg_size, host_buf, msg_size, ACL_MEMCPY_HOST_TO_DEVICE);
            }

            auto end = std::chrono::high_resolution_clock::now();
            double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
            latencies.push_back(latency_us);
        }

        // 清理
        aclrtFree(device_buf);
        aclrtFreeHost(host_buf);

        return compute_stats(latencies);
    }

    // 带宽测试 (CPU中转)
    StatsResult bandwidth(size_t msg_size, int iterations, int warmup) {
        std::vector<double> bw_values;

        void* device_buf;
        void* host_buf;
        aclrtMalloc(&device_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMallocHost(&host_buf, msg_size);

        // Warmup
        for (int i = 0; i < warmup; i++) {
            aclrtMemcpy(host_buf, msg_size, device_buf, msg_size, ACL_MEMCPY_DEVICE_TO_HOST);
        }

        // 正式测试: 单向D2H带宽
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();

            aclrtMemcpy(host_buf, msg_size, device_buf, msg_size, ACL_MEMCPY_DEVICE_TO_HOST);

            auto end = std::chrono::high_resolution_clock::now();
            double time_us = std::chrono::duration<double, std::micro>(end - start).count();
            double bw_gb_s = compute_bandwidth(msg_size, time_us);
            bw_values.push_back(bw_gb_s);
        }

        aclrtFree(device_buf);
        aclrtFreeHost(host_buf);

        return compute_stats(bw_values);
    }

private:
    int rank_;
    int world_size_;
};

} // namespace benchmark