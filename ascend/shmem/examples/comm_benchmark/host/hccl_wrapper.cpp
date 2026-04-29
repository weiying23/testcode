/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * HCCL通信测试实现 - 华为集合通信库对比
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#include "benchmark_config.h"
#include "benchmark_utils.h"

namespace benchmark {

class HCCLBenchmark {
public:
    HCCLBenchmark(int rank, int world_size, const char* ipport)
        : rank_(rank), world_size_(world_size), comm_(nullptr) {

        // 初始化HCCL
        init_hccl(rank, world_size, ipport);
    }

    ~HCCLBenchmark() {
        if (comm_) {
            HcclCommDestroy(comm_);
        }
    }

    // 初始化HCCL通信域
    int init_hccl(int rank, int world_size, const char* ipport) {
        // 创建HCCL通信域
        HcclRootInfo root_info;

        if (rank == 0) {
            // Rank 0 初始化root信息
            HcclGetRootInfo(&root_info);

            // 广播root信息给其他rank (实际需要通过socket/MPI等)
            // 这里简化处理，假设通过某种方式广播
        }

        // 等待root_info广播完成 (实际需要同步)
        // 简化：直接使用本地初始化

        // 创建通信域
        HcclCommInitClusterInfo(rank, world_size, &root_info, &comm_);

        // 验证rank和size
        uint32_t hccl_rank = HcclGetRankId(comm_);
        uint32_t hccl_size = HcclGetRankSize(comm_);

        std::cout << "[HCCL] Rank " << hccl_rank << " of " << hccl_size << " initialized\n";

        return 0;
    }

    // PingPong延迟测试 (使用Send/Recv)
    StatsResult pingpong_latency(aclrtStream stream, size_t msg_size,
                                  int iterations, int warmup) {
        std::vector<double> latencies;

        // 分配Device内存
        void* send_buf;
        void* recv_buf;
        aclrtMalloc(&send_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&recv_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);

        uint32_t peer = (rank_ == 0) ? 1 : 0;

        // Warmup
        for (int i = 0; i < warmup; i++) {
            if (rank_ == 0) {
                // Send → Recv
                HcclSend(send_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
                HcclRecv(recv_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
            } else {
                // Recv → Send
                HcclRecv(recv_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
                HcclSend(send_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
            }
            aclrtSynchronizeStream(stream);
        }

        // 正式测试
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();

            if (rank_ == 0) {
                HcclSend(send_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
                HcclRecv(recv_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
            } else {
                HcclRecv(recv_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
                HcclSend(send_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
            }
            aclrtSynchronizeStream(stream);

            auto end = std::chrono::high_resolution_clock::now();
            double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
            latencies.push_back(latency_us);
        }

        aclrtFree(send_buf);
        aclrtFree(recv_buf);

        return compute_stats(latencies);
    }

    // 单向延迟测试 (只Send或只Recv)
    StatsResult one_way_latency(aclrtStream stream, size_t msg_size,
                                 int iterations, int warmup) {
        std::vector<double> latencies;

        void* send_buf;
        void* recv_buf;
        aclrtMalloc(&send_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&recv_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);

        uint32_t peer = (rank_ == 0) ? 1 : 0;

        // Warmup
        for (int i = 0; i < warmup; i++) {
            if (rank_ == 0) {
                HcclSend(send_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
            } else {
                HcclRecv(recv_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
            }
            aclrtSynchronizeStream(stream);
        }

        // 正式测试
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();

            if (rank_ == 0) {
                HcclSend(send_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
            } else {
                HcclRecv(recv_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
            }
            aclrtSynchronizeStream(stream);

            auto end = std::chrono::high_resolution_clock::now();
            double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
            latencies.push_back(latency_us);
        }

        aclrtFree(send_buf);
        aclrtFree(recv_buf);

        return compute_stats(latencies);
    }

    // 带宽测试 (使用AllReduce连续传输)
    StatsResult bandwidth(aclrtStream stream, size_t msg_size, int iterations, int warmup) {
        std::vector<double> bw_values;

        void* buf;
        aclrtMalloc(&buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);

        // Warmup
        for (int i = 0; i < warmup; i++) {
            HcclAllReduce(buf, buf, msg_size / sizeof(float),
                         HcclDataType::HCCL_DATA_TYPE_FLOAT,
                         HcclReduceOp::HCCL_REDUCE_SUM, comm_, stream);
            aclrtSynchronizeStream(stream);
        }

        // 正式测试 - 连续多次AllReduce
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            HcclAllReduce(buf, buf, msg_size / sizeof(float),
                         HcclDataType::HCCL_DATA_TYPE_FLOAT,
                         HcclReduceOp::HCCL_REDUCE_SUM, comm_, stream);
        }
        aclrtSynchronizeStream(stream);
        auto end = std::chrono::high_resolution_clock::now();

        double total_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        // 带宽 = msg_size * iterations * 2 (AllReduce双向) / time
        double bw_gb_s = compute_bandwidth(msg_size * iterations * 2, total_time_us);

        aclrtFree(buf);

        return {bw_gb_s, 0, bw_gb_s, bw_gb_s, bw_gb_s};
    }

    // AllGather带宽测试
    StatsResult allgather_bandwidth(aclrtStream stream, size_t msg_size, int iterations) {
        std::vector<double> bw_values;

        void* send_buf;
        void* recv_buf;
        size_t recv_size = msg_size * world_size_;
        aclrtMalloc(&send_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&recv_buf, recv_size, ACL_MEM_MALLOC_HUGE_FIRST);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            HcclAllGather(send_buf, recv_buf, msg_size / sizeof(float),
                         HcclDataType::HCCL_DATA_TYPE_FLOAT, comm_, stream);
        }
        aclrtSynchronizeStream(stream);
        auto end = std::chrono::high_resolution_clock::now();

        double total_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        // AllGather带宽 = msg_size * world_size * iterations / time
        double bw_gb_s = compute_bandwidth(msg_size * world_size_ * iterations, total_time_us);

        aclrtFree(send_buf);
        aclrtFree(recv_buf);

        return {bw_gb_s, 0, bw_gb_s, bw_gb_s, bw_gb_s};
    }

    // ReduceScatter带宽测试
    StatsResult reducescatter_bandwidth(aclrtStream stream, size_t msg_size, int iterations) {
        std::vector<double> bw_values;

        void* send_buf;
        void* recv_buf;
        size_t send_size = msg_size * world_size_;
        aclrtMalloc(&send_buf, send_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&recv_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            HcclReduceScatter(send_buf, recv_buf, msg_size / sizeof(float),
                             HcclDataType::HCCL_DATA_TYPE_FLOAT,
                             HcclReduceOp::HCCL_REDUCE_SUM, comm_, stream);
        }
        aclrtSynchronizeStream(stream);
        auto end = std::chrono::high_resolution_clock::now();

        double total_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        double bw_gb_s = compute_bandwidth(send_size * iterations, total_time_us);

        aclrtFree(send_buf);
        aclrtFree(recv_buf);

        return {bw_gb_s, 0, bw_gb_s, bw_gb_s, bw_gb_s};
    }

    // 非阻塞通信测试 (HCCL异步版本)
    StatsResult async_pingpong_latency(aclrtStream stream, size_t msg_size,
                                        int iterations, int warmup) {
        std::vector<double> latencies;

        void* send_buf;
        void* recv_buf;
        aclrtMalloc(&send_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&recv_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);

        uint32_t peer = (rank_ == 0) ? 1 : 0;

        // Warmup
        for (int i = 0; i < warmup; i++) {
            HcclSend(send_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
            HcclRecv(recv_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
            aclrtSynchronizeStream(stream);
        }

        // 正式测试 - 非阻塞版本
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();

            // 发起通信后立即返回，不等待
            if (rank_ == 0) {
                HcclSend(send_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
                // 可以在这里执行计算...
                HcclRecv(recv_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
            } else {
                HcclRecv(recv_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
                // 可以在这里执行计算...
                HcclSend(send_buf, msg_size, HcclDataType::HCCL_DATA_TYPE_UINT8, peer, comm_, stream);
            }

            // 等待完成
            aclrtSynchronizeStream(stream);

            auto end = std::chrono::high_resolution_clock::now();
            double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
            latencies.push_back(latency_us);
        }

        aclrtFree(send_buf);
        aclrtFree(recv_buf);

        return compute_stats(latencies);
    }

    // 获取HCCL通信域
    HcclComm get_comm() { return comm_; }

    // 获取rank
    uint32_t get_rank() { return HcclGetRankId(comm_); }

    // 获取world_size
    uint32_t get_size() { return HcclGetRankSize(comm_); }

private:
    int rank_;
    int world_size_;
    HcclComm comm_;
};

} // namespace benchmark