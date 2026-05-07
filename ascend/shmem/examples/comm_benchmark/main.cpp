/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Comm Benchmark主程序 - NPU通信性能对比测试
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include <cinttypes>
#include <unistd.h>  // for usleep

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "utils.h"
#include "benchmark_config.h"
#include "benchmark_utils.h"

// HCCL头文件（仅在启用HCCL时包含）
#ifdef ENABLE_HCCL
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#endif

// ========== 调试宏定义 ==========
#define DEBUG_LOG(rank, fmt, ...) \
    do { \
        fprintf(stderr, "[DEBUG][Rank %d] %s:%d: " fmt "\n", \
                rank, __func__, __LINE__, ##__VA_ARGS__); \
    } while(0)

#define CHECK_ACL_STATUS(rank, call, desc) \
    do { \
        aclError ret = call; \
        if (ret != ACL_SUCCESS) { \
            fprintf(stderr, "[ERROR][Rank %d] %s:%d: %s failed, ret=%d\n", \
                    rank, __func__, __LINE__, desc, ret); \
            return -1; \
        } \
        DEBUG_LOG(rank, "%s success, ret=%d", desc, ret); \
    } while(0)

#define CHECK_SHMEM_STATUS(rank, call, desc) \
    do { \
        int ret = call; \
        if (ret != ACLSHMEM_SUCCESS) { \
            fprintf(stderr, "[ERROR][Rank %d] %s:%d: %s failed, ret=%d\n", \
                    rank, __func__, __LINE__, desc, ret); \
            return -1; \
        } \
        DEBUG_LOG(rank, "%s success", desc); \
    } while(0)

#define CHECK_PTR(rank, ptr, desc) \
    do { \
        if (ptr == nullptr) { \
            fprintf(stderr, "[ERROR][Rank %d] %s:%d: %s is nullptr\n", \
                    rank, __func__, __LINE__, desc); \
            return StatsResult{0,0,0,0,0}; \
        } \
        DEBUG_LOG(rank, "%s allocated at %p", desc, ptr); \
    } while(0)

// 时间戳打印宏
#define TIMESTAMP(rank, label) \
    do { \
        auto ts = std::chrono::high_resolution_clock::now(); \
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(ts.time_since_epoch()).count(); \
        DEBUG_LOG(rank, "[%s] timestamp=%ld ms", label, (long)ms); \
    } while(0)

// 结果验证宏
#define VERIFY_RESULT(rank, expected, actual, desc) \
    do { \
        if (expected != actual) { \
            fprintf(stderr, "[ERROR][Rank %d] Verification failed: %s, expected=%ld, actual=%ld\n", \
                    rank, desc, (long)expected, (long)actual); \
        } else { \
            DEBUG_LOG(rank, "Verification passed: %s, value=%ld", desc, (long)expected); \
        } \
    } while(0)

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

using namespace benchmark;

// 全局配置变量
int g_npus = 8;                 // 节点内NPU总数，用于计算物理设备ID
const char* ipport;             // rendezvous地址（TCP socket地址）
int f_rank = 0;                 // rank编号偏移量（用于多节点场景）
int f_npu = 0;                  // NPU编号偏移量（物理设备ID的起点）

aclshmemx_uniqueid_t default_flag_uid;  // uniqueid结构体（DEFAULT模式下使用）

// NPU频率 (MHz) - 用于将cycles转换为时间
const double NPU_FREQ_MHZ = 1000.0;

/**
 * 初始化ACL环境（只调用一次，在引擎循环外）
 * 参考rdma_perftest/mte_perftest: aclInit → aclrtSetDevice → aclrtCreateStream
 */
int init_acl_environment(int rank, aclrtStream* stream) {
    int32_t device_id = f_npu;
    DEBUG_LOG(rank, "=== init_acl_environment START ===");
    DEBUG_LOG(rank, "device_id=%d (f_npu=%d)", device_id, f_npu);

    CHECK_ACL_STATUS(rank, aclInit(nullptr), "aclInit");
    CHECK_ACL_STATUS(rank, aclrtSetDevice(device_id), "aclrtSetDevice");
    // 参考实现: stream在aclshmemx_init_attr之前创建
    CHECK_ACL_STATUS(rank, aclrtCreateStream(stream), "aclrtCreateStream");

    DEBUG_LOG(rank, "stream created at %p", *stream);
    DEBUG_LOG(rank, "=== init_acl_environment END ===");
    return 0;
}

/**
 * 初始化SHMEM环境（每个引擎调用一次）
 */
int init_shmem_environment(int rank, int world_size, uint64_t mem_size, EngineType engine) {
    DEBUG_LOG(rank, "=== init_shmem_environment START ===");
    DEBUG_LOG(rank, "world_size=%d, mem_size=%lu MB, engine=%s",
              world_size, (unsigned long)(mem_size / (1024 * 1024)), engine_name(engine).c_str());
    TIMESTAMP(rank, "SHMEM_INIT_START");

    // aclshmemx_init_attr_t: shmem初始化属性结构体
    aclshmemx_init_attr_t attributes;

    // test_set_attr: 辅助函数，填充shmem初始化属性结构体
    test_set_attr(rank, world_size, mem_size, ipport, default_flag_uid, &attributes);
    DEBUG_LOG(rank, "test_set_attr done: my_pe=%d, n_pes=%d, ip_port=%s, mem_size=%lu",
              attributes.my_pe, attributes.n_pes, ipport, (unsigned long)attributes.local_mem_size);

    // 根据引擎类型设置数据传输引擎和超时配置:
    // 参考mte_perftest: MTE引擎不显式设置engine_type，使用默认值
    // 参考rdma_perftest: RDMA引擎设置ACLSHMEM_DATA_OP_ROCE
    if (engine == EngineType::RDMA) {
        attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
        attributes.option_attr.shm_init_timeout = 10;
        attributes.option_attr.shm_create_timeout = 10;
        attributes.option_attr.control_operation_timeout = 10;
        DEBUG_LOG(rank, "Engine set to RDMA (ROCE) with timeout=10s");
    } else if (engine == EngineType::SDMA) {
        attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;
        DEBUG_LOG(rank, "Engine set to SDMA");
    } else {
        // MTE引擎：不设置engine_type，使用默认MTE（参考mte_perftest）
        DEBUG_LOG(rank, "Engine set to MTE (default, no explicit engine_type)");
    }

    // aclshmemx_set_conf_store_tls: 设置配置存储TLS（可选）
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    DEBUG_LOG(rank, "aclshmemx_set_conf_store_tls done");

    // aclshmemx_init_attr: 初始化shmem运行时
    int init_ret = aclshmemx_init_attr(BENCHMARK_INIT_FLAG, &attributes);
    if (init_ret != ACLSHMEM_SUCCESS) {
        fprintf(stderr, "[ERROR][Rank %d] aclshmemx_init_attr failed for engine %s, ret=%d\n",
                rank, engine_name(engine).c_str(), init_ret);
        switch (init_ret) {
            case ACLSHMEM_INVALID_PARAM:
                fprintf(stderr, "[ERROR][Rank %d] Error: ACLSHMEM_INVALID_PARAM (-1)\n", rank);
                break;
            case ACLSHMEM_INVALID_VALUE:
                fprintf(stderr, "[ERROR][Rank %d] Error: ACLSHMEM_INVALID_VALUE (-2)\n", rank);
                break;
            case ACLSHMEM_SMEM_ERROR:
                fprintf(stderr, "[ERROR][Rank %d] Error: ACLSHMEM_SMEM_ERROR (-3) - Shared memory problem\n", rank);
                fprintf(stderr, "[HINT][Rank %d] Check: memory limits, /dev/shmem permissions, NPU device status\n", rank);
                break;
            case ACLSHMEM_INNER_ERROR:
                fprintf(stderr, "[ERROR][Rank %d] Error: ACLSHMEM_INNER_ERROR (-4) - Internal error\n", rank);
                fprintf(stderr, "[HINT][Rank %d] Check: NPU device status, peer process running\n", rank);
                break;
            case ACLSHMEM_NOT_INITED:
                fprintf(stderr, "[ERROR][Rank %d] Error: ACLSHMEM_NOT_INITED (-5)\n", rank);
                break;
            case ACLSHMEM_BOOTSTRAP_ERROR:
                fprintf(stderr, "[ERROR][Rank %d] Error: ACLSHMEM_BOOTSTRAP_ERROR (-6) - Bootstrap connection failed\n", rank);
                fprintf(stderr, "[HINT][Rank %d] Check: TCP port %s availability, firewall settings\n", rank, ipport);
                break;
            case ACLSHMEM_TIMEOUT_ERROR:
                fprintf(stderr, "[ERROR][Rank %d] Error: ACLSHMEM_TIMEOUT_ERROR (-7) - Connection timeout\n", rank);
                fprintf(stderr, "[HINT][Rank %d] Peer may not have started, or network unreachable\n", rank);
                break;
            case ACLSHMEM_MALLOC_FAILED:
                fprintf(stderr, "[ERROR][Rank %d] Error: ACLSHMEM_MALLOC_FAILED (-8) - Memory allocation failed\n", rank);
                fprintf(stderr, "[HINT][Rank %d] Check: system memory availability\n", rank);
                break;
            default:
                fprintf(stderr, "[ERROR][Rank %d] Error: Unknown error code %d\n", rank, init_ret);
        }
        return -1;
    }

    TIMESTAMP(rank, "SHMEM_INIT_END");
    DEBUG_LOG(rank, "aclshmemx_init_attr success for engine %s", engine_name(engine).c_str());

    // 验证初始化后的状态
    int my_pe = aclshmem_my_pe();
    int n_pes = aclshmem_n_pes();
    DEBUG_LOG(rank, "aclshmem_my_pe=%d, aclshmem_n_pes=%d (expected: my_pe=%d, n_pes=%d)",
              my_pe, n_pes, rank, world_size);

    if (my_pe != rank || n_pes != world_size) {
        fprintf(stderr, "[ERROR][Rank %d] PE mismatch! my_pe=%d vs rank=%d, n_pes=%d vs world_size=%d\n",
                rank, my_pe, rank, n_pes, world_size);
        return -1;
    }

    DEBUG_LOG(rank, "=== init_shmem_environment END ===");
    return 0;
}

/**
 * 终止SHMEM环境（每个引擎结束时调用）
 * 参考参考实现：确保彻底清理状态
 */
void finalize_shmem_environment(int rank) {
    DEBUG_LOG(rank, "=== finalize_shmem_environment START ===");
    TIMESTAMP(rank, "SHMEM_FINALIZE_START");

    // 参考rdma_perftest/mte_perftest: 调用aclshmem_finalize清理所有SHMEM资源
    DEBUG_LOG(rank, "calling aclshmem_finalize...");
    aclshmem_finalize();
    DEBUG_LOG(rank, "aclshmem_finalize done");

    // 参考参考实现: 添加短暂延迟让系统稳定
    // 多引擎迭代时需要让前一个引擎的状态彻底清理
    usleep(100000);  // 100ms
    DEBUG_LOG(rank, "cleanup stabilization delay done");

    TIMESTAMP(rank, "SHMEM_FINALIZE_END");
    DEBUG_LOG(rank, "=== finalize_shmem_environment END ===");
}

/**
 * 终止ACL环境（最后调用一次）
 * 参考rdma_perftest/mte_perftest: aclrtResetDevice 在 aclFinalize 之前调用
 */
void finalize_acl_environment(int rank) {
    int32_t device_id = f_npu;
    DEBUG_LOG(rank, "=== finalize_acl_environment START ===");

    DEBUG_LOG(rank, "calling aclrtResetDevice(%d)...", device_id);
    aclrtResetDevice(device_id);
    DEBUG_LOG(rank, "aclrtResetDevice done");

    DEBUG_LOG(rank, "calling aclFinalize...");
    aclFinalize();
    DEBUG_LOG(rank, "aclFinalize done");

    DEBUG_LOG(rank, "=== finalize_acl_environment END ===");
}

/**
 * ========== RDMA PingPong延迟测试 ==========
 */
StatsResult test_rdma_pingpong_latency(aclrtStream stream, uint64_t ffts_config,
                                         uint8_t* gva, size_t msg_size,
                                         int iterations, int warmup) {
    int rank = aclshmem_my_pe();
    DEBUG_LOG(rank, "=== test_rdma_pingpong_latency START ===");
    DEBUG_LOG(rank, "msg_size=%zu bytes (%zu KB), iterations=%d, warmup=%d",
              msg_size, msg_size / 1024, iterations, warmup);
    DEBUG_LOG(rank, "stream=%p, ffts_config=0x%lx, gva=%p", stream, (unsigned long)ffts_config, gva);

    CHECK_PTR(rank, stream, "stream");
    CHECK_PTR(rank, gva, "gva");

    // 参考rdma_perftest: 初始化测试数据
    // 数据布局：PE i的数据位于 gva + i * msg_size
    int64_t* init_data;
    size_t total_data_size = msg_size * 2;  // 2个PE的数据
    aclrtMallocHost((void**)&init_data, total_data_size);
    // 参考rdma_perftest: 每个PE的数据值为 pe_id + 10
    for (size_t i = 0; i < msg_size / sizeof(int64_t); i++) {
        init_data[i] = rank + 10;  // 每个PE初始化自己的数据
    }
    // 拷贝数据到对称内存
    aclrtMemcpy(gva + rank * msg_size, msg_size, init_data, msg_size, ACL_MEMCPY_HOST_TO_DEVICE);
    DEBUG_LOG(rank, "Test data initialized: gva[%zu] = %ld", rank * msg_size / sizeof(int64_t), (long)(rank + 10));
    aclrtFreeHost(init_data);

    // 分配结果buffer：存储每次迭代的cycles值
    uint8_t* result_buffer;
    size_t result_size = iterations * sizeof(int64_t) + sizeof(int64_t);
    DEBUG_LOG(rank, "allocating result_buffer, size=%zu bytes", result_size);

    aclError ret = aclrtMalloc((void**)&result_buffer, result_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR][Rank %d] aclrtMalloc result_buffer failed, ret=%d\n", rank, ret);
        return StatsResult{0,0,0,0,0};
    }
    DEBUG_LOG(rank, "result_buffer allocated at %p", result_buffer);

    // 初始化结果buffer为0xFF（方便调试）
    int64_t* init_buf;
    aclrtMallocHost((void**)&init_buf, result_size);
    memset(init_buf, 0xFF, result_size);
    aclrtMemcpy(result_buffer, result_size, init_buf, result_size, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtFreeHost(init_buf);
    DEBUG_LOG(rank, "result_buffer initialized to 0xFF");

    TIMESTAMP(rank, "KERNEL_LAUNCH_START");

    // launch_rdma_pingpong_latency: 启动RDMA PingPong延迟测试Kernel
    DEBUG_LOG(rank, "launching rdma_pingpong_latency kernel...");
    launch_rdma_pingpong_latency(1, stream, ffts_config, gva,
                                  msg_size, iterations, warmup, result_buffer);
    DEBUG_LOG(rank, "kernel launched, waiting for synchronization...");

    TIMESTAMP(rank, "KERNEL_LAUNCH_END");

    // aclrtSynchronizeStream: 同步stream，等待Kernel执行完成
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR][Rank %d] aclrtSynchronizeStream failed, ret=%d\n", rank, ret);
        aclrtFree(result_buffer);
        return StatsResult{0,0,0,0,0};
    }
    DEBUG_LOG(rank, "stream synchronized, kernel completed");

    TIMESTAMP(rank, "KERNEL_SYNC_END");

    // 分配Host端内存，用于接收结果数据
    int64_t* host_result;
    DEBUG_LOG(rank, "allocating host memory for result...");
    ret = aclrtMallocHost((void**)&host_result, result_size);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR][Rank %d] aclrtMallocHost failed, ret=%d\n", rank, ret);
        aclrtFree(result_buffer);
        return StatsResult{0,0,0,0,0};
    }
    DEBUG_LOG(rank, "host_result allocated at %p", host_result);

    // aclrtMemcpy: 将结果从Device拷贝到Host
    DEBUG_LOG(rank, "copying result from device to host...");
    ret = aclrtMemcpy(host_result, result_size, result_buffer, result_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR][Rank %d] aclrtMemcpy failed, ret=%d\n", rank, ret);
        aclrtFreeHost(host_result);
        aclrtFree(result_buffer);
        return StatsResult{0,0,0,0,0};
    }
    DEBUG_LOG(rank, "result copied to host");

    // 打印前10个结果值（调试）
    DEBUG_LOG(rank, "=== First 10 raw cycles values ===");
    for (int i = 0; i < std::min(10, iterations); i++) {
        DEBUG_LOG(rank, "iteration[%d]: cycles=%ld (0x%lx)", i, (long)host_result[i], (unsigned long)host_result[i]);
    }

    // 统计结果中的有效值数量
    int valid_count = 0;
    int zero_count = 0;
    int invalid_count = 0;
    for (int i = 0; i < iterations; i++) {
        if (host_result[i] == 0) zero_count++;
        else if (host_result[i] == -1 || host_result[i] == 0xFFFFFFFFFFFFFFFFLL) invalid_count++;
        else valid_count++;
    }
    DEBUG_LOG(rank, "Result statistics: valid=%d, zero=%d, invalid(-1/0xFF..)=%d, total=%d",
              valid_count, zero_count, invalid_count, iterations);

    if (zero_count > 0 && valid_count == 0) {
        fprintf(stderr, "[WARN][Rank %d] All results are 0! Kernel may not have executed correctly.\n", rank);
    }

    // 将cycles转换为延迟时间（us）
    std::vector<double> latencies;
    for (int i = 0; i < iterations; i++) {
        double latency_us = cycles_to_us(host_result[i], NPU_FREQ_MHZ);
        latencies.push_back(latency_us);
    }

    // aclrtFreeHost: 释放Host端内存
    aclrtFreeHost(host_result);
    DEBUG_LOG(rank, "host_result freed");

    // ========== 数据验证 ==========
    // 参考mte_perftest: 验证数据传输是否正确
    DEBUG_LOG(rank, "=== Verifying data transfer ===");
    int64_t* verify_buf;
    aclrtMallocHost((void**)&verify_buf, msg_size);
    uint32_t peer = (rank == 0) ? 1 : 0;
    // 检查对端数据是否到达：读取 peer slot 的数据
    aclrtMemcpy(verify_buf, msg_size, gva + peer * msg_size, msg_size, ACL_MEMCPY_DEVICE_TO_HOST);
    // 验证第一个数据值：应该是 peer + 10
    int64_t expected_val = peer + 10;
    VERIFY_RESULT(rank, expected_val, verify_buf[0], "peer data transfer");
    aclrtFreeHost(verify_buf);
    DEBUG_LOG(rank, "=== Data verification done ===");

    // aclrtFree: 释放NPU设备内存
    aclrtFree(result_buffer);
    DEBUG_LOG(rank, "result_buffer freed");

    // compute_stats: 计算统计结果（平均值、标准差、最小值、最大值、中位数）
    StatsResult stats = compute_stats(latencies);
    DEBUG_LOG(rank, "=== test_rdma_pingpong_latency END: mean=%.2f us ===", stats.mean);

    return stats;
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
    int rank = aclshmem_my_pe();
    DEBUG_LOG(rank, "=== test_rdma_bandwidth START ===");
    DEBUG_LOG(rank, "msg_size=%zu bytes, iterations=%d, warmup_rounds=%d, test_rounds=%d",
              msg_size, iterations, warmup_rounds, test_rounds);

    // 分配结果buffer
    uint8_t* result_buffer;
    aclError ret = aclrtMalloc((void**)&result_buffer, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR][Rank %d] aclrtMalloc result_buffer failed, ret=%d\n", rank, ret);
        return StatsResult{0,0,0,0,0};
    }
    DEBUG_LOG(rank, "result_buffer allocated at %p", result_buffer);

    std::vector<double> bandwidths;

    // 多轮测试取平均
    for (int round = 0; round < warmup_rounds + test_rounds; round++) {
        DEBUG_LOG(rank, "Bandwidth round %d (warmup=%d, test=%d)...",
                  round, warmup_rounds, test_rounds);

        // launch_rdma_bandwidth: 启动RDMA带宽测试Kernel
        launch_rdma_bandwidth(1, stream, ffts_config, gva, msg_size, iterations, result_buffer);

        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR][Rank %d] aclrtSynchronizeStream failed in round %d, ret=%d\n",
                    rank, round, ret);
            aclrtFree(result_buffer);
            return StatsResult{0,0,0,0,0};
        }

        int64_t* host_result;
        aclrtMallocHost((void**)&host_result, sizeof(int64_t));
        aclrtMemcpy(host_result, sizeof(int64_t), result_buffer, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

        DEBUG_LOG(rank, "round %d: cycles=%ld", round, (long)host_result[0]);

        double total_time_us = cycles_to_us(host_result[0], NPU_FREQ_MHZ);

        // 带宽计算（单向带宽）
        double bw_gb_s = compute_bandwidth(iterations * msg_size, total_time_us);
        DEBUG_LOG(rank, "round %d: time=%.2f us, bandwidth=%.2f GB/s", round, total_time_us, bw_gb_s);

        // 跳过 warmup 轮次
        if (round >= warmup_rounds) {
            bandwidths.push_back(bw_gb_s);
        }

        aclrtFreeHost(host_result);
    }

    aclrtFree(result_buffer);

    // 计算统计结果（平均带宽、标准差等）
    StatsResult stats = compute_stats(bandwidths);
    DEBUG_LOG(rank, "=== test_rdma_bandwidth END: mean=%.2f GB/s ===", stats.mean);

    return stats;
}

/**
 * ========== MTE PingPong延迟测试 ==========
 */
StatsResult test_mte_pingpong_latency(aclrtStream stream, uint64_t ffts_config,
                                       uint8_t* gva, size_t msg_size,
                                       int iterations, int warmup) {
    int rank = aclshmem_my_pe();
    DEBUG_LOG(rank, "=== test_mte_pingpong_latency START ===");
    DEBUG_LOG(rank, "msg_size=%zu bytes (%zu KB), iterations=%d, warmup=%d",
              msg_size, msg_size / 1024, iterations, warmup);
    DEBUG_LOG(rank, "stream=%p, ffts_config=0x%lx, gva=%p", stream, (unsigned long)ffts_config, gva);

    CHECK_PTR(rank, stream, "stream");
    CHECK_PTR(rank, gva, "gva");

    // 参考mte_perftest: 初始化测试数据
    int64_t* init_data;
    size_t total_data_size = msg_size * 2;  // 2个PE的数据
    aclrtMallocHost((void**)&init_data, total_data_size);
    for (size_t i = 0; i < msg_size / sizeof(int64_t); i++) {
        init_data[i] = rank + 10;
    }
    aclrtMemcpy(gva + rank * msg_size, msg_size, init_data, msg_size, ACL_MEMCPY_HOST_TO_DEVICE);
    DEBUG_LOG(rank, "Test data initialized: gva[%zu] = %ld", rank * msg_size / sizeof(int64_t), (long)(rank + 10));
    aclrtFreeHost(init_data);

    // 分配结果buffer
    uint8_t* result_buffer;
    size_t result_size = iterations * sizeof(int64_t) + sizeof(int64_t);
    DEBUG_LOG(rank, "allocating result_buffer, size=%zu bytes", result_size);

    aclError ret = aclrtMalloc((void**)&result_buffer, result_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR][Rank %d] aclrtMalloc result_buffer failed, ret=%d\n", rank, ret);
        return StatsResult{0,0,0,0,0};
    }
    DEBUG_LOG(rank, "result_buffer allocated at %p", result_buffer);

    // 初始化结果buffer为0xFF（方便调试）
    int64_t* init_buf;
    aclrtMallocHost((void**)&init_buf, result_size);
    memset(init_buf, 0xFF, result_size);
    aclrtMemcpy(result_buffer, result_size, init_buf, result_size, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtFreeHost(init_buf);
    DEBUG_LOG(rank, "result_buffer initialized to 0xFF");

    TIMESTAMP(rank, "KERNEL_LAUNCH_START");

    // launch_mte_pingpong_latency: 启动MTE PingPong延迟测试Kernel
    DEBUG_LOG(rank, "launching mte_pingpong_latency kernel...");
    launch_mte_pingpong_latency(1, stream, ffts_config, gva,
                                 msg_size, iterations, warmup, result_buffer);
    DEBUG_LOG(rank, "kernel launched, waiting for synchronization...");

    TIMESTAMP(rank, "KERNEL_LAUNCH_END");

    // aclrtSynchronizeStream: 同步stream
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR][Rank %d] aclrtSynchronizeStream failed, ret=%d\n", rank, ret);
        aclrtFree(result_buffer);
        return StatsResult{0,0,0,0,0};
    }
    DEBUG_LOG(rank, "stream synchronized, kernel completed");

    TIMESTAMP(rank, "KERNEL_SYNC_END");

    // 分配Host端内存
    int64_t* host_result;
    DEBUG_LOG(rank, "allocating host memory for result...");
    ret = aclrtMallocHost((void**)&host_result, result_size);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR][Rank %d] aclrtMallocHost failed, ret=%d\n", rank, ret);
        aclrtFree(result_buffer);
        return StatsResult{0,0,0,0,0};
    }
    DEBUG_LOG(rank, "host_result allocated at %p", host_result);

    // aclrtMemcpy: 拷贝结果到Host
    DEBUG_LOG(rank, "copying result from device to host...");
    ret = aclrtMemcpy(host_result, result_size, result_buffer, result_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR][Rank %d] aclrtMemcpy failed, ret=%d\n", rank, ret);
        aclrtFreeHost(host_result);
        aclrtFree(result_buffer);
        return StatsResult{0,0,0,0,0};
    }
    DEBUG_LOG(rank, "result copied to host");

    // 打印前10个结果值（调试）
    DEBUG_LOG(rank, "=== First 10 raw cycles values ===");
    for (int i = 0; i < std::min(10, iterations); i++) {
        DEBUG_LOG(rank, "iteration[%d]: cycles=%ld (0x%lx)", i, (long)host_result[i], (unsigned long)host_result[i]);
    }

    // 统计结果中的有效值数量
    int valid_count = 0;
    int zero_count = 0;
    int invalid_count = 0;
    for (int i = 0; i < iterations; i++) {
        if (host_result[i] == 0) zero_count++;
        else if (host_result[i] == -1 || host_result[i] == 0xFFFFFFFFFFFFFFFFLL) invalid_count++;
        else valid_count++;
    }
    DEBUG_LOG(rank, "Result statistics: valid=%d, zero=%d, invalid(-1/0xFF..)=%d, total=%d",
              valid_count, zero_count, invalid_count, iterations);

    if (zero_count > 0 && valid_count == 0) {
        fprintf(stderr, "[WARN][Rank %d] All results are 0! Kernel may not have executed correctly.\n", rank);
    }

    // 将cycles转换为延迟时间
    std::vector<double> latencies;
    for (int i = 0; i < iterations; i++) {
        double latency_us = cycles_to_us(host_result[i], NPU_FREQ_MHZ);
        latencies.push_back(latency_us);
    }

    // 释放Host端内存
    aclrtFreeHost(host_result);
    DEBUG_LOG(rank, "host_result freed");

    // ========== 数据验证 ==========
    DEBUG_LOG(rank, "=== Verifying data transfer ===");
    int64_t* verify_buf;
    aclrtMallocHost((void**)&verify_buf, msg_size);
    uint32_t peer = (rank == 0) ? 1 : 0;
    aclrtMemcpy(verify_buf, msg_size, gva + peer * msg_size, msg_size, ACL_MEMCPY_DEVICE_TO_HOST);
    int64_t expected_val = peer + 10;
    VERIFY_RESULT(rank, expected_val, verify_buf[0], "peer data transfer");
    aclrtFreeHost(verify_buf);
    DEBUG_LOG(rank, "=== Data verification done ===");

    // 释放Device端内存
    aclrtFree(result_buffer);
    DEBUG_LOG(rank, "result_buffer freed");

    // 计算统计结果
    StatsResult stats = compute_stats(latencies);
    DEBUG_LOG(rank, "=== test_mte_pingpong_latency END: mean=%.2f us ===", stats.mean);

    return stats;
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
    int rank = aclshmem_my_pe();
    DEBUG_LOG(rank, "=== test_mte_bandwidth START ===");
    DEBUG_LOG(rank, "msg_size=%zu bytes, iterations=%d, warmup_rounds=%d, test_rounds=%d",
              msg_size, iterations, warmup_rounds, test_rounds);

    // 分配结果buffer
    uint8_t* result_buffer;
    aclError ret = aclrtMalloc((void**)&result_buffer, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR][Rank %d] aclrtMalloc result_buffer failed, ret=%d\n", rank, ret);
        return StatsResult{0,0,0,0,0};
    }
    DEBUG_LOG(rank, "result_buffer allocated at %p", result_buffer);

    std::vector<double> bandwidths;

    // 多轮测试取平均
    for (int round = 0; round < warmup_rounds + test_rounds; round++) {
        DEBUG_LOG(rank, "Bandwidth round %d (warmup=%d, test=%d)...",
                  round, warmup_rounds, test_rounds);

        // launch_mte_bandwidth: 启动MTE带宽测试Kernel
        launch_mte_bandwidth(1, stream, ffts_config, gva, msg_size, iterations, result_buffer);

        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR][Rank %d] aclrtSynchronizeStream failed in round %d, ret=%d\n",
                    rank, round, ret);
            aclrtFree(result_buffer);
            return StatsResult{0,0,0,0,0};
        }

        int64_t* host_result;
        aclrtMallocHost((void**)&host_result, sizeof(int64_t));
        aclrtMemcpy(host_result, sizeof(int64_t), result_buffer, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

        DEBUG_LOG(rank, "round %d: cycles=%ld", round, (long)host_result[0]);

        double total_time_us = cycles_to_us(host_result[0], NPU_FREQ_MHZ);

        // 带宽计算（单向带宽）
        double bw_gb_s = compute_bandwidth(iterations * msg_size, total_time_us);
        DEBUG_LOG(rank, "round %d: time=%.2f us, bandwidth=%.2f GB/s", round, total_time_us, bw_gb_s);

        // 跳过 warmup 轮次
        if (round >= warmup_rounds) {
            bandwidths.push_back(bw_gb_s);
        }

        aclrtFreeHost(host_result);
    }

    aclrtFree(result_buffer);

    // 计算统计结果（平均带宽、标准差等）
    StatsResult stats = compute_stats(bandwidths);
    DEBUG_LOG(rank, "=== test_mte_bandwidth END: mean=%.2f GB/s ===", stats.mean);

    return stats;
}

/**
 * ========== CPU中转测试 ==========
 *
 * 测试CPU Host端作为中转的传输延迟（Device->Host->Device）
 * 用于对比NPU直连通信的性能优势
 */
StatsResult test_cpu_transfer(aclrtStream stream, size_t msg_size, int iterations, int warmup) {
    int rank = aclshmem_my_pe();
    DEBUG_LOG(rank, "=== test_cpu_transfer START ===");
    DEBUG_LOG(rank, "msg_size=%zu bytes, iterations=%d, warmup=%d", msg_size, iterations, warmup);

    // 延迟结果数组
    std::vector<double> latencies;

    // 分配Device端内存
    void* device_buf;
    DEBUG_LOG(rank, "allocating device_buf...");
    aclError ret = aclrtMalloc(&device_buf, msg_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR][Rank %d] aclrtMalloc device_buf failed, ret=%d\n", rank, ret);
        return StatsResult{0,0,0,0,0};
    }
    DEBUG_LOG(rank, "device_buf allocated at %p", device_buf);

    // 分配Host端内存
    void* host_buf;
    DEBUG_LOG(rank, "allocating host_buf...");
    ret = aclrtMallocHost(&host_buf, msg_size);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR][Rank %d] aclrtMallocHost host_buf failed, ret=%d\n", rank, ret);
        aclrtFree(device_buf);
        return StatsResult{0,0,0,0,0};
    }
    DEBUG_LOG(rank, "host_buf allocated at %p", host_buf);

    // memset: 初始化Host端数据为0xAA
    memset(host_buf, 0xAA, msg_size);

    // warmup迭代：预热数据传输路径
    DEBUG_LOG(rank, "warmup iterations...");
    for (int i = 0; i < warmup; i++) {
        aclrtMemcpy(host_buf, msg_size, device_buf, msg_size, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(device_buf, msg_size, host_buf, msg_size, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    // 正式迭代：测量传输延迟
    DEBUG_LOG(rank, "test iterations...");
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        aclrtMemcpy(host_buf, msg_size, device_buf, msg_size, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(device_buf, msg_size, host_buf, msg_size, ACL_MEMCPY_HOST_TO_DEVICE);

        auto end = std::chrono::high_resolution_clock::now();

        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(latency_us);
    }

    // aclrtFree: 释放Device端内存
    DEBUG_LOG(rank, "freeing device_buf...");
    aclrtFree(device_buf);

    // aclrtFreeHost: 释放Host端内存
    DEBUG_LOG(rank, "freeing host_buf...");
    aclrtFreeHost(host_buf);

    // 计算统计结果
    StatsResult stats = compute_stats(latencies);
    DEBUG_LOG(rank, "=== test_cpu_transfer END: mean=%.2f us ===", stats.mean);

    return stats;
}

/**
 * ========== 主测试流程 ==========
 */
int run_benchmark(int rank, int world_size) {
    // 参考mte_perftest: 使用1GB local_mem_size
    // 参考rdma_perftest: 使用64MB local_mem_size
    // 多引擎测试需要更大内存，使用1GB
    uint64_t mem_size = 1024UL * 1024UL * 1024UL;  // 1GB

    // make_dir: 创建结果目录
    make_dir("results");

    // CSVWriter: 创建CSV文件写入器
    // 用于记录测试结果
    CSVWriter latency_csv("results/latency_results.csv");
    CSVWriter bandwidth_csv("results/bandwidth_results.csv");

    // 定义要测试的引擎类型列表（优先测试节点内通信引擎）
    // MTE: 节点内通信，使用片上互联，不需要网卡
    // SDMA: 节点内通信，使用片上SDMA单元，不需要网卡
    // RDMA: 跨节点通信，需要RoCE网卡，单节点测试时可能需要网卡loopback配置
    std::vector<EngineType> engines = {
        EngineType::MTE,   // MTE引擎（节点内）- 优先测试
        EngineType::SDMA,  // SDMA引擎（节点内）- 第二优先
        EngineType::RDMA,  // RDMA引擎（跨节点）- 最后测试
    };

    // 打印测试信息（仅 Rank 0 打印）
    if (rank == 0) {
        std::cout << "\n==================== Comm Benchmark ====================\n";
        std::cout << "Mode: " << BENCHMARK_MODE_NAME << "\n";
        std::cout << "HCCL: " << BENCHMARK_HCCL_MODE_NAME << "\n";
        std::cout << "Rank: " << rank << ", WorldSize: " << world_size << "\n";
        std::cout << "Device: " << f_npu << " (f_npu)\n";
        print_separator();
    }

    // ========== ACL初始化（只调用一次，参考rdma_perftest顺序）==========
    // 参考实现顺序: aclInit → aclrtSetDevice → aclrtCreateStream
    DEBUG_LOG(rank, "=== Initializing ACL (one-time) ===");
    aclrtStream stream = nullptr;
    int acl_ret = init_acl_environment(rank, &stream);
    if (acl_ret != 0) {
        fprintf(stderr, "[ERROR][Rank %d] ACL initialization failed, cannot proceed\n", rank);
        return -1;
    }
    DEBUG_LOG(rank, "ACL initialized, stream=%p", stream);

    // ========== RDMA/MTE/SDMA测试 ==========
    int successful_engines = 0;
    for (EngineType engine : engines) {
        if (rank == 0) {
            std::cout << "\n---------- Testing Engine: " << engine_name(engine) << " ----------\n";
        }

        DEBUG_LOG(rank, "=== Starting engine %s ===", engine_name(engine).c_str());
        TIMESTAMP(rank, "ENGINE_TEST_START");

        // init_shmem_environment: 初始化SHMEM环境（ACL已初始化，stream已创建）
        int init_ret = init_shmem_environment(rank, world_size, mem_size, engine);
        if (init_ret != 0) {
            fprintf(stderr, "[WARN][Rank %d] Engine %s SHMEM initialization failed, skipping this engine\n",
                    rank, engine_name(engine).c_str());
            if (rank == 0) {
                std::cout << "[SKIP] Engine " << engine_name(engine) << " unavailable on this system\n";
            }
            continue;
        }

        successful_engines++;

        // stream已在ACL初始化时创建，这里直接使用
        DEBUG_LOG(rank, "using existing stream=%p", stream);

        // util_get_ffts_config: 获取FFTS配置地址
        uint64_t ffts_config = util_get_ffts_config();
        DEBUG_LOG(rank, "ffts_config=0x%lx", (unsigned long)ffts_config);

        // 参考rdma_perftest: 使用6MB (size6M = 6 * 1024 * 1024)
        // 参考mte_perftest: 根据测试需求动态计算 datasize * block_size
        // 计算实际需要的内存大小：
        // - 延迟测试：2个slot * max_msg_size (2个PE各需要一个发送区)
        // - 带宽测试：2个slot * max_msg_size * block_dim
        // - 同步区域：额外的空间用于magic value同步
        size_t max_msg_size = 8 * 1024 * 1024;  // 8MB (最大测试消息)
        size_t latency_mem = 2 * max_msg_size + 32;  // 延迟测试内存
        size_t bandwidth_mem = 2 * max_msg_size * 32 + 32;  // 带宽测试内存 (block_dim=32)
        size_t required_mem = std::max(latency_mem, bandwidth_mem);
        // 参考rdma_perftest: 使用至少6MB
        size_t alloc_size = std::max(required_mem, (size_t)(6 * 1024 * 1024));
        DEBUG_LOG(rank, "allocating symmetric memory, size=%zu MB (calculated from test requirements)...",
                  alloc_size / (1024 * 1024));
        TIMESTAMP(rank, "SHMEM_MALLOC_START");
        uint8_t* gva = (uint8_t*)aclshmem_malloc(alloc_size);
        TIMESTAMP(rank, "SHMEM_MALLOC_END");
        if (gva == nullptr) {
            fprintf(stderr, "[ERROR][Rank %d] aclshmem_malloc failed, gva is nullptr\n", rank);
            // 不销毁stream，只finalize SHMEM
            finalize_shmem_environment(rank);
            continue;
        }
        DEBUG_LOG(rank, "symmetric memory allocated at %p", gva);

        // Barrier 确保两个 PE 都完成了初始化
        DEBUG_LOG(rank, "calling aclshmem_barrier_all to sync before testing...");
        aclshmem_barrier_all();
        DEBUG_LOG(rank, "barrier completed, both PEs ready");

        // PingPong延迟测试（仅 Rank 0 打印）
        if (rank == 0) {
            std::cout << "\n[PingPong Latency Test]\n";
        }
        for (size_t msg_size : MSG_SIZES) {
            // get_iterations: 根据消息大小获取迭代次数
            int iterations = get_iterations(msg_size);

            // get_warmup_iterations: 根据消息大小获取warmup次数
            int warmup = get_warmup_iterations(msg_size);

            StatsResult result;
            if (engine == EngineType::RDMA) {
                // test_rdma_pingpong_latency: 执行RDMA pingpong延迟测试
                result = test_rdma_pingpong_latency(stream, ffts_config, gva,
                                                     msg_size, iterations, warmup);
            } else if (engine == EngineType::MTE) {
                // test_mte_pingpong_latency: 执行MTE pingpong延迟测试
                result = test_mte_pingpong_latency(stream, ffts_config, gva,
                                                    msg_size, iterations, warmup);
            } else if (engine == EngineType::SDMA) {
                // SDMA 使用与 MTE 相同的测试函数（都是节点内通信）
                result = test_mte_pingpong_latency(stream, ffts_config, gva,
                                                    msg_size, iterations, warmup);
            }

            // 仅 Rank 0 打印结果
            if (rank == 0) {
                print_test_header({rank, world_size, engine, TestType::PINGPONG_LATENCY,
                                   msg_size, iterations, warmup, ipport});
                print_result(result);
                latency_csv.write_row(engine_name(engine), "pingpong_latency",
                                       msg_size, iterations, result);
            }
        }

        // 带宽测试（仅 Rank 0 打印）
        if (rank == 0) {
            std::cout << "\n[Bandwidth Test]\n";
        }
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

            StatsResult result;
            if (engine == EngineType::RDMA) {
                // test_rdma_bandwidth: 执行RDMA带宽测试
                result = test_rdma_bandwidth(stream, ffts_config, gva, msg_size,
                                              iterations, warmup_rounds, test_rounds);
            } else if (engine == EngineType::MTE) {
                // test_mte_bandwidth: 执行MTE带宽测试
                result = test_mte_bandwidth(stream, ffts_config, gva, msg_size,
                                             iterations, warmup_rounds, test_rounds);
            } else if (engine == EngineType::SDMA) {
                // SDMA 使用与 MTE 相同的测试函数（都是节点内通信）
                result = test_mte_bandwidth(stream, ffts_config, gva, msg_size,
                                             iterations, warmup_rounds, test_rounds);
            }

            // 仅 Rank 0 打印结果
            if (rank == 0) {
                print_test_header({rank, world_size, engine, TestType::BANDWIDTH,
                                   msg_size, iterations, 0, ipport});
                std::cout << "WarmupRounds: " << warmup_rounds << ", TestRounds: " << test_rounds << "\n";
                std::cout << "Bandwidth: " << result.mean << " +/- " << result.std
                          << " GB/s (min=" << result.min << ", max=" << result.max << ")\n";
                bandwidth_csv.write_row(engine_name(engine), "bandwidth",
                                          msg_size, iterations, result);
            }
        }

        DEBUG_LOG(rank, "=== Engine %s tests completed ===", engine_name(engine).c_str());

        // Barrier 确保所有测试完成后再释放资源
        DEBUG_LOG(rank, "calling aclshmem_barrier_all before cleanup...");
        aclshmem_barrier_all();
        DEBUG_LOG(rank, "barrier completed, ready to cleanup");

        // aclshmem_free: 释放对称内存
        DEBUG_LOG(rank, "freeing symmetric memory at %p...", gva);
        aclshmem_free(gva);
        DEBUG_LOG(rank, "symmetric memory freed");

        // finalize_shmem_environment: 终止SHMEM环境
        // 注意：不在这里销毁stream，stream在整个生命周期内保持
        finalize_shmem_environment(rank);
        DEBUG_LOG(rank, "=== Engine %s cleanup done ===", engine_name(engine).c_str());

        TIMESTAMP(rank, "ENGINE_TEST_END");
    }

    // 检查是否有成功的引擎
    if (successful_engines == 0) {
        fprintf(stderr, "[ERROR][Rank %d] No SHMEM engine initialization succeeded!\n", rank);
        if (rank == 0) {
            std::cout << "\n[FAILED] All SHMEM engines failed. Check:\n";
            std::cout << "  - NPU device availability\n";
            std::cout << "  - Peer process running on same node\n";
        }
        // 销毁stream再finalize
        aclrtDestroyStream(stream);
        finalize_acl_environment(rank);
        return -1;
    }

    if (rank == 0) {
        std::cout << "\n[SUMMARY] " << successful_engines << " engine(s) tested successfully.\n";
    }

    // ========== CPU中转测试（在ACL finalize之前，使用已有stream）==========
    // 参考rdma_perftest/mte_perftest: CPU测试不需要SHMEM初始化，只需要ACL
    if (rank == 0) {
        std::cout << "\n---------- Testing Engine: CPU_D2H_H2D ----------\n";
    }

    DEBUG_LOG(rank, "=== Starting CPU transfer test ===");
    if (rank == 0) {
        std::cout << "\n[CPU Transfer Latency Test]\n";
    }
    for (size_t msg_size : MSG_SIZES) {
        int iterations = get_iterations(msg_size);
        int warmup = get_warmup_iterations(msg_size);

        DEBUG_LOG(rank, "CPU test: msg_size=%zu, iterations=%d, warmup=%d", msg_size, iterations, warmup);

        StatsResult result = test_cpu_transfer(stream, msg_size, iterations, warmup);

        if (rank == 0) {
            latency_csv.write_row("CPU_D2H_H2D", "pingpong_latency",
                                   msg_size, iterations, result);
        }
    }
    DEBUG_LOG(rank, "=== CPU test completed ===");

// ========== HCCL测试（条件编译，在ACL finalize之前）==========
#ifdef ENABLE_HCCL
    if (rank == 0) {
        std::cout << "\n---------- Testing Engine: HCCL ----------\n";
    }
    std::cout << "[HCCL] Huawei Collective Communication Library Test\n";

    // HCCL需要自己的stream
    aclrtStream hccl_stream = nullptr;
    aclrtCreateStream(&hccl_stream);
    DEBUG_LOG(rank, "HCCL stream created at %p", hccl_stream);

    // HcclComm: HCCL通信组句柄
    HcclComm hccl_comm = nullptr;

    // HcclRootInfo: HCCL根信息结构体
    HcclRootInfo root_info;

    // HcclGetRootInfo: 获取根信息
    HcclGetRootInfo(&root_info);

    // HcclCommInitRootInfo: 使用根信息初始化通信组
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

    // aclrtDestroyStream: 销毁HCCL流
    aclrtDestroyStream(hccl_stream);
    DEBUG_LOG(rank, "HCCL stream destroyed");
#else
    if (rank == 0) {
        std::cout << "\n---------- HCCL Test Skipped ----------\n";
        std::cout << "[INFO] HCCL not enabled. Define ENABLE_HCCL in benchmark_config.h to enable.\n";
    }
#endif

    // ========== 终止ACL环境（最后调用一次）==========
    // 参考rdma_perftest/mte_perftest: aclrtDestroyStream → aclrtResetDevice → aclFinalize
    DEBUG_LOG(rank, "destroying main stream at %p...", stream);
    aclrtDestroyStream(stream);
    DEBUG_LOG(rank, "main stream destroyed");

    finalize_acl_environment(rank);
    DEBUG_LOG(rank, "=== All tests completed, ACL finalized ===");

    if (rank == 0) {
        std::cout << "\n==================== Benchmark Complete ====================\n";
        std::cout << "Results saved to results/ directory\n";
    }

    return 0;
}

/**
 * 主函数：解析命令行参数并运行测试
 */
int main(int argc, char* argv[]) {
    // 检查命令行参数数量
    if (argc < 7) {
        std::cout << "Usage: ./comm_benchmark <n_ranks> <rank_id> <ipport> <g_npus> <f_rank> <device_id>\n";
        std::cout << "Example: ./comm_benchmark 2 0 tcp://127.0.0.1:8765 8 0 1\n";
        std::cout << "         (rank 0 runs on device 1, rank 1 runs on device 2)\n";
        return -1;
    }

    // 解析命令行参数
    int argIdx = 1;

    // n_ranks: 总进程数量（通信组大小）
    int n_ranks = atoi(argv[argIdx++]);

    // rank_id: 当前进程编号（PE编号）
    int rank_id = atoi(argv[argIdx++]);

    // ipport: rendezvous地址（TCP socket地址）
    ipport = argv[argIdx++];

    // g_npus: 节点内NPU总数（用于计算物理设备ID时的取模）
    g_npus = atoi(argv[argIdx++]);

    // f_rank: rank编号偏移量（用于多节点场景）
    f_rank = atoi(argv[argIdx++]);

    // device_id: 直接指定物理设备ID（不再作为偏移量计算）
    int device_id = atoi(argv[argIdx++]);

    // f_npu 设为 device_id（用于 init_acl_environment 和设备操作）
    f_npu = device_id;

    // 打印启动信息
    fprintf(stderr, "\n=== Comm Benchmark START ===\n");
    fprintf(stderr, "[Rank %d] n_ranks=%d, ipport=%s, g_npus=%d, f_rank=%d, device_id=%d\n",
            rank_id, n_ranks, ipport, g_npus, f_rank, device_id);
    fprintf(stderr, "============================\n\n");

    // check_env: 检查环境变量和配置
    if (!check_env()) {
        fprintf(stderr, "[ERROR][Rank %d] Environment check failed\n", rank_id);
        return -1;
    }

    // 打印benchmark模式信息
    std::cout << "\n[Benchmark Mode] " << BENCHMARK_MODE_NAME << "\n";
    std::cout << "[HCCL Status] " << BENCHMARK_HCCL_MODE_NAME << "\n";

    TIMESTAMP(rank_id, "BENCHMARK_START");

    // run_benchmark: 执行benchmark测试
    DEBUG_LOG(rank_id, "Calling run_benchmark...");
    int status = run_benchmark(rank_id, n_ranks);
    DEBUG_LOG(rank_id, "run_benchmark returned, status=%d", status);

    TIMESTAMP(rank_id, "BENCHMARK_END");

    if (status == 0) {
        std::cout << "[SUCCESS] Benchmark completed for rank " << rank_id << "\n";
    } else {
        fprintf(stderr, "[ERROR][Rank %d] Benchmark failed with status=%d\n", rank_id, status);
    }

    fprintf(stderr, "\n=== Comm Benchmark END (Rank %d) ===\n", rank_id);
    return status;
}