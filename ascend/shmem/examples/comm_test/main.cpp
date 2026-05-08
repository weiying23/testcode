/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <iomanip>
#include <getopt.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

#include "acl/acl.h"
#include "shmem.h"
#include "host/shmem_host_def.h"
#include "benchmark_config.h"

using namespace engine_bench;

// ========== 全局配置 ==========
int g_npus = 8;
const char *ipport = "tcp://127.0.0.1:8898";
int f_pe = 0;
int f_npu = 0;
int device_id_override = -1;  // 直接指定device ID，-1表示使用自动计算

// ========== 外部Kernel启动函数 ==========
extern "C" void launch_mte_bench_kernel(uint32_t block_dim, void *stream,
                                        uint8_t *dst_gva, uint8_t *src_gva,
                                        int elements, int peer_pe, int ub_size_kb,
                                        int loop_count, int warmup, int mode, int dtype);

extern "C" void launch_sdma_bench_kernel(uint32_t block_dim, void *stream,
                                         uint8_t *dst_gva, uint8_t *src_gva,
                                         int elements, int peer_pe, int ub_size_kb,
                                         int loop_count, int warmup, int mode, int dtype);

// ========== 性能结果存储 ==========
static aclshmem_prof_pe_t *out_profs = nullptr;

// ========== 辅助函数 ==========
static char g_ipport[ACLSHMEM_MAX_IP_PORT_LEN] = {0};
static aclshmemx_uniqueid_t g_uid = {0};

int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size,
                      const char *ip_port, aclshmemx_init_attr_t *attributes)
{
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), static_cast<size_t>(ACLSHMEM_MAX_IP_PORT_LEN) - 1);
        std::copy_n(ip_port, ip_len, g_ipport);
        std::copy_n(ip_port, ip_len, attributes->ip_port);
        g_ipport[ip_len] = '\0';
        attributes->ip_port[ip_len] = '\0';
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT,
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};

    // 使用全局变量 g_uid
    attributes->comm_args = reinterpret_cast<void *>(&g_uid);
    g_uid.my_pe = my_pe;
    g_uid.n_pes = n_pes;

    return ACLSHMEM_SUCCESS;
}

// ========== 创建目录 ==========
bool make_dir(const std::string& path) {
    if (path.empty()) return true;
    if (mkdir(path.c_str(), 0755) == 0) return true;
    if (errno == EEXIST) return true;
    return false;
}

std::string get_dir(const std::string& filename) {
    size_t pos = filename.find_last_of("/");
    if (pos == std::string::npos) return "";
    return filename.substr(0, pos);
}

// ========== CSV写入 ==========
void write_csv(const std::string& filename,
               const std::vector<std::vector<std::string>>& data)
{
    std::string dir = get_dir(filename);
    if (!dir.empty() && !make_dir(dir)) {
        std::cerr << "Error: cannot create dir " << dir << std::endl;
        return;
    }

    std::ofstream out_file(filename);
    if (!out_file.is_open()) {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return;
    }

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            out_file << row[i];
            if (i < row.size() - 1) out_file << ",";
        }
        out_file << "\n";
    }
    out_file.close();
}

// ========== 性能数据解析 ==========
PerfResult parse_perf_result(aclshmem_prof_pe_t *profs, size_t msg_size,
                             int block_size, int iterations, int g_npus)
{
    PerfResult result;
    result.msg_size = msg_size;
    result.iterations = iterations;

    // 获取硬件周期到微秒的转换系数
    int64_t cycle2us = get_cycle_to_us_ratio();

    double max_core_time = 0.0;
    int actual_blocks = std::min(block_size, ACLSHMEM_CYCLE_PROF_MAX_BLOCK);

    for (int block_id = 0; block_id < actual_blocks; block_id++) {
        aclshmem_prof_block_t *prof = &profs->block_prof[block_id];
        if (prof->ccount[0] == 0) continue;

        double avg_us = (double)prof->cycles[0] / prof->ccount[0] / cycle2us;
        if (avg_us > max_core_time) {
            max_core_time = avg_us;
        }
    }

    result.time_us = max_core_time;
    result.latency_us = max_core_time / iterations;

    // 计算带宽: 总数据量 / 时间
    if (max_core_time > 0) {
        double total_bytes = (double)msg_size * (double)block_size;
        result.bandwidth_gbs = total_bytes / max_core_time * 1000000.0 / 1024.0 / 1024.0 / 1024.0;
    } else {
        result.bandwidth_gbs = 0.0;
    }

    return result;
}

// ========== 引擎类型字符串解析 ==========
EngineType parse_engine_type(const char *str) {
    if (strcmp(str, "mte_intra") == 0) return EngineType::MTE_INTRA_CARD;
    if (strcmp(str, "mte_inter") == 0) return EngineType::MTE_INTER_CARD;
    if (strcmp(str, "sdma_inter") == 0) return EngineType::SDMA_INTER_CARD;
    if (strcmp(str, "all") == 0) return EngineType::MTE_INTER_CARD;  // 默认
    return EngineType::MTE_INTER_CARD;
}

TestMode parse_test_mode(const char *str) {
    if (strcmp(str, "put") == 0) return TestMode::PUT;
    if (strcmp(str, "get") == 0) return TestMode::GET;
    if (strcmp(str, "bi_put") == 0) return TestMode::BI_PUT;
    if (strcmp(str, "bi_get") == 0) return TestMode::BI_GET;
    return TestMode::PUT;
}

DataType parse_data_type(const char *str) {
    if (strcmp(str, "float") == 0) return DataType::FLOAT;
    if (strcmp(str, "int32") == 0) return DataType::INT32;
    if (strcmp(str, "int64") == 0) return DataType::INT64;
    return DataType::FLOAT;
}

// ========== 单引擎性能测试模板 ==========
template<typename T>
int run_engine_benchmark(TestConfig config, std::vector<PerfResult>& results)
{
    int pe_id = config.pe_id;
    int n_pes = config.n_pes;
    int device_id;
    if (config.device_id >= 0) {
        // 直接使用指定的device ID
        device_id = config.device_id;
    } else {
        // 自动计算（原来的方式）
        device_id = pe_id % config.g_npus + config.f_npu;
    }

    int status = 0;
    aclrtStream stream = nullptr;

    // ========== ACL初始化 ==========
    status = aclInit(nullptr);
    if (status != ACL_ERROR_NONE) {
        std::cerr << "aclInit failed: " << status << std::endl;
        return status;
    }

    status = aclrtSetDevice(device_id);
    if (status != ACL_ERROR_NONE) {
        std::cerr << "aclrtSetDevice failed: " << status << std::endl;
        return status;
    }

    status = aclrtCreateStream(&stream);
    if (status != ACL_ERROR_NONE) {
        std::cerr << "aclrtCreateStream failed: " << status << std::endl;
        return status;
    }

    // ========== SHMEM初始化 ==========
    uint64_t local_mem_size = 1024UL * 1024UL * 1024 * 4;  // 4GB
    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, config.ipport.c_str(), &attributes);

    // 根据引擎类型设置数据操作引擎
    if (config.engine == EngineType::SDMA_INTER_CARD) {
        attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;
    } else {
        attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
    }

    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (status != ACLSHMEM_SUCCESS) {
        std::cerr << "aclshmemx_init_attr failed: " << status << std::endl;
        return status;
    }

    // ========== 测试循环 ==========
    int peer_pe = (pe_id + 1) % n_pes;
    int prof_pe = 0;  // 性能采集PE

    // 对于卡内测试（MTE_INTRA），peer_pe应该是自己（同一NPU内的不同内存区域）
    // 注意：当前SHMEM架构下，真正的"卡内"需要特殊处理
    // 这里我们通过设置不同的通信目标来模拟

    std::vector<size_t> msg_sizes = get_msg_sizes();

    for (size_t msg_size : msg_sizes) {
        int iterations = get_iterations(msg_size);
        int warmup = get_warmup_iterations(msg_size);

        std::cout << "Testing " << engine_name(config.engine)
                  << " msg_size=" << msg_size << " bytes"
                  << " iterations=" << iterations << std::endl;

        // 分配对称内存
        size_t alloc_size = msg_size * config.block_size;
        void *dst_ptr = aclshmem_malloc(alloc_size);
        void *src_ptr = aclshmem_malloc(alloc_size);

        if (dst_ptr == nullptr || src_ptr == nullptr) {
            std::cerr << "aclshmem_malloc failed" << std::endl;
            break;
        }

        // 初始化数据
        int elements = alloc_size / sizeof(T);
        std::vector<T> src_data(elements, static_cast<T>(pe_id + 10));
        std::vector<T> dst_data(elements, static_cast<T>(pe_id + 100));

        aclrtMemcpy(src_ptr, alloc_size, src_data.data(), alloc_size, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dst_ptr, alloc_size, dst_data.data(), alloc_size, ACL_MEMCPY_HOST_TO_DEVICE);

        // 屏障同步
        aclshmem_barrier_all();

        // 执行Kernel
        if (config.engine == EngineType::SDMA_INTER_CARD) {
            launch_sdma_bench_kernel(config.block_size, stream,
                                     (uint8_t *)dst_ptr, (uint8_t *)src_ptr,
                                     elements, peer_pe, config.ub_size_kb,
                                     iterations, warmup,
                                     static_cast<int>(config.mode),
                                     static_cast<int>(config.dtype));
        } else {
            launch_mte_bench_kernel(config.block_size, stream,
                                    (uint8_t *)dst_ptr, (uint8_t *)src_ptr,
                                    elements, peer_pe, config.ub_size_kb,
                                    iterations, warmup,
                                    static_cast<int>(config.mode),
                                    static_cast<int>(config.dtype));
        }

        aclrtSynchronizeStream(stream);

        // 收集性能数据
        aclshmemx_show_prof(&out_profs, false);

        if (pe_id == prof_pe && out_profs != nullptr) {
            PerfResult result = parse_perf_result(out_profs, msg_size,
                                                  config.block_size, iterations, config.g_npus);
            results.push_back(result);
        }

        aclshmemx_show_prof(nullptr, true);  // 清空

        aclshmem_free(dst_ptr);
        aclshmem_free(src_ptr);
    }

    // ========== 资源释放 ==========
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();

    return 0;
}

// ========== 主函数 ==========
int main(int argc, char *argv[])
{
    TestConfig config;
    config.pe_id = 0;
    config.n_pes = 2;
    config.g_npus = 2;
    config.f_npu = 0;
    config.device_id = -1;  // 默认自动计算
    config.engine = EngineType::MTE_INTER_CARD;
    config.mode = TestMode::PUT;
    config.dtype = DataType::FLOAT;
    config.msg_size = 0;  // 由get_msg_sizes决定
    config.block_size = 32;
    config.ub_size_kb = 16;
    config.ipport = "tcp://127.0.0.1:8898";

    bool test_all_engines = false;

    static struct option long_options[] = {
        {"pes", required_argument, 0, 0},
        {"pe-id", required_argument, 0, 0},
        {"device", required_argument, 0, 'D'},  // 直接指定NPU ID
        {"ipport", required_argument, 0, 0},
        {"gnpus", required_argument, 0, 0},
        {"fnpu", required_argument, 0, 0},
        {"engine", required_argument, 0, 'e'},
        {"mode", required_argument, 0, 'm'},
        {"dtype", required_argument, 0, 'd'},
        {"block-size", required_argument, 0, 'b'},
        {"ub-size", required_argument, 0, 0},
        {"all", no_argument, 0, 'a'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "e:m:d:b:aD:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'D':
                config.device_id = std::atoi(optarg);
                break;
            case 'e':
                config.engine = parse_engine_type(optarg);
                if (strcmp(optarg, "all") == 0) test_all_engines = true;
                break;
            case 'm':
                config.mode = parse_test_mode(optarg);
                break;
            case 'd':
                config.dtype = parse_data_type(optarg);
                break;
            case 'b':
                config.block_size = std::atoi(optarg);
                break;
            case 'a':
                test_all_engines = true;
                break;
            case 0:
                if (strcmp(long_options[option_index].name, "pes") == 0) {
                    config.n_pes = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "pe-id") == 0) {
                    config.pe_id = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "ipport") == 0) {
                    config.ipport = optarg;
                } else if (strcmp(long_options[option_index].name, "gnpus") == 0) {
                    config.g_npus = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "fnpu") == 0) {
                    config.f_npu = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "ub-size") == 0) {
                    config.ub_size_kb = std::atoi(optarg);
                }
                break;
            default:
                std::cerr << "Unknown option" << std::endl;
                return 1;
        }
    }

    // 计算实际使用的device_id
    int actual_device_id;
    if (config.device_id >= 0) {
        actual_device_id = config.device_id;
    } else {
        actual_device_id = config.pe_id % config.g_npus + config.f_npu;
    }

    std::cout << "========== Engine Benchmark Configuration ==========" << std::endl;
    std::cout << "PE: " << config.pe_id << " / " << config.n_pes << std::endl;
    std::cout << "NPU Device: " << actual_device_id << std::endl;
    std::cout << "Engine: " << engine_name(config.engine) << std::endl;
    std::cout << "Mode: " << mode_name(config.mode) << std::endl;
    std::cout << "DataType: " << type_name(config.dtype) << std::endl;
    std::cout << "BlockSize: " << config.block_size << std::endl;
    std::cout << "UB Size: " << config.ub_size_kb << " KB" << std::endl;
    std::cout << "IP: " << config.ipport << std::endl;
    std::cout << "====================================================" << std::endl;

    std::vector<PerfResult> results;

    // 执行测试
    #define RUN_BENCH(type) \
        run_engine_benchmark<type>(config, results)

    switch (config.dtype) {
        case DataType::FLOAT: RUN_BENCH(float); break;
        case DataType::INT32: RUN_BENCH(int32_t); break;
        case DataType::INT64: RUN_BENCH(int64_t); break;
        default: RUN_BENCH(float); break;
    }

    #undef RUN_BENCH

    // 输出结果到CSV
    if (config.pe_id == 0) {
        std::string csv_filename = "output/" + engine_name(config.engine) + "_" +
                                   mode_name(config.mode) + "_" +
                                   type_name(config.dtype) + ".csv";

        std::vector<std::vector<std::string>> csv_data;
        csv_data.push_back({"MsgSize(B)", "Bandwidth(GB/s)", "Latency(us)",
                            "Time(us)", "Iterations"});

        for (const auto& r : results) {
            csv_data.push_back({std::to_string(r.msg_size),
                                std::to_string(r.bandwidth_gbs),
                                std::to_string(r.latency_us),
                                std::to_string(r.time_us),
                                std::to_string(r.iterations)});
        }

        write_csv(csv_filename, csv_data);
        std::cout << "Results saved to: " << csv_filename << std::endl;
    }

    std::cout << "[SUCCESS] Engine benchmark completed in PE " << config.pe_id << std::endl;

    return 0;
}