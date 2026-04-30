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
#include <iomanip>
#include "utils.h"
#include "perftest_common_types.h"
#include "mte_perftest_common.h"

int g_npus = 8;
const char *ipport;
static const char *fuc_data_type;
static const char *fuc_test_type;
int f_pe = 0;
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

static aclshmem_prof_pe_t *out_profs;
extern "C" void launch_mte_perf_kernel(uint32_t block_dim, void *stream, uint8_t *dst_gva, uint8_t *src_gva, int elements, int32_t frame_id, int test_mode, int data_type, int ub_size_kb, int64_t prof_pe_val, int loop_count);

static perftest::mte_mode_t get_mte_mode(const char *test_type_str) {
    if (strcmp(test_type_str, "put") == 0) return perftest::TEST_MODE_MTE_PUT;
    else if (strcmp(test_type_str, "bi_put") == 0) return perftest::TEST_MODE_BI_PUT;
    else if (strcmp(test_type_str, "get") == 0) return perftest::TEST_MODE_MTE_GET;
    else if (strcmp(test_type_str, "bi_get") == 0) return perftest::TEST_MODE_BI_GET;
    return perftest::TEST_MODE_MTE_PUT;
}

template<typename T>
int test_shmem_mte_perf_test_impl(int pe_id, int n_pes, uint64_t local_mem_size,
                                   int min_block_size, int max_block_size,
                                   int min_exponent, int max_exponent, int loop_count, perftest::mte_mode_t test_mode, perftest::perf_data_type_t data_type_enum, int prof_pe, int ub_size_kb, std::vector<std::vector<std::string>>& csv_data)
{
    // 计算物理设备ID：pe_id % g_npus + f_npu
    int32_t device_id = (pe_id % g_npus + f_npu);
    int status = 0;
    aclrtStream stream = nullptr;

    // ========== ACL初始化 ==========
    // aclInit: 初始化ACL（Ascend Computing Language）运行时环境
    // 参数: nullptr表示使用默认配置
    // 必须在调用任何ACL API之前执行
    status = aclInit(nullptr);
    // aclrtSetDevice: 设置当前进程使用的NPU设备
    // 将进程绑定到指定NPU，后续所有ACL操作在该设备上执行
    status = aclrtSetDevice(device_id);
    // aclrtCreateStream: 创建ACL流（用于异步操作队列）
    status = aclrtCreateStream(&stream);

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
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // aclshmemx_init_attr: 初始化shmem运行时（默认socket模式）
    // 参数详解:
    // - ACLSHMEMX_INIT_WITH_DEFAULT: 初始化模式标志
    //   使用TCP socket进行进程间rendezvous
    // - &attributes: 初始化属性结构体指针
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 执行后完成:
    // 1. 建立进程间通信通道
    // 2. 分配对称内存堆
    // 3. 初始化MTE通信引擎（默认引擎）
    // 4. 设置PE编号和通信组信息
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    int frame_id = 0;
    for (int block_size = min_block_size; block_size <= max_block_size; block_size++) {
        for (int exponent = min_exponent; exponent <= max_exponent; exponent++) {
            int datasize = std::pow(2, exponent);
            std::cout << "pe: " << pe_id << " block_size: " << block_size << " size: " << datasize << " frame_id: " << frame_id << std::endl;

            // aclshmem_malloc: 分配对称内存（用于MTE性能测试数据）
            // 参数详解:
            // - datasize * block_size: 对称内存大小
            //   根据测试的数据大小和核数计算
            // 返回值: 对称内存指针（GVA格式）
            // 对称内存用途：
            // - 存存MTE Put/Get测试的源数据和目标数据
            // - MTE引擎通过GVA地址直接访问其他PE的数据
            // 对称内存核心特点：
            // 1. 所有PE在同一虚拟地址上拥有相同大小的内存块
            // 2. PE i可以直接通过GVA地址访问PE j的数据
            // 3. 用于存放通信数据和同步标志
            void *dst_ptr = aclshmem_malloc(datasize * block_size);
            void *src_ptr = aclshmem_malloc(datasize * block_size);

            int all_size = datasize * block_size;
            int trans_size = all_size / sizeof(T);

            // 初始化测试数据
            std::vector<T> src_input(trans_size, 0);
            std::vector<T> dst_input(trans_size, 0);
            for (int i = 0; i < trans_size; i++) {
                src_input[i] = (T)(pe_id + 10);
                dst_input[i] = (T)(pe_id + 100);
            }

            // aclrtMemcpy: 将Host端数据拷贝到Device端对称内存
            status = aclrtMemcpy(src_ptr, all_size,
                                src_input.data(), all_size, ACL_MEMCPY_HOST_TO_DEVICE);
            status = aclrtMemcpy(dst_ptr, all_size,
                                dst_input.data(), all_size, ACL_MEMCPY_HOST_TO_DEVICE);

            // ========== MTE性能测试 ==========
            // launch_mte_perf_kernel: 执行MTE性能测试Kernel
            // 参数详解:
            // - block_size: 核数（AIV核心数量）
            // - stream: ACL流
            // - dst_ptr: 目标地址（对称内存指针，GVA格式）
            // - src_ptr: 源地址（对称内存指针，GVA格式）
            // - trans_size: 元素数量
            // - frame_id: 测试帧ID（用于性能统计）
            // - test_mode: 测试模式（Put/Get/Bi_Put/Bi_Get）
            // - data_type_enum: 数据类型（float/int8/int16/int32/int64等）
            // - ub_size_kb: UB缓冲区大小（KB）
            // - prof_pe: 性能采集的PE编号
            // - loop_count: 测试循环次数
            // MTE引擎特点：
            // - 使用片上MTE单元进行数据传输
            // - 仅支持节点内NPU间通信
            // - 高带宽、低延迟
            // - 适合大规模数据传输性能测试
            launch_mte_perf_kernel(block_size, stream, (uint8_t *)dst_ptr, (uint8_t *)src_ptr, trans_size, frame_id, static_cast<int>(test_mode), static_cast<int>(data_type_enum), ub_size_kb, prof_pe, loop_count);
            status = aclrtSynchronizeStream(stream);

            // ========== 结果校验 ==========
            bool verify_success = true;

            // compare_values: 比较数据的lambda函数，用于校验传输结果
            auto compare_values = [&](T *ptr1, T *ptr2, size_t count, const char *label1, const char *label2) -> bool {
                for (size_t i = 0; i < count; i++) {
                    if (ptr1[i] != ptr2[i]) {
                        std::cout << "  [ERROR] Mismatch at index " << i << ": " << label1 << "=" << (double)ptr1[i] << ", " << label2 << "=" << (double)ptr2[i] << std::endl;
                        return false;
                    }
                }
                return true;
            };

            std::vector<T> dst_host(trans_size, 0);
            std::vector<T> src_host(trans_size, 0);

            // aclrtMemcpy: 将Device端结果拷贝到Host端进行校验
            status = aclrtMemcpy(dst_host.data(), all_size, dst_ptr, all_size, ACL_MEMCPY_DEVICE_TO_HOST);
            status = aclrtMemcpy(src_host.data(), all_size, src_ptr, all_size, ACL_MEMCPY_DEVICE_TO_HOST);

            // peer_pe: 对端PE编号，用于校验数据来源
            int peer_pe = (pe_id + 1) % n_pes;
            bool is_unilateral = (test_mode == perftest::TEST_MODE_MTE_PUT || test_mode == perftest::TEST_MODE_MTE_GET);
            bool is_put = (test_mode == perftest::TEST_MODE_MTE_PUT || test_mode == perftest::TEST_MODE_BI_PUT);

            // ========== 不同测试模式的校验逻辑 ==========
            if (test_mode == perftest::TEST_MODE_MTE_PUT) {
                std::cout << "\n[Verification] put operation: Checking data transfer..." << std::endl;
                if (pe_id != prof_pe) {
                    T expected_val = static_cast<T>(prof_pe + 10);
                    if (!compare_values(dst_host.data(), &expected_val, 1, "dst[0]", "peer_src[0]")) {
                        verify_success = false;
                        std::cout << "  [ERROR] put operation: destination data does not match source data!" << std::endl;
                    }
                }
            } else if (test_mode == perftest::TEST_MODE_MTE_GET) {
                std::cout << "\n[Verification] get operation: Checking data transfer..." << std::endl;
                if (pe_id == prof_pe) {
                    T expected_val = static_cast<T>(peer_pe + 100);
                    if (!compare_values(src_host.data(), &expected_val, 1, "src[0]", "peer_dst[0]")) {
                        verify_success = false;
                        std::cout << "  [ERROR] get operation: source data does not match destination data!" << std::endl;
                    }
                }
            } else if (test_mode == perftest::TEST_MODE_BI_PUT) {
                std::cout << "\n[Verification] bi_put operation: Checking data transfer..." << std::endl;
                T expected_val = static_cast<T>(peer_pe + 10);
                if (!compare_values(dst_host.data(), &expected_val, 1, "dst[0]", "peer_src[0]")) {
                    verify_success = false;
                    std::cout << "  [ERROR] bi_put operation: destination data does not match source data!" << std::endl;
                }
            } else if (test_mode == perftest::TEST_MODE_BI_GET) {
                std::cout << "\n[Verification] bi_get operation: Checking data transfer..." << std::endl;
                T expected_val = static_cast<T>(peer_pe + 100);
                if (!compare_values(src_host.data(), &expected_val, 1, "src[0]", "peer_dst[0]")) {
                    verify_success = false;
                    std::cout << "  [ERROR] bi_get operation: source data does not match destination data!" << std::endl;
                }
            }

            if (verify_success) {
                std::cout << "[Verification] SUCCESS: Data transferred correctly!" << std::endl;
            } else {
                std::cout << "[Verification] FAILED: Data transfer verification failed!" << std::endl;
            }
            std::cout << "" << std::endl;

            // ========== 性能数据采集 ==========
            // aclshmemx_show_prof: 显示性能统计信息
            // 参数详解:
            // - &out_profs: 输出性能统计结构体指针
            // - false: 不清空统计数据（保留数据用于后续分析）
            aclshmemx_show_prof(&out_profs, false);
            // collect_prof_data_to_csv: 收集性能数据到CSV格式
            collect_prof_data_to_csv(out_profs, frame_id, block_size, datasize, g_npus, ub_size_kb, csv_data);

            // aclshmem_free: 释放对称内存
            // 参数: aclshmem_malloc返回的对称内存指针
            // 必须与aclshmem_malloc配对使用
            aclshmem_free(dst_ptr);
            aclshmem_free(src_ptr);
            status = aclrtSynchronizeStream(stream);
            
            frame_id++;
            if (frame_id >= ACLSHMEM_CYCLE_PROF_FRAME_CNT) {
                std::cerr << "警告: frame_id 超过最大值 " << ACLSHMEM_CYCLE_PROF_FRAME_CNT << ", 停止测试" << std::endl;
                break;
            }
        }
        if (frame_id >= ACLSHMEM_CYCLE_PROF_FRAME_CNT) {
            break;
        }
    }
    // aclshmemx_show_prof: 清空性能统计数据
    // 参数详解:
    // - nullptr: 不输出统计数据
    // - true: 清空统计数据
    aclshmemx_show_prof(nullptr, true);

    // ========== 资源释放 ==========
    // aclshmem_finalize: 终止shmem运行时，释放所有shmem资源
    // 功能详解：
    // - 释放对称内存堆（Symmetric Heap）
    // - 关闭进程间通信通道（TCP socket连接）
    // - 清理MTE通信引擎状态
    // - 释放内部同步机制资源（barrier、quiet等）
    // 返回值: ACLSHMEM_SUCCESS表示成功
    status = aclshmem_finalize();
    status = aclrtDestroyStream(stream);
    status = aclrtResetDevice(device_id);
    status = aclFinalize();

    return 0;
}

int main(int argc, char *argv[])
{
    int status = 0;
    int n_pes = 2;
    int pe_id = 0;
    ipport = "tcp://127.0.0.1:8764";
    g_npus = 2;
    f_pe = 0;
    f_npu = 4;
    const char *test_type = "put";
    fuc_data_type = "float";
    int min_block_size = 32;
    int max_block_size = 32;
    int min_exponent = 3;
    int max_exponent = 17;
    int loop_count = 1000;
    int ub_size_kb = 16;
    
    static struct option long_options[] = {
        {"pes", required_argument, 0, 0},
        {"pe-id", required_argument, 0, 0},
        {"ipport", required_argument, 0, 0},
        {"gnpus", required_argument, 0, 0},
        {"fpe", required_argument, 0, 0},
        {"fnpu", required_argument, 0, 0},
        {"test-type", required_argument, 0, 't'},
        {"datatype", required_argument, 0, 'd'},
        {"block-size", required_argument, 0, 'b'},
        {"block-range", required_argument, 0, 0},
        {"exponent", required_argument, 0, 'e'},
        {"exponent-range", required_argument, 0, 0},
        {"loop-count", required_argument, 0, 0},
        {"ub-size", required_argument, 0, 0},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "t:d:b:e:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 't':
                test_type = optarg;
                break;
            case 'd':
                fuc_data_type = optarg;
                break;
            case 'b':
                min_block_size = max_block_size = std::atoi(optarg);
                break;
            case 'e':
                min_exponent = max_exponent = std::atoi(optarg);
                break;
            case 0:
                if (strcmp(long_options[option_index].name, "pes") == 0) {
                    n_pes = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "pe-id") == 0) {
                    pe_id = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "ipport") == 0) {
                    ipport = optarg;
                } else if (strcmp(long_options[option_index].name, "gnpus") == 0) {
                    g_npus = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "fpe") == 0) {
                    f_pe = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "fnpu") == 0) {
                    f_npu = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "block-range") == 0) {
                    min_block_size = std::atoi(optarg);
                    if (optind < argc) {
                        max_block_size = std::atoi(argv[optind]);
                        optind++;
                    }
                } else if (strcmp(long_options[option_index].name, "exponent-range") == 0) {
                    min_exponent = std::atoi(optarg);
                    if (optind < argc) {
                        max_exponent = std::atoi(argv[optind]);
                        optind++;
                    }
                } else if (strcmp(long_options[option_index].name, "loop-count") == 0) {
                    loop_count = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "ub-size") == 0) {
                    ub_size_kb = std::atoi(optarg);
                }
                break;
            default:
                std::cerr << "错误: 未知参数" << std::endl;
                std::cerr << "使用方法: " << argv[0] << " --pes <n_pes> --pe-id <pe_id> --ipport <ip:port> --gnpus <gnpu_num> --fpe <first_pe> --fnpu <first_npu> [-t <put|bi_put|get|bi_get>] [-d <float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char>] [-b <block_size>] [-e <exponent>] [--block-range <min> <max>] [--exponent-range <min> <max>] [--loop-count <count>] [--ub-size <size>]" << std::endl;
                return 1;
        }
    }
    
    std::cout << "[SUCCESS] demo run start in pe " << pe_id << ", test type: " << test_type << ", data type: " << fuc_data_type << std::endl;
    std::cout << "n_pes: " << n_pes << ", pe_id: " << pe_id << ", g_npus: " << g_npus << std::endl;
    std::cout << "核数范围: " << min_block_size << "-" << max_block_size << std::endl;
    std::cout << "幂数范围: " << min_exponent << "-" << max_exponent << std::endl;
    std::cout << "循环次数: " << loop_count << std::endl;
    std::cout << "UB size (KB): " << ub_size_kb << std::endl;
    
    fuc_test_type = test_type;
    perftest::mte_mode_t test_mode = get_mte_mode(test_type);
    perftest::perf_data_type_t data_type_enum = get_data_type(fuc_data_type);
    
    uint64_t max_datasize = (1ULL << max_exponent);
    uint64_t max_required_size = max_datasize * max_block_size * 2;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    const uint64_t ONE_GB = 1024UL * 1024UL * 1024;
    const uint64_t MAX_GB = 40;
    
    if (max_required_size > local_mem_size) {
        uint64_t gb_needed = (max_required_size + ONE_GB - 1) / ONE_GB;
        if (gb_needed > MAX_GB) {
            std::cerr << "Error: Required memory exceeds 40GB! Max need " << gb_needed << " GB" << std::endl;
            std::cerr << "Please adjust block-range or exponent-range parameters" << std::endl;
            return 1;
        }
        local_mem_size = gb_needed * ONE_GB;
        std::cout << "INFO: Auto-adjust local_mem_size to " << gb_needed << " GB" << std::endl;
    }
    
    const char *prof_pe_env = std::getenv("SHMEM_CYCLE_PROF_PE");
    int prof_pe = 0;
    if (prof_pe_env != nullptr) {
        prof_pe = std::atoi(prof_pe_env);
    }
    
    std::vector<std::vector<std::string>> csv_data = {
        {"DataSize/B", "Npus", "Blocks", "UBsize/KB", "Bandwidth/GB/s", "CoreMaxTime/us", "SingleCoreTime/us"},
    };
    
    #define TEST_IMPL_OP(type) \
        status = test_shmem_mte_perf_test_impl<type>(pe_id, n_pes, local_mem_size, \
                                                      min_block_size, max_block_size, \
                                                      min_exponent, max_exponent, loop_count, test_mode, data_type_enum, prof_pe, ub_size_kb, csv_data)
    
    DISPATCH_BY_TYPE(fuc_data_type, TEST_IMPL_OP);
    
    #undef TEST_IMPL_OP
    
    if (pe_id == prof_pe) {
        std::string csv_filename = "output/" + std::string(fuc_test_type) + "_" + std::string(fuc_data_type) + "_" + int_to_string(prof_pe) + ".csv";
        write_csv(csv_filename, csv_data);
    }
    
    std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    
    return 0;
}
