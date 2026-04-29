/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <iomanip>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <algorithm>

#include "opdev/fp16_t.h"
#include "opdev/bfloat16.h"
#include "utils.h"
#include "param.h"

using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;

#include "acl/acl.h"
#include "shmem.h"
#include "allgather_kernel.h"

int g_npus = 8;
const char *ipport = "tcp://127.0.0.1:8998";
int f_pe = 0;
int f_npu = 0;
const char *data_type = "int";
int perf_times = 50;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr uint32_t MAGIC_MULTIPLIER = 1024;
constexpr uint32_t DATA_SIZE_THRESHOLD = 2097152;
constexpr uint32_t BLOCK_NUM_SMALL_DATA = 8;
constexpr uint32_t BLOCK_NUM_LARGE_DATA = 16;

aclshmemx_uniqueid_t default_flag_uid;

template <class T>
int test_aclshmem_all_gather(int pe_id, int n_pes)
{
    // ACLStream init
    int status = 0;
    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);

    // Prepare FFTS address
    uint64_t fftsAddr = util_get_ffts_config();

    int case_num = 24;
    std::vector<uint32_t> test_cases = {};
    for (int i = 0; i < case_num; i++) {
        int data_len = 16 * (1 << i);
        test_cases.push_back(data_len);
    }

    uint32_t BLOCK_NUM = 8;

    std::string exec_path = __FILE__;
    size_t pos = exec_path.find_last_of("/\\");
    std::string dir_path = exec_path.substr(0, pos);
    std::string result_path = dir_path + "/results.csv";
    std::ofstream outFile(result_path);
    if (!outFile.is_open()) {
        std::cerr << "错误：无法创建文件！" << std::endl;
        return 1;
    }
    outFile << "M,N,Time(us)\n";

    // magic is used to sync.
    int magic = 1;

    for (int i = 0; i < test_cases.size(); i++) {
        if (pe_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Started." << std::endl;
        }
        uint32_t trans_size = test_cases[i];

        //  Small data kernel needs 8 AIV core, Big data kernel needs 16 AIV.
        if (trans_size * sizeof(T) < DATA_SIZE_THRESHOLD) {
            BLOCK_NUM = BLOCK_NUM_SMALL_DATA;
        } else {
            BLOCK_NUM = BLOCK_NUM_LARGE_DATA;
        }

        void *input_ptr;
        aclrtMalloc(&input_ptr, trans_size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        uint8_t *input_host;
        aclrtMallocHost(reinterpret_cast<void**>(&input_host), trans_size * sizeof(T));
        std::string inputFile = "../../examples/allgather/golden/allgather_" + std::to_string(trans_size) + "_" +
                                std::to_string(n_pes) + "/input_gm_" + std::to_string(pe_id) + ".bin";
        ReadFile(inputFile, input_host, trans_size * sizeof(T));
        aclrtMemcpy(input_ptr, trans_size * sizeof(T), input_host, trans_size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

        void *output_ptr;
        aclrtMalloc(&output_ptr, trans_size * n_pes * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

        // sync Buffer + data Buffer
        // BLOCK_NUM: AIV核心数量，用于计算同步缓冲区大小
        int aiv_num = BLOCK_NUM;

        // aclshmem_malloc: 分配对称内存（Symmetric Heap）
        // 对称内存是shmem通信的核心机制：
        // - 所有PE在同一虚拟地址上拥有相同大小的内存块
        // - PE i可以直接通过GVA地址访问PE j的内存
        // - 用于存放通信数据和同步标志
        // 参数详解:
        // - aiv_num * SYNC_FLAG_INTERVAL * sizeof(T): 同步缓冲区大小
        //   用于存放同步标志（magic number等）
        // - GVA_BUFF_MAX_SIZE / sizeof(T): 数据缓冲区大小
        //   用于存放实际通信数据
        // 返回值: 对称内存指针（GVA格式）
        // GVA = Global Virtual Address，全局虚拟地址
        // 示例：如果PE 0在地址ptr存放数据X，PE 1可以通过相同的ptr地址读取到X
        // 注意：
        // 1. 对称内存必须通过aclshmem_free释放，不能使用aclrtFree
        // 2. 分配的内存对所有PE可见，可以通过RMA操作访问
        // 3. 对称内存大小不能超过初始化时设置的local_mem_size
        void *ptr = aclshmem_malloc(aiv_num * SYNC_FLAG_INTERVAL * sizeof(T) + GVA_BUFF_MAX_SIZE / sizeof(T));

        // AllGather
        for (int zz = 0; zz < perf_times; zz++) {
            magic++;
            allgather_demo<T>(BLOCK_NUM, stream, fftsAddr, (uint8_t *)input_ptr,
                              (uint8_t *)output_ptr, (uint8_t *)ptr, trans_size, magic * MAGIC_MULTIPLIER);
        }
        status = aclrtSynchronizeStream(stream);

        aclshmemx_show_prof(nullptr, true);

        // Result Check
        T *output_host;
        size_t output_size = n_pes * trans_size * sizeof(T);
        status = aclrtMallocHost(reinterpret_cast<void**>(&output_host), output_size);
        status = aclrtMemcpy(output_host, output_size, output_ptr, output_size, ACL_MEMCPY_DEVICE_TO_HOST);

        T *golden_host;
        status = aclrtMallocHost(reinterpret_cast<void**>(&golden_host), output_size);
        std::string goldenFile = "../../examples/allgather/golden/allgather_" +
            std::to_string(trans_size) + "_" + std::to_string(n_pes) + "/golden.bin";
        ReadFile(goldenFile, golden_host, n_pes * trans_size * sizeof(T));
        for (int zz = 0; zz < n_pes * trans_size; zz++) {
            if (static_cast<float>(output_host[zz]) != static_cast<float>(golden_host[zz])) {
                std::cout << static_cast<float>(output_host[zz]) << " != " << static_cast<float>(golden_host[zz])
                          << ", trans_size is : " << trans_size << ", idx is: " << zz
                          << ", pe_id is: "<< pe_id << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // ========== 资源释放 ==========
        // aclrtFreeHost: 释放Host端内存
        // 必须与aclrtMallocHost配对使用
        // 参数: aclrtMallocHost返回的Host端内存指针
        // 注意：释放前确保数据已拷贝完成
        status = aclrtFreeHost(input_host);
        status = aclrtFreeHost(output_host);
        status = aclrtFreeHost(golden_host);

        // aclshmem_free: 释放对称内存
        // 参数: aclshmem_malloc返回的对称内存指针
        // 必须与aclshmem_malloc配对使用
        // 执行效果:
        // - 将对称内存归还到Symmetric Heap
        // - 其他shmem操作可以重新分配此内存
        // - 释放后该地址不再可用于通信
        // 重要提示：
        // 1. 不能使用aclrtFree释放对称内存，必须使用aclshmem_free
        // 2. 所有PE应同时释放对称内存，避免内存碎片
        // 3. 释放前确保所有RMA操作已完成（可调用shmem_quiet）
        aclshmem_free(ptr);

        // aclrtFree: 释放普通NPU设备内存
        // 用于释放非对称内存（通过aclrtMalloc分配）
        // 参数: aclrtMalloc返回的设备内存指针
        // 注意：与aclshmem_free区分，此函数释放的是普通设备内存
        aclrtFree(input_ptr);
        aclrtFree(output_ptr);

        outFile << 1 << "," << trans_size << "," << " " << "\n";

        if (pe_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Finised !! Result Correct !!" << std::endl;
        }
    }

    outFile.close();
    status = aclrtDestroyStream(stream);
    return status;
}

int main(int argc, char *argv[])
{
    int status = 0;
    int n_pes = atoi(argv[INDEX1]);
    int pe_id = atoi(argv[INDEX2]);
    ipport = argv[INDEX3];
    g_npus = atoi(argv[INDEX4]);
    f_pe = atoi(argv[INDEX5]);
    f_npu = atoi(argv[INDEX6]);
    data_type = argv[INDEX7];
    perf_times = atoi(argv[INDEX8]);

    // ========== ACL && Shmem 初始化 ==========
    // 计算物理设备ID：pe_id % g_npus + f_npu
    // pe_id: 当前进程的逻辑编号（Process ID）
    // g_npus: 节点内NPU总数
    // f_npu: NPU编号偏移量（物理设备ID的起点）
    // 示例：如果pe_id=3, g_npus=8, f_npu=0，则device_id=3
    int32_t device_id = pe_id % g_npus + f_npu;

    // aclInit: 初始化ACL（Ascend Computing Language）运行时环境
    // 参数: nullptr表示使用默认配置
    // 必须在调用任何ACL API之前执行
    // 初始化CANN软件栈，加载驱动，准备NPU资源
    // 返回值: ACL_SUCCESS表示成功，否则返回错误码
    status = aclInit(nullptr);

    // aclrtSetDevice: 设置当前进程使用的NPU设备
    // 参数: device_id - 物理NPU设备编号
    // 将进程绑定到指定的NPU，后续所有ACL操作在该设备上执行
    // 设置设备上下文，创建计算资源队列
    // 每个进程必须调用此函数后才能执行NPU计算
    status = aclrtSetDevice(device_id);

    // 定义对称内存大小：1GB
    // 对称内存是shmem通信的核心机制：
    // - 所有PE在同一虚拟地址上拥有相同大小的内存块
    // - PE i可以直接通过GVA地址访问PE j的内存
    // - 用于存放通信数据和同步标志
    // - GVA = Global Virtual Address，全局虚拟地址
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;

    // aclshmemx_init_attr_t: shmem初始化属性结构体
    // 包含以下关键字段：
    // - my_pe: 当前PE编号（进程ID），范围[0, n_pes-1]
    // - n_pes: 总PE数量（进程总数）
    // - ip_port: rendezvous地址（TCP socket地址），如"tcp://127.0.0.1:8998"
    //   PE 0监听此地址，其他PE连接到此地址进行握手
    // - local_mem_size: 对称内存大小（字节）
    // - option_attr: 可选属性（引擎类型、超时等）
    //   .data_op_engine_type: 数据传输引擎类型
    //     - ACLSHMEM_DATA_OP_MTE: MTE引擎（片上互联，节点内）
    //     - ACLSHMEM_DATA_OP_ROCE: RDMA引擎（RoCE网络，跨节点）
    //     - ACLSHMEM_DATA_OP_SDMA: SDMA引擎（片上SDMA单元，节点内）
    //     - ACLSHMEM_DATA_OP_UDMA: UDMA引擎（高性能互联）
    //   .timeout: 各阶段超时设置（秒）
    // - instance_id: 多实例模式下的实例编号（可选）
    //   允许同一PE参与多个不同的通信组
    // - comm_args: 通信参数指针（用于传递uniqueid等）
    aclshmemx_init_attr_t attributes;

    // test_set_attr: 辅助函数，填充shmem初始化属性结构体
    // 参数详解:
    // - pe_id: 当前PE编号（进程ID），用于标识本进程在通信组中的位置
    // - n_pes: 总PE数量（进程总数），用于计算数据分布和通信范围
    // - local_mem_size: 对称内存大小（字节），所有PE分配相同大小
    // - ipport: rendezvous地址字符串，格式如"tcp://127.0.0.1:8998"
    //   PE 0作为server监听端口，其他PE作为client连接
    // - default_flag_uid: uniqueid结构体（DEFAULT模式下使用）
    //   用于传递bootstrap握手信息
    // - &attributes: 属性结构体指针（输出参数）
    // 设置完成后attributes包含完整的初始化配置
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    // aclshmemx_init_attr: 初始化shmem运行时
    // 参数详解:
    // - ACLSHMEMX_INIT_WITH_DEFAULT: 初始化模式标志
    //   表示使用默认socket/bootstrap模式，不需要MPI
    //   可选模式:
    //   * ACLSHMEMX_INIT_WITH_DEFAULT: TCP socket模式（推荐，最常用）
    //     PE 0监听ip_port端口，其他PE连接进行握手
    //     不依赖MPI，适合纯shmem应用
    //   * ACLSHMEMX_INIT_WITH_MPI: 使用MPI进行初始化
    //     利用MPI的进程协调机制，适合MPI+shmem混合应用
    //   * ACLSHMEMX_INIT_WITH_UNIQUEID: 使用唯一ID模式
    //     PE 0生成uniqueid并广播给其他PE
    //     支持子组通信和多实例场景
    // - &attributes: 初始化属性结构体指针
    // 返回值: ACLSHMEM_SUCCESS表示成功，否则返回错误码
    // 执行后完成:
    // 1. 建立进程间通信通道（TCP socket连接）
    // 2. 分配对称内存堆（Symmetric Heap）
    //    所有PE在同一虚拟地址分配相同大小的内存
    // 3. 初始化通信引擎（根据option_attr.data_op_engine_type）
    // 4. 设置PE编号和通信组信息（my_pe, n_pes）
    // 5. 创建内部同步机制（barrier、quiet、handle等）
    // 注意: 必须在所有PE上都调用此函数，否则会造成初始化阻塞
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    if (std::string(data_type) == "int") {
        status = test_aclshmem_all_gather<int>(pe_id, n_pes);
    } else if (std::string(data_type) == "int32_t") {
        status = test_aclshmem_all_gather<int32_t>(pe_id, n_pes);
    } else if (std::string(data_type) == "float16_t") {
        status = test_aclshmem_all_gather<fp16_t>(pe_id, n_pes);
    } else if (std::string(data_type) == "bfloat16_t") {
        status = test_aclshmem_all_gather<bfloat16>(pe_id, n_pes);
    }
    // aclshmem_finalize: 终止shmem运行时，释放所有shmem资源
    // 功能详解：
    // - 释放对称内存堆（Symmetric Heap）
    // - 关闭进程间通信通道（TCP socket连接）
    // - 清理通信引擎状态
    // - 释放内部同步机制资源
    // 执行流程：
    // 1. 等待所有pending的RMA操作完成
    // 2. 通知其他PE本PE即将退出
    // 3. 释放所有对称内存资源
    // 4. 关闭bootstrap通信通道
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 注意：
    // 1. 每个PE必须调用此函数后才能退出程序
    // 2. 所有PE应同时调用此函数，避免资源泄露
    // 3. 调用后不能再执行任何shmem操作
    status = aclshmem_finalize();

    // aclrtResetDevice: 重置NPU设备状态
    // 参数: device_id - 要重置的NPU设备编号
    // 执行效果：
    // - 清除设备上下文
    // - 释放设备上的计算资源
    // - 不影响其他进程对该设备的使用
    status = aclrtResetDevice(device_id);

    // aclFinalize: 终止ACL运行时环境
    // 执行效果：
    // - 释放ACL内部资源
    // - 关闭驱动连接
    // - 清理CANN软件栈状态
    // 返回值: ACL_SUCCESS表示成功
    // 注意：必须在所有ACL操作完成后调用
    status = aclFinalize();
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    return 0;
}