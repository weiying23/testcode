/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include <cstring>
#include <cerrno>
#include <algorithm>
#include <random>
#include <cmath>
#include <type_traits>

#include "acl/acl.h"
#include "kernel_operator.h"

#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

#include "utils.h"
#include "param.h"
#include "shmem.h"

#define CHECK_RET(func)                                                                                                \
    do {                                                                                                               \
        int ret = func;                                                                                                \
        if (ret != 0) {                                                                                                \
            std::cerr << __FILE__ << ":" << __LINE__ << " error: " << ret << std::endl;                                \
            return ret;                                                                                                \
        }                                                                                                              \
    } while (0)


int g_npus = 8;
const char *ipport = "tcp://127.0.0.1:8998";
int f_pe = 0;
int f_npu = 0;
const char *data_type = "int";
static char g_ipport[ACLSHMEM_MAX_IP_PORT_LEN] = {0};
aclshmemx_uniqueid_t default_flag_uid;

constexpr float FLOAT_EPS = 1e-5f;
constexpr double DOUBLE_EPS = 1e-8;
constexpr int INT_EPS = 0;

template <typename T>
bool check_accuracy(T actual, T expected)
{
    if constexpr (std::is_integral_v<T>) {
        return actual == expected;
    } else if constexpr (std::is_same_v<T, half>) {
        return std::fabs(actual - expected) < FLOAT_EPS;
    } else if constexpr (std::is_same_v<T, double>) {
        return std::fabs(actual - expected) < DOUBLE_EPS;
    } else {
        return actual == expected;
    }
}

template <typename T>
__global__ __aicore__ void allgather_sdma(GM_ADDR gva, int elem_size, GM_ADDR dump, bool is_put)
{
    AscendC::TPipe pipe;
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#endif
    if ASCEND_IS_AIV {
        // aclshmem_my_pe(): 获取当前PE编号（在Kernel内调用）
        // 返回当前进程在通信组中的编号，范围 [0, n_pes-1]
        // 用于确定本PE的数据位置和通信目标
        int my_pe = aclshmem_my_pe();

        // aclshmem_n_pes(): 获取通信组中的总PE数量
        // 返回参与通信的进程总数
        // 用于计算数据分布和循环范围
        int n_pes = aclshmem_n_pes();

        // Define temporary UB buffer for SDMA operations
        constexpr uint32_t ub_offset = 1024;
        constexpr uint32_t ub_size = 64;  // 64B for temporary buffer
        __ubuf__ uint8_t *tmp_buff = reinterpret_cast<__ubuf__ uint8_t *>(uint64_t(ub_offset));

        uint32_t data_length = elem_size * sizeof(T);
        // allgather
        const auto cur_block_idx = AscendC::GetBlockIdx();
        const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint64_t base_per_core = data_length / comm_block_dim;
        uint64_t extra_bytes = data_length % comm_block_dim;
        uint64_t data_offset = 0;
        if (cur_block_idx < extra_bytes) {
            data_offset = cur_block_idx * (base_per_core + 1);
        } else {
            data_offset = extra_bytes * (base_per_core + 1) +
                                    (cur_block_idx - extra_bytes) * base_per_core;
        }
        if (cur_block_idx < extra_bytes) {
            base_per_core += 1;
        }
        if (base_per_core == 0) {
            return;
        }
        for (int i = 0; i < n_pes; i++) {
            // 跳过自己，不需要向自己发送数据
            if (i == my_pe) {
                continue;
            }
            if (is_put) {
                // aclshmemx_sdma_put_nbi: 非阻塞SDMA Put操作（发送数据到目标PE）
                // SDMA引擎使用片上SDMA单元进行节点内NPU间通信
                // 参数详解:
                // - gva + data_length * my_pe + data_offset: 目标地址（目标PE的对称内存地址，GVA格式）
                //   GVA = Global Virtual Address，所有PE看到的相同虚拟地址
                // - gva + data_length * my_pe + data_offset: 源地址（本PE的对称内存地址）
                // - tmp_buff: UB缓冲区地址（用于SDMA引擎中转数据）
                //   UB = Unified Buffer，AI Core内部的临时存储区
                // - ub_size: UB缓冲区大小（字节），通常为64B或更小
                // - base_per_core: 要传输的数据长度（字节）
                // - i: 目标PE编号（接收数据的PE）
                // - EVENT_ID0: 事件ID（用于硬件同步）
                //   用于跟踪SDMA操作的完成状态
                // 执行流程:
                // 1. SDMA引擎从源地址读取数据到UB缓冲区
                // 2. 通过片上互联将数据发送到目标PE
                // 3. 目标PE的SDMA引擎接收数据并写入目标地址
                // 非阻塞特性: 函数立即返回，不等待传输完成
                aclshmemx_sdma_put_nbi(gva + data_length * my_pe + data_offset, gva + data_length * my_pe + data_offset,
                    tmp_buff, ub_size, base_per_core, i, EVENT_ID0);
            } else {
                // aclshmemx_sdma_get_nbi: 非阻塞SDMA Get操作（从目标PE拉取数据）
                // 参数详解:
                // - gva + data_length * i + data_offset: 目标地址（本PE的接收地址，GVA格式）
                // - gva + data_length * i + data_offset: 源地址（目标PE的发送地址）
                // - tmp_buff: UB缓冲区地址
                // - ub_size: UB缓冲区大小
                // - base_per_core: 数据长度
                // - i: 目标PE编号（数据来源PE）
                // - EVENT_ID0: 事件ID
                // 执行流程:
                // 1. 本PE的SDMA引擎向目标PE发起读取请求
                // 2. 目标PE通过片上互联发送数据
                // 3. 本PE的SDMA引擎接收数据并写入目标地址
                aclshmemx_sdma_get_nbi(gva + data_length * i + data_offset, gva + data_length * i + data_offset,
                    tmp_buff, ub_size, base_per_core, i, EVENT_ID0);
            }
        }
        // aclshmemx_sdma_quiet: 等待所有SDMA操作完成
        // 参数:
        // - tmp_buff: UB缓冲区地址
        // - ub_size: UB缓冲区大小
        // - EVENT_ID0: 事件ID（与put/get操作使用相同ID）
        // 执行效果:
        // - 阻塞直到所有之前发起的sdma_put_nbi/sdma_get_nbi操作完成
        // - 确保数据已完全传输到目标地址
        // - 相当于同步屏障，保证数据一致性
        // 必须在put/get操作后调用，否则数据可能未完全传输
        aclshmemx_sdma_quiet(tmp_buff, ub_size, EVENT_ID0);
    }
}

template <typename T>
__global__ __aicore__ void allgather_sdma_tensor(GM_ADDR gva, int elem_size, GM_ADDR dump, bool is_put)
{
    AscendC::TPipe pipe;
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#endif
    if ASCEND_IS_AIV {
        int my_pe = aclshmem_my_pe();
        int n_pes = aclshmem_n_pes();

        // Define temporary UB buffer as LocalTensor for SDMA operations
        constexpr uint32_t ub_offset = 1024;
        constexpr uint32_t ub_size = 64;  // 64B for temporary buffer
        AscendC::LocalTensor<T> tmp_local;
        tmp_local.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        tmp_local.address_.bufferAddr = ub_offset;
        tmp_local.address_.dataLen = ub_size;

        const auto cur_block_idx = AscendC::GetBlockIdx();
        const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint64_t base_per_core = elem_size / comm_block_dim;
        uint64_t extra_size = elem_size % comm_block_dim;
        uint64_t data_offset = 0;
        if (cur_block_idx < extra_size) {
            data_offset = cur_block_idx * (base_per_core + 1);
        } else {
            data_offset = extra_size * (base_per_core + 1) +
                                    (cur_block_idx - extra_size) * base_per_core;
        }
        if (cur_block_idx < extra_size) {
            base_per_core += 1;
        }
        if (base_per_core == 0) {
            return;
        }
        for (int i = 0; i < n_pes; i++) {
            if (i == my_pe) {
                continue;
            }
            AscendC::GlobalTensor<T> src_tensor;
            AscendC::GlobalTensor<T> dst_tensor;

            if (is_put) {
                __gm__ T* data_addr =
                    reinterpret_cast<__gm__ T*>(gva + my_pe * elem_size * sizeof(T) + data_offset * sizeof(T));
                src_tensor.SetGlobalBuffer(data_addr, base_per_core);
                dst_tensor.SetGlobalBuffer(data_addr, base_per_core);
                aclshmemx_sdma_put_nbi(dst_tensor, src_tensor, tmp_local, base_per_core, i, EVENT_ID0);
            } else {
                __gm__ T* data_addr =
                    reinterpret_cast<__gm__ T*>(gva + i * elem_size * sizeof(T) + data_offset * sizeof(T));
                src_tensor.SetGlobalBuffer(data_addr, base_per_core);
                dst_tensor.SetGlobalBuffer(data_addr, base_per_core);
                aclshmemx_sdma_get_nbi(dst_tensor, src_tensor, tmp_local, base_per_core, i, EVENT_ID0);
            }
        }
        aclshmemx_sdma_quiet(tmp_local, EVENT_ID0);
    }
}

template <class T>
void allgather_kernel(uint32_t block_dim, void *stream, uint8_t *gva, int n_elements, uint8_t *device_dump,
    bool test_tensor_mode, bool is_put)
{
    if (!test_tensor_mode) {
        allgather_sdma<T><<<block_dim, nullptr, stream>>>(gva, n_elements, device_dump, is_put);
    } else {
        allgather_sdma_tensor<T><<<block_dim, nullptr, stream>>>(gva, n_elements, device_dump, is_put);
    }
}

int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port,
                      aclshmemx_init_attr_t *attributes)
{
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), sizeof(g_ipport) - 1);

        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, 
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(&default_flag_uid);
    aclshmemx_uniqueid_t *uid_args = (aclshmemx_uniqueid_t *)(attributes->comm_args);
    uid_args->my_pe = my_pe;
    uid_args->n_pes = n_pes;
    return ACLSHMEM_SUCCESS;
}

template <class T>
int test_allgather_sdma(int my_pe, int n_pes)
{
    // ACLStream init
    aclrtStream stream = nullptr;
    CHECK_RET(aclrtCreateStream(&stream));

    constexpr uint32_t n_blocks = 20;
    constexpr int num10 = 10;

    uint8_t *device_dump = nullptr;
#if defined(ENABLE_ASCENDC_DUMP)
    CHECK_RET(aclrtMalloc(reinterpret_cast<void **>(&device_dump), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
#endif

    // aclshmem_malloc: 分配对称内存（Symmetric Heap）
    // 参数: (128 * 1024 * 1024) * sizeof(T) - 约128MB的对称内存
    // 返回值: 对称内存指针（GVA格式）
    // 对称内存核心特点：
    // - 所有PE在同一虚拟地址上拥有相同大小的内存块
    // - PE i可以直接通过GVA地址访问PE j的数据
    // - 用于存放通信数据和同步标志
    // - 必须通过aclshmem_free释放，不能使用aclrtFree
    // GVA = Global Virtual Address，全局虚拟地址
    // 示例：如果PE 0在地址ptr存放数据X，PE 1可以通过相同的ptr地址读取到X
    void *gva = aclshmem_malloc((128 * 1024 * 1024) * sizeof(T));

    // 初始化数据
    size_t trans_size = 16 * 1024 * 1024;
    std::vector<T> input(trans_size, 0);
    for (size_t i = 0; i < trans_size; i++) {
        input[i] = (T)(my_pe + num10);
    }

    CHECK_RET(aclrtMemcpy(reinterpret_cast<uint8_t *>(gva) + aclshmem_my_pe() * trans_size * sizeof(T),
        trans_size * sizeof(T), input.data(), trans_size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE));

    allgather_kernel<T>(n_blocks, stream, reinterpret_cast<uint8_t *>(gva), trans_size, device_dump, false, true);

    CHECK_RET(aclrtSynchronizeStream(stream));

    // aclshmem_barrier_all: 全局屏障同步
    // 功能：所有PE都调用此函数后才能继续执行
    // 执行流程：
    // 1. 当前PE到达屏障，标记自己已完成
    // 2. 等待所有其他PE也到达屏障
    // 3. 所有PE都到达后，一起释放继续执行
    // 用途：
    // - 确保所有PE的通信操作都已完成
    // - 用于结果验证前的同步
    // - 保证数据一致性
    // 注意：必须所有PE都调用此函数，否则会造成死锁
    aclshmem_barrier_all();

#if defined(ENABLE_ASCENDC_DUMP)
    Adx::AdumpPrintWorkSpace(device_dump, ALL_DUMPSIZE, stream, "test");
#endif

    // 结果校验
    T* y_host;
    size_t input_size = n_pes * trans_size * sizeof(T);
    CHECK_RET(aclrtMallocHost(reinterpret_cast<void**>(&y_host), input_size));
    CHECK_RET(aclrtMemcpy(y_host, input_size, gva, input_size, ACL_MEMCPY_DEVICE_TO_HOST));

    const int check_step = 1; // 数据校验的步长
    for (int i = 0; i < n_pes; i++) {
        for (int j = 0; j < trans_size; j+= check_step) {
            int y = (int)(y_host[trans_size * i + j]);
            if (y != i + num10) {
                printf("ERROR in pe%d:%d %d != %d\n", i, j, y, i + num10);
                break;
            }
        }
    }

    CHECK_RET(aclrtFreeHost(y_host));

    // aclshmem_free: 释放对称内存
    // 参数: aclshmem_malloc返回的对称内存指针
    // 必须与aclshmem_malloc配对使用
    // 注意：不能使用aclrtFree释放对称内存，必须使用aclshmem_free
    // 释放后，该内存块可以被其他shmem操作重新分配
    aclshmem_free(gva);

    std::cout << " Pe " << my_pe << "Finised !! Result Correct !!" << std::endl;

    CHECK_RET(aclrtDestroyStream(stream));
    return 0;
}

int main(int argc, char *argv[])
{
    int status = 0;
    int n_pes = atoi(argv[INDEX1]);
    int my_pe = atoi(argv[INDEX2]);
    ipport = argv[INDEX3];
    g_npus = atoi(argv[INDEX4]);
    f_pe = atoi(argv[INDEX5]);
    f_npu = atoi(argv[INDEX6]);
    data_type = argv[INDEX7];

    // ========== ACL && Shmem 初始化 ==========
    // 计算物理设备ID：my_pe % g_npus + f_npu
    // my_pe: 当前进程的逻辑编号
    // g_npus: 节点内NPU总数
    // f_npu: NPU编号偏移量（物理设备ID的起点）
    int32_t device_id = my_pe % g_npus + f_npu;

    // aclInit: 初始化ACL（Ascend Computing Language）运行时环境
    // 参数: nullptr表示使用默认配置
    // 必须在调用任何ACL API之前执行
    // 初始化CANN软件栈，加载驱动，准备NPU资源
    CHECK_RET(aclInit(nullptr));

    // aclrtSetDevice: 设置当前进程使用的NPU设备
    // 参数: device_id - 物理NPU设备编号
    // 将进程绑定到指定的NPU，后续所有ACL操作在该设备上执行
    // 设置设备上下文，准备计算资源
    CHECK_RET(aclrtSetDevice(device_id));

    // 定义对称内存大小：1GB
    // 对称内存是所有PE在同一虚拟地址上的相同大小内存
    // 用于存放通信数据和同步标志
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;

    // aclshmemx_init_attr_t: shmem初始化属性结构体
    // 包含以下关键字段：
    // - my_pe: 当前PE编号
    // - n_pes: 总PE数量
    // - ip_port: rendezvous地址（TCP socket地址）
    // - local_mem_size: 对称内存大小
    // - option_attr: 可选属性（引擎类型、超时等）
    //   .data_op_engine_type: 数据传输引擎类型
    //   .timeout: 各阶段超时设置
    aclshmemx_init_attr_t attributes;

    // test_set_attr: 辅助函数，填充shmem初始化属性结构体
    // 参数详解:
    // - my_pe: 当前PE编号（进程ID）
    // - n_pes: 总PE数量（进程总数）
    // - local_mem_size: 对称内存大小（字节）
    // - ipport: rendezvous地址字符串，如"tcp://127.0.0.1:8998"
    //   PE 0监听此地址，其他PE连接到此地址进行握手
    // - &attributes: 属性结构体指针（输出参数）
    CHECK_RET(test_set_attr(my_pe, n_pes, local_mem_size, ipport, &attributes));

    // ACLSHMEM_DATA_OP_SDMA: 设置数据传输引擎类型为SDMA
    // SDMA引擎特点：
    // - 使用片上SDMA单元进行数据传输
    // - 仅支持节点内NPU间通信（不支持跨节点）
    // - 高带宽、低延迟
    // - 适合大规模数据传输
    // 其他可选引擎类型:
    // - ACLSHMEM_DATA_OP_MTE: MTE引擎（片上互联，节点内）
    // - ACLSHMEM_DATA_OP_ROCE: RDMA引擎（RoCE网络，跨节点）
    // - ACLSHMEM_DATA_OP_UDMA: UDMA引擎（高性能互联）
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;

    // aclshmemx_init_attr: 初始化shmem运行时
    // 参数详解:
    // - ACLSHMEMX_INIT_WITH_DEFAULT: 初始化模式标志
    //   表示使用默认socket/bootstrap模式，不需要MPI
    //   可选模式:
    //   * ACLSHMEMX_INIT_WITH_DEFAULT: TCP socket模式（推荐）
    //   * ACLSHMEMX_INIT_WITH_MPI: 使用MPI进行初始化
    //   * ACLSHMEMX_INIT_WITH_UNIQUEID: 使用唯一ID模式
    // - &attributes: 初始化属性结构体指针
    // 返回值: ACLSHMEM_SUCCESS表示成功，否则返回错误码
    // 执行后完成：
    // 1. 建立进程间通信通道（TCP socket连接）
    // 2. 分配对称内存堆（Symmetric Heap）
    // 3. 初始化SDMA通信引擎
    // 4. 设置PE编号和通信组信息
    // 5. 创建内部同步机制（barrier、quiet等）
    CHECK_RET(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    if (std::string(data_type) == "int") {
        CHECK_RET(test_allgather_sdma<int>(my_pe, n_pes));
    } else if (std::string(data_type) == "uint8") {
        CHECK_RET(test_allgather_sdma<uint8_t>(my_pe, n_pes));
    } else if (std::string(data_type) == "int64") {
        CHECK_RET(test_allgather_sdma<int64_t>(my_pe, n_pes));
    } else if (std::string(data_type) == "fp32") {
        CHECK_RET(test_allgather_sdma<float>(my_pe, n_pes));
    } else {
        printf("ERROR: Unsupport type\n");
        return -1;
    }

    // aclshmem_finalize: 终止shmem运行时，释放所有shmem资源
    // 包括对称内存、通信通道、内部状态等
    CHECK_RET(aclshmem_finalize());
    CHECK_RET(aclrtResetDevice(device_id));
    CHECK_RET(aclFinalize());

    std::cout << "[SUCCESS] demo run success in pe " << my_pe << std::endl;
    return 0;
}