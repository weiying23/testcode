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
#include <cstdio>
#include <cstring>
#include <algorithm>

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
extern aclshmem_host_state_t g_state_host;

template <typename T>
__global__ __aicore__ void allgather_sdma(GM_ADDR gva, int elem_size, GM_ADDR dump, bool is_put)
{
    AscendC::TPipe pipe;
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#endif
    if ASCEND_IS_AIV {
        int my_pe = aclshmem_my_pe();
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
            if (i == my_pe) {
                continue;
            }
            if (is_put) {
                // aclshmemx_sdma_put_nbi: 非阻塞SDMA Put操作
                // 将数据从本PE发送到目标PE
                aclshmemx_sdma_put_nbi(gva + data_length * my_pe + data_offset, gva + data_length * my_pe + data_offset,
                    tmp_buff, ub_size, base_per_core, i, EVENT_ID0);
            } else {
                // aclshmemx_sdma_get_nbi: 非阻塞SDMA Get操作
                // 从目标PE拉取数据到本PE
                aclshmemx_sdma_get_nbi(gva + data_length * i + data_offset, gva + data_length * i + data_offset,
                    tmp_buff, ub_size, base_per_core, i, EVENT_ID0);
            }
        }
        // aclshmemx_sdma_notify_record: 记录SDMA完成通知
        // 用于通知其他PE数据传输已完成
        // 参数详解:
        // - tmp_buff: UB缓冲区地址（用于通知记录）
        // - ub_size: UB缓冲区大小
        // - EVENT_ID0: 事件ID（与put/get操作使用相同ID）
        // 执行效果:
        // - 在指定事件上记录完成通知
        // - 其他PE可以通过aclrtWaitAndResetNotify等待此通知
        // - 用于异步SDMA操作的同步
        // 使用场景：
        // - 与aclrtWaitAndResetNotify配合使用进行同步
        // - 替代aclshmemx_sdma_quiet，提供更细粒度的同步控制
        // - 适合需要精确控制同步时机的场景
        aclshmemx_sdma_notify_record(tmp_buff, ub_size, EVENT_ID0);
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
        aclshmemx_sdma_notify_record(tmp_local, EVENT_ID0);
    }
}

template <typename T>
__global__ __aicore__ void device_copy(GM_ADDR src, GM_ADDR dst, int message_length)
{
    // aclshmemi_get_state(): 获取shmem运行时状态（在Kernel内调用）
    // 返回设备端状态结构体，包含UB地址、同步ID等配置信息
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();

    // 从状态结构体获取UB缓冲区地址和大小
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    int64_t my_pe = aclshmem_my_pe();
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;
    // aclshmemx_mte_put_nbi: 非阻塞MTE Put操作
    // 参数:
    //   - dst: 目标地址（目标PE的GVA地址）
    //   - src: 源地址（本PE的GVA地址）
    //   - copy_ub: UB缓冲区地址（用于MTE引擎中转）
    //   - copy_ub_size: UB缓冲区大小
    //   - message_length: 数据长度（字节）
    //   - my_pe: 目标PE编号（此处为自身，用于本地拷贝测试）
    //   - copy_event_id: 同步事件ID
    // MTE引擎使用片上MTE单元进行数据传输
    aclshmemx_mte_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub),
                          copy_ub_size, message_length, my_pe, copy_event_id);
    // aclshmem_quiet: 等待所有shmem通信操作完成
    // 阻塞直到所有之前发起的put/get操作完成
    aclshmem_quiet();
}

template <class T>
void copy_demo(uint32_t block_dim, void* stream, uint8_t* src, uint8_t* dst, int elements)
{
    device_copy<T><<<block_dim, nullptr, stream>>>(src, dst, elements);
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

    constexpr uint32_t total_block_num = 20;
    constexpr int num10 = 10;
    constexpr int sub_block_num = 2;
    uint8_t *device_dump = nullptr;
#if defined(ENABLE_ASCENDC_DUMP)
    CHECK_RET(aclrtMalloc(reinterpret_cast<void **>(&device_dump), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
#endif

    // aclshmem_malloc: 分配对称内存
    // 用于存放AllGather操作的通信数据
    // 参数详解:
    // - (128 * 1024 * 1024) * sizeof(T): 对称内存大小
    //   约128MB * sizeof(T)的对称内存
    // 返回值: 对称内存指针（GVA格式）
    // 对称内存用途：
    // - 存存AllGather操作的数据
    // - 存存其他PE的数据分片
    // 对称内存核心特点：
    // 1. 所有PE在同一虚拟地址上拥有相同大小的内存块
    // 2. PE i可以直接通过GVA地址访问PE j的数据
    // 3. 用于存放通信数据和同步标志
    // 注意：
    // - 必须通过aclshmem_free释放
    // - 分配大小不能超过local_mem_size
    void *gva = aclshmem_malloc((128 * 1024 * 1024) * sizeof(T));

    // 初始化数据
    size_t trans_size = 16 * 1024 * 1024;
    std::vector<T> input(trans_size, 0);
    for (size_t i = 0; i < trans_size; i++) {
        input[i] = (T)(my_pe + num10);
    }

    CHECK_RET(aclrtMemcpy(reinterpret_cast<uint8_t *>(gva) + my_pe * trans_size * sizeof(T),
        trans_size * sizeof(T), input.data(), trans_size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE));
    uint8_t *ptr = reinterpret_cast<uint8_t *>(gva);
    uint8_t *ptr_A = ptr + n_pes * trans_size * sizeof(T);
    allgather_kernel<T>(total_block_num, stream, ptr, trans_size, device_dump, false, true);

    for(int i = 0; i < total_block_num * sub_block_num; i++) {
        // aclrtWaitAndResetNotify: 等待并重置通知
        // 参数详解:
        // - g_state_host.notify_arr[i]: 通知ID（硬件通知对象）
        // - g_state_host.default_stream: ACL流
        // - 0: timeout（0表示无限等待）
        // 执行效果:
        // - 阻塞直到收到对应的notify_record通知
        // - 收到通知后自动重置通知状态
        // - 用于等待特定SDMA操作完成
        // 与notify_record配合使用：
        // - notify_record在Kernel中记录完成通知
        // - wait_notify在Host端等待完成通知
        // - 提供细粒度的异步操作同步控制
        CHECK_RET(aclrtWaitAndResetNotify(g_state_host.notify_arr[i], g_state_host.default_stream, 0));
    }
    // aclshmem_barrier_all: 全局屏同步
    // 功能详解：
    // - 所有PE都调用此函数后才能继续执行
    // - 确保所有notify操作都已完成
    // - 用于copy_demo前的同步
    // 注意：必须所有PE都调用此函数
    aclshmem_barrier_all();
    copy_demo<T>(1, g_state_host.default_stream, ptr, ptr_A, n_pes * trans_size * sizeof(T));

    CHECK_RET(aclrtSynchronizeStream(g_state_host.default_stream));

#if defined(ENABLE_ASCENDC_DUMP)
    Adx::AdumpPrintWorkSpace(device_dump, ALL_DUMPSIZE, stream, "test");
#endif

    // 操作结果校验
    T *y_host;
    size_t input_size = n_pes * trans_size * sizeof(T);
    uint32_t pe_id = aclshmem_my_pe();
    // 校验 ptr_A 中的内容
    uint32_t status = aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size);
    status = aclrtMemcpy(y_host, input_size, ptr_A, input_size, ACL_MEMCPY_DEVICE_TO_HOST);
    std::cout << "Pe " << pe_id << " AllGather result in ptr_A after notify_wait:" << std::endl;
    int unexpected_count = 0;
    for (int i = 0; i < n_pes; i++) {
        for (int j = 0; j < trans_size; j++) {
            int y = (int)(y_host[trans_size * i + j]);
            if (y != num10 + i) {
                unexpected_count++;
            }
        }
    }
    std::cout << "Pe " << pe_id << " has " << unexpected_count << " unexpected values." << std::endl;

    CHECK_RET(aclrtFreeHost(y_host));

    // aclshmem_free: 释放对称内存
    // 参数: aclshmem_malloc返回的对称内存指针（gva）
    // 必须与aclshmem_malloc配对使用
    // 执行效果:
    // - 将对称内存归还到Symmetric Heap
    // - 其他shmem操作可以重新分配此内存
    // - 释放后该地址不再可用于通信
    // 重要提示：
    // 1. 不能使用aclrtFree释放对称内存
    // 2. 所有PE应同时释放对称内存
    // 3. 释放前确保所有SDMA操作已完成
    aclshmem_free(gva);

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

    // Acl && Shmem init
    int32_t device_id = my_pe % g_npus + f_npu;
    CHECK_RET(aclInit(nullptr));
    CHECK_RET(aclrtSetDevice(device_id));

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    CHECK_RET(test_set_attr(my_pe, n_pes, local_mem_size, ipport, &attributes));

    // ACLSHMEM_DATA_OP_SDMA: 设置数据传输引擎为SDMA
    // SDMA引擎特点：
    // - 使用片上SDMA单元进行数据传输
    // - 仅支持节点内NPU间通信（不支持跨节点）
    // - 高带宽、低延迟
    // - 适合大规模数据传输
    // - 支持notify_record/wait_notify同步机制
    // 其他可选引擎类型:
    // - ACLSHMEM_DATA_OP_MTE: MTE引擎（片上互联，节点内）
    // - ACLSHMEM_DATA_OP_ROCE: RDMA引擎（RoCE网络，跨节点）
    // - ACLSHMEM_DATA_OP_UDMA: UDMA引擎（高性能互联）
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;

    // aclshmemx_init_attr: 初始化shmem运行时
    // 参数详解:
    // - ACLSHMEMX_INIT_WITH_DEFAULT: 初始化模式标志
    //   使用TCP socket进行进程间rendezvous
    // - &attributes: 初始化属性结构体指针
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 执行后完成:
    // 1. 建立进程间通信通道
    // 2. 分配对称内存堆
    // 3. 初始化SDMA通信引擎
    // 4. 设置PE编号和通信组信息
    // 5. 初始化notify机制（notify_arr等）
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
    // 功能详解：
    // - 释放对称内存堆
    // - 关闭进程间通信通道
    // - 清理SDMA通信引擎状态
    // - 释放notify机制资源（notify_arr等）
    // 执行流程：
    // 1. 等待所有pending的SDMA操作完成
    // 2. 通知其他PE本PE即将退出
    // 3. 释放所有对称内存资源
    // 4. 关闭bootstrap通信通道
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 注意：
    // 1. 每个PE必须调用此函数后才能退出程序
    // 2. 所有PE应同时调用此函数
    // 3. 调用后不能再执行任何shmem操作
    CHECK_RET(aclshmem_finalize());
    CHECK_RET(aclrtResetDevice(device_id));
    CHECK_RET(aclFinalize());

    std::cout << "[SUCCESS] demo run success in pe " << my_pe << std::endl;
    return 0;
}