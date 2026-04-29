/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

// aclshmem_host
#include "host/shmem_host_def.h"
#include "host/mem/shmem_host_heap.h"
#include "host/init/shmem_host_init.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/team/shmem_host_team.h"

// utils
#include "utils.h"

#include "catcoc/catcoc.h"
#include "catcoc/comm_epilogue/comm_dispatch_policy.h"
#include "catcoc/comm_epilogue/block/comm_block_epilogue.h"
#include "catcoc/comm_epilogue/block/comm_block_swizzle.h"
#include "catcoc/comm_epilogue/tile/tile_remote_copy.h"
#include "catcoc/detail/remote_copy_type.h"
#include "catcoc/dgemm/kernel/matmul_allreduce.h"

constexpr size_t NPU_MALLOC_SPACE = 1024UL * 1024 * 1024;

constexpr uint32_t BLOCK_NUM = 20;

using namespace AscendC;
using namespace Catcoc;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;

using ElementA = half;
using ElementB = half;
using ElementC = half;
using ElementD = half;

CATLASS_GLOBAL
void ShmemMatmulAllReduce(
    uint64_t fftsAddr,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD, GM_ADDR gmSymmetric,
    uint32_t m, uint32_t n, uint32_t k
)
{
    util_set_ffts_config(fftsAddr);

    using ArchTag = Catlass::Arch::AtlasA2;

    uint32_t peIdx = aclshmem_my_pe();
    // aclshmem_n_pes(): 获取总PE数量，用于ReduceScatter分块计算
    // 返回值: 总PE数量
    // 在Matmul-AllReduce中的用途：
    // 1. 计算ReduceScatter的数据分块大小（m/peSize）
    // 2. 确定AllGather的数据来源PE数量
    // 3. 计算通信组的数据分布和循环范围
    uint32_t peSize = aclshmem_n_pes();

    Catlass::GemmCoord problemShape{m, n, k};
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutD layoutD{m, n};

    constexpr bool ENABLE_UNIT_FLAG = true;
    constexpr int L1TILEM = 128;
    constexpr int L1TILEN = 256;
    constexpr int L1TILEK = 256;
    constexpr int L0TILEM = 128;
    constexpr int L0TILEN = 256;
    constexpr int L0TILEK = 64;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;
    using L1TileShape = Catlass::GemmShape<L1TILEM, L1TILEN, L1TILEK>;
    using L0TileShape = Catlass::GemmShape<L0TILEM, L0TILEN, L0TILEK>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType
    >;

    constexpr uint32_t SWIZZLE_GROUP_SIZE = 7;
    constexpr uint32_t SWIZZLE_DIRECTION = 1;
    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<SWIZZLE_GROUP_SIZE, SWIZZLE_DIRECTION>;
    using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0>;

    using RemoteSrcType = CType;
    using RemoteDstType = DType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Get>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    constexpr uint32_t COMM_BLOCK_ROWS = 64;
    constexpr uint32_t COMM_BLOCK_COLUMNS = 256;
    constexpr uint32_t CORE_SPLIT_ROWS = 20;
    constexpr uint32_t CORE_SPLIT_COLUMNS = 1;
    using CommBlockShape = Catlass::MatrixShape<COMM_BLOCK_ROWS, COMM_BLOCK_COLUMNS>;
    using CommCoreSplit = Catlass::MatrixShape<CORE_SPLIT_ROWS, CORE_SPLIT_COLUMNS>;

    constexpr uint32_t UB_STAGES = 2;
    constexpr uint32_t SCATTER_TILE_ROWS = 32;
    constexpr uint32_t SCATTER_TILE_COLUMNS = 256;
    using EpilogueReduceScatterTileShape = Catlass::MatrixShape<SCATTER_TILE_ROWS, SCATTER_TILE_COLUMNS>;
    using EpilogueReduceScatterDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
        Catcoc::detail::CopyMode::Scatter>;
    using BlockEpilogueReduceScatter = CommEpilogue::Block::CommBlockEpilogue<
        EpilogueReduceScatterDispatch,
        RemoteSrcType, RemoteDstType,
        CommCoreSplit,
        CommBlockShape,
        EpilogueReduceScatterTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t ALLGATHER_TILE_ROWS = 32;
    constexpr uint32_t ALLGATHER_TILE_COLUMNS = 256;
    using EpilogueAllGatherTileShape = Catlass::MatrixShape<ALLGATHER_TILE_ROWS, ALLGATHER_TILE_COLUMNS>;
    using EpilogueAllGatherDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
        Catcoc::detail::CopyMode::Gather>;
    using BlockEpilogueAllGather = CommEpilogue::Block::CommBlockEpilogue<
        EpilogueAllGatherDispatch,
        RemoteSrcType, RemoteDstType,
        CommCoreSplit,
        CommBlockShape,
        EpilogueAllGatherTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    using MatmulAllReduceKernel = DGemm::Kernel::MatmulAllReduce<
        BlockMmad,
        BlockEpilogueReduceScatter,
        BlockEpilogueAllGather,
        BlockMmadScheduler,
        BlockEpilogueScheduler,
        WORKSPACE_STAGES
    >;

    typename BlockEpilogueReduceScatter::Params reduceScatterParams{};
    typename BlockEpilogueAllGather::Params allGatherParams{};

    typename MatmulAllReduceKernel::Params params{
        problemShape, peIdx, peSize,
        COMM_INTERVAL,
        gmA, layoutA,
        gmB, layoutB,
        gmD, layoutD,
        gmSymmetric,
        reduceScatterParams,
        allGatherParams
    };

    MatmulAllReduceKernel matmulAllReduceKernel;
    matmulAllReduceKernel(params);
}

struct Options {
    static constexpr auto HELPER =
       "Usage: matmul_allreduce pe_size pe_id ip_port m n k [device_id_list]\n";

    int peSize;
    int peId;
    std::string ipPort;
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::string dataPath;
    std::vector<int> deviceIdList{};

    int Parse(int argc, char **argv)
    {
        enum ArgsIndex {
            PE_SIZE_INDEX = 1,
            PE_ID_INDEX,
            IP_PORT_INDEX,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            DATA_PATH_INDEX,
            DEVICE_LIST_INDEX,
            INDEX_MAX
        };

        if (argc > INDEX_MAX) {
            printf(HELPER);
            return -1;
        }

        peSize = std::atoi(argv[PE_SIZE_INDEX]);
        peId = std::atoi(argv[PE_ID_INDEX]);
        ipPort = argv[IP_PORT_INDEX];
        m = std::atoi(argv[M_INDEX]);
        n = std::atoi(argv[N_INDEX]);
        k = std::atoi(argv[K_INDEX]);
        dataPath = argv[DATA_PATH_INDEX];
        if (argc > DEVICE_LIST_INDEX) {
            char *idListStr = argv[DEVICE_LIST_INDEX];
            for (char *idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (size_t i = 0; i < peSize; ++i) {
                deviceIdList.push_back(i);
            }
        }
        return 0;
    }

    std::string GetDataPath(std::string const &fileName = "") const
    {
        return dataPath + "/" + fileName;
    }
};

aclshmemx_uniqueid_t default_flag_uid;

int main(int argc, char **argv)
{
    int status = ACLSHMEM_SUCCESS;
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }
    int n_pes = options.peSize;
    int pe_id = options.peId;
    std::string ipPort = options.ipPort;
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;
    int32_t device_id = options.deviceIdList[pe_id];

    std::cout << "[TEST] input pe_size: " << n_pes << " pe_id:" << pe_id << " input_ip: " << ipPort << "\n";

    // Acl && Shmem init
    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
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

    // test_set_attr: 辅助函数，填充shmem初始化属性结构体
    // 参数详解:
    // - pe_id: 当前PE编号
    // - n_pes: 总PE数量
    // - local_mem_size: 对称内存大小（1GB）
    // - ipPort.c_str(): rendezvous地址字符串
    // - default_flag_uid: uniqueid结构体
    // - &attributes: 属性结构体指针（输出参数）
    test_set_attr(pe_id, n_pes, local_mem_size, ipPort.c_str(), default_flag_uid, &attributes);

    // aclshmemx_init_attr: 初始化shmem运行时（默认socket模式）
    // 参数详解:
    // - ACLSHMEMX_INIT_WITH_DEFAULT: 初始化模式标志
    //   使用TCP socket进行进程间rendezvous
    // - &attributes: 初始化属性结构体指针
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // Matmul-AllReduce场景中的作用：
    // 1. 建立所有PE之间的通信通道
    // 2. 分配对称内存堆，用于ReduceScatter/AllGather数据交换
    // 3. 初始化通信引擎，支持ReduceScatter和AllGather集合操作
    // 4. 设置pe_id和n_pes信息
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);


    // ACLStream init
    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);

    size_t aSize = static_cast<size_t>(m) * k * sizeof(__fp16);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(__fp16);
    size_t dSize = static_cast<size_t>(m) * n * sizeof(__fp16);

    uint8_t *aDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *aHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&aHost), aSize));
    ReadFile(options.GetDataPath("pe_" + std::to_string(pe_id) + "_a.bin"), aHost, aSize);
    ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *bHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&bHost), bSize));
    ReadFile(options.GetDataPath("pe_" + std::to_string(pe_id) + "_b.bin"), bHost, bSize);
    ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *dDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&dDevice), dSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *dHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&dHost), dSize));
    aclrtMemset(dHost, dSize, 0, dSize);  // 零初始化 C 矩阵
    ACL_CHECK(aclrtMemcpy(dDevice, dSize, dHost, dSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // aclshmem_malloc: 分配对称内存（用于ReduceScatter/AllGather通信缓冲区）
    // 参数详解:
    // - (204 * 1024 * 1024) * sizeof(__fp16): 对称内存大小
    //   约204MB * 2字节 = ~408MB的对称内存
    // 返回值: 对称内存指针（GVA格式）
    // Matmul-AllReduce场景中对称内存的用途：
    // - 存存ReduceScatter操作的中间结果
    //   每个PE将矩阵乘法结果按行分块scatter到对应PE
    // - 存存AllGather操作的数据缓冲区
    //   每个PE从其他PEgather部分结果
    // - 用于ReduceScatter-AllGather融合通信的数据交换
    // Matmul-AllReduce执行流程：
    // 1. 各PE独立执行本地矩阵乘法：C = A × B
    // 2. 执行ReduceScatter：每个PE将结果矩阵按行分块
    //    PE i负责第i行的数据和reduce
    // 3. 执行AllGather：每个PE将reduce后的结果广播给所有PE
    // 对称内存核心特点：
    // 1. 所有PE在同一虚拟地址上拥有相同大小的内存块
    // 2. PE i可以直接通过GVA地址访问PE j的数据
    // 3. 用于存放通信数据和同步标志
    // 注意：
    // - 必须通过aclshmem_free释放
    // - 分配大小不能超过local_mem_size
    void *symmPtr = aclshmem_malloc((204 * 1024 * 1024) * sizeof(__fp16));
    uint8_t *symmetricPtr = reinterpret_cast<uint8_t *>(symmPtr);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "Before calling MM_AR kernel " << std::endl;
    for (int i = 0; i < 1; i++) {
        uint64_t fftsAddr = util_get_ffts_config();
        ShmemMatmulAllReduce<<<BLOCK_NUM, nullptr, stream>>>(
            fftsAddr,
            aDevice, bDevice, dDevice, symmetricPtr,
            m, n, k
        );
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "After calling MM_AR kernel " << std::endl;

    // aclshmemx_show_prof: 显示性能统计信息（可选调试功能）
    // 参数详解:
    // - nullptr: 默认输出位置
    // - true: 启用性能统计输出
    // 执行效果:
    // - 输出ReduceScatter和AllGather的性能统计
    // - 显示数据传输时间、带宽等信息
    // - 用于性能分析和优化调试
    aclshmemx_show_prof(nullptr, true);

    if (pe_id == 0) {
        ACL_CHECK(aclrtMemcpy(dHost, dSize, dDevice, dSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(options.GetDataPath("aclshmem_output.bin"), dHost, dSize);
        std::printf("test finished\n");
    }

    // aclshmem_free: 释放对称内存
    // 参数: aclshmem_malloc返回的对称内存指针（symmPtr）
    // 必须与aclshmem_malloc配对使用
    // 执行效果:
    // - 将对称内存归还到Symmetric Heap
    // - 其他shmem操作可以重新分配此内存
    // - 释放后该地址不再可用于通信
    // 重要提示：
    // 1. 不能使用aclrtFree释放对称内存
    // 2. 所有PE应同时释放对称内存
    // 3. 释放前确保ReduceScatter和AllGather操作已完成
    aclshmem_free(symmPtr);

    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(dHost));
    ACL_CHECK(aclrtFree(aDevice));
    ACL_CHECK(aclrtFree(bDevice));
    ACL_CHECK(aclrtFree(dDevice));

    status = aclrtDestroyStream(stream);

    // aclshmem_finalize: 终止shmem运行时，释放所有shmem资源
    // 功能详解：
    // - 释放对称内存堆
    // - 关闭进程间通信通道
    // - 清理通信引擎状态
    // - 释放内部同步机制资源
    // 执行流程：
    // 1. 等待所有pending的ReduceScatter/AllGather操作完成
    // 2. 通知其他PE本PE即将退出
    // 3. 释放所有对称内存资源
    // 4. 关闭bootstrap通信通道
    // 返回值: ACLSHMEM_SUCCESS表示成功
    // 注意：
    // 1. 每个PE必须调用此函数后才能退出程序
    // 2. 所有PE应同时调用此函数
    // 3. 调用后不能再执行任何shmem操作
    status = aclshmem_finalize();
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    return 0;
}
