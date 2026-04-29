/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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

// shmem_host
#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_team.h"

// utils
#include "utils.h"

#include "catcoc/catcoc.h"
#include "catcoc/comm_epilogue/comm_dispatch_policy.h"
#include "catcoc/comm_epilogue/block/comm_block_epilogue.h"
#include "catcoc/comm_epilogue/block/comm_block_swizzle.h"
#include "catcoc/comm_epilogue/tile/tile_remote_copy.h"
#include "catcoc/detail/remote_copy_type.h"
#include "catcoc/dgemm/kernel/matmul_reduce_scatter.h"

using namespace AscendC;
using namespace Catcoc;

constexpr size_t NPU_MALLOC_SPACE = 1024UL * 1024 * 1024;

constexpr uint32_t BLOCK_NUM = 20;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;

using ElementA = half;
using ElementB = half;
using ElementC = half;
using ElementD = half;

CATLASS_GLOBAL
void ShmemMatmulReduceScatter(
    uint64_t fftsAddr,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD, GM_ADDR gmSymmetric,
    uint32_t m, uint32_t n, uint32_t k
)
{
    // shmemx_set_ffts_config(): 设置FFTS（Fast Flag Task Sync）配置地址，用于核间快速同步通信
    shmemx_set_ffts_config(fftsAddr);

    using ArchTag = Catlass::Arch::AtlasA2;

    // shmem_my_pe(): 获取当前PE（Processing Element）的编号，返回当前进程在通信组中的唯一标识（rank ID）
    uint32_t rankIdx = shmem_my_pe();
    // shmem_n_pes(): 获取通信组中总PE数量，返回参与通信的所有进程总数（rank总数）
    // 用于确定ReduceScatter操作的参与者数量和每个rank分得的数据大小
    // 返回值: 通信组中rank的总数
    uint32_t rankSize = shmem_n_pes();

    Catlass::GemmCoord problemShape{m, n, k};
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    if (rankSize == 0) {
        return;
    }
    LayoutD layoutD{m / rankSize, n};

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
    // GemmIdentityBlockSwizzle: Matmul专用的块调度器
    // 参数: SWIZZLE_GROUP_SIZE=7 - 调度组大小，定义每个调度组包含7个block
    // 参数: SWIZZLE_DIRECTION=1 - 调度方向，定义block的调度顺序方向
    // 该调度器负责将矩阵乘法计算块分配到不同的AIV核心上执行
    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<SWIZZLE_GROUP_SIZE, SWIZZLE_DIRECTION>;
    // BlockCommSwizzle: 通信块调度器，负责分配通信操作到不同的AIV核心上执行
    // 参数0表示通信块调度偏移量为0，参数true表示启用特殊调度模式
    // 该调度器确保ReduceScatter通信操作均匀分配到各核心，避免通信负载不均衡
    using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0, true>;

    // RemoteSrcType: 远程数据源类型，指定从远端rank读取数据的类型为CType(half)
    // 用于shmem get操作时确定远端数据的类型和布局
    // 在ReduceScatter操作中，从各rank的对称内存读取部分矩阵乘法结果进行规约
    using RemoteSrcType = CType;
    // RemoteDstType: 远程数据目标类型，指定写入到本地或远端rank的数据类型为DType(half)
    // 用于shmem put/get操作时确定目标数据的类型和布局
    // 在ReduceScatter操作中，规约后的结果写入到本rank对应的输出位置
    using RemoteDstType = DType;
    // CopyDirect: 拷贝方向枚举类型，定义shmem RMA(Remote Memory Access)操作的数据传输方向
    // CopyDirect::Get表示使用shmem_get方式，从远端rank主动拉取数据到本地
    // ReduceScatter通常使用Get方式：从对称内存读取其他rank的数据进行规约
    using CopyDirect = Catcoc::detail::CopyDirect;
    // TileRemoteCopy: 远程数据拷贝Tile类，封装shmem RMA操作的具体实现
    // 参数: ArchTag-架构类型, RemoteSrcType-源数据类型, RemoteDstType-目标数据类型, CopyDirect::Get-使用Get方式
    // 该类实现了基于shmem的跨rank数据传输，包括数据打包、RDMA传输、数据解包等操作
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Get>;
    // TileScheduler: Tile调度器，负责将通信任务分配到不同的tile上执行
    // EpilogueIdentityTileSwizzle表示使用身份映射调度，即按原始顺序分配tile
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    // COMM_BLOCK_ROWS: 通信块行数，定义每次shmem通信操作处理的数据行数为64行
    // 该参数影响通信操作的粒度和效率，较大的块可以提高带宽利用率
    constexpr uint32_t COMM_BLOCK_ROWS = 64;
    // COMM_BLOCK_COLUMNS: 通信块列数，定义每次shmem通信操作处理的数据列数为256列
    constexpr uint32_t COMM_BLOCK_COLUMNS = 256;
    // CORE_SPLIT_ROWS: 核分裂行数，定义参与通信操作的AIV核心数量为20个
    // 多核并行执行通信操作可以提高整体通信吞吐量
    constexpr uint32_t CORE_SPLIT_ROWS = 20;
    // CORE_SPLIT_COLUMNS: 核分裂列数，定义每个核心处理的列方向分裂数为1
    constexpr uint32_t CORE_SPLIT_COLUMNS = 1;
    // CommBlockShape: 通信块形状类型，定义shmem通信操作的基本数据块维度
    // 用于确定每次RDMA操作传输的数据量大小
    using CommBlockShape = Catlass::MatrixShape<COMM_BLOCK_ROWS, COMM_BLOCK_COLUMNS>;
    // CommCoreSplit: 通信核分裂配置类型，定义参与通信的核心分配方案
    // 指定多少个核心参与通信，以及每个核心处理的数据范围
    using CommCoreSplit = Catlass::MatrixShape<CORE_SPLIT_ROWS, CORE_SPLIT_COLUMNS>;

    // UB_STAGES: UB(Unified Buffer)缓冲区阶段数，定义用于通信操作的UB缓冲区数量为2
    // 多阶段缓冲可以实现流水线操作，在传输一个阶段数据时同时处理另一个阶段的数据
    constexpr uint32_t UB_STAGES = 2;
    // SCATTER_TILE_ROWS: Scatter操作的Tile行数，定义ReduceScatter时每个tile处理32行数据
    // ReduceScatter将各rank的部分结果规约后分发到对应的rank
    constexpr uint32_t SCATTER_TILE_ROWS = 32;
    // SCATTER_TILE_COLUMNS: Scatter操作的Tile列数，定义ReduceScatter时每个tile处理256列数据
    constexpr uint32_t SCATTER_TILE_COLUMNS = 256;
    // EpilogueReduceScatterTileShape: ReduceScatter Tile形状类型
    // 定义shmem ReduceScatter通信操作中每个tile处理的数据维度
    // ReduceScatter操作: 各rank的矩阵乘法结果按行分块，规约后每个rank只保留自己对应的部分
    using EpilogueReduceScatterTileShape = Catlass::MatrixShape<SCATTER_TILE_ROWS, SCATTER_TILE_COLUMNS>;
    // EpilogueReduceScatterDispatch: ReduceScatter分发策略类型
    // 参数: UB_STAGES-缓冲区阶段数, CopyMode::Scatter-使用Scatter模式进行数据分发
    // Scatter模式对应shmem的reduce-scatter语义：先规约再分发，每个rank得到规约结果的一部分
    using EpilogueReduceScatterDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
        Catcoc::detail::CopyMode::Scatter>;
    // BlockEpilogueReduceScatter: ReduceScatter Epilogue块类
    // 封装完整的ReduceScatter通信操作实现，包括:
    // 1. 从各rank的对称内存读取部分矩阵乘法结果(使用shmem_get)
    // 2. 在本地进行规约累加操作
    // 3. 将规约后的结果按rank分发到对应的对称内存位置
    // 参数: EpilogueReduceScatterDispatch-分发策略, RemoteSrcType/RemoteDstType-数据类型,
    //       CommCoreSplit-核分裂配置, CommBlockShape-通信块形状,
    //       EpilogueReduceScatterTileShape-tile形状, TileRemoteCopy-远程拷贝实现, TileScheduler-tile调度
    using BlockEpilogueReduceScatter = CommEpilogue::Block::CommBlockEpilogue<
        EpilogueReduceScatterDispatch,
        RemoteSrcType, RemoteDstType,
        CommCoreSplit,
        CommBlockShape,
        EpilogueReduceScatterTileShape, TileRemoteCopy, TileScheduler
    >;

    // WORKSPACE_STAGES: Workspace工作空间阶段数，定义用于存储中间通信结果的缓冲区数量为2
    // 多阶段workspace可以实现矩阵乘法和通信操作的流水线并行
    constexpr uint32_t WORKSPACE_STAGES = 2;
    // COMM_INTERVAL: 通信间隔，定义每隔10次矩阵乘法tile计算后执行一次通信操作
    // 该参数控制计算和通信的交替频率，平衡计算效率和通信带宽利用
    constexpr uint32_t COMM_INTERVAL = 10;
    // MatmulReduceScatterKernel: Matmul+ReduceScatter组合Kernel类
    // 实现矩阵乘法与ReduceScatter通信的融合执行，利用shmem实现高效的分布式计算
    // 执行流程:
    // 1. 执行矩阵乘法计算(本地A×B得到完整C结果)
    // 2. 使用shmem ReduceScatter将C结果按行分块规约并分发到各rank
    // 3. 每个rank得到自己对应部分的规约结果 C_local = sum(C_i[block])
    // 参数: BlockMmad-矩阵乘法块, BlockEpilogueReduceScatter-ReduceScatter通信块,
    //       BlockMmadScheduler-计算调度器, BlockEpilogueScheduler-通信调度器,
    //       WORKSPACE_STAGES-工作空间阶段数
    using MatmulReduceScatterKernel = DGemm::Kernel::MatmulReduceScatter<
        BlockMmad,
        BlockEpilogueReduceScatter,
        BlockMmadScheduler,
        BlockEpilogueScheduler,
        WORKSPACE_STAGES
    >;

    // reduceScatterParams: ReduceScatter通信参数结构体
    // 存储ReduceScatter操作的具体配置参数，如数据偏移、通信目标rank等
    typename BlockEpilogueReduceScatter::Params reduceScatterParams{};

    // MatmulReduceScatterKernel::Params: Matmul+ReduceScatter Kernel参数结构体
    // 包含完整的kernel执行所需的所有参数
    typename MatmulReduceScatterKernel::Params params{
        problemShape,               // 问题规模: m×k×n的矩阵乘法维度
        rankIdx,                    // 当前rank编号: 通过shmem_my_pe()获取，标识当前进程
        rankSize,                   // rank总数: 通过shmem_n_pes()获取，标识参与通信的进程数
        COMM_INTERVAL,              // 通信间隔: 每隔多少次计算后执行ReduceScatter通信
        gmA, layoutA,               // 矩阵A: 全局内存地址和数据布局
        gmB, layoutB,               // 矩阵B: 全局内存地址和数据布局
        gmD, layoutD,               // 矩阵D(输出): 全局内存地址和数据布局（ReduceScatter后本rank对应的部分）
        gmSymmetric,                // 对称内存地址: shmem_malloc分配的对称内存，用于跨rank通信
                                    // 所有rank通过相同偏移访问此内存，实现RDMA数据交换
        reduceScatterParams         // ReduceScatter参数: 规约分发操作的配置
    };

    MatmulReduceScatterKernel matmulReduceScatterKernel;
    matmulReduceScatterKernel(params);
}

struct Options {
    static constexpr auto HELPER =
       "Usage: matmul_reduce_scatter rank_size rank_id ip_port m n k [device_id_list]\n";

    int rankSize;
    int rankId;
    std::string ipPort;
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::string dataPath;
    std::vector<int> deviceIdList{};

    int Parse(int argc, char **argv)
    {
        enum ArgsIndex {
            RANK_SIZE_INDEX = 1,
            RANK_ID_INDEX,
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

        rankSize = std::atoi(argv[RANK_SIZE_INDEX]);
        rankId = std::atoi(argv[RANK_ID_INDEX]);
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
            for (size_t i = 0; i < rankSize; ++i) {
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

int main(int argc, char **argv)
{
    int status = SHMEM_SUCCESS;
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }
    int rankSize = options.rankSize;
    int rankId = options.rankId;
    std::string ipPort = options.ipPort;
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;
    int32_t deviceId = options.deviceIdList[rankId];

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipPort << "\n";

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    // shmem_set_conf_store_tls(): 禁用TLS(Thread Local Storage)存储配置方式
    // 参数: false表示禁用TLS，nullptr和0表示不使用默认配置文件路径和长度
    // 设置为false后使用shmem_set_attr/shmem_init_attr自定义配置方式初始化shmem环境
    status = shmem_set_conf_store_tls(false, nullptr, 0);
    // shmem_init_attr_t: shmem初始化属性结构体，用于存储rank信息、内存大小、网络配置等初始化参数
    shmem_init_attr_t *attributes;
    // shmem_set_attr(): 设置shmem初始化属性参数
    // 参数1 rankId: 当前进程的rank编号（进程在通信组中的唯一标识）
    // 参数2 rankSize: 通信组中总进程数量（所有参与分布式计算的rank总数）
    // 参数3 NPU_MALLOC_SPACE: 每个rank分配的对称内存空间大小（1GB）
    // 参数4 ipPort.c_str(): 网络通信的IP地址和端口字符串，用于rank间网络连接
    // 参数5 &attributes: 输出参数，返回配置好的初始化属性结构体指针
    status = shmem_set_attr(rankId, rankSize, NPU_MALLOC_SPACE, ipPort.c_str(), &attributes);
    // shmem_init_attr(): 根据attributes中的配置参数初始化shmem运行环境
    // 此函数会建立rank间的网络连接、分配对称内存堆、初始化通信通道等
    status = shmem_init_attr(attributes);
    // shmem_init_status(): 检查并返回shmem初始化的状态结果
    // 返回SHMEM_SUCCESS表示初始化成功，否则表示初始化失败需要处理错误
    status = shmem_init_status();

    size_t aSize = static_cast<size_t>(m) * k * sizeof(__fp16);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(__fp16);
    size_t dSize = static_cast<size_t>(m) * n * sizeof(__fp16);
    size_t dSizeScatter = dSize / options.rankSize;

    uint8_t *aDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *aHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&aHost), aSize));
    ReadFile(options.GetDataPath("rank_" + std::to_string(rankId) + "_a.bin"), aHost, aSize);
    ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *bHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&bHost), bSize));
    ReadFile(options.GetDataPath("rank_" + std::to_string(rankId) + "_b.bin"), bHost, bSize);
    ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *dDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&dDevice), dSizeScatter, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *dHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&dHost), dSize));

    // shmem_malloc(): 从对称共享内存堆(Symmetric Heap)中分配指定大小的内存空间
    // 对称内存是指所有rank在相同偏移位置都能访问的共享内存区域，用于跨rank通信
    // 参数: (204 * 1024 * 1024) * sizeof(__fp16) = 约408MB的fp16类型内存空间
    // 返回: 对称内存指针，所有rank都可以通过相同偏移访问该内存区域
    // 该内存用于存储ReduceScatter操作中的中间通信数据，实现跨rank的数据聚合和分发
    // 在Matmul+ReduceScatter场景中，各rank将矩阵乘法结果写入对称内存
    // 然后从对称内存读取其他rank的数据进行规约
    void *symmPtr = shmem_malloc((204 * 1024 * 1024) * sizeof(__fp16));
    uint8_t *symmetricPtr = reinterpret_cast<uint8_t *>(symmPtr);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "Before calling MM_RS kernel " << std::endl;
    for (int i = 0; i < 1; i++) {
        // shmemx_get_ffts_config(): 获取FFTS(Fast Flag Task Sync)硬件同步配置地址
        // FFTS是NPU核间快速同步机制，用于在kernel执行时实现核间的轻量级同步操作
        // 返回: FFTS配置寄存器的物理地址，传递给kernel用于设置同步基址
        // kernel内部会使用此地址进行ReduceScatter操作的核间同步
        uint64_t fftsAddr = shmemx_get_ffts_config();
        ShmemMatmulReduceScatter<<<BLOCK_NUM, nullptr, stream>>>(
            fftsAddr,
            aDevice, bDevice, dDevice, symmetricPtr,
            m, n, k
        );
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "After calling MM_RS kernel " << std::endl;

    ACL_CHECK(aclrtMemcpy(dHost, dSizeScatter, dDevice, dSizeScatter, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(options.GetDataPath("shmem_output.bin"), dHost, dSizeScatter, rankId * dSizeScatter);
    if (rankId == 0) {
        std::printf("test finished\n");
    }

    // shmem_free(): 释放之前通过shmem_malloc()分配的对称共享内存空间
    // 参数: symmPtr - 要释放的对称内存指针
    // 此函数会将内存归还到对称内存堆，供后续shmem_malloc调用重新使用
    // 注意：释放后不应再访问该内存区域，否则会导致未定义行为
    shmem_free(symmPtr);

    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(dHost));
    ACL_CHECK(aclrtFree(aDevice));
    ACL_CHECK(aclrtFree(bDevice));
    ACL_CHECK(aclrtFree(dDevice));

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    // shmem_finalize(): 结束并清理shmem运行环境，释放所有shmem相关资源
    // 此函数会执行以下操作:
    // 1. 释放所有未释放的对称内存资源
    // 2. 关闭rank间的网络通信连接
    // 3. 清理通信通道和同步资源
    // 4. 重置shmem运行状态
    // 调用此函数后，所有shmem API都不应再被调用，直到重新初始化
    // 返回: SHMEM_SUCCESS表示成功清理，否则表示清理过程中出现错误
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}
