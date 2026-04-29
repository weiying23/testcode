/*
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
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

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
#include "catcoc/dgemm/block/block_swizzle_allgather.h"
#include "catcoc/dgemm/kernel/allgather_matmul.h"

static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

using namespace AscendC;
using namespace Catcoc;

constexpr uint32_t BLOCK_NUM = 20;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;

using ElementA = half;
using ElementB = half;
using ElementC = half;

#if defined(ENABLE_ASCENDC_DUMP)
CATLASS_GLOBAL
void ShmemAllGatherMatmul(
    uint64_t fftsAddr,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmSymmetric,
    uint32_t m, uint32_t n, uint32_t k, GM_ADDR dump)
{
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#else
CATLASS_GLOBAL
void ShmemAllGatherMatmul(
    uint64_t fftsAddr,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmSymmetric,
    uint32_t m, uint32_t n, uint32_t k)
{
#endif
    // shmemx_set_ffts_config()/SetSyncBaseAddr: 设置FFTS(Fast Flag Task Sync)硬件同步配置地址
    // FFTS是NPU核间快速同步机制，用于在kernel执行时实现核间的轻量级同步操作
    // 参数: fftsAddr - FFTS配置寄存器的物理地址
    // kernel内部会使用此地址进行AllGather操作的核间同步，确保各rank的数据收集顺序正确
    AscendC::SetSyncBaseAddr(fftsAddr);

    // Define ArchTag
    using ArchTag = Catlass::Arch::AtlasA2;

    // shmem_my_pe(): 获取当前PE（Processing Element）的编号，返回当前进程在通信组中的唯一标识
    // 在AllGather操作中，每个rank需要知道自己的编号以确定数据在收集结果中的位置
    // 返回值: 当前rank的ID（0到rankSize-1之间的整数）
    uint32_t rank = shmem_my_pe();
    // shmem_n_pes(): 获取通信组中总PE数量，返回参与通信的所有进程总数（rank总数）
    // 用于确定AllGather操作的参与者数量和最终收集结果的大小
    // 返回值: 通信组中rank的总数
    uint32_t rankSize = shmem_n_pes();

    Catlass::GemmCoord problemShape{m, n, k};
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    // LayoutC布局大小为 m * rankSize * n，因为AllGather会将所有rank的矩阵乘法结果收集在一起
    // 每个rank贡献m行，收集后总共有 m * rankSize 行
    LayoutC layoutC{m * rankSize, n};

    // Block level, define BlockMmad
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
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType
    >;

    constexpr uint32_t SWIZZLE_GROUP_SIZE = 7;
    constexpr uint32_t SWIZZLE_DIRECTION = 1;
    // GemmBlockSwizzleAllGatherMesh: AllGather+Matmul专用的块调度器
    // 参数: SWIZZLE_GROUP_SIZE=7 - 调度组大小，定义每个调度组包含7个block
    // 参数: SWIZZLE_DIRECTION=1 - 调度方向，定义block的调度顺序方向
    // 该调度器专门优化了AllGather通信与Matmul计算的流水线并行执行
    // 在AllGather场景中，计算块和通信块交替执行，此调度器负责协调两者
    using BlockMmadScheduler = typename Catcoc::DGemm::Block::GemmBlockSwizzleAllGatherMesh<SWIZZLE_GROUP_SIZE,
                                                                                            SWIZZLE_DIRECTION>;
    // BlockCommSwizzle: 通信块调度器，负责分配通信操作到不同的AIV核心上执行
    // 参数0表示通信块调度偏移量为0，即从第一个核心开始分配通信任务
    // 该调度器确保AllGather通信操作均匀分配到各核心，避免通信负载不均衡
    using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0>;

    // RemoteSrcType: 远程数据源类型，指定从远端rank读取数据的类型为AType(half)
    // 在AllGather操作中，RemoteSrcType是各rank本地矩阵A的数据类型
    // 用于shmem put操作时确定源数据的类型和布局（本场景使用Put方式推送数据）
    using RemoteSrcType = AType;
    // RemoteDstType: 远程数据目标类型，指定写入到远端rank或本地对称内存的数据类型
    // 在AllGather操作中，各rank将自己的矩阵A推送到对称内存，RemoteDstType也是AType
    // 用于shmem put/get操作时确定目标数据的类型和布局
    using RemoteDstType = AType;
    // CopyDirect: 拷贝方向枚举类型，定义shmem RMA(Remote Memory Access)操作的数据传输方向
    // CopyDirect::Put表示使用shmem_put方式，从本地主动推送数据到对称内存供其他rank读取
    // AllGather通常使用Put方式：各rank将数据推送到对称内存，其他rank从对称内存读取
    using CopyDirect = Catcoc::detail::CopyDirect;
    // TileRemoteCopy: 远程数据拷贝Tile类，封装shmem RMA操作的具体实现
    // 参数: ArchTag-架构类型, RemoteSrcType-源数据类型, RemoteDstType-目标数据类型, CopyDirect::Put-使用Put方式
    // 该类实现了基于shmem的跨rank数据传输，包括数据打包、RDMA Put传输、数据解包等操作
    // Put方式: 本rank将矩阵A数据推送到对称内存，其他rank从对称内存读取该数据
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Put>;
    // TileScheduler: Tile调度器，负责将通信任务分配到不同的tile上执行
    // EpilogueIdentityTileSwizzle表示使用身份映射调度，即按原始顺序分配tile
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    // COMM_BLOCK_ROWS: 通信块行数，定义每次shmem通信操作处理的数据行数为64行
    // 该参数影响通信操作的粒度和效率，较大的块可以提高RDMA带宽利用率
    constexpr uint32_t COMM_BLOCK_ROWS = 64;
    // COMM_BLOCK_COLUMNS_DIVISOR: 通信块列数除数，用于计算通信块的列数上限
    // 实际列数 = UINT_MAX / COMM_BLOCK_COLUMNS_DIVISOR，表示通信块列数不受限制（动态计算）
    constexpr uint32_t COMM_BLOCK_COLUMNS_DIVISOR = 2;
    // CORE_SPLIT_ROWS: 核分裂行数，定义参与AllGather通信操作的AIV核心数量为20个
    // 多核并行执行通信操作可以提高整体通信吞吐量，充分利用RDMA带宽
    constexpr uint32_t CORE_SPLIT_ROWS = 20;
    // CORE_SPLIT_COLUMNS: 核分裂列数，定义每个核心处理的列方向分裂数为1
    constexpr uint32_t CORE_SPLIT_COLUMNS = 1;
    // CommBlockShape: 通信块形状类型，定义shmem AllGather通信操作的基本数据块维度
    // 用于确定每次RDMA Put操作传输的矩阵A数据量大小
    // 行数固定为64，列数动态计算以适应不同矩阵宽度
    using CommBlockShape = Catlass::MatrixShape<COMM_BLOCK_ROWS, UINT_MAX / COMM_BLOCK_COLUMNS_DIVISOR>;
    // CommCoreSplit: 通信核分裂配置类型，定义参与AllGather通信的核心分配方案
    // 指定20个核心参与通信，每个核心处理特定的数据范围
    using CommCoreSplit = Catlass::MatrixShape<CORE_SPLIT_ROWS, CORE_SPLIT_COLUMNS>;

    // UB_STAGES: UB(Unified Buffer)缓冲区阶段数，定义用于通信操作的UB缓冲区数量为2
    // 多阶段缓冲可以实现流水线操作，在传输一个阶段数据时同时处理另一个阶段的数据
    // 提高通信效率，减少等待时间
    constexpr uint32_t UB_STAGES = 2;
    // ALLGATHER_TILE_ROWS: AllGather操作的Tile行数，定义AllGather时每个tile处理32行数据
    // AllGather将各rank的矩阵A收集到对称内存，每个tile处理32行的收集工作
    constexpr uint32_t ALLGATHER_TILE_ROWS = 32;
    // ALLGATHER_TILE_COLUMNS: AllGather操作的Tile列数，定义AllGather时每个tile处理256列数据
    constexpr uint32_t ALLGATHER_TILE_COLUMNS = 256;
    // EpilogueAllGatherTileShape: AllGather Tile形状类型
    // 定义shmem AllGather通信操作中每个tile处理的数据维度
    // AllGather操作: 各rank的矩阵A被收集并存储到对称内存，供后续矩阵乘法使用
    // 执行流程: rank i将自己的矩阵A推送到对称内存的对应位置，然后所有rank读取完整的矩阵A集合
    using EpilogueAllGatherTileShape = Catlass::MatrixShape<ALLGATHER_TILE_ROWS, ALLGATHER_TILE_COLUMNS>;
    // EpilogueAllGatherDispatch: AllGather分发策略类型
    // 参数: UB_STAGES-缓冲区阶段数, CopyMode::Gather-使用Gather模式进行数据收集
    // Gather模式对应shmem的all-gather语义：收集各rank的数据并广播到所有rank
    // 在AllGather+Matmul场景中，先收集所有rank的矩阵A，然后每个rank用完整的A集合进行矩阵乘法
    using EpilogueAllGatherDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
        Catcoc::detail::CopyMode::Gather>;
    // BlockEpilogueAllGather: AllGather Epilogue块类
    // 封装完整的AllGather通信操作实现，包括:
    // 1. 各rank将自己部分的矩阵A数据推送到对称内存(使用shmem_put)
    // 2. 各rank从对称内存读取所有rank推送的矩阵A数据
    // 3. 将收集到的完整矩阵A集合组装成连续的内存布局
    // 参数: EpilogueAllGatherDispatch-分发策略, RemoteSrcType/RemoteDstType-数据类型,
    //       CommCoreSplit-核分裂配置, CommBlockShape-通信块形状,
    //       EpilogueAllGatherTileShape-tile形状, TileRemoteCopy-远程拷贝实现(Put方式), TileScheduler-tile调度
    using BlockEpilogueAllGather = CommEpilogue::Block::CommBlockEpilogue<
        EpilogueAllGatherDispatch,
        RemoteSrcType, RemoteDstType,
        CommCoreSplit,
        CommBlockShape,
        EpilogueAllGatherTileShape, TileRemoteCopy, TileScheduler
    >;

    // WORKSPACE_STAGES: Workspace工作空间阶段数，定义用于存储中间通信结果的缓冲区数量为2
    // 多阶段workspace可以实现AllGather通信和矩阵乘法计算的流水线并行
    // 一个阶段执行通信时，另一个阶段的数据可以同时进行计算
    constexpr uint32_t WORKSPACE_STAGES = 2;
    // COMM_INTERVAL: 通信间隔，定义每隔3次计算tile后执行一次AllGather通信操作
    // 该参数控制计算和通信的交替频率，平衡计算效率和通信带宽利用
    // 较小的间隔增加通信频率，较大的间隔减少通信开销但增加计算等待时间
    constexpr uint32_t COMM_INTERVAL = 3;
    // AllGatherMatmulKernel: AllGather+Matmul组合Kernel类
    // 实现AllGather通信与矩阵乘法计算的融合执行，利用shmem实现高效的分布式矩阵乘法
    // 执行流程:
    // 1. 各rank并行执行本地矩阵乘法计算的部分工作(A_local × B_local)
    // 2. 同时执行AllGather通信，将各rank的矩阵A收集到对称内存
    // 3. 使用收集到的完整矩阵A集合继续执行矩阵乘法
    // 4. 最终每个rank都得到完整的矩阵乘法结果 C = A_all × B_local
    // 参数: BlockMmad-矩阵乘法块, BlockEpilogueAllGather-AllGather通信块,
    //       BlockMmadScheduler-计算调度器(AllGather专用), BlockEpilogueScheduler-通信调度器,
    //       WORKSPACE_STAGES-工作空间阶段数
    using AllGatherMatmulKernel = DGemm::Kernel::AllGatherMatmul<
        BlockMmad,
        BlockEpilogueAllGather,
        BlockMmadScheduler,
        BlockEpilogueScheduler,
        WORKSPACE_STAGES
    >;

    // allGatherParams: AllGather通信参数结构体
    // 存储AllGather操作的具体配置参数，如数据偏移、通信目标rank列表等
    // 初始化为空，后续kernel会根据rank和rankSize自动填充
    typename BlockEpilogueAllGather::Params allGatherParams{};

    // AllGatherMatmulKernel::Params: AllGather+Matmul Kernel参数结构体
    // 包含完整的kernel执行所需的所有参数
    // Prepare params
    typename AllGatherMatmulKernel::Params params{
        problemShape,               // 问题规模: m×k×n的矩阵乘法维度
        rank,                       // 当前rank编号: 通过shmem_my_pe()获取，标识当前进程
        rankSize,                   // rank总数: 通过shmem_n_pes()获取，标识参与通信的进程数
        COMM_INTERVAL,              // 通信间隔: 每隔多少次计算后执行AllGather通信
        gmA, layoutA,               // 矩阵A: 全局内存地址和数据布局（本地部分）
        gmB, layoutB,               // 矩阵B: 全局内存地址和数据布局（本地完整）
        gmC, layoutC,               // 矩阵C(输出): 全局内存地址和数据布局（收集后的完整结果）
        gmSymmetric,                // 对称内存地址: shmem_malloc分配的对称内存，用于跨rank通信
                                    // 所有rank通过相同偏移访问此内存，实现AllGather数据交换
                                    // 各rank将自己的矩阵A推送到此内存的对应位置
        allGatherParams             // AllGather参数: 全收集操作的配置
    };

    // Call kernel: 执行AllGather+Matmul融合kernel
    // kernel内部会先执行AllGather收集矩阵A，然后执行矩阵乘法计算
    AllGatherMatmulKernel matmulCommKernel;
    matmulCommKernel(params);
}

struct Options {
    static constexpr auto HELPER =
       "Usage: allgather_matmul rank_size rank_id ip_port m n k [device_id_list]\n";

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

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId <<
        " input_ip: " << ipPort << std::endl;

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
    // 参数3 gNpuMallocSpace: 每个rank分配的对称内存空间大小（1GB）
    // 参数4 ipPort.c_str(): 网络通信的IP地址和端口字符串，用于rank间RDMA网络连接建立
    // 参数5 &attributes: 输出参数，返回配置好的初始化属性结构体指针
    status = shmem_set_attr(rankId, rankSize, gNpuMallocSpace, ipPort.c_str(), &attributes);
    // shmem_init_attr(): 根据attributes中的配置参数初始化shmem运行环境
    // 此函数会执行: 建立rank间RDMA网络连接、分配对称内存堆、初始化通信通道和同步资源等
    status = shmem_init_attr(attributes);
    // shmem_init_status(): 检查并返回shmem初始化的状态结果
    // 返回SHMEM_SUCCESS表示初始化成功，否则表示初始化失败需要处理错误
    status = shmem_init_status();

    size_t aSize = static_cast<size_t>(m) * k * sizeof(__fp16);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(__fp16);
    size_t cSize = static_cast<size_t>(m) * rankSize * n * sizeof(__fp16);

    uint8_t *aDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *aHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&aHost), aSize));
    ReadFile(options.GetDataPath("rank_" + std::to_string(rankId) + "_a.bin"), aHost, aSize);
    ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *bHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&bHost), bSize));
    ReadFile(options.GetDataPath("rank_" + std::to_string(rankId) + "_b.bin"), bHost, bSize);
    ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *cHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&cHost), cSize));

    // shmem_malloc(): 从对称共享内存堆(Symmetric Heap)中分配指定大小的内存空间
    // 对称内存是指所有rank在相同偏移位置都能访问的共享内存区域，用于跨rank RDMA通信
    // 参数: (204 * 1024 * 1024) * sizeof(__fp16) = 约408MB的fp16类型内存空间
    // 返回: 对称内存指针，所有rank都可以通过相同偏移访问该内存区域
    // 该内存用于存储AllGather操作中的矩阵A通信数据，实现跨rank的矩阵A收集
    // 在AllGather+Matmul场景中，各rank将本地矩阵A推送到此对称内存的对应位置
    // 然后所有rank从对称内存读取完整的矩阵A集合进行矩阵乘法计算
    void *symmPtr = shmem_malloc((204 * 1024 * 1024) * sizeof(__fp16));
    uint8_t *gmSymmetric = (uint8_t *)symmPtr;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "Before calling AG_MM kernel " << std::endl;
    for (int i = 0; i < 1; i++) {
        // shmemx_get_ffts_config(): 获取FFTS(Fast Flag Task Sync)硬件同步配置地址
        // FFTS是NPU核间快速同步机制，用于在kernel执行时实现核间的轻量级同步操作
        // 返回: FFTS配置寄存器的物理地址，传递给kernel用于设置同步基址
        // kernel内部会使用此地址进行AllGather通信操作的核间同步，确保数据收集顺序正确
        uint64_t fftsAddr = shmemx_get_ffts_config();
#if defined(ENABLE_ASCENDC_DUMP)
        uint8_t *deviceDump{nullptr};
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceDump), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
        ShmemAllGatherMatmul<<<BLOCK_NUM, nullptr, stream>>>(
            fftsAddr,
            aDevice, bDevice, cDevice, gmSymmetric,
            m, n, k, deviceDump);
        ACL_CHECK(aclrtSynchronizeStream(stream));
        Adx::AdumpPrintWorkSpace(deviceDump, ALL_DUMPSIZE, stream, "test");
#else
        ShmemAllGatherMatmul<<<BLOCK_NUM, nullptr, stream>>>(
            fftsAddr,
            aDevice, bDevice, cDevice, gmSymmetric,
            m, n, k);
#endif
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "After calling AG_MM kernel " << std::endl;

    ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(options.GetDataPath("shmem_output.bin"), cHost, cSize);
    if (rankId == 0) {
        std::printf("test finished\n");
    }

    // shmem_free(): 释放之前通过shmem_malloc()分配的对称共享内存空间
    // 参数: symmPtr - 要释放的对称内存指针
    // 此函数会将内存归还到对称内存堆，供后续shmem_malloc调用重新使用
    // 注意：释放后不应再访问该内存区域，否则会导致未定义行为或数据损坏
    shmem_free(symmPtr);

    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(cHost));
    ACL_CHECK(aclrtFree(aDevice));
    ACL_CHECK(aclrtFree(bDevice));
    ACL_CHECK(aclrtFree(cDevice));

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    // shmem_finalize(): 结束并清理shmem运行环境，释放所有shmem相关资源
    // 此函数会执行以下操作:
    // 1. 释放所有未释放的对称内存资源（如果还有未释放的会自动释放）
    // 2. 关闭rank间的RDMA网络通信连接
    // 3. 清理通信通道和同步资源（FFTS、信号量等）
    // 4. 重置shmem运行状态，使后续shmem API调用无效
    // 调用此函数后，所有shmem API都不应再被调用，直到重新初始化
    // 返回: SHMEM_SUCCESS表示成功清理，否则表示清理过程中出现错误
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}