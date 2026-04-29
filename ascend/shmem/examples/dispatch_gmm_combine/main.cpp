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

// misc
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "dispatch_policy_custom.h"

// from shmem-templates
#include "dispatch_gmm_combine.h"

// shmem_host
#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_team.h"
#include "shmem_api.h"
// utils
#include "utils.h"
#include "select_helper.h"

static uint32_t gNpuNum = 16;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

using namespace AscendC;
using namespace Catlass;
using fp16_t = op::fp16_t;

struct CoCTiling {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    uint32_t swizzleDirect = 0;
    uint32_t swizzleOffset = 0;
    int32_t ubMoveNum = 0;
    uint32_t pValue = 0;
    uint32_t commNpuSplit = 0;
    uint32_t commDataSplit = 0;
    uint32_t lenPerLoop = 0;
    uint32_t EP = 0;
    uint32_t expertPerRank = 0;
    uint32_t maxOutputSize = 0;

    int64_t topK;
    int64_t activeNum;
    int64_t expertCapacity;
    int64_t expertNum;
    int64_t dropPadMode;
    int64_t expertTokensCountOrCumsumFlag;
    bool expertTokensBeforeCapacityFlag;
    int64_t quantMode;
    uint64_t initRoutingQuantTilingKey;
};

constexpr uint32_t
BLOCK_NUM = 8;
constexpr int32_t
BLOCK_SIZE_16 = 16;

template<class AType_,
    class BType_,
    class CType_,
    bool TB_,
    bool Nz_>
class DispatchGMMClass {
public:
    CATLASS_DEVICE
    DispatchGMMClass()
    {}

    CATLASS_DEVICE
    void Run(uint64_t fftsAddr, GemmCoord problemShape, GM_ADDR a, GM_ADDR b1, GM_ADDR b2, GM_ADDR c, GM_ADDR scale1,
             GM_ADDR scale2, GM_ADDR symmetricPtr,
             GM_ADDR expertIdx, GM_ADDR moeInitRoutingQuantV2Scale, GM_ADDR moeInitRoutingQuantV2Offset,
             GM_ADDR expertTokensBeforeCapacity, GM_ADDR probs,
             GM_ADDR ptrWorkspace, CoCTiling cocTiling,
             optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData)
    {
        // Define ArchTag
        using ArchTag = Arch::AtlasA2;
        constexpr bool enableUnitFlag = false;
        constexpr bool enableShuffleK = true;
        // unzip cocTiling
        uint32_t m = cocTiling.m;
        uint32_t n = cocTiling.n;
        uint32_t k = cocTiling.k;

        uint32_t epilogueCoreNum = 20;
        uint32_t epilogueGranularity = 17;

        uint32_t n2 = k;
        uint32_t k2 = n / 2;

        int32_t ubMoveNum = cocTiling.ubMoveNum;
        uint32_t EP = cocTiling.EP;
        uint32_t expertPerRank = cocTiling.expertPerRank;
        uint32_t maxOutputSize = cocTiling.maxOutputSize;
        int64_t activeNum = cocTiling.activeNum;
        int64_t expertCapacity = cocTiling.expertCapacity;
        int64_t expertNum = cocTiling.expertNum;
        int64_t dropPadMode = cocTiling.dropPadMode;
        int64_t expertTokensCountOrCumsumFlag = cocTiling.expertTokensCountOrCumsumFlag;
        bool expertTokensBeforeCapacityFlag = cocTiling.expertTokensBeforeCapacityFlag;
        int64_t quantMode = cocTiling.quantMode;
        int64_t topK = cocTiling.topK;
        uint64_t initRoutingQuantTilingKey = cocTiling.initRoutingQuantTilingKey;

        // 获取当前PE（Processing Element）的编号，即当前rank的ID
        // shmem_my_pe(): 返回当前进程在通信组中的唯一标识(rank编号)
        // 在MOE(Mixture of Experts)场景中，每个rank负责一部分专家的计算
        // rank编号用于确定当前rank应该处理哪些专家以及需要从哪些rank获取专家权重
        uint32_t rank = shmem_my_pe();
        // shmem_n_pes(): 获取通信组中总PE数量，返回参与通信的所有进程总数（rank总数）
        // 用于确定专家分发的目标rank数量和通信操作的参与者规模
        uint32_t rankSize = shmem_n_pes();

        using LayoutA = layout::RowMajor;
        using LayoutB = typename std::conditional<
            Nz_,
            layout::zN,
            typename std::conditional<TB_, layout::ColumnMajor, layout::RowMajor>::type
        >::type;

        LayoutB layoutB1 = LayoutBInitializer<LayoutB, BType_>::create(k, n);
        LayoutB layoutB2 = LayoutBInitializer<LayoutB, BType_>::create(k2, n2);
        using LayoutC = layout::RowMajor;
        constexpr int L1TILEM = 128;
        constexpr int L1TILEN = 256;
        constexpr int L1TILEK = 512;
        using L1TileShape = GemmShape<L1TILEM, L1TILEN, L1TILEK>;   // M, N, K

        constexpr
        uint32_t workspaceStages = 2;
        constexpr
        uint32_t preloadStages = 1;
        constexpr
        uint32_t l1Stages = 2;
        constexpr
        uint32_t l0AStages = 2;
        constexpr
        uint32_t l0BStages = 2;
        constexpr
        uint32_t l0CStages = 1;
        constexpr
        uint32_t l1StagesNormal = 1;
        constexpr
        uint32_t l0BStagesNormal = 1;
        using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsyncFixpipe<
            preloadStages,
            l1Stages, l0AStages, l0BStages, l0CStages,
            enableUnitFlag, enableShuffleK
        >;

        constexpr int L0TILEM = 128;
        constexpr int L0TILEN = 256;
        constexpr int L0TILEK = 128;
        using L0TileShape = GemmShape<L0TILEM, L0TILEN, L0TILEK>;
        using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
        using BType = Gemm::GemmType<int8_t, LayoutB>;
        using CType = Gemm::GemmType<float16_t, layout::RowMajor>;
        using D1Type = Gemm::GemmType<int8_t, layout::RowMajor>;
        using D2Type = typename std::conditional<
            std::is_same_v<CType_, bfloat16_t>,
            Gemm::GemmType<bfloat16_t, layout::RowMajor>,
            Gemm::GemmType<CType_, layout::RowMajor>> ::type;

        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        constexpr
        uint32_t ubStages = 2;

        using EpilogueDispatchPolicy1 = Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<ubStages>;

        using ScaleType = Gemm::GemmType<uint64_t, layout::VectorLayout>;
        using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
        using ElementMulType = Gemm::GemmType<float, layout::RowMajor>;
        using TileElemWiseMuls = Epilogue::Tile::TileElemWiseMuls<ArchTag, ElementMulType, 0>;

        using TileCopy1 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D1Type>;
        using BlockEpilogue1 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy1, CType, PerTokenScaleType,
                D1Type, TileElemWiseMuls, TileCopy1>;

        using EpilogueDispatchPolicy2 = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
        using TileCopy2 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D2Type>;
        using BlockEpilogue2 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType,
                D2Type, TileCopy2>;

        constexpr uint32_t SWIZZLE_GROUP_SIZE = 9;
        constexpr uint32_t SWIZZLE_DIRECTION = 1;
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<SWIZZLE_GROUP_SIZE, SWIZZLE_DIRECTION>;
        using ElementGroupList = int64_t;
        using MatmulKernel = Gemm::Kernel::DispatchGmmCombineKernel<BlockMmad,
                BlockScheduler, ElementGroupList, BlockEpilogue1, BlockEpilogue2>;

        LayoutA layoutA1{m, k};
        LayoutA layoutA2{m, k2};
        layout::VectorLayout layoutScale1{n};
        layout::VectorLayout layoutScale2{n2};
        layout::RowMajor layoutD1{maxOutputSize, k2};
        layout::RowMajor layoutD2{static_cast<uint32_t>(m * topK), n2};
        // Prepare params
        typename MatmulKernel::Params params{
            problemShape, cocTiling.EP, cocTiling.expertPerRank, cocTiling.maxOutputSize,
            rank, rankSize,
            activeNum, expertCapacity, expertNum, dropPadMode, expertTokensCountOrCumsumFlag,
            expertTokensBeforeCapacityFlag, quantMode, topK, initRoutingQuantTilingKey,
            epilogueCoreNum, epilogueGranularity,
            a, layoutA1, layoutA2,
            b1, layoutB1,
            b2, layoutB2,
            scale1, layoutScale1,
            scale2, layoutScale2,
            c, layoutD1, layoutD2,
            expertIdx, moeInitRoutingQuantV2Scale, moeInitRoutingQuantV2Offset,
            expertTokensBeforeCapacity, probs,
            ptrWorkspace,
            symmetricPtr, ubMoveNum, moeInitRoutingQuantV2TilingData};

        MatmulKernel kernel(params);
        kernel(params);
    }
};

CATLASS_GLOBAL
void DispatchGMM(
    uint64_t fftsAddr, GemmCoord problemShape, GM_ADDR a, GM_ADDR b1, GM_ADDR b2, GM_ADDR c, GM_ADDR scale1,
    GM_ADDR scale2, GM_ADDR symmetricPtr,
    GM_ADDR expertIdx, GM_ADDR moeInitRoutingQuantV2Scale, GM_ADDR moeInitRoutingQuantV2Offset,
    GM_ADDR expertTokensBeforeCapacity, GM_ADDR probs,
    GM_ADDR ptrWorkspace, CoCTiling cocTiling, int64_t tilingKey,
    optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // Set FFTS address
    AscendC::SetSyncBaseAddr(reinterpret_cast<uint64_t>(fftsAddr));

    DispatchGMMClass<int8_t, int8_t, float16_t, false, true> op;
    op.Run(fftsAddr, problemShape, a, b1, b2, c, scale1, scale2, symmetricPtr, expertIdx, moeInitRoutingQuantV2Scale,
        moeInitRoutingQuantV2Offset, expertTokensBeforeCapacity, probs,
        ptrWorkspace, cocTiling, moeInitRoutingQuantV2TilingData);
}

void InitData(uint8_t **hostPtr, uint8_t **devicePtr, size_t aSize, std::string path = "")
{
    std::cout << path << std::endl;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**> (devicePtr), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(hostPtr), aSize));
    if (path.length() == 0) {
        return;
    }
    ReadFile(path, *hostPtr, aSize);
    ACL_CHECK(aclrtMemcpy(*devicePtr, aSize, *hostPtr, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
}

int main(int argc, char **argv)
{
    int status = SHMEM_SUCCESS;
    int rankSize = atoi(argv[1]);
    int rankId = atoi(argv[2]);
    std::string ipport = argv[3];

    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = atoi(argv[4]) + rankId % gNpuNum;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
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
    // 参数4 ipport.c_str(): 网络通信的IP地址和端口字符串，用于rank间RDMA网络连接建立
    // 参数5 &attributes: 输出参数，返回配置好的初始化属性结构体指针
    status = shmem_set_attr(rankId, rankSize, gNpuMallocSpace, ipport.c_str(), &attributes);
    // shmem_init_attr(): 根据attributes中的配置参数初始化shmem运行环境
    // 此函数会执行: 建立rank间RDMA网络连接、分配对称内存堆、初始化通信通道和同步资源等
    status = shmem_init_attr(attributes);
    // shmem_init_status(): 检查并返回shmem初始化的状态结果
    // 返回SHMEM_SUCCESS表示初始化成功，否则表示初始化失败需要处理错误
    status = shmem_init_status();

    uint32_t m = atoi(argv[5]);
    uint32_t k = atoi(argv[6]);
    uint32_t n = atoi(argv[7]);
    uint32_t EP = rankSize;
    uint32_t expertPerRank = atoi(argv[8]);
    uint32_t dataType = atoi(argv[9]);
    uint32_t weightNz = atoi(argv[10]);
    uint32_t transB = atoi(argv[11]);
    uint32_t ubMoveNum = 3584;
    uint32_t topK = 8;
    uint32_t maxOutputSize = m * topK * 2;

    uint32_t k2 = n / 2;
    uint32_t n2 = k;

    // m, n, k
    GemmCoord problemShape{m, n, k};
    size_t aSize = static_cast<size_t>(m) * k * sizeof(float16_t);
    size_t b1Size = static_cast<size_t>(k) * n * expertPerRank * sizeof(int8_t);
    size_t b2Size = static_cast<size_t>(k2) * n2 * expertPerRank * sizeof(int8_t);
    size_t cSize = static_cast<size_t>(m) * n2 * sizeof(float16_t);
    size_t dequantScale1Size = static_cast<size_t>(expertPerRank) * n * sizeof(int64_t);
    size_t dequantScale2Size = static_cast<size_t>(expertPerRank) * n2 * sizeof(int64_t);
    size_t perTokenScaleSize = static_cast<size_t>(maxOutputSize) * sizeof(float);
    size_t probsSize = m * topK * sizeof(float);
    size_t lenTokenPerExpert = EP * EP * expertPerRank * sizeof(int32_t);

    uint32_t aivNum = 2 * BLOCK_NUM;

    size_t workspaceSize = m * topK * sizeof(int32_t) +
                           EP * EP * expertPerRank * sizeof(int32_t) * 3 +
                           maxOutputSize * sizeof(float32_t) * 2 +
                           std::max(maxOutputSize * n * sizeof(float16_t), maxOutputSize * n2 * sizeof(float16_t)) +
                           std::max(maxOutputSize * k * sizeof(int8_t), maxOutputSize * k2 * sizeof(int8_t));

    uint8_t *aDevice;
    uint8_t *aHost;
    uint8_t *b1Device;
    uint8_t *b1Host;
    uint8_t *b2Device;
    uint8_t *b2Host;
    uint8_t *cDevice;
    uint8_t *cHost;
    uint8_t *scale1Device;
    uint8_t *scale1Host;
    uint8_t *scale2Device;
    uint8_t *scale2Host;
    uint8_t *ptrWorkspace;
    uint8_t *probsDevice;
    uint8_t *probsHost;

    std::string filePrefix;
    const char *env_var = std::getenv("INPUT_PATH");
    if (env_var) {
        filePrefix = env_var;
    } else {
        std::cout << "请设置input 文件路径: export INPUT_PATH =" << std::endl;
    }
    std::string fileSuffix =
            "_" + std::to_string(dataType) + "_1_" + std::to_string(m) + "_" + std::to_string(k) + "_" +
            std::to_string(n) + "_" + std::to_string(expertPerRank) + "_" + std::to_string(EP) + "_1.bin";

    InitData(&b1Host, &b1Device, b1Size, filePrefix + "matrix_b1_" + std::to_string(rankId) + fileSuffix);
    InitData(&b2Host, &b2Device, b2Size, filePrefix + "matrix_b2_" + std::to_string(rankId) + fileSuffix);
    InitData(&cHost, &cDevice, cSize);
    InitData(&scale1Host, &scale1Device, dequantScale1Size,
             filePrefix + "matrix_dequant_scale1_" + std::to_string(rankId) + fileSuffix);
    InitData(&scale2Host, &scale2Device, dequantScale2Size,
             filePrefix + "matrix_dequant_scale2_" + std::to_string(rankId) + fileSuffix);
    InitData(&probsHost, &probsDevice, probsSize, filePrefix + "probs" + fileSuffix);

    uint8_t *expertIdx;
    uint8_t *expertIdxHost;
    uint8_t *moeInitRoutingQuantV2Scale;
    uint8_t *moeInitRoutingQuantV2Offset;
    uint8_t *expandedX;
    uint8_t *expandedXHost;

    uint8_t *expertTokensBeforeCapacity;
    int64_t activeNum = 0;
    int64_t expertCapacity = 0;
    int64_t expertNum = expertPerRank * EP;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t quantMode = 1;
    std::string dispatchFileSuffix = "";
    InitData(&aHost, &aDevice, m * k * sizeof(float16_t),
             filePrefix + "matrix_a_" + std::to_string(rankId) + fileSuffix);
    InitData(&expertIdxHost, &expertIdx, m * topK * sizeof(int32_t),
             filePrefix + "expert_idx_" + std::to_string(rankId) + fileSuffix);

    moeInitRoutingQuantV2Scale = nullptr;
    moeInitRoutingQuantV2Offset = nullptr;
    expertTokensBeforeCapacity = nullptr;

    optiling::MoeInitRoutingQuantV2TilingBase moeInitRoutingQuantV2TilingBase;
    int64_t inuptXDtypeSize = sizeof(float16_t);
    int64_t scaleDim0 = 0;
    int64_t ubSize = 196352;
    moeInitRoutingQuantV2TilingBase.DoTiling(m, k, topK, expertCapacity, expertNum, activeNum, dropPadMode,
                                             expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
                                             inuptXDtypeSize, quantMode, scaleDim0, aivNum, ubSize);
    uint64_t initRoutingQuantTilingKey = moeInitRoutingQuantV2TilingBase.tilingKey_;
    size_t initRoutingWorkspace = moeInitRoutingQuantV2TilingBase.workspaceSize_;
    workspaceSize += initRoutingWorkspace;
    printf("!!!!!!!!!! initRoutingQuantTilingKey %lu\n\n", initRoutingQuantTilingKey);
    if (rankId == 0) {
        moeInitRoutingQuantV2TilingBase.ShowTilingData();
    }

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&ptrWorkspace), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    int32_t shmem_size = (504 * 1024 * 1024) * sizeof(__fp16);
    // shmem_malloc(): 从对称共享内存堆(Symmetric Heap)中分配指定大小的内存空间
    // 对称内存是指所有rank在相同偏移位置都能访问的共享内存区域，用于跨rank RDMA通信
    // 参数: shmem_size = (504 * 1024 * 1024) * sizeof(__fp16) = 约1008MB的fp16类型内存空间
    // 返回: 对称内存指针，所有rank都可以通过相同偏移访问该内存区域
    // 该内存用于存储DispatchGMM操作中的专家分发通信数据，实现跨rank的专家权重交换
    // 在MOE(Mixture of Experts)场景中，不同rank需要通过shmem访问其他rank上的专家权重矩阵
    void *symmPtr = shmem_malloc(shmem_size);
    uint8_t *symmetricPtr = (uint8_t *) symmPtr;

    CoCTiling cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    cocTiling.ubMoveNum = ubMoveNum;
    cocTiling.maxOutputSize = maxOutputSize;
    cocTiling.EP = EP;
    cocTiling.expertPerRank = expertPerRank;
    cocTiling.activeNum = activeNum;
    cocTiling.expertCapacity = expertCapacity;
    cocTiling.expertNum = expertNum;
    cocTiling.dropPadMode = dropPadMode;
    cocTiling.expertTokensCountOrCumsumFlag = expertTokensCountOrCumsumFlag;
    cocTiling.expertTokensBeforeCapacityFlag = expertTokensBeforeCapacityFlag;
    cocTiling.quantMode = quantMode;
    cocTiling.topK = topK;
    cocTiling.initRoutingQuantTilingKey = initRoutingQuantTilingKey;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    for (int i = 0; i < 1; ++i) {
        // shmemx_get_ffts_config(): 获取FFTS(Fast Flag Task Sync)硬件同步配置地址
        // FFTS是NPU核间快速同步机制，用于在kernel执行时实现核间的轻量级同步操作
        // 返回: FFTS配置寄存器的物理地址，传递给kernel用于设置同步基址
        // kernel内部会使用此地址进行MOE专家分发操作的核间同步和通信协调
        uint64_t fftsAddr = shmemx_get_ffts_config();
        ACL_CHECK(aclrtMemcpy(b1Device, b1Size, b1Host, b1Size, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(b2Device, b2Size, b2Host, b2Size, ACL_MEMCPY_HOST_TO_DEVICE));
        DispatchGMM<<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, problemShape, aDevice, b1Device, b2Device,
                cDevice, scale1Device, scale2Device, symmetricPtr,
                expertIdx, moeInitRoutingQuantV2Scale, moeInitRoutingQuantV2Offset,
                expertTokensBeforeCapacity, probsDevice,
                ptrWorkspace, cocTiling, 0, moeInitRoutingQuantV2TilingBase.quantTilingData);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./out/output_" + std::to_string(rankId) + ".bin", cHost, cSize);
    if (rankId == 0) {
        std::printf("\ntest finished\n");
    }
    // shmem_free(): 释放之前通过shmem_malloc()分配的对称共享内存空间
    // 参数: symmPtr - 要释放的对称内存指针
    // 此函数会将内存归还到对称内存堆，供后续shmem_malloc调用重新使用
    // 注意：释放后不应再访问该内存区域，否则会导致未定义行为或数据损坏
    shmem_free(symmPtr);
    ACL_CHECK(aclrtFreeHost(b1Host));
    ACL_CHECK(aclrtFreeHost(b2Host));
    ACL_CHECK(aclrtFreeHost(cHost));
    ACL_CHECK(aclrtFree(b1Device));
    ACL_CHECK(aclrtFree(b2Device));
    ACL_CHECK(aclrtFree(cDevice));
    ACL_CHECK(aclrtFreeHost(expertIdxHost));
    ACL_CHECK(aclrtFree(expertIdx));

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
