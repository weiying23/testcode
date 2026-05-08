/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * MTE vs SDMA 跨卡带宽测试 Kernel
 *
 * 功能说明：
 *   对比 MTE 和 SDMA 两种通信引擎在跨卡（同节点不同 NPU）数据传输时的带宽性能。
 *   - MTE: GM → UB → GM 路径，数据经过 UB 缓冲区中转
 *   - SDMA: GM → GM 直接传输，无需 UB 中转
 *
 * 测试方法：
 *   Rank 0 发送数据到 Rank 1，测量发送带宽。
 *   使用 host 端 chrono 计时，避免 profiling 系统的 overflow 问题。
 */

#include "kernel_operator.h"
#include "shmem.h"

// ============================================================================
// MTE 带宽测试 Kernel
// ============================================================================
//
// MTE (Memory Transfer Engine) 数据路径：GM → UB → GM
//   1. 源数据从 GM（Global Memory）搬运到 UB（Unified Buffer）
//   2. 从 UB 经过 MTE 硬件搬运到目标 NPU 的 GM
//
// API: aclshmemx_mte_put_nbi (非阻塞单边写)
//   - 参数：dst_gm, src_gm, ub_buf, ub_size, count, peer_pe, event_id
//   - 需要提供 UB 缓冲区用于数据中转
//   - 完成同步：SetFlag/WaitFlag 等待 MTE3_S 事件
//
// Kernel 逻辑：
//   1. 只有 Rank 0 执行发送（单边操作，接收方被动）
//   2. 每个 AI Core 处理 msg_size/block_dim 的数据切片
//   3. 循环 iterations 次 put_nbi 调用
//   4. Core 0 负责等待所有 DMA 完成
//   5. 全局 barrier 确保所有 core 结束
// ============================================================================
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void mte_bw_kernel(GM_ADDR gva,        // 对称内存起始地址
                   int64_t msg_size,   // 单次传输的数据量（字节）
                   int64_t iterations, // 循环次数
                   int64_t block_dim)  // AI Core 数量
{
    // 获取当前 PE 编号（0 或 1）和 AI Core 编号
    int64_t rank     = aclshmem_my_pe();      // PE 编号
    int64_t core_idx = AscendC::GetBlockIdx(); // AI Core 编号（0~31）
    uint32_t peer    = 1 - (uint32_t)rank;     // 对端 PE 编号

    // Rank 1 不执行发送，直接等待 barrier
    // PUT 是单边操作：发送方主动写，接收方被动接受
    if (rank != 0) {
        aclshmemx_barrier_all_vec();
        return;
    }

    // ========== 从 device state 获取 MTE 配置参数 ==========
    // MTE 需要以下硬件配置：
    //   - copy_ub:   UB 缓冲区地址（数据中转站）
    //   - copy_size: UB 缓冲区大小（通常是 16KB）
    //   - sync_id:   同步事件 ID（用于等待 DMA 完成）
    __gm__ aclshmem_device_host_state_t *st = aclshmemi_get_state();
    uint64_t copy_ub   = st->mte_config.aclshmem_ub;   // UB 地址
    uint32_t copy_size = st->mte_config.ub_size;       // UB 大小
    AscendC::TEventID ev = (AscendC::TEventID)st->mte_config.sync_id; // 事件 ID

    // ========== 计算每个 Core 处理的数据切片 ==========
    // 总数据量按 block_dim 平分给各个 AI Core
    int64_t slice  = msg_size / block_dim;            // 每个 Core 的数据量
    int64_t offset = core_idx * slice;                // 该 Core 的偏移量
    __gm__ uint8_t *src = (__gm__ uint8_t *)gva + offset;  // 源地址（本 NPU）
    __gm__ uint8_t *dst = (__gm__ uint8_t *)gva + offset;  // 目标地址（对端 NPU）

    // 管道屏障：确保前面的操作完成
    AscendC::PipeBarrier<PIPE_ALL>();

    // ========== 执行 iterations 次 MTE PUT ==========
    // aclshmemx_mte_put_nbi 流程：
    //   1. GM → UB: MTE2 通道搬运源数据到 UB
    //   2. UB → GM: MTE3 通道搬运数据到目标 NPU 的 GM
    // 非阻塞调用，DMA 在后台执行
    for (int64_t i = 0; i < iterations; i++) {
        aclshmemx_mte_put_nbi(dst, src,
                              reinterpret_cast<__ubuf__ uint8_t *>(copy_ub),
                              copy_size,               // UB 大小
                              (int32_t)slice,          // 传输字节数
                              (int32_t)peer,           // 目标 PE
                              ev);                     // 事件 ID
    }

    // ========== 等待所有 DMA 完成 ==========
    // SyncAll: 所有 AI Core 同步
    // SetFlag/WaitFlag: Core 0 等待 MTE3_S 事件（DMA 完成信号）
    AscendC::SyncAll();
    if (core_idx == 0) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(ev);   // 设置等待事件
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(ev);  // 等待 DMA 完成
    }

    // 全局 barrier：确保所有 Core 都完成
    aclshmemx_barrier_all_vec();
}

// ============================================================================
// SDMA 带宽测试 Kernel
// ============================================================================
//
// SDMA (System DMA) 数据路径：GM → GM 直接传输
//   - 不经过 UB 缓冲区，直接从源 GM 到目标 GM
//   - 延迟更低，适合中小数据量传输
//
// API: aclshmemx_sdma_put_nbi (非阻塞单边写)
//   - 参数：dst_gm, src_gm, ub_buf, ub_size, count, peer_pe, event_id
//   - 虽然 API 有 ub_buf 参数，但 SDMA 不实际使用 UB 做数据中转
//   - ub_buf 仅用于内部状态管理
//   - 完成同步：aclshmemx_sdma_quiet 或 PipeBarrier
//
// Kernel 逻辑：
//   1. 只有 Rank 0 执行发送
//   2. 每个 Core 处理 msg_size/block_dim 的数据切片
//   3. 循环 iterations 次 put_nbi
//   4. 调用 sdma_quiet 等待所有 DMA 完成
//   5. 全局 barrier
// ============================================================================
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void sdma_bw_kernel(GM_ADDR gva,        // 对称内存起始地址
                    int64_t msg_size,   // 单次传输的数据量
                    int64_t iterations, // 循环次数
                    int64_t block_dim)  // AI Core 数量
{
    int64_t rank     = aclshmem_my_pe();
    int64_t core_idx = AscendC::GetBlockIdx();
    uint32_t peer    = 1 - (uint32_t)rank;

    // PUT 是单边操作，Rank 1 被动接收
    if (rank != 0) {
        aclshmemx_barrier_all_vec();
        return;
    }

    // ========== SDMA 配置参数 ==========
    // SDMA API 需要 ub_buf 参数，但实际不用于数据中转
    // 这里用固定的 UB 偏移和大小（与原 comm_test 保持一致）
    constexpr uint32_t UB_OFFSET  = 1024;         // UB 偏移地址
    constexpr uint32_t SDMA_UB_SZ = 16 * 1024;    // 16KB（仅用于 API 参数）
    __ubuf__ uint8_t *tmp = reinterpret_cast<__ubuf__ uint8_t *>(UB_OFFSET);

    // ========== 计算数据切片 ==========
    int64_t slice  = msg_size / block_dim;
    int64_t offset = core_idx * slice;

    // 数据太小，该 Core 无需处理
    if (slice == 0) {
        aclshmemx_barrier_all_vec();
        return;
    }

    __gm__ uint8_t *src = (__gm__ uint8_t *)gva + offset;
    __gm__ uint8_t *dst = (__gm__ uint8_t *)gva + offset;

    AscendC::PipeBarrier<PIPE_ALL>();

    // ========== 执行 iterations 次 SDMA PUT ==========
    // aclshmemx_sdma_put_nbi：直接 GM → GM DMA
    for (int64_t i = 0; i < iterations; i++) {
        aclshmemx_sdma_put_nbi(dst, src, tmp, SDMA_UB_SZ,
                               (uint64_t)slice,        // 传输字节数
                               (int32_t)peer,          // 目标 PE
                               EVENT_ID0);             // 事件 ID
    }

    // ========== 等待所有 DMA 完成 ==========
    // aclshmemx_sdma_quiet：阻塞等待指定 event 的所有 DMA 完成
    aclshmemx_sdma_quiet(tmp, SDMA_UB_SZ, EVENT_ID0);

    aclshmemx_barrier_all_vec();
}

// ============================================================================
// Host 端 Kernel 启动函数
// ============================================================================
// 这些函数由 main.cpp 调用，负责启动上述 kernel
extern "C" void launch_mte_bw(uint32_t bdim, void *stream, uint8_t *gva,
                               int64_t msg_size, int64_t iters)
{
    mte_bw_kernel<<<bdim, nullptr, stream>>>(gva, msg_size, iters, (int64_t)bdim);
}

extern "C" void launch_sdma_bw(uint32_t bdim, void *stream, uint8_t *gva,
                                int64_t msg_size, int64_t iters)
{
    sdma_bw_kernel<<<bdim, nullptr, stream>>>(gva, msg_size, iters, (int64_t)bdim);
}