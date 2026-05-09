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
// 【为什么用 MTE】
// MTE (Memory Transfer Engine) 是 Ascend AI Core 内置的数据搬运引擎：
//   - 硬件位置：每个 AI Core 内部
//   - 设计目的：在 AI Core 和外部内存（GM/UB）之间搬运数据
//   - 跨卡通信时：需要把数据从本 NPU 的 GM 搬到 UB，再通过 HCCS 发送到目标 NPU
//
// 【MTE 数据路径：GM → UB → GM】
// 为什么需要 UB 中转？
//   - 原因 1：AI Core 只能直接访问自己的 UB，不能直接访问其他 NPU 的 GM
//   - 原因 2：跨卡通信需要通过 HCCS 链路，HCCS 连接的是 UB 级别的接口
//   - 原因 3：MTE 硬件设计就是把 UB 作为数据搬运的"站台"
//
// 流程：
//   1. MTE2 通道：GM → UB（把源数据从全局内存搬到 UB）
//   2. HCCS 链路：UB → 目标 NPU（跨卡传输）
//   3. MTE3 通道：数据到达目标 NPU 的 GM
//
// 【API: aclshmemx_mte_put_nbi】
// 为什么用非阻塞版本？
//   - nbi = non-blocking immediate
//   - 阻塞版本会等待每次传输完成，无法流水线
//   - 非阻塞版本可以连续发起多个 DMA，让硬件并行处理
//   - 最后用 quiet/wait 等待所有 DMA 完成，效率更高
//
// 参数说明：
//   - dst_gm: 目标地址（对端 NPU 的对称内存偏移）
//   - src_gm: 源地址（本 NPU 的对称内存偏移）
//   - ub_buf: UB 缓冲区地址（为什么必须有：数据必须先到 UB 才能跨卡）
//   - ub_size: UB 大小（为什么重要：决定了每次搬运的最大 chunk）
//   - count: 传输字节数（可以大于 ub_size，会自动分多次搬运）
//   - peer_pe: 目标 PE 编号
//   - event_id: 同步事件 ID（用于等待 DMA 完成）
//
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void mte_bw_kernel(GM_ADDR gva,        // 对称内存起始地址
                   int64_t msg_size,   // 单次传输的数据量（字节）
                   int64_t iterations, // 循环次数
                   int64_t block_dim)  // AI Core 数量
{
    int64_t rank     = aclshmem_my_pe();      // PE 编号
    int64_t core_idx = AscendC::GetBlockIdx(); // AI Core 编号（0~31）
    uint32_t peer    = 1 - (uint32_t)rank;     // 对端 PE 编号

    // 【为什么只有 Rank 0 发送】
    // PUT 是单边操作（one-sided operation）：
    //   - 发送方主动：调用 put_nbi 把数据写到接收方的内存
    //   - 接收方被动：不需要执行任何代码，数据自动到达其内存
    //   - 带宽测试测的是发送方的发送能力，所以只需要 Rank 0 执行
    //   - Rank 1 只需要 barrier 等待，确保 Rank 0 的数据确实写到了自己的内存
    if (rank != 0) {
        aclshmemx_barrier_all_vec();
        return;
    }

    // 【为什么从 device_state 获取配置】
    // MTE 硬件配置不是硬编码的，而是由 SHMEM 初始化时根据硬件情况动态设置：
    //   - copy_ub: UB 地址是在初始化时从可用 UB 空间中分配的
    //   - copy_size: UB 大小根据芯片型号和配置决定（通常 16KB）
    //   - sync_id: 事件 ID 由 SHMEM 库管理，避免冲突
    // 如果硬编码这些值，可能在不同硬件配置下不工作
    __gm__ aclshmem_device_host_state_t *st = aclshmemi_get_state();
    uint64_t copy_ub   = st->mte_config.aclshmem_ub;
    uint32_t copy_size = st->mte_config.ub_size;
    AscendC::TEventID ev = (AscendC::TEventID)st->mte_config.sync_id;

    // 【为什么按 block_dim 切分数据】
    // 一个 kernel 启动 block_dim 个 AI Core（这里默认 32）：
    //   - 每个 Core 独立执行，并行处理不同的数据切片
    //   - 总带宽 = 所有 Core 的带宽之和
    //   - 如果不分片，只有一个 Core 处理所有数据，无法发挥并行优势
    int64_t slice  = msg_size / block_dim;
    int64_t offset = core_idx * slice;
    __gm__ uint8_t *src = (__gm__ uint8_t *)gva + offset;
    __gm__ uint8_t *dst = (__gm__ uint8_t *)gva + offset;

    // 【为什么需要 PipeBarrier<PIPE_ALL>】
    // Ascend AI Core 有多条并行管道（PIPE_MTE2, PIPE_MTE3, PIPE_S, PIPE_V 等）
    //   - 不同管道可以并行执行
    //   - 但某些操作需要所有管道都完成后才能继续
    //   - PipeBarrier<PIPE_ALL> 确保所有管道的前面操作都完成
    //   - 防止：前面未完成的操作干扰后面的 DMA
    AscendC::PipeBarrier<PIPE_ALL>();

    // 【为什么循环 iterations 次】
    // 带宽测试需要多次传输才能得到稳定的结果：
    //   - 单次传输时间太短，计时精度不够
    //   - 多次传输可以平均掉硬件抖动
    //   - iterations 根据消息大小调整（大消息次数少，小消息次数多）
    for (int64_t i = 0; i < iterations; i++) {
        aclshmemx_mte_put_nbi(dst, src,
                              reinterpret_cast<__ubuf__ uint8_t *>(copy_ub),
                              copy_size,
                              (int32_t)slice,
                              (int32_t)peer,
                              ev);
    }

    // 【为什么需要 SyncAll + SetFlag/WaitFlag】
    // iterations 次 put_nbi 都是异步的，DMA 还在后台执行：
    //   - SyncAll：所有 AI Core 同步到这里（确保所有 Core 都发完请求）
    //   - SetFlag/WaitFlag：等待 DMA 真正完成
    //   - 为什么只有 Core 0 做：event_id 是共享的，一个 Core 等待就够了
    //   - 如果不等待：DMA 可能还没完成，数据还没到目标，后续操作会出错
    AscendC::SyncAll();
    if (core_idx == 0) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(ev);
    }

    // 【为什么需要 barrier_all_vec】
    // 确保所有 AI Core 都到达这里：
    //   - 某些 Core 可能处理更快，提前结束
    //   - barrier 让它们都等待，直到所有 Core 完成
    //   - 这样 kernel 返回时，所有 Core 的数据都已经传输完成
    aclshmemx_barrier_all_vec();
}

// ============================================================================
// SDMA 带宽测试 Kernel
// ============================================================================
//
// 【为什么用 SDMA】
// SDMA (System DMA) 是系统级 DMA 引擎：
//   - 硬件位置：NPU 芯片级别，不在 AI Core 内部
//   - 设计目的：在 NPU 之间直接搬运数据，无需经过 AI Core 的 UB
//   - 优势：减少一次数据搬运步骤，延迟更低
//   - 适用场景：中小数据量（UB 中转的开销占主导时）
//
// 【SDMA 数据路径：GM → GM】
// 为什么不需要 UB 中转？
//   - SDMA 硬件直接连接不同 NPU 的 GM
//   - 数据路径：本 NPU GM → HCCS → 目标 NPU GM
//   - 没有中间的 UB 站台，一步到位
//
// 【API 有 ub_buf 参数但不用于数据中转】
// aclshmemx_sdma_put_nbi(dst, src, ub_buf, ub_size, count, peer_pe, event_id)
// 为什么 API 设计成有 ub_buf？
//   - 原因：API 统一设计，MTE 和 SDMA 用相同的参数接口
//   - SDMA 内部：ub_buf 只用于存储 DMA 请求的状态信息，不是数据缓冲区
//   - 实际传输：数据直接从 src_gm 到 dst_gm，不经过 ub_buf
//
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void sdma_bw_kernel(GM_ADDR gva,        // 对称内存起始地址
                    int64_t msg_size,   // 单次传输的数据量
                    int64_t iterations, // 循环次数
                    int64_t block_dim)  // AI Core 数量
{
    int64_t rank     = aclshmem_my_pe();
    int64_t core_idx = AscendC::GetBlockIdx();
    uint32_t peer    = 1 - (uint32_t)rank;

    // 【为什么只有 Rank 0 发送】
    // 同 MTE：PUT 是单边操作，接收方被动
    if (rank != 0) {
        aclshmemx_barrier_all_vec();
        return;
    }

    // 【为什么硬编码 UB 参数】
    // SDMA 不从 device_state 获取配置，因为：
    //   - SDMA 不需要 UB 地址来做数据中转
    //   - ub_buf 只是 API 参数，传什么值都可以（只要地址有效）
    //   - 用固定值简化代码，不需要额外的配置查询
    constexpr uint32_t UB_OFFSET  = 1024;
    constexpr uint32_t SDMA_UB_SZ = 64;
    __ubuf__ uint8_t *tmp = reinterpret_cast<__ubuf__ uint8_t *>(UB_OFFSET);

    // 【数据切片逻辑】
    // 同 MTE：按 block_dim 平分给各个 AI Core
    int64_t slice  = msg_size / block_dim;
    int64_t offset = core_idx * slice;

    // 【为什么检查 slice == 0】
    // 消息大小可能小于 block_dim：
    //   - 例如 msg_size=4KB, block_dim=32 → slice=128B
    //   - 但如果 msg_size < block_dim，某个 Core 的 slice=0
    //   - slice=0 的 Core 无法发起有效的 DMA，直接退出
    if (slice == 0) {
        aclshmemx_barrier_all_vec();
        return;
    }

    __gm__ uint8_t *src = (__gm__ uint8_t *)gva + offset;
    __gm__ uint8_t *dst = (__gm__ uint8_t *)gva + offset;

    AscendC::PipeBarrier<PIPE_ALL>();

    // 【循环 iterations 次】
    // 同 MTE：多次传输稳定带宽测量
    for (int64_t i = 0; i < iterations; i++) {
        aclshmemx_sdma_put_nbi(dst, src, tmp, SDMA_UB_SZ,
                               (uint64_t)slice,
                               (int32_t)peer,
                               EVENT_ID0);
    }

    // 【为什么用 sdma_quiet 而不是 SetFlag/WaitFlag】
    // SDMA 完成同步方式不同：
    //   - MTE：用 event flag 同步（因为 DMA 在 AI Core 内部的 MTE 管道）
    //   - SDMA：用专门的 quiet API（因为 DMA 在芯片级别的 SDMA 引擎）
    //   - quiet 会阻塞等待所有发出去的 SDMA 请求完成
    aclshmemx_sdma_quiet(tmp, SDMA_UB_SZ, EVENT_ID0);

    aclshmemx_barrier_all_vec();
}

// ============================================================================
// Host 端 Kernel 启动函数
// ============================================================================
// 【为什么需要单独的 launch 函数】
// Kernel 定义在 .cpp 文件中（fusion 模式），Host 调用需要 extern "C" 接口：
//   - Ascend kernel 编译：kernel<<<>>>语法只能在同一个编译单元内使用
//   - Host 和 kernel 分开编译：kernel 在 device 侧，host 在 host 侧
//   - launch 函数：作为桥梁，host 调用 launch，launch 启动 kernel
//   - extern "C"：确保 C 链接，避免 C++ name mangling 导致找不到符号
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
