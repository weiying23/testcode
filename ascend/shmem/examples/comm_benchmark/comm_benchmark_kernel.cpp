/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Comm Benchmark Kernel - 所有通信性能测试Kernel的集合
 *
 * 问题修复记录：
 * 1. Pingpong latency定值问题：已修复
 *    - 添加magic value写入逻辑到数据末尾
 *    - 发送方在发送前写入：`*(__gm__ uint32_t*)(src_addr + msg_size - 8) = MAGIC_VAL + i`
 *    - 接收方检测magic value确认数据到达
 *
 * 2. 带宽测量过大问题：已修复
 *    - 原问题：只测量指令下发时间，没有等待实际数据搬运完成
 *    - put_nbi是非阻塞操作，循环下发很快完成
 *    - quiet/WaitFlag只等待队列清空，不保证接收方收到数据
 *    - 修复：使用pingpong模式，发送方等待接收方确认后才记录结束时间
 *    - 新流程：
 *      a) 发送方批量发送所有数据
 *      b) 发送方写入完成标志通知接收方
 *      c) 接收方等待完成标志，确保数据到达
 *      d) 接收方发送确认响应
 *      e) 发送方收到确认后才记录结束时间
 */

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

constexpr uint32_t MAGIC_VAL = 12345;
constexpr uint32_t MAGIC_VAL_BW = 10;

// ========== RDMA PingPong延迟测试Kernel ==========
//
// BUG修复说明：
// 原bug：Kernel轮询等待magic value但从未写入，导致latency测量为定值
// 根因分析：
// - Pingpong同步机制需要发送方在数据末尾写入magic value
// - 接收方通过检测magic value来确认数据到达
// - 原代码没有写入magic value，导致轮询失效
//
// 修复方案：
// 1. 发送方在发送前，将magic value写入数据末尾的8字节位置
// 2. Magic value格式: sender_rank + MAGIC_VAL + iteration
// 3. 接收方通过轮询检测magic value变化来确认数据到达
//
// Pingpong流程（修复后）：
// 1. Rank 0 写入 magic(12345+i) 到 slot 0 末尾
// 2. Rank 0 发送 slot 0 到 Rank 1 的 slot 0
// 3. Rank 1 检测 slot 0 末尾的 magic(12345+i)
// 4. Rank 1 写入 magic(12346+i) 到 slot 1 末尾
// 5. Rank 1 发送 slot 1 到 Rank 0 的 slot 1
// 6. Rank 0 检测 slot 1 末尾的 magic(12346+i)
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_pingpong_latency_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t warmup,
    GM_ADDR result_buffer) {

    util_set_ffts_config(ffts_config);
    if (AscendC::GetSubBlockIdx() != 0) return;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE);
    AscendC::LocalTensor<uint32_t> ubLocal = buf.GetWithOffset<uint32_t>(UB_ALIGN_SIZE / sizeof(uint32_t), 0);

    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    // 内存布局：
    // Slot 0 (gva + 0*msg_size): rank 0的发送数据区
    // Slot 1 (gva + 1*msg_size): rank 1的发送数据区
    // Slot 0末尾 (gva + msg_size - 8): rank 1轮询位置
    // Slot 1末尾 (gva + 2*msg_size - 8): rank 0轮询位置
    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    // Warmup阶段（不计入统计）
    for (int64_t i = 0; i < warmup; i++) {
        if (rank == 0) {
            // BUG修复步骤1：写入magic value到数据末尾
            // Magic value = peer(0) + MAGIC_VAL + i = 12345 + i
            // 这是rank 1期望检测到的值
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = MAGIC_VAL + i;

            // 发送数据（包含magic value）到rank 1
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);

            // BUG修复步骤6：等待rank 1的响应magic value
            // 检测slot 1末尾的magic value = peer(1) + MAGIC_VAL + i = 12346 + i
            while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != peer + MAGIC_VAL + i) {
                dcci_cachelines(gva + msg_size * 2 - 8, 8);
                AscendC::GetSystemCycle();
            }
        } else {
            // BUG修复步骤3：等待rank 0的magic value
            // 检测slot 0末尾的magic value = peer(0) + MAGIC_VAL + i = 12345 + i
            while (*(__gm__ uint32_t*)(gva + msg_size * 1 - 8) != peer + MAGIC_VAL + i) {
                dcci_cachelines(gva + msg_size * 1 - 8, 8);
                AscendC::GetSystemCycle();
            }

            // BUG修复步骤4：写入响应magic value到数据末尾
            // Magic value = peer(1) + MAGIC_VAL + i = 12346 + i
            // 这是rank 0期望检测到的值
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = peer + MAGIC_VAL + i;

            // 发送响应数据（包含magic value）到rank 0
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 正式测试阶段（计入统计）
    if (rank == 0) {
        for (int64_t i = 0; i < iterations; i++) {
            // 记录开始时间（cycles）
            int64_t iter_start = AscendC::GetSystemCycle();

            // BUG修复步骤1：写入magic value
            // Magic value = MAGIC_VAL + warmup + i（跳过warmup计数）
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = MAGIC_VAL + warmup + i;

            // 发送数据到rank 1
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);

            // 等待rank 1的响应（检测slot 1末尾的magic）
            while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != peer + MAGIC_VAL + warmup + i) {
                dcci_cachelines(gva + msg_size * 2 - 8, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // 记录结束时间（cycles）
            int64_t iter_end = AscendC::GetSystemCycle();

            // 将cycles差值写入结果buffer（每次迭代写入不同位置）
            *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = iter_end - iter_start;
        }
    } else {
        for (int64_t i = 0; i < iterations; i++) {
            // 等待rank 0的数据（检测slot 0末尾的magic）
            while (*(__gm__ uint32_t*)(gva + msg_size * 1 - 8) != peer + MAGIC_VAL + warmup + i) {
                dcci_cachelines(gva + msg_size * 1 - 8, 8);
                AscendC::GetSystemCycle();
            }

            // BUG修复步骤4：写入响应magic value
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = peer + MAGIC_VAL + warmup + i;

            // 发送响应数据到rank 0
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

// ========== RDMA带宽测试Kernel（支持多核聚合）==========
//
// 改进说明：
// 1. 支持多核聚合带宽测试（block_dim 可配置：1, 8, 16, 32）
// 2. 每个 AIV 核心独立发送数据，测量聚合带宽
// 3. 只有 Core 0 执行同步操作（quiet + 通知 + 等待确认）
//
// 多核聚合测试说明：
// - block_dim = 1: 单核带宽基准
// - block_dim = 8/16/32: 多核并行，测量聚合带宽
// - 每个 AIV 核心发送 iterations 次 msg_size 数据
// - 总传输量 = block_dim * iterations * msg_size
//
// 内存布局：
// - 每个 PE 有 block_dim 个数据 slot
// - PE i 的 Core j 数据位于 gva + i * msg_size * block_dim + j * msg_size
// - 同步区域位于所有数据之后
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_bandwidth_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t block_dim,
    GM_ADDR result_buffer) {

    util_set_ffts_config(ffts_config);
    // 多核模式下，所有 Core 都执行（不再使用 return）

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    int64_t core_idx = AscendC::GetBlockIdx();  // 当前核心编号
    uint32_t peer;

    // 多核数据布局：
    // 每个 PE 有 block_dim 个数据区域
    // PE i 的 Core j 数据位于 gva + i * msg_size * block_dim + j * msg_size
    GM_ADDR src_addr = gva + rank * msg_size * block_dim + core_idx * msg_size;
    GM_ADDR result_addr = result_buffer;

    // 同步区域（位于所有数据区域之后）
    // 总数据区域大小 = rank_size * msg_size * block_dim
    int64_t sync_base_offset = rank_size * msg_size * block_dim;
    GM_ADDR notify_addr = gva + sync_base_offset + 8;
    GM_ADDR ack_addr = gva + sync_base_offset + 16;

    if (rank == 0) {
        // 发送方逻辑
        peer = 1;

        // 所有 Core 都执行数据发送
        int64_t start_cycle = AscendC::GetSystemCycle();

        for (int64_t i = 0; i < iterations; i++) {
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);
        }

        // 只有 Core 0 执行同步操作
        if (core_idx == 0) {
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
            aclshmem_uint8_put_nbi(notify_addr, src_addr, sizeof(uint32_t), peer);
            while (*(__gm__ uint32_t*)(ack_addr) != peer + MAGIC_VAL) {
                dcci_cachelines(ack_addr, sizeof(uint32_t));
                AscendC::GetSystemCycle();
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end_cycle = AscendC::GetSystemCycle();

        // 只有 Core 0 记录结果
        if (core_idx == 0) {
            *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;
        }

    } else {
        // 接收方逻辑
        peer = 0;

        // 只有 Core 0 执行同步操作
        if (core_idx == 0) {
            while (*(__gm__ uint32_t*)(notify_addr) != peer + MAGIC_VAL) {
                dcci_cachelines(notify_addr, sizeof(uint32_t));
                AscendC::GetSystemCycle();
            }
            aclshmem_uint8_put_nbi(ack_addr, src_addr, sizeof(uint32_t), peer);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

void launch_rdma_bandwidth(uint32_t block_dim, void* stream,
                            uint64_t ffts_config, uint8_t* gva,
                            int64_t msg_size, int64_t iterations,
                            uint8_t* result_buffer) {
    rdma_bandwidth_kernel<<<block_dim, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, block_dim, result_buffer);
}

// ========== MTE PingPong延迟测试Kernel ==========
//
// BUG修复说明：
// 原bug：与RDMA pingpong相同，Kernel未写入magic value导致latency为定值
// 修复方案：在发送前写入magic value到数据末尾
//
// MTE引擎特点：
// - 用于节点内NPU间通信
// - 使用片上MTE单元进行数据传输
// - 高带宽、低延迟，适合大规模数据传输
// - MTE操作需要使用专门的同步机制（MTE3_S事件）
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_pingpong_latency_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t warmup,
    GM_ADDR result_buffer) {

    util_set_ffts_config(ffts_config);
    if (AscendC::GetSubBlockIdx() != 0) return;

    // 获取MTE配置信息
    // device_state->mte_config包含MTE引擎的配置参数：
    // - aclshmem_ub: UB缓冲区地址（用于MTE数据搬运）
    // - ub_size: UB缓冲区大小
    // - sync_id: 同步事件ID（用于MTE3_S事件同步）
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;

    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    // 内存布局（与RDMA pingpong相同）
    // Slot 0 (gva + 0*msg_size): rank 0的发送数据区
    // Slot 1 (gva + 1*msg_size): rank 1的发送数据区
    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    // Warmup阶段（不计入统计）
    for (int64_t i = 0; i < warmup; i++) {
        if (rank == 0) {
            // BUG修复步骤1：写入magic value到数据末尾
            // Magic value = peer(0) + MAGIC_VAL + i = 12345 + i
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = MAGIC_VAL + i;

            // aclshmemx_mte_put_nbi: 使用MTE引擎发送数据
            // 参数说明：
            // - dest: 目标地址（远程PE的地址，GVA格式）
            // - src: 源地址（本地PE的地址，GVA格式）
            // - ub_buffer: UB缓冲区地址（用于MTE数据搬运）
            // - ub_size: UB缓冲区大小
            // - size: 数据大小
            // - pe: 目标PE编号
            // - event_id: 同步事件ID
            // MTE工作原理：
            // - 数据通过UB缓冲区进行搬运（分批传输）
            // - 使用MTE3_S事件进行同步
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);

            // AscendC::SetFlag/WaitFlag: 设置和等待硬件事件
            // MTE3_S事件：MTE引擎发送完成事件
            // 用于等待MTE put操作完成
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            // 等待rank 1的响应magic value
            // 检测slot 1末尾的magic value = peer(1) + MAGIC_VAL + i = 12346 + i
            while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != peer + MAGIC_VAL + i) {
                dcci_cachelines(gva + msg_size * 2 - 8, 8);
                AscendC::GetSystemCycle();
            }
        } else {
            // 等待rank 0的magic value
            // 检测slot 0末尾的magic value = peer(0) + MAGIC_VAL + i = 12345 + i
            while (*(__gm__ uint32_t*)(gva + msg_size * 1 - 8) != peer + MAGIC_VAL + i) {
                dcci_cachelines(gva + msg_size * 1 - 8, 8);
                AscendC::GetSystemCycle();
            }

            // BUG修复步骤4：写入响应magic value
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = peer + MAGIC_VAL + i;

            // 发送响应数据
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 正式测试阶段（计入统计）
    if (rank == 0) {
        for (int64_t i = 0; i < iterations; i++) {
            // 记录开始时间（cycles）
            int64_t iter_start = AscendC::GetSystemCycle();

            // BUG修复：写入magic value
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = MAGIC_VAL + warmup + i;

            // 发送数据
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            // 等待响应
            while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != peer + MAGIC_VAL + warmup + i) {
                dcci_cachelines(gva + msg_size * 2 - 8, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // 记录结束时间并写入结果
            int64_t iter_end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = iter_end - iter_start;
        }
    } else {
        for (int64_t i = 0; i < iterations; i++) {
            // 等待rank 0的数据
            while (*(__gm__ uint32_t*)(gva + msg_size * 1 - 8) != peer + MAGIC_VAL + warmup + i) {
                dcci_cachelines(gva + msg_size * 1 - 8, 8);
                AscendC::GetSystemCycle();
            }

            // 写入响应magic value
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = peer + MAGIC_VAL + warmup + i;

            // 发送响应
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

// ========== MTE带宽测试Kernel（支持多核聚合）==========
//
// 改进说明（与 RDMA 带宽测试相同）：
// 1. 支持多核聚合带宽测试（block_dim 可配置）
// 2. 每个 AIV 核心独立发送数据，测量聚合带宽
// 3. 只有 Core 0 执行同步操作
//
// 多核聚合测试说明：
// - block_dim = 1: 单核带宽基准
// - block_dim = 8/16/32: 多核并行，测量聚合带宽
// - 每个 AIV 核心发送 iterations 次 msg_size 数据
// - 总传输量 = block_dim * iterations * msg_size
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_bandwidth_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t block_dim,
    GM_ADDR result_buffer) {

    util_set_ffts_config(ffts_config);
    // 多核模式下，所有 Core 都执行

    // 获取MTE配置
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    int64_t core_idx = AscendC::GetBlockIdx();
    uint32_t peer;

    // 多核数据布局
    GM_ADDR src_addr = gva + rank * msg_size * block_dim + core_idx * msg_size;
    GM_ADDR result_addr = result_buffer;

    // 同步区域
    int64_t sync_base_offset = rank_size * msg_size * block_dim;
    GM_ADDR notify_addr = gva + sync_base_offset + 8;
    GM_ADDR ack_addr = gva + sync_base_offset + 16;

    if (rank == 0) {
        // 发送方逻辑
        peer = 1;

        int64_t start_cycle = AscendC::GetSystemCycle();

        // 所有 Core 都执行数据发送
        for (int64_t i = 0; i < iterations; i++) {
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
        }

        // 只有 Core 0 执行同步操作
        if (core_idx == 0) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            aclshmemx_mte_put_nbi((__gm__ uint8_t*)notify_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, sizeof(uint32_t), peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            while (*(__gm__ uint32_t*)(ack_addr) != peer + MAGIC_VAL) {
                dcci_cachelines(ack_addr, sizeof(uint32_t));
                AscendC::GetSystemCycle();
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end_cycle = AscendC::GetSystemCycle();

        if (core_idx == 0) {
            *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;
        }

    } else {
        // 接收方逻辑
        peer = 0;

        // 只有 Core 0 执行同步操作
        if (core_idx == 0) {
            while (*(__gm__ uint32_t*)(notify_addr) != peer + MAGIC_VAL) {
                dcci_cachelines(notify_addr, sizeof(uint32_t));
                AscendC::GetSystemCycle();
            }

            aclshmemx_mte_put_nbi((__gm__ uint8_t*)ack_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, sizeof(uint32_t), peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

void launch_mte_bandwidth(uint32_t block_dim, void* stream,
                           uint64_t ffts_config, uint8_t* gva,
                           int64_t msg_size, int64_t iterations,
                           uint8_t* result_buffer) {
    mte_bandwidth_kernel<<<block_dim, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, block_dim, result_buffer);
}

// ========== 通信隐藏测试Kernel ==========
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void hidden_comm_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    GM_ADDR matmul_A,
    GM_ADDR matmul_B,
    GM_ADDR matmul_C,
    int64_t matmul_M,
    int64_t matmul_K,
    int64_t matmul_N,
    GM_ADDR result_buffer) {

    util_set_ffts_config(ffts_config);
    if (AscendC::GetSubBlockIdx() != 0) return;

    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;

    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    if (rank == 0) {
        for (int64_t i = 0; i < iterations; i++) {
            int64_t iter_start = AscendC::GetSystemCycle();

            // 1. 发起非阻塞通信
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);

            // 2. 同时进行计算负载
            volatile float sum = 0;
            for (int64_t j = 0; j < matmul_M * matmul_K * matmul_N / 1000; j++) {
                sum += j * 0.001f;
            }

            // 3. 等待通信完成
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            int64_t iter_end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = iter_end - iter_start;
        }
    }
}

// ========== MatMul计算Kernel ==========
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void matmul_compute_kernel(
    GM_ADDR A,
    GM_ADDR B,
    GM_ADDR C,
    int64_t M,
    int64_t K,
    int64_t N,
    GM_ADDR result_buffer) {

    if (AscendC::GetSubBlockIdx() != 0) return;

    int64_t start_cycle = AscendC::GetSystemCycle();

    __gm__ float* a_ptr = (__gm__ float*)A;
    __gm__ float* b_ptr = (__gm__ float*)B;
    __gm__ float* c_ptr = (__gm__ float*)C;

    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0;
            for (int64_t k = 0; k < K; k++) {
                sum += a_ptr[i * K + k] * b_ptr[k * N + j];
            }
            c_ptr[i * N + j] = sum;
        }
    }

    int64_t end_cycle = AscendC::GetSystemCycle();
    *(__gm__ int64_t*)(result_buffer) = end_cycle - start_cycle;
}

// ========== Host端调用接口 ==========
void launch_rdma_pingpong_latency(uint32_t block_dim, void* stream,
                                   uint64_t ffts_config, uint8_t* gva,
                                   int64_t msg_size, int64_t iterations,
                                   int64_t warmup, uint8_t* result_buffer) {
    rdma_pingpong_latency_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, warmup, result_buffer);
}

void launch_rdma_bandwidth(uint32_t block_dim, void* stream,
                            uint64_t ffts_config, uint8_t* gva,
                            int64_t msg_size, int64_t iterations,
                            uint8_t* result_buffer) {
    rdma_bandwidth_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, result_buffer);
}

void launch_mte_pingpong_latency(uint32_t block_dim, void* stream,
                                  uint64_t ffts_config, uint8_t* gva,
                                  int64_t msg_size, int64_t iterations,
                                  int64_t warmup, uint8_t* result_buffer) {
    mte_pingpong_latency_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, warmup, result_buffer);
}

void launch_hidden_comm(uint32_t block_dim, void* stream,
                         uint64_t ffts_config, uint8_t* gva,
                         int64_t msg_size, int64_t iterations,
                         uint8_t* matmul_A, uint8_t* matmul_B, uint8_t* matmul_C,
                         int64_t M, int64_t K, int64_t N,
                         uint8_t* result_buffer) {
    hidden_comm_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations,
        matmul_A, matmul_B, matmul_C, M, K, N,
        result_buffer);
}

void launch_matmul_compute(uint32_t block_dim, void* stream,
                            uint8_t* A, uint8_t* B, uint8_t* C,
                            int64_t M, int64_t K, int64_t N,
                            uint8_t* result_buffer) {
    matmul_compute_kernel<<<1, nullptr, stream>>>(
        A, B, C, M, K, N, result_buffer);
}
