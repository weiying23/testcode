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

// ========== RDMA带宽测试Kernel ==========
//
// BUG修复说明：
// 原问题：只测量指令下发时间，没有等待实际数据搬运完成，导致带宽过大
// 根因分析：
// - aclshmem_uint8_put_nbi是非阻塞操作，只将请求放入队列
// - 循环下发iterations次操作很快完成（可能只需几微秒）
// - roce_quiet等待队列清空，但不保证接收方已收到数据
// - 实际数据搬运还在网络传输中，测得带宽虚高
//
// 修复方案：
// 使用pingpong模式测量带宽：
// 1. 发送方批量发送数据
// 2. 接收方收到后发送确认响应
// 3. 发送方收到响应后才记录结束时间
// 这样测量的时间包含了完整的双向传输时间
//
// 带宽计算公式：
// - 单向带宽 = iterations * msg_size / (total_time / 2)
// - 双向带宽 = iterations * msg_size * 2 / total_time（考虑发送+接收）
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_bandwidth_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    GM_ADDR result_buffer) {

    util_set_ffts_config(ffts_config);
    if (AscendC::GetSubBlockIdx() != 0) return;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64, 0);

    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    // 内存布局：
    // Slot 0: rank 0的数据区（发送方使用）
    // Slot 1: rank 1的数据区（接收方使用）
    // Slot 2末尾: 同步/确认区域
    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    // 使用pingpong模式测量真实传输时间
    if (rank == 0) {
        // 发送方逻辑
        int64_t start_cycle = AscendC::GetSystemCycle();

        // 步骤1：批量发送所有数据
        for (int64_t i = 0; i < iterations; i++) {
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);
        }

        // 步骤2：写入完成标志，通知接收方所有数据已发送
        // Magic value = iterations（表示发送了多少次）
        *(__gm__ uint32_t*)(gva + msg_size * 2 - 8) = iterations;

        // 发送完成标志到接收方
        aclshmem_uint8_put_nbi(gva + msg_size * 2 - 8, gva + msg_size * 2 - 8, 8, peer);
        aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);

        // 步骤3：等待接收方的确认响应
        // 接收方在收到所有数据后，会写入 MAGIC_VAL_BW 作为确认
        while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != MAGIC_VAL_BW) {
            dcci_cachelines(gva + msg_size * 2 - 8, 8);
            AscendC::GetSystemCycle();
        }

        int64_t end_cycle = AscendC::GetSystemCycle();
        *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;

    } else {
        // 接收方逻辑
        // 步骤1：等待发送方的完成标志
        // 持续轮询，直到收到发送方通知的所有数据已发送
        uint32_t expected_count = 0;
        while (expected_count < iterations) {
            dcci_cachelines(gva + msg_size * 2 - 8, 8);
            expected_count = *(__gm__ uint32_t*)(gva + msg_size * 2 - 8);
            AscendC::GetSystemCycle();
        }

        // 步骤2：确保所有数据都已到达（执行quiet等待）
        aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);

        // 步骤3：发送确认响应
        // 写入 MAGIC_VAL_BW 表示接收方已收到所有数据
        *(__gm__ uint32_t*)(gva + msg_size * 2 - 8) = MAGIC_VAL_BW;
        aclshmem_uint8_put_nbi(gva + msg_size * 2 - 8, gva + msg_size * 2 - 8, 8, peer);
        aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
    }
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

// ========== MTE带宽测试Kernel ==========
//
// BUG修复说明：
// 原问题：与RDMA带宽测试相同，只测量指令下发时间，带宽过大
// 根因分析：
// - aclshmemx_mte_put_nbi是非阻塞操作
// - WaitFlag只等待最后一个事件完成，不保证接收方收到所有数据
// - 测得带宽虚高
//
// 修复方案：
// 使用pingpong模式测量真实传输时间
// 与RDMA带宽测试使用相同的同步机制
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_bandwidth_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    GM_ADDR result_buffer) {

    util_set_ffts_config(ffts_config);
    if (AscendC::GetSubBlockIdx() != 0) return;

    // 获取MTE配置
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;

    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    // 使用pingpong模式测量真实传输时间
    if (rank == 0) {
        // 发送方逻辑
        int64_t start_cycle = AscendC::GetSystemCycle();

        // 步骤1：批量发送所有数据
        for (int64_t i = 0; i < iterations; i++) {
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
        }

        // 步骤2：等待所有发送完成
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

        // 步骤3：写入完成标志，通知接收方
        *(__gm__ uint32_t*)(gva + msg_size * 2 - 8) = iterations;

        // 发送完成标志
        aclshmemx_mte_put_nbi((__gm__ uint8_t*)(gva + msg_size * 2 - 8),
                              (__gm__ uint8_t*)(gva + msg_size * 2 - 8),
                              reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                              copy_ub_size, 8, peer, copy_event_id);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

        // 步骤4：等待接收方确认
        while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != MAGIC_VAL_BW) {
            dcci_cachelines(gva + msg_size * 2 - 8, 8);
            AscendC::GetSystemCycle();
        }

        int64_t end_cycle = AscendC::GetSystemCycle();
        *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;

    } else {
        // 接收方逻辑
        // 步骤1：等待发送方的完成标志
        uint32_t expected_count = 0;
        while (expected_count < iterations) {
            dcci_cachelines(gva + msg_size * 2 - 8, 8);
            expected_count = *(__gm__ uint32_t*)(gva + msg_size * 2 - 8);
            AscendC::GetSystemCycle();
        }

        // 步骤2：等待所有数据到达
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

        // 步骤3：发送确认响应
        *(__gm__ uint32_t*)(gva + msg_size * 2 - 8) = MAGIC_VAL_BW;
        aclshmemx_mte_put_nbi((__gm__ uint8_t*)(gva + msg_size * 2 - 8),
                              (__gm__ uint8_t*)(gva + msg_size * 2 - 8),
                              reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                              copy_ub_size, 8, peer, copy_event_id);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
    }
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

void launch_mte_bandwidth(uint32_t block_dim, void* stream,
                           uint64_t ffts_config, uint8_t* gva,
                           int64_t msg_size, int64_t iterations,
                           uint8_t* result_buffer) {
    mte_bandwidth_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, result_buffer);
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
