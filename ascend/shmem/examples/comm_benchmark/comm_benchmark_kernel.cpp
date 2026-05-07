/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Comm Benchmark Kernel - 通信性能测试Kernel
 *
 * 关键改进：
 * 1. 双向信号区分（rank0_expect vs rank1_expect + 5000）防止回环误判
 * 2. 正确的缓存刷新（AscendC::DataCacheCleanAndInvalidate）
 * 3. 内存屏障确保写入可见（MemoryBarrier W_MTE1）
 * 4. Iteration间同步防止竞争（aclshmem_barrier_all）
 * 5. 合并warmup和iterations循环
 * 6. 使用aclshmem_quiet确保传输完成
 */

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

// 宏定义
#define MAGIC_VAL 1000
#define TIMEOUT_CYCLES 100000000LL  // 约0.1秒，根据实际频率调整
#define TIMEOUT_ERROR_CODE -1

// ========== RDMA PingPong延迟测试Kernel ==========
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_pingpong_latency_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t warmup,
    GM_ADDR result_buffer) {

    util_set_ffts_config(ffts_config);
    if (AscendC::GetSubBlockIdx() != 0) return;

    // 基础环境初始化
    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    // 地址计算
    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR peer_slot_flag_addr = gva + peer * msg_size + msg_size - 8;
    GM_ADDR self_slot_flag_addr = src_addr + msg_size - 8;
    GM_ADDR result_addr = result_buffer;

    // 等待函数：带超时的peer信号等待
    auto wait_for_peer = [&](uint32_t target_val) -> bool {
        int64_t wait_start = AscendC::GetSystemCycle();
        while (true) {
            // 显式刷新 Cache，确保读到的是远程写入物理内存的值
            AscendC::DataCacheCleanAndInvalidate<uint32_t>(peer_slot_flag_addr);

            if (*(__gm__ uint32_t*)peer_slot_flag_addr == target_val) {
                return true;
            }
            if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                return false;
            }
            // 稍微让出指令发射，避免过度占用总线
            for (int k = 0; k < 100; k++) { __asm__ __volatile__("" : : : "memory"); }
        }
    };

    // --- 测试开始 ---
    bool timeout_occurred = false;
    int64_t total_iters = warmup + iterations;

    for (int64_t i = 0; i < total_iters; i++) {
        uint32_t rank0_expect = MAGIC_VAL + i;
        uint32_t rank1_expect = MAGIC_VAL + i + 5000; // 区分双向信号，防止回环误判

        if (rank == 0) {
            int64_t t_start = (i >= warmup) ? AscendC::GetSystemCycle() : 0;

            // 1. 准备数据并打屏障
            *(__gm__ uint32_t*)self_slot_flag_addr = rank0_expect;
            AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1); // 确保 Store 对搬运引擎可见

            // 2. 发送全量数据（含 Flag）
            aclshmem_uint8_put_nbi(gva + peer * msg_size, src_addr, msg_size, peer);
            aclshmem_quiet(); // 确保指令已发出

            // 3. 等待 Rank 1 的回信
            if (!wait_for_peer(rank1_expect)) { timeout_occurred = true; break; }

            if (i >= warmup) {
                int64_t t_end = AscendC::GetSystemCycle();
                *(__gm__ int64_t*)(result_addr + (i - warmup) * sizeof(int64_t)) = t_end - t_start;
            }
        }
        else {
            // Rank 1: 等待 Rank 0 的信号
            if (!wait_for_peer(rank0_expect)) { timeout_occurred = true; break; }

            // 响应：准备回传数据
            *(__gm__ uint32_t*)self_slot_flag_addr = rank1_expect;
            AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);

            aclshmem_uint8_put_nbi(gva + peer * msg_size, src_addr, msg_size, peer);
            aclshmem_quiet();
        }

        // --- 关键：Iteration 间同步 ---
        // 必须确保两边都完成了这一轮，才能进入下一轮复位标志位
        aclshmem_barrier_all();

        // 清理标志位（复位），为下一轮做准备
        *(__gm__ uint32_t*)self_slot_flag_addr = 0;
        AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);
        aclshmem_barrier_all();
    }

    if (timeout_occurred && rank == 0) {
        *(__gm__ int64_t*)(result_addr) = TIMEOUT_ERROR_CODE;
    }
}

// ========== SDMA PingPong延迟测试Kernel ==========
// SDMA使用与RDMA相同的逻辑，但使用SDMA专用的put接口
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void sdma_pingpong_latency_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t warmup,
    GM_ADDR result_buffer) {

    util_set_ffts_config(ffts_config);
    if (AscendC::GetSubBlockIdx() != 0) return;

    // 基础环境初始化
    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    // 地址计算
    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR peer_slot_flag_addr = gva + peer * msg_size + msg_size - 8;
    GM_ADDR self_slot_flag_addr = src_addr + msg_size - 8;
    GM_ADDR result_addr = result_buffer;

    // 等待函数：带超时的peer信号等待
    auto wait_for_peer = [&](uint32_t target_val) -> bool {
        int64_t wait_start = AscendC::GetSystemCycle();
        while (true) {
            AscendC::DataCacheCleanAndInvalidate<uint32_t>(peer_slot_flag_addr);

            if (*(__gm__ uint32_t*)peer_slot_flag_addr == target_val) {
                return true;
            }
            if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                return false;
            }
            for (int k = 0; k < 100; k++) { __asm__ __volatile__("" : : : "memory"); }
        }
    };

    // --- 测试开始 ---
    bool timeout_occurred = false;
    int64_t total_iters = warmup + iterations;

    for (int64_t i = 0; i < total_iters; i++) {
        uint32_t rank0_expect = MAGIC_VAL + i;
        uint32_t rank1_expect = MAGIC_VAL + i + 5000;

        if (rank == 0) {
            int64_t t_start = (i >= warmup) ? AscendC::GetSystemCycle() : 0;

            *(__gm__ uint32_t*)self_slot_flag_addr = rank0_expect;
            AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);

            // SDMA put接口
            aclshmem_uint8_put_nbi(gva + peer * msg_size, src_addr, msg_size, peer);
            aclshmem_quiet();

            if (!wait_for_peer(rank1_expect)) { timeout_occurred = true; break; }

            if (i >= warmup) {
                int64_t t_end = AscendC::GetSystemCycle();
                *(__gm__ int64_t*)(result_addr + (i - warmup) * sizeof(int64_t)) = t_end - t_start;
            }
        }
        else {
            if (!wait_for_peer(rank0_expect)) { timeout_occurred = true; break; }

            *(__gm__ uint32_t*)self_slot_flag_addr = rank1_expect;
            AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);

            aclshmem_uint8_put_nbi(gva + peer * msg_size, src_addr, msg_size, peer);
            aclshmem_quiet();
        }

        aclshmem_barrier_all();
        *(__gm__ uint32_t*)self_slot_flag_addr = 0;
        AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);
        aclshmem_barrier_all();
    }

    if (timeout_occurred && rank == 0) {
        *(__gm__ int64_t*)(result_addr) = TIMEOUT_ERROR_CODE;
    }
}

// ========== MTE PingPong延迟测试Kernel ==========
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
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;

    // 基础环境初始化
    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    // 地址计算
    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR peer_slot_flag_addr = gva + peer * msg_size + msg_size - 8;
    GM_ADDR self_slot_flag_addr = src_addr + msg_size - 8;
    GM_ADDR result_addr = result_buffer;

    // 等待函数：带超时的peer信号等待
    auto wait_for_peer = [&](uint32_t target_val) -> bool {
        int64_t wait_start = AscendC::GetSystemCycle();
        while (true) {
            AscendC::DataCacheCleanAndInvalidate<uint32_t>(peer_slot_flag_addr);

            if (*(__gm__ uint32_t*)peer_slot_flag_addr == target_val) {
                return true;
            }
            if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                return false;
            }
            for (int k = 0; k < 100; k++) { __asm__ __volatile__("" : : : "memory"); }
        }
    };

    // --- 测试开始 ---
    bool timeout_occurred = false;
    int64_t total_iters = warmup + iterations;

    for (int64_t i = 0; i < total_iters; i++) {
        uint32_t rank0_expect = MAGIC_VAL + i;
        uint32_t rank1_expect = MAGIC_VAL + i + 5000;

        if (rank == 0) {
            int64_t t_start = (i >= warmup) ? AscendC::GetSystemCycle() : 0;

            *(__gm__ uint32_t*)self_slot_flag_addr = rank0_expect;
            AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);

            // MTE put接口
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)(gva + peer * msg_size), (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            if (!wait_for_peer(rank1_expect)) { timeout_occurred = true; break; }

            if (i >= warmup) {
                int64_t t_end = AscendC::GetSystemCycle();
                *(__gm__ int64_t*)(result_addr + (i - warmup) * sizeof(int64_t)) = t_end - t_start;
            }
        }
        else {
            if (!wait_for_peer(rank0_expect)) { timeout_occurred = true; break; }

            *(__gm__ uint32_t*)self_slot_flag_addr = rank1_expect;
            AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);

            aclshmemx_mte_put_nbi((__gm__ uint8_t*)(gva + peer * msg_size), (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        }

        aclshmem_barrier_all();
        *(__gm__ uint32_t*)self_slot_flag_addr = 0;
        AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);
        aclshmem_barrier_all();
    }

    if (timeout_occurred && rank == 0) {
        *(__gm__ int64_t*)(result_addr) = TIMEOUT_ERROR_CODE;
    }
}

// ========== RDMA带宽测试Kernel（支持多核聚合）==========
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_bandwidth_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t block_dim,
    GM_ADDR result_buffer) {

    util_set_ffts_config(ffts_config);

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
    GM_ADDR self_slot_flag_addr = src_addr + sizeof(uint32_t); // 用于发送通知

    // 等待函数
    auto wait_for_signal = [&](GM_ADDR wait_addr, uint32_t target_val) -> bool {
        int64_t wait_start = AscendC::GetSystemCycle();
        while (true) {
            AscendC::DataCacheCleanAndInvalidate<uint32_t>(wait_addr);
            if (*(__gm__ uint32_t*)wait_addr == target_val) {
                return true;
            }
            if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES * 10) { // 带宽测试允许更长超时
                return false;
            }
            for (int k = 0; k < 100; k++) { __asm__ __volatile__("" : : : "memory"); }
        }
    };

    if (rank == 0) {
        peer = 1;

        int64_t start_cycle = AscendC::GetSystemCycle();

        // 所有 Core 都执行数据发送
        for (int64_t i = 0; i < iterations; i++) {
            GM_ADDR dst_addr = gva + peer * msg_size * block_dim + core_idx * msg_size;
            aclshmem_uint8_put_nbi(dst_addr, src_addr, msg_size, peer);
        }

        // 只有 Core 0 执行同步操作
        if (core_idx == 0) {
            aclshmem_quiet(); // 确保所有传输完成

            // 发送完成通知
            *(__gm__ uint32_t*)notify_addr = MAGIC_VAL;
            AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);
            aclshmem_uint8_put_nbi(notify_addr, notify_addr, sizeof(uint32_t), peer);
            aclshmem_quiet();

            // 等待接收方确认
            if (!wait_for_signal(ack_addr, MAGIC_VAL + 5000)) {
                *(__gm__ int64_t*)(result_addr) = TIMEOUT_ERROR_CODE;
                return;
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end_cycle = AscendC::GetSystemCycle();

        if (core_idx == 0) {
            *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;
        }

    } else {
        peer = 0;

        // 只有 Core 0 执行同步操作
        if (core_idx == 0) {
            if (!wait_for_signal(notify_addr, MAGIC_VAL)) {
                return;
            }

            // 发送确认
            *(__gm__ uint32_t*)ack_addr = MAGIC_VAL + 5000;
            AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);
            aclshmem_uint8_put_nbi(ack_addr, ack_addr, sizeof(uint32_t), peer);
            aclshmem_quiet();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// ========== SDMA带宽测试Kernel（支持多核聚合）==========
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void sdma_bandwidth_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t block_dim,
    GM_ADDR result_buffer) {

    util_set_ffts_config(ffts_config);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    int64_t core_idx = AscendC::GetBlockIdx();
    uint32_t peer;

    GM_ADDR src_addr = gva + rank * msg_size * block_dim + core_idx * msg_size;
    GM_ADDR result_addr = result_buffer;

    int64_t sync_base_offset = rank_size * msg_size * block_dim;
    GM_ADDR notify_addr = gva + sync_base_offset + 8;
    GM_ADDR ack_addr = gva + sync_base_offset + 16;

    auto wait_for_signal = [&](GM_ADDR wait_addr, uint32_t target_val) -> bool {
        int64_t wait_start = AscendC::GetSystemCycle();
        while (true) {
            AscendC::DataCacheCleanAndInvalidate<uint32_t>(wait_addr);
            if (*(__gm__ uint32_t*)wait_addr == target_val) return true;
            if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES * 10) return false;
            for (int k = 0; k < 100; k++) { __asm__ __volatile__("" : : : "memory"); }
        }
    };

    if (rank == 0) {
        peer = 1;
        int64_t start_cycle = AscendC::GetSystemCycle();

        for (int64_t i = 0; i < iterations; i++) {
            GM_ADDR dst_addr = gva + peer * msg_size * block_dim + core_idx * msg_size;
            aclshmem_uint8_put_nbi(dst_addr, src_addr, msg_size, peer);
        }

        if (core_idx == 0) {
            aclshmem_quiet();
            *(__gm__ uint32_t*)notify_addr = MAGIC_VAL;
            AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);
            aclshmem_uint8_put_nbi(notify_addr, notify_addr, sizeof(uint32_t), peer);
            aclshmem_quiet();

            if (!wait_for_signal(ack_addr, MAGIC_VAL + 5000)) {
                *(__gm__ int64_t*)(result_addr) = TIMEOUT_ERROR_CODE;
                return;
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end_cycle = AscendC::GetSystemCycle();

        if (core_idx == 0) {
            *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;
        }

    } else {
        peer = 0;

        if (core_idx == 0) {
            if (!wait_for_signal(notify_addr, MAGIC_VAL)) return;

            *(__gm__ uint32_t*)ack_addr = MAGIC_VAL + 5000;
            AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);
            aclshmem_uint8_put_nbi(ack_addr, ack_addr, sizeof(uint32_t), peer);
            aclshmem_quiet();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// ========== MTE带宽测试Kernel（支持多核聚合）==========
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_bandwidth_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t block_dim,
    GM_ADDR result_buffer) {

    util_set_ffts_config(ffts_config);

    // 获取MTE配置
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    int64_t core_idx = AscendC::GetBlockIdx();
    uint32_t peer;

    GM_ADDR src_addr = gva + rank * msg_size * block_dim + core_idx * msg_size;
    GM_ADDR result_addr = result_buffer;

    int64_t sync_base_offset = rank_size * msg_size * block_dim;
    GM_ADDR notify_addr = gva + sync_base_offset + 8;
    GM_ADDR ack_addr = gva + sync_base_offset + 16;

    auto wait_for_signal = [&](GM_ADDR wait_addr, uint32_t target_val) -> bool {
        int64_t wait_start = AscendC::GetSystemCycle();
        while (true) {
            AscendC::DataCacheCleanAndInvalidate<uint32_t>(wait_addr);
            if (*(__gm__ uint32_t*)wait_addr == target_val) return true;
            if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES * 10) return false;
            for (int k = 0; k < 100; k++) { __asm__ __volatile__("" : : : "memory"); }
        }
    };

    if (rank == 0) {
        peer = 1;
        int64_t start_cycle = AscendC::GetSystemCycle();

        for (int64_t i = 0; i < iterations; i++) {
            GM_ADDR dst_addr = gva + peer * msg_size * block_dim + core_idx * msg_size;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)dst_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
        }

        if (core_idx == 0) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            *(__gm__ uint32_t*)notify_addr = MAGIC_VAL;
            AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)notify_addr, (__gm__ uint8_t*)notify_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, sizeof(uint32_t), peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            if (!wait_for_signal(ack_addr, MAGIC_VAL + 5000)) {
                *(__gm__ int64_t*)(result_addr) = TIMEOUT_ERROR_CODE;
                return;
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end_cycle = AscendC::GetSystemCycle();

        if (core_idx == 0) {
            *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;
        }

    } else {
        peer = 0;

        if (core_idx == 0) {
            if (!wait_for_signal(notify_addr, MAGIC_VAL)) return;

            *(__gm__ uint32_t*)ack_addr = MAGIC_VAL + 5000;
            AscendC::MemoryBarrier(AscendC::MemoryBarrierRole::W_MTE1);
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)ack_addr, (__gm__ uint8_t*)ack_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, sizeof(uint32_t), peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// ========== Host端调用接口 ==========
void launch_rdma_pingpong_latency(uint32_t block_dim, void* stream,
                                   uint64_t ffts_config, uint8_t* gva,
                                   int64_t msg_size, int64_t iterations,
                                   int64_t warmup, uint8_t* result_buffer) {
    rdma_pingpong_latency_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, warmup, result_buffer);
}

void launch_sdma_pingpong_latency(uint32_t block_dim, void* stream,
                                   uint64_t ffts_config, uint8_t* gva,
                                   int64_t msg_size, int64_t iterations,
                                   int64_t warmup, uint8_t* result_buffer) {
    sdma_pingpong_latency_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, warmup, result_buffer);
}

void launch_mte_pingpong_latency(uint32_t block_dim, void* stream,
                                  uint64_t ffts_config, uint8_t* gva,
                                  int64_t msg_size, int64_t iterations,
                                  int64_t warmup, uint8_t* result_buffer) {
    mte_pingpong_latency_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, warmup, result_buffer);
}

void launch_rdma_bandwidth(uint32_t block_dim, void* stream,
                            uint64_t ffts_config, uint8_t* gva,
                            int64_t msg_size, int64_t iterations,
                            uint8_t* result_buffer) {
    rdma_bandwidth_kernel<<<block_dim, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, block_dim, result_buffer);
}

void launch_sdma_bandwidth(uint32_t block_dim, void* stream,
                            uint64_t ffts_config, uint8_t* gva,
                            int64_t msg_size, int64_t iterations,
                            uint8_t* result_buffer) {
    sdma_bandwidth_kernel<<<block_dim, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, block_dim, result_buffer);
}

void launch_mte_bandwidth(uint32_t block_dim, void* stream,
                           uint64_t ffts_config, uint8_t* gva,
                           int64_t msg_size, int64_t iterations,
                           uint8_t* result_buffer) {
    mte_bandwidth_kernel<<<block_dim, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, block_dim, result_buffer);
}