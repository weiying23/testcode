/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Comm Benchmark Kernel - 通信性能测试Kernel
 *
 * ========== 延迟测量语义说明 ==========
 *
 * 本文件实现了 RDMA 和 MTE 两种引擎的 pingpong 延迟测试。
 * 两个引擎测量的延迟在语义上是等价的，但硬件路径不同：
 *
 *   RDMA (roce_quiet 路径)：
 *     put_nbi(data) → roce_quiet → write_flag → wait_peer_flag
 *     roce_quiet 确保数据已到达远端 NIC 内存，flag 写入发生在数据确认之后。
 *     测量的是"远端确认收到数据"的完整往返延迟（RTT/2 为单向延迟）。
 *
 *   MTE (WaitFlag 路径)：
 *     mte_put_nbi(data) → SetFlag/WaitFlag(MTE3_S) → write_flag → wait_peer_flag
 *     WaitFlag 确保 MTE DMA 已将数据写入共享 GVA，共享内存对所有 PE 立即可见。
 *     对共享内存而言，WaitFlag 等价于 roce_quiet，两者测量语义一致。
 *
 * 带宽测量语义：
 *   计时窗口 = [start_cycle, end_cycle]，其中：
 *     start_cycle 在第一次 put_nbi 之前
 *     end_cycle   在 quiet/WaitFlag 之后、notify/ack 握手之前
 *   因此带宽 = iterations × msg_size / (end - start) 不含握手 RTT 开销。
 *
 * 同步机制说明：
 *   - aclshmem_barrier_all()：kernel 入口双端同步，避免先启动的 PE 误超时
 *   - AscendC::SyncAll()：带宽 kernel 内跨核同步，确保所有 core put 都发出后再 quiet
 *   - aclshmem_uint32_test()：pingpong 等待（带超时，shmem 原语替代手写 dcci 轮询）
 *   - aclshmem_uint32_wait_until()：带宽 notify/ack 等待（无需超时，barrier 已保证连通）
 */

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

// 参考rdma_perftest：MAGIC_VAL = 10，用于数据初始化和同步
constexpr uint32_t MAGIC_VAL = 10;

// 超时配置：10秒超时（假设NPU频率1GHz，10秒 = 10^10 cycles）
constexpr int64_t TIMEOUT_CYCLES = 10000000000LL;  // 10 seconds timeout

// 超时检测结果码
constexpr int64_t TIMEOUT_ERROR_CODE = -1;  // 超时错误标记

// ========== RDMA PingPong延迟测试Kernel ==========
//
// 内存布局（与 MTE 版本相同）：
//   [0,          msg_size)         Rank 0 数据区
//   [msg_size,   2*msg_size)       Rank 1 数据区
//   [2*msg_size, 2*msg_size+8)     Rank 0 的通知 flag（仅 Rank 0 写，Rank 1 轮询）
//   [2*msg_size+8, 2*msg_size+16)  Rank 1 的响应 flag（仅 Rank 1 写，Rank 0 轮询）
//
// 修复说明（与 MTE 版本相同）：
//   原实现把 flag 放在数据 slot 末尾，put 整个 slot 时会覆盖对端刚写入的 flag，
//   且每轮末尾的复位操作有窗口期，可能在对端读到前就清掉通知。
//   修复：使用独立于数据区的 flag 地址，put 只传数据，flag 单调递增不复位。
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

    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    GM_ADDR rank0_flag_addr = gva + 2 * msg_size;
    GM_ADDR rank1_flag_addr = gva + 2 * msg_size + 8;
    GM_ADDR my_flag_addr   = (rank == 0) ? rank0_flag_addr : rank1_flag_addr;
    GM_ADDR peer_flag_addr = (rank == 0) ? rank1_flag_addr : rank0_flag_addr;

    uint32_t my_seq   = MAGIC_VAL + rank;
    uint32_t peer_seq = MAGIC_VAL + (uint32_t)peer;
    bool timeout_detected = false;

    // 入口屏障：确保双端都已进入 kernel，避免先启动的 PE 因对端未就绪而超时
    aclshmem_barrier_all();

    // Warmup 阶段
    for (int64_t i = 0; i < warmup && !timeout_detected; i++) {
        if (rank == 0) {
            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmem_uint8_put_nbi(dst_addr, src_addr, msg_size, peer);
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
            my_seq++;
            *(__gm__ uint32_t*)(my_flag_addr) = my_seq;

            peer_seq++;
            int64_t wait_start = AscendC::GetSystemCycle();
            while (!aclshmem_uint32_test((__gm__ uint32_t*)peer_flag_addr, ACLSHMEM_CMP_EQ, peer_seq)) {
                if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                    timeout_detected = true;
                    break;
                }
            }
        } else {
            peer_seq++;
            int64_t wait_start = AscendC::GetSystemCycle();
            while (!aclshmem_uint32_test((__gm__ uint32_t*)peer_flag_addr, ACLSHMEM_CMP_EQ, peer_seq)) {
                if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                    timeout_detected = true;
                    break;
                }
            }
            if (!timeout_detected) {
                GM_ADDR dst_addr = gva + peer * msg_size;
                aclshmem_uint8_put_nbi(dst_addr, src_addr, msg_size, peer);
                aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
                my_seq++;
                *(__gm__ uint32_t*)(my_flag_addr) = my_seq;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    if (timeout_detected) {
        if (rank == 0) *(__gm__ int64_t*)(result_addr) = TIMEOUT_ERROR_CODE;
        return;
    }

    // 正式测试阶段
    // 延迟语义：包含 put_nbi + roce_quiet（远端确认收到）+ flag 信令的完整 RTT。
    // roce_quiet 保证数据到达对端 NIC，因此测量的是"确认送达"的往返延迟，
    // 与 MTE 版本的"共享内存写完成"RTT 语义等价但硬件路径不同。
    if (rank == 0) {
        for (int64_t i = 0; i < iterations; i++) {
            int64_t iter_start = AscendC::GetSystemCycle();

            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmem_uint8_put_nbi(dst_addr, src_addr, msg_size, peer);
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
            my_seq++;
            *(__gm__ uint32_t*)(my_flag_addr) = my_seq;

            peer_seq++;
            int64_t wait_start = AscendC::GetSystemCycle();
            while (!aclshmem_uint32_test((__gm__ uint32_t*)peer_flag_addr, ACLSHMEM_CMP_EQ, peer_seq)) {
                if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                    *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = TIMEOUT_ERROR_CODE;
                    return;
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            int64_t iter_end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = iter_end - iter_start;
        }
    } else {
        for (int64_t i = 0; i < iterations; i++) {
            peer_seq++;
            int64_t wait_start = AscendC::GetSystemCycle();
            while (!aclshmem_uint32_test((__gm__ uint32_t*)peer_flag_addr, ACLSHMEM_CMP_EQ, peer_seq)) {
                if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                    return;
                }
            }
            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmem_uint8_put_nbi(dst_addr, src_addr, msg_size, peer);
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
            my_seq++;
            *(__gm__ uint32_t*)(my_flag_addr) = my_seq;

            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

// ========== RDMA带宽测试Kernel（支持多核聚合）==========
//
// 内存布局：
// - 每个 PE 有 block_dim 个数据 slot
// - PE i 的 Core j 数据位于 gva + i * msg_size * block_dim + j * msg_size
// - 同步区域（notify/ack）位于所有数据之后：
//     notify_addr = gva + rank_size * msg_size * block_dim + 8
//     ack_addr    = gva + rank_size * msg_size * block_dim + 16
//
// 修复说明：
// 1. 计时点：end_cycle 在 quiet 完成、notify/ack 开始之前采样，
//    排除通知握手的 RTT 开销，让带宽结果只反映数据传输时间。
// 2. round_id 参数：每轮使用 (rank + MAGIC_VAL + round_id) 作为 flag 值，
//    避免上一轮残留值在下一轮被误读。
// 3. 数据发完后 quiet，再停表，保证所有 put_nbi 都已到达对端。
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_bandwidth_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t block_dim,
    GM_ADDR result_buffer,
    int64_t round_id) {

    util_set_ffts_config(ffts_config);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    int64_t core_idx = AscendC::GetBlockIdx();
    uint32_t peer;

    GM_ADDR src_addr = gva + rank * msg_size * block_dim + core_idx * msg_size;
    GM_ADDR result_addr = result_buffer;

    int64_t sync_base_offset = rank_size * msg_size * block_dim;
    GM_ADDR notify_addr = gva + sync_base_offset + 8;
    GM_ADDR ack_addr    = gva + sync_base_offset + 16;

    // 每轮使用不同的 flag 值，防止上轮残留干扰
    uint32_t expected_notify = (uint32_t)(0 + MAGIC_VAL + round_id);
    uint32_t expected_ack    = (uint32_t)(1 + MAGIC_VAL + round_id);

    // 入口屏障：确保双端都已进入 kernel
    aclshmem_barrier_all();

    if (rank == 0) {
        peer = 1;
        int64_t start_cycle = AscendC::GetSystemCycle();

        for (int64_t i = 0; i < iterations; i++) {
            GM_ADDR dst_addr = gva + peer * msg_size * block_dim + core_idx * msg_size;
            aclshmem_uint8_put_nbi(dst_addr, src_addr, msg_size, peer);
        }

        // 跨核屏障：确保所有 core 的 put_nbi 都已发出，再由 core 0 调 quiet
        AscendC::SyncAll();

        if (core_idx == 0) {
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
            int64_t end_cycle = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;

            *(__gm__ uint32_t*)(notify_addr) = expected_notify;
            aclshmem_uint32_wait_until((__gm__ uint32_t*)ack_addr, ACLSHMEM_CMP_EQ, expected_ack);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

    } else {
        peer = 0;

        if (core_idx == 0) {
            aclshmem_uint32_wait_until((__gm__ uint32_t*)notify_addr, ACLSHMEM_CMP_EQ, expected_notify);
            *(__gm__ uint32_t*)(ack_addr) = expected_ack;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

void launch_rdma_bandwidth(uint32_t block_dim, void* stream,
                            uint64_t ffts_config, uint8_t* gva,
                            int64_t msg_size, int64_t iterations,
                            uint8_t* result_buffer, int64_t round_id) {
    rdma_bandwidth_kernel<<<block_dim, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, block_dim, result_buffer, round_id);
}

// ========== MTE PingPong延迟测试Kernel ==========
//
// 内存布局（调用方需保证 gva 分配足够空间）：
//   [0,          msg_size)         Rank 0 数据区（put 的 src/dst）
//   [msg_size,   2*msg_size)       Rank 1 数据区（put 的 src/dst）
//   [2*msg_size, 2*msg_size+8)     Rank 0 的通知 flag（仅 Rank 0 写，Rank 1 轮询）
//   [2*msg_size+8, 2*msg_size+16)  Rank 1 的通知 flag（仅 Rank 1 写，Rank 0 轮询）
//
// 修复说明：
//   原实现把 flag 放在数据区末尾（msg_size-8 处），导致两个 bug：
//   1. put 整个 slot 时会把对端刚写入的响应 flag 用旧值覆盖，Rank 0 永远看不到期望值。
//   2. 每次迭代末尾的 "复位" 操作有窗口期，可能在对端读到值之前就清掉了通知。
//   修复方法：使用独立于数据区的 flag 地址，put 只传数据（不含 flag），
//   且 flag 单调递增，不再做复位。
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_pingpong_latency_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t warmup,
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

    GM_ADDR rank0_flag_addr = gva + 2 * msg_size;
    GM_ADDR rank1_flag_addr = gva + 2 * msg_size + 8;
    GM_ADDR my_flag_addr   = (rank == 0) ? rank0_flag_addr : rank1_flag_addr;
    GM_ADDR peer_flag_addr = (rank == 0) ? rank1_flag_addr : rank0_flag_addr;

    uint32_t my_seq   = MAGIC_VAL + rank;
    uint32_t peer_seq = MAGIC_VAL + (uint32_t)peer;
    bool timeout_detected = false;

    // 入口屏障：确保双端都已进入 kernel
    aclshmem_barrier_all();

    // Warmup 阶段
    for (int64_t i = 0; i < warmup && !timeout_detected; i++) {
        if (rank == 0) {
            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)dst_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            my_seq++;
            *(__gm__ uint32_t*)(my_flag_addr) = my_seq;

            peer_seq++;
            int64_t wait_start = AscendC::GetSystemCycle();
            while (!aclshmem_uint32_test((__gm__ uint32_t*)peer_flag_addr, ACLSHMEM_CMP_EQ, peer_seq)) {
                if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                    timeout_detected = true;
                    break;
                }
            }
        } else {
            peer_seq++;
            int64_t wait_start = AscendC::GetSystemCycle();
            while (!aclshmem_uint32_test((__gm__ uint32_t*)peer_flag_addr, ACLSHMEM_CMP_EQ, peer_seq)) {
                if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                    timeout_detected = true;
                    break;
                }
            }
            if (!timeout_detected) {
                GM_ADDR dst_addr = gva + peer * msg_size;
                aclshmemx_mte_put_nbi((__gm__ uint8_t*)dst_addr, (__gm__ uint8_t*)src_addr,
                                      reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                      copy_ub_size, msg_size, peer, copy_event_id);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
                my_seq++;
                *(__gm__ uint32_t*)(my_flag_addr) = my_seq;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    if (timeout_detected) {
        if (rank == 0) *(__gm__ int64_t*)(result_addr) = TIMEOUT_ERROR_CODE;
        return;
    }

    // 正式测试阶段
    // 延迟语义：包含 mte_put_nbi + WaitFlag（本地 MTE 完成，数据写入共享 GVA，
    // 对端通过 dcci/wait_until 可立即读到）+ flag 信令的完整 RTT。
    // WaitFlag 对共享内存等价于 RDMA 的 roce_quiet，两者测量语义一致。
    if (rank == 0) {
        for (int64_t i = 0; i < iterations; i++) {
            int64_t iter_start = AscendC::GetSystemCycle();

            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)dst_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            my_seq++;
            *(__gm__ uint32_t*)(my_flag_addr) = my_seq;

            peer_seq++;
            int64_t wait_start = AscendC::GetSystemCycle();
            while (!aclshmem_uint32_test((__gm__ uint32_t*)peer_flag_addr, ACLSHMEM_CMP_EQ, peer_seq)) {
                if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                    *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = TIMEOUT_ERROR_CODE;
                    return;
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            int64_t iter_end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = iter_end - iter_start;
        }
    } else {
        for (int64_t i = 0; i < iterations; i++) {
            peer_seq++;
            int64_t wait_start = AscendC::GetSystemCycle();
            while (!aclshmem_uint32_test((__gm__ uint32_t*)peer_flag_addr, ACLSHMEM_CMP_EQ, peer_seq)) {
                if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                    return;
                }
            }
            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)dst_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            my_seq++;
            *(__gm__ uint32_t*)(my_flag_addr) = my_seq;

            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

// ========== MTE带宽测试Kernel（支持多核聚合）==========
//
// 修复说明（与 RDMA 带宽版本相同）：
// 1. end_cycle 在 SetFlag/WaitFlag 之后、notify/ack 之前采样。
// 2. round_id 参数保证每轮 flag 值唯一，避免跨轮污染。
// 3. 数据全部发完且 MTE WaitFlag 后再停表，保证数据已到达对端。
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_bandwidth_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t block_dim,
    GM_ADDR result_buffer,
    int64_t round_id) {

    util_set_ffts_config(ffts_config);

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
    GM_ADDR ack_addr    = gva + sync_base_offset + 16;

    uint32_t expected_notify = (uint32_t)(0 + MAGIC_VAL + round_id);
    uint32_t expected_ack    = (uint32_t)(1 + MAGIC_VAL + round_id);

    // 入口屏障：确保双端都已进入 kernel
    aclshmem_barrier_all();

    if (rank == 0) {
        peer = 1;
        int64_t start_cycle = AscendC::GetSystemCycle();

        for (int64_t i = 0; i < iterations; i++) {
            GM_ADDR dst_addr = gva + peer * msg_size * block_dim + core_idx * msg_size;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)dst_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
        }

        // 跨核屏障：确保所有 core 的 mte_put_nbi 都已发出，再由 core 0 做 WaitFlag
        AscendC::SyncAll();

        if (core_idx == 0) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            int64_t end_cycle = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;

            *(__gm__ uint32_t*)(notify_addr) = expected_notify;
            aclshmem_uint32_wait_until((__gm__ uint32_t*)ack_addr, ACLSHMEM_CMP_EQ, expected_ack);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

    } else {
        peer = 0;

        if (core_idx == 0) {
            aclshmem_uint32_wait_until((__gm__ uint32_t*)notify_addr, ACLSHMEM_CMP_EQ, expected_notify);
            *(__gm__ uint32_t*)(ack_addr) = expected_ack;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

void launch_mte_bandwidth(uint32_t block_dim, void* stream,
                           uint64_t ffts_config, uint8_t* gva,
                           int64_t msg_size, int64_t iterations,
                           uint8_t* result_buffer, int64_t round_id) {
    mte_bandwidth_kernel<<<block_dim, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, block_dim, result_buffer, round_id);
}

// ========== Host端调用接口 ==========
void launch_rdma_pingpong_latency(uint32_t block_dim, void* stream,
                                   uint64_t ffts_config, uint8_t* gva,
                                   int64_t msg_size, int64_t iterations,
                                   int64_t warmup, uint8_t* result_buffer) {
    rdma_pingpong_latency_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, warmup, result_buffer);
}

void launch_mte_pingpong_latency(uint32_t block_dim, void* stream,
                                  uint64_t ffts_config, uint8_t* gva,
                                  int64_t msg_size, int64_t iterations,
                                  int64_t warmup, uint8_t* result_buffer) {
    mte_pingpong_latency_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, warmup, result_buffer);
}
