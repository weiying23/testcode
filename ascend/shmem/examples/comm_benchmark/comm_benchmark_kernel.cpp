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
 *   - aclshmem_barrier_all()：仅在 host 端调用（run_benchmark 中），
 *     确保两个 PE 的 kernel 都启动完毕再开始收发，不可在 kernel 内调用。
 *   - AscendC::SyncAll()：带宽 kernel 内跨核同步，确保所有 core put 都发出后再 quiet
 *   - dcci_cachelines + while 轮询：pingpong 等待，接收方轮询本地 GVA（sender PUT 进来的位置）
 *   - put_nbi(notify/ack_addr, src_addr, 4, peer)：带宽 notify/ack 通过 PUT 传递到对端
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
// 内存布局（与参考实现 rdma_perftest_kernel.cpp 一致）：
//   [0,        msg_size)   Rank 0 数据区，末 8 字节为 Rank 0 的发送 flag
//   [msg_size, 2*msg_size) Rank 1 数据区，末 8 字节为 Rank 1 的发送 flag
//
// 同步机制：
//   - flag 内嵌在自己的数据 slot 末尾（src_addr + msg_size - 8）
//   - put_nbi(src_addr, src_addr, msg_size, peer)：把自己的整个 slot（含 flag）PUT 到对端
//     对端收到后，在其本地的 gva + rank*msg_size 就能读到数据和 flag
//   - 接收方轮询自己本地 GVA（发送方 PUT 进来的位置），用 dcci_cachelines 刷缓存
//   - flag 值 = MAGIC_VAL + sender_rank，固定不变，通过 PUT 写入对端
//   - roce_quiet 确保 put_nbi 已完成送达，之后再轮询对端响应
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

    // src_addr: 自己的数据 slot 起始地址
    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    // flag 内嵌在 slot 末尾：发送方写自己 slot 的 flag，接收方轮询 peer slot 的 flag
    // Rank 0 发送后，Rank 1 在 gva+0*msg_size+msg_size-8 处看到 Rank 0 的 flag
    // Rank 1 发送后，Rank 0 在 gva+1*msg_size+msg_size-8 处看到 Rank 1 的 flag
    GM_ADDR my_flag_addr   = src_addr + msg_size - 8;
    GM_ADDR peer_flag_addr = gva + peer * msg_size + msg_size - 8;

    // 单调递增序列号：每次 PUT 前先把 my_seq 写入 slot 末尾，随 PUT 一起传到对端
    // host 已将 src_addr[-8] 初始化为 MAGIC_VAL + rank，内核从该值开始
    uint32_t my_seq   = MAGIC_VAL + (uint32_t)rank;
    uint32_t peer_seq = MAGIC_VAL + (uint32_t)peer;

    // Warmup 阶段
    for (int64_t i = 0; i < warmup; i++) {
        if (rank == 0) {
            *(__gm__ uint32_t*)my_flag_addr = my_seq;
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
            while (*(__gm__ uint32_t*)peer_flag_addr != peer_seq) {
                dcci_cachelines(peer_flag_addr, 8);
                AscendC::GetSystemCycle();
            }
            my_seq++;
            peer_seq++;
        } else {
            while (*(__gm__ uint32_t*)peer_flag_addr != peer_seq) {
                dcci_cachelines(peer_flag_addr, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            *(__gm__ uint32_t*)my_flag_addr = my_seq;
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
            my_seq++;
            peer_seq++;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 正式测试阶段
    if (rank == 0) {
        for (int64_t i = 0; i < iterations; i++) {
            int64_t iter_start = AscendC::GetSystemCycle();

            *(__gm__ uint32_t*)my_flag_addr = my_seq;
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);

            int64_t wait_start = AscendC::GetSystemCycle();
            while (*(__gm__ uint32_t*)peer_flag_addr != peer_seq) {
                dcci_cachelines(peer_flag_addr, 8);
                AscendC::GetSystemCycle();
                if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                    *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = TIMEOUT_ERROR_CODE;
                    return;
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            my_seq++;
            peer_seq++;

            int64_t iter_end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = iter_end - iter_start;
        }
    } else {
        for (int64_t i = 0; i < iterations; i++) {
            int64_t wait_start = AscendC::GetSystemCycle();
            while (*(__gm__ uint32_t*)peer_flag_addr != peer_seq) {
                dcci_cachelines(peer_flag_addr, 8);
                AscendC::GetSystemCycle();
                if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                    return;
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            *(__gm__ uint32_t*)my_flag_addr = my_seq;
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
            my_seq++;
            peer_seq++;
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

// ========== RDMA带宽测试Kernel ==========
//
// 内存布局（与参考实现 rdma_highlevel_put_bw 一致）：
//   [0,              rank_size*msg_size)    所有 PE 的数据区（每 PE 一个 slot）
//   [rank_size*msg_size+8,  +4)             notify 区（Rank 0 PUT 给 Rank 1，值=0+MAGIC_VAL+round）
//   [rank_size*msg_size+16, +4)             ack    区（Rank 1 PUT 回 Rank 0，值=1+MAGIC_VAL+round）
//
// 同步机制：
//   - notify/ack 全部通过 put_nbi 传递到对端本地内存，接收方用 dcci_cachelines 轮询
//   - Rank 0：发完数据 → roce_quiet → PUT notify → 等待 ack（dcci 轮询）
//   - Rank 1：等待 notify（dcci 轮询）→ PUT ack
//   - round_id 保证每轮 flag 值唯一，避免跨轮污染
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

    // notify/ack 区偏移与参考实现保持一致
    int64_t sync_base_offset = rank_size * msg_size * block_dim;
    GM_ADDR notify_addr = gva + sync_base_offset + 8;
    GM_ADDR ack_addr    = gva + sync_base_offset + 16;

    // 固定 flag 值（host 每轮前清零 notify/ack 区，保证跨轮安全）
    // notify_val = MAGIC_VAL + 0 (来自 Rank 0 的 src_addr[0])
    // ack_val    = MAGIC_VAL + 1 (来自 Rank 1 的 src_addr[0])
    uint32_t notify_val = (uint32_t)(MAGIC_VAL);      // Rank 0 的 src_addr[0]
    uint32_t ack_val    = (uint32_t)(MAGIC_VAL + 1);  // Rank 1 的 src_addr[0]
    (void)round_id;

    if (rank == 0) {
        peer = 1;
        int64_t start_cycle = AscendC::GetSystemCycle();

        for (int64_t i = 0; i < iterations; i++) {
            GM_ADDR dst_addr = gva + peer * msg_size * block_dim + core_idx * msg_size;
            aclshmem_uint8_put_nbi(dst_addr, src_addr, msg_size, peer);
        }

        AscendC::SyncAll();

        if (core_idx == 0) {
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
            int64_t end_cycle = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;

            // PUT notify 到 Rank 1 的本地内存（src_addr 首 4 字节已由 host 写为 notify_val）
            aclshmem_uint8_put_nbi(notify_addr, src_addr, sizeof(uint32_t), peer);
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);

            // 等待 Rank 1 PUT 回来的 ack（轮询本地 ack_addr）
            while (*(__gm__ uint32_t*)ack_addr != ack_val) {
                dcci_cachelines(ack_addr, 8);
                AscendC::GetSystemCycle();
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

    } else {
        peer = 0;

        if (core_idx == 0) {
            // 等待 Rank 0 PUT 来的 notify（轮询本地 notify_addr）
            while (*(__gm__ uint32_t*)notify_addr != notify_val) {
                dcci_cachelines(notify_addr, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // PUT ack 回 Rank 0（src_addr 首 4 字节已由 host 写为 ack_val）
            aclshmem_uint8_put_nbi(ack_addr, src_addr, sizeof(uint32_t), peer);
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
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
// 内存布局（与 RDMA 版本相同，与参考实现一致）：
//   [0,        msg_size)   Rank 0 数据区，末 8 字节为 Rank 0 的发送 flag
//   [msg_size, 2*msg_size) Rank 1 数据区，末 8 字节为 Rank 1 的发送 flag
//
// 同步机制：
//   - flag 内嵌在自己的数据 slot 末尾（src_addr + msg_size - 8），host 预写入 MAGIC_VAL+rank
//   - aclshmemx_mte_put_nbi(src, src, msg_size, peer)：把整个 slot（含 flag）PUT 到对端
//     对端在其本地 gva+peer*msg_size 处直接读到数据和 flag
//   - SetFlag/WaitFlag(MTE3_S)：确保 MTE DMA 已完成写入共享 GVA，对端可见
//   - 接收方用 dcci_cachelines 刷缓存后轮询本地 GVA（发送方 PUT 进来的位置）
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

    // 单调递增序列号，host 已将 src_addr[-8] 初始化为 MAGIC_VAL + rank
    GM_ADDR my_flag_addr  = src_addr + msg_size - 8;
    GM_ADDR peer_flag_addr = gva + peer * msg_size + msg_size - 8;
    uint32_t my_seq   = MAGIC_VAL + (uint32_t)rank;
    uint32_t peer_seq = MAGIC_VAL + (uint32_t)peer;

    // Warmup 阶段
    for (int64_t i = 0; i < warmup; i++) {
        if (rank == 0) {
            *(__gm__ uint32_t*)my_flag_addr = my_seq;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            while (*(__gm__ uint32_t*)peer_flag_addr != peer_seq) {
                dcci_cachelines(peer_flag_addr, 8);
                AscendC::GetSystemCycle();
            }
            my_seq++;
            peer_seq++;
        } else {
            while (*(__gm__ uint32_t*)peer_flag_addr != peer_seq) {
                dcci_cachelines(peer_flag_addr, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            *(__gm__ uint32_t*)my_flag_addr = my_seq;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            my_seq++;
            peer_seq++;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 正式测试阶段
    if (rank == 0) {
        for (int64_t i = 0; i < iterations; i++) {
            int64_t iter_start = AscendC::GetSystemCycle();

            *(__gm__ uint32_t*)my_flag_addr = my_seq;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            int64_t wait_start = AscendC::GetSystemCycle();
            while (*(__gm__ uint32_t*)peer_flag_addr != peer_seq) {
                dcci_cachelines(peer_flag_addr, 8);
                AscendC::GetSystemCycle();
                if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                    *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = TIMEOUT_ERROR_CODE;
                    return;
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            my_seq++;
            peer_seq++;

            int64_t iter_end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = iter_end - iter_start;
        }
    } else {
        for (int64_t i = 0; i < iterations; i++) {
            int64_t wait_start = AscendC::GetSystemCycle();
            while (*(__gm__ uint32_t*)peer_flag_addr != peer_seq) {
                dcci_cachelines(peer_flag_addr, 8);
                AscendC::GetSystemCycle();
                if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
                    return;
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            *(__gm__ uint32_t*)my_flag_addr = my_seq;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            my_seq++;
            peer_seq++;
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

// ========== MTE带宽测试Kernel ==========
//
// 内存布局、同步机制与 RDMA 带宽版本完全对称，区别仅在于使用 MTE 引擎传输数据：
//   - aclshmemx_mte_put_nbi + SetFlag/WaitFlag(MTE3_S) 替代 put_nbi + roce_quiet
//   - notify/ack 握手仍通过 aclshmem_uint8_put_nbi（shmem 原语，走 MTE 路径）传递
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

    uint32_t notify_val = (uint32_t)(MAGIC_VAL);
    uint32_t ack_val    = (uint32_t)(MAGIC_VAL + 1);
    (void)round_id;

    if (rank == 0) {
        peer = 1;
        int64_t start_cycle = AscendC::GetSystemCycle();

        for (int64_t i = 0; i < iterations; i++) {
            GM_ADDR dst_addr = gva + peer * msg_size * block_dim + core_idx * msg_size;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)dst_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
        }

        AscendC::SyncAll();

        if (core_idx == 0) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            int64_t end_cycle = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;

            // PUT notify 到 Rank 1（src_addr 首 4 字节 = notify_val，由 host 预写）
            aclshmem_uint8_put_nbi(notify_addr, src_addr, sizeof(uint32_t), peer);

            // 等待 Rank 1 的 ack
            while (*(__gm__ uint32_t*)ack_addr != ack_val) {
                dcci_cachelines(ack_addr, 8);
                AscendC::GetSystemCycle();
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

    } else {
        peer = 0;

        if (core_idx == 0) {
            // 等待 Rank 0 的 notify
            while (*(__gm__ uint32_t*)notify_addr != notify_val) {
                dcci_cachelines(notify_addr, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // PUT ack 回 Rank 0（src_addr 首 4 字节 = ack_val，由 host 预写）
            aclshmem_uint8_put_nbi(ack_addr, src_addr, sizeof(uint32_t), peer);
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
