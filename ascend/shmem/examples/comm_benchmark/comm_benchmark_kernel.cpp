/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Comm Benchmark Kernel - 通信性能测试Kernel
 */

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

constexpr uint32_t MAGIC_VAL = 12345;
constexpr uint32_t MAGIC_VAL_BW = 10;

// ========== RDMA PingPong延迟测试Kernel ==========
// 参考rdma_perftest的pingpong逻辑：
// - Rank 0: 写入 MAGIC_VAL，发送到 peer slot，等待 peer+MAGIC_VAL 响应
// - Rank 1: 等待 MAGIC_VAL，写入 peer+MAGIC_VAL 响应，发送回去
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

    // 内存布局（参考rdma_perftest）：
    // Slot 0 (gva + 0*msg_size): 数据区
    // Slot 0末尾 (gva + msg_size - 8): rank 0写入magic value，rank 1等待检测
    // Slot 1末尾 (gva + 2*msg_size - 8): rank 1写入magic value，rank 0等待检测
    // 注意：put操作会把整个slot数据复制到peer slot，包括magic value
    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    // Warmup阶段
    for (int64_t i = 0; i < warmup; i++) {
        if (rank == 0) {
            // Rank 0: 写入 MAGIC_VAL + i，发送到 peer slot 0
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = MAGIC_VAL + i;

            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmem_uint8_put_nbi(dst_addr, src_addr, msg_size, peer);

            // 等待 rank 1 响应：检测 slot 1 末尾的 peer(1) + MAGIC_VAL + i
            // 注意：这里检测的是 gva + msg_size * 2 - 8（slot 1末尾）
            while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != peer + MAGIC_VAL + i) {
                dcci_cachelines(gva + msg_size * 2 - 8, 8);
                AscendC::GetSystemCycle();
            }
        } else {
            // Rank 1: 等待 rank 0 的 MAGIC_VAL + i
            // 注意：检测 slot 0 末尾，即 gva + msg_size - 8（不是 msg_size * 1 - 8）
            while (*(__gm__ uint32_t*)(gva + msg_size - 8) != MAGIC_VAL + i) {
                dcci_cachelines(gva + msg_size - 8, 8);
                AscendC::GetSystemCycle();
            }

            // 写入响应：peer(0) + MAGIC_VAL + i
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = peer + MAGIC_VAL + i;

            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmem_uint8_put_nbi(dst_addr, src_addr, msg_size, peer);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 正式测试阶段
    if (rank == 0) {
        for (int64_t i = 0; i < iterations; i++) {
            int64_t iter_start = AscendC::GetSystemCycle();

            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = MAGIC_VAL + warmup + i;

            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmem_uint8_put_nbi(dst_addr, src_addr, msg_size, peer);

            while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != peer + MAGIC_VAL + warmup + i) {
                dcci_cachelines(gva + msg_size * 2 - 8, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            int64_t iter_end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = iter_end - iter_start;
        }
    } else {
        for (int64_t i = 0; i < iterations; i++) {
            while (*(__gm__ uint32_t*)(gva + msg_size - 8) != MAGIC_VAL + warmup + i) {
                dcci_cachelines(gva + msg_size - 8, 8);
                AscendC::GetSystemCycle();
            }

            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = peer + MAGIC_VAL + warmup + i;

            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmem_uint8_put_nbi(dst_addr, src_addr, msg_size, peer);
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

        // 记录开始时间
        int64_t start_cycle = AscendC::GetSystemCycle();

        // 所有 Core 都执行数据发送
        for (int64_t i = 0; i < iterations; i++) {
            // 目标地址：peer的对应slot
            GM_ADDR dst_addr = gva + peer * msg_size * block_dim + core_idx * msg_size;
            aclshmem_uint8_put_nbi(dst_addr, src_addr, msg_size, peer);
        }

        // 只有 Core 0 执行同步操作
        if (core_idx == 0) {
            // aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
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
// MTE引擎特点：
// - 用于节点内NPU间通信
// - 使用片上MTE单元进行数据传输
// - 高带宽、低延迟，适合大规模数据传输
//
// 参考rdma_perftest的pingpong逻辑：
// - Rank 0 写入 MAGIC_VAL，发送到 peer，等待 peer+MAGIC_VAL 响应
// - Rank 1 等待 peer+MAGIC_VAL，写入 peer+MAGIC_VAL 响应，发送回去
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

    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    // 内存布局（参考rdma_perftest）
    // Slot 0 (gva + 0*msg_size): rank 0的发送数据区，末尾是rank 1的轮询位置
    // Slot 1 (gva + 1*msg_size): rank 1的发送数据区，末尾是rank 0的轮询位置
    // 等待位置：gva + msg_size * 2 - 8 用于存储结果（不用于同步）
    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    // Warmup阶段（参考rdma_perftest的逻辑）
    for (int64_t i = 0; i < warmup; i++) {
        if (rank == 0) {
            // Rank 0: 写入 MAGIC_VAL + i 到自己 slot 末尾，发送到 peer
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = MAGIC_VAL + i;

            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)dst_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            // 等待 rank 1 的响应：检测 slot 1 末尾的 peer(1) + MAGIC_VAL + i
            while (*(__gm__ uint32_t*)(gva + peer * msg_size + msg_size - 8) != peer + MAGIC_VAL + i) {
                dcci_cachelines(gva + peer * msg_size + msg_size - 8, 8);
                AscendC::GetSystemCycle();
            }
        } else {
            // Rank 1: 等待 rank 0 的数据：检测 slot 0 末尾的 MAGIC_VAL + i
            while (*(__gm__ uint32_t*)(gva + peer * msg_size + msg_size - 8) != MAGIC_VAL + i) {
                dcci_cachelines(gva + peer * msg_size + msg_size - 8, 8);
                AscendC::GetSystemCycle();
            }

            // 写入响应：peer(0) + MAGIC_VAL + i
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = peer + MAGIC_VAL + i;

            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)dst_addr, (__gm__ uint8_t*)src_addr,
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
            int64_t iter_start = AscendC::GetSystemCycle();

            // 写入 magic value
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = MAGIC_VAL + warmup + i;

            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)dst_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            // 等待响应：peer(1) + MAGIC_VAL + warmup + i
            while (*(__gm__ uint32_t*)(gva + peer * msg_size + msg_size - 8) != peer + MAGIC_VAL + warmup + i) {
                dcci_cachelines(gva + peer * msg_size + msg_size - 8, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            int64_t iter_end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(result_addr + i * sizeof(int64_t)) = iter_end - iter_start;
        }
    } else {
        for (int64_t i = 0; i < iterations; i++) {
            // 等待 rank 0：MAGIC_VAL + warmup + i
            while (*(__gm__ uint32_t*)(gva + peer * msg_size + msg_size - 8) != MAGIC_VAL + warmup + i) {
                dcci_cachelines(gva + peer * msg_size + msg_size - 8, 8);
                AscendC::GetSystemCycle();
            }

            // 写入响应：peer(0) + MAGIC_VAL + warmup + i
            *(__gm__ uint32_t*)(src_addr + msg_size - 8) = peer + MAGIC_VAL + warmup + i;

            GM_ADDR dst_addr = gva + peer * msg_size;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)dst_addr, (__gm__ uint8_t*)src_addr,
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
            // 目标地址：peer的对应slot
            GM_ADDR dst_addr = gva + peer * msg_size * block_dim + core_idx * msg_size;
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)dst_addr, (__gm__ uint8_t*)src_addr,
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
