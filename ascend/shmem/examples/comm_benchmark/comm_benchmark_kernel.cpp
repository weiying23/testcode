/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Comm Benchmark Kernel - 通信性能测试Kernel
 */

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

#define MAGIC_VAL 1000
#define TIMEOUT_CYCLES 100000000LL
#define TIMEOUT_ERROR_CODE -1

// 显式定义等待逻辑，避免 Lambda 导致的 host/device 属性冲突
__aicore__ inline bool perform_wait(uint64_t addr, uint32_t target_val) {
    int64_t wait_start = AscendC::GetSystemCycle();
    while (true) {
        // 使用针对地址的底层 Cache 刷新指令
        dcci_cachelines(addr, sizeof(uint32_t));

        // 显式从内存读取
        if (*(__gm__ uint32_t*)addr == target_val) {
            return true;
        }
        if (AscendC::GetSystemCycle() - wait_start > TIMEOUT_CYCLES) {
            return false;
        }
        // 简单的执行流挂起
        for (int k = 0; k < 50; k++) {
            __asm__ __volatile__("" : : : "memory");
        }
    }
}

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

    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR peer_flag_ptr = gva + peer * msg_size + msg_size - 8;
    GM_ADDR self_flag_ptr = src_addr + msg_size - 8;

    bool timeout_occurred = false;
    int64_t total_iters = warmup + iterations;

    for (int64_t i = 0; i < total_iters; i++) {
        uint32_t rank0_expect = MAGIC_VAL + i;
        uint32_t rank1_expect = MAGIC_VAL + i + 5000;

        if (rank == 0) {
            int64_t t_start = (i >= warmup) ? AscendC::GetSystemCycle() : 0;

            // 1. 准备数据
            *(__gm__ uint32_t*)self_flag_ptr = rank0_expect;

            // 2. 使用 PipeBarrier 确保 Store 落地
            AscendC::PipeBarrier<PIPE_ALL>();

            // 3. 发送
            aclshmem_uint8_put_nbi(gva + peer * msg_size, src_addr, msg_size, peer);
            aclshmem_quiet();

            // 4. 等待回信
            if (!perform_wait((uint64_t)peer_flag_ptr, rank1_expect)) { timeout_occurred = true; break; }

            if (i >= warmup) {
                int64_t t_end = AscendC::GetSystemCycle();
                *(__gm__ int64_t*)(result_buffer + (i - warmup) * sizeof(int64_t)) = t_end - t_start;
            }
        } else {
            // Rank 1 等待
            if (!perform_wait((uint64_t)peer_flag_ptr, rank0_expect)) { timeout_occurred = true; break; }

            *(__gm__ uint32_t*)self_flag_ptr = rank1_expect;
            AscendC::PipeBarrier<PIPE_ALL>();

            aclshmem_uint8_put_nbi(gva + peer * msg_size, src_addr, msg_size, peer);
            aclshmem_quiet();
        }

        // Iteration 同步与复位
        aclshmem_barrier_all();
        *(__gm__ uint32_t*)self_flag_ptr = 0;
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmem_barrier_all();
    }

    if (timeout_occurred && rank == 0) {
        *(__gm__ int64_t*)result_buffer = TIMEOUT_ERROR_CODE;
    }
}

// ========== SDMA PingPong延迟测试Kernel ==========
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void sdma_pingpong_latency_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    int64_t warmup,
    GM_ADDR result_buffer) {

    util_set_ffts_config(ffts_config);
    if (AscendC::GetSubBlockIdx() != 0) return;

    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR peer_flag_ptr = gva + peer * msg_size + msg_size - 8;
    GM_ADDR self_flag_ptr = src_addr + msg_size - 8;

    bool timeout_occurred = false;
    int64_t total_iters = warmup + iterations;

    for (int64_t i = 0; i < total_iters; i++) {
        uint32_t rank0_expect = MAGIC_VAL + i;
        uint32_t rank1_expect = MAGIC_VAL + i + 5000;

        if (rank == 0) {
            int64_t t_start = (i >= warmup) ? AscendC::GetSystemCycle() : 0;

            *(__gm__ uint32_t*)self_flag_ptr = rank0_expect;
            AscendC::PipeBarrier<PIPE_ALL>();

            // SDMA使用相同的put接口
            aclshmem_uint8_put_nbi(gva + peer * msg_size, src_addr, msg_size, peer);
            aclshmem_quiet();

            if (!perform_wait((uint64_t)peer_flag_ptr, rank1_expect)) { timeout_occurred = true; break; }

            if (i >= warmup) {
                int64_t t_end = AscendC::GetSystemCycle();
                *(__gm__ int64_t*)(result_buffer + (i - warmup) * sizeof(int64_t)) = t_end - t_start;
            }
        } else {
            if (!perform_wait((uint64_t)peer_flag_ptr, rank0_expect)) { timeout_occurred = true; break; }

            *(__gm__ uint32_t*)self_flag_ptr = rank1_expect;
            AscendC::PipeBarrier<PIPE_ALL>();

            aclshmem_uint8_put_nbi(gva + peer * msg_size, src_addr, msg_size, peer);
            aclshmem_quiet();
        }

        aclshmem_barrier_all();
        *(__gm__ uint32_t*)self_flag_ptr = 0;
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmem_barrier_all();
    }

    if (timeout_occurred && rank == 0) {
        *(__gm__ int64_t*)result_buffer = TIMEOUT_ERROR_CODE;
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

    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR peer_flag_ptr = gva + peer * msg_size + msg_size - 8;
    GM_ADDR self_flag_ptr = src_addr + msg_size - 8;

    bool timeout_occurred = false;
    int64_t total_iters = warmup + iterations;

    for (int64_t i = 0; i < total_iters; i++) {
        uint32_t rank0_expect = MAGIC_VAL + i;
        uint32_t rank1_expect = MAGIC_VAL + i + 5000;

        if (rank == 0) {
            int64_t t_start = (i >= warmup) ? AscendC::GetSystemCycle() : 0;

            *(__gm__ uint32_t*)self_flag_ptr = rank0_expect;
            AscendC::PipeBarrier<PIPE_ALL>();

            // MTE put接口
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)(gva + peer * msg_size), (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            if (!perform_wait((uint64_t)peer_flag_ptr, rank1_expect)) { timeout_occurred = true; break; }

            if (i >= warmup) {
                int64_t t_end = AscendC::GetSystemCycle();
                *(__gm__ int64_t*)(result_buffer + (i - warmup) * sizeof(int64_t)) = t_end - t_start;
            }
        } else {
            if (!perform_wait((uint64_t)peer_flag_ptr, rank0_expect)) { timeout_occurred = true; break; }

            *(__gm__ uint32_t*)self_flag_ptr = rank1_expect;
            AscendC::PipeBarrier<PIPE_ALL>();

            aclshmemx_mte_put_nbi((__gm__ uint8_t*)(gva + peer * msg_size), (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        }

        aclshmem_barrier_all();
        *(__gm__ uint32_t*)self_flag_ptr = 0;
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmem_barrier_all();
    }

    if (timeout_occurred && rank == 0) {
        *(__gm__ int64_t*)result_buffer = TIMEOUT_ERROR_CODE;
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

    GM_ADDR src_addr = gva + rank * msg_size * block_dim + core_idx * msg_size;
    GM_ADDR result_addr = result_buffer;

    int64_t sync_base_offset = rank_size * msg_size * block_dim;
    GM_ADDR notify_addr = gva + sync_base_offset + 8;
    GM_ADDR ack_addr = gva + sync_base_offset + 16;

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
            AscendC::PipeBarrier<PIPE_ALL>();
            aclshmem_uint8_put_nbi(notify_addr, notify_addr, sizeof(uint32_t), peer);
            aclshmem_quiet();

            if (!perform_wait((uint64_t)ack_addr, MAGIC_VAL + 5000)) {
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
            if (!perform_wait((uint64_t)notify_addr, MAGIC_VAL)) return;

            *(__gm__ uint32_t*)ack_addr = MAGIC_VAL + 5000;
            AscendC::PipeBarrier<PIPE_ALL>();
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
            AscendC::PipeBarrier<PIPE_ALL>();
            aclshmem_uint8_put_nbi(notify_addr, notify_addr, sizeof(uint32_t), peer);
            aclshmem_quiet();

            if (!perform_wait((uint64_t)ack_addr, MAGIC_VAL + 5000)) {
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
            if (!perform_wait((uint64_t)notify_addr, MAGIC_VAL)) return;

            *(__gm__ uint32_t*)ack_addr = MAGIC_VAL + 5000;
            AscendC::PipeBarrier<PIPE_ALL>();
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
            AscendC::PipeBarrier<PIPE_ALL>();
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)notify_addr, (__gm__ uint8_t*)notify_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, sizeof(uint32_t), peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            if (!perform_wait((uint64_t)ack_addr, MAGIC_VAL + 5000)) {
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
            if (!perform_wait((uint64_t)notify_addr, MAGIC_VAL)) return;

            *(__gm__ uint32_t*)ack_addr = MAGIC_VAL + 5000;
            AscendC::PipeBarrier<PIPE_ALL>();
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