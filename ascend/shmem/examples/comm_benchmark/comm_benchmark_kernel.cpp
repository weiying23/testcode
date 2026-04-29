/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Comm Benchmark Kernel - 所有通信性能测试Kernel的集合
 */

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

constexpr uint32_t MAGIC_VAL = 12345;
constexpr uint32_t MAGIC_VAL_BW = 10;

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

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE);
    AscendC::LocalTensor<uint32_t> ubLocal = buf.GetWithOffset<uint32_t>(UB_ALIGN_SIZE / sizeof(uint32_t), 0);

    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    // Warmup阶段
    for (int64_t i = 0; i < warmup; i++) {
        if (rank == 0) {
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);
            while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != peer + MAGIC_VAL + i) {
                dcci_cachelines(gva + msg_size * 2 - 8, 8);
                AscendC::GetSystemCycle();
            }
        } else {
            while (*(__gm__ uint32_t*)(gva + msg_size * 1 - 8) != peer + MAGIC_VAL + i) {
                dcci_cachelines(gva + msg_size * 1 - 8, 8);
                AscendC::GetSystemCycle();
            }
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 正式测试阶段
    if (rank == 0) {
        for (int64_t i = 0; i < iterations; i++) {
            int64_t iter_start = AscendC::GetSystemCycle();

            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);
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
            while (*(__gm__ uint32_t*)(gva + msg_size * 1 - 8) != peer + MAGIC_VAL + warmup + i) {
                dcci_cachelines(gva + msg_size * 1 - 8, 8);
                AscendC::GetSystemCycle();
            }
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

// ========== RDMA带宽测试Kernel ==========
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

    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    if (rank == 0) {
        int64_t start_cycle = AscendC::GetSystemCycle();

        for (int64_t i = 0; i < iterations; i++) {
            aclshmem_uint8_put_nbi(src_addr, src_addr, msg_size, peer);
        }

        aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);

        int64_t end_cycle = AscendC::GetSystemCycle();
        *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;
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

    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;

    int64_t rank = aclshmem_my_pe();
    uint32_t peer = (rank == 0) ? 1 : 0;

    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    // Warmup阶段
    for (int64_t i = 0; i < warmup; i++) {
        if (rank == 0) {
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

            while (*(__gm__ uint32_t*)(gva + msg_size * 2 - 8) != peer + MAGIC_VAL + i) {
                dcci_cachelines(gva + msg_size * 2 - 8, 8);
                AscendC::GetSystemCycle();
            }
        } else {
            while (*(__gm__ uint32_t*)(gva + msg_size * 1 - 8) != peer + MAGIC_VAL + i) {
                dcci_cachelines(gva + msg_size * 1 - 8, 8);
                AscendC::GetSystemCycle();
            }
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 正式测试
    if (rank == 0) {
        for (int64_t i = 0; i < iterations; i++) {
            int64_t iter_start = AscendC::GetSystemCycle();

            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

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
            while (*(__gm__ uint32_t*)(gva + msg_size * 1 - 8) != peer + MAGIC_VAL + warmup + i) {
                dcci_cachelines(gva + msg_size * 1 - 8, 8);
                AscendC::GetSystemCycle();
            }
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
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_bandwidth_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
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
        int64_t start_cycle = AscendC::GetSystemCycle();

        for (int64_t i = 0; i < iterations; i++) {
            aclshmemx_mte_put_nbi((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr,
                                  reinterpret_cast<__ubuf__ uint8_t*>(copy_ub),
                                  copy_ub_size, msg_size, peer, copy_event_id);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);

        int64_t end_cycle = AscendC::GetSystemCycle();
        *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;
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
