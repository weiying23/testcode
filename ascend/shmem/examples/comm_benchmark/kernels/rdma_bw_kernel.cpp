/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * RDMA带宽测试Kernel
 */

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem_api.h"

constexpr int64_t BW_ITERATIONS = 10000;

extern "C" __global__ __aicore__ void rdma_bandwidth_kernel(
    uint64_t ffts_config,
    GM_ADDR gva,
    int64_t msg_size,
    int64_t iterations,
    GM_ADDR result_buffer) {

    shmemx_set_ffts_config(ffts_config);
    if (AscendC::GetSubBlockIdx() != 0) return;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE);
    AscendC::LocalTensor<uint32_t> ubLocal = buf.GetWithOffset<uint32_t>(UB_ALIGN_SIZE / sizeof(uint32_t), 0);

    int64_t rank = smem_shm_get_global_rank();
    int64_t rank_size = smem_shm_get_global_rank_size();
    uint32_t peer = (rank == 0) ? 1 : 0;

    GM_ADDR src_addr = gva + rank * msg_size;
    GM_ADDR result_addr = result_buffer;

    if (rank == 0) {
        int64_t start_cycle = AscendC::GetSystemCycle();

        // 连续发送iterations次
        for (int64_t i = 0; i < iterations; i++) {
            shmemi_roce_write((__gm__ uint8_t*)src_addr, (__gm__ uint8_t*)src_addr, peer, 0, msg_size, ubLocal, ubLocal);
        }

        // 等待所有发送完成
        shmemi_roce_quiet(peer, 0, ubLocal, ubLocal);

        int64_t end_cycle = AscendC::GetSystemCycle();

        // 记录总时间
        *(__gm__ int64_t*)(result_addr) = end_cycle - start_cycle;

        // 计算带宽: bandwidth = msg_size * iterations / time
        // 单位: cycles, 需要在host端转换为实际带宽
    }
}

void launch_rdma_bandwidth(uint32_t block_dim, void* stream,
                            uint64_t ffts_config, uint8_t* gva,
                            int64_t msg_size, int64_t iterations,
                            uint8_t* result_buffer) {
    rdma_bandwidth_kernel<<<1, nullptr, stream>>>(
        ffts_config, gva, msg_size, iterations, result_buffer);
}