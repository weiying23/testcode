/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "kernel_operator.h"
#include "shmem_api.h"
#include "shmem_ptr_kernel.h"

class kernel_ptr_test {
public:
    __aicore__ inline kernel_ptr_test()
    {
    }
    __aicore__ inline void Init(GM_ADDR gva)
    {
        gva_gm = (__gm__ int *)gva;
        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();
    }
    __aicore__ inline void Process()
    {
        AscendC::PipeBarrier<PIPE_ALL>();

        __gm__ int *p0 = static_cast<__gm__ int *>(shmem_ptr(gva_gm, rank));
        __gm__ int *p1 = static_cast<__gm__ int *>(shmem_ptr(gva_gm + 1, rank));
        int32_t delta_self = p1 - p0;
        shmem_int32_p(gva_gm, delta_self, rank);

        int peer = (rank + 1) % rank_size;
        __gm__ int *q0 = static_cast<__gm__ int *>(shmem_ptr(gva_gm, peer));
        __gm__ int *q1 = static_cast<__gm__ int *>(shmem_ptr(gva_gm + 1, peer));
        int32_t delta_remote = q1 - q0;
        shmem_int32_p(gva_gm + 1, delta_remote, rank);
    }

private:
    __gm__ int *gva_gm;
    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void device_ptr_test(GM_ADDR gva)
{
    kernel_ptr_test op;
    op.Init(gva);
    op.Process();
}

void get_device_ptr(uint32_t block_dim, void *stream, uint8_t *gva)
{
    device_ptr_test<<<block_dim, nullptr, stream>>>(gva);
}
