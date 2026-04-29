# AllGather
该样例工程位于examples\allgather文件夹下。

在该样例中，实现了一个通信量较小(单PE通信量小于2MB)情况下，有着更低时延的AllGather纯通信算子。 各PE首先将存在本端input地址下的数据push到本PE的对称内存上；确认远端PE的任务完成后，从远端PE的对称内存拉取对应PE的数据，从而整体完成AllGather的操作。这个样例展示了多种shmem API的用法，包括shmem_mte_put_mem_nbi、shmemx_signal_op以及shmem_mte_get_mem_nbi等，用于p2p的通信以及同步任务。

## 核函数实现
```c++

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem_api.h"
using namespace AscendC;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;

template<typename T>
SHMEM_DEVICE void all_gather_small_data(uint64_t fftsAddr, __gm__ T* input, __gm__ T* output, __gm__ T* gva, int elements, int magic)
{
#ifdef __DAV_C220_VEC__
    const int64_t aivNum = GetBlockNum() * 2;
    const int64_t aivIndex = GetBlockIdx();

    const int64_t data_offset = aivNum * SYNC_FLAG_INTERVAL;
    const int64_t flag_offset = aivIndex * SYNC_FLAG_INTERVAL;

    int64_t my_rank = shmem_my_pe();
    int64_t pe_size = shmem_n_pes();

    __gm__ T *input_gm = (__gm__ T *)input;
    __gm__ T *output_gm = (__gm__ T *)output;

    __gm__ T *gva_data_gm = (__gm__ T*)((__gm__ int32_t*)gva + data_offset);
    __gm__ int32_t *gva_sync_gm = (__gm__ int32_t *)gva;
    
    __ubuf__ T* tmp_buff = (__ubuf__ T*)(64);

    // data move parameters
    const uint32_t ub_size = UB_DMA_MAX_SIZE;
    uint32_t input_offset, output_offset, gva_offset, num_per_core;

    // [AllGather Step 1] local input gm -> symmetric mem.
    num_per_core = elements / aivNum;
    input_offset = aivIndex * num_per_core;
    gva_offset = aivIndex * num_per_core;
    if (aivIndex == aivNum - 1) {
        num_per_core = elements - num_per_core * aivIndex;
    }
    shmem_mte_put_mem_nbi(gva_data_gm + gva_offset, input_gm + input_offset, tmp_buff, ub_size, num_per_core, my_rank, EVENT_ID0);

    const int64_t core_per_rank = aivNum / pe_size;
    const int64_t core_rank_idx = aivIndex % core_per_rank;
    const int64_t x = aivIndex / core_per_rank;

    // Sync Ensure Corresponding Tasks Done.
    shmem_quiet();
    shmemi_barrier_core_soft();

    shmemx_signal_op(gva_sync_gm + flag_offset, magic, SHMEM_SIGNAL_SET, my_rank);
    shmem_signal_wait_until((__gm__ int32_t *)shmem_ptr(gva_sync_gm, x) + flag_offset, SHMEM_CMP_EQ, magic);

    // [AllGather Step 2] symmetric mem -> local output.
    num_per_core = elements / core_per_rank;
    output_offset = x * elements + core_rank_idx * num_per_core;
    gva_offset = core_rank_idx * num_per_core;
    if (core_rank_idx == core_per_rank - 1) {
        num_per_core = elements - num_per_core * core_rank_idx;
    }
    shmem_mte_get_mem_nbi(output_gm + output_offset, gva_data_gm + gva_offset, tmp_buff, ub_size, num_per_core, x, EVENT_ID0);
#endif
}
```