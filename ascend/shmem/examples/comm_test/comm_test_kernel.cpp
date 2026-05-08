/**
 * MTE vs SDMA cross-card bandwidth benchmark kernels.
 * Kernel only does the puts + completion sync.
 * All timing is done host-side with chrono.
 */

#include "kernel_operator.h"
#include "shmem.h"

// ===== MTE Bandwidth Kernel =====
// All cores issue mte_put_nbi, core 0 waits for DMA completion, then barrier.
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void mte_bw_kernel(GM_ADDR gva, int64_t msg_size, int64_t iterations, int64_t block_dim)
{
    int64_t rank     = aclshmem_my_pe();
    int64_t core_idx = AscendC::GetBlockIdx();
    uint32_t peer    = 1 - (uint32_t)rank;

    if (rank != 0) {
        aclshmemx_barrier_all_vec();
        return;
    }

    __gm__ aclshmem_device_host_state_t *st = aclshmemi_get_state();
    uint64_t copy_ub   = st->mte_config.aclshmem_ub;
    uint32_t copy_size = st->mte_config.ub_size;
    AscendC::TEventID ev = (AscendC::TEventID)st->mte_config.sync_id;

    int64_t slice  = msg_size / block_dim;
    int64_t offset = core_idx * slice;
    __gm__ uint8_t *src = (__gm__ uint8_t *)gva + offset;
    __gm__ uint8_t *dst = (__gm__ uint8_t *)gva + offset;

    AscendC::PipeBarrier<PIPE_ALL>();

    for (int64_t i = 0; i < iterations; i++) {
        aclshmemx_mte_put_nbi(dst, src,
                              reinterpret_cast<__ubuf__ uint8_t *>(copy_ub),
                              copy_size, (int32_t)slice, (int32_t)peer, ev);
    }

    AscendC::SyncAll();
    if (core_idx == 0) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(ev);
    }

    aclshmemx_barrier_all_vec();
}

// ===== SDMA Bandwidth Kernel =====
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void sdma_bw_kernel(GM_ADDR gva, int64_t msg_size, int64_t iterations, int64_t block_dim)
{
    int64_t rank     = aclshmem_my_pe();
    int64_t core_idx = AscendC::GetBlockIdx();
    uint32_t peer    = 1 - (uint32_t)rank;

    if (rank != 0) {
        aclshmemx_barrier_all_vec();
        return;
    }

    constexpr uint32_t UB_OFFSET  = 1024;
    constexpr uint32_t SDMA_UB_SZ = 16 * 1024;   // 16KB, matches original comm_test default
    __ubuf__ uint8_t *tmp = reinterpret_cast<__ubuf__ uint8_t *>(UB_OFFSET);

    // divide by block_dim only (same as MTE), so each core gets a reasonable slice
    int64_t slice  = msg_size / block_dim;
    int64_t offset = core_idx * slice;

    if (slice == 0) {
        aclshmemx_barrier_all_vec();
        return;
    }

    __gm__ uint8_t *src = (__gm__ uint8_t *)gva + offset;
    __gm__ uint8_t *dst = (__gm__ uint8_t *)gva + offset;

    AscendC::PipeBarrier<PIPE_ALL>();

    for (int64_t i = 0; i < iterations; i++) {
        aclshmemx_sdma_put_nbi(dst, src, tmp, SDMA_UB_SZ,
                               (uint64_t)slice, (int32_t)peer, EVENT_ID0);
    }
    aclshmemx_sdma_quiet(tmp, SDMA_UB_SZ, EVENT_ID0);

    aclshmemx_barrier_all_vec();
}

// ===== Host-side launchers =====
extern "C" void launch_mte_bw(uint32_t bdim, void *stream, uint8_t *gva,
                               int64_t msg_size, int64_t iters)
{
    mte_bw_kernel<<<bdim, nullptr, stream>>>(gva, msg_size, iters, (int64_t)bdim);
}

extern "C" void launch_sdma_bw(uint32_t bdim, void *stream, uint8_t *gva,
                                int64_t msg_size, int64_t iters)
{
    sdma_bw_kernel<<<bdim, nullptr, stream>>>(gva, msg_size, iters, (int64_t)bdim);
}
