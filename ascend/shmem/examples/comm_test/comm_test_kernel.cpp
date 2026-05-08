/**
 * MTE vs SDMA cross-card bandwidth benchmark kernels.
 * Timing: GetSystemCycle() directly in kernel, result written to result_buf.
 * Rank 0 does all the puts and timing; Rank 1 waits barrier.
 */

#include "kernel_operator.h"
#include "shmem.h"

// ===== MTE Bandwidth Kernel =====
// Rank 0: times (iterations) mte_put_nbi calls, writes elapsed cycles to result_buf[0].
// Rank 1: waits barrier and returns.
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void mte_bw_kernel(GM_ADDR gva, int64_t msg_size, int64_t iterations,
                   int64_t block_dim, GM_ADDR result_buf)
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
    int64_t start = AscendC::GetSystemCycle();

    for (int64_t i = 0; i < iterations; i++) {
        aclshmemx_mte_put_nbi(dst, src,
                              reinterpret_cast<__ubuf__ uint8_t *>(copy_ub),
                              copy_size, (int32_t)slice, (int32_t)peer, ev);
    }

    AscendC::SyncAll();

    if (core_idx == 0) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(ev);
        int64_t end = AscendC::GetSystemCycle();
        *(__gm__ int64_t *)result_buf = end - start;
    }

    aclshmemx_barrier_all_vec();
}

// ===== SDMA Bandwidth Kernel =====
// Same structure, but uses aclshmemx_sdma_put_nbi + sdma_quiet.
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void sdma_bw_kernel(GM_ADDR gva, int64_t msg_size, int64_t iterations,
                    int64_t block_dim, GM_ADDR result_buf)
{
    int64_t rank     = aclshmem_my_pe();
    int64_t core_idx = AscendC::GetBlockIdx();
    uint32_t peer    = 1 - (uint32_t)rank;

    if (rank != 0) {
        aclshmemx_barrier_all_vec();
        return;
    }

    constexpr uint32_t UB_OFFSET  = 1024;
    constexpr uint32_t SDMA_UB_SZ = 256 * 1024;
    __ubuf__ uint8_t *tmp = reinterpret_cast<__ubuf__ uint8_t *>(UB_OFFSET);

    int64_t comm_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    int64_t base     = msg_size / comm_dim;
    int64_t extra    = msg_size % comm_dim;
    int64_t my_bytes = base + (core_idx < extra ? 1 : 0);
    int64_t my_off   = core_idx < extra
                           ? core_idx * (base + 1)
                           : extra * (base + 1) + (core_idx - extra) * base;

    if (my_bytes == 0) {
        aclshmemx_barrier_all_vec();
        return;
    }

    __gm__ uint8_t *src = (__gm__ uint8_t *)gva + my_off;
    __gm__ uint8_t *dst = (__gm__ uint8_t *)gva + my_off;

    AscendC::PipeBarrier<PIPE_ALL>();
    int64_t start = AscendC::GetSystemCycle();

    for (int64_t i = 0; i < iterations; i++) {
        aclshmemx_sdma_put_nbi(dst, src, tmp, SDMA_UB_SZ,
                               (uint64_t)my_bytes, (int32_t)peer, EVENT_ID0);
    }
    aclshmemx_sdma_quiet(tmp, SDMA_UB_SZ, EVENT_ID0);

    AscendC::SyncAll();

    if (core_idx == 0) {
        int64_t end = AscendC::GetSystemCycle();
        *(__gm__ int64_t *)result_buf = end - start;
    }

    aclshmemx_barrier_all_vec();
}

// ===== Host-side launchers =====
extern "C" void launch_mte_bw(uint32_t bdim, void *stream, uint8_t *gva,
                               int64_t msg_size, int64_t iters, uint8_t *result_buf)
{
    mte_bw_kernel<<<bdim, nullptr, stream>>>(gva, msg_size, iters, (int64_t)bdim, result_buf);
}

extern "C" void launch_sdma_bw(uint32_t bdim, void *stream, uint8_t *gva,
                                int64_t msg_size, int64_t iters, uint8_t *result_buf)
{
    sdma_bw_kernel<<<bdim, nullptr, stream>>>(gva, msg_size, iters, (int64_t)bdim, result_buf);
}
