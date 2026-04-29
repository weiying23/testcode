/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
    WARNING：Restrictions of Barrier APIs.

    1. Barrier APIs can be used only in MIX kernels. The compiler will optimize the kernel to VEC or CUBE if
       it lacks effective cube instructions (eg. Mmad) or vector instructions (eg: DataCopy).
    Need compiler updates to remove this feature, or insert Mmad/DataCopy calls manully.

    2. Barrier APIs conflict with SyncAll. Avoid mixing them together.

    3. We provide 2 kinds of barrier:
        a. shmem_barrier_xxx
            Barrier of all cores. On systems with only HCCS: All operations of all ranks of a team on excuting
            stream before the barrier are visiable to all ranks of the team after the barrier.
        b. shmemx_barrier_xxx_vec
            Barrier of all VEC cores. On systems with only HCCS: All operations of ALL VEC CORES of all ranks of a
            team on excuting stream before the barrier are visiable to ALL VEC CORES of all ranks of the team after
            the barrier.

        This subtle difference is beneficial to compute-communiction overlapping (usually UNI_DIRECTIONAL dependency),
        and could achieve better performance. Refer to examples/matmul_allreduce for details.

    4. The scalar unit of cube core is not affected by shmem_barrier_xxx. Make sure don't use that.
*/

#ifndef SHMEM_DEVICE_SYNC_H
#define SHMEM_DEVICE_SYNC_H

#include "host_device/shmem_types.h"
#include "internal/device/sync/shmemi_device_quiet.h"
#include "internal/device/sync/shmemi_device_p2p.h"
#include "internal/device/sync/shmemi_device_barrier.h"
#include "internal/device/sync/shmemi_device_handle.h"
#include "internal/device/sync/shmemi_device_partial_barrier.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn SHMEM_DEVICE void shmemx_set_ffts_config(uint64_t config)
 * @brief Set runtime ffts address. Call this at MIX Kernel entry point (if the kernel contains barrier calls).
 *
 * @param config              [config] ffts config, acquired by shmemx_get_ffts_config()
 */
SHMEM_DEVICE void shmemx_set_ffts_config(uint64_t config)
{
    AscendC::SetSyncBaseAddr(config);
}

/**
 * @fn SHMEM_DEVICE void shmem_barrier(shmem_team_t tid)
 * @brief shmem_barrier is a collective synchronization routine over a team. Control returns from shmem_barrier
 *        after all PEs in the team have called shmem_barrier.
 *        shmem_barrier ensures that all previously issued stores and remote memory updates, including AMOs and
 *        RMA operations, done by any of the PEs in the active set are complete before returning. On systems with
 *        only scale-up network (HCCS), updates are globally visible, whereas on systems with both scale-up network
 *        HCCS and scale-out network (RDMA), SHMEM only guarantees that updates to the memory of a given PE are
 *        visible to that PE.
 *        Barrier operations issued on the CPU and the NPU only complete communication operations that were issued
 *        from the CPU and the NPU, respectively. To ensure completion of GPU-side operations from the CPU, using
 *        aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 * @param tid              [in] team to do barrier
 */
SHMEM_DEVICE void shmem_barrier(shmem_team_t tid)
{
    shmemi_barrier<false>(tid);
}

/**
 * @fn SHMEM_DEVICE void shmem_barrier_all()
 * @brief shmem_barrier of all PEs.
 */
SHMEM_DEVICE void shmem_barrier_all()
{
    shmem_barrier(SHMEM_TEAM_WORLD);
}

/**
 * @brief Similar to shmem_barrier except that only vector cores participate. Useful in communication-over-compute
 *        operators. Cube core may call the api but takes no effect.
 *
 * @param tid              [in] team to do barrier
 */
SHMEM_DEVICE void shmemx_barrier_vec(shmem_team_t tid)
{
    shmemi_barrier<true>(tid);
}

/**
 * @brief shmemx_barrier_vec of all PEs.
 *
 * @param tid              [in] team to do barrier
 */
SHMEM_DEVICE void shmemx_barrier_all_vec()
{
    shmemx_barrier_vec(SHMEM_TEAM_WORLD);
}

/**
 * @brief Partial barrier: synchronize only the PEs listed in pes[0..count-1].
 *
 * @param tid   [in] Team handle.
 * @param pes   [in] Team PE ID array participating in this partial barrier.
 * @param count [in] Length of the pes array.
 */
SHMEM_DEVICE void shmemx_partial_barrier(shmem_team_t tid, __gm__ uint32_t *pes, uint32_t count)
{
    shmemi_partial_barrier<false>(tid, pes, count);
}

/**
 * @brief Partial barrier (vector version): synchronize only the subset of VEC cores
 *        corresponding to the PEs listed in pes[0..count-1].
 *
 * @param tid   [in] Team handle.
 * @param pes   [in] Team PE ID array participating in this partial barrier.
 * @param count [in] Length of the pes array.
 */
SHMEM_DEVICE void shmemx_partial_barrier_vec(shmem_team_t tid, __gm__ uint32_t *pes, uint32_t count)
{
    shmemi_partial_barrier<true>(tid, pes, count);
}

/**
 * @brief The shmem_quiet routine ensures completion of all operations on symmetric data objects issued by the
 *        calling PE.
 *        On systems with only scale-up network (HCCS), updates are globally visible, whereas on systems with
 *        both scale-up network HCCS and scale-out network (RDMA), SHMEM only guarantees that updates to the
 *        memory of a given PE are visible to that PE.
 *        Quiet operations issued on the CPU and the NPU only complete communication operations that were
 *        issued from the CPU and the NPU, respectively. To ensure completion of GPU-side operations from
 *        the CPU, using aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 */
SHMEM_DEVICE void shmem_quiet()
{
    shmemi_quiet();
}

/**
 * @brief In OpenSHMEM specification, shmem_fence assures ordering of delivery of Put, AMOs, and memory store routines
 *        to symmetric data objects, but does not guarantee the completion of these operations.
 *        However, due to hardware capabilities, we implemented shmem_fence same as shmem_quiet, ensuring both ordering
 *        and completion.
 *        Fence operations issued on the CPU and the NPU only order communication operations that were issued from the
 *        CPU and the NPU, respectively. To ensure completion of GPU-side operations from the CPU,
 *        using aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 */
SHMEM_DEVICE void shmem_fence()
{
    shmemi_quiet();
}

/**
 * @brief The shmemx_signal_op operation updates sig_addr with signal using operation sig_op on the specified PE.
 *        This operation can be used together with shmem_signal_wait_until for efficient point-to-point synchronization.
 * WARNING: Atomicity NOT Guaranteed.
 *
 * @param sig_addr              [in] Symmetric address of the signal word to be updated.
 * @param signal                [in] The value used to update sig_addr.
 * @param sig_op                [in] Operation used to update sig_addr with signal. Supported operations:
 *                                   SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD
 * @param pe                    [in] PE number of the remote PE.
 */
SHMEM_DEVICE void shmemx_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
{
    shmemix_signal_op(sig_addr, signal, sig_op, pe);
}

/**
 * @brief This routine can be used to implement point-to-point synchronization between PEs or between threads within
 *        the same PE. A call to this routine blocks until the value of sig_addr at the calling PE satisfies the wait
 *        condition specified by the comparison operator, cmp, and comparison value, cmp_val.
 *
 * @param sig_addr              [in] Local address of the source signal variable.
 * @param cmp                   [in] The comparison operator that compares sig_addr with cmp_val. Supported operators:
 *                                   SHMEM_CMP_EQ/SHMEM_CMP_NE/SHMEM_CMP_GT/SHMEM_CMP_GE/SHMEM_CMP_LT/SHMEM_CMP_LE.
 * @param cmp_val               [in] The value against which the object pointed to by sig_addr will be compared.
 * @return Return the contents of the signal data object, sig_addr, at the calling PE that satisfies the wait condition.
 */
SHMEM_DEVICE int32_t shmem_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val)
{
    return shmemi_signal_wait_until(sig_addr, cmp, cmp_val);
}

#ifdef __cplusplus
}
#endif

#endif