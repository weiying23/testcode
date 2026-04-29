/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "shmem_api.h"

extern "C" SHMEM_GLOBAL void p2p_chain(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    shmemx_set_ffts_config(config);
    auto sig_addr = (__gm__ int32_t *)addr;
    int32_t val = *sig_addr;
    int next = (rank_id + 1) % rank_size;

    shmem_barrier_all();

#ifdef __DAV_C220_VEC__
    if (rank_id == 0) {
        shmemx_signal_op(sig_addr, 1, SHMEM_SIGNAL_SET, next);
        shmem_signal_wait_until(sig_addr, SHMEM_CMP_EQ, 1);
    } else {
        shmem_signal_wait_until(sig_addr, SHMEM_CMP_EQ, 1);
        shmemx_signal_op(sig_addr, 1, SHMEM_SIGNAL_SET, next);
    }
#endif

    shmem_barrier_all();

#ifdef __DAV_C220_VEC__
    if (rank_id == 0) {
        shmemx_signal_op(sig_addr, 1, SHMEM_SIGNAL_ADD, next);
        shmem_signal_wait_until(sig_addr, SHMEM_CMP_EQ, 1 + AscendC::GetBlockNum() * AscendC::GetTaskRation());
    } else {
        shmem_signal_wait_until(sig_addr, SHMEM_CMP_EQ, 1 + AscendC::GetBlockNum() * AscendC::GetTaskRation());
        shmemx_signal_op(sig_addr, 1, SHMEM_SIGNAL_ADD, next);
    }
#endif

    shmem_barrier_all();
}

void p2p_chain_do(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size)
{
    p2p_chain<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size);
}