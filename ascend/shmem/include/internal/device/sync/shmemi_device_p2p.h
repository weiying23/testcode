/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHEMEI_P2P_H
#define SHEMEI_P2P_H

#include "shmemi_device_quiet.h"

SHMEM_DEVICE void shmemi_signal_set(__gm__ int32_t *addr, int32_t val)
{
    shmemi_store(addr, val);

    // flush data cache to GM after signal to ensure it is visiable to other ranks
    dcci_cacheline((__gm__ uint8_t *)addr);
}

SHMEM_DEVICE void shmemi_signal_set(__gm__ int32_t *addr, int pe, int32_t val)
{
    shmemi_signal_set(shmemi_ptr(addr, pe), val);
}

SHMEM_DEVICE void shmemi_highlevel_signal_set(__gm__ int32_t *dst, __gm__ int32_t *src, int pe)
{
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(SHMEM_INTERNAL_UB_BUF_START_ADDR
                                                                        + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    shmemi_roce_write((__gm__ uint8_t*)shmem_ptr(dst, pe), (__gm__ uint8_t*)src, pe, 0, sizeof(int32_t),
        ub_tensor_64, ub_tensor_32);
    shmemi_roce_quiet(pe, 0, ub_tensor_64, ub_tensor_32);
}

SHMEM_DEVICE void shmemi_signal_add(__gm__ int32_t *addr, int pe, int32_t val)
{
    // ensure previous atomic operations end
    dcci_atomic();
    dsb_all();

    // atomic add
    set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
    st_atomic<int32_t>(val, shmemi_ptr(addr, pe));
    dcci_atomic();
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);

        if (*sig_addr == cmp_val) {
            return *sig_addr;
        }

        // in case when peer pe enters next barrier
        if (*sig_addr == cmp_val + 1) {
            return *sig_addr;
        }
    } while (true);

    // never reach
    return -1;
}

// Atomicity of SHMEM_SIGNAL_SET not guaranteed
SHMEM_DEVICE void shmemix_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
{
    switch (sig_op) {
        case SHMEM_SIGNAL_SET:
            shmemi_signal_set(sig_addr, pe, signal);
            break;
        case SHMEM_SIGNAL_ADD:
            shmemi_signal_add(sig_addr, pe, signal);
            break;
    }
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until_eq(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) != cmp_val);

    return ret;
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until_ne(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) == cmp_val);

    return ret;
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until_gt(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) <= cmp_val);

    return ret;
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until_ge(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) < cmp_val);

    return ret;
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until_lt(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) >= cmp_val);

    return ret;
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until_le(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) > cmp_val);

    return ret;
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val)
{
    switch (cmp) {
        case SHMEM_CMP_EQ:
            return shmemi_signal_wait_until_eq(sig_addr, cmp_val);
        case SHMEM_CMP_NE:
            return shmemi_signal_wait_until_ne(sig_addr, cmp_val);
        case SHMEM_CMP_GT:
            return shmemi_signal_wait_until_gt(sig_addr, cmp_val);
        case SHMEM_CMP_GE:
            return shmemi_signal_wait_until_ge(sig_addr, cmp_val);
        case SHMEM_CMP_LT:
            return shmemi_signal_wait_until_lt(sig_addr, cmp_val);
        case SHMEM_CMP_LE:
            return shmemi_signal_wait_until_le(sig_addr, cmp_val);
    }
    return -1;
}

SHMEM_DEVICE int32_t shmemi_signal_fetch(__gm__ int32_t *sig_addr)
{
    dcci_cacheline((__gm__ uint8_t *)sig_addr);
    return *sig_addr;
}

template <typename T>
SHMEM_DEVICE void shmemi_wait_until(__gm__ T *sig_addr, int cmp, T cmp_val)
{
    switch (cmp) {
        case SHMEM_CMP_EQ:
            do {
                dcci_cacheline((__gm__ uint8_t *)sig_addr);
            } while (!(*sig_addr == cmp_val));
            break;
        case SHMEM_CMP_NE:
            do {
                dcci_cacheline((__gm__ uint8_t *)sig_addr);
            } while (!(*sig_addr != cmp_val));
            break;
        case SHMEM_CMP_GT:
            do {
                dcci_cacheline((__gm__ uint8_t *)sig_addr);
            } while (!(*sig_addr > cmp_val));
            break;
        case SHMEM_CMP_GE:
            do {
                dcci_cacheline((__gm__ uint8_t *)sig_addr);
            } while (!(*sig_addr >= cmp_val));
            break;
        case SHMEM_CMP_LT:
            do {
                dcci_cacheline((__gm__ uint8_t *)sig_addr);
            } while (!(*sig_addr < cmp_val));
            break;
        case SHMEM_CMP_LE:
            do {
                dcci_cacheline((__gm__ uint8_t *)sig_addr);
            } while (!(*sig_addr <= cmp_val));
            break;
    }
}

template <typename T>
SHMEM_DEVICE int shmemi_test(__gm__ T *sig_addr, int cmp, T cmp_val)
{
    dcci_cacheline((__gm__ uint8_t *)sig_addr);
    switch (cmp) {
        case SHMEM_CMP_EQ:
            return *sig_addr == cmp_val;
        case SHMEM_CMP_NE:
            return *sig_addr != cmp_val;
        case SHMEM_CMP_GT:
            return *sig_addr > cmp_val;
        case SHMEM_CMP_GE:
            return *sig_addr >= cmp_val;
        case SHMEM_CMP_LT:
            return *sig_addr < cmp_val;
        case SHMEM_CMP_LE:
            return *sig_addr <= cmp_val;
    }
    return 0;
}

template <typename T>
SHMEM_DEVICE void shmemi_wait_until_all(__gm__ T *sig_addr, size_t nelems, const int *status, int cmp, T cmp_val,
                                        int stride = SHMEMI_SYNCBIT_SIZE / sizeof(T))
{
    for (int i = 0; i < nelems; i++) {
        if (status && status[i] != 0) {
            continue;
        }

        shmemi_wait_until(sig_addr + i * stride, cmp, cmp_val);
    }
}

template <typename T>
SHMEM_DEVICE int shmemi_test_all(__gm__ T *sig_addr, size_t nelems, const int *status, int cmp, T cmp_val,
                                 int stride = SHMEMI_SYNCBIT_SIZE / sizeof(T))
{
    int ret = 1;

    for (int i = 0; i < nelems; i++) {
        if (status && status[i] != 0) {
            continue;
        }

        dcci_cacheline((__gm__ uint8_t *)(sig_addr + i * stride));
        switch (cmp) {
            case SHMEM_CMP_EQ:
                ret &= (*(sig_addr + i * stride) == cmp_val);
                break;
            case SHMEM_CMP_NE:
                ret &= (*(sig_addr + i * stride) != cmp_val);
                break;
            case SHMEM_CMP_GT:
                ret &= (*(sig_addr + i * stride) > cmp_val);
                break;
            case SHMEM_CMP_GE:
                ret &= (*(sig_addr + i * stride) >= cmp_val);
                break;
            case SHMEM_CMP_LT:
                ret &= (*(sig_addr + i * stride) < cmp_val);
                break;
            case SHMEM_CMP_LE:
                ret &= (*(sig_addr + i * stride) <= cmp_val);
                break;
        }
    }

    return ret;
}

#endif