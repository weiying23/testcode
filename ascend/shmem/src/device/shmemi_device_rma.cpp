/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acl/acl.h"
#include "kernel_operator.h"
#include "shmemi_device_rma.h"

using namespace std;

// kernels
SHMEM_GLOBAL void shmemi_putmem_nbi(GM_ADDR lptr, GM_ADDR rptr, uint32_t elem_size, int32_t pe)
{
    shmem_put_uint8_mem_nbi(lptr, rptr, elem_size, pe);
}

SHMEM_GLOBAL void shmemi_putmem(GM_ADDR lptr, GM_ADDR rptr, uint32_t elem_size, int32_t pe)
{
    shmem_put_uint8_mem(lptr, rptr, elem_size, pe);
}

SHMEM_GLOBAL void shmemi_getmem_nbi(GM_ADDR lptr, GM_ADDR rptr, uint32_t elem_size, int32_t pe)
{
    shmem_get_uint8_mem_nbi(lptr, rptr, elem_size, pe);
}

SHMEM_GLOBAL void shmemi_getmem(GM_ADDR lptr, GM_ADDR rptr, uint32_t elem_size, int32_t pe)
{
    shmem_get_uint8_mem(lptr, rptr, elem_size, pe);
}

SHMEM_GLOBAL void shmemi_putmem_signal(GM_ADDR lptr, GM_ADDR rptr, uint32_t elem_size, GM_ADDR sig_addr, int32_t signal,
                                       int sig_op, int pe)
{
    __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);
    shmem_put_uint8_mem_signal(lptr, rptr, elem_size, sig_addr_int32, signal, sig_op, pe);
}

SHMEM_GLOBAL void shmemi_putmem_signal_nbi(GM_ADDR lptr, GM_ADDR rptr, uint32_t elem_size, GM_ADDR sig_addr,
                                           int32_t signal, int sig_op, int pe)
{
    __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);
    shmem_put_uint8_mem_signal_nbi(lptr, rptr, elem_size, sig_addr_int32, signal, sig_op, pe);
}

SHMEM_GLOBAL void k_shmem_getmem(GM_ADDR dst, GM_ADDR src, size_t elem_size, int32_t pe)
{
    shmem_getmem(dst, src, elem_size, pe);
}

// kernel function calling entrance
int32_t shmemi_prepare_and_post_rma(const char *api_name, shmemi_op_t desc, bool is_nbi, uint8_t *lptr, uint8_t *rptr,
                                    size_t n_elems, size_t elem_bytes, int pe, uint8_t *sig_addr, int32_t signal,
                                    int sig_op, ptrdiff_t lstride, ptrdiff_t rstride, aclrtStream acl_strm,
                                    size_t block_size)
{
    if ((lstride > 1) || (rstride > 1)) {
        return -1;
    }

    if (is_nbi) {
        switch (desc) {
            case SHMEMI_OP_PUT:
                shmemi_putmem_nbi<<<block_size, 0, acl_strm>>>(lptr, rptr, n_elems * elem_bytes, pe);
                break;
            case SHMEMI_OP_GET:
                shmemi_getmem_nbi<<<block_size, 0, acl_strm>>>(lptr, rptr, n_elems * elem_bytes, pe);
                break;
            case SHMEMI_OP_PUT_SIGNAL:
                shmemi_putmem_signal_nbi<<<block_size, 0, acl_strm>>>(lptr, rptr, n_elems * elem_bytes, sig_addr,
                                                                      signal, sig_op, pe);
            default:
                break;
        }
    } else {
        switch (desc) {
            case SHMEMI_OP_PUT:
                shmemi_putmem<<<block_size, 0, acl_strm>>>(lptr, rptr, n_elems * elem_bytes, pe);
                break;
            case SHMEMI_OP_GET:
                shmemi_getmem<<<block_size, 0, acl_strm>>>(lptr, rptr, n_elems * elem_bytes, pe);
                break;
            case SHMEMI_OP_PUT_SIGNAL:
                shmemi_putmem_signal<<<block_size, 0, acl_strm>>>(lptr, rptr, n_elems * elem_bytes, sig_addr, signal,
                                                                  sig_op, pe);
            default:
                break;
        }
    }
    return 0;
}

int32_t shmemi_getmem_on_stream(uint8_t *dst, uint8_t *src, size_t elem_size, int32_t pe, aclrtStream stream)
{
    k_shmem_getmem<<<1, nullptr, stream>>>(dst, src, elem_size, pe);
    return aclrtSynchronizeStream(stream);
}

#define SHMEMI_TYPENAME_P(NAME, TYPE)                                                \
    SHMEM_GLOBAL void shmemi_##NAME##_p(GM_ADDR dest_addr, const TYPE value, int pe) \
    {                                                                                \
        __gm__ TYPE *dst = (__gm__ TYPE *)dest_addr;                                 \
        shmem_##NAME##_p(dst, value, pe);                                            \
    }

SHMEM_TYPE_FUNC(SHMEMI_TYPENAME_P)

#undef SHMEMI_TYPENAME_P

// shmem_g
#define SHMEMI_TYPENAME_G(NAME, TYPE)                                            \
    SHMEM_GLOBAL void shmemi_##NAME##_g(GM_ADDR src, int pe, GM_ADDR value_addr) \
    {                                                                            \
        __gm__ TYPE *src_addr = (__gm__ TYPE *)src;                              \
        __gm__ TYPE *dst_addr = (__gm__ TYPE *)value_addr;                       \
        *dst_addr = shmem_##NAME##_g(src_addr, pe);                              \
    }

SHMEM_TYPE_FUNC(SHMEMI_TYPENAME_G)
#undef SHMEMI_TYPENAME_G

#define SHMEMI_TYPENAME_PREPARE_RMA_P(NAME, TYPE)                                                           \
    void shmemi_prepare_and_post_rma_##NAME##_p(const char *api_name, uint8_t *dst_ptr, TYPE value, int pe, \
                                                aclrtStream acl_strm, size_t block_size)                    \
    {                                                                                                       \
        shmemi_##NAME##_p<<<block_size, 0, acl_strm>>>(dst_ptr, value, pe);                                 \
    }

SHMEM_TYPE_FUNC(SHMEMI_TYPENAME_PREPARE_RMA_P)
#undef SHMEMI_TYPENAME_PREPARE_RMA_P
