/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_RMA_H
#define SHMEMI_DEVICE_RMA_H

#include <cstdint>
#include <cstddef>
#include <acl/acl.h>
#include "shmem_api.h"
#include "host_device/shmem_types.h"

#define SHMEMI_TYPENAME_PREPARE_RMA_P(NAME, TYPE)                                                           \
    void shmemi_prepare_and_post_rma_##NAME##_p(const char *api_name, uint8_t *dst_ptr, TYPE value, int pe, \
                                                aclrtStream acl_strm, size_t block_size);

SHMEM_TYPE_FUNC(SHMEMI_TYPENAME_PREPARE_RMA_P)
#undef SHMEMI_TYPENAME_PREPARE_RMA_P

// internal kernels calling
int32_t shmemi_prepare_and_post_rma(const char *api_name, shmemi_op_t desc, bool is_nbi, uint8_t *lptr, uint8_t *rptr,
                                    size_t n_elems, size_t elem_bytes, int pe, uint8_t *sig_addr, int32_t signal,
                                    int sig_op, ptrdiff_t lstride = 1, ptrdiff_t rstride = 1,
                                    aclrtStream acl_strm = nullptr, size_t block_size = 1);

int32_t shmemi_getmem_on_stream(uint8_t *dst, uint8_t *src, size_t elem_size, int32_t pe, aclrtStream stream);

#endif