/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <iomanip>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "fp16_t.h"
#include "bfloat16.h"

#include <acl/acl.h>
#include "shmem_api.h"
#include "shmem_kernel.h"
#include "kv_shuffle_kernel.h"

using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;

namespace ShmemKernel {

extern void kv_shuffle(
    uint32_t block_dim, void* stream, uint64_t fftsAddr,
    uint8_t* k_cache,
    uint8_t* v_cache,
    uint8_t* global_shuffle_table,
    uint8_t* src_block_table,
    uint8_t* dst_block_table,
    uint8_t* sync_ptr,
    int64_t block_num,
    int64_t kv_head_num, int64_t page_size, int64_t head_dim, int32_t sync_count);

int shmem_kv_shuffle(uint32_t block_dim, aclrtStream stream, uint64_t fftsAddr, void* k_cache,
    void* v_cache,
    void* global_shuffle_table,
    void* src_block_table,
    void* dst_block_table,
    void* sync_ptr,
    int64_t block_nums,
    int64_t kv_head_num, int64_t page_size, int64_t head_dim, int32_t sync_count)
{
    int status = 0;
    // kv_shuffle
    
    kv_shuffle(block_dim, stream, fftsAddr,
        (uint8_t *)k_cache,
        (uint8_t *)v_cache,
        (uint8_t *)global_shuffle_table,
        (uint8_t *)src_block_table,
        (uint8_t *)dst_block_table,
        (uint8_t *)sync_ptr,
        block_nums,
        kv_head_num, page_size, head_dim, sync_count);
    return status;
}

} // namespace ShmemKernel