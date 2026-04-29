/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef KV_SHUFFLE_KERNEL_H
#define KV_SHUFFLE_KERNEL_H

void kv_shuffle(
    uint32_t block_dim, void* stream, uint64_t fftsAddr,
    uint8_t* k_cache,
    uint8_t* v_cache,
    uint8_t* global_shuffle_table,
    uint8_t* src_block_table,
    uint8_t* dst_block_table,
    uint8_t* sync_ptr,
    int64_t block_num,
    int64_t kv_head_num, int64_t page_size, int64_t head_dim, int32_t sync_count);

#endif