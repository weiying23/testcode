/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEM_FABRIC_HYBRID_HYBRID_BIG_MEM_DL_H
#define MEM_FABRIC_HYBRID_HYBRID_BIG_MEM_DL_H

#include <stdint.h>
#include <stdbool.h>

#ifndef __cplusplus
extern "C" {
#endif

typedef void *hybm_entity_t;
typedef void *hybm_mem_slice_t;

#define HYBM_FREE_SINGLE_SLICE 0x00
#define HYBM_FREE_ALL_SLICE 0x01

#define HYBM_EXPORT_PARTIAL_SLICE 0x00
#define HYBM_EXPORT_ALL_SLICE 0x01


typedef enum {
    HYBM_TYPE_AI_CORE_INITIATE = 0,
    HYBM_TYPE_BUTT
} hybm_type;

typedef enum {
    HYBM_DOP_TYPE_DEFAULT = 0U,
    HYBM_DOP_TYPE_MTE = 1U << 0,
    HYBM_DOP_TYPE_DEVICE_RDMA = 1U << 1,

    HYBM_DOP_TYPE_BUTT
} hybm_data_op_type;

typedef enum {
    HYBM_SCOPE_IN_NODE = 0,
    HYBM_SCOPE_CROSS_NODE,

    HYBM_SCOPE_BUTT
} hybm_scope;

typedef enum {
    HYBM_MEM_TYPE_DEVICE = 0,

    HYBM_MEM_TYPE_BUTT
} hybm_mem_type;

typedef enum {
    HYBM_ROLE_PEER = 0,
    HYBM_ROLE_SENDER,
    HYBM_ROLE_RECEIVER,
    HYBM_ROLE_BUTT
} hybm_role_type;

typedef struct {
    uint8_t desc[512L];
    uint32_t descLen;
} hybm_exchange_info;

typedef struct {
    hybm_type bmType;
    hybm_mem_type memType;
    hybm_data_op_type bmDataOpType;
    hybm_scope bmScope;
    uint16_t rankCount;
    uint16_t rankId;
    uint16_t devId;
    uint64_t singleRankVASpace;
    uint64_t preferredGVA;
    bool globalUniqueAddress; // 是否使用全局统一内存地址
    hybm_role_type role;
    char nic[64];
} hybm_options;

#ifndef __cplusplus
}
#endif

#endif // MEM_FABRIC_HYBRID_HYBRID_BIG_MEM_DL_H
