/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acl/acl.h"
#include "kernel_operator.h"

#include "shmem_api.h"

// kernels
SHMEM_GLOBAL void k_shmem_handle_wait(int32_t tid)
{
    shmemi_handle(tid);
}

// interfaces
void shmemi_handle_wait_on_stream(shmem_handle_t handle, aclrtStream stream)
{
    // call barrier kernel
    k_shmem_handle_wait<<<1, nullptr, stream>>>((int32_t)handle.team_id);
}