/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "shmemi_device_intf.h"

extern "C" int rtGetC2cCtrlAddr(uint64_t *config, uint32_t *len);

namespace shm {
static uint64_t ffts_config;

int32_t shmemi_sync_init()
{
    uint32_t len;
    return rtGetC2cCtrlAddr(&ffts_config, &len);
}

} // namespace

uint64_t shmemx_get_ffts_config()
{
    return shm::ffts_config;
}

void shmem_barrier(shmem_team_t tid)
{
    // using default stream to do barrier
    shmemi_barrier_on_stream(tid, nullptr);
}

void shmem_barrier_all()
{
    shmem_barrier(SHMEM_TEAM_WORLD);
}

void shmemx_barrier_on_stream(shmem_team_t tid, aclrtStream stream)
{
    shmemi_barrier_on_stream(tid, stream);
}

void shmemx_barrier_all_on_stream(aclrtStream stream)
{
    shmemi_barrier_on_stream(SHMEM_TEAM_WORLD, stream);
}

void shmem_handle_wait(shmem_handle_t handle, aclrtStream stream)
{
    shmemi_handle_wait_on_stream(handle, stream);
}
