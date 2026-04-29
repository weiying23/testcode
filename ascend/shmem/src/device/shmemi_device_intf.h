/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_INTF_H
#define SHMEMI_DEVICE_INTF_H

#include "stdint.h"
#include "host_device/shmem_types.h"

// internal kernels
int32_t shmemi_memset(int32_t *array, int32_t len, int32_t val, int32_t count);

int32_t shmemi_barrier_on_stream(shmem_team_t tid, void *stream);

void shmemi_handle_wait_on_stream(shmem_handle_t handle, aclrtStream stream);

#endif