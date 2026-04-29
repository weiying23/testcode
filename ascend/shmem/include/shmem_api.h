/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_API_H
#define SHMEM_API_H

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#include "device/shmem_device_def.h"
#include "device/shmem_device_rma.h"
#include "device/shmemx_device_rma.h"
#include "device/shmem_device_sync.h"
#include "device/shmem_device_team.h"
#include "device/shmem_device_atomic.h"
#endif

#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_sync.h"
#include "host/shmem_host_team.h"

#endif // SHMEM_API_H
