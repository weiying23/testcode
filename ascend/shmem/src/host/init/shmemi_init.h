/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_INIT_H
#define SHMEMI_INIT_H

#include "stdint.h"
#include "internal/host_device/shmemi_types.h"

namespace shm {
extern shmemi_device_host_state_t g_state;
extern shmemi_host_state_t g_state_host;

int32_t update_device_state(void);

int32_t shmemi_control_barrier_all();

}  // namespace shm

#endif  // SHMEMI_INIT_H
