/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHEMEI_QUIET_H
#define SHEMEI_QUIET_H

#include "internal/device/shmemi_device_common.h"

SHMEM_DEVICE void shmemi_quiet()
{
    // clear instruction pipes
    AscendC::PipeBarrier<PIPE_ALL>();

    // flush data cache to GM
    dcci_entire_cache();
}

#endif