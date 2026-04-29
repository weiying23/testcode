/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
    WARNING：

    Our barrier implementation ensures that:
        On systems with only HCCS: All operations of all ranks of a team ON EXECUTING/INTERNAL STREAMs
        before the barrier are visiable to all ranks of the team after the barrier.

    Refer to shmem_device_sync.h for using restrictions.
*/

#ifndef SHMEM_HOST_SYNC_H
#define SHMEM_HOST_SYNC_H

#include "acl/acl.h"
#include "shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn SHMEM_HOST_API uint64_t shmemx_get_ffts_config()
 * @brief Get runtime ffts config. This config should be passed to MIX Kernel and set by MIX Kernel
 * using shmemx_set_ffts. Refer to shmemx_set_ffts for more details.
 *
 * @return ffts config
 */
SHMEM_HOST_API uint64_t shmemx_get_ffts_config();

/**
 * @brief The shmemx_barrier_on_stream is a collective synchronization routine over a team.
 * @param tid              [in] team to do barrier
 * @param stream           [in] copy used stream (use default stream if stream == NULL)
 */
SHMEM_HOST_API void shmemx_barrier_on_stream(shmem_team_t tid, aclrtStream stream);

/**
 * @brief The shmemx_barrier_all_on_stream routine is a mechanism for synchronizing all PEs at once.
 * @param stream           [in] copy used stream (use default stream if stream == NULL)
 */
SHMEM_HOST_API void shmemx_barrier_all_on_stream(aclrtStream stream);

/**
 * @fn SHMEM_HOST_API void shmem_handle_wait(shmem_handle_t handle)
 * @brief Wait asynchronous RMA operations to finish.
 */
SHMEM_HOST_API void shmem_handle_wait(shmem_handle_t handle, aclrtStream stream);

/**
 * @brief The shmemx_barrier_all_on_stream routine is a mechanism for synchronizing all PEs at once.
 * @param stream           [in] copy used stream (use default stream if stream == NULL)
 */
SHMEM_HOST_API void shmemx_barrier_all_on_stream(aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif