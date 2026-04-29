/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_HOST_TEAM_H
#define SHMEM_HOST_TEAM_H

#include "host_device/shmem_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Collective Interface. Creates a new SHMEM team from an existing parent team.
 *
 * @param parent_team        [in] A team handle.
 * @param pe_start           [in] The first PE number of the subset of PEs from the parent team.
 * @param pe_stride          [in] The stride between team PE numbers in the parent team.
 * @param pe_size            [in] The total number of PEs in new team.
 * @param new_team           [out] A team handle.
 *
 * @return 0 on successful creation of new_team; otherwise nonzero.
 */
SHMEM_HOST_API int shmem_team_split_strided(shmem_team_t parent_team, int pe_start, int pe_stride, int pe_size,
                                            shmem_team_t *new_team);

/**
 * @brief Collective Interface. Split team from an existing parent team based on a 2D Cartsian Space.
 *
 * @param parent_team       [in] A team handle.
 * @param x_range           [in] represents the number of elements in the first dimensions
 * @param x_team            [in] A new x-axis team after team split.
 * @param y_team            [in] A new y-axis team after team split.
 *
 * @return 0 on successful creation of new_team; otherwise nonzero.
 */
SHMEM_HOST_API int shmem_team_split_2d(shmem_team_t parent_team, int x_range, shmem_team_t *x_team,
                                       shmem_team_t *y_team);

/**
 * @brief Translate a given PE number in one team into the corresponding PE number in another team.
 *
 * @param src_team           [in] A SHMEM team handle.
 * @param src_pe             [in] The PE number in src_team.
 * @param dest_team          [in] A SHMEM team handle.
 *
 * @return The number of PEs in the specified team.
 *         If the team handle is SHMEM_TEAM_INVALID, returns -1.
 */
SHMEM_HOST_API int shmem_team_translate_pe(shmem_team_t src_team, int src_pe, shmem_team_t dest_team);

/**
 * @brief Collective Interface. Destroys the team referenced by the team handle.
 *
 * @param team              [in] A team handle.
 */
SHMEM_HOST_API void shmem_team_destroy(shmem_team_t team);

/**
 * @brief Returns the PE number of the local PE
 *
 * @return Integer between 0 and npes - 1
 */
SHMEM_HOST_API int shmem_my_pe(void);

/**
 * @brief Returns the number of PEs running in the program.
 *
 * @return Number of PEs in the program.
 */
SHMEM_HOST_API int shmem_n_pes(void);

/**
 * @brief Returns the number of the calling PE in the specified team.
 *
 * @param team              [in] A team handle.
 *
 * @return The number of the calling PE within the specified team.
 *         If the team handle is SHMEM_TEAM_INVALID, returns -1.
 */
SHMEM_HOST_API int shmem_team_my_pe(shmem_team_t team);

/**
 * @brief Returns the number of PEs in the specified team.
 *
 * @param team              [in] A team handle.
 *
 * @return The number of PEs in the specified team.
 *         If the team handle is SHMEM_TEAM_INVALID, returns -1.
 */
SHMEM_HOST_API int shmem_team_n_pes(shmem_team_t team);

/**
 * @brief return team config which pass in as team created
 *
 * @param team [IN] team handle
 * @param config [OUT] the config associated with team, reserved for future use
 */
SHMEM_HOST_API int shmem_team_get_config(shmem_team_t team, shmem_team_config_t *config);

#ifdef __cplusplus
}
#endif

#endif