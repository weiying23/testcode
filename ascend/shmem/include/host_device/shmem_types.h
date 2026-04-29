/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_TYPES_H
#define SHMEM_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @private
*/
#define SHMEM_GLOBAL __global__ __aicore__

/// \def SHMEM_DEVICE
/// \brief A macro that identifies a function on the device side.
#define SHMEM_DEVICE __attribute__((always_inline)) __aicore__ __inline__

/**
 * @addtogroup group_enums
 * @{
*/

/**
* @brief The state of the SHMEM host OP type.
*/
enum shmemi_op_t {
    SHMEMI_OP_PUT = 0,
    SHMEMI_OP_P,
    SHMEMI_OP_PUT_SIGNAL,
    SHMEMI_OP_GET,
    SHMEMI_OP_G,
    // SHMEMI_OP_FENCE,
    // SHMEMI_OP_AMO,
    // SHMEMI_OP_QUIET,
    // SHMEMI_OP_SENTINEL = INT_MAX,
};

/**
 * @brief Team's index.
*/
enum shmem_team_index_t {
    SHMEM_TEAM_INVALID = -1,
    SHMEM_TEAM_WORLD = 0
};

/**
 * @brief Data op engine type.
*/
enum data_op_engine_type_t {
    SHMEM_DATA_OP_MTE = 0x01,
    SHMEM_DATA_OP_SDMA = 0x02,
    SHMEM_DATA_OP_ROCE = 0x04,
};

/**
 * @brief signal ops, used by signaler in p2p synchronization
 */
enum {
    SHMEM_SIGNAL_SET,
    SHMEM_SIGNAL_ADD
};

/**
 * @brief signal compare ops, used by signalee in p2p synchronization
 */
enum {
    SHMEM_CMP_EQ = 0,
    SHMEM_CMP_NE,
    SHMEM_CMP_GT,
    SHMEM_CMP_GE,
    SHMEM_CMP_LT,
    SHMEM_CMP_LE
};

/**
 * @brief Reserved for future use.
 */
typedef struct {
    int num_contexts;
} shmem_team_config_t;

/**@} */ // end of group_enums

/**
 * @defgroup group_typedef Typedef
 * @{

*/
/**
 * @brief A typedef of int
*/
typedef int shmem_team_t;

/**@} */ // end of group_typedef

/**
 * @addtogroup group_structs
 * @{
*/
/**
 * @struct shmem_handle_t
 * @brief Handle info used for non-blocking API synchronization.
 *
 * - shmem_team_t team_id: Team ID used for synchronization.
*/
struct shmem_handle_t {
    shmem_team_t team_id;
};

/**@} */ // end of group_structs
#ifdef __cplusplus
}
#endif

#endif /* SHMEM_TYPES_H */