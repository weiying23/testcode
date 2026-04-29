/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UT_FUNC_TYPE_H
#define UT_FUNC_TYPE_H

#define SHMEM_FUNC_TYPE_HOST(FUNC)   \
    FUNC(half, op::fp16_t);          \
    FUNC(float, float);              \
    FUNC(double, double);            \
    FUNC(int8, int8_t);              \
    FUNC(int16, int16_t);            \
    FUNC(int32, int32_t);            \
    FUNC(int64, int64_t);            \
    FUNC(uint8, uint8_t);            \
    FUNC(uint16, uint16_t);          \
    FUNC(uint32, uint32_t);          \
    FUNC(uint64, uint64_t);          \
    FUNC(char, char);                \
    FUNC(bfloat16, op::bfloat16)

#define SHMEM_FUNC_TYPE_KERNEL(FUNC) \
    FUNC(half, half);                \
    FUNC(float, float);              \
    FUNC(double, double);            \
    FUNC(int8, int8_t);              \
    FUNC(int16, int16_t);            \
    FUNC(int32, int32_t);            \
    FUNC(int64, int64_t);            \
    FUNC(uint8, uint8_t);            \
    FUNC(uint16, uint16_t);          \
    FUNC(uint32, uint32_t);          \
    FUNC(uint64, uint64_t);          \
    FUNC(char, char);                \
    FUNC(bfloat16, bfloat16_t)

#define SHMEM_MEM_PUT_GET_FUNC(FUNC) \
    FUNC(float, float);       \
    FUNC(double, double);     \
    FUNC(int8, int8_t);       \
    FUNC(int16, int16_t);     \
    FUNC(int32, int32_t);     \
    FUNC(int64, int64_t);     \
    FUNC(uint8, uint8_t);     \
    FUNC(uint16, uint16_t);   \
    FUNC(uint32, uint32_t);   \
    FUNC(uint64, uint64_t);   \
    FUNC(char, char)

#define SHMEM_ATOMIC_ADD_FUNC_TYPE_HOST(FUNC) \
    FUNC(half, op::fp16_t);                   \
    FUNC(float, float);                       \
    FUNC(int8, int8_t);                       \
    FUNC(int16, int16_t);                     \
    FUNC(int32, int32_t)

#define SHMEM_ATOMIC_ADD_FUNC_TYPE_KERNEL(FUNC) \
    FUNC(half, half);                           \
    FUNC(float, float);                         \
    FUNC(int8, int8_t);                         \
    FUNC(int16, int16_t);                       \
    FUNC(int32, int32_t)

#endif  // UT_FUNC_TYPE_H