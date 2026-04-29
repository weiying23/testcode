/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#ifndef FUZZ_FUNC_TYPE_H
#define FUZZ_FUNC_TYPE_H

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

#endif  // FUZZ_FUNC_TYPE_H
