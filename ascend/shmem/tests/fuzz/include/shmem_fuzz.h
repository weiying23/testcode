/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DT_FUZZ_H
#define DT_FUZZ_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <type_traits>

#include "acl/acl_base.h"
#include "host/shmem_host_def.h"
#include "secodefuzz/secodeFuzz.h"

#undef inline

#define MOCKER_CPP(api, TT) MOCKCPP_NS::mockAPI(#api, reinterpret_cast<TT>(api))

// dt-fuzz 要求3小时或者3000万次
constexpr int SHMEM_FUZZ_COUNT = 1;                     // 30_000_000;
constexpr int SHMEM_FUZZ_RUNNING_SECONDS = 3 * 60 * 60; // 3小时，单位秒

// fuzz变量ID (每个fuzz用例中都必须从0开始按顺序使用)
constexpr int FUZZ_VALUE_0_ID = 0;
constexpr int FUZZ_VALUE_1_ID = 1;
constexpr int FUZZ_VALUE_2_ID = 2;
constexpr int FUZZ_VALUE_3_ID = 3;
constexpr int FUZZ_VALUE_4_ID = 4;
constexpr int FUZZ_VALUE_5_ID = 5;
constexpr int FUZZ_VALUE_6_ID = 6;
constexpr int FUZZ_VALUE_7_ID = 7;
constexpr int FUZZ_VALUE_8_ID = 8;
constexpr int FUZZ_VALUE_9_ID = 9;

// fuzz数字类型约束
template <class T>
inline constexpr bool IS_FUZZ_NUMBER_TYPE = std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                                            std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
                                            std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
                                            std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
                                            std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

// fuzz获取数字类型变异值
template <class T, std::enable_if_t<IS_FUZZ_NUMBER_TYPE<T>, bool> = true>
inline T fuzz_get_number(int id, T init_value)
{
    if constexpr (std::is_same_v<T, uint8_t>) {
        return *(T *)DT_SetGetU8V3(id, init_value);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return *(T *)DT_SetGetU16V3(id, init_value);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return *(T *)DT_SetGetU32V3(id, init_value);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return *(T *)DT_SetGetU64V3(id, init_value);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return *(T *)DT_SetGetS8V3(id, init_value);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return *(T *)DT_SetGetS16V3(id, init_value);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return *(T *)DT_SetGetS32V3(id, init_value);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return *(T *)DT_SetGetS64V3(id, init_value);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return *(T *)DT_SetGetFloat32(id, &init_value);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return *(T *)DT_SetGetFloat64(id, &init_value);
    }
}

// fuzz获取数字类型范围变异值 (secodefuzz只支持i32类型的范围变异)
inline int32_t fuzz_get_ranged_number(int id, int32_t init_value, int32_t min, int32_t max)
{
    return *(int32_t *)DT_SetGetNumberRangeV3(id, init_value, min, max);
}

constexpr uint64_t KiB = 1024;
constexpr uint64_t MiB = 1024 * KiB;
constexpr uint64_t GiB = 1024 * MiB;

// 获取全局指定的 gnpu 数量
int shmem_fuzz_gnpu_num(void);
// 获取 rank_id 对应的 device_id
int32_t shmem_fuzz_device_id(int rank_id);
// 使用参数进行默认逻辑的shmem初始化
void shmem_fuzz_test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *stream);
// 根据参数构造默认shmem初始化attrs
void shmem_fuzz_test_set_attr(int rank_id, int n_ranks, uint64_t local_mem_size, shmem_init_attr_t **attributes);
// 使用指定attrs初始化shmem
void shmem_fuzz_test_init_attr(shmem_init_attr_t *attributes, aclrtStream *stream);
// shmem反初始化
void shmem_fuzz_test_deinit(aclrtStream stream, int device_id);

// 在用例中执行多进程并行逻辑
void shmem_fuzz_multi_task(std::function<void(int, int, uint64_t)> task, uint64_t local_mem_size, int process_count);

// shmem初始化RAII包装类，适合在 shmem_fuzz_multi_task 中使用
class shmem_init_scope {
public:
    shmem_init_scope(int rank_id, int n_ranks, uint64_t local_mem_size) : rank_id(rank_id)
    {
        std::lock_guard lock(mutex_);
        if (count_ == 0) {
            shmem_fuzz_test_init(rank_id, n_ranks, local_mem_size, &stream);
        } else {
            // shmem_init_scope should not overlap
            std::cerr << "WARN: shmem_init_scope overlapped!" << std::endl;
        }
        count_++;
    }

    ~shmem_init_scope()
    {
        std::lock_guard lock(mutex_);
        count_--;
        if (count_ == 0) {
            shmem_fuzz_test_deinit(stream, shmem_fuzz_device_id(rank_id));
        }
    }

    shmem_init_scope(const shmem_init_scope &) = delete;
    shmem_init_scope(shmem_init_scope &&) = delete;
    shmem_init_scope &operator=(const shmem_init_scope &) = delete;
    shmem_init_scope &operator=(shmem_init_scope &&) = delete;

    int rank_id;
    aclrtStream stream = nullptr;

private:
    std::mutex mutex_;
    static inline size_t count_ = 0;
};

#endif // DT_FUZZ_H
