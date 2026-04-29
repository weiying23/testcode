/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "kernel_operator.h"
#include "shmem_api.h"
#include "func_type.h"

#define KERNEL_P(NAME, TYPE)                                           \
    class kernel_##NAME##_p {                                          \
    public:                                                            \
        __aicore__ inline kernel_##NAME##_p()                          \
        {                                                              \
        }                                                              \
        __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)          \
        {                                                              \
            gva_gm = (__gm__ TYPE *)gva;                               \
            dev_gm = (__gm__ TYPE *)dev;                               \
                                                                       \
            rank = shmem_my_pe();                                      \
            rank_size = shmem_n_pes();                                 \
        }                                                              \
        __aicore__ inline void Process(uint64_t config)                \
        {                                                              \
            shmemx_set_ffts_config(config);                            \
            shmem_##NAME##_p(gva_gm, *dev_gm, (rank + 1) % rank_size); \
            shmemx_barrier_all_vec();                                  \
        }                                                              \
                                                                       \
    private:                                                           \
        __gm__ TYPE *gva_gm;                                           \
        __gm__ TYPE *dev_gm;                                           \
                                                                       \
        int64_t rank;                                                  \
        int64_t rank_size;                                             \
    }

SHMEM_FUNC_TYPE_KERNEL(KERNEL_P);

#define P_NUM_TEST(NAME, TYPE)                                                                           \
    extern "C" __global__ __aicore__ void p_##NAME##_num_test(GM_ADDR gva, GM_ADDR dev, uint64_t config) \
    {                                                                                                    \
        kernel_##NAME##_p op;                                                                            \
        op.Init(gva, dev);                                                                               \
        op.Process(config);                                                                              \
    }

SHMEM_FUNC_TYPE_KERNEL(P_NUM_TEST);

#define PUT_ONE_NUM_DO(NAME, TYPE)                                                                              \
    void put_##NAME##_one_num_do(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, uint8_t *dev) \
    {                                                                                                           \
        p_##NAME##_num_test<<<block_dim, nullptr, stream>>>(gva, dev, config);                                  \
    }

SHMEM_FUNC_TYPE_KERNEL(PUT_ONE_NUM_DO);

#define KERNEL_G(NAME, TYPE)                                             \
    class kernel_##NAME##_g {                                            \
    public:                                                              \
        __aicore__ inline kernel_##NAME##_g()                            \
        {                                                                \
        }                                                                \
        __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)            \
        {                                                                \
            gva_gm = (__gm__ TYPE *)gva;                                 \
            dev_gm = (__gm__ TYPE *)dev;                                 \
                                                                         \
            rank = shmem_my_pe();                                        \
            rank_size = shmem_n_pes();                                   \
        }                                                                \
        __aicore__ inline void Process(uint64_t config)                  \
        {                                                                \
            shmemx_set_ffts_config(config);                              \
            TYPE val = shmem_##NAME##_g(gva_gm, (rank + 1) % rank_size); \
            *dev_gm = val;                                               \
            shmemx_barrier_all_vec();                                    \
        }                                                                \
                                                                         \
    private:                                                             \
        __gm__ TYPE *gva_gm;                                             \
        __gm__ TYPE *dev_gm;                                             \
                                                                         \
        int64_t rank;                                                    \
        int64_t rank_size;                                               \
    }

SHMEM_FUNC_TYPE_KERNEL(KERNEL_G);

#define G_NUM_TEST(NAME, TYPE)                                                                           \
    extern "C" __global__ __aicore__ void g_##NAME##_num_test(GM_ADDR gva, GM_ADDR dev, uint64_t config) \
    {                                                                                                    \
        kernel_##NAME##_g op;                                                                            \
        op.Init(gva, dev);                                                                               \
        op.Process(config);                                                                              \
    }

SHMEM_FUNC_TYPE_KERNEL(G_NUM_TEST);

#define GET_ONE_NUM_DO(NAME, TYPE)                                                                              \
    void get_##NAME##_one_num_do(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, uint8_t *dev) \
    {                                                                                                           \
        g_##NAME##_num_test<<<block_dim, nullptr, stream>>>(gva, dev, config);                                  \
    }

SHMEM_FUNC_TYPE_KERNEL(GET_ONE_NUM_DO);
