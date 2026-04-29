/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEMI_HOST_DEF_H
#define SHMEMI_HOST_DEF_H

#include <sys/socket.h>
#include <netinet/in.h>

typedef enum {
    ADDR_IPv4,
    ADDR_IPv6
} shmem_addr_type_t;

typedef struct {
    union {
        struct sockaddr_in addr4;
        struct sockaddr_in6 addr6;
    } addr;
    shmem_addr_type_t type;
} shmem_sockaddr_t;

typedef struct {
    int32_t version;
    int32_t inner_sockFd;
    shmem_sockaddr_t addr;
    uint64_t magic;
} shmem_uniqueid_inner_t;

#endif // SHMEMI_HOST_DEF_H