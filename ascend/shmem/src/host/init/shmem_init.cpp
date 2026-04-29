/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <stdint.h>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <random>
#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "internal/host/shmemi_host_def.h"

using namespace std;

namespace shm {
constexpr uint64_t MIN_PORT = 1024;
constexpr uint64_t MAX_PORT = 65536;
constexpr uint64_t MAX_ATTEMPTS = 1000;
constexpr uint64_t MAX_IFCONFIG_LENGTH = 23;
constexpr uint64_t MAX_IP = 48;
constexpr int DEFAULT_MY_PE = -1;
constexpr int DEFAULT_N_PES = -1;

constexpr int DEFAULT_FLAG = 0;
constexpr int DEFAULT_ID = 0;
constexpr int DEFAULT_TIMEOUT = 120;
constexpr int DEFAULT_TEVENT = 0;
constexpr int DEFAULT_BLOCK_NUM = 1;
constexpr int DEFAULT_IFNAME_LNEGTH = 4;

// initializer
#define SHMEM_DEVICE_HOST_STATE_INITIALIZER                                            \
    {                                                                                 \
        (1 << 16) + sizeof(shmemi_device_host_state_t),  /* version */                     \
            (DEFAULT_MY_PE),                           /* mype */                       \
            (DEFAULT_N_PES),                           /* npes */                       \
            NULL,                                    /* heap_base */                   \
            NULL,                                  /* p2p_heap_host_base */           \
            NULL,                                  /* sdma_heap_host_base */          \
            NULL,                                  /* roce_heap_host_base */          \
            NULL,                                  /* p2p_heap_device_base */        \
            NULL,                                  /* sdma_heap_device_base */       \
            NULL,                                  /* roce_heap_device_base */       \
            {},                                     /* topo_list */                     \
            SIZE_MAX,                                /* heap_size */                   \
            {NULL},                                  /* team_pools */                  \
            0,                                       /* sync_pool */                  \
            0,                                       /* sync_counter */                \
            0,                                      /* core_sync_pool */             \
            0,                                      /* core_sync_counter */          \
            0,                                        /* partial_barrier_pool */      \
            false,                                   /* shmem_is_shmem_initialized */ \
            false,                                   /* shmem_is_shmem_created */     \
            {0, 16 * 1024, 0},                       /* shmem_mte_config */           \
    }

shmemi_device_host_state_t g_state = SHMEM_DEVICE_HOST_STATE_INITIALIZER;
shmemi_host_state_t g_state_host = {nullptr, DEFAULT_TEVENT, DEFAULT_BLOCK_NUM};
shmem_init_attr_t g_attr;
static smem_shm_t g_smem_handle = nullptr;
static bool g_attr_init = false;
static char g_ipport[SHMEM_MAX_IP_PORT_LEN] = {0};

int32_t version_compatible()
{
    int32_t status = SHMEM_SUCCESS;
    return status;
}

int32_t bind_tcp_port_v4(int &sockfd, int port, shmem_uniqueid_inner_t *innerUId, char *ip_str)
{
    sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd != -1) {
        int on_v4 = 1;
        if (::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on_v4, sizeof(on_v4)) == 0) {
            innerUId->addr.addr.addr4.sin_port = htons(port);
            sockaddr *cur_addr = reinterpret_cast<sockaddr *>(&innerUId->addr.addr.addr4);
            if (::bind(sockfd, cur_addr, sizeof(innerUId->addr.addr.addr4)) == 0) {
                SHM_LOG_INFO("bind ipv4 success " << ", fd:" << sockfd << ", " << ip_str << ":" << port);
                return 0;
            } else {
                SHM_LOG_ERROR("bind socket fail:" << errno << "," << ip_str << ":" << port);
            }
        } else {
            SHM_LOG_ERROR("set socket opt fail:" << errno << ","  << ip_str << ":" << port);
        }
        close(sockfd);
        sockfd = -1;
    } else {
        SHM_LOG_ERROR("create socket fail:" << errno << ", " << ip_str << ":" << port);
    }
    return -1;
}

int32_t bind_tcp_port_v6(int &sockfd, int port, shmem_uniqueid_inner_t *innerUId, char *ip_str)
{
    sockfd = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (sockfd != -1) {
        int on_v6 = 1;
        if (::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on_v6, sizeof(on_v6)) == 0) {
            innerUId->addr.addr.addr6.sin6_port = htons(port);
            sockaddr *cur_addr = reinterpret_cast<sockaddr *>(&innerUId->addr.addr.addr6);
            if (::bind(sockfd, cur_addr, sizeof(innerUId->addr.addr.addr6)) == 0) {
                SHM_LOG_INFO("bind ipv6 success " << ", fd:" << sockfd << ", " << ip_str << ":" << port);
                return 0;
            } else {
                SHM_LOG_ERROR("bind socket6 fail:" << errno << "," << ip_str << ":" << port);
            }
        } else {
            SHM_LOG_ERROR("set socket6 opt fail:" << errno << "," << ip_str << ":" << port);
        }
        close(sockfd);
        sockfd = -1;
    } else {
        SHM_LOG_ERROR("create socket6 fail:" << errno << "," << ip_str << ":" << port);
    }
    return -1;
}

int32_t shmemi_options_init()
{
    int32_t status = SHMEM_SUCCESS;
    return status;
}

int32_t shmemi_state_init_attr(shmem_init_attr_t *attributes)
{
    int32_t status = SHMEM_SUCCESS;
    g_state.mype = attributes->my_rank;
    g_state.npes = attributes->n_ranks;
    g_state.heap_size = attributes->local_mem_size + SHMEM_EXTRA_SIZE;

    aclrtStream stream = nullptr;
    SHMEM_CHECK_RET(aclrtCreateStream(&stream), aclrtCreateStream);
    g_state_host.default_stream = stream;
    g_state_host.default_event_id = DEFAULT_TEVENT;
    g_state_host.default_block_num = DEFAULT_BLOCK_NUM;
    return status;
}

void shmemi_reach_info_init(void *&gva)
{
    uint32_t reach_info = 0;
    int32_t status = SHMEM_SUCCESS;
    for (int32_t i = 0; i < g_state.npes; i++) {
        status = smem_shm_topology_can_reach(g_smem_handle, i, &reach_info);
        if (status != SHMEM_SUCCESS) {
            SHM_LOG_ERROR("smem_shm_topology_can_reach failed");
        }
        g_state.p2p_heap_host_base[i] = (void *)((uintptr_t)gva + g_state.heap_size * static_cast<uint32_t>(i));
        if (reach_info & SMEMS_DATA_OP_MTE) {
            g_state.topo_list[i] |= SHMEM_TRANSPORT_MTE;
        }
        if (reach_info & SMEMS_DATA_OP_SDMA) {
            g_state.sdma_heap_host_base[i] = (void *)((uintptr_t)gva + g_state.heap_size * static_cast<uint32_t>(i));
        } else {
            g_state.sdma_heap_host_base[i] = NULL;
        }
        if (reach_info & SMEMS_DATA_OP_RDMA) {
            g_state.topo_list[i] |= SHMEM_TRANSPORT_ROCE;
        }
    }
}

int32_t shmemi_heap_init(shmem_init_attr_t *attributes)
{
    void *gva = nullptr;
    int32_t status = SHMEM_SUCCESS;
    int32_t device_id;
    SHMEM_CHECK_RET(aclrtGetDevice(&device_id), aclrtGetDevice);

    status = smem_init(DEFAULT_FLAG);
    if (status != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("smem_init Failed");
        return SHMEM_SMEM_ERROR;
    }
    smem_shm_config_t config;
    status = smem_shm_config_init(&config);
    if (status != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("smem_shm_config_init Failed");
        return SHMEM_SMEM_ERROR;
    }
    // set config.sockFd value
    config.sockFd = attributes->option_attr.sockFd;
    status = smem_shm_init(attributes->ip_port, attributes->n_ranks, attributes->my_rank, device_id, &config);
    if (status != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("smem_shm_init Failed");
        return SHMEM_SMEM_ERROR;
    }

    config.shmInitTimeout = attributes->option_attr.shm_init_timeout;
    config.shmCreateTimeout = attributes->option_attr.shm_create_timeout;
    config.controlOperationTimeout = attributes->option_attr.control_operation_timeout;

    g_smem_handle = smem_shm_create(DEFAULT_ID, attributes->n_ranks, attributes->my_rank, g_state.heap_size,
                                    static_cast<smem_shm_data_op_type>(attributes->option_attr.data_op_engine_type),
                                    DEFAULT_FLAG, &gva);
    if (g_smem_handle == nullptr || gva == nullptr) {
        SHM_LOG_ERROR("smem_shm_create Failed");
        return SHMEM_SMEM_ERROR;
    }
    SHMEM_CHECK_RET(
        aclrtMallocHost(((void **)&g_state.p2p_heap_host_base), g_state.npes * sizeof(void *)));
    SHMEM_CHECK_RET(
        aclrtMallocHost(((void **)&g_state.sdma_heap_host_base), g_state.npes * sizeof(void *)));
    SHMEM_CHECK_RET(
        aclrtMallocHost(((void **)&g_state.roce_heap_host_base), g_state.npes * sizeof(void *)));

    SHMEM_CHECK_RET(aclrtMalloc(((void **)&g_state.p2p_heap_device_base), g_state.npes * sizeof(void *),
        ACL_MEM_MALLOC_HUGE_FIRST));
    SHMEM_CHECK_RET(aclrtMalloc(((void **)&g_state.sdma_heap_device_base), g_state.npes * sizeof(void *),
        ACL_MEM_MALLOC_HUGE_FIRST));
    SHMEM_CHECK_RET(aclrtMalloc(((void **)&g_state.roce_heap_device_base), g_state.npes * sizeof(void *),
        ACL_MEM_MALLOC_HUGE_FIRST));

    auto alignedSize = ALIGN_UP(g_state.heap_size, SHMEM_HEAP_SEGMENT_SIZE);
    g_state.heap_base = (void *)((uintptr_t)gva + alignedSize * static_cast<uint32_t>(attributes->my_rank));
    g_state.heap_size = alignedSize;

    shmemi_reach_info_init(gva);
    if (shm::g_ipport[0] != '\0') {
        g_ipport[0] = '\0';
        bzero(attributes->ip_port, sizeof(attributes->ip_port));
    } else {
        SHM_LOG_WARN("my_rank:" << attributes->my_rank << " shm::g_ipport is released in advance!");
        bzero(attributes->ip_port, sizeof(attributes->ip_port));
    }
    g_state.is_shmem_created = true;
    return status;
}

int32_t shmemi_control_barrier_all()
{
    SHM_ASSERT_RETURN(g_smem_handle != nullptr, SHMEM_INVALID_PARAM);
    auto ret = smem_shm_control_barrier(g_smem_handle);
    if (ret != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("Barrier failed");
        return ret;
    }
    return SHMEM_SUCCESS;
}

int32_t update_device_state()
{
    if (!g_state.is_shmem_created) {
        return SHMEM_NOT_INITED;
    }

    SHMEM_CHECK_RET(aclrtMemcpy(g_state.p2p_heap_device_base, g_state.npes * sizeof(void *),
        g_state.p2p_heap_host_base, g_state.npes * sizeof(void *), ACL_MEMCPY_HOST_TO_DEVICE), aclrtMemcpy);
    SHMEM_CHECK_RET(aclrtMemcpy(g_state.sdma_heap_device_base, g_state.npes * sizeof(void *),
        g_state.sdma_heap_host_base, g_state.npes * sizeof(void *), ACL_MEMCPY_HOST_TO_DEVICE), aclrtMemcpy);
    SHMEM_CHECK_RET(aclrtMemcpy(g_state.roce_heap_device_base, g_state.npes * sizeof(void *),
        g_state.roce_heap_host_base, g_state.npes * sizeof(void *), ACL_MEMCPY_HOST_TO_DEVICE), aclrtMemcpy);
    auto ret = smem_shm_set_extra_context(g_smem_handle, (void *)&g_state, sizeof(shmemi_device_host_state_t));
    if (ret != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("Failed to attach extra context to segment");
        return ret;
    }
    return SHMEM_SUCCESS;
}

int32_t check_attr(shmem_init_attr_t *attributes)
{
    if ((attributes->my_rank < 0) || (attributes->n_ranks <= 0)) {
        SHM_LOG_ERROR("my_rank:" << attributes->my_rank << " and n_ranks: " << attributes->n_ranks
                                 << " cannot be less 0 , n_ranks still cannot be equal 0");
        return SHMEM_INVALID_VALUE;
    } else if (attributes->n_ranks > SHMEM_MAX_RANKS) {
        SHM_LOG_ERROR("n_ranks: " << attributes->n_ranks << " cannot be more than " << SHMEM_MAX_RANKS);
        return SHMEM_INVALID_VALUE;
    } else if (attributes->my_rank >= attributes->n_ranks) {
        SHM_LOG_ERROR("n_ranks:" << attributes->n_ranks << " cannot be less than my_rank:" << attributes->my_rank);
        return SHMEM_INVALID_PARAM;
    } else if (attributes->local_mem_size <= 0) {
        SHM_LOG_ERROR("local_mem_size:" << attributes->local_mem_size << " cannot be less or equal 0");
        return SHMEM_INVALID_VALUE;
    }
    return SHMEM_SUCCESS;
}

}  // namespace shm

int32_t shmem_set_data_op_engine_type(shmem_init_attr_t *attributes, data_op_engine_type_t value)
{
    SHM_ASSERT_RETURN(attributes != nullptr, SHMEM_INVALID_PARAM);
    attributes->option_attr.data_op_engine_type = value;
    return SHMEM_SUCCESS;
}

int32_t shmem_set_timeout(shmem_init_attr_t *attributes, uint32_t value)
{
    SHM_ASSERT_RETURN(attributes != nullptr, SHMEM_INVALID_PARAM);
    attributes->option_attr.shm_init_timeout = value;
    attributes->option_attr.shm_create_timeout = value;
    attributes->option_attr.control_operation_timeout = value;
    return SHMEM_SUCCESS;
}

int32_t shmem_set_attr(int32_t my_rank, int32_t n_ranks, uint64_t local_mem_size, const char *ip_port,
                       shmem_init_attr_t **attributes)
{
    SHM_ASSERT_RETURN(local_mem_size <= SHMEM_MAX_LOCAL_SIZE, SHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(n_ranks <= SHMEM_MAX_RANKS, SHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(my_rank < SHMEM_MAX_RANKS, SHMEM_INVALID_VALUE);
    *attributes = &shm::g_attr;
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), sizeof(shm::g_ipport) - 1);

        std::copy_n(ip_port, ip_len, shm::g_ipport);
        shm::g_ipport[ip_len] = '\0';
        std::copy_n(shm::g_ipport, ip_len, shm::g_attr.ip_port);
        if (shm::g_ipport[0] == '\0') {
            SHM_LOG_ERROR("my_rank:" << my_rank << " shm::g_ipport is nullptr!");
            return SHMEM_INVALID_VALUE;
        }
    } else {
        SHM_LOG_WARN("init with my_rank:" << my_rank << " ip_port is nullptr!");
    }

    int attr_version = static_cast<int>((1 << 16) + sizeof(shmem_init_attr_t));
    shm::g_attr.my_rank = my_rank;
    shm::g_attr.n_ranks = n_ranks;
    shm::g_attr.ip_port[ip_len] = '\0';
    shm::g_attr.local_mem_size = local_mem_size;
    shm::g_attr.option_attr = {attr_version, SHMEM_DATA_OP_MTE, shm::DEFAULT_TIMEOUT,
                               shm::DEFAULT_TIMEOUT, shm::DEFAULT_TIMEOUT, 0};
    shm::g_attr_init = true;
    return SHMEM_SUCCESS;
}

int32_t shmem_get_uid_magic(shmem_uniqueid_inner_t *innerUId)
{
    std::ifstream urandom("/dev/urandom", std::ios::binary);
    if (!urandom) {
        SHM_LOG_ERROR("open random failed");
        return SHMEM_INNER_ERROR;
    }

    urandom.read(reinterpret_cast<char *>(&innerUId->magic), sizeof(innerUId->magic));
    if (urandom.fail()) {
        SHM_LOG_ERROR("read random failed.");
        return SHMEM_INNER_ERROR;
    }
    SHM_LOG_DEBUG("init magic id to " << innerUId->magic);
    return SHMEM_SUCCESS;
}

int32_t shmem_get_port_magic(shmem_uniqueid_inner_t *innerUId, char *ip_str)
{
    static std::random_device rd;
    const int min_port = shm::MIN_PORT;
    const int max_port = shm::MAX_PORT;
    const int max_attempts = shm::MAX_ATTEMPTS;
    const int offset_bit = 32;
    uint64_t seed = 1;
    seed |= static_cast<uint64_t>(getpid()) << offset_bit;
    seed |= static_cast<uint64_t>(static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count())
                                  & 0xFFFFFFFF);
    static std::mt19937_64 gen(seed);
    std::uniform_int_distribution<> dis(min_port, max_port);

    int sockfd = -1;
    int32_t ret;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        int port = dis(gen);
        if (innerUId->addr.type == ADDR_IPv4) {
            ret = shm::bind_tcp_port_v4(sockfd, port, innerUId, ip_str);
            if (ret == 0) {
                innerUId->inner_sockFd = sockfd;
                return 0;
            }
        } else {
            ret = shm::bind_tcp_port_v6(sockfd, port, innerUId, ip_str);
            if (ret == 0) {
                innerUId->inner_sockFd = sockfd;
                return 0;
            }
        }
    }
    SHM_LOG_ERROR("Not find a available tcp port");
    return -1;
}

int32_t shmem_using_env_port(shmem_uniqueid_inner_t *innerUId, char *ip_str, uint16_t envPort)
{
    if (envPort < shm::MIN_PORT) {   // envPort > MAX_PORT always false
        SHM_LOG_ERROR("env port is invalid. " << envPort);
        return SHMEM_INVALID_PARAM;
    }

    int sockfd = -1;
    int32_t ret;
    if (innerUId->addr.type == ADDR_IPv4) {
        ret = shm::bind_tcp_port_v4(sockfd, envPort, innerUId, ip_str);
        if (ret == 0) {
            innerUId->inner_sockFd = sockfd;
            return 0;
        }
    } else {
        ret = shm::bind_tcp_port_v6(sockfd, envPort, innerUId, ip_str);
        if (ret == 0) {
            innerUId->inner_sockFd = sockfd;
            return 0;
        }
    }
    SHM_LOG_ERROR("init with env port fialed " << envPort << ", ret=" << ret);
    return ret;
}

int32_t ParseInterfaceWithType(const char *ipInfo, char *IP, sa_family_t &sockType, bool &flag)
{
    const char *delim = ":";
    const char *sep = strchr(ipInfo, delim[0]);
    if (sep != nullptr) {
        size_t leftLen = sep - ipInfo;
        if (leftLen >= shm::MAX_IFCONFIG_LENGTH - 1 || leftLen == 0) {
            return SHMEM_INVALID_VALUE;
        }
        std::copy_n(ipInfo, leftLen, IP);
        IP[leftLen] = '\0';
        sockType = (strcmp(sep + 1, "inet6") != 0) ? AF_INET : AF_INET6;
        flag = true;
    }
    return SHMEM_SUCCESS;
}

int32_t shmem_auto_get_ip(struct sockaddr *ifaAddr, char *local, sa_family_t &sockType)
{
    sockType = ifaAddr->sa_family;
    if (sockType == AF_INET) {
        auto localIp = reinterpret_cast<struct sockaddr_in *>(ifaAddr)->sin_addr;
        if (inet_ntop(sockType, &localIp, local, shm::MAX_IP) == nullptr) {
            SHM_LOG_ERROR("convert local ipv4 to string failed. ");
            return SHMEM_INVALID_PARAM;
        }
        return SHMEM_SUCCESS;
    } else if (sockType == AF_INET6) {
        auto localIp = reinterpret_cast<struct sockaddr_in6 *>(ifaAddr)->sin6_addr;
        if (inet_ntop(sockType, &localIp, local, shm::MAX_IP) == nullptr) {
            SHM_LOG_ERROR("convert local ipv6 to string failed. ");
            return SHMEM_INVALID_PARAM;
        }
        return SHMEM_SUCCESS;
    }
    return SHMEM_INVALID_PARAM;
}

bool shmem_check_ifa(struct ifaddrs *ifa, sa_family_t sockType, bool flag, char *ifaName, size_t ifaLen)
{
    if (ifa->ifa_addr == nullptr || ifa->ifa_netmask == nullptr || ifa->ifa_name == nullptr) {
        SHM_LOG_DEBUG("loop ifa_addr/ifa_netmask/ifa_name is nullptr");
        return false;
    }

    // socket type match and input env ifa valid
    if (ifa->ifa_addr->sa_family != sockType && flag) {
        SHM_LOG_DEBUG("sa family is not match, get " << ifa->ifa_addr->sa_family << ", expect " << sockType);
        return false;
    }

    //  prefix match with input ifa name
    if (strncmp(ifa->ifa_name, ifaName, ifaLen) != 0) {
        SHM_LOG_DEBUG("ifa name prefix un-match, get " << ifa->ifa_name << ", expect " << ifaName);
        return false;
    }

    // ignore ifa which is down or loopback or not running
    if ((ifa->ifa_flags & IFF_LOOPBACK) || !(ifa->ifa_flags & IFF_RUNNING) || !(ifa->ifa_flags & IFF_UP)) {
        SHM_LOG_DEBUG("ifa flag un-match, flag=" << ifa->ifa_flags);
        return false;
    }

    if (sockType == AF_INET6) {
        struct sockaddr_in6 *sa6 = reinterpret_cast<struct sockaddr_in6 *>(ifa->ifa_addr);
        if (IN6_IS_ADDR_LINKLOCAL(&sa6->sin6_addr)) {
            SHM_LOG_DEBUG("ifa is scope link addr " << ifaName);
            return false;
        }
    }
    return true;
}

int32_t shmem_get_ip_from_ifa(char *local, sa_family_t &sockType, const string ipInfo)
{
    struct ifaddrs *ifaddr;
    char ifaName[shm::MAX_IFCONFIG_LENGTH];
    sockType = AF_INET;
    bool flag = false;
    if (ipInfo.empty()) {
        std::copy_n("eth", shm::DEFAULT_IFNAME_LNEGTH, ifaName);
        ifaName[shm::DEFAULT_IFNAME_LNEGTH - 1] = '\0';
        SHM_LOG_INFO("use default if to find IP:" << ifaName);
    } else if (ParseInterfaceWithType(ipInfo.c_str(), ifaName, sockType, flag) != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("IP size set in SHMEM_CONF_STORE_MASTER_IF format has wrong length");
        return SHMEM_INVALID_PARAM;
    }
    if (getifaddrs(&ifaddr) == -1) {
        SHM_LOG_ERROR("get local net interfaces failed: " << errno);
        return SHMEM_INVALID_PARAM;
    }
    int32_t result = SHMEM_INVALID_PARAM;
    const int IP_STR_BUFFER_SIZE = 64;
    for (auto ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!shmem_check_ifa(ifa, sockType, flag, ifaName, strlen(ifaName))) {
            continue;
        }
        if (sockType == AF_INET && flag) {
            auto localIp = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr)->sin_addr;
            if (inet_ntop(sockType, &localIp, local, IP_STR_BUFFER_SIZE) == nullptr) {
                SHM_LOG_ERROR("convert local ipv4 to string failed. ");
                continue;
            }
            result = SHMEM_SUCCESS;
            break;
        } else if (sockType == AF_INET6 && flag) {
            auto localIp = reinterpret_cast<struct sockaddr_in6 *>(ifa->ifa_addr)->sin6_addr;
            if (inet_ntop(sockType, &localIp, local, IP_STR_BUFFER_SIZE) == nullptr) {
                SHM_LOG_ERROR("convert local ipv6 to string failed. ");
                continue;
            }
            result = SHMEM_SUCCESS;
            break;
        } else {
            auto ret = shmem_auto_get_ip(ifa->ifa_addr, local, sockType);
            if (ret != SHMEM_SUCCESS) {
                continue;
            }
            result = SHMEM_SUCCESS;
            break;
        }
    }
    freeifaddrs(ifaddr);
    return result;
}

int32_t shmem_get_ip_from_env(char *ip, uint16_t &port, sa_family_t &sockType, const string ipPort)
{
    if (!ipPort.empty()) {
        SHM_LOG_DEBUG("get env SHMEM_UID_SESSION_ID value:" << ipPort);
        std::string ipPortStr = ipPort;

        if (ipPort[0] == '[') {
            sockType = AF_INET6;
            size_t found = ipPortStr.find_last_of(']');
            if (found == std::string::npos || ipPortStr.length() - found <= 1) {
                SHM_LOG_ERROR("get env SHMEM_UID_SESSION_ID is invalid");
                return SHMEM_INVALID_PARAM;
            }
            std::string ipStr = ipPortStr.substr(1, found - 1);
            std::string portStr = ipPortStr.substr(found + 2);

            std::string result = ipStr;
            if (result.length() >= shm::MAX_IP) {
                SHM_LOG_ERROR("IP address is too long");
                return SHMEM_INVALID_PARAM;
            }
            std::copy(result.begin(), result.end(), ip);
            ip[result.length()] = '\0';

            port = std::stoi(portStr);
        } else {
            sockType = AF_INET;
            size_t found = ipPortStr.find_last_of(':');
            if (found == std::string::npos || ipPortStr.length() - found <= 1) {
                SHM_LOG_ERROR("get env SHMEM_UID_SESSION_ID is invalid");
                return SHMEM_INVALID_PARAM;
            }
            std::string ipStr = ipPortStr.substr(0, found);
            std::string portStr = ipPortStr.substr(found + 1);

            std::string result = ipStr;
            if (result.length() >= shm::MAX_IP) {
                SHM_LOG_ERROR("IP address is too long");
                return SHMEM_INVALID_PARAM;
            }
            std::copy(result.begin(), result.end(), ip);
            ip[result.length()] = '\0';

            port = std::stoi(portStr);
        }
        return SHMEM_SUCCESS;
    }
    return SHMEM_INVALID_PARAM;
}

int32_t shmem_set_ip_info(shmem_uniqueid_t *uid, sa_family_t &sockType, char *pta_env_ip, uint16_t pta_env_port,
                          bool is_from_ifa)
{
    // init default uid
    SHM_ASSERT_RETURN(uid != nullptr, SHMEM_INVALID_PARAM);
    *uid = SHMEM_UNIQUEID_INITIALIZER;
    shmem_uniqueid_inner_t *innerUID = reinterpret_cast<shmem_uniqueid_inner_t *>(uid);
    if (sockType == AF_INET) {
        innerUID->addr.addr.addr4.sin_family = AF_INET;
        if (inet_pton(AF_INET, pta_env_ip, &(innerUID->addr.addr.addr4.sin_addr)) <= 0) {
            SHM_LOG_ERROR("inet_pton IPv4 failed");
            return SHMEM_NOT_INITED;
        }
        innerUID->addr.type = ADDR_IPv4;
    } else if (sockType == AF_INET6) {
        innerUID->addr.addr.addr6.sin6_family = AF_INET6;
        if (inet_pton(AF_INET6, pta_env_ip, &(innerUID->addr.addr.addr6.sin6_addr)) <= 0) {
            SHM_LOG_ERROR("inet_pton IPv6 failed");
            return SHMEM_NOT_INITED;
        }
        innerUID->addr.type = ADDR_IPv6;
    } else {
        SHM_LOG_ERROR("IP Type is not IPv4 or IPv6");
        return SHMEM_INVALID_PARAM;
    }

    // fill ip port as part of uid
    if (is_from_ifa) {
        int32_t ret = shmem_get_port_magic(innerUID, pta_env_ip);
        if (ret != 0) {
            SHM_LOG_ERROR("get available port failed.");
            return SHMEM_INVALID_PARAM;
        }
    } else {
        int32_t ret = shmem_using_env_port(innerUID, pta_env_ip, pta_env_port);
        if (ret != 0) {
            SHM_LOG_ERROR("using env port failed.");
            return SHMEM_INVALID_PARAM;
        }
    }

    SHM_LOG_INFO("gen unique id success.");
    return SHMEM_SUCCESS;
}

int32_t shmem_get_uniqueid(shmem_uniqueid_t *uid)
{
    if (shmem_set_log_level(shm::WARN_LEVEL) != 0) {
        SHM_LOG_ERROR("failed to set log level");
        return SHMEM_INNER_ERROR;
    }
    char pta_env_ip[shm::MAX_IP];
    uint16_t pta_env_port{};
    sa_family_t sockType;
    const char *ipPortInput = std::getenv("SHMEM_UID_SESSION_ID");
    const char *ipInfoInput = std::getenv("SHMEM_UID_SOCK_IFNAM");
    const string ipPort = ipPortInput ? ipPortInput : "";
    const string ipInfo = ipInfoInput ? ipInfoInput : "";
    bool is_from_ifa = false;
    if (!ipPort.empty()) {
        if (shmem_get_ip_from_env(pta_env_ip, pta_env_port, sockType, ipPort) != SHMEM_SUCCESS) {
            SHM_LOG_ERROR("cant get pta master addr.");
            return SHMEM_INVALID_PARAM;
        }
    } else {
        is_from_ifa = true;
        if (shmem_get_ip_from_ifa(pta_env_ip, sockType, ipInfo) != SHMEM_SUCCESS) {
            SHM_LOG_ERROR("cant get available ip port.");
            return SHMEM_INVALID_PARAM;
        }
    }
    SHM_LOG_INFO("get master IP value:" << pta_env_ip);
    return shmem_set_ip_info(uid, sockType, pta_env_ip, pta_env_port, is_from_ifa);
}

int32_t shmem_set_attr_uniqueid_args(int rank_id, int nranks, const shmem_uniqueid_t *uid, shmem_init_attr_t *attr)
{
    if (attr == nullptr || uid == nullptr) {
        SHM_LOG_ERROR("set unique id attr/uid is null");
        return SHMEM_INVALID_PARAM;
    }

    if (rank_id != shm::g_attr.my_rank || nranks != shm::g_attr.n_ranks) {
        SHM_LOG_ERROR("rankid/nranks invalid, maybe call shmem_set_attr firstly.");
        return SHMEM_INVALID_PARAM;
    }

    if (uid->version != SHMEM_UNIQUEID_VERSION) {
        SHM_LOG_ERROR("uid version invalid, init unique id with shmem_get_uniqueid firstly.");
        return SHMEM_INVALID_PARAM;
    }

    // extract ip port from inner unique id
    shmem_uniqueid_inner_t *innerUID = reinterpret_cast<shmem_uniqueid_inner_t *>(const_cast<shmem_uniqueid_t *>(uid));

    // compatibility with shmem_init_attr, init ip_port from unique id
    std::string ipPort;
    if (innerUID->addr.type == ADDR_IPv6) {
        char ipStr[INET6_ADDRSTRLEN] = {0};
        if (inet_ntop(AF_INET6, &(innerUID->addr.addr.addr6.sin6_addr), ipStr, sizeof(ipStr)) == nullptr) {
            SHM_LOG_ERROR("inet_ntop failed for IPv6");
            return SHMEM_INNER_ERROR;
        }
        uint16_t port = ntohs(innerUID->addr.addr.addr6.sin6_port);
        ipPort = "tcp6://[" + std::string(ipStr) + "]:" + std::to_string(port);
    } else {
        char ipStr[INET_ADDRSTRLEN] = {0};
        if (inet_ntop(AF_INET, &(innerUID->addr.addr.addr4.sin_addr), ipStr, sizeof(ipStr)) == nullptr) {
            SHM_LOG_ERROR("inet_ntop failed for IPv4");
            return SHMEM_INNER_ERROR;
        }
        uint16_t port = ntohs(innerUID->addr.addr.addr4.sin_port);
        ipPort = "tcp://" + std::string(ipStr) + ":" + std::to_string(port);
    }
    std::copy(ipPort.begin(), ipPort.end(), shm::g_ipport);
    std::copy(ipPort.begin(), ipPort.end(), shm::g_attr.ip_port);
    std::copy(ipPort.begin(), ipPort.end(), attr->ip_port);
    shm::g_ipport[ipPort.size()] = '\0';
    shm::g_attr.ip_port[ipPort.size()] = '\0';
    attr->ip_port[ipPort.size()] = '\0';
    attr->option_attr.sockFd = innerUID->inner_sockFd;
    SHM_LOG_INFO("extract ip port:" << ipPort);

    int32_t status = shmem_init_attr(attr);
    if (status != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("shmem_init_attr failed");
        return status;
    }
    return SHMEM_SUCCESS;
}

int32_t shmem_init_status(void)
{
    if (!shm::g_state.is_shmem_created)
        return SHMEM_STATUS_NOT_INITIALIZED;
    else if (!shm::g_state.is_shmem_initialized)
        return SHMEM_STATUS_SHM_CREATED;
    else if (shm::g_state.is_shmem_initialized)
        return SHMEM_STATUS_IS_INITIALIZED;
    else
        return SHMEM_STATUS_INVALID;
}

void shmem_rank_exit(int status)
{
    SHM_LOG_DEBUG("shmem_rank_exit is work ,status: " << status);
    exit(status);
}

int32_t shmem_init_attr(shmem_init_attr_t *attributes)
{
    int32_t ret;

    SHM_ASSERT_RETURN(attributes != nullptr, SHMEM_INVALID_PARAM);
    SHMEM_CHECK_RET(shmem_set_log_level(shm::WARN_LEVEL), shmem_set_log_level);
    SHMEM_CHECK_RET(shm::check_attr(attributes), check_attr);
    SHMEM_CHECK_RET(shm::version_compatible(), version_compatible);
    SHMEM_CHECK_RET(shm::shmemi_options_init(), shmemi_options_init);

    SHMEM_CHECK_RET(shm::shmemi_state_init_attr(attributes), shmemi_state_init_attr);
    SHMEM_CHECK_RET(shm::shmemi_heap_init(attributes), shmemi_heap_init);
    SHMEM_CHECK_RET(shm::update_device_state(), update_device_state);

    // heap_size is aligned, use actual local_mem_size to init mm
    SHMEM_CHECK_RET(shm::memory_manager_initialize(shm::g_state.heap_base, attributes->local_mem_size + SHMEM_EXTRA_SIZE),
                    memory_manager_initialize);
    SHMEM_CHECK_RET(shm::shmemi_team_init(shm::g_state.mype, shm::g_state.npes), shmemi_team_init);
    SHMEM_CHECK_RET(shm::update_device_state(), update_device_state);
    SHMEM_CHECK_RET(shm::shmemi_sync_init(), shmemi_sync_init);
    SHMEM_CHECK_RET(smem_shm_register_exit(shm::g_smem_handle, &shmem_rank_exit), smem_shm_register_exit);
    shm::g_state.is_shmem_initialized = true;
    SHMEM_CHECK_RET(shm::shmemi_control_barrier_all(), shmemi_control_barrier_all);
    return SHMEM_SUCCESS;
}

int32_t shmem_set_config_store_tls_key(const char *tls_pk, const uint32_t tls_pk_len,
    const char *tls_pk_pw, const uint32_t tls_pk_pw_len, const shmem_decrypt_handler decrypt_handler)
{
    return smem_set_config_store_tls_key(tls_pk, tls_pk_len, tls_pk_pw, tls_pk_pw_len, decrypt_handler);
}

int32_t shmem_set_extern_logger(void (*func)(int level, const char *msg))
{
    SHM_ASSERT_RETURN(func != nullptr, SHMEM_INVALID_PARAM);
    shm::shm_out_logger::Instance().set_extern_log_func(func, true);
    return smem_set_extern_logger(func);
}

int32_t shmem_set_log_level(int level)
{
    // use env first, input level secondly, user may change level from env instead call func
    const char *in_level = std::getenv("SHMEM_LOG_LEVEL");
    if (in_level != nullptr) {
        auto tmp_level = std::string(in_level);
        if (tmp_level == "DEBUG") {
            level = shm::DEBUG_LEVEL;
        } else if (tmp_level == "INFO") {
            level = shm::INFO_LEVEL;
        } else if (tmp_level == "WARN") {
            level = shm::WARN_LEVEL;
        } else if (tmp_level == "ERROR") {
            level = shm::ERROR_LEVEL;
        } else if (tmp_level == "FATAL") {
            level = shm::FATAL_LEVEL;
        }
    }
    shm::shm_out_logger::Instance().set_log_level(static_cast<shm::log_level>(level));
    if (smem_set_log_level(level) != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("Failed to set ock::mf::OutLogger level");
    }
    return SHMEM_SUCCESS;
}

int32_t shmem_set_conf_store_tls(bool enable, const char *tls_info, const uint32_t tls_info_len)
{
    return smem_set_conf_store_tls(enable, tls_info, tls_info_len);
}

int32_t shmem_finalize(void)
{
    SHMEM_CHECK_RET(shm::shmemi_team_finalize());

    if (shm::g_state.p2p_heap_host_base != nullptr) {
        aclrtFree(shm::g_state.p2p_heap_host_base);
    }
    if (shm::g_state.sdma_heap_host_base != nullptr) {
        aclrtFree(shm::g_state.sdma_heap_host_base);
    }
    if (shm::g_state.roce_heap_host_base != nullptr) {
        aclrtFree(shm::g_state.roce_heap_host_base);
    }

    if (shm::g_state.p2p_heap_device_base != nullptr) {
        aclrtFree(shm::g_state.p2p_heap_device_base);
    }
    if (shm::g_state.sdma_heap_device_base != nullptr) {
        aclrtFree(shm::g_state.sdma_heap_device_base);
    }
    if (shm::g_state.roce_heap_device_base != nullptr) {
        aclrtFree(shm::g_state.roce_heap_device_base);
    }

    if (shm::g_smem_handle != nullptr) {
        int32_t status = smem_shm_destroy(shm::g_smem_handle, 0);
        if (status != SHMEM_SUCCESS) {
            SHM_LOG_ERROR("smem_shm_destroy Failed");
            return SHMEM_SMEM_ERROR;
        }
        shm::g_smem_handle = nullptr;
    }
    smem_shm_uninit(0);
    smem_uninit();
    return SHMEM_SUCCESS;
}

void shmem_info_get_version(int *major, int *minor)
{
    SHM_ASSERT_RET_VOID(major != nullptr && minor != nullptr);
    *major = SHMEM_MAJOR_VERSION;
    *minor = SHMEM_MINOR_VERSION;
}

void shmem_info_get_name(char *name)
{
    SHM_ASSERT_RET_VOID(name != nullptr);
    std::ostringstream oss;
    oss << "SHMEM v" << SHMEM_VENDOR_MAJOR_VER << "." << SHMEM_VENDOR_MINOR_VER << "." << SHMEM_VENDOR_PATCH_VER;
    auto version_str = oss.str();
    size_t i;
    for (i = 0; i < SHMEM_MAX_NAME_LEN - 1 && version_str[i] != '\0'; i++) {
        name[i] = version_str[i];
    }
    name[i] = '\0';
}

void shmem_global_exit(int status)
{
    smem_shm_global_exit(shm::g_smem_handle, status);
}
