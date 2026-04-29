/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "hybm_define.h"
#include "hybm_logger.h"
#include "dl_acl_api.h"
#include "dl_hccp_api.h"
#include "device_rdma_common.h"
#include "device_rdma_helper.h"
#include "fixed_ranks_qp_manager.h"
#include "dynamic_ranks_qp_manager.h"
#include "device_rdma_transport_manager.h"

namespace ock {
namespace mf {
namespace transport {
namespace device {
bool RdmaTransportManager::tsdOpened_ = false;
bool RdmaTransportManager::deviceIpRetired_ = false;
bool RdmaTransportManager::raInitialized_ = false;
void* RdmaTransportManager::storedRdmaHandle_ = nullptr;

RdmaTransportManager::~RdmaTransportManager()
{
    ClearAllRegisterMRs();
    tsdOpened_ = false;
    raInitialized_ = false;
    deviceIpRetired_ = false;
    storedRdmaHandle_ = nullptr;
}

void RdmaTransportManager::InitializeDeviceAddress(mf_sockaddr &deviceAddr)
{
    if (deviceIp_.type == IpV4) {
        deviceAddr.ip.ipv4.sin_family = AF_INET;
        deviceAddr.ip.ipv4.sin_addr = deviceIp_.ip.ipv4;
        deviceAddr.ip.ipv4.sin_port = devicePort_;
        deviceAddr.type = IpV4;
    } else if (deviceIp_.type == IpV6) {
        deviceAddr.ip.ipv6.sin6_family = AF_INET6;
        deviceAddr.ip.ipv6.sin6_addr = deviceIp_.ip.ipv6;
        deviceAddr.ip.ipv6.sin6_port = devicePort_;
        deviceAddr.type = IpV6;
    }
}

Result RdmaTransportManager::OpenDevice(const TransportOptions &options)
{
    int32_t deviceId = -1;

    BM_LOG_DEBUG(rankId_ << " begin to open device with " << options);
    auto ret = DlAclApi::AclrtGetDevice(&deviceId);
    if (ret != 0 || deviceId < 0) {
        BM_LOG_ERROR(rankId_ << " AclrtGetDevice() return=" << ret << ", output deviceId=" << deviceId);
        return BM_DL_FUNCTION_FAILED;
    }
    deviceId_ = static_cast<uint32_t>(deviceId);
    rankId_ = options.rankId;
    rankCount_ = options.rankCount;
    role_ = options.role;
    ret = ParseDeviceNic(options.nic, devicePort_);
    if (ret != BM_OK) {
        BM_LOG_ERROR(rankId_ << " parse input nic(" << options.nic << ") failed!");
        return BM_INVALID_PARAM;
    }

    if (options.type == IpV4) {
        deviceIp_.type = IpV4;
    } else if (options.type == IpV6) {
        deviceIp_.type = IpV6;
    }
    if (!PrepareOpenDevice(deviceId_, rankCount_, deviceIp_, rdmaHandle_)) {
        BM_LOG_ERROR(rankId_ << " PrepareOpenDevice failed.");
        return BM_ERROR;
    }

    nicInfo_ = GenerateDeviceNic(deviceIp_, devicePort_);

    mf_sockaddr deviceAddr;
    InitializeDeviceAddress(deviceAddr);
    if (role_ == HYBM_ROLE_PEER) {
        qpManager_ = std::make_shared<FixedRanksQpManager>(deviceId_, rankId_, rankCount_, deviceAddr);
    } else {
        qpManager_ = std::make_shared<DynamicRanksQpManager>(deviceId_, rankId_, rankCount_, deviceAddr,
                                                             role_ == HYBM_ROLE_RECEIVER);
    }

    deviceChipInfo_ = std::make_shared<DeviceChipInfo>(deviceId_);
    ret = deviceChipInfo_->Init();
    if (ret != BM_OK) {
        BM_LOG_ERROR(rankId_ << " device info init failed: " << ret);
        return ret;
    }
    BM_LOG_INFO(rankId_ << " open device with " << options << " success.");
    return BM_OK;
}

Result RdmaTransportManager::CloseDevice()
{
    if (qpManager_ != nullptr) {
        qpManager_->Shutdown();
        qpManager_ = nullptr;
    }
    return BM_OK;
}

Result RdmaTransportManager::RegisterMemoryRegion(const TransportMemoryRegion &mr)
{
    void *mrHandle = nullptr;
    HccpMrInfo info{};
    info.addr = (void *)(ptrdiff_t)mr.addr;
    info.size = mr.size;
    info.access = mr.access;
    auto ret = DlHccpApi::RaRegisterMR(rdmaHandle_, &info, mrHandle);
    if (ret != 0) {
        BM_LOG_ERROR(rankId_ << " register MR=" << mr << " failed: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }

    RegMemResult result{mr.addr, mr.size, mrHandle, info.lkey, info.rkey};
    BM_LOG_DEBUG(rankId_ << " register MR result=" << result);

    registerMRS_.emplace(mr.addr, result);
    ret = qpManager_->SetLocalMemories(registerMRS_);
    if (ret != BM_OK) {
        BM_LOG_ERROR(rankId_ << " qp manager set mr failed: " << ret);
        return ret;
    }

    return BM_OK;
}

Result RdmaTransportManager::UnregisterMemoryRegion(uint64_t addr)
{
    auto pos = registerMRS_.find(addr);
    if (pos == registerMRS_.end()) {
        BM_LOG_ERROR(rankId_ << " input address not register!");
        return BM_INVALID_PARAM;
    }

    auto ret = DlHccpApi::RaDeregisterMR(rdmaHandle_, pos->second.mrHandle);
    if (ret != 0) {
        BM_LOG_ERROR(rankId_ << " Unregister MR addr failed: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }

    registerMRS_.erase(pos);
    ret = qpManager_->SetLocalMemories(registerMRS_);
    if (ret != BM_OK) {
        BM_LOG_ERROR(rankId_ << " qp manager set mr failed: " << ret);
        return ret;
    }
    return BM_OK;
}

Result RdmaTransportManager::QueryMemoryKey(uint64_t addr, TransportMemoryKey &key)
{
    RegMemKeyUnion keyUnion{};
    auto pos = registerMRS_.lower_bound(addr);
    if (pos == registerMRS_.end() || pos->first + pos->second.size <= addr) {
        BM_LOG_ERROR(rankId_ << " input address not register");
        return BM_INVALID_PARAM;
    }

    keyUnion.deviceKey = pos->second;

    key = keyUnion.commonKey;
    return BM_OK;
}

Result RdmaTransportManager::ParseMemoryKey(const TransportMemoryKey &key, uint64_t &addr, uint64_t &size)
{
    RegMemKeyUnion keyUnion{};
    keyUnion.commonKey = key;
    if (keyUnion.deviceKey.type != TT_HCCP) {
        BM_LOG_ERROR(rankId_ << " parse memory key type invalid: " << keyUnion.deviceKey.type);
        return BM_ERROR;
    }

    addr = keyUnion.deviceKey.address;
    size = keyUnion.deviceKey.size;
    return BM_OK;
}

Result RdmaTransportManager::Prepare(const HybmTransPrepareOptions &options)
{
    BM_LOG_DEBUG(rankId_ << " RdmaTransportManager Prepare with : " << options);
    int ret;
    if ((ret = CheckPrepareOptions(options)) != 0) {
        return ret;
    }

    mf_sockaddr deviceNetwork;
    std::unordered_map<uint32_t, ConnectRankInfo> rankInfo;
    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        ret = ParseDeviceNic(it->second.nic, deviceNetwork);
        if (ret != BM_OK) {
            BM_LOG_ERROR(rankId_ << " parse networks[" << it->first << "]=" << it->second.nic << " failed: " << ret);
            return BM_INVALID_PARAM;
        }

        rankInfo.emplace(it->first, ConnectRankInfo{it->second.role, deviceNetwork, it->second.memKeys});
    }
    BM_LOG_DEBUG(rankId_ << " SetRemoteRankInfo rankInfo.size=" << rankInfo.size());

    ret = qpManager_->SetRemoteRankInfo(rankInfo);
    if (ret != BM_OK) {
        BM_LOG_ERROR(rankId_ << " qp manager set remote rank info failed: " << ret);
        return ret;
    }

    ret = qpManager_->Startup(rdmaHandle_);
    if (ret != BM_OK) {
        BM_LOG_ERROR(rankId_ << " qp manager startup failed: " << ret);
        return ret;
    }

    return BM_OK;
}

Result RdmaTransportManager::Connect()
{
    auto ret = AsyncConnect();
    if (ret != BM_OK) {
        BM_LOG_ERROR(rankId_ << " AsyncConnect() failed: " << ret);
        return ret;
    }

    ret = WaitForConnected(-1L);
    if (ret != BM_OK) {
        BM_LOG_ERROR(rankId_ << " WaitForConnected(-1) failed: " << ret);
        return ret;
    }

    return BM_OK;
}

Result RdmaTransportManager::AsyncConnect()
{
    return BM_OK;
}

Result RdmaTransportManager::WaitForConnected(int64_t timeoutNs)
{
    if (qpManager_ == nullptr) {
        BM_LOG_ERROR(rankId_ << " server side not listen!");
        return BM_ERROR;
    }

    auto ret = qpManager_->WaitingConnectionReady();
    if (ret != BM_OK) {
        BM_LOG_ERROR(rankId_ << " wait for server side connected on device failed: " << ret);
        return ret;
    }

    return BM_OK;
}

Result RdmaTransportManager::UpdateRankOptions(const HybmTransPrepareOptions &options)
{
    BM_LOG_DEBUG(rankId_ << " RdmaTransportManager Prepare with : " << options);
    if (qpManager_ == nullptr) {
        BM_LOG_ERROR(rankId_ << " qp manager not created");
        return BM_ERROR;
    }

    mf_sockaddr deviceNetwork;
    std::unordered_map<uint32_t, ConnectRankInfo> ranksInfo;
    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        auto ret = ParseDeviceNic(it->second.nic, deviceNetwork);
        if (ret != BM_OK) {
            BM_LOG_ERROR(rankId_ << " update rank network(" << it->second.nic << ") invalid.");
            return BM_INVALID_PARAM;
        }
        BM_LOG_INFO(rankId_ << " UpdateRankOptions update rank: " << it->first);
        ranksInfo.emplace(it->first, ConnectRankInfo{it->second.role, deviceNetwork, it->second.memKeys});
    }
    BM_LOG_DEBUG(rankId_ << " UpdateRankOptions ranksInfo.size=" << ranksInfo.size());

    auto ret = qpManager_->SetRemoteRankInfo(ranksInfo);
    if (ret != BM_OK) {
        BM_LOG_ERROR(rankId_ << " update rank options failed: " << ret);
        return ret;
    }

    return BM_OK;
}

const std::string &RdmaTransportManager::GetNic() const
{
    return nicInfo_;
}

const void *RdmaTransportManager::GetQpInfo() const
{
    if (qpManager_ == nullptr) {
        BM_LOG_ERROR(rankId_ << " GetQpInfo(): connection manager not created.");
        return nullptr;
    }
    return qpManager_->GetQpInfoAddress();
}

bool RdmaTransportManager::PrepareOpenDevice(uint32_t device, uint32_t rankCount,
                                             net_addr_t &deviceIp, void *&rdmaHandle)
{
    // If can get rdmaHandle, maybe the device has been opened, can try get rdmaHandle directly.
    if (DlHccpApi::RaRdevGetHandle(device, rdmaHandle) == 0) {
        if (rdmaHandle != nullptr) {
            if (!RetireDeviceIp(device, deviceIp)) {
                BM_LOG_ERROR(device << " RetireDeviceIp failed.");
                return false;
            }
            BM_LOG_DEBUG(device << " Had prepared device and get rdmaHandle success.");
            return true;
        }
        BM_LOG_INFO(device << " Had prepared device, but rdmaHandle is null, need init again.");
    }
    if (!OpenTsd(device, rankCount)) {
        BM_LOG_ERROR(device << " open tsd failed.");
        return false;
    }

    if (!RaInit(device)) {
        BM_LOG_ERROR(device << " RaInit failed.");
        return false;
    }

    if (!RetireDeviceIp(device, deviceIp)) {
        BM_LOG_ERROR(device << " RetireDeviceIp failed.");
        return false;
    }

    if (!RaRdevInit(device, deviceIp, rdmaHandle)) {
        BM_LOG_ERROR(device << " RaRdevInit failed.");
        return false;
    }
    return true;
}

bool RdmaTransportManager::OpenTsd(uint32_t deviceId, uint32_t rankCount)
{
    if (tsdOpened_) {
        BM_LOG_INFO(deviceId << " tsd already opened.");
        return true;
    }

    auto res = DlHccpApi::TsdOpen(deviceId, rankCount);
    if (res != 0) {
        BM_LOG_ERROR("TsdOpen for (deviceId=" << deviceId << ", rankCount=" << rankCount << ") failed: " << res);
        return false;
    }

    BM_LOG_DEBUG("open tsd for device id: " << deviceId << ", rank count: " << rankCount << " success.");
    tsdOpened_ = true;
    return true;
}

bool RdmaTransportManager::RaInit(uint32_t deviceId)
{
    if (raInitialized_) {
        BM_LOG_INFO(deviceId << " ra already initialized.");
        return true;
    }
    const std::chrono::seconds WAIT_TIME(3);
    HccpRaInitConfig initConfig{};
    initConfig.phyId = deviceId;
    initConfig.nicPosition = NETWORK_OFFLINE;
    initConfig.hdcType = 6;  // HDC_SERVICE_TYPE_RDMA = 6
    BM_LOG_DEBUG(deviceId << " RaInit=" << initConfig);
    std::this_thread::sleep_for(WAIT_TIME); // avoid hccl init conflict
    auto ret = DlHccpApi::RaInit(initConfig);
    if (ret != 0) {
        BM_LOG_WARN(deviceId << " Hccp Init RA failed: " << ret);
        // maybe hccl have already initialized ra, wait 3s then return true.
        std::this_thread::sleep_for(WAIT_TIME);
        raInitialized_ = true;
        return true;
    }

    BM_LOG_DEBUG(deviceId << " ra init for device id: " << deviceId << " success.");
    raInitialized_ = true;
    return true;
}

bool RdmaTransportManager::HandleRetiredDeviceIp(uint32_t deviceId, net_addr_t &deviceIp, net_addr_t &retiredIp)
{
    if (deviceIpRetired_ && deviceIp.type == IpV4) {
        BM_LOG_INFO(deviceId << " device ip already retired : " << inet_ntoa(retiredIp.ip.ipv4));
        deviceIp = retiredIp;
        return true;
    } else if (deviceIpRetired_ && deviceIp.type == IpV6) {
        char ipv6Str[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, &retiredIp.ip.ipv6, ipv6Str, INET6_ADDRSTRLEN);
        BM_LOG_INFO(deviceId << " device ip already retired : " << ipv6Str);
        deviceIp = retiredIp;
        return true;
    }
    return false;
}

bool RdmaTransportManager::RetireDeviceIp(uint32_t deviceId, net_addr_t &deviceIp)
{
    net_addr_t retiredIp{};

    auto isRetire = HandleRetiredDeviceIp(deviceId, deviceIp, retiredIp);
    if (isRetire) {
        return true;
    }

    uint32_t count = 0;
    std::vector<HccpInterfaceInfo> infos;

    HccpRaGetIfAttr config;
    config.phyId = deviceId;
    config.nicPosition = NETWORK_OFFLINE;
    config.isAll = true;

    auto ret = DlHccpApi::RaGetIfNum(config, count);
    if (ret != 0 || count == 0) {
        BM_LOG_ERROR(deviceId << " get interface count failed: " << ret << ", count: " << count);
        return false;
    }

    infos.resize(count);
    ret = DlHccpApi::RaGetIfAddrs(config, infos.data(), count);
    if (ret != 0) {
        BM_LOG_ERROR(deviceId << " get interface information failed: " << ret);
        return false;
    }

    for (auto &info : infos) {
        if (info.family == AF_INET) {
            deviceIp.ip.ipv4 = retiredIp.ip.ipv4 = info.ifaddr.ip.addr;
            deviceIp.type = IpV4;
            deviceIpRetired_ = true;
            BM_LOG_DEBUG(deviceId << " retire device ip success : " << inet_ntoa(deviceIp.ip.ipv4));
            return true;
        }
        if (info.family == AF_INET6) {
            deviceIp.ip.ipv6 = retiredIp.ip.ipv6 = info.ifaddr.ip.addr6;
            deviceIp.type = IpV6;
            deviceIpRetired_ = true;
            char ipv6Str[INET6_ADDRSTRLEN];
            inet_ntop(AF_INET6, &deviceIp.ip.ipv6, ipv6Str, INET6_ADDRSTRLEN);
            BM_LOG_DEBUG(deviceId << " retire device ip success : " << ipv6Str);
            return true;
        }
    }

    BM_LOG_ERROR(deviceId << " not found network device of AF_INET or AF_INET6 on NPU.");
    return false;
}

bool RdmaTransportManager::RaRdevInit(uint32_t deviceId, net_addr_t deviceIp, void *&rdmaHandle)
{
    if (storedRdmaHandle_ != nullptr) {
        BM_LOG_INFO(deviceId << " ra rdev already initialized.");
        rdmaHandle = storedRdmaHandle_;
        return true;
    }

    HccpRdevInitInfo info{};
    HccpRdev rdev{};

    info.mode = NETWORK_OFFLINE;
    info.notifyType = NOTIFY;
    info.enabled2mbLite = true;
    rdev.phyId = deviceId;
    rdev.family = (deviceIp.type == IpV4) ? AF_INET : AF_INET6;
    if (deviceIp.type == IpV4) {
        rdev.localIp.addr = deviceIp.ip.ipv4;
    } else if (deviceIp.type == IpV6) {
        rdev.localIp.addr6 = deviceIp.ip.ipv6;
    }
    BM_LOG_DEBUG(deviceId << " RaRdevInitV2, info=" << info << "rdev=" << rdev);
    auto ret = DlHccpApi::RaRdevInitV2(info, rdev, rdmaHandle);
    if (ret != 0) {
        BM_LOG_ERROR(deviceId << " Hccp Init RDev failed: " << ret);
        return false;
    }

    storedRdmaHandle_ = rdmaHandle;
    BM_LOG_INFO(deviceId << " initialize RDev success.");
    return true;
}

void RdmaTransportManager::ClearAllRegisterMRs()
{
    for (auto it = registerMRS_.begin(); it != registerMRS_.end(); ++it) {
        auto ret = DlHccpApi::RaDeregisterMR(rdmaHandle_, it->second.mrHandle);
        if (ret != 0) {
            BM_LOG_WARN(rankId_ << " Unregister:" << (void *)(ptrdiff_t)it->first << " : " << it->second << " failed: " << ret);
        }
    }
    registerMRS_.clear();
}

int RdmaTransportManager::CheckPrepareOptions(const ock::mf::transport::HybmTransPrepareOptions &options)
{
    if (role_ != HYBM_ROLE_PEER) {
        BM_LOG_INFO(rankId_ << " transport role: " << role_ << " check options passed.");
        return BM_OK;
    }

    if (options.options.size() > rankCount_) {
        BM_LOG_ERROR(rankId_ << " options size():" << options.options.size() << " larger than rank count: " << rankCount_);
        return BM_INVALID_PARAM;
    }

    if (options.options.find(rankId_) == options.options.end()) {
        BM_LOG_ERROR(rankId_ << " options not contains self rankId: " << rankId_);
        return BM_INVALID_PARAM;
    }

    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        if (it->first >= rankCount_) {
            BM_LOG_ERROR(rankId_ << " input options of nics contains rankId:" << it->first << ", rank count: " << rankCount_);
            return BM_INVALID_PARAM;
        }
    }

    return BM_OK;
}
}
}
}
}
