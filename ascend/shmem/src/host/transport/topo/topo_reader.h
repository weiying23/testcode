/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_TRANSPORT_TOPO_READER_H
#define MF_HYBRID_TRANSPORT_TOPO_READER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

namespace shm {
namespace transport {

constexpr std::size_t HCCP_EID_RAW_SIZE = 16;
constexpr std::size_t HCCP_EID_HEX_SIZE = HCCP_EID_RAW_SIZE * 2;

struct RankInfo {
    uint32_t device_id{0};
    uint32_t local_id{0};
};

struct RootInfo {
    std::string topo_file_path;
    std::vector<RankInfo> rank_list;
    // deviceId -> localId
    std::unordered_map<uint32_t, uint32_t> deviceLocalIdMap;
    uint32_t device_id_offset{0};
    // eid size, shared by all devices in one rootinfo
    uint32_t eidCount{0};
    // localId -> port -> eidIndex
    std::unordered_map<uint32_t, std::unordered_map<std::string, uint32_t>> portEidMap;
    // localId -> eidIndex -> eid raw bytes from rootinfo addr
    std::unordered_map<uint32_t, std::map<uint32_t, std::array<uint8_t, HCCP_EID_RAW_SIZE>>> eidAddrMap;
};

struct TopoEdge {
    uint32_t local_a{0};
    std::vector<std::string> local_a_ports;
    uint32_t local_b{0};
    std::vector<std::string> local_b_ports;
};

struct TopoInfo {
    std::vector<TopoEdge> edge_list;
};

class TopoReader {
public:
    static constexpr const char* ROOTINFO_PATH = "/etc/hccl_rootinfo.json";

    static bool LoadJsonFile(const std::string& path, nlohmann::json& data);
    static bool ParseRootInfo(RootInfo& out);
    static bool ParseTopoInfo(const std::string& path, TopoInfo& out);
    static bool GetLocalEidRouteForPeer(
        const RootInfo& root, const TopoInfo& topo, uint32_t myLocalId, uint32_t peerLocalId, uint32_t& localEidIndex,
        std::array<uint8_t, HCCP_EID_RAW_SIZE>& localEidRaw);
    static bool GetLocalId(const RootInfo& root, uint32_t deviceId, uint32_t& localId);
    static bool GetLocalIdWithDeviceIdOffset(const RootInfo& root, uint32_t deviceId, uint32_t& localId);
    static uint32_t GetDeviceIdOffset(const RootInfo& root);

    static bool GetEidCount(const RootInfo& root, uint32_t& count);

private:
    static bool ParseEidRaw(const nlohmann::json& jsonValue, std::array<uint8_t, HCCP_EID_RAW_SIZE>& raw);
    static bool ParseUint(const nlohmann::json& jsonValue, uint32_t& value);
};

} // namespace transport
} // namespace shm

#endif // MF_HYBRID_TRANSPORT_TOPO_READER_H
