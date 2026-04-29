/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cctype>
#include <exception>
#include <fstream>
#include <utility>

#include "shmemi_host_common.h"
#include "topo_reader.h"

namespace shm {
namespace transport {

bool TopoReader::LoadJsonFile(const std::string& path, nlohmann::json& data)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        SHM_LOG_ERROR("Open json file failed, the path is " << path);
        return false;
    }
    try {
        input >> data;
        return true;
    } catch (const std::exception& ex) {
        SHM_LOG_ERROR("Parse json failed, the path is " << path << ", error: " << ex.what());
        return false;
    }
}

bool TopoReader::ParseRootInfo(RootInfo& out)
{
    nlohmann::json rootInfoJson;
    if (!LoadJsonFile(ROOTINFO_PATH, rootInfoJson)) {
        SHM_LOG_ERROR("Failed to load rootinfo json from " << ROOTINFO_PATH);
        return false;
    }

    if (!rootInfoJson.contains("topo_file_path") || !rootInfoJson["topo_file_path"].is_string()) {
        SHM_LOG_ERROR("Topo file path not found in rootinfo.");
        return false;
    }
    out.topo_file_path = rootInfoJson["topo_file_path"].get<std::string>();

    if (!rootInfoJson.contains("rank_list") || !rootInfoJson["rank_list"].is_array()) {
        SHM_LOG_ERROR("Rootinfo: rank_list not found.");
        return false;
    }
    const auto& rankListJson = rootInfoJson["rank_list"];
    out.rank_list.reserve(rankListJson.size());
    for (const auto& rankJson : rankListJson) {
        if (!rankJson.contains("device_id") || !rankJson.contains("local_id")) {
            continue;
        }

        RankInfo rankInfo;
        if (!ParseUint(rankJson["device_id"], rankInfo.device_id) ||
            !ParseUint(rankJson["local_id"], rankInfo.local_id)) {
            SHM_LOG_ERROR("Rootinfo: invalid device_id or local_id entry.");
            return false;
        }

        if (out.rank_list.empty()) {
            out.device_id_offset = rankInfo.device_id;
        }

        out.rank_list.push_back(rankInfo);
        out.deviceLocalIdMap[rankInfo.device_id] = rankInfo.local_id;

        // Build portEidMap: localId -> {port -> eidIndex}
        if (!rankJson.contains("level_list") || !rankJson["level_list"].is_array() || rankJson["level_list"].empty()) {
            SHM_LOG_WARN("Rootinfo: missing level_list for localId " << rankInfo.local_id);
            continue;
        }
        const auto& levelListJson = rankJson["level_list"];
        bool foundTopoLevel = false;
        for (const auto& levelJson : levelListJson) {
            if (levelJson.contains("net_type") && levelJson["net_type"].is_string() &&
                levelJson["net_type"].get<std::string>() == "TOPO_FILE_DESC") {
                foundTopoLevel = true;
                if (!levelJson.contains("rank_addr_list") || !levelJson["rank_addr_list"].is_array()) {
                    SHM_LOG_WARN("Rootinfo: missing rank_addr_list for localId " << rankInfo.local_id);
                    break;
                }
                const auto& rankAddrListJson = levelJson["rank_addr_list"];
                const uint32_t eidCount = static_cast<uint32_t>(rankAddrListJson.size());
                if (out.eidCount == 0) {
                    out.eidCount = eidCount;
                }
                uint32_t eidIndex = 0;
                for (const auto& rankAddrJson : rankAddrListJson) {
                    std::array<uint8_t, HCCP_EID_RAW_SIZE> rawAddr{};
                    if (rankAddrJson.contains("addr") && ParseEidRaw(rankAddrJson["addr"], rawAddr)) {
                        out.eidAddrMap[rankInfo.local_id][eidIndex] = rawAddr;
                    }
                    if (rankAddrJson.contains("ports") && rankAddrJson["ports"].is_array()) {
                        for (const auto& port : rankAddrJson["ports"]) {
                            if (port.is_string()) {
                                out.portEidMap[rankInfo.local_id][port.get<std::string>()] = eidIndex;
                            }
                        }
                    }
                    ++eidIndex;
                }
                break;
            }
        }
        if (!foundTopoLevel) {
            SHM_LOG_WARN("Rootinfo: missing topo level for localId " << rankInfo.local_id);
            continue;
        }
    }

    if (out.rank_list.empty()) {
        SHM_LOG_ERROR("Rootinfo: no valid rank entries parsed.");
        return false;
    }
    SHM_LOG_INFO("Temporary rootinfo device_id offset inferred as " << out.device_id_offset);
    return true;
}

bool TopoReader::ParseTopoInfo(const std::string& path, TopoInfo& out)
{
    std::ifstream topoFile(path);
    if (!topoFile.is_open()) {
        SHM_LOG_ERROR("Failed to open topo file: " << path);
        return false;
    }
    nlohmann::json topoInfoJson;
    try {
        topoFile >> topoInfoJson;
    } catch (const nlohmann::json::exception& e) {
        SHM_LOG_ERROR("Topo parse failed: " << e.what());
        return false;
    }

    if (!topoInfoJson.contains("edge_list") || !topoInfoJson["edge_list"].is_array()) {
        SHM_LOG_ERROR("Topo parse failed: edge_list not found.");
        return false;
    }
    const auto& edgeListJson = topoInfoJson["edge_list"];
    out.edge_list.reserve(edgeListJson.size());
    for (const auto& edgeObj : edgeListJson) {
        if (!edgeObj.contains("local_a") || !edgeObj.contains("local_b")) {
            continue;
        }
        TopoEdge edge;
        if (!ParseUint(edgeObj["local_a"], edge.local_a) || !ParseUint(edgeObj["local_b"], edge.local_b)) {
            SHM_LOG_ERROR("Topo parse failed: local_a or local_b is invalid.");
            return false;
        }

        if (edgeObj.contains("local_a_ports") && edgeObj["local_a_ports"].is_array()) {
            edge.local_a_ports = edgeObj["local_a_ports"].get<std::vector<std::string>>();
        }
        if (edgeObj.contains("local_b_ports") && edgeObj["local_b_ports"].is_array()) {
            edge.local_b_ports = edgeObj["local_b_ports"].get<std::vector<std::string>>();
        }
        out.edge_list.push_back(std::move(edge));
    }

    if (out.edge_list.empty()) {
        SHM_LOG_ERROR("Topo parse failed: no valid edge entries parsed.");
        return false;
    }
    return true;
}

// Resolves the local route used to connect from myLocalId to peerLocalId.
// The remote EID index must be provided by the peer, because rootinfo only contains
// port -- EID
bool TopoReader::GetLocalEidRouteForPeer(
    const RootInfo& root, const TopoInfo& topo, uint32_t myLocalId, uint32_t peerLocalId, uint32_t& localEidIndex,
    std::array<uint8_t, HCCP_EID_RAW_SIZE>& localEidRaw)
{
    std::string localPort;
    for (const auto& edge : topo.edge_list) {
        if (edge.local_a == myLocalId && edge.local_b == peerLocalId && !edge.local_a_ports.empty()) {
            localPort = edge.local_a_ports[0];
            break;
        }
        if (edge.local_b == myLocalId && edge.local_a == peerLocalId && !edge.local_b_ports.empty()) {
            localPort = edge.local_b_ports[0];
            break;
        }
    }
    if (localPort.empty()) {
        SHM_LOG_ERROR(
            "Failed to get local eid route, no usable edge between localId " << myLocalId << " and " << peerLocalId);
        return false;
    }

    const auto localRankItem = root.portEidMap.find(myLocalId);
    if (localRankItem == root.portEidMap.end()) {
        SHM_LOG_ERROR("Failed to get local eid index, localId " << myLocalId << " not found in portEidMap");
        return false;
    }
    const auto localPortItem = localRankItem->second.find(localPort);
    if (localPortItem == localRankItem->second.end()) {
        SHM_LOG_ERROR("Failed to get local eid index, port " << localPort << " not found for localId " << myLocalId);
        return false;
    }
    localEidIndex = localPortItem->second;

    const auto localIt = root.eidAddrMap.find(myLocalId);
    if (localIt == root.eidAddrMap.end()) {
        SHM_LOG_ERROR("Failed to get eid route, localId " << myLocalId << " not found in eidAddrMap");
        return false;
    }
    const auto eidIt = localIt->second.find(localEidIndex);
    if (eidIt == localIt->second.end()) {
        SHM_LOG_ERROR("Failed to get eid route, eidIndex " << localEidIndex << " not found for localId " << myLocalId);
        return false;
    }
    localEidRaw = eidIt->second;
    SHM_LOG_INFO(
        "Get local eid route success, myLocalId: " << myLocalId << ", peerLocalId: " << peerLocalId << ", localPort: "
                                                   << localPort << ", localEidIndex: " << localEidIndex);
    return true;
}

bool TopoReader::GetLocalId(const RootInfo& root, uint32_t deviceId, uint32_t& localId)
{
    const auto it = root.deviceLocalIdMap.find(deviceId);
    if (it != root.deviceLocalIdMap.end()) {
        localId = it->second;
        return true;
    }
    SHM_LOG_ERROR("RootInfo is invalid or incomplete, failed to find localId for deviceId " << deviceId);
    return false;
}

bool TopoReader::GetLocalIdWithDeviceIdOffset(const RootInfo& root, uint32_t deviceId, uint32_t& localId)
{
    const uint32_t rootInfoDeviceId = deviceId + root.device_id_offset;
    const auto it = root.deviceLocalIdMap.find(rootInfoDeviceId);
    if (it != root.deviceLocalIdMap.end()) {
        localId = it->second;
        return true;
    }

    SHM_LOG_ERROR(
        "Temporary rootinfo offset lookup failed, local deviceId "
        << deviceId << ", device_id_offset " << root.device_id_offset << ", rootinfo deviceId " << rootInfoDeviceId);
    return false;
}

uint32_t TopoReader::GetDeviceIdOffset(const RootInfo& root) {
    return root.device_id_offset;
}

bool TopoReader::GetEidCount(const RootInfo& root, uint32_t& count)
{
    if (root.eidCount > 0) {
        count = root.eidCount;
        return true;
    }

    SHM_LOG_ERROR("RootInfo is invalid or incomplete, failed to find a valid eid count.");
    return false;
}

bool TopoReader::ParseEidRaw(const nlohmann::json& jsonValue, std::array<uint8_t, HCCP_EID_RAW_SIZE>& raw)
{
    raw.fill(0);
    if (jsonValue.is_array()) {
        if (jsonValue.size() != raw.size()) {
            SHM_LOG_ERROR("Rootinfo addr array size is invalid: " << jsonValue.size());
            return false;
        }
        for (size_t i = 0; i < raw.size(); ++i) {
            if (!jsonValue[i].is_number_unsigned() && !jsonValue[i].is_number_integer()) {
                SHM_LOG_ERROR("Rootinfo addr array contains non-numeric value at index " << i);
                return false;
            }
            auto value = jsonValue[i].get<int32_t>();
            if (value < 0 || value > 0xff) {
                SHM_LOG_ERROR("Rootinfo addr array value out of byte range at index " << i << ", value " << value);
                return false;
            }
            raw[i] = static_cast<uint8_t>(value);
        }
        return true;
    }

    if (!jsonValue.is_string()) {
        SHM_LOG_ERROR("Rootinfo addr format is unsupported.");
        return false;
    }

    const std::string& jsonString = jsonValue.get_ref<const std::string&>();
    std::string normalized;
    normalized.reserve(jsonString.size());
    for (const char ch : jsonString) {
        if (std::isxdigit(static_cast<unsigned char>(ch))) {
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }

    if (normalized.size() != HCCP_EID_HEX_SIZE) {
        SHM_LOG_ERROR("Rootinfo addr hex length is invalid: " << normalized.size());
        return false;
    }

    try {
        for (size_t i = 0; i < raw.size(); ++i) {
            const std::string byteStr = normalized.substr(i * 2, 2);
            raw[i] = static_cast<uint8_t>(std::stoul(byteStr, nullptr, 16));
        }
    } catch (const std::exception& ex) {
        SHM_LOG_ERROR("Failed to parse rootinfo addr hex string, error: " << ex.what());
        return false;
    }
    return true;
}

bool TopoReader::ParseUint(const nlohmann::json& jsonValue, uint32_t& value)
{
    try {
        if (jsonValue.is_string()) {
            value = static_cast<uint32_t>(std::stoul(jsonValue.get<std::string>()));
            return true;
        }
        value = jsonValue.get<uint32_t>();
        return true;
    } catch (const std::exception& ex) {
        SHM_LOG_ERROR("Failed to parse uint value, error: " << ex.what());
        return false;
    }
}

} // namespace transport
} // namespace shm
