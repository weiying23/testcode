#!/bin/bash
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Comm Benchmark运行脚本
#

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))

# 参数解析
IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
RANK_SIZE=${#DEVICE_ID_LIST[@]}

if [ $RANK_SIZE -gt 8 ]; then
    echo "Error: Rank size > 8 is not supported"
    exit 1
fi

if [ $RANK_SIZE -lt 2 ]; then
    echo "Error: At least 2 ranks required"
    exit 1
fi

# 设置环境
source ${PROJECT_ROOT}/install/set_env.sh

IPPORT="tcp://127.0.0.1:8789"
EXEC_BIN=${PROJECT_ROOT}/build/bin/comm_benchmark

# 创建结果目录
mkdir -p ${SCRIPT_DIR}/../results

echo "========================================"
echo "Comm Benchmark"
echo "========================================"
echo "Rank Size: ${RANK_SIZE}"
echo "Devices: ${DEVICE_ID_LIST[@]}"
echo "IP Port: ${IPPORT}"
echo "========================================"

# 启动多个进程
for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
    device_id=${DEVICE_ID_LIST[$idx]}
    echo "Starting Rank ${idx} on Device ${device_id}"
    ${EXEC_BIN} ${RANK_SIZE} ${idx} ${IPPORT} ${RANK_SIZE} 0 ${device_id} &
done

# 等待所有进程完成
wait

echo "========================================"
echo "Benchmark Complete!"
echo "Results saved to results/ directory"
echo "========================================"

# 显示结果摘要
if [ -f "${SCRIPT_DIR}/../results/latency_results.csv" ]; then
    echo ""
    echo "Latency Results:"
    cat ${SCRIPT_DIR}/../results/latency_results.csv
fi

if [ -f "${SCRIPT_DIR}/../results/bandwidth_results.csv" ]; then
    echo ""
    echo "Bandwidth Results:"
    cat ${SCRIPT_DIR}/../results/bandwidth_results.csv
fi

cd ${CURRENT_DIR}