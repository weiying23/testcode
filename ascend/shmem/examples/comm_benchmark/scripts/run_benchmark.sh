#!/bin/bash
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Comm Benchmark运行脚本
#

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))

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

# 启动延迟参数（秒），让 Rank 0 先启动建立监听
# 可通过第二个参数覆盖，如: ./run_benchmark.sh 1,2 0.5
STARTUP_DELAY=${2:-0.5}

# 设置环境
source ${PROJECT_ROOT}/install/set_env.sh

IPPORT="tcp://127.0.0.1:8789"
EXEC_BIN=${PROJECT_ROOT}/build/bin/comm_benchmark
G_NPUS=8

# 创建结果目录
mkdir -p ${SCRIPT_DIR}/../results

echo "========================================"
echo "Comm Benchmark"
echo "========================================"
echo "Rank Size: ${RANK_SIZE}"
echo "Devices: ${DEVICE_ID_LIST[@]}"
echo "IP Port: ${IPPORT}"
echo "Startup Delay: ${STARTUP_DELAY}s (Rank 0 first, others follow)"
echo "========================================"

# 信号处理：ctrl+c时杀死所有子进程
trap 'echo "Interrupted, killing all processes..."; kill $(jobs -p) 2>/dev/null; exit 1' SIGINT SIGTERM

# 启动 Rank 0（监听端，先启动）
device_id=${DEVICE_ID_LIST[0]}
echo "Starting Rank 0 on Device ${device_id} (listener, starting first)"
${EXEC_BIN} ${RANK_SIZE} 0 ${IPPORT} ${G_NPUS} 0 ${device_id} &
RANK0_PID=$!

# 等待 Rank 0 启动并建立监听
sleep ${STARTUP_DELAY}

# 启动其他 Ranks（连接端）
for (( idx = 1; idx < ${RANK_SIZE}; idx = idx + 1 )); do
    device_id=${DEVICE_ID_LIST[$idx]}
    echo "Starting Rank ${idx} on Device ${device_id} (connector)"
    ${EXEC_BIN} ${RANK_SIZE} ${idx} ${IPPORT} ${G_NPUS} 0 ${device_id} &
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