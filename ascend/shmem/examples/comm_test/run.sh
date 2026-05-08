#!/bin/bash
# MTE vs SDMA cross-card bandwidth benchmark
# Usage: ./run.sh <npu0> <npu1>   OR   ./run.sh <npu0>,<npu1>

set -e

if [[ "$1" == *","* ]]; then
    NPU0=$(echo "$1" | cut -d',' -f1)
    NPU1=$(echo "$1" | cut -d',' -f2)
else
    NPU0=${1:-0}
    NPU1=${2:-1}
fi

PORT="tcp://127.0.0.1:8899"
BIN="../../build/bin/comm_test"

if [ ! -f "$BIN" ]; then
    echo "Error: $BIN not found. Build the project first."
    exit 1
fi

pkill -f "comm_test" 2>/dev/null || true
sleep 1

echo "Testing NPU ${NPU0} <-> NPU ${NPU1}"

$BIN 2 0 $PORT $NPU0 &
$BIN 2 1 $PORT $NPU1 &
wait
