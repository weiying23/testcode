#!/bin/bash
# MTE vs SDMA cross-card bandwidth benchmark
# Usage: ./run.sh <npu0> <npu1>   OR   ./run.sh <npu0>,<npu1>
# Example: ./run.sh 6 7
#          ./run.sh 6,7

set -e

# parse both "6,7" and "6 7" formats
if [[ "$1" == *","* ]]; then
    NPU0=$(echo "$1" | cut -d',' -f1)
    NPU1=$(echo "$1" | cut -d',' -f2)
else
    NPU0=${1:-0}
    NPU1=${2:-1}
fi

MTE_PORT="tcp://127.0.0.1:8899"
SDMA_PORT="tcp://127.0.0.1:8900"
BIN="../../build/bin/comm_test"

if [ ! -f "$BIN" ]; then
    echo "Error: $BIN not found. Build the project first."
    exit 1
fi

echo "Testing NPU ${NPU0} <-> NPU ${NPU1}"

echo "[1/2] Running MTE..."
$BIN 2 0 $MTE_PORT $NPU0 mte &
$BIN 2 1 $MTE_PORT $NPU1 mte &
wait

echo "[2/2] Running SDMA..."
$BIN 2 0 $SDMA_PORT $NPU0 sdma &
$BIN 2 1 $SDMA_PORT $NPU1 sdma &
wait
