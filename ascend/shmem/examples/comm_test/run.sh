#!/bin/bash
# MTE vs SDMA cross-card bandwidth benchmark
# Usage: ./run.sh <npu0> <npu1>
# Example: ./run.sh 0 1

set -e

NPU0=${1:-0}
NPU1=${2:-1}
IPPORT="tcp://127.0.0.1:8899"
BIN="../../build/bin/comm_test"

if [ ! -f "$BIN" ]; then
    echo "Error: $BIN not found. Build the project first."
    exit 1
fi

echo "Testing NPU ${NPU0} <-> NPU ${NPU1}"

$BIN 2 0 $IPPORT $NPU0 &
$BIN 2 1 $IPPORT $NPU1 &
wait
