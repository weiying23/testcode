#!/bin/bash
# Engine Benchmark运行脚本
# 使用方式: bash run.sh <npu_ids> [options]
#   bash run.sh 0,1 -e mte_inter
#   bash run.sh 4,5 -e sdma_inter -m put

set -e

# ========== 默认参数 ==========
NPU_IDS="0,1"
IPPORT="tcp://127.0.0.1:8898"
ENGINE="mte_inter"
MODE="put"
DTYPE="float"
BLOCK_SIZE=32
UB_SIZE=16

# ========== 解析第一个参数（NPU ID列表）==========
if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    NPU_IDS="$1"
    shift
fi

# 解析 NPU ID
NPU0=$(echo $NPU_IDS | cut -d',' -f1)
NPU1=$(echo $NPU_IDS | cut -d',' -f2)

# ========== 解析其他参数 ==========
while [[ $# -gt 0 ]]; do
    case $1 in
        -ipport|--ipport)
            IPPORT="$2"
            shift 2
            ;;
        -e|--engine)
            ENGINE="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -dtype|--dtype)
            DTYPE="$2"
            shift 2
            ;;
        -b|--block-size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --ub-size)
            UB_SIZE="$2"
            shift 2
            ;;
        -all|--all)
            ENGINE="all"
            shift
            ;;
        -h|--help)
            echo "Usage: bash run.sh <npu_ids> [options]"
            echo "  npu_ids: NPU ID列表，如 0,1 或 4,5 (default: 0,1)"
            echo ""
            echo "Options:"
            echo "  -ipport <ip>      IP and port (default: tcp://127.0.0.1:8898)"
            echo "  -e <type>         Engine: mte_inter|sdma_inter|all (default: mte_inter)"
            echo "  -m <mode>         Mode: put|get (default: put)"
            echo "  -dtype <type>     Data type: float|int32|int64 (default: float)"
            echo "  -b <num>          Block size (default: 32)"
            echo "  --ub-size <num>   UB size KB (default: 16)"
            echo "  -all              Test all engines"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ========== 环境检查 ==========
if [ ! -f "./build/bin/comm_test" ]; then
    echo "Error: Binary not found. Please compile first."
    echo "Run: bash scripts/build.sh -examples"
    exit 1
fi

mkdir -p output
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
export SHMEM_UID_SESSION_ID=${IPPORT#tcp://}

echo "=========================================="
echo "NPU: ${NPU0}, ${NPU1}"
echo "Engine: ${ENGINE}, Mode: ${MODE}"
echo "=========================================="

if [ "$ENGINE" == "all" ]; then
    echo "===== MTE Inter-Card ====="
    ./build/bin/comm_test --pe-id 0 --pes 2 --ipport ${IPPORT} \
        -D ${NPU0} --engine mte_inter -m ${MODE} -dtype ${DTYPE} \
        -b ${BLOCK_SIZE} --ub-size ${UB_SIZE} &

    ./build/bin/comm_test --pe-id 1 --pes 2 --ipport ${IPPORT} \
        -D ${NPU1} --engine mte_inter -m ${MODE} -dtype ${DTYPE} \
        -b ${BLOCK_SIZE} --ub-size ${UB_SIZE} &
    wait

    echo "===== SDMA Inter-Card ====="
    ./build/bin/comm_test --pe-id 0 --pes 2 --ipport ${IPPORT} \
        -D ${NPU0} --engine sdma_inter -m ${MODE} -dtype ${DTYPE} \
        -b ${BLOCK_SIZE} --ub-size ${UB_SIZE} &

    ./build/bin/comm_test --pe-id 1 --pes 2 --ipport ${IPPORT} \
        -D ${NPU1} --engine sdma_inter -m ${MODE} -dtype ${DTYPE} \
        -b ${BLOCK_SIZE} --ub-size ${UB_SIZE} &
    wait
else
    ./build/bin/comm_test --pe-id 0 --pes 2 --ipport ${IPPORT} \
        -D ${NPU0} --engine ${ENGINE} -m ${MODE} -dtype ${DTYPE} \
        -b ${BLOCK_SIZE} --ub-size ${UB_SIZE} &

    ./build/bin/comm_test --pe-id 1 --pes 2 --ipport ${IPPORT} \
        -D ${NPU1} --engine ${ENGINE} -m ${MODE} -dtype ${DTYPE} \
        -b ${BLOCK_SIZE} --ub-size ${UB_SIZE} &
    wait
fi

echo ""
echo "Results:"
ls -la output/*.csv 2>/dev/null || echo "No CSV files"
echo "[SUCCESS]"