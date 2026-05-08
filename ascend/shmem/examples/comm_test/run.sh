#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Engine Benchmark运行脚本

set -e

# ========== 默认参数 ==========
PES=2
GNPUS=2
FNPU=0
IPPORT="tcp://127.0.0.1:8898"
ENGINE="mte_inter"
MODE="put"
DTYPE="float"
BLOCK_SIZE=32
UB_SIZE=16

# ========== 解析参数 ==========
while [[ $# -gt 0 ]]; do
    case $1 in
        -pes|--pes)
            PES="$2"
            shift 2
            ;;
        -gnpus|--gnpus)
            GNPUS="$2"
            shift 2
            ;;
        -fnpu|--fnpu)
            FNPU="$2"
            shift 2
            ;;
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
        -d|--dtype)
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
            echo "Usage: bash run.sh [options]"
            echo "Options:"
            echo "  -pes, --pes <num>       Number of PEs (default: 2)"
            echo "  -gnpus, --gnpus <num>   Number of NPUs (default: 2)"
            echo "  -fnpu, --fnpu <num>     First NPU ID (default: 0)"
            echo "  -ipport, --ipport <ip>  IP and port (default: tcp://127.0.0.1:8898)"
            echo "  -e, --engine <type>     Engine type: mte_intra|mte_inter|sdma_inter|all (default: mte_inter)"
            echo "  -m, --mode <mode>       Test mode: put|get|bi_put|bi_get (default: put)"
            echo "  -d, --dtype <type>      Data type: float|int32|int64 (default: float)"
            echo "  -b, --block-size <num>  Block size (default: 32)"
            echo "  --ub-size <num>         UB size in KB (default: 16)"
            echo "  -all, --all             Test all engines"
            echo "  -h, --help              Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ========== 环境检查 ==========
if [ ! -f "./build/bin/engine_benchmark" ]; then
    echo "Error: Binary not found. Please compile first."
    echo "Run: bash scripts/build.sh -examples"
    exit 1
fi

# ========== 创建输出目录 ==========
mkdir -p output

# ========== 设置环境变量 ==========
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
export SHMEM_UID_SESSION_ID=${IPPORT#tcp://}

# ========== 运行测试 ==========
echo "=========================================="
echo "Starting Engine Benchmark"
echo "=========================================="
echo "PES: ${PES}"
echo "GNPUS: ${GNPUS}"
echo "FNPU: ${FNPU}"
echo "IPPORT: ${IPPORT}"
echo "ENGINE: ${ENGINE}"
echo "MODE: ${MODE}"
echo "DTYPE: ${DTYPE}"
echo "BLOCK_SIZE: ${BLOCK_SIZE}"
echo "UB_SIZE: ${UB_SIZE}"
echo "=========================================="

# 根据引擎类型选择测试模式
if [ "$ENGINE" == "all" ]; then
    # 测试所有引擎
    echo "Testing all engines..."

    # MTE Inter-Card
    echo ""
    echo "===== Testing MTE Inter-Card ====="
    ./build/bin/engine_benchmark \
        --pes ${PES} --pe-id 0 --ipport ${IPPORT} \
        --gnpus ${GNPUS} --fnpu ${FNPU} \
        --engine mte_inter --mode ${MODE} --dtype ${DTYPE} \
        --block-size ${BLOCK_SIZE} --ub-size ${UB_SIZE} &

    ./build/bin/engine_benchmark \
        --pes ${PES} --pe-id 1 --ipport ${IPPORT} \
        --gnpus ${GNPUS} --fnpu ${FNPU} \
        --engine mte_inter --mode ${MODE} --dtype ${DTYPE} \
        --block-size ${BLOCK_SIZE} --ub-size ${UB_SIZE} &

    wait
    echo "MTE Inter-Card test completed"

    # SDMA Inter-Card
    echo ""
    echo "===== Testing SDMA Inter-Card ====="
    ./build/bin/engine_benchmark \
        --pes ${PES} --pe-id 0 --ipport ${IPPORT} \
        --gnpus ${GNPUS} --fnpu ${FNPU} \
        --engine sdma_inter --mode ${MODE} --dtype ${DTYPE} \
        --block-size ${BLOCK_SIZE} --ub-size ${UB_SIZE} &

    ./build/bin/engine_benchmark \
        --pes ${PES} --pe-id 1 --ipport ${IPPORT} \
        --gnpus ${GNPUS} --fnpu ${FNPU} \
        --engine sdma_inter --mode ${MODE} --dtype ${DTYPE} \
        --block-size ${BLOCK_SIZE} --ub-size ${UB_SIZE} &

    wait
    echo "SDMA Inter-Card test completed"

else
    # 单引擎测试
    echo ""
    echo "===== Testing ${ENGINE} ====="

    ./build/bin/engine_benchmark \
        --pes ${PES} --pe-id 0 --ipport ${IPPORT} \
        --gnpus ${GNPUS} --fnpu ${FNPU} \
        --engine ${ENGINE} --mode ${MODE} --dtype ${DTYPE} \
        --block-size ${BLOCK_SIZE} --ub-size ${UB_SIZE} &

    ./build/bin/engine_benchmark \
        --pes ${PES} --pe-id 1 --ipport ${IPPORT} \
        --gnpus ${GNPUS} --fnpu ${FNPU} \
        --engine ${ENGINE} --mode ${MODE} --dtype ${DTYPE} \
        --block-size ${BLOCK_SIZE} --ub-size ${UB_SIZE} &

    wait
    echo "${ENGINE} test completed"
fi

echo ""
echo "=========================================="
echo "Benchmark Results Summary"
echo "=========================================="
echo "Results saved to: output/"
ls -la output/*.csv 2>/dev/null || echo "No CSV files generated"
echo ""
echo "[SUCCESS] Engine benchmark completed"