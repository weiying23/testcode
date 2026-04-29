#!/bin/bash
set -e
# usage: bash run.sh [comm_type] [data_type] [test_start_line] [test_collect_rows] [device_list]
# eg. bash run.sh 0 1 0,1      # 在 0/1 卡上运行 all reduce 精度测试, 数据类型 FP16, rank size = 2
# eg. bash run.sh 1 27 0 10 4,5,6,7  # 在 4/5/6/7 卡上运行 allgather matmul 性能测试, 数据类型 BF16, 从test_shapes.csv的第0个shape开始, 每10个shape采集一次msprof数据, rank size = 4

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))
DATA_PATH=${PROJECT_ROOT}/examples/dynamic_tiling/output
TILING_UTILS_PATH=${PROJECT_ROOT}/examples/dynamic_tiling/utils
UTILS_PATH=${PROJECT_ROOT}/examples/utils
PARENT_PATH=${PROJECT_ROOT}/examples/dynamic_tiling/

# eg. 精度测试WARM_UP_TIMES设置成0, PERF_TEST_CYCLE_TIMES成1
# eg. 性能测试WARM_UP_TIMES设置成10, PERF_TEST_CYCLE_TIMES成3
export WARM_UP_TIMES=10
export PERF_TEST_CYCLE_TIMES=3
export SEARCH_PARAMS=0

CSV_FILE="${SCRIPT_DIR}/test_shapes.csv"

NUM_ARGS=$#

case "$NUM_ARGS" in
  3)
    COMM_TYPE="$1"
    DATA_TYPE="$2"
    TEST_START_LINE=0
    TEST_COLLECT_ROWS=1
    DEVICE_ID_STR="$3"
    TEST_TYPE=0
    ;;

  5)
    COMM_TYPE="$1"
    DATA_TYPE="$2"
    TEST_START_LINE="$3"
    TEST_COLLECT_ROWS="$4"
    DEVICE_ID_STR="$5"
    TEST_TYPE=1
    ;;

  *)
    echo "Error: invalid number of arguments: $NUM_ARGS"
    usage
    return 1
    ;;
esac

IFS=',' read -ra DEVICE_ID_LIST <<< "$DEVICE_ID_STR"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ $RANK_SIZE -gt 8 ]; then
    echo "Rank size is illegal"
    exit 1
fi

cd ${PROJECT_ROOT}/examples/dynamic_tiling/
EXEC_BIN=${PROJECT_ROOT}/build/bin/dynamic_tiling

if [ "$TEST_START_LINE" = "0" ]; then
    rm -rf output
    mkdir -p output
    mkdir -p output/tiling
fi

IDX=0

if [ "$TEST_TYPE" = "0" ]; then
    tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N TA TB; do
        if [ "$IDX" -lt "$TEST_START_LINE" ]; then
            (( IDX+=1 ))
            continue
        fi

        echo "Processing test case: M=${M}, K=${K}, N=${N}, TransA=${TA}, TransB=${TB}"

        rm -rf output/*.bin
        python3 ${UTILS_PATH}/gen_data.py ${COMM_TYPE} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} ${DATA_PATH}

        # Set necessary parameters
        IPPORT="tcp://127.0.0.1:27008"

        # Start Process
        for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
            APP="$EXEC_BIN $COMM_TYPE $DATA_TYPE $RANK_SIZE $idx $IPPORT $M $N $K $TEST_START_LINE $TEST_COLLECT_ROWS $PARENT_PATH $CSV_FILE $DEVICE_ID_STR $DATA_PATH"
            ${APP}&
        done

        # Wait until all process exit
        wait

        if [ "$COMM_TYPE" = "1" ]; then
            python3 ${UTILS_PATH}/verify_result.py ./output/output.bin ./output/golden.bin ${DATA_TYPE} ${M} ${N} ${K}
        elif [ "$COMM_TYPE" = "4" ]; then
            python3 ${UTILS_PATH}/verify_result.py ./output/output.bin ./output/golden.bin ${DATA_TYPE} ${M} ${N} ${K}
            python3 ${UTILS_PATH}/verify_result.py ./output/output_gather_a.bin ./output/gather_a.bin ${DATA_TYPE} ${M} ${N} ${K}
        else
            python3 ${UTILS_PATH}/verify_result.py ./output/output.bin ./output/golden.bin ${DATA_TYPE} ${M} ${N} $((K * RANK_SIZE))
        fi

        ret=$?
        [[ $ret -eq 0 ]] || exit 1

        (( TEST_START_LINE+=TEST_COLLECT_ROWS ))
        (( IDX+=1 ))
    done
else
    tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N TA TB; do
        if [ "$IDX" -lt "$TEST_START_LINE" ]; then
            (( IDX+=1 ))
            continue
        fi

        echo "Processing test case: M=${M}, K=${K}, N=${N}, TransA=${TA}, TransB=${TB}"

        # Set necessary parameters
        IPPORT="tcp://127.0.0.1:27009"

        OUTPUT_PATH="./output/msprof/start_line${IDX}_run_rows${TEST_COLLECT_ROWS}/"

        # Start Process
        for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
            APP="$EXEC_BIN $COMM_TYPE $DATA_TYPE $RANK_SIZE $idx $IPPORT $M $N $K $TEST_START_LINE $TEST_COLLECT_ROWS $PARENT_PATH $CSV_FILE $DEVICE_ID_STR"
            msprof --application="${APP}" --output="${OUTPUT_PATH}"&
        done

        # Wait until all process exit
        wait

        python3 ${TILING_UTILS_PATH}/process_data.py "${OUTPUT_PATH}"

        (( TEST_START_LINE+=TEST_COLLECT_ROWS ))
        (( IDX+=1 ))
    done
    python3 ${TILING_UTILS_PATH}/get_best_result.py "${CSV_FILE}"
fi

cd ${CURRENT_DIR}