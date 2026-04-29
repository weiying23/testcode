#!/bin/bash
#bash scripts/build.sh
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))
EXAMPLE_DIR=${SCRIPT_DIR}/../

# Default Args
RANK_SIZE="2"
IPPORT="tcp://127.0.0.1:18878"
FIRST_NPU="0"

# Args Parse
while [[ $# -gt 0 ]]; do
    case "$1" in
        -ranks)
            if [ -n "$2" ]; then
                RANK_SIZE="$2"
                shift 2
            else
                echo "Error: -ranks requires a value."
                exit 1
            fi
            ;;
        -fnpu)
            if [ -n "$2" ]; then
                FIRST_NPU="$2"
                shift 2
            else
                echo "Error: -fnpu requires a value."
                exit 1
            fi
            ;;
        -ipport)
            if [ -n "$2" ]; then
                IPPORT="$2"
                shift 2
            else
                echo "Error: -ipport requires a value."
                exit 1
            fi
            ;;
        -M)
            if [ -n "$2" ]; then
                M="$2"
                shift 2
            else
                echo "Error: -M requires a value."
                exit 1
            fi
            ;;
        -K)
            if [ -n "$2" ]; then
                K="$2"
                shift 2
            else
                echo "Error: -K requires a value."
                exit 1
            fi
            ;;
        -N)
            if [ -n "$2" ]; then
                N="$2"
                shift 2
            else
                echo "Error: -N requires a value."
                exit 1
            fi
            ;;
        -expertPerRank)
            if [ -n "$2" ]; then
                expertPerRank="$2"
                shift 2
            else
                echo "Error: -expertPerRank requires a value."
                exit 1
            fi
            ;;
        -dataType)
            if [ -n "$2" ]; then
                dataType="$2"
                shift 2
            else
                echo "Error: -dataType requires a value."
                exit 1
            fi
            ;;
        -weightNz)
            if [ -n "$2" ]; then
                weightNz="$2"
                shift 2
            else
                echo "Error: -weightNz requires a value."
                exit 1
            fi
            ;;
        -transB)
            if [ -n "$2" ]; then
                transB="$2"
                shift 2
            else
                echo "Error: -transB requires a value."
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1."
            exit 1
            ;;
    esac
done

echo "DATA_DIR: $DATA_DIR"
EXEC_BIN=${PROJECT_ROOT}/build/bin/dispatch_gmm_combine

cd ${PROJECT_ROOT}/examples/dispatch_gmm_combine/
python3 utils/gen_data.py
if [[ $? -ne 0 ]]; then
    echo "gen data failed."
    exit 1
fi

echo "Test Case, M: ${M}, K: ${K}, N: ${N}, expertPerRank: ${expertPerRank}"
export LD_LIBRARY_PATH=${PROJECT_ROOT}/install/shmem/lib:${ASCEND_HOME_PATH}/lib64:${PROJECT_ROOT}/install/memfabric_hybrid/lib:$LD_LIBRARY_PATH
for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
    export INPUT_PATH=${EXAMPLE_DIR}/utils/test_data/
    ${EXEC_BIN} "$RANK_SIZE" "$idx" "$IPPORT" "$FIRST_NPU" "$M" "$K" "$N" "$expertPerRank" "$dataType" "$weightNz" "$transB" &
done

# Wait until all process exit
wait

cd ${CURRENT_DIR}
python ${CURRENT_DIR}/utils/check_result.py --rank_size $RANK_SIZE --dataType $dataType --m $M --k $K --n $N --expert_per_rank $expertPerRank --EP $RANK_SIZE
exit $?