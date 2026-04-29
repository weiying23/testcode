#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))

cd ${PROJECT_ROOT}/examples/kv_shuffle/
EXEC_BIN=${PROJECT_ROOT}/build/bin/kv_shuffle

# Set necessary parameters
IPPORT="tcp://127.0.0.1:27010"
RANK_SIZE=$1

rm -rf scripts/output/*.bin
python3 scripts/golden.py $RANK_SIZE

# Start Process
for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
    APP="$EXEC_BIN $RANK_SIZE $idx $IPPORT"
    ${APP}&
done

# Wait until all process exit
wait

# Verify output
DATA_PATH=${PROJECT_ROOT}/examples/kv_shuffle/scripts/output
for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
    python3 scripts/result_compare.py ${DATA_PATH}/k_cache_output_rank_${idx}.bin ${DATA_PATH}/k_cache_golden_rank_${idx}.bin
    [[ $? -eq 0 ]] || exit 1
    python3 scripts/result_compare.py ${DATA_PATH}/v_cache_output_rank_${idx}.bin ${DATA_PATH}/v_cache_golden_rank_${idx}.bin
    [[ $? -eq 0 ]] || exit 1
done

cd -
