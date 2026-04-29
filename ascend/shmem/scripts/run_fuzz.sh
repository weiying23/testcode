#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
readonly CURRENT_DIR=$(pwd)
readonly SCRIPT_DIR=$(dirname $(readlink -f "$0"))
readonly PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
readonly BUILD_PATH="$PROJECT_ROOT/build"
readonly COVERAGE_PATH="$BUILD_PATH/coverage"

cd ${PROJECT_ROOT}
rm -rf "$COVERAGE_PATH"

set -e
RANK_SIZE="8"
IPPORT="tcp://127.0.0.1:8666"
FIRST_NPU="0"
FIRST_RANK="0"
GNPU_NUM="8"
if [ -z "${GTEST_FILTER}" ]; then
    TEST_FILTER="*.*"
else
    TEST_FILTER="${GTEST_FILTER}"
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        -ranks)
            if [ -n "$2" ]; then
                RANK_SIZE="$2"
                if [[ "$GNPU_NUM" -gt "$RANK_SIZE" ]]; then
                    GNPU_NUM="$RANK_SIZE"
                    echo "Because RANK_SIZE is less than GNPU_NUM, GNPU_NUM is assigned the value of RANK_SIZE=${RANK_SIZE}."
                fi
                shift 2
            else
                echo "Error: -ranks requires a value. Please add a parameter after -ranks."
                exit 1
            fi
            ;;
        -frank)
            if [ -n "$2" ]; then
                FIRST_RANK="$2"
                shift 2
            else
                echo "Error: -frank requires a value. Please add a parameter after -frank."
                exit 1
            fi
            ;;
        -ipport)
            if [ -n "$2" ]; then
                IPPORT="$2"
                shift 2
            else
                echo "Error: -ipport requires a value. Please add a parameter after -ipport."
                exit 1
            fi
            ;;
        -gnpus)
            if [ -n "$2" ]; then
                GNPU_NUM="$2"
                shift 2
            else
                echo "Error: -gnpus requires a value. Please add a parameter after -gnpus."
                exit 1
            fi
            ;;
        -fnpu)
            if [ -n "$2" ]; then
                FIRST_NPU="$2"
                shift 2
            else
                echo "Error: -fnpu requires a value. Please add a parameter after -fnpu."
                exit 1
            fi
            ;;
        -test_filter)
            if [ -n "$2" ]; then
                TEST_FILTER="*$2*"
                shift 2
            else
                echo "Error: -test_filter requires a value. Please add a parameter after -test_filter."
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1. Please revise."
            exit 1
            ;;
    esac
done
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${PROJECT_ROOT}/install/memfabric_hybrid/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
export SMEM_CONF_STORE_TLS_ENABLE=0

# Run fuzz test
cd "$BUILD_PATH"
"$BUILD_PATH/bin/shmem_fuzztest" "$RANK_SIZE" "$IPPORT" "$GNPU_NUM" "$FIRST_RANK" "$FIRST_NPU" --gtest_output=xml:gcover_report/test_detail.xml --gtest_filter=${TEST_FILTER}

# Collect coverage
mkdir -p "$COVERAGE_PATH"
cd "$COVERAGE_PATH"
lcov --d "$BUILD_PATH" --c --output-file "$COVERAGE_PATH/coverage.info" -rc lcov_branch_coverage=1 --ignore-errors mismatch --keep-going --rc lcov_excl_br_line="LCOV_EXCL_BR_LINE|SHM_LOG_*|SHM_ASSERT*|SHMEM_CHECK_RET"
lcov -e "$COVERAGE_PATH/coverage.info" "*/src/host/*" -o "$COVERAGE_PATH/coverage.info" --ignore-errors mismatch --keep-going --rc lcov_branch_coverage=1
genhtml -o "$COVERAGE_PATH/result" "$COVERAGE_PATH/coverage.info" --show-details --legend --rc lcov_branch_coverage=1

cd "$CURRENT_DIR"
