#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

function check_env() {
    if ! command -v python3 >/dev/null 2>&1; then
        echo "Error: python3 not found in PATH"
        return 1
    fi
    if ! python3 -c "import pybind11" >/dev/null 2>&1; then
        echo "Error: pybind11 is not installed in current python environment"
        echo "Try: pip install pybind11"
        return 1
    fi
}

BUILD_MODE=$1
BUILD_OPEN_ABI=$2
BUILD_COMPILER=$3

if [ -z "$BUILD_MODE" ]; then
    BUILD_MODE="RELEASE"
fi

if [ -z "$BUILD_OPEN_ABI" ]; then
    BUILD_OPEN_ABI="ON"
fi

if [ -z "$BUILD_COMPILER" ]; then
    BUILD_COMPILER="gcc"
fi

readonly ROOT_PATH=$(dirname $(readlink -f "$0"))

set -e
CURRENT_DIR=$(pwd)

cd ${ROOT_PATH}/..
PROJ_DIR=$(pwd)

rm -rf ./build ./output

mkdir build/
cmake -DCMAKE_BUILD_TYPE="${BUILD_MODE}" -DBUILD_OPEN_ABI="${BUILD_OPEN_ABI}" -DBUILD_COMPILER="${BUILD_COMPILER}" -S . -B build/
make install -j5 -C build/

mkdir -p "${PROJ_DIR}/src/smem/python/mf_smem/lib"
\cp -v "${PROJ_DIR}/output/smem/lib64/libmf_smem.so" "${PROJ_DIR}/src/smem/python/mf_smem/lib"

cd ${CURRENT_DIR}