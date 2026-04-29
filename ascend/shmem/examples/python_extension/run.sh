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
readonly PROJECT_ROOT=${SCRIPT_DIR}/../../

function pre_check()
{
    cd $PROJECT_ROOT
    pip show shmem >/dev/null 2>&1
    if [[ $? -eq 0 ]]; then
        echo "begin uninstall old shmem whl package"
        pip uninstall --yes shmem
    fi

    echo "begin install shmem whl package"
    whl_file=`ls package/*/*.whl`
    if [[ $? -ne 0 ]]; then
        echo "execute 'bash script/build.sh -package' to build the whl package"
        exit 1
    fi
    pip install $whl_file
    [[ $? -eq 0 ]] || exit 1

    run_file=`ls package/*/*.run`
    $run_file --install
    if [[ $? -ne 0 ]]; then
        echo "execute 'bash script/build.sh -package' to build the run package"
        exit 1
    fi
    source /usr/local/Ascend/shmem/latest/set_env.sh
    [[ $? -eq 0 ]] || exit 2
}

function run_py_test()
{
    cd $SCRIPT_DIR/test/
    torchrun --nproc-per-node 2 init_test.py
    [[ $? -eq 0 ]] || return 1

    torchrun --nproc-per-node 2 tls_test.py
    [[ $? -eq 0 ]] || return 1

    torchrun --nproc-per-node 2 global_exit_test.py
    [[ $? -eq 0 ]] || return 1
}

function main()
{
    pre_check

    run_py_test
    [[ $? -eq 0 ]] || return 1
}

main
exit $?