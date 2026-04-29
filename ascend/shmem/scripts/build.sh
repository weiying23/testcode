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
if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

source "$(dirname "$_ASCEND_INSTALL_PATH")/set_env.sh"

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
VERSION="1.0.0"
OUTPUT_DIR=$PROJECT_ROOT/install
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
THIRD_PARTY_DIR=$PROJECT_ROOT/3rdparty
mkdir -p $THIRD_PARTY_DIR
RELEASE_DIR=$PROJECT_ROOT/ci/release
UNDER_DIR=$PROJECT_ROOT/src/

BUILD_TYPE=RELEASE
PYEXPAND_TYPE=OFF
PACKAGE=OFF

COMPILE_OPTIONS=""

COVERAGE_TYPE=""
GEN_DOC=OFF

cann_default_path="/usr/local/Ascend/ascend-toolkit"

cd ${PROJECT_ROOT}

function fn_build()
{
    fn_build_under_memfabric
    [ -d build ] && rm -rf build
    mkdir -p build

    cd build
    cmake -DBUILD_PYTHON=$PYEXPAND_TYPE $COMPILE_OPTIONS -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
    make install -j17
    cd -
}

function fn_whl_build()
{
  echo "Python extension enabled. Copying and packaging Python wheel..."

  cd "${PROJECT_ROOT}/scripts"
  source set_env.sh

  cd "${PROJECT_ROOT}/src/python"
  rm -rf build shmem.egg-info build dist
  GIT_COMMIT=`git rev-parse HEAD`
  {
  echo "commit_id: ${GIT_COMMIT}"
  } > "${PROJECT_ROOT}/src/python/shmem/VERSION"
  python3 setup.py bdist_wheel

  cd "${PROJECT_ROOT}"
}

function make_package()
{
    rm -rf "${PROJECT_ROOT}/package"
    if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
        ARCH="x86_64"
    elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
        ARCH="aarch64"
    else
        exit 1
    fi

    mkdir -p "${PROJECT_ROOT}"/package/$ARCH/
    if [ "$PYEXPAND_TYPE" = "ON" ]; then
         cp "${PROJECT_ROOT}"/src/python/dist/*.whl "${PROJECT_ROOT}"/package/$ARCH/
         whl_name=`basename ${PROJECT_ROOT}/src/python/dist/*.whl`
         echo "${whl_name} is copy to ${PROJECT_ROOT}/package"
    fi
    cp -r "${PROJECT_ROOT}"/install/$ARCH "${PROJECT_ROOT}"/package
    echo "SHMEM_${VERSION}_linux-${ARCH}.run is copy to ${PROJECT_ROOT}/package"
}

function fn_make_run_package()
{
    if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
        echo "it is system of x86_64"
        ARCH="x86_64"
    elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
        echo "it is system of aarch64"
        ARCH="aarch64"
    else
        echo "it is not system of x86_64 or aarch64"
        exit 1
    fi

    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    commit_id=$(git rev-parse HEAD)
    mkdir -p $OUTPUT_DIR
    touch $OUTPUT_DIR/version.info
    cat>$OUTPUT_DIR/version.info<<EOF
        SHMEM Version :  ${VERSION}
        Platform : ${ARCH}
        branch : ${branch}
        commit id : ${commit_id}
EOF

    mkdir -p $OUTPUT_DIR/scripts
    mkdir -p $RELEASE_DIR/$ARCH
    cp $PROJECT_ROOT/scripts/install.sh $OUTPUT_DIR
    cp $PROJECT_ROOT/scripts/set_env.sh $OUTPUT_DIR
    cp $PROJECT_ROOT/scripts/uninstall.sh $OUTPUT_DIR/scripts

    sed -i "s/SHMEMPKGARCH/${ARCH}/" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/scripts/uninstall.sh

    chmod +x $OUTPUT_DIR/*.sh

    makeself_dir=${ASCEND_HOME_PATH}/toolkit/tools/op_project_templates/ascendc/customize/cmake/util/makeself/
    ${makeself_dir}/makeself.sh --header ${makeself_dir}/makeself-header.sh \
        --help-header $PROJECT_ROOT/scripts/help.info --gzip --complevel 4 --nomd5 --sha256 --chown \
        ${OUTPUT_DIR} $RELEASE_DIR/$ARCH/SHMEM_${VERSION}_linux-${ARCH}.run "SHMEM-api" ./install.sh
    [ -d "$OUTPUT_DIR/$ARCH" ] && rm -rf "$OUTPUT_DIR/$ARCH"
    cp -r $RELEASE_DIR/$ARCH $OUTPUT_DIR
    echo "SHMEM_${VERSION}_linux-${ARCH}.run is successfully generated in $OUTPUT_DIR/$ARCH"
}

function fn_build_googletest()
{
    if [ -d "$THIRD_PARTY_DIR/googletest/lib" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    [[ ! -d "googletest" ]] && git clone --branch v1.14.0 --depth 1 https://gitee.com/mirrors/googletest.git
    cd googletest

    rm -rf build && mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/googletest -DCMAKE_SKIP_RPATH=TRUE -DCMAKE_CXX_FLAGS="-fPIC"
    cmake --build . --parallel $(nproc)
    cmake --install . > /dev/null
    [[ -d "$THIRD_PARTY_DIR/googletest/lib64" ]] && cp -rf $THIRD_PARTY_DIR/googletest/lib64 $THIRD_PARTY_DIR/googletest/lib
    echo "Googletest is successfully installed to $THIRD_PARTY_DIR/googletest"
    cd ${PROJECT_ROOT}
}

function fn_build_secodefuzz()
{
    if [ -f "$THIRD_PARTY_DIR/secodefuzz/lib/libSecodefuzz.a" ] && [ -f "$THIRD_PARTY_DIR/secodefuzz/include/secodefuzz/secodeFuzz.h" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR

    # Need to replace git link before build fuzz test
    [[ ! -d "secodefuzz" ]] && git clone --branch v2.4.8 --depth 1 secodefuzz.git
    cd secodefuzz

    # build secodefuzz
    # -- HACK: enable PIC
    sed -i 's/cmake ../cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON ../g' ./build.sh
    # -- HACK: remove signal handlers, to support running multi-task tests.
    sed -i 's/#define HAS_SIGNAL/#undef HAS_SIGNAL/g' ./Secodefuzz/secodeFuzz.h
    bash build.sh
    if [ $? -ne 0 ]; then
        echo "secodefuzz build failed."
        return 1
    fi

    # install lib and headers into target directory
    mkdir -p "$THIRD_PARTY_DIR/secodefuzz/lib"
    cp ./examples/out-bin-x64/out/* "$THIRD_PARTY_DIR/secodefuzz/lib"
    cp ./examples/out-bin-x64/libSecodefuzz.a "$THIRD_PARTY_DIR/secodefuzz/lib"
    mkdir -p "$THIRD_PARTY_DIR/secodefuzz/include/secodefuzz"
    cp ./Secodefuzz/secodeFuzz.h "$THIRD_PARTY_DIR/secodefuzz/include/secodefuzz"
    echo "secodefuzz is successfully installed to $THIRD_PARTY_DIR/secodefuzz"
    cd ${PROJECT_ROOT}
}

function fn_build_memfabric()
{
    if [ -d "$THIRD_PARTY_DIR/memfabric_hybrid/output/smem/lib64" ]; then
        echo "Memfabric_hybrid already build"
    else
        git submodule update 3rdparty/memfabric_hybrid # not with recursive
        cd $THIRD_PARTY_DIR/memfabric_hybrid
        bash script/build.sh $BUILD_TYPE OFF OFF $PYEXPAND_TYPE
        find output
        cd ${PROJECT_ROOT}
    fi

    mkdir -p $OUTPUT_DIR/memfabric_hybrid/lib
    mkdir -p $OUTPUT_DIR/memfabric_hybrid/include
    cp -r $THIRD_PARTY_DIR/memfabric_hybrid/output/hybm/lib64/* $OUTPUT_DIR/memfabric_hybrid/lib
    cp -r $THIRD_PARTY_DIR/memfabric_hybrid/output/smem/lib64/* $OUTPUT_DIR/memfabric_hybrid/lib
    cp -r $THIRD_PARTY_DIR/memfabric_hybrid/output/smem/include/smem $OUTPUT_DIR/memfabric_hybrid/include
    echo "Memfabric_hybrid is successfully installed to $THIRD_PARTY_DIR/memfabric_hybrid"
}

function fn_build_under_memfabric()
{
    cd $UNDER_DIR/memfabric_hybrid
    bash script/build.sh $BUILD_TYPE OFF
    ls -l output/smem
    cd ${PROJECT_ROOT}

    mkdir -p $OUTPUT_DIR/memfabric_hybrid/lib
    mkdir -p $OUTPUT_DIR/memfabric_hybrid/include
    cp -r $UNDER_DIR/memfabric_hybrid/output/hybm/lib64/* $OUTPUT_DIR/memfabric_hybrid/lib
    cp -r $UNDER_DIR/memfabric_hybrid/output/smem/lib64/* $OUTPUT_DIR/memfabric_hybrid/lib
    cp -r $UNDER_DIR/memfabric_hybrid/output/smem/include/smem $OUTPUT_DIR/memfabric_hybrid/include
    echo "Memfabric_hybrid is successfully installed to $UNDER_DIR/memfabric_hybrid"
}

function fn_build_doxygen()
{
    if [ -d "$THIRD_PARTY_DIR/doxygen" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    wget --no-check-certificate https://github.com/doxygen/doxygen/releases/download/Release_1_9_6/doxygen-1.9.6.src.tar.gz
    tar -xzvf doxygen-1.9.6.src.tar.gz
    cd doxygen-1.9.6
    rm -rf build && mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/doxygen
    cmake --build . --parallel $(nproc)
    cmake --install . > /dev/null
    rm -rf $THIRD_PARTY_DIR/doxygen-1.9.6
    cd ${PROJECT_ROOT}
}

function fn_build_sphinx()
{
    [[ "$COVERAGE_TYPE" != "" ]] && return 0
    pip install -U sphinx
    pip install sphinx_rtd_theme
    pip install myst_parser
    pip install breathe
}

function fn_gen_doc()
{
    cd $PROJECT_ROOT
    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    local doxyfile=$PROJECT_ROOT/docs/Doxyfile
    local doxygen_output_dir=$PROJECT_ROOT/docs/$branch
    [[ -f "$doxyfile" ]] && rm -rf $doxyfile
    [[ -d "$doxygen_output_dir" ]] && rm -rf $doxygen_output_dir
    mkdir -p $doxygen_output_dir
    $THIRD_PARTY_DIR/doxygen/bin/doxygen -g $doxyfile
    sed -i "s#PROJECT_NAME           =.*#PROJECT_NAME           = "Shmem"#g" $doxyfile
    sed -i "s#PROJECT_NUMBER         =.*#PROJECT_NUMBER         = $branch#g" $doxyfile
    sed -i "s#OUTPUT_DIRECTORY       =.*#OUTPUT_DIRECTORY       = $doxygen_output_dir#g" $doxyfile
    sed -i "s#OUTPUT_LANGUAGE        =.*#OUTPUT_LANGUAGE        = English#g" $doxyfile
    sed -i "s#INPUT                  =.*#INPUT                  = $PROJECT_ROOT/include/host $PROJECT_ROOT/include/device $PROJECT_ROOT/include/host_device#g" $doxyfile
    sed -i "s#RECURSIVE              =.*#RECURSIVE              = YES#g" $doxyfile
    sed -i "s#USE_MDFILE_AS_MAINPAGE =.*#USE_MDFILE_AS_MAINPAGE = $PROJECT_ROOT/README.md#g" $doxyfile
    sed -i "s#HTML_EXTRA_STYLESHEET  =.*#HTML_EXTRA_STYLESHEET  = $PROJECT_ROOT/docs/doxygen/custom.css#g" $doxyfile
    sed -i "s#GENERATE_LATEX         =.*#GENERATE_LATEX         = NO#g" $doxyfile
    sed -i "s#HAVE_DOT               =.*#HAVE_DOT               = NO#g" $doxyfile
    sed -i "s#WARNINGS_AS_ERROR      =.*#WARNINGS_AS_ERROR      = NO#g" $doxyfile
    sed -i "s#EXTRACT_ALL            =.*#EXTRACT_ALL            = YES#g" $doxyfile
    sed -i "s#USE_MATHJAX            =.*#USE_MATHJAX            = YES#g" $doxyfile
    sed -i "s#WARN_NO_PARAMDOC       =.*#WARN_NO_PARAMDOC       = YES#g" $doxyfile
    sed -i "s#GENERATE_TREEVIEW      =.*#GENERATE_TREEVIEW      = YES#g" $doxyfile
    sed -i "s#WARN_AS_ERROR          =.*#WARN_AS_ERROR          = YES#g" $doxyfile
    sed -i "s#GENERATE_XML           =.*#GENERATE_XML           = YES#g" $doxyfile
    sed -i "s#MACRO_EXPANSION        =.*#MACRO_EXPANSION        = YES#g" $doxyfile
    sed -i "s#EXPAND_ONLY_PREDEF     =.*#EXPAND_ONLY_PREDEF     = YES#g" $doxyfile
    sed -i "s#EXPAND_AS_DEFINED      =.*#EXPAND_AS_DEFINED      = SHMEM_TYPE_FUNC SHMEM_TYPENAME_P_AICORE SHMEM_TYPENAME_G_AICORE SHMEM_GET_TYPENAME_MEM SHMEM_GET_TYPENAME_MEM_TENSOR SHMEM_PUT_TYPENAME_MEM SHMEM_PUT_TYPENAME_MEM_TENSOR SHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED SHMEM_PUT_TYPENAME_MEM_DETAILED SHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED#g" $doxyfile
    sed -i "s#EXCLUDE_SYMBOLS        =.*#EXCLUDE_SYMBOLS        = SHMEM_GLOBAL SHMEM_TYPENAME_P_AICORE SHMEM_TYPENAME_G_AICORE SHMEM_GET_TYPENAME_MEM SHMEM_GET_TYPENAME_MEM_TENSOR SHMEM_PUT_TYPENAME_MEM SHMEM_PUT_TYPENAME_MEM_TENSOR SHMEM_GET_TYPENAME_MEM_DETAILED SHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED SHMEM_PUT_TYPENAME_MEM_DETAILED SHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED DcciCacheline addrGm#g" $doxyfile

    $THIRD_PARTY_DIR/doxygen/bin/doxygen $doxyfile
    [[ "$COVERAGE_TYPE" != "" ]] && return 0
    local sphinx_out_dir=$PROJECT_ROOT/docs/$branch/guide
    [[ -d "$sphinx_out_dir" ]] && rm -rf $sphinx_out_dir
    mkdir -p $sphinx_out_dir
    sphinx-build -M html $PROJECT_ROOT/docs $sphinx_out_dir
}

set -e
while [[ $# -gt 0 ]]; do
    case "$1" in
        -uttests)
            fn_build_googletest
            cd $THIRD_PARTY_DIR; [[ ! -d "catlass" ]] && git clone https://gitee.com/ascend/catlass; cd $PROJECT_ROOT
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_UNIT_TEST=ON"
            shift
            ;;
        -fuzz)
            fn_build_secodefuzz
            fn_build_googletest
            cd $THIRD_PARTY_DIR; [[ ! -d "catlass" ]] && git clone https://gitee.com/ascend/catlass; cd $PROJECT_ROOT
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_FUZZ_TEST=ON"
            shift
            ;;
        -debug)
            BUILD_TYPE=Debug
            fn_build_googletest
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_UNIT_TEST=ON"
            shift
            ;;
        -examples)
            cd $THIRD_PARTY_DIR; [[ ! -d "catlass" ]] && git clone https://gitee.com/ascend/catlass; cd $PROJECT_ROOT
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_EXAMPLES=ON"
            shift
            ;;
        -python_extension)
            PYEXPAND_TYPE=ON
            shift
            ;;
        -gendoc)
            fn_build_doxygen
            fn_build_sphinx
            GEN_DOC=ON
            shift
            ;;
        -onlygendoc)
            fn_build_doxygen
            fn_build_sphinx
            fn_gen_doc
            exit 0
            shift
            ;;
        -enable_ascendc_dump)
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DENABLE_ASCENDC_DUMP=ON"
            shift
            ;;
        -package)
            PACKAGE=ON
            PYEXPAND_TYPE=ON
            shift
            ;;
        *)
            echo "Error: Unknown option $1."
            exit 1
            ;;
    esac
done

fn_build

if [ "$PYEXPAND_TYPE" = "ON" ]; then
    fn_whl_build
fi

fn_make_run_package
if [ ${GEN_DOC} == "ON" ]; then
    fn_gen_doc
fi

if [ "$PACKAGE" == "ON" ]; then
    make_package
fi

cd ${CURRENT_DIR}
