#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd ${script_dir}/../../ && pwd)"
export PROJECT_ROOT=${project_root}
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${PROJECT_ROOT}/src/memfabric_hybrid/output/smem/lib64/:${PROJECT_ROOT}/src/memfabric_hybrid/output/hybm/lib64/:$LD_LIBRARY_PATH
cd $PROJECT_ROOT
pids=()
./build/bin/rdma_perftest 2 0 tcp://127.0.0.1:8765 2 0 0 highlevel_put_pingpong_latency 64 & # rank 0
pid=$!
pids+=("$pid")

./build/bin/rdma_perftest 2 1 tcp://127.0.0.1:8765 2 0 0 highlevel_put_pingpong_latency 64 & # rank 1
pid=$!
pids+=("$pid")

ret=0
for pid in ${pids[@]}; do
    wait $pid
    r=$?
    if [[ $r -ne 0 ]]; then
        ret=$r
    fi
    echo "wait $pid finished"
done
exit $ret