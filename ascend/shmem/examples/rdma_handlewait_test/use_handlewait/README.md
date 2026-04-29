使用方式: 
1.在shmem/目录编译:
```bash
bash scripts/build.sh -examples
```
2.在shmem/目录运行:
```bash
export PROJECT_ROOT=<shmem-root-directory>
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${PROJECT_ROOT}/3rdparty/memfabric_hybrid/output/smem/lib64:${PROJECT_ROOT}/3rdparty/memfabric_hybrid/output/hybm/lib64:$LD_LIBRARY_PATH
./build/bin/use_handlewait 2 0 tcp://127.0.0.1:8765 2 0 0 # rank 0
./build/bin/use_handlewait 2 1 tcp://127.0.0.1:8765 2 0 0 # rank 1
```

3.命令行参数说明
    ./use_handlewait <n_ranks> <rank_id> <ipport> <g_npus> <f_rank> <f_npu>

- n_ranks: 全局Rank数量，只支持2个Rank。
- rank_id: 当前进程的Rank号。
- ipport: SHMEM初始化需要的IP及端口号，格式为tcp://<IP>:<端口号>。如果执行跨机测试，需要讲IP设为rank0所在Host的IP。
- g_npus: 当前卡上启动的NPU数量。
- f_rank: 当前卡上使用的第一个Rank号。
- f_npu: 当前卡上使用的第一个NPU卡号。