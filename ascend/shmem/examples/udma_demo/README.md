使用方式:

1.在shmem/目录编译:
```bash
bash scripts/build.sh -examples -soc_type Ascend950
```

2.在shmem/目录运行:
```bash
export PROJECT_ROOT=<shmem-root-directory>
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
export SHMEM_UID_SESSION_ID=127.0.0.1:8899

bash examples/udma_demo/run.sh 0 # allgather测试
bash examples/udma_demo/run.sh 1 # put signal 测试
```
默认按单机8卡启动，脚本依次拉起`pe 0`到`pe 7`，并等待所有进程退出。

3.脚本命令行参数说明
```bash
./udma_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> [test_type]
```

- n_pes: 全局Pe数量。
- pe_id: 当前NPU对应的Pe号。
- ipport: SHMEM初始化需要的IP及端口号，格式为tcp://<IP>:<端口号>。如果执行跨机测试，需要将IP设为pe0所在Host的IP。
- g_npus: 当前卡上启动的NPU数量。
- f_pe: 当前卡上使用的第一个Pe号。
- f_npu: 当前卡上使用的第一个NPU卡号。
- test_type: 测试类型（可选），0表示运行all-gather测试（默认），1表示运行put signal测试。
