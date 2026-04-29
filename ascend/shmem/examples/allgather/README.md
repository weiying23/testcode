使用方式: 
1.在shmem/目录编译: 
    bash scripts/build.sh

2.在shmem/examples/allgather目录执行demo:
    # 完成RANKS卡下的allgather同时验证精度，性能数据会输出在result.csv中。
    # RANKS : [2, 4, 8]
    # TYPES : [int, int32_t, float16_t, bfloat16_t]
    bash run.sh -ranks ${RANKS} -type ${TYPES}