# comm_test - MTE/SDMA卡间通信性能对比测试

## 测试内容

对比两种通信引擎在同节点内不同NPU之间传输数据的性能：

| 引擎 | 数据路径 | 特点 |
|------|---------|------|
| **MTE** | GM → UB → GM | 需要UB缓冲区中转，适合大数据传输 |
| **SDMA** | GM → GM 直接 | 无需UB中转，延迟更低 |

## 编译

```bash
cd shmem/
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -examples
```

## 运行

```bash
cd examples/comm_test

# 使用 NPU 0 和 1 测试 MTE
bash run.sh 0,1

# 使用 NPU 4 和 5 测试 SDMA
bash run.sh 4,5 -e sdma_inter

# 测试两种引擎对比
bash run.sh 0,1 -all

# 直接运行二进制（需要两个进程同时运行）
./build/bin/comm_test --pe-id 0 -D 0 &
./build/bin/comm_test --pe-id 1 -D 1 &
wait
```

## 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `0,1` | NPU ID列表 | 0,1 |
| `-e mte_inter` | MTE引擎 | 默认 |
| `-e sdma_inter` | SDMA引擎 | - |
| `-all` | 测试两种引擎 | - |
| `-m put/get` | 测试模式 | put |
| `-dtype float/int32/int64` | 数据类型 | float |

## 输出

结果保存在 `output/` 目录：

- `MTE_INTER_put_float.csv`
- `SDMA_INTER_put_float.csv`

CSV内容：消息大小、带宽(GB/s)、延迟(μs)、总时间、迭代次数