# Engine Benchmark - MTE/SDMA通信引擎性能对比测试

## 概述

本示例整合了 `mte_perftest` 和 `sdma` 的基准测试代码，用于对比以下通信场景的性能差异：

| 测试场景 | 通信引擎 | 通信范围 | 说明 |
|---------|---------|---------|------|
| **MTE_INTRA** | MTE | 卡内（同一NPU） | 同一NPU内的内存传输（暂未实现） |
| **MTE_INTER** | MTE | 卡间（同节点） | 同节点内不同NPU之间的通信 |
| **SDMA_INTER** | SDMA | 卡间（同节点） | 同节点内不同NPU之间的通信 |

### 引擎对比

| 特性 | MTE | SDMA |
|------|-----|------|
| 数据路径 | GM → UB → GM | GM → GM（直接） |
| 需要UB缓冲区 | **是** | **否** |
| 延迟 | 较低 | 更低 |
| 适用场景 | 大数据传输+计算融合 | 直接内存传输 |

## 环境要求

- CANN 9.0.0 及以上（尝鲜版）
- SDMA功能需要额外安装 ops-legacy 包
- 编译时需启用 `-examples` 选项

## 编译与运行

### 1. 编译

在 `shmem/` 根目录下编译：

```bash
# 设置CANN环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 编译所有示例
bash scripts/build.sh -examples
```

### 2. 运行

进入 `examples/comm_test` 目录执行：

```bash
# 添加执行权限
chmod +x run.sh

# 运行单个引擎测试
bash run.sh -e mte_inter -m put -d float

# 运行所有引擎对比测试
bash run.sh -all

# 跨NPU测试（指定NPU ID）
bash run.sh -pes 2 -gnpus 2 -fnpu 0 -e mte_inter
```

### 3. 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-pes` | PE数量 | 2 |
| `-gnpus` | NPU数量 | 2 |
| `-fnpu` | 第一个NPU ID | 0 |
| `-ipport` | IP和端口 | tcp://127.0.0.1:8898 |
| `-e, --engine` | 引擎类型: `mte_intra`|`mte_inter`|`sdma_inter`|`all` | mte_inter |
| `-m, --mode` | 测试模式: `put`|`get`|`bi_put`|`bi_get` | put |
| `-d, --dtype` | 数据类型: `float`|`int32`|`int64` | float |
| `-b, --block-size` | Block大小（核数） | 32 |
| `--ub-size` | UB大小(KB) | 16 |
| `-all` | 测试所有引擎 | - |

## 测试模式说明

### PUT模式（单向发送）

```
PE 0: 发送数据 → PE 1
PE 1: 接收数据
```

适用于测量单向传输带宽。

### GET模式（单向接收）

```
PE 0: 从PE 1拉取数据
PE 1: 被拉取数据
```

适用于测量主动拉取的带宽。

### BI_PUT模式（双向发送）

```
PE 0: 发送数据 → PE 1
PE 1: 发送数据 → PE 0
```

同时进行双向传输，测量双向带宽。

### BI_GET模式（双向接收）

```
PE 0: 从PE 1拉取数据
PE 1: 从PE 0拉取数据
```

同时进行双向拉取。

## 输出结果

运行完成后，结果保存在 `output/` 目录下：

```
output/
├── MTE_INTER_put_float.csv      # MTE卡间Put测试结果
├── MTE_INTER_get_float.csv      # MTE卡间Get测试结果
├── SDMA_INTER_put_float.csv     # SDMA卡间Put测试结果
└── SDMA_INTER_get_float.csv     # SDMA卡间Get测试结果
```

### CSV格式

| 列名 | 说明 |
|------|------|
| MsgSize(B) | 消息大小（字节） |
| Bandwidth(GB/s) | 带宽（GB/s） |
| Latency(us) | 单次传输延迟（微秒） |
| Time(us) | 总传输时间（微秒） |
| Iterations | 测试迭代次数 |

## 性能对比分析

### 预期性能差异

| 场景 | 数据量 | MTE优势 | SDMA优势 |
|------|--------|---------|---------|
| 小数据(<1KB) | 64B-1KB | - | 低延迟，无UB开销 |
| 中等数据(1KB-1MB) | 1KB-1MB | 流水线优化 | 直接传输效率 |
| 大数据(>1MB) | 1MB-16MB | 高带宽 | 高带宽 |

### 测试建议

1. **对比MTE和SDMA卡间性能**：
   ```bash
   bash run.sh -all -m put -d float
   ```

2. **测试不同数据类型的影响**：
   ```bash
   bash run.sh -e mte_inter -m put -d int64
   bash run.sh -e sdma_inter -m put -d int64
   ```

3. **测试不同核数的影响**：
   ```bash
   bash run.sh -e mte_inter -b 64
   bash run.sh -e sdma_inter -b 64
   ```

## 注意事项

1. **SDMA需要CANN 9.0.0**：SDMA功能在9.0.0及以上版本新增支持
2. **对称内存**：所有测试使用 `aclshmem_malloc` 分配对称内存
3. **同步机制**：
   - MTE使用 `aclshmemx_barrier_all_vec()` 同步
   - SDMA使用 `aclshmemx_sdma_quiet()` 同步
4. **UB缓冲区**：MTE需要UB缓冲区中转数据，SDMA不需要

## 相关示例

- `mte_perftest`: MTE引擎详细性能测试
- `sdma`: SDMA引擎功能验证
- `comm_benchmark`: 综合通信性能基准测试（含RDMA）

## 技术原理

### MTE引擎

```
数据路径：远程GM → 本地UB → 本地GM
         (HCCS)    (片上)    (片上)
```

- 通过HCCS片上互联访问远程GM
- 数据需要通过UB缓冲区中转
- 适合通信-计算融合场景

### SDMA引擎

```
数据路径：本地GM → 远程GM
         (HCCS直接传输)
```

- 片上DMA控制器直接传输
- 无需UB缓冲区中转
- 适合直接内存传输场景

## 参考资料

- [通信引擎对比文档](../docs/communication_engines_comparison.md)
- [SHMEM初始化指南](../init/README.md)
- [SDMA使用说明](../sdma/README.md)