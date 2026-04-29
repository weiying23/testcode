# Comm Benchmark - NPU通信性能对比测试

## 概述

本Benchmark用于测试和对比昇腾NPU之间不同通信方式的性能，包括：

- **RDMA通信**：通过RoCE进行跨NPU通信
- **MTE通信**：同节点内最高带宽的内存传输引擎
- **HCCL通信**：华为官方集合通信库（标准对比基准）
- **CPU中转通信**：D2H + H2D方式（对比最低性能）
- **通信隐藏效果**：非阻塞通信 + 计算重叠

## MPI开关配置

本Benchmark支持两种进程间通信模式：

| 模式 | 宏定义 | 初始化标志 | 说明 |
|------|--------|-----------|------|
| **Socket模式（默认）** | 无需定义 | `ACLSHMEMX_INIT_WITH_DEFAULT` | 使用socket进行进程间通信，无需MPI |
| **MPI模式** | `ENABLE_MPI` | `ACLSHMEMX_INIT_WITH_MPI` | 使用MPI进行进程间通信 |

### 切换MPI模式

编辑 `benchmark_config.h` 文件：

```cpp
// ========== MPI开关配置 ==========
// 默认关闭MPI，使用socket模式进行进程间通信
// 取消下面注释以启用MPI:
// #define ENABLE_MPI    // <-- 取消此行注释启用MPI
```

**当前默认：Socket模式（关闭MPI）**

## 测试维度

| 测试类型 | 测试内容 | 输出指标 |
|---------|---------|---------|
| **PingPong延迟** | 双向往返延迟 | mean, std, min, max, median (us) |
| **带宽测试** | 单向传输带宽 | GB/s |
| **通信隐藏** | 非阻塞+计算重叠 | 隐藏率 (%) |

## 消息大小范围

| 类别 | 大小范围 | 迭代次数 | Warmup次数 |
|------|---------|---------|-----------|
| 小消息 | 1KB - 256KB | 10000 | 100 |
| 中消息 | 1MB - 8MB | 1000 | 10 |
| 大消息 | 16MB - 128MB | 100 | 5 |

## 目录结构

```
comm_benchmark/
├── CMakeLists.txt              # 构建配置
├── main.cpp                    # 主程序入口
├── comm_benchmark_kernel.cpp   # 所有NPU Kernel实现
├── benchmark_config.h          # 配置文件（含MPI开关）
├── benchmark_utils.h           # 工具函数
├── scripts/
│   └── run_benchmark.sh        # 运行脚本
├── results/                    # 结果输出目录
│   ├── latency_results.csv
│   ├── bandwidth_results.csv
│   └── hidden_results.csv
└── README.md                   # 本文档
```

## 编译

### 1. 设置环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2. 编译（Socket模式，默认）

```bash
cd shmem/
bash scripts/build.sh -examples

# 编译产物位置
# build/bin/comm_benchmark
# build/lib/libcomm_benchmark_kernel.so
```

### 3. 编译（MPI模式）

如需使用MPI模式，先修改 `benchmark_config.h` 启用 `ENABLE_MPI`，然后：

```bash
cd shmem/
bash scripts/build.sh -examples

# MPI编译需要确保环境中已安装MPI库
# 如：openmpi, mpich等
```

## 运行

### 方式一：使用运行脚本

```bash
cd examples/comm_benchmark
bash scripts/run_benchmark.sh <device_list>

# 示例：2卡测试
bash scripts/run_benchmark.sh 0,1

# 示例：4卡测试
bash scripts/run_benchmark.sh 0,1,2,3
```

### 方式二：手动运行

```bash
# 基本用法
./comm_benchmark <n_ranks> <rank_id> <ipport> <g_npus> <f_rank> <f_npu>

# 示例：2卡测试，在两个终端分别执行
# 终端1 (Rank 0):
./comm_benchmark 2 0 tcp://127.0.0.1:8789 8 0 0

# 终端2 (Rank 1):
./comm_benchmark 2 1 tcp://127.0.0.1:8789 8 0 1
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| n_ranks | 总进程数 | 2 |
| rank_id | 当前进程编号 | 0 或 1 |
| ipport | 通信地址（Socket模式） | tcp://127.0.0.1:8789 |
| g_npus | 节点内NPU数量 | 8 |
| f_rank | rank偏移量 | 0 |
| f_npu | NPU编号偏移量 | 0 或 1 |

## 结果输出

### latency_results.csv
```csv
engine,test,msg_size_bytes,iterations,mean_us,std_us,min_us,max_us,median_us
RDMA,pingpong_latency,1024,10000,5.2,0.3,4.8,6.5,5.1
MTE,pingpong_latency,1024,10000,3.8,0.2,3.5,4.2,3.7
HCCL,pingpong_latency,1024,10000,6.0,0.4,5.5,7.0,5.9
CPU_D2H_H2D,pingpong_latency,1024,10000,50.5,2.1,48.0,55.0,50.2
```

### bandwidth_results.csv
```csv
engine,test,msg_size_bytes,iterations,mean_us,std_us,min_us,max_us,median_us
RDMA,bandwidth,1048576,1000,25.5,0,25.5,25.5,25.5
MTE,bandwidth,1048576,1000,45.2,0,45.2,45.2,45.2
HCCL,allreduce_bandwidth,1048576,1000,28.0,0,28.0,28.0,28.0
HCCL,allgather_bandwidth,1048576,1000,30.0,0,30.0,30.0,30.0
HCCL,reducescatter_bandwidth,1048576,1000,26.0,0,26.0,26.0,26.0
```

### hidden_results.csv
```csv
engine,msg_size,comm_time_us,compute_time_us,overlap_time_us,hidden_rate
RDMA,1048576,500,200,400,60
MTE,1048576,300,200,250,83
```

## 性能对比预期

| 通信方式 | PingPong延迟 | 带宽 | 适用场景 |
|---------|-------------|------|---------|
| **MTE** | 最低 | 最高 | 同节点多卡通信 |
| **RDMA** | 较低 | 较高 | 跨节点通信 |
| **HCCL** | 中等 | 较高 | 官方标准库，通用场景 |
| **CPU中转** | 最高 | 最低 | 无RDMA环境的fallback |

## HCCL测试内容

| 操作 | 说明 | 测试指标 |
|------|------|---------|
| **Send/Recv** | 点对点通信 | PingPong延迟 |
| **AllReduce** | 全局归约 | 带宽 (GB/s) |
| **AllGather** | 全局收集 | 带宽 (GB/s) |
| **ReduceScatter** | 归约分发 | 带宽 (GB/s) |

## 通信隐藏效果

测试原理：
```
1. 测量纯通信时间 T_comm
2. 发起非阻塞通信
3. 立即开始计算 (MatMul)
4. 等待通信完成
5. 测量重叠时间 T_overlap
6. 隐藏率 = (T_comm - (T_overlap - T_compute)) / T_comm
```

## 注意事项

1. **MPI模式**：默认关闭，使用Socket模式运行更便捷
2. **单节点环境**：MTE测试仅在单节点内有效
3. **多次迭代**：每组测试多次迭代取平均值
4. **Warmup**：丢弃前1-5%迭代作为预热
5. **统计指标**：输出mean, std, min, max, median
6. **HCCL依赖**：需要CANN环境包含HCCL库

## 已实现功能

| 功能 | 状态 |
|------|------|
| RDMA PingPong延迟 | ✅ 已实现 |
| RDMA带宽测试 | ✅ 已实现 |
| MTE PingPong延迟 | ✅ 已实现 |
| MTE带宽测试 | ✅ 已实现 |
| HCCL Send/Recv延迟 | ✅ 已实现 |
| HCCL AllReduce带宽 | ✅ 已实现 |
| HCCL AllGather带宽 | ✅ 已实现 |
| HCCL ReduceScatter带宽 | ✅ 已实现 |
| CPU中转测试 | ✅ 已实现 |
| 通信隐藏测试 | ✅ 已实现 |
| 统计分析输出 | ✅ 已实现 |
| CSV结果输出 | ✅ 已实现 |
| MPI开关控制 | ✅ 已实现 |