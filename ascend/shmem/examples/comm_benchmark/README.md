# Comm Benchmark - NPU通信性能对比测试

## 概述

本Benchmark用于测试和对比昇腾NPU之间不同通信方式的性能，包括：

- **RDMA通信**：通过RoCE进行跨NPU通信
- **MTE通信**：同节点内最高带宽的内存传输引擎
- **HCCL通信**：华为官方集合通信库（标准对比基准）- 需启用
- **CPU中转通信**：D2H + H2D方式（对比最低性能）
- **通信隐藏效果**：非阻塞通信 + 计算重叠

## 编译开关配置

### MPI开关

| 模式 | 宏定义 | 说明 |
|------|--------|------|
| **Socket模式（默认）** | 无需定义 | 使用socket进行进程间通信，无需MPI |
| **MPI模式** | `ENABLE_MPI` | 使用MPI进行进程间通信 |

### HCCL开关

| 模式 | 宏定义 | 说明 |
|------|--------|------|
| **HCCL关闭（默认）** | 无需定义 | 无需HCCL库，可正常编译运行 |
| **HCCL启用** | `ENABLE_HCCL` | 需CANN环境包含HCCL库 |

### 配置方式

编辑 `benchmark_config.h` 文件：

```cpp
// ========== MPI开关配置 ==========
// #define ENABLE_MPI    // 取消注释启用MPI

// ========== HCCL开关配置 ==========
// #define ENABLE_HCCL   // 取消注释启用HCCL
```

**当前默认：Socket模式 + HCCL关闭（无需MPI和HCCL即可编译运行）**

## 编译

### 1. 设置环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2. 编译（默认模式，无需MPI/HCCL）

```bash
cd shmem/
bash scripts/build.sh -examples

# 编译产物
# build/bin/comm_benchmark
```

### 3. 启用HCCL编译

修改 `benchmark_config.h` 启用 `ENABLE_HCCL`：

```cpp
#define ENABLE_HCCL   // 取消注释
```

然后重新编译。需要CANN环境包含HCCL头文件和库。

## 运行

### 方式一：手动运行（推荐）

```bash
# 基本用法
./comm_benchmark <n_ranks> <rank_id> <ipport> <g_npus> <f_rank> <f_npu>

# 示例：2卡测试，在两个终端分别执行
# 终端1 (Rank 0):
./comm_benchmark 2 0 tcp://127.0.0.1:8789 8 0 0

# 终端2 (Rank 1):
./comm_benchmark 2 1 tcp://127.0.0.1:8789 8 0 1
```

### 方式二：使用脚本

```bash
cd examples/comm_benchmark
bash scripts/run_benchmark.sh <device_list>

# 示例：2卡测试
bash scripts/run_benchmark.sh 0,1
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| n_ranks | 总进程数 | 2 |
| rank_id | 当前进程编号 | 0 或 1 |
| ipport | 通信地址 | tcp://127.0.0.1:8789 |
| g_npus | 节点内NPU数量 | 8 |
| f_rank | rank偏移量 | 0 |
| f_npu | NPU编号偏移量 | 0 或 1 |

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
├── benchmark_config.h          # 配置文件（含MPI/HCCL开关）
├── benchmark_utils.h           # 工具函数
├── scripts/run_benchmark.sh    # 运行脚本
├── results/                    # 结果输出目录
└── README.md                   # 本文档
```

## 结果输出

### latency_results.csv
```csv
engine,test,msg_size_bytes,iterations,mean_us,std_us,min_us,max_us,median_us
RDMA,pingpong_latency,1024,10000,5.2,0.3,4.8,6.5,5.1
MTE,pingpong_latency,1024,10000,3.8,0.2,3.5,4.2,3.7
CPU_D2H_H2D,pingpong_latency,1024,10000,50.5,2.1,48.0,55.0,50.2
```

### bandwidth_results.csv
```csv
engine,test,msg_size_bytes,iterations,mean_us,std_us,min_us,max_us,median_us
RDMA,bandwidth,1048576,1000,25.5,0,25.5,25.5,25.5
MTE,bandwidth,1048576,1000,45.2,0,45.2,45.2,45.2
```

## HCCL测试内容（需启用ENABLE_HCCL）

| 操作 | 说明 | 测试指标 |
|------|------|---------|
| **Send/Recv** | 点对点通信 | PingPong延迟 |
| **AllReduce** | 全局归约 | 带宽 (GB/s) |
| **AllGather** | 全局收集 | 带宽 (GB/s) |
| **ReduceScatter** | 归约分发 | 带宽 (GB/s) |

## 性能对比预期

| 通信方式 | PingPong延迟 | 带宽 | 适用场景 |
|---------|-------------|------|---------|
| **MTE** | 最低 | 最高 | 同节点多卡通信 |
| **RDMA** | 较低 | 较高 | 跨节点通信 |
| **CPU中转** | 最高 | 最低 | 无RDMA环境的fallback |

## 注意事项

1. **无需MPI/HCCL**：默认配置可无MPI、无HCCL环境编译运行
2. **HCCL可选**：需要HCCL测试时，启用 `ENABLE_HCCL` 宏
3. **单节点环境**：MTE测试仅在单节点内有效
4. **多次迭代**：每组测试多次迭代取平均值

## 已实现功能

| 功能 | 状态 | 依赖 |
|------|------|------|
| RDMA PingPong延迟 | ✅ 已实现 | 无 |
| RDMA带宽测试 | ✅ 已实现 | 无 |
| MTE PingPong延迟 | ✅ 已实现 | 无 |
| MTE带宽测试 | ✅ 已实现 | 无 |
| CPU中转测试 | ✅ 已实现 | 无 |
| 通信隐藏测试 | ✅ 已实现 | 无 |
| HCCL Send/Recv延迟 | ✅ 已实现 | HCCL库 |
| HCCL AllReduce带宽 | ✅ 已实现 | HCCL库 |
| HCCL AllGather带宽 | ✅ 已实现 | HCCL库 |
| HCCL ReduceScatter带宽 | ✅ 已实现 | HCCL库 |
| 统计分析输出 | ✅ 已实现 | 无 |
| CSV结果输出 | ✅ 已实现 | 无 |
| MPI开关控制 | ✅ 已实现 | 无 |
| HCCL开关控制 | ✅ 已实现 | 无 |