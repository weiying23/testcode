# Comm Benchmark — NPU 通信性能对比测试

## 概述

本 Benchmark 覆盖昇腾 NPU 间**所有主要通信路径**的延迟与带宽，提供横向对比数据，帮助开发者选择合适的通信方式，并量化非阻塞通信的计算隐藏效果。

**可进行的对比测试**

| # | 测试名称 | 引擎 | 指标 | 需要额外库 |
|---|---------|------|------|-----------|
| 1 | RDMA PingPong 延迟 | RoCE (RDMA) | mean/std/min/max/median (µs) | 无 |
| 2 | RDMA 单向带宽 | RoCE (RDMA) | GB/s | 无 |
| 3 | MTE PingPong 延迟 | MTE | mean/std/min/max/median (µs) | 无 |
| 4 | MTE 单向带宽 | MTE | GB/s | 无 |
| 5 | CPU 中转延迟 (D2H+H2D) | CPU memcpy | mean/std/min/max/median (µs) | 无 |
| 6 | 通信隐藏效果 | MTE | 隐藏率 (%) | 无 |
| 7 | HCCL Send/Recv 延迟 | HCCL | mean/std/min/max/median (µs) | HCCL 库 |
| 8 | HCCL AllReduce 带宽 | HCCL | GB/s | HCCL 库 |
| 9 | HCCL AllGather 带宽 | HCCL | GB/s | HCCL 库 |
| 10 | HCCL ReduceScatter 带宽 | HCCL | GB/s | HCCL 库 |

---

## 目录结构

```
comm_benchmark/
├── CMakeLists.txt              # 构建配置
├── main.cpp                    # Host 端主程序
├── comm_benchmark_kernel.cpp   # 所有 NPU Kernel 实现
├── benchmark_config.h          # 消息大小、迭代次数、引擎/测试枚举、MPI/HCCL 开关
├── benchmark_utils.h           # 统计计算、CSV 写入、打印辅助、环境检查
├── scripts/
│   └── run_benchmark.sh        # 多进程自动启动脚本
├── results/                    # 运行后自动创建，存放 CSV 结果
└── README.md
```

---

## 测试详解

### 测试 1 & 2 — RDMA PingPong 延迟 / 带宽

**原理**：通过 `aclshmem_uint8_put_nbi` 发起 RoCE RDMA 单向写操作，用轮询目标地址魔数的方式进行同步。

**延迟测试**（Kernel: `rdma_pingpong_latency_kernel`）
- Rank 0 向 Rank 1 发送消息，Rank 1 收到后立即回发，往返一次记为一个样本
- 记录每次往返的 cycle 数，除以 NPU 频率（1000 MHz）转换为 µs
- 消息内容末尾 8 字节用魔数轮询判断收到

**带宽测试**（Kernel: `rdma_bandwidth_kernel`）
- Rank 0 连续发出 `iterations` 次 put，再调用 `aclshmemx_roce_quiet` 等待所有传输完成
- 总时间 ÷ 总传输量 = 单向带宽
- 起始消息大小 ≥ 64 KB

---

### 测试 3 & 4 — MTE PingPong 延迟 / 带宽

**原理**：通过 `aclshmemx_mte_put_nbi` 经片上 MTE 引擎在节点内 NPU 间传输数据，完成后用 `SetFlag<MTE3_S>` / `WaitFlag<MTE3_S>` 等待传输结束。

**延迟测试**（Kernel: `mte_pingpong_latency_kernel`）
- 逻辑与 RDMA 延迟测试相同，同步机制替换为 MTE flag
- 从 `aclshmemi_get_state()` 获取运行时分配的 UB 地址和大小作为中转缓冲区

**带宽测试**（Kernel: `mte_bandwidth_kernel`）
- Rank 0 连续发出 `iterations` 次 MTE put，最后 SetFlag/WaitFlag 整体完成
- 起始消息大小 ≥ 64 KB

---

### 测试 5 — CPU 中转延迟 (D2H + H2D)

**原理**：通过 `aclrtMemcpy` 先做 Device→Host 再做 Host→Device，用 `std::chrono::high_resolution_clock` 在 CPU 侧计时，模拟最差情况下的通信路径。

- 测试范围：全部消息大小（1 KB–128 MB）
- 不依赖 shmem 通信，仅使用 ACL 标准接口
- 结果写入 `latency_results.csv`，引擎名为 `CPU_D2H_H2D`

---

### 测试 6 — 通信隐藏效果

**原理**：测量非阻塞 MTE 通信与计算负载同时运行时的实际耗时，量化通信被计算隐藏的比例。

**流程**（Kernel: `hidden_comm_kernel`，仅 Rank 0 参与）
1. 发起非阻塞 `aclshmemx_mte_put_nbi`
2. 同时执行浮点累加计算负载（负载量与消息大小匹配，见下表）
3. SetFlag/WaitFlag 等待通信完成
4. 记录整段时间为 `overlap_time`

**计算负载与消息大小的对应关系**

| 消息大小 | 矩阵规模 (M×K×N) | 浮点运算量 |
|---------|-----------------|-----------|
| ≤ 64 KB | 512 × 512 × 512 | ~268 MFlops |
| ≤ 1 MB | 1024 × 1024 × 1024 | ~2 GFlops |
| ≤ 8 MB | 2048 × 2048 × 2048 | ~17 GFlops |
| > 8 MB | 4096 × 4096 × 4096 | ~137 GFlops |

**隐藏率计算**

```
hidden_rate = (1 - overlap_time / (comm_time × 2)) × 100%
```

- `comm_time`：同等消息大小下 MTE PingPong 延迟的均值（µs）
- `comm_time × 2`：视作无隐藏时通信的理论上限
- 隐藏率越高说明通信被计算覆盖的比例越大
- 消息大小范围：≥ 256 KB

---

### 测试 7–10 — HCCL 集合通信（需启用 `ENABLE_HCCL`）

**测试 7 — HCCL Send/Recv PingPong 延迟**
- Rank 0 Send→Rank 1 Recv，Rank 1 Send→Rank 0 Recv，一来一回为一个样本
- 用 `std::chrono` 在 Host 侧计时（含 stream 提交 + 同步开销）
- 结果写入 `latency_results.csv`，引擎名为 `HCCL`

**测试 8 — HCCL AllReduce 带宽**
- 所有 Rank 参与 AllReduce（HCCL_REDUCE_SUM，float32）
- 带宽计算：`msg_size × iterations × 2 ÷ total_time`（×2 体现双向流量）
- 起始消息大小 ≥ 64 KB

**测试 9 — HCCL AllGather 带宽**
- 每个 Rank 发送 `msg_size` 字节，接收 `msg_size × world_size` 字节
- 带宽计算基于接收总量：`recv_size × iterations ÷ total_time`
- 起始消息大小 ≥ 64 KB

**测试 10 — HCCL ReduceScatter 带宽**
- 每个 Rank 发送 `msg_size × world_size` 字节，接收 `msg_size` 字节
- 带宽计算基于发送总量：`send_size × iterations ÷ total_time`
- 起始消息大小 ≥ 64 KB

---
## 消息大小与迭代次数

所有测试共用以下消息大小序列（定义于 `benchmark_config.h`）：

```
1 KB, 4 KB, 16 KB, 64 KB, 256 KB, 1 MB, 2 MB, 4 MB, 8 MB, 16 MB, 32 MB, 64 MB, 128 MB
```

迭代次数策略：

| 消息大小 | 正式迭代次数 | Warmup 次数 |
|---------|------------|------------|
| ≤ 256 KB | 10000 | 100 |
| ≤ 8 MB | 1000 | 10 |
| > 8 MB | 100 | 5 |

带宽测试（RDMA/MTE）和 HCCL 集合通信带宽测试的迭代次数略有不同：

| 消息大小 | 带宽测试迭代次数 |
|---------|--------------|
| < 64 KB | 跳过（不测带宽） |
| ≤ 8 MB | 1000 |
| > 8 MB | 100 |

通信隐藏测试仅对 ≥ 256 KB 的消息进行，迭代次数为 `≤ 8 MB: 100`，`> 8 MB: 20`。

---

## 编译开关配置

编辑 `benchmark_config.h`：

```cpp
// ========== MPI开关（默认关闭）==========
// #define ENABLE_MPI        // 取消注释使用 MPI 初始化

// ========== HCCL开关（默认关闭）==========
// #define ENABLE_HCCL       // 取消注释启用测试 7–10
```

| 宏 | 默认 | 效果 |
|----|------|------|
| `ENABLE_MPI` | 关闭 | 开启后使用 `ACLSHMEMX_INIT_WITH_MPI` 初始化，需 MPI 运行时 |
| `ENABLE_HCCL` | 关闭 | 开启后编译 HCCL 测试段，CMake 自动链接 `hccl` 库 |

---

## 编译

### 1. 设置环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 或
source ${ASCEND_HOME_PATH}/set_env.sh
```

### 2. 默认编译（无 MPI / 无 HCCL）

```bash
cd shmem/
bash scripts/build.sh -examples
# 产物：build/bin/comm_benchmark
```

### 3. 启用 HCCL 编译

在 `benchmark_config.h` 中取消注释 `#define ENABLE_HCCL`，然后重新编译。CMakeLists.txt 会自动添加 `hccl` 库链接，要求 CANN 环境中包含 `libhccl.so` 及对应头文件。

---

## 运行

### 方式一：手动启动（推荐调试）

在**两个独立终端**分别执行，Rank 0 先启动（它会监听端口等待 Rank 1 连接）：

```bash
# 终端 1 — Rank 0
./build/bin/comm_benchmark 2 0 tcp://127.0.0.1:8789 8 0 0

# 终端 2 — Rank 1
./build/bin/comm_benchmark 2 1 tcp://127.0.0.1:8789 8 0 1
```

### 方式二：脚本自动启动

```bash
# 2 卡测试（设备 0 和 1）
bash examples/comm_benchmark/scripts/run_benchmark.sh 0,1

# 4 卡测试（设备 0,1,2,3）
bash examples/comm_benchmark/scripts/run_benchmark.sh 0,1,2,3
```

### 参数说明

```
comm_benchmark <n_ranks> <rank_id> <ipport> <g_npus> <f_rank> <f_npu>
```

| 参数 | 说明 | 示例 |
|------|------|------|
| `n_ranks` | 总进程数（当前仅支持 2） | `2` |
| `rank_id` | 当前进程编号（0 或 1） | `0` |
| `ipport` | Rendezvous 地址 | `tcp://127.0.0.1:8789` |
| `g_npus` | 节点内 NPU 总数 | `8` |
| `f_rank` | rank 编号偏移量 | `0` |
| `f_npu` | NPU 编号偏移量（物理设备 ID 的起点） | `0` 或 `1` |

> `f_npu` 决定每个进程绑定的物理设备 ID：`device_id = rank_id % g_npus + f_npu`。  
> 脚本模式下，`f_npu` 由 `device_list` 中各位置的设备号自动填入。

---

## 结果输出

程序运行后，结果自动保存至 `results/` 目录（不存在时自动创建）：

### `results/latency_results.csv`

记录 PingPong 延迟测试结果，列格式：

```
engine,test,msg_size_bytes,iterations,mean_us,std_us,min_us,max_us,median_us
RDMA,pingpong_latency,1024,10000,5.20,0.31,4.80,6.50,5.10
MTE,pingpong_latency,1024,10000,3.80,0.20,3.50,4.20,3.70
CPU_D2H_H2D,pingpong_latency,1024,10000,50.50,2.10,48.00,55.00,50.20
HCCL,pingpong_latency,1024,10000,12.40,0.55,11.80,14.20,12.30
```

### `results/bandwidth_results.csv`

记录带宽测试结果，单位为 GB/s（存入 `mean_us` 列）：

```
engine,test,msg_size_bytes,iterations,mean_us,std_us,min_us,max_us,median_us
RDMA,bandwidth,1048576,1000,25.50,0,25.50,25.50,25.50
MTE,bandwidth,1048576,1000,45.20,0,45.20,45.20,45.20
HCCL,allreduce_bandwidth,1048576,1000,38.60,0,38.60,38.60,38.60
HCCL,allgather_bandwidth,1048576,100,42.10,0,42.10,42.10,42.10
HCCL,reducescatter_bandwidth,1048576,100,40.80,0,40.80,40.80,40.80
```

### `results/hidden_results.csv`

记录通信隐藏测试结果：

```
engine,test,msg_size_bytes,comm_time_us,compute_time_us,overlap_time_us,hidden_rate_pct
RDMA,hidden_comm,262144,5.20,0,4.10,60.58
MTE,hidden_comm,262144,3.80,0,2.90,61.84
```

---

## 性能预期对比

### 延迟（PingPong RTT，参考值）

| 引擎 | 1 KB | 64 KB | 1 MB | 适用场景 |
|------|------|-------|------|---------|
| **MTE** | 最低 | 最低 | 最低 | 同节点 NPU 间 |
| **RDMA** | 较低 | 较低 | 较低 | 跨节点 NPU 间 |
| **HCCL** | 中等 | 中等 | 中等 | 框架层集合通信 |
| **CPU D2H+H2D** | 最高 | 最高 | 最高 | 无 RDMA 时 fallback |

### 带宽（单向，参考值）

| 引擎 | 峰值 | 适用消息大小 |
|------|------|------------|
| **MTE** | 最高 | ≥ 256 KB |
| **RDMA** | 较高 | ≥ 1 MB |
| **HCCL AllReduce** | 受 Ring-AllReduce 影响 | ≥ 1 MB |
| **CPU D2H+H2D** | 受 PCIe 带宽限制 | 全范围 |

---

## 注意事项

- **进程数**：当前 Kernel 实现固定假设 Rank 0 与 Rank 1 进行点对点通信，`n_ranks` 实际只支持 `2`，扩展到更多 Rank 需修改 Kernel。
- **MTE 仅限单节点**：MTE 引擎使用片上互联，不支持跨节点通信，跨节点环境下请只使用 RDMA 引擎。
- **内存用量**：每个进程分配 256 MB shmem GVA，确保 NPU 设备内存充足。
- **端口冲突**：多次运行若端口未释放，换一个 `ipport` 端口号即可。
- **HCCL rootInfo 广播**：多进程场景下 `HcclGetRootInfo` 返回的信息需要由 Rank 0 通过带外方式（文件或 socket）广播给所有其他 Rank，再统一调用 `HcclCommInitRootInfo`。当前实现依赖 shmem 的初始化 socket 完成隐式同步，若单独使用 HCCL 测试段，需确认 rootInfo 已正确共享。
- **HCCL 需要额外链接**：启用 `ENABLE_HCCL` 前，确认 CANN 环境包含 `libhccl.so` 及 `hccl/hccl.h`。
