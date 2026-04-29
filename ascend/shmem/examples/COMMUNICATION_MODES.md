# SHMEM Examples 通信模式分析

## 概述

本目录包含多个基于SHMEM库的分布式通信示例程序，展示了NPU间的高性能RDMA通信模式。这些示例涵盖了常见的分布式计算场景，如AllGather、ReduceScatter、AllReduce等集合通信操作与矩阵乘法的融合。

---

## 通信模式分类

### 1. D2D (Device-to-Device) 通信

D2D通信是本示例集的核心通信模式，通过**对称共享内存 + RDMA**实现跨rank通信。

#### 数据传输方向

| 方向 | API | 说明 | 使用场景 |
|------|-----|------|---------|
| **Put (推送)** | `shmem_mte_put_mem_nbi` | 本地NPU主动推送数据到远端rank的对称内存 | AllGather系列示例 |
| **Get (拉取)** | `shmem_mte_get_mem_nbi` | 本地NPU主动从远端rank拉取数据 | ReduceScatter/AllReduce系列示例 |

#### 数据传输模式

| 模式 | 数据流向 | 说明 | 使用场景 |
|------|---------|------|---------|
| **Gather** | 多对一收集 | 多个rank的数据收集到一个位置 | AllGather操作 |
| **Scatter** | 一对多分发 | 一个位置的数据分发到多个rank | ReduceScatter操作 |

#### 组合使用示例

| 组合 | 通信操作 | 说明 |
|------|---------|------|
| `CopyDirect::Put + CopyMode::Gather` | AllGather | 每个rank主动Put自己数据到对称内存，其他rank从对称内存读取 |
| `CopyDirect::Get + CopyMode::Scatter` | ReduceScatter | 每个rank主动Get其他rank的数据到自己位置进行归约 |

### 2. H2D/D2H (Host-to-Device/Device-to-Host) 通信

用于数据初始化和结果校验，通过ACL接口实现：

| 方向 | API | 说明 | 使用场景 |
|------|-----|------|---------|
| **H2D** | `aclrtMemcpy(..., ACL_MEMCPY_HOST_TO_DEVICE)` | Host数据拷贝到NPU Device内存 | 输入数据初始化 |
| **D2H** | `aclrtMemcpy(..., ACL_MEMCPY_DEVICE_TO_HOST)` | NPU Device数据拷贝到Host | 结果校验、数据输出 |

---

## 硬件组件说明

| 硬件组件 | 作用 | 相关API |
|---------|------|---------|
| **NPU (Neural Processing Unit)** | 矩阵计算核心，执行MatMul/GMM等计算 | `aclrtSetDevice`, `aclrtCreateStream` |
| **RoCE网卡** | RDMA over Converged Ethernet，提供高性能跨rank通信 | `SHMEM_DATA_OP_ROCE` |
| **MTE引擎 (Memory Transfer Engine)** | NPU内存传输引擎，负责数据搬运 | `shmem_mte_put_mem_nbi`, `shmem_mte_get_mem_nbi`, `shmem_mte_set_ub_params` |
| **FFTS (Fast Flag Task Sync)** | NPU核间快速同步机制，实现轻量级同步 | `shmemx_get_ffts_config` |
| **对称内存堆 (Symmetric Heap)** | 所有rank在相同偏移位置可访问的共享内存区域 | `shmem_malloc`, `shmem_free` |

---

## 示例程序详解

### 综合表格

| 示例名称 | 通信模式 | 硬件 | 计算目的 | 主要接口 |
|---------|---------|------|---------|---------|
| **allgather_matmul** | D2D (RDMA Put + Gather)<br>H2D/D2H | NPU + RoCE网卡 | 矩阵乘法前先AllGather收集输入，实现分布式矩阵计算 | `shmem_set_attr`, `shmem_init_attr`, `shmem_malloc`, `shmem_mte_put_mem_nbi`, `shmem_handle_wait`, `shmem_free`, `shmem_finalize` |
| **allgather_matmul_padding** | D2D (RDMA Put + Gather)<br>H2D/D2H | NPU + RoCE网卡 | AllGather + MatMul，支持矩阵Padding对齐 | 同上 |
| **allgather_matmul_with_gather_result** | D2D (RDMA Put + Gather)<br>H2D/D2H | NPU + RoCE网卡 | AllGather + MatMul，保留Gather中间结果 | 同上 |
| **matmul_allreduce** | D2D (RDMA Get + Scatter/Gather)<br>H2D/D2H | NPU + RoCE网卡 | MatMul后AllReduce归约结果，实现分布式矩阵计算 | `shmem_set_attr`, `shmem_init_attr`, `shmem_malloc`, `shmem_mte_get_mem_nbi`, `shmem_handle_wait`, `shmem_free`, `shmem_finalize` |
| **matmul_reduce_scatter** | D2D (RDMA Get + Scatter)<br>H2D/D2H | NPU + RoCE网卡 | MatMul后ReduceScatter分发结果，实现分布式矩阵计算 | 同上 |
| **matmul_reduce_scatter_padding** | D2D (RDMA Get + Scatter)<br>H2D/D2H | NPU + RoCE网卡 | MatMul + ReduceScatter，支持矩阵Padding对齐 | 同上 |
| **allgather** | D2D (RDMA Put/Get)<br>H2D/D2H | NPU + RoCE网卡 | 纯AllGather通信测试，验证跨rank数据收集 | `shmem_set_attr`, `shmem_init_attr`, `shmem_malloc`, `shmem_mte_put_mem_nbi`, `shmem_mte_get_mem_nbi`, `shmem_free`, `shmem_finalize` |
| **dynamic_tiling** | D2D (RDMA Get/Put)<br>H2D/D2H | NPU + RoCE网卡 | 动态分块通信-计算融合，支持三种模式：MatMul-AllReduce、AllGather-MatMul、MatMul-ReduceScatter | `shmem_set_attr`, `shmem_init_attr`, `shmem_malloc`, `shmem_mte_get_mem_nbi/put`, `shmem_handle_wait`, `shmem_free`, `shmem_finalize` |
| **dispatch_gmm_combine** | D2D (RDMA)<br>H2D/D2H | NPU + RoCE网卡 | MoE专家分发+分组矩阵乘法，AllToAll通信+GMM计算融合 | `shmem_set_attr`, `shmem_init_attr`, `shmem_malloc`, `shmem_mte_get_mem_nbi`, `shmem_quiet`, `shmem_free`, `shmem_finalize` |
| **kv_shuffle** | D2D (RDMA Put)<br>H2D/D2H | NPU + RoCE网卡 | KV Cache重排，分布式推理场景的KV cache跨rank交换 | `shmem_set_attr`, `shmem_init_attr`, `shmem_malloc`, `shmem_mte_put_mem_nbi`, `shmem_free`, `shmem_finalize` |
| **rdma_demo** | D2D (RDMA)<br>H2D/D2H | NPU + RoCE网卡 | RDMA基础功能演示，AllGather通信验证 | `shmem_set_attr`, `shmem_init_attr`, `shmem_malloc`, `shmem_handle_wait`, `shmem_free`, `shmem_finalize` |
| **rdma_perftest** | D2D (RDMA Put)<br>H2D/D2H | NPU + RoCE网卡 + MTE引擎 | RDMA性能测试：PingPong延迟、PostSend开销、带宽、MTE+RDMA并行带宽 | `shmem_set_attr`, `shmem_init_attr`, `shmem_malloc`, `shmem_mte_put_mem_nbi`, `shmem_mte_set_ub_params`, `shmemx_get_ffts_config`, `shmem_finalize` |
| **rdma_handlewait_test/use_handlewait** | D2D (RDMA Put)<br>H2D/D2H | NPU + RoCE网卡 | 验证使用shmem_handle_wait的正确同步方式 | `shmem_set_attr`, `shmem_init_attr`, `shmem_malloc`, `shmem_handle_wait`, `shmem_mte_put_mem_nbi`, `shmem_quiet`, `shmem_free`, `shmem_finalize` |
| **rdma_handlewait_test/unuse_handlewait** | D2D (RDMA Put)<br>H2D/D2H | NPU + RoCE网卡 | 验证不使用shmem_handle_wait可能导致的数据不一致问题（对比示例） | 同上（无shmem_handle_wait） |

### 示例分类

#### 通信-计算融合示例

| 示例 | 通信操作 | 计算操作 | 适用场景 |
|------|---------|---------|---------|
| allgather_matmul系列 | AllGather | MatMul | 需要先收集输入再计算的分布式矩阵乘法 |
| matmul_allreduce | AllReduce | MatMul | 计算后需要归约结果的分布式矩阵乘法 |
| matmul_reduce_scatter系列 | ReduceScatter | MatMul | 计算后需要分发结果的分布式矩阵乘法 |
| dynamic_tiling | 多种可选 | MatMul | 动态分块，灵活选择通信模式 |

#### 纯通信示例

| 示例 | 通信操作 | 目的 |
|------|---------|------|
| allgather | AllGather | 验证基础AllGather通信功能 |
| rdma_demo | AllGather | RDMA基础功能演示 |
| rdma_handlewait_test | RDMA Put | 验证同步机制的重要性 |

#### 性能测试示例

| 示例 | 测试类型 | 目的 |
|------|---------|------|
| rdma_perftest | PingPong延迟、带宽、MTE并行 | RDMA性能基准测试 |

#### 应用场景示例

| 示例 | 应用场景 | 目的 |
|------|---------|------|
| dispatch_gmm_combine | MoE (Mixture of Experts) | 专家分发+分组矩阵乘法融合 |
| kv_shuffle | 分布式推理 | KV Cache跨rank重排 |

---

## 主要API说明

### 初始化与配置

| API | 说明 |
|-----|------|
| `shmem_set_attr(rank_id, n_ranks, mem_size, ipport, &attributes)` | 设置SHMEM初始化属性参数 |
| `shmem_init_attr(attributes)` | 根据配置初始化SHMEM运行环境，建立RDMA连接 |
| `shmem_set_conf_store_tls(false, nullptr, 0)` | 禁用TLS存储配置方式 |
| `shmem_finalize()` | 结束并清理SHMEM运行环境，释放所有资源 |

### 对称内存管理

| API | 说明 |
|-----|------|
| `shmem_malloc(size)` | 从对称内存堆分配指定大小的内存 |
| `shmem_free(ptr)` | 释放对称内存 |
| `shmem_my_pe()` | 获取当前rank编号 |
| `shmem_n_pes()` | 获取总rank数量 |

### RDMA数据传输

| API | 说明 |
|-----|------|
| `shmem_mte_put_mem_nbi(dst, src, ub, params, peer, event_id)` | 非阻塞RDMA Put操作 |
| `shmem_mte_get_mem_nbi(dst, src, ub, params, peer, event_id)` | 非阻塞RDMA Get操作 |
| `shmem_quiet()` | 等待所有RDMA操作完成 |
| `shmem_handle_wait(handle, stream)` | 等待特定通信操作完成 |

### 硬件配置

| API | 说明 |
|-----|------|
| `shmemx_get_ffts_config()` | 获取FFTS硬件同步配置地址 |
| `shmem_mte_set_ub_params(idx, size, params)` | 设置MTE引擎UB缓冲区参数 |

---

## 通信流程示意图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        D2D RDMA通信流程                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Rank 0                          Rank 1                             │
│  ┌─────┐                         ┌─────┐                            │
│  │ NPU │                         │ NPU │                            │
│  └─┬───┘                         └─┬───┘                            │
│    │                               │                                │
│    ▼                               ▼                                │
│  ┌─────────────┐               ┌─────────────┐                      │
│  │ 对称内存堆  │◄────────────►│ 对称内存堆  │                      │
│  │ (Symmetric) │   RDMA/RoCE   │ (Symmetric) │                      │
│  └─────┬───────┘               └─────┬───────┘                      │
│        │                             │                              │
│  Put:  │ ───────►                    │  Get: │ ───────►             │
│        │        推送数据              │       拉取数据               │
│        │                             │                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        H2D/D2H通信流程                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Host CPU                        NPU Device                         │
│  ┌─────┐                         ┌─────┐                            │
│  │ CPU │                         │ NPU │                            │
│  └─┬───┘                         └─┬───┘                            │
│    │                               │                                │
│    │ H2D: 输入数据初始化            │                                │
│    │ ───────────────────────────► │                                │
│    │                               │                                │
│    │                               │ D2H: 结果校验                   │
│    │ ◄─────────────────────────── │                                │
│    │                               │                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 快速入门

### 编译

```bash
cd shmem/
bash scripts/build.sh
```

### 运行示例

```bash
# 运行allgather_matmul示例
cd examples/allgather_matmul
bash scripts/run.sh 6,7

# 运行matmul_allreduce示例
cd examples/matmul_allreduce
bash scripts/run.sh 6,7

# 运行rdma性能测试
./build/bin/rdma_perftest 2 0 tcp://127.0.0.1:8765 2 0 0 highlevel_put_pingpong_latency 64
./build/bin/rdma_perftest 2 1 tcp://127.0.0.1:8765 2 0 0 highlevel_put_pingpong_latency 64
```

---

## NPU间通信手段总结

### 一、通信架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NPU间通信方式分类                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    直接通信 (NPU-to-NPU)                             │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  1. RDMA Put (单边推送)                                              │   │
│  │  2. RDMA Get (单边拉取)                                              │   │
│  │  3. MTE直接传输                                                      │   │
│  │  4. 片间互联 (同一芯片内不同AI Core)                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    间接通信 (经CPU中转)                              │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  1. Host端MPI通信                                                    │   │
│  │  2. Host端Socket通信                                                 │   │
│  │  3. ACL D2H + H2D (先拷贝到Host再转发)                               │   │
│  │  4. 共享内存 (Host端共享内存作为中转站)                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    隐藏通信 (通信计算重叠)                           │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  1. 非阻塞RDMA (shmem_mte_put_nbi)                                   │   │
│  │  2. 通信-计算融合算子                                                │   │
│  │  3. Pipeline分块 (边计算边通信)                                      │   │
│  │  4. MTE+RDMA并行                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 二、直接通信方式 (NPU-to-NPU)

#### 1. RDMA Put (单边推送)

| 特性 | 说明 |
|------|------|
| **原理** | 本地NPU主动写入远端NPU内存，远端被动接收 |
| **API** | `shmem_mte_put_mem_nbi`, `shmemi_roce_write` |
| **远端CPU参与** | ❌ 不参与 |
| **数据路径** | `本地NPU内存 → RoCE网卡 → 网络 → 远端NPU内存` |
| **同步方式** | 轮询信号或`shmem_quiet` |
| **隐藏通信** | ✅ 支持非阻塞版本 |

```cpp
// Kernel内直接调用
shmem_mte_put_mem_nbi(dst_addr, src_addr, ub_buffer, ub_size, length, peer, event_id);
// 数据直接写入peer的NPU内存，peer无需调用任何接收函数
```

**适用场景**：
- AllGather操作 (每个rank把自己的数据推送到对称内存)
- KV Shuffle (推送KV cache到目标rank)
- 低延迟小消息通信

---

#### 2. RDMA Get (单边拉取)

| 特性 | 说明 |
|------|------|
| **原理** | 本地NPU主动从远端NPU读取数据到本地 |
| **API** | `shmem_mte_get_mem_nbi`, `shmemi_roce_read` |
| **远端CPU参与** | ❌ 不参与 |
| **数据路径** | `本地NPU → RoCE网卡 → 网络 → 远端NPU内存 → 本地NPU内存` |
| **同步方式** | 轮询信号或事件等待 |
| **隐藏通信** | ✅ 支持非阻塞版本 |

```cpp
// Kernel内直接调用
shmem_mte_get_mem_nbi(dst_addr, src_addr, ub_buffer, ub_size, length, peer, event_id);
// 数据直接从peer的NPU内存拉取到本地NPU内存
```

**适用场景**：
- ReduceScatter操作 (拉取其他rank数据归约)
- MatMul-AllReduce (拉取部分结果进行归约)
- 需要控制数据获取时机的场景

---

#### 3. MTE直接传输 (同节点内)

| 特性 | 说明 |
|------|------|
| **原理** | MTE引擎直接在NPU内存间搬运数据 |
| **API** | `shmem_mte_put_mem_nbi`, `shmem_mte_get_mem_nbi` |
| **适用范围** | 同节点内不同NPU卡之间 |
| **远端CPU参与** | ❌ 不参与 |
| **带宽** | 比RDMA更高 (无网络开销) |
| **隐藏通信** | ✅ 支持非阻塞 |

```cpp
// 同节点内MTE传输
shmem_mte_put_mem_nbi(dst, src, ub, size, length, peer, event_id);
// 当peer在同一节点时，MTE引擎直接搬运，不走RoCE网络
```

**适用场景**：
- 单节点多卡通信
- 高带宽需求的大数据传输
- 同芯片内AI Core间通信

---

#### 4. 片间互联 (同一芯片内)

| 特性 | 说明 |
|------|------|
| **原理** | 同一Ascend芯片内不同AI Core间的直接通信 |
| **机制** | FFTS同步 + 片内互联总线 |
| **API** | `shmemx_get_ffts_config`, 片内同步机制 |
| **延迟** | 极低 (纳秒级) |
| **隐藏通信** | ✅ 核间异步同步 |

```cpp
// 片内同步
uint64_t fftsConfig = shmemx_get_ffts_config();  // 获取FFTS配置
shmemx_set_ffts_config(fftsConfig);              // Kernel内设置同步基址
// AI Core间通过FFTS进行快速同步
```

**适用场景**：
- 同芯片内多核并行计算
- 核间数据交换
- 极低延迟同步

---

### 三、间接通信方式 (经CPU中转)

#### 1. Host端MPI通信

| 特性 | 说明 |
|------|------|
| **原理** | 先D2H拷贝到Host，再通过MPI发送，对端H2D拷贝回NPU |
| **数据路径** | `NPU → D2H → Host → MPI → Host' → H2D → NPU'` |
| **远端CPU参与** | ✅ 必须参与 (调用MPI_Recv) |
| **隐藏通信** | ⚠️ 部分支持 (MPI_Isend) |
| **延迟** | 高 (多次拷贝 + CPU处理) |

```
数据流向：
┌─────┐    D2H    ┌─────┐    MPI    ┌─────┐    H2D    ┌─────┐
│NPU 0├──────────►│CPU 0├──────────►│CPU 1├──────────►│NPU 1│
└─────┘           └─────┘           └─────┘           └─────┘
```

**适用场景**：
- 跨节点通信且无RDMA支持
- 兼容现有MPI程序
- 需要CPU参与处理数据的场景

---

#### 2. Host端Socket通信

| 特性 | 说明 |
|------|------|
| **原理** | Host端通过TCP/UDP Socket转发数据 |
| **数据路径** | `NPU → D2H → Host → Socket → Host' → H2D → NPU'` |
| **远端CPU参与** | ✅ 必须参与 |
| **隐藏通信** | ❌ 不支持 |
| **延迟** | 很高 |

**适用场景**：
- RDMA不可用时的fallback方案
- 跨网络通信
- 调试和验证阶段

---

#### 3. ACL D2H + H2D 中转

| 特性 | 说明 |
|------|------|
| **原理** | 手动控制数据从NPU拷贝到Host，再从Host拷贝到另一个NPU |
| **API** | `aclrtMemcpy(ACL_MEMCPY_DEVICE_TO_HOST)` + `aclrtMemcpy(ACL_MEMCPY_HOST_TO_DEVICE)` |
| **隐藏通信** | ❌ 完全暴露 |
| **延迟** | 高 |

```cpp
// Host端手动中转
aclrtMemcpy(host_buf, size, npu_buf, size, ACL_MEMCPY_DEVICE_TO_HOST);  // NPU0 → Host
// 通过某种方式将host_buf传输到对端Host
aclrtMemcpy(npu_buf, size, host_buf, size, ACL_MEMCPY_HOST_TO_DEVICE);  // Host → NPU1
```

**适用场景**：
- 单节点多卡数据交换
- 需要CPU处理数据的场景
- 无RDMA环境

---

#### 4. Host端共享内存中转

| 特性 | 说明 |
|------|------|
| **原理** | 多个进程的Host端通过共享内存交换数据，再H2D到NPU |
| **数据路径** | `NPU → D2H → 共享内存 → H2D → NPU'` |
| **远端CPU参与** | ✅ 参与 (读写共享内存) |
| **隐藏通信** | ❌ 不支持 |
| **适用范围** | 同节点内 |

**适用场景**：
- 同节点多进程协作
- Host端数据共享
- 低成本数据交换

---

### 四、隐藏通信方式 (通信计算重叠)

#### 1. 非阻塞RDMA (通信隐藏)

| 特性 | 说明 |
|------|------|
| **原理** | 发起RDMA后立即返回，通信在后台进行，不阻塞计算 |
| **API** | `shmem_mte_put_mem_nbi`, `shmem_mte_get_mem_nbi` (nbi = non-blocking immediate) |
| **隐藏效果** | ✅ 通信与计算并行 |
| **同步方式** | `shmem_quiet` 或 `shmem_handle_wait` |

```cpp
// 非阻塞Put
shmem_mte_put_mem_nbi(dst, src, ub, size, len, peer, event_id);  // 发起后立即返回
// 此时可以继续进行计算...
compute_kernel(...);  // 计算与通信并行进行
// 需要同步时
shmem_quiet();  // 等待所有RDMA完成
// 或
shmem_handle_wait(handle, stream);  // 等待特定通信完成
```

**隐藏示意**：
```
时间线：
────────────────────────────────────────────────────────►
通信:  [====Put发起====]     [====数据传输====]     [====完成====]
计算:                    [====计算Kernel====]    [====继续计算====]
                           ↑通信和计算并行重叠
```

---

#### 2. 通信-计算融合算子

| 特性 | 说明 |
|------|------|
| **原理** | 在单个Kernel内同时执行通信和计算，利用数据依赖隐藏通信 |
| **示例** | AllGather-MatMul, MatMul-ReduceScatter |
| **隐藏效果** | ✅ 高效隐藏 |
| **实现** | 分块Pipeline执行 |

```cpp
// AllGather-MatMul融合Kernel
// 块0: 通信块0数据的同时，计算块1
for (int block = 0; block < num_blocks; block++) {
    // 发起下一块通信 (隐藏在当前计算中)
    shmem_mte_put_mem_nbi(next_block_src, next_block_dst, ...);
    
    // 计算当前块
    matmul_compute(current_block_A, current_block_B, current_block_C);
    
    // 等待通信完成 (此时计算已完成，通信开销被隐藏)
    wait_for_comm_event();
}
```

**隐藏示意**：
```
传统分离模式：
通信:  [====AllGather====] [等待]
计算:                    [====MatMul====]

融合隐藏模式：
通信:  [块0][块1][块2][块3]  ← 分块通信
计算:       [块0][块1][块2][块3]  ← 分块计算，通信被隐藏
       ↑
       块0通信时块1计算，块1通信时块2计算...
```

---

#### 3. Pipeline分块通信

| 特性 | 说明 |
|------|------|
| **原理** | 将大数据分块，流水线执行：块i通信时，块i-1计算 |
| **隐藏效果** | ✅ 深度隐藏 |
| **实现** | 多级UB缓冲交替使用 |

```cpp
// Pipeline分块示例
constexpr int UB_STAGES = 2;  // 2级缓冲交替使用

// Stage 0: 发起通信
shmem_mte_put_mem_nbi(block[0].dst, block[0].src, ub[0], ...);

// Stage 1: 使用ub[1]计算block[1]，同时block[0]在通信
compute_block(ub[1], block[1]);

// Stage 2: 等待block[0]通信完成，开始计算
wait_event(event[0]);
// block[0]通信完成，可以开始计算block[0]或发起下一块通信
```

---

#### 4. MTE + RDMA并行

| 特性 | 说明 |
|------|------|
| **原理** | MTE引擎处理本地数据搬运，RDMA处理远端数据传输，两者并行 |
| **API** | `shmem_mte_set_ub_params`, `shmemi_roce_write` |
| **隐藏效果** | ✅ 本地+远端并行 |
| **示例** | `rdma_mte_put_bw` 性能测试 |

```cpp
// rdma_mte_put_bw示例 (rdma_perftest_kernel.cpp)
// Core 0: RDMA发送
shmemi_roce_write(remote_addr, local_addr, peer, ...);

// Core 1: MTE本地搬运
shmem_mte_put_mem_nbi(local_dst, local_src, ub, ...);

// 两个Core并行工作，RDMA和MTE同时执行
```

---

### 五、通信方式对比总表

| 方式 | NPU直接通信 | CPU参与 | 隐藏支持 | 延迟 | 带宽 | 适用场景 |
|------|-------------|---------|---------|------|------|---------|
| **RDMA Put** | ✅ 是 | ❌ 否 | ✅ 非阻塞 | 极低 | 高 | AllGather, KV Shuffle |
| **RDMA Get** | ✅ 是 | ❌ 否 | ✅ 非阻塞 | 极低 | 高 | ReduceScatter, AllReduce |
| **MTE直接** | ✅ 是 | ❌ 否 | ✅ 非阻塞 | 低 | 极高 | 同节点内传输 |
| **片间互联** | ✅ 是 | ❌ 否 | ✅ 异步同步 | 纳秒级 | 最高 | 同芯片内核间 |
| **Host MPI** | ❌ 否 | ✅ 是 | ⚠️ 部分 | 高 | 中 | 无RDMA环境 |
| **Host Socket** | ❌ 否 | ✅ 是 | ❌ 否 | 很高 | 低 | 调试/Fallback |
| **D2H+H2D** | ❌ 否 | ✅ 是 | ❌ 否 | 高 | 中 | 需CPU处理 |
| **共享内存** | ❌ 否 | ✅ 是 | ❌ 否 | 中 | 高 | 同节点Host共享 |
| **融合算子** | ✅ 是 | ❌ 否 | ✅ 深度隐藏 | 极低 | 高 | 分布式训练 |
| **Pipeline** | ✅ 是 | ❌ 否 | ✅ 深度隐藏 | 极低 | 高 | 大数据分块 |
| **MTE+RDMA** | ✅ 是 | ❌ 否 | ✅ 并行 | 极低 | 极高 | 性能极限场景 |

---

### 六、选择指南

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        通信方式选择决策树                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 是否在同节点内？                                                         │
│     ├── YES → 优先选择 MTE直接传输 (最高带宽)                               │
│     │        或 RDMA Put/Get (跨节点也支持)                                 │
│     │                                                                       │
│     └── NO → 是否支持RDMA？                                                 │
│              ├── YES → RDMA Put/Get (最低延迟)                              │
│              │       + 隐藏通信 (非阻塞/Pipeline)                            │
│              │                                                              │
│              └── NO → Host端通信                                            │
│                        同节点 → 共享内存中转                                 │
│                        跨节点 → MPI/Socket                                   │
│                                                                             │
│  2. 是否需要隐藏通信？                                                       │
│     ├── YES → 通信-计算融合算子                                             │
│     │        或 Pipeline分块                                                │
│     │        或 非阻塞RDMA                                                  │
│     │                                                                       │
│     └── NO → 阻塞式通信 (简单易用)                                          │
│              shmem_quiet等待完成                                            │
│                                                                             │
│  3. 数据大小？                                                               │
│     ├── 小数据 (< 1KB) → RDMA Put/Get                                       │
│     │       关注延迟而非带宽                                                 │
│     │                                                                       │
│     ├── 中等数据 (1KB - 1MB) → RDMA + Pipeline                              │
│     │       分块隐藏通信                                                     │
│     │                                                                       │
│     └── 大数据 (> 1MB) → 融合算子                                           │
│             深度隐藏，最大化计算通信重叠                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 七、典型应用场景匹配

| 应用场景 | 推荐通信方式 | 原因 |
|---------|-------------|------|
| **分布式矩阵乘法** | 融合算子 + Pipeline | MatMul计算量大，可深度隐藏通信 |
| **分布式推理** | RDMA Put (KV Shuffle) | 推理阶段通信频繁，需要低延迟 |
| **梯度同步** | RDMA Get + 非阻塞 | AllReduce梯度，异步更新参数 |
| **MoE专家分发** | RDMA AllToAll | 专家数据量大，RDMA高带宽 |
| **参数服务器** | RDMA Get | Worker拉取参数，低延迟更新 |
| **同节点多卡** | MTE直接传输 | 最高带宽，无网络开销 |
| **调试验证** | Host端D2H+H2D | 数据可控，便于观察 |

---

### 八、隐藏通信效果对比

| 隐藏程度 | 方式 | 通信隐藏率 | 实现复杂度 |
|---------|------|-----------|-----------|
| **无隐藏** | 阻塞式RDMA | 0% | 简单 |
| **部分隐藏** | 非阻塞RDMA | 30-50% | 中等 |
| **中等隐藏** | Pipeline分块 | 50-70% | 中等 |
| **深度隐藏** | 融合算子 | 70-90% | 复杂 |
| **极限隐藏** | MTE+RDMA并行 | >90% | 很复杂 |

**隐藏率公式**：
```
隐藏率 = (通信时间 - 额外等待时间) / 通信时间 × 100%

理想隐藏：通信完全被计算覆盖，额外等待时间 = 0，隐藏率 = 100%
```

---

### 九、RDMA PingPong延迟测试详解

#### 测试原理

PingPong延迟测试是一种经典的**网络性能测试方法**，用于测量两个节点之间**单次通信往返的时间延迟**。

```
时间线：
    Rank 0                      Rank 1
       │                           │
       │  ─────── Put数据 ───────► │   (1) 发送消息
       │                           │
       │                           │  收到数据，立即回复
       │                           │
       │  ◄─────── Put数据 ─────── │   (2) 返回消息
       │                           │
   ────┴───────────────────────────┴────
   
   latency = T_end - T_start (往返时间)
   单向延迟 ≈ latency / 2
```

#### 与MPI Send/Recv的差异

| 特性 | MPI Send/Recv | RDMA Put PingPong |
|------|---------------|-------------------|
| **通信模式** | 双侧主动 | 单侧主动 |
| **发送端** | 调用 `MPI_Send()` | 调用 `shmem_put()` |
| **接收端** | **必须调用 `MPI_Recv()`** | **无需调用接收函数** |
| **远端CPU参与** | **必须参与** | **不参与** |
| **数据拷贝** | 可能多次拷贝 | **零拷贝** |
| **延迟来源** | CPU处理+缓冲拷贝+握手 | RDMA启动开销+网络延迟 |

**Send/Recv转换为单边通信**：

- `MPI Send` → `shmem_put` (主动写入对方内存)
- `MPI Recv` → `shmem_get` (主动从对方内存读取)

转换后远端无需调用接收函数，但需要额外同步机制。

---

### 十、昇腾NPU代码运行机制

#### 编译架构

```
源代码
├── main.cpp (Host端代码)
│   └── 标准C++，运行在CPU上
│   └── 调用ACL API控制NPU
│
└── *_kernel.cpp (Device端Kernel)
    └── AscendC编程，带 __aicore__ 标记
    └── 运行在NPU AI Core上

编译器
├── Host端: bisheng编译器 (C++编译器)
└── Device端: CCE编译器 (昇腾专用编译器)
    └── 编译选项: -xcce --cce-aicore-arch=dav-c220
```

#### 关键标记说明

| 标记 | 含义 |
|------|------|
| `__global__` | Kernel入口函数，Host调用，Device执行 |
| `__aicore__` | 运行在AI Core上（计算核心） |
| `__gm__` | Global Memory指针（NPU全局内存） |
| `__ubuf__` | Unified Buffer指针（NPU片上缓冲） |
| `<<<blocks, nullptr, stream>>>` | Kernel启动语法，指定block数和stream |

#### 运行流程

```
1. Host端启动程序
   ↓
2. aclInit() → 初始化ACL运行时
   ↓
3. aclrtSetDevice(device_id) → 选择NPU设备
   ↓
4. shmem_init_attr() → 初始化SHMEM，建立RDMA连接
   ↓
5. shmem_malloc() → 分配对称共享内存
   ↓
6. 启动Kernel到NPU
   kernel<<<block_dim, stream>>>(args...)
   ┌───────────────────────────────┐
   │ NPU AI Core执行Kernel代码     │
   │ - shmem_put/get RDMA通信      │
   │ - 矩阵计算                    │
   └───────────────────────────────┘
   ↓
7. aclrtSynchronizeStream() → 等待Kernel完成
   ↓
8. shmem_finalize() / aclFinalize() → 清理资源
```

---

## 参考文献

- [OpenSHMEM Specification](https://openshmem.org/)
- RoCE (RDMA over Converged Ethernet) 协议
- 华为CANN开发文档