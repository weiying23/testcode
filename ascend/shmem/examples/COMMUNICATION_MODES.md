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

## 参考文献

- [OpenSHMEM Specification](https://openshmem.org/)
- RoCE (RDMA over Converged Ethernet) 协议
- 华为CANN开发文档