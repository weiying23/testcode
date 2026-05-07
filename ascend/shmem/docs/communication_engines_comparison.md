# Ascend SHMEM 通信引擎对比

本文档详细介绍Ascend SHMEM支持的四种通信引擎：MTE、SDMA、RDMA、UDMA的硬件机制、技术特点和适用场景。

---

## 一、引擎概述

SHMEM支持以下四种数据传输引擎：

| 引擎 | 枚举值 | 适用范围 | 芯片支持 |
|------|--------|----------|----------|
| MTE | `ACLSHMEM_DATA_OP_MTE = 0x01` | 节点内通信 | Ascend910/950 |
| SDMA | `ACLSHMEM_DATA_OP_SDMA = 0x02` | 节点内通信 | Ascend910/950 |
| RDMA | `ACLSHMEM_DATA_OP_ROCE = 0x04` | 跨节点通信 | Ascend910/950 |
| UDMA | `ACLSHMEM_DATA_OP_UDMA = 0x08` | 节点内+跨节点 | **仅Ascend950** |

引擎类型定义位于：`include/host_device/shmem_common_types.h`

```cpp
enum data_op_engine_type_t {
    ACLSHMEM_DATA_OP_MTE = 0x01,
    ACLSHMEM_DATA_OP_SDMA = 0x02,
    ACLSHMEM_DATA_OP_ROCE = 0x04,
    ACLSHMEM_DATA_OP_UDMA = 0x08,
    ACLSHMEM_DATA_OP_MAX = 0x08,
};
```

---

## 二、HCCS互联网络

**HCCS (Huawei Cache Coherence System)** 是华为的片上高速互联技术，是理解引擎差异的关键基础设施。

### HCCS特点

- **物理位置**：芯片内部互联网络
- **通信范围**：节点内NPU间通信
- **核心能力**：提供缓存一致性保证
- **带宽特性**：高带宽、低延迟的片上互联

### HCCS与引擎的关系

| 引擎 | HCCS依赖 | 说明 |
|------|----------|------|
| MTE | **必须** | 通过HCCS访问远程NPU的GM内存 |
| SDMA | **必须** | 通过HCCS进行DMA传输 |
| RDMA | 不依赖 | 使用以太网/RoCE网络 |
| UDMA | 可选 | 节点内使用HCCS，跨节点使用网络 |

**重要提示**：当HCCS连通时，MTE可以支持跨机通信（物理上仍属于节点内互联场景）。

---

## 三、引擎详细对比

### 3.1 MTE (Memory Transfer Engine)

#### 硬件机制

MTE是NPU内部的内存传输引擎，负责GM(Global Memory)与UB(Unified Buffer)之间的数据搬运。

```
数据路径：远程GM → 本地UB → 本地GM
         (HCCS)    (片上)    (片上)
```

#### 核心特点

| 特性 | 说明 |
|------|------|
| 物理位置 | NPU芯片内部的MTE单元 |
| 互联方式 | HCCS片上互联 |
| 数据路径 | GM → UB → GM（需要UB缓冲区中转） |
| 缓冲要求 | **必须使用UB缓冲区**，最小64字节 |
| API实现 | `AscendC::DataCopy`: `copy_gm2ub`, `copy_ub2gm` |

#### 使用示例

```cpp
// MTE需要通过UB缓冲区中转数据
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(__gm__ T *dst, __gm__ T *src,
                                            __ubuf__ T *buf, uint32_t ub_size,
                                            uint32_t elem_size, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr(src, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    // 远程GM → 本地UB
    aclshmemi_copy_gm2ub(buf, remote_ptr, block_size);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
    // 本地UB → 本地GM
    aclshmemi_copy_ub2gm(dst, buf, block_size);
}
```

#### 初始化配置

```cpp
aclshmemx_init_attr_t attributes;
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
```

#### 适用场景

- 节点内大规模数据传输
- 通信-计算融合（流水线优化）
- 需要UB缓冲区的场景（如矩阵乘法融合）

---

### 3.2 SDMA (System DMA)

#### 硬件机制

SDMA是片上DMA控制器，支持GM到GM的直接传输，无需UB中转。

```
数据路径：本地GM → 远程GM
         (HCCS直接传输)
```

#### 核心特点

| 特性 | 说明 |
|------|------|
| 物理位置 | NPU芯片内部的SDMA控制器 |
| 互联方式 | HCCS片上互联 |
| 数据路径 | GM → GM（直接传输） |
| 缓冲要求 | **无需UB缓冲区** |
| 异步机制 | notify_record / wait_notify |
| API实现 | SDMA专用API |

#### 使用示例

```cpp
// SDMA异步通知机制
// 记录通知
aclshmemx_notify_record(record_addr, pe);

// 等待通知到达
aclshmemx_wait_notify(record_addr);
```

#### 初始化配置

```cpp
aclshmemx_init_attr_t attributes;
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 设置SDMA参数
aclshmemx_set_sdma_config(offset, ub_size, sync_id);
```

#### 适用场景

- 节点内直接内存传输（无需UB）
- 需要异步通知机制的场景
- 低延迟小规模数据传输

---

### 3.3 RDMA (Remote DMA / RoCE)

#### 硬件机制

RDMA使用RoCE(RDMA over Converged Ethernet)协议，通过以太网进行跨节点零拷贝传输。

```
数据路径：本地HBM → RoCE NIC → 网络 → 远端NIC → 远端HBM
         (零拷贝远程直接内存访问)
```

#### 核心特点

| 特性 | 说明 |
|------|------|
| 物理位置 | RoCE网卡硬件 |
| 互联方式 | 以太网/RoCE网络 |
| 通信范围 | **跨节点通信** |
| 数据路径 | HBM → Network → HBM |
| 内存要求 | **必须使用对称内存**（`aclshmem_malloc`） |
| 同步要求 | **必须使用handle_wait同步** |
| 编译要求 | 需要 `-enable_rdma` 编译选项 |

#### 使用示例

```cpp
// RDMA初始化配置
aclshmemx_init_attr_t attributes;
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 跨节点必须使用对称内存（不能用aclrtMalloc）
void* src_ptr = aclshmem_malloc(data_size);
void* dst_ptr = aclshmem_malloc(data_size);

// 执行RDMA操作
aclshmemx_putmem_on_stream(dst_ptr, src_ptr, data_size, target_pe, stream);

// 必须使用handle_wait等待完成
aclshmem_handle_t handle;
handle.team_id = ACLSHMEM_TEAM_WORLD;
aclshmemx_handle_wait(handle, stream);
```

#### 编译要求

```bash
# 编译时需要启用RDMA支持
bash scripts/build.sh -enable_rdma -examples
```

#### 适用场景

- 跨节点NPU通信
- 分布式训练（多机场景）
- 需要零拷贝的大规模数据传输

#### 注意事项

1. **内存限制**：跨节点必须使用`aclshmem_malloc`分配对称内存，不能使用`aclrtMalloc`
2. **同步限制**：必须调用`aclshmemx_handle_wait`等待传输完成
3. **网络配置**：需要正确配置RoCE网络环境
4. **信号操作限制**：
   - `ACLSHMEM_SIGNAL_SET`：支持RDMA跨机
   - `ACLSHMEM_SIGNAL_ADD`：不支持RDMA跨机

---

### 3.4 UDMA (Unified DMA) - Ascend950专属

#### 硬件机制

UDMA是Ascend950特有的新一代统一DMA引擎，融合了节点内和跨节点通信能力，采用类似RDMA的Queue Pair机制。

```
硬件架构：
┌─────────────────────────────────────────────┐
│  UDMA引擎 (Ascend950芯片内部)                │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐    ┌──────────┐              │
│  │ Send Queue│    │Recv Queue│              │
│  │   (SQ)   │    │   (RQ)   │              │
│  └──────────┘    └──────────┘              │
│       ↓              ↓                      │
│  ┌──────────┐    ┌──────────┐              │
│  │ Send CQ  │    │ Recv CQ  │              │
│  └──────────┘    └──────────┘              │
│       ↓              ↓                      │
│  ┌──────────┐                               │
│  │ Doorbell │  (硬件通知机制)                │
│  └──────────┘                               │
│                                             │
└─────────────────────────────────────────────┘
```

#### 核心特点

| 特性 | 说明 |
|------|------|
| 物理位置 | Ascend950芯片内部 |
| 互联方式 | HCCS（节点内）+ Network（跨节点） |
| 通信范围 | 节点内 + 跨节点 |
| 硬件机制 | WQ(Work Queue) + CQ(Completion Queue) + Jetty端点 |
| 数据路径 | HBM → HBM（零拷贝直接传输） |
| 缓冲要求 | 仅需64字节UB空间（用于控制信息） |
| 原子操作 | **支持丰富的原子操作** |
| 并发限制 | **不支持对同一PE的并发操作** |
| 芯片支持 | **仅Ascend950** |
| 编译要求 | 需要 `-soc_type Ascend950` |

#### Queue Pair机制详解

```cpp
// UDMA Queue Pair信息结构
struct ACLSHMEMAIVUDMAInfo {
    uint32_t qpNum;   // Queue Pair数量（每个连接的QP数）
    uint64_t sqPtr;   // Send Queue地址数组 [PE_NUM][qpNum]
    uint64_t rqPtr;   // Receive Queue地址数组 [PE_NUM][qpNum]
    uint64_t scqPtr;  // Send Completion Queue地址数组 [PE_NUM][qpNum]
    uint64_t rcqPtr;  // Receive Completion Queue地址数组 [PE_NUM][qpNum]
    uint64_t memPtr;  // Memory Region数组 [MAX_PE_NUM]
};

// Work Queue上下文
struct ACLSHMEMUDMAWQCtx {
    uint32_t wqn;         // Work Queue Number
    uint64_t bufAddr;     // Ring Buffer起始地址
    uint32_t baseBkShift; // log2(每个WQE的大小)
    uint32_t depth;       // Ring Buffer深度
    uint64_t headAddr;    // Producer Index地址
    uint64_t tailAddr;    // Consumer Index地址
    ACLSHMEMUDMADBMode dbMode;  // Doorbell模式(HW_DB/SW_DB)
    uint64_t dbAddr;      // Doorbell地址
    uint32_t sl;          // Service Level
    uint64_t wqeCntAddr;  // WQE计数地址
    uint64_t amoAddr;     // AMO地址（存储fetch数据）
};
```

#### UDMA操作码

```cpp
enum class aclshmemi_udma_opcode_t : uint32_t {
    UDMA_OP_SEND = 0,              // 发送操作
    UDMA_OP_SEND_WITH_IMM,         // 带立即数的发送
    UDMA_OP_SEND_WITH_INV,         // 带失效的发送
    UDMA_OP_WRITE,                 // 写操作（Put）
    UDMA_OP_WRITE_WITH_IMM,        // 带立即数的写
    UDMA_OP_WRITE_WITH_NOTIFY,     // 带通知的写（PutSignal）
    UDMA_OP_READ,                  // 读操作（Get）
    UDMA_OP_CAS,                   // Compare And Swap
    UDMA_OP_ATOMIC_SWAP,           // 原子交换
    UDMA_OP_ATOMIC_STORE,          // 原子存储
    UDMA_OP_ATOMIC_LOAD,           // 原子加载
    UDMA_OPCODE_FAA = 0xb,         // Fetch And Add
    UDMA_OP_WRITE_WITH_REDUCE,     // 带Reduce的写
    UDMA_OPCODE_NOP = 0x11         // 无操作
};
```

#### 丰富的原子操作

UDMA支持最丰富的原子操作集：

| 原子操作 | 函数 | 说明 |
|----------|------|------|
| Atomic Add | `aclshmemx_udma_atomic_add` | 远程原子加 |
| Fetch Add | `aclshmemx_udma_atomic_fetch_add` | 原子加并返回原值 |
| Compare Swap | `aclshmemx_udma_atomic_compare_swap` | 条件交换（CAS） |
| Fetch | `aclshmemx_udma_atomic_fetch` | 获取远程值 |
| Set | `aclshmemx_udma_atomic_set` | 设置远程值 |
| Swap | `aclshmemx_udma_atomic_swap` | 交换并返回原值 |
| Fetch Inc | `aclshmemx_udma_atomic_fetch_inc` | 原子递增并返回原值 |
| Inc | `aclshmemx_udma_atomic_inc` | 原子递增 |
| Fetch And | `aclshmemx_udma_atomic_fetch_and` | 原子AND并返回原值 |
| And | `aclshmemx_udma_atomic_and` | 原子AND |
| Fetch Or | `aclshmemx_udma_atomic_fetch_or` | 原子OR并返回原值 |
| Or | `aclshmemx_udma_atomic_or` | 原子OR |
| Fetch XOR | `aclshmemx_udma_atomic_fetch_xor` | 原子XOR并返回原值 |
| XOR | `aclshmemx_udma_atomic_xor` | 原子XOR |

#### 使用示例

```cpp
// UDMA初始化（仅Ascend950）
aclshmemx_init_attr_t attributes;
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// UDMA Put操作（异步）
aclshmemx_udma_put_nbi(dst, src, buf, elem_size, pe);

// UDMA Get操作（异步）
aclshmemx_udma_get_nbi(dst, src, buf, elem_size, pe);

// UDMA PutSignal操作
aclshmemx_udma_put_signal_nbi(dst, src, elem_size, sig_addr, signal, pe);

// UDMA Quiet（等待完成）
aclshmemx_udma_quiet(pe);

// UDMA原子操作示例
int32_t old_value = aclshmemx_udma_atomic_fetch_add(dst, value, pe);
int32_t old_value = aclshmemx_udma_atomic_compare_swap(dst, cond, value, pe);
```

#### 编译要求

```bash
# 必须指定Ascend950芯片类型
bash scripts/build.sh -examples -soc_type Ascend950
```

#### 重要限制

**并发限制**：所有UDMA API都有明确的并发警告：

```
WARNING: When using UDMA as the underlying transport,
concurrent RMA/AMO operations to the same PE are not supported.
```

这意味着：
- **同一PE不能同时发起多个UDMA操作**
- 必须等待前一个操作完成（通过`aclshmemx_udma_quiet`）才能发起下一个

**调试工具限制**：

```
UDMA相关接口和用例不支持使用mssanitizer进行内存检测
```

#### 适用场景

- Ascend950平台的首选通信引擎
- 需要复杂原子操作的场景（CAS、FAA等）
- 需要统一处理节点内和跨节点通信
- 高性能零拷贝传输

---

## 四、架构对比图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Ascend NPU 通信引擎架构                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────┐         ┌─────────────────────────────┐  │
│   │      节点内通信              │         │      跨节点通信              │  │
│   ├─────────────────────────────┤         ├─────────────────────────────┤  │
│   │                             │         │                             │  │
│   │  ┌───────────────────────┐  │         │  ┌───────────────────────┐  │  │
│   │  │   NPU 0               │  │         │  │   Node 0              │  │  │
│   │  │  ┌─────┐              │  │         │  │  ┌─────┐              │  │  │
│   │  │  │ MTE │──┐           │  │         │  │  │NPU 0│              │  │  │
│   │  │  └─────┘  │           │  │         │  │  └─────┘              │  │  │
│   │  │           │           │  │         │  │      │                │  │  │
│   │  │  ┌─────┐  │  HCCS     │  │         │  │      │ RDMA           │  │  │
│   │  │  │SDMA │──┼───────────│  │         │  │      │ (RoCE)         │  │  │
│   │  │  └─────┘  │           │  │         │  │      │                │  │  │
│   │  │           │           │  │         │  │      ▼                │  │  │
│   │  │  ┌─────┐  │           │  │         │  │  ┌─────────────┐      │  │  │
│   │  │  │UDMA │──┘           │  │         │  │  │ RoCE NIC    │      │  │  │
│   │  │  └─────┘              │  │         │  │  └─────────────┘      │  │  │
│   │  │  (仅950)              │  │         │  │      │                │  │  │
│   │  └───────────────────────┘  │         │  │      │ Ethernet      │  │  │
│   │                             │         │  │      │ Network       │  │  │
│   │  ┌───────────────────────┐  │         │  │      ▼                │  │  │
│   │  │   NPU 1               │  │         │  │  ┌─────────────┐      │  │  │
│   │  │  ┌─────┐              │  │         │  │  │ Network     │      │  │  │
│   │  │  │ MTE │              │  │         │  │  │ Switch      │      │  │  │
│   │  │  └─────┘              │  │         │  │  └─────────────┘      │  │  │
│   │  │                       │  │         │  │      │                │  │  │
│   │  │  ┌─────┐              │  │         │  │      │                │  │  │
│   │  │  │SDMA │              │  │         │  │      ▼                │  │  │
│   │  │  └─────┘              │  │         │  │  ┌─────────────┐      │  │  │
│   │  │                       │  │         │  │  │ RoCE NIC    │      │  │  │
│   │  │  ┌─────┐              │  │         │  │  └─────────────┘      │  │  │
│   │  │  │UDMA │              │  │         │  │      │                │  │  │
│   │  │  └─────┘              │  │         │  │      ▼                │  │  │
│   │  │  (仅950)              │  │         │  │  ┌───────────────────┐│  │  │
│   │  └───────────────────────┘  │         │  │  │   Node 1         ││  │  │
│   │                             │         │  │  │  ┌─────┐         ││  │  │
│   └─────────────────────────────┘         │  │  │  │NPU 0│         ││  │  │
│                                             │  │  │  └─────┘         ││  │  │
│   数据路径对比:                             │  │  └───────────────────┘│  │  │
│                                             │  └─────────────────────────────┘  │
│   MTE:  GM → UB → GM (需要UB中转)           │                             │  │
│   SDMA: GM → GM (直接传输)                  │                             │  │
│   RDMA: HBM → NIC → Network → NIC → HBM    │                             │  │
│   UDMA: WQ/CQ机制，支持节点内+跨节点        │                             │  │
│                                             │                             │  │
│                                             └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 五、关键差异总结表

| 对比维度 | MTE | SDMA | RDMA | UDMA |
|----------|------|------|------|------|
| **通信范围** | 节点内 | 节点内 | 跨节点 | 节点内+跨节点 |
| **硬件位置** | NPU内部MTE单元 | NPU内部SDMA单元 | RoCE网卡 | NPU内部UDMA引擎 |
| **互联方式** | HCCS | HCCS | Ethernet/RoCE | HCCS + Network |
| **数据路径** | GM→UB→GM | GM→GM直接 | HBM→Network→HBM | HBM→HBM直接 |
| **需要UB缓冲** | **是** | **否** | **否** | 仅需64B |
| **原子操作** | 基础 | 基础 | 有限 | **丰富**（CAS, FAA等） |
| **芯片支持** | 910/950 | 910/950 | 910/950 | **仅950** |
| **并发支持** | 支持 | 支持 | 支持 | **不支持**同一PE并发 |
| **编译选项** | 默认 | 默认 | `-enable_rdma` | `-soc_type Ascend950` |
| **同步机制** | Stream同步 | notify/wait | handle_wait | udma_quiet |
| **内存要求** | 普通内存 | 普通内存 | **对称内存** | 对称内存 |

---

## 六、混合引擎配置

SHMEM支持同时配置多种引擎，实现自动路径选择：

```cpp
// 配置MTE和RDMA混合模式
aclshmemx_init_attr_t attributes;
attributes.option_attr.data_op_engine_type =
    static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE);
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
```

### 混合模式路径选择策略

| API | 跨机支持 | 路径选择策略 |
|-----|----------|--------------|
| `aclshmemx_putmem_on_stream` | 支持 | HCCS连通时优先走MTE，否则RDMA可用时走RDMA |
| `aclshmemx_getmem_on_stream` | 支持 | HCCS连通时优先走MTE，否则RDMA可用时走RDMA |
| `aclshmemx_signal_op_on_stream` | 部分支持 | `SIGNAL_SET`: 支持RDMA跨机；`SIGNAL_ADD`: 不支持RDMA跨机 |
| `aclshmemx_signal_wait_until_on_stream` | 不支持 | 仅HCCS可通时支持MTE跨机 |

---

## 七、引擎选择建议

### 按场景选择

| 场景 | 推荐引擎 | 原因 |
|------|----------|------|
| **节点内大规模数据传输** | MTE | 高带宽，适合批量数据 |
| **节点内直接传输** | SDMA | 无需UB中转，延迟低 |
| **跨节点通信** | RDMA | 唯一支持跨节点的通用引擎 |
| **需要复杂原子操作** | UDMA | 唯一支持CAS、FAA等丰富原子操作 |
| **通信-计算融合** | MTE | 流水线优化，UB可复用 |
| **Ascend950通用场景** | UDMA | 功能最全，节点内+跨节点统一 |

### 按芯片平台选择

| 芯片 | 首选引擎 | 备选引擎 |
|------|----------|----------|
| **Ascend910** | MTE（节点内）/ RDMA（跨节点） | SDMA |
| **Ascend950** | UDMA（统一引擎） | MTE/SDMA/RDMA |

### 按数据规模选择

| 数据规模 | 推荐引擎 | 说明 |
|----------|----------|------|
| **小数据（<1KB）** | SDMA | 低延迟，无UB开销 |
| **中等数据（1KB-1MB）** | MTE/SDMA | 根据是否需要UB选择 |
| **大数据（>1MB）** | MTE/RDMA | 高带宽优化 |

---

## 八、性能对比参考

### 理论性能指标

| 引擎 | 带宽范围 | 延迟范围 | 特点 |
|------|----------|----------|------|
| MTE | 高带宽（HCCS） | 低延迟 | UB中转有额外开销 |
| SDMA | 高带宽（HCCS） | 极低延迟 | 直接传输，无中转 |
| RDMA | 中高带宽（网络） | 中等延迟 | 跨节点受限网络性能 |
| UDMA | 高带宽 | 低延迟 | 零拷贝，功能最全 |

### 实测建议

建议使用 `comm_benchmark` 示例进行实际性能测试：

```bash
cd examples/comm_benchmark
bash run.sh
```

该测试支持：
- RDMA/MTE/SDMA/HCCL对比
- 多数据类型测试
- 多消息大小测试
- 带宽和延迟测量

---

## 九、调试与Profiling

### Profiling工具

SHMEM提供Profiling打点工具，用于量化MTE搬运性能：

```cpp
// 在Kernel代码中添加埋点
SHMEMI_PROF_BEGIN(0);  // 开始埋点
// ... MTE操作 ...
SHMEMI_PROF_END(0);    // 结束埋点

// 在Host端输出性能数据
aclshmemx_show_prof(&out_profs, false);
```

详细使用参考：[Profiling工具使用指南](docs/debug/profiling.md)

### 调试工具限制

| 引擎 | mssanitizer支持 | 说明 |
|------|-----------------|------|
| MTE | 支持 | 可进行内存检测 |
| SDMA | **不支持** | 工具限制 |
| RDMA | **不支持** | 工具限制 |
| UDMA | **不支持** | 工具限制 |

---

## 十、常见问题

### Q1: 为什么MTE需要UB缓冲区？

A: MTE引擎的设计是GM(Global Memory)与UB(Unified Buffer)之间的搬运引擎。远程数据需要先搬运到本地UB，再从UB搬运到本地GM。这种设计适合通信-计算融合场景，UB可以作为计算单元的直接输入。

### Q2: UDMA为什么仅支持Ascend950？

A: UDMA是Ascend950芯片新增的硬件引擎，采用新一代统一DMA架构，融合了节点内和跨节点通信能力。Ascend910芯片没有UDMA硬件单元。

### Q3: RDMA为什么必须使用对称内存？

A: RDMA引擎通过GVA(Global Virtual Address)直接访问远程内存。对称内存保证了所有PE在同一虚拟地址上拥有相同大小的内存块，使得PE i可以通过GVA地址直接访问PE j的数据。

### Q4: UDMA的并发限制如何解决？

A: UDMA不支持对同一PE的并发操作。解决方案：
1. 使用`aclshmemx_udma_quiet(pe)`等待前一个操作完成
2. 使用多个QP（Queue Pair）对不同PE并发操作
3. 使用其他引擎（如MTE）处理并发需求

### Q5: 如何选择节点内通信引擎？

A: 选择建议：
- Ascend910: MTE（需要UB场景）/ SDMA（直接传输场景）
- Ascend950: 首选UDMA（功能最全）

---

## 十一、参考资料

- [SHMEM初始化指南](docs/example/api_demo.md)
- [Stream API使用指南](docs/api/stream_api_usage.md)
- [Profiling工具使用](docs/debug/profiling.md)
- [编译构建指南](docs/compilation_build_guide.md)

---

## 附录：示例代码索引

| 示例 | 引擎 | 说明 |
|------|------|------|
| `init` | 全引擎 | 初始化模式测试 |
| `allgather` | MTE | AllGather集合通信 |
| `mte_perftest` | MTE | MTE性能测试 |
| `sdma` | SDMA | SDMA AllGather测试 |
| `notifywait` | SDMA | SDMA通知机制测试 |
| `rdma_demo` | RDMA | RDMA基础通信 |
| `rdma_perftest` | RDMA | RDMA性能测试 |
| `rdma_handlewait_test` | RDMA | Handle Wait同步测试 |
| `udma_demo` | UDMA | UDMA AllGather和PutSignal测试 |
| `udma_atomic_add` | UDMA | UDMA原子加测试 |
| `comm_benchmark` | 全引擎 | 综合通信性能基准测试 |