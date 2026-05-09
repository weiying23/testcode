# comm_test - MTE vs SDMA 跨卡带宽对比测试

## 测试目的

对比 MTE 和 SDMA 两种通信引擎在同节点内不同 NPU 之间传输数据的带宽性能。

## 两种引擎的区别

| 引擎 | 数据传输路径 | 特点 | 适用场景 |
|------|-------------|------|---------|
| **MTE** (Memory Transfer Engine) | GM → UB → GM | 数据经过 UB 缓冲区中转 | 大数据传输 |
| **SDMA** (System DMA) | GM → GM 直接 | 不经过 UB，延迟更低 | 中小数据传输 |

- **GM** (Global Memory): NPU 的全局内存（HBM）
- **UB** (Unified Buffer): AI Core 内部的统一缓冲区

## 测试方法

### 两轮独立测试

```
第一轮：MTE 测试
  ┌─ shmem_init(ACLSHMEM_DATA_OP_MTE) ─┐  用 MTE 引擎初始化
  ├─ aclshmem_malloc                   ├  分配对称内存
  ├─ verify + sweep                    ├  MTE 带宽测试
  ├─ aclshmem_free + finalize          ├  释放资源
  └─────────────────────────────────────┘

第二轮：SDMA 测试
  ┌─ shmem_init(ACLSHMEM_DATA_OP_SDMA) ─┐  用 SDMA 引擎初始化
  ├─ aclshmem_malloc                   ├  分配对称内存
  ├─ verify + sweep                    ├  SDMA 带宽测试
  ├─ aclshmem_free + finalize          ├  释放资源
  └─────────────────────────────────────┘
```

### 单轮带宽测试流程

```
Rank 0 (发送方)                      Rank 1 (接收方)
    │                                    │
    ├─ aclshmem_malloc ──────────────────┤  分配对称内存
    │                                    │
    ├─ warmup (20 iterations) ───────────┤  预热
    │                                    │
    ├─ 记录开始时间 t0                    │
    ├─ PUT (N iterations) ──────────────►│  循环发送数据
    ├─ 等待 DMA 完成                     │
    ├─ 记录结束时间 t1                    │
    │                                    │
    ├─ 计算 BW = msg_size / (t1-t0)      │
    └ barrier_all ───────────────────────┤  同步结束
```

### 测试参数

- **消息大小**: 4KB → 16MB（8 种大小）
- **迭代次数**: 大消息 100 次，小消息 1000 次
- **AI Core 数**: 32（并行发送）

## 编译

```bash
cd shmem/
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -examples
```

## 运行

```bash
cd examples/comm_test

# 使用 NPU 0 和 NPU 1 测试
bash run.sh 0 1

# 使用 NPU 4 和 NPU 5 测试
bash run.sh 4 5
```

## 输出示例

```
[P2P] device 0 -> 1: can_access=1, EnablePeerAccess=0
[verify][MTE] PASS — PE1.gva == 0xAA (数据传输成功)

=== MTE (aclshmemx_mte_put_nbi) ===
MsgSize(B)    Iters         Time/iter(us)     BW(GB/s)
------------------------------------------------------------
4096          1000          12.34             0.33
16384         1000          45.67             0.36
...

[verify][SDMA] PASS — PE1.gva == 0xAA (数据传输成功)

=== SDMA (aclshmemx_sdma_put_nbi) ===
MsgSize(B)    Iters         Time/iter(us)     BW(GB/s)
------------------------------------------------------------
4096          1000          8.56              0.48
...

[DONE]
```

## 代码结构

### comm_test_kernel.cpp - Device Kernel

```
mte_bw_kernel():
  1. Rank 0 发送，Rank 1 等待
  2. 从 device_state 获取 UB 配置
  3. 每个 Core 处理 msg_size/block_dim 的数据切片
  4. 循环调用 aclshmemx_mte_put_nbi
  5. SetFlag/WaitFlag 等待 DMA 完成
  6. barrier_all_vec 结束

sdma_bw_kernel():
  1. Rank 0 发送，Rank 1 等待
  2. 每个 Core 处理数据切片
  3. 循环调用 aclshmemx_sdma_put_nbi（UB 大小 64B）
  4. sdma_quiet 等待 DMA 完成
  5. barrier_all_vec 结束
```

### main.cpp - Host 程序

```
main():
  1. 解析参数：pe_id, device_id, ipport
  2. ACL 初始化 + P2P 访问设置
  
  第一轮（MTE）:
    3. shmem_init(engine_type=MTE)
    4. 分配对称内存
    5. verify() + sweep() 测试 MTE
    6. 释放资源
  
  第二轮（SDMA）:
    7. shmem_init(engine_type=SDMA)
    8. 分配对称内存
    9. verify() + sweep() 测试 SDMA
    10. 释放资源
```

## 关键 API

### 初始化引擎类型

```cpp
shmem_init(pe, n_pes, ipport, engine_type)
  // engine_type:
  //   ACLSHMEM_DATA_OP_MTE  - MTE 引擎
  //   ACLSHMEM_DATA_OP_SDMA - SDMA 引擎
```

### MTE API

```cpp
aclshmemx_mte_put_nbi(dst_gm, src_gm, ub_buf, ub_size, count, peer_pe, event_id)
```
- `dst_gm`: 目标地址（对端 NPU 的 GM）
- `src_gm`: 源地址（本 NPU 的 GM）
- `ub_buf`: UB 缓冲区地址（数据中转站）
- `ub_size`: UB 大小（从 device_state 获取，通常 16KB）
- `count`: 传输字节数
- `peer_pe`: 目标 PE 编号
- `event_id`: 同步事件 ID

### SDMA API

```cpp
aclshmemx_sdma_put_nbi(dst_gm, src_gm, ub_buf, ub_size, count, peer_pe, event_id)
aclshmemx_sdma_quiet(ub_buf, ub_size, event_id)
```
- `ub_size`: 64B（仅用于状态管理，不用于数据中转）

### 对称内存 API

```cpp
aclshmem_malloc(size)   // 分配对称内存
aclshmem_free(ptr)      // 释放对称内存
aclshmem_barrier_all()  // 全局屏障同步
```

## 注意事项

1. **两个进程必须同时启动**：run.sh 会并行启动两个进程
2. **IP 地址必须相同**：两个进程使用相同的 ipport 进行通信
3. **P2P 访问**：测试前会检查并启用 NPU 之间的 P2P 访问
4. **计时方式**：使用 Host 端 chrono 计时，避免 Device profiling 的 overflow 问题
5. **两轮独立测试**：MTE 和 SDMA 分别初始化、测试、释放，确保测试环境独立