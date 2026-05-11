# SHMEM MTE/SDMA 通信示例伪代码

本文档从 `comm_test` 代码中提取出使用 SHMEM 进行 MTE 和 SDMA 通信的关键步骤、接口及调用顺序。

---

## 1. Host 端初始化流程

```伪代码
// ============================================================
// 第一步：ACL 运行时初始化
// ============================================================
aclInit(nullptr)                    // 初始化 ACL 运行时环境
aclrtSetDevice(device_id)           // 设置当前进程使用的 NPU 设备

// ============================================================
// 第二步：P2P 访问设置（跨卡通信必须）
// ============================================================
// WHY：跨卡通信本质是 P2P（Peer-to-Peer）访问，两个 NPU 直接访问对方内存
aclrtDeviceCanAccessPeer(&can_access, device_id, peer_device)  // 检查 P2P 是否支持
if (can_access) {
    aclrtDeviceEnablePeerAccess(peer_device, 0)                // 启用 P2P 访问权限
}

aclrtCreateStream(&stream)          // 创建 ACL 流（管理 kernel 执行）

// ============================================================
// 第三步：SHMEM 初始化（选择通信引擎）
// ============================================================
// 填充初始化属性结构体
aclshmemx_init_attr_t attr
attr.my_pe          = pe_id         // 当前 PE 编号（进程 ID）
attr.n_pes          = n_pes         // 总 PE 数量
attr.ip_port        = "tcp://127.0.0.1:8998"  // rendezvous 地址
attr.local_mem_size = 512MB         // 对称内存大小

// 关键参数：选择通信引擎类型
// - ACLSHMEM_DATA_OP_MTE：MTE 引擎（GM → UB → GM）
// - ACLSHMEM_DATA_OP_SDMA：SDMA 引擎（GM → GM）
// - ACLSHMEM_DATA_OP_ROCE：RoCE 引擎（跨节点 RDMA）
attr.option_attr.data_op_engine_type = engine_type

attr.comm_args      = &g_uid        // uniqueid（必须是全局/静态变量）

aclshmemx_set_conf_store_tls(false, nullptr, 0)
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr)  // 初始化 SHMEM 运行时

// ============================================================
// 第四步：分配对称内存
// ============================================================
// WHY：对称内存是所有 PE 在同一虚拟地址上的相同大小内存
//      PE i 可以直接通过 GVA 地址访问 PE j 的数据
uint8_t *gva = aclshmem_malloc(size)  // 分配对称内存（返回 GVA 指针）

aclshmem_barrier_all()              // 屏障同步：确保所有 PE 完成初始化

// ============================================================
// 第五步：数据初始化（可选）
// ============================================================
aclrtMemset(gva, size, fill_value, size)  // 初始化对称内存数据
aclrtMemcpy(gva, size, host_data, size, ACL_MEMCPY_HOST_TO_DEVICE)
aclshmem_barrier_all()              // 等待所有 PE 完成数据准备
```

---

## 2. Kernel 端通信流程

### 2.1 MTE 通信（GM → UB → GM）

```伪代码
// ============================================================
// MTE Kernel 流程
// ============================================================

// 获取 PE 信息
int64_t rank = aclshmem_my_pe()     // 当前 PE 编号
int64_t n_pes = aclshmem_n_pes()    // 总 PE 数量
uint32_t peer = 1 - rank            // 对端 PE 编号（假设 2 PE 场景）

// 从 device_state 获取 MTE 配置（WHY：硬件配置动态设置，非硬编码）
aclshmem_device_host_state_t *st = aclshmemi_get_state()
uint64_t copy_ub = st->mte_config.aclshmem_ub      // UB 缓冲区地址
uint32_t copy_size = st->mte_config.ub_size        // UB 大小（通常 16KB）
TEventID event_id = st->mte_config.sync_id         // 事件 ID

// 按 block_dim 切分数据（WHY：多核并行，每个核处理不同数据切片）
int64_t slice = msg_size / block_dim
int64_t offset = core_idx * slice
__gm__ uint8_t *src = gva + offset
__gm__ uint8_t *dst = gva + offset

PipeBarrier<PIPE_ALL>()             // 确保所有管道同步

// MTE 非阻塞 PUT 循环
for (int64_t i = 0; i < iterations; i++) {
    // aclshmemx_mte_put_nbi：MTE 非阻塞 PUT 操作
    // 参数：
    //   - dst：目标地址（对端 PE 的 GVA 偏移）
    //   - src：源地址（本 PE 的 GVA 偏移）
    //   - ub_buf：UB 缓冲区（WHY：数据必须先到 UB 才能跨卡传输）
    //   - ub_size：UB 大小（决定每次搬运的最大 chunk）
    //   - count：传输字节数
    //   - peer_pe：目标 PE 编号
    //   - event_id：同步事件 ID
    aclshmemx_mte_put_nbi(dst, src, ub_buf, ub_size, count, peer_pe, event_id)
}

// 同步等待 DMA 完成
SyncAll()                           // 所有 AI Core 同步
if (core_idx == 0) {
    SetFlag<HardEvent::MTE3_S>(event_id)   // 设置事件标志
    WaitFlag<HardEvent::MTE3_S>(event_id)  // 等待 DMA 完成
}

aclshmemx_barrier_all_vec()         // 所有核屏障同步
```

### 2.2 SDMA 通信（GM → GM）

```伪代码
// ============================================================
// SDMA Kernel 流程
// ============================================================

// 获取 PE 信息
int64_t rank = aclshmem_my_pe()
int64_t n_pes = aclshmem_n_pes()
uint32_t peer = 1 - rank

// SDMA UB 配置（WHY：SDMA 不需要 UB 中转数据，只用于状态存储）
constexpr uint32_t UB_OFFSET = 1024
constexpr uint32_t SDMA_UB_SIZE = 64
__ubuf__ uint8_t *tmp_ub = reinterpret_cast<__ubuf__ uint8_t *>(UB_OFFSET)

// 按 block_dim 切分数据
int64_t slice = msg_size / block_dim
int64_t offset = core_idx * slice
__gm__ uint8_t *src = gva + offset
__gm__ uint8_t *dst = gva + offset

PipeBarrier<PIPE_ALL>()

// SDMA 非阻塞 PUT 循环
for (int64_t i = 0; i < iterations; i++) {
    // aclshmemx_sdma_put_nbi：SDMA 非阻塞 PUT 操作
    // 参数：
    //   - dst：目标地址（对端 PE 的 GVA）
    //   - src：源地址（本 PE 的 GVA）
    //   - ub_buf：UB 缓冲区（WHY：存储 DMA 状态，不用于数据中转）
    //   - ub_size：UB 大小（64B 即可）
    //   - count：传输字节数
    //   - peer_pe：目标 PE 编号
    //   - event_id：同步事件 ID（如 EVENT_ID0）
    // 数据路径：本 NPU GM → HCCS → 目标 NPU GM（直接传输）
    aclshmemx_sdma_put_nbi(dst, src, ub_buf, ub_size, count, peer_pe, event_id)
}

// SDMA quiet：等待所有 SDMA 操作完成
// WHY：SDMA 使用专门的 quiet API（因为 DMA 在芯片级别的 SDMA 引擎）
aclshmemx_sdma_quiet(ub_buf, ub_size, event_id)

aclshmemx_barrier_all_vec()
```

---

## 3. 资源释放流程

```伪代码
// ============================================================
// 释放对称内存
// ============================================================
// WHY：必须使用 aclshmem_free，不能用 aclrtFree
aclshmem_free(gva)

// ============================================================
// 终止 SHMEM 运行时
// ============================================================
// WHY：释放对称内存堆、关闭通信通道、清理引擎状态
aclshmem_finalize()

// ============================================================
// ACL 资源释放
// ============================================================
aclrtDestroyStream(stream)
aclrtResetDevice(device_id)
aclFinalize()
```

---

## 4. 完整调用顺序图

```
┌─────────────────────────────────────────────────────────────┐
│                    Host 端初始化                              │
├─────────────────────────────────────────────────────────────┤
│  1. aclInit()                                                │
│  2. aclrtSetDevice()                                         │
│  3. aclrtDeviceCanAccessPeer() + EnablePeerAccess()          │
│  4. aclrtCreateStream()                                      │
│  5. shmem_init(MTE/SDMA引擎)                                 │
│  6. aclshmem_malloc() → gva                                  │
│  7. aclshmem_barrier_all()                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Kernel 端通信                             │
├─────────────────────────────────────────────────────────────┤
│  MTE 流程：                                                  │
│  ├─ aclshmem_my_pe() + aclshmem_n_pes()                     │
│  ├─ aclshmemi_get_state() → ub/size/event_id                │
│  ├─ PipeBarrier<PIPE_ALL>()                                 │
│  ├─ 循环：aclshmemx_mte_put_nbi()                           │
│  ├─ SyncAll() + SetFlag/WaitFlag                            │
│  └─ aclshmemx_barrier_all_vec()                             │
│                                                              │
│  SDMA 流程：                                                 │
│  ├─ aclshmem_my_pe() + aclshmem_n_pes()                     │
│  ├─ PipeBarrier<PIPE_ALL>()                                 │
│  ├─ 循环：aclshmemx_sdma_put_nbi()                          │
│  ├─ aclshmemx_sdma_quiet()                                  │
│  └─ aclshmemx_barrier_all_vec()                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Host 端等待 + 释放                         │
├─────────────────────────────────────────────────────────────┤
│  1. aclrtSynchronizeStream(stream)                          │
│  2. aclshmem_barrier_all()                                  │
│  3. aclshmem_free(gva)                                      │
│  4. aclshmem_finalize()                                     │
│  5. aclrtDestroyStream() + aclrtResetDevice()               │
│  6. aclFinalize()                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 关键 API 对照表

| API | 作用 | MTE | SDMA |
|-----|------|-----|------|
| `aclshmemx_init_attr` | 初始化 SHMEM 运行时 | ✓ | ✓ |
| `aclshmem_malloc` | 分配对称内存（GVA） | ✓ | ✓ |
| `aclshmem_barrier_all` | 全局屏障同步 | ✓ | ✓ |
| `aclshmemx_mte_put_nbi` | MTE 非阻塞 PUT | ✓ | - |
| `aclshmemx_sdma_put_nbi` | SDMA 非阻塞 PUT | - | ✓ |
| `SetFlag/WaitFlag` | MTE DMA 同步 | ✓ | - |
| `aclshmemx_sdma_quiet` | SDMA DMA 同步 | - | ✓ |
| `aclshmem_free` | 释放对称内存 | ✓ | ✓ |
| `aclshmem_finalize` | 终止 SHMEM | ✓ | ✓ |

---

## 6. MTE vs SDMA 核心差异

| 特性 | MTE | SDMA |
|------|-----|------|
| **数据路径** | GM → UB → GM | GM → GM（直接） |
| **UB 作用** | 数据中转缓冲区 | 仅存储状态信息 |
| **UB 大小要求** | 通常 16KB | 64B 即可 |
| **同步方式** | SetFlag/WaitFlag | aclshmemx_sdma_quiet |
| **适用场景** | 大数据量传输 | 中小数据量传输 |
| **硬件位置** | AI Core 内部 | NPU 芯片级别 |
| **延迟** | 较高（两步搬运） | 较低（一步到位） |

---

## 7. 重要注意事项

1. **对称内存必须用 aclshmem_free 释放**
   - 不能使用 aclrtFree
   - 必须与 aclshmem_malloc 配对

2. **PUT 是单边操作**
   - 发送方主动调用 put_nbi
   - 接收方被动等待，不需要执行代码
   - 接收方用 barrier 确保数据到达

3. **P2P 必须启用**
   - 跨卡通信依赖 P2P 访问
   - 不启用 P2P 会经过 Host CPU 中转

4. **g_uid 必须是全局变量**
   - comm_args 指针指向的内存必须是全局/静态
   - 局部变量可能导致初始化失败

5. **SHMEM 初始化引擎与 kernel API 可不同**
   - 初始化用 MTE：建立对称内存映射
   - kernel 可调用 SDMA API：实际使用 SDMA 引擎传输

---

## 8. 带宽测试伪代码

```伪代码
// Host 端带宽测试
for (msg_size in [4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 8MB, 16MB]) {
    iters = get_iters(msg_size)      // 根据大小调整迭代次数
    
    // warmup：让硬件进入稳定状态
    launch_kernel(warmup_iters = 20)
    aclrtSynchronizeStream()
    
    // 正式测试计时
    t0 = chrono::now()
    launch_kernel(iters)
    aclrtSynchronizeStream()
    t1 = chrono::now()
    
    // 计算带宽
    if (pe == 0) {
        time_per_iter = (t1 - t0) / iters
        bandwidth = msg_size / time_per_iter / 1000  // GB/s
        print(msg_size, iters, time_per_iter, bandwidth)
    }
}
```