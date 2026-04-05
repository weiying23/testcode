---
title: Distributed MPI s-step GMRES Implementation Design
date: 2026-04-05
---

# 分布式MPI s-step GMRES实现设计

## 目标

将当前冗余存储的s-step GMRES实现改为真正的分布式MPI版本，实现：
- 行分区矩阵存储
- 分布式向量 + 幽灵层
- 非阻塞边界通信
- 与冗余版本对比验证性能提升

## 参数约束

| 参数 | 值 |
|-----|---|
| 问题规模 | n = 100000 |
| 最大进程数 | np = 10 |
| 收敛容忍度 | tol = 1e-8 |

## 支持矩阵类型

1. 五对角矩阵 (Five-diagonal) - 2D Poisson五点差分
2. 各向异性扩散矩阵 (Anisotropic) - eps=0.01
3. 通用CSR稀疏矩阵

---

## 第一部分：分布式矩阵类

### DistributedCSRMatrix

```cpp
class DistributedCSRMatrix {
    int n_global;      // 全局矩阵维度
    int n_local;       // 本地行数 (≈ n_global / nprocs)
    int row_start;     // 本地起始行号 (rank * n_local)
    int row_end;       // 本地结束行号 (row_start + n_local - 1)

    // CSR存储 (只存本地行)
    std::vector<int> rowptr;    // 本地行的行指针
    std::vector<int> colidx;    // 列索引 (全局列号)
    std::vector<double> values; // 非零值

    // 边界信息
    std::vector<int> send_indices;  // 需发送给邻居的本地索引
    std::vector<int> recv_indices;  // 需从邻居接收的全局行号映射

    int neighbor_left;   // 左邻居进程号 (-1表示无)
    int neighbor_right;  // 右邻居进程号 (-1表示无)
    int n_send_left, n_send_right;
    int n_recv_left, n_recv_right;

    // 构建方法
    void buildFiveDiagonal(int nx, double diag, double offdiag);
    void buildAnisotropic(int nx, double eps);
    void loadCSR(int n_global, int* rowptr_g, int* colidx_g, double* values_g);

    // Mat-vec
    void mv(const double* x_local, const double* x_ghost, double* y_local);

    // 边界信息初始化
    void setupHalo(HaloExchange& halo);
};
```

### 行分区策略

采用连续分区：
```
n_global = 100000, np = 10:
  进程0: row_start=0,    row_end=9999,   n_local=10000
  进程1: row_start=10000, row_end=19999, n_local=10000
  ...
  进程9: row_start=90000, row_end=99999, n_local=10000
```

处理非整除情况：
```cpp
n_local = n_global / nprocs;
remainder = n_global % nprocs;
if (rank < remainder) n_local++;  // 前remainder个进程多分配一行
```

### colidx使用全局列号

原因：便于索引幽灵层数据
```
本地行i的非零元:
  colidx[j] = 全局列号k

若 k 在本地范围 [row_start, row_end]:
  使用 x_local[k - row_start]
否则:
  使用 x_ghost[映射索引]
```

---

## 第二部分：分布式向量与幽灵层

### DistributedVector

```cpp
class DistributedVector {
    int n_local;           // 本地分量数
    int n_ghost;           // 幽灵层分量数

    std::vector<double> data;  // [本地 | 幽灵层]

    // 访问
    double& local(int i) { return data[i]; }
    double& ghost(int i) { return data[n_local + i]; }
    const double* local_data() const { return data.data(); }
    const double* ghost_data() const { return data.data() + n_local; }

    // 本地BLAS操作
    void zero();
    void copyFromLocal(const double* src);
    double dotLocal(const DistributedVector& other) const;  // 本地部分点积
};
```

### HaloExchange

```cpp
class HaloExchange {
    MPI_Comm comm;
    int rank, nprocs;

    int neighbor_left, neighbor_right;
    int n_send_left, n_send_right;
    int n_recv_left, n_recv_right;

    std::vector<double> send_buf_left, send_buf_right;
    std::vector<double> recv_buf_left, recv_buf_right;
    std::vector<int> send_idx_left, send_idx_right;  // 发送数据的本地索引

    MPI_Request req_send[2], req_recv[2];

    // 初始化
    void init(int n_left, int n_right, int left_rank, int right_rank);

    // 开始交换 (非阻塞)
    void start_exchange(DistributedVector& vec);

    // 等待完成
    void wait_exchange(DistributedVector& vec);
};
```

### 幽灵层交换流程

```
HaloExchange::start_exchange(vec):
  1. MPI_Irecv(recv_buf_left, neighbor_left)
  2. MPI_Irecv(recv_buf_right, neighbor_right)
  3. 从vec.local打包数据到send_buf
  4. MPI_Isend(send_buf_left, neighbor_left)
  5. MPI_Isend(send_buf_right, neighbor_right)

HaloExchange::wait_exchange(vec):
  1. MPI_Wait(req_recv_left)
  2. MPI_Wait(req_recv_right)
  3. 将recv_buf复制到vec.ghost
  4. MPI_Wait(req_send_left)  // 确保发送完成
  5. MPI_Wait(req_send_right)
```

---

## 第三部分：分布式ILU0预处理

### DistributedILU0

```cpp
class DistributedILU0 {
    int n_local;

    std::vector<int> rowptr;
    std::vector<int> colidx;
    std::vector<double> lu;

    bool factored;

    void factorize(const DistributedCSRMatrix& mat);
    void apply(const double* r_local, const double* r_ghost, double* z_local);
};
```

### 简化局部ILU0方案

每进程独立进行局部ILU0分解，忽略跨进程填充：
- 实现简单
- 边界行预处理效果略降
- 预计迭代次数增加10-30%

分解流程 (factorize):
```
for i = row_start to row_end:
  for k = rowptr[i] to rowptr[i+1]:
    if colidx[k] < row_start:  // 跨进程的填充，跳过
      continue
    // 本地ILU分解...
```

应用流程 (apply):
```
// L求解 (下三角)
for i = 0 to n_local-1:
  z[i] = r[i]
  for j = rowptr[i] to diag[i]:
    if colidx[j] >= row_start:
      z[i] -= lu[j] * z[colidx[j] - row_start]

// U求解 (上三角)
for i = n_local-1 to 0:
  for j = diag[i]+1 to rowptr[i+1]:
    if colidx[j] >= row_start:
      z[i] -= lu[j] * z[colidx[j] - row_start]
  z[i] /= lu[diag[i]]
```

---

## 第四部分：分布式s-step GMRES

### 主程序结构

基于 `sstep_gmres_paper.cpp`，修改关键部分：

1. **矩阵构建**:
```cpp
// 原代码
CSRMatrix A(n);
A.buildFiveDiagonal(...);

// 分布式代码
DistributedCSRMatrix A;
A.init(MPI_COMM_WORLD, n_global);
A.buildFiveDiagonal(nx, diag, offdiag);
```

2. **向量操作**:
```cpp
// 原代码
std::vector<double> r(n), z(n);

// 分布式代码
DistributedVector r(A.n_local, A.n_ghost);
DistributedVector z(A.n_local, A.n_ghost);
```

3. **Mat-vec**:
```cpp
// 原代码
A.mv(x.data(), Atmp.data());

// 分布式代码
halo.start_exchange(x);
halo.wait_exchange(x);
A.mv(x.local_data(), x.ghost_data(), Atmp.local_data());
```

4. **内积 (Allreduce)**:
```cpp
// 原代码
double loc = vdot(n, a, b);
MPI_Allreduce(&loc, &glob, 1, ...);

// 分布式代码
double loc = vdot(A.n_local, a.local_data(), b.local_data());
MPI_Allreduce(&loc, &glob, 1, ...);
```

### 每块通信模式

```
Block k:
  1. 幽灵层交换 (边界通信): s次
  2. 内积Allreduce (全局通信): 1次

总通信:
  - 边界交换: m × s 次 (每块s次matvec)
  - Allreduce: m+1 次 (与冗余版本相同)
```

---

## 第五部分：测试与验证

### 测试计划

**阶段1: 单进程验证 (np=1)**
```
目的: 确保分布式版本在单进程时正确
方法: 与冗余版本对比结果
测试: n=400, s=3, m=15, tol=1e-8
预期: 收敛残差相同
```

**阶段2: 多进程正确性 (np=4)**
```
目的: 验证幽灵层交换和内积正确
方法: 与冗余版本对比收敛曲线
测试: n=400, s=3, m=15, tol=1e-8
预期: 收敛曲线一致
```

**阶段3: 大规模性能 (np=10)**
```
目的: 测量真实性能提升
测试: n=100000, s=3, m=30, tol=1e-8
测量:
  - 总时间
  - 计算时间分解
  - 通信时间分解
预期: 分布式比冗余快20-30x
```

### 性能对比表

| 测试配置 | 冗余存储时间 | 分布式时间 | 加速比 |
|---------|-------------|-----------|-------|
| np=1, n=400 | 0.15ms | 0.15ms | 1x |
| np=4, n=400 | 0.21ms | ~0.10ms | ~2x |
| np=10, n=100000 | 1.0s | ~0.03-0.05s | 20-30x |

### 成功标准

1. 分布式版本在np≥2时比冗余版本更快
2. 收敛残差达到1e-8
3. 通信次数保持m+1次Allreduce
4. 边界交换正确，无数据错误

---

## 文件结构

```
sstepgmres/
├── dist_matrix.h      // DistributedCSRMatrix
├── dist_vector.h      // DistributedVector + HaloExchange
├── dist_ilu.h         // DistributedILU0
├── sstep_gmres_dist.cpp // 分布式GMRES主程序
├── test_dist.sh       // 分布式测试脚本
```

---

## 实现优先级

1. **P0 (必须)**: DistributedCSRMatrix + 五对角构建
2. **P0 (必须)**: HaloExchange + 幽灵层交换
3. **P0 (必须)**: DistributedVector + 本地BLAS
4. **P0 (必须)**: 分布式s-step GMRES主程序
5. **P1 (重要)**: DistributedILU0
6. **P2 (可选)**: 各向异性矩阵构建
7. **P2 (可选)**: 通用CSR加载