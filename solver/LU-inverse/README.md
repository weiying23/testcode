# LU-inverse 矩阵求逆性能测试

## 快速开始

### 编译和运行

```bash
cd /Users/yingwei/Documents/code/testcode/solver/LU-inverse

# 编译三种方法对比程序
OMP_NUM_THREADS=8 VECLIB_MAX_THREADS=8 \
clang++ -O3 -mcpu=apple-m4 \
    -Xpreprocessor -fopenmp \
    -I/opt/homebrew/opt/libomp/include \
    -L/opt/homebrew/opt/libomp/lib -lomp \
    -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 \
    -framework Accelerate -lm \
    -I/opt/homebrew/include/eigen3 \
    final_compare.cpp -o final_compare

# 运行
./final_compare
```

### 输出示例

```
LAPACK vs Eigen vs Blockwise 性能对比
每矩阵重复测试 1000 次

| 文件 | 方法 | 时间 (s) | 残差 |
|------|------|---------|------|
| benchmark1_1.mtx | LAPACK    | 0.000021 | 1.32e-13 |
| benchmark1_1.mtx | Eigen     | 0.000016 | 6.73e-14 |
| benchmark1_1.mtx | Blockwise | 0.000015 | 1.82e-13 |
| benchmark10_1.mtx | LAPACK    | 0.000021 | 1.39e-13 |
| benchmark10_1.mtx | Eigen     | 0.000016 | 7.12e-14 |
| benchmark10_1.mtx | Blockwise | 0.000015 | 1.55e-13 |
| benchmark1000_1.mtx | LAPACK    | 0.000021 | 1.41e-13 |
| benchmark1000_1.mtx | Eigen     | 0.000016 | 6.93e-14 |
| benchmark1000_1.mtx | Blockwise | 0.000015 | 1.80e-13 |
```

**性能总结** (40×40 矩阵，288 个样本，每矩阵 1000 次迭代):
- **Eigen**: 0.000016s (精度最高 ~7e-14)
- **Blockwise**: 0.000015s (比 Eigen 快 7%，残差 ~1e-13)
- **LAPACK**: 0.000021s (残差 ~1e-13)

---

## 目录

1. [项目结构](#项目结构)
2. [测试程序说明](#测试程序说明)
3. [算法说明](#算法说明)
4. [测试结果](#测试结果)
5. [分析报告](#分析报告)

---

## 项目结构

```
LU-inverse/
├── final_compare.cpp           # 三种方法对比 (LAPACK/Eigen/Blockwise)
├── invertible_perf_test.c      # LAPACK vs Blockwise 对比
├── check_invertible.c          # 矩阵可逆性检测
├── run.sh                      # 编译和运行脚本
├── benchmark*_1.mtx            # 测试矩阵文件 (3 文件 × 96 矩阵)
│
├── README.md                   # 本文件
├── matrix-analysis.md          # 矩阵特性分析报告
└── invertible-analysis.md      # 矩阵可逆性分析
```

---

## 测试程序说明

### 1. final_compare.cpp - 三种方法对比

**功能**: 对比 LAPACK、Eigen 和 Blockwise (2×2) 三种求逆方法

**对比算法**:
| 方法 | 描述 | 精度 |
|------|------|------|
| LAPACK | `dgetrf_` + `dgetri_` | ~1e-13 |
| Eigen | `Matrix::inverse()` | ~7e-14 |
| Blockwise | 2×2 分块 + Schur 补 | ~1e-13 |

**运行方法**:
```bash
./run.sh   # 或手动编译 final_compare.cpp
./final_compare
```

---

## 算法说明

### 1. LAPACK 直接法

```cpp
// LU 分解
dgetrf_(&n, &n, A, &lda, ipiv, &info);

// 求逆
dgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
```

**特点**: 稳定可靠，macOS Accelerate 框架高度优化

### 2. Eigen 库

```cpp
Eigen::Map<Eigen::MatrixXd> mat(A, n, n);
MatrixXd A_inv = mat.inverse();
```

**特点**: C++ 模板库，代码简洁，精度最高

### 3. Blockwise (2×2) 分块求逆

**舒尔补公式**:
```
[A B]⁻¹   [A⁻¹ + A⁻¹BS⁻¹CA⁻¹   -A⁻¹BS⁻¹]
[C D]   = [-S⁻¹CA⁻¹            S⁻¹     ]
```
其中 S = D - CA⁻¹B 为 Schur 补

**步骤**:
1. 将 n×n 矩阵分为 4 个 (n/2)×(n/2) 子块
2. 用 LAPACK 求 A11⁻¹ 和 S⁻¹
3. 计算四个结果分块
4. 组装结果

**特点**: 分治策略，缓存效率高，速度最快

---

## 测试结果

### 40×40 矩阵性能对比 (288 个矩阵，每矩阵 1000 次)

| 方法 | 平均时间 | 相对速度 | 残差 |
|------|----------|----------|------|
| **Blockwise** | **0.000015s** | **1.07x 快** | **~1e-13** |
| Eigen | 0.000016s | - | **~7e-14** |
| LAPACK | 0.000021s | 1.32x | ~1e-13 |

### 不同规模矩阵性能对比

| 规模 | Eigen | Blockwise | 加速比 (Blockwise/Eigen) |
|------|-------|-----------|-------------------------|
| 40×40 | 0.000042s | 0.000032s | **1.31x** |
| 100×100 | 0.000202s | 0.000147s | **1.37x** |
| 200×200 | 0.002111s | 0.000443s | **4.77x** |
| 500×500 | 0.019459s | 0.004374s | **4.45x** |

**关键发现**：
- **小规模 (n≤40)**：Eigen 和 Blockwise 性能相当，差距 <10%
- **中规模 (n≈100)**：Blockwise 开始领先，快 1.4x
- **大规模 (n≥200)**：Blockwise 优势显著，快 **4.5x+**

### 方法选择建议

| 矩阵规模 | 推荐方法 | 理由 |
|----------|---------|------|
| n ≤ 50 | Eigen | 代码简洁，精度最高，性能相当 |
| 50 < n < 200 | Blockwise | 性能开始领先，快 1.3-1.5x |
| n ≥ 200 | **Blockwise** | 性能显著领先，快 4x+ |

---

## 分析报告

| 文件 | 内容 |
|------|------|
| `matrix-analysis.md` | 矩阵特性分析 + 三种方法性能对比 |
| `invertible-analysis.md` | 矩阵可逆性判定和占比分析 |

---

## 编译选项说明

```bash
# 标准编译选项 (推荐)
clang++ -O3 -mcpu=apple-m4 \
    -Xpreprocessor -fopenmp \
    -I/opt/homebrew/opt/libomp/include \
    -L/opt/homebrew/opt/libomp/lib -lomp \
    -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 \
    -framework Accelerate -lm \
    -I/opt/homebrew/include/eigen3 \
    source.cpp -o program

# 设置线程数
export OMP_NUM_THREADS=8
export VECLIB_MAX_THREADS=8
```

---

*文档更新时间：2026-03-12*
*测试平台：macOS (Apple M4), Accelerate Framework, Eigen 5.0.1*
