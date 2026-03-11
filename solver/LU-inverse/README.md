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
LAPACK vs Eigen vs Blockwise (2x2) 性能对比 (40x40 矩阵)

| 文件 | 方法 | 平均时间 (s) | 平均残差 |
|------|------|--------------|----------|
| benchmark1_1.mtx | LAPACK   | 0.000031 | 1.32e-13 |
| benchmark1_1.mtx | Eigen    | 0.000037 | 6.73e-14 |
| benchmark1_1.mtx | Blockwise| 0.000020 | 1.82e-13 |
| benchmark10_1.mtx | LAPACK   | 0.000021 | 1.39e-13 |
| benchmark10_1.mtx | Eigen    | 0.000017 | 7.12e-14 |
| benchmark10_1.mtx | Blockwise| 0.000015 | 1.55e-13 |
| benchmark1000_1.mtx | LAPACK   | 0.000020 | 1.41e-13 |
| benchmark1000_1.mtx | Eigen    | 0.000016 | 6.93e-14 |
| benchmark1000_1.mtx | Blockwise| 0.000015 | 1.80e-13 |
```

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
├── advanced_methods.c          # 高级求逆算法 (Newton-Schulz, Hyperpower)
├── main.c                      # 主测试程序
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

### 2. advanced_methods.c - 高级算法对比

**功能**: 对比多种迭代求逆算法

**测试算法**:
| 算法 | 描述 | 收敛阶 |
|------|------|--------|
| LAPACK | LU 分解 + 求逆 | 直接法 |
| Newton-Schulz | Xₖ₊₁ = Xₖ(I + Rₖ) | 2 阶 |
| Hyperpower (3 阶) | Xₖ₊₁ = Xₖ(I + Rₖ + Rₖ²) | 3 阶 |
| Hyperpower (5 阶) | Xₖ₊₁ = Xₖ(Σᵢ₌₀⁴ Rₖⁱ) | 5 阶 |

**运行方法**:
```bash
./run.sh
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

### 4. Newton-Schulz 迭代

**公式**: Xₖ₊₁ = Xₖ(I + Rₖ)，Rₖ = I - AXₖ

**初始猜测**: X₀ = αAᵀ，α = 1/(||A||₁·||A||∞)

**特点**: 需要 27-28 次迭代，小矩阵上不如直接法

---

## 测试结果

### 性能对比 (288 个矩阵)

| 方法 | 平均时间 | 相对速度 | 残差 | 速度获胜 |
|------|----------|----------|------|----------|
| **Blockwise (2×2)** | **0.000017s** | **1.4x 快** | **~1e-13** | **253/288 (87.8%)** |
| Eigen | 0.000023s | 1.05x 快 | ~7e-14 | - |
| LAPACK | 0.000024s | 基准 | ~1e-13 | 35/288 |
| Newton-Schulz | 0.000190s | 0.09x (慢 11 倍) | ~1e-12 | 0/288 |

### 方法选择建议

| 优先级 | 方法 | 速度 | 精度 | 推荐度 |
|--------|------|------|------|--------|
| 🥇 | **Blockwise (2×2)** | 最快 | 高 | ⭐⭐⭐ |
| 🥈 | Eigen | 快 | 最高 | ⭐⭐ |
| 🥉 | LAPACK | 快 | 高 | ⭐⭐ |
| ❌ | 迭代法 | 慢 | 中 | 不推荐 |

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

*文档更新时间：2026-03-11*
*测试平台：macOS (Apple M4), Accelerate Framework, Eigen 5.0.1*
