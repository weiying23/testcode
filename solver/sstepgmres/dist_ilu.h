// dist_ilu.h - 分布式 ILU0 预处理器
// ============================================================
// 本文件实现分布式不完全 LU 分解 (ILU0) 预处理器
//
// 核心概念：
// -----------
// 1. ILU0 (Incomplete LU Factorization, Level 0):
//    - LU 分解但保持原矩阵的稀疏结构
//    - 不引入新的填充 (fill-in)
//    - 对角线以下的元素归入 L，以上的归入 U
//    - L 的对角线隐含为 1 (不存储)
//
// 2. 分布式 ILU0 的特点:
//    - 每进程独立对其本地行做 ILU0
//    - 忽略跨进程的填充 (简化实现)
//    - 效果类似于 Block Jacobi 预处理器
//
// 3. Block Jacobi 效果:
//    - 完全 ILU0 应考虑全局填充
//    - 分布式版本忽略跨进程填充，相当于:
//      M = diag(M_0, M_1, ..., M_{np-1})
//      其中 M_p 是进程 p 的局部 ILU0
//    - 弱点: 边界行的处理不完整
//    - 优点: 数值更稳定，无需额外通信
//
// 4. 应用过程:
//    解 Mz = r:
//    - Forward solve: L * y = r
//    - Backward solve: U * z = y
//    - 完全本地操作，无需 MPI 通信
// ============================================================

#ifndef DIST_ILU_H
#define DIST_ILU_H

#include <vector>
#include <cmath>
#include "dist_matrix.h"

// ============================================================
// DistributedILU0 - 分布式 ILU0 预处理器类
// ============================================================
// 存储结构:
//   - rowptr_, colidx_: 复制矩阵的 CSR 结构
//   - lu_: LU 分解后的值 (L 和 U 混合存储)
//   - diag_idx_[i]: 第 i 行对角元在 CSR 中的位置
//
// LU 存储示意 (第 i 行 CSR 部分):
//   k < diag_idx_[i]: L 元素 (对角线以下)
//   k = diag_idx_[i]: U 的对角元
//   k > diag_idx_[i]: U 元素 (对角线以上)
//
// 注意:
//   L 的对角线元素不存储 (隐含为 1)
//   只有 U 存储对角元
// ============================================================
class DistributedILU0 {
    int n_local_;       // 本地行数
    int row_start_;     // 本地起始行索引

    std::vector<int> rowptr_;      // 行指针 (复制自矩阵)
    std::vector<int> colidx_;      // 列索引 (复制自矩阵)
    std::vector<double> lu_;       // LU 值 (分解后)
    std::vector<int> diag_idx_;    // 每行对角元位置

public:
    DistributedILU0() : n_local_(0), row_start_(0) {}

    // ============================================================
    // factorize() - 执行 ILU0 分解
    // ============================================================
    // 算法 (标准 ILU0):
    //   for i = 1, 2, ..., n-1:
    //     for k = rowptr[i] to diag_idx[i]-1 (L 部分):
    //       j = colidx[k]
    //       if j < i:
    //         a[i,j] = a[i,j] / a[j,j]      (L 元素归一化)
    //         for p = k+1 to rowptr[i+1]-1:
    //           a[i,p] -= a[i,j] * a[j,p]   (更新 U 部分)
    //
    // 分布式简化:
    //   - 只处理本地列 (colidx >= row_start)
    //   - 忽略跨进程填充
    //   - 这使得分解变为 Block Jacobi 形式
    //
    // 数学意义:
    //   完全 ILU0: M ≈ A, 更强的预处理器
    //   分布式 ILU0: M ≈ Block Jacobi(A), 较弱但无通信
    // ============================================================
    void factorize(const DistributedCSRMatrix& mat) {
        n_local_ = mat.n_local();
        row_start_ = mat.row_start();

        // 复制矩阵结构和值
        int nnz = mat.rowptr_[mat.n_local()];
        rowptr_ = mat.rowptr_;
        colidx_ = mat.colidx_;
        lu_ = mat.values_;

        // 找到每行对角元位置
        // 对角元: colidx[k] == row_start + i (本地行 i 的全局对角索引)
        diag_idx_.resize(n_local_);
        for (int i = 0; i < n_local_; i++) {
            for (int k = rowptr_[i]; k < rowptr_[i + 1]; k++) {
                if (colidx_[k] == row_start_ + i) {
                    diag_idx_[i] = k;
                    break;
                }
            }
        }

        // ILU0 分解 (仅本地元素)
        // ==================================
        // 关键: 分布式版本只处理 colidx >= row_start 的列
        // 这简化了实现，但牺牲了预处理器强度
        // ==================================
        for (int i = 1; i < n_local_; i++) {
            for (int k = rowptr_[i]; k < rowptr_[i + 1]; k++) {
                int j_global = colidx_[k];

                // 跳过非本地列 (这是分布式简化的关键)
                if (j_global < row_start_) continue;

                int j_local = j_global - row_start_;
                if (j_local >= i) break;  // 已进入 U 部分

                // L 元素归一化: a[i,j] /= a[j,j]
                double diag = lu_[diag_idx_[j_local]];
                if (std::abs(diag) > 1e-14) {
                    lu_[k] /= diag;
                }

                // 更新剩余元素
                // a[i,p] -= a[i,j] * a[j,p] for p > k
                for (int p = k + 1; p < rowptr_[i + 1]; p++) {
                    int col_p_global = colidx_[p];
                    if (col_p_global < row_start_) continue;

                    // 在行 j_local 中找匹配元素 a[j,p]
                    for (int q = rowptr_[j_local] + 1; q < rowptr_[j_local + 1]; q++) {
                        if (colidx_[q] == col_p_global) {
                            lu_[p] -= lu_[k] * lu_[q];
                            break;
                        }
                    }
                }
            }
        }
    }

    // ============================================================
    // apply() - 应用预处理器: z = M^{-1} * r
    // ============================================================
    // 解三角系统 Mz = r:
    //   1. Forward solve: L * y = r
    //      y[i] = r[i] - sum_{j<i} L[i,j] * y[j]
    //      (注意: L 的对角线隐含为 1)
    //
    //   2. Backward solve: U * z = y
    //      z[i] = (y[i] - sum_{j>i} U[i,j] * z[j]) / U[i,i]
    //
    // 分布式特点:
    //   - 完全本地操作，无需 MPI 通信
    //   - 边界行的预解效果较弱 (Block Jacobi 弱点)
    //   - 对大多数问题足够有效
    //
    // 输入:
    //   r_local: 本地残差向量
    // 输出:
    //   z_local: 本地预处理后向量
    // ============================================================
    void apply(const double* r_local, double* z_local) {
        std::vector<double> y(n_local_);

        // ==================================
        // Forward solve: L * y = r
        // ==================================
        // L 元素: rowptr[i] 到 diag_idx[i]-1
        // L[i,j] = lu_[k] where colidx[k] = j
        // L 的对角线为 1 (不存储)
        // ==================================
        for (int i = 0; i < n_local_; i++) {
            double sum = r_local[i];
            for (int k = rowptr_[i]; k < diag_idx_[i]; k++) {
                int j_global = colidx_[k];
                if (j_global >= row_start_) {
                    sum -= lu_[k] * y[j_global - row_start_];
                }
            }
            y[i] = sum;  // L 的对角线为 1，无需除法
        }

        // ==================================
        // Backward solve: U * z = y
        // ==================================
        // U 元素: diag_idx[i] 到 rowptr[i+1]-1
        // U[i,i] = lu_[diag_idx[i]]
        // U[i,j] = lu_[k] for k > diag_idx[i]
        // ==================================
        for (int i = n_local_ - 1; i >= 0; i--) {
            double sum = y[i];
            for (int k = diag_idx_[i] + 1; k < rowptr_[i + 1]; k++) {
                int j_global = colidx_[k];
                if (j_global >= row_start_ && j_global <= row_start_ + n_local_ - 1) {
                    sum -= lu_[k] * z_local[j_global - row_start_];
                }
            }
            double diag = lu_[diag_idx_[i]];
            if (std::abs(diag) > 1e-14) {
                z_local[i] = sum / diag;
            } else {
                z_local[i] = 0.0;  // 防止除零
            }
        }
    }

    int n_local() const { return n_local_; }
};

#endif // DIST_ILU_H