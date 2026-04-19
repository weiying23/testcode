// dist_matrix.h - 分布式 CSR 稀疏矩阵
// ============================================================
// 本文件定义分布式稀疏矩阵的存储和操作
//
// 核心概念：
// -----------
// 1. 行分区 (Row Partition):
//    - 全局矩阵按行分给各进程
//    - 进程 p 负责 rows [row_start, row_end]
//    - 每进程只存储自己负责的行 (CSR 格式)
//
// 2. CSR 格式 (Compressed Sparse Row):
//    - rowptr[i]: 第 i 行的起始位置
//    - colidx[k]: 第 k 个非零元的全局列索引
//    - values[k]: 第 k 个非零元的值
//    - 第 i 行的非零元: rowptr[i] 到 rowptr[i+1]-1
//
// 3. 幽灵点 (Ghost Points):
//    - 本地行可能引用其他进程的列
//    - 这些"外部"列称为幽灵点
//    - 矩阵-向量乘法前需通过幽灵层交换获取这些数据
//
// 4. 五对角矩阵:
//    - 来自二维 Poisson 问题离散化
//    - 每行最多 5 个非零元: 对角、左、右、上、下
//    - 网格 (nx × nx), 总维度 n = nx²
// ============================================================

#ifndef DIST_MATRIX_H
#define DIST_MATRIX_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <mpi.h>
#include "dist_vector.h"
#include "blas_utils.h"

// ============================================================
// DistributedCSRMatrix - 分布式 CSR 稀疏矩阵类
// ============================================================
// 存储结构:
//   - rowptr_: 本地行的指针数组 (大小 n_local+1)
//   - colidx_: 全局列索引 (大小 nnz)
//   - values_: 非零值 (大小 nnz)
//
// 幽灵点映射:
//   - ghost_global_idx_: 幽灵点的全局索引列表
//   - ghost_local_map_: 每个非零元对应的幽灵层位置
//     (若列是本地列，值为 -1；若列是幽灵点，值为其在幽灵层中的位置)
//
// 一维分区示意:
//   进程 0: rows [0, 99]
//   进程 1: rows [100, 199]
//   进程 2: rows [200, 299]
//   ...
//
// 幽灵点示例 (进程 1):
//   本地行 row=100 可能引用列 99 (来自进程 0)
//   列 99 就是进程 1 的幽灵点
// ============================================================
class DistributedCSRMatrix {
    friend class DistributedILU0;

    MPI_Comm comm_;
    int rank_, nprocs_;

    int n_global_;      // 全局矩阵维度
    int n_local_;       // 本地行数
    int row_start_;     // 本地起始行索引 (全局)
    int row_end_;       // 本地结束行索引 (全局)

    // CSR 存储 (仅本地行)
    std::vector<int> rowptr_;    // 本地行指针
    std::vector<int> colidx_;    // 全局列索引
    std::vector<double> values_; // 非零值

    // 幽灵点映射
    std::vector<int> ghost_global_idx_;  // 幽灵点的全局索引
    std::vector<int> ghost_local_map_;   // CSR 元素到幽灵层的映射

    // Halo 信息
    int neighbor_left_, neighbor_right_;  // 左/右邻居进程编号
    int n_send_left_, n_send_right_;      // 发送数据量
    int n_recv_left_, n_recv_right_;      // 接收数据量
    std::vector<int> send_idx_left_, send_idx_right_;  // 发送索引

public:
    DistributedCSRMatrix() : comm_(MPI_COMM_NULL), rank_(0), nprocs_(0),
                              n_global_(0), n_local_(0), row_start_(0), row_end_(-1),
                              neighbor_left_(-1), neighbor_right_(-1),
                              n_send_left_(0), n_send_right_(0),
                              n_recv_left_(0), n_recv_right_(0) {}

    int n_global() const { return n_global_; }
    int n_local() const { return n_local_; }
    int n_ghost() const { return ghost_global_idx_.size(); }
    int row_start() const { return row_start_; }
    int row_end() const { return row_end_; }

    // ============================================================
    // init() - 初始化 MPI 通信器并计算行分区
    // ============================================================
    // 分区策略:
    //   基础大小: base = n_global / nprocs
    //   余数分配: 前 remainder 个进程各多分 1 行
    //
    // 示例 (n=100, np=4):
    //   base=25, remainder=0
    //   进程 0: rows 0-24 (25行)
    //   进程 1: rows 25-49 (25行)
    //   进程 2: rows 50-74 (25行)
    //   进程 3: rows 75-99 (25行)
    //
    // 示例 (n=100, np=3):
    //   base=33, remainder=1
    //   进程 0: rows 0-33 (34行, 多分1行)
    //   进程 1: rows 34-66 (33行)
    //   进程 2: rows 67-99 (33行)
    // ============================================================
    void init(MPI_Comm comm, int n_global) {
        comm_ = comm;
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &nprocs_);

        n_global_ = n_global;

        // 计算本地分区
        int base = n_global_ / nprocs_;
        int remainder = n_global_ % nprocs_;

        if (rank_ < remainder) {
            n_local_ = base + 1;
            row_start_ = rank_ * (base + 1);
        } else {
            n_local_ = base;
            row_start_ = remainder * (base + 1) + (rank_ - remainder) * base;
        }
        row_end_ = row_start_ + n_local_ - 1;

        // 确定邻居进程
        neighbor_left_ = (rank_ > 0) ? rank_ - 1 : -1;
        neighbor_right_ = (rank_ < nprocs_ - 1) ? rank_ + 1 : -1;
    }

    // 设置 Halo 交换对象 (矩阵构建后调用)
    void setupHalo(HaloExchange& halo) {
        halo.init(comm_, n_send_left_, n_send_right_,
                  n_recv_left_, n_recv_right_,
                  neighbor_left_, neighbor_right_);
        halo.setSendIndices(send_idx_left_, send_idx_right_);
    }

    // ============================================================
    // buildFiveDiagonal() - 构建五对角矩阵
    // ============================================================
    // 来自二维网格 Poisson 问题:
    //   -∇²u = f  在 (nx × nx) 网格上离散化
    //
    // 五点差分格式:
    //   4*u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1] = h²*f[i,j]
    //
    // 网格索引到矩阵索引:
    //   i_global = iy * nx + ix
    //   (其中 ix, iy 为网格坐标)
    //
    // 非零元位置:
    //   - 对角: 列 = i_global (自身)
    //   - 左邻居: 列 = i_global - 1 (ix > 0 时)
    //   - 右邻居: 列 = i_global + 1 (ix < nx-1 时)
    //   - 上邻居: 列 = i_global - nx (iy > 0 时)
    //   - 下邻居: 列 = i_global + nx (iy < nx-1 时)
    //
    // 参数:
    //   diag_val: 对角元值 (通常为 4.0 或 2+2ε)
    //   offdiag_val: 非对角元值 (通常为 -1.0)
    // ============================================================
    void buildFiveDiagonal(double diag_val, double offdiag_val) {
        int nx = static_cast<int>(std::sqrt(static_cast<double>(n_global_)));

        rowptr_.resize(n_local_ + 1, 0);
        std::vector<std::vector<std::pair<int, double>>> rows(n_local_);

        for (int i_local = 0; i_local < n_local_; i_local++) {
            int i_global = row_start_ + i_local;
            rows[i_local].push_back({i_global, diag_val});

            // 左邻居 (同一行前一列)
            if (i_global > 0) {
                rows[i_local].push_back({i_global - 1, offdiag_val});
            }
            // 右邻居 (同一行后一列)
            if (i_global < n_global_ - 1) {
                rows[i_local].push_back({i_global + 1, offdiag_val});
            }
            // 上邻居 (上一行同列)
            if (i_global >= nx) {
                rows[i_local].push_back({i_global - nx, offdiag_val});
            }
            // 下邻居 (下一行同列)
            if (i_global + nx < n_global_) {
                rows[i_local].push_back({i_global + nx, offdiag_val});
            }

            // 按列索引排序 (CSR 格式要求)
            std::sort(rows[i_local].begin(), rows[i_local].end());
            rowptr_[i_local + 1] = rowptr_[i_local] + rows[i_local].size();
        }

        // 构建 CSR 数组
        int nnz = rowptr_[n_local_];
        colidx_.resize(nnz);
        values_.resize(nnz);

        int idx = 0;
        for (int i_local = 0; i_local < n_local_; i_local++) {
            for (auto& p : rows[i_local]) {
                colidx_[idx] = p.first;
                values_[idx++] = p.second;
            }
        }

        // 识别幽灵点并设置 Halo
        identifyGhostPoints();
    }

    // ============================================================
    // buildAnisotropic() - 构建各向异性扩散矩阵
    // ============================================================
    // 各向异性 Poisson 问题:
    //   -ε * ∂²u/∂x² - ∂²u/∂y² = f
    //
    // 离散化:
    //   (2ε+2)u[i,j] - ε*u[i-1,j] - ε*u[i+1,j] - u[i,j-1] - u[i,j+1] = h²f
    //
    // 非零元值:
    //   - 对角: 2.0 + 2.0*eps
    //   - x方向邻居: -eps
    //   - y方向邻居: -1.0
    //
    // 参数 eps 的意义:
    //   eps=1: 各向同性 (标准 Poisson)
    //   eps<1: y方向扩散更强 (垂直条纹)
    //   eps>1: x方向扩散更强 (水平条纹)
    //
    // 收敛难度:
    //   eps 过小 (如 0.01) 时矩阵条件数很差，收敛困难
    // ============================================================
    void buildAnisotropic(double eps) {
        int nx = static_cast<int>(std::sqrt(static_cast<double>(n_global_)));

        rowptr_.resize(n_local_ + 1, 0);
        std::vector<std::vector<std::pair<int, double>>> rows(n_local_);

        for (int i_local = 0; i_local < n_local_; i_local++) {
            int i_global = row_start_ + i_local;
            int ix = i_global % nx;
            int iy = i_global / nx;

            // 对角元
            rows[i_local].push_back({i_global, 2.0 + 2.0 * eps});

            // x方向邻居 (系数为 -eps)
            if (ix > 0) {
                rows[i_local].push_back({i_global - 1, -eps});
            }
            if (ix < nx - 1) {
                rows[i_local].push_back({i_global + 1, -eps});
            }

            // y方向邻居 (系数为 -1.0)
            if (iy > 0) {
                rows[i_local].push_back({i_global - nx, -1.0});
            }
            if (iy < nx - 1) {
                rows[i_local].push_back({i_global + nx, -1.0});
            }

            std::sort(rows[i_local].begin(), rows[i_local].end());
            rowptr_[i_local + 1] = rowptr_[i_local] + rows[i_local].size();
        }

        // 构建 CSR 数组
        int nnz = rowptr_[n_local_];
        colidx_.resize(nnz);
        values_.resize(nnz);

        int idx = 0;
        for (int i_local = 0; i_local < n_local_; i_local++) {
            for (auto& p : rows[i_local]) {
                colidx_[idx] = p.first;
                values_[idx++] = p.second;
            }
        }

        // 识别幽灵点
        identifyGhostPoints();
    }

    // ============================================================
    // mv() - 分布式矩阵-向量乘法
    // ============================================================
    // 计算: y_local = A * x
    //
    // 输入:
    //   x_local: 本进程拥有的 x 分量
    //   x_ghost: 幽灵层数据 (通过 HaloExchange 获得)
    //
    // 输出:
    //   y_local: 结果向量 (仅本地部分)
    //
    // 算法:
    //   for each local row i:
    //     y[i] = sum_{k in row i} A[i,k] * x[k]
    //     if k is local: use x_local[k - row_start]
    //     if k is ghost: use x_ghost[ghost_local_map[k]]
    //
    // 幽灵点索引:
    //   ghost_local_map_[k] 给出 CSR 位置 k 对应的幽灵层位置
    //   若 k 是本地列，ghost_local_map_[k] = -1
    // ============================================================
    void mv(const double* x_local, const double* x_ghost, double* y_local) {
        for (int i_local = 0; i_local < n_local_; i_local++) {
            double sum = 0.0;
            for (int k = rowptr_[i_local]; k < rowptr_[i_local + 1]; k++) {
                int j_global = colidx_[k];

                double x_val;
                if (j_global >= row_start_ && j_global <= row_end_) {
                    // 本地列: 直接使用 x_local
                    x_val = x_local[j_global - row_start_];
                } else {
                    // 幽灵点列: 使用 x_ghost
                    // ghost_local_map_[k] 给出幽灵层位置
                    int ghost_pos = ghost_local_map_[k];
                    x_val = x_ghost[ghost_pos];
                }
                sum += values_[k] * x_val;
            }
            y_local[i_local] = sum;
        }
    }

private:
    // ============================================================
    // identifyGhostPoints() - 识别幽灵点并建立映射
    // ============================================================
    // 扫描所有本地行的列索引:
    //   - 若列索引在 [row_start, row_end] 外，则为幽灵点
    //   - 使用 unordered_map 避免重复记录同一幽灵点
    //
    // 建立两个映射:
    //   1. ghost_global_idx_: 幽灵点的全局索引列表
    //   2. ghost_local_map_: CSR 每个元素对应的幽灵层位置
    //
    // 发送/接收设置:
    //   - 幽灵点 < row_start: 来自左邻居
    //   - 幽灵点 > row_end: 来自右邻居
    //   - 发送数据: 边界行的数据 (邻居需要)
    //
    // 一维分区的特殊情况:
    //   对于五对角矩阵，幽灵点只来自直接邻居
    //   发送: 本进程的第一行和最后一行
    // ============================================================
    void identifyGhostPoints() {
        ghost_global_idx_.clear();
        ghost_local_map_.clear();
        send_idx_left_.clear();
        send_idx_right_.clear();

        // 使用哈希表避免重复幽灵点
        std::unordered_map<int, int> ghost_map;

        int ghost_counter = 0;
        for (int i_local = 0; i_local < n_local_; i_local++) {
            for (int k = rowptr_[i_local]; k < rowptr_[i_local + 1]; k++) {
                int j_global = colidx_[k];

                // 检查列是否在本地范围外
                if (j_global < row_start_ || j_global > row_end_) {
                    if (ghost_map.find(j_global) == ghost_map.end()) {
                        // 新幽灵点: 记录并分配位置
                        ghost_map[j_global] = ghost_counter;
                        ghost_global_idx_.push_back(j_global);
                        ghost_counter++;
                    }
                    ghost_local_map_.push_back(ghost_map[j_global]);
                } else {
                    // 本地列: 标记为 -1
                    ghost_local_map_.push_back(-1);
                }
            }
        }

        // 统计来自左/右邻居的幽灵点数
        n_recv_left_ = 0;
        n_recv_right_ = 0;
        for (int g : ghost_global_idx_) {
            if (g < row_start_) n_recv_left_++;
            else if (g > row_end_) n_recv_right_++;
        }

        // 设置发送数据 (边界行)
        // 对于一维分区: 左邻居需要本进程第一行，右邻居需要最后一行
        if (neighbor_left_ >= 0) {
            send_idx_left_.push_back(0);  // 第一行
            n_send_left_ = 1;
        } else {
            n_send_left_ = 0;
        }

        if (neighbor_right_ >= 0) {
            send_idx_right_.push_back(n_local_ - 1);  // 最后一行
            n_send_right_ = 1;
        } else {
            n_send_right_ = 0;
        }
    }
};

#endif // DIST_MATRIX_H