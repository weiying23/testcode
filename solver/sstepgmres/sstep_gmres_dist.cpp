//==============================================================================
// sstep_gmres_dist.cpp - 分布式 s-step GMRES 求解器
//
// 基于论文: arXiv:2001.04886v2 - "s-Step Orthomin and GMRES implemented on parallel computers"
//
// 核心思想:
//   将 s 次 GMRES 迭代合并为一个"块"迭代，减少全局通信次数
//   - 传统 GMRES: 每个 Krylov 向量需要 ~k 次全局通信 (逐步正交化)
//   - s-step GMRES: 每 s 个向量只需要 1 次全局通信 (打包所有内积)
//
// 文件结构:
//   - sstepGMRES(): 纯算法函数，不涉及问题构建
//   - main(): 问题构建、预条件计算、调用 GMRES
//
// 作者: Claude
// 日期: 2026-04-07
//==============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <mpi.h>
#include <functional>

// macOS 使用 Accelerate 框架进行 LAPACK 调用
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern "C" {
void dposv_(char* uplo, int* n, int* nrhs, double* a, int* lda, double* b, int* ldb, int* info);
}
#endif

#include "blas_utils.h"
#include "dist_vector.h"
#include "dist_matrix.h"
#include "dist_ilu.h"

//==============================================================================
// GMRES 结果结构体
//==============================================================================
struct GMRESResult {
    bool converged;           // 是否收敛
    double final_residual;    // 最终相对残差 ||b-Ax||/||b||
    int iterations;           // 总迭代块数（所有 restart 轮次累计）
    int communications;       // 总通信次数
    int restarts_used;        // 实际使用的 restart 次数

    GMRESResult() : converged(false), final_residual(0.0),
                    iterations(0), communications(0), restarts_used(0) {}
};

//==============================================================================
// GMRES 参数结构体
//==============================================================================
struct GMRESParams {
    int s;          // s-step 参数 (推荐 2-3)
                    // - s=2: 最稳定，但收敛可能慢
                    // - s=3: 推荐，收敛快且通信少
                    // - s≥4: 可能数值不稳定
    int m;          // 每轮最大块数，总 Krylov 维度 = s * m
    double tol;     // 收敛容忍度 (相对残差)
    int max_restarts;   // 最大 restart 次数

    GMRESParams(int s_val = 3, int m_val = 10, double tol_val = 1e-8, int max_rst = 25)
        : s(s_val), m(m_val), tol(tol_val), max_restarts(max_rst) {}
};

//==============================================================================
// 全局内积计算 (需要 MPI_Allreduce)
//
// 在分布式环境下，每个进程只有部分数据
// 全局内积 = Σ(各进程的本地内积) / nprocs
// 注意: 除以 nprocs 是因为本地向量已经归一化
//==============================================================================
double globalDot(MPI_Comm comm, int nprocs, const double* a, const double* b, int n) {
    double local_sum = vdot(n, a, b);
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_sum / nprocs;
}

//==============================================================================
// 全局范数计算: ||v|| = sqrt(<v, v>)
//==============================================================================
double globalNorm(MPI_Comm comm, int nprocs, const double* v, int n) {
    return std::sqrt(globalDot(comm, nprocs, v, v, n));
}

//==============================================================================
// SPD 矩阵求解 (用于 Scalar1 正交化)
//
// 求解 W * h = projections，其中 W 是 s×s 对称正定矩阵
// 使用 LAPACK 的 Cholesky 分解 (dposv)
//
// 数学背景:
//   Power basis 向量 V_k^0, V_k^1, ..., V_k^{s-1} 之间不正交
//   Gram 矩阵 W[i,j] = <V_k^i, V_k^j> 描述了它们的"夹角"
//   正交化系数通过解 W * h = projections 得到
//==============================================================================
void solveSPD(int n, double* A, double* b_in, double* x_out) {
    std::vector<double> A_copy(A, A + n * n);  // 复制矩阵 (dposv 会覆盖)
    std::vector<double> b_copy(b_in, b_in + n);
    int info, nrhs = 1;
    dposv_((char*)"U", &n, &nrhs, A_copy.data(), &n, b_copy.data(), &n, &info);
    for (int i = 0; i < n; i++) x_out[i] = b_copy[i];
}

//==============================================================================
//                        s-step GMRES 核心算法
//==============================================================================
//
// 【算法概述】
//
// GMRES 的数学目标:
//   找 x_m ∈ x₀ + K_m(A,r₀)，使得 ||b - Ax_m|| 最小
//   K_m(A,r₀) = span{r₀, Ar₀, A²r₀, ..., A^(m-1)r₀} 为 Krylov 子空间
//
// s-step 的关键改进:
//   将 Krylov 子空间的基向量分组为"块"
//   每块包含 s 个 Power basis 向量: [v, Av, A²v, ..., A^(s-1)v]
//
// 【为什么 s-step 可以减少通信】
//
// 传统 GMRES (Arnoldi 过程):
//   第 k 步: w = A*v_k
//            h_{j,k} = <w, v_j>  ← 需要 Allreduce (逐步依赖)
//            w = w - h_{j,k}*v_j ← w 被修改！
//            ...
//   内积之间有依赖: h_{j+1,k} 依赖 h_{j,k} 的结果
//   总通信: ~m²/2 次 Allreduce
//
// s-step GMRES:
//   Power basis: V_0, V_1, ..., V_{s-1} 一次性生成 (本地 mat-vec)
//   Gram 矩阵: W[i,j] = <V_i, V_j> 所有内积可并行计算
//   打包发送: 一次 Allreduce 传输所有 s² 个内积
//   总通信: m+1 次 Allreduce
//
// 【数据结构说明】
//
// V[k]: 第 k 个 Krylov 块，包含 s 个向量
//       V[k][j*n_local + i] = V_k^j 的第 i 个分量
//       即 V_k^j = V[k][j*n_local ... (j+1)*n_local-1]
//
// W[k]: 第 k 个块的 Gram 矩阵 (s×s)
//       W[k][i*s + j] = <V_k^i, V_k^j>
//       用于 Scalar1 正交化 (因为 Power basis 向量不正交)
//
// H: Hessenberg 矩阵 ((ms+1) × ms)，列主序存储
//    H[col*(ms+1) + row] = H(row, col)
//    描述 Krylov 基向量的递推关系
//
// g: 最小二乘问题的右端向量 (经过 Givens 旋转后)
//    g[m] = 当前残差估计 (收敛判断依据)
//
// 【算法流程】
//
// Phase 1: 初始化
//   1. 计算初始残差 r = b - Ax (x=0 时 r=b)
//   2. 应用预处理器 z = M^{-1}*r
//   3. 构建第一个 Power basis 块 V_0
//   4. 计算 beta = ||z|| 和 Gram 矩阵 W_0
//   5. 归一化 V_0
//
// Phase 2: 主循环 (每个块迭代)
//   Step 1: 计算 w = M^{-1}*A*V_k^{s-1} (新块的起点)
//   Step 2: 构建 V_{k+1} 的完整 Power basis
//   Step 3: 打包计算所有内积 (一次 Allreduce)
//   Step 4: Scalar1 正交化 (通过 W 矩阵)
//   Step 5: 提取 Gram 矩阵 W_{k+1}
//   Step 6: 从归一化的起点重建 V_{k+1}
//   Step 7: Scalar2 设置 (幂基结构)
//   Step 8: Givens 旋转更新
//   Step 9: 收敛检查
//
// Phase 3: 收敛后计算解
//   解上三角系统得到 y
//   x = x₀ + Σ V_k * y_k
//
//==============================================================================
void sstepGMRES(
    MPI_Comm comm,
    int nprocs,
    int rank,
    int n_local,
    int n_ghost,
    const double* b_local,
    const GMRESParams& params,
    std::function<void(const double*, double*, double*)> matVec,  // (in, ghost_out, out)
    std::function<void(const double*, double*)> precond,          // (in, out)
    double* x_local,
    GMRESResult& result
) {
    //--------------------------------------------------------------------------
    // 参数提取
    //--------------------------------------------------------------------------
    int s = params.s;      // s-step 参数 (每块向量数)
    int m = params.m;      // 最大块数
    double tol = params.tol;
    int ms = m * s;        // 总 Krylov 维度

    //--------------------------------------------------------------------------
    // 工作向量初始化
    //--------------------------------------------------------------------------
    std::vector<double> r_local(n_local);      // 残差向量 (本地部分)
    std::vector<double> z_local(n_local);      // 预处理后的残差 M^{-1}*r
    std::vector<double> tmp_local(n_local);    // 临时向量
    std::vector<double> Atmp_local(n_local);   // 矩阵-向量乘结果 A*v
    std::vector<double> ghost_data(n_ghost);   // 幽灵层数据 (邻居进程边界值)

    // 解向量初始化为零 (初始猜测 x₀ = 0)
    std::fill(x_local, x_local + n_local, 0.0);

    // 初始残差: r = b - A*x₀ = b (因为 x₀=0)
    std::copy(b_local, b_local + n_local, r_local.begin());

    // 计算 ||b|| 用于相对残差判断
    // 相对残差 = ||b - Ax|| / ||b||
    double bnorm = globalNorm(comm, nprocs, b_local, n_local);

    //--------------------------------------------------------------------------
    // Krylov 子空间存储结构
    //
    // V[k]: 第 k 个块的 s 个 Power basis 向量
    //       存储: [V_k^0 | V_k^1 | ... | V_k^{s-1}]
    //       每个 V_k^j 长度为 n_local
    //--------------------------------------------------------------------------
    std::vector<std::vector<double>> V(m + 1);
    for (int k = 0; k <= m; k++) {
        V[k].resize(s * n_local);  // 每块 s 个向量
    }

    //--------------------------------------------------------------------------
    // Gram 矩阵 W
    //
    // Power basis 向量之间不正交，W 矩阵描述它们的"夹角"
    // W[k][i*s + j] = <V_k^i, V_k^j>
    //
    // 数学意义:
    //   若 V_k^i, V_k^j 正交，则 W[k][i*s+j] = 0 (i≠j)
    //   实际上 Power basis 不正交，W[k] 是稠密的
    //   正交化时需要解 W * h = projections
    //--------------------------------------------------------------------------
    std::vector<std::vector<double>> W(m + 1);
    for (int k = 0; k <= m; k++) {
        W[k].resize(s * s);
    }

    //--------------------------------------------------------------------------
    // Hessenberg 矩阵 H
    //
    // H 是 (ms+1) × ms 的上 Hessenberg 矩阵
    // H(col, row) 存储在 H[col*(ms+1) + row]
    //
    // 数学意义:
    //   描述 Krylov 基的递推关系: A*V_j = Σ H(i,j)*V_i + H(j+1,j)*V_{j+1}
    //
    // Givens 旋转后:
    //   H 变为上三角 R
    //   最小二乘问题变为解 R*y = g
    //--------------------------------------------------------------------------
    std::vector<double> H((ms + 1) * ms, 0.0);
    std::vector<double> g(ms + 1, 0.0);        // 最小二乘右端项
    std::vector<double> cs(ms), sn(ms);        // Givens 旋转参数 (cos, sin)
    int givens_count = 0;                      // 已执行的 Givens 旋转次数

    int ncomm = 0;  // 全局通信计数器

    //==========================================================================
    // Phase 1: 初始化 - 构建第一个 Krylov 块 V_0
    //==========================================================================

    //--------------------------------------------------------------------------
    // Step 1.1: 计算初始预处理残差
    //
    // z = M^{-1} * r = M^{-1} * b
    //
    // 数学意义:
    //   预处理改变了 Krylov 子空间
    //   原始: K_m(A, b)
    //   预处理后: K_m(M^{-1}A, M^{-1}b)
    //--------------------------------------------------------------------------
    precond(b_local, z_local.data());

    //--------------------------------------------------------------------------
    // Step 1.2: 构建 V_0 的 Power basis
    //
    // V_0^0 = z = M^{-1}*b
    // V_0^1 = M^{-1}*A*z
    // V_0^2 = M^{-1}*A*V_0^1 = M^{-1}*A²*z
    // ...
    // V_0^{s-1} = M^{-1}*A^(s-1)*z
    //
    // 注意: 这些向量之间不正交！
    //       需要 Gram 矩阵 W_0 来处理正交化
    //--------------------------------------------------------------------------
    // V_0^0 = z
    for (int i = 0; i < n_local; i++) {
        V[0][i] = z_local[i];
    }

    // 生成剩余的 Power basis 向量 (本地计算，无需全局通信)
    // 循环: V_0^j = M^{-1} * A * V_0^{j-1}
    for (int j = 1; j < s; j++) {
        // 取出 V_0^{j-1}
        std::copy(&V[0][(j - 1) * n_local], &V[0][j * n_local], tmp_local.begin());
        // 计算 A * V_0^{j-1} (包含幽灵层交换)
        matVec(tmp_local.data(), ghost_data.data(), Atmp_local.data());
        // 应用预处理器: V_0^j = M^{-1} * A * V_0^{j-1}
        precond(Atmp_local.data(), tmp_local.data());
        // 存入 V_0^j
        std::copy(tmp_local.begin(), tmp_local.end(), &V[0][j * n_local]);
    }

    //==========================================================================
    // Step 1.3: 第一次全局通信
    //
    // 打包发送内容:
    //   [0]:     beta² = ||z||² (初始残差范数)
    //   [1..s²]: W_0 矩阵元素
    //
    // 关键技巧: 将所有需要的信息打包到一个 Allreduce
    //           避免多次小消息传输
    //==========================================================================
    int init_size = 1 + s * s;
    std::vector<double> init_loc(init_size, 0.0);

    // beta² = ||z||² = <z, z>
    init_loc[0] = vdot(n_local, z_local.data(), z_local.data());

    // 计算 W_0[i,j] = <V_0^i, V_0^j>
    int idx = 1;
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
            init_loc[idx++] = vdot(n_local, &V[0][i * n_local], &V[0][j * n_local]);
        }
    }

    // 一次 Allreduce 获取所有全局值
    std::vector<double> init_glb(init_size);
    MPI_Allreduce(init_loc.data(), init_glb.data(), init_size, MPI_DOUBLE, MPI_SUM, comm);
    ncomm++;

    // beta = ||z|| (初始预处理残差范数)
    // 注意: init_glb[0] 是各进程本地平方和，除以 nprocs 得到平均值
    //       然后开方得到全局范数
    double beta_sq = init_glb[0] / nprocs;
    double beta = std::sqrt(beta_sq);

    //--------------------------------------------------------------------------
    // Step 1.4: 检查初始残差是否已满足收敛条件
    //--------------------------------------------------------------------------
    if (beta < tol * bnorm) {
        result.converged = true;
        result.final_residual = beta / bnorm;
        result.iterations = 0;
        result.communications = ncomm;
        return;  // 无需迭代，直接返回
    }

    //--------------------------------------------------------------------------
    // Step 1.5: 归一化 V_0
    //
    // 将所有 V_0 的向量除以 beta
    // 使得 ||V_0^0|| = 1 (Arnoldi 基的起点)
    //--------------------------------------------------------------------------
    for (int j = 0; j < s; j++) {
        vscal(n_local, 1.0 / beta, &V[0][j * n_local]);
    }

    //--------------------------------------------------------------------------
    // Step 1.6: 提取归一化后的 W_0
    //
    // 归一化后: W_0[i,j] = <V_0^i/beta, V_0^j/beta> = 原值 / beta²
    //--------------------------------------------------------------------------
    idx = 1;
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
            W[0][i * s + j] = init_glb[idx++] / (beta_sq * nprocs);
        }
    }

    //--------------------------------------------------------------------------
    // Step 1.7: 初始化最小二乘问题
    //
    // g 向量初始化: g[0] = beta
    // 这是 ||b||/||b|| * beta 的简化形式
    // 经过 Givens 旋转后，g[m] 将是残差估计
    //--------------------------------------------------------------------------
    g[0] = beta;

    //--------------------------------------------------------------------------
    // Step 1.8: Scalar2 设置 - 幂基结构
    //
    // 对于 Power basis，H 矩阵有固定的结构:
    //   H[j, j+1] = 1.0 (表示 V_{j+1} = A * V_j 的关系)
    //
    // 数学意义:
    //   Power basis: V^j = A^j * V^0
    //   递推关系: V^{j+1} = A * V^j
    //   在 H 矩阵中表示为 H[j, j+1] = 1.0
    //--------------------------------------------------------------------------
    for (int j = 0; j < s - 1; j++) {
        H[j * (ms + 1) + j + 1] = 1.0;
    }

    //==========================================================================
    // Phase 2: 主循环 - 每次迭代生成一个 Krylov 块
    //==========================================================================
    for (int k = 0; k < m; k++) {
        int nblk = k + 1;  // 当前已有的块数

        //======================================================================
        // Step 2.1: 计算新块的起点
        //
        // w = M^{-1} * A * V_k^{s-1}
        //
        // 这是新块 V_{k+1}^0 的初始值 (正交化前)
        //
        // 数学意义:
        //   V_k^{s-1} 是第 k 块的最后一个向量
        //   A * V_k^{s-1} 扩展 Krylov 子空间到下一个"维度"
        //   M^{-1} 应用预处理器
        //======================================================================
        // 取出 V_k^{s-1} (第 k 块的最后一个向量)
        std::copy(&V[k][(s - 1) * n_local], &V[k][s * n_local], tmp_local.begin());
        // 计算 A * V_k^{s-1}
        matVec(tmp_local.data(), ghost_data.data(), Atmp_local.data());
        // 应用预处理器
        precond(Atmp_local.data(), tmp_local.data());
        // 存入 V_{k+1}^0
        std::copy(tmp_local.begin(), tmp_local.end(), &V[k + 1][0]);

        //======================================================================
        // Step 2.2: 构建 V_{k+1} 的完整 Power basis
        //
        // 从 V_{k+1}^0 生成 V_{k+1}^1, V_{k+1}^2, ..., V_{k+1}^{s-1}
        //
        // V_{k+1}^j = M^{-1} * A * V_{k+1}^{j-1}
        //
        // 注意: 这是本地计算，不涉及全局通信
        //       mat-vec 的幽灵层交换是点对点通信，不是 Allreduce
        //======================================================================
        for (int j = 1; j < s; j++) {
            std::copy(&V[k + 1][(j - 1) * n_local], &V[k + 1][j * n_local], tmp_local.begin());
            matVec(tmp_local.data(), ghost_data.data(), Atmp_local.data());
            precond(Atmp_local.data(), tmp_local.data());
            std::copy(tmp_local.begin(), tmp_local.end(), &V[k + 1][j * n_local]);
        }

        //======================================================================
        // Step 2.3: 一次性计算所有内积 (核心通信优化)
        //
        // 打包内容:
        //   [0 .. nblk*s-1]:     Scalar1 投影 (V_{k+1}^0 在之前所有块的投影)
        //   [nblk*s .. nblk*s+s²-1]: W_{k+1} 矩阵元素
        //   [nblk*s+s²]:         ||V_{k+1}^0||² (正交化前的范数)
        //
        // 关键技巧:
        //   - 这些内积的输入向量都已固定，没有依赖关系
        //   - 可以全部打包到一个 Allreduce
        //   - 传统 GMRES 需要逐步等待，这里一次搞定
        //======================================================================
        int total_size = nblk * s + s * s + 1;
        std::vector<double> local_data(total_size, 0.0);
        idx = 0;

        // Scalar1 投影: <V_p^j, V_{k+1}^0> for all p <= k, all j < s
        // 这些用于正交化 V_{k+1}^0 到之前所有块
        for (int pk = 0; pk < nblk; pk++) {
            for (int j = 0; j < s; j++) {
                local_data[idx++] = vdot(n_local, &V[pk][j * n_local], &V[k + 1][0]);
            }
        }

        // W_{k+1} 矩阵: <V_{k+1}^i, V_{k+1}^j>
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                local_data[idx++] = vdot(n_local, &V[k + 1][i * n_local], &V[k + 1][j * n_local]);
            }
        }

        // ||V_{k+1}^0||² (正交化前的范数平方)
        local_data[idx] = vdot(n_local, &V[k + 1][0], &V[k + 1][0]);

        // 一次 Allreduce 获取所有全局值
        std::vector<double> global_data(total_size);
        MPI_Allreduce(local_data.data(), global_data.data(), total_size, MPI_DOUBLE, MPI_SUM, comm);
        ncomm++;

        // 归一化 (除以 nprocs)
        for (int i = 0; i < total_size; i++) global_data[i] /= nprocs;
        double w_norm_sq = global_data[total_size - 1];

        //======================================================================
        // Step 2.4: Scalar1 正交化
        //
        // 目标: 将 V_{k+1}^0 正交化到之前所有块 V_0, V_1, ..., V_k
        //
        // 数学推导:
        //   设投影 p_j = <V_{k+1}^0, V_k^j> (已通过 Allreduce 获取)
        //   正交化系数 h_j 满足: W_k * h = p
        //   (因为 Power basis 向量不正交，需要通过 W 矩阵修正)
        //
        // 正交化过程:
        //   V_{k+1}^0 = V_{k+1}^0 - Σ h_j * V_k^j
        //======================================================================
        idx = 0;
        double energy = 0;  // 正交化消耗的"能量" Σ h_j²
        int col_last = k * s + (s - 1);  // H 矩阵中当前处理的列索引

        // 对每个之前的块执行正交化
        for (int pk = 0; pk < nblk; pk++) {
            // 提取投影向量 h_raw (未修正的投影系数)
            std::vector<double> h_raw(s), h_ortho(s);
            for (int j = 0; j < s; j++) h_raw[j] = global_data[idx++];

            // 解 SPD 系统: W[pk] * h_ortho = h_raw
            // 得到正确的正交化系数
            solveSPD(s, W[pk].data(), h_raw.data(), h_ortho.data());

            // 更新 H 矩阵和 V_{k+1}^0
            for (int j = 0; j < s; j++) {
                // 存入 H 矩阵
                H[col_last * (ms + 1) + pk * s + j] = h_ortho[j];
                // 正交化: V_{k+1}^0 -= h_ortho[j] * V_pk^j
                vaxpy(n_local, -h_ortho[j], &V[pk][j * n_local], &V[k + 1][0]);
                // 记录能量贡献
                energy += h_ortho[j] * h_ortho[j];
            }
        }

        //======================================================================
        // Step 2.5: 计算正交化后的范数
        //
        // 数学推导:
        //   正交化前: ||V_{k+1}^0||² = w_norm_sq
        //   正交化后: ||V_{k+1}^0||² = w_norm_sq - Σ h_j²
        //   (因为 V_{k+1}^0 -= Σ h_j * V_j，且 V_j 正交)
        //======================================================================
        double norm_sq = w_norm_sq - energy;
        // 数值稳定性检查: 如果 norm_sq < 0，重新计算
        if (norm_sq < 0) {
            norm_sq = vdot(n_local, &V[k + 1][0], &V[k + 1][0]);
        }
        double norm = std::sqrt(std::max(norm_sq, 0.0));

        // 存入 H 矩阵的最后一个元素 (H(col_last, col_last+1))
        H[col_last * (ms + 1) + (k + 1) * s] = norm;

        //--------------------------------------------------------------------------
        // Step 2.6: 归一化 V_{k+1}^0
        //--------------------------------------------------------------------------
        if (norm > 1e-14) {
            vscal(n_local, 1.0 / norm, &V[k + 1][0]);
        }

        //======================================================================
        // Step 2.7: 提取 Gram 矩阵 W_{k+1}
        //
        // W_{k+1}[i,j] = <V_{k+1}^i, V_{k+1}^j>
        // 用于下一轮正交化
        //======================================================================
        idx = nblk * s;
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                W[k + 1][i * s + j] = global_data[idx++];
            }
        }

        //======================================================================
        // Step 2.8: 从归一化的 V_{k+1}^0 重建整个块
        //
        // 为什么需要重建？
        //   - V_{k+1}^0 已被正交化和归一化
        //   - 但 V_{k+1}^1, V_{k+1}^2, ... 还是原始的 Power basis
        //   - 需要从新的 V_{k+1}^0 重新生成它们
        //
        // 数学意义:
        //   新的 Power basis: V_{k+1}^j = M^{-1}*A^j*V_{k+1}^0 (归一化后)
        //======================================================================
        for (int j = 1; j < s; j++) {
            std::copy(&V[k + 1][(j - 1) * n_local], &V[k + 1][j * n_local], tmp_local.begin());
            matVec(tmp_local.data(), ghost_data.data(), Atmp_local.data());
            precond(Atmp_local.data(), tmp_local.data());
            std::copy(tmp_local.begin(), tmp_local.end(), &V[k + 1][j * n_local]);
        }

        //======================================================================
        // Step 2.9: Scalar2 设置 - 幂基结构
        //
        // 对于新块 V_{k+1}，设置 H 矩阵的固定结构:
        //   H[(k+1)*s+j, (k+1)*s+j+1] = 1.0
        //======================================================================
        for (int j = 0; j < s - 1; j++) {
            int col_j = (k + 1) * s + j;
            H[col_j * (ms + 1) + (k + 1) * s + j + 1] = 1.0;
        }

        //======================================================================
        // Step 2.10: Givens 旋转
        //
        // 目标: 将 H 矩阵转换为上三角形式
        //
        // 数学推导:
        //   H 是上 Hessenberg 矩阵 (只有下三角有一行非零)
        //   通过 Givens 旋转逐行消去下三角元素
        //   每次旋转同时更新 g 向量
        //
        // Givens 旋转:
        //   G(i,j) = [c  s; -s  c]
        //   作用: 消去 H(i+1,i)，使 H(i,i) 变为 r = sqrt(a²+b²)
        //
        // g 向量更新:
        //   g[i] = c*g[i] + s*g[i+1]
        //   g[i+1] = -s*g[i] + c*g[i+1]
        //   g[m] 保持为残差估计
        //======================================================================
        for (int j = 0; j < s; j++) {
            int col = k * s + j;

            // 应用之前的 Givens 旋转到新列
            for (int i = 0; i < givens_count; i++) {
                double t1 = H[col * (ms + 1) + i];
                double t2 = H[col * (ms + 1) + i + 1];
                H[col * (ms + 1) + i] = cs[i] * t1 + sn[i] * t2;
                H[col * (ms + 1) + i + 1] = -sn[i] * t1 + cs[i] * t2;
            }

            // 创建新的 Givens 旋转
            int row = givens_count;
            double a = H[col * (ms + 1) + row];
            double b_val = H[col * (ms + 1) + row + 1];

            if (std::abs(a) < 1e-14 && std::abs(b_val) < 1e-14) {
                // 数值特殊情况: 设为单位旋转
                cs[givens_count] = 1.0;
                sn[givens_count] = 0.0;
            } else {
                // 正常情况: 计算 cos, sin
                double r = std::sqrt(a * a + b_val * b_val);
                cs[givens_count] = a / r;
                sn[givens_count] = b_val / r;
                // 更新 H 矩阵元素
                H[col * (ms + 1) + row] = r;
                H[col * (ms + 1) + row + 1] = 0.0;
            }

            // 更新 g 向量
            double gt = g[givens_count];
            g[givens_count] = cs[givens_count] * gt + sn[givens_count] * g[givens_count + 1];
            g[givens_count + 1] = -sn[givens_count] * gt + cs[givens_count] * g[givens_count + 1];
            givens_count++;
        }

        //======================================================================
        // Step 2.11: 收敛检查
        //
        // 残差估计 = |g[givens_count]| / bnorm
        //
        // 数学推导:
        //   经过 Givens 旋转后，最小二乘问题变为:
        //     min ||R*y - g||
        //   其中 R 是上三角，g 经过旋转
        //   残差 = g[m] (未被消去的最后一个元素)
        //======================================================================
        double res_est = std::abs(g[givens_count]) / bnorm;

        // 输出迭代信息 (仅 rank 0)
        if (rank == 0) {
            std::cout << "Block " << (k + 1) << " (Krylov=" << ((k + 1) * s)
                      << "): residual=" << std::scientific << res_est << "\n";
        }

        //======================================================================
        // Step 2.12: 收敛后计算解
        //======================================================================
        if (res_est < tol) {
            //--------------------------------------------------------------
            // 解上三角系统: R * y = g
            //
            // 从最后一个元素开始回代
            // y[i] = (g[i] - Σ_{j>i} R[i,j]*y[j]) / R[i,i]
            //--------------------------------------------------------------
            std::vector<double> y(ms, 0.0);
            for (int i = givens_count - 1; i >= 0; i--) {
                y[i] = g[i];
                for (int j = i + 1; j < givens_count; j++) {
                    y[i] -= H[j * (ms + 1) + i] * y[j];
                }
                if (std::abs(H[i * (ms + 1) + i]) > 1e-14) {
                    y[i] /= H[i * (ms + 1) + i];
                }
            }

            //--------------------------------------------------------------
            // 计算解向量: x = Σ V_k * y_k
            //
            // x = x₀ + Σ_{k=0}^{nblk-1} Σ_{j=0}^{s-1} y[k*s+j] * V_k^j
            //--------------------------------------------------------------
            for (int kk = 0; kk <= k; kk++) {
                for (int j = 0; j < s; j++) {
                    vaxpy(n_local, y[kk * s + j], &V[kk][j * n_local], x_local);
                }
            }

            //--------------------------------------------------------------
            // 计算真实残差 ||b - Ax||
            //
            // 验证: 比较 Givens 残差估计与真实残差
            //--------------------------------------------------------------
            std::vector<double> Ax_local(n_local), res_local(n_local);
            matVec(x_local, ghost_data.data(), Ax_local.data());
            for (int i = 0; i < n_local; i++) {
                res_local[i] = b_local[i] - Ax_local[i];
            }
            result.final_residual = globalNorm(comm, nprocs, res_local.data(), n_local) / bnorm;

            result.converged = true;
            result.iterations = k + 1;
            result.communications = ncomm;
            return;
        }
    }

    //==========================================================================
    // Phase 3: 未收敛 - 计算当前最优解
    //
    // 使用全部 m 个块的 Krylov 子空间
    // 计算 x = Σ V_k * y_k 使得 ||b - Ax|| 最小
    //==========================================================================
    std::vector<double> y(ms, 0.0);
    for (int i = ms - 1; i >= 0; i--) {
        y[i] = g[i];
        for (int j = i + 1; j < ms; j++) {
            y[i] -= H[j * (ms + 1) + i] * y[j];
        }
        if (std::abs(H[i * (ms + 1) + i]) > 1e-14) {
            y[i] /= H[i * (ms + 1) + i];
        }
    }

    // 组合解向量
    for (int k_iter = 0; k_iter < m; k_iter++) {
        for (int j = 0; j < s; j++) {
            vaxpy(n_local, y[k_iter * s + j], &V[k_iter][j * n_local], x_local);
        }
    }

    // 计算真实残差
    std::vector<double> Ax_local(n_local), res_local(n_local);
    matVec(x_local, ghost_data.data(), Ax_local.data());
    for (int i = 0; i < n_local; i++) {
        res_local[i] = b_local[i] - Ax_local[i];
    }
    result.final_residual = globalNorm(comm, nprocs, res_local.data(), n_local) / bnorm;

    result.converged = false;
    result.iterations = m;
    result.communications = ncomm;
}

//==============================================================================
// 主程序: 问题构建与 GMRES 调用
//==============================================================================
int main(int argc, char** argv) {
    //--------------------------------------------------------------------------
    // MPI 初始化
    //--------------------------------------------------------------------------
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    //--------------------------------------------------------------------------
    // 参数解析
    //--------------------------------------------------------------------------
    if (argc < 5) {
        if (rank == 0) {
            std::cout << "Usage: " << argv[0] << " <n_global> <s> <m> <type> [tol]\n";
            std::cout << "  n_global: global matrix dimension\n";
            std::cout << "  s: s-step parameter (2-3)\n";
            std::cout << "  m: number of blocks\n";
            std::cout << "  type: 0=five-diagonal, 1=anisotropic\n";
            std::cout << "  tol: convergence tolerance (default 1e-8)\n";
        }
        MPI_Finalize();
        return 1;
    }

    int n_global = std::atoi(argv[1]);
    int s = std::atoi(argv[2]);
    int m = std::atoi(argv[3]);
    int type = std::atoi(argv[4]);
    double tol = (argc > 5) ? std::atof(argv[5]) : 1e-8;

    // 参数约束: s 建议 2-3
    if (s < 2) s = 2;
    if (s > 3) s = 3;

    //--------------------------------------------------------------------------
    // 分布式矩阵构建
    //--------------------------------------------------------------------------
    DistributedCSRMatrix A;
    A.init(MPI_COMM_WORLD, n_global);

    std::string mat_name;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.5, 1.5);

    if (type == 0) {
        // 五对角矩阵 (2D Poisson 五点差分格式)
        A.buildFiveDiagonal(4.0 + dis(gen), -0.5 * dis(gen));
        mat_name = "Easy";
    } else {
        // 各向异性扩散矩阵 (条件数差，收敛困难)
        A.buildAnisotropic(0.01);
        mat_name = "Anisotropic(0.01)";
    }

    //--------------------------------------------------------------------------
    // 幽灵层交换初始化
    //--------------------------------------------------------------------------
    HaloExchange halo;
    A.setupHalo(halo);

    //--------------------------------------------------------------------------
    // ILU0 预处理器
    //--------------------------------------------------------------------------
    DistributedILU0 M;
    M.factorize(A);

    //--------------------------------------------------------------------------
    // 获取本地维度
    //--------------------------------------------------------------------------
    int n_local = A.n_local();
    int n_ghost = A.n_ghost();

    //--------------------------------------------------------------------------
    // 右端项构建: b = 1.0
    //--------------------------------------------------------------------------
    std::vector<double> b_local(n_local, 1.0);

    //--------------------------------------------------------------------------
    // 打印问题信息
    //--------------------------------------------------------------------------
    if (rank == 0) {
        std::cout << "==============================================\n";
        std::cout << "Distributed s-step GMRES\n";
        std::cout << "==============================================\n";
        std::cout << "Matrix: " << mat_name << ", n_global=" << n_global << "\n";
        std::cout << "np=" << nprocs << ", s=" << s << ", m=" << m << "\n";
        std::cout << "tol=" << tol << "\n";
        std::cout << "==============================================\n\n";
    }

    //--------------------------------------------------------------------------
    // 定义回调函数
    //
    // 回调函数实现了算法与问题实现的解耦
    // GMRES 算法只关心抽象操作，不关心具体实现
    //--------------------------------------------------------------------------

    // 矩阵-向量乘法回调: y = A * x
    auto matVecCallback = [&](const double* v_in, double* ghost_out, double* v_out) {
        // 使用临时 DistributedVector 进行幽灵层交换
        DistributedVector tmp_vec(n_local, n_ghost);
        for (int i = 0; i < n_local; i++) tmp_vec.local(i) = v_in[i];

        // 非阻塞幽灵层交换
        halo.start_exchange(tmp_vec);
        halo.wait_exchange(tmp_vec);

        // 复制幽灵层数据
        for (int i = 0; i < n_ghost; i++) {
            ghost_out[i] = tmp_vec.ghost(i);
        }

        // 执行本地矩阵-向量乘法
        A.mv(v_in, ghost_out, v_out);
    };

    // 预处理器回调: z = M^{-1} * r
    auto precondCallback = [&](const double* r_in, double* z_out) {
        M.apply(r_in, z_out);
    };

    //--------------------------------------------------------------------------
    // 调用 GMRES 求解器
    //--------------------------------------------------------------------------
    std::vector<double> x_local(n_local);
    GMRESParams params(s, m, tol);
    GMRESResult result;

    sstepGMRES(
        MPI_COMM_WORLD,
        nprocs,
        rank,
        n_local,
        n_ghost,
        b_local.data(),
        params,
        matVecCallback,
        precondCallback,
        x_local.data(),
        result
    );

    //--------------------------------------------------------------------------
    // 输出结果
    //--------------------------------------------------------------------------
    if (rank == 0) {
        std::cout << "\n==============================================\n";
        std::cout << "Final Result:\n";
        std::cout << "==============================================\n";
        std::cout << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
        std::cout << "||b-Ax||/||b|| = " << std::scientific << result.final_residual << "\n";
        std::cout << "Iterations (blocks): " << result.iterations << "\n";
        std::cout << "Krylov vectors: " << result.iterations * s << "\n";
        std::cout << "Global communications: " << result.communications
                  << " (expected max: " << (m + 1) << ")\n";
        std::cout << "==============================================\n";
    }

    MPI_Finalize();
    return result.converged ? 0 : 1;
}