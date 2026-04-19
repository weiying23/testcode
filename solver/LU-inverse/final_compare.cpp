// ============================================================================
// 矩阵求逆方法对比程序
// 比较三种方法的性能：LAPACK、Eigen、分块法（Blockwise）
// 使用 OpenMP 并行化：每个线程计算一次矩阵分解
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>                      // OpenMP 并行化
#include <Accelerate/Accelerate.h>  // macOS 加速框架，提供 LAPACK 和 CBLAS
//#define EIGEN_USE_OPENMP
#include <Eigen/Dense>              // Eigen 线性代数库

#ifdef ACCELERATE_NEW_LAPACK
typedef __LAPACK_int lapack_int;  // 新 LAPACK 接口（ILP64，使用 long）
#else
typedef __CLPK_integer lapack_int;    // 旧 CLAPACK 接口（使用 int）
#endif

// 全局数据区（40x40 矩阵，最大 1600 元素）
// 每个线程需要独立的存储空间
#define MAX_THREADS 64
double A[MAX_THREADS][1600];       // 行优先存储的原始矩阵（每个线程独立）
double A_col[MAX_THREADS][1600];   // 列优先存储的副本（LAPACK 需要，每个线程独立）
double Ai_l[MAX_THREADS][1600], Ai_e[MAX_THREADS][1600], Ai_b[MAX_THREADS][1600];  // 每个线程独立的求逆结果存储

// 每个线程的计时和残差结果
double thread_time_l[MAX_THREADS], thread_time_e[MAX_THREADS], thread_time_b[MAX_THREADS];
double thread_res_l[MAX_THREADS], thread_res_e[MAX_THREADS], thread_res_b[MAX_THREADS];

// Blockwise 工作缓冲区已移除，改用 Eigen 内部管理

/**
 * 从 .mtx 文件加载矩阵
 * @param fn  文件名
 * @param n   输出：矩阵维度
 * @param idx 要加载的矩阵索引（文件中可能包含多个矩阵）
 * @param tot 输出：文件中矩阵总数
 * @param tid 线程 ID，用于选择独立的数据存储区
 */
void load_matrix(const char *fn, int *n, int idx, int *tot, int tid) {
    FILE *f = fopen(fn, "r");
    char line[256];
    // 跳过前两行（注释行和头部信息）
    fgets(line, sizeof(line), f);
    fgets(line, sizeof(line), f);
    int rows, cols, num;
    // 解析头部：Number of matrices:X Matrix length:R C
    sscanf(line, "Number of matrices:%d Matrix length:%d %d", &num, &rows, &cols);
    *tot = num; *n = rows;
    // 跳过前面的矩阵数据
    for (int k = 0; k < idx * rows * cols; k++) fgets(line, sizeof(line), f);
    // 读取当前矩阵的列优先数据
    double *col = (double *)malloc(rows * cols * sizeof(double));
    for (int k = 0; k < rows * cols; k++) { fgets(line, sizeof(line), f); col[k] = atof(line); }
    fclose(f);
    // 列优先转行优先 (A)，同时准备列优先副本 (A_col)，使用线程独立的存储区
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            A[tid][i * cols + j] = col[i + j * rows];
            A_col[tid][i + j * cols] = col[i + j * rows];  // 保持列优先
        }
    free(col);
}

/**
 * 使用 LAPACK 进行矩阵求逆（仅核心操作，计时内）
 * @param n 矩阵维度
 * @param tid 线程 ID，使用独立的数据存储区
 * 结果存储于 Ai_l[tid]（行优先）
 */
void lapack_inv_core(int n, int tid) {
    double *Ac = (double *)malloc(n*n*sizeof(double));
    lapack_int *ipiv = (lapack_int *)malloc(n*sizeof(lapack_int));
    // 直接使用预转换的列优先数据 A_col[tid]
    for (int i = 0; i < n*n; i++) Ac[i] = A_col[tid][i];
    // LU 分解：A = P * L * U
    lapack_int nn = n, lda = n, info = 0;
    dgetrf_(&nn, &nn, Ac, &lda, ipiv, &info);
    if (info != 0) { free(Ac); free(ipiv); return; }  // 奇异矩阵
    // 查询最优工作区大小
    double wq; lapack_int lw = -1;
    dgetri_(&nn, Ac, &lda, ipiv, &wq, &lw, &info);
    lw = (lapack_int)wq;
    double *wk = (double *)malloc(lw*sizeof(double));
    // 从 LU 分解计算逆矩阵
    dgetri_(&nn, Ac, &lda, ipiv, wk, &lw, &info);
    free(wk); free(ipiv);
    // 列优先转行优先存储结果
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Ai_l[tid][i*n + j] = Ac[j*n + i];
    free(Ac);
}

/**
 * 计算残差 ||A × A^(-1) - I||_F（计时外）
 * @param n 矩阵维度
 * @param tid 线程 ID，使用独立的数据存储区
 * @param Ai 逆矩阵（行优先，线程独立）
 * @return Frobenius 范数残差
 */
double compute_residual(int n, int tid, double *Ai) {
    double *P = (double *)malloc(n*n*sizeof(double));
    double *I = (double *)malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++) I[i] = 0;
    for (int i = 0; i < n; i++) I[i*n + i] = 1;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A[tid], n, Ai, n, 0.0, P, n);
    double r = 0;
    for (int k = 0; k < n*n; k++) { double d = P[k] - I[k]; r += d*d; }
    free(P); free(I);
    return sqrt(r);
}

/**
 * 使用 Eigen 库进行矩阵求逆（仅核心操作，计时内）
 * @param n 矩阵维度
 * @param tid 线程 ID，使用独立的数据存储区
 * 结果存储于 Ai_e[tid]（行优先）
 */
void eigen_inv_core(int n, int tid) {
    // 使用 Map 将原始数组包装为 Eigen 矩阵（行优先，使用线程独立数据）
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(A[tid], n, n);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mati(Ai_e[tid], n, n);
    mati = mat.inverse();  // Eigen 内置求逆
}

// ============================================================================
// 分块求逆法（Blockwise Inversion）- 优化版本
// 利用 2x2 分块矩阵求逆公式：
//   [A11 A12]^(-1)   [ A11^(-1) + A11^(-1)*A12*S^(-1)*A21*A11^(-1)   -A11^(-1)*A12*S^(-1) ]
//   [A21 A22]      = [                    -S^(-1)*A21*A11^(-1))            S^(-1)        ]
// 其中 S = A22 - A21*A11^(-1)*A12 (Schur 补)
//
// 优化：缓存中间矩阵 M = A21*A11^(-1) 和 N = A11^(-1)*A12，减少重复的矩阵乘法次数
// ============================================================================

/**
 * 使用分块法进行矩阵求逆（仅核心操作，计时内）
 * 混合版本：Eigen 求逆 + cblas_dgemm 矩阵乘法（单线程）
 * @param n 矩阵维度（必须为偶数）
 * @param tid 线程 ID，使用独立的数据存储区和缓冲区
 * 结果存储于 Ai_b[tid]（行优先）
 */
void blockwise_inv_core(int n, int tid) {
    int n2 = n / 2;
    int n2sq = n2 * n2;

    // 从行优先 A[tid] 提取子块到 Eigen（自动转为列优先）
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMat;
    Eigen::Map<RowMat> A_full(A[tid], n, n);

    Eigen::MatrixXd A11 = A_full.block(0, 0, n2, n2);
    Eigen::MatrixXd A12 = A_full.block(0, n2, n2, n2);
    Eigen::MatrixXd A21 = A_full.block(n2, 0, n2, n2);
    Eigen::MatrixXd A22 = A_full.block(n2, n2, n2, n2);

    // 步骤 1: 求 A11^(-1) — Eigen 求逆
    Eigen::MatrixXd A11_inv = A11.inverse();

    // 步骤 2: M = A21 * A11_inv — cblas_dgemm
    Eigen::MatrixXd M(n2, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2,
                1.0, A21.data(), n2, A11_inv.data(), n2, 0.0, M.data(), n2);

    // 步骤 3: N = A11_inv * A12 — cblas_dgemm
    Eigen::MatrixXd N(n2, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2,
                1.0, A11_inv.data(), n2, A12.data(), n2, 0.0, N.data(), n2);

    // 步骤 4: S = A22 - M * A12 — cblas_dgemm + 标量减法
    Eigen::MatrixXd S(n2, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2,
                1.0, M.data(), n2, A12.data(), n2, 0.0, S.data(), n2);
    for (int i = 0; i < n2sq; i++) S.data()[i] = A22.data()[i] - S.data()[i];

    // 步骤 5: 求 S^(-1) — Eigen 求逆
    Eigen::MatrixXd S_inv = S.inverse();

    // 步骤 6: TopRight = -N * S_inv — cblas_dgemm
    Eigen::MatrixXd TopRight(n2, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2,
                -1.0, N.data(), n2, S_inv.data(), n2, 0.0, TopRight.data(), n2);

    // 步骤 7: BotLeft = -S_inv * M — cblas_dgemm
    Eigen::MatrixXd BotLeft(n2, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2,
                -1.0, S_inv.data(), n2, M.data(), n2, 0.0, BotLeft.data(), n2);

    // 步骤 8: TopLeft = A11_inv - TopRight * M — cblas_dgemm + 标量减法
    Eigen::MatrixXd TopLeft(n2, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2,
                1.0, TopRight.data(), n2, M.data(), n2, 0.0, TopLeft.data(), n2);
    for (int i = 0; i < n2sq; i++) TopLeft.data()[i] = A11_inv.data()[i] - TopLeft.data()[i];

    // 组装结果（行优先输出）
    Eigen::Map<RowMat> result(Ai_b[tid], n, n);
    result.block(0, 0, n2, n2) = TopLeft;
    result.block(0, n2, n2, n2) = TopRight;
    result.block(n2, 0, n2, n2) = BotLeft;
    result.block(n2, n2, n2, n2) = S_inv;
}

/**
 * 主函数：性能对比测试（OpenMP 并行版本）
 * 每个线程独立计算矩阵分解，进行多次求逆运算
 *
 * 计时说明：
 * - 只计时核心求逆操作（dgetrf+dgetri / eigen inverse / blockwise）
 * - 残差检查在计时外进行，仅用于验证正确性
 * - 输出所有线程中的最大/最小时间
 *
 * 线程数控制：
 * - 通过环境变量 OMP_NUM_THREADS 设置
 */
int main() {
    // 测试文件列表
    const char *files[] = {"benchmark1_1.mtx", "benchmark10_1.mtx", "benchmark1000_1.mtx"};
    int n, tot;
    int num_runs = 1000;  // 每个矩阵重复次数

    // 从环境变量获取线程数
    int num_threads = omp_get_max_threads();
    printf("LAPACK vs Eigen vs Blockwise 性能对比 (OpenMP 并行版本)\n");
    printf("线程数：%d (通过 OMP_NUM_THREADS 设置)\n", num_threads);
    printf("每矩阵重复测试 %d 次\n", num_runs);
    printf("计时范围：仅核心求逆操作（不含残差检查）\n\n");

    // 输出 Markdown 表格格式
    printf("| 文件 | 方法 | 时间 (s) | 残差 |\n");
    printf("|------|------|---------|------|\n");

    for (int f = 0; f < 3; f++) {
        // 加载文件，获取矩阵信息（使用线程 0 读取）
        load_matrix(files[f], &n, 0, &tot, 0);

        // 初始化线程结果
        for (int t = 0; t < num_threads; t++) {
            thread_time_l[t] = 0; thread_time_e[t] = 0; thread_time_b[t] = 0;
            thread_res_l[t] = 0; thread_res_e[t] = 0; thread_res_b[t] = 0;
        }

        // 遍历文件中所有矩阵
        for (int i = 0; i < tot; i++) {
            // 每个线程加载自己的矩阵副本
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();

                // 每个线程加载矩阵到自己的存储区
                load_matrix(files[f], &n, i, &tot, tid);

                // 每种方法重复测试 num_runs 次
                for (int run = 0; run < num_runs; run++) {
                    // === 只计时核心求逆操作 ===
                    double s;

                    s = omp_get_wtime();
                    lapack_inv_core(n, tid);
                    thread_time_l[tid] += omp_get_wtime() - s;

                    s = omp_get_wtime();
                    eigen_inv_core(n, tid);
                    thread_time_e[tid] += omp_get_wtime() - s;

                    s = omp_get_wtime();
                    blockwise_inv_core(n, tid);
                    thread_time_b[tid] += omp_get_wtime() - s;

                    // === 残差检查在计时外进行 ===
                    thread_res_l[tid] += compute_residual(n, tid, Ai_l[tid]);
                    thread_res_e[tid] += compute_residual(n, tid, Ai_e[tid]);
                    thread_res_b[tid] += compute_residual(n, tid, Ai_b[tid]);
                }
            }
        }

        // 计算并输出最大/最小时间
        int total = tot * num_runs;

        // 找最大/最小时间
        double min_l = thread_time_l[0], max_l = thread_time_l[0];
        double min_e = thread_time_e[0], max_e = thread_time_e[0];
        double min_b = thread_time_b[0], max_b = thread_time_b[0];

        for (int t = 1; t < num_threads; t++) {
            if (thread_time_l[t] < min_l) min_l = thread_time_l[t];
            if (thread_time_l[t] > max_l) max_l = thread_time_l[t];
            if (thread_time_e[t] < min_e) min_e = thread_time_e[t];
            if (thread_time_e[t] > max_e) max_e = thread_time_e[t];
            if (thread_time_b[t] < min_b) min_b = thread_time_b[t];
            if (thread_time_b[t] > max_b) max_b = thread_time_b[t];
        }

        // 计算所有线程的平均时间
        double avg_l = 0, avg_e = 0, avg_b = 0;
        for (int t = 0; t < num_threads; t++) {
            avg_l += thread_time_l[t];
            avg_e += thread_time_e[t];
            avg_b += thread_time_b[t];
        }
        avg_l /= num_threads;
        avg_e /= num_threads;
        avg_b /= num_threads;

        // 输出统计结果（平均时间，括号内为 min/max 范围）
        printf("| %s | LAPACK    | %.6f (min=%.6f, max=%.6f) | %.2e |\n",
               files[f], avg_l/total, min_l/total, max_l/total, thread_res_l[0]/total);
        printf("| %s | Eigen     | %.6f (min=%.6f, max=%.6f) | %.2e |\n",
               files[f], avg_e/total, min_e/total, max_e/total, thread_res_e[0]/total);
        printf("| %s | Blockwise | %.6f (min=%.6f, max=%.6f) | %.2e |\n",
               files[f], avg_b/total, min_b/total, max_b/total, thread_res_b[0]/total);
    }

    return 0;
}
