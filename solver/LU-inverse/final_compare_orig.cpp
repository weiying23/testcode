// ============================================================================
// 矩阵求逆方法对比程序
// 比较三种方法的性能：LAPACK、Eigen、分块法（Blockwise）
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Accelerate/Accelerate.h>  // macOS 加速框架，提供 LAPACK 和 CBLAS
//#define EIGEN_USE_OPENMP
#include <Eigen/Dense>              // Eigen 线性代数库

#ifdef ACCELERATE_NEW_LAPACK
typedef __LAPACK_int lapack_int;  // 新 LAPACK 接口（ILP64，使用 long）
#else
typedef __CLPK_integer lapack_int;    // 旧 CLAPACK 接口（使用 int）
#endif

// 全局数据区（40x40 矩阵，最大 1600 元素）
double A[1600];       // 行优先存储的原始矩阵
double A_col[1600];   // 列优先存储的副本（LAPACK 需要）
double Ai_l[1600], Ai_e[1600];  // 存储 LAPACK 和 Eigen 的求逆结果

/**
 * 从 .mtx 文件加载矩阵
 * @param fn  文件名
 * @param n   输出：矩阵维度
 * @param idx 要加载的矩阵索引（文件中可能包含多个矩阵）
 * @param tot 输出：文件中矩阵总数
 */
void load_matrix(const char *fn, int *n, int idx, int *tot) {
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
    // 列优先转行优先 (A)，同时准备列优先副本 (A_col)
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            A[i * cols + j] = col[i + j * rows];
            A_col[i + j * cols] = col[i + j * rows];  // 保持列优先
        }
    free(col);
}

/**
 * 使用 LAPACK 进行矩阵求逆（仅核心操作，计时内）
 * @param n 矩阵维度
 * 结果存储于 Ai_l（行优先）
 */
void lapack_inv_core(int n) {
    double *Ac = (double *)malloc(n*n*sizeof(double));
    lapack_int *ipiv = (lapack_int *)malloc(n*sizeof(lapack_int));
    // 直接使用预转换的列优先数据 A_col
    for (int i = 0; i < n*n; i++) Ac[i] = A_col[i];
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
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Ai_l[i*n + j] = Ac[j*n + i];
    free(Ac);
}

/**
 * 计算残差 ||A × A^(-1) - I||_F（计时外）
 * @param n 矩阵维度
 * @param Ai 逆矩阵（行优先）
 * @return Frobenius 范数残差
 */
double compute_residual(int n, double *Ai) {
    double *P = (double *)malloc(n*n*sizeof(double));
    double *I = (double *)malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++) I[i] = 0;
    for (int i = 0; i < n; i++) I[i*n + i] = 1;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, Ai, n, 0.0, P, n);
    double r = 0;
    for (int k = 0; k < n*n; k++) { double d = P[k] - I[k]; r += d*d; }
    free(P); free(I);
    return sqrt(r);
}

/**
 * 使用 Eigen 库进行矩阵求逆（仅核心操作，计时内）
 * @param n 矩阵维度
 * 结果存储于 Ai_e（行优先）
 */
void eigen_inv_core(int n) {
    // 使用 Map 将原始数组包装为 Eigen 矩阵（行优先）
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(A, n, n);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mati(Ai_e, n, n);
    mati = mat.inverse();  // Eigen 内置求逆
}

// ============================================================================
// 分块求逆法（Blockwise Inversion）
// 利用 2x2 分块矩阵求逆公式：
//   [A11 A12]^(-1)   [ A11^(-1) + A11^(-1)*A12*S^(-1)*A21*A11^(-1)   -A11^(-1)*A12*S^(-1) ]
//   [A21 A22]      = [                    -S^(-1)*A21*A11^(-1))            S^(-1)        ]
// 其中 S = A22 - A21*A11^(-1)*A12 (Schur 补)
// ============================================================================

// Blockwise 工作缓冲区（静态分配，避免 malloc 开销）
// 最大支持 20x20 子块（即 40x40 矩阵）
double buf_A11[400], buf_A12[400], buf_A21[400], buf_A22[400];
double buf_S[400], buf_TR[400], buf_BL[400], buf_Temp[400], buf_Temp2[400];
lapack_int buf_ipiv[20];
double buf_wk[200];

/**
 * 使用分块法进行矩阵求逆（仅核心操作，计时内）
 * @param n 矩阵维度（必须为偶数）
 * 结果存储于 Ai_e（行优先）
 */
void blockwise_inv_core(int n) {
    int n2 = n / 2;
    int n2sq = n2 * n2;

    // 从行优先提取子块到列优先缓冲区（LAPACK 需要列优先）
    for (int j = 0; j < n2; j++) {
        for (int i = 0; i < n2; i++) {
            buf_A11[i + j * n2] = A[i * n + j];
            buf_A12[i + j * n2] = A[i * n + (j + n2)];
            buf_A21[i + j * n2] = A[(i + n2) * n + j];
            buf_A22[i + j * n2] = A[(i + n2) * n + (j + n2)];
        }
    }

    // 步骤 1: 求 A11^(-1) (原地，结果存于 buf_A11)
    lapack_int nn = n2, lda = n2, info = 0;
    dgetrf_(&nn, &nn, buf_A11, &lda, buf_ipiv, &info);
    if (info != 0) return;
    double wq; lapack_int lw = -1;
    dgetri_(&nn, buf_A11, &lda, buf_ipiv, &wq, &lw, &info);
    lw = (lapack_int)wq;
    dgetri_(&nn, buf_A11, &lda, buf_ipiv, buf_wk, &lw, &info);

    // 步骤 2: 计算 Schur 补 S = A22 - A21 × A11^(-1) × A12
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, 1.0, buf_A21, n2, buf_A11, n2, 0.0, buf_Temp, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, 1.0, buf_Temp, n2, buf_A12, n2, 0.0, buf_S, n2);
    for (int i = 0; i < n2sq; i++) buf_S[i] = buf_A22[i] - buf_S[i];

    // 步骤 3: 求 S^(-1) (原地，结果存于 buf_S)
    dgetrf_(&nn, &nn, buf_S, &lda, buf_ipiv, &info);
    if (info != 0) return;
    lw = -1;
    dgetri_(&nn, buf_S, &lda, buf_ipiv, &wq, &lw, &info);
    lw = (lapack_int)wq;
    dgetri_(&nn, buf_S, &lda, buf_ipiv, buf_wk, &lw, &info);

    // 步骤 4: 计算右上块 TopRight = -A11^(-1) × A12 × S^(-1)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, 1.0, buf_A11, n2, buf_A12, n2, 0.0, buf_Temp, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, -1.0, buf_Temp, n2, buf_S, n2, 0.0, buf_TR, n2);

    // 步骤 5: 计算左下块 BotLeft = -S^(-1) × A21 × A11^(-1)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, 1.0, buf_S, n2, buf_A21, n2, 0.0, buf_Temp, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, -1.0, buf_Temp, n2, buf_A11, n2, 0.0, buf_BL, n2);

    // 步骤 6: 计算左上块 TopLeft = A11^(-1) + A11^(-1) × A12 × S^(-1) × A21 × A11^(-1)
    //   = A11^(-1) - TopRight × A21 × A11^(-1)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, -1.0, buf_TR, n2, buf_A21, n2, 0.0, buf_Temp, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, 1.0, buf_Temp, n2, buf_A11, n2, 0.0, buf_Temp2, n2);
    for (int i = 0; i < n2sq; i++) buf_Temp2[i] = buf_A11[i] + buf_Temp2[i];

    // 组装结果（列优先转行优先）
    double *dst = Ai_e;
    for (int j = 0; j < n2; j++) {
        for (int i = 0; i < n2; i++) {
            int i2 = i + n2;
            dst[i * n + j] = buf_Temp2[i + j * n2];       // 左上
            dst[i * n + j + n2] = buf_TR[i + j * n2];     // 右上
            dst[i2 * n + j] = buf_BL[i + j * n2];         // 左下
            dst[i2 * n + j + n2] = buf_S[i + j * n2];     // 右下
        }
    }
}

/**
 * 主函数：性能对比测试
 * 测试三个基准文件，每个矩阵重复测试多次取平均
 *
 * 计时说明：
 * - 只计时核心求逆操作（dgetrf+dgetri / eigen inverse / blockwise）
 * - 残差检查在计时外进行，仅用于验证正确性
 */
int main() {
    // 测试文件列表
    const char *files[] = {"benchmark1_1.mtx", "benchmark10_1.mtx", "benchmark1000_1.mtx"};
    int n, tot;
    int num_runs = 1000;  // 每个矩阵重复次数

    printf("LAPACK vs Eigen vs Blockwise 性能对比\n");
    printf("每矩阵重复测试 %d 次\n", num_runs);
    printf("计时范围：仅核心求逆操作（不含残差检查）\n\n");

    // 输出 Markdown 表格格式
    printf("| 文件 | 方法 | 时间 (s) | 残差 |\n");
    printf("|------|------|---------|------|\n");

    for (int f = 0; f < 3; f++) {
        // 加载文件，获取矩阵信息
        load_matrix(files[f], &n, 0, &tot);
        double t_l = 0, t_e = 0, t_b = 0, r_l = 0, r_e = 0, r_b = 0;

        // 遍历文件中所有矩阵
        for (int i = 0; i < tot; i++) {
            load_matrix(files[f], &n, i, &tot);

            // 每种方法重复测试 num_runs 次
            for (int run = 0; run < num_runs; run++) {
                // === 只计时核心求逆操作 ===
                clock_t s;

                s = clock();
                lapack_inv_core(n);
                t_l += (clock()-s)/(double)CLOCKS_PER_SEC;

                s = clock();
                eigen_inv_core(n);
                t_e += (clock()-s)/(double)CLOCKS_PER_SEC;

                s = clock();
                blockwise_inv_core(n);
                t_b += (clock()-s)/(double)CLOCKS_PER_SEC;

                // === 残差检查在计时外进行 ===
                r_l += compute_residual(n, Ai_l);
                r_e += compute_residual(n, Ai_e);
                r_b += compute_residual(n, Ai_e);
            }
        }

        // 输出平均时间和平均残差
        int total = tot * num_runs;
        printf("| %s | LAPACK    | %.6f | %.2e |\n", files[f], t_l/total, r_l/total);
        printf("| %s | Eigen     | %.6f | %.2e |\n", files[f], t_e/total, r_e/total);
        printf("| %s | Blockwise | %.6f | %.2e |\n", files[f], t_b/total, r_b/total);
    }

    return 0;
}
