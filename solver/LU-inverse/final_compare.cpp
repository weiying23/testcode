#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Accelerate/Accelerate.h>
#include <Eigen/Dense>

typedef long lapack_int;
double A[1600];       // 行优先
double A_col[1600];   // 列优先（预转换）
double Ai_l[1600], Ai_e[1600], Ai_n[1600];

void load_matrix(const char *fn, int *n, int idx, int *tot) {
    FILE *f = fopen(fn, "r");
    char line[256];
    fgets(line, sizeof(line), f);
    fgets(line, sizeof(line), f);
    int rows, cols, num;
    sscanf(line, "Number of matrices:%d Matrix length:%d %d", &num, &rows, &cols);
    *tot = num; *n = rows;
    for (int k = 0; k < idx * rows * cols; k++) fgets(line, sizeof(line), f);
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

double lapack_inv(int n) {
    double *Ac = (double *)malloc(n*n*sizeof(double));
    lapack_int *ipiv = (lapack_int *)malloc(n*sizeof(lapack_int));
    // 直接使用预转换的列优先数据 A_col
    for (int i = 0; i < n*n; i++) Ac[i] = A_col[i];
    lapack_int nn = n, lda = n, info = 0;
    dgetrf_(&nn, &nn, Ac, &lda, ipiv, &info);
    if (info != 0) { free(Ac); free(ipiv); return 1e300; }
    double wq; lapack_int lw = -1;
    dgetri_(&nn, Ac, &lda, ipiv, &wq, &lw, &info);
    lw = (lapack_int)wq;
    double *wk = (double *)malloc(lw*sizeof(double));
    dgetri_(&nn, Ac, &lda, ipiv, wk, &lw, &info);
    free(wk); free(ipiv);
    // 列优先转行优先存储结果
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Ai_l[i*n + j] = Ac[j*n + i];
    free(Ac);
    // 计算残差
    double *P = (double *)malloc(n*n*sizeof(double));
    double *I = (double *)malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++) I[i] = 0;
    for (int i = 0; i < n; i++) I[i*n + i] = 1;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, Ai_l, n, 0.0, P, n);
    double r = 0;
    for (int k = 0; k < n*n; k++) { double d = P[k] - I[k]; r += d*d; }
    free(P); free(I);
    return sqrt(r);
}

double eigen_inv(int n) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(A, n, n);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mati(Ai_e, n, n);
    mati = mat.inverse();
    double *P = (double *)malloc(n*n*sizeof(double));
    double *I = (double *)malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++) I[i] = 0;
    for (int i = 0; i < n; i++) I[i*n + i] = 1;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, Ai_e, n, 0.0, P, n);
    double r = 0;
    for (int k = 0; k < n*n; k++) { double d = P[k] - I[k]; r += d*d; }
    free(P); free(I);
    return sqrt(r);
}

double blockwise_inv(int n) {
    int n2 = n / 2;
    double *A11 = (double *)malloc(n2 * n2 * sizeof(double));
    double *A12 = (double *)malloc(n2 * n2 * sizeof(double));
    double *A21 = (double *)malloc(n2 * n2 * sizeof(double));
    double *A22 = (double *)malloc(n2 * n2 * sizeof(double));
    double *S = (double *)malloc(n2 * n2 * sizeof(double));
    double *Temp = (double *)malloc(n2 * n2 * sizeof(double));
    double *Temp2 = (double *)malloc(n2 * n2 * sizeof(double));

    // 从预转换的列优先数据 A_col 提取子块（已经是列优先）
    for (int j = 0; j < n2; j++) {
        for (int i = 0; i < n2; i++) {
            A11[i + j * n2] = A_col[i + j * n];
            A12[i + j * n2] = A_col[i + (j + n2) * n];
            A21[i + j * n2] = A_col[(i + n2) + j * n];
            A22[i + j * n2] = A_col[(i + n2) + (j + n2) * n];
        }
    }

    // 调用 LAPACK 求 A11_inv（列优先）
    lapack_int nn = n2, lda = n2, info = 0;
    lapack_int *ipiv = (lapack_int *)malloc(n2 * sizeof(lapack_int));
    dgetrf_(&nn, &nn, A11, &lda, ipiv, &info);
    if (info != 0) { free(ipiv); free(A11); free(A12); free(A21); free(A22); free(S); free(Temp); free(Temp2); return 1e300; }
    double wq; lapack_int lw = -1;
    dgetri_(&nn, A11, &lda, ipiv, &wq, &lw, &info);
    lw = (lapack_int)wq;
    double *wk = (double *)malloc(lw * sizeof(double));
    dgetri_(&nn, A11, &lda, ipiv, wk, &lw, &info);
    free(wk); free(ipiv);

    // S = A22 - A21 × A11_inv × A12
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, 1.0, A21, n2, A11, n2, 0.0, Temp, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, 1.0, Temp, n2, A12, n2, 0.0, S, n2);
    for (int i = 0; i < n2 * n2; i++) S[i] = A22[i] - S[i];

    // 求 S_inv
    ipiv = (lapack_int *)malloc(n2 * sizeof(lapack_int));
    dgetrf_(&nn, &nn, S, &lda, ipiv, &info);
    if (info != 0) { free(ipiv); free(A11); free(A12); free(A21); free(A22); free(S); free(Temp); free(Temp2); return 1e300; }
    lw = -1;
    dgetri_(&nn, S, &lda, ipiv, &wq, &lw, &info);
    lw = (lapack_int)wq;
    wk = (double *)malloc(lw * sizeof(double));
    dgetri_(&nn, S, &lda, ipiv, wk, &lw, &info);
    free(wk); free(ipiv);

    double *TopLeft = (double *)malloc(n2 * n2 * sizeof(double));
    double *TopRight = (double *)malloc(n2 * n2 * sizeof(double));
    double *BotLeft = (double *)malloc(n2 * n2 * sizeof(double));
    double *BotRight = (double *)malloc(n2 * n2 * sizeof(double));

    // BotRight = S_inv
    memcpy(BotRight, S, n2 * n2 * sizeof(double));

    // TopRight = -A11_inv × A12 × S_inv
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, 1.0, A11, n2, A12, n2, 0.0, Temp, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, -1.0, Temp, n2, S, n2, 0.0, TopRight, n2);

    // BotLeft = -S_inv × A21 × A11_inv
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, 1.0, S, n2, A21, n2, 0.0, Temp, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, -1.0, Temp, n2, A11, n2, 0.0, BotLeft, n2);

    // TopLeft = A11_inv + A11_inv × A12 × S_inv × A21 × A11_inv
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, -1.0, TopRight, n2, A21, n2, 0.0, Temp, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, 1.0, Temp, n2, A11, n2, 0.0, Temp2, n2);
    for (int i = 0; i < n2 * n2; i++) TopLeft[i] = A11[i] + Temp2[i];

    // 组装结果（列优先转行优先）
    for (int j = 0; j < n2; j++) {
        for (int i = 0; i < n2; i++) {
            Ai_n[i * n + j] = TopLeft[i + j * n2];
            Ai_n[i * n + (j + n2)] = TopRight[i + j * n2];
            Ai_n[(i + n2) * n + j] = BotLeft[i + j * n2];
            Ai_n[(i + n2) * n + (j + n2)] = BotRight[i + j * n2];
        }
    }

    free(A11);free(A12);free(A21);free(A22);free(S);free(Temp);free(Temp2);
    free(TopLeft);free(TopRight);free(BotLeft);free(BotRight);

    // 计算残差
    double *P = (double *)malloc(n*n*sizeof(double));
    double *I = (double *)malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++) I[i] = 0;
    for (int i = 0; i < n; i++) I[i*n + i] = 1;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, Ai_n, n, 0.0, P, n);
    double r = 0;
    for (int k = 0; k < n*n; k++) { double d = P[k] - I[k]; r += d*d; }
    free(P); free(I);
    return sqrt(r);
}

int main() {
    const char *files[] = {"benchmark1_1.mtx", "benchmark10_1.mtx", "benchmark1000_1.mtx"};
    int n, tot;
    int num_runs = 1000;  // 每矩阵重复测试次数

    printf("LAPACK vs Eigen vs Blockwise (2x2) 性能对比 (40x40 矩阵)\n");
    printf("每矩阵重复测试 %d 次，总计 %d 矩阵\n\n", num_runs, 3 * 96);

    printf("| 文件 | 方法 | 平均时间 (s) | 平均残差 |\n");
    printf("|------|------|--------------|----------|\n");

    for (int f = 0; f < 3; f++) {
        load_matrix(files[f], &n, 0, &tot);
        double t_l = 0, t_e = 0, t_b = 0, r_l = 0, r_e = 0, r_b = 0;

        for (int i = 0; i < tot; i++) {
            load_matrix(files[f], &n, i, &tot);

            // 多轮重复测试
            for (int run = 0; run < num_runs; run++) {
                clock_t s;
                s = clock(); r_l += lapack_inv(n); t_l += (clock()-s)/(double)CLOCKS_PER_SEC;
                s = clock(); r_e += eigen_inv(n); t_e += (clock()-s)/(double)CLOCKS_PER_SEC;
                s = clock(); r_b += blockwise_inv(n); t_b += (clock()-s)/(double)CLOCKS_PER_SEC;
            }
        }

        int total = tot * num_runs;
        printf("| %s | LAPACK   | %.6f | %.2e |\n", files[f], t_l/total, r_l/total);
        printf("| %s | Eigen    | %.6f | %.2e |\n", files[f], t_e/total, r_e/total);
        printf("| %s | Blockwise| %.6f | %.2e |\n", files[f], t_b/total, r_b/total);
    }

    return 0;
}
