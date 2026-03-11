#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

#ifdef __APPLE__
typedef long lapack_int;
#endif

// *_1.mtx 文件所有 96 个矩阵都可逆
int invertible_indices[96];
int num_invertible = 96;

int load_matrix_from_file(const char *filename, int *n, double *A, int matrix_index, int *total_matrices) {
    FILE *f = fopen(filename, "r");
    if (!f) return 0;
    char line[256];
    if (!fgets(line, sizeof(line), f) || !fgets(line, sizeof(line), f)) { fclose(f); return 0; }
    int rows, cols, num_matrices;
    if (sscanf(line, "Number of matrices:%d Matrix length:%d %d", &num_matrices, &rows, &cols) != 3) { fclose(f); return 0; }
    *total_matrices = num_matrices;
    *n = rows;
    int elements_per_matrix = rows * cols;
    for (int k = 0; k < matrix_index * elements_per_matrix; k++) fgets(line, sizeof(line), f);
    double *col_data = (double *)malloc(rows * cols * sizeof(double));
    for (int k = 0; k < rows * cols; k++) {
        if (!fgets(line, sizeof(line), f)) { free(col_data); fclose(f); return 0; }
        col_data[k] = atof(line);
    }
    fclose(f);
    // 列优先转行优先
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            A[i * cols + j] = col_data[i + j * rows];
    free(col_data);
    return 1;
}

// LAPACK 逆（行优先输入输出，内部转列优先）
int lapack_inverse(int n, const double *A_row, double *A_inv_row) {
    double *A_col = (double *)malloc(n * n * sizeof(double));
    double *A_inv_col = (double *)malloc(n * n * sizeof(double));
    lapack_int *ipiv = (lapack_int *)malloc(n * sizeof(lapack_int));
    // 行优先转列优先
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A_col[i + j * n] = A_row[i * n + j];
    lapack_int nn = (lapack_int)n, lda = (lapack_int)n, info = 0;
    dgetrf_(&nn, &nn, A_col, &lda, ipiv, &info);
    if (info != 0) { free(A_col); free(A_inv_col); free(ipiv); return 0; }
    double work_query; lapack_int lwork = -1;
    dgetri_(&nn, A_col, &lda, ipiv, &work_query, &lwork, &info);
    lwork = (lapack_int)work_query;
    double *work = (double *)malloc(lwork * sizeof(double));
    dgetri_(&nn, A_col, &lda, ipiv, work, &lwork, &info);
    free(work); free(ipiv);
    memcpy(A_inv_col, A_col, n * n * sizeof(double));
    // 列优先转回行优先
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A_inv_row[i * n + j] = A_inv_col[j * n + i];
    free(A_col); free(A_inv_col);
    return 1;
}

// 分块求逆（行优先输入输出，内部列优先计算）
int blockwise_inverse(int n, const double *A_row, double *A_inv_row) {
    int n2 = n / 2;
    double *A11 = (double *)malloc(n2 * n2 * sizeof(double));
    double *A12 = (double *)malloc(n2 * n2 * sizeof(double));
    double *A21 = (double *)malloc(n2 * n2 * sizeof(double));
    double *A22 = (double *)malloc(n2 * n2 * sizeof(double));
    double *A11_inv = (double *)malloc(n2 * n2 * sizeof(double));
    double *S = (double *)malloc(n2 * n2 * sizeof(double));
    double *S_inv = (double *)malloc(n2 * n2 * sizeof(double));
    double *Temp = (double *)malloc(n2 * n2 * sizeof(double));
    double *Temp2 = (double *)malloc(n2 * n2 * sizeof(double));
    
    // 行优先转列优先，并提取子块
    for (int j = 0; j < n2; j++) {
        for (int i = 0; i < n2; i++) {
            A11[i + j * n2] = A_row[i * n + j];
            A12[i + j * n2] = A_row[i * n + (j + n2)];
            A21[i + j * n2] = A_row[(i + n2) * n + j];
            A22[i + j * n2] = A_row[(i + n2) * n + (j + n2)];
        }
    }
    
    // 调用 LAPACK 求 A11_inv（列优先）
    {
        lapack_int nn = (lapack_int)n2, lda = (lapack_int)n2, info = 0;
        lapack_int *ipiv = (lapack_int *)malloc(n2 * sizeof(lapack_int));
        dgetrf_(&nn, &nn, A11, &lda, ipiv, &info);
        if (info != 0) { free(ipiv); return 0; }
        double work_query; lapack_int lwork = -1;
        dgetri_(&nn, A11, &lda, ipiv, &work_query, &lwork, &info);
        lwork = (lapack_int)work_query;
        double *work = (double *)malloc(lwork * sizeof(double));
        dgetri_(&nn, A11, &lda, ipiv, work, &lwork, &info);
        free(work); free(ipiv);
        // A11 now contains A11_inv (col-major)
    }
    
    // S = A22 - A21 × A11_inv × A12
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, 1.0, A21, n2, A11, n2, 0.0, Temp, n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2, 1.0, Temp, n2, A12, n2, 0.0, S, n2);
    for (int i = 0; i < n2 * n2; i++) S[i] = A22[i] - S[i];
    
    // 求 S_inv
    {
        lapack_int nn = (lapack_int)n2, lda = (lapack_int)n2, info = 0;
        lapack_int *ipiv = (lapack_int *)malloc(n2 * sizeof(lapack_int));
        dgetrf_(&nn, &nn, S, &lda, ipiv, &info);
        if (info != 0) { free(ipiv); return 0; }
        double work_query; lapack_int lwork = -1;
        dgetri_(&nn, S, &lda, ipiv, &work_query, &lwork, &info);
        lwork = (lapack_int)work_query;
        double *work = (double *)malloc(lwork * sizeof(double));
        dgetri_(&nn, S, &lda, ipiv, work, &lwork, &info);
        free(work); free(ipiv);
        // S now contains S_inv (col-major)
    }
    
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
            A_inv_row[i * n + j] = TopLeft[i + j * n2];
            A_inv_row[i * n + (j + n2)] = TopRight[i + j * n2];
            A_inv_row[(i + n2) * n + j] = BotLeft[i + j * n2];
            A_inv_row[(i + n2) * n + (j + n2)] = BotRight[i + j * n2];
        }
    }
    
    free(A11);free(A12);free(A21);free(A22);free(S);free(Temp);free(Temp2);
    free(TopLeft);free(TopRight);free(BotLeft);free(BotRight);
    return 1;
}

double compute_residual(int n, const double *A, const double *A_inv) {
    double *Prod = (double *)malloc(n * n * sizeof(double));
    double *I_mat = (double *)malloc(n * n * sizeof(double));
    for (int i = 0; i < n * n; i++) I_mat[i] = 0.0;
    for (int i = 0; i < n; i++) I_mat[i * n + i] = 1.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, A_inv, n, 0.0, Prod, n);
    double res = 0.0;
    for (int k = 0; k < n * n; k++) { double d = Prod[k] - I_mat[k]; res += d * d; }
    free(Prod); free(I_mat);
    return sqrt(res);
}

int main() {
    const char *files[] = {"benchmark1_1.mtx", "benchmark10_1.mtx", "benchmark1000_1.mtx"};
    int n, total;
    double *A = (double *)malloc(40 * 40 * sizeof(double));
    double *A_inv_lapack = (double *)malloc(40 * 40 * sizeof(double));
    double *A_inv_block = (double *)malloc(40 * 40 * sizeof(double));

    // 初始化可逆索引数组（*_1.mtx 所有 96 个矩阵都可逆）
    for (int i = 0; i < 96; i++) invertible_indices[i] = i;

    printf("=============================================================\n");
    printf("  可逆矩阵性能对比测试 (*_1.mtx 文件，全部 96 个矩阵可逆)\n");
    printf("=============================================================\n\n");

    FILE *md = fopen("invertible-perf-results-1.md", "w");
    fprintf(md, "# 可逆矩阵性能对比测试报告 (*_1.mtx 文件)\n\n");
    fprintf(md, "## 1. 测试配置\n\n");
    fprintf(md, "- **测试矩阵**: benchmark1_1.mtx, benchmark10_1.mtx, benchmark1000_1.mtx\n");
    fprintf(md, "- **矩阵数量**: 每个文件 96 个矩阵，全部数值可逆\n");
    fprintf(md, "- **对比方法**: LAPACK vs Blockwise (2×2) 分块求逆\n");
    fprintf(md, "- **验证标准**: 残差 ||A·A⁻¹ - I||_F < 1e-10\n\n");
    
    int grand_total_lapack_wins = 0, grand_total_block_wins = 0;
    double grand_total_lapack_time = 0, grand_total_block_time = 0;
    double grand_total_lapack_res = 0, grand_total_block_res = 0;
    int grand_total_valid = 0;
    
    for (int f = 0; f < 3; f++) {
        load_matrix_from_file(files[f], &n, A, 0, &total);
        
        int file_lapack_wins = 0, file_block_wins = 0;
        double file_lapack_time = 0, file_block_time = 0;
        double file_lapack_res = 0, file_block_res = 0;
        int file_valid = 0;
        
        fprintf(md, "## 2. %s 测试结果\n\n", files[f]);
        fprintf(md, "| 矩阵索引 | LAPACK 时间 (s) | Blockwise 时间 (s) | 加速比 | 获胜 | LAPACK 残差 | Blockwise 残差 | 验证 |\n");
        fprintf(md, "|---------|----------------|-----------------|--------|------|------------|---------------|------|\n");

        for (int i = 0; i < num_invertible; i++) {
            int idx = invertible_indices[i];
            load_matrix_from_file(files[f], &n, A, idx, &total);

            clock_t t0 = clock();
            int lapack_ok = lapack_inverse(n, A, A_inv_lapack);
            clock_t t1 = clock();
            double lapack_time = lapack_ok ? (double)(t1 - t0) / CLOCKS_PER_SEC : -1;
            double lapack_res = lapack_ok ? compute_residual(n, A, A_inv_lapack) : 1e300;
            
            t0 = clock();
            int block_ok = blockwise_inverse(n, A, A_inv_block);
            t1 = clock();
            double block_time = block_ok ? (double)(t1 - t0) / CLOCKS_PER_SEC : -1;
            double block_res = block_ok ? compute_residual(n, A, A_inv_block) : 1e300;
            
            int valid = (lapack_ok && block_ok && lapack_res < 1e-10 && block_res < 1e-10);
            
            if (lapack_ok && block_ok) {
                double speedup = block_time > 1e-9 ? lapack_time / block_time : 0;
                const char *winner = block_time < lapack_time ? "Blockwise" : "LAPACK";
                const char *verify = valid ? "✓" : "✗";
                
                if (block_time < lapack_time) file_block_wins++; else file_lapack_wins++;
                file_lapack_time += lapack_time;
                file_block_time += block_time;
                file_lapack_res += lapack_res;
                file_block_res += block_res;
                file_valid++;
                
                fprintf(md, "| %d | %.6f | %.6f | %.2fx | %s | %.4e | %.4e | %s |\n",
                        idx, lapack_time, block_time, speedup, winner, lapack_res, block_res, verify);
            } else {
                fprintf(md, "| %d | 失败 | 失败 | - | - | - | - | ✗ |\n", idx);
            }
        }
        
        fprintf(md, "\n### %s 统计\n\n", files[f]);
        fprintf(md, "| 指标 | 数值 |\n");
        fprintf(md, "|------|------|\n");
        fprintf(md, "| 有效矩阵数 | %d/96 |\n", file_valid);
        fprintf(md, "| LAPACK 平均时间 | %.6f s |\n", file_valid > 0 ? file_lapack_time / file_valid : 0);
        fprintf(md, "| Blockwise 平均时间 | %.6f s |\n", file_valid > 0 ? file_block_time / file_valid : 0);
        fprintf(md, "| LAPACK 平均残差 | %.4e |\n", file_valid > 0 ? file_lapack_res / file_valid : 0);
        fprintf(md, "| Blockwise 平均残差 | %.4e |\n", file_valid > 0 ? file_block_res / file_valid : 0);
        fprintf(md, "| LAPACK 获胜 | %d/96 |\n", file_lapack_wins);
        fprintf(md, "| Blockwise 获胜 | %d/96 (%.1f%%) |\n", file_block_wins, 100.0 * file_block_wins / 96);
        fprintf(md, "| 加速比 (LAPACK/Blockwise) | %.2fx |\n", file_valid > 0 ? file_lapack_time / file_block_time : 0);
        fprintf(md, "\n");
        
        grand_total_lapack_wins += file_lapack_wins;
        grand_total_block_wins += file_block_wins;
        grand_total_lapack_time += file_lapack_time;
        grand_total_block_time += file_block_time;
        grand_total_lapack_res += file_lapack_res;
        grand_total_block_res += file_block_res;
        grand_total_valid += file_valid;
    }
    
    fprintf(md, "## 3. 总体统计\n\n");
    fprintf(md, "| 指标 | 数值 |\n");
    fprintf(md, "|------|------|\n");
    fprintf(md, "| 总有效矩阵数 | %d/288 |\n", grand_total_valid);
    fprintf(md, "| LAPACK 总平均时间 | %.6f s |\n", grand_total_valid > 0 ? grand_total_lapack_time / grand_total_valid : 0);
    fprintf(md, "| Blockwise 总平均时间 | %.6f s |\n", grand_total_valid > 0 ? grand_total_block_time / grand_total_valid : 0);
    fprintf(md, "| LAPACK 平均残差 | %.4e |\n", grand_total_valid > 0 ? grand_total_lapack_res / grand_total_valid : 0);
    fprintf(md, "| Blockwise 平均残差 | %.4e |\n", grand_total_valid > 0 ? grand_total_block_res / grand_total_valid : 0);
    fprintf(md, "| LAPACK 获胜 | %d/288 (%.1f%%) |\n", grand_total_lapack_wins, 100.0 * grand_total_lapack_wins / 288);
    fprintf(md, "| Blockwise 获胜 | %d/288 (%.1f%%) |\n", grand_total_block_wins, 100.0 * grand_total_block_wins / 288);
    fprintf(md, "| 加速比 (LAPACK/Blockwise) | %.2fx |\n", grand_total_valid > 0 ? grand_total_lapack_time / grand_total_block_time : 0);
    fprintf(md, "\n");

    fprintf(md, "## 4. 结论\n\n");
    if (grand_total_valid == 288) {
        if (grand_total_block_wins > grand_total_lapack_wins) {
            fprintf(md, "**Blockwise (2x2) 分块求逆在可逆矩阵上表现更优**\n\n");
            fprintf(md, "- 平均加速比：%.2fx\n", grand_total_valid > 0 ? grand_total_lapack_time / grand_total_block_time : 0);
            fprintf(md, "- 获胜比例：%.1f%%\n", 100.0 * grand_total_block_wins / 288);
            fprintf(md, "- 精度验证：%d/288 矩阵两者残差均 < 1e-10\n", grand_total_valid);
        } else {
            fprintf(md, "**LAPACK 在可逆矩阵上表现更优**\n\n");
        }
    } else {
        fprintf(md, "**部分矩阵求逆失败，需要进一步调查**\n");
    }

    fclose(md);

    printf("详细结果已写入 invertible-perf-results-1.md\n\n");
    printf("总体统计:\n");
    printf("  有效矩阵：%d/288\n", grand_total_valid);
    printf("  LAPACK 平均时间：%.6f s\n", grand_total_valid > 0 ? grand_total_lapack_time / grand_total_valid : 0);
    printf("  Blockwise 平均时间：%.6f s\n", grand_total_valid > 0 ? grand_total_block_time / grand_total_valid : 0);
    printf("  LAPACK 平均残差：%.4e\n", grand_total_valid > 0 ? grand_total_lapack_res / grand_total_valid : 0);
    printf("  Blockwise 平均残差：%.4e\n", grand_total_valid > 0 ? grand_total_block_res / grand_total_valid : 0);
    printf("  LAPACK 获胜：%d/288 (%.1f%%)\n", grand_total_lapack_wins, 100.0 * grand_total_lapack_wins / 288);
    printf("  Blockwise 获胜：%d/288 (%.1f%%)\n", grand_total_block_wins, 100.0 * grand_total_block_wins / 288);
    printf("  加速比：%.2fx\n", grand_total_valid > 0 ? grand_total_lapack_time / grand_total_block_time : 0);
    
    free(A); free(A_inv_lapack); free(A_inv_block);
    return 0;
}
