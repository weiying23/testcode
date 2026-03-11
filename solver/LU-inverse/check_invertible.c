/**
 * ============================================================================
 * 文件名：check_invertible.c
 * 功能：检测 benchmark 文件中哪些矩阵是数值可逆的
 * 平台：macOS (使用 Accelerate 框架)
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

#ifdef __APPLE__
    typedef long lapack_int;
#endif

// 从文件加载矩阵
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
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            A[i * cols + j] = col_data[i + j * rows];
    free(col_data);
    return 1;
}

// 检查矩阵可逆性，返回最小主元和条件数估计
int check_invertible(int n, const double *A, double *min_pivot, double *cond_est) {
    double *A_copy = (double *)malloc(n * n * sizeof(double));
    lapack_int *ipiv = (lapack_int *)malloc(n * sizeof(lapack_int));

    // 转置为列优先
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A_copy[i + j * n] = A[i * n + j];

    lapack_int nn = (lapack_int)n, lda = (lapack_int)n, info = 0;
    dgetrf_(&nn, &nn, A_copy, &lda, ipiv, &info);

    // 找最小和最大主元
    double max_pivot = 0.0;
    *min_pivot = 1e300;
    for (int i = 0; i < n; i++) {
        double p = fabs(A_copy[i + i * n]);
        if (p < *min_pivot) *min_pivot = p;
        if (p > max_pivot) max_pivot = p;
    }

    *cond_est = (*min_pivot > 1e-300) ? max_pivot / *min_pivot : 1e300;

    // 可逆判定：info=0 且最小主元 > 1e-12
    int invertible = (info == 0 && *min_pivot > 1e-12);

    free(A_copy); free(ipiv);
    return invertible;
}

int main() {
    const char *files[] = {"benchmark1_1.mtx", "benchmark10_1.mtx", "benchmark1000_1.mtx"};
    int n, total;
    double *A = (double *)malloc(40 * 40 * sizeof(double));

    printf("=============================================================\n");
    printf("  矩阵可逆性检测程序\n");
    printf("=============================================================\n\n");
    printf("判定标准：最小主元 > 1e-12 为可逆\n\n");

    int grand_total_invertible = 0;
    int grand_total = 0;

    // 存储可逆矩阵索引
    int invertible_indices[100];

    for (int f = 0; f < 3; f++) {
        load_matrix_from_file(files[f], &n, A, 0, &total);

        int file_invertible = 0;
        int idx_count = 0;
        double min_cond = 1e300, max_cond = 0.0;

        printf("文件：%s\n", files[f]);
        printf("------------------------------------------\n");

        for (int i = 0; i < total; i++) {
            load_matrix_from_file(files[f], &n, A, i, &total);
            double min_pivot, cond;
            int inv = check_invertible(n, A, &min_pivot, &cond);

            if (inv) {
                file_invertible++;
                invertible_indices[idx_count++] = i;
                if (cond < min_cond) min_cond = cond;
                if (cond > max_cond) max_cond = cond;
            }
        }

        printf("  可逆矩阵：%d/%d (%.1f%%)\n", file_invertible, total, 100.0 * file_invertible / total);
        printf("  可逆索引：");
        for (int i = 0; i < idx_count; i++) {
            printf("%d", invertible_indices[i]);
            if (i < idx_count - 1) printf(", ");
        }
        printf("\n");
        printf("  条件数范围：%.4e ~ %.4e\n\n", min_cond, max_cond);

        grand_total_invertible += file_invertible;
        grand_total += total;
    }

    printf("=============================================================\n");
    printf("  总计：%d/%d 可逆 (%.1f%%)\n", grand_total_invertible, grand_total,
           100.0 * grand_total_invertible / grand_total);
    printf("=============================================================\n");

    printf("\n可逆矩阵索引列表 (三个文件相同):\n");
    printf("invertible_indices[] = {");
    for (int i = 0; i < 28; i++) {
        printf("%d", invertible_indices[i]);
        if (i < 27) printf(", ");
    }
    printf("};\n");

    free(A);
    return 0;
}
