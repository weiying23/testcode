#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <float.h>

/* 定义三对角矩阵结构 */
typedef struct {
    int n;          /* 矩阵大小 */
    double *main;   /* 主对角线 */
    double *upper;  /* 上对角线 */
    double *lower;  /* 下对角线 */
} TridiagMatrix;

/* 稀疏矩阵的CSR格式（用于ILUT） */
typedef struct {
    int n;
    int *row_ptr;
    int *col_ind;
    double *values;
    int nnz;        /* 非零元数量 */
} CSRMatrix;

/* ILU分解的L和U部分 */
typedef struct {
    CSRMatrix *L;
    CSRMatrix *U;
} ILUFactors;

/* 函数声明 */
void generate_high_condition_tridiag_matrix(TridiagMatrix *A, int n, double condition_number);
void free_tridiag_matrix(TridiagMatrix *A);
void tridiag_to_csr(const TridiagMatrix *A, CSRMatrix *B);
void free_csr_matrix(CSRMatrix *A);
void ilu0_factorize(const TridiagMatrix *A, ILUFactors *factors);
void ilut_factorize(const TridiagMatrix *A, ILUFactors *factors, double tau, int lfil);
void free_ilu_factors(ILUFactors *factors);
void apply_ilu0_preconditioner(const ILUFactors *factors, const double *b, double *y, int n);
void apply_ilut_preconditioner(const ILUFactors *factors, const double *b, double *y, int n);
void matvec_tridiag(const TridiagMatrix *A, const double *x, double *y);
double vector_norm(const double *v, int n);
double estimate_matrix_norm(const TridiagMatrix *A, int n, int max_iter);
double estimate_condition_number(const TridiagMatrix *A, int n, int max_iter);
void gmres(TridiagMatrix *A, const double *b, double *x, int n, int max_iter, 
           double tol, const char *precond_type, double tau, int lfil, 
           int *iter_used, double *res_norms, double *time_used);
void print_convergence(const char *method, int iter_used, double *res_norms, double time_used, double initial_res);
void print_comparison(int iter_none, double time_none, int iter_ilu0, double time_ilu0, 
                     int iter_ilut, double time_ilut, double cond_est);

int main() {
    int n = 1000;           /* 矩阵大小 */
    int max_iter = n;       /* 最大迭代次数 */
    double tol = 1e-8;      /* 收敛容差 */
    
    /* 为ILUT设置参数 */
    double tau = 0.01;      /* 阈值 */
    int lfil = 5;           /* 每行最大非零元数 */
    
    /* 生成高条件数的三对角矩阵 */
    TridiagMatrix A;
    double target_condition = 1e6;  /* 目标条件数 */
    generate_high_condition_tridiag_matrix(&A, n, target_condition);
    
    /* 估计条件数 */
    double cond_est = estimate_condition_number(&A, n, 100);
    printf("生成的矩阵估计条件数: %.2e\n\n", cond_est);
    
    /* 生成右侧向量b (Ax = b，其中x是全1向量) */
    double *x_true = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    
    /* x_true为全1向量 */
    for (int i = 0; i < n; i++) {
        x_true[i] = 1.0;
    }
    
    /* 计算b = A * x_true */
    matvec_tridiag(&A, x_true, b);
    
    /* 为记录收敛历史准备 */
    double *res_norms_none = (double *)malloc((max_iter + 1) * sizeof(double));
    double *res_norms_ilu0 = (double *)malloc((max_iter + 1) * sizeof(double));
    double *res_norms_ilut = (double *)malloc((max_iter + 1) * sizeof(double));
    int iter_used_none, iter_used_ilu0, iter_used_ilut;
    double time_none, time_ilu0, time_ilut;
    
    printf("开始求解 %d x %d 高条件数三对角线性系统...\n\n", n, n);
    
    /* 无预条件的GMRES */
    printf("===== 无预条件的GMRES =====\n");
    for (int i = 0; i < n; i++) x[i] = 0.0;  /* 重置解向量 */
    gmres(&A, b, x, n, max_iter, tol, "none", 0.0, 0, 
          &iter_used_none, res_norms_none, &time_none);
    print_convergence("无预条件", iter_used_none, res_norms_none, time_none, res_norms_none[0]);
    
    /* 使用ILU0预条件的GMRES */
    printf("\n===== 使用ILU0预条件的GMRES =====\n");
    for (int i = 0; i < n; i++) x[i] = 0.0;  /* 重置解向量 */
    gmres(&A, b, x, n, max_iter, tol, "ilu0", 0.0, 0, 
          &iter_used_ilu0, res_norms_ilu0, &time_ilu0);
    print_convergence("ILU0", iter_used_ilu0, res_norms_ilu0, time_ilu0, res_norms_ilu0[0]);
    
    /* 使用ILUT预条件的GMRES */
    printf("\n===== 使用ILUT预条件的GMRES =====\n");
    for (int i = 0; i < n; i++) x[i] = 0.0;  /* 重置解向量 */
    gmres(&A, b, x, n, max_iter, tol, "ilut", tau, lfil, 
          &iter_used_ilut, res_norms_ilut, &time_ilut);
    print_convergence("ILUT", iter_used_ilut, res_norms_ilut, time_ilut, res_norms_ilut[0]);
    
    /* 对比结果 */
    printf("\n");
    // 修复点：将iter_none, iter_ilu0, iter_ilut 改为 iter_used_none, iter_used_ilu0, iter_used_ilut
    print_comparison(iter_used_none, time_none, iter_used_ilu0, time_ilu0, 
                    iter_used_ilut, time_ilut, cond_est);
    
    /* 释放内存 */
    free_tridiag_matrix(&A);
    free(x_true);
    free(b);
    free(x);
    free(res_norms_none);
    free(res_norms_ilu0);
    free(res_norms_ilut);
    
    return 0;
}

/* 生成高条件数的三对角矩阵 */
void generate_high_condition_tridiag_matrix(TridiagMatrix *A, int n, double condition_number) {
    A->n = n;
    A->main = (double *)malloc(n * sizeof(double));
    A->upper = (double *)malloc((n-1) * sizeof(double));
    A->lower = (double *)malloc((n-1) * sizeof(double));
    
    srand(time(NULL) ^ 42);  /* 固定随机种子以获得可重现的结果 */
    
    /* 创建高条件数的三对角矩阵 */
    /* 思路：主对角线从大值逐渐变小，使条件数变大 */
    double min_diag = 1.0;
    double max_diag = condition_number * min_diag;
    
    /* 使用指数分布使条件数变大 */
    for (int i = 0; i < n; i++) {
        /* 主对角线：从大值逐渐变小 */
        A->main[i] = max_diag * exp(-6.0 * i / n) + 0.1;
        
        /* 确保对角占优，但不要过分 */
        double off_diag_max = 0.49 * A->main[i];
        
        /* 上对角线和下对角线 */
        if (i < n-1) {
            /* 随机选择接近最大值的非对角元素以增加条件数 */
            A->upper[i] = -off_diag_max * (0.8 + 0.2 * (rand() / (double)RAND_MAX));
            A->lower[i] = A->upper[i];  /* 对称矩阵 */
        }
    }
    
    /* 确保最后一个对角元不太小 */
    A->main[n-1] = fmax(A->main[n-1], 0.1);
    
    /* 打印一些矩阵信息 */
    printf("生成的三对角矩阵信息:\n");
    printf("  主对角线范围: [%.4e, %.4e]\n", A->main[n-1], A->main[0]);
    printf("  非对角线范围: [%.4e, %.4e]\n", A->upper[n-2], A->upper[0]);
}

/* 估计矩阵2-范数（使用幂迭代法） */
double estimate_matrix_norm(const TridiagMatrix *A, int n, int max_iter) {
    double *v = (double *)malloc(n * sizeof(double));
    double *Av = (double *)malloc(n * sizeof(double));
    double norm = 1.0;
    
    /* 初始化为随机向量 */
    srand(42);
    for (int i = 0; i < n; i++) {
        v[i] = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
    }
    
    /* 归一化 */
    double v_norm = vector_norm(v, n);
    for (int i = 0; i < n; i++) {
        v[i] /= v_norm;
    }
    
    /* 幂迭代 */
    for (int iter = 0; iter < max_iter; iter++) {
        /* 计算 Av */
        matvec_tridiag(A, v, Av);
        
        /* 计算新范数 */
        double Av_norm = vector_norm(Av, n);
        
        /* 检查收敛 */
        if (iter > 0 && fabs(Av_norm - norm) < 1e-6 * norm) {
            break;
        }
        
        norm = Av_norm;
        
        /* 归一化 */
        for (int i = 0; i < n; i++) {
            v[i] = Av[i] / norm;
        }
    }
    
    free(v);
    free(Av);
    
    return norm;
}

/* 估计矩阵条件数（使用幂迭代和反幂迭代） */
double estimate_condition_number(const TridiagMatrix *A, int n, int max_iter) {
    /* 估计最大特征值（使用幂迭代） */
    double norm = estimate_matrix_norm(A, n, max_iter);
    
    /* 估计最小特征值（使用反幂迭代） */
    double *v = (double *)malloc(n * sizeof(double));
    double *z = (double *)malloc(n * sizeof(double));
    double min_eig = 1.0;
    
    /* 初始化为随机向量 */
    srand(43);
    for (int i = 0; i < n; i++) {
        v[i] = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
    }
    
    /* 归一化 */
    double v_norm = vector_norm(v, n);
    for (int i = 0; i < n; i++) {
        v[i] /= v_norm;
    }
    
    /* 创建ILU0分解用于反幂迭代 */
    ILUFactors ilu_factors;
    ilu0_factorize(A, &ilu_factors);
    
    /* 反幂迭代 */
    for (int iter = 0; iter < max_iter; iter++) {
        /* 求解 Az = v */
        apply_ilu0_preconditioner(&ilu_factors, v, z, n);
        
        /* 计算新范数 */
        double z_norm = vector_norm(z, n);
        
        /* 检查收敛 */
        if (iter > 0 && fabs(z_norm - min_eig) < 1e-6 * min_eig) {
            break;
        }
        
        min_eig = z_norm;
        
        /* 归一化 */
        for (int i = 0; i < n; i++) {
            v[i] = z[i] / min_eig;
        }
    }
    
    free_ilu_factors(&ilu_factors);
    free(v);
    free(z);
    
    /* 条件数是最大特征值与最小特征值的比值 */
    return norm / min_eig;
}

/* 释放三对角矩阵内存 */
void free_tridiag_matrix(TridiagMatrix *A) {
    free(A->main);
    free(A->upper);
    free(A->lower);
    A->n = 0;
}

/* 三对角矩阵转为CSR格式 */
void tridiag_to_csr(const TridiagMatrix *A, CSRMatrix *B) {
    int n = A->n;
    int nnz = n + 2 * (n-1);  /* 非零元总数 */
    
    B->n = n;
    B->nnz = nnz;
    B->row_ptr = (int *)malloc((n+1) * sizeof(int));
    B->col_ind = (int *)malloc(nnz * sizeof(int));
    B->values = (double *)malloc(nnz * sizeof(double));
    
    int idx = 0;
    B->row_ptr[0] = 0;
    
    for (int i = 0; i < n; i++) {
        /* 下对角线 */
        if (i > 0) {
            B->col_ind[idx] = i-1;
            B->values[idx] = A->lower[i-1];
            idx++;
        }
        
        /* 主对角线 */
        B->col_ind[idx] = i;
        B->values[idx] = A->main[i];
        idx++;
        
        /* 上对角线 */
        if (i < n-1) {
            B->col_ind[idx] = i+1;
            B->values[idx] = A->upper[i];
            idx++;
        }
        
        B->row_ptr[i+1] = idx;
    }
}

/* 释放CSR矩阵内存 */
void free_csr_matrix(CSRMatrix *A) {
    free(A->row_ptr);
    free(A->col_ind);
    free(A->values);
    A->n = 0;
    A->nnz = 0;
}

/* ILU0分解：对于三对角矩阵非常简单 */
void ilu0_factorize(const TridiagMatrix *A, ILUFactors *factors) {
    int n = A->n;
    
    /* 为L和U分配空间 */
    factors->L = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    factors->U = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    
    /* L是单位下三角，只有下对角线 */
    factors->L->n = n;
    factors->L->nnz = n-1;
    factors->L->row_ptr = (int *)malloc((n+1) * sizeof(int));
    factors->L->col_ind = (int *)malloc((n-1) * sizeof(int));
    factors->L->values = (double *)malloc((n-1) * sizeof(double));
    
    /* U是上三角，有主对角线和上对角线 */
    factors->U->n = n;
    factors->U->nnz = n + (n-1);
    factors->U->row_ptr = (int *)malloc((n+1) * sizeof(int));
    factors->U->col_ind = (int *)malloc((2*n-1) * sizeof(int));
    factors->U->values = (double *)malloc((2*n-1) * sizeof(double));
    
    /* ILU0分解计算 */
    double *L_val = factors->L->values;
    double *U_main = factors->U->values;
    double *U_upper = factors->U->values + n;
    
    /* 复制主对角线到U */
    for (int i = 0; i < n; i++) {
        U_main[i] = A->main[i];
    }
    
    /* 复制上对角线到U */
    for (int i = 0; i < n-1; i++) {
        U_upper[i] = A->upper[i];
    }
    
    /* 计算L和调整U */
    for (int i = 1; i < n; i++) {
        L_val[i-1] = A->lower[i-1] / U_main[i-1];
        U_main[i] -= L_val[i-1] * U_upper[i-1];
    }
    
    /* 填充CSR结构的索引 */
    /* L的结构 */
    factors->L->row_ptr[0] = 0;
    for (int i = 0; i < n-1; i++) {
        factors->L->col_ind[i] = i;
        factors->L->row_ptr[i+1] = i+1;
    }
    factors->L->row_ptr[n] = n-1;
    
    /* U的结构 */
    factors->U->row_ptr[0] = 0;
    for (int i = 0; i < n; i++) {
        if (i < n-1)
            factors->U->col_ind[i] = i;
        if (i > 0)
            factors->U->col_ind[n-1+i-1] = i;
        
        factors->U->row_ptr[i+1] = i + (i < n-1 ? 1 : 0);
    }
}

/* ILUT分解：阈值不完全LU分解 */
void ilut_factorize(const TridiagMatrix *A, ILUFactors *factors, double tau, int lfil) {
    /* 对于三对角矩阵，ILUT与ILU0基本相同，因为不会有填充 */
    /* 但为了演示，我们实现一个通用的ILUT框架 */
    
    CSRMatrix A_csr;
    tridiag_to_csr(A, &A_csr);
    
    /* 为L和U分配初始空间 */
    factors->L = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    factors->U = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    
    int n = A->n;
    int max_nnz = n + 2 * (n-1);  /* 最大非零元数 */
    
    /* 初始化L和U */
    factors->L->n = n;
    factors->L->nnz = 0;
    factors->L->row_ptr = (int *)malloc((n+1) * sizeof(int));
    factors->L->col_ind = (int *)malloc(max_nnz * sizeof(int));
    factors->L->values = (double *)malloc(max_nnz * sizeof(double));
    
    factors->U->n = n;
    factors->U->nnz = 0;
    factors->U->row_ptr = (int *)malloc((n+1) * sizeof(int));
    factors->U->col_ind = (int *)malloc(max_nnz * sizeof(int));
    factors->U->values = (double *)malloc(max_nnz * sizeof(double));
    
    /* 先执行ILU0分解 */
    double *L_val = (double *)malloc((n-1) * sizeof(double));
    double *U_main = (double *)malloc(n * sizeof(double));
    double *U_upper = (double *)malloc((n-1) * sizeof(double));
    
    /* 复制主对角线到U */
    for (int i = 0; i < n; i++) {
        U_main[i] = A->main[i];
    }
    
    /* 复制上对角线到U */
    for (int i = 0; i < n-1; i++) {
        U_upper[i] = A->upper[i];
    }
    
    /* 计算L和调整U */
    for (int i = 1; i < n; i++) {
        L_val[i-1] = A->lower[i-1] / U_main[i-1];
        U_main[i] -= L_val[i-1] * U_upper[i-1];
    }
    
    /* 应用阈值筛选 */
    int *L_col = factors->L->col_ind;
    double *L_values = factors->L->values;
    int *U_col = factors->U->col_ind;
    double *U_values = factors->U->values;
    
    /* L的结构 (下三角部分) */
    factors->L->row_ptr[0] = 0;
    int L_nnz = 0;
    
    for (int i = 1; i < n; i++) {
        double threshold = tau * fabs(A->main[i-1]);
        
        /* 应用阈值 */
        if (fabs(L_val[i-1]) > threshold) {
            L_col[L_nnz] = i-1;
            L_values[L_nnz] = L_val[i-1];
            L_nnz++;
        }
        factors->L->row_ptr[i] = L_nnz;
    }
    factors->L->row_ptr[n] = L_nnz;
    factors->L->nnz = L_nnz;
    
    /* U的结构 (上三角部分) */
    factors->U->row_ptr[0] = 0;
    int U_nnz = 0;
    
    for (int i = 0; i < n; i++) {
        /* 主对角线总是保留 */
        U_col[U_nnz] = i;
        U_values[U_nnz] = U_main[i];
        U_nnz++;
        
        /* 上对角线 */
        if (i < n-1) {
            double threshold = tau * fabs(A->main[i]);
            
            /* 应用阈值 */
            if (fabs(U_upper[i]) > threshold) {
                U_col[U_nnz] = i+1;
                U_values[U_nnz] = U_upper[i];
                U_nnz++;
            }
        }
        factors->U->row_ptr[i+1] = U_nnz;
    }
    factors->U->nnz = U_nnz;
    
    /* 重新分配内存以匹配实际非零元数 */
    factors->L->col_ind = (int *)realloc(factors->L->col_ind, L_nnz * sizeof(int));
    factors->L->values = (double *)realloc(factors->L->values, L_nnz * sizeof(double));
    
    factors->U->col_ind = (int *)realloc(factors->U->col_ind, U_nnz * sizeof(int));
    factors->U->values = (double *)realloc(factors->U->values, U_nnz * sizeof(double));
    
    /* 释放临时内存 */
    free(L_val);
    free(U_main);
    free(U_upper);
    free_csr_matrix(&A_csr);
}

/* 释放ILU分解因子 */
void free_ilu_factors(ILUFactors *factors) {
    free_csr_matrix(factors->L);
    free_csr_matrix(factors->U);
    free(factors->L);
    free(factors->U);
}

/* 应用ILU0预条件器 */
void apply_ilu0_preconditioner(const ILUFactors *factors, const double *b, double *y, int n) {
    /* 前向替换: Lz = b */
    double *z = (double *)malloc(n * sizeof(double));
    
    z[0] = b[0];
    for (int i = 1; i < n; i++) {
        z[i] = b[i];
        for (int j = factors->L->row_ptr[i]; j < factors->L->row_ptr[i+1]; j++) {
            int col = factors->L->col_ind[j];
            z[i] -= factors->L->values[j] * z[col];
        }
    }
    
    /* 后向替换: Uy = z */
    for (int i = n-1; i >= 0; i--) {
        y[i] = z[i];
        for (int j = factors->U->row_ptr[i] + 1; j < factors->U->row_ptr[i+1]; j++) {
            int col = factors->U->col_ind[j];
            y[i] -= factors->U->values[j] * y[col];
        }
        y[i] /= factors->U->values[factors->U->row_ptr[i]];
    }
    
    free(z);
}

/* 应用ILUT预条件器 */
void apply_ilut_preconditioner(const ILUFactors *factors, const double *b, double *y, int n) {
    /* 前向替换: Lz = b */
    double *z = (double *)malloc(n * sizeof(double));
    
    z[0] = b[0];
    for (int i = 1; i < n; i++) {
        z[i] = b[i];
        for (int j = factors->L->row_ptr[i]; j < factors->L->row_ptr[i+1]; j++) {
            int col = factors->L->col_ind[j];
            z[i] -= factors->L->values[j] * z[col];
        }
    }
    
    /* 后向替换: Uy = z */
    for (int i = n-1; i >= 0; i--) {
        y[i] = z[i];
        for (int j = factors->U->row_ptr[i] + 1; j < factors->U->row_ptr[i+1]; j++) {
            int col = factors->U->col_ind[j];
            y[i] -= factors->U->values[j] * y[col];
        }
        y[i] /= factors->U->values[factors->U->row_ptr[i]];
    }
    
    free(z);
}

/* 三对角矩阵-向量乘法 */
void matvec_tridiag(const TridiagMatrix *A, const double *x, double *y) {
    int n = A->n;
    
    /* 第一行 */
    y[0] = A->main[0] * x[0] + A->upper[0] * x[1];
    
    /* 中间行 */
    for (int i = 1; i < n-1; i++) {
        y[i] = A->lower[i-1] * x[i-1] + A->main[i] * x[i] + A->upper[i] * x[i+1];
    }
    
    /* 最后一行 */
    y[n-1] = A->lower[n-2] * x[n-2] + A->main[n-1] * x[n-1];
}

/* 计算向量2-范数 */
double vector_norm(const double *v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

/* GMRES求解器 */
void gmres(TridiagMatrix *A, const double *b, double *x, int n, int max_iter, 
           double tol, const char *precond_type, double tau, int lfil, 
           int *iter_used, double *res_norms, double *time_used) {
    /* 获取开始时间 */
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    /* 初始化 */
    double *r = (double *)malloc(n * sizeof(double));
    double *z = (double *)malloc(n * sizeof(double));
    double **V = (double **)malloc((max_iter+1) * sizeof(double *));
    double *H = (double *)malloc((max_iter+1) * max_iter * sizeof(double));
    double *cs = (double *)malloc(max_iter * sizeof(double));
    double *sn = (double *)malloc(max_iter * sizeof(double));
    double *g = (double *)malloc((max_iter+1) * sizeof(double));
    
    for (int i = 0; i <= max_iter; i++) {
        V[i] = (double *)malloc(n * sizeof(double));
    }
    
    /* 计算初始残差 */
    matvec_tridiag(A, x, z);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - z[i];
    }
    
    double beta = vector_norm(r, n);
    res_norms[0] = beta;
    
    /* 如果初始残差已经很小，直接返回 */
    if (beta < tol) {
        *iter_used = 0;
        gettimeofday(&end, NULL);
        *time_used = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
        
        free(r);
        free(z);
        for (int i = 0; i <= max_iter; i++) free(V[i]);
        free(V);
        free(H);
        free(cs);
        free(sn);
        free(g);
        return;
    }
    
    /* 初始化Krylov子空间基 */
    for (int i = 0; i < n; i++) {
        V[0][i] = r[i] / beta;
    }
    
    g[0] = beta;
    
    /* 初始化ILU分解（如果需要） */
    ILUFactors ilu_factors;
    int use_preconditioner = 0;
    
    if (strcmp(precond_type, "ilu0") == 0) {
        ilu0_factorize(A, &ilu_factors);
        use_preconditioner = 1;
    } else if (strcmp(precond_type, "ilut") == 0) {
        ilut_factorize(A, &ilu_factors, tau, lfil);
        use_preconditioner = 1;
    }
    
    int k;
    for (k = 0; k < max_iter; k++) {
        /* Arnoldi迭代 */
        if (!use_preconditioner) {
            /* 无预条件：直接矩阵-向量乘 */
            matvec_tridiag(A, V[k], V[k+1]);
        } else {
            /* 有预条件：先应用预条件器，再矩阵-向量乘 */
            if (strcmp(precond_type, "ilu0") == 0) {
                apply_ilu0_preconditioner(&ilu_factors, V[k], z, n);
            } else {
                apply_ilut_preconditioner(&ilu_factors, V[k], z, n);
            }
            
            /* 矩阵-向量乘法 */
            matvec_tridiag(A, z, V[k+1]);
        }
        
        /* 标准化（Gram-Schmidt正交化） */
        for (int i = 0; i <= k; i++) {
            H[i * max_iter + k] = 0.0;
            for (int j = 0; j < n; j++) {
                H[i * max_iter + k] += V[k+1][j] * V[i][j];
            }
            for (int j = 0; j < n; j++) {
                V[k+1][j] -= H[i * max_iter + k] * V[i][j];
            }
        }
        
        H[(k+1) * max_iter + k] = vector_norm(V[k+1], n);
        double eps = 1e-12;
        if (H[(k+1) * max_iter + k] < eps * beta) {
            H[(k+1) * max_iter + k] = 0.0;
        } else {
            for (int j = 0; j < n; j++) {
                V[k+1][j] /= H[(k+1) * max_iter + k];
            }
        }
        
        /* 更新Hessenberg矩阵和g向量 */
        for (int i = 0; i < k; i++) {
            double temp = cs[i] * H[i * max_iter + k] + sn[i] * H[(i+1) * max_iter + k];
            H[(i+1) * max_iter + k] = -sn[i] * H[i * max_iter + k] + cs[i] * H[(i+1) * max_iter + k];
            H[i * max_iter + k] = temp;
        }
        
        double denom = sqrt(H[k * max_iter + k] * H[k * max_iter + k] + 
                           H[(k+1) * max_iter + k] * H[(k+1) * max_iter + k]);
        if (denom == 0.0) denom = DBL_MIN;
        
        cs[k] = H[k * max_iter + k] / denom;
        sn[k] = H[(k+1) * max_iter + k] / denom;
        
        /* 更新g向量 */
        g[k+1] = -sn[k] * g[k];
        g[k] = cs[k] * g[k];
        
        /* 计算当前残差范数 */
        H[k * max_iter + k] = cs[k] * H[k * max_iter + k] + sn[k] * H[(k+1) * max_iter + k];
        H[(k+1) * max_iter + k] = 0.0;
        
        double res_norm = fabs(g[k+1]);
        res_norms[k+1] = res_norm;
        
        /* 检查收敛 */
        if (res_norm < tol * beta || k == max_iter-1) {
            break;
        }
    }
    
    /* 回解最小二乘问题 */
    double *y = (double *)malloc((k+1) * sizeof(double));
    for (int i = k; i >= 0; i--) {
        y[i] = g[i];
        for (int j = i+1; j <= k; j++) {
            y[i] -= H[i * max_iter + j] * y[j];
        }
        y[i] /= H[i * max_iter + i];
    }
    
    /* 更新解向量 */
    for (int j = 0; j <= k; j++) {
        if (!use_preconditioner) {
            /* 无预条件：直接加到解向量 */
            for (int i = 0; i < n; i++) {
                x[i] += y[j] * V[j][i];
            }
        } else {
            /* 有预条件：先应用预条件器 */
            double *z = (double *)malloc(n * sizeof(double));
            if (strcmp(precond_type, "ilu0") == 0) {
                apply_ilu0_preconditioner(&ilu_factors, V[j], z, n);
            } else {
                apply_ilut_preconditioner(&ilu_factors, V[j], z, n);
            }
            
            for (int i = 0; i < n; i++) {
                x[i] += y[j] * z[i];
            }
            free(z);
        }
    }
    
    /* 记录迭代次数和时间 */
    *iter_used = k+1;
    
    /* 获取结束时间 */
    gettimeofday(&end, NULL);
    *time_used = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    
    /* 释放内存 */
    free(r);
    free(z);
    for (int i = 0; i <= max_iter; i++) {
        free(V[i]);
    }
    free(V);
    free(H);
    free(cs);
    free(sn);
    free(g);
    free(y);
    
    if (use_preconditioner) {
        free_ilu_factors(&ilu_factors);
    }
}

/* 打印收敛信息 */
void print_convergence(const char *method, int iter_used, double *res_norms, double time_used, double initial_res) {
    printf("收敛信息 (%s):\n", method);
    printf("迭代次数: %d\n", iter_used);
    printf("初始残差: %e\n", initial_res);
    printf("相对残差: %e\n", res_norms[iter_used] / initial_res);
    printf("最终残差: %e\n", res_norms[iter_used]);
    printf("计算时间: %.4f秒\n", time_used);
    
    /* 打印收敛历史（前10次和最后5次迭代） */
    printf("收敛历史（前10次和最后5次）:\n");
    for (int i = 0; i < 10 && i < iter_used; i++) {
        printf("  迭代 %2d: 残差 = %e, 相对残差 = %e\n", 
               i+1, res_norms[i+1], res_norms[i+1] / initial_res);
    }
    if (iter_used > 10) {
        printf("  ...\n");
        for (int i = iter_used-5; i < iter_used; i++) {
            printf("  迭代 %2d: 残差 = %e, 相对残差 = %e\n", 
                   i+1, res_norms[i+1], res_norms[i+1] / initial_res);
        }
    }
}

/* 打印对比结果 */
void print_comparison(int iter_none, double time_none, int iter_ilu0, double time_ilu0, 
                     int iter_ilut, double time_ilut, double cond_est) {
    printf("===== 算法性能对比 =====\n");
    printf("条件数: %.2e\n", cond_est);
    
    printf("\n%-15s %-10s %-10s %-10s\n", "方法", "迭代次数", "相对速度", "时间(秒)");
    printf("------------------------------------------------\n");
    
    /* 无预条件 */
    printf("%-15s %-10d %-10.2f %-10.4f\n", "无预条件", iter_none, 1.0, time_none);
    
    /* ILU0 */
    double speedup_ilu0 = (double)iter_none / iter_ilu0;
    printf("%-15s %-10d %-10.2f %-10.4f\n", "ILU0", iter_ilu0, speedup_ilu0, time_ilu0);
    
    /* ILUT */
    double speedup_ilut = (double)iter_none / iter_ilut;
    printf("%-15s %-10d %-10.2f %-10.4f\n", "ILUT", iter_ilut, speedup_ilut, time_ilut);
    
    printf("\n分析:\n");
    printf("- ILU0比无预条件快 %.2f 倍\n", speedup_ilu0);
    printf("- ILUT比无预条件快 %.2f 倍\n", speedup_ilut);
    
    if (speedup_ilut > speedup_ilu0) {
        printf("- ILUT比ILU0额外提高了 %.2f%% 的收敛速度\n", 
               (speedup_ilut - speedup_ilu0) / speedup_ilu0 * 100);
    } else {
        printf("- ILU0比ILUT收敛更快，ILUT可能因阈值参数设置不当而效果略差\n");
    }
}