#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_ORDER 14
#define MAX_STENCIL (MAX_ORDER / 2)
#define PI 3.14159265358979323846

// 高斯消元法求解线性方程组 Ax = B
// A: N x N 矩阵，B: N 向量，x: 解向量
void solve_linear_system(int N, double A[N][N], double B[N], double x[N]) {
    int i, j, k;
    double max, temp;
    int pivot_row;

    // 复制以避免修改原数组（可选，此处直接修改）
    double mat[N][N];
    double vec[N];
    for(i=0; i<N; i++) {
        vec[i] = B[i];
        for(j=0; j<N; j++) mat[i][j] = A[i][j];
    }

    // 消元
    for (i = 0; i < N; i++) {
        // 选主元
        max = fabs(mat[i][i]);
        pivot_row = i;
        for (k = i + 1; k < N; k++) {
            if (fabs(mat[k][i]) > max) {
                max = fabs(mat[k][i]);
                pivot_row = k;
            }
        }

        // 交换行
        if (pivot_row != i) {
            for (j = 0; j < N; j++) {
                temp = mat[i][j];
                mat[i][j] = mat[pivot_row][j];
                mat[pivot_row][j] = temp;
            }
            temp = vec[i];
            vec[i] = vec[pivot_row];
            vec[pivot_row] = temp;
        }

        // 消去
        for (k = i + 1; k < N; k++) {
            double factor = mat[k][i] / mat[i][i];
            for (j = i; j < N; j++) {
                mat[k][j] -= factor * mat[i][j];
            }
            vec[k] -= factor * vec[i];
        }
    }

    // 回代
    for (i = N - 1; i >= 0; i--) {
        x[i] = vec[i];
        for (j = i + 1; j < N; j++) {
            x[i] -= mat[i][j] * x[j];
        }
        x[i] /= mat[i][i];
    }
}

// 生成中心差分格式系数
// order: 精度阶数 (4, 8, 12, 14)
// coeffs: 输出系数数组 a_1, a_2, ...
// 返回使用的半模板宽度 N (即 stencil 半径)
int generate_coeffs(int order, double *coeffs) {
    int N = order / 2; // 4 阶->2 个点，8 阶->4 个点
    double A[MAX_STENCIL][MAX_STENCIL];
    double B[MAX_STENCIL];
    double x[MAX_STENCIL];
    int i, j;

    // 构建线性方程组
    // 条件：sum(a_m * m^(2k-1)) = 0.5 (k=1), 0 (k>1)
    // 矩阵行 i 对应幂次 2*i + 1
    for (i = 0; i < N; i++) {
        B[i] = (i == 0) ? 0.5 : 0.0;
        for (j = 0; j < N; j++) {
            int m = j + 1;
            int power = 2 * i + 1;
            A[i][j] = pow((double)m, power);
        }
    }

    solve_linear_system(N, A, B, x);

    for (i = 0; i < N; i++) {
        coeffs[i] = x[i];
    }
    return N;
}

// 计算修正波数 k* dx
// kh: 无量纲波数 k * dx
// coeffs: 差分系数
// N: 系数个数
double get_modified_wavenumber(double kh, double *coeffs, int N) {
    double k_star = 0.0;
    int m;
    // 公式：k* dx = 2 * sum( a_m * sin(m * kh) )
    for (m = 0; m < N; m++) {
        k_star += 2.0 * coeffs[m] * sin((m + 1) * kh);
    }
    return k_star;
}

int main() {
    int orders[] = {4, 8, 12, 14};
    int num_schemes = 4;
    double coeffs[MAX_STENCIL];
    FILE *fp = fopen("dispersion_data.csv", "w");
    
    if (!fp) {
        printf("Error: Cannot create output file.\n");
        return 1;
    }

    // 写入 CSV 表头
    fprintf(fp, "kh,Exact,");
    for (int i = 0; i < num_schemes; i++) {
        fprintf(fp, "Order_%d", orders[i]);
        if (i < num_schemes - 1) fprintf(fp, ",");
    }
    fprintf(fp, "\n");

    printf("=== Spectral Analysis of Explicit Central Schemes ===\n\n");
    printf("%-10s %-10s %-15s %-15s\n", "Order", "Stencil", "1% Error Limit", "Max Dispersion Err");
    printf("------------------------------------------------------------\n");

    for (int i = 0; i < num_schemes; i++) {
        int order = orders[i];
        int N = generate_coeffs(order, coeffs);
        
        // 打印系数供验证
        // printf("Order %d Coeffs: ", order);
        // for(int k=0; k<N; k++) printf("%.6f ", coeffs[k]);
        // printf("\n");

        double max_err = 0.0;
        double kh_1percent = 0.0;
        int found_1percent = 0;

        // 遍历波数 kh 从 0 到 PI
        for (double kh = 0.01; kh <= PI; kh += 0.01) {
            double k_exact = kh;
            double k_num = get_modified_wavenumber(kh, coeffs, N);
            
            // 计算相对误差 (色散误差)
            // 注意：在 kh 接近 0 时误差很小，主要看高波数
            double err = fabs(k_num - k_exact) / k_exact; 
            // 对于高波数，使用绝对误差更合理，因为相速度误差正比于 (k* - k)
            double abs_err = fabs(k_num - k_exact); 

            if (abs_err > max_err) max_err = abs_err;
            
            // 记录误差超过 0.01 (1% 波长分辨率) 的临界点
            // 这里定义：当修正波数偏离真实波数超过 1% 时，认为该波数不可解
            if (!found_1percent && abs_err > 0.01 * kh) { 
                 // 更严格的标准：相速度误差 > 1%
                 // c_num / c_exact = k* / k. 所以 |k* - k|/k > 0.01
                 kh_1percent = kh - 0.01; 
                 found_1percent = 1;
            }

            // 写入数据 (每隔 0.05 写一次，避免文件过大)
            if (fmod(kh, 0.05) < 0.001 || kh > PI - 0.01) {
                fprintf(fp, "%.4f,%.4f,", kh, k_exact);
                // 为了对齐，我们需要在循环外写，这里简化处理：
                // 实际做法应该是先存数组再写列，这里为了代码简单，
                // 我们只写当前行的数据，但 CSV 结构需要调整。
                // 修正：为了简化 C 代码，我们只输出统计表，绘图数据用简化逻辑。
                // 这里为了演示，我们只输出统计结果到控制台，CSV 仅用于 Python 绘图。
            }
        }
        
        // 重新循环写 CSV (为了代码结构清晰，实际应用中应分离逻辑)
        // 这里为了单文件演示，我们只做控制台统计，CSV 生成放在下面单独循环
    }
    
    // 重新生成 CSV 数据以便绘图
    rewind(fp); // 不行，需要关闭重开，这里简化：直接追加或重新逻辑
    fclose(fp);
    fp = fopen("dispersion_data.csv", "w");
    fprintf(fp, "kh,Exact,Order_4,Order_8,Order_12,Order_14\n");

    for (double kh = 0.0; kh <= PI; kh += 0.02) {
        fprintf(fp, "%.4f,%.4f", kh, kh);
        for (int i = 0; i < num_schemes; i++) {
            int order = orders[i];
            int N = generate_coeffs(order, coeffs);
            double k_num = get_modified_wavenumber(kh, coeffs, N);
            fprintf(fp, ",%.4f", k_num);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    // 再次运行统计输出
    for (int i = 0; i < num_schemes; i++) {
        int order = orders[i];
        int N = generate_coeffs(order, coeffs);
        double kh_1percent = 0.0;
        
        for (double kh = 0.01; kh <= PI; kh += 0.001) {
            double k_num = get_modified_wavenumber(kh, coeffs, N);
            if (kh > 0.1 && fabs(k_num - kh) / kh > 0.01) {
                kh_1percent = kh;
                break;
            }
        }
        if (kh_1percent == 0) kh_1percent = PI;

        printf("%-10d %-10d %-15.4f (rad) %-15.4f\n", 
               order, 2*N+1, kh_1percent, 
               fabs(get_modified_wavenumber(PI, coeffs, N) - PI));
    }

    printf("\nData saved to 'dispersion_data.csv'.\n");
    printf("Use the provided Python script to plot the results.\n");
    printf("Note: Dissipation is 0 for all central schemes (Imaginary part = 0).\n");

    return 0;
}
