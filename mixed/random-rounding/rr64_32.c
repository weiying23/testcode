#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// ==================== 随机舍入系统 ====================
typedef struct {
    unsigned int seed;
    double resolution;    // 舍入分辨率，模拟32位精度
} RandomRoundSystem;

RandomRoundSystem init_random_round(double resolution) {
    RandomRoundSystem sys;
    sys.seed = (unsigned int)time(NULL);
    sys.resolution = resolution > 0 ? resolution : 1e-6;  // 32位浮点数典型分辨率
    return sys;
}

static unsigned int simple_rand(unsigned int *state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

// 线性插值随机舍入函数（64位到32位模拟）
float random_round_value(RandomRoundSystem *sys, double x) {
    if (sys->resolution <= 0.0) {
        return x;
    }
    
    if (fabs(x) < 1e-15) return 0.0;
    
    double x1 = floor(x / sys->resolution) * sys->resolution;
    double x2 = x1 + sys->resolution;
    
    if (fabs(x - x1) < 1e-12) return (float)x1;
    if (fabs(x2 - x) < 1e-12) return (float)x2;
    
    double p = (x - x1) / (x2 - x1);
    unsigned int r = simple_rand(&sys->seed);
    double rand_val = (double)(r % 10000) / 10000.0;
    
    return (rand_val < p) ? (float)x2 : (float)x1;
}

// ==================== PDE求解器 ====================
// 1. 64位直接计算（高精度参考解）
void solve_pde_64bit_direct(double* u, int n, double alpha, double dx, double dt, int steps) {
    double* temp = (double*)malloc(n * sizeof(double));
    double factor = alpha * dt / (dx * dx);
    
    for (int step = 0; step < steps; step++) {
        temp[0] = 0.0;
        temp[n-1] = 0.0;
        
        for (int i = 1; i < n-1; i++) {
            temp[i] = u[i] + factor * (u[i+1] - 2*u[i] + u[i-1]);
        }
        
        double* swap = u;
        u = temp;
        temp = swap;
    }
    
    free(temp);
}

// 2. 64位降精度到32位（随机舍入）
void solve_pde_64bit_to_32bit(RandomRoundSystem *sys, float* u, int n, 
                             double alpha, double dx, double dt, int steps) {
    float* temp = (float*)malloc(n * sizeof(float));
    
    float factor = alpha * dt / (dx * dx);
    
    for (int step = 0; step < steps; step++) {
        temp[0] = 0.0;
        temp[n-1] = 0.0;
        
        for (int i = 1; i < n-1; i++) {
            /*
            double diff1 = u[i+1] - u[i];
            diff1 = random_round_value(sys, diff1);
            
            double diff2 = u[i] - u[i-1];
            diff2 = random_round_value(sys, diff2);
            
            double laplacian = diff1 - diff2;
            laplacian = random_round_value(sys, laplacian);
            
            double update = factor * laplacian;
            update = random_round_value(sys, update);
            
            temp[i] = u[i] + update;
            temp[i] = random_round_value(sys, temp[i]);
            */

            temp[i] = random_round_value(sys, u[i]) \
            + factor * (random_round_value(sys, u[i+1]) \
            - 2*random_round_value(sys, u[i]) + random_round_value(sys, u[i-1]));
        }
        
        //double* swap = u;
        //u = temp;
        //temp = swap;
        float* swap = u;
        u = temp;
        temp = swap;
    }
    
    free(temp);
}

// 3. 直接使用32位进行计算（使用原生float类型）
void solve_pde_32bit_direct(float* u, int n, double alpha, double dx, double dt, int steps) {
    // 分配32位浮点数数组
    float* u32 = (float*)malloc(n * sizeof(float));
    float* temp32 = (float*)malloc(n * sizeof(float));
    
    // 转换初始条件到32位（使用原生转换）
    for (int i = 0; i < n; i++) {
        u32[i] = (float)u[i];  // 原生64位到32位转换
    }
    
    // 转换参数到32位
    float alpha32 = (float)alpha;
    float dx32 = (float)dx;
    float dt32 = (float)dt;
    
    // 计算系数（使用32位运算）
    float factor = alpha32 * dt32 / (dx32 * dx32);
    
    for (int step = 0; step < steps; step++) {
        // 边界条件
        temp32[0] = 0.0f;
        temp32[n-1] = 0.0f;
        
        // 更新内部点（全部使用32位运算）
        for (int i = 1; i < n-1; i++) {
            float diff1 = u32[i+1] - u32[i];      // 32位减法
            float diff2 = u32[i] - u32[i-1];      // 32位减法
            float laplacian = diff1 - diff2;      // 32位减法
            float update = factor * laplacian;    // 32位乘法
            temp32[i] = u32[i] + update;          // 32位加法
        }
        
        // 交换数组
        float* swap = u32;
        u32 = temp32;
        temp32 = swap;
    }
    
    //转换回64位用于比较
    for (int i = 0; i < n; i++) {
        u[i] = (double)u32[i];  // 原生32位到64位转换
    }
    
    free(u32);
    free(temp32);
}

// ==================== 辅助函数 ====================
void init_temperature(double* u, int n, double L) {
    for (int i = 0; i < n; i++) {
        double x = (double)i * L / (n - 1);
        u[i] = 0.5 * sin(3.141592653589793 * x / L) + 0.5;
    }
}

void copy_array(double* dest, double* src, int n) {
    memcpy(dest, src, n * sizeof(double));
}

void copy_array32(float* dest, double* src, int n) {
    for (int i = 0; i < n; i++) {
        dest[i] = (float)src[i];  // 类型转换
    }
}

void calculate_errors(double* ref, double* test, int n, 
                     double* max_abs_error, double* avg_abs_error,
                     double* max_rel_error, double* avg_rel_error) {
    *max_abs_error = 0.0;
    *avg_abs_error = 0.0;
    *max_rel_error = 0.0;
    *avg_rel_error = 0.0;
    
    int valid_rel_points = 0;
    
    for (int i = 0; i < n; i++) {
        double abs_err = fabs(ref[i] - test[i]);
        *avg_abs_error += abs_err;
        
        if (abs_err > *max_abs_error) {
            *max_abs_error = abs_err;
        }
        
        if (fabs(ref[i]) > 1e-15) {
            double rel_err = abs_err / fabs(ref[i]);
            *avg_rel_error += rel_err;
            valid_rel_points++;
            
            if (rel_err > *max_rel_error) {
                *max_rel_error = rel_err;
            }
        }
    }
    
    *avg_abs_error /= n;
    if (valid_rel_points > 0) {
        *avg_rel_error /= valid_rel_points;
    }
}

// ==================== 主函数 ====================
int main() {
    printf("三种精度模式PDE求解器对比（64位->32位）\n");
    printf("1. 64位直接计算\n");
    printf("2. 64位降精度到32位（随机舍入）\n");
    printf("3. 直接使用32位计算\n");
    printf("========================================\n\n");
    
    const int N = 51;
    const double L = 1.0;
    const double dx = L / (N-1);
    const double dt = 0.0005;
    const double alpha = 0.01;
    const int steps = 200;
    
    printf("PDE参数:\n");
    printf("  网格点数: %d\n", N);
    printf("  空间长度: %.2f\n", L);
    printf("  空间步长: %.8f\n", dx);
    printf("  时间步长: %.8f\n", dt);
    printf("  扩散系数: %.6f\n", alpha);
    printf("  时间步数: %d\n\n", steps);
    
    double* u_ref = (double*)malloc(N * sizeof(double));
    float* u_rand = (float*)malloc(N * sizeof(float));
    float* u_direct32 = (float*)malloc(N * sizeof(float));
    double* u_initial = (double*)malloc(N * sizeof(double));
    
    init_temperature(u_initial, N, L);
    
    // 测试1: 64位直接计算
    printf("测试1: 64位直接计算\n");
    printf("----------------------------------------\n");
    
    copy_array(u_ref, u_initial, N);
    solve_pde_64bit_direct(u_ref, N, alpha, dx, dt, steps);
    
    // 测试2: 64位降精度到32位（随机舍入）
    printf("\n测试2: 64位降精度到32位（随机舍入）\n");
    printf("----------------------------------------\n");
    
    copy_array32(u_rand, u_initial, N);
    RandomRoundSystem sys = init_random_round(1e-6);  // 32位典型分辨率
    sys.seed = 12345;
    
    solve_pde_64bit_to_32bit(&sys, u_rand, N, alpha, dx, dt, steps);
    
    double max_abs, avg_abs, max_rel, avg_rel;
    calculate_errors(u_ref, u_rand, N, &max_abs, &avg_abs, &max_rel, &avg_rel);
    
    printf("与64位直接计算的误差:\n");
    printf("  最大绝对误差: %.10f\n", max_abs);
    printf("  平均绝对误差: %.10f\n", avg_abs);
    printf("  最大相对误差: %.6f%%\n", max_rel * 100.0);
    printf("  平均相对误差: %.6f%%\n", avg_rel * 100.0);
    
    // 测试3: 直接使用32位计算
    printf("\n测试3: 直接使用32位计算\n");
    printf("----------------------------------------\n");
    
    copy_array32(u_direct32, u_initial, N);
    solve_pde_32bit_direct(u_direct32, N, alpha, dx, dt, steps);
    
    calculate_errors(u_ref, u_direct32, N, &max_abs, &avg_abs, &max_rel, &avg_rel);
    
    printf("与64位直接计算的误差:\n");
    printf("  最大绝对误差: %.10f\n", max_abs);
    printf("  平均绝对误差: %.10f\n", avg_abs);
    printf("  最大相对误差: %.6f%%\n", max_rel * 100.0);
    printf("  平均相对误差: %.6f%%\n", avg_rel * 100.0);
    
    //calculate_errors(u_rand, u_direct32, N, &max_abs, &avg_abs, &max_rel, &avg_rel);
    //printf("\n32位直接与随机舍入的差异:\n");
    //printf("  最大绝对差异: %.10f\n", max_abs);
    //printf("  平均绝对差异: %.10f\n", avg_abs);
    
    printf("\n\n三种方法结果对比（每隔5个点）:\n");
    printf("位置\t64位直接\t64->32随机舍入\t32位直接\t随机舍入误差\t32位直接误差\n");
    printf("-------------------------------------------------------------------------------\n");
    
    for (int i = 0; i < N; i += 5) {
        double x = i * dx;
        double err_rand = fabs(u_ref[i] - u_rand[i]);
        double err_direct32 = fabs(u_ref[i] - u_direct32[i]);
        
        printf("%.3f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n", 
               x, u_ref[i], u_rand[i], u_direct32[i], err_rand, err_direct32);
    }
    
    printf("\n\n精度对比统计汇总:\n");
    printf("========================================\n");
    
    double err_rand_max = 0.0, err_rand_avg = 0.0;
    double err_direct_max = 0.0, err_direct_avg = 0.0;
    double diff_max = 0.0, diff_avg = 0.0;
    
    for (int i = 0; i < N; i++) {
        double e1 = fabs(u_ref[i] - u_rand[i]);
        double e2 = fabs(u_ref[i] - u_direct32[i]);
        double d = fabs(u_rand[i] - u_direct32[i]);
        
        err_rand_avg += e1;
        err_direct_avg += e2;
        diff_avg += d;
        
        if (e1 > err_rand_max) err_rand_max = e1;
        if (e2 > err_direct_max) err_direct_max = e2;
        if (d > diff_max) diff_max = d;
    }
    
    err_rand_avg /= N;
    err_direct_avg /= N;
    diff_avg /= N;
    
    printf("方法                   最大绝对误差   平均绝对误差\n");
    printf("--------------------------------------------------\n");
    printf("64位降精度到32位(随机) %.10f   %.10f\n", err_rand_max, err_rand_avg);
    printf("直接32位计算           %.10f   %.10f\n", err_direct_max, err_direct_avg);
    //printf("两种低精度方法差异     %.10f   %.10f\n", diff_max, diff_avg);
    
    
    free(u_ref);
    free(u_rand);
    free(u_direct32);
    free(u_initial);
    
    printf("\n测试完成！\n");
    
    return 0;
}