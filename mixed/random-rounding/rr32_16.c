#include <stdio.h>      
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <string.h>

// ==================== 随机舍入系统 ====================
// 定义随机舍入系统的结构体
typedef struct {
    unsigned int seed;    // 随机数种子
    float resolution;     // 舍入分辨率，即舍入的最小间隔
} RandomRoundSystem;

// 初始化随机舍入系统
RandomRoundSystem init_random_round(float resolution) {
    RandomRoundSystem sys;        // 声明随机舍入系统变量
    sys.seed = (unsigned int)time(NULL);  // 使用当前时间作为随机种子
    sys.resolution = resolution > 0 ? resolution : 0.001f;  // 确保分辨率为正数
    return sys;                   // 返回初始化后的系统
}

// 简单随机数生成器（线性同余法）
static unsigned int simple_rand(unsigned int *state) {
    *state = *state * 1103515245 + 12345;  // 线性同余公式生成新随机数
    return *state;               // 返回生成的随机数
}

// 线性插值随机舍入函数
float random_round_value(RandomRoundSystem *sys, float x) {
    if (sys->resolution <= 0.0f) {  // 如果分辨率非正，直接返回原值
        return x;
    }
    
    // 处理特殊值：接近0的值直接返回0
    if (fabsf(x) < 1e-12f) return 0.0f;
    
    // 计算相邻的舍入点：x1为向下取整的舍入点，x2为向上取整的舍入点
    float x1 = floorf(x / sys->resolution) * sys->resolution;
    float x2 = x1 + sys->resolution;
    
    // 如果x接近边界，直接返回边界值（避免浮点精度问题）
    if (fabsf(x - x1) < 0.00001f) return x1;
    if (fabsf(x2 - x) < 0.00001f) return x2;
    
    // 计算概率: P(x→x2) = (x-x1)/(x2-x1)
    float p = (x - x1) / (x2 - x1);
    
    // 生成0-1之间的随机数
    unsigned int r = simple_rand(&sys->seed);
    float rand_val = (float)(r % 10000) / 10000.0f;
    
    // 根据概率决定舍入方向：随机数小于p则向上舍入，否则向下舍入
    return (rand_val < p) ? x2 : x1;
}

// ==================== 16位模拟系统 ====================
// 定义16位半精度浮点数结构体（简化模拟）
typedef struct {
    unsigned short value;  // 16位表示，使用无符号短整型存储
} Float16;

// 32位浮点数转16位（确定性舍入，模拟四舍五入）
Float16 float32_to_float16(float f32) {
    Float16 result;                        // 定义16位浮点数结果
    // 简化模拟：将32位浮点数乘以1024（2^10）后四舍五入到整数
    float scaled = f32 * 1024.0f;          // 乘以1024相当于保留10位小数精度
    result.value = (unsigned short)(scaled + 0.5f);  // 四舍五入到最近的整数
    return result;                         // 返回16位表示
}

// 16位浮点数转32位
float float16_to_float32(Float16 f16) {
    // 将16位整数除以1024.0，恢复为32位浮点数
    return (float)f16.value / 1024.0f;
}

// 16位加法模拟
Float16 float16_add(Float16 a, Float16 b) {
    float f32_a = float16_to_float32(a);      // 将a转换为32位
    float f32_b = float16_to_float32(b);      // 将b转换为32位
    float f32_result = f32_a + f32_b;         // 32位加法
    return float32_to_float16(f32_result);    // 结果转回16位
}

// 16位减法模拟
Float16 float16_sub(Float16 a, Float16 b) {
    float f32_a = float16_to_float32(a);      // 将a转换为32位
    float f32_b = float16_to_float32(b);      // 将b转换为32位
    float f32_result = f32_a - f32_b;         // 32位减法
    return float32_to_float16(f32_result);    // 结果转回16位
}

// 16位乘法模拟
Float16 float16_mul(Float16 a, Float16 b) {
    float f32_a = float16_to_float32(a);      // 将a转换为32位
    float f32_b = float16_to_float32(b);      // 将b转换为32位
    float f32_result = f32_a * f32_b;         // 32位乘法
    return float32_to_float16(f32_result);    // 结果转回16位
}

// ==================== PDE求解器 ====================
// 1. 32位直接计算（高精度参考解）
void solve_pde_32bit_direct(float* u, int n, float alpha, float dx, float dt, int steps) {
    float* temp = (float*)malloc(n * sizeof(float));  // 分配临时数组内存
    float factor = alpha * dt / (dx * dx);            // 计算显式格式的系数
    
    for (int step = 0; step < steps; step++) {        // 时间步循环
        // 边界条件：固定为0（Dirichlet边界）
        temp[0] = 0.0f;
        temp[n-1] = 0.0f;
        
        // 更新内部点：使用显式有限差分格式
        for (int i = 1; i < n-1; i++) {
            temp[i] = u[i] + factor * (u[i+1] - 2*u[i] + u[i-1]);
        }
        
        // 交换数组：当前解成为下一时间步的旧解
        float* swap = u;
        u = temp;
        temp = swap;
    }
    
    free(temp);  // 释放临时数组内存
}

// 2. 32位降精度到16位（随机舍入）
void solve_pde_32bit_to_16bit(RandomRoundSystem *sys, float* u, int n, 
                             float alpha, float dx, float dt, int steps) {
    float* temp = (float*)malloc(n * sizeof(float));  // 分配临时数组内存
    
    // 计算系数并进行随机舍入（模拟16位精度）
    float factor = alpha * dt / (dx * dx);
    factor = random_round_value(sys, factor);
    
    for (int step = 0; step < steps; step++) {        // 时间步循环
        // 边界条件：固定为0
        temp[0] = 0.0f;
        temp[n-1] = 0.0f;
        
        // 更新内部点（每个计算步骤都进行随机舍入）
        for (int i = 1; i < n-1; i++) {
            // 计算拉普拉斯项：使用中心差分，每个差分进行舍入
            float diff1 = u[i+1] - u[i];
            diff1 = random_round_value(sys, diff1);
            
            float diff2 = u[i] - u[i-1];
            diff2 = random_round_value(sys, diff2);
            
            float laplacian = diff1 - diff2;
            laplacian = random_round_value(sys, laplacian);
            
            float update = factor * laplacian;
            update = random_round_value(sys, update);
            
            temp[i] = u[i] + update;
            temp[i] = random_round_value(sys, temp[i]);  // 对结果进行舍入
        }
        
        // 交换数组
        float* swap = u;
        u = temp;
        temp = swap;
    }
    
    free(temp);  // 释放临时数组内存
}

// 3. 直接使用16位进行计算（模拟16位硬件）
void solve_pde_16bit_direct(float* u, int n, float alpha, float dx, float dt, int steps) {
    // 分配16位数组
    Float16* u16 = (Float16*)malloc(n * sizeof(Float16));
    Float16* temp16 = (Float16*)malloc(n * sizeof(Float16));
    
    // 转换初始条件到16位
    for (int i = 0; i < n; i++) {
        u16[i] = float32_to_float16(u[i]);
    }
    
    // 转换参数到16位
    Float16 alpha16 = float32_to_float16(alpha);
    Float16 dx16 = float32_to_float16(dx);
    Float16 dt16 = float32_to_float16(dt);
    
    // 计算系数：alpha * dt / (dx * dx)
    Float16 dx_sq = float16_mul(dx16, dx16);           // dx的平方
    Float16 factor_num = float16_mul(alpha16, dt16);   // 分子 alpha * dt
    Float16 factor = float32_to_float16(0.0f);         // 初始化系数
    
    // 避免除零：如果dx_sq不为0，则计算系数
    if (float16_to_float32(dx_sq) > 1e-12f) {
        // 模拟除法：factor = factor_num / dx_sq
        // 由于没有实现16位除法，通过转换到32位计算
        float f32_num = float16_to_float32(factor_num);
        float f32_den = float16_to_float32(dx_sq);
        float f32_factor = f32_num / f32_den;
        factor = float32_to_float16(f32_factor);
    }
    
    for (int step = 0; step < steps; step++) {        // 时间步循环
        // 边界条件：固定为0
        temp16[0] = float32_to_float16(0.0f);
        temp16[n-1] = float32_to_float16(0.0f);
        
        // 更新内部点（全部使用16位运算）
        for (int i = 1; i < n-1; i++) {
            Float16 diff1 = float16_sub(u16[i+1], u16[i]);      // u[i+1] - u[i]
            Float16 diff2 = float16_sub(u16[i], u16[i-1]);      // u[i] - u[i-1]
            Float16 laplacian = float16_sub(diff1, diff2);      // 拉普拉斯项
            Float16 update = float16_mul(factor, laplacian);    // 更新量
            temp16[i] = float16_add(u16[i], update);            // 新值
        }
        
        // 交换数组
        Float16* swap = u16;
        u16 = temp16;
        temp16 = swap;
    }
    
    // 转换回32位用于后续比较
    for (int i = 0; i < n; i++) {
        u[i] = float16_to_float32(u16[i]);
    }
    
    // 释放16位数组内存
    free(u16);
    free(temp16);
}

// ==================== 辅助函数 ====================
// 初始化温度分布：使用正弦函数，避免边界为0
void init_temperature(float* u, int n, float L) {
    for (int i = 0; i < n; i++) {
        float x = (float)i * L / (n - 1);              // 计算位置坐标
        // 使用正弦分布，值域在[0,1]之间
        u[i] = 0.5f * sinf(3.14159265f * x / L) + 0.5f;
    }
}

// 复制数组：将源数组复制到目标数组
void copy_array(float* dest, float* src, int n) {
    memcpy(dest, src, n * sizeof(float));             // 使用memcpy高效复制
}

// 计算误差统计：比较参考解和测试解的误差
void calculate_errors(float* ref, float* test, int n, 
                     float* max_abs_error, float* avg_abs_error,
                     float* max_rel_error, float* avg_rel_error) {
    *max_abs_error = 0.0f;     // 初始化最大绝对误差
    *avg_abs_error = 0.0f;     // 初始化平均绝对误差
    *max_rel_error = 0.0f;     // 初始化最大相对误差
    *avg_rel_error = 0.0f;     // 初始化平均相对误差
    
    int valid_rel_points = 0;  // 有效相对误差点数（避免除以0）
    
    for (int i = 0; i < n; i++) {
        float abs_err = fabsf(ref[i] - test[i]);      // 计算绝对误差
        *avg_abs_error += abs_err;                    // 累加绝对误差
        
        if (abs_err > *max_abs_error) {               // 更新最大绝对误差
            *max_abs_error = abs_err;
        }
        
        // 计算相对误差（避免除以0）
        if (fabsf(ref[i]) > 1e-12f) {
            float rel_err = abs_err / fabsf(ref[i]);  // 相对误差 = 绝对误差 / |参考值|
            *avg_rel_error += rel_err;                // 累加相对误差
            valid_rel_points++;                       // 增加有效点数
            
            if (rel_err > *max_rel_error) {           // 更新最大相对误差
                *max_rel_error = rel_err;
            }
        }
    }
    
    *avg_abs_error /= n;                              // 计算平均绝对误差
    if (valid_rel_points > 0) {
        *avg_rel_error /= valid_rel_points;           // 计算平均相对误差
    }
}

// ==================== 主函数 ====================
int main() {
    // 打印程序标题和说明
    printf("三种精度模式PDE求解器对比\n");
    printf("1. 32位直接计算\n");
    printf("2. 32位降精度到16位（随机舍入）\n");
    printf("3. 直接使用16位计算\n");
    printf("========================================\n\n");
    
    // PDE参数设置
    const int N = 31;           // 网格点数（较小以便观察）
    const float L = 1.0f;       // 空间长度
    const float dx = L / (N-1); // 空间步长
    const float dt = 0.0005f;   // 时间步长（满足稳定性条件）
    const float alpha = 0.01f;  // 热扩散系数
    const int steps = 200;      // 时间步数
    
    // 打印参数信息
    printf("PDE参数:\n");
    printf("  网格点数: %d\n", N);
    printf("  空间长度: %.2f\n", L);
    printf("  空间步长: %.6f\n", dx);
    printf("  时间步长: %.6f\n", dt);
    printf("  扩散系数: %.4f\n", alpha);
    printf("  时间步数: %d\n\n", steps);
    
    // 分配内存：为四种情况分配数组
    float* u_ref = (float*)malloc(N * sizeof(float));      // 参考解（32位直接）
    float* u_rand = (float*)malloc(N * sizeof(float));     // 随机舍入解
    float* u_direct16 = (float*)malloc(N * sizeof(float)); // 16位直接解
    float* u_initial = (float*)malloc(N * sizeof(float));  // 初始条件
    
    // 初始化温度分布
    init_temperature(u_initial, N, L);
    
    // ========== 测试1: 32位直接计算 ==========
    printf("测试1: 32位直接计算\n");
    printf("----------------------------------------\n");
    
    copy_array(u_ref, u_initial, N);                    // 复制初始条件到参考解
    solve_pde_32bit_direct(u_ref, N, alpha, dx, dt, steps);  // 执行32位直接计算
    
    // ========== 测试2: 32位降精度到16位（随机舍入） ==========
    printf("\n测试2: 32位降精度到16位（随机舍入）\n");
    printf("----------------------------------------\n");
    
    copy_array(u_rand, u_initial, N);                   // 复制初始条件到随机舍入解
    RandomRoundSystem sys = init_random_round(0.001f);  // 初始化随机舍入系统，分辨率0.001
    sys.seed = 12345;  // 固定随机种子确保结果可重复
    
    solve_pde_32bit_to_16bit(&sys, u_rand, N, alpha, dx, dt, steps);  // 执行随机舍入计算
    
    // 计算与32位直接计算的误差
    float max_abs, avg_abs, max_rel, avg_rel;
    calculate_errors(u_ref, u_rand, N, &max_abs, &avg_abs, &max_rel, &avg_rel);
    
    printf("与32位直接计算的误差:\n");
    printf("  最大绝对误差: %.6f\n", max_abs);
    printf("  平均绝对误差: %.6f\n", avg_abs);
    printf("  最大相对误差: %.2f%%\n", max_rel * 100.0f);
    printf("  平均相对误差: %.2f%%\n", avg_rel * 100.0f);
    
    // ========== 测试3: 直接使用16位计算 ==========
    printf("\n测试3: 直接使用16位计算\n");
    printf("----------------------------------------\n");
    
    copy_array(u_direct16, u_initial, N);               // 复制初始条件到16位直接解
    solve_pde_16bit_direct(u_direct16, N, alpha, dx, dt, steps);  // 执行16位直接计算
    
    // 计算与32位直接计算的误差
    calculate_errors(u_ref, u_direct16, N, &max_abs, &avg_abs, &max_rel, &avg_rel);
    
    printf("与32位直接计算的误差:\n");
    printf("  最大绝对误差: %.6f\n", max_abs);
    printf("  平均绝对误差: %.6f\n", avg_abs);
    printf("  最大相对误差: %.2f%%\n", max_rel * 100.0f);
    printf("  平均相对误差: %.2f%%\n", avg_rel * 100.0f);
    
    // 计算16位直接与随机舍入的误差
    calculate_errors(u_rand, u_direct16, N, &max_abs, &avg_abs, &max_rel, &avg_rel);
    printf("\n16位直接与随机舍入的差异:\n");
    printf("  最大绝对差异: %.6f\n", max_abs);
    printf("  平均绝对差异: %.6f\n", avg_abs);
    
    // ========== 结果对比表格 ==========
    printf("\n\n三种方法结果对比（每隔3个点）:\n");
    printf("位置\t32位直接\t32->16随机舍入\t16位直接\t随机舍入误差\t16位直接误差\n");
    printf("-------------------------------------------------------------------------------\n");
    
    for (int i = 0; i < N; i += 3) {                    // 每隔3个点打印结果
        float x = i * dx;                               // 计算位置
        float err_rand = fabsf(u_ref[i] - u_rand[i]);   // 随机舍入误差
        float err_direct16 = fabsf(u_ref[i] - u_direct16[i]);  // 16位直接误差
        
        printf("%.3f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n", 
               x, u_ref[i], u_rand[i], u_direct16[i], err_rand, err_direct16);
    }
    
    // ========== 统计汇总 ==========
    printf("\n\n精度对比统计汇总:\n");
    printf("========================================\n");
    
    // 重新计算所有误差进行汇总
    float err_rand_max = 0.0f, err_rand_avg = 0.0f;
    float err_direct_max = 0.0f, err_direct_avg = 0.0f;
    float diff_max = 0.0f, diff_avg = 0.0f;
    
    for (int i = 0; i < N; i++) {
        float e1 = fabsf(u_ref[i] - u_rand[i]);         // 随机舍入误差
        float e2 = fabsf(u_ref[i] - u_direct16[i]);     // 16位直接误差
        float d = fabsf(u_rand[i] - u_direct16[i]);     // 两种低精度方法差异
        
        err_rand_avg += e1;                             // 累加随机舍入误差
        err_direct_avg += e2;                           // 累加16位直接误差
        diff_avg += d;                                  // 累加方法差异
        
        if (e1 > err_rand_max) err_rand_max = e1;       // 更新最大随机舍入误差
        if (e2 > err_direct_max) err_direct_max = e2;   // 更新最大16位直接误差
        if (d > diff_max) diff_max = d;                 // 更新最大方法差异
    }
    
    err_rand_avg /= N;                                  // 计算平均随机舍入误差
    err_direct_avg /= N;                                // 计算平均16位直接误差
    diff_avg /= N;                                      // 计算平均方法差异
    
    // 打印统计汇总表
    printf("方法                   最大绝对误差   平均绝对误差\n");
    printf("--------------------------------------------------\n");
    printf("32位降精度到16位(随机)  %.6f      %.6f\n", err_rand_max, err_rand_avg);
    printf("直接16位计算            %.6f      %.6f\n", err_direct_max, err_direct_avg);
    printf("两种低精度方法差异      %.6f      %.6f\n", diff_max, diff_avg);
    

    
    // 清理内存：释放所有动态分配的内存
    free(u_ref);
    free(u_rand);
    free(u_direct16);
    free(u_initial);
    
    printf("\n测试完成！\n");
    
    return 0;  // 程序正常结束
}