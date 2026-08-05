#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>

// ==================== IEEE 754 binary16 舍入函数 ====================
// 基于 rr64_32_new.c 的位操作实现，precision_bits=11 (binary16)
// binary16: 1 符号 + 5 指数 + 10 尾数(存储) + 1 隐含 = 11 位有效精度
// 相对 binary32 (24 位) 需丢弃 k = 24 - 11 = 13 位

#define TOTAL_PREC 24       // binary32 总精度 (23 存储 + 1 隐含)
#define BINARY16_PREC 11    // binary16 有效精度

// 线性同余随机数生成器（可复现）
static unsigned int g_rng_state = 12345;

static unsigned int sr_rand(void) {
    g_rng_state = g_rng_state * 1103515245u + 12345u;
    return g_rng_state;
}

// 随机舍入到 binary16 (Mode 1 Stochastic Rounding)
// 原理：尾数 + k 位随机数后截断低 k 位
//   P[向上舍入] = m_low / 2^k = 距下界的相对距离
//   E[SR(x)] = x （无偏性）
float stochastic_round_16(float x) {
    if (!isfinite(x) || x == 0.0f) return x;

    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));

    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;

    int is_subnormal = (exp == 0);
    if (!is_subnormal) mantissa |= 0x800000;  // 添加隐含位

    int k = TOTAL_PREC - BINARY16_PREC;   // 13
    uint32_t mask = (1U << k) - 1;         // 0x1FFF
    uint32_t random_bits = sr_rand() & mask;

    uint64_t m64 = (uint64_t)mantissa + random_bits;

    // 处理进位 (尾数溢出 24 位 -> 指数 +1)
    if (m64 >= (1ULL << TOTAL_PREC)) {
        m64 >>= 1;
        exp += 1;
        if (exp >= 0xFF) {
            uint32_t inf_bits = (sign << 31) | (0xFF << 23);
            float inf;
            memcpy(&inf, &inf_bits, sizeof(float));
            return inf;
        }
    }

    // 截断低 k 位
    uint32_t new_mantissa = (uint32_t)m64 & (~mask);
    if (!is_subnormal && exp > 0) {
        new_mantissa &= 0x7FFFFF;  // 清除隐含位用于存储
    }

    uint32_t new_bits = (sign << 31) | (exp << 23) | new_mantissa;
    float result;
    memcpy(&result, &new_bits, sizeof(float));
    return result;
}

// 确定性舍入到 binary16 (Round-to-Nearest)
// 原理：加 0.5 ulp 偏置后截断 (Round Half Up)
float round_to_16_rn(float x) {
    if (!isfinite(x) || x == 0.0f) return x;

    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));

    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;

    int is_subnormal = (exp == 0);
    if (!is_subnormal) mantissa |= 0x800000;

    int k = TOTAL_PREC - BINARY16_PREC;   // 13
    uint32_t mask = (1U << k) - 1;
    uint32_t bias = (1U << (k - 1));      // 0x1000 = 半个 ulp

    uint64_t m64 = (uint64_t)mantissa + bias;

    if (m64 >= (1ULL << TOTAL_PREC)) {
        m64 >>= 1;
        exp += 1;
        if (exp >= 0xFF) {
            uint32_t inf_bits = (sign << 31) | (0xFF << 23);
            float inf;
            memcpy(&inf, &inf_bits, sizeof(float));
            return inf;
        }
    }

    uint32_t new_mantissa = (uint32_t)m64 & (~mask);
    if (!is_subnormal && exp > 0) {
        new_mantissa &= 0x7FFFFF;
    }

    uint32_t new_bits = (sign << 31) | (exp << 23) | new_mantissa;
    float result;
    memcpy(&result, &new_bits, sizeof(float));
    return result;
}

// ==================== PDE求解器 ====================
// 1. 32位直接计算（高精度参考解）
void solve_pde_32bit_direct(float* u, int n, float alpha, float dx, float dt, int steps) {
    float* temp = (float*)malloc(n * sizeof(float));
    float factor = alpha * dt / (dx * dx);

    for (int step = 0; step < steps; step++) {
        temp[0] = 0.0f;
        temp[n-1] = 0.0f;

        for (int i = 1; i < n-1; i++) {
            temp[i] = u[i] + factor * (u[i+1] - 2*u[i] + u[i-1]);
        }

        // 复制回 u，避免局部指针交换导致的结果丢失
        memcpy(u, temp, n * sizeof(float));
    }

    free(temp);
}

// 2. 32位降精度到16位（随机舍入）
void solve_pde_32bit_to_16bit(float* u, int n,
                              float alpha, float dx, float dt, int steps) {
    float* temp = (float*)malloc(n * sizeof(float));

    // 系数也舍入到 binary16
    float factor = stochastic_round_16(alpha * dt / (dx * dx));

    for (int step = 0; step < steps; step++) {
        temp[0] = 0.0f;
        temp[n-1] = 0.0f;

        // 每步运算后进行 binary16 随机舍入
        for (int i = 1; i < n-1; i++) {
            float diff1 = stochastic_round_16(u[i+1] - u[i]);
            float diff2 = stochastic_round_16(u[i] - u[i-1]);
            float laplacian = stochastic_round_16(diff1 - diff2);
            float update = stochastic_round_16(factor * laplacian);
            temp[i] = stochastic_round_16(u[i] + update);
        }

        memcpy(u, temp, n * sizeof(float));
    }

    free(temp);
}

// 3. 直接使用16位进行计算（模拟16位硬件，每步 RN 舍入）
void solve_pde_16bit_direct(float* u, int n, float alpha, float dx, float dt, int steps) {
    float* u16 = (float*)malloc(n * sizeof(float));
    float* temp = (float*)malloc(n * sizeof(float));

    // 转换初始条件到 binary16 (RN)
    for (int i = 0; i < n; i++) {
        u16[i] = round_to_16_rn(u[i]);
    }

    // 转换参数到 binary16
    float alpha16 = round_to_16_rn(alpha);
    float dx16 = round_to_16_rn(dx);
    float dt16 = round_to_16_rn(dt);

    // 计算系数：alpha * dt / (dx * dx)，每步 RN 舍入
    float dx_sq = round_to_16_rn(dx16 * dx16);
    float factor_num = round_to_16_rn(alpha16 * dt16);
    float factor = round_to_16_rn(factor_num / dx_sq);

    for (int step = 0; step < steps; step++) {
        temp[0] = 0.0f;
        temp[n-1] = 0.0f;

        // 全部使用 binary16 RN 舍入运算
        for (int i = 1; i < n-1; i++) {
            float diff1 = round_to_16_rn(u16[i+1] - u16[i]);
            float diff2 = round_to_16_rn(u16[i] - u16[i-1]);
            float laplacian = round_to_16_rn(diff1 - diff2);
            float update = round_to_16_rn(factor * laplacian);
            temp[i] = round_to_16_rn(u16[i] + update);
        }

        memcpy(u16, temp, n * sizeof(float));
    }

    // 转换回用于比较
    for (int i = 0; i < n; i++) {
        u[i] = u16[i];
    }

    free(u16);
    free(temp);
}

// ==================== 辅助函数 ====================
// 初始化温度分布：使用正弦函数，避免边界为0
void init_temperature(float* u, int n, float L) {
    for (int i = 0; i < n; i++) {
        float x = (float)i * L / (n - 1);
        u[i] = 0.5f * sinf(3.14159265f * x / L) + 0.5f;
    }
}

// 复制数组：将源数组复制到目标数组
void copy_array(float* dest, float* src, int n) {
    memcpy(dest, src, n * sizeof(float));
}

// 计算误差统计：比较参考解和测试解的误差
void calculate_errors(float* ref, float* test, int n,
                      float* max_abs_error, float* avg_abs_error,
                      float* max_rel_error, float* avg_rel_error) {
    *max_abs_error = 0.0f;
    *avg_abs_error = 0.0f;
    *max_rel_error = 0.0f;
    *avg_rel_error = 0.0f;

    int valid_rel_points = 0;

    for (int i = 0; i < n; i++) {
        float abs_err = fabsf(ref[i] - test[i]);
        *avg_abs_error += abs_err;

        if (abs_err > *max_abs_error) {
            *max_abs_error = abs_err;
        }

        if (fabsf(ref[i]) > 1e-12f) {
            float rel_err = abs_err / fabsf(ref[i]);
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
    printf("三种精度模式PDE求解器对比\n");
    printf("1. 32位直接计算\n");
    printf("2. 32位降精度到16位（随机舍入 SR）\n");
    printf("3. 直接使用16位计算（RN 舍入）\n");
    printf("========================================\n\n");

    // PDE参数设置
    const int N = 31;
    const float L = 1.0f;
    const float dx = L / (N-1);
    const float dt = 0.0005f;
    const float alpha = 0.01f;
    const int steps = 200;

    printf("PDE参数:\n");
    printf("  网格点数: %d\n", N);
    printf("  空间长度: %.2f\n", L);
    printf("  空间步长: %.6f\n", dx);
    printf("  时间步长: %.6f\n", dt);
    printf("  扩散系数: %.4f\n", alpha);
    printf("  时间步数: %d\n\n", steps);

    // 分配内存
    float* u_ref = (float*)malloc(N * sizeof(float));
    float* u_rand = (float*)malloc(N * sizeof(float));
    float* u_direct16 = (float*)malloc(N * sizeof(float));
    float* u_initial = (float*)malloc(N * sizeof(float));

    // 初始化温度分布
    init_temperature(u_initial, N, L);

    // ========== 测试1: 32位直接计算 ==========
    printf("测试1: 32位直接计算\n");
    printf("----------------------------------------\n");

    copy_array(u_ref, u_initial, N);
    solve_pde_32bit_direct(u_ref, N, alpha, dx, dt, steps);

    // ========== 测试2: 32位降精度到16位（随机舍入） ==========
    printf("\n测试2: 32位降精度到16位（随机舍入 SR）\n");
    printf("----------------------------------------\n");

    copy_array(u_rand, u_initial, N);
    g_rng_state = 12345;  // 固定随机种子确保结果可重复

    solve_pde_32bit_to_16bit(u_rand, N, alpha, dx, dt, steps);

    // 计算与32位直接计算的误差
    float max_abs, avg_abs, max_rel, avg_rel;
    calculate_errors(u_ref, u_rand, N, &max_abs, &avg_abs, &max_rel, &avg_rel);

    printf("与32位直接计算的误差:\n");
    printf("  最大绝对误差: %.6f\n", max_abs);
    printf("  平均绝对误差: %.6f\n", avg_abs);
    printf("  最大相对误差: %.2f%%\n", max_rel * 100.0f);
    printf("  平均相对误差: %.2f%%\n", avg_rel * 100.0f);

    // ========== 测试3: 直接使用16位计算 ==========
    printf("\n测试3: 直接使用16位计算（RN 舍入）\n");
    printf("----------------------------------------\n");

    copy_array(u_direct16, u_initial, N);
    solve_pde_16bit_direct(u_direct16, N, alpha, dx, dt, steps);

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

    for (int i = 0; i < N; i += 3) {
        float x = i * dx;
        float err_rand = fabsf(u_ref[i] - u_rand[i]);
        float err_direct16 = fabsf(u_ref[i] - u_direct16[i]);

        printf("%.3f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
               x, u_ref[i], u_rand[i], u_direct16[i], err_rand, err_direct16);
    }

    // ========== 统计汇总 ==========
    printf("\n\n精度对比统计汇总:\n");
    printf("========================================\n");

    float err_rand_max = 0.0f, err_rand_avg = 0.0f;
    float err_direct_max = 0.0f, err_direct_avg = 0.0f;
    float diff_max = 0.0f, diff_avg = 0.0f;

    for (int i = 0; i < N; i++) {
        float e1 = fabsf(u_ref[i] - u_rand[i]);
        float e2 = fabsf(u_ref[i] - u_direct16[i]);
        float d = fabsf(u_rand[i] - u_direct16[i]);

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
    printf("32位降精度到16位(随机)  %.6f      %.6f\n", err_rand_max, err_rand_avg);
    printf("直接16位计算            %.6f      %.6f\n", err_direct_max, err_direct_avg);
    printf("两种低精度方法差异      %.6f      %.6f\n", diff_max, diff_avg);

    // 清理内存
    free(u_ref);
    free(u_rand);
    free(u_direct16);
    free(u_initial);

    printf("\n测试完成！\n");

    return 0;
}
