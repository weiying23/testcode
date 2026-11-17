#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ============================================================================
// 计时
// ============================================================================
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ============================================================================
// 软件 SR (随机舍入, 含 RNG)
// ============================================================================
static uint32_t g_rng = 12345;

static float sr_round16(float x) {
    if (!isfinite(x) || x == 0.0f) return x;
    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));
    uint32_t sign = (bits >> 31) & 1u;
    uint32_t exp  = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x7FFFFFu;
    int sub = (exp == 0);
    if (!sub) mant |= 0x800000u;
    uint32_t mask = (1u << 13) - 1u;
    uint32_t r = (g_rng = g_rng * 1103515245u + 12345u) & mask;
    uint64_t m = (uint64_t)mant + r;
    if (m >= (1ULL << 24)) { m >>= 1; exp++; }
    uint32_t nm = (uint32_t)m & ~mask;
    if (!sub && exp > 0) nm &= 0x7FFFFFu;
    uint32_t nb = (sign << 31) | (exp << 23) | nm;
    float res;
    memcpy(&res, &nb, sizeof(float));
    return res;
}

// ============================================================================
// 软件 RN (确定性舍入, 不含 RNG — 纯位操作开销)
// ============================================================================
static float rn_round16(float x) {
    if (!isfinite(x) || x == 0.0f) return x;
    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));
    uint32_t sign = (bits >> 31) & 1u;
    uint32_t exp  = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x7FFFFFu;
    int sub = (exp == 0);
    if (!sub) mant |= 0x800000u;
    uint32_t mask = (1u << 13) - 1u;
    uint32_t bias = (1u << 12);  // 0.5 ulp
    uint64_t m = (uint64_t)mant + bias;
    if (m >= (1ULL << 24)) { m >>= 1; exp++; }
    uint32_t nm = (uint32_t)m & ~mask;
    if (!sub && exp > 0) nm &= 0x7FFFFFu;
    uint32_t nb = (sign << 31) | (exp << 23) | nm;
    float res;
    memcpy(&res, &nb, sizeof(float));
    return res;
}

// ============================================================================
// 硬件 FP32→FP16 RN (单条 FCVT 指令, 用 _Float16 转换)
// ============================================================================
static float hw_round16(float x) {
    // 编译器生成单条 FCVT 指令
    return (float)(_Float16)x;
}

// ============================================================================
// 微基准测试
// ============================================================================
// 每个 case 处理 n 个元素, 重复 reps 次, 取最优
#define BENCH(name, body) \
    do { \
        double best = 1e18; \
        for (int r = 0; r < reps; r++) { \
            memcpy(c, a, n * sizeof(float)); \
            g_rng = 12345; \
            double t0 = now_sec(); \
            for (int i = 2; i < n; i++) { body; } \
            double t1 = now_sec(); \
            if (t1 - t0 < best) best = t1 - t0; \
        } \
        g_checksum += c[n/2]; \
        printf("  %-28s %8.3f ms  %7.1f ns/op  %5.1fx\n", \
               name, best * 1e3, best / (n - 2) * 1e9, \
               best / t_compute); \
    } while (0)

static double g_checksum = 0.0;

int main(void) {
    const int n = 1000000;   // 1M 元素
    const int reps = 50;

    float* a = malloc(n * sizeof(float));
    float* b = malloc(n * sizeof(float));
    float* c = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        a[i] = 0.5f * sinf((float)i * 0.001f) + 0.5f;
        b[i] = 0.3f * cosf((float)i * 0.001f) + 0.3f;
    }

    float factor = 0.25f;
    double t_compute = 0;

    printf("================================================================\n");
    printf("软件 SR 开销微基准 (Apple M4, 单线程, 1M 元素 x 50 次取最优)\n");
    printf("每点计算: c[i] = a[i] + factor*(a[i+1]-2*a[i]+a[i-1])  (5 FLOP)\n");
    printf("================================================================\n\n");

    // --- 基线: 纯 FP32 计算 (无舍入) ---
    BENCH("A. 纯 FP32 计算 (5 FLOP)",
          c[i] = a[i] + factor * (a[i+1] - 2.0f*a[i] + a[i-1]));
    t_compute = 1e18;
    { // 重新测量基线, 用于后续 ratio
        double best = 1e18;
        for (int r = 0; r < reps; r++) {
            memcpy(c, a, n * sizeof(float));
            double t0 = now_sec();
            for (int i = 2; i < n; i++)
                c[i] = a[i] + factor * (a[i+1] - 2.0f*a[i] + a[i-1]);
            double t1 = now_sec();
            if (t1 - t0 < best) best = t1 - t0;
        }
        t_compute = best;
        g_checksum += c[n/2];
    }

    printf("  %-28s %8.3f ms  %7.1f ns/op  %5.1fx  (基线)\n\n",
           "A. 纯 FP32 计算 (5 FLOP)", t_compute * 1e3,
           t_compute / (n - 2) * 1e9, 1.0);

    printf("--- 转换开销: 计算 + 1 次转换 ---\n");
    BENCH("C. 计算 + 1x hw_round16 (FCVT)",
          float tmp = a[i] + factor * (a[i+1] - 2.0f*a[i] + a[i-1]);
          c[i] = hw_round16(tmp));
    BENCH("D. 计算 + 1x rn_round16 (软件RN,无RNG)",
          float tmp = a[i] + factor * (a[i+1] - 2.0f*a[i] + a[i-1]);
          c[i] = rn_round16(tmp));
    BENCH("E. 计算 + 1x sr_round16 (软件SR,含RNG)",
          float tmp = a[i] + factor * (a[i+1] - 2.0f*a[i] + a[i-1]);
          c[i] = sr_round16(tmp));

    printf("\n--- 混合: 计算 + 5 次转换 (当前 rr32_16.c 模式) ---\n");
    BENCH("F. 计算 + 5x sr_round16 (软件SR)",
          float d1 = sr_round16(a[i+1] - a[i]);
          float d2 = sr_round16(a[i] - a[i-1]);
          float lap = sr_round16(d1 - d2);
          float upd = sr_round16(factor * lap);
          c[i] = sr_round16(a[i] + upd));

    printf("\n================================================================\n");
    printf("总结:\n");
    printf("  纯计算 (5 FLOP)           = 1.0x  (基线)\n");
    printf("  + 1x 硬件 FCVT             = ~1.0x  (几乎免费)\n");
    printf("  + 1x 软件 RN (位操作)      = 计算 + 1x 转换开销\n");
    printf("  + 1x 软件 SR (位操作+RNG)   = 计算 + 1x 转换开销\n");
    printf("  + 5x 软件 SR (当前代码)    = 计算 + 5x 转换开销\n");
    printf("(checksum: %.4f)\n", g_checksum);

    free(a); free(b); free(c);
    return 0;
}
