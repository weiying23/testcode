#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

#define PI_F 3.14159265358979323846f

// ============================================================================
// 计时
// ============================================================================
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ============================================================================
// 防优化校验和
// ============================================================================
static double g_checksum = 0.0;

// ============================================================================
// 软件 SR (随机舍入到 binary16, from rr32_16.c)
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
// PDE: u_new[i] = u[i] + factor * (u[i+1] - 2*u[i] + u[i-1])
// 边界: u[0] = u[n-1] = 0
// ============================================================================

// 1. FP32 标量 (基线, 禁止向量化)
__attribute__((noinline))
void solve_fp32_scalar(float* restrict u, int n, float factor, int steps) {
    float* restrict t = malloc(n * sizeof(float));
    for (int s = 0; s < steps; s++) {
        t[0] = 0.0f; t[n-1] = 0.0f;
        #pragma clang loop vectorize(disable) interleave(disable)
        for (int i = 1; i < n-1; i++)
            t[i] = u[i] + factor * (u[i+1] - 2.0f*u[i] + u[i-1]);
        memcpy(u, t, n * sizeof(float));
    }
    free(t);
}

// 2. FP16 原生标量 (_Float16, 禁止向量化)
__attribute__((noinline))
void solve_fp16_scalar(_Float16* restrict u, int n, _Float16 factor, int steps) {
    _Float16* restrict t = malloc(n * sizeof(_Float16));
    _Float16 two = (_Float16)2.0;
    for (int s = 0; s < steps; s++) {
        t[0] = 0; t[n-1] = 0;
        #pragma clang loop vectorize(disable) interleave(disable)
        for (int i = 1; i < n-1; i++)
            t[i] = u[i] + factor * (u[i+1] - two*u[i] + u[i-1]);
        memcpy(u, t, n * sizeof(_Float16));
    }
    free(t);
}

// 3. FP32 NEON 向量化 (4 元素/寄存器)
__attribute__((noinline))
void solve_fp32_neon(float* restrict u, int n, float factor, int steps) {
    float* restrict t = malloc(n * sizeof(float));
    float32x4_t vf = vdupq_n_f32(factor);
    float32x4_t v2 = vdupq_n_f32(2.0f);
    for (int s = 0; s < steps; s++) {
        t[0] = 0.0f; t[n-1] = 0.0f;
        int i = 1;
        for (; i + 5 <= n; i += 4) {
            float32x4_t left   = vld1q_f32(&u[i-1]);
            float32x4_t center = vld1q_f32(&u[i]);
            float32x4_t right  = vld1q_f32(&u[i+1]);
            // lap = left + right - 2*center
            float32x4_t lap = vaddq_f32(left, right);
            lap = vmlsq_f32(lap, v2, center);       // lap - 2*center
            // res = center + factor * lap
            float32x4_t res = vmlaq_f32(center, vf, lap);
            vst1q_f32(&t[i], res);
        }
        for (; i < n-1; i++)
            t[i] = u[i] + factor * (u[i+1] - 2.0f*u[i] + u[i-1]);
        memcpy(u, t, n * sizeof(float));
    }
    free(t);
}

// 4. FP16 NEON 向量化 (8 元素/寄存器, 原生 FP16 运算)
__attribute__((noinline))
void solve_fp16_neon(_Float16* restrict u, int n, _Float16 factor, int steps) {
    _Float16* restrict t = malloc(n * sizeof(_Float16));
    const float16_t* pu = (const float16_t*)u;
    float16_t* pt = (float16_t*)t;
    float16x8_t vf = vdupq_n_f16((float16_t)factor);
    float16x8_t v2 = vdupq_n_f16((float16_t)2.0);
    for (int s = 0; s < steps; s++) {
        t[0] = 0; t[n-1] = 0;
        int i = 1;
        for (; i + 9 <= n; i += 8) {
            float16x8_t left   = vld1q_f16(&pu[i-1]);
            float16x8_t center = vld1q_f16(&pu[i]);
            float16x8_t right  = vld1q_f16(&pu[i+1]);
            // lap = left + right - 2*center
            float16x8_t lap = vaddq_f16(left, right);
            lap = vsubq_f16(lap, vmulq_f16(v2, center));
            // res = center + factor * lap  (使用 add+mul 代替 vmlaq)
            float16x8_t res = vaddq_f16(center, vmulq_f16(vf, lap));
            vst1q_f16(&pt[i], res);
        }
        for (; i < n-1; i++)
            t[i] = u[i] + factor * (u[i+1] - (_Float16)2*u[i] + u[i-1]);
        memcpy(u, t, n * sizeof(_Float16));
    }
    free(t);
}

// 5. FP32 存储 + 5 次软件 SR/点 (当前代码模式: 每步运算后都 SR)
__attribute__((noinline))
void solve_sr5(float* restrict u, int n, float factor, int steps) {
    float* restrict t = malloc(n * sizeof(float));
    factor = sr_round16(factor);
    for (int s = 0; s < steps; s++) {
        t[0] = 0.0f; t[n-1] = 0.0f;
        #pragma clang loop vectorize(disable) interleave(disable)
        for (int i = 1; i < n-1; i++) {
            float d1 = sr_round16(u[i+1] - u[i]);
            float d2 = sr_round16(u[i] - u[i-1]);
            float lap = sr_round16(d1 - d2);
            float upd = sr_round16(factor * lap);
            t[i] = sr_round16(u[i] + upd);
        }
        memcpy(u, t, n * sizeof(float));
    }
    free(t);
}

// 6. FP32 存储 + 1 次软件 SR/点 (优化: 只对最终结果 SR)
__attribute__((noinline))
void solve_sr1(float* restrict u, int n, float factor, int steps) {
    float* restrict t = malloc(n * sizeof(float));
    for (int s = 0; s < steps; s++) {
        t[0] = 0.0f; t[n-1] = 0.0f;
        #pragma clang loop vectorize(disable) interleave(disable)
        for (int i = 1; i < n-1; i++) {
            float res = u[i] + factor * (u[i+1] - 2.0f*u[i] + u[i-1]);
            t[i] = sr_round16(res);
        }
        memcpy(u, t, n * sizeof(float));
    }
    free(t);
}

// 7. 混合精度标量: FP16 存储 → FP32 计算 → 硬件 RN 转 FP16 存储
__attribute__((noinline))
void solve_mixed_scalar(_Float16* restrict u, int n, _Float16 factor_h, int steps) {
    _Float16* restrict t = malloc(n * sizeof(_Float16));
    float factor = (float)factor_h;
    for (int s = 0; s < steps; s++) {
        t[0] = 0; t[n-1] = 0;
        #pragma clang loop vectorize(disable) interleave(disable)
        for (int i = 1; i < n-1; i++) {
            // FP16→FP32 (隐式转换) → FP32 计算 → FP32→FP16 (硬件 RN, 单指令)
            float val = (float)u[i] + factor * ((float)u[i+1] - 2.0f*(float)u[i] + (float)u[i-1]);
            t[i] = (_Float16)val;
        }
        memcpy(u, t, n * sizeof(_Float16));
    }
    free(t);
}

// 8. 混合精度 NEON: FP16 存储 → FP32 NEON 计算 → 硬件 RN 转 FP16 存储
// 优化版: 1 次宽加载(128位) + vextq 提取, 替代 3 次窄加载
__attribute__((noinline))
void solve_mixed_neon(_Float16* restrict u, int n, _Float16 factor_h, int steps) {
    _Float16* restrict t = malloc(n * sizeof(_Float16));
    const float16_t* pu = (const float16_t*)u;
    float16_t* pt = (float16_t*)t;
    float factor = (float)factor_h;
    float32x4_t vf = vdupq_n_f32(factor);
    float32x4_t v2 = vdupq_n_f32(2.0f);
    for (int s = 0; s < steps; s++) {
        t[0] = 0; t[n-1] = 0;
        int i = 1;
        for (; i + 5 <= n; i += 4) {
            // 1 次宽加载 8 个 FP16 (u[i-1..i+6]), 转成 2 组 FP32x4
            float16x8_t raw = vld1q_f16(&pu[i-1]);
            float32x4_t lo = vcvt_f32_f16(vget_low_f16(raw));   // u[i-1..i+2]
            float32x4_t hi = vcvt_high_f32_f16(raw);            // u[i+3..i+6]
            // 移位提取 left/center/right (无需额外加载)
            float32x4_t left   = lo;                             // u[i-1..i+2]
            float32x4_t center = vextq_f32(lo, hi, 1);           // u[i..i+3]
            float32x4_t right  = vextq_f32(lo, hi, 2);           // u[i+1..i+4]
            // FP32 计算
            float32x4_t lap = vaddq_f32(left, right);
            lap = vmlsq_f32(lap, v2, center);
            float32x4_t res = vmlaq_f32(center, vf, lap);
            // FP32→FP16 硬件 RN + 存储
            vst1_f16(&pt[i], vcvt_f16_f32(res));
        }
        for (; i < n-1; i++) {
            float val = (float)u[i] + factor * ((float)u[i+1] - 2.0f*(float)u[i] + (float)u[i-1]);
            t[i] = (_Float16)val;
        }
        memcpy(u, t, n * sizeof(_Float16));
    }
    free(t);
}

// ============================================================================
// 辅助: 校验和
// ============================================================================
static double checksum_f32(const float* u, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)u[i];
    return s;
}
static double checksum_f16(const _Float16* u, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)u[i];
    return s;
}

// ============================================================================
// 主函数
// ============================================================================
int main(void) {
    printf("================================================================\n");
    printf("FP16 vs FP32 性能基准测试 (Apple M4, ARM NEON)\n");
    printf("PDE: 1D 热方程 u_new[i] = u[i] + factor*(u[i+1]-2*u[i]+u[i-1])\n");
    printf("factor = 0.25 (稳定, FP16/FP32 精确表示)\n");
    printf("================================================================\n\n");

    float factor = 0.25f;

    struct { int n; int steps; const char* cache; } sizes[] = {
        {1024,     5000, "L1"},
        {8192,     1000, "L1"},
        {65536,     200, "L2"},
        {524288,     50, "L2/Mem"},
        {4194304,    10, "Mem"},
    };
    int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int reps = 5;

    // 表头
    printf("%-9s %-6s %-7s | %-9s %-9s %-9s | %-9s %-9s %-9s | %-9s %-9s\n",
           "N", "Steps", "Cache",
           "FP32_s", "FP16_s", "Mix_s",
           "FP32_NEON", "FP16_NEON", "Mix_NEON",
           "SR1(1x)", "SR5(5x)");
    printf("%-9s %-6s %-7s | %-9s %-9s %-9s | %-9s %-9s %-9s | %-9s %-9s\n",
           "---------", "------", "-------",
           "---------", "---------", "---------",
           "---------", "---------", "---------",
           "---------", "---------");

    for (int si = 0; si < nsizes; si++) {
        int n = sizes[si].n;
        int steps = sizes[si].steps;

        // 初始化
        float* u32_init = malloc(n * sizeof(float));
        _Float16* u16_init = malloc(n * sizeof(_Float16));
        for (int i = 0; i < n; i++) {
            float x = (float)i / (n - 1);
            float v = 0.5f * sinf(PI_F * x) + 0.5f;
            u32_init[i] = v;
            u16_init[i] = (_Float16)v;
        }

        double t_fp32s = 0, t_fp16s = 0, t_fp32n = 0, t_fp16n = 0, t_sr5 = 0, t_sr1 = 0;
        double t_mixed_s = 0, t_mixed_n = 0;

        // --- FP32 scalar ---
        {
            float* u = malloc(n * sizeof(float));
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                memcpy(u, u32_init, n * sizeof(float));
                double t0 = now_sec();
                solve_fp32_scalar(u, n, factor, steps);
                double t1 = now_sec();
                if (t1 - t0 < best) best = t1 - t0;
            }
            t_fp32s = best;
            g_checksum += checksum_f32(u, n);
            free(u);
        }

        // --- FP16 scalar ---
        {
            _Float16* u = malloc(n * sizeof(_Float16));
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                memcpy(u, u16_init, n * sizeof(_Float16));
                double t0 = now_sec();
                solve_fp16_scalar(u, n, (_Float16)factor, steps);
                double t1 = now_sec();
                if (t1 - t0 < best) best = t1 - t0;
            }
            t_fp16s = best;
            g_checksum += checksum_f16(u, n);
            free(u);
        }

        // --- FP32 NEON ---
        {
            float* u = malloc(n * sizeof(float));
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                memcpy(u, u32_init, n * sizeof(float));
                double t0 = now_sec();
                solve_fp32_neon(u, n, factor, steps);
                double t1 = now_sec();
                if (t1 - t0 < best) best = t1 - t0;
            }
            t_fp32n = best;
            g_checksum += checksum_f32(u, n);
            free(u);
        }

        // --- FP16 NEON ---
        {
            _Float16* u = malloc(n * sizeof(_Float16));
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                memcpy(u, u16_init, n * sizeof(_Float16));
                double t0 = now_sec();
                solve_fp16_neon(u, n, (_Float16)factor, steps);
                double t1 = now_sec();
                if (t1 - t0 < best) best = t1 - t0;
            }
            t_fp16n = best;
            g_checksum += checksum_f16(u, n);
            free(u);
        }

        // --- SR5 (5 software SR/point) ---
        {
            float* u = malloc(n * sizeof(float));
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                memcpy(u, u32_init, n * sizeof(float));
                g_rng = 12345;
                double t0 = now_sec();
                solve_sr5(u, n, factor, steps);
                double t1 = now_sec();
                if (t1 - t0 < best) best = t1 - t0;
            }
            t_sr5 = best;
            g_checksum += checksum_f32(u, n);
            free(u);
        }

        // --- SR1 (1 software SR/point) ---
        {
            float* u = malloc(n * sizeof(float));
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                memcpy(u, u32_init, n * sizeof(float));
                g_rng = 12345;
                double t0 = now_sec();
                solve_sr1(u, n, factor, steps);
                double t1 = now_sec();
                if (t1 - t0 < best) best = t1 - t0;
            }
            t_sr1 = best;
            g_checksum += checksum_f32(u, n);
            free(u);
        }

        // --- Mixed scalar (FP16 store → FP32 compute → FP16 RN) ---
        {
            _Float16* u = malloc(n * sizeof(_Float16));
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                memcpy(u, u16_init, n * sizeof(_Float16));
                double t0 = now_sec();
                solve_mixed_scalar(u, n, (_Float16)factor, steps);
                double t1 = now_sec();
                if (t1 - t0 < best) best = t1 - t0;
            }
            t_mixed_s = best;
            g_checksum += checksum_f16(u, n);
            free(u);
        }

        // --- Mixed NEON (FP16 store → FP32 NEON compute → FP16 RN) ---
        {
            _Float16* u = malloc(n * sizeof(_Float16));
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                memcpy(u, u16_init, n * sizeof(_Float16));
                double t0 = now_sec();
                solve_mixed_neon(u, n, (_Float16)factor, steps);
                double t1 = now_sec();
                if (t1 - t0 < best) best = t1 - t0;
            }
            t_mixed_n = best;
            g_checksum += checksum_f16(u, n);
            free(u);
        }

        // 打印结果
        printf("%-9d %-6d %-7s | %7.2fms %7.2fms %7.2fms | %7.2fms %7.2fms %7.2fms | %7.2fms %7.2fms\n",
               n, steps, sizes[si].cache,
               t_fp32s * 1e3, t_fp16s * 1e3, t_mixed_s * 1e3,
               t_fp32n * 1e3, t_fp16n * 1e3, t_mixed_n * 1e3,
               t_sr1 * 1e3, t_sr5 * 1e3);

        // 关键加速比
        double r_fp16s = (t_fp16s > 1e-12) ? t_fp32s / t_fp16s : 0;
        double r_mixs  = (t_mixed_s > 1e-12) ? t_fp32s / t_mixed_s : 0;
        double r_fp16n = (t_fp16n > 1e-12) ? t_fp32n / t_fp16n : 0;
        double r_mixn  = (t_mixed_n > 1e-12) ? t_fp32n / t_mixed_n : 0;
        double r_sr1   = (t_sr1 > 1e-12) ? t_fp32s / t_sr1 : 0;
        printf("  → speedup | FP16_s=%.2fx  Mix_s=%.2fx | FP16_N=%.2fx  Mix_N=%.2fx | SR1=%.2fx\n",
               r_fp16s, r_mixs, r_fp16n, r_mixn, r_sr1);

        free(u32_init);
        free(u16_init);
    }

    printf("\n================================================================\n");
    printf("说明:\n");
    printf("  FP32_s    : FP32 标量 (基线, 4 字节/元素)\n");
    printf("  FP16_s    : FP16 原生标量 _Float16 (2 字节/元素)\n");
    printf("  Mix_s     : 混合精度标量 (FP16 存储→FP32 计算→FP16 RN, 2 字节/元素)\n");
    printf("  FP32_NEON : FP32 NEON (float32x4_t, 4 元素/寄存器)\n");
    printf("  FP16_NEON : FP16 原生 NEON (float16x8_t, 8 元素/寄存器)\n");
    printf("  Mix_NEON  : 混合精度 NEON (FP16 存储→FP32 NEON 计算→FP16 RN)\n");
    printf("  SR1/SR5   : 软件 SR (1次/5次 每点, FP32 存储)\n");
    printf("----------------------------------------------------------------\n");
    printf("  Mix_N vs FP32_N: 混合精度 NEON 相对 FP32 NEON 的加速比\n");
    printf("  → 如果访存瓶颈: 接近 2x (FP16 存储减半流量)\n");
    printf("  → 转换开销: 硬件 FCVT 单指令, 几乎免费\n");
    printf("(checksum: %.4f)\n", g_checksum);

    return 0;
}
