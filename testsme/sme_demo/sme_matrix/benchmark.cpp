/*
 * benchmark.cpp
 *
 * 对 SME GEMM kernel 进行：
 *   1. 多维度正确性验证（方阵 + 非方阵 + 非 8 倍数边界）
 *      以 naive 三重循环作为参考，计算最大绝对误差
 *   2. 与 Apple Accelerate 框架（cblas_dgemm）的性能对比
 *      报告各实现的执行时间和 GFLOPS
 *
 * 编译命令：
 *   /usr/bin/clang++ -O2 -march=armv9-a+sve+sve2+sme+sme-f64f64 \
 *       -framework Accelerate matrix_multiply.cpp benchmark.cpp -o benchmark
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <sys/time.h>
#include <Accelerate/Accelerate.h>
#include "matrix_methods.h"

/* ── 工具函数 ──────────────────────────────────────────────────────────── */

static double now_sec() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* 随机初始化矩阵，值域 [lo, hi]；n 用 int64_t 避免大矩阵溢出 */
static void rand_fill(double *mat, int64_t n, double lo = 0.5, double hi = 1.5) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(lo, hi);
    for (int64_t i = 0; i < n; i++) mat[i] = dist(rng);
}

/* Naive 三重循环参考实现（仅用于小矩阵正确性验证） */
static void naive_gemm(const double *A, const double *B, double *C,
                       int M, int N, int K) {
    memset(C, 0, sizeof(double) * M * N);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                C[(int64_t)i * N + j] += A[(int64_t)i * K + k]
                                       * B[(int64_t)k * N + j];
}

/* 两个矩阵间的最大绝对误差 */
static double max_abs_err(const double *ref, const double *got, int n) {
    double err = 0.0;
    for (int i = 0; i < n; i++)
        err = std::max(err, std::fabs(ref[i] - got[i]));
    return err;
}

/* 计算 GEMM 浮点操作数：2×M×N×K（用 int64_t 避免大规模溢出）*/
static double gflops(int M, int N, int K, double sec) {
    return 2.0 * (int64_t)M * N * K / sec * 1e-9;
}

/* ── 正确性测试 ────────────────────────────────────────────────────────── */

struct CorrectnessCase { int M, N, K; const char *desc; };

static void run_correctness_tests() {
    printf("════════════════════════════════════════════════════════════\n");
    printf("  正确性验证（SME 结果 vs Naive 参考，误差阈值 1e-9）\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  %-30s  %8s  %8s  %8s\n", "维度 M×K×N", "Naive", "SME误差", "BLAS误差");
    printf("  %-30s  %8s  %8s  %8s\n", "──────────────────────────────",
           "────────", "────────", "────────");

    CorrectnessCase cases[] = {
        /* 正好一个 8×8 tile */
        {  8,   8,   8, "方阵 8×8×8（1 tile）"},
        /* 整数倍 tile */
        { 16,  16,  16, "方阵 16×16×16"},
        { 64,  64,  64, "方阵 64×64×64"},
        /* 非 8 倍数——测试边界 tile */
        {  9,   7,  11, "非方阵 9×11×7（不规则）"},
        { 17,  13,  19, "非方阵 17×19×13"},
        { 37,  53,  73, "非方阵 37×73×53"},
        /* 宽矩阵 */
        { 32, 128,  64, "宽矩阵 32×64×128"},
        { 15,  33,  17, "宽矩阵 15×17×33（奇数）"},
        /* 高矩阵 */
        {128,  32,  64, "高矩阵 128×64×32"},
        /* 中等规模非方阵 */
        {100, 200, 150, "非方阵 100×150×200"},
        {256, 128, 192, "非方阵 256×192×128"},
    };

    const double ERR_THRESHOLD = 1e-9;
    bool all_pass = true;

    for (auto &c : cases) {
        int sz_a = c.M * c.K, sz_b = c.K * c.N, sz_c = c.M * c.N;
        double *A   = (double*)malloc(sz_a * sizeof(double));
        double *B   = (double*)malloc(sz_b * sizeof(double));
        double *ref = (double*)malloc(sz_c * sizeof(double)); /* naive 参考 */
        double *sme = (double*)malloc(sz_c * sizeof(double)); /* SME 结果   */
        double *bls = (double*)malloc(sz_c * sizeof(double)); /* BLAS 结果  */

        rand_fill(A, sz_a);
        rand_fill(B, sz_b);

        /* 参考：naive 三重循环 */
        naive_gemm(A, B, ref, c.M, c.N, c.K);

        /* SME kernel */
        memset(sme, 0, sz_c * sizeof(double));
        gemmkernel(A, B, sme, c.M, c.N, c.K, 1.0);

        /* Accelerate cblas_dgemm（行主序，C = 1×A×B + 0×C） */
        memset(bls, 0, sz_c * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    c.M, c.N, c.K, 1.0,
                    A, c.K, B, c.N, 0.0,
                    bls, c.N);

        double sme_err = max_abs_err(ref, sme, sz_c);
        double bls_err = max_abs_err(ref, bls, sz_c);
        bool   pass    = (sme_err < ERR_THRESHOLD);

        printf("  %-30s  %8.3g  %8.3g  %8.3g  %s\n",
               c.desc, 0.0, sme_err, bls_err,
               pass ? "✓" : "✗ FAIL");
        if (!pass) all_pass = false;

        free(A); free(B); free(ref); free(sme); free(bls);
    }

    printf("\n  总体结论：%s\n\n",
           all_pass ? "全部通过 ✓" : "存在失败项 ✗");
}

/* ── 性能测试 ──────────────────────────────────────────────────────────── */

struct PerfCase { int M, N, K; const char *desc; int reps; };

static void run_perf_tests() {
    printf("════════════════════════════════════════════════════════════\n");
    printf("  性能对比：SME vs Apple Accelerate（cblas_dgemm）\n");
    printf("  时间取多次运行最小值，单位 ms；GFLOPS = 2MNK / time\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  %-26s  %7s %7s  %7s %7s  %6s\n",
           "维度 M×K×N", "SME(ms)", "GFLOPS", "BLAS(ms)", "GFLOPS", "加速比");
    printf("  %-26s  %7s %7s  %7s %7s  %6s\n",
           "──────────────────────────",
           "───────", "───────", "────────", "───────", "──────");

    PerfCase cases[] = {
        /* 小规模 */
        {   64,   64,   64, "方阵 64³",             200},
        {  128,  128,  128, "方阵 128³",             100},
        {  256,  256,  256, "方阵 256³",              20},
        /* 中等方阵 */
        {  512,  512,  512, "方阵 512³",               5},
        { 1024, 1024, 1024, "方阵 1024³",              3},
        { 2048, 2048, 2048, "方阵 2048³",              1},
        { 4096, 4096, 4096, "方阵 4096³",              1},
        { 8192, 8192, 8192, "方阵 8192³",              1},
        /* 大规模非方阵 */
        { 4096, 8192, 2048, "4096×2048×8192",          1},
        { 8192, 4096, 1024, "8192×1024×4096",          1},
        /* 非 8 倍数边界 */
        {  100,  100,  100, "方阵 100³（非8倍）",      50},
        {  200,  300,  250, "200×250×300（非8倍）",    10},
    };

    for (auto &c : cases) {
        int64_t sz_a = (int64_t)c.M * c.K;
        int64_t sz_b = (int64_t)c.K * c.N;
        int64_t sz_c = (int64_t)c.M * c.N;
        double mem_mb = (sz_a + sz_b + sz_c) * 8.0 / (1024.0 * 1024.0);
        double *A   = (double*)malloc(sz_a * sizeof(double));
        double *B   = (double*)malloc(sz_b * sizeof(double));
        double *C   = (double*)malloc(sz_c * sizeof(double));

        if (!A || !B || !C) {
            printf("  %-26s  [跳过：内存分配失败，需 %.0f MB]\n",
                   c.desc, mem_mb);
            free(A); free(B); free(C);
            continue;
        }

        if (mem_mb > 500.0)
            printf("  [内存占用 %.0f MB，正在运行 %s ...]\n", mem_mb, c.desc);

        rand_fill(A, sz_a);
        rand_fill(B, sz_b);

        /* ── SME 计时 ── */
        double sme_best = 1e18;
        for (int r = 0; r < c.reps; r++) {
            memset(C, 0, sz_c * sizeof(double));
            double t0 = now_sec();
            gemmkernel(A, B, C, c.M, c.N, c.K, 1.0);
            double t1 = now_sec();
            sme_best = std::min(sme_best, t1 - t0);
        }

        /* ── Accelerate BLAS 计时 ── */
        double blas_best = 1e18;
        for (int r = 0; r < c.reps; r++) {
            memset(C, 0, sz_c * sizeof(double));
            double t0 = now_sec();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        c.M, c.N, c.K, 1.0,
                        A, c.K, B, c.N, 0.0,
                        C, c.N);
            double t1 = now_sec();
            blas_best = std::min(blas_best, t1 - t0);
        }

        double sme_ms   = sme_best  * 1000.0;
        double blas_ms  = blas_best * 1000.0;
        double sme_gf   = gflops(c.M, c.N, c.K, sme_best);
        double blas_gf  = gflops(c.M, c.N, c.K, blas_best);
        double speedup  = sme_ms / blas_ms;   /* >1 表示 BLAS 更快 */

        printf("  %-26s  %7.2f %7.2f  %8.2f %7.2f  %5.1fx\n",
               c.desc, sme_ms, sme_gf, blas_ms, blas_gf, speedup);

        free(A); free(B); free(C);
    }
    printf("\n  注：加速比 >1 表示 Accelerate 更快，<1 表示 SME 更快\n\n");
}

int main() {
    run_correctness_tests();
    run_perf_tests();
    return 0;
}
