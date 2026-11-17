// single_bench.cpp - single-core SME GEMM vs Apple Accelerate
// (P-core via QoS_USER_INTERACTIVE; E-core via: taskpolicy -b ./single_bench)
//
// Requires: matrix_multiply_single.o (gemmkernel forced to 1 thread, -O1)
//           packing.o (-O2), built by run_peak.sh
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <pthread.h>
#include <Accelerate/Accelerate.h>
#include "matrix_methods.h"

using namespace std::chrono;

static inline double now_sec() {
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static void rand_fill(double *mat, int64_t n) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.5, 1.5);
    for (int64_t i = 0; i < n; i++) mat[i] = dist(rng);
}

static double max_abs_err(const double *a, const double *b, int64_t n) {
    double err = 0.0;
    for (int64_t i = 0; i < n; i++)
        err = std::max(err, std::fabs(a[i] - b[i]));
    return err;
}

static double gflops(int M, int N, int K, double sec) {
    return 2.0 * (int64_t)M * N * K / sec / 1e9;
}

int main() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);

    int sizes[] = {64, 128, 256, 512, 1024, 2048};
    int nsizes = 6;

    printf("%-10s | %-14s | %-14s | %-10s | %s\n",
           "M=N=K", "SME GFLOPS", "BLAS GFLOPS", "SME/BLAS", "err");
    printf("-----------+----------------+----------------+------------+----------\n");

    for (int si = 0; si < nsizes; si++) {
        int M = sizes[si], N = sizes[si], K = sizes[si];
        int64_t sz_a = (int64_t)M * K, sz_b = (int64_t)K * N, sz_c = (int64_t)M * N;
        double *A = (double*)aligned_alloc(64, sz_a * 8);
        double *B = (double*)aligned_alloc(64, sz_b * 8);
        double *C_sme = (double*)aligned_alloc(64, sz_c * 8);
        double *C_blas = (double*)aligned_alloc(64, sz_c * 8);
        rand_fill(A, sz_a);
        rand_fill(B, sz_b);

        memset(C_sme, 0, sz_c * 8);
        gemmkernel(A, B, C_sme, M, N, K, 1.0);
        memset(C_blas, 0, sz_c * 8);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0, A, K, B, N, 0.0, C_blas, N);

        double sme_best = 1e9;
        for (int r = 0; r < 3; r++) {
            memset(C_sme, 0, sz_c * 8);
            double t0 = now_sec();
            gemmkernel(A, B, C_sme, M, N, K, 1.0);
            double t1 = now_sec();
            sme_best = std::min(sme_best, t1 - t0);
        }

        double blas_best = 1e9;
        for (int r = 0; r < 3; r++) {
            memset(C_blas, 0, sz_c * 8);
            double t0 = now_sec();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0, A, K, B, N, 0.0, C_blas, N);
            double t1 = now_sec();
            blas_best = std::min(blas_best, t1 - t0);
        }

        double sme_gf = gflops(M, N, K, sme_best);
        double blas_gf = gflops(M, N, K, blas_best);
        double err = max_abs_err(C_sme, C_blas, sz_c);

        printf("%-10d | %14.2f | %14.2f | %9.1f%% | %.2e | sme %.3f ms blas %.3f ms\n",
               M, sme_gf, blas_gf, sme_gf / blas_gf * 100.0, err,
               sme_best*1e3, blas_best*1e3);

        free(A); free(B); free(C_sme); free(C_blas);
    }

    printf("\nnote: VECLIB_MAXIMUM_THREADS=1, QoS=USER_INTERACTIVE (P core)\n");
    return 0;
}
