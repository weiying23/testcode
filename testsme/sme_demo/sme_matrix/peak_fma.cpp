// peak_fma.cpp - NEON vfmaq_f64 peak FMA stress (intrinsics version)
// build: /usr/bin/clang++ -O3 peak_fma.cpp -o peak_fma
#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>
#include <atomic>

using namespace std::chrono;

#define FLOP_PER_ITER 64.0

static volatile int sink_guard = 0;

// 16 independent accumulators to fill the FP pipelines
static int64_t fma_worker_seconds(double duration_sec, double &sink) {
    const float64x2_t a = vdupq_n_f64(1.0000000001);
    const float64x2_t b = vdupq_n_f64(0.9999999999);
    float64x2_t acc0  = vdupq_n_f64(0.5);
    float64x2_t acc1  = vdupq_n_f64(1.5);
    float64x2_t acc2  = vdupq_n_f64(2.5);
    float64x2_t acc3  = vdupq_n_f64(3.5);
    float64x2_t acc4  = vdupq_n_f64(4.5);
    float64x2_t acc5  = vdupq_n_f64(5.5);
    float64x2_t acc6  = vdupq_n_f64(6.5);
    float64x2_t acc7  = vdupq_n_f64(7.5);
    float64x2_t acc8  = vdupq_n_f64(8.5);
    float64x2_t acc9  = vdupq_n_f64(9.5);
    float64x2_t acc10 = vdupq_n_f64(10.5);
    float64x2_t acc11 = vdupq_n_f64(11.5);
    float64x2_t acc12 = vdupq_n_f64(12.5);
    float64x2_t acc13 = vdupq_n_f64(13.5);
    float64x2_t acc14 = vdupq_n_f64(14.5);
    float64x2_t acc15 = vdupq_n_f64(15.5);
    int64_t iters = 0;
    const int BLOCK = 50000;
    auto t0 = steady_clock::now();
    auto tend = t0 + duration<double>(duration_sec);
    while (steady_clock::now() < tend) {
        for (int j = 0; j < BLOCK; j++) {
            acc0  = vfmaq_f64(acc0,  a, b);
            acc1  = vfmaq_f64(acc1,  a, b);
            acc2  = vfmaq_f64(acc2,  a, b);
            acc3  = vfmaq_f64(acc3,  a, b);
            acc4  = vfmaq_f64(acc4,  a, b);
            acc5  = vfmaq_f64(acc5,  a, b);
            acc6  = vfmaq_f64(acc6,  a, b);
            acc7  = vfmaq_f64(acc7,  a, b);
            acc8  = vfmaq_f64(acc8,  a, b);
            acc9  = vfmaq_f64(acc9,  a, b);
            acc10 = vfmaq_f64(acc10, a, b);
            acc11 = vfmaq_f64(acc11, a, b);
            acc12 = vfmaq_f64(acc12, a, b);
            acc13 = vfmaq_f64(acc13, a, b);
            acc14 = vfmaq_f64(acc14, a, b);
            acc15 = vfmaq_f64(acc15, a, b);
            iters++;
        }
    }
    float64x2_t s = vaddq_f64(acc0, acc1);
    s = vaddq_f64(s, acc2);  s = vaddq_f64(s, acc3);
    s = vaddq_f64(s, acc4);  s = vaddq_f64(s, acc5);
    s = vaddq_f64(s, acc6);  s = vaddq_f64(s, acc7);
    s = vaddq_f64(s, acc8);  s = vaddq_f64(s, acc9);
    s = vaddq_f64(s, acc10); s = vaddq_f64(s, acc11);
    s = vaddq_f64(s, acc12); s = vaddq_f64(s, acc13);
    s = vaddq_f64(s, acc14); s = vaddq_f64(s, acc15);
    sink = vaddvq_f64(s);
    return iters;
}

struct WorkerArgs { double dur; int64_t iters; double sink; };

static void worker_main(WorkerArgs *a) {
    a->iters = fma_worker_seconds(a->dur, a->sink);
}

static double measure_n_threads_gflops(int nthreads, double dur) {
    std::vector<WorkerArgs> args(nthreads);
    for (int t = 0; t < nthreads; t++) args[t].dur = dur;

    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (int t = 0; t < nthreads; t++)
        pool.emplace_back(worker_main, &args[t]);
    for (auto &th : pool) th.join();

    double total_iters = 0.0;
    double total_sink = 0.0;
    for (int t = 0; t < nthreads; t++) {
        total_iters += (double)args[t].iters;
        total_sink += args[t].sink;
    }
    if (total_sink == -1.0) sink_guard++;
    return total_iters * FLOP_PER_ITER / dur / 1e9;
}

int main(int argc, char **argv) {
    double dur = (argc > 1) ? atof(argv[1]) : 0.5;
    if (dur < 0.2) dur = 0.5;

    printf("Apple M4 FP64 NEON vfmaq_f64 peak FMA test, dur=%.2fs\n\n", dur);

    // keep_sink is volatile: without it the compiler eliminates the whole
    // FMA loop in the inlined single-thread path (result: bogus 99000 GFLOPS)
    volatile double keep_sink = 0.0;
    double best1 = 0.0;
    for (int r = 0; r < 3; r++) {
        double sink0 = 0.0;
        int64_t it0 = fma_worker_seconds(dur, sink0);
        keep_sink += sink0;
        double g = (double)it0 * FLOP_PER_ITER / dur / 1e9;
        if (g > best1) best1 = g;
    }
    printf("single-thread : %8.1f GFLOPS (sink=%f)\n\n", best1, keep_sink);

    int hw = (int)std::thread::hardware_concurrency();
    if (hw < 1) hw = 1;
    printf("hardware_concurrency = %d\n", hw);
    for (int n : {1, 2, 4, 6, 8, 10}) {
        if (n > hw) break;
        double g = measure_n_threads_gflops(n, dur);
        printf("%2d threads    : %8.1f GFLOPS\n", n, g);
    }
    printf("\ndone.\n");
    return 0;
}
