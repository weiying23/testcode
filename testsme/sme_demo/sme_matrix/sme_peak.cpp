// sme_peak.cpp - SME fmopa peak FLOPS stress
// build (MUST be -O0):
//   /usr/bin/clang++ -O0 -march=armv9-a+sve+sve2+sme+sme-f64f64 sme_peak.cpp -o sme_peak
// run: ./sme_peak 20000000
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <thread>
#include <vector>
#include <cstdlib>

using namespace std::chrono;

static double g_a[8] __attribute__((aligned(64))) = {1,1,1,1,1,1,1,1};
static double g_b[8] __attribute__((aligned(64))) = {1,1,1,1,1,1,1,1};

// one group = 8 fmopa (za0..za7) = 8*64 FMA = 1024 FLOP
__attribute__((always_inline))
static inline void sme_fma_block(int64_t times) {
    __asm__ volatile(
        "1:\n\t"
        "fmopa za0.d, p0/m, p0/m, z1.d, z2.d\n\t"
        "fmopa za1.d, p0/m, p0/m, z1.d, z2.d\n\t"
        "fmopa za2.d, p0/m, p0/m, z1.d, z2.d\n\t"
        "fmopa za3.d, p0/m, p0/m, z1.d, z2.d\n\t"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n\t"
        "fmopa za5.d, p0/m, p0/m, z1.d, z2.d\n\t"
        "fmopa za6.d, p0/m, p0/m, z1.d, z2.d\n\t"
        "fmopa za7.d, p0/m, p0/m, z1.d, z2.d\n\t"
        "subs %[t], %[t], #1\n\t"
        "b.ne 1b\n\t"
        : [t] "+r"(times)
        :
        : "z1", "z2", "p0", "cc", "memory");
}

static inline void sme_init_za() {
    __asm__ volatile("smstart" ::: "memory");
    __asm__ volatile("zero {za0.d}" ::: "memory");
    __asm__ volatile("zero {za1.d}" ::: "memory");
    __asm__ volatile("zero {za2.d}" ::: "memory");
    __asm__ volatile("zero {za3.d}" ::: "memory");
    __asm__ volatile("zero {za4.d}" ::: "memory");
    __asm__ volatile("zero {za5.d}" ::: "memory");
    __asm__ volatile("zero {za6.d}" ::: "memory");
    __asm__ volatile("zero {za7.d}" ::: "memory");
    __asm__ volatile("ptrue p0.d" ::: "p0");
    __asm__ volatile("ld1d {z1.d}, p0/z, [%[p]]" :: [p]"r"(g_a) : "z1", "memory");
    __asm__ volatile("ld1d {z2.d}, p0/z, [%[p]]" :: [p]"r"(g_b) : "z2", "memory");
}

static inline void sme_shutdown() {
    __asm__ volatile("smstop" ::: "memory");
}

// fixed-work measurement: time is taken outside smstart..smstop
static double sme_worker_once(int64_t groups) {
    auto t0 = steady_clock::now();
    sme_init_za();
    sme_fma_block(groups);
    sme_shutdown();
    auto t1 = steady_clock::now();
    return duration<double>(t1 - t0).count();
}

static double gflops_of(int64_t groups, double sec) {
    return (double)groups * 8.0 * 128.0 / sec / 1e9;
}

int main(int argc, char **argv) {
    int64_t groups = (argc > 1) ? atoll(argv[1]) : 500000;
    if (groups < 1000) groups = 1000;

    printf("M4 SME FP64 fmopa peak stress\n");
    printf("one group = 8 fmopa = 8*64 FMA = 1024 FLOP\n");
    printf("fixed groups per thread = %lld\n\n", (long long)groups);

    double wsec = sme_worker_once(10000);
    printf("warmup 10000 groups: %.3f ms, %.1f GFLOPS\n\n",
           wsec * 1e3, gflops_of(10000, wsec));

    double best1 = 0.0;
    for (int r = 0; r < 3; r++) {
        double sec = sme_worker_once(groups);
        double gf = gflops_of(groups, sec);
        if (gf > best1) best1 = gf;
    }
    printf("single-thread SME fmopa: %8.1f GFLOPS\n\n", best1);

    int hw = (int)std::thread::hardware_concurrency();
    if (hw < 1) hw = 1;
    printf("hardware_concurrency = %d\n", hw);

    for (int n : {1, 2, 4, 6, 8, 10}) {
        if (n > hw) break;
        std::vector<double> secs(n, 0.0);
        std::vector<std::thread> pool;
        for (int t = 0; t < n; t++) {
            pool.emplace_back([&, t]() {
                secs[t] = sme_worker_once(groups);
            });
        }
        for (auto &th : pool) th.join();
        double agg = 0.0;
        for (int t = 0; t < n; t++) agg += gflops_of(groups, secs[t]);
        printf("%2d threads SME fmopa: %8.1f GFLOPS (aggregate)", n, agg);
        if (n >= 4) {
            printf("   per-thread:");
            for (int t = 0; t < n; t++) printf(" %.0f", gflops_of(groups, secs[t]));
        }
        printf("\n");
    }

    printf("\ndone.\n");
    return 0;
}
