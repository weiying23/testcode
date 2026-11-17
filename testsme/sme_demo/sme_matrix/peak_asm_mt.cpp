// peak_asm_mt.cpp - NEON FP64 peak FMA stress (inline asm, multi-thread)
// build: /usr/bin/clang++ -O3 peak_asm_mt.cpp -o peak_asm_mt
// 16 fmla.2d per loop on independent accumulators v0-v15,
// constants held in v30/v31. This is the reference NEON peak tool.
#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>
using namespace std::chrono;

__attribute__((always_inline))
static inline void fma_block16(int64_t times) {
    __asm__ volatile(
        "1:\n"
        "fmla v0.2d, v30.2d, v31.2d\n"
        "fmla v1.2d, v30.2d, v31.2d\n"
        "fmla v2.2d, v30.2d, v31.2d\n"
        "fmla v3.2d, v30.2d, v31.2d\n"
        "fmla v4.2d, v30.2d, v31.2d\n"
        "fmla v5.2d, v30.2d, v31.2d\n"
        "fmla v6.2d, v30.2d, v31.2d\n"
        "fmla v7.2d, v30.2d, v31.2d\n"
        "fmla v8.2d, v30.2d, v31.2d\n"
        "fmla v9.2d, v30.2d, v31.2d\n"
        "fmla v10.2d, v30.2d, v31.2d\n"
        "fmla v11.2d, v30.2d, v31.2d\n"
        "fmla v12.2d, v30.2d, v31.2d\n"
        "fmla v13.2d, v30.2d, v31.2d\n"
        "fmla v14.2d, v30.2d, v31.2d\n"
        "fmla v15.2d, v30.2d, v31.2d\n"
        "subs %[t], %[t], #1\n"
        "b.ne 1b\n"
        : [t] "+r"(times)
        :
        : "v0","v1","v2","v3","v4","v5","v6","v7",
          "v8","v9","v10","v11","v12","v13","v14","v15",
          "v30","v31","cc","memory");
}

static void init_vec_regs() {
    __asm__ volatile("fmov d30, #0.5" ::: "v30");
    __asm__ volatile("mov v30.d[1], v30.d[0]" ::: "v30");
    __asm__ volatile("fmov d31, #0.5" ::: "v31");
    __asm__ volatile("mov v31.d[1], v31.d[0]" ::: "v31");
    __asm__ volatile(
        "movi v0.2d, #0\nmovi v1.2d, #0\nmovi v2.2d, #0\nmovi v3.2d, #0\n"
        "movi v4.2d, #0\nmovi v5.2d, #0\nmovi v6.2d, #0\nmovi v7.2d, #0\n"
        "movi v8.2d, #0\nmovi v9.2d, #0\nmovi v10.2d, #0\nmovi v11.2d, #0\n"
        "movi v12.2d, #0\nmovi v13.2d, #0\nmovi v14.2d, #0\nmovi v15.2d, #0\n"
        ::: "v0","v1","v2","v3","v4","v5","v6","v7",
            "v8","v9","v10","v11","v12","v13","v14","v15","v30","v31");
}

// run FMA blocks for dur seconds, return block count
static int64_t run_seconds(double dur) {
    init_vec_regs();
    int64_t warm = 500000;
    fma_block16(warm);
    auto t0 = steady_clock::now();
    auto tend = t0 + duration<double>(dur);
    int64_t blocks = 0;
    while (steady_clock::now() < tend) {
        fma_block16(1000);
        blocks += 1000;
    }
    return blocks;
}

// each block = 16 fmla.2d = 64 FLOP
int main(int argc, char** argv) {
    double dur = argc > 1 ? atof(argv[1]) : 0.5;
    if (dur < 0.2) dur = 0.5;
    int hw = (int)std::thread::hardware_concurrency();
    for (int n : {1, 2, 4, 6, 8, 10}) {
        if (n > hw) break;
        std::vector<int64_t> blocks(n, 0);
        std::vector<std::thread> pool;
        for (int t = 0; t < n; t++) {
            pool.emplace_back([&, t]() { blocks[t] = run_seconds(dur); });
        }
        for (auto &th : pool) th.join();
        double total_blocks = 0;
        for (auto b : blocks) total_blocks += (double)b;
        double gf = total_blocks * 16.0 * 4.0 / dur / 1e9;
        printf("%2d threads asm fma16: %.1f GFLOPS\n", n, gf);
    }
    return 0;
}
