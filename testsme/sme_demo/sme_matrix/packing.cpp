/*
 * packing.cpp — A/B panel packing（矩阵子块打包）
 *
 * ══════════════════════════════════════════════════════════════════════════
 * 【为什么需要 packing】
 *
 * 在 GEMM 的三重循环里，访问 B 矩阵时的内存跨步（stride）极大：
 *
 *   B 行 k 在内存中的地址 = B_base + k * N * sizeof(double)
 *
 *   当 N = 4096 时：相邻两行间距 = 4096 × 8 = 32 KB
 *   当 N = 8192 时：相邻两行间距 = 8192 × 8 = 64 KB
 *
 * 每次访问一行 B，CPU 需要一个全新的 cache line 甚至 TLB 条目。
 * 若 B 很大（超出 L2/L3），每次访问都会导致 DRAM 访问，极度浪费带宽。
 *
 * Packing 把 B[kc:kc+Kc, jc:jc+Nc] 这个小子块复制到连续缓冲区 B_pack：
 *
 *   打包后行间距 = Nc × 8 = 96 × 8 = 768 字节 ≈ 12 条 cache line
 *
 * 整个 B_pack 只有 256 × 96 × 8 = 192 KB，远小于 L2（4~16 MB），
 * 打包之后 micro-kernel 的热循环全部在 L1/L2 命中。
 *
 * A 同理：A_pack = 64 × 256 × 8 = 128 KB，也驻留 L2。
 *
 * ══════════════════════════════════════════════════════════════════════════
 * 【为什么是独立编译单元，而不是在 matrix_multiply.cpp 里】
 *
 * 整个项目必须用 -O0 编译 matrix_multiply.cpp，原因：
 *   - SME 内联汇编依赖精确的寄存器分配；-O1 及以上可能重排指令或重用寄存器，
 *     破坏 za6/z1/w12 的使用约定，导致结果错误（已实验证实）。
 *
 * 但 packing 函数只含标量 C++ + memcpy，完全不涉及 SME 内联汇编，可以安全优化：
 *   -O2：让编译器展开循环、内联 memcpy、使用 NEON 向量化搬运
 *   -march=armv9-a（不加 +sve）：只允许 NEON，不生成非流式 SVE 指令
 *
 * 非流式 SVE 在 macOS 用户态是非法指令（SIGILL），所以不能带 +sve。
 * 分离编译后分别链接，两段代码互不干扰。
 *
 * ══════════════════════════════════════════════════════════════════════════
 * 【软件预取策略】
 *
 * pack_b 从原始 B 矩阵按大步长读取（可达 32~64 KB/行），L3 miss 的代价很高。
 * 通过 __builtin_prefetch 提前 4 行发出预取请求，可以在当前行 memcpy 执行时，
 * 让内存控制器把未来的行提前拉进 L2，从而把内存延迟和计算时间重叠（掩盖延迟）。
 *
 * 预取距离选择 4 行：
 *   Apple M4 的 DRAM 延迟约 100 ns，@3.5 GHz ≈ 350 个时钟周期。
 *   memcpy(768 B) 大约需要 50~100 周期。
 *   因此需要提前 3~6 行发出请求。4 行是一个保守而安全的选择。
 */

#include <cstdint>
#include <cstring>
#include "packing.h"

/* ─────────────────────────────────────────────────────────────────────────
 * pack_a — 把 A[0:mc, 0:kc_sz] 从大步长原始矩阵复制到行连续缓冲区
 *
 * 参数：
 *   dst        — 目标缓冲区（A_pack），行跨度为 kc_stride
 *   src        — 原始 A 矩阵中的子块起始位置（A[ic, kc]）
 *   mc         — 本次处理的行数（≤ Mc = 64）
 *   kc_sz      — 本次处理的列数（≤ Kc = 256）
 *   lda        — 原始 A 的行跨度（= 完整矩阵的 K，可达 8192）
 *   kc_stride  — A_pack 的行跨度（= Kc = 256，固定值）
 *
 * 内存布局变换示意（mc=3, kc_sz=5, Kc=8）：
 *
 *   原始 A（行间距 lda*8 字节）：     A_pack（行间距 Kc*8 字节）：
 *   [a00 a01 a02 a03 a04 ... ... ...]  [a00 a01 a02 a03 a04 0 0 0]
 *   [a10 a11 a12 a13 a14 ... ... ...]  [a10 a11 a12 a13 a14 0 0 0]
 *   [a20 a21 a22 a23 a24 ... ... ...]  [a20 a21 a22 a23 a24 0 0 0]
 *
 * 补零（0 填充）的原因：
 *   sme_load_a 使用 p2/z（k 谓词）加载 8 个 double，若尾部 kc_sz < Kc，
 *   超出 kc_sz 的位置会被 ZA load 读到；若不清零，ZA 里会有垃圾值影响 fmopa。
 * ───────────────────────────────────────────────────────────────────────── */
void pack_a(double *dst, const double *src,
            int mc, int kc_sz, int lda, int kc_stride)
{
    for (int i = 0; i < mc; i++) {
        /*
         * 软件预取：在处理第 i 行时，向硬件发出第 i+4 行的预取请求。
         * lda 可能高达 8192，两行之间相差 64 KB，远超 cache line（64 B），
         * 不提前预取必然触发 L2/L3 miss。
         *
         * __builtin_prefetch 参数：
         *   - 第 2 个参数 0 = 读预取（不是写）
         *   - 第 3 个参数 1 = 建议放入 L2（locality hint，0 = L1 stream, 3 = L1 keep）
         */
        if (i + 4 < mc)
            __builtin_prefetch(src + (int64_t)(i + 4) * lda, 0, 1);

        /* 将第 i 行的 kc_sz 个 double 复制到 A_pack 的连续行 */
        memcpy(dst + (int64_t)i * kc_stride,
               src + (int64_t)i * lda,
               kc_sz * sizeof(double));

        /* 尾部补零：当 kc_sz < kc_stride 时，填充剩余槽位为 0 */
        if (kc_sz < kc_stride)
            memset(dst + (int64_t)i * kc_stride + kc_sz, 0,
                   (kc_stride - kc_sz) * sizeof(double));
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * pack_b — 把 B[0:kc_sz, 0:nc_sz] 从大步长原始矩阵复制到行连续缓冲区
 *
 * 参数：
 *   dst        — 目标缓冲区（B_pack），行跨度为 nc_stride
 *   src        — 原始 B 矩阵中的子块起始位置（B[kc, jc]）
 *   kc_sz      — 本次处理的行数（≤ Kc = 256）
 *   nc_sz      — 本次处理的列数（≤ Nc = 96）
 *   ldb        — 原始 B 的行跨度（= 完整矩阵的 N，可达 8192）
 *   nc_stride  — B_pack 的行跨度（= Nc = 96，固定值）
 *
 * 打包后的收益（以 N=4096 为例）：
 *
 *   micro-kernel 里访问 B 的地址计算：
 *     原始：b_row = B + (k+t) * 4096          → 每次跨 32 KB
 *     packed：b_row = B_pack + (k+t) * 96     → 每次跨 768 B
 *
 *   整个 B_pack 大小 = 256 × 96 × 8 = 192 KB，在 L2 里全部命中。
 *   即使 N=8192，打包后也只需访问 192 KB，与 N 无关。
 *
 * 补零策略与 pack_a 相同，确保 sme_load_fmopa 读到的 padding 区域为 0。
 * ───────────────────────────────────────────────────────────────────────── */
void pack_b(double *dst, const double *src,
            int kc_sz, int nc_sz, int ldb, int nc_stride)
{
    for (int k = 0; k < kc_sz; k++) {
        /*
         * 软件预取：提前 4 行预热 B 的下一行。
         * ldb = N，当 N=8192 时行间距 64 KB，没有预取几乎必然 L3 miss。
         * 这是 pack_b 性能最关键的一行。
         */
        if (k + 4 < kc_sz)
            __builtin_prefetch(src + (int64_t)(k + 4) * ldb, 0, 1);

        /* 将第 k 行的 nc_sz 个 double 复制到 B_pack 的连续行 */
        memcpy(dst + (int64_t)k * nc_stride,
               src + (int64_t)k * ldb,
               nc_sz * sizeof(double));

        /* 尾部补零：nc_sz < nc_stride 时处理最后一个 jc block 的边界 */
        if (nc_sz < nc_stride)
            memset(dst + (int64_t)k * nc_stride + nc_sz, 0,
                   (nc_stride - nc_sz) * sizeof(double));
    }
}
