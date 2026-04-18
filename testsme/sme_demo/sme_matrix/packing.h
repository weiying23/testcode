/*
 * packing.h — A / B 矩阵打包（Panel Packing）接口声明
 *
 * ══════════════════════════════════════════════════════════════════════════
 * 【为什么要做 Packing？】
 *
 *  原始矩阵 A（M×K）和 B（K×N）存储在行主序的大缓冲区中，相邻行之间
 *  的内存间距（stride）等于整行的字节数：
 *
 *    A 行间距 = K × 8 字节（例如 K=256 → 2 KB）
 *    B 行间距 = N × 8 字节（例如 N=8192 → 65 KB）
 *
 *  在 micro-kernel 的 k 循环中，每次加载 A 的下一行或 B 的下一行都需要
 *  跨越这么大的 stride；当 stride > L1 cache 大小时，几乎每次加载都是
 *  L1 miss，严重拖慢内核性能。
 *
 *  Packing 把子块复制到行连续的临时缓冲区（A_pack / B_pack）：
 *
 *    A_pack 行间距 = Kc × 8 = 256 × 8 = 2 KB（与 K 无关）
 *    B_pack 行间距 = Nc × 8 =  96 × 8 = 768 B（与 N 无关）
 *
 *  micro-kernel 只访问 A_pack/B_pack，每次 k 步只需顺序读取 768 B，
 *  整个 B_pack（192 KB）常驻 L1/L2，大幅减少 cache miss。
 *
 * ══════════════════════════════════════════════════════════════════════════
 * 【为什么 packing.cpp 独立编译？】
 *
 *  matrix_multiply.cpp 必须以 -O0 编译（防止编译器干预 SME 内联汇编）。
 *  但 pack_a / pack_b 是纯 C++ 内存复制，用 -O2 编译可以让编译器把内层
 *  循环向量化为 NEON 128-bit / 256-bit 指令，比 -O0 下的标量复制快 3~5 倍。
 *
 *  把 packing.cpp 单独以 -O2 -march=armv9-a（不带 +sve）编译：
 *    - 享受 NEON 自动向量化
 *    - 不触发非流式 SVE（不加 +sve 就不会生成 cntd → 不会 SIGILL）
 *
 * ══════════════════════════════════════════════════════════════════════════
 */
#pragma once

/*
 * pack_a — 将 A 的子块从原始矩阵复制到行连续缓冲区
 *
 * 参数：
 *   dst        — 输出缓冲区，A_pack，大小 ≥ mc × kc_stride × 8 字节
 *   src        — 原始 A 矩阵子块起始指针：&A[ic, kc]
 *   mc         — 本次要打包的行数（≤ Mc，处理尾块时可能更小）
 *   kc_sz      — 本次要打包的列数（≤ Kc，处理尾块时可能更小）
 *   lda        — 原始 A 矩阵的行跨度（= 整矩阵的 K）
 *   kc_stride  — A_pack 的行跨度（= Kc，固定值，尾块不足部分补零）
 *
 * 内存布局变换（示意）：
 *
 *   原始 A（行主序，stride = K）：          A_pack（行主序，stride = Kc）：
 *   A[ic+0, kc .. kc+kc_sz-1]               A_pack[0, 0 .. kc_sz-1]
 *   A[ic+1, kc .. kc+kc_sz-1]      →        A_pack[1, 0 .. kc_sz-1]
 *   ...                                      ...
 *   A[ic+mc-1, kc .. kc+kc_sz-1]            A_pack[mc-1, 0 .. kc_sz-1]
 *
 *   若 kc_sz < Kc，A_pack 每行末尾补零至 Kc 列（保证 micro-kernel 谓词
 *   屏蔽的 lane 读到 0，不影响计算结果）。
 */
void pack_a(double *dst, const double *src,
            int mc, int kc_sz, int lda, int kc_stride);

/*
 * pack_b — 将 B 的子块从原始矩阵复制到行连续缓冲区
 *
 * 参数：
 *   dst        — 输出缓冲区，B_pack，大小 ≥ kc_sz × nc_stride × 8 字节
 *   src        — 原始 B 矩阵子块起始指针：&B[kc, jc]
 *   kc_sz      — 本次要打包的行数（≤ Kc）
 *   nc_sz      — 本次要打包的列数（≤ Nc，处理尾块时可能更小）
 *   ldb        — 原始 B 矩阵的行跨度（= 整矩阵的 N）
 *   nc_stride  — B_pack 的行跨度（= Nc，固定值，尾块不足部分补零）
 *
 * 内存布局变换（示意）：
 *
 *   原始 B（行主序，stride = N，例如 N=8192 → 65 KB/行）：
 *   B[kc+0, jc .. jc+nc_sz-1]               B_pack[0, 0 .. nc_sz-1]
 *   B[kc+1, jc .. jc+nc_sz-1]      →        B_pack[1, 0 .. nc_sz-1]
 *   ...                                      ...
 *   B[kc+kc_sz-1, jc .. jc+nc_sz-1]         B_pack[kc_sz-1, 0 .. nc_sz-1]
 *
 *   打包后 B_pack 行间距 = Nc × 8 = 768 B，比原始最大 65 KB 小 85 倍，
 *   整个 B_pack 仅 192 KB，可以完全驻留 M4 E 核 L2（4 MB）。
 *
 *   若 nc_sz < Nc，B_pack 每行末尾补零至 Nc 列（保证列谓词全假时读到 0）。
 */
void pack_b(double *dst, const double *src,
            int kc_sz, int nc_sz, int ldb, int nc_stride);
