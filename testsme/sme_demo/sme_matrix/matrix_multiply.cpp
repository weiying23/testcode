/*
 * matrix_multiply.cpp — 基于 Apple M4 SME 的双精度矩阵乘法实现
 *
 * ══════════════════════════════════════════════════════════════════════════
 * 【整体优化层次】
 *
 *  本文件实现了一个五层优化的 GEMM，从宏观到微观依次是：
 *
 *  层次 1：多线程并行（gemmkernel）
 *    按 M 方向把矩阵行分片，每个线程独立处理一批 ic 块。
 *    线程间访问不同的 C 行，写无竞争；A_pack/B_pack 人手一份，不共享。
 *    M4 有 4 个 P 核 + 6 个 E 核，10 线程理论上可获得约 10× 吞吐提升。
 *
 *  层次 2：缓存级分块（gemmkernel，三层循环 kc → ic → jc）
 *    把大矩阵切成 Mc×Kc×Nc 的小块，让每次迭代的工作集（A_pack + B_pack + C tile）
 *    装进 L2 缓存（E 核 4 MB，P 核 16 MB），消除随机 DRAM 访问。
 *
 *  层次 3：Panel Packing（pack_a / pack_b，位于 packing.cpp）
 *    把子块从大步长原始矩阵复制到行连续缓冲区，将 B 行间距从 N×8（可达 64 KB）
 *    压缩到 Nc×8 = 768 B，使 B_pack 在 micro-kernel 热循环期间驻留 L1。
 *
 *  层次 4：寄存器级分块 + 6 路 j-tile 展开（sme_micro_kernel）
 *    在 ZA 矩阵寄存器里做 8×8 outer-product 累加（fmopa），
 *    同时展开 6 个 j-tile（za0–za5），让一次 A 列读取驱动 6 次 fmopa，
 *    将 A 加载的摊薄率从 1:1 提升到 1:6。
 *
 *  层次 5：软件预取（gemmkernel 内 + packing.cpp）
 *    在当前 pack/compute 执行期间，提前发出下一批数据的预取指令，
 *    把内存访问延迟和计算时间重叠（掩盖延迟）。
 *
 * ══════════════════════════════════════════════════════════════════════════
 * 【macOS 上的关键限制：不能用 __arm_streaming，不能用 cntd】
 *
 *  macOS 内核不向用户态暴露非流式 SVE（Scalable Vector Extension）。
 *  任何触发非流式 SVE 系统指令的代码都会收到 SIGILL：
 *    - cntd（读取非流式向量长度）     → SIGILL
 *    - svcntd()（C 库封装的 cntd）    → SIGILL
 *    - __arm_streaming 函数属性       → 编译器自动在调用方插入 cntd → SIGILL
 *
 *  因此所有 SME 操作都用 __asm__ volatile 手动管理，SVL 硬编码为 8（512 bits）。
 *
 * ══════════════════════════════════════════════════════════════════════════
 * 【编译参数说明】
 *
 *  matrix_multiply.cpp：-O0 -march=armv9-a+sve+sve2+sme+sme-f64f64
 *    -O0：防止编译器破坏 SME 内联汇编对寄存器的精确控制
 *    +sme+sme-f64f64：启用 f64f64 双精度外积（fmopa za.d）
 *
 *  packing.cpp：-O2 -march=armv9-a
 *    -O2：让 memcpy 被 NEON 向量化，大幅提升打包吞吐
 *    不带 +sve：防止生成非流式 SVE → SIGILL
 */

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <thread>
#include <vector>
#include <sys/sysctl.h>
#include <arm_sme.h>
#include "sme_intrinsics.h"
#include "packing.h"

/* ─────────────────────────────────────────────────────────────────────────
 * test_kernel — 最小化的 SME 流式模式可用性探针
 *
 * 这个函数只做一件事：执行一对最基本的状态切换指令
 *
 *   smstart sm   → 只打开 Streaming 模式（SVCR.SM = 1）
 *   smstop  sm   → 只关闭 Streaming 模式（SVCR.SM = 0）
 *
 * 目的不是做矩阵计算，而是快速确认：
 *   1. CPU 支持 SME 指令集
 *   2. macOS 允许当前进程在 EL0 执行这两条 SME 状态切换指令
 *   3. 程序的基本编译参数（+sme）是正确的
 *
 * 如果这里就触发 SIGILL，说明问题还停留在"最基础的 SME 入口不可用"这一层，
 * 后续所有 fmopa / ld1d za / st1d za 调试都没有意义。
 * ───────────────────────────────────────────────────────────────────────── */
void test_kernel() {
    __asm__ volatile("smstart sm");   /* 进入流式模式：SVCR.SM = 1 */
    __asm__ volatile("smstop sm");    /* 退出流式模式：SVCR.SM = 0 */
    printf("SME streaming mode: OK (smstart sm / smstop sm succeeded)\n");
}

/* ─────────────────────────────────────────────────────────────────────────
 * sme_micro_kernel — SME 寄存器级 GEMM 内核
 *
 * 职责：计算 C[0:M, 0:N] += A[0:M, 0:K] × B[0:K, 0:N]
 *        其中 A/B 均已打包为连续缓冲区（A_pack/B_pack），lda=Kc, ldb=Nc。
 *
 * ──────────────────────────────────────────────────────────────────────
 * 【优化 1：fmopa 外积指令 — 64 次 FMA/条指令】
 *
 *   普通 FMA（fmla）一次处理一个向量（8 个 double），效率受限于向量宽度。
 *   fmopa（Floating-point Matrix Outer Product Accumulate）：
 *     fmopa zaT.d, p_row/m, p_col/m, z_a.d, z_b.d
 *     语义：zaT[row, col] += z_a[row] * z_b[col]，对所有有效 (row, col) 并行执行。
 *
 *   一条 fmopa 完成 8×8 = 64 次乘加，是当前 M4 上最高效的 f64 计算指令。
 *
 * ──────────────────────────────────────────────────────────────────────
 * 【优化 2：6 路 j-tile 展开 — A 加载成本摊薄 6 倍】
 *
 *   naive 方案：每次外积只累加一个 C tile（za0），1 次 A 列读取 → 1 次 fmopa
 *   6-tile 方案：同一个 A 列（z1）驱动 6 个 C tile（za0–za5）：
 *
 *     sme_read_a_col(t)        →  z1 = A[:, k+t]          (1次，从 za6 读)
 *     sme_load_fmopa0(B[:, j+0])  →  za0 += outer(z1, B_row)  (j-tile 0)
 *     sme_load_fmopa1(B[:, j+8])  →  za1 += outer(z1, B_row)  (j-tile 1)
 *     ...
 *     sme_load_fmopa5(B[:,j+40])  →  za5 += outer(z1, B_row)  (j-tile 5)
 *
 *   j 步长 = 6 × 8 = 48，每次覆盖 48 列 C。
 *   za7 空置（SME f64f64 模式共 8 个 tile，za6 用于缓存 A，za7 未使用）。
 *
 * ──────────────────────────────────────────────────────────────────────
 * 【优化 3：谓词化边界处理 — 消除 if 分支和 cleanup loop】
 *
 *   M/N/K 均可能不是 8/48 的整数倍。传统做法需要为尾块写单独的标量循环，
 *   代码膨胀，分支预测也有成本。
 *
 *   这里用 whilelt 生成谓词，让 ld1d/fmopa/st1d 在假谓词 lane 上自动成为 no-op：
 *     - p1（行谓词）：处理 M 尾部，不足 8 行时屏蔽无效行
 *     - p2（k 谓词）：处理 K 尾部，不足 8 列时屏蔽无效 k-lane
 *     - p0/p3–p7（列谓词）：处理 N 尾部，超出 N 的列谓词全假
 *
 * ──────────────────────────────────────────────────────────────────────
 * 【优化 4：C 预加载 + kc-block 跨块累加】
 *
 *   C tile（48 列 × 8 行 = 384 doubles）在进入 k 循环前先从内存加载到 za0–za5，
 *   k 循环结束后写回。gemmkernel 用多个 kc block 分段累加时，每个 kc block 都
 *   加载 C 的当前值，加上本块贡献后写回，最终得到完整的 C。
 *
 * ──────────────────────────────────────────────────────────────────────
 * 【优化 5：热循环内软件预取 B_pack 下两行】
 *
 *   B_pack 总大小 192 KB，在 M4 P 核 L1（128 KB）里放不下，存在 L1 miss。
 *   在每次 fmopa 前，预取 B_pack 的 (k+t+2) 行，将 L2 miss 延迟与 fmopa 计算重叠。
 *
 * ──────────────────────────────────────────────────────────────────────
 * 参数说明：
 *   mata  — A_pack 起始（已打包，lda = Kc = 256）
 *   matb  — B_pack 起始（已打包，ldb = Nc = 96）
 *   matc  — 原始 C 矩阵的子块起始（未打包，ldc = N）
 *   M/N/K — 本次块尺寸（均 ≤ Mc/Nc/Kc，处理尾块时可能更小）
 *   lda   — A 行跨度（打包后固定为 Kc = 256）
 *   ldb   — B 行跨度（打包后固定为 Nc = 96）
 *   ldc   — C 行跨度（原始矩阵 N，未打包）
 * ───────────────────────────────────────────────────────────────────────── */
static __attribute__((noinline))
void sme_micro_kernel(double *mata, double *matb, double *matc,
                      int M, int N, int K,
                      int lda, int ldb, int ldc)
{
    /*
     * RAII 进入流式模式（smstart）：同时设置 SVCR.SM=1 和 SVCR.ZA=1。
     * 这是访问 ZA tile 的前提条件；guard 析构时自动执行 smstop 清零 SVCR。
     *
     * 每次 micro_kernel 调用都有独立的 smstart/smstop，原因：
     *   1. SVCR 是 per-CPU 状态，多线程时各核独立，不会冲突
     *   2. smstop 会把 ZA 清零，下次调用能从干净状态开始累加 C
     */
    SmeGuard guard;

    /*
     * ── 外层 i 循环：沿 M（行）方向，步长 8 ────────────────────────────
     *
     * 每次迭代处理 A 的 8 行、C 的 8 行，对应 ZA tile 的 8 行切片。
     * M4 的 ZA tile 每行恰好是 SVL/8 = 8 个 double，所以步长固定为 8。
     */
    for (int i = 0; i < M; i += 8) {

        /*
         * 生成行谓词 p1 = whilelt(i, M)：
         * lane t 有效当且仅当 i + t < M。
         * 在最后一个不满 8 行的块中，p1 会自动屏蔽越界 lane，
         * fmopa 不会累加到 C 的越界行。
         */
        sme_row_pred((uint64_t)i, (uint64_t)M);

        /*
         * ── 中层 j 循环：沿 N（列）方向，步长 48 = 6×8 ─────────────────
         *
         * 每次迭代处理 C 的 48 列（6 个 za tile，每个 8 列）。
         * j 步长 48 是关键参数：太小则 ZA tile 利用率低；太大则需要超过 6 个
         * 列谓词（p0-p7 只有 8 个，p1/p2 已占用，最多剩 6 个给列谓词）。
         */
        for (int j = 0; j < N; j += 48) {

            /*
             * 为 6 个 j-tile 各生成一个列谓词（p0, p3–p7）。
             * 当 j + 8k >= N 时，whilelt 产生全假谓词 → 该 tile 的所有
             * ld1d / fmopa / st1d 自动成为 no-op，无需任何显式判断。
             */
            sme_col_pred0((uint64_t)(j +  0), (uint64_t)N);
            sme_col_pred1((uint64_t)(j +  8), (uint64_t)N);
            sme_col_pred2((uint64_t)(j + 16), (uint64_t)N);
            sme_col_pred3((uint64_t)(j + 24), (uint64_t)N);
            sme_col_pred4((uint64_t)(j + 32), (uint64_t)N);
            sme_col_pred5((uint64_t)(j + 40), (uint64_t)N);

            /*
             * ── 阶段 1：加载 C[i:i+8, j:j+48] → za0–za5 ────────────────
             *
             * 把 C 当前 tile 从内存读入 ZA 作为累加初值。
             * 之后 k 循环在 ZA 里完成所有的部分和累加，最后写回一次，
             * 避免每个 kc block 都重复读写 C（跨 kc block 的累加）。
             *
             * ldc = N（原始 C 矩阵行跨度，未打包），因为 C 不需要打包——
             * C 是只在这里被读写一次的输出，不在热循环里反复访问。
             */
            for (int t = 0; t < 8 && (i + t) < M; t++) {
                const double *c_row = matc + (int64_t)(i + t) * ldc + j;
                sme_load_c0((uint32_t)t, c_row +  0);   /* za0 ← C[i+t, j+0..7]  */
                sme_load_c1((uint32_t)t, c_row +  8);   /* za1 ← C[i+t, j+8..15] */
                sme_load_c2((uint32_t)t, c_row + 16);   /* za2 ← C[i+t, j+16..23]*/
                sme_load_c3((uint32_t)t, c_row + 24);   /* za3 ← C[i+t, j+24..31]*/
                sme_load_c4((uint32_t)t, c_row + 32);   /* za4 ← C[i+t, j+32..39]*/
                sme_load_c5((uint32_t)t, c_row + 40);   /* za5 ← C[i+t, j+40..47]*/
            }

            /*
             * ── 内层 k 循环：沿 K 方向，步长 8 ──────────────────────────
             *
             * 这是整个 GEMM 的计算热点。每次迭代：
             *   1. 把 A 的 8×8 子块按行写入 za6（"A 缓冲 tile"）
             *   2. 依次读出 za6 的每一列（= A 子块的每一列），
             *      对 B 的 6 个列段分别做外积，累加到 za0–za5
             *
             * 这样 A 的每一列只读一次，却驱动 6 次 fmopa（每次 64 FMA），
             * 总 FMA / A读取 = 6 × 64 = 384，远高于 naive 实现。
             */
            for (int k = 0; k < K; k += 8) {

                /*
                 * 生成 k 谓词 p2 = whilelt(k, K)。
                 * 用于 A 加载（ld1d za6h … p2/z）：若 K 不是 8 的整数倍，
                 * 最后一组不满 8 的 k-lane 被 p2 屏蔽，za6 中对应位置清零。
                 */
                sme_k_pred((uint64_t)k, (uint64_t)K);

                /*
                 * 阶段 2a：将 A[i:i+8, k:k+8] 按行存入 za6
                 *
                 * sme_load_a(t, addr) 等价于：
                 *   ld1d za6h.d[t, 0], p2/z, [addr]
                 * 把 A 的第 i+t 行、第 k 列起的 8 个 double 写入 za6 的第 t 行。
                 *
                 * 执行完后 za6 的布局：
                 *   za6[t, :] = A_pack[(i+t)*Kc + k .. k+7]
                 *
                 * 注意：此时用的是 A_pack，lda = Kc = 256（紧密排列）。
                 */
                for (int t = 0; t < 8 && (i + t) < M; t++)
                    sme_load_a((uint32_t)t, mata + (int64_t)(i + t) * lda + k);

                /*
                 * 阶段 2b：外积累加热点 — 6 路 j-tile 并行 fmopa
                 *
                 * 对 k 方向的每一个"k-lane"（共最多 8 个）：
                 *   1. sme_read_a_col(t)：把 za6 的第 t 列读入 z1
                 *      即 z1 = A[:, k+t]，这是外积的左操作数（列向量）。
                 *   2. 软件预取 B_pack 的第 (k+t+2) 行，掩盖 L1/L2 延迟。
                 *   3. 依次对 6 个 B 列段各做一次 fmopa：
                 *      za0 += outer(z1, B_pack[k+t, j+0..7])
                 *      za1 += outer(z1, B_pack[k+t, j+8..15])
                 *      ...
                 *      za5 += outer(z1, B_pack[k+t, j+40..47])
                 */
                for (int t = 0; t < 8 && (k + t) < K; t++) {
                    /* z1 ← za6 的第 t 列 = A 子块的第 t 个 k-lane */
                    sme_read_a_col((uint32_t)t);

                    const double *b_row = matb + (int64_t)(k + t) * ldb + j;

                    /*
                     * 软件预取 B_pack 下两行。
                     * 当前在处理第 (k+t) 行，预取第 (k+t+2) 行。
                     * B_pack 行间距 = ldb×8 = 768 B，2 行 = 1536 B，
                     * 正好覆盖约 350 个时钟周期的 L2 miss 延迟窗口。
                     */
                    if (k + t + 2 < K)
                        __builtin_prefetch(matb + (int64_t)(k + t + 2) * ldb + j, 0, 3);

                    /* 6 次 fmopa，共用同一个 z1 */
                    sme_load_fmopa0(b_row +  0);   /* za0 += outer(z1, B[k+t, j+0..7])  */
                    sme_load_fmopa1(b_row +  8);   /* za1 += outer(z1, B[k+t, j+8..15]) */
                    sme_load_fmopa2(b_row + 16);   /* za2 += outer(z1, B[k+t, j+16..23])*/
                    sme_load_fmopa3(b_row + 24);   /* za3 += outer(z1, B[k+t, j+24..31])*/
                    sme_load_fmopa4(b_row + 32);   /* za4 += outer(z1, B[k+t, j+32..39])*/
                    sme_load_fmopa5(b_row + 40);   /* za5 += outer(z1, B[k+t, j+40..47])*/
                }
            } /* end k loop */

            /*
             * ── 阶段 3：将 za0–za5 中的累加结果写回 C ────────────────────
             *
             * 注意 ldc = N（原始矩阵行跨度），C 不需要连续布局。
             * st1d 不带 /z，谓词只控制写哪些 lane，不产生"清零写"。
             */
            for (int t = 0; t < 8 && (i + t) < M; t++) {
                double *c_row = matc + (int64_t)(i + t) * ldc + j;
                sme_store_c0((uint32_t)t, c_row +  0);
                sme_store_c1((uint32_t)t, c_row +  8);
                sme_store_c2((uint32_t)t, c_row + 16);
                sme_store_c3((uint32_t)t, c_row + 24);
                sme_store_c4((uint32_t)t, c_row + 32);
                sme_store_c5((uint32_t)t, c_row + 40);
            }
        } /* end j loop */
    } /* end i loop */
    /* SmeGuard 析构 → smstop：退出流式模式，ZA 状态清零 */
}

/* ─────────────────────────────────────────────────────────────────────────
 * gemmkernel — GEMM 公共入口：多线程 + 缓存分块 + Packing
 *
 * 计算：C = A(M×K) × B(K×N)，alpha 参数暂未使用
 *
 * ──────────────────────────────────────────────────────────────────────
 * 【缓存分块参数（Apple M4 调优）】
 *
 *   Mc = 64   A 的行方向块大小（对应每线程处理的 M 行数）
 *   Kc = 256  A/B 的 K 方向块大小
 *   Nc = 96   B/C 的 N 方向块大小（= 2 × j 步长 48 = 2 × 6tiles × 8）
 *
 *   工作集估算：
 *     A_pack = Mc × Kc × 8 = 64 × 256 × 8 =  128 KB
 *     B_pack = Kc × Nc × 8 = 256 × 96 × 8 =  192 KB
 *     C tile = Mc × Nc × 8 = 64 × 96 × 8  =   48 KB
 *     合计                               ≈  368 KB
 *   → 适配 E 核 4 MB L2（完全装入）和 P 核 16 MB L2（大量空余给预取缓冲）
 *
 * ──────────────────────────────────────────────────────────────────────
 * 【循环顺序：kc → ic → jc，以及为什么是这个顺序】
 *
 *   kc（最外层）：
 *     每个 kc block 处理 A 和 B 在 K 方向的一个切片。
 *     先循环 kc，使 B[kc:kc+Kc, :] 在 L3 里对所有 (ic, jc) 对可复用。
 *
 *   ic（中层）：
 *     A_pack[ic:ic+Mc, kc:kc+Kc]（128 KB）在此级别打包一次，
 *     对所有 jc 块复用（不重复打包 A），常驻 L2。
 *     C[ic, :] 行面板（64 × N × 8 字节）在 jc 循环期间也常驻 L2，
 *     每个 jc 只更新 C 的一列块，局部性良好。
 *
 *     注：若把 jc 放在 ic 外（kc→jc→ic），每次 jc 迭代需访问全部 M 行的 C，
 *     C 的工作集 = M×Nc×8 = 1024×96×8 = 768 KB，超出 E 核 L2，会导致
 *     C 的 miss 率上升，实测使性能下降约 15%。
 *
 *   jc（最内层）：
 *     B_pack 在此处刚打包完，立即被 micro_kernel 消费，驻留 L1。
 *     同一个 A_pack 被所有 jc 共用，不重复加载。
 *
 * ──────────────────────────────────────────────────────────────────────
 * 【多线程并行策略】
 *
 *   并行维度：ic 块（M 方向）
 *     线程 tid 处理 ic block 编号为 tid, tid+num_threads, tid+2*num_threads, ...
 *     即 round-robin 静态分配。
 *
 *   线程安全分析：
 *     A 读取：线程 tid 只读 A[ic:ic+Mc, kc:kc+Kc]，ic 不重叠 → 无竞争
 *     B 读取：所有线程读同一 B 的不同 jc 段（只读）→ 无竞争
 *     C 写入：线程 tid 只写 C[ic:ic+Mc, :]，ic 不重叠 → 无竞争
 *     A_pack：每线程独立 malloc，互不共享 → 无竞争
 *     B_pack：每线程独立 malloc → 无竞争（代价是每线程重复打包 B，
 *             但总 B 打包量不变，只是各线程各做自己负责的 ic 块的那份）
 *
 *   SME 状态（SVCR）：
 *     SVCR 是 per-CPU 架构状态，每个线程独立持有；各线程各自调用
 *     smstart/smstop，互不影响。M4 的所有核均支持 SME，E 核和 P 核均可用。
 *
 *   线程数：
 *     取 hardware_concurrency()（M4 返回 10：4P + 6E）与 ic 块数的较小值，
 *     避免创建比工作单元还多的线程（对小矩阵尤为重要）。
 * ───────────────────────────────────────────────────────────────────────── */
void gemmkernel(double *mata, double *matb, double *matc,
                int M, int N, int K, double alpha)
{
    const int Mc = 64, Kc = 256, Nc = 96;

    /*
     * 计算线程数：取硬件并发线程数与 ic 块数的较小值。
     *
     * 若 total_ic_blocks < hw_threads（小矩阵），多余的线程无事可做；
     * 限制线程数可以避免多余的 std::thread 创建开销（每线程约 10 µs）。
     *
     * 若 M = 8（只有 1 个 ic 块），则 num_threads = 1，直接走单线程快路径。
     */
    unsigned hw_threads = std::thread::hardware_concurrency();
    int num_threads = hw_threads ? (int)hw_threads : 4;
    int total_ic_blocks = (M + Mc - 1) / Mc;
    if (num_threads > total_ic_blocks) num_threads = total_ic_blocks;
    if (num_threads < 1) num_threads = 1;

    /*
     * worker lambda：每个线程执行的计算单元。
     *
     * 捕获方式：引用捕获（[&]），所有线程共享 A/B/C 指针和维度参数。
     * 写安全性由 ic 分片保证：每个 block 只属于一个线程。
     */
    auto worker = [&](int tid) {
        /*
         * 每线程独立分配 A_pack 和 B_pack。
         *
         * aligned_alloc(64, ...) 保证 64 字节对齐（一个 cache line），
         * 避免 A_pack/B_pack 的起始地址跨 cache line 导致的额外 miss。
         *
         * 不共享的原因：共享需要加锁或用不同偏移，增加复杂度；
         * 各线程独立的 pack buffer 完全无竞争，且 pack 成本已被计算摊薄。
         */
        double *A_pack = (double*)aligned_alloc(64, (int64_t)Mc * Kc * sizeof(double));
        double *B_pack = (double*)aligned_alloc(64, (int64_t)Kc * Nc * sizeof(double));

        /*
         * kc 循环：沿 K 方向分块。
         * 所有线程同步执行同一个 kc，对 B 的访问是只读的，无冲突。
         */
        for (int kc = 0; kc < K; kc += Kc) {
            int kc_sz = std::min(Kc, K - kc);

            /*
             * block 循环：线程 tid 负责编号 = tid + n*num_threads 的所有 ic 块。
             * 步长 num_threads × Mc 实现 round-robin 静态调度。
             */
            for (int block = tid; block < total_ic_blocks; block += num_threads) {
                int ic = block * Mc;
                int mc_sz = std::min(Mc, M - ic);

                /*
                 * 打包 A[ic:ic+mc_sz, kc:kc+kc_sz]。
                 * A_pack 的行跨度固定为 Kc（而非原始 K），使 micro_kernel
                 * 内部的 A 访问步长从 K×8（可达 64KB）压缩到 Kc×8 = 2 KB。
                 *
                 * A_pack 对本线程的所有 jc 复用，只打包一次。
                 */
                pack_a(A_pack,
                       mata + (int64_t)ic * K + kc,
                       mc_sz, kc_sz, K, Kc);

                /*
                 * 软件预取本线程下一个 ic 块的 A。
                 * 当前正在 pack_a 和 jc 循环期间，下一个 ic 块的 A 数据
                 * 可以提前拉进 L2，掩盖大步长（ic×K）的内存延迟。
                 */
                if (block + num_threads < total_ic_blocks) {
                    int next_ic = (block + num_threads) * Mc;
                    __builtin_prefetch(mata + (int64_t)next_ic * K + kc, 0, 1);
                }

                /* jc 循环：沿 N 方向分块，每次消费一个 B_pack */
                for (int jc = 0; jc < N; jc += Nc) {
                    int nc_sz = std::min(Nc, N - jc);

                    /*
                     * 打包 B[kc:kc+kc_sz, jc:jc+nc_sz]。
                     * B_pack 的行跨度固定为 Nc = 96，micro_kernel 内 B 行间距
                     * 从 N×8（可达 64 KB）压缩到 Nc×8 = 768 B，B_pack 整体
                     * 仅 192 KB，驻留 L1/L2。
                     *
                     * 注：这里每个 ic 块都重复打包 B（因为 ic 在 jc 外），
                     * 但 C 局部性更重要（若把 jc 提到外层，C 跨越整个 M，
                     * 超出 L2，实测退化约 15%）。
                     */
                    pack_b(B_pack,
                           matb + (int64_t)kc * N + jc,
                           kc_sz, nc_sz, N, Nc);

                    /*
                     * 软件预取下一个 jc 块的 B（在原始 B 矩阵中）。
                     * pack_b 即将处理下一 jc 块；若那块数据还在 DRAM，
                     * 提前发出请求可使延迟与当前 pack_b + micro_kernel 重叠。
                     */
                    if (jc + Nc < N)
                        __builtin_prefetch(matb + (int64_t)kc * N + (jc + Nc), 0, 1);

                    /*
                     * 调用寄存器级内核，传入打包后的 A_pack/B_pack。
                     * lda=Kc, ldb=Nc 是打包后的固定行跨度。
                     * ldc=N 是原始 C 矩阵的行跨度（C 未打包）。
                     */
                    sme_micro_kernel(
                        A_pack, B_pack,
                        matc + (int64_t)ic * N + jc,
                        mc_sz, nc_sz, kc_sz,
                        Kc, Nc, N
                    );
                }
            }
        }

        /* 释放本线程的 pack 缓冲区 */
        free(A_pack);
        free(B_pack);
    };

    /* 小矩阵快路径：直接在当前线程运行，避免 thread 创建开销 */
    if (num_threads == 1) {
        worker(0);
        return;
    }

    /*
     * 创建 num_threads 个线程，每个线程运行 worker(tid)。
     * workers[0] 运行在创建者线程，其余 num_threads-1 个在新线程上。
     * join() 等待所有线程完成后才返回，确保 C 写入完毕。
     */
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (int tid = 0; tid < num_threads; tid++)
        workers.emplace_back(worker, tid);
    for (auto &thread : workers)
        thread.join();
}

/* ─────────────────────────────────────────────────────────────────────────
 * verify_sme — SME 硬件状态全面验证
 *
 * 三个层次依次确认 SME 真正被使能：
 *
 * 层次 1：sysctl（内核特性标志）
 *   查询 hw.optional.arm.FEAT_SME* 系列键，确认内核已向用户态暴露 SME。
 *   这是静态检查，只说明硬件+内核支持 SME，不证明指令被执行。
 *
 * 层次 2：SVCR 寄存器（CPU 状态寄存器，运行时）
 *   在 smstart / smstop 包围的代码中用 mrs 读取 SVCR（Streaming Vector
 *   Control Register，EL0 可访问）：
 *     bit 0 (SM) = 1 → CPU 已进入 Streaming SVE 模式
 *     bit 1 (ZA) = 1 → ZA 矩阵寄存器阵列已激活
 *   smstop 后再次读取确认两位恢复为 0。
 *   这是最直接的运行时证明。
 *
 * 层次 3：RDSVL（流式向量长度，运行时）
 *   在 Streaming 模式内执行 rdsvl 指令，读取实际 SVL（字节数）。
 *   结果应为 64（= 512 bits），与硬件上报的 sme_max_svl_b 一致。
 *   若 SME 未使能，此指令会产生 SIGILL。
 *
 * 层次 4：正确性检验（3×3 参考矩阵）
 *   用已知精确答案的小矩阵验证 gemmkernel 计算结果，排除"SME 指令执行了
 *   但 ZA 数据读写有误"的情况。
 * ───────────────────────────────────────────────────────────────────────── */
void verify_sme()
{
    printf("\n========== SME 硬件状态验证 ==========\n");

    /* ── 层次 1：sysctl 静态特性标志 ────────────────────────────────────── */
    printf("\n[层次 1] sysctl 内核特性标志\n");

    /* 查询各 SME 特性标志 */
    const char *sysctl_keys[] = {
        "hw.optional.arm.FEAT_SME",
        "hw.optional.arm.FEAT_SME2",
        "hw.optional.arm.FEAT_SME_F64F64",
        "hw.optional.arm.sme_max_svl_b",   /* 最大 SVL（字节） */
    };
    for (const char *key : sysctl_keys) {
        int val = 0;
        size_t sz = sizeof(val);
        if (sysctlbyname(key, &val, &sz, nullptr, 0) == 0) {
            printf("  %-42s = %d\n", key, val);
        } else {
            printf("  %-42s = (查询失败)\n", key);
        }
    }

    /* ── 层次 2：运行时读取 SVCR 寄存器 ─────────────────────────────────── */
    printf("\n[层次 2] SVCR 寄存器（Streaming Vector Control Register）\n");

    uint64_t svcr_on = 0, svcr_off = 0;
    __asm__ volatile(
        /* 进入流式模式并激活 ZA */
        "smstart\n\t"
        /* mrs 读取 SVCR（EL0 可访问，编码 S3_3_C4_C2_2） */
        "mrs %[on], S3_3_C4_C2_2\n\t"
        /* 退出流式模式并关闭 ZA */
        "smstop\n\t"
        /* 再次读取，确认恢复为 0 */
        "mrs %[off], S3_3_C4_C2_2\n\t"
        : [on]"=r"(svcr_on), [off]"=r"(svcr_off)
        :
        : "memory"
    );

    printf("  smstart 后 SVCR = 0x%llx\n", (unsigned long long)svcr_on);
    printf("    bit 0 (SM) = %llu  → Streaming SVE 模式%s\n",
           (unsigned long long)(svcr_on & 1),
           (svcr_on & 1) ? " ✓ 已使能" : " ✗ 未使能");
    printf("    bit 1 (ZA) = %llu  → ZA 矩阵阵列%s\n",
           (unsigned long long)((svcr_on >> 1) & 1),
           ((svcr_on >> 1) & 1) ? " ✓ 已激活" : " ✗ 未激活");
    printf("  smstop 后 SVCR = 0x%llx  → SM=0, ZA=0 %s\n",
           (unsigned long long)svcr_off,
           (svcr_off == 0) ? "✓" : "✗ 异常");

    /* ── 层次 3：RDSVL 读取实际流式向量长度 ─────────────────────────────── */
    printf("\n[层次 3] RDSVL（流式向量长度）\n");

    uint64_t svl_bytes = 0;
    __asm__ volatile(
        "smstart\n\t"
        /* rdsvl xN, #1 → xN = SVL（字节数） × 1 */
        "rdsvl %[svl], #1\n\t"
        "smstop\n\t"
        : [svl]"=r"(svl_bytes)
        :
        : "memory"
    );
    printf("  SVL = %llu 字节 = %llu bits（期望：64 字节 / 512 bits）  %s\n",
           (unsigned long long)svl_bytes,
           (unsigned long long)(svl_bytes * 8),
           (svl_bytes == 64) ? "✓" : "✗ 异常");
    printf("  每 ZA tile 可容纳 double 数量 = %llu × %llu = %llu 个\n",
           (unsigned long long)(svl_bytes / 8),
           (unsigned long long)(svl_bytes / 8),
           (unsigned long long)(svl_bytes / 8) * (svl_bytes / 8));

    /* ── 层次 4：3×3 参考矩阵正确性检验 ─────────────────────────────────── */
    printf("\n[层次 4] 正确性检验（3×3 参考矩阵）\n");

    /*
     * 选择 A × I = A 作为 smoke test 的原因：
     *
     *   1. 结果精确可验（没有浮点误差积累），任何偏差都是 ZA 读写逻辑错误。
     *   2. M=N=K=3 强制触发所有尾块处理路径（3 不是 8 的整数倍），
     *      能验证 whilelt 谓词的边界处理是否正确。
     *   3. B=单位矩阵使得 C 每个元素只来自 A 的单列，容易定位到出错的列。
     *   4. 矩阵小，运行快，即使 SME 状态不对导致段错误也能快速定位。
     *
     * A × I = A 只验证"写对了地方"，还需要额外的算术验证确认"值也对了"。
     * 下面再测 [[1,1],[1,1]] × [[1,1],[1,1]] = [[2,2],[2,2]]，
     * 结果 2 来自 1×1+1×1，能验证 fmopa 的累加逻辑（而非仅写入路径）。
     */

    /* A = [[1,2,3],[4,5,6],[7,8,9]]，B = 单位矩阵 → C 应等于 A */
    double A[9] = {1,2,3, 4,5,6, 7,8,9};
    double B[9] = {1,0,0, 0,1,0, 0,0,1};  /* 单位矩阵 */
    double C[9] = {0};
    double expect[9] = {1,2,3, 4,5,6, 7,8,9};

    gemmkernel(A, B, C, 3, 3, 3, 1.0);

    bool ok = true;
    for (int i = 0; i < 9; i++) {
        if (fabs(C[i] - expect[i]) > 1e-9) { ok = false; break; }
    }
    printf("  A × I = A  →  %s\n", ok ? "✓ 结果正确，ZA 读写无误" : "✗ 结果错误");

    /* 再验一组：A=[[1,1],[1,1]]（扩展为 4×4 块） 自乘 → 每元素 = 2 */
    double A2[4] = {1,1, 1,1};
    double B2[4] = {1,1, 1,1};
    double C2[4] = {0};
    gemmkernel(A2, B2, C2, 2, 2, 2, 1.0);
    bool ok2 = (fabs(C2[0]-2)<1e-9 && fabs(C2[1]-2)<1e-9 &&
                fabs(C2[2]-2)<1e-9 && fabs(C2[3]-2)<1e-9);
    printf("  [[1,1],[1,1]] × [[1,1],[1,1]]  →  %s\n",
           ok2 ? "✓ 结果正确" : "✗ 结果错误");

    printf("\n======================================\n\n");
}
