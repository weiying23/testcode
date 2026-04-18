/*
 * sme_demo.cpp — 最小化 SME 矩阵乘法 demo
 *
 * 功能：用 SME fmopa 指令计算 C = A × B，矩阵大小固定为 8×8 双精度浮点。
 *
 * 8×8 恰好是 SVL=512bits 下一个 ZA tile 的尺寸（8 行 × 8 列 = 64 个 double），
 * 这样代码最简洁，无需任何分块、循环展开或多线程——只展示 SME 的核心用法。
 *
 * SME 编程的 7 个核心步骤：
 *   1. smstart  — 进入流式模式，激活 ZA 矩阵寄存器
 *   2. zero     — 清零 ZA 累加器
 *   3. ptrue    — 设置全真谓词（所有 8 个 lane 都参与计算）
 *   4. ld1d za  — 将矩阵 A 逐行加载到 ZA 暂存 tile
 *   5. fmopa    — 外积累加：za0 += A 列向量 × B 行向量（64 次 FMA / 条指令）
 *   6. st1d za  — 将 ZA 结果逐行写回内存 C
 *   7. smstop   — 退出流式模式，清零 ZA 状态
 *
 * 注意（macOS 限制）：
 *   不能使用 __arm_streaming / svcntd() / cntd，
 *   这些会触发编译器插入非流式 SVE 指令 → macOS 内核报 SIGILL。
 *   因此所有 SME 状态切换都用 __asm__ volatile 手动管理。
 */

#include <cstdio>
#include <cmath>
#include <cstring>

static const int N = 8;   /* 矩阵维度：8×8，恰好一个 ZA tile */

/* ── 辅助：打印矩阵 ────────────────────────────────────────────── */
static void print_matrix(const char *name, const double *M)
{
    printf("%s:\n", name);
    for (int i = 0; i < N; i++) {
        printf("  [");
        for (int j = 0; j < N; j++)
            printf(" %7.2f", M[i * N + j]);
        printf(" ]\n");
    }
}

/* ── 辅助：标量参考实现，用于验证结果 ─────────────────────────── */
static void naive_gemm(const double *A, const double *B, double *C)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0;
            for (int k = 0; k < N; k++)
                s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

/* ────────────────────────────────────────────────────────────────
 * sme_gemm_8x8 — 用 SME 计算 C = A × B（8×8 双精度）
 *
 * 算法核心：
 *   C = sum_k  outer_product( A[:,k],  B[k,:] )
 *
 *   对每个 k，A[:,k] 是 A 的第 k 列（长度 8 的列向量），
 *           B[k,:] 是 B 的第 k 行（长度 8 的行向量）。
 *   两者做外积得到一个 8×8 矩阵，对所有 k 累加即得 C。
 *
 *   fmopa 指令正是硬件级的外积累加，一条指令完成 8×8=64 次乘加。
 *
 * ZA tile 使用：
 *   za1.d — 暂存整个矩阵 A（8 行按水平切片存入）
 *   za0.d — 累加结果，最终存的就是 C
 * ──────────────────────────────────────────────────────────────── */
static void sme_gemm_8x8(const double *A, const double *B, double *C)
{
    /* ── 步骤 1：smstart ────────────────────────────────────────
     * 同时设置 SVCR.SM=1（进入 Streaming SVE 模式）
     *          SVCR.ZA=1（激活 ZA 矩阵寄存器阵列）
     * 没有这一步，ld1d za / fmopa / st1d za 等指令都是非法指令。
     */
    __asm__ volatile("smstart" ::: "memory");

    /* ── 步骤 2：zero {za0.d} ───────────────────────────────────
     * 把 za0 的全部 64 个 double 清零，作为干净的累加起点。
     * 如果不清零，za0 中可能有上一次调用留下的残余值，导致结果错误。
     */
    __asm__ volatile("zero {za0.d}" ::: "memory");

    /* ── 步骤 3：ptrue p0.d ─────────────────────────────────────
     * 将谓词寄存器 p0 的所有 8 个 lane 设为 true（全真谓词）。
     * 后续的 ld1d / fmopa / st1d 都用 p0 作为谓词，
     * 全真意味着"处理所有 8 个元素，不做任何屏蔽"。
     */
    __asm__ volatile("ptrue p0.d" ::: "p0");

    /* ── 步骤 4：将矩阵 A 逐行加载到 za1（水平切片）────────────
     *
     * ld1d za1h.d[w12, 0], p0/z, [addr]
     *   za1h.d[w12, 0] = za1 的第 w12 行（水平切片，8 个 double）
     *   p0/z = 以 p0 为谓词，/z 表示假 lane 清零（这里全真，无影响）
     *   [addr]  = A[row, 0] 的地址，即 A 第 row 行的起始位置
     *
     * 执行完后 za1[row, col] = A[row, col]，za1 存储了完整的 A。
     * 后续步骤可以通过"垂直切片"读出 A 的列向量。
     */
    for (int row = 0; row < N; row++) {
        __asm__ volatile(
            "mov  w12, %w[r]\n\t"
            "ld1d za1h.d[w12, 0], p0/z, [%[p]]"
            :: [r]"r"(row), [p]"r"(A + row * N)
            : "w12", "memory"
        );
    }

    /* ── 步骤 5：外积累加循环（k = 0 … 7）──────────────────────
     *
     * 每次迭代处理矩阵 A 的第 k 列与矩阵 B 的第 k 行：
     *
     *   (a) 读 za1 的第 k 列（垂直切片）→ z1
     *       za1v.d[w12, 0] = za1 的第 w12 列 = A[:, k]（A 的列向量）
     *       mov z1.d, p0/m, za1v.d[w12, 0]：把该列复制到向量寄存器 z1
     *
     *   (b) 从内存加载 B 的第 k 行 → z2
     *       ld1d {z2.d}, p0/z, [addr]：从 B[k, 0] 加载 8 个 double 到 z2
     *
     *   (c) 外积累加：za0 += outer(z1, z2)
     *       fmopa za0.d, p0/m, p0/m, z1.d, z2.d
     *       语义：za0[i, j] += z1[i] * z2[j]，对所有 8×8 对同时执行
     *       一条指令完成 64 次乘加（FMA），这是 SME 的核心算力来源。
     */
    for (int k = 0; k < N; k++) {

        /* (a) z1 ← za1 第 k 列 = A[:, k] */
        __asm__ volatile(
            "mov  w12, %w[c]\n\t"
            "mov  z1.d, p0/m, za1v.d[w12, 0]"
            :: [c]"r"(k) : "w12", "z1"
        );

        /* (b) z2 ← B[k, 0..7] */
        __asm__ volatile(
            "ld1d {z2.d}, p0/z, [%[p]]"
            :: [p]"r"(B + k * N) : "z2", "memory"
        );

        /* (c) za0 += outer(z1, z2) */
        __asm__ volatile(
            "fmopa za0.d, p0/m, p0/m, z1.d, z2.d"
            ::: "memory"
        );
    }

    /* ── 步骤 6：将 za0 逐行写回内存 C ─────────────────────────
     *
     * st1d za0h.d[w12, 0], p0, [addr]
     *   za0h.d[w12, 0] = za0 的第 w12 行（8 个 double）
     *   p0 = 谓词（全真，写出所有 8 个元素）
     *   注意：st1d 的谓词不加 /z 后缀（st1d 没有"清零写"语义，/z 会报汇编错误）
     */
    for (int row = 0; row < N; row++) {
        __asm__ volatile(
            "mov  w12, %w[r]\n\t"
            "st1d za0h.d[w12, 0], p0, [%[p]]"
            :: [r]"r"(row), [p]"r"(C + row * N)
            : "w12", "memory"
        );
    }

    /* ── 步骤 7：smstop ─────────────────────────────────────────
     * 清除 SVCR.SM=0, SVCR.ZA=0，退出流式模式，ZA 内容归零。
     * 必须与 smstart 配对，否则后续普通 C++ 代码在流式模式下行为未定义。
     */
    __asm__ volatile("smstop" ::: "memory");
}

/* ────────────────────────────────────────────────────────────────
 * main — 运行两个验证 case
 * ──────────────────────────────────────────────────────────────── */
int main()
{
    printf("╔══════════════════════════════════╗\n");
    printf("║   SME Matrix Multiply Demo       ║\n");
    printf("║   8x8 double precision (1 tile)  ║\n");
    printf("╚══════════════════════════════════╝\n\n");

    double A[N * N], B[N * N], C[N * N];

    /* ── Case 1：A × I = A ──────────────────────────────────────
     * 用单位矩阵作为 B，结果应等于 A，便于目视验证。
     */
    printf("── Case 1: A × I = A ──────────────────────────────\n\n");

    for (int i = 0; i < N * N; i++) A[i] = i + 1.0;   /* A = 1,2,3,...,64 */
    memset(B, 0, sizeof(B));
    for (int i = 0; i < N; i++) B[i * N + i] = 1.0;   /* B = 单位矩阵 */
    memset(C, 0, sizeof(C));

    sme_gemm_8x8(A, B, C);

    print_matrix("A", A);
    printf("\n");
    print_matrix("C = A × I  (SME result)", C);

    bool ok1 = true;
    for (int i = 0; i < N * N; i++)
        if (fabs(C[i] - A[i]) > 1e-9) { ok1 = false; break; }
    printf("\nCase 1 结果：%s\n\n", ok1 ? "✓ PASS" : "✗ FAIL");

    /* ── Case 2：SME 结果 vs 标量参考实现 ──────────────────────
     * 用非平凡矩阵验证 fmopa 外积累加的数值正确性。
     */
    printf("── Case 2: SME vs 标量参考（随机值矩阵）────────────\n\n");

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i + 1) * 1.1 + j * 0.7;
            B[i * N + j] = (i * 0.5 + j + 1) * 0.9;
        }

    double C_sme[N * N], C_ref[N * N];
    memset(C_sme, 0, sizeof(C_sme));
    memset(C_ref, 0, sizeof(C_ref));

    sme_gemm_8x8(A, B, C_sme);
    naive_gemm(A, B, C_ref);

    double max_err = 0;
    for (int i = 0; i < N * N; i++)
        max_err = fmax(max_err, fabs(C_sme[i] - C_ref[i]));

    print_matrix("C (SME result)", C_sme);
    printf("\n");
    print_matrix("C (scalar reference)", C_ref);
    printf("\n最大绝对误差：%.2e\n", max_err);
    printf("Case 2 结果：%s\n\n", max_err < 1e-9 ? "✓ PASS" : "✗ FAIL");

    return (ok1 && max_err < 1e-9) ? 0 : 1;
}
