/*
 * sme_intrinsics.h — SME 内联汇编封装层
 *
 * ══════════════════════════════════════════════════════════════════════════
 * 【为什么不用编译器生成的 SME intrinsic（arm_sme.h 中的 __arm_streaming）】
 *
 * arm_sme.h 提供了 svmopa_za64_f64_m() 等高级接口，但在 macOS 上全部不可用：
 *   - 当函数带有 __arm_streaming 属性时，编译器会在**调用方的 prologue** 插入
 *     cntd 指令（用于保存可变向量长度 VG，以支持 C++ 异常展开栈回溯）。
 *   - cntd 是"非流式 SVE"系统指令，macOS 内核不向用户态暴露非流式 SVE，
 *     执行后立即触发 SIGILL（非法指令信号）。
 *   - 解决方案：完全抛弃 __arm_streaming 属性，改用 __asm__ volatile 手动
 *     管理 smstart/smstop，编译器便不再生成 cntd。
 *
 * ══════════════════════════════════════════════════════════════════════════
 * 【SME 两个独立的使能开关】
 *
 *   smstart sm  → 仅设置 SVCR.SM=1（进入 Streaming SVE 模式）
 *   smstart za  → 仅设置 SVCR.ZA=1（激活 ZA 矩阵寄存器阵列）
 *   smstart     → 同时设置 SM 和 ZA（等价于 smstart sm + smstart za）
 *   smstop      → 同时清除 SM 和 ZA（退出流式模式并归零 ZA 状态）
 *
 *   我们始终使用无后缀的 smstart/smstop，因为 fmopa 需要 SM=1 且 ZA=1 同时成立。
 *
 * ══════════════════════════════════════════════════════════════════════════
 * 【ZA 寄存器阵列结构（SVL = 512 bits = 64 bytes，双精度 f64f64 模式）】
 *
 *   ZA 是一块 SVL×SVL bits 的二维矩阵寄存器，按 double 计算：
 *     SVL/64 = 8 个 double/向量，ZA 共 8×8 = 64 个 double，称为一个"ZA tile"。
 *
 *   SME f64f64 模式下，ZA 被划分为 8 个独立的 tile（za0.d … za7.d），
 *   每个 tile 可独立累加一个 8×8 double 矩阵。
 *
 *   访问方式分两种：
 *     水平切片（zaT h.d[w12, 0]）：第 w12 行的 8 个 double，按行存储
 *     垂直切片（zaT v.d[w12, 0]）：第 w12 列的 8 个 double，按列存储
 *   切片索引寄存器必须是 w12–w15（32 位），这是 SME 硬件约束。
 *
 * ══════════════════════════════════════════════════════════════════════════
 * 【谓词寄存器约束：只能使用 p0–p7】
 *
 *   ld1d/st1d/fmopa 等 SME 访存和计算指令的谓词操作数只接受 p0–p7（共 8 个）。
 *   p8–p15 是通用谓词寄存器，但 SME 访存指令编码不支持，强制使用会汇编报错。
 *   因此 6 路 j-tile 展开（za0–za5）恰好用完 p0, p3–p7 共 6 个列谓词，
 *   p1 留给行谓词，p2 留给 k 方向谓词，合计 8 个刚好耗尽。
 *
 * ══════════════════════════════════════════════════════════════════════════
 * 【寄存器分配总表（sme_micro_kernel 内）】
 *
 *   谓词寄存器：
 *     p0       — j-tile 0 的列谓词  [j+0,  j+8)  whilelt(j+0,  N)
 *     p1       — 行谓词             [i,    i+8)  whilelt(i,    M)；
 *                                   同时用作 mov z1.d, p1/m, za6v 的 merge 谓词
 *     p2       — k 方向谓词        [k,    k+8)  whilelt(k,    K)
 *     p3       — j-tile 1 的列谓词  [j+8,  j+16)
 *     p4       — j-tile 2 的列谓词  [j+16, j+24)
 *     p5       — j-tile 3 的列谓词  [j+24, j+32)
 *     p6       — j-tile 4 的列谓词  [j+32, j+40)
 *     p7       — j-tile 5 的列谓词  [j+40, j+48)
 *
 *   ZA tile（矩阵寄存器）：
 *     za0.d    — C 列块 0 的 8×8 累加器（对应 j+0..j+7 列）
 *     za1.d    — C 列块 1 的 8×8 累加器（j+8..j+15）
 *     za2.d    — C 列块 2 的 8×8 累加器（j+16..j+23）
 *     za3.d    — C 列块 3 的 8×8 累加器（j+24..j+31）
 *     za4.d    — C 列块 4 的 8×8 累加器（j+32..j+39）
 *     za5.d    — C 列块 5 的 8×8 累加器（j+40..j+47）
 *     za6.d    — A 子块缓存（水平加载 A 行，垂直读出 A 列供 fmopa 使用）
 *     za7.d    — 未使用（smstart 后自动激活，保持全零）
 *
 *   通用向量寄存器：
 *     z1.d     — 当前 A 列的 8 个 double（从 za6 垂直切片读出，供 fmopa 左操作数）
 *     z2.d     — 当前 B 行段的 8 个 double（从 B_pack 加载，供 fmopa 右操作数）
 *
 *   通用标量寄存器：
 *     w12      — ZA tile 切片索引（SME 硬件要求必须是 w12–w15）
 */
#pragma once
/*
 * SmeGuard — 用 RAII 自动管理流式模式生命周期
 *
 * 设计原因：
 *   SME 状态不是普通寄存器，而是 CPU 的执行模式；如果某条路径提前 return、
 *   throw（理论上）或 break，手写 smstop 很容易遗漏，导致后续代码仍在 streaming
 *   mode 下运行，行为不可预测。
 *
 * 优化意义：
 *   这里本身不是性能优化，而是正确性基础设施。通过 RAII 把 smstart/smstop 成对
 *   绑定，保证每个 micro-kernel 调用都在一个干净的 SME 上下文里执行。
 *
 * 注意：
 *   - 构造函数执行 smstart：同时打开 SM 和 ZA
 *   - 析构函数执行 smstop：同时关闭 SM 和 ZA
 *   - 使用 always_inline，避免 guard 本身引入额外函数调用开销
 */
struct SmeGuard {
    __attribute__((always_inline)) SmeGuard()  { __asm__ volatile("smstart" ::: "memory"); }
    __attribute__((always_inline)) ~SmeGuard() { __asm__ volatile("smstop"  ::: "memory"); }
    SmeGuard(const SmeGuard &) = delete;
    SmeGuard &operator=(const SmeGuard &) = delete;
};

/*
 * sme_row_pred — 生成行方向谓词 p1
 *
 * 含义：
 *   p1[d] = (i + lane < M) ? true : false
 *
 * 用途：
 *   1. 处理 micro-kernel 最后一个 i-block 时的尾行（不足 8 行）
 *   2. 作为 fmopa 左操作数的行掩码；无效行不会参与累加
 *   3. 作为 mov z1.d, p1/m, za6v 的 merge 谓词，仅将有效行写入 z1
 *
 * 优化意义：
 *   谓词化替代显式 if 分支，避免尾块走单独的慢路径；主循环和尾块共用同一段 SME
 *   汇编，减少分支预测开销，代码也更紧凑。
 */
__attribute__((always_inline)) static inline void sme_row_pred(uint64_t a, uint64_t b) {
    __asm__ volatile("whilelt p1.d, %0, %1" :: "r"(a), "r"(b) : "p1");
}

/*
 * sme_k_pred — 生成 k 方向谓词 p2
 *
 * 含义：
 *   p2[d] = (k + lane < K) ? true : false
 *
 * 用途：
 *   A 的每次加载是 8 个 double；若当前 k-block 的尾部不足 8 个元素，就用 p2 屏蔽
 *   无效 lane，保证尾部读取安全且结果正确。
 *
 * 优化意义：
 *   与行谓词类似，消除对 K 尾块的单独标量 cleanup loop。
 */
__attribute__((always_inline)) static inline void sme_k_pred(uint64_t a, uint64_t b) {
    __asm__ volatile("whilelt p2.d, %0, %1" :: "r"(a), "r"(b) : "p2");
}

/*
 * sme_col_pred0..5 — 生成 6 个 j-tile 的列方向谓词
 *
 * 每个谓词对应 8 列：
 *   p0 → [j+0,  j+8)
 *   p3 → [j+8,  j+16)
 *   p4 → [j+16, j+24)
 *   p5 → [j+24, j+32)
 *   p6 → [j+32, j+40)
 *   p7 → [j+40, j+48)
 *
 * 设计原因：
 *   6 个谓词正好对应 6 个 C accumulator tile（za0–za5）。当 N 不是 48 的整数倍时，
 *   超出边界的列谓词自动全假，于是对应的 ld1d/fmopa/st1d 全部成为 no-op。
 *
 * 优化意义：
 *   让 j 主循环可以固定按 48 列推进，而无需在 C++ 层判断“剩余列数 < 48”并切换到
 *   慢路径。边界统一由硬件谓词处理。
 */
__attribute__((always_inline)) static inline void sme_col_pred0(uint64_t a, uint64_t b) { __asm__ volatile("whilelt p0.d, %0, %1" :: "r"(a), "r"(b) : "p0"); }
__attribute__((always_inline)) static inline void sme_col_pred1(uint64_t a, uint64_t b) { __asm__ volatile("whilelt p3.d, %0, %1" :: "r"(a), "r"(b) : "p3"); }
__attribute__((always_inline)) static inline void sme_col_pred2(uint64_t a, uint64_t b) { __asm__ volatile("whilelt p4.d, %0, %1" :: "r"(a), "r"(b) : "p4"); }
__attribute__((always_inline)) static inline void sme_col_pred3(uint64_t a, uint64_t b) { __asm__ volatile("whilelt p5.d, %0, %1" :: "r"(a), "r"(b) : "p5"); }
__attribute__((always_inline)) static inline void sme_col_pred4(uint64_t a, uint64_t b) { __asm__ volatile("whilelt p6.d, %0, %1" :: "r"(a), "r"(b) : "p6"); }
__attribute__((always_inline)) static inline void sme_col_pred5(uint64_t a, uint64_t b) { __asm__ volatile("whilelt p7.d, %0, %1" :: "r"(a), "r"(b) : "p7"); }

/*
 * sme_load_a — 将 A 的一整行（最多 8 个 double）加载到 za6 的某一水平切片
 *
 * 汇编语义：
 *   mov  w12, row
 *   ld1d za6h.d[w12, 0], p2/z, [addr]
 *
 * 解释：
 *   - za6h.d[w12, 0] 表示 za6 的第 row 行（水平切片）
 *   - p2/z 表示由 k 谓词控制；无效 lane 自动补零
 *   - 这里把 A 的 8 行 × 8 列子块按“行”的形式临时存在 za6 中
 *
 * 优化意义：
 *   先把 A 子块全部搬进 ZA，再在后续阶段按“列”读出，就能做到：
 *     一次 A 加载 → 6 次 fmopa 复用
 *   这是 outer-product 微内核的关键：A 的加载成本被多个 C tile 分摊。
 */
__attribute__((always_inline))
static inline void sme_load_a(uint32_t row, const double *addr) {
    __asm__ volatile(
        "mov w12, %w[r]\n\t"
        "ld1d za6h.d[w12, 0], p2/z, [%[p]]"
        :: [r]"r"(row), [p]"r"(addr) : "w12", "memory"
    );
}

/*
 * sme_read_a_col — 从 za6 读取一列到 z1.d
 *
 * 汇编语义：
 *   mov w12, col
 *   mov z1.d, p1/m, za6v.d[w12, 0]
 *
 * 解释：
 *   - za6v.d[w12, 0] 取出 za6 的第 col 列，也就是 A 子块中固定 k 位置上的 8 个行值
 *   - 这 8 个 double 组成 outer-product 的左操作数向量 z1
 *   - p1/m 表示仅对有效行进行写入；越界行保持 merge 语义
 *
 * 为什么可以用 p1/m 而不是“全真谓词”：
 *   因为后续 fmopa 也用 p1/m，所有无效行本来就不会参与计算，所以 z1 中无效 lane
 *   保持未定义也不影响结果；这是一个省谓词寄存器的小技巧。
 *
 * 优化意义：
 *   通过“先按行写 ZA，再按列读 ZA”，把原本需要从内存按列收集的 A 数据，转换成
 *   一次本地 ZA 读操作。这正是 SME 相比普通 NEON/SVE 的优势：ZA 天然适合 outer-product。
 */
__attribute__((always_inline))
static inline void sme_read_a_col(uint32_t col) {
    __asm__ volatile(
        "mov w12, %w[c]\n\t"
        "mov z1.d, p1/m, za6v.d[w12, 0]"
        :: [c]"r"(col) : "w12", "z1"
    );
}

/*
 * _LOAD_C 宏族 — 将当前 C 子块的一行从内存加载到 ZA 累加器
 *
 * 生成 sme_load_c0 … sme_load_c5，分别写入 za0 … za5 的第 row 行。
 *
 * 展开示例（sme_load_c0）：
 *   mov  w12, row
 *   ld1d za0h.d[w12, 0], p0/z, [addr]
 *
 * 优化意义：
 *   将 C 的 8×48 tile 整体预加载进 ZA，在寄存器里完成整个 kc-block 的累加，
 *   最后只写回一次——大幅减少 C 的主存读写。
 *
 * 为什么使用 /z 谓词：
 *   ld1d … /z 把谓词为 false 的 lane 直接清零。超出 N 边界的列谓词全假，
 *   那些 ZA lane 被清零，不污染后续累加，实现安全的尾边界处理。
 */
#define _LOAD_C(T, P) \
    __attribute__((always_inline)) static inline \
    void sme_load_c##T(uint32_t row, const double *addr) { \
        __asm__ volatile("mov w12, %w[r]\n\t" \
                         "ld1d za" #T "h.d[w12, 0], " #P "/z, [%[p]]" \
                         :: [r]"r"(row), [p]"r"(addr) : "w12", "memory"); }
_LOAD_C(0,p0) _LOAD_C(1,p3) _LOAD_C(2,p4) _LOAD_C(3,p5) _LOAD_C(4,p6) _LOAD_C(5,p7)
#undef _LOAD_C

/*
 * _LOAD_FMOPA 宏族 — 加载 B 的一个 8 列行段并做 8×8 外积累加
 *
 * 生成 sme_load_fmopa0 … sme_load_fmopa5，分别累加到 za0 … za5。
 *
 * 展开示例（sme_load_fmopa0）：
 *   ld1d {z2.d}, p0/z, [b]         ; 从 B_pack 加载 8 个 double 到 z2
 *   fmopa za0.d, p1/m, p0/m, z1.d, z2.d  ; za0 += outer_product(z1, z2)
 *
 * 数学含义：
 *   za0[i_lane, j_lane] += z1[i_lane] * z2[j_lane]
 *   其中 i_lane 由 p1（行谓词）控制，j_lane 由 p0（列谓词）控制。
 *   一条 fmopa 完成 8×8 = 64 次融合乘加（FMA），是整个实现的算力核心。
 *
 * 关键复用收益：
 *   z1 在外层已经一次性从 za6 读出，接下来连续调用 sme_load_fmopa0 … fmopa5，
 *   用同一个 z1 分别乘以 B 的 6 个列段，6 个 C tile 并行累加。
 *   这等价于 1 次 A 列读取驱动 6 次 64-FMA，FMA 效率比原始单 tile 高 6 倍。
 */
#define _LOAD_FMOPA(T, P) \
    __attribute__((always_inline)) static inline \
    void sme_load_fmopa##T(const double *b) { \
        __asm__ volatile("ld1d {z2.d}, " #P "/z, [%[p]]\n\t" \
                         "fmopa za" #T ".d, p1/m, " #P "/m, z1.d, z2.d" \
                         :: [p]"r"(b) : "z2", "memory"); }
_LOAD_FMOPA(0,p0) _LOAD_FMOPA(1,p3) _LOAD_FMOPA(2,p4) _LOAD_FMOPA(3,p5) _LOAD_FMOPA(4,p6) _LOAD_FMOPA(5,p7)
#undef _LOAD_FMOPA

/*
 * _STORE_C 宏族 — 将 ZA 累加器的一行结果写回内存 C
 *
 * 生成 sme_store_c0 … sme_store_c5，分别写出 za0 … za5 的第 row 行。
 *
 * 展开示例（sme_store_c0）：
 *   mov  w12, row
 *   st1d za0h.d[w12, 0], p0, [addr]
 *
 * 为什么不写 p0/z：
 *   st1d 是"条件写"而非"条件读"；谓词控制哪些 lane 真正写回内存，没有"清零写"
 *   的语义。语法上 st1d 不接受 /z 后缀，硬要写会报汇编错误。
 *
 * 优化意义：
 *   整个 kc-block 都在 ZA 寄存器里累加，此处才是唯一一次写 C 的机会。
 *   写回后 C tile 经由 D-cache 写入内存，若后续 kc-block 还要累加，
 *   下一次 _LOAD_C 会从 cache 中命中（C tile 很小，通常驻留 L1）。
 */
#define _STORE_C(T, P) \
    __attribute__((always_inline)) static inline \
    void sme_store_c##T(uint32_t row, double *addr) { \
        __asm__ volatile("mov w12, %w[r]\n\t" \
                         "st1d za" #T "h.d[w12, 0], " #P ", [%[p]]" \
                         :: [r]"r"(row), [p]"r"(addr) : "w12", "memory"); }
_STORE_C(0,p0) _STORE_C(1,p3) _STORE_C(2,p4) _STORE_C(3,p5) _STORE_C(4,p6) _STORE_C(5,p7)
#undef _STORE_C
