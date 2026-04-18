#!/bin/sh
# run.sh — SME GEMM 项目构建 & 测试脚本
#
# 用法：
#   bash run.sh           # 完整构建 + 运行 test（SME 硬件验证）+ 运行 benchmark
#   bash run.sh build     # 仅编译，不运行
#   bash run.sh test      # 仅运行 ./test（SME 硬件状态验证 + 小矩阵正确性）
#   bash run.sh bench     # 仅运行 ./benchmark（正确性 + 多维度性能对比）
#
# ─────────────────────────────────────────────────────────────────
# 编译策略（分离编译）
#
#   本项目必须对不同文件使用不同优化级别：
#
#   packing.cpp  →  -O2 -march=armv9-a
#     -O2：让编译器把内层 memcpy 循环向量化为 NEON 指令，
#           比 -O0 标量循环快 3~5 倍。
#     不带 +sve：macOS 不支持用户态非流式 SVE，
#                带上会在向量化代码里插入 cntd → SIGILL。
#
#   matrix_multiply.cpp / benchmark.cpp  →  -O0 -march=armv9-a+sve+sve2+sme+sme-f64f64
#     -O0：防止编译器重排/重用寄存器，破坏 SME 内联汇编对
#           za6/z1/w12 等寄存器的精确控制。
#     +sme：启用 SME 指令集（smstart/smstop/fmopa/ld1d za…）。
#     +sme-f64f64：启用双精度外积（fmopa za.d），
#                  即 za[i,j] += A[i] * B[j]（64 次 FMA/条指令）。
#     +sve +sve2：编译器识别 SVE 向量类型需要此 flag，
#                  运行时实际 SME 的流式 SVE 由 smstart 控制。
#
# ─────────────────────────────────────────────────────────────────
# benchmark 测试维度说明
#
#   正确性测试（对比 naive 标量和 Apple Accelerate cblas_dgemm）：
#     - 方阵：8³、16³、64³（覆盖 1 tile、2 tile、8 tile 等规模）
#     - 非方阵：9×11×7、17×19×13、37×73×53（覆盖所有尾边界路径）
#     - 宽/高矩阵：32×64×128、15×17×33、128×64×32
#     - 其他：100×150×200、256×192×128
#     误差阈值：1e-9（双精度相对误差远小于此值时视为正确）
#
#   性能测试（时间取多次运行最小值，GFLOPS = 2MNK / time）：
#     方阵（测试 cache 命中率随规模变化）：
#       64³   → 工作集 ~0.1 MB，完全驻留 L1
#       128³  → 工作集 ~0.4 MB，完全驻留 L2
#       256³  → 工作集 ~1.5 MB，驻留 L2
#       512³  → 工作集 ~12 MB，溢出 P 核 L2，压 L3
#       1024³ → 工作集 ~48 MB，L3 压力大
#       2048³ → 工作集 ~192 MB，主要测内存带宽
#       4096³ → 工作集 ~768 MB，DRAM 带宽瓶颈
#       8192³ → 工作集 ~3 GB，最大规模，测极限吞吐
#     非方阵：
#       4096×2048×8192 → 宽 K，测 k 方向分块效率
#       8192×1024×4096 → 高 M，测多线程 ic 分片效率
#     非 8 倍数：
#       100³、200×250×300 → 验证谓词化边界处理的性能代价
# ─────────────────────────────────────────────────────────────────

set -e

CXX=/usr/bin/clang++

# SME 编译 flags（用于 matrix_multiply.cpp 和 benchmark.cpp）
SME_FLAGS="-march=armv9-a+sve+sve2+sme+sme-f64f64"

# Packing 编译 flags（纯 NEON，不带 SVE，避免 SIGILL）
PACK_FLAGS="-march=armv9-a"

# Apple Accelerate 框架 flags（用于 benchmark 对比 cblas_dgemm）
ACCEL_FLAGS="-DACCELERATE_NEW_LAPACK -Wno-deprecated-declarations -framework Accelerate"

# ── 构建函数 ──────────────────────────────────────────────────────
do_build() {
    echo "=== 编译 packing.o  (-O2, NEON only, no SVE) ==="
    $CXX -c -O2 $PACK_FLAGS packing.cpp -o packing.o

    echo "=== 编译 matrix_multiply.o  (-O0, SME+SVE2) ==="
    $CXX -c -O0 $SME_FLAGS matrix_multiply.cpp -o matrix_multiply.o

    echo "=== 编译 matrix.o  (-O0, SME+SVE2) ==="
    $CXX -c -O0 $SME_FLAGS matrix.cpp -o matrix.o

    echo "=== 链接 test ==="
    $CXX matrix.o matrix_multiply.o packing.o -o test
    chmod 755 test

    echo "=== 编译 benchmark.o  (-O0, SME+SVE2 + Accelerate) ==="
    $CXX -c -O0 $SME_FLAGS $ACCEL_FLAGS benchmark.cpp -o benchmark_main.o

    echo "=== 链接 benchmark ==="
    $CXX matrix_multiply.o packing.o benchmark_main.o $ACCEL_FLAGS -o benchmark
    chmod 755 benchmark

    echo ""
    echo "构建完成。运行：./test 或 ./benchmark"
    echo ""
}

# ── 入口 ─────────────────────────────────────────────────────────
case "${1:-all}" in
    build)
        do_build
        ;;
    test)
        echo "=== 运行 ./test（SME 硬件验证 + 小矩阵正确性）==="
        ./test
        ;;
    bench)
        echo "=== 运行 ./benchmark（完整正确性 + 多维度性能对比）==="
        ./benchmark
        ;;
    all|*)
        do_build
        echo "=== 运行 ./test ==="
        ./test
        echo "=== 运行 ./benchmark ==="
        ./benchmark
        ;;
esac
