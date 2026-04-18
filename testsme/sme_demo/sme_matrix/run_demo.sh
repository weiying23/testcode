#!/bin/sh
# run_demo.sh — 最小 SME 矩阵乘 demo 的编译与运行脚本
#
# 功能：编译并运行 `sme_demo.cpp`
# 这个 demo 只做一件事：
#   用 SME 的 `fmopa` 指令计算一个最简单的 8×8 双精度矩阵乘法。
#
# 为什么选 8×8：
#   Apple M4 上 SME 的 SVL = 512 bits = 64 bytes
#   double 每个 8 字节，因此一行可放 8 个 double
#   一个 ZA tile 就是 8×8 double，正好适合做最小 demo
#
# 编译参数说明：
#   -O0
#     不做高级优化，保证 demo 行为最直观，便于看汇编和调试。
#
#   -march=armv9-a+sve+sve2+sme+sme-f64f64
#     +sme        : 启用 SME 指令集
#     +sme-f64f64 : 启用双精度矩阵外积 `fmopa za?.d`
#     +sve/+sve2  : 让汇编器和头文件接受相关指令/寄存器语法
#
# 运行结果包含两个 case：
#   1. A × I = A
#   2. SME 结果 vs 标量参考实现

set -e

CXX=/usr/bin/clang++
FLAGS="-O0 -march=armv9-a+sve+sve2+sme+sme-f64f64"
SRC="sme_demo.cpp"
OUT="sme_demo"

echo "=== 编译最小 SME demo ==="
$CXX $FLAGS "$SRC" -o "$OUT"
chmod 755 "$OUT"

echo ""
echo "=== 运行 SME demo ==="
./"$OUT"
