#!/bin/bash
# 诊断残余停滞问题
# 对比Givens估计残余 vs 真实残余

echo "=============================================="
echo "残余停滞诊断分析"
echo "=============================================="
echo ""

echo "=== 现象分析 ==="
echo ""
echo "测试结果 (n=100000):"
echo "  Givens估计残余: ~3.5e-10"
echo "  真实残余 ||b-Ax||: ~4e-9"
echo "  差距: 约10倍"
echo ""

echo "=== 原因分析 ==="
echo ""
echo "1. 功率基向量不正交:"
echo "   V_k = [v, Av, A²v, ..., A^(s-1)v]"
echo "   这些向量本身不正交，需要通过W矩阵处理"
echo ""

echo "2. CGS正交化误差累积:"
echo "   Classical Gram-Schmidt (CGS) 会累积正交化误差"
echo "   每个块迭代都会引入小的正交性损失"
echo "   多个块后，误差显著累积"
echo ""

echo "3. W矩阵条件数问题:"
echo "   功率基的Gram矩阵W会随迭代变得更病态"
echo "   W的数值精度下降 → 正交化精度下降"
echo ""

echo "=== 验证测试 ==="
echo ""

echo "--- 测试1: 不同规模问题的正交性损失 ---"
for n in 100 1000 10000 100000; do
    echo "n=$n:"
    timeout 30 mpirun -np 1 ./sstep_gmres_paper $n 3 30 0 2>&1 | grep -E "residual=|b-Ax" | tail -3
    echo ""
done

echo "--- 测试2: s参数对正交性的影响 ---"
for s in 2 3; do
    echo "s=$s (n=100000):"
    timeout 30 mpirun -np 1 ./sstep_gmres_paper 100000 $s 30 0 2>&1 | grep -E "residual=|b-Ax" | tail -3
    echo ""
done

echo "=== 改进方案 ==="
echo ""
echo "1. 使用改进正交化:"
echo "   - Reorthogonalization (重复CGS)"
echo "   - MGS (Modified Gram-Schmidt) 替代 CGS"
echo ""

echo "2. 使用更好的基:"
echo "   - Chebyshev多项式基 (比功率基更稳定)"
echo "   - Newton多项式基"
echo ""

echo "3. 重启策略:"
echo "   - GMRES(m) 重启: 定期重新开始，重置正交性"
echo "   - 但每次重启丢失之前的子空间信息"
echo ""

echo "4. 更好的预处理:"
echo "   - ILU(k) 带更多填充"
echo "   - 代数多重网格 (AMG)"
echo "   - 这些可以减小条件数，减少迭代次数"