#!/bin/bash
# s-step GMRES 参数选择指南
# 帮助用户根据问题特性选择合适的 n, s, m

echo "=============================================="
echo "s-step GMRES 参数选择指南"
echo "=============================================="
echo ""

echo "=== 参数速查表 ==="
echo ""
echo "参数 | 含义         | 建议值      | 影响"
echo "-----|-------------|------------|------------------"
echo " n   | 矩阵维度     | 完全平方数   | 条件数 O(n)"
echo " s   | s-step参数   | 2-3 (推荐3) | 通信效率 vs 稳定性"
echo " m   | 块数         | 10-50       | Krylov维度 vs 内存"
echo ""

echo "=== 组合建议 ==="
echo ""
echo "问题类型              | n      | s | m  | Krylov | 说明"
echo "----------------------|--------|---|----|----|------"
echo "小规模测试            | 100    | 3 | 10 | 30 | 快速验证"
echo "中等规模 (简单问题)   | 400    | 3 | 15 | 45 | 通常足够"
echo "中等规模 (困难问题)   | 400    | 3 | 30 | 90 | 各向异性"
echo "大规模 (简单问题)     | 1000   | 3 | 20 | 60 | 可能需重启"
echo "大规模 (困难问题)     | 1000   | 3 | 50 | 150| 多次重启"
echo ""

echo "=== 选择原则 ==="
echo ""
echo "1. n 的选择:"
echo "   - 应为完全平方数 (n = nx × nx)"
echo "   - nx = sqrt(n) 是网格边长"
echo "   - 条件数 ≈ O(n)，越大越难收敛"
echo ""

echo "2. s 的选择:"
echo "   - s=2: 最稳定，适合困难问题"
echo "   - s=3: 平衡选择，论文推荐"
echo "   - s=4-5: 通信效率高，但可能不稳定"
echo "   - s>5: 不推荐 (数值不稳定)"
echo ""

echo "3. m 的选择:"
echo "   - 简单问题: m ≈ 10-20"
echo "   - 困难问题: m ≈ 30-50"
echo "   - 经验公式: m × s ≈ 期望的 Krylov 维度"
echo "   - Krylov 维度不足 → 需要重启 (额外计算)"
echo ""

echo "=== Krylov 维度经验值 ==="
echo ""
echo "问题难度   | 条件数    | 需要 Krylov 维度"
echo "-----------|----------|------------------"
echo "简单       | <100     | 10-30"
echo "中等       | 100-1000 | 30-60"
echo "困难       | >1000    | 60-150+"
echo ""

echo "=== 实际测试 ==="
echo ""
echo "测试不同 m 对收敛的影响 (n=400, s=3, 五对角):"
echo ""

for m in 5 10 15; do
    echo "--- m=$m (Krylov维度=$((m*3))) ---"
    timeout 5 mpirun -np 1 ./sstep_gmres_paper 400 3 $m 0 2>&1 | grep -E "Block|Global|Converged|Not" | head -5
    echo ""
done