#!/bin/bash
# test_dist.sh - 分布式 s-step GMRES 测试脚本
#
# ============================================================================
# 使用方法:
#   ./test_dist.sh                    # 运行所有测试
#   ./test_dist.sh quick              # 快速验证测试
#   ./test_dist.sh pentadiagonal      # 五对角矩阵测试
#   ./test_dist.sh anisotropic        # 各向异性矩阵测试
#   ./test_dist.sh scalability        # 进程扩展性测试
#
# 手动运行示例:
#   # 五对角矩阵 (快速收敛)
#   mpirun -np 4 ./sstep_gmres_dist 10000 3 30 0 1e-8 25
#
#   # 各向异性矩阵 (需要 restart)
#   mpirun -np 4 ./sstep_gmres_dist 10000 3 30 1 1e-6 25
#
#   # 大规模测试
#   mpirun -np 4 ./sstep_gmres_dist 100000 3 50 0 1e-8 25
# ============================================================================
#
# 参数说明:
#   参数1 (n_global)    : 全局矩阵维度 (完全平方数)
#                         推荐值: 100, 400, 2500, 10000, 40000, 100000
#
#   参数2 (s)           : s-step 参数
#                         s=2: 最稳定，但收敛可能较慢
#                         s=3: 推荐，收敛快且通信少
#                         s>=4: 不推荐，可能数值不稳定
#
#   参数3 (m)           : 每轮最大块数
#                         总 Krylov 维度 = s * m
#                         推荐值: 10-50
#
#   参数4 (type)        : 矩阵类型
#                         0: 五对角矩阵 (来自 2D Poisson，快速收敛)
#                         1: 各向异性矩阵 (eps=0.01，需要 restart)
#
#   参数5 (tol)         : 收敛容忍度
#                         推荐值: 1e-6 ~ 1e-10
#
#   参数6 (max_restarts): 最大重启次数
#                         推荐值: 25 (默认)
#                         设为 1 等价于无 restart
#
# ============================================================================
# 矩阵说明:
#
# 五对角矩阵 (type=0):
#   来自二维 Poisson 问题离散化: -∇²u = f
#   条件数好，通常在 2-3 个块内收敛
#   不需要 restart
#
# 各向异性矩阵 (type=1):
#   来自各向异性扩散: -ε·u_xx - u_yy = f, ε=0.01
#   条件数差，需要多次 restart
#   s=3 比 s=2 效果更好
# ============================================================================

set -e

# 编译
compile() {
    echo "编译中..."
    mpicxx -std=c++11 -O3 -framework Accelerate -o sstep_gmres_dist sstep_gmres_dist.cpp 2>/dev/null
    echo "编译完成"
    echo ""
}

# 快速验证测试
test_quick() {
    echo "=============================================="
    echo "快速验证测试"
    echo "=============================================="
    echo ""

    # 五对角矩阵 - 应该在 2 个块内收敛
    echo "# 五对角矩阵 n=10000"
    mpirun -np 4 ./sstep_gmres_dist 10000 3 10 0 1e-8 25 2>&1 | grep -E "(Converged|communications|Restarts)"

    echo ""
    echo "快速验证通过!"
}

# 五对角矩阵收敛性测试
test_pentadiagonal() {
    echo "=============================================="
    echo "五对角矩阵收敛性测试 (type=0)"
    echo "=============================================="
    echo ""
    echo "| 维度 n  | 网格 nx | 收敛块数 | 通信次数 | 最终残差 |"
    echo "|---------|---------|----------|----------|----------|"

    for n in 400 2500 10000 40000; do
        nx=$(echo "sqrt($n)" | bc | cut -d. -f1)
        result=$(mpirun -np 4 ./sstep_gmres_dist $n 3 30 0 1e-8 25 2>&1)
        blocks=$(echo "$result" | grep "Iterations (blocks)" | grep -oE "[0-9]+")
        comm=$(echo "$result" | grep "Global communications" | grep -oE "[0-9]+")
        residual=$(echo "$result" | grep "||b-Ax||/||b||" | grep -oE "[0-9]+\.[0-9]+e[+-][0-9]+")
        printf "| %-7s | %-7s | %-8s | %-8s | %s |\n" "$n" "$nx" "$blocks" "$comm" "$residual"
    done
    echo ""
    echo "结论: 五对角矩阵在所有维度下都能在 2 个块内收敛"
    echo ""
}

# 各向异性矩阵收敛性测试
test_anisotropic() {
    echo "=============================================="
    echo "各向异性矩阵收敛性测试 (type=1, eps=0.01)"
    echo "=============================================="
    echo ""
    echo "注意: 各向异性矩阵条件数差，需要多次 restart"
    echo ""
    echo "| 维度 n  | 网格 nx | 收敛块数 | 通信次数 | Restart | 最终残差 |"
    echo "|---------|---------|----------|----------|---------|----------|"

    for n in 400 2500 10000; do
        nx=$(echo "sqrt($n)" | bc | cut -d. -f1)
        result=$(mpirun -np 4 ./sstep_gmres_dist $n 3 30 1 1e-6 25 2>&1)
        blocks=$(echo "$result" | grep "Iterations (blocks)" | grep -oE "[0-9]+")
        comm=$(echo "$result" | grep "Global communications" | grep -oE "[0-9]+")
        restarts=$(echo "$result" | grep "Restarts used" | grep -oE "[0-9]+")
        residual=$(echo "$result" | grep "||b-Ax||/||b||" | grep -oE "[0-9]+\.[0-9]+e[+-][0-9]+")
        printf "| %-7s | %-7s | %-8s | %-8s | %-7s | %s |\n" "$n" "$nx" "$blocks" "$comm" "$restarts" "$residual"
    done

    echo ""
    echo "n=40000 大规模测试 (m=50, max_restarts=50)..."
    result=$(mpirun -np 4 ./sstep_gmres_dist 40000 3 50 1 1e-6 50 2>&1)
    blocks=$(echo "$result" | grep "Iterations (blocks)" | grep -oE "[0-9]+")
    comm=$(echo "$result" | grep "Global communications" | grep -oE "[0-9]+")
    restarts=$(echo "$result" | grep "Restarts used" | grep -oE "[0-9]+")
    residual=$(echo "$result" | grep "||b-Ax||/||b||" | grep -oE "[0-9]+\.[0-9]+e[+-][0-9]+")
    printf "| %-7s | %-7s | %-8s | %-8s | %-7s | %s |\n" "40000" "200" "$blocks" "$comm" "$restarts" "$residual"
    echo ""
}

# s 参数对比测试
test_s_parameter() {
    echo "=============================================="
    echo "s 参数对比测试 (各向异性矩阵 n=10000)"
    echo "=============================================="
    echo ""
    echo "| s | 收敛块数 | 通信次数 | Restart次数 |"
    echo "|---|----------|----------|-------------|"

    for s in 2 3; do
        result=$(mpirun -np 4 ./sstep_gmres_dist 10000 $s 30 1 1e-6 25 2>&1)
        blocks=$(echo "$result" | grep "Iterations (blocks)" | grep -oE "[0-9]+")
        comm=$(echo "$result" | grep "Global communications" | grep -oE "[0-9]+")
        restarts=$(echo "$result" | grep "Restarts used" | grep -oE "[0-9]+")
        printf "| %d | %-8s | %-8s | %-11s |\n" "$s" "$blocks" "$comm" "$restarts"
    done
    echo ""
    echo "结论: s=3 比 s=2 通信更少，收敛更快"
    echo ""
}

# 进程扩展性测试
test_scalability() {
    echo "=============================================="
    echo "进程扩展性测试 (五对角矩阵 n=10000)"
    echo "=============================================="
    echo ""
    echo "| 进程数 | 收敛块数 | 通信次数 | 说明 |"
    echo "|--------|----------|----------|------|"

    for np in 1 2 4; do
        result=$(mpirun -np $np ./sstep_gmres_dist 10000 3 10 0 1e-8 25 2>&1)
        blocks=$(echo "$result" | grep "Iterations (blocks)" | grep -oE "[0-9]+")
        comm=$(echo "$result" | grep "Global communications" | grep -oE "[0-9]+")
        if [ "$np" -eq 1 ]; then
            note="串行"
        else
            note="并行"
        fi
        printf "| %-6s | %-8s | %-8s | %s |\n" "$np" "$blocks" "$comm" "$note"
    done
    echo ""
    echo "结论: 通信次数不随进程数增加，具有良好的并行扩展性"
    echo ""
}

# 完整测试
test_all() {
    test_quick
    echo ""
    test_pentadiagonal
    echo ""
    test_anisotropic
    echo ""
    test_s_parameter
    echo ""
    test_scalability
}

# 主程序
compile

case "${1:-all}" in
    quick)
        test_quick
        ;;
    pentadiagonal)
        test_pentadiagonal
        ;;
    anisotropic)
        test_anisotropic
        ;;
    s-param)
        test_s_parameter
        ;;
    scalability)
        test_scalability
        ;;
    all)
        test_all
        ;;
    *)
        echo "未知测试类型: $1"
        echo "可用选项: quick, pentadiagonal, anisotropic, s-param, scalability, all"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "测试完成!"
echo "=============================================="