#!/bin/bash
# ============================================================================
# MTE vs SDMA 跨卡带宽对比测试运行脚本
# ============================================================================
#
# 功能：
#   在同节点的两个 NPU 上并行运行 comm_test 程序，测试 MTE 和 SDMA 带宽。
#
# 使用方式：
#   bash run.sh <npu0> <npu1>     # 两个参数形式
#   bash run.sh <npu0>,<npu1>     # 逗号分隔形式
#
# 示例：
#   bash run.sh 0 1               # 使用 NPU 0 和 NPU 1
#   bash run.sh 4 5               # 使用 NPU 4 和 NPU 5
#   bash run.sh 0,1               # 逗号分隔，等同于 bash run.sh 0 1
#
# 参数说明：
#   npu0: 第一个 NPU 的设备 ID（Rank 0 运行在此 NPU）
#   npu1: 第二个 NPU 的设备 ID（Rank 1 运行在此 NPU）
#
# 输出：
#   在屏幕上打印 MTE 和 SDMA 的带宽对比结果
# ============================================================================

set -e

# ========== 解析 NPU ID 参数 ==========
# 支持两种格式：
#   1. bash run.sh 0 1      (两个参数)
#   2. bash run.sh 0,1      (逗号分隔)
if [[ "$1" == *","* ]]; then
    NPU0=$(echo "$1" | cut -d',' -f1)
    NPU1=$(echo "$1" | cut -d',' -f2)
else
    NPU0=${1:-0}   # 默认 NPU 0
    NPU1=${2:-1}   # 默认 NPU 1
fi

# ========== 配置参数 ==========
PORT="tcp://127.0.0.1:8899"  # SHMEM 通信端口（两个进程必须使用相同端口）
BIN="../../build/bin/comm_test"  # 编译后的二进制路径

# ========== 检查二进制文件 ==========
if [ ! -f "$BIN" ]; then
    echo "Error: $BIN not found. Build the project first."
    echo "Run: bash scripts/build.sh -examples"
    exit 1
fi

# ========== 清理残留进程 ==========
# 杀掉可能残留的 comm_test 进程
pkill -f "comm_test" 2>/dev/null || true
sleep 1

# ========== 输出测试配置 ==========
echo "=========================================="
echo "MTE vs SDMA Cross-Card Bandwidth Test"
echo "=========================================="
echo "NPU Pair: ${NPU0} <-> ${NPU1}"
echo "Port: ${PORT}"
echo "=========================================="

# ========== 并行启动两个进程 ==========
# 进程 0 (Rank 0):
#   参数: 2 0 $PORT $NPU0 $NPU1
#   含义: n_pes=2, pe_id=0, ipport, device_id=NPU0, peer_device=NPU1
#
# 进程 1 (Rank 1):
#   参数: 2 1 $PORT $NPU1 $NPU0
#   含义: n_pes=2, pe_id=1, ipport, device_id=NPU1, peer_device=NPU0
#
# 注意：两个进程必须同时启动，因为 SHMEM 初始化需要双方都在线

$BIN 2 0 $PORT $NPU0 $NPU1 &
$BIN 2 1 $PORT $NPU1 $NPU0 &

# 等待两个进程都完成
wait

echo ""
echo "[SUCCESS] Test completed"