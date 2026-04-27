#!/usr/bin/env bash
# 昇腾环境诊断脚本
# 用途：检查驱动、CANN、HCCL、torch_npu 等所有关键组件

echo "=========================================="
echo "昇腾环境诊断工具"
echo "=========================================="
echo ""

# ========== 1. 系统信息 ==========
echo "[1] 系统信息"
echo "----------------------------------------"
echo "操作系统: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "内核版本: $(uname -r)"
echo "CPU架构: $(uname -m)"
echo "主机名: $(hostname)"
echo ""

# ========== 2. 昇腾驱动 ==========
echo "[2] 昇腾驱动"
echo "----------------------------------------"
if command -v npu-smi &> /dev/null; then
    echo "✓ npu-smi 已安装"
    echo ""
    echo "驱动版本:"
    npu-smi info -t board -i 0 | grep "Version" || echo "  无法获取版本信息"
    echo ""
    echo "NPU 设备列表:"
    npu-smi info -t board -l || npu-smi info
else
    echo "✗ npu-smi 未找到（驱动可能未安装）"
fi
echo ""

# ========== 3. 设备文件 ==========
echo "[3] NPU 设备文件"
echo "----------------------------------------"
if ls /dev/davinci* &> /dev/null; then
    echo "✓ NPU 设备文件存在:"
    ls -l /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc 2>/dev/null | head -20
else
    echo "✗ /dev/davinci* 不存在（驱动未加载或权限问题）"
fi
echo ""

# ========== 4. CANN 安装 ==========
echo "[4] CANN 安装"
echo "----------------------------------------"

# 检查常见安装路径
CANN_PATHS=(
    "/usr/local/Ascend/ascend-toolkit/latest"
    "/usr/local/Ascend/ascend-toolkit"
    "/usr/local/Ascend/nnae/latest"
    "/usr/local/Ascend/nnrt/latest"
    "${ASCEND_HOME_PATH}"
)

CANN_FOUND=0
for path in "${CANN_PATHS[@]}"; do
    if [[ -n "$path" && -d "$path" ]]; then
        echo "✓ CANN 安装路径: $path"
        CANN_FOUND=1
        
        # 读取版本信息
        if [[ -f "$path/version.cfg" ]]; then
            echo "  版本信息:"
            cat "$path/version.cfg" | grep -E "Version|version" | head -5
        fi
        
        # 检查关键库文件
        echo ""
        echo "  关键库文件:"
        [[ -f "$path/lib64/libascendcl.so" ]] && echo "    ✓ libascendcl.so (CANN 运行时)" || echo "    ✗ libascendcl.so"
        [[ -f "$path/lib64/libge_runner.so" ]] && echo "    ✓ libge_runner.so (图引擎)" || echo "    ✗ libge_runner.so"
        [[ -f "$path/lib64/libgraph.so" ]] && echo "    ✓ libgraph.so (图编译)" || echo "    ✗ libgraph.so"
        
        # 检查环境变量脚本
        echo ""
        echo "  环境变量脚本:"
        [[ -f "$path/set_env.sh" ]] && echo "    ✓ set_env.sh" || echo "    ✗ set_env.sh"
        
        break
    fi
done

if [[ $CANN_FOUND -eq 0 ]]; then
    echo "✗ CANN 未找到（请检查安装路径）"
fi
echo ""

# ========== 5. HCCL 库 ==========
echo "[5] HCCL 集合通信库"
echo "----------------------------------------"
HCCL_PATHS=(
    "/usr/local/Ascend/ascend-toolkit/latest/lib64/libhccl.so"
    "/usr/local/Ascend/nnae/latest/lib64/libhccl.so"
    "${ASCEND_HOME_PATH}/lib64/libhccl.so"
)

HCCL_FOUND=0
for hccl_path in "${HCCL_PATHS[@]}"; do
    if [[ -n "$hccl_path" && -f "$hccl_path" ]]; then
        echo "✓ HCCL 库: $hccl_path"
        HCCL_FOUND=1
        
        # 尝试获取版本信息
        if command -v strings &> /dev/null; then
            VERSION=$(strings "$hccl_path" | grep -E "HCCL.*[0-9]+\.[0-9]+" | head -1)
            [[ -n "$VERSION" ]] && echo "  版本: $VERSION"
        fi
        break
    fi
done

if [[ $HCCL_FOUND -eq 0 ]]; then
    echo "✗ HCCL 库未找到"
fi
echo ""

# ========== 6. 环境变量 ==========
echo "[6] 关键环境变量"
echo "----------------------------------------"
echo "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH:-<未设置>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<未设置>}" | head -c 200
echo "..."
echo "PYTHONPATH: ${PYTHONPATH:-<未设置>}" | head -c 200
echo "..."
echo "ASCEND_RT_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES:-<未设置>}"
echo "RANK: ${RANK:-<未设置>}"
echo "WORLD_SIZE: ${WORLD_SIZE:-<未设置>}"
echo "MASTER_ADDR: ${MASTER_ADDR:-<未设置>}"
echo "MASTER_PORT: ${MASTER_PORT:-<未设置>}"
echo ""

# ========== 7. Python 环境 ==========
echo "[7] Python 环境"
echo "----------------------------------------"
if command -v python3 &> /dev/null; then
    echo "✓ Python: $(python3 --version)"
    echo "  路径: $(which python3)"
    echo ""
    
    # 检查 torch
    echo "  检查 PyTorch:"
    python3 -c "import torch; print(f'    ✓ torch {torch.__version__}')" 2>/dev/null || echo "    ✗ torch 未安装"
    
    # 检查 torch_npu
    echo "  检查 torch_npu:"
    python3 -c "import torch_npu; print(f'    ✓ torch_npu {torch_npu.__version__}')" 2>/dev/null || echo "    ✗ torch_npu 未安装"
    
    # 检查 NPU 可用性
    echo "  检查 NPU 可用性:"
    python3 -c "import torch; import torch_npu; print(f'    torch.npu.is_available() = {torch.npu.is_available()}')" 2>/dev/null || echo "    ✗ 无法检查（导入失败）"
    
    # 检查设备数量
    echo "  检查 NPU 设备数量:"
    python3 -c "import torch; import torch_npu; print(f'    torch.npu.device_count() = {torch.npu.device_count()}')" 2>/dev/null || echo "    ✗ 无法检查"
    
else
    echo "✗ python3 未找到"
fi
echo ""

# ========== 8. 网络和 RDMA（多卡通信） ==========
echo "[8] 网络和 RDMA（多卡通信）"
echo "----------------------------------------"
if command -v ibstat &> /dev/null; then
    echo "✓ RDMA 工具已安装"
    echo ""
    echo "InfiniBand 设备:"
    ibstat 2>/dev/null | grep -E "CA type|Port|State|Rate" | head -20 || echo "  无设备或无权限"
else
    echo "✗ ibstat 未找到（RDMA 工具未安装，多卡通信可能受限）"
fi
echo ""

# ========== 9. 常见问题诊断 ==========
echo "[9] 常见问题诊断"
echo "----------------------------------------"

# 检查驱动和设备
if ! command -v npu-smi &> /dev/null; then
    echo "⚠ 驱动未安装或 npu-smi 不在 PATH 中"
    echo "  解决: 安装昇腾驱动固件包"
fi

if ! ls /dev/davinci* &> /dev/null; then
    echo "⚠ NPU 设备文件不存在"
    echo "  解决: 检查驱动是否正确加载 (lsmod | grep drv_davinci)"
fi

# 检查 CANN
if [[ $CANN_FOUND -eq 0 ]]; then
    echo "⚠ CANN 未安装或路径不标准"
    echo "  解决: 安装 CANN toolkit 并设置 ASCEND_HOME_PATH"
fi

# 检查环境变量
if [[ -z "${LD_LIBRARY_PATH}" ]] || [[ ! "${LD_LIBRARY_PATH}" =~ "Ascend" ]]; then
    echo "⚠ LD_LIBRARY_PATH 未包含 CANN 路径"
    echo "  解决: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
fi

# 检查 torch_npu
if ! python3 -c "import torch_npu" 2>/dev/null; then
    echo "⚠ torch_npu 未安装或版本不匹配"
    echo "  解决: pip install torch_npu（确保版本与 CANN 匹配）"
fi

# 检查 NPU 可用性
if ! python3 -c "import torch; import torch_npu; assert torch.npu.is_available()" 2>/dev/null; then
    echo "⚠ torch.npu.is_available() 返回 False"
    echo "  解决: 检查驱动、CANN 环境变量、设备权限"
fi

echo ""
echo "=========================================="
echo "诊断完成"
echo "=========================================="
echo ""
echo "提示: 如果发现问题，请按以下顺序排查:"
echo "  1. 安装驱动 → 检查 npu-smi info 是否正常"
echo "  2. 安装 CANN → 检查 /usr/local/Ascend/ascend-toolkit/"
echo "  3. 加载环境变量 → source set_env.sh"
echo "  4. 安装 torch_npu → pip install torch_npu"
echo "  5. 验证环境 → python3 -c 'import torch_npu; print(torch.npu.is_available())'"
