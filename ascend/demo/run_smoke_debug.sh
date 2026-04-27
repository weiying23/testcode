#!/usr/bin/env bash
# 昇腾环境单卡验证脚本（调试版）

set -x  # 打印每条执行的命令，方便定位卡在哪里
set -e  # 遇到错误立即退出

echo "[DEBUG] Script started at $(date)"

# 获取脚本所在目录（兼容性更好的写法）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[DEBUG] SCRIPT_DIR=${SCRIPT_DIR}"

# 尝试加载 CANN 环境变量
echo "[DEBUG] Checking CANN environment..."
if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
  echo "[DEBUG] Loading CANN from ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
  source "${ASCEND_HOME_PATH}/set_env.sh"
elif [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
  echo "[DEBUG] Loading CANN from /usr/local/Ascend/ascend-toolkit/"
  source "/usr/local/Ascend/ascend-toolkit/set_env.sh"
else
  echo "[WARN] CANN set_env.sh not found, continuing without it..."
fi

# 检查 Python 和依赖
echo "[DEBUG] Checking Python environment..."
which python3
python3 --version

echo "[DEBUG] Checking torch and torch_npu..."
python3 -c "import torch; print(f'torch={torch.__version__}')" || echo "[ERROR] torch import failed"
python3 -c "import torch_npu; print(f'torch_npu={torch_npu.__version__}')" || echo "[ERROR] torch_npu import failed"

# 环境信息检查
echo "[DEBUG] Running environment check..."
python3 - <<'PY'
import os
import sys
try:
    import torch
    import torch_npu
    print(f"[ENV] torch={torch.__version__}")
    print(f"[ENV] torch_npu={getattr(torch_npu, '__version__', 'unknown')}")
    print(f"[ENV] npu_available={torch.npu.is_available()}")
    print(f"[ENV] npu_device_count={torch.npu.device_count() if torch.npu.is_available() else 0}")
    print(f"[ENV] visible_devices={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<unset>')}")
    if not torch.npu.is_available():
        print("[ERROR] NPU not available, check driver and CANN installation", file=sys.stderr)
        sys.exit(1)
except Exception as exc:
    print(f"[ENV-ERROR] {exc}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PY

if [[ $? -ne 0 ]]; then
    echo "[ERROR] Environment check failed, stopping here"
    exit 1
fi

# 运行主测试脚本
echo "[DEBUG] Running smoke test..."
python3 "${SCRIPT_DIR}/ascend_torch_npu_smoke.py" --size 1024 --steps 10 --warmup 3

echo "[INFO] =========================================="
echo "[INFO] Single-card smoke test finished!"
echo "[INFO] =========================================="
echo "[INFO] For 8-card HCCL validation, run:"
echo "  torchrun --nproc_per_node=8 ${SCRIPT_DIR}/ascend_torch_npu_smoke.py --distributed --size 1024 --steps 10 --warmup 3"
