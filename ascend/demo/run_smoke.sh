#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
  source "${ASCEND_HOME_PATH}/set_env.sh"
elif [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
  source "/usr/local/Ascend/ascend-toolkit/set_env.sh"
fi

python3 - <<'PY'
import os
try:
    import torch
    import torch_npu
    print(f"[ENV] torch={torch.__version__}")
    print(f"[ENV] torch_npu={getattr(torch_npu, '__version__', 'unknown')}")
    print(f"[ENV] npu_available={torch.npu.is_available()}")
    print(f"[ENV] visible_devices={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<unset>')}")
except Exception as exc:
    print(f"[ENV-ERROR] {exc}")
    raise
PY

python3 "${SCRIPT_DIR}/ascend_torch_npu_smoke.py" --size 1024 --steps 10 --warmup 3

echo "[INFO] single-card smoke finished"
echo "[INFO] for 8-card validation, run:"
echo "torchrun --nproc_per_node=8 ${SCRIPT_DIR}/ascend_torch_npu_smoke.py --distributed --size 1024 --steps 10 --warmup 3"
