"""
昇腾 NPU 环境验证脚本 (torch_npu + HCCL)

功能：
1. 检查 torch_npu 是否正确安装并能识别 NPU 设备
2. 在 NPU 上执行基础算子 (matmul) 并测量性能
3. 可选：通过 HCCL 验证多卡通信 (all_reduce)

关键概念：
- torch_npu: 华为提供的 PyTorch NPU 后端插件，底层调用 CANN 算子库
- CANN: 昇腾异构计算架构 (Compute Architecture for Neural Networks)
- HCCL: 华为集合通信库 (Huawei Collective Communication Library)，类似 NVIDIA NCCL
"""

import argparse
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Ascend torch_npu smoke case")
    parser.add_argument("--size", type=int, default=2048, help="matrix size")
    parser.add_argument("--warmup", type=int, default=5, help="warmup steps")
    parser.add_argument("--steps", type=int, default=20, help="timed steps")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="tensor dtype",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="enable 8-card HCCL validation when launched by torchrun",
    )
    return parser.parse_args()


def resolve_dtype(torch_module, dtype_name):
    mapping = {
        "float16": torch_module.float16,
        "float32": torch_module.float32,
        "bfloat16": torch_module.bfloat16,
    }
    return mapping[dtype_name]


def tflops(size, steps, seconds):
    ops = 2.0 * size * size * size * steps
    return ops / seconds / 1e12


def main():
    args = parse_args()

    # ========== 步骤 1: 导入 torch 和 torch_npu ==========
    # torch_npu 是华为提供的 PyTorch NPU 后端插件
    # 导入后会自动注册 NPU 设备类型和算子实现
    try:
        import torch
    except ImportError as exc:
        print(f"[ERROR] torch import failed: {exc}", file=sys.stderr)
        print("[HINT] 先安装 PyTorch。", file=sys.stderr)
        return 1

    try:
        import torch_npu  # 这一步会加载 CANN 运行时库 (libascendcl.so 等)
    except ImportError as exc:
        print(f"[ERROR] torch_npu import failed: {exc}", file=sys.stderr)
        print("[HINT] 先安装与当前 CANN 匹配的 torch_npu。", file=sys.stderr)
        return 1

    # ========== 步骤 2: 检查 NPU 设备可用性 ==========
    # torch.npu.is_available() 会调用 CANN 的 aclrtGetDeviceCount 等 API
    # 如果返回 False，可能是：
    # - 驱动未安装或版本不匹配
    # - CANN 环境变量未设置 (LD_LIBRARY_PATH, ASCEND_HOME_PATH 等)
    # - 容器内设备未正确映射 (需要 --device /dev/davinci0:/dev/davinci0 等)
    if not torch.npu.is_available():
        print("[ERROR] torch.npu.is_available() == False", file=sys.stderr)
        print("[HINT] 检查驱动、CANN 环境变量、容器设备映射。", file=sys.stderr)
        return 1

    # ========== 步骤 3: 设置当前进程使用的 NPU 设备 ==========
    # 单卡模式：使用 npu:0
    # 多卡模式：torchrun 会为每个进程设置 LOCAL_RANK 环境变量，对应不同的 NPU 卡
    use_distributed = args.distributed or "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"npu:{local_rank if use_distributed else 0}"
    torch.npu.set_device(device)  # 类似 torch.cuda.set_device，设置当前默认设备

    # ========== 步骤 4: 初始化分布式通信 (可选) ==========
    # HCCL 是华为的集合通信库，类似 NVIDIA 的 NCCL
    # PyTorch 通过 torch.distributed 统一接口调用 HCCL
    rank = 0
    world_size = 1
    dist = None
    if use_distributed:
        import torch.distributed as dist

        # backend="hccl" 会加载 HCCL 库 (libhccl.so)
        # HCCL 底层通过 RDMA 或 RoCE 实现卡间高速通信
        # 初始化时会读取环境变量：
        # - RANK: 全局进程编号
        # - WORLD_SIZE: 总进程数
        # - MASTER_ADDR, MASTER_PORT: 主节点地址和端口
        dist.init_process_group(backend="hccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    # ========== 步骤 5: 创建测试数据并执行算子 ==========
    # torch.randn(..., device=device) 会在 NPU 上分配显存并初始化随机数
    # 底层调用 CANN 的 aclrtMalloc 和随机数生成算子
    dtype = resolve_dtype(torch, args.dtype)
    a = torch.randn(args.size, args.size, device=device, dtype=dtype)
    b = torch.randn(args.size, args.size, device=device, dtype=dtype)

    # Warmup: 首次执行会触发算子编译和缓存，跳过这部分时间
    for _ in range(args.warmup):
        _ = torch.matmul(a, b)  # 调用 CANN 的 MatMul 算子
    torch.npu.synchronize()  # 等待 NPU 上所有操作完成，类似 torch.cuda.synchronize()

    # 正式计时：测量 matmul 性能
    start = time.perf_counter()
    result = None
    for _ in range(args.steps):
        result = torch.matmul(a, b)  # torch_npu 会将此操作下发到 NPU 执行
    torch.npu.synchronize()  # 确保所有计算完成后再停止计时
    elapsed = time.perf_counter() - start

    # ========== 步骤 6: 打印性能和校验结果 ==========
    checksum = float(result.float().mean().item())  # 将结果拷贝回 CPU 并计算均值
    perf = tflops(args.size, args.steps, elapsed)
    print(
        f"[RANK {rank}] device={device} world_size={world_size} "
        f"dtype={args.dtype} elapsed={elapsed:.4f}s tflops={perf:.2f} checksum={checksum:.6f}",
        flush=True,
    )

    # ========== 步骤 7: HCCL 通信验证 (多卡模式) ==========
    if use_distributed:
        # 每个进程创建一个包含自己 rank+1 的 tensor
        verify = torch.tensor([rank + 1.0], device=device)
        
        # all_reduce 会调用 HCCL 的 HcclAllReduce API
        # 默认操作是 SUM，所有卡的数据会求和并广播回每张卡
        # 例如 8 卡时：1+2+3+4+5+6+7+8 = 36
        dist.all_reduce(verify)
        
        expected = world_size * (world_size + 1) / 2.0  # 等差数列求和公式
        if rank == 0:
            print(
                f"[RANK 0] hccl all_reduce sum={verify.item():.1f}, expected={expected:.1f}",
                flush=True,
            )
        
        # 校验 HCCL 通信是否正确
        if abs(verify.item() - expected) > 1e-5:
            raise RuntimeError("HCCL all_reduce 校验失败")
        
        dist.barrier()  # 同步所有进程
        dist.destroy_process_group()  # 清理 HCCL 资源

    print(
        f"[OK] torch={torch.__version__}, torch_npu={getattr(torch_npu, '__version__', 'unknown')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
