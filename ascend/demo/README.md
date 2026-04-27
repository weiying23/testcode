# 昇腾 NPU 环境验证样例

这是一个最小化的鲲鹏 + 昇腾 910 环境验证样例，用于快速打通 PyTorch + torch_npu + CANN + HCCL 的完整链路。

## 目录结构

```
demo/
├── ascend_torch_npu_smoke.py  # 主验证脚本（单卡/多卡）
├── run_smoke.sh               # 一键运行脚本（单卡）
└── README.md                  # 本文档
```

## 核心概念

### CANN (Compute Architecture for Neural Networks)
- 华为昇腾异构计算架构，类似 NVIDIA 的 CUDA Toolkit
- 包含算子库、编译器、运行时、调试工具等
- 安装路径通常为 `/usr/local/Ascend/ascend-toolkit/`
- 需要通过 `set_env.sh` 设置环境变量（`LD_LIBRARY_PATH`、`PYTHONPATH` 等）

### torch_npu
- 华为提供的 PyTorch NPU 后端插件，类似 `torch.cuda`
- 将 PyTorch 算子调用转换为 CANN 算子执行
- 版本必须与 CANN 版本匹配（例如 CANN 8.0.RC1 对应 torch_npu 2.1.0.post3）
- 安装方式：`pip install torch_npu` 或从源码编译

### HCCL (Huawei Collective Communication Library)
- 华为集合通信库，类似 NVIDIA 的 NCCL
- 支持 `all_reduce`、`all_gather`、`broadcast` 等多卡通信原语
- 底层通过 RDMA 或 RoCE 实现高速卡间互联
- PyTorch 通过 `torch.distributed` 统一接口调用

## 环境要求

### 硬件
- 鲲鹏 CPU（ARM64 架构）
- 昇腾 910 NPU（单卡或多卡）

### 软件
- 操作系统：Ubuntu 20.04/22.04 或 openEuler
- 昇腾驱动：与 NPU 型号匹配
- CANN：8.0.RC1 或更高版本
- Python：3.8-3.10
- PyTorch：2.1.0 或更高版本
- torch_npu：与 CANN 版本匹配

### 环境变量检查
```bash
# 检查 CANN 是否正确安装
ls /usr/local/Ascend/ascend-toolkit/set_env.sh

# 加载 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查驱动和设备
npu-smi info

# 检查 Python 环境
python3 -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

## 使用方法

### 单卡验证（快速开始）

```bash
# 方式 1：使用一键脚本
bash demo/run_smoke.sh

# 方式 2：手动执行
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 demo/ascend_torch_npu_smoke.py --size 1024 --steps 10 --warmup 3
```

**预期输出：**
```
[ENV] torch=2.1.0
[ENV] torch_npu=2.1.0.post3
[ENV] npu_available=True
[ENV] visible_devices=0,1,2,3,4,5,6,7
[RANK 0] device=npu:0 world_size=1 dtype=float16 elapsed=0.0234s tflops=92.15 checksum=0.123456
[OK] torch=2.1.0, torch_npu=2.1.0.post3
```

### 8 卡 HCCL 验证

```bash
# 使用 torchrun 启动 8 个进程，每个进程绑定一张 NPU 卡
source /usr/local/Ascend/ascend-toolkit/set_env.sh
torchrun --nproc_per_node=8 demo/ascend_torch_npu_smoke.py \
    --distributed \
    --size 1024 \
    --steps 10 \
    --warmup 3
```

**预期输出（8 个进程会并行打印）：**
```
[RANK 0] device=npu:0 world_size=8 dtype=float16 elapsed=0.0234s tflops=92.15 checksum=0.123456
[RANK 1] device=npu:1 world_size=8 dtype=float16 elapsed=0.0235s tflops=91.87 checksum=0.123457
...
[RANK 7] device=npu:7 world_size=8 dtype=float16 elapsed=0.0236s tflops=91.56 checksum=0.123458
[RANK 0] hccl all_reduce sum=36.0, expected=36.0
[OK] torch=2.1.0, torch_npu=2.1.0.post3
```

**关键验证点：**
- `world_size=8`：确认 8 个进程都正确初始化
- `hccl all_reduce sum=36.0`：验证 HCCL 通信正确（1+2+3+4+5+6+7+8=36）
- 所有进程都打印 `[OK]`：无异常退出

## 参数说明

```bash
python3 demo/ascend_torch_npu_smoke.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--size` | 2048 | 矩阵大小（N×N） |
| `--warmup` | 5 | 预热迭代次数（跳过算子编译时间） |
| `--steps` | 20 | 正式计时迭代次数 |
| `--dtype` | float16 | 数据类型（float16/float32/bfloat16） |
| `--distributed` | False | 启用多卡 HCCL 验证 |

## 代码详解

### 关键步骤

#### 1. 导入 torch_npu
```python
import torch_npu  # 自动注册 NPU 设备和算子
```
- 导入时会加载 CANN 运行时库（`libascendcl.so`、`libge_runner.so` 等）
- 注册 `npu` 设备类型到 PyTorch
- 注册 CANN 算子实现（MatMul、Conv2d、ReLU 等）

#### 2. 检查设备可用性
```python
torch.npu.is_available()  # 调用 CANN 的 aclrtGetDeviceCount
```
- 返回 `False` 的常见原因：
  - 驱动未安装或版本不匹配
  - CANN 环境变量未设置
  - 容器内设备未映射（需要 `--device /dev/davinci*`）

#### 3. 设置当前设备
```python
torch.npu.set_device("npu:0")  # 类似 torch.cuda.set_device
```
- 单卡模式：固定使用 `npu:0`
- 多卡模式：根据 `LOCAL_RANK` 环境变量选择设备

#### 4. 初始化 HCCL
```python
dist.init_process_group(backend="hccl")
```
- 加载 HCCL 库（`libhccl.so`）
- 读取环境变量：`RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT`
- 建立卡间通信拓扑（通过 RDMA 或 RoCE）

#### 5. 执行算子
```python
result = torch.matmul(a, b)  # 下发到 NPU 执行
torch.npu.synchronize()      # 等待 NPU 完成
```
- `torch.matmul` 会调用 CANN 的 `MatMul` 算子
- 首次执行会触发算子编译（TBE 或 AIC 编译器）
- 编译结果会缓存到 `~/.cache/torch_npu/`

#### 6. HCCL 通信
```python
dist.all_reduce(tensor)  # 调用 HCCL 的 HcclAllReduce
```
- 所有卡的 `tensor` 求和并广播回每张卡
- 验证公式：`sum = world_size * (world_size + 1) / 2`
- 8 卡时：`1+2+3+4+5+6+7+8 = 36`

## 常见问题

### 1. `torch.npu.is_available()` 返回 `False`

**排查步骤：**
```bash
# 检查驱动
npu-smi info

# 检查 CANN 环境变量
echo $LD_LIBRARY_PATH | grep ascend
echo $PYTHONPATH | grep ascend

# 手动加载环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查设备文件权限
ls -l /dev/davinci*
ls -l /dev/davinci_manager
ls -l /dev/devmm_svm
```

### 2. `ImportError: libascendcl.so: cannot open shared object file`

**原因：** CANN 环境变量未设置

**解决：**
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 3. `torch_npu` 版本不匹配

**症状：** 运行时报错 `version mismatch` 或 `symbol not found`

**解决：**
```bash
# 查看 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# 安装匹配的 torch_npu
pip install torch_npu==<对应版本>
```

**版本对应关系：**
| CANN 版本 | torch_npu 版本 |
|-----------|----------------|
| 8.0.RC1   | 2.1.0.post3    |
| 8.0.RC2   | 2.1.0.post6    |
| 7.0.0     | 2.0.1          |

### 4. HCCL 初始化失败

**症状：** `RuntimeError: HCCL error in: init_process_group`

**排查步骤：**
```bash
# 检查网络连通性
ping <MASTER_ADDR>

# 检查端口是否被占用
netstat -tuln | grep <MASTER_PORT>

# 检查 HCCL 配置文件（如果使用）
cat /etc/hccl.conf

# 检查 RDMA 设备
ibstat
```

### 5. 容器内运行失败

**Docker 启动参数：**
```bash
docker run -it --rm \
    --device /dev/davinci0:/dev/davinci0 \
    --device /dev/davinci1:/dev/davinci1 \
    --device /dev/davinci2:/dev/davinci2 \
    --device /dev/davinci3:/dev/davinci3 \
    --device /dev/davinci4:/dev/davinci4 \
    --device /dev/davinci5:/dev/davinci5 \
    --device /dev/davinci6:/dev/davinci6 \
    --device /dev/davinci7:/dev/davinci7 \
    --device /dev/davinci_manager:/dev/davinci_manager \
    --device /dev/devmm_svm:/dev/devmm_svm \
    --device /dev/hisi_hdc:/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    <镜像名>
```

## 性能参考

### 昇腾 910B（单卡）

| 矩阵大小 | dtype | TFLOPS (理论峰值 256) |
|----------|-------|-----------------------|
| 1024×1024 | FP16 | ~90 |
| 2048×2048 | FP16 | ~180 |
| 4096×4096 | FP16 | ~220 |
| 8192×8192 | FP16 | ~240 |

**注意：** 实际性能受算子实现、显存带宽、编译优化等因素影响。

## 下一步

环境验证通过后，可以进行：

1. **模型训练/推理**
   - 将现有 PyTorch 代码中的 `.cuda()` 改为 `.npu()`
   - 使用 `torch.distributed` 进行多卡训练

2. **性能优化**
   - 使用 `torch_npu.npu.amp` 进行混合精度训练
   - 使用 `torch_npu.npu.utils.profiler` 进行性能分析
   - 调整算子 Tiling 参数（需要 AscendC 开发）

3. **算子开发**
   - 使用 AscendC 开发自定义算子
   - 使用 TBE (Tensor Boost Engine) 开发融合算子

## 参考资料

- [CANN 官方文档](https://www.hiascend.com/document)
- [torch_npu GitHub](https://github.com/Ascend/pytorch)
- [昇腾社区](https://www.hiascend.com/forum)
- [算子开发指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha001/operatordev/opdevg/opdevg_000001.html)
