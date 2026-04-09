# s-step GMRES Restart 机制设计

## 概述

为分布式 s-step GMRES 求解器增加 restart 机制，解决以下问题：
- **控制内存使用**：限制 Krylov 子空间维度，避免存储过多向量
- **改善收敛性**：对于收敛慢的问题（如各向异性矩阵），通过 restart 避免停滞

## 需求

| 需求项 | 决定 |
|--------|------|
| 触发条件 | 固定周期 restart，每 m 个块后自动 restart |
| 解的处理 | 保留当前解 x，残差 r = b - Ax 重新开始 |
| 最大次数 | 用户指定参数 `max_restarts`，默认 25 |
| 输出模式 | 汇总模式，最后汇总所有 restart 轮次信息 |

## 设计细节

### 1. 参数结构体扩展

```cpp
struct GMRESParams {
    int s;              // s-step 参数 (2-3)
    int m;              // 每轮最大块数
    double tol;         // 收敛容忍度
    int max_restarts;   // 最大 restart 次数 (新增，默认 25)

    GMRESParams(int s_val = 3, int m_val = 10, double tol_val = 1e-8, int max_rst = 25)
        : s(s_val), m(m_val), tol(tol_val), max_restarts(max_rst) {}
};
```

### 2. 结果结构体扩展

```cpp
struct GMRESResult {
    bool converged;           // 是否收敛
    double final_residual;    // 最终相对残差
    int iterations;           // 总迭代块数（所有 restart 轮次累计）
    int communications;       // 总通信次数
    int restarts_used;        // 实际使用的 restart 次数 (新增)

    GMRESResult() : converged(false), final_residual(0.0),
                    iterations(0), communications(0), restarts_used(0) {}
};
```

### 3. Restart 循环结构

```cpp
void sstepGMRES(...) {
    for (int rst = 0; rst < params.max_restarts; rst++) {

        // Step A: 计算当前残差
        // 第 0 轮: x=0, 所以 r = b
        // 后续轮: x 已更新, 计算 r = b - A*x

        // Step B: 初始化本轮数据结构
        // 清零并重新分配 V, W, H, g, cs, sn

        // Step C: 构建第一个块 V_0
        // z = M^{-1} * r
        // 生成 Power basis V_0^0, V_0^1, ..., V_0^{s-1}
        // 计算 beta 和 W_0

        // Step D: 主循环 m 个块
        // 如果收敛: 更新解 x, 返回

        // Step E: 未收敛处理
        // 更新解 x = x + Σ V_k * y_k
        // 累加统计，继续下一轮 restart
    }

    // 达到最大 restart 次数仍未收敛
    result.converged = false;
}
```

### 4. 每轮 Restart 的额外开销

| 操作 | 通信次数 |
|------|---------|
| 计算 r = b - Ax | 1 次 Allreduce (||r||) |
| 初始化 V_0, W_0 | 1 次 Allreduce |
| 每轮主循环 | m+1 次 Allreduce |
| **每轮 restart 额外** | **2 次 Allreduce** |
| **总计（k 轮 restart）** | **k×(m+1) + 2k 次** |

### 5. 命令行参数

```bash
Usage: ./sstep_gmres_dist <n_global> <s> <m> <type> [tol] [max_restarts]

参数:
  n_global:     全局矩阵维度
  s:            s-step 参数 (2-3)
  m:            每轮最大块数
  type:         矩阵类型 (0=五对角, 1=各向异性)
  tol:          收敛容忍度 (默认 1e-8)
  max_restarts: 最大 restart 次数 (默认 25)
```

### 6. 输出格式

**收敛时：**
```
==============================================
Distributed s-step GMRES (with restart)
==============================================
Matrix: Anisotropic(0.01), n_global=10000
np=4, s=3, m=10, max_restarts=25
tol=1e-8
==============================================

Restart 1: Krylov=30, residual=1.2e-03
Restart 2: Krylov=30, residual=3.4e-05
Restart 3: Krylov=30, residual=8.7e-08
Restart 4: Krylov=21, residual=2.1e-09  [converged]

==============================================
Final Result:
==============================================
Converged: Yes
||b-Ax||/||b|| = 2.1e-09
Restarts used: 4
Total Krylov vectors: 111
Total communications: 50
==============================================
```

**未收敛时：**
```
...
Restart 25: Krylov=30, residual=1.5e-04

==============================================
Final Result:
==============================================
Converged: No
||b-Ax||/||b|| = 1.5e-04
Restarts used: 25
Total Krylov vectors: 750
Total communications: 325
==============================================
```

## 代码修改清单

| 组件 | 改动内容 | 行数估计 |
|------|---------|---------|
| `GMRESParams` | 新增 `max_restarts` 参数 | ~5 行 |
| `GMRESResult` | 新增 `restarts_used` 字段 | ~3 行 |
| `sstepGMRES` | 内置 restart 循环 | ~120-150 行 |
| `main` | 参数解析 + 输出格式 | ~30-40 行 |
| **总计** | | **~160-200 行** |

## 实现方案

采用**方案 B：内置 restart 循环**

在现有 `sstepGMRES` 函数内部直接添加 restart 循环，保持单一函数结构，逻辑集中。

**优点：**
- 代码改动集中在一个函数
- 无需新增外部 wrapper
- 与现有代码风格一致

**缺点：**
- 函数略长，但职责仍清晰

## 不改动的部分

- 辅助函数 `globalDot`, `globalNorm`, `solveSPD`
- 回调函数 `matVec`, `precond` 的接口
- 分布式矩阵、向量、预处理器的实现
- 测试脚本 `test_dist.sh`（仅需更新参数）

## 测试计划

1. **五对角矩阵**（收敛快）：验证 restart 不影响正常收敛
2. **各向异性矩阵**（收敛慢）：验证 restart 能够继续迭代
3. **通信次数验证**：确认每轮 restart 增加 2 次通信
4. **边界情况**：max_restarts=1 等价于无 restart