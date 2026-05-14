# SME Matrix Multiply Demo

基于 Apple M4 SME (Scalable Matrix Extension) 的双精度矩阵乘法实现。

## 项目简介

本项目演示如何利用 Apple Silicon M4 芯片的 SME 扩展进行高性能双精度矩阵乘法 (GEMM) 计算。SME 是 ARMv9 架构新增的矩阵计算扩展，提供专门的 ZA 矩阵寄存器阵列和 fmopa 外积累加指令，一条指令可完成 8×8 = 64 次融合乘加 (FMA)。

## 系统要求

- **硬件**：Apple M4 芯片（支持 SME F64F64）
- **系统**：macOS Sequoia 15.x 或更高版本
- **编译器**：Apple Clang 16+（支持 `-march=armv9-a+sme+sme-f64f64`）

## 文件结构

```
sme_matrix/
├── sme_demo.cpp          # 最小化 8×8 SME 演示
├── matrix_multiply.cpp   # 完整五层优化 GEMM 实现
├── sme_intrinsics.h      # SME 内联汇编封装
├── matrix_methods.h      # 函数声明
├── packing.h             # Panel Packing 接口声明
├── packing.cpp           # Panel Packing 实现
├── benchmark.cpp         # 性能基准测试
├── run.sh                # 编译脚本
└── run_demo.sh           # 演示编译脚本
```

## 编译与运行

### 最小化演示 (sme_demo)

```bash
./run_demo.sh
```

或手动编译：

```bash
/usr/bin/clang++ -O2 -march=armv9-a+sve+sve2+sme+sme-f64f64 \
    sme_demo.cpp -o sme_demo
./sme_demo
```

输出 8×8 矩阵乘法的 SME 计算结果与标量参考的对比验证。

### 完整 GEMM + 性能测试

```bash
./run.sh
```

或手动编译：

```bash
# packing.cpp 需用 -O2 以启用 NEON 向量化
/usr/bin/clang++ -O2 -march=armv9-a -c packing.cpp -o packing.o

# matrix_multiply.cpp 必须用 -O0 防止编译器破坏 SME 汇编
/usr/bin/clang++ -O0 -march=armv9-a+sve+sve2+sme+sme-f64f64 \
    -c matrix_multiply.cpp -o matrix_multiply.o

# benchmark.cpp 用 -O2 + Accelerate 框架
/usr/bin/clang++ -O2 -march=armv9-a+sve+sve2+sme+sme-f64f64 \
    -framework Accelerate \
    matrix_multiply.o packing.o benchmark.cpp -o benchmark

./benchmark
```

## 技术要点

### SME 编程核心步骤

1. **smstart** — 进入流式模式，激活 ZA 矩阵寄存器 (SVCR.SM=1, SVCR.ZA=1)
2. **zero** — 清零 ZA 累加器
3. **ptrue / whilelt** — 设置谓词寄存器
4. **ld1d za** — 加载矩阵数据到 ZA tile
5. **fmopa** — 外积累加：`za[i,j] += a[i] * b[j]` (64 FMA/指令)
6. **st1d za** — 写回结果
7. **smstop** — 退出流式模式

### macOS 限制

macOS 内核不向用户态暴露非流式 SVE 指令，以下操作会触发 SIGILL：

- `cntd` / `svcntd()` — 读取非流式向量长度
- `__arm_streaming` 函数属性 — 编译器自动插入 cntd

**解决方案**：所有 SME 状态切换用 `__asm__ volatile` 手动管理，SVL 硬编码为 8 (512 bits)。

### 五层优化策略

| 层次 | 优化 | 说明 |
|------|------|------|
| 1 | 多线程并行 | M 方向分片，利用 M4 的 10 个核 (4P+6E) |
| 2 | 缓存分块 | Mc×Kc×Nc = 64×256×96，工作集 368KB 驻留 L2 |
| 3 | Panel Packing | 压缩矩阵行间距，改善内存局部性 |
| 4 | 6 路 j-tile 展开 | 一次 A 列读取驱动 6 次 fmopa |
| 5 | 软件预取 | 掩盖内存访问延迟 |

### 缓存分块参数

```
Mc = 64    A 行方向块大小
Kc = 256   A/B K 方向块大小
Nc = 96    B/C N 方向块大小

工作集估算：
  A_pack = 64 × 256 × 8 = 128 KB
  B_pack = 256 × 96 × 8 = 192 KB
  C tile = 64 × 96 × 8  =  48 KB
  合计                 ≈ 368 KB
```

适配 M4 E 核 4MB L2 和 P 核 16MB L2 缓存。

### ZA Tile 结构 (SVL = 512 bits)

```
ZA 是 512×512 bits 的矩阵寄存器阵列
双精度 f64f64 模式下划分为 8 个独立 tile (za0.d … za7.d)
每个 tile 可容纳 8×8 = 64 个 double

访问方式：
  水平切片 zaTh.d[w12, 0] — 第 w12 行
  垂直切片 zaTv.d[w12, 0] — 第 w12 列
```

## 性能测试输出示例

```
════════════════════════════════════════════════════════════
  正确性验证（SME 结果 vs Naive 参考，误差阈值 1e-9）
════════════════════════════════════════════════════════════
  维度 M×K×N                       Naive    SME误差   BLAS误差
  ──────────────────────────────  ────────  ────────  ────────
  方阵 8×8×8（1 tile）               0        0        0  ✓
  方阵 64×64×64                     0        0        0  ✓
  ...

════════════════════════════════════════════════════════════
  性能对比：SME vs Apple Accelerate（cblas_dgemm）
════════════════════════════════════════════════════════════
  维度 M×K×N                   SME(ms)  GFLOPS  BLAS(ms)  GFLOPS  加速比
  ─────────────────────────  ──────── ────────  ──────── ────────  ──────
  方阵 512³                      12.34   21.89     10.56   25.64   1.17x
  方阵 1024³                     89.12   23.92     78.45   27.17   1.14x
  ...
```

注：加速比 >1 表示 Accelerate 更快，<1 表示 SME 更快。

## 参考资料

- [ARM SME Introduction](https://developer.arm.com/documentation/109368/latest/)
- [SME Programmer's Guide](https://developer.arm.com/documentation/den0040/a/)
- [Apple Silicon Performance Optimization](https://developer.apple.com/documentation/performance)

## 许可证

本项目仅供学习和研究使用。