# SME Demo

Apple M4 SME (Scalable Matrix Extension) 探索项目，按时间线分三个阶段。

## 目录结构

| 目录 | 阶段 | 内容 |
| --- | --- | --- |
| [`01-hello-sme/`](01-hello-sme) | 2026-01 入门 | SME 指令试水：`smeop.s` 手写汇编（smstart/ld1w/faddv/mopa）、`print_smeop.c` 驱动程序、`sme_test.cpp` 16×16 SME 矩阵乘验证 |
| [`02-debug-notes/`](02-debug-notes) | 2026-02 调试 | `matrix_multiply.cpp` Illegal Instruction 错误的三轮根因分析：DIAGNOSIS（初诊）→ REAL_DIAGNOSIS（真因：声明/定义不匹配）→ BUGS_FOUND（2 个实锤 bug） |
| [`sme_matrix/`](sme_matrix) | 2026-04~10 主项目 | 完整五层优化 SME GEMM（ZA 寄存器 + fmopa 外积 + Panel Packing），详见其内部 [README](sme_matrix/README.md) 与 [optimization_log.md](sme_matrix/optimization_log.md) |

## 参考

- [m4-sme-exploration](../m4-sme-exploration)（同级目录）：vendored 的系统性 M4 SME 微基准研究
- [hellosme](../hellosme)：libxsmm 与耶拿大学 SME GEMM 参考实现（clone，不入库）
