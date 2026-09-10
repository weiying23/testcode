# testcode

HPC 高性能计算研究与实验仓库，聚焦并行编程模型（MPI/OpenMP/pthreads）、数值线性代数求解器、ARM SIMD/SVE/SME（Apple Silicon）、昇腾 NPU 通信库、stencil 缓存分块优化、混合精度随机舍入、以及 Docker 跨容器 MPI 等方向的性能探索与基准测试。

## 目录结构

按主题分组：

### 并行编程模型

| 目录 | 语言 | 用途 |
| --- | --- | --- |
| [`multithread_simple_case`](multithread_simple_case) | C++/asm | MPI/OpenMP/pthreads 与 ARM SVE/SME 的 hello-world 对比，研究线程绑定对性能的影响 |
| [`coll_mpi_demo`](coll_mpi_demo) | C/C++ | MPI collective（gather/scatter）演示与基准，含手写 mympi 与原生 MPI 对比 |
| [`graphschedule`](graphschedule) | C++ | 基于 DAG 的任务调度算法，配合自定义线程池 |
| [`test_thread_sync`](test_thread_sync) | C | pthread mutex 同步与竞态条件演示 |
| [`test_thread_user_defined_barrier`](test_thread_user_defined_barrier) | C | 用户定义 barrier 的并行热传导仿真（条件变量 vs 忙等） |
| [`test_allocate_mem_compare`](test_allocate_mem_compare) | C | tcmalloc 与默认分配器在单线程/多线程下的性能对比 |

### 数值线性代数求解器

| 目录 | 语言 | 用途 |
| --- | --- | --- |
| [`solver`](solver) | C/C++ | GMRES（ILU0/ILUT 预处理）、s-step GMRES、LU 求逆、稀疏三角求解 |

### ARM SIMD / SME（Apple Silicon）

| 目录 | 语言 | 用途 |
| --- | --- | --- |
| [`testsme`](testsme) | C/asm/Py | ARM SME 探索（vendored libxsmm、M4 微基准、ZA 寄存器 GEMM demo） |
| [`testblas`](testblas) | C++ | GEMM 基准：pthread 行并行 vs OpenMP vs Accelerate/BLAS + 线程池 |
| [`mixed`](mixed) | C | 随机舍入（stochastic rounding）的混合精度 fp16/fp32 实现 |

### stencil / 缓存分块优化

| 目录 | 语言 | 用途 |
| --- | --- | --- |
| [`12order-scheme`](12order-scheme) | C | 3D 12 阶有限差分 stencil 性能优化（融合/分块/OpenMP，2.35x 加速） |
| [`test_stencil_tiling`](test_stencil_tiling) | C | 5 阶 stencil 缓存分块实验与 cache 分析 |

### 昇腾 NPU 生态

| 目录 | 语言 | 用途 |
| --- | --- | --- |
| [`ascend`](ascend) | C/C++/Py | 昇腾 NPU 相关项目集合：hcomm/shmem 通信库、MatterSim/LAMMPS 分子模拟、MatRIS 训练 |

### Docker 跨容器 MPI

| 目录 | 语言 | 用途 |
| --- | --- | --- |
| [`twin_docker`](twin_docker) | C/Docker | 双 Docker 容器隔离网络运行 MPI，含 passwordless SSH 与数据校验 |

## 构建与运行

原生 C/C++ 项目通用编译模式：

```bash
clang[-++] -O3 -mcpu=apple-m4 -pthread/-fopenmp ... -o <bin>
mpirun -np <N> ./<bin>
```

各子目录多提供 `run.sh` 一键编译运行脚本；`ascend/` 与 `testsme/` 下为 vendored CMake 项目，按各自子目录 README 构建。

## 环境依赖

- clang（支持 `-mcpu=apple-m1`/`apple-m4`，ARM SVE/SME）
- OpenMPI（`mpicc`/`mpirun`）
- libomp（Homebrew：`brew install libomp`）
- Apple Accelerate framework（LAPACK/BLAS）
- Docker（仅 `twin_docker` 需要）
- CANN / 昇腾 NPU 工具链（仅 `ascend/` 需要）
