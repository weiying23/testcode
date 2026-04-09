# s-step GMRES Restart 机制实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为分布式 s-step GMRES 求解器增加 restart 机制，支持固定周期 restart，控制内存使用并改善收敛性。

**Architecture:** 在现有 `sstepGMRES` 函数内部添加 restart 循环，每轮 m 个块后用当前解重新计算残差并开始新一轮迭代。保持单一函数结构，逻辑集中。

**Tech Stack:** C++11, MPI, LAPACK (dposv)

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `sstep_gmres_dist.cpp` | 唯一需要修改的文件，包含所有改动 |

---

### Task 1: 扩展参数和结果结构体

**Files:**
- Modify: `sstep_gmres_dist.cpp:43-66`

- [ ] **Step 1: 扩展 GMRESResult 结构体**

在 `GMRESResult` 结构体中添加 `restarts_used` 字段：

```cpp
struct GMRESResult {
    bool converged;           // 是否收敛
    double final_residual;    // 最终相对残差 ||b-Ax||/||b||
    int iterations;           // 总迭代块数（所有 restart 轮次累计）
    int communications;       // 总通信次数
    int restarts_used;        // 实际使用的 restart 次数 (新增)

    GMRESResult() : converged(false), final_residual(0.0),
                    iterations(0), communications(0), restarts_used(0) {}
};
```

- [ ] **Step 2: 扩展 GMRESParams 结构体**

在 `GMRESParams` 结构体中添加 `max_restarts` 参数：

```cpp
struct GMRESParams {
    int s;          // s-step 参数 (推荐 2-3)
                    // - s=2: 最稳定，但收敛可能慢
                    // - s=3: 推荐，收敛快且通信少
                    // - s>=4: 可能数值不稳定
    int m;          // 每轮最大块数，总 Krylov 维度 = s * m
    double tol;     // 收敛容忍度 (相对残差)
    int max_restarts;   // 最大 restart 次数 (新增，默认 25)

    GMRESParams(int s_val = 3, int m_val = 10, double tol_val = 1e-8, int max_rst = 25)
        : s(s_val), m(m_val), tol(tol_val), max_restarts(max_rst) {}
};
```

- [ ] **Step 3: 编译验证**

```bash
cd /Users/yingwei/Documents/code/testcode/solver/sstepgmres
mpicxx -std=c++11 -O3 -framework Accelerate sstep_gmres_dist.cpp -o sstep_gmres_dist 2>&1
```

Expected: 编译成功（可能有 deprecated 警告，忽略）

- [ ] **Step 4: Commit**

```bash
git add sstep_gmres_dist.cpp
git commit -m "feat: add max_restarts parameter and restarts_used to result struct

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: 重构 sstepGMRES 函数 - 添加 restart 外层循环

**Files:**
- Modify: `sstep_gmres_dist.cpp:180-760`

此任务将现有代码重构为 restart 循环结构。分多个步骤完成。

- [ ] **Step 1: 添加 restart 循环框架**

在函数开头（参数提取后）添加 restart 循环框架，将现有代码结构包入循环：

找到第 199 行 `int ms = m * s;` 之后，添加变量定义并重构：

```cpp
    //--------------------------------------------------------------------------
    // 参数提取
    //--------------------------------------------------------------------------
    int s = params.s;      // s-step 参数 (每块向量数)
    int m = params.m;      // 每轮最大块数
    double tol = params.tol;
    int ms = m * s;        // 总 Krylov 维度

    //--------------------------------------------------------------------------
    // 工作向量初始化 (在 restart 循环外部)
    //--------------------------------------------------------------------------
    std::vector<double> r_local(n_local);      // 残差向量 (本地部分)
    std::vector<double> z_local(n_local);      // 预处理后的残差 M^{-1}*r
    std::vector<double> tmp_local(n_local);    // 临时向量
    std::vector<double> Atmp_local(n_local);   // 矩阵-向量乘结果 A*v
    std::vector<double> ghost_data(n_ghost);   // 幽灵层数据

    // 解向量初始化为零 (初始猜测 x₀ = 0)
    std::fill(x_local, x_local + n_local, 0.0);

    // 计算 ||b|| 用于相对残差判断
    double bnorm = globalNorm(comm, nprocs, b_local, n_local);

    //--------------------------------------------------------------------------
    // Krylov 子空间存储结构 (在 restart 循环外部，复用内存)
    //--------------------------------------------------------------------------
    std::vector<std::vector<double>> V(m + 1);
    for (int k = 0; k <= m; k++) {
        V[k].resize(s * n_local);
    }

    std::vector<std::vector<double>> W(m + 1);
    for (int k = 0; k <= m; k++) {
        W[k].resize(s * s);
    }

    std::vector<double> H((ms + 1) * ms, 0.0);
    std::vector<double> g(ms + 1, 0.0);
    std::vector<double> cs(ms), sn(ms);

    //--------------------------------------------------------------------------
    // Restart 循环
    //--------------------------------------------------------------------------
    int total_iterations = 0;       // 累计迭代块数
    int total_communications = 0;   // 累计通信次数

    for (int rst = 0; rst < params.max_restarts; rst++) {

        // ==============================================================
        // 现有的初始化和主循环代码将放在这里
        // ==============================================================
```

- [ ] **Step 2: 移动数据结构清零到 restart 循环内**

在 restart 循环开始处添加数据结构重置：

```cpp
        // 每轮 restart 重置数据结构
        std::fill(H.begin(), H.end(), 0.0);
        std::fill(g.begin(), g.end(), 0.0);
        std::fill(cs.begin(), cs.end(), 0.0);
        std::fill(sn.begin(), sn.end(), 0.0);
        int givens_count = 0;
        int ncomm = 0;  // 本轮通信计数
```

- [ ] **Step 3: 添加残差计算逻辑**

在 restart 循环内，数据结构重置后添加残差计算：

```cpp
        //==============================================================
        // Step A: 计算当前残差
        //==============================================================
        if (rst == 0) {
            // 第一轮: x=0, 所以 r = b
            std::copy(b_local, b_local + n_local, r_local.begin());
        } else {
            // 后续轮: x 已更新, 计算 r = b - A*x
            std::vector<double> Ax_local(n_local);
            matVec(x_local, ghost_data.data(), Ax_local.data());
            for (int i = 0; i < n_local; i++) {
                r_local[i] = b_local[i] - Ax_local[i];
            }
        }
```

- [ ] **Step 4: 添加本轮 V_0 初始化代码**

将现有的 V_0 初始化代码（Step 1.1-1.8）移入 restart 循环内，并修改为使用 `r_local`：

```cpp
        //==============================================================
        // Step B: 计算初始预处理残差 z = M^{-1} * r
        //==============================================================
        precond(r_local.data(), z_local.data());

        //==============================================================
        // Step C: 构建 V_0 的 Power basis
        //==============================================================
        for (int i = 0; i < n_local; i++) {
            V[0][i] = z_local[i];
        }

        for (int j = 1; j < s; j++) {
            std::copy(&V[0][(j - 1) * n_local], &V[0][j * n_local], tmp_local.begin());
            matVec(tmp_local.data(), ghost_data.data(), Atmp_local.data());
            precond(Atmp_local.data(), tmp_local.data());
            std::copy(tmp_local.begin(), tmp_local.end(), &V[0][j * n_local]);
        }

        //==============================================================
        // Step D: 计算 beta² 和 W_0 (一次 Allreduce)
        //==============================================================
        int init_size = 1 + s * s;
        std::vector<double> init_loc(init_size, 0.0);
        init_loc[0] = vdot(n_local, z_local.data(), z_local.data());

        int idx = 1;
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                init_loc[idx++] = vdot(n_local, &V[0][i * n_local], &V[0][j * n_local]);
            }
        }

        std::vector<double> init_glb(init_size);
        MPI_Allreduce(init_loc.data(), init_glb.data(), init_size, MPI_DOUBLE, MPI_SUM, comm);
        ncomm++;

        double beta_sq = init_glb[0] / nprocs;
        double beta = std::sqrt(beta_sq);

        // 检查是否已经收敛
        if (beta < tol * bnorm) {
            result.converged = true;
            result.final_residual = beta / bnorm;
            result.iterations = total_iterations;
            result.communications = total_communications + ncomm;
            result.restarts_used = rst + 1;
            return;
        }

        // 归一化 V_0
        for (int j = 0; j < s; j++) {
            vscal(n_local, 1.0 / beta, &V[0][j * n_local]);
        }

        // 提取归一化后的 W_0
        idx = 1;
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                W[0][i * s + j] = init_glb[idx++] / (beta_sq * nprocs);
            }
        }

        g[0] = beta;

        // Scalar2 设置
        for (int j = 0; j < s - 1; j++) {
            H[j * (ms + 1) + j + 1] = 1.0;
        }
```

- [ ] **Step 5: 修改主循环收敛处理**

修改主循环内的收敛处理，累积统计并返回：

在收敛块（原 Step 2.12）中，修改 `result` 赋值和 `return`：

```cpp
        if (res_est < tol) {
            // 解上三角系统
            std::vector<double> y(ms, 0.0);
            for (int i = givens_count - 1; i >= 0; i--) {
                y[i] = g[i];
                for (int j = i + 1; j < givens_count; j++) {
                    y[i] -= H[j * (ms + 1) + i] * y[j];
                }
                if (std::abs(H[i * (ms + 1) + i]) > 1e-14) {
                    y[i] /= H[i * (ms + 1) + i];
                }
            }

            // 更新解向量
            for (int kk = 0; kk <= k; kk++) {
                for (int j = 0; j < s; j++) {
                    vaxpy(n_local, y[kk * s + j], &V[kk][j * n_local], x_local);
                }
            }

            // 计算真实残差
            std::vector<double> Ax_local(n_local), res_local(n_local);
            matVec(x_local, ghost_data.data(), Ax_local.data());
            for (int i = 0; i < n_local; i++) {
                res_local[i] = b_local[i] - Ax_local[i];
            }
            result.final_residual = globalNorm(comm, nprocs, res_local.data(), n_local) / bnorm;

            result.converged = true;
            result.iterations = total_iterations + k + 1;
            result.communications = total_communications + ncomm;
            result.restarts_used = rst + 1;
            return;
        }
```

- [ ] **Step 6: 添加未收敛轮次的处理**

在主循环结束后（原 Phase 3 位置），改为更新解并准备下一轮 restart：

```cpp
        //==============================================================
        // 本轮未收敛: 更新解并准备下一轮 restart
        //==============================================================
        std::vector<double> y(ms, 0.0);
        for (int i = ms - 1; i >= 0; i--) {
            y[i] = g[i];
            for (int j = i + 1; j < ms; j++) {
                y[i] -= H[j * (ms + 1) + i] * y[j];
            }
            if (std::abs(H[i * (ms + 1) + i]) > 1e-14) {
                y[i] /= H[i * (ms + 1) + i];
            }
        }

        // 更新解向量
        for (int k_iter = 0; k_iter < m; k_iter++) {
            for (int j = 0; j < s; j++) {
                vaxpy(n_local, y[k_iter * s + j], &V[k_iter][j * n_local], x_local);
            }
        }

        // 累积统计
        total_iterations += m;
        total_communications += ncomm;

        // 输出本轮 restart 信息 (汇总模式)
        if (rank == 0) {
            std::cout << "Restart " << (rst + 1) 
                      << ": Krylov=" << (m * s)
                      << ", residual=" << std::scientific << res_est << "\n";
        }
    }  // end of restart loop
```

- [ ] **Step 7: 添加达到最大 restart 次数的处理**

在 restart 循环结束后添加：

```cpp
    //==========================================================================
    // 达到最大 restart 次数仍未收敛
    //==========================================================================
    // 计算最终残差
    std::vector<double> Ax_final(n_local), res_final(n_local);
    matVec(x_local, ghost_data.data(), Ax_final.data());
    for (int i = 0; i < n_local; i++) {
        res_final[i] = b_local[i] - Ax_final[i];
    }
    result.final_residual = globalNorm(comm, nprocs, res_final.data(), n_local) / bnorm;

    result.converged = false;
    result.iterations = total_iterations;
    result.communications = total_communications;
    result.restarts_used = params.max_restarts;
```

- [ ] **Step 8: 编译验证**

```bash
cd /Users/yingwei/Documents/code/testcode/solver/sstepgmres
mpicxx -std=c++11 -O3 -framework Accelerate sstep_gmres_dist.cpp -o sstep_gmres_dist 2>&1
```

Expected: 编译成功

- [ ] **Step 9: 功能测试 - max_restarts=1 等价于无 restart**

```bash
mpirun -np 4 ./sstep_gmres_dist 10000 3 10 0 1e-8 1
```

Expected: 结果与无 restart 版本一致（收敛，残差 ~4e-9）

- [ ] **Step 10: Commit**

```bash
git add sstep_gmres_dist.cpp
git commit -m "feat: implement restart loop in sstepGMRES

- Add outer restart loop with max_restarts iterations
- Reset data structures each restart cycle
- Preserve solution x across restarts
- Accumulate iteration and communication counts

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: 更新 main 函数 - 命令行参数和输出格式

**Files:**
- Modify: `sstep_gmres_dist.cpp:765-927`

- [ ] **Step 1: 更新用法提示**

修改 `main` 函数中的用法提示，添加 `max_restarts` 参数：

```cpp
    if (argc < 5) {
        if (rank == 0) {
            std::cout << "Usage: " << argv[0] << " <n_global> <s> <m> <type> [tol] [max_restarts]\n";
            std::cout << "  n_global: global matrix dimension\n";
            std::cout << "  s: s-step parameter (2-3)\n";
            std::cout << "  m: blocks per restart cycle\n";
            std::cout << "  type: 0=five-diagonal, 1=anisotropic\n";
            std::cout << "  tol: convergence tolerance (default 1e-8)\n";
            std::cout << "  max_restarts: max restart cycles (default 25)\n";
        }
        MPI_Finalize();
        return 1;
    }
```

- [ ] **Step 2: 解析 max_restarts 参数**

在参数解析部分添加：

```cpp
    int n_global = std::atoi(argv[1]);
    int s = std::atoi(argv[2]);
    int m = std::atoi(argv[3]);
    int type = std::atoi(argv[4]);
    double tol = (argc > 5) ? std::atof(argv[5]) : 1e-8;
    int max_restarts = (argc > 6) ? std::atoi(argv[6]) : 25;  // 新增

    // 参数约束
    if (s < 2) s = 2;
    if (s > 3) s = 3;
    if (max_restarts < 1) max_restarts = 1;  // 新增
```

- [ ] **Step 3: 更新问题信息输出**

```cpp
    if (rank == 0) {
        std::cout << "==============================================\n";
        std::cout << "Distributed s-step GMRES (with restart)\n";
        std::cout << "==============================================\n";
        std::cout << "Matrix: " << mat_name << ", n_global=" << n_global << "\n";
        std::cout << "np=" << nprocs << ", s=" << s << ", m=" << m 
                  << ", max_restarts=" << max_restarts << "\n";
        std::cout << "tol=" << tol << "\n";
        std::cout << "==============================================\n\n";
    }
```

- [ ] **Step 4: 更新 GMRESParams 构造**

```cpp
    GMRESParams params(s, m, tol, max_restarts);
```

- [ ] **Step 5: 更新最终结果输出**

```cpp
    if (rank == 0) {
        std::cout << "\n==============================================\n";
        std::cout << "Final Result:\n";
        std::cout << "==============================================\n";
        std::cout << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
        std::cout << "||b-Ax||/||b|| = " << std::scientific << result.final_residual << "\n";
        std::cout << "Restarts used: " << result.restarts_used << "\n";
        std::cout << "Total Krylov vectors: " << result.iterations * s << "\n";
        std::cout << "Total communications: " << result.communications << "\n";
        std::cout << "==============================================\n";
    }
```

- [ ] **Step 6: 编译验证**

```bash
cd /Users/yingwei/Documents/code/testcode/solver/sstepgmres
mpicxx -std=c++11 -O3 -framework Accelerate sstep_gmres_dist.cpp -o sstep_gmres_dist 2>&1
```

Expected: 编译成功

- [ ] **Step 7: Commit**

```bash
git add sstep_gmres_dist.cpp
git commit -m "feat: update main function for restart parameters

- Add max_restarts command line argument
- Update usage message and output format
- Show restart count in final results

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: 测试验证

**Files:**
- Run: 可执行文件测试

- [ ] **Step 1: 测试五对角矩阵（收敛快）**

```bash
mpirun -np 4 ./sstep_gmres_dist 10000 3 10 0 1e-8 25
```

Expected: 收敛，restarts_used=1（或更少），残差 < 1e-8

- [ ] **Step 2: 测试各向异性矩阵（收敛慢）**

```bash
mpirun -np 4 ./sstep_gmres_dist 10000 3 10 1 1e-6 25
```

Expected: 可能需要多次 restart，最终收敛或显著降低残差

- [ ] **Step 3: 验证 max_restarts=1 等价于无 restart**

```bash
mpirun -np 4 ./sstep_gmres_dist 10000 3 10 0 1e-8 1
```

Expected: 行为与原版本完全一致

- [ ] **Step 4: 验证通信次数计算**

检查输出中的通信次数是否正确：
- 第一轮 restart: init (1) + m blocks (m) = m+1 次
- 后续轮次: 残差计算额外 + init + m blocks

Expected: 通信次数符合预期公式

- [ ] **Step 5: Commit 测试通过标记**

```bash
git add sstep_gmres_dist.cpp
git commit -m "test: verify restart mechanism works correctly

- Five-diagonal matrix converges quickly
- Anisotropic matrix benefits from restart
- max_restarts=1 equivalent to no restart
- Communication count correct

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: 更新压缩包

**Files:**
- Create: `sstep_gmres_dist.zip`

- [ ] **Step 1: 重新打包**

```bash
cd /Users/yingwei/Documents/code/testcode/solver/sstepgmres
rm -f sstep_gmres_dist.zip
zip sstep_gmres_dist.zip \
    sstep_gmres_dist.cpp \
    blas_utils.h \
    dist_vector.h \
    dist_matrix.h \
    dist_ilu.h \
    test_dist.sh \
    sstep_gmres_dist
```

- [ ] **Step 2: Commit**

```bash
git add sstep_gmres_dist.zip
git commit -m "chore: update zip archive with restart feature

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## 验收标准

| 标准 | 验证方法 |
|------|---------|
| max_restarts=1 行为等价于原版本 | 测试对比 |
| 五对角矩阵收敛 | 运行测试 |
| 各向异性矩阵可通过 restart 改善 | 运行测试 |
| 通信次数正确 | 检查输出 |
| 输出格式正确 | 检查输出 |