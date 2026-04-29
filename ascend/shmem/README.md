SHMEM
===
昇腾平台多机多卡内存访问加速库，为模型与算子开发者提供高效、易用的跨设备内存通信能力，简化卡间数据同步与通算融合算子开发流程。

## 最新动态
🔥 [2025/12] SHMEM项目首次上线。

## 一、项目简介
SHMEM 是面向昇腾平台的多机多卡内存通信库，通过封装 Host 侧与 Device 侧接口，实现跨设备的高效内存访问与数据同步。其核心价值在于：
- 支持 AICore直驱 MTE、xDMA 使能 D2D/D2H/H2D/D2rH/rH2D 通信
- 简化分布式场景下的卡间通信逻辑，降低算子开发门槛
- 与 CANN 生态深度适配，支持通算融合类算子的快速部署
- 更多详细资料请参考[SHMEM](https://shmem-doc.pages.dev/)


## 二、核心功能
![核心功能](docs/images/readme-features.png)

**1. 双侧接口体系**
- Host侧：负责初始化、内存堆管理、通信域（Team）创建及全局同步
- Device侧：提供远程内存访问（RMA）、设备级同步及通信域操作

接口设计贴合昇腾算子开发范式，支持 Host 与 Device 协同工作流。

**2. 高性能通信优化**
- 内置 MTE、xDMA 引擎支持，实现远程内存直接读写，减少数据传输延迟
- 兼容 MPI 通信框架，支持集合通信（Allgather/Allreduce 等）场景
- 针对昇腾硬件特性优化数据传输路径，提升多卡协同效率

**3. 安全通信机制**
- 默认启用 TLS 加密保护跨设备数据传输，支持接口级关闭控制：
   ```c
   int32_t ret = aclshmemx_set_conf_store_tls(false, NULL, 0);
   ```

- 提供安全加固指南，包括权限配置、加密套件选择等企业级安全策略

**4. 通信通路覆盖**

下图展示了 SHMEM 支持的全链路通信通路（以910A3为例），覆盖 Host 侧与 Device 侧的不同传输引擎：

<img src="docs/images/dma.png" width="800"/>

如图所示，SHMEM 支持多种通信引擎和通路：
- **MTE Engine**：芯片级内存传输引擎，支持 D2D、D2H、H2D、D2rH、rH2D 多种通路
- **xDMA Engine**：高速直接内存访问引擎，支撑主机内/主机间高效数据传输

**5. 多语言与扩展支持**
- 提供 C++ 原生接口与 Python 封装，满足不同开发场景需求
- 模块化设计支持通信后端（MTE/xDMA）动态切换，便于功能扩展

**6. 丰富场景样例**
覆盖基础通信到复杂算子融合场景：
- rdma_demo：RDMA 协议通信演示
- matmul_allreduce：通算融合算子实现（矩阵乘 + Allreduce）


## 三、环境准备

### 3.1 硬件要求
- Atlas 系列：800I A2/A3、800T A2/A3
- 架构兼容：aarch64、x86

### 3.2 软件依赖
#### 3.2.1 CANN 版本说明
| 驱动固件 | CANN版本 | D2D | D2H/H2D | D2rH/rH2D | 其他依赖 |
| --- | --- | --- | --- | --- | --- |
| Ascend HDK 25.0.RC1.1 | 8.3.RC1 及以上<br>[社区版资源](https://www.hiascend.com/developer/download/community/result?module=cann) | MTE<br>RDMA |  |  | |
| Ascend HDK 25.0.RC1.1 | 8.5.0 及以上<br>[社区版资源](https://www.hiascend.com/developer/download/community/result?module=cann) | MTE<br>RDMA | MTE | MTE | 使能 A3 D2rH/rH2D：LingQu Computing Network [1.5.0版本](https://support.huawei.com/enterprise/zh/ascend-computing/lingqu-computing-network-pid-258003841/software)<br>升级指导：[安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/lingqu-computing-network-pid-258003841) |
| Ascend HDK 25.0.RC1.1 | 9.0.0 及以上 <br> 尝鲜版toolkit 包：[x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260305000326487/x86_64/Ascend-cann-toolkit_9.0.0_linux-x86_64.run)/[aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260305000326487/aarch64/Ascend-cann-toolkit_9.0.0_linux-aarch64.run)| MTE<br>RDMA<br>SDMA | MTE | MTE | 使能 SDMA 需下载 ops-legacy 包（按硬件平台）：[A2 x86_64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-910b-ops-legacy_9.0.0_linux-x86_64.run)/[A2 aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-910b-ops-legacy_9.0.0_linux-aarch64.run)/[A3 x86_64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-A3-ops-legacy_9.0.0_linux-x86_64.run)/[A3 aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-A3-ops-legacy_9.0.0_linux-aarch64.run)  |


#### 3.2.2 CANN 包安装
参考[快速安装CANN](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)

配置 CANN 环境变量（默认安装路径）：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

配置 CANN 环境变量（自定义安装路径）：
```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

### 3.3 其他软件依赖
- 安装 PyTorch 框架和 torch_npu 插件
  编译运行 torch 输入输出 tensor 的算子时必须安装本包。
  根据实际环境，选择对应的版本进行安装，具体可以查看[Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html)文档。
- 工具链：
  - cmake ≥ 3.19
  - GLIBC ≥ 2.28

### 3.4 python依赖安装
```bash
python3 -m pip install -r requirements.txt
```
### 3.5 可选依赖
- MPI：OpenMPI 4.0+（分布式通信场景）
- Python：3.7+（Python 接口使用）
- PyTorch：1.12+（Python 示例运行）
## 四、快速开始

### 4.1 安装方式
#### 4.1.1 方式一：源码编译
```bash
# 克隆代码仓
git clone https://gitcode.com/cann/shmem.git
cd shmem

# 编译核心库（默认不包含xDMA能力，也不包含示例和测试）
bash scripts/build.sh

# 配置环境变量
source install/set_env.sh
```
备注：build.sh 的参数可以参考 [compilation_build_guide.md](./docs/compilation_build_guide.md)

#### 4.1.2 方式二：二进制包安装
获取方式：```bash scripts/build.sh -package```

软件包格式：```SHMEM_{version}_linux-{arch}.run```
```bash
#进入对应的目录
cd {project_root}/package/{arch}/
# 权限配置与校验
chmod +x SHMEM_1.0.0_linux-aarch64.run
./SHMEM_1.0.0_linux-aarch64.run --check

# 安装（默认路径：/usr/local/Ascend/shmem）
./SHMEM_1.0.0_linux-aarch64.run --install

# 配置环境变量
#（默认路径：/usr/local/Ascend/shmem）
source /usr/local/Ascend/shmem/latest/set_env.sh
#（自定义路径：${install_path}/shmem）
source ${install_path}/shmem/latest/set_env.sh
```

### 4.2 验证安装
以```matmul_allreduce```为例，验证核心功能：

1. 在源码 shmem/ 目录编译:

   ```sh
   bash scripts/build.sh -examples
   ```

2. 在 shmem/examples/matmul_allreduce 目录执行 demo:

   ```sh
   bash scripts/run.sh -ranks 2 -M 1024 -K 2048 -N 8192
   ```

注意：example 及其他样例代码仅供参考，在生产环境中请谨慎使用。

### 4.3 debug 模式使用
在源码 shmem/ 目录编译:

   ```sh
   bash scripts/build.sh -examples -debug
   ```

注意：此处 `-examples` 参数非必选项，仅提供[使用参考](./docs/debug/Troubleshooting_FAQs.md#SHMEM-常见问题)。

### 4.4 Python接口使用
注意：python 接口 API 列表可参考：[python接口API列表](./docs/api/pythonAPI.md)。

1. 在 `scripts` 目录下编译时，带上 `build python` 的选项

   ```sh
   bash build.sh -package
   ```

2. 在 install 目录下，source 环境变量

   ```sh
   source set_env.sh
   ```

3. 设置是否开启 TLS 认证，默认开启，若关闭 TLS 认证，请使用如下接口

   ```python
   import shmem as shm
   shm.set_conf_store_tls(False, "")   # 关闭tls认证
   ```

   ```python
   import shmem as shm
   tls_info = "xxx"
   shm.set_conf_store_tls(True, tls_info)# 开启TLS认证
   ```

4. 在 `examples/python_extension` 运行测试 demo

   ```sh
   bash run.sh
   ```

看到日志中打印出 `test.py running success!` 即为 demo 运行成功。


## 五、代码结构
```
shmem/                                  # 项目根目录
├── docs/                               # 文档与说明
├── examples/                           # 示例工程集合
├── include/                            # 对外头文件
│   ├── shmem.h                         # shmem所有对外API汇总
│   ├── device/                         # device侧头文件
│   │   ├── gm2gm/                      # aicore驱动gm2gm数据面高阶和低阶接口
│   │   │   └── engine/                 # aicore 直驱gm2gm低阶接口
│   │   ├── team/                       # devic侧通信域管理
│   │   └── ub2gm/                      # aicore驱动ub2gm数据面低阶接口
│   │       └── engine/                 # aicore直驱ub2gm低阶接口
│   ├── host/                           # host侧头文件
│   │   ├── data_plane/                 # host侧CPU驱动数据面接口
│   │   ├── init/                       # host侧初始化接口
│   │   ├── mem/                        # host侧内存管理接口
│   │   ├── team/                       # host侧通信域管理接口
│   │   └── utils/                      # 工具与通用辅助代码
│   └── host_device/                    # 共用目录
├── scripts/                            # 示例脚本（编译/运行）
├── src/                                # 源码实现
│   ├── device/                         # device侧实现
│   │   ├── gm2gm/                      # aicore直驱gm2gm数据面高阶和低阶接口
│   │   │   └── engine/                 # aicore直驱gm2gm低阶接口
│   │   ├── team/                       # device侧通信域管理
│   │   └── ub2gm/                      # aicore驱动ub2gm数据面低阶接口
│   │       └── mte/                    # aicore直驱ub2gm低阶接口
│   ├── host/                           # host侧实现
│   │   ├── bootstrap/                  # bootstrap
│   │   ├── hybm/                       # Hybrid Memory（hybm）实现
│   │   ├── init/                       # 初始化
│   │   ├── mem/                        # 内存管理相关
│   │   ├── python_wrapper/             # Python 封装/绑定
│   │   ├── sync/                       # 同步原语（barrier/p2p/order）
│   │   ├── team/                       # team（通信域）相关
│   │   ├── transport/                  # 传输层实现（如 RDMA）
│   │   └── utils/                      # 工具与通用辅助代码
│   ├── host_device/                    # 共用目录
│   └── python/                         # python相关目录
└── tests/                              # 测试用例集合（UT/功能测试）
```

## 六、典型使用场景
**1. 通算融合类算子开发**：基于 Device 侧内存直接访问接口，开发融合「计算+通信」的自定义算子（如 matmul+allreduce），减少卡间数据拷贝，提升算子执行效率。

**2. 多机多卡数据同步**：通过 Host 侧通信域管理接口，快速搭建多机多卡集群的内存共享通道，实现跨节点数据同步，适配分布式训练场景。

**3. 低延迟卡间通信**：利用 RDMA 优化的 Device 侧接口，实现卡间毫秒级数据传输，满足实时性要求高的 AI 推理场景。

**4. Python 分布式训练适配**：通过 Python 扩展接口，将 SHMEM 集成到 PyTorch 分布式训练流程中，替代传统 MPI 通信，降低训练通信开销。

## 七、测试框架
- **单元测试：** 覆盖核心接口（初始化、内存操作、同步等），位于 `tests/unittest/`

- **算子泛化性测试：** 针对 `matmul_allreduce` 等样例，支持动态生成测试数据与精度校验

### 7.1 运行单元测试

```bash
# 编译并运行单元测试
bash scripts/build.sh -uttests
bash scripts/run.sh
```
run.sh 脚本提供 `-ranks`、`-test_filter` 等参数自定义执行用例的卡数、gtest_filter 等，例如：
```bash
# 8卡，运行所有*Init*用例
bash scripts/run.sh -ranks 8 -test_filter Init
```
具体参数请见[相关脚本-run.sh](docs/compilation_build_guide.md#SHMEM关键文件介绍)


### 7.2 样例编译与执行

   ```bash
   bash scripts/build.sh -examples
   bash scripts/run_examples.sh
   ```

### 7.3 样例单独执行
```bash
可查看 examples 目录下对应的样例目录下的 README.md
```

### 7.4 运行 Python 测试用例
```bash
# 编译 Python 扩展
bash build.sh -python_extension
# 安装 wheel 包
pip3 install src/python/dist/shmem-xxx.whl --force-reinstall
# 2卡运行 Python 测试
torchrun --nproc-per-node=2 examples/python_extension/test/init_test.py
```

### 7.5 自定义测试
基于 ```tests/``` 目录的 gtest 框架，新增测试用例需遵循：
- 测试文件命名：```{module}_test.cc```
- 测试用例命名：```{FunctionName}_{Scenario}_Test```

## 八、配置与调优（可选，进阶使用）
**1. 关闭 TLS 加密（提升通信性能）**

默认开启 TLS 加密，若为内网可信环境，可关闭以提升通信速度：
```c
int32_t ret = aclshmemx_set_conf_store_tls(false, NULL, 0);
```
```python
import shmem as shm
shm.set_conf_store_tls(False, "")
```

**2. 调整共享内存大小**

初始化时指定共享内存池大小（默认 16GB），适配大内存场景：
```c
aclshmemx_init_attr_t attr;
attr.local_mem_size = 32 * 1024 * 1024 * 1024; // 32GB
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
```

**3. 性能调优建议**
- 优先使用 Device 侧接口（减少 Host-Device 交互）；
- 通信域分组时，按物理机节点划分，减少跨节点通信；
- 大批次数据传输时，开启 RDMA 协议（编译时加 -DSHMEM_RDMA=ON）。

## 九、常见问题（FAQ）
**Q1：编译时报「CANN 环境未找到」？**

A：确认已执行 ```source /usr/local/Ascend/ascend-toolkit/set_env.sh```，且 CANN 版本满足[环境依赖](#32-软件依赖)。

**Q2：运行示例时报「卡间通信超时」？**

A：检查网卡是否开启 RDMA、防火墙是否放行通信端口（默认 8666）、各节点时钟是否同步。

**Q3：Python 导入 shmem 时报「找不到模块」？**

A：确认已安装 wheel 包，且 ```source``` 了 install 目录下的 ```set_env.sh```，环境变量 ```PYTHONPATH``` 包含 shmem 路径。

**Q4：关闭 TLS 后仍提示加密失败？**

A：需在 ```aclshmemx_init_attr``` 前调用 ```aclshmemx_set_conf_store_tls```，初始化后无法修改 TLS 配置。

**Q5：googletest、catlass、这两个插件执行build.sh时提示git失败？**

A：确认 git 配置与是否可以访问网站，如果环境不能连接网站可以尝试手动下载文件到 3rdparty 目录下。

**Q6：CANN包安装失败？可访问常见问题地址，查看是否有解决的方案？**

A：查看[常见问题](https://www.hiascend.com/document/detail/zh/AscendFAQ/CommuFunc/resdl/rdl_011.html)

> 更多故障排查见：[Troubleshooting](docs/debug/Troubleshooting_FAQs.md)

## 十、贡献
### 贡献者列表
- [华南理工大学 陆璐教授团队](https://www2.scut.edu.cn/cs/2017/0629/c22284a328108/page.htm)

### 参与贡献指南
我们欢迎所有形式的贡献（代码、文档、bug反馈）：

**1. 提 Issue**

- 提交 bug：明确环境（硬件/软件版本）、复现步骤、错误日志；
- 提功能需求：说明场景、预期效果、适配的硬件/软件版本。

**2. 提 PR**

- 分支规范：功能开发用 ```feature/xxx```，bug 修复用 ```bugfix/xxx```；
- 代码规范：遵循 代码风格文档，新增代码需补充单元测试；
- PR 描述：说明修改目的、核心逻辑、测试验证结果。

**3. 代码审核**

- PR 需通过 CI 自动测试（编译、单元测试、代码规范检查）；
- 至少 1 名维护者审核通过后，方可合并。

详细步骤可参考[贡献指南](CONTRIBUTING.md)

## 十一、安全声明
- 通信安全：默认启用 TLS 加密，支持自定义加密套件
- 公网依赖：依赖的开源仓库与工具地址参见[公网地址清单](SECURITY.md#公网地址声明)
- 加固指南：参考[安全加固建议](SECURITY.md#安全加固)配置系统权限与防火墙

## 十二、版权与许可
Copyright (c) 2025 Huawei Technologies Co., Ltd.
本项目基于 CANN Open Software License Agreement Version 2.0 授权，仅允许用于昇腾处理器相关开发。

## 十三、注意事项
1. 本项目仅适配昇腾平台，不支持其他硬件架构（如 x86 通用服务器、NVIDIA GPU）；
2. 示例代码仅供学习参考，生产环境使用前需完成充分的功能和性能测试；
3. CANN 版本升级可能导致接口兼容问题，建议锁定文档指定的 CANN 版本；
4. 关闭 TLS 加密后，需确保通信网络为可信内网，避免数据泄露风险。
