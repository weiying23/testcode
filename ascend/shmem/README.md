SHMEM
===
🔥 [2025/10] SHMEM项目首次上线。

## 一、什么是SHMEM
### 介绍
本系统主要面向昇腾平台上的模型和算子开发者，提供便携易用的多机多卡内存访问方式，方便用户开发在卡间同步数据，加速通信或通算融合类算子开发。

### 软件架构
共享内存库接口主要分为host和device接口部分：
- host侧接口提供初始化、内存管理、通信域管理以及同步功能。
- device侧接口提供内存访问、同步以及通信域管理功能。

### 目录结构说明
详细介绍见[code_organization](docs/code_organization.md)
```
├── 3rdparty // 依赖的第三方库
├── docs     // 文档
├── examples // 使用样例
├── include  // 头文件
├── scripts  // 相关脚本
├── src      // 源代码
└── tests    // 测试用例
```
## 二、环境构建

### 软件硬件配套说明
- 硬件型号支持
  - Atlas 800I A2/A3 系列产品
  - Atlas 800T A2/A3 系列产品
- 平台：aarch64/x86
- 配套软件：驱动固件 Ascend HDK 25.0.RC1.1、 CANN 8.2.RC1及之后版本。
cmake >= 3.19
GLIBC >= 2.28

### 快速安装CANN软件
本节提供快速安装CANN软件的示例命令，更多安装步骤请参考[详细安装指南](#cann详细安装指南)。

#### 安装前准备
在线安装和离线安装时，需确保已具备Python环境及pip3，当前CANN支持Python3.7.x至3.11.4版本。
离线安装时，请单击[获取链接](https://www.hiascend.com/developer/download/community/result?module=cann)下载CANN软件包，并上传到安装环境任意路径。
#### 安装CANN
```shell
chmod +x Ascend-cann-toolkit_8.2.RC1_linux-$(arch).run
./Ascend-cann-toolkit_8.2.RC1_linux-$(arch).run --install
```
#### 安装后配置
配置环境变量脚本set_env.sh，当前安装路径以${HOME}/Ascend为例。
```
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
```
安装业务运行时依赖的Python第三方库（如果使用root用户安装，请将命令中的--user删除）。
```
pip3 install attrs cython 'numpy>=1.19.2,<=1.24.0' decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20.0 scipy requests absl-py --user
```
### CANN详细安装指南
开发者可访问[昇腾文档-昇腾社区](https://www.hiascend.com/document)->CANN社区版->软件安装，查看CANN软件安装引导，根据机器环境、操作系统和业务场景选择后阅读详细安装步骤。

## 三、快速上手
### SHMEM编译
 - 设置CANN环境变量<br>
    ```sh
    # root用户安装（默认路径）
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```
 - 共享内存库编译<br>
    编译共享内存库，设置共享内存库环境变量：
    ```sh
    cd shmem
    bash scripts/build.sh
    source install/set_env.sh
    ```
 - run包使用<br>
    软件包名为：SHMEM_{version}_linux-{arch}.run <br>
    其中，{version}表示软件版本号，{arch}表示CPU架构。<br>
    安装run包（需要依赖cann环境）<br>

    ```sh
    chmod +x 软件包名.run # 增加对软件包的可执行权限
    ./软件包名.run --check # 校验软件包安装文件的一致性和完整性
    ./软件包名.run --install # 安装软件，可使用--help查询相关安装选项
    ```
    出现提示`xxx install success!`则安装成功

注意：shmem 默认开启tls通信加密。如果需要关闭，需要调用接口主动关闭：
```c
int32_t ret = shmem_set_conf_store_tls(false, null, 0);
```
具体细节详见安全声明章节

### 执行样例算子Demo
以执行一个样例matmul_allreduce算子Demo为例：
1. 在源码shmem/目录编译:

   ```sh
   bash scripts/build.sh -examples
   ```

2. 在shmem/examples/matmul_allreduce目录执行demo:

   ```sh
   bash scripts/run.sh -ranks 2 -M 1024 -K 2048 -N 8192
   ```
   注意：example及其他样例代码仅供参考，在生产环境中请谨慎使用。

### 功能自测用例

共享内存库接口单元测试，在工程目录下执行
```sh
bash scripts/build.sh -uttests
bash scripts/run.sh
```
run.sh脚本提供-ranks -ipport -test_filter等参数自定义执行用例的卡数、ip端口、gtest_filter等，例如：

```sh
# 8卡，ip:port 127.0.0.1:8666，运行所有*Init*用例
bash scripts/run.sh -ranks 8 -ipport tcp://127.0.0.1:8666 -test_filter Init
```

### python侧test用例
注意：python接口API列表可参考：[python接口API列表](./docs/pythonAPI.md)。

1. 在scripts目录下编译的时候，带上build python的选项

   ```sh
   bash build.sh -python_extension
   ```
2. 在install目录下，source环境变量

   ```sh
   source set_env.sh
   ```
3. 在src/python目录下，进行setup，获取到wheel安装包

   ```sh
   python3 setup.py bdist_wheel
   ```
4. 在src/python/dist目录下，安装wheel包

   ```sh
   pip3 install shmem-xxx.whl --force-reinstall
   ```
5. 设置是否开启TLS认证，默认开启，若关闭TLS认证，请使用如下接口

   ```python
   import shmem as shm
   shm.set_conf_store_tls(False, "")   # 关闭tls认证
   ```

   ```python
   import shmem as shm
   tls_info = "xxx"
   shm.set_conf_store_tls(True, tls_info)      # 开启TLS认证
   ```
6. 使用torchrun运行测试demo

   ```sh
   torchrun --nproc-per-node=k test.py // k为想运行的ranksize
   ```
   看到日志中打印出“test.py running success!”即为demo运行成功

## 四、在样例工程使用调测功能
### cce::printf
#### 介绍
在example及其他样例代码中可使用设备侧打印函数`cce::printf`功能，用法与C标准库的printf一致。
#### 开启方法
若想使用该功能需要修改`examples\CMakeLists.txt`，为`target_compile_options`添加编译选项`--cce-enable-print`。

注意：cce::printf功能为编译器侧提供的能力，仅在CANN 8.2 T103版本支持，若您不想安装该版本的CANN，可使用[Ascend C算子调测API](#ascend-c算子调测api)提供的方法。
### Ascend C算子调测API
AscendC算子调测API是AscendC提供的调试能力，可进行kernel内部的打印、Tensor内容的查看(Dump)。

关于kernel调测api的详细介绍，可参考[DumpTensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/ascendcopapi/atlasascendc_api_07_0192.html)和[printf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/ascendcopapi/atlasascendc_api_07_0193.html)。

#### 插入调试代码

1. 修改使用该功能的核函数入口和相关调用代码，增加开启调测功能（`#if defined(ENABLE_ASCENDC_DUMP)`）的编译时代码，具体可参考`examples/matmul_allreduce/main.cpp`。
2. 在想进行调试的层级，增加调测API调用。

   ```diff
   // examples/matmul_allreduce/kernel/matmul_epilogue_comm.hpp
   template <>
   __forceinline__ __aicore__
   void operator()<AscendC::AIC>(Params &params)
   {
      BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
      uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

      BlockMmad blockMmad(resource);

      // Represent the full gm
      AscendC::GlobalTensor<ElementA> gmA;
      gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
      AscendC::GlobalTensor<ElementB> gmB;
      gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);

   +  AscendC::printf("coreLoops is %d\n", coreLoops);
   +  AscendC::DumpTensor(gmA, coreLoops, 16);
      ...
   }
   ```

#### 编译运行

1. 打开工具的编译开关`-enable_ascendc_dump`， 使能AscendC算子调测API编译算子样例。

   ```sh
   bash scripts/build.sh -enable_ascendc_dump -examples
   ```
2. 在shmem/examples/matmul_allreduce目录执行demo:

   ```sh
   bash scripts/run.sh -ranks 2 -M 1024 -K 2048 -N 8192
   ```
- ⚠ 注意事项
  - 目前`AscendC算子调测API`**不**支持打印`FixPipe`上的数值。

## 五、贡献

### 贡献者列表

- [华南理工大学 陆璐教授团队](https://www2.scut.edu.cn/cs/2017/0629/c22284a328108/page.htm)


### 参与贡献指南

1.  fork仓库 & 修改并提交代码
1.  新建 Pull-Request

详细步骤可参考[贡献指南](docs/CONTRIBUTING.md)

## 六、学习资源
- [api_demo](docs/api_demo.md)：api调用示例
- [code_organization](docs/code_organization.md)：
- [example](docs/example.md)：AllGather算子demo
- [quickstart](docs/quickstart.md)：编译运行流程说明
- [related_scripts](docs/related_scripts.md)：相关脚本介绍
- [pythonAPI](docs/pythonAPI.md)：SHMEM对外接口说明
- [Troubleshooting_FAQs](docs/Troubleshooting_FAQs.md)：使用限制&常见问题
- [CONTRIBUTING](docs/CONTRIBUTING.md)：如何向SHMEM贡献代码
## 七、参考文档
- **[CANN社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/index/index.html)**
- **[SHMEM文档](https://shmem-doc.pages.dev/)**
