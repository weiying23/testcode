# 编译与构建
## SHMEM编译
### 下载SHMEM源码

```shell
git clone https://gitcode.com/cann/shmem.git
```

您可自行选择需要的分支。

### 编译

进入shmem的根目录，编译

```shell
cd shmem
bash scripts/build.sh
```

更多命令介绍可查看shmem仓主目录下的`README.md`和`scripts/build.sh`文件。

### SHMEM编译相关说明
SHMEM的基本编译命令是`bash build.sh`，默认构建模式下生成版本信息，并创建安装包(默认情况下不会编译RDMA能力、用例、测试、python接口)。后可跟参数，实现不同功能：
- `-use_cxx11_abi1`：启用 `C++11 ABI`，默认
- `-use_cxx11_abi0`：禁用 `C++11 ABI`
- `-cann`：CANN 8.5以上版本可以使用CANN开放接口编译
- `-uttests`：构建`tests/unittest`目录下所有ut用例
- `-examples`：构建examples目录下所有用例
- `-python_example`：对部分examples目录下用例提供torch接入能力
- `-enable_rdma`：构建并启用RDMA相关能力
- `-enable_ascendc_dump`：启用`AscendC_Dump`模式，用于对算子内核代码进行调测
- `-package`：构建py扩展的whl包
    * 生成run包SHMEM\_{version}\_linux-{arch}.run，生成路径为{project_root}/package/{arch}/
    * 生成python whl包shmem-xxx.whl，生成路径为{project_root}/package/{arch}/
- `-python_extension`：生成python whl包shmem-xxx.whl，生成路径为{project_root}/dist/
- `-gendoc`：生成文档
- `-onlygendoc`: 生成文档，不构建源码
- `-debug`：设置构建类型为 `Debug` 模式
- `-mssanitizer`：example启用mssanitizer内存检测工具，脚本执行需用mssanitizer拉起任务才能实际生效，allgather样例运行脚本提供了tool选项使用工具。工具使用方法参见[异常检测工具（msSanitizer，MindStudio Sanitizer）](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/optool/atlasopdev_16_0039.html) 注：如果开启该选项请确保使用mssanitizer工具拉起算子，如使用其他方式拉起算子可能导致未知错误！
- `-soc_type`：如果SOC是Ascend950类别，需要增加`-soc_type Ascend950`参数，可以通过`npu-smi info`命令查看，其他SOC可不加该参数。
- `-enable_simt`：使能simt编程模式接口。
- `-full`：编译 package + python_extension + uttests + examples。SHMEM自动编译脚本，会自动完成依赖库的下载，工程编译，UT用例编译，库打包。
## SHMEM关键文件介绍
### `scripts`目录：
   - `install.sh`: 安装脚本
   - `uninstall.sh`: 卸载脚本
   - `build.sh`: 编译脚本
   - `release.sh`：全自动构建与打包脚本
   - `set_env.sh`：SHMEM的环境变量设置文件
### run.sh脚本使用
UT测试用例运行脚本。
```sh
bash scripts/run.sh
```
提供多种参数支持自定义用例执行
```sh
-ranks          # 总rank数
-frank          # 该服务器第一个rank
-ipport         # ip端口
-fnpu           # 每个服务器起的第一个npu
-gnpus          # 单机使用的卡数
-test_filter    # gtest_filter

# 例如
bash run.sh -ranks 4 -fnpu 2 -gnpus 4 -test_filter ScalarP # 会在2-6卡

```
### install.sh
打包生成的run包安装卸载依赖的脚本，提供安装卸载功能。
安装目录
```
${INSTALL_PATH}
    |--shmem
        |--latest
        |--${version}
            |--shmem
                |--include  (头文件)
                |--lib      (so库)
            |--scripts      (卸载脚本)
```
### uninstall.sh
卸载脚本，可以卸载对应路径安装的shmem库或通过run包的--uninstall卸载默认路径下的shmem。

### release.sh
出包脚本，编译后使用。对编译产物打包后会删除install目录下其他文件。推荐使用build.sh完成打包。


### 编译文件`build.sh`
文件名：`scripts/build.sh`  
SHMEM编译文件，一般无需更改。

### 环境变量设置文件`set_env.sh`
​**文件名**​：`scripts/set_env.sh`  
SHMEM安装完成后，提供进程级环境变量设置脚本`set_env.sh`，以自动完成环境变量设置，用户进程结束后自动失效。
