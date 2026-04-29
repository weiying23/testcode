# 相关脚本

## build.sh
使用方式
```sh
bash build.sh
```
SHMEM自动编译脚本，会自动完成依赖库的下载，工程编译，UT用例编译，库打包

## install.sh
打包生成的run包安装卸载依赖的脚本，提供安装卸载功能。
### 安装目录
```
${INSTALL_PATH}
    |--shmem
        |--latest
        |--${version}
            |--shmem
                |--include  (头文件)
                |--lib      (so库)
            |--memfabric_hybrid
                |--include  (头文件)
                |--lib      (so库)
            |--scripts      (卸载脚本)
```
## run.sh
UT测试用例运行脚本。
```sh
bash run.sh
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

## set_env.sh
环境变量设置脚本。

## uninstall.sh
卸载脚本，可以卸载对应路径安装的shmem库或通过run包的--uninstall卸载默认路径下的shmem。

## release.sh
出包脚本，编译后使用。对编译产物打包后会删除install目录下其他文件。推荐使用build.sh完成打包。

