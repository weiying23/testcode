# 代码组织结构
## SHMEM组织结构
``` 
├── 3rdparty // 依赖的第三方库
├── docs     // 文档
├── examples // 使用样例
├── include  // 头文件
├── scripts  // 相关脚本
├── src      // 源代码
└── tests    // 测试用例
```

## include
```
include目录下的头文件是按照如下文件层级进行组织的
|── include
|    |── shmem_api.h                            // shmem所有对外API
|    |── device
|       |── low_level
|           |── shmem_device_low_level_rma.h    // device侧远端内存访问低阶接口
|        |── shmem_device_def.h                 // device侧定义的宏
|        |── shmem_device_rma.h                 // device侧远端内存访问接口
|        |── shmem_device_sync.h                // device侧同步接口
|        |── shmem_device_team.h                // device侧通信域管理接口
|    |── host
|        |── shmem_host_def.h                   // host侧定义的宏和数据类型
|        |── shmem_host_heap.h                  // host侧内存堆管理接口
|        |── shmem_host_init.h                  // host侧初始化接口
|        |── shmem_host_rma.h                   // host侧远端内存访问接口
|        |── shmem_host_sync.h                  // host侧同步接口
|        |── shmem_host_team.h                  // host侧通信域管理接口
|    |── host_device
|        |── shmem_types.h                      // host和device共用的数据类型
|    |── internal
|        |── device                             // device侧内部头文件
|        └── host_device                        // host侧内部头文件
```   
## src
```
|── src
|    |── device             // device侧接口实现
|    |── host           
│    │    ├─common          // host侧通用接口实现、如日志模块
│    │    ├─init            // host侧初始化接口实现
│    │    ├─mem             // host侧内存管理接口实现
│    │    ├─python_wrapper  // Py接口封装
│    │    ├─team            // host侧通信域管理接口实现
|    └── transport          // 建链相关内容
```
## examples
```
├─examples
│  ├─helloworld         // shmem简易调用示例
│  └─matmul_allreduce   // 通算融合算子实现样例
```
## tests
```
└─tests
    └─unittest
        ├─init  // 初始化接口单元测试
        ├─mem   // 内存管理接口单元测试
        ├─sync  // 同步管理接口单元测试
        └─team  // 通信域管理接口单元测试
```
## docs
```
├─docs
│    ├─api_demo.md              // shmem api调用demo
│    ├─code_organization.md     // 工程组织架构（本文件）
│    ├─example.md               // 使用样例
│    ├─quickstart.md            // 快速开始
│    ├─related_scripts.md       // 相关脚本介绍
|    ├─pythonAPI.md             // shmem python api列表
│    └─Troubleshooting_FAQs.md  // QA
```

## scripts
存放相关脚本。  
[脚本具体功能和使用](related_scripts.md)