# Twin Docker MPI 容器通信

通过 Docker 容器运行 MPI 程序，实现两个容器之间的数据传输和验证。

## 功能说明

本项目实现以下功能：

1. **容器隔离**：MPI 程序完全在 Docker 容器内运行，与宿主机 MPI 环境隔离
2. **容器间通信**：两个容器通过 MPI Send/Recv 进行数据传输
3. **数据验证**：发送和接收的数据与理论值比对，确保传输正确性
4. **自动清理**：测试完成后自动删除容器和网络，不影响宿主机环境

## 运行方法

```bash
# 构建镜像
make build

# 运行测试（默认 64KB）
./run_mpi.sh

# 自定义数据大小
./run_mpi.sh 131072

# 清理（保留镜像）
./cleanup.sh

# 清理（删除镜像）
./cleanup.sh --remove-image
```

## 文件说明

### Dockerfile

构建 Docker 镜像的配置文件。

```
FROM ubuntu:22.04
├── 安装 OpenMPI（mpirun, mpicc, 库文件）
├── 安装 OpenSSH（容器间免密访问）
├── 生成 SSH 密钥（id_rsa, id_rsa.pub）
├── 编译 mpi_app.c → /app/mpi_app
└── 启动 SSH 服务 + 保持容器运行
```

**关键配置：**
- `OMPI_ALLOW_RUN_AS_ROOT=1`：允许容器内以 root 运行 MPI
- SSH 密钥在构建时生成，运行时交换公钥实现免密访问

### mpi_app.c

MPI 数据传输程序，核心逻辑：

```
Rank 0（发送方）              Rank 1（接收方）
    │                              │
    │── 生成理论数据 pattern 0     │── 生成理论数据 pattern 1
    │                              │
    │──── MPI_Send(64KB) ──────────│── MPI_Recv
    │                              │── 与 pattern 0 比对验证
    │                              │── MPI_Send(确认数据)
    │── MPI_Recv                   │
    │── 与 pattern 1 比对验证       │
    │                              │
    └── 输出 PASSED                 └── 输出 PASSED
```

**数据验证原理：**
- 每个字节值 = `(rank + index) % 256`
- 例如 rank 0 的数据：`[0, 1, 2, 3, ..., 255, 0, 1, ...]`
- 接收方与理论 pattern 比对，逐字节验证

### run_mpi.sh

主启动脚本，分三个阶段执行：

**Phase 1 - 创建网络和容器：**
```bash
docker network create mpi_net          # 创建隔离网络
docker run --network=mpi_net ...       # 启动两个容器
```

**Phase 2 - 配置 SSH + 运行 MPI：**
```bash
# 步骤 1: 交换 SSH 公钥（实现免密）
CONTAINER_0_KEY → CONTAINER_1 的 authorized_keys
CONTAINER_1_KEY → CONTAINER_0 的 authorized_keys

# 步骤 2: 配置 SSH 跳过 host key 验证
StrictHostKeyChecking no

# 步骤 3: 创建 hostfile（告诉 MPI 有哪些节点）
172.19.0.2 slots=1    # 容器 0
172.19.0.3 slots=1    # 容器 1

# 步骤 4: 运行 MPI
mpirun --hostfile /tmp/hostfile -np 2 /app/mpi_app
```

**Phase 3 - 清理：**
```bash
docker rm -f mpi_container_0 mpi_container_1
docker network rm mpi_net
```

### cleanup.sh

清理脚本，提供两个选项：

```bash
./cleanup.sh             # 仅删除容器和网络
./cleanup.sh --remove-image  # 同时删除镜像
```

删除镜像后，宿主机 MPI 完全不受影响。

### Makefile

构建自动化：

```makefile
make build     # docker build -t mpi_app:latest .
make clean     # docker rmi mpi_app:latest
```

## 技术架构

```
宿主机
│
├── Docker 网络 (mpi_net)
│   │
│   ├── 容器 0 (mpi_container_0)
│   │   ├── IP: 172.19.0.2
│   │   ├── OpenMPI 4.1.2
│   │   ├── SSH 服务
│   │   └── /app/mpi_app (rank 0)
│   │
│   └── 容器 1 (mpi_container_1)
│       ├── IP: 172.19.0.3
│       ├── OpenMPI 4.1.2
│       ├── SSH 服务
│       └── /app/mpi_app (rank 1)
│
└── mpirun 通过 SSH 启动远程进程
    容器间 MPI 通过 TCP socket 通信
```

## 为什么需要 SSH？

OpenMPI 默认使用 SSH（plm_rsh）启动远程节点上的进程：

```
容器 0 的 mpirun
    │
    │── SSH 到 172.19.0.2 → 启动 rank 0 进程
    │── SSH 到 172.19.0.3 → 启动 rank 1 进程
    │
    └── rank 0 和 rank 1 通过 MPI 通信
```

所以需要：
1. 容器内运行 SSH 服务
2. 交换公钥实现免密登录
3. 跳过 host key 验证

## 运行结果示例

```
=== Twin Docker MPI Communication Test ===
Data size: 65536 bytes

Phase 1: Creating Docker network and starting containers...
Containers started
Container 0 IP: 172.19.0.2
Container 1 IP: 172.19.0.3

Phase 2: Running MPI test...
Process 0 started, world size: 2, data size: 65536 bytes
Rank 0: Sending 65536 bytes to rank 1...
Process 1 started, world size: 2, data size: 65536 bytes
Rank 1: Waiting for data from rank 0...
Verification PASSED: all 65536 bytes match
Rank 1: Received data verified!
Rank 1: Communication complete!
Verification PASSED: all 65536 bytes match
Rank 0: Communication complete and verified!

Phase 3: Cleanup...
Containers and network removed

=== Test Complete ===
```

## 前提条件

- Docker Desktop 已启动
- 宿主机已安装 OpenMPI（用于构建镜像时编译 mpi_app.c，实际 MPI 运行在容器内）