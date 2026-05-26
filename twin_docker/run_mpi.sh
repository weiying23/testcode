#!/bin/bash

#===============================================================================
# Twin Docker MPI 通信测试脚本
#
# 功能：启动两个 Docker 容器，通过 MPI 进行数据传输和验证
#
# 使用方法：
#   ./run_mpi.sh              # 默认 64KB 数据
#   ./run_mpi.sh 131072       # 自定义数据大小（如 128KB）
#
# 流程：
#   Phase 1: 创建 Docker 网络 + 启动容器
#   Phase 2: 配置 SSH 免密 + 运行 MPI 测试
#   Phase 3: 清理容器和网络
#===============================================================================

set -e  # 遇到错误立即退出

# 参数：数据大小（字节），默认 64KB
DATA_SIZE=${1:-65536}

echo "=== Twin Docker MPI Communication Test ==="
echo "Data size: ${DATA_SIZE} bytes"

#-------------------------------------------------------------------------------
# Phase 1: 创建 Docker 网络 + 启动容器
#-------------------------------------------------------------------------------
echo ""
echo "Phase 1: Creating Docker network and starting containers..."

# 清理可能残留的容器和网络（忽略错误）
docker rm -f mpi_container_0 mpi_container_1 2>/dev/null || true
docker network rm mpi_net 2>/dev/null || true

# 创建隔离的 Docker 网络，让两个容器能互相通信
docker network create mpi_net

# 在同一网络中启动两个容器
# --network=mpi_net: 使用刚创建的网络
# -d: 后台运行
# --name: 容器名称
docker run --network=mpi_net -d --name mpi_container_0 mpi_app:latest
docker run --network=mpi_net -d --name mpi_container_1 mpi_app:latest

echo "Containers started"
sleep 2  # 等待容器完全启动

# 获取容器的 IP 地址（用于 MPI hostfile）
CONTAINER_0_IP=$(docker inspect mpi_container_0 --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
CONTAINER_1_IP=$(docker inspect mpi_container_1 --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')

echo "Container 0 IP: ${CONTAINER_0_IP}"
echo "Container 1 IP: ${CONTAINER_1_IP}"

#-------------------------------------------------------------------------------
# Phase 2: 配置 SSH 免密访问 + 运行 MPI 测试
#-------------------------------------------------------------------------------
echo ""
echo "Phase 2: Running MPI test..."

# 步骤 2.1: 交换 SSH 公钥，实现容器间免密访问
# OpenMPI 默认使用 SSH 启动远程进程
CONTAINER_0_KEY=$(docker exec mpi_container_0 cat /root/.ssh/id_rsa.pub)
CONTAINER_1_KEY=$(docker exec mpi_container_1 cat /root/.ssh/id_rsa.pub)

# 将容器 1 的公钥添加到容器 0 的 authorized_keys
docker exec mpi_container_0 sh -c "echo '${CONTAINER_1_KEY}' >> /root/.ssh/authorized_keys"

# 将容器 0 的公钥添加到容器 1 的 authorized_keys
docker exec mpi_container_1 sh -c "echo '${CONTAINER_0_KEY}' >> /root/.ssh/authorized_keys"

# 步骤 2.2: 配置 SSH 跳过 host key 验证
# StrictHostKeyChecking no: 不验证远程主机指纹
# UserKnownHostsFile /dev/null: 不保存 host key
docker exec mpi_container_0 sh -c "cat > /root/.ssh/config << EOF
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF"
docker exec mpi_container_1 sh -c "cat > /root/.ssh/config << EOF
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF"

# 步骤 2.3: 创建 MPI hostfile
# hostfile 告诉 mpirun 有哪些节点可用
# slots=1 表示每个节点运行 1 个 MPI 进程
docker exec mpi_container_0 sh -c "cat > /tmp/hostfile << EOF
${CONTAINER_0_IP} slots=1
${CONTAINER_1_IP} slots=1
EOF"

docker exec mpi_container_1 sh -c "cat > /tmp/hostfile << EOF
${CONTAINER_0_IP} slots=1
${CONTAINER_1_IP} slots=1
EOF"

# 步骤 2.4: 从容器 0 运行 MPI 程序
# --allow-run-as-root: 允许以 root 运行（容器内默认是 root）
# --hostfile: 指定参与 MPI 的节点列表
# -np 2: 总共 2 个 MPI 进程
docker exec mpi_container_0 mpirun --allow-run-as-root \
    --hostfile /tmp/hostfile \
    -np 2 /app/mpi_app ${DATA_SIZE}

#-------------------------------------------------------------------------------
# Phase 3: 清理
#-------------------------------------------------------------------------------
echo ""
echo "Phase 3: Cleanup..."

# 删除容器和网络
docker rm -f mpi_container_0 mpi_container_1 2>/dev/null || true
docker network rm mpi_net 2>/dev/null || true
echo "Containers and network removed"

echo ""
echo "=== Test Complete ==="