// dist_vector.h - 分布式向量与幽灵层交换
// ============================================================
// 本文件定义分布式线性代数求解器中的向量数据结构和通信机制
//
// 核心概念：
// -----------
// 1. 分布式向量 (DistributedVector):
//    - 每个进程只存储部分数据 (local 部分)
//    - 需要邻居进程的数据用于矩阵-向量乘法 (ghost 部分)
//    - 数据布局: [local(0..n_local-1) | ghost(0..n_ghost-1)]
//
// 2. 幽灵层 (Ghost Layer/Halo):
//    - 矩阵-向量乘法时，边界行的计算需要邻居进程的数据
//    - 这些"外部"数据在本进程中称为幽灵点
//    - 通过 HaloExchange 类进行非阻塞 MPI 交换
//
// 3. 非阻塞通信:
//    - MPI_Isend/Irecv: 发送/接收不阻塞主进程
//    - MPI_Wait: 等待通信完成
//    - 好处: 可在等待时做其他计算 (通信-计算重叠)
// ============================================================

#ifndef DIST_VECTOR_H
#define DIST_VECTOR_H

#include <vector>
#include <cstring>
#include <mpi.h>
#include "blas_utils.h"

// ============================================================
// DistributedVector - 分布式向量类
// ============================================================
// 存储结构:
//   data_ = [local_0, local_1, ..., local_{n_local-1},
//            ghost_0, ghost_1, ..., ghost_{n_ghost-1}]
//
// 内存布局示例 (n_local=100, n_ghost=10):
//   索引 0-99:   本进程拥有的向量分量
//   索引 100-109: 从邻居进程接收的幽灵点数据
// ============================================================
class DistributedVector {
    int n_local_;       // 本进程拥有的分量数
    int n_ghost_;       // 幽灵层分量数 (来自邻居进程)
    std::vector<double> data_;  // 连续存储: [local | ghost]

public:
    DistributedVector() : n_local_(0), n_ghost_(0) {}

    DistributedVector(int n_local, int n_ghost)
        : n_local_(n_local), n_ghost_(n_ghost), data_(n_local + n_ghost, 0.0) {}

    // 初始化向量大小
    void init(int n_local, int n_ghost) {
        n_local_ = n_local;
        n_ghost_ = n_ghost;
        data_.resize(n_local + n_ghost);
        zero();
    }

    int n_local() const { return n_local_; }
    int n_ghost() const { return n_ghost_; }
    int size() const { return n_local_ + n_ghost_; }

    // ============================================================
    // 数据访问方法
    // ============================================================
    // local(i): 访问本进程的第 i 个分量
    // ghost(i): 访问幽灵层的第 i 个分量
    // ============================================================
    double& local(int i) { return data_[i]; }
    double& ghost(int i) { return data_[n_local_ + i]; }
    double local(int i) const { return data_[i]; }
    double ghost(int i) const { return data_[n_local_ + i]; }

    // 获取原始数据指针 (用于 BLAS 调用)
    double* local_data() { return data_.data(); }
    double* ghost_data() { return data_.data() + n_local_; }
    const double* local_data() const { return data_.data(); }
    const double* ghost_data() const { return data_.data() + n_local_; }

    // ============================================================
    // 局部 BLAS 操作
    // ============================================================
    // 注意: 这些操作只涉及 local 部分，不包含 ghost
    // 全局操作需要在主程序中调用 globalDot/globalNorm
    // ============================================================

    // 向量清零
    void zero() { std::memset(data_.data(), 0, data_.size() * sizeof(double)); }

    // 从本地数据复制
    void copyFromLocal(const double* src) {
        std::memcpy(local_data(), src, n_local_ * sizeof(double));
    }

    // 局部内积: sum = x · y (仅 local 部分)
    double dotLocal(const DistributedVector& other) const {
        return vdot(n_local_, local_data(), other.local_data());
    }

    // 局部范数: ||x||_2 = sqrt(x · x) (仅 local 部分)
    double normLocal() const {
        return std::sqrt(dotLocal(*this));
    }

    // 局部向量加法: y = y + a * x (仅 local 部分)
    void axpyLocal(double a, const DistributedVector& x) {
        vaxpy(n_local_, a, x.local_data(), local_data());
    }

    // 局部向量缩放: x = a * x (仅 local 部分)
    void scalLocal(double a) {
        vscal(n_local_, a, local_data());
    }
};

// ============================================================
// HaloExchange - 幽灵层交换类
// ============================================================
// 用于分布式矩阵-向量乘法中的边界数据交换
//
// 工作流程:
// 1. init(): 初始化邻居信息和缓冲区大小
// 2. setSendIndices(): 设置需要发送给邻居的本地索引
// 3. start_exchange(): 启动非阻塞 MPI 交换
//    - 先 post Irecv (准备接收)
//    - 打包发送数据
//    - post Isend (发送)
// 4. wait_exchange(): 等待完成并解包接收数据
//
// 通信模式:
// - 一维行分区: 每个进程最多有两个邻居 (左、右)
// - 左邻居: row_start-1 的进程 (负责更小的行索引)
// - 右邻居: row_end+1 的进程 (负责更大的行索引)
//
// MPI Tag 约定:
// - Tag 0: 从右邻居接收 / 发送给左邻居
// - Tag 1: 从左邻居接收 / 发送给右邻居
// ============================================================
class HaloExchange {
    MPI_Comm comm_;
    int rank_, nprocs_;

    // 邻居进程编号 (-1 表示无该邻居)
    int neighbor_left_, neighbor_right_;

    // 发送/接收计数
    int n_send_left_, n_send_right_;   // 发送给左/右邻居的数据量
    int n_recv_left_, n_recv_right_;   // 从左/右邻居接收的数据量

    // 通信缓冲区
    std::vector<double> send_buf_left_, send_buf_right_;
    std::vector<double> recv_buf_left_, recv_buf_right_;

    // 发送索引: 本地向量中哪些分量需要发送
    std::vector<int> send_idx_left_, send_idx_right_;

    // MPI 请求对象 (用于非阻塞通信)
    MPI_Request req_send_[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Request req_recv_[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    bool exchange_started_;  // 标记交换是否已开始

public:
    HaloExchange() : neighbor_left_(-1), neighbor_right_(-1),
                     n_send_left_(0), n_send_right_(0),
                     n_recv_left_(0), n_recv_right_(0),
                     exchange_started_(false) {}

    // ============================================================
    // init() - 初始化幽灵层交换
    // ============================================================
    // 参数:
    //   comm: MPI 通信器
    //   n_send_left/right: 发送给左/右邻居的数据量
    //   n_recv_left/right: 从左/右邻居接收的数据量
    //   left/right_rank: 邻居进程编号 (-1 表示无该邻居)
    // ============================================================
    void init(MPI_Comm comm, int n_send_left, int n_send_right,
              int n_recv_left, int n_recv_right,
              int left_rank, int right_rank) {
        comm_ = comm;
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &nprocs_);

        neighbor_left_ = left_rank;
        neighbor_right_ = right_rank;
        n_send_left_ = n_send_left;
        n_send_right_ = n_send_right;
        n_recv_left_ = n_recv_left;
        n_recv_right_ = n_recv_right;

        // 分配缓冲区
        send_buf_left_.resize(n_send_left_);
        send_buf_right_.resize(n_send_right_);
        recv_buf_left_.resize(n_recv_left_);
        recv_buf_right_.resize(n_recv_right_);

        exchange_started_ = false;
    }

    // 设置发送索引 (哪些本地分量需要发送)
    void setSendIndices(const std::vector<int>& idx_left, const std::vector<int>& idx_right) {
        send_idx_left_ = idx_left;
        send_idx_right_ = idx_right;
    }

    // ============================================================
    // start_exchange() - 启动非阻塞交换
    // ============================================================
    // 步骤:
    // 1. Post Irecv: 先发布接收请求 (避免死锁)
    // 2. 打包发送数据: 从本地向量提取需要发送的分量
    // 3. Post Isend: 发布发送请求
    //
    // 注意: 此函数调用后，通信在后台进行
    //       必须随后调用 wait_exchange() 完成交换
    // ============================================================
    void start_exchange(DistributedVector& vec) {
        if (exchange_started_) return;

        // 先发布接收请求 (标准 MPI 非阻塞通信模式)
        if (neighbor_left_ >= 0 && n_recv_left_ > 0) {
            MPI_Irecv(recv_buf_left_.data(), n_recv_left_, MPI_DOUBLE,
                      neighbor_left_, 0, comm_, &req_recv_[0]);
        }
        if (neighbor_right_ >= 0 && n_recv_right_ > 0) {
            MPI_Irecv(recv_buf_right_.data(), n_recv_right_, MPI_DOUBLE,
                      neighbor_right_, 1, comm_, &req_recv_[1]);
        }

        // 打包发送数据 (从本地向量提取)
        for (int i = 0; i < n_send_left_; i++) {
            send_buf_left_[i] = vec.local(send_idx_left_[i]);
        }
        for (int i = 0; i < n_send_right_; i++) {
            send_buf_right_[i] = vec.local(send_idx_right_[i]);
        }

        // 发布发送请求
        if (neighbor_left_ >= 0 && n_send_left_ > 0) {
            MPI_Isend(send_buf_left_.data(), n_send_left_, MPI_DOUBLE,
                      neighbor_left_, 1, comm_, &req_send_[0]);
        }
        if (neighbor_right_ >= 0 && n_send_right_ > 0) {
            MPI_Isend(send_buf_right_.data(), n_send_right_, MPI_DOUBLE,
                      neighbor_right_, 0, comm_, &req_send_[1]);
        }

        exchange_started_ = true;
    }

    // ============================================================
    // wait_exchange() - 等待交换完成并解包数据
    // ============================================================
    // 步骤:
    // 1. Wait for Irecv: 等待接收完成
    // 2. 解包数据: 将接收缓冲区写入幽灵层
    // 3. Wait for Isend: 等待发送完成 (确保缓冲区可重用)
    //
    // 幽灵层布局:
    //   ghost[0..n_recv_left-1]: 从左邻居接收的数据
    //   ghost[n_recv_left..n_recv_left+n_recv_right-1]: 从右邻居接收的数据
    // ============================================================
    void wait_exchange(DistributedVector& vec) {
        if (!exchange_started_) return;

        // 等待接收并解包
        if (neighbor_left_ >= 0 && n_recv_left_ > 0) {
            MPI_Wait(&req_recv_[0], MPI_STATUS_IGNORE);
            for (int i = 0; i < n_recv_left_; i++) {
                vec.ghost(i) = recv_buf_left_[i];
            }
        }
        if (neighbor_right_ >= 0 && n_recv_right_ > 0) {
            MPI_Wait(&req_recv_[1], MPI_STATUS_IGNORE);
            // 右邻居数据放在幽灵层后半部分
            for (int i = 0; i < n_recv_right_; i++) {
                vec.ghost(n_recv_left_ + i) = recv_buf_right_[i];
            }
        }

        // 等待发送完成
        if (neighbor_left_ >= 0 && n_send_left_ > 0) {
            MPI_Wait(&req_send_[0], MPI_STATUS_IGNORE);
        }
        if (neighbor_right_ >= 0 && n_send_right_ > 0) {
            MPI_Wait(&req_send_[1], MPI_STATUS_IGNORE);
        }

        exchange_started_ = false;
    }

    // 返回总接收数据量
    int n_recv_total() const { return n_recv_left_ + n_recv_right_; }
};

#endif // DIST_VECTOR_H