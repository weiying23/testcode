// dist_vector.h - Distributed vector with ghost layer
#ifndef DIST_VECTOR_H
#define DIST_VECTOR_H

#include <vector>
#include <cstring>
#include <mpi.h>
#include "blas_utils.h"

class DistributedVector {
    int n_local_;       // Local component count
    int n_ghost_;       // Ghost layer component count
    std::vector<double> data_;  // Storage: [local(0..n_local-1) | ghost(0..n_ghost-1)]

public:
    DistributedVector() : n_local_(0), n_ghost_(0) {}

    DistributedVector(int n_local, int n_ghost)
        : n_local_(n_local), n_ghost_(n_ghost), data_(n_local + n_ghost, 0.0) {}

    void init(int n_local, int n_ghost) {
        n_local_ = n_local;
        n_ghost_ = n_ghost;
        data_.resize(n_local + n_ghost);
        zero();
    }

    int n_local() const { return n_local_; }
    int n_ghost() const { return n_ghost_; }
    int size() const { return n_local_ + n_ghost_; }

    // Access
    double& local(int i) { return data_[i]; }
    double& ghost(int i) { return data_[n_local_ + i]; }
    double local(int i) const { return data_[i]; }
    double ghost(int i) const { return data_[n_local_ + i]; }

    double* local_data() { return data_.data(); }
    double* ghost_data() { return data_.data() + n_local_; }
    const double* local_data() const { return data_.data(); }
    const double* ghost_data() const { return data_.data() + n_local_; }

    // Local BLAS operations
    void zero() { std::memset(data_.data(), 0, data_.size() * sizeof(double)); }

    void copyFromLocal(const double* src) {
        std::memcpy(local_data(), src, n_local_ * sizeof(double));
    }

    double dotLocal(const DistributedVector& other) const {
        return vdot(n_local_, local_data(), other.local_data());
    }

    double normLocal() const {
        return std::sqrt(dotLocal(*this));
    }

    void axpyLocal(double a, const DistributedVector& x) {
        vaxpy(n_local_, a, x.local_data(), local_data());
    }

    void scalLocal(double a) {
        vscal(n_local_, a, local_data());
    }
};

// HaloExchange - Non-blocking ghost layer exchange
class HaloExchange {
    MPI_Comm comm_;
    int rank_, nprocs_;

    int neighbor_left_, neighbor_right_;
    int n_send_left_, n_send_right_;
    int n_recv_left_, n_recv_right_;

    std::vector<double> send_buf_left_, send_buf_right_;
    std::vector<double> recv_buf_left_, recv_buf_right_;
    std::vector<int> send_idx_left_, send_idx_right_;

    MPI_Request req_send_[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Request req_recv_[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    bool exchange_started_;

public:
    HaloExchange() : neighbor_left_(-1), neighbor_right_(-1),
                     n_send_left_(0), n_send_right_(0),
                     n_recv_left_(0), n_recv_right_(0),
                     exchange_started_(false) {}

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

        send_buf_left_.resize(n_send_left_);
        send_buf_right_.resize(n_send_right_);
        recv_buf_left_.resize(n_recv_left_);
        recv_buf_right_.resize(n_recv_right_);

        exchange_started_ = false;
    }

    void setSendIndices(const std::vector<int>& idx_left, const std::vector<int>& idx_right) {
        send_idx_left_ = idx_left;
        send_idx_right_ = idx_right;
    }

    // Start non-blocking exchange
    void start_exchange(DistributedVector& vec) {
        if (exchange_started_) return;

        // Post receives first
        if (neighbor_left_ >= 0 && n_recv_left_ > 0) {
            MPI_Irecv(recv_buf_left_.data(), n_recv_left_, MPI_DOUBLE,
                      neighbor_left_, 0, comm_, &req_recv_[0]);
        }
        if (neighbor_right_ >= 0 && n_recv_right_ > 0) {
            MPI_Irecv(recv_buf_right_.data(), n_recv_right_, MPI_DOUBLE,
                      neighbor_right_, 1, comm_, &req_recv_[1]);
        }

        // Pack send data
        for (int i = 0; i < n_send_left_; i++) {
            send_buf_left_[i] = vec.local(send_idx_left_[i]);
        }
        for (int i = 0; i < n_send_right_; i++) {
            send_buf_right_[i] = vec.local(send_idx_right_[i]);
        }

        // Post sends
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

    // Wait for exchange to complete and unpack
    void wait_exchange(DistributedVector& vec) {
        if (!exchange_started_) return;

        // Wait for receives
        if (neighbor_left_ >= 0 && n_recv_left_ > 0) {
            MPI_Wait(&req_recv_[0], MPI_STATUS_IGNORE);
            for (int i = 0; i < n_recv_left_; i++) {
                vec.ghost(i) = recv_buf_left_[i];
            }
        }
        if (neighbor_right_ >= 0 && n_recv_right_ > 0) {
            MPI_Wait(&req_recv_[1], MPI_STATUS_IGNORE);
            for (int i = 0; i < n_recv_right_; i++) {
                vec.ghost(n_recv_left_ + i) = recv_buf_right_[i];
            }
        }

        // Wait for sends to complete
        if (neighbor_left_ >= 0 && n_send_left_ > 0) {
            MPI_Wait(&req_send_[0], MPI_STATUS_IGNORE);
        }
        if (neighbor_right_ >= 0 && n_send_right_ > 0) {
            MPI_Wait(&req_send_[1], MPI_STATUS_IGNORE);
        }

        exchange_started_ = false;
    }

    int n_recv_total() const { return n_recv_left_ + n_recv_right_; }
};

#endif // DIST_VECTOR_H