// dist_matrix.h - Distributed CSR sparse matrix
#ifndef DIST_MATRIX_H
#define DIST_MATRIX_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include "dist_vector.h"
#include "blas_utils.h"

class DistributedCSRMatrix {
    MPI_Comm comm_;
    int rank_, nprocs_;

    int n_global_;      // Global matrix dimension
    int n_local_;       // Local row count
    int row_start_;     // Local starting row index
    int row_end_;       // Local ending row index

    // CSR storage (local rows only)
    std::vector<int> rowptr_;    // Local row pointers
    std::vector<int> colidx_;    // Global column indices
    std::vector<double> values_;

    // Ghost mapping
    std::vector<int> ghost_global_idx_;  // Global indices of ghost points
    std::vector<int> ghost_local_map_;   // Local index mapping for matvec

    // Halo info
    int neighbor_left_, neighbor_right_;
    int n_send_left_, n_send_right_;
    int n_recv_left_, n_recv_right_;
    std::vector<int> send_idx_left_, send_idx_right_;

public:
    DistributedCSRMatrix() : comm_(MPI_COMM_NULL), rank_(0), nprocs_(0),
                              n_global_(0), n_local_(0), row_start_(0), row_end_(-1),
                              neighbor_left_(-1), neighbor_right_(-1),
                              n_send_left_(0), n_send_right_(0),
                              n_recv_left_(0), n_recv_right_(0) {}

    int n_global() const { return n_global_; }
    int n_local() const { return n_local_; }
    int n_ghost() const { return ghost_global_idx_.size(); }
    int row_start() const { return row_start_; }
    int row_end() const { return row_end_; }

    // Initialize MPI communicator and compute partition
    void init(MPI_Comm comm, int n_global) {
        comm_ = comm;
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &nprocs_);

        n_global_ = n_global;

        // Compute local partition
        int base = n_global_ / nprocs_;
        int remainder = n_global_ % nprocs_;

        if (rank_ < remainder) {
            n_local_ = base + 1;
            row_start_ = rank_ * (base + 1);
        } else {
            n_local_ = base;
            row_start_ = remainder * (base + 1) + (rank_ - remainder) * base;
        }
        row_end_ = row_start_ + n_local_ - 1;

        // Determine neighbors
        neighbor_left_ = (rank_ > 0) ? rank_ - 1 : -1;
        neighbor_right_ = (rank_ < nprocs_ - 1) ? rank_ + 1 : -1;
    }

    // Setup halo exchange object (to be called after matrix build)
    void setupHalo(HaloExchange& halo) {
        halo.init(comm_, n_send_left_, n_send_right_,
                  n_recv_left_, n_recv_right_,
                  neighbor_left_, neighbor_right_);
        halo.setSendIndices(send_idx_left_, send_idx_right_);
    }
};

#endif // DIST_MATRIX_H