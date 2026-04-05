// dist_vector.h - Distributed vector with ghost layer
#ifndef DIST_VECTOR_H
#define DIST_VECTOR_H

#include <vector>
#include <cstring>
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

#endif // DIST_VECTOR_H