// dist_ilu.h - Distributed ILU0 preconditioner
#ifndef DIST_ILU_H
#define DIST_ILU_H

#include <vector>
#include <cmath>
#include "dist_matrix.h"

class DistributedILU0 {
    int n_local_;
    int row_start_;

    std::vector<int> rowptr_;
    std::vector<int> colidx_;
    std::vector<double> lu_;
    std::vector<int> diag_idx_;  // Index of diagonal element in each row

public:
    DistributedILU0() : n_local_(0), row_start_(0) {}

    // Local ILU0 factorization (ignores cross-process fill)
    void factorize(const DistributedCSRMatrix& mat) {
        n_local_ = mat.n_local();
        row_start_ = mat.row_start();

        // Copy matrix structure
        int nnz = mat.rowptr_[mat.n_local()];
        rowptr_ = mat.rowptr_;
        colidx_ = mat.colidx_;
        lu_ = mat.values_;

        // Find diagonal indices
        diag_idx_.resize(n_local_);
        for (int i = 0; i < n_local_; i++) {
            for (int k = rowptr_[i]; k < rowptr_[i + 1]; k++) {
                if (colidx_[k] == row_start_ + i) {
                    diag_idx_[i] = k;
                    break;
                }
            }
        }

        // ILU0 factorization (only for local elements)
        for (int i = 1; i < n_local_; i++) {
            for (int k = rowptr_[i]; k < rowptr_[i + 1]; k++) {
                int j_global = colidx_[k];

                // Skip if column is outside local range
                if (j_global < row_start_) continue;

                int j_local = j_global - row_start_;
                if (j_local >= i) break;

                // Get diagonal of row j_local
                double diag = lu_[diag_idx_[j_local]];
                if (std::abs(diag) > 1e-14) {
                    lu_[k] /= diag;
                }

                // Update remaining elements (only local columns)
                for (int p = k + 1; p < rowptr_[i + 1]; p++) {
                    int col_p_global = colidx_[p];
                    if (col_p_global < row_start_) continue;

                    // Find matching element in row j_local
                    for (int q = rowptr_[j_local] + 1; q < rowptr_[j_local + 1]; q++) {
                        if (colidx_[q] == col_p_global) {
                            lu_[p] -= lu_[k] * lu_[q];
                            break;
                        }
                    }
                }
            }
        }
    }

    // Apply preconditioner: z = M^{-1} * r
    void apply(const double* r_local, double* z_local) {
        std::vector<double> y(n_local_);

        // Forward solve: L * y = r
        for (int i = 0; i < n_local_; i++) {
            double sum = r_local[i];
            for (int k = rowptr_[i]; k < diag_idx_[i]; k++) {
                int j_global = colidx_[k];
                if (j_global >= row_start_) {
                    sum -= lu_[k] * y[j_global - row_start_];
                }
            }
            y[i] = sum;
        }

        // Backward solve: U * z = y
        for (int i = n_local_ - 1; i >= 0; i--) {
            double sum = y[i];
            for (int k = diag_idx_[i] + 1; k < rowptr_[i + 1]; k++) {
                int j_global = colidx_[k];
                if (j_global >= row_start_ && j_global <= row_start_ + n_local_ - 1) {
                    sum -= lu_[k] * z_local[j_global - row_start_];
                }
            }
            double diag = lu_[diag_idx_[i]];
            if (std::abs(diag) > 1e-14) {
                z_local[i] = sum / diag;
            } else {
                z_local[i] = 0.0;
            }
        }
    }

    int n_local() const { return n_local_; }
};

#endif // DIST_ILU_H