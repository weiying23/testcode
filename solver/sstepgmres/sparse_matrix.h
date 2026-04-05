// sparse_matrix.h - CSR sparse matrix and ILU0 preconditioner
#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <vector>
#include <algorithm>
#include <cmath>

class CSRMatrix {
    int n_;
    std::vector<int> rowptr_;
    std::vector<int> colidx_;
    std::vector<double> values_;

public:
    CSRMatrix(int n) : n_(n) {}

    int size() const { return n_; }
    const int* rows() const { return rowptr_.data(); }
    const int* cols() const { return colidx_.data(); }
    const double* vals() const { return values_.data(); }

    // Build standard 5-diagonal matrix (2D Laplacian-like)
    void buildFiveDiagonal(double diag_val, double offdiag_val) {
        rowptr_.resize(n_ + 1, 0);
        std::vector<std::vector<std::pair<int, double>>> rows(n_);
        int sq = static_cast<int>(std::sqrt(static_cast<double>(n_)));

        for (int i = 0; i < n_; i++) {
            rows[i].push_back({i, diag_val});
            if (i > 0) rows[i].push_back({i - 1, offdiag_val});
            if (i < n_ - 1) rows[i].push_back({i + 1, offdiag_val});
            if (i >= sq) rows[i].push_back({i - sq, offdiag_val});
            if (i + sq < n_) rows[i].push_back({i + sq, offdiag_val});
            std::sort(rows[i].begin(), rows[i].end());
            rowptr_[i + 1] = rowptr_[i] + rows[i].size();
        }

        colidx_.resize(rowptr_[n_]);
        values_.resize(rowptr_[n_]);
        int idx = 0;
        for (int i = 0; i < n_; i++) {
            for (auto& p : rows[i]) {
                colidx_[idx] = p.first;
                values_[idx++] = p.second;
            }
        }
    }

    // Build anisotropic diffusion matrix: -eps*u_xx - u_yy
    void buildAnisotropic(double eps) {
        rowptr_.resize(n_ + 1, 0);
        std::vector<std::vector<std::pair<int, double>>> rows(n_);
        int sq = static_cast<int>(std::sqrt(static_cast<double>(n_)));

        for (int i = 0; i < n_; i++) {
            int ix = i % sq;
            int iy = i / sq;
            rows[i].push_back({i, 2.0 + 2.0 * eps});
            if (ix > 0) rows[i].push_back({i - 1, -eps});
            if (ix < sq - 1) rows[i].push_back({i + 1, -eps});
            if (iy > 0) rows[i].push_back({i - sq, -1.0});
            if (iy < sq - 1) rows[i].push_back({i + sq, -1.0});
            std::sort(rows[i].begin(), rows[i].end());
            rowptr_[i + 1] = rowptr_[i] + rows[i].size();
        }

        colidx_.resize(rowptr_[n_]);
        values_.resize(rowptr_[n_]);
        int idx = 0;
        for (int i = 0; i < n_; i++) {
            for (auto& p : rows[i]) {
                colidx_[idx] = p.first;
                values_[idx++] = p.second;
            }
        }
    }

    // Matrix-vector product: y = A*x
    void mv(const double* x, double* y) const {
        for (int i = 0; i < n_; i++) {
            double sum = 0.0;
            for (int j = rowptr_[i]; j < rowptr_[i + 1]; j++) {
                sum += values_[j] * x[colidx_[j]];
            }
            y[i] = sum;
        }
    }
};

// ILU0 preconditioner (incomplete LU factorization with zero fill)
class ILU0 {
    int n_;
    std::vector<int> rowptr_;
    std::vector<int> colidx_;
    std::vector<double> lu_;

public:
    void factorize(const CSRMatrix& A) {
        n_ = A.size();
        int nnz = A.rows()[n_];
        rowptr_.assign(A.rows(), A.rows() + n_ + 1);
        colidx_.assign(A.cols(), A.cols() + nnz);
        lu_.assign(A.vals(), A.vals() + nnz);

        // In-place ILU0 factorization
        for (int i = 1; i < n_; i++) {
            for (int k = rowptr_[i]; k < rowptr_[i + 1]; k++) {
                int j = colidx_[k];
                if (j >= i) break;

                // Find diagonal element of row j
                double diag = 0.0;
                for (int p = rowptr_[j]; p < rowptr_[j + 1]; p++) {
                    if (colidx_[p] == j) {
                        diag = lu_[p];
                        break;
                    }
                }

                if (std::abs(diag) > 1e-14) {
                    lu_[k] /= diag;
                }

                // Update remaining elements
                for (int p = k + 1; p < rowptr_[i + 1]; p++) {
                    for (int q = rowptr_[j] + 1; q < rowptr_[j + 1]; q++) {
                        if (colidx_[q] == colidx_[p]) {
                            lu_[p] -= lu_[k] * lu_[q];
                            break;
                        }
                    }
                }
            }
        }
    }

    // Apply preconditioner: solve M*x = b (forward + backward solve)
    void apply(const double* b, double* x) const {
        std::vector<double> y(n_);

        // Forward solve: L*y = b
        for (int i = 0; i < n_; i++) {
            double sum = b[i];
            for (int k = rowptr_[i]; k < rowptr_[i + 1]; k++) {
                if (colidx_[k] < i) {
                    sum -= lu_[k] * y[colidx_[k]];
                }
            }
            y[i] = sum;
        }

        // Backward solve: U*x = y
        for (int i = n_ - 1; i >= 0; i--) {
            double sum = y[i];
            double diag = 1.0;
            for (int k = rowptr_[i]; k < rowptr_[i + 1]; k++) {
                if (colidx_[k] > i) {
                    sum -= lu_[k] * x[colidx_[k]];
                } else if (colidx_[k] == i) {
                    diag = lu_[k];
                }
            }
            if (std::abs(diag) > 1e-14) {
                x[i] = sum / diag;
            } else {
                x[i] = 0.0;  // Protection for near-zero diagonal
            }
        }
    }

    int size() const { return n_; }
};

#endif // SPARSE_MATRIX_H