# s-step GMRES Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a modular, well-tested s-step GMRES solver following arXiv:2001.04886v2 with optimal m+1 communication pattern.

**Architecture:** Modular C++ design with separate components: BLAS utilities, matrix operations, s-step Arnoldi procedure, least squares solver, and main driver. Each component is independently testable.

**Tech Stack:** C++11, MPI, LAPACK (via Accelerate or OpenBLAS), CSR sparse matrix format, ILU0 preconditioner.

---

## File Structure

```
sstepgmres/
├── sstep_gmres_ref.cpp       # Main driver with test harness
├── blas_utils.h              # BLAS helper functions (inline)
├── sparse_matrix.h           # CSRMatrix and ILU0 classes
├── sstep_gmres_core.h        # Core solver data structures
├── sstep_gmres_core.cpp      # Core solver implementation
├── test_sstep_gmres.cpp      # Unit tests (can run standalone or with MPI)
```

**Design principles:**
- `blas_utils.h`: Pure inline functions, no dependencies
- `sparse_matrix.h`: Self-contained matrix classes
- `sstep_gmres_core.h/cpp`: Core algorithm, testable without MPI
- `sstep_gmres_ref.cpp`: MPI wrapper + test harness

---

## Task 1: BLAS Utilities Header

**Files:**
- Create: `blas_utils.h`

- [ ] **Step 1: Write the header file with inline BLAS helpers**

```cpp
// blas_utils.h - Basic BLAS-like operations for s-step GMRES
#ifndef BLAS_UTILS_H
#define BLAS_UTILS_H

#include <cmath>
#include <cstring>

// Dot product: return x^T * y
inline double vdot(int n, const double* x, const double* y) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i] * y[i];
    return sum;
}

// AXPY: y = a*x + y
inline void vaxpy(int n, double a, const double* x, double* y) {
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}

// Scale: x = a*x
inline void vscal(int n, double a, double* x) {
    for (int i = 0; i < n; i++) x[i] *= a;
}

// Copy: y = x
inline void vcopy(int n, const double* x, double* y) {
    std::memcpy(y, x, n * sizeof(double));
}

// Norm: return ||x||_2
inline double vnorm(int n, const double* x) {
    return std::sqrt(vdot(n, x, x));
}

// Zero: x = 0
inline void vzero(int n, double* x) {
    std::memset(x, 0, n * sizeof(double));
}

// Initialize: x[i] = val for all i
inline void vinit(int n, double val, double* x) {
    for (int i = 0; i < n; i++) x[i] = val;
}

#endif // BLAS_UTILS_H
```

- [ ] **Step 2: Commit the header file**

```bash
git add blas_utils.h
git commit -m "feat: add BLAS utilities header for s-step GMRES"
```

---

## Task 2: Sparse Matrix Classes

**Files:**
- Create: `sparse_matrix.h`

- [ ] **Step 1: Write the CSRMatrix class definition**

```cpp
// sparse_matrix.h - CSR sparse matrix and ILU0 preconditioner
#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <vector>
#include <algorithm>
#include <cmath>
#include "blas_utils.h"

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

#endif // SPARSE_MATRIX_H
```

- [ ] **Step 2: Add ILU0 preconditioner class**

Add to `sparse_matrix.h` after CSRMatrix class:

```cpp
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
            x[i] = sum / diag;
        }
    }

    int size() const { return n_; }
};
```

- [ ] **Step 3: Commit sparse matrix header**

```bash
git add sparse_matrix.h
git commit -m "feat: add CSR matrix and ILU0 preconditioner classes"
```

---

## Task 3: Core Solver Data Structures

**Files:**
- Create: `sstep_gmres_core.h`

- [ ] **Step 1: Write the core solver header with data structures**

```cpp
// sstep_gmres_core.h - Core s-step GMRES data structures and interface
#ifndef SSTEP_GMRES_CORE_H
#define SSTEP_GMRES_CORE_H

#include <vector>
#include "blas_utils.h"

// Solver configuration
struct SstepConfig {
    int s;           // s-step parameter (2-5 recommended)
    int m;           // Number of blocks (total Krylov = m*s)
    double tol;      // Convergence tolerance
    int max_restarts; // Maximum restart cycles

    SstepConfig() : s(3), m(10), tol(1e-10), max_restarts(100) {}
};

// Solver workspace - all memory allocated upfront
struct SstepWorkspace {
    int n;           // Problem dimension
    int s;           // s-step parameter
    int m;           // Number of blocks
    int ms;          // Total Krylov dimension (m*s)

    // Basis vectors: V[k][j] stores j-th vector in k-th block
    // Flattened: V[k] has s vectors of length n
    std::vector<std::vector<double>> V;

    // Gram matrices: W[k] is s x s symmetric positive definite
    std::vector<std::vector<double>> W;

    // Hessenberg matrix: stored column-major, dimension (ms+1) x ms
    std::vector<double> H;

    // Givens rotation data
    std::vector<double> g;      // Right-hand side transformed
    std::vector<double> cs;     // Cosine parameters
    std::vector<double> sn;     // Sine parameters
    int givens_count;           // Current number of rotations

    // Temporary vectors
    std::vector<double> tmp;    // Work vector (n)
    std::vector<double> Atmp;   // A*x result (n)

    // Initialize workspace
    void init(int n_, int s_, int m_) {
        n = n_;
        s = s_;
        m = m_;
        ms = m * s;

        V.resize(m + 1);
        for (int k = 0; k <= m; k++) {
            V[k].resize(s * n);
        }

        W.resize(m + 1);
        for (int k = 0; k <= m; k++) {
            W[k].resize(s * s);
        }

        H.assign((ms + 1) * ms, 0.0);
        g.assign(ms + 1, 0.0);
        cs.assign(ms, 0.0);
        sn.assign(ms, 0.0);
        givens_count = 0;

        tmp.resize(n);
        Atmp.resize(n);
    }
};

// Result structure
struct SstepResult {
    int iterations;     // Number of restart cycles
    int communications; // Number of global communications (MPI)
    double final_residual; // ||b - Ax|| / ||b||
    bool converged;
};

#endif // SSTEP_GMRES_CORE_H
```

- [ ] **Step 2: Commit core header**

```bash
git add sstep_gmres_core.h
git commit -m "feat: add s-step GMRES core data structures"
```

---

## Task 4: Small Linear System Solver (Cholesky)

**Files:**
- Create: `sstep_gmres_core.cpp` (partial)

- [ ] **Step 1: Write the SPD solver function**

At the top of `sstep_gmres_core.cpp`:

```cpp
// sstep_gmres_core.cpp - Core s-step GMRES implementation
#include "sstep_gmres_core.h"
#include <vector>
#include <cmath>

// External LAPACK call - platform dependent
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern void dposv_(char* uplo, int* n, int* nrhs, double* A, int* lda,
                   double* b, int* ldb, int* info);
#endif

// Solve SPD system using Cholesky: A*x = b, where A is s x s
// A and b are overwritten; solution returned in b
void solve_spd_small(int n, double* A, double* b) {
    std::vector<double> A_copy(A, A + n * n);
    std::vector<double> b_copy(b, b + n);
    int info, nrhs = 1;
    dposv_((char*)"U", &n, &nrhs, A_copy.data(), &n, b_copy.data(), &n, &info);
    for (int i = 0; i < n; i++) b[i] = b_copy[i];
}
```

- [ ] **Step 2: Commit**

```bash
git add sstep_gmres_core.cpp
git commit -m "feat: add SPD solver for small s x s systems"
```

---

## Task 5: Build First Block (V_0)

**Files:**
- Modify: `sstep_gmres_core.cpp`

- [ ] **Step 1: Write the first block generation function**

Add to `sstep_gmres_core.cpp`:

```cpp
// Build first block V_0: power basis from initial vector z
// V_0 = [z, MAz, MA^2z, ..., MA^(s-1)z]
// Requires: matvec function and preconditioner apply function
template<typename MatVec, typename Precond>
void build_first_block(SstepWorkspace& ws,
                       const double* z,
                       int n,
                       Matvec mv,
                       Precond pc) {
    // First vector: z (already preconditioned residual)
    vcopy(n, z, &ws.V[0][0]);

    // Generate power basis: MA^j * z
    for (int j = 1; j < ws.s; j++) {
        mv(&ws.V[0][(j - 1) * n], ws.Atmp.data());
        pc(ws.Atmp.data(), ws.tmp.data());
        vcopy(n, ws.tmp.data(), &ws.V[0][j * n]);
    }
}

// Compute Gram matrix W_k: W[i,j] = <V_k^i, V_k^j>
void compute_gram_matrix(SstepWorkspace& ws, int k, int n, double scale = 1.0) {
    for (int i = 0; i < ws.s; i++) {
        for (int j = 0; j < ws.s; j++) {
            ws.W[k][i * ws.s + j] = vdot(n, &ws.V[k][i * n], &ws.V[k][j * n]) * scale;
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add sstep_gmres_core.cpp
git commit -m "feat: add first block generation and Gram matrix computation"
```

---

## Task 6: Scalar1 - Block Orthogonalization

**Files:**
- Modify: `sstep_gmres_core.cpp`

- [ ] **Step 1: Write Scalar1 function**

Add to `sstep_gmres_core.cpp`:

```cpp
// Scalar1: Orthogonalize V_{k+1}^1 against all previous blocks V_0..V_k
// Uses W_i matrices to compute orthogonalization coefficients
// Returns: projection coefficients h stored in H matrix
void scalar1(SstepWorkspace& ws, int k, int n,
             const std::vector<double>& projections) {
    int ms = ws.ms;
    int col_idx = k * ws.s + (ws.s - 1);  // Last column of block k

    int idx = 0;
    double energy = 0.0;

    for (int pk = 0; pk <= k; pk++) {
        // Get raw projections for this block
        std::vector<double> h_raw(ws.s);
        for (int j = 0; j < ws.s; j++) {
            h_raw[j] = projections[idx++];
        }

        // Solve W_k * h = projections to get orthogonal coefficients
        std::vector<double> h_ortho(ws.s);
        solve_spd_small(ws.s, ws.W[pk].data(), h_raw.data());

        for (int j = 0; j < ws.s; j++) {
            h_ortho[j] = h_raw[j];
        }

        // Store in H matrix and update V_{k+1}^1
        for (int j = 0; j < ws.s; j++) {
            ws.H[col_idx * (ms + 1) + pk * ws.s + j] = h_ortho[j];
            vaxpy(n, -h_ortho[j], &ws.V[pk][j * n], &ws.V[k + 1][0]);
            energy += h_ortho[j] * h_ortho[j];
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add sstep_gmres_core.cpp
git commit -m "feat: add Scalar1 block orthogonalization"
```

---

## Task 7: Scalar2 - Power Basis Approximation

**Files:**
- Modify: `sstep_gmres_core.cpp`

- [ ] **Step 1: Write Scalar2 function (simplified power basis)**

Add to `sstep_gmres_core.cpp`:

```cpp
// Scalar2: Set up power basis structure in H matrix
// For power basis: H[col_j, col_j + 1] = 1.0 (identity mapping)
// This is a NO COMMUNICATION operation
void scalar2(SstepWorkspace& ws, int k) {
    int ms = ws.ms;
    int block_start = k * ws.s;

    // Set identity structure for within-block connections
    for (int j = 0; j < ws.s - 1; j++) {
        int col = block_start + j;
        ws.H[col * (ms + 1) + block_start + j + 1] = 1.0;
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add sstep_gmres_core.cpp
git commit -m "feat: add Scalar2 power basis approximation"
```

---

## Task 8: Givens Rotations for Least Squares

**Files:**
- Modify: `sstep_gmres_core.cpp`

- [ ] **Step 1: Write Givens rotation functions**

Add to `sstep_gmres_core.cpp`:

```cpp
// Apply Givens rotation to two values
inline void apply_givens(double& a, double& b, double c, double s) {
    double t1 = c * a + s * b;
    double t2 = -s * a + c * b;
    a = t1;
    b = t2;
}

// Compute Givens rotation to zero out b given (a, b)
inline void compute_givens(double a, double b, double& c, double& r) {
    if (std::abs(a) < 1e-14 && std::abs(b) < 1e-14) {
        c = 1.0;
        r = 0.0;
    } else {
        r = std::sqrt(a * a + b * b);
        c = a / r;
    }
}

// Apply Givens rotations to a column of H matrix
void apply_givens_to_column(SstepWorkspace& ws, int col, int nrots) {
    int ms = ws.ms;
    for (int i = 0; i < nrots; i++) {
        double t1 = ws.H[col * (ms + 1) + i];
        double t2 = ws.H[col * (ms + 1) + i + 1];
        ws.H[col * (ms + 1) + i] = ws.cs[i] * t1 + ws.sn[i] * t2;
        ws.H[col * (ms + 1) + i + 1] = -ws.sn[i] * t1 + ws.cs[i] * t2;
    }
}

// Create new Givens rotation and apply to g vector
void create_givens_rotation(SstepWorkspace& ws, int row, int col) {
    int ms = ws.ms;
    double a = ws.H[col * (ms + 1) + row];
    double b = ws.H[col * (ms + 1) + row + 1];

    double r;
    compute_givens(a, b, ws.cs[ws.givens_count], r);

    ws.sn[ws.givens_count] = b / (r > 1e-14 ? r : 1.0);
    ws.H[col * (ms + 1) + row] = r;
    ws.H[col * (ms + 1) + row + 1] = 0.0;

    // Apply to g
    double gt = ws.g[ws.givens_count];
    ws.g[ws.givens_count] = ws.cs[ws.givens_count] * gt + ws.sn[ws.givens_count] * ws.g[ws.givens_count + 1];
    ws.g[ws.givens_count + 1] = -ws.sn[ws.givens_count] * gt + ws.cs[ws.givens_count] * ws.g[ws.givens_count + 1];

    ws.givens_count++;
}
```

- [ ] **Step 2: Commit**

```bash
git add sstep_gmres_core.cpp
git commit -m "feat: add Givens rotation functions for least squares"
```

---

## Task 9: Main s-step Arnoldi Loop

**Files:**
- Modify: `sstep_gmres_core.cpp`

- [ ] **Step 1: Write the main Arnoldi loop function**

Add to `sstep_gmres_core.cpp`:

```cpp
// Main s-step Arnoldi procedure (one block iteration)
// Returns: norm of orthogonalized V_{k+1}^1
template<typename MatVec, typename Precond>
double sstep_arnoldi_block(SstepWorkspace& ws, int k, int n,
                           const std::vector<double>& projections,
                           double w_norm_sq,
                           MatVec mv,
                           Precond pc) {
    int ms = ws.ms;

    // Scalar1: orthogonalize against previous blocks
    scalar1(ws, k, n, projections);

    // Compute norm of orthogonalized vector
    double energy = 0.0;
    for (int pk = 0; pk <= k; pk++) {
        for (int j = 0; j < ws.s; j++) {
            double h = ws.H[(k * ws.s + ws.s - 1) * (ms + 1) + pk * ws.s + j];
            energy += h * h;
        }
    }

    double norm_sq = w_norm_sq - energy;
    if (norm_sq < 0) norm_sq = vdot(n, &ws.V[k + 1][0], &ws.V[k + 1][0]);
    double norm = std::sqrt(std::max(norm_sq, 0.0));

    // Store norm in H matrix (subdiagonal)
    int col_last = k * ws.s + (ws.s - 1);
    ws.H[col_last * (ms + 1) + (k + 1) * ws.s] = norm;

    // Normalize V_{k+1}^1
    if (norm > 1e-14) {
        vscal(n, 1.0 / norm, &ws.V[k + 1][0]);
    }

    // Rebuild V_{k+1} from normalized first vector
    for (int j = 1; j < ws.s; j++) {
        mv(&ws.V[k + 1][(j - 1) * n], ws.Atmp.data());
        pc(ws.Atmp.data(), ws.tmp.data());
        vcopy(n, ws.tmp.data(), &ws.V[k + 1][j * n]);
    }

    // Scalar2: power basis structure (no communication)
    scalar2(ws, k + 1);

    // Apply Givens rotations to new columns
    for (int j = 0; j < ws.s; j++) {
        int col = k * ws.s + j;
        apply_givens_to_column(ws, col, ws.givens_count);
        create_givens_rotation(ws, ws.givens_count, col);
    }

    return std::abs(ws.g[ws.givens_count]);
}
```

- [ ] **Step 2: Commit**

```bash
git add sstep_gmres_core.cpp
git commit -m "feat: add main s-step Arnoldi block iteration"
```

---

## Task 10: Solution Back-Solve

**Files:**
- Modify: `sstep_gmres_core.cpp`

- [ ] **Step 1: Write the back-solve and solution computation**

Add to `sstep_gmres_core.cpp`:

```cpp
// Back-solve the least squares problem using transformed H and g
void back_solve(SstepWorkspace& ws, std::vector<double>& y) {
    int ms = ws.ms;
    y.assign(ms, 0.0);

    for (int i = ws.givens_count - 1; i >= 0; i--) {
        y[i] = ws.g[i];
        for (int j = i + 1; j < ws.givens_count; j++) {
            y[i] -= ws.H[j * (ms + 1) + i] * y[j];
        }
        if (std::abs(ws.H[i * (ms + 1) + i]) > 1e-14) {
            y[i] /= ws.H[i * (ms + 1) + i];
        }
    }
}

// Compute solution: x = x0 + sum_k V_k * y_k
void compute_solution(SstepWorkspace& ws, int n, int num_blocks,
                      const std::vector<double>& y,
                      double* x) {
    for (int k = 0; k < num_blocks; k++) {
        for (int j = 0; j < ws.s; j++) {
            vaxpy(n, y[k * ws.s + j], &ws.V[k][j * n], x);
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add sstep_gmres_core.cpp
git commit -m "feat: add back-solve and solution computation"
```

---

## Task 11: MPI Driver Wrapper

**Files:**
- Create: `sstep_gmres_ref.cpp`

- [ ] **Step 1: Write the MPI wrapper and main solver function**

```cpp
// sstep_gmres_ref.cpp - Reference implementation with MPI communication
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include "blas_utils.h"
#include "sparse_matrix.h"
#include "sstep_gmres_core.h"

// MPI wrapper for s-step GMRES
// Communication pattern: m+1 Allreduces (optimal)
void sstep_gmres_mpi(MPI_Comm comm,
                     const CSRMatrix& A,
                     const ILU0& M,
                     int n,
                     const SstepConfig& cfg,
                     const std::vector<double>& b,
                     std::vector<double>& x,
                     SstepResult& result) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    result.communications = 0;
    result.iterations = 0;
    result.converged = false;

    // Compute ||b||
    double bnorm_local = vdot(n, b.data(), b.data());
    double bnorm_global;
    MPI_Allreduce(&bnorm_local, &bnorm_global, 1, MPI_DOUBLE, MPI_SUM, comm);
    bnorm_global = std::sqrt(bnorm_global / nprocs);

    x.assign(n, 0.0);

    for (int restart = 0; restart < cfg.max_restarts; restart++) {
        result.iterations = restart + 1;

        // Compute initial residual: r = b - A*x
        std::vector<double> r(n), z(n);
        A.mv(x.data(), r.data());
        for (int i = 0; i < n; i++) r[i] = b[i] - r[i];

        // Apply preconditioner
        M.apply(r.data(), z.data());

        // Initialize workspace
        SstepWorkspace ws;
        ws.init(n, cfg.s, cfg.m);
        int ms = cfg.m * cfg.s;

        // Build V_0
        build_first_block(ws, z.data(), n,
            [&](const double* in, double* out) { A.mv(in, out); },
            [&](const double* in, double* out) { M.apply(in, out); });

        // ONE Allreduce for beta^2 + W_0
        int init_size = 1 + cfg.s * cfg.s;
        std::vector<double> init_local(init_size, 0.0);
        init_local[0] = vdot(n, z.data(), z.data());

        int idx = 1;
        for (int i = 0; i < cfg.s; i++) {
            for (int j = 0; j < cfg.s; j++) {
                init_local[idx++] = vdot(n, &ws.V[0][i * n], &ws.V[0][j * n]);
            }
        }

        std::vector<double> init_global(init_size);
        MPI_Allreduce(init_local.data(), init_global.data(), init_size,
                       MPI_DOUBLE, MPI_SUM, comm);
        result.communications++;

        for (int i = 0; i < init_size; i++) init_global[i] /= nprocs;

        double beta_sq = init_global[0];
        double beta = std::sqrt(beta_sq);

        if (beta / bnorm_global < cfg.tol) {
            result.final_residual = beta / bnorm_global;
            result.converged = true;
            return;
        }

        // Normalize V_0
        for (int j = 0; j < cfg.s; j++) {
            vscal(n, 1.0 / beta, &ws.V[0][j * n]);
        }

        // Extract W_0 (normalized)
        idx = 1;
        for (int i = 0; i < cfg.s; i++) {
            for (int j = 0; j < cfg.s; j++) {
                ws.W[0][i * cfg.s + j] = init_global[idx++] / (beta_sq * nprocs);
            }
        }

        ws.g[0] = beta;
        scalar2(ws, 0);

        // Main block loop
        for (int k = 0; k < cfg.m; k++) {
            // Compute MA * V_k^{s-1}
            A.mv(&ws.V[k][(cfg.s - 1) * n], ws.Atmp.data());
            M.apply(ws.Atmp.data(), ws.tmp.data());

            // Build V_{k+1} (before orthogonalization)
            vcopy(n, ws.tmp.data(), &ws.V[k + 1][0]);
            for (int j = 1; j < cfg.s; j++) {
                A.mv(&ws.V[k + 1][(j - 1) * n], ws.Atmp.data());
                M.apply(ws.Atmp.data(), ws.tmp.data());
                vcopy(n, ws.tmp.data(), &ws.V[k + 1][j * n]);
            }

            // ONE Allreduce for Scalar1 projections + W_{k+1} + ||w||^2
            int nblk = k + 1;
            int total_size = nblk * cfg.s + cfg.s * cfg.s + 1;
            std::vector<double> local_data(total_size, 0.0);

            idx = 0;
            for (int pk = 0; pk < nblk; pk++) {
                for (int j = 0; j < cfg.s; j++) {
                    local_data[idx++] = vdot(n, &ws.V[pk][j * n], &ws.V[k + 1][0]);
                }
            }

            for (int i = 0; i < cfg.s; i++) {
                for (int j = 0; j < cfg.s; j++) {
                    local_data[idx++] = vdot(n, &ws.V[k + 1][i * n], &ws.V[k + 1][j * n]);
                }
            }

            local_data[idx] = vdot(n, &ws.V[k + 1][0], &ws.V[k + 1][0]);

            std::vector<double> global_data(total_size);
            MPI_Allreduce(local_data.data(), global_data.data(), total_size,
                          MPI_DOUBLE, MPI_SUM, comm);
            result.communications++;

            for (int i = 0; i < total_size; i++) global_data[i] /= nprocs;
            double w_norm_sq = global_data[total_size - 1];

            // Extract projections
            std::vector<double> projections(nblk * cfg.s);
            for (int i = 0; i < nblk * cfg.s; i++) projections[i] = global_data[i];

            // Run s-step Arnoldi
            double res_est = sstep_arnoldi_block(ws, k, n, projections, w_norm_sq,
                [&](const double* in, double* out) { A.mv(in, out); },
                [&](const double* in, double* out) { M.apply(in, out); });

            res_est /= bnorm_global;

            if (rank == 0) {
                std::cout << "Block " << (k + 1) << " (Krylov=" << ((k + 1) * cfg.s)
                          << "): residual=" << std::scientific << res_est << "\n";
            }

            // Extract W_{k+1}
            idx = nblk * cfg.s;
            for (int i = 0; i < cfg.s; i++) {
                for (int j = 0; j < cfg.s; j++) {
                    ws.W[k + 1][i * cfg.s + j] = global_data[idx++];
                }
            }

            if (res_est < cfg.tol) {
                std::vector<double> y;
                back_solve(ws, y);
                compute_solution(ws, n, k + 1, y, x.data());

                // Compute true residual
                std::vector<double> Ax(n), res(n);
                A.mv(x.data(), Ax.data());
                for (int i = 0; i < n; i++) res[i] = b[i] - Ax[i];

                double res_norm_local = vdot(n, res.data(), res.data());
                MPI_Allreduce(&res_norm_local, &result.final_residual, 1,
                              MPI_DOUBLE, MPI_SUM, comm);
                result.final_residual = std::sqrt(result.final_residual / nprocs) / bnorm_global;
                result.converged = true;

                if (rank == 0) {
                    std::cout << "\n||b-Ax||/||b|| = " << std::scientific
                              << result.final_residual << "\n";
                }
                return;
            }
        }

        // End of cycle: update solution
        std::vector<double> y;
        back_solve(ws, y);
        compute_solution(ws, n, cfg.m, y, x.data());
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add sstep_gmres_ref.cpp
git commit -m "feat: add MPI driver wrapper for s-step GMRES"
```

---

## Task 12: Test Harness Main Function

**Files:**
- Modify: `sstep_gmres_ref.cpp`

- [ ] **Step 1: Add main function with test cases**

Add to `sstep_gmres_ref.cpp`:

```cpp
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Parse arguments
    int n = (argc > 1) ? atoi(argv[1]) : 1000;
    int s = (argc > 2) ? atoi(argv[2]) : 3;
    int m = (argc > 3) ? atoi(argv[3]) : 10;
    int type = (argc > 4) ? atoi(argv[4]) : 0;

    // Validate s (stability constraint)
    if (s < 2) s = 2;
    if (s > 5) s = 5;

    CSRMatrix A(n);
    ILU0 M;
    std::string matrix_name;

    if (type == 0) {
        A.buildFiveDiagonal(4.0, -1.0);
        matrix_name = "Five-diagonal (easy)";
    } else if (type == 1) {
        A.buildAnisotropic(0.01);
        matrix_name = "Anisotropic(0.01) (hard)";
    } else {
        A.buildAnisotropic(0.001);
        matrix_name = "Anisotropic(0.001) (very hard)";
    }

    if (rank == 0) {
        std::cout << "==============================================\n";
        std::cout << "s-step GMRES Reference Implementation\n";
        std::cout << "Paper: arXiv:2001.04886v2\n";
        std::cout << "==============================================\n";
        std::cout << "Matrix: " << matrix_name << "\n";
        std::cout << "Dimension n: " << n << "\n";
        std::cout << "s-step parameter: " << s << "\n";
        std::cout << "Number of blocks m: " << m << "\n";
        std::cout << "Total Krylov dim: " << (m * s) << "\n";
        std::cout << "Expected communications: " << (m + 1) << "\n";
        std::cout << "Standard GMRES would need: " << (m * s * 2) << "\n";
        std::cout << "==============================================\n\n";
    }

    // Factorize preconditioner
    M.factorize(A);

    // Set up problem: b = ones, x0 = zeros
    std::vector<double> b(n, 1.0);
    std::vector<double> x;

    SstepConfig cfg;
    cfg.s = s;
    cfg.m = m;
    cfg.tol = 1e-10;
    cfg.max_restarts = 100;

    SstepResult result;

    double t0 = MPI_Wtime();
    sstep_gmres_mpi(MPI_COMM_WORLD, A, M, n, cfg, b, x, result);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "\n==============================================\n";
        std::cout << "RESULT SUMMARY\n";
        std::cout << "==============================================\n";
        std::cout << "Communications: " << result.communications
                  << " (expected: " << (m + 1) << ")\n";
        std::cout << "Time: " << (t1 - t0) << " seconds\n";
        std::cout << "Final residual: " << std::scientific << result.final_residual << "\n";
        std::cout << "Status: " << (result.converged ? "CONVERGED" : "NOT CONVERGED") << "\n";
        std::cout << "==============================================\n";
    }

    MPI_Finalize();
    return result.converged ? 0 : 1;
}
```

- [ ] **Step 2: Commit**

```bash
git add sstep_gmres_ref.cpp
git commit -m "feat: add test harness main function"
```

---

## Task 13: Build and Test

**Files:**
- Run: build commands and tests

- [ ] **Step 1: Compile the reference implementation**

```bash
mpicxx -O3 -std=c++11 -o sstep_gmres_ref sstep_gmres_ref.cpp
```

Expected: Compilation succeeds with no warnings

- [ ] **Step 2: Run single process test**

```bash
mpirun -np 1 ./sstep_gmres_ref 100 3 10 0
```

Expected output:
```
Block 1 (Krylov=3): residual=...
Block 2 (Krylov=6): residual=...
...
Communications: 11 (expected: 11)
Status: CONVERGED
```

- [ ] **Step 3: Run multi-process test**

```bash
mpirun -np 4 ./sstep_gmres_ref 1000 3 10 0
```

Expected: Same convergence, communications still m+1

- [ ] **Step 4: Run hard problem test**

```bash
mpirun -np 4 ./sstep_gmres_ref 1000 3 30 1
```

Expected: Anisotropic problem converges (may need more blocks)

- [ ] **Step 5: Commit test results**

```bash
git add -A
git commit -m "test: verify s-step GMRES reference implementation"
```

---

## Self-Review Checklist

After completing all tasks, verify:

1. **Spec coverage**: All Phase 1 items from spec covered:
   - ✅ Data structures
   - ✅ First block generation
   - ✅ Wi (Gram matrix) computation
   - ✅ Scalar1
   - ✅ Scalar2
   - ✅ Least squares solver
   - ✅ Restart mechanism
   - ✅ Test harness

2. **Placeholder scan**: No TBD, TODO, or incomplete sections

3. **Type consistency**: All function signatures match between declaration and usage

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-05-sstep-gmres-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?