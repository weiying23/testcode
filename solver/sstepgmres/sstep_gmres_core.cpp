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

// Build first block V_0: power basis from initial vector z
// V_0 = [z, MAz, MA^2z, ..., MA^(s-1)z]
// Requires: matvec function and preconditioner apply function
template<typename MatVec, typename Precond>
void build_first_block(SstepWorkspace& ws,
                       const double* z,
                       int n,
                       MatVec mv,
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
