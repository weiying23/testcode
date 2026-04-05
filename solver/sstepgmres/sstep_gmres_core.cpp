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
    if (info != 0) {
        // Cholesky failed - matrix not SPD, return zeros
        for (int i = 0; i < n; i++) b[i] = 0.0;
        return;
    }
    for (int i = 0; i < n; i++) b[i] = b_copy[i];
}

// Compute Gram matrix W_k: W[i,j] = <V_k^i, V_k^j>
void compute_gram_matrix(SstepWorkspace& ws, int k, int n, double scale) {
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
    int col_idx = k * ws.s + (ws.s - 1);

    int idx = 0;
    double energy = 0.0;

    for (int pk = 0; pk <= k; pk++) {
        // Get raw projections for this block
        std::vector<double> h_raw(ws.s);
        for (int j = 0; j < ws.s; j++) {
            h_raw[j] = projections[idx++];
        }

        // Solve W_k * h = projections (result in h_raw)
        solve_spd_small(ws.s, ws.W[pk].data(), h_raw.data());

        // Store in H matrix and update V_{k+1}^1
        for (int j = 0; j < ws.s; j++) {
            ws.H[col_idx * (ms + 1) + pk * ws.s + j] = h_raw[j];
            vaxpy(n, -h_raw[j], &ws.V[pk][j * n], &ws.V[k + 1][0]);
            energy += h_raw[j] * h_raw[j];
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
