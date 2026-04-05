// sstep_gmres_core.h - Core s-step GMRES data structures and interface
#ifndef SSTEP_GMRES_CORE_H
#define SSTEP_GMRES_CORE_H

#include <vector>
#include <cmath>
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
            V[k].assign(s * n, 0.0);
        }

        W.resize(m + 1);
        for (int k = 0; k <= m; k++) {
            W[k].assign(s * s, 0.0);
        }

        H.assign((ms + 1) * ms, 0.0);
        g.assign(ms + 1, 0.0);
        cs.assign(ms, 0.0);
        sn.assign(ms, 0.0);
        givens_count = 0;

        tmp.assign(n, 0.0);
        Atmp.assign(n, 0.0);
    }
};

// Result structure
struct SstepResult {
    int iterations;     // Number of restart cycles
    int communications; // Number of global communications (MPI)
    double final_residual; // ||b - Ax|| / ||b||
    bool converged;
};

// Solve SPD system using Cholesky: A*x = b, where A is s x s
// A and b are overwritten; solution returned in b
void solve_spd_small(int n, double* A, double* b);

// Compute Gram matrix W_k: W[i,j] = <V_k^i, V_k^j>
void compute_gram_matrix(SstepWorkspace& ws, int k, int n, double scale = 1.0);

// Scalar1: Orthogonalize V_{k+1}^1 against all previous blocks V_0..V_k
// Uses W_i matrices to compute orthogonalization coefficients
// Returns: projection coefficients h stored in H matrix
void scalar1(SstepWorkspace& ws, int k, int n,
             const std::vector<double>& projections);

// Scalar2: Set up power basis structure in H matrix
// For power basis: H[col_j, col_j + 1] = 1.0 (identity mapping)
// This is a NO COMMUNICATION operation
void scalar2(SstepWorkspace& ws, int k);

// Apply Givens rotations to a column of H matrix
void apply_givens_to_column(SstepWorkspace& ws, int col, int nrots);

// Create new Givens rotation and apply to g vector
void create_givens_rotation(SstepWorkspace& ws, int row, int col);

// Back-solve the least squares problem using transformed H and g
void back_solve(SstepWorkspace& ws, std::vector<double>& y);

// Compute solution: x = x0 + sum_k V_k * y_k
void compute_solution(SstepWorkspace& ws, int n, int num_blocks,
                      const std::vector<double>& y,
                      double* x);

// Template function declarations (definitions must be in header)
template<typename MatVec, typename Precond>
void build_first_block(SstepWorkspace& ws,
                       const double* z,
                       int n,
                       MatVec mv,
                       Precond pc);

template<typename MatVec, typename Precond>
double sstep_arnoldi_block(SstepWorkspace& ws, int k, int n,
                           const std::vector<double>& projections,
                           double w_norm_sq,
                           MatVec mv,
                           Precond pc);

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
    // Safety check: if numerical issues cause negative norm_sq, recompute locally
    // In redundant computation mode, all processes have same data, so local = global
    if (norm_sq < 0 || std::isnan(norm_sq)) {
        norm_sq = vdot(n, &ws.V[k + 1][0], &ws.V[k + 1][0]);
        // Clear energy contribution since we're using local norm
        for (int pk = 0; pk <= k; pk++) {
            for (int j = 0; j < ws.s; j++) {
                double h = ws.H[(k * ws.s + ws.s - 1) * (ms + 1) + pk * ws.s + j];
                norm_sq -= h * h;
            }
        }
    }
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

    // Return residual estimate from the last g vector element
    // After s Givens rotations, residual estimate is at g[givens_count]
    return std::abs(ws.g[ws.givens_count]);
}

#endif // SSTEP_GMRES_CORE_H