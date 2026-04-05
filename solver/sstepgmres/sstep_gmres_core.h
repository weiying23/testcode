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