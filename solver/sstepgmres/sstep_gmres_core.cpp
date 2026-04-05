// sstep_gmres_core.cpp - Core s-step GMRES implementation
#include "sstep_gmres_core.h"
#include <vector>
#include <cmath>

// External LAPACK call - platform dependent
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern "C" void dposv_(char* uplo, int* n, int* nrhs, double* A, int* lda,
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
