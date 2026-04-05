// sstep_gmres_dist.cpp - Distributed s-step GMRES solver
#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include "blas_utils.h"
#include "dist_vector.h"
#include "dist_matrix.h"
#include "dist_ilu.h"

// Global dot product (requires Allreduce)
double globalDot(MPI_Comm comm, const DistributedVector& a, const DistributedVector& b) {
    double local_sum = a.dotLocal(b);
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_sum;
}

double globalNorm(MPI_Comm comm, const DistributedVector& v) {
    return std::sqrt(globalDot(comm, v, v));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Parse arguments
    if (argc < 5) {
        if (rank == 0) {
            std::cout << "Usage: " << argv[0] << " <n_global> <s> <m> <type> [tol]\n";
            std::cout << "  n_global: global matrix dimension\n";
            std::cout << "  s: s-step parameter (2-3)\n";
            std::cout << "  m: number of blocks\n";
            std::cout << "  type: 0=five-diagonal, 1=anisotropic\n";
            std::cout << "  tol: convergence tolerance (default 1e-8)\n";
        }
        MPI_Finalize();
        return 1;
    }

    int n_global = std::atoi(argv[1]);
    int s = std::atoi(argv[2]);
    int m = std::atoi(argv[3]);
    int type = std::atoi(argv[4]);
    double tol = (argc > 5) ? std::atof(argv[5]) : 1e-8;

    // Clamp s to recommended range
    if (s < 2) s = 2;
    if (s > 3) s = 3;

    // Initialize distributed matrix
    DistributedCSRMatrix A;
    A.init(MPI_COMM_WORLD, n_global);

    std::string mat_name;
    if (type == 0) {
        A.buildFiveDiagonal(4.0, -1.0);
        mat_name = "Five-diagonal";
    } else {
        A.buildAnisotropic(0.01);
        mat_name = "Anisotropic(0.01)";
    }

    // Initialize halo exchange
    HaloExchange halo;
    A.setupHalo(halo);

    // ILU0 preconditioner
    DistributedILU0 M;
    M.factorize(A);

    if (rank == 0) {
        std::cout << "==============================================\n";
        std::cout << "Distributed s-step GMRES\n";
        std::cout << "==============================================\n";
        std::cout << "Matrix: " << mat_name << ", n_global=" << n_global << "\n";
        std::cout << "np=" << nprocs << ", s=" << s << ", m=" << m << "\n";
        std::cout << "tol=" << tol << "\n";
        std::cout << "==============================================\n\n";
    }

    // TODO: Solver implementation will go here

    MPI_Finalize();
    return 0;
}