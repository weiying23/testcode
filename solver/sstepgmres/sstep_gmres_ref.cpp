// sstep_gmres_ref.cpp - Reference implementation with MPI communication
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>

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

        // Handle zero initial residual
        if (beta < 1e-15) {
            result.final_residual = 0.0;
            result.converged = true;
            return;
        }

        if (beta / bnorm_global < cfg.tol) {
            result.final_residual = beta / bnorm_global;
            result.converged = true;
            return;
        }

        // Normalize V_0
        if (beta > 1e-15) {
            for (int j = 0; j < cfg.s; j++) {
                vscal(n, 1.0 / beta, &ws.V[0][j * n]);
            }
        }

        // Extract W_0 (normalized)
        idx = 1;
        for (int i = 0; i < cfg.s; i++) {
            for (int j = 0; j < cfg.s; j++) {
                ws.W[0][i * cfg.s + j] = init_global[idx++] / (beta_sq);
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Parse arguments
    int n = (argc > 1) ? std::atoi(argv[1]) : 1000;
    int s = (argc > 2) ? std::atoi(argv[2]) : 3;
    int m = (argc > 3) ? std::atoi(argv[3]) : 10;
    int type = (argc > 4) ? std::atoi(argv[4]) : 0;

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