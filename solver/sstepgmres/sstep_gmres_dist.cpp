// sstep_gmres_dist.cpp - Distributed s-step GMRES solver
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <mpi.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern "C" {
void dposv_(char* uplo, int* n, int* nrhs, double* a, int* lda, double* b, int* ldb, int* info);
}
#endif

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
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.5, 1.5);

    if (type == 0) {
        A.buildFiveDiagonal(4.0 + dis(gen), -0.5 * dis(gen));
        mat_name = "Easy";
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

    // ==========================================
    // Distributed s-step GMRES Implementation
    // ==========================================

    int n_local = A.n_local();
    int n_ghost = A.n_ghost();
    int ms = m * s;
    int ncomm = 0;

    // Setup vectors
    DistributedVector b(n_local, n_ghost);
    DistributedVector x(n_local, n_ghost);
    DistributedVector r(n_local, n_ghost);
    DistributedVector z(n_local, n_ghost);
    DistributedVector Atmp(n_local, n_ghost);
    DistributedVector tmp(n_local, n_ghost);

    // Initialize b = 1.0
    for (int i = 0; i < n_local; i++) {
        b.local(i) = 1.0;
    }

    // Compute initial residual r = b - A*x (x = 0 initially)
    x.zero();
    r.copyFromLocal(b.local_data());

    // Compute bnorm
    double bnorm_loc = b.dotLocal(b);
    double bnorm;
    MPI_Allreduce(&bnorm_loc, &bnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    bnorm = std::sqrt(bnorm / nprocs);

    // Storage for V blocks: V[k] has s vectors, each of size (n_local + n_ghost)
    std::vector<std::vector<double>> V(m + 1);
    for (int k = 0; k <= m; k++) {
        V[k].resize(s * n_local);
    }

    // W matrices: W[k] is s x s Gram matrix
    std::vector<std::vector<double>> W(m + 1);
    for (int k = 0; k <= m; k++) {
        W[k].resize(s * s);
    }

    // Hessenberg matrix H (ms+1 x ms, column-major)
    std::vector<double> H((ms + 1) * ms, 0.0);
    std::vector<double> g(ms + 1, 0.0);
    std::vector<double> cs(ms), sn(ms);
    int givens_count = 0;

    // Helper function: compute z = M^{-1} * r (local preconditioner apply)
    auto applyPrecond = [&](const double* r_in, double* z_out) {
        M.apply(r_in, z_out);
    };

    // Helper function: distributed mat-vec with halo exchange
    auto distributedMatVec = [&](DistributedVector& v_in, DistributedVector& v_out) {
        halo.start_exchange(v_in);
        halo.wait_exchange(v_in);
        A.mv(v_in.local_data(), v_in.ghost_data(), v_out.local_data());
    };

    // Compute initial residual and apply preconditioner
    // Since x = 0, r = b, z = M^{-1} * b
    applyPrecond(b.local_data(), z.local_data());

    // ==========================================
    // Build V_0: power basis (unnormalized)
    // ==========================================
    for (int i = 0; i < n_local; i++) {
        V[0][i] = z.local(i);
    }

    // Build power basis: V_0^{j} = A * V_0^{j-1} for j = 1 to s-1
    for (int j = 1; j < s; j++) {
        // Copy V[0][(j-1)*n_local, ...] to a DistributedVector for mat-vec
        for (int i = 0; i < n_local; i++) {
            tmp.local(i) = V[0][(j - 1) * n_local + i];
        }
        distributedMatVec(tmp, Atmp);
        applyPrecond(Atmp.local_data(), tmp.local_data());
        for (int i = 0; i < n_local; i++) {
            V[0][j * n_local + i] = tmp.local(i);
        }
    }

    // ==========================================
    // ONE Allreduce for: beta^2 + W_0 (unnormalized)
    // ==========================================
    int init_size = 1 + s * s;
    std::vector<double> init_loc(init_size, 0.0);
    init_loc[0] = vdot(n_local, z.local_data(), z.local_data());  // beta^2

    int idx = 1;
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
            init_loc[idx++] = vdot(n_local, &V[0][i * n_local], &V[0][j * n_local]);
        }
    }

    std::vector<double> init_glb(init_size);
    MPI_Allreduce(init_loc.data(), init_glb.data(), init_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ncomm++;

    double beta_sq = init_glb[0] / nprocs;
    double beta = std::sqrt(beta_sq);

    double final_res = 0.0;

    if (beta < tol) {
        final_res = beta / bnorm;
        if (rank == 0) {
            std::cout << "Converged at iteration 0\n";
            std::cout << "Final residual: " << std::scientific << final_res << "\n";
        }
        MPI_Finalize();
        return 0;
    }

    // Normalize V_0
    for (int j = 0; j < s; j++) {
        for (int i = 0; i < n_local; i++) {
            V[0][j * n_local + i] /= beta;
        }
    }

    // Extract and normalize W_0
    idx = 1;
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
            W[0][i * s + j] = init_glb[idx++] / (beta_sq * nprocs);
        }
    }

    g[0] = beta;

    // Scalar2 for V_0: power basis structure (NO COMMUNICATION)
    for (int j = 0; j < s - 1; j++) {
        H[j * (ms + 1) + j + 1] = 1.0;
    }

    // Helper function for SPD solve (for Scalar1 orthogonalization)
    auto solveSPD = [](int n, double* A, double* b_in, double* x_out) {
        std::vector<double> A_copy(A, A + n * n);
        std::vector<double> b_copy(b_in, b_in + n);
        int info, nrhs = 1;
        // Use Cholesky factorization
        dposv_((char*)"U", &n, &nrhs, A_copy.data(), &n, b_copy.data(), &n, &info);
        for (int i = 0; i < n; i++) x_out[i] = b_copy[i];
    };

    // ==========================================
    // Main loop: ONE Allreduce per iteration
    // ==========================================
    for (int k = 0; k < m; k++) {
        int nblk = k + 1;

        // Compute w = MA * V_k^{s-1}
        // First copy V[k][(s-1)*n_local, ...] to tmp
        for (int i = 0; i < n_local; i++) {
            tmp.local(i) = V[k][(s - 1) * n_local + i];
        }
        distributedMatVec(tmp, Atmp);
        applyPrecond(Atmp.local_data(), tmp.local_data());

        // Build V_{k+1} from w (before orthogonalization)
        for (int i = 0; i < n_local; i++) {
            V[k + 1][i] = tmp.local(i);
        }

        for (int j = 1; j < s; j++) {
            for (int i = 0; i < n_local; i++) {
                tmp.local(i) = V[k + 1][(j - 1) * n_local + i];
            }
            distributedMatVec(tmp, Atmp);
            applyPrecond(Atmp.local_data(), tmp.local_data());
            for (int i = 0; i < n_local; i++) {
                V[k + 1][j * n_local + i] = tmp.local(i);
            }
        }

        // ==========================================
        // ONE Allreduce for:
        // 1. Scalar1: projections of w onto V_0..V_k (nblk * s values)
        // 2. W_{k+1}: Gram matrix of V_{k+1} (s * s values)
        // 3. ||w||^2 (1 value)
        // Total: nblk*s + s*s + 1
        // ==========================================
        int total_size = nblk * s + s * s + 1;
        std::vector<double> local_data(total_size, 0.0);
        idx = 0;

        // Scalar1: project w onto all previous blocks
        for (int pk = 0; pk < nblk; pk++) {
            for (int j = 0; j < s; j++) {
                local_data[idx++] = vdot(n_local, &V[pk][j * n_local], &V[k + 1][0]);
            }
        }

        // W_{k+1}: Gram matrix of V_{k+1}
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                local_data[idx++] = vdot(n_local, &V[k + 1][i * n_local], &V[k + 1][j * n_local]);
            }
        }

        // ||w||^2
        local_data[idx] = vdot(n_local, &V[k + 1][0], &V[k + 1][0]);

        std::vector<double> global_data(total_size);
        MPI_Allreduce(local_data.data(), global_data.data(), total_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        ncomm++;

        for (int i = 0; i < total_size; i++) global_data[i] /= nprocs;
        double w_norm_sq = global_data[total_size - 1];

        // ==========================================
        // Process Scalar1: orthogonalize w against V_0..V_k
        // ==========================================
        idx = 0;
        double energy = 0;
        int col_last = k * s + (s - 1);

        for (int pk = 0; pk < nblk; pk++) {
            std::vector<double> h_raw(s), h_ortho(s);
            for (int j = 0; j < s; j++) h_raw[j] = global_data[idx++];

            solveSPD(s, W[pk].data(), h_raw.data(), h_ortho.data());

            for (int j = 0; j < s; j++) {
                H[col_last * (ms + 1) + pk * s + j] = h_ortho[j];
                vaxpy(n_local, -h_ortho[j], &V[pk][j * n_local], &V[k + 1][0]);
                energy += h_ortho[j] * h_ortho[j];
            }
        }

        double norm_sq = w_norm_sq - energy;
        if (norm_sq < 0) {
            norm_sq = vdot(n_local, &V[k + 1][0], &V[k + 1][0]);
        }
        double norm = std::sqrt(std::max(norm_sq, 0.0));
        H[col_last * (ms + 1) + (k + 1) * s] = norm;

        if (norm > 1e-14) {
            vscal(n_local, 1.0 / norm, &V[k + 1][0]);
        }

        // ==========================================
        // Extract W_{k+1} from global_data
        // ==========================================
        idx = nblk * s;
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                W[k + 1][i * s + j] = global_data[idx++];
            }
        }

        // Rebuild V_{k+1} from orthogonalized first vector
        for (int j = 1; j < s; j++) {
            for (int i = 0; i < n_local; i++) {
                tmp.local(i) = V[k + 1][(j - 1) * n_local + i];
            }
            distributedMatVec(tmp, Atmp);
            applyPrecond(Atmp.local_data(), tmp.local_data());
            for (int i = 0; i < n_local; i++) {
                V[k + 1][j * n_local + i] = tmp.local(i);
            }
        }

        // Scalar2 for V_{k+1}: power basis structure (NO COMMUNICATION)
        for (int j = 0; j < s - 1; j++) {
            int col_j = (k + 1) * s + j;
            H[col_j * (ms + 1) + (k + 1) * s + j + 1] = 1.0;
        }

        // ==========================================
        // Givens rotations
        // ==========================================
        for (int j = 0; j < s; j++) {
            int col = k * s + j;
            for (int i = 0; i < givens_count; i++) {
                double t1 = H[col * (ms + 1) + i];
                double t2 = H[col * (ms + 1) + i + 1];
                H[col * (ms + 1) + i] = cs[i] * t1 + sn[i] * t2;
                H[col * (ms + 1) + i + 1] = -sn[i] * t1 + cs[i] * t2;
            }

            int row = givens_count;
            double a = H[col * (ms + 1) + row];
            double b_val = H[col * (ms + 1) + row + 1];

            if (std::abs(a) < 1e-14 && std::abs(b_val) < 1e-14) {
                cs[givens_count] = 1.0;
                sn[givens_count] = 0.0;
            } else {
                double r = std::sqrt(a * a + b_val * b_val);
                cs[givens_count] = a / r;
                sn[givens_count] = b_val / r;
                H[col * (ms + 1) + row] = r;
                H[col * (ms + 1) + row + 1] = 0.0;
            }

            double gt = g[givens_count];
            g[givens_count] = cs[givens_count] * gt + sn[givens_count] * g[givens_count + 1];
            g[givens_count + 1] = -sn[givens_count] * gt + cs[givens_count] * g[givens_count + 1];
            givens_count++;
        }

        double res_est = std::abs(g[givens_count]) / bnorm;
        if (rank == 0) {
            std::cout << "Block " << (k + 1) << " (Krylov=" << ((k + 1) * s)
                      << "): residual=" << std::scientific << res_est << "\n";
        }

        if (res_est < tol) {
            // Compute solution
            std::vector<double> y(ms, 0.0);
            for (int i = givens_count - 1; i >= 0; i--) {
                y[i] = g[i];
                for (int j = i + 1; j < givens_count; j++) {
                    y[i] -= H[j * (ms + 1) + i] * y[j];
                }
                if (std::abs(H[i * (ms + 1) + i]) > 1e-14) {
                    y[i] /= H[i * (ms + 1) + i];
                }
            }

            // Update x = x + sum(V_k^j * y_{k*s + j})
            for (int kk = 0; kk <= k; kk++) {
                for (int j = 0; j < s; j++) {
                    vaxpy(n_local, y[kk * s + j], &V[kk][j * n_local], x.local_data());
                }
            }

            // True residual
            std::vector<double> Ax_local(n_local), res_local(n_local);
            halo.start_exchange(x);
            halo.wait_exchange(x);
            A.mv(x.local_data(), x.ghost_data(), Ax_local.data());
            for (int i = 0; i < n_local; i++) {
                res_local[i] = b.local(i) - Ax_local[i];
            }
            double res_norm_loc = vdot(n_local, res_local.data(), res_local.data());
            MPI_Allreduce(&res_norm_loc, &final_res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            final_res = std::sqrt(final_res / nprocs) / bnorm;

            if (rank == 0) {
                std::cout << "\n||b-Ax||/||b|| = " << std::scientific << final_res << "\n";
                std::cout << "Global communications: " << ncomm << " (expected: " << (m + 1) << ")\n";
            }
            MPI_Finalize();
            return 0;
        }
    }

    // Final solution (if not converged within m blocks)
    std::vector<double> y(ms, 0.0);
    for (int i = ms - 1; i >= 0; i--) {
        y[i] = g[i];
        for (int j = i + 1; j < ms; j++) {
            y[i] -= H[j * (ms + 1) + i] * y[j];
        }
        if (std::abs(H[i * (ms + 1) + i]) > 1e-14) {
            y[i] /= H[i * (ms + 1) + i];
        }
    }

    for (int k_iter = 0; k_iter < m; k_iter++) {
        for (int j = 0; j < s; j++) {
            vaxpy(n_local, y[k_iter * s + j], &V[k_iter][j * n_local], x.local_data());
        }
    }

    // True residual
    std::vector<double> Ax_local(n_local), res_local(n_local);
    halo.start_exchange(x);
    halo.wait_exchange(x);
    A.mv(x.local_data(), x.ghost_data(), Ax_local.data());
    for (int i = 0; i < n_local; i++) {
        res_local[i] = b.local(i) - Ax_local[i];
    }
    double res_norm_loc = vdot(n_local, res_local.data(), res_local.data());
    MPI_Allreduce(&res_norm_loc, &final_res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    final_res = std::sqrt(final_res / nprocs) / bnorm;

    if (rank == 0) {
        std::cout << "\n||b-Ax||/||b|| = " << std::scientific << final_res << "\n";
        std::cout << "Global communications: " << ncomm << " (expected: " << (m + 1) << ")\n";
        std::cout << "Did not converge within " << m << " blocks\n";
    }

    MPI_Finalize();
    return 0;
}