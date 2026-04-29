/*
 * 3D Second-Order Derivative with 12th Order Finite Difference Scheme
 * Grid: 256 x 256 x 256
 * Stencil: 13 points in each direction (6 points on each side + center)
 *
 * This is the ORIGINAL (unoptimized) version for reference.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NX 256
#define NY 256
#define NZ 256

// Grid spacing
#define DX 1.0
#define DY 1.0
#define DZ 1.0

// 12th order central difference coefficients for 2nd derivative
// These coefficients are derived from Taylor series expansion
// For 2nd derivative: f''(x) ≈ (1/dx²) * Σ c[i] * f(x + i*dx)
// Coefficients for 12th order accuracy (13-point stencil)
const double coef_2nd_deriv_12th[7] = {
    -2.0555555555555556,   // c[0] - center point coefficient
     1.0285714285714286,   // c[1] - first neighbor
    -0.12857142857142856,  // c[2] - second neighbor
     0.025396825396825397, // c[3] - third neighbor
    -0.006349206349206349,// c[4] - fourth neighbor
     0.0009523809523809524,// c[5] - fifth neighbor
    -0.00007936507936507937 // c[6] - sixth neighbor
};

// Allocate 3D array (flattened to 1D for performance)
double* allocate_3d_array(int nx, int ny, int nz) {
    return (double*)malloc(nx * ny * nz * sizeof(double));
}

// Access 3D array element at (i, j, k)
#define IDX(arr, i, j, k, ny, nz) ((arr)[(i) * (ny) * (nz) + (j) * (nz) + (k)])

// Initialize with a test function: f(x,y,z) = sin(x) * sin(y) * sin(z)
void initialize_field(double *field, int nx, int ny, int nz) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                double x = i * DX;
                double y = j * DY;
                double z = k * DZ;
                IDX(field, i, j, k, ny, nz) = sin(x) * sin(y) * sin(z);
            }
        }
    }
}

// Compute 2nd derivative in X direction using 12th order scheme
// This is the UNOPTIMIZED version - straightforward but cache-unfriendly
void compute_d2dx_12th_original(double *input, double *output, int nx, int ny, int nz) {
    double inv_dx2 = 1.0 / (DX * DX);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                double result = 0.0;

                // Handle boundaries with lower-order schemes or zero-padding
                if (i < 6 || i >= nx - 6) {
                    // Boundary: use lower order or set to zero
                    result = 0.0;
                } else {
                    // Interior: full 12th order stencil
                    result = coef_2nd_deriv_12th[0] * IDX(input, i, j, k, ny, nz);

                    for (int d = 1; d <= 6; d++) {
                        result += coef_2nd_deriv_12th[d] * (
                            IDX(input, i - d, j, k, ny, nz) +
                            IDX(input, i + d, j, k, ny, nz)
                        );
                    }

                    result *= inv_dx2;
                }

                IDX(output, i, j, k, ny, nz) = result;
            }
        }
    }
}

// Compute 2nd derivative in Y direction using 12th order scheme
void compute_d2dy_12th_original(double *input, double *output, int nx, int ny, int nz) {
    double inv_dy2 = 1.0 / (DY * DY);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                double result = 0.0;

                if (j < 6 || j >= ny - 6) {
                    result = 0.0;
                } else {
                    result = coef_2nd_deriv_12th[0] * IDX(input, i, j, k, ny, nz);

                    for (int d = 1; d <= 6; d++) {
                        result += coef_2nd_deriv_12th[d] * (
                            IDX(input, i, j - d, k, ny, nz) +
                            IDX(input, i, j + d, k, ny, nz)
                        );
                    }

                    result *= inv_dy2;
                }

                IDX(output, i, j, k, ny, nz) = result;
            }
        }
    }
}

// Compute 2nd derivative in Z direction using 12th order scheme
void compute_d2dz_12th_original(double *input, double *output, int nx, int ny, int nz) {
    double inv_dz2 = 1.0 / (DZ * DZ);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                double result = 0.0;

                if (k < 6 || k >= nz - 6) {
                    result = 0.0;
                } else {
                    result = coef_2nd_deriv_12th[0] * IDX(input, i, j, k, ny, nz);

                    for (int d = 1; d <= 6; d++) {
                        result += coef_2nd_deriv_12th[d] * (
                            IDX(input, i, j, k - d, ny, nz) +
                            IDX(input, i, j, k + d, ny, nz)
                        );
                    }

                    result *= inv_dz2;
                }

                IDX(output, i, j, k, ny, nz) = result;
            }
        }
    }
}

// Compute Laplacian (sum of 2nd derivatives in all directions)
void compute_laplacian_12th_original(double *input, double *output, int nx, int ny, int nz) {
    double *temp1 = allocate_3d_array(nx, ny, nz);
    double *temp2 = allocate_3d_array(nx, ny, nz);

    // Compute d²/dx²
    compute_d2dx_12th_original(input, temp1, nx, ny, nz);

    // Compute d²/dy²
    compute_d2dy_12th_original(input, temp2, nx, ny, nz);

    // Add d²/dz² directly to output
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                double dz2;
                if (k < 6 || k >= nz - 6) {
                    dz2 = 0.0;
                } else {
                    dz2 = coef_2nd_deriv_12th[0] * IDX(input, i, j, k, ny, nz);
                    for (int d = 1; d <= 6; d++) {
                        dz2 += coef_2nd_deriv_12th[d] * (
                            IDX(input, i, j, k - d, ny, nz) +
                            IDX(input, i, j, k + d, ny, nz)
                        );
                    }
                    dz2 /= (DZ * DZ);
                }
                IDX(output, i, j, k, ny, nz) = IDX(temp1, i, j, k, ny, nz) +
                                               IDX(temp2, i, j, k, ny, nz) + dz2;
            }
        }
    }

    free(temp1);
    free(temp2);
}

// Verify result against analytical solution
// For f = sin(x)*sin(y)*sin(z), ∇²f = -3*sin(x)*sin(y)*sin(z)
double verify_laplacian(double *numerical, double *analytical, int nx, int ny, int nz) {
    double max_error = 0.0;
    double l2_error = 0.0;
    int count = 0;

    for (int i = 6; i < nx - 6; i++) {
        for (int j = 6; j < ny - 6; j++) {
            for (int k = 6; k < nz - 6; k++) {
                double error = fabs(IDX(numerical, i, j, k, ny, nz) -
                                   IDX(analytical, i, j, k, ny, nz));
                if (error > max_error) max_error = error;
                l2_error += error * error;
                count++;
            }
        }
    }

    l2_error = sqrt(l2_error / count);
    printf("Max error: %e\n", max_error);
    printf("L2 error: %e\n", l2_error);
    return max_error;
}

int main() {
    printf("3D 2nd Derivative - 12th Order Scheme (ORIGINAL VERSION)\n");
    printf("Grid size: %d x %d x %d = %.2f million points\n",
           NX, NY, NZ, (double)NX*NY*NZ/1e6);
    printf("Stencil: 13 points per direction (12th order accuracy)\n\n");

    // Allocate memory
    double *input = allocate_3d_array(NX, NY, NZ);
    double *output = allocate_3d_array(NX, NY, NZ);
    double *analytical = allocate_3d_array(NX, NY, NZ);

    // Initialize input field
    initialize_field(input, NX, NY, NZ);

    // Compute analytical Laplacian: -3*sin(x)*sin(y)*sin(z)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                double x = i * DX;
                double y = j * DY;
                double z = k * DZ;
                IDX(analytical, i, j, k, NY, NZ) = -3.0 * sin(x) * sin(y) * sin(z);
            }
        }
    }

    // Warm-up run
    printf("Running warm-up...\n");
    compute_laplacian_12th_original(input, output, NX, NY, NZ);

    // Benchmark
    int n_runs = 5;
    printf("Running %d benchmark iterations...\n\n", n_runs);

    clock_t start = clock();
    for (int run = 0; run < n_runs; run++) {
        compute_laplacian_12th_original(input, output, NX, NY, NZ);
    }
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double per_run = elapsed / n_runs;
    double gflops = (double)NX * NY * NZ * 13.0 * 3.0 * 2.0 / 1e9 / per_run;

    printf("Timing Results:\n");
    printf("  Total time: %.3f seconds\n", elapsed);
    printf("  Time per run: %.3f seconds\n", per_run);
    printf("  Performance: %.2f GFLOPS\n", gflops);
    printf("\n");

    // Verify accuracy
    printf("Verification (interior points only):\n");
    verify_laplacian(output, analytical, NX, NY, NZ);

    free(input);
    free(output);
    free(analytical);

    printf("\nDone.\n");

    return 0;
}
