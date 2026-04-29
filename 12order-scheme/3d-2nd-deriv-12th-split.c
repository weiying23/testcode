/*
 * 3D Second-Order Derivative with 12th Order Finite Difference Scheme
 * Grid: 256 x 256 x 256
 *
 * SPLIT VERSION: Computes d²/dx², d²/dy², d²/dz² separately and sums
 * This tests whether dimension splitting helps performance.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr, rw, locality) __builtin_prefetch((addr), (rw), (locality))
#else
#define PREFETCH(addr, rw, locality) ((void)0)
#endif

#define NX 256
#define NY 256
#define NZ 256

#define DX 1.0
#define DY 1.0
#define DZ 1.0

#define BLOCK_X 32
#define BLOCK_Y 32
#define BLOCK_Z 32

const double coef_2nd_deriv_12th[7] = {
    -2.0555555555555556,
     1.0285714285714286,
    -0.12857142857142856,
     0.025396825396825397,
    -0.006349206349206349,
     0.0009523809523809524,
    -0.00007936507936507937
};

#define PAD 8
#define IDX_PAD(arr, i, j, k, ny, nz) \
    ((arr)[((i) + PAD) * ((ny) + 2*PAD) * ((nz) + 2*PAD) + \
           ((j) + PAD) * ((nz) + 2*PAD) + ((k) + PAD)])

double* allocate_3d_array_padded(int nx, int ny, int nz) {
    size_t size = (size_t)(nx + 2*PAD) * (ny + 2*PAD) * (nz + 2*PAD);
    double *arr = (double*)malloc(size * sizeof(double));
    if (arr) {
        for (size_t i = 0; i < size; i++) arr[i] = 0.0;
    }
    return arr;
}

// Compute d²/dx² - X direction stencil
void compute_d2dx_12th(double *input, double *output, int nx, int ny, int nz) {
    double inv_dx2 = 1.0 / (DX * DX);
    int nz_pad = nz + 2*PAD;
    int ny_pad = ny + 2*PAD;
    int stride_x = ny_pad * nz_pad;  // X-direction stride

    double c0 = coef_2nd_deriv_12th[0];
    double c1 = coef_2nd_deriv_12th[1];
    double c2 = coef_2nd_deriv_12th[2];
    double c3 = coef_2nd_deriv_12th[3];
    double c4 = coef_2nd_deriv_12th[4];
    double c5 = coef_2nd_deriv_12th[5];
    double c6 = coef_2nd_deriv_12th[6];

    for (int ti = 6; ti < nx - 6; ti += BLOCK_X) {
        int bx_end = (ti + BLOCK_X < nx - 6) ? ti + BLOCK_X : nx - 6;
        for (int i = ti; i < bx_end; i++) {
            for (int j = 6; j < ny - 6; j++) {
                int base_idx = ((i + PAD) * ny_pad + (j + PAD)) * nz_pad + PAD;
                for (int k = 6; k < nz - 6; k++) {
                    int idx = base_idx + k;
                    double center = input[idx];
                    double result = c0 * center;
                    result += c1 * (input[idx - stride_x] + input[idx + stride_x]);
                    result += c2 * (input[idx - 2*stride_x] + input[idx + 2*stride_x]);
                    result += c3 * (input[idx - 3*stride_x] + input[idx + 3*stride_x]);
                    result += c4 * (input[idx - 4*stride_x] + input[idx + 4*stride_x]);
                    result += c5 * (input[idx - 5*stride_x] + input[idx + 5*stride_x]);
                    result += c6 * (input[idx - 6*stride_x] + input[idx + 6*stride_x]);
                    output[idx] = result * inv_dx2;
                }
            }
        }
    }
}

// Compute d²/dy² - Y direction stencil
void compute_d2dy_12th(double *input, double *output, int nx, int ny, int nz) {
    double inv_dy2 = 1.0 / (DY * DY);
    int nz_pad = nz + 2*PAD;
    int ny_pad = ny + 2*PAD;
    int stride_y = nz_pad;

    double c0 = coef_2nd_deriv_12th[0];
    double c1 = coef_2nd_deriv_12th[1];
    double c2 = coef_2nd_deriv_12th[2];
    double c3 = coef_2nd_deriv_12th[3];
    double c4 = coef_2nd_deriv_12th[4];
    double c5 = coef_2nd_deriv_12th[5];
    double c6 = coef_2nd_deriv_12th[6];

    for (int i = 6; i < nx - 6; i++) {
        for (int tj = 6; tj < ny - 6; tj += BLOCK_Y) {
            int by_end = (tj + BLOCK_Y < ny - 6) ? tj + BLOCK_Y : ny - 6;
            for (int j = tj; j < by_end; j++) {
                int base_idx = ((i + PAD) * ny_pad + (j + PAD)) * nz_pad + PAD;
                for (int k = 6; k < nz - 6; k++) {
                    int idx = base_idx + k;
                    double center = input[idx];
                    double result = c0 * center;
                    result += c1 * (input[idx - stride_y] + input[idx + stride_y]);
                    result += c2 * (input[idx - 2*stride_y] + input[idx + 2*stride_y]);
                    result += c3 * (input[idx - 3*stride_y] + input[idx + 3*stride_y]);
                    result += c4 * (input[idx - 4*stride_y] + input[idx + 4*stride_y]);
                    result += c5 * (input[idx - 5*stride_y] + input[idx + 5*stride_y]);
                    result += c6 * (input[idx - 6*stride_y] + input[idx + 6*stride_y]);
                    output[idx] = result * inv_dy2;
                }
            }
        }
    }
}

// Compute d²/dz² - Z direction stencil (contiguous access!)
void compute_d2dz_12th(double *input, double *output, int nx, int ny, int nz) {
    double inv_dz2 = 1.0 / (DZ * DZ);
    int nz_pad = nz + 2*PAD;
    int ny_pad = ny + 2*PAD;

    double c0 = coef_2nd_deriv_12th[0];
    double c1 = coef_2nd_deriv_12th[1];
    double c2 = coef_2nd_deriv_12th[2];
    double c3 = coef_2nd_deriv_12th[3];
    double c4 = coef_2nd_deriv_12th[4];
    double c5 = coef_2nd_deriv_12th[5];
    double c6 = coef_2nd_deriv_12th[6];

    // Z-direction: contiguous memory access - optimal for SIMD/prefetch
    for (int i = 6; i < nx - 6; i++) {
        for (int j = 6; j < ny - 6; j++) {
            int base_idx = ((i + PAD) * ny_pad + (j + PAD)) * nz_pad + PAD;
            // Prefetch ahead for contiguous access
            for (int k = 6; k < nz - 6; k += 8) {
                PREFETCH(&input[base_idx + k + 16], 0, 3);
            }
            for (int k = 6; k < nz - 6; k++) {
                int idx = base_idx + k;
                double center = input[idx];
                double result = c0 * center;
                result += c1 * (input[idx - 1] + input[idx + 1]);
                result += c2 * (input[idx - 2] + input[idx + 2]);
                result += c3 * (input[idx - 3] + input[idx + 3]);
                result += c4 * (input[idx - 4] + input[idx + 4]);
                result += c5 * (input[idx - 5] + input[idx + 5]);
                result += c6 * (input[idx - 6] + input[idx + 6]);
                output[idx] = result * inv_dz2;
            }
        }
    }
}

// Split Laplacian: compute each direction separately and sum
void compute_laplacian_12th_split(double *input, double *output, int nx, int ny, int nz) {
    double *temp = allocate_3d_array_padded(nx, ny, nz);

    // Pass 1: d²/dx² -> temp
    compute_d2dx_12th(input, temp, nx, ny, nz);

    // Pass 2: d²/dy² -> output
    compute_d2dy_12th(input, output, nx, ny, nz);

    // Pass 3: d²/dz², add d2x (from temp) and d2z to output
    int nz_pad = nz + 2*PAD;
    int ny_pad = ny + 2*PAD;
    double inv_dz2 = 1.0 / (DZ * DZ);
    double c0 = coef_2nd_deriv_12th[0];
    double c1 = coef_2nd_deriv_12th[1];
    double c2 = coef_2nd_deriv_12th[2];
    double c3 = coef_2nd_deriv_12th[3];
    double c4 = coef_2nd_deriv_12th[4];
    double c5 = coef_2nd_deriv_12th[5];
    double c6 = coef_2nd_deriv_12th[6];

    for (int i = 6; i < nx - 6; i++) {
        for (int j = 6; j < ny - 6; j++) {
            int base_idx = ((i + PAD) * ny_pad + (j + PAD)) * nz_pad + PAD;
            for (int k = 6; k < nz - 6; k++) {
                int idx = base_idx + k;
                double center = input[idx];
                double d2z = c0 * center;
                d2z += c1 * (input[idx - 1] + input[idx + 1]);
                d2z += c2 * (input[idx - 2] + input[idx + 2]);
                d2z += c3 * (input[idx - 3] + input[idx + 3]);
                d2z += c4 * (input[idx - 4] + input[idx + 4]);
                d2z += c5 * (input[idx - 5] + input[idx + 5]);
                d2z += c6 * (input[idx - 6] + input[idx + 6]);
                output[idx] += temp[idx] + d2z * inv_dz2;  // Add d2x + d2z to d2y
            }
        }
    }

    free(temp);
}

void initialize_field_padded(double *field, int nx, int ny, int nz) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                double x = i * DX;
                double y = j * DY;
                double z = k * DZ;
                IDX_PAD(field, i, j, k, ny, nz) = sin(x) * sin(y) * sin(z);
            }
        }
    }
}

double verify_laplacian_padded(double *numerical, double *analytical, int nx, int ny, int nz) {
    double max_error = 0.0;
    double l2_error = 0.0;
    int count = 0;
    for (int i = 6; i < nx - 6; i++) {
        for (int j = 6; j < ny - 6; j++) {
            for (int k = 6; k < nz - 6; k++) {
                double error = fabs(IDX_PAD(numerical, i, j, k, ny, nz) -
                                   IDX_PAD(analytical, i, j, k, ny, nz));
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
    printf("3D 2nd Derivative - 12th Order (SPLIT VERSION)\n");
    printf("Grid size: %d x %d x %d = %.2f million points\n", NX, NY, NZ, (double)NX*NY*NZ/1e6);
    printf("Approach: 3 separate 1D passes (d2dx + d2dy + d2dz)\n");
    printf("This tests if dimension splitting helps vs. fused approach.\n\n");

    double *input = allocate_3d_array_padded(NX, NY, NZ);
    double *output = allocate_3d_array_padded(NX, NY, NZ);
    double *analytical = allocate_3d_array_padded(NX, NY, NZ);

    if (!input || !output || !analytical) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    initialize_field_padded(input, NX, NY, NZ);

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                double x = i * DX;
                double y = j * DY;
                double z = k * DZ;
                IDX_PAD(analytical, i, j, k, NY, NZ) = -3.0 * sin(x) * sin(y) * sin(z);
            }
        }
    }

    printf("Running warm-up...\n");
    compute_laplacian_12th_split(input, output, NX, NY, NZ);

    int n_runs = 5;
    printf("Running %d benchmark iterations...\n\n", n_runs);

    clock_t start = clock();
    for (int run = 0; run < n_runs; run++) {
        compute_laplacian_12th_split(input, output, NX, NY, NZ);
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

    printf("Verification (interior points only):\n");
    verify_laplacian_padded(output, analytical, NX, NY, NZ);

    free(input);
    free(output);
    free(analytical);

    printf("\nDone.\n");

    return 0;
}
