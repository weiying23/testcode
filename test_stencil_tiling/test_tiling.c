#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define NX 256
#define NY 256
#define NZ 256
#define TILE_X 12
#define TILE_Y 16
#define TILE_Z 256
#define ORDER 12
#define RADIUS 6

// 分配三维数组
float*** allocate_3d_array(int nx, int ny, int nz) {
    float*** array = (float***)malloc(nx * sizeof(float**));
    for (int i = 0; i < nx; i++) {
        array[i] = (float**)malloc(ny * sizeof(float*));
        for (int j = 0; j < ny; j++) {
            array[i][j] = (float*)malloc(nz * sizeof(float));
        }
    }
    return array;
}

// 分配一维数组（以三维方式访问）
float* allocate_1d_array(int nx, int ny, int nz) {
    return (float*)malloc(nx * ny * nz * sizeof(float));
}

// 释放三维数组
void free_3d_array(float*** array, int nx, int ny) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            free(array[i][j]);
        }
        free(array[i]);
    }
    free(array);
}

// 初始化三维数组（使用一维数组表示）
void initialize_1d(float* data, int nx, int ny, int nz) {
    srand(time(NULL));
    int total = nx * ny * nz;
    for (int i = 0; i < total; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

// 初始化三维数组（使用三维数组表示）
void initialize_3d(float*** data, int nx, int ny, int nz) {
    srand(time(NULL));
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                data[i][j][k] = (float)rand() / RAND_MAX;
            }
        }
    }
}

// 基础版本（无优化）- 使用一维数组
void stencil_baseline_1d(float* input, float* output) {
    for (int x = RADIUS; x < NX - RADIUS; ++x) {
        for (int y = RADIUS; y < NY - RADIUS; ++y) {
            for (int z = RADIUS; z < NZ - RADIUS; ++z) {
                float sum = 0.0f;
                int idx = x * NY * NZ + y * NZ + z;
                
                // 十二阶 stencil (x方向)
                for (int k = -RADIUS; k <= RADIUS; ++k) {
                    sum += input[idx + k * NY * NZ];
                }
                // 十二阶 stencil (y方向)
                for (int k = -RADIUS; k <= RADIUS; ++k) {
                    sum += input[idx + k * NZ];
                }
                // 十二阶 stencil (z方向)
                for (int k = -RADIUS; k <= RADIUS; ++k) {
                    sum += input[idx + k];
                }
                
                output[idx] = sum;
            }
        }
    }
}

// 基础版本（无优化）- 使用三维数组
void stencil_baseline_3d(float*** input, float*** output) {
    for (int x = RADIUS; x < NX - RADIUS; ++x) {
        for (int y = RADIUS; y < NY - RADIUS; ++y) {
            for (int z = RADIUS; z < NZ - RADIUS; ++z) {
                float sum = 0.0f;
                
                // 十二阶 stencil (x方向)
                for (int k = -RADIUS; k <= RADIUS; ++k) {
                    sum += input[x + k][y][z];
                }
                // 十二阶 stencil (y方向)
                for (int k = -RADIUS; k <= RADIUS; ++k) {
                    sum += input[x][y + k][z];
                }
                // 十二阶 stencil (z方向)
                for (int k = -RADIUS; k <= RADIUS; ++k) {
                    sum += input[x][y][z + k];
                }
                
                output[x][y][z] = sum;
            }
        }
    }
}

// Tiling优化版本 - 使用一维数组
void stencil_tiling_1d(float* input, float* output) {
    // 对计算区域进行分块
    for (int tx = RADIUS; tx < NX - RADIUS; tx += TILE_X) {
        int tile_x_end = (tx + TILE_X) < (NX - RADIUS) ? (tx + TILE_X) : (NX - RADIUS);
        
        for (int ty = RADIUS; ty < NY - RADIUS; ty += TILE_Y) {
            int tile_y_end = (ty + TILE_Y) < (NY - RADIUS) ? (ty + TILE_Y) : (NY - RADIUS);
            
            for (int tz = RADIUS; tz < NZ - RADIUS; tz += TILE_Z) {
                int tile_z_end = (tz + TILE_Z) < (NZ - RADIUS) ? (tz + TILE_Z) : (NZ - RADIUS);
                
                // 处理当前 tile
                for (int x = tx; x < tile_x_end; ++x) {
                    for (int y = ty; y < tile_y_end; ++y) {
                        // 预取当前行的起始位置，提高缓存效率
                        int row_start_idx = x * NY * NZ + y * NZ + tz;
                        
                        for (int z = tz; z < tile_z_end; ++z) {
                            float sum = 0.0f;
                            int idx = row_start_idx + (z - tz);
                            
                            // x方向 stencil
                            int offset_x = idx - RADIUS * NY * NZ;
                            for (int k = 0; k < 2 * RADIUS + 1; ++k) {
                                sum += input[offset_x + k * NY * NZ];
                            }
                            
                            // y方向 stencil
                            int offset_y = idx - RADIUS * NZ;
                            for (int k = 0; k < 2 * RADIUS + 1; ++k) {
                                sum += input[offset_y + k * NZ];
                            }
                            
                            // z方向 stencil
                            int offset_z = idx - RADIUS;
                            for (int k = 0; k < 2 * RADIUS + 1; ++k) {
                                sum += input[offset_z + k];
                            }
                            
                            output[idx] = sum;
                        }
                    }
                }
            }
        }
    }
}

// Tiling优化版本 - 使用三维数组
void stencil_tiling_3d(float*** input, float*** output) {
    // 对计算区域进行分块
    for (int tx = RADIUS; tx < NX - RADIUS; tx += TILE_X) {
        int tile_x_end = (tx + TILE_X) < (NX - RADIUS) ? (tx + TILE_X) : (NX - RADIUS);
        
        for (int ty = RADIUS; ty < NY - RADIUS; ty += TILE_Y) {
            int tile_y_end = (ty + TILE_Y) < (NY - RADIUS) ? (ty + TILE_Y) : (NY - RADIUS);
            
            for (int tz = RADIUS; tz < NZ - RADIUS; tz += TILE_Z) {
                int tile_z_end = (tz + TILE_Z) < (NZ - RADIUS) ? (tz + TILE_Z) : (NZ - RADIUS);
                
                // 处理当前 tile
                for (int x = tx; x < tile_x_end; ++x) {
                    for (int y = ty; y < tile_y_end; ++y) {
                        for (int z = tz; z < tile_z_end; ++z) {
                            float sum = 0.0f;
                            
                            // x方向 stencil
                            for (int k = -RADIUS; k <= RADIUS; ++k) {
                                sum += input[x + k][y][z];
                            }
                            
                            // y方向 stencil
                            for (int k = -RADIUS; k <= RADIUS; ++k) {
                                sum += input[x][y + k][z];
                            }
                            
                            // z方向 stencil
                            for (int k = -RADIUS; k <= RADIUS; ++k) {
                                sum += input[x][y][z + k];
                            }
                            
                            output[x][y][z] = sum;
                        }
                    }
                }
            }
        }
    }
}

// 带有边界扩展的tiling优化版本 - 使用一维数组
void stencil_tiling_optimized_1d(float* input, float* output) {
    // HALO区域等于RADIUS，因为需要左右各RADIUS个点
    const int HALO = RADIUS;
    
    for (int tx = RADIUS; tx < NX - RADIUS; tx += TILE_X) {
        // 计算tile的起始和结束位置（包含halo区域）
        int tile_x_start = tx - HALO;
        int tile_x_end = (tx + TILE_X + HALO) < NX ? (tx + TILE_X + HALO) : NX;
        int tile_x_size = tile_x_end - tile_x_start;
        
        for (int ty = RADIUS; ty < NY - RADIUS; ty += TILE_Y) {
            int tile_y_start = ty - HALO;
            int tile_y_end = (ty + TILE_Y + HALO) < NY ? (ty + TILE_Y + HALO) : NY;
            int tile_y_size = tile_y_end - tile_y_start;
            
            for (int tz = RADIUS; tz < NZ - RADIUS; tz += TILE_Z) {
                int tile_z_start = tz - HALO;
                int tile_z_end = (tz + TILE_Z + HALO) < NZ ? (tz + TILE_Z + HALO) : NZ;
                int tile_z_size = tile_z_end - tile_z_start;
                
                // 创建局部 tile 缓冲区（包含halo区域）
                float* tile_buffer = (float*)malloc(tile_x_size * tile_y_size * tile_z_size * sizeof(float));
                
                // 复制数据到 tile 缓冲区（包括 halo 区域）
                for (int x = 0; x < tile_x_size; ++x) {
                    for (int y = 0; y < tile_y_size; ++y) {
                        for (int z = 0; z < tile_z_size; ++z) {
                            int src_x = tile_x_start + x;
                            int src_y = tile_y_start + y;
                            int src_z = tile_z_start + z;
                            
                            // 安全检查
                            if (src_x >= 0 && src_x < NX && 
                                src_y >= 0 && src_y < NY && 
                                src_z >= 0 && src_z < NZ) {
                                int src_idx = src_x * NY * NZ + src_y * NZ + src_z;
                                int dst_idx = x * tile_y_size * tile_z_size + 
                                            y * tile_z_size + z;
                                tile_buffer[dst_idx] = input[src_idx];
                            }
                        }
                    }
                }
                
                // 计算 tile 内部区域（排除 halo）
                for (int x = HALO; x < tile_x_size - HALO; ++x) {
                    for (int y = HALO; y < tile_y_size - HALO; ++y) {
                        int row_start_idx = x * tile_y_size * tile_z_size + 
                                          y * tile_z_size + HALO;
                        
                        for (int z = HALO; z < tile_z_size - HALO; ++z) {
                            float sum = 0.0f;
                            int idx = row_start_idx + (z - HALO);
                            
                            // 在局部缓冲区中计算 stencil
                            // x方向
                            int x_offset = idx - HALO * tile_y_size * tile_z_size;
                            for (int k = 0; k < 2 * RADIUS + 1; ++k) {
                                sum += tile_buffer[x_offset + k * tile_y_size * tile_z_size];
                            }
                            
                            // y方向
                            int y_offset = idx - HALO * tile_z_size;
                            for (int k = 0; k < 2 * RADIUS + 1; ++k) {
                                sum += tile_buffer[y_offset + k * tile_z_size];
                            }
                            
                            // z方向
                            int z_offset = idx - HALO;
                            for (int k = 0; k < 2 * RADIUS + 1; ++k) {
                                sum += tile_buffer[z_offset + k];
                            }
                            
                            // 写回结果到输出数组
                            int out_x = tile_x_start + x;
                            int out_y = tile_y_start + y;
                            int out_z = tile_z_start + z;
                            int out_idx = out_x * NY * NZ + out_y * NZ + out_z;
                            output[out_idx] = sum;
                        }
                    }
                }
                
                free(tile_buffer);
            }
        }
    }
}

// 验证结果
int verify_results(float* arr1, float* arr2, int nx, int ny, int nz, float tolerance) {
    int total = nx * ny * nz;
    for (int i = 0; i < total; i++) {
        if (fabs(arr1[i] - arr2[i]) > tolerance) {
            printf("Mismatch at index %d: %.6f vs %.6f\n", i, arr1[i], arr2[i]);
            return 0;
        }
    }
    return 1;
}

// 计时函数
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main() {
    printf("3D 12th-order Stencil Computation with Tiling Optimization\n");
    printf("Array size: %d x %d x %d\n", NX, NY, NZ);
    printf("Stencil radius: %d (12th-order, 13-point)\n", RADIUS);
    printf("Tile size: %d x %d x %d\n\n", TILE_X, TILE_Y, TILE_Z);
    
    // 分配内存（使用一维数组表示）
    float* input_1d = allocate_1d_array(NX, NY, NZ);
    float* output_baseline_1d = allocate_1d_array(NX, NY, NZ);
    float* output_tiling_1d = allocate_1d_array(NX, NY, NZ);
    float* output_optimized_1d = allocate_1d_array(NX, NY, NZ);
    
    // 初始化数据
    printf("Initializing data...\n");
    initialize_1d(input_1d, NX, NY, NZ);
    memset(output_baseline_1d, 0, NX * NY * NZ * sizeof(float));
    memset(output_tiling_1d, 0, NX * NY * NZ * sizeof(float));
    memset(output_optimized_1d, 0, NX * NY * NZ * sizeof(float));
    
    // 测试基础版本
    printf("\nRunning 一维数组 version...\n");
    double start = get_time();
    stencil_baseline_1d(input_1d, output_baseline_1d);
    double end = get_time();
    double duration_baseline = end - start;
    printf("Baseline version: %.3f seconds\n", duration_baseline);
    
    // 测试tiling版本
    printf("\nRunning tiling版本 version...\n");
    start = get_time();
    stencil_tiling_1d(input_1d, output_tiling_1d);
    end = get_time();
    double duration_tiling = end - start;
    printf("Tiling version: %.3f seconds\n", duration_tiling);
    
    // 测试优化版tiling版本
    printf("\nRunning 优化的tiling版本 version...\n");
    start = get_time();
    stencil_tiling_optimized_1d(input_1d, output_optimized_1d);
    end = get_time();
    double duration_optimized = end - start;
    printf("Optimized tiling version: %.3f seconds\n", duration_optimized);
    
    // 验证结果
    printf("\nVerifying results...\n");
    if (verify_results(output_baseline_1d, output_tiling_1d, NX, NY, NZ, 1e-6)) {
        printf("Baseline and tiling results match!\n");
    } else {
        printf("ERROR: Baseline and tiling results do NOT match!\n");
    }
    
    if (verify_results(output_baseline_1d, output_optimized_1d, NX, NY, NZ, 1e-6)) {
        printf("Baseline and optimized tiling results match!\n");
    } else {
        printf("ERROR: Baseline and optimized tiling results do NOT match!\n");
    }
    
    // 计算加速比
    printf("\nPerformance Summary:\n");
    printf("基线 version: %.3f seconds\n", duration_baseline);
    printf("Tiling version: %.3f seconds (Speedup: %.2fx)\n", 
           duration_tiling, duration_baseline / duration_tiling);
    printf("优化tiling version: %.3f seconds (Speedup: %.2fx)\n", 
           duration_optimized, duration_baseline / duration_optimized);
    
    // 测试三维数组版本
    printf("\n\nTesting 3D array 三维数组 version...\n");
    
    // 分配三维数组
    float*** input_3d = allocate_3d_array(NX, NY, NZ);
    float*** output_baseline_3d = allocate_3d_array(NX, NY, NZ);
    float*** output_tiling_3d = allocate_3d_array(NX, NY, NZ);
    
    // 初始化数据
    printf("Initializing 3D arrays...\n");
    initialize_3d(input_3d, NX, NY, NZ);
    
    // 测试基础版本（3D数组）
    printf("\nRunning 基础版本 (3D arrays)...\n");
    start = get_time();
    stencil_baseline_3d(input_3d, output_baseline_3d);
    end = get_time();
    double duration_baseline_3d = end - start;
    printf("Baseline version (3D arrays): %.3f seconds\n", duration_baseline_3d);
    
    // 测试tiling版本（3D数组）
    printf("\nRunning tiling版本 version (3D arrays)...\n");
    start = get_time();
    stencil_tiling_3d(input_3d, output_tiling_3d);
    end = get_time();
    double duration_tiling_3d = end - start;
    printf("Tiling version (3D arrays): %.3f seconds (Speedup: %.2fx)\n", 
           duration_tiling_3d, duration_baseline_3d / duration_tiling_3d);
    
    // 释放内存
    printf("\nCleaning up memory...\n");
    free(input_1d);
    free(output_baseline_1d);
    free(output_tiling_1d);
    free(output_optimized_1d);
    
    free_3d_array(input_3d, NX, NY);
    free_3d_array(output_baseline_3d, NX, NY);
    free_3d_array(output_tiling_3d, NX, NY);
    
    printf("\nDone!\n");
    
    return 0;
}
