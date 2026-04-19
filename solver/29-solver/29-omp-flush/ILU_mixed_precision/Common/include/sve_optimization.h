
#ifndef SVE_OPTIMIZATION_H
#define SVE_OPTIMIZATION_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cassert>
#include <string>
#include <iostream>
#include <algorithm>
#include <arm_sve.h>
#include <math.h>

#include "number_type.h"
#include "zone.h"
#include "grid_polyhedra.h"
#include "utility_functions.h"
#include "solver_ns.h"
#include "io_base_format.h"
#include "io_log.h"
#include "parallel_base_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"
#include "linsys_solver.h"
#include "temporal_discretisation_implicit.h"

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif

#include "kblas.h"

#ifdef MPICH
#include <mpi.h>
MPI_Comm comm_L2;
MPI_Comm comm_L3;
MPI_Comm comm_L4;
#endif

#ifdef FS_OPENMP
#include "omp.h"
#endif

using namespace std;

extern double  time_a, time_b;

// -- MPI ranks for switch -- //
#define L2_SIZE 2048
#define L3_SIZE 8096
// -- MPI ranks for switch -- //
// size for cacheline
#define CACHE_LINE_SIZE 64
// --- GMRES of max kspan ---//
#define MAXK 31

namespace mflow
{
    // struct to avoid false sharing in cacheline for thread-level parallelism
    struct thread_private {
        MATRIXTYPE values[MAXK];
        char padding[CACHE_LINE_SIZE - (MAXK * sizeof(MATRIXTYPE)) % CACHE_LINE_SIZE];
    };
#ifdef MPICH
// 初始化层次通信子，基于全局通信域 world
void init_hierarchy(MPI_Comm world) {
    int rank, size;
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);

    // L2 组：连续 2048 个进程一组
    int color_L2 = rank / L2_SIZE;
    MPI_Comm_split(world, color_L2, rank, &comm_L2);

    int is_L2_leader = (rank % L2_SIZE == 0);

    // L3 组：仅 L2 leader 参与，连续 8096 个进程一组（基于原始 rank）
    int color_L3 = is_L2_leader ? (rank / L3_SIZE) : MPI_UNDEFINED;
    MPI_Comm_split(world, color_L3, rank, &comm_L3);

    int is_L3_leader = 0;
    if (comm_L3 != MPI_COMM_NULL) {
        int rank_L3;
        MPI_Comm_rank(comm_L3, &rank_L3);
        is_L3_leader = (rank_L3 == 0);
    }

    // L4 组：所有 L3 leader 在同一组
    int color_L4 = is_L3_leader ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(world, color_L4, rank, &comm_L4);
}

// 释放层次通信子
void free_hierarchy() {
    if (comm_L2 != MPI_COMM_NULL) MPI_Comm_free(&comm_L2);
    if (comm_L3 != MPI_COMM_NULL) MPI_Comm_free(&comm_L3);
    if (comm_L4 != MPI_COMM_NULL) MPI_Comm_free(&comm_L4);
}

// 层次化 Allreduce：对 count 个数据执行 op 操作，结果存入 recvbuf
void hierarchical_allreduce(const void *sendbuf, void *recvbuf, int count,
                            MPI_Datatype datatype, MPI_Op op) {
    int type_size;
    MPI_Type_size(datatype, &type_size);
    void *tmp = malloc(count * type_size);
    memcpy(tmp, sendbuf, count * type_size);

    // ----- 上升归约 -----
    if (comm_L2 != MPI_COMM_NULL) {
        int rank;
        MPI_Comm_rank(comm_L2, &rank);
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : tmp, tmp, count, datatype, op, 0, comm_L2);
    }
    if (comm_L3 != MPI_COMM_NULL) {
        int rank;
        MPI_Comm_rank(comm_L3, &rank);
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : tmp, tmp, count, datatype, op, 0, comm_L3);
    }
    if (comm_L4 != MPI_COMM_NULL) {
        MPI_Allreduce(MPI_IN_PLACE, tmp, count, datatype, op, comm_L4);
    }

    // ----- 下行广播 -----
    if (comm_L3 != MPI_COMM_NULL) {
        MPI_Bcast(tmp, count, datatype, 0, comm_L3);
    }
    if (comm_L2 != MPI_COMM_NULL) {
        MPI_Bcast(tmp, count, datatype, 0, comm_L2);
    }

    memcpy(recvbuf, tmp, count * type_size);
    free(tmp);
}

void use_MPI(const void *local_sum, void *global_sum, int m, MPI_Op op){
    MPI_Barrier(MPI_COMM_WORLD);
    double time_t0 = MPI_Wtime();

    //hierarchical_allreduce(local_sum, global_sum, m, MATRIXMPITYPE, op);
    MPI_Allreduce(local_sum, global_sum, m, MATRIXMPITYPE, op, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double time_t1 = MPI_Wtime();
    time_a += time_t1 - time_t0;
}
#endif

float dot_product_parallel_sve_float(const float* __restrict a, 
    const float* __restrict b, IntType n) {
    //scalar = a[] * b[]
    float global_sum = 0.0f;
    #pragma omp parallel reduction(+:global_sum)
    {
        // 每个线程私有的SVE向量累加器
        svfloat32_t local_sum_vec = svdup_n_f32(0.0f);
        
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i += svcntw()) {
            svbool_t pg = svwhilelt_b32(i, n);  // 生成谓词，处理边界
            svfloat32_t vec_a = svld1_f32(pg, &a[i]);
            svfloat32_t vec_b = svld1_f32(pg, &b[i]);
            
            // 乘积累加：local_sum_vec += vec_a * vec_b
            local_sum_vec = svmla_f32_m(pg, local_sum_vec, vec_a, vec_b);
        }
        int indx = n / svcntw() * svcntw();
        if ( indx < n) {
            svbool_t pg = svwhilelt_b32((int64_t)indx, (int64_t)n);
            svfloat32_t vec1 = svld1_f32(pg, &a[indx]);
            svfloat32_t vec2 = svld1_f32(pg, &b[indx]);
            local_sum_vec = svmla_f32_m(pg, local_sum_vec, vec1, vec2);
        }
        // 将当前线程的向量累加器归约为标量
        float thread_sum = svaddv_f32(svptrue_b32(), local_sum_vec);
        
        // 线程间归约（OpenMP自动处理reduction）
        global_sum += thread_sum;
    }
    return global_sum;
}

float AdotA_self_sve_omp_float(const float* __restrict a, int n, const IntType MAX_THREADS) {

    float sum = 0.0f;
    thread_private *tmpbuf = new thread_private[ MAX_THREADS ];
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        tmpbuf[tid].values[0] = 0.0f;

        int chunk = (n + nth - 1) / nth;
        int start = tid * chunk;
        int end = (start + chunk < n) ? start + chunk : n;
        svfloat32_t vacc = svdup_f32(0.0f);

        int i = start;
        for (; i + svcntw() <= end; i += svcntw()) {
            svbool_t pg = svptrue_b32();                 // 全真谓词
            svfloat32_t avec = svld1_f32(pg, &a[i]);     // 加载 a 的向量
            // FMA 指令：vacc = vacc + avec * avec
            vacc = svmla_f32_x(pg, vacc, avec, avec);
        }

        if (i < end) {
            svbool_t pg = svwhilelt_b32(i, end);         // 部分真谓词
            svfloat32_t avec = svld1_f32(pg, &a[i]);
            vacc = svmla_f32_m(pg, vacc, avec, avec);
        }

        tmpbuf[tid].values[0] = svaddv_f32(svptrue_b32(), vacc);
    }
    for(int l=0; l<MAX_THREADS; l++) sum += tmpbuf[l].values[0];
    delete[] tmpbuf;
    return sum;
}


void vector_div_scalar_sve_omp_float(float* __restrict b,
    float scalar, int n) {

    float inv_scalar = 1.0f / scalar;
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();

        int chunk = (n + nth - 1) / nth;
        int start = tid * chunk;
        int end = (start + chunk < n) ? start + chunk : n;

        svfloat32_t inv_vec = svdup_f32(inv_scalar);
        int i = start;
        for (; i + svcntw() <= end; i += svcntw()) {
            svbool_t pg = svptrue_b32();               // 全真谓词
            svfloat32_t b_vec = svld1_f32(pg, &b[i]);  // 加载 b 向量
            b_vec = svmul_f32_x(pg, b_vec, inv_vec);   // b *= inv_scalar
            svst1_f32(pg, &b[i], b_vec);               // 存储结果
        }

        if (i < end) {
            svbool_t pg = svwhilelt_b32(i, end);       // 部分真谓词
            svfloat32_t b_vec = svld1_f32(pg, &b[i]);
            b_vec = svmul_f32_m(pg, b_vec, inv_vec);
            svst1_f32(pg, &b[i], b_vec);
        }
    }
}


void vector_div_scalar_sve_omp_double(double* __restrict b,
                                      double scalar,
                                      int n) {
    double inv_scalar = 1.0 / scalar;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int chunk = (n + nth - 1) / nth;
        int start = tid * chunk;
        int end = (start + chunk < n) ? start + chunk : n;

        svfloat64_t inv_vec = svdup_f64(inv_scalar);
        int i = start;
        for (; i + svcntd() <= end; i += svcntd()) {
            svbool_t pg = svptrue_b64();              
            svfloat64_t b_vec = svld1_f64(pg, &b[i]); 
            b_vec = svmul_f64_x(pg, b_vec, inv_vec);  
            svst1_f64(pg, &b[i], b_vec);              
        }

        if (i < end) {
            svbool_t pg = svwhilelt_b64((int64_t)i, (int64_t)end);
            svfloat64_t b_vec = svld1_f64(pg, &b[i]);
            b_vec = svmul_f64_m(pg, b_vec, inv_vec);
            svst1_f64(pg, &b[i], b_vec);
        }
    }
}

__attribute__((always_inline)) svfloat64_t scalar_sve_double(const svfloat64_t a_vec,
                       const double* __restrict b_ptr,
                       const svbool_t pg,
                       svfloat64_t vacc){
    svfloat64_t b_vec = svld1_f64(pg, b_ptr);
    return svmla_f64_m(pg, vacc, a_vec, b_vec);
}

void projection_sve_omp_double(const double* __restrict a,
                   const double* __restrict w,
                   IntType len,
                   IntType stride,
                   IntType n,
                   double* __restrict result) {
    #pragma omp parallel reduction(+:result[:len])
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int chunk = (n + nth - 1) / nth;
        int start = tid * chunk;
        int end = (start + chunk < n) ? start + chunk : n;

        int j=0;
        for(; j+4<=len; j+=4){
            int i = start;
            const double* ptr0 = &w[j    *stride];
            const double* ptr1 = &w[(j+1)*stride];
            const double* ptr2 = &w[(j+2)*stride];
            const double* ptr3 = &w[(j+3)*stride];
            
            svfloat64_t vacc0 = svdup_f64(0.0);
            svfloat64_t vacc1 = svdup_f64(0.0);
            svfloat64_t vacc2 = svdup_f64(0.0);
            svfloat64_t vacc3 = svdup_f64(0.0);
            for ( ; i + svcntd() <= end; i += svcntd()) {
                svbool_t pg = svptrue_b64();                 
                svfloat64_t a_vec = svld1_f64(pg, &a[i]);     
            
                vacc0 = svmla_f64_x(pg, vacc0, a_vec, svld1_f64(pg, &ptr0[i]));
                vacc1 = svmla_f64_x(pg, vacc1, a_vec, svld1_f64(pg, &ptr1[i]));
                vacc2 = svmla_f64_x(pg, vacc2, a_vec, svld1_f64(pg, &ptr2[i]));
                vacc3 = svmla_f64_x(pg, vacc3, a_vec, svld1_f64(pg, &ptr3[i]));
            }
            if (i < end) {
                svbool_t pg       = svwhilelt_b64((int64_t)i, (int64_t)end);
                svfloat64_t a_vec = svld1_f64(pg, &a[i]);

                vacc0 = svmla_f64_m(pg, vacc0, a_vec, svld1_f64(pg, &ptr0[i]));
                vacc1 = svmla_f64_m(pg, vacc1, a_vec, svld1_f64(pg, &ptr1[i]));
                vacc2 = svmla_f64_m(pg, vacc2, a_vec, svld1_f64(pg, &ptr2[i]));
                vacc3 = svmla_f64_m(pg, vacc3, a_vec, svld1_f64(pg, &ptr3[i]));
            }

            double sum_vec0 = svaddv_f64(svptrue_b64(), vacc0);
            double sum_vec1 = svaddv_f64(svptrue_b64(), vacc1);
            double sum_vec2 = svaddv_f64(svptrue_b64(), vacc2);
            double sum_vec3 = svaddv_f64(svptrue_b64(), vacc3);

            result[j]   += sum_vec0;
            result[j+1] += sum_vec1;
            result[j+2] += sum_vec2;
            result[j+3] += sum_vec3;
                        
        }
        for (; j < len; j++) {
            const double *b = &w[j*stride];
            svfloat64_t local_sum_vec = svdup_f64(0.0);

            int i=start;
            for (; i + svcntd() <= end; i += svcntd()) {
                svbool_t pg = svptrue_b64();  // 生成谓词，处理边界
                svfloat64_t vec_a = svld1_f64(pg, &a[i]);
                svfloat64_t vec_b = svld1_f64(pg, &b[i]);
                local_sum_vec = svmla_f64_x(pg, local_sum_vec, vec_a, vec_b);
            }
            if ( i < end) {
                svbool_t pg = svwhilelt_b64((int64_t)i, (int64_t)end);
                svfloat64_t vec_a = svld1_f64(pg, &a[i]);
                svfloat64_t vec_b = svld1_f64(pg, &b[i]);
                local_sum_vec = svmla_f64_m(pg, local_sum_vec, vec_a, vec_b);
            }
            double thread_sum = svaddv_f64(svptrue_b64(), local_sum_vec);
            result[j] += thread_sum;
        }
    }
}

__attribute__((always_inline)) svfloat32_t scalar_sve(const svfloat32_t a_vec,
                       const float* __restrict b_ptr,
                       const svbool_t pg,
                       svfloat32_t vacc){
    svfloat32_t b_vec = svld1_f32(pg, b_ptr);
    return svmla_f32_m(pg, vacc, a_vec, b_vec);
}

void projection_sve_omp_float(const float* __restrict a,
                   const float* __restrict w,
                   IntType len,
                   IntType stride,
                   IntType n,
                   float* __restrict result,
                   const IntType MAX_THREADS) {
    thread_private *tmpbuf = new thread_private[ MAX_THREADS ];
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int chunk = (n + nth - 1) / nth;
        int start = tid * chunk;
        int end = (start + chunk < n) ? start + chunk : n;
        for(int l=0; l<MAXK; l++) tmpbuf[tid].values[l] = 0.0f;

        int j=0;
        for(; j+4<=len; j+=4){
            int i = start;
            const float* ptr0 = &w[j    *stride];
            const float* ptr1 = &w[(j+1)*stride];
            const float* ptr2 = &w[(j+2)*stride];
            const float* ptr3 = &w[(j+3)*stride];
            
            svfloat32_t vacc0 = svdup_f32(0.0f);
            svfloat32_t vacc1 = svdup_f32(0.0f);
            svfloat32_t vacc2 = svdup_f32(0.0f);
            svfloat32_t vacc3 = svdup_f32(0.0f);
            for ( ; i + svcntw() <= end; i += svcntw()) {
                svbool_t pg = svptrue_b32();                 
                svfloat32_t a_vec = svld1_f32(pg, &a[i]);     
                vacc0 = svmla_f32_x(pg, vacc0, a_vec, svld1_f32(pg, &ptr0[i]));
                vacc1 = svmla_f32_x(pg, vacc1, a_vec, svld1_f32(pg, &ptr1[i]));
                vacc2 = svmla_f32_x(pg, vacc2, a_vec, svld1_f32(pg, &ptr2[i]));
                vacc3 = svmla_f32_x(pg, vacc3, a_vec, svld1_f32(pg, &ptr3[i]));
            }
            if (i < end) {
                svbool_t pg       = svwhilelt_b32((int64_t)i, (int64_t)end);
                svfloat32_t a_vec = svld1_f32(pg, &a[i]);
                vacc0 = svmla_f32_m(pg, vacc0, a_vec, svld1_f32(pg, &ptr0[i]));
                vacc1 = svmla_f32_m(pg, vacc1, a_vec, svld1_f32(pg, &ptr1[i]));
                vacc2 = svmla_f32_m(pg, vacc2, a_vec, svld1_f32(pg, &ptr2[i]));
                vacc3 = svmla_f32_m(pg, vacc3, a_vec, svld1_f32(pg, &ptr3[i]));
            }

            float sum_vec0 = svaddv_f32(svptrue_b32(), vacc0);
            float sum_vec1 = svaddv_f32(svptrue_b32(), vacc1);
            float sum_vec2 = svaddv_f32(svptrue_b32(), vacc2);
            float sum_vec3 = svaddv_f32(svptrue_b32(), vacc3);

            tmpbuf[tid].values[j+0] = sum_vec0;
            tmpbuf[tid].values[j+1] = sum_vec1;
            tmpbuf[tid].values[j+2] = sum_vec2;
            tmpbuf[tid].values[j+3] = sum_vec3;
        }
        for (; j < len; j++) {
            const float *b = &w[j*stride];
            svfloat32_t local_sum_vec = svdup_n_f32(0.0f);

            int i=start;
            for (; i + svcntw() <= end; i += svcntw()) {
                svbool_t pg = svptrue_b32();  // 生成谓词，处理边界
                svfloat32_t vec_a = svld1_f32(pg, &a[i]);
                svfloat32_t vec_b = svld1_f32(pg, &b[i]);
                local_sum_vec = svmla_f32_x(pg, local_sum_vec, vec_a, vec_b);
            }
            if ( i < end) {
                svbool_t pg = svwhilelt_b32((int64_t)i, (int64_t)end);
                svfloat32_t vec_a = svld1_f32(pg, &a[i]);
                svfloat32_t vec_b = svld1_f32(pg, &b[i]);
                local_sum_vec = svmla_f32_m(pg, local_sum_vec, vec_a, vec_b);
            }
            float thread_sum = svaddv_f32(svptrue_b32(), local_sum_vec);
            tmpbuf[tid].values[j+0] = thread_sum;
        }
    }
    for(int k=0; k<len; k++){
        for(int l=0; l<MAX_THREADS; l++){
            result[k] += tmpbuf[l].values[k];
        }
    } 
    delete[] tmpbuf;
}

void vector_sub_scaled_sve_omp_float(
    float* __restrict a,
    const float* __restrict b,
    const float scalar,
    int n) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int chunk = (n + nth - 1) / nth;
        int start = tid * chunk;
        int end = (start + chunk < n) ? start + chunk : n;

        svfloat32_t scalar_vec = svdup_f32(scalar);
        int i = start;
        for (; i + svcntw() <= end; i += svcntw()) {
            svbool_t pg = svptrue_b32();               // 全真谓词
            svfloat32_t a_vec = svld1_f32(pg, &a[i]);  // 加载 a 向量
            svfloat32_t b_vec = svld1_f32(pg, &b[i]);  // 加载 b 向量
            a_vec = svmla_f32_x(pg, a_vec, scalar_vec, b_vec);
            svst1_f32(pg, &a[i], a_vec);               
        }

        // 处理剩余不足一个向量的元素
        if (i < end) {
            svbool_t pg = svwhilelt_b32(i, end);       // 部分真谓词
            svfloat32_t a_vec = svld1_f32(pg, &a[i]);
            svfloat32_t b_vec = svld1_f32(pg, &b[i]);

            a_vec = svmla_f32_m(pg, a_vec, scalar_vec, b_vec);
            svst1_f32(pg, &a[i], a_vec);
        }
    }
}

void vector_sub_scaled_sve_omp_double(
    double* __restrict a,
    const double* __restrict b,
    const double scalar,
    int n) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int chunk = (n + nth - 1) / nth;
        int start = tid * chunk;
        int end = (start + chunk < n) ? start + chunk : n;

        svfloat64_t scalar_vec = svdup_f64(scalar);   
        int i = start;

        for (; i + svcntd() <= end; i += svcntd()) {
            svbool_t pg = svptrue_b64();              
            svfloat64_t a_vec = svld1_f64(pg, &a[i]);  
            svfloat64_t b_vec = svld1_f64(pg, &b[i]);  
            a_vec = svmla_f64_x(pg, a_vec, scalar_vec, b_vec); 
            svst1_f64(pg, &a[i], a_vec);              
        }

        // 处理剩余不足一个向量的元素
        if (i < end) {
            svbool_t pg = svwhilelt_b64(i, end);      
            svfloat64_t a_vec = svld1_f64(pg, &a[i]);  
            svfloat64_t b_vec = svld1_f64(pg, &b[i]);  
            a_vec = svmla_f64_m(pg, a_vec, scalar_vec, b_vec); 
            svst1_f64(pg, &a[i], a_vec);
        }
    }
}

void spmv_bsr200_sve_omp(const IntType* __restrict brow_ptr,
                       const IntType* __restrict bcol_ind,
                       const float* __restrict bvals,
                       const float* __restrict x_blk,
                       float* __restrict y_blk,
                       IntType Mb,
                       IntType BS)   // 块大小，可变为200
{
    const int VL = svcntw();  // 向量长度（单精度元素数）
    const bool use_sve = true; // 总是使用SVE（假设编译时启用）

#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (int br = 0; br < Mb; ++br) {
            float* __restrict yrow = y_blk + br * BS;
            // 清零yrow
            for (int i = 0; i < BS; ++i) yrow[i] = 0.0f;  // 可用SVE优化清零，但先保持简单

            const int row_beg = brow_ptr[br];
            const int row_end = brow_ptr[br+1];

            for (int bi = row_beg; bi < row_end; ++bi) {
                const int bc = bcol_ind[bi];
                const float* __restrict A = bvals + (size_t)bi * BS * BS;
                const float* __restrict x = x_blk + (size_t)bc * BS;

                // 块乘法：yrow += A * x, 逐行计算
                for (int i = 0; i < BS; ++i) {
                    svfloat32_t acc = svdup_f32(0.0f);
                    const float* A_row = A + i * BS;
                    int j = 0;
                    // 处理完整向量块
                    for (; j + VL <= BS; j += VL) {
                        svfloat32_t v_a = svld1_f32(svptrue_b32(), A_row + j);
                        svfloat32_t v_x = svld1_f32(svptrue_b32(), x + j);
                        acc = svmla_f32_m(svptrue_b32(), acc, v_a, v_x);
                    }
                    // 处理剩余部分
                    if (j < BS) {
                        svbool_t pg = svwhilelt_b32(j, BS);
                        svfloat32_t v_a = svld1_f32(pg, A_row + j);
                        svfloat32_t v_x = svld1_f32(pg, x + j);
                        acc = svmla_f32_m(pg, acc, v_a, v_x);
                    }
                    yrow[i] += svaddv_f32(svptrue_b32(), acc);
                }
            }
        }
    }
}

void spmv_bsr5_sve_omp_float(const IntType* __restrict brow_ptr,
                       const IntType* __restrict bcol_ind,
                       const float* __restrict bvals,   // nnzb * 25
                       float* __restrict y_blk,
                       const float* __restrict x_blk,   // Nb * 5
                       IntType Mb,
                       const IntType BS=5)         // Mb * 5
{
    // OpenMP 并行块行
#pragma omp parallel
    {
        // 每个线程本地的 SVE 谓词（前 5 lane 有效）
        const svbool_t pg5 = svwhilelt_b32(0, BS);
        const bool vec5_ok = (svcntw() >= BS);

        //printf("Thread %d: svcntw() = %d, use_sve = %d\n", omp_get_thread_num(), svcntw(), vec5_ok);

#pragma omp for schedule(dynamic)
        for (int br = 0; br < Mb; ++br) {
            float* __restrict yrow = &y_blk[br * BS];
            // 本线程负责的块行，先清零 5 个输出
            yrow[0] = yrow[1] = yrow[2] = yrow[3] = yrow[4] = 0.0;

            if (!vec5_ok) {
                // 罕见：VL < 5 的回退（标量微内核）
                for (int bi = brow_ptr[br]; bi < brow_ptr[br+1]; ++bi) {
                    const int bc = bcol_ind[bi];
                    const float* __restrict blk  = &bvals[(size_t)bi * BS * BS];
                    const float* __restrict xseg = &x_blk[(size_t)bc * BS];
                    const float x0 = xseg[0], x1 = xseg[1], x2 = xseg[2], x3 = xseg[3], x4 = xseg[4];

                    const float* a = blk + 0*BS; yrow[0] += a[0]*x0 + a[1]*x1 + a[2]*x2 + a[3]*x3 + a[4]*x4;
                    a = blk + 1*BS;               yrow[1] += a[0]*x0 + a[1]*x1 + a[2]*x2 + a[3]*x3 + a[4]*x4;
                    a = blk + 2*BS;               yrow[2] += a[0]*x0 + a[1]*x1 + a[2]*x2 + a[3]*x3 + a[4]*x4;
                    a = blk + 3*BS;               yrow[3] += a[0]*x0 + a[1]*x1 + a[2]*x2 + a[3]*x3 + a[4]*x4;
                    a = blk + 4*BS;               yrow[4] += a[0]*x0 + a[1]*x1 + a[2]*x2 + a[3]*x3 + a[4]*x4;
                }
                continue;
            }

            // SVE 快速路径：向量累加器，块行结束再做一次水平归约
            svfloat32_t acc0 = svdup_f32(0.0);
            svfloat32_t acc1 = svdup_f32(0.0);
            svfloat32_t acc2 = svdup_f32(0.0);
            svfloat32_t acc3 = svdup_f32(0.0);
            svfloat32_t acc4 = svdup_f32(0.0);

            const int row_beg = brow_ptr[br];
            const int row_end = brow_ptr[br+1];

            //if(ir < 20)
                //printf("x_blk[0] = %f, bvals[0] = %f\n", x_blk[bcol_ind[row_beg] * BS], bvals[0]);

            for (int bi = row_beg; bi < row_end; ++bi) {
                const int bc = bcol_ind[bi];
                const float* __restrict blk  = &bvals[bi * BS * BS];
                const float* __restrict xseg = &x_blk[bc * BS];

                // 轻量预取下一块（可选）
#if defined(__GNUC__) || defined(__clang__)
                if (bi + 1 < row_end) {
                    __builtin_prefetch(&bvals[(bi + 1) * BS * BS], 0, 1);
                    __builtin_prefetch(&x_blk[bcol_ind[bi + 1] * BS], 0, 1);
                }
#endif
                const svfloat32_t xv = svld1_f32(pg5, xseg);

                const svfloat32_t a0 = svld1_f32(pg5, blk + 0*BS);
                acc0 = svmla_f32_m(pg5, acc0, a0, xv);

                const svfloat32_t a1 = svld1_f32(pg5, blk + 1*BS);
                acc1 = svmla_f32_m(pg5, acc1, a1, xv);

                const svfloat32_t a2 = svld1_f32(pg5, blk + 2*BS);
                acc2 = svmla_f32_m(pg5, acc2, a2, xv);

                const svfloat32_t a3 = svld1_f32(pg5, blk + 3*BS);
                acc3 = svmla_f32_m(pg5, acc3, a3, xv);

                const svfloat32_t a4 = svld1_f32(pg5, blk + 4*BS);
                acc4 = svmla_f32_m(pg5, acc4, a4, xv);
            }

            // 每行只归约一次
            yrow[0] += svaddv_f32(pg5, acc0);
            yrow[1] += svaddv_f32(pg5, acc1);
            yrow[2] += svaddv_f32(pg5, acc2);
            yrow[3] += svaddv_f32(pg5, acc3);
            yrow[4] += svaddv_f32(pg5, acc4);
        } // omp for
    } // omp parallel
}

void spmv_bsr5_sve_omp_double(const IntType* __restrict brow_ptr,
                       const IntType* __restrict bcol_ind,
                       const double* __restrict bvals,   // nnzb * 25
                       double* __restrict y_blk,
                       const double* __restrict x_blk,   // Nb * 5
                       IntType Mb,
                       const IntType BS=5)         // Mb * 5
{
    // OpenMP 并行块行
#pragma omp parallel
    {
        // 每个线程本地的 SVE 谓词（前 5 lane 有效）
        const svbool_t pg5 = svwhilelt_b64(0, BS);
        const bool vec5_ok = (svcntd() >= BS);

        //printf("Thread %d: svcntw() = %d, use_sve = %d\n", omp_get_thread_num(), svcntw(), vec5_ok);

#pragma omp for schedule(dynamic)
        for (int br = 0; br < Mb; ++br) {
            double* __restrict yrow = &y_blk[br * BS];
            // 本线程负责的块行，先清零 5 个输出
            yrow[0] = yrow[1] = yrow[2] = yrow[3] = yrow[4] = 0.0;

            if (!vec5_ok) {
                // 罕见：VL < 5 的回退（标量微内核）
                for (int bi = brow_ptr[br]; bi < brow_ptr[br+1]; ++bi) {
                    const int bc = bcol_ind[bi];
                    const double* __restrict blk  = &bvals[(size_t)bi * BS * BS];
                    const double* __restrict xseg = &x_blk[(size_t)bc * BS];
                    const double x0 = xseg[0], x1 = xseg[1], x2 = xseg[2], x3 = xseg[3], x4 = xseg[4];

                    const double* a = blk + 0*BS; yrow[0] += a[0]*x0 + a[1]*x1 + a[2]*x2 + a[3]*x3 + a[4]*x4;
                    a = blk + 1*BS;               yrow[1] += a[0]*x0 + a[1]*x1 + a[2]*x2 + a[3]*x3 + a[4]*x4;
                    a = blk + 2*BS;               yrow[2] += a[0]*x0 + a[1]*x1 + a[2]*x2 + a[3]*x3 + a[4]*x4;
                    a = blk + 3*BS;               yrow[3] += a[0]*x0 + a[1]*x1 + a[2]*x2 + a[3]*x3 + a[4]*x4;
                    a = blk + 4*BS;               yrow[4] += a[0]*x0 + a[1]*x1 + a[2]*x2 + a[3]*x3 + a[4]*x4;
                }
                continue;
            }

            // SVE 快速路径：向量累加器，块行结束再做一次水平归约
            svfloat64_t acc0 = svdup_f64(0.0);
            svfloat64_t acc1 = svdup_f64(0.0);
            svfloat64_t acc2 = svdup_f64(0.0);
            svfloat64_t acc3 = svdup_f64(0.0);
            svfloat64_t acc4 = svdup_f64(0.0);

            const int row_beg = brow_ptr[br];
            const int row_end = brow_ptr[br+1];

            //if(ir < 20)
                //printf("x_blk[0] = %f, bvals[0] = %f\n", x_blk[bcol_ind[row_beg] * BS], bvals[0]);

            for (int bi = row_beg; bi < row_end; ++bi) {
                const int bc = bcol_ind[bi];
                const double* __restrict blk  = &bvals[bi * BS * BS];
                const double* __restrict xseg = &x_blk[bc * BS];

                // 轻量预取下一块（可选）
#if defined(__GNUC__) || defined(__clang__)
                if (bi + 1 < row_end) {
                    __builtin_prefetch(&bvals[(bi + 1) * BS * BS], 0, 1);
                    __builtin_prefetch(&x_blk[bcol_ind[bi + 1] * BS], 0, 1);
                }
#endif
                const svfloat64_t xv = svld1_f64(pg5, xseg);

                const svfloat64_t a0 = svld1_f64(pg5, blk + 0*BS);
                acc0 = svmla_f64_m(pg5, acc0, a0, xv);

                const svfloat64_t a1 = svld1_f64(pg5, blk + 1*BS);
                acc1 = svmla_f64_m(pg5, acc1, a1, xv);

                const svfloat64_t a2 = svld1_f64(pg5, blk + 2*BS);
                acc2 = svmla_f64_m(pg5, acc2, a2, xv);

                const svfloat64_t a3 = svld1_f64(pg5, blk + 3*BS);
                acc3 = svmla_f64_m(pg5, acc3, a3, xv);

                const svfloat64_t a4 = svld1_f64(pg5, blk + 4*BS);
                acc4 = svmla_f64_m(pg5, acc4, a4, xv);
            }

            // 每行只归约一次
            yrow[0] += svaddv_f64(pg5, acc0);
            yrow[1] += svaddv_f64(pg5, acc1);
            yrow[2] += svaddv_f64(pg5, acc2);
            yrow[3] += svaddv_f64(pg5, acc3);
            yrow[4] += svaddv_f64(pg5, acc4);
        } // omp for
    } // omp parallel
}
void vector_sub_sve_omp_float(float* __restrict ptr1,
                      const float* __restrict w,
                      IntType m,                     // = i+1 (当前正交化步数)
                      IntType stride,               // = matrixN * nvar
                      IntType n,
                      const float* __restrict coeffs) { 
    #pragma omp parallel 
    //proc_bind(close)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int chunk = (n + nth - 1) / nth;
        int start = tid * chunk;
        int end = (start + chunk < n) ? start + chunk : n;

        int i = start;
        for (; i + svcntw() <= end; i += svcntw()) {
            svbool_t pg = svptrue_b32();                // 全真谓词
            svfloat32_t acc = svld1_f32(pg, &ptr1[i]);  // 加载当前块到累加器

            for (int k = 0; k < m; ++k) {
                svfloat32_t vec_k = svld1_f32(pg, &w[k * stride + i]); 
                acc = svmla_f32_x(pg, acc, svdup_f32(-coeffs[k]), vec_k); 
            }

            svst1_f32(pg, &ptr1[i], acc);               // 存回更新后的块
        }

        if (i < end) {
            svbool_t pg = svwhilelt_b32(i, end);        // 生成部分激活谓词
            svfloat32_t acc = svld1_f32(pg, &ptr1[i]);

            for (int k = 0; k < m; ++k) {
                svfloat32_t vec_k = svld1_f32(pg, &w[k * stride + i]);
                acc = svmla_f32_m(pg, acc, svdup_f32(-coeffs[k]), vec_k);
            }

            svst1_f32(pg, &ptr1[i], acc);
        }
    }
}

void matrix_vector_sub_sve_float(float* __restrict low_prod, const float* __restrict matrix,
                           const float* __restrict prod, int nvar) {
    for (int j = 0; j < nvar; ++j) {
        const float *row = &matrix[j * nvar];
        svfloat32_t sum = svdup_f32(0.0f);

        int k = 0;
        for (; k + svcntw() <= nvar; k += svcntw()) {
            svbool_t pg = svptrue_b32();
            svfloat32_t vec_row = svld1_f32(pg, &row[k]);
            svfloat32_t vec_prod = svld1_f32(pg, &prod[k]);
            sum = svmla_f32_x(pg, sum, vec_row, vec_prod);
        }

        if (k < nvar) {
            svbool_t pg = svwhilelt_b32(k, nvar);
            svfloat32_t vec_row = svld1_f32(pg, &row[k]);
            svfloat32_t vec_prod = svld1_f32(pg, &prod[k]);
            sum = svmla_f32_m(pg, sum, vec_row, vec_prod);
        }

        float dot = svaddv_f32(svptrue_b32(), sum);
        low_prod[j] -= dot;
    }
}

void matrix_vector_sub_sve_unroll_float(
    float* __restrict low_prod, 
    const float* __restrict matrix,
    const float* __restrict prod, 
    int nvar) {

    const int unroll = 4;
    int j = 0;
    for (; j + unroll <= nvar; j += unroll) {
        const float *row0 = &matrix[j       * nvar];
        const float *row1 = &matrix[(j + 1) * nvar];
        const float *row2 = &matrix[(j + 2) * nvar];
        const float *row3 = &matrix[(j + 3) * nvar];
        svfloat32_t acc0 = svdup_f32(0.0f);
        svfloat32_t acc1 = svdup_f32(0.0f);
        svfloat32_t acc2 = svdup_f32(0.0f);
        svfloat32_t acc3 = svdup_f32(0.0f);

        int k = 0;
        for (; k + svcntw() <= nvar; k += svcntw()) {
            svbool_t pg = svptrue_b32();
            svfloat32_t prod_vec = svld1_f32(pg, &prod[k]);

            acc0 = svmla_f32_x(pg, acc0, svld1_f32(pg, &row0[k]), prod_vec);
            acc1 = svmla_f32_x(pg, acc1, svld1_f32(pg, &row1[k]), prod_vec);
            acc2 = svmla_f32_x(pg, acc2, svld1_f32(pg, &row2[k]), prod_vec);
            acc3 = svmla_f32_x(pg, acc3, svld1_f32(pg, &row3[k]), prod_vec);
        }
        if (k < nvar) {
            svbool_t pg = svwhilelt_b32(k, nvar);
            svfloat32_t prod_vec = svld1_f32(pg, &prod[k]);

            acc0 = svmla_f32_m(pg, acc0, svld1_f32(pg, &row0[k]), prod_vec);
            acc1 = svmla_f32_m(pg, acc1, svld1_f32(pg, &row1[k]), prod_vec);
            acc2 = svmla_f32_m(pg, acc2, svld1_f32(pg, &row2[k]), prod_vec);
            acc3 = svmla_f32_m(pg, acc3, svld1_f32(pg, &row3[k]), prod_vec);
        }
        low_prod[j  ] -= svaddv_f32(svptrue_b32(), acc0);
        low_prod[j+1] -= svaddv_f32(svptrue_b32(), acc1);
        low_prod[j+2] -= svaddv_f32(svptrue_b32(), acc2);
        low_prod[j+3] -= svaddv_f32(svptrue_b32(), acc3);
    }

    for (; j < nvar; ++j) {
        const float *row = &matrix[j * nvar];
        svfloat32_t acc = svdup_f32(0.0f);
        int k = 0;

        for (; k + svcntw() <= nvar; k += svcntw()) {
            svbool_t pg = svptrue_b32();
            svfloat32_t prod_vec = svld1_f32(pg, &prod[k]);
            svfloat32_t row_vec = svld1_f32(pg, &row[k]);
            acc = svmla_f32_x(pg, acc, row_vec, prod_vec);
        }
        if (k < nvar) {
            svbool_t pg = svwhilelt_b32(k, nvar);
            svfloat32_t prod_vec = svld1_f32(pg, &prod[k]);
            svfloat32_t row_vec = svld1_f32(pg, &row[k]);
            acc = svmla_f32_m(pg, acc, row_vec, prod_vec);
        }
        low_prod[j] -= svaddv_f32(svptrue_b32(), acc);
    }
}
void matrix_vector_sve_unroll_float(
    float* __restrict low_prod, 
    const float* __restrict matrix,
    const float* __restrict prod, 
    int nvar) {

    const int unroll = 4;
    int j = 0;
    for (; j + unroll <= nvar; j += unroll) {
        const float *row0 = &matrix[j       * nvar];
        const float *row1 = &matrix[(j + 1) * nvar];
        const float *row2 = &matrix[(j + 2) * nvar];
        const float *row3 = &matrix[(j + 3) * nvar];
        svfloat32_t acc0 = svdup_f32(0.0f);
        svfloat32_t acc1 = svdup_f32(0.0f);
        svfloat32_t acc2 = svdup_f32(0.0f);
        svfloat32_t acc3 = svdup_f32(0.0f);

        int k = 0;
        for (; k + svcntw() <= nvar; k += svcntw()) {
            svbool_t pg = svptrue_b32();
            svfloat32_t prod_vec = svld1_f32(pg, &prod[k]);

            acc0 = svmla_f32_x(pg, acc0, svld1_f32(pg, &row0[k]), prod_vec);
            acc1 = svmla_f32_x(pg, acc1, svld1_f32(pg, &row1[k]), prod_vec);
            acc2 = svmla_f32_x(pg, acc2, svld1_f32(pg, &row2[k]), prod_vec);
            acc3 = svmla_f32_x(pg, acc3, svld1_f32(pg, &row3[k]), prod_vec);
        }
        if (k < nvar) {
            svbool_t pg = svwhilelt_b32(k, nvar);
            svfloat32_t prod_vec = svld1_f32(pg, &prod[k]);

            acc0 = svmla_f32_m(pg, acc0, svld1_f32(pg, &row0[k]), prod_vec);
            acc1 = svmla_f32_m(pg, acc1, svld1_f32(pg, &row1[k]), prod_vec);
            acc2 = svmla_f32_m(pg, acc2, svld1_f32(pg, &row2[k]), prod_vec);
            acc3 = svmla_f32_m(pg, acc3, svld1_f32(pg, &row3[k]), prod_vec);
        }
        low_prod[j  ] = svaddv_f32(svptrue_b32(), acc0);
        low_prod[j+1] = svaddv_f32(svptrue_b32(), acc1);
        low_prod[j+2] = svaddv_f32(svptrue_b32(), acc2);
        low_prod[j+3] = svaddv_f32(svptrue_b32(), acc3);
    }

    for (; j < nvar; ++j) {
        const float *row = &matrix[j * nvar];
        svfloat32_t acc = svdup_f32(0.0f);
        int k = 0;

        for (; k + svcntw() <= nvar; k += svcntw()) {
            svbool_t pg = svptrue_b32();
            svfloat32_t prod_vec = svld1_f32(pg, &prod[k]);
            svfloat32_t row_vec = svld1_f32(pg, &row[k]);
            acc = svmla_f32_x(pg, acc, row_vec, prod_vec);
        }
        if (k < nvar) {
            svbool_t pg = svwhilelt_b32(k, nvar);
            svfloat32_t prod_vec = svld1_f32(pg, &prod[k]);
            svfloat32_t row_vec = svld1_f32(pg, &row[k]);
            acc = svmla_f32_m(pg, acc, row_vec, prod_vec);
        }
        low_prod[j] = svaddv_f32(svptrue_b32(), acc);
    }
}

void ClassicalGramSchmidt_hybrid_sve(IntType i, MATRIXTYPE **&Hsbg, 
    MATRIXTYPE* __restrict w,
    IntType nTCell, 
    IntType nBFace, 
    IntType nvar, 
    IntType matrixN, 
    const IntType MAX_THREADS,
    const int iter) {
    const IntType nT5 = nTCell * nvar;
    const IntType stride = matrixN * nvar;
    MATRIXTYPE *ptr1 = &w[(i + 1) * stride];
    const IntType m = i + 1;

    MATRIXTYPE *local_sum1 = new MATRIXTYPE[m]();
    MATRIXTYPE *global_sum1 = new MATRIXTYPE[m]();
    const IntType bufsize = m + 1;
    MATRIXTYPE *local_buf = new MATRIXTYPE[bufsize]();
    MATRIXTYPE *global_buf = new MATRIXTYPE[bufsize]();
    MATRIXTYPE nrm = 0.0;
    thread_private *tmpbuf = new thread_private[ MAX_THREADS ];
    #pragma omp parallel shared(nrm) 
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        int chunk = (nT5 + nth - 1) / nth;
        int start = tid * chunk;
        int end = (start + chunk < nT5) ? start + chunk : nT5;
        for(int l=0; l<MAXK; l++) tmpbuf[tid].values[l] = 0.0;

        int j=0;
        
        for(; j+4 <= m; j+=4){
            int a = start;
            const float* wptr0 = &w[j    *stride];
            const float* wptr1 = &w[(j+1)*stride];
            const float* wptr2 = &w[(j+2)*stride];
            const float* wptr3 = &w[(j+3)*stride];
            
            svfloat32_t vacc0 = svdup_f32(0.0f);
            svfloat32_t vacc1 = svdup_f32(0.0f);
            svfloat32_t vacc2 = svdup_f32(0.0f);
            svfloat32_t vacc3 = svdup_f32(0.0f);
            for ( ; a + svcntw() <= end; a += svcntw()) {
                svbool_t pg = svptrue_b32();                 
                svfloat32_t a_vec = svld1_f32(pg, &ptr1[a]);     
            
                vacc0 = svmla_f32_x(pg, vacc0, a_vec, svld1_f32(pg, &wptr0[a]));
                vacc1 = svmla_f32_x(pg, vacc1, a_vec, svld1_f32(pg, &wptr1[a]));
                vacc2 = svmla_f32_x(pg, vacc2, a_vec, svld1_f32(pg, &wptr2[a]));
                vacc3 = svmla_f32_x(pg, vacc3, a_vec, svld1_f32(pg, &wptr3[a]));
            }
            if (a < end) {
                svbool_t pg       = svwhilelt_b32((int64_t)a, (int64_t)end);
                svfloat32_t a_vec = svld1_f32(pg, &ptr1[a]);

                vacc0 = svmla_f32_m(pg, vacc0, a_vec, svld1_f32(pg, &wptr0[a]));
                vacc1 = svmla_f32_m(pg, vacc1, a_vec, svld1_f32(pg, &wptr1[a]));
                vacc2 = svmla_f32_m(pg, vacc2, a_vec, svld1_f32(pg, &wptr2[a]));
                vacc3 = svmla_f32_m(pg, vacc3, a_vec, svld1_f32(pg, &wptr3[a]));
            }
            tmpbuf[tid].values[j+0] = svaddv_f32(svptrue_b32(), vacc0);
            tmpbuf[tid].values[j+1] = svaddv_f32(svptrue_b32(), vacc1);
            tmpbuf[tid].values[j+2] = svaddv_f32(svptrue_b32(), vacc2);
            tmpbuf[tid].values[j+3] = svaddv_f32(svptrue_b32(), vacc3);
        }

        for (; j < m; j++) {
            const float *b = &w[j*stride];
            svfloat32_t local_sum_vec = svdup_n_f32(0.0f);
            int a=start;
            for (; a + svcntw() <= end; a += svcntw()) {
                svbool_t pg = svptrue_b32();
                svfloat32_t vec_a = svld1_f32(pg, &ptr1[a]);
                svfloat32_t vec_b = svld1_f32(pg, &b[a]);
                local_sum_vec = svmla_f32_x(pg, local_sum_vec, vec_a, vec_b);
            }
            if ( a < end) {
                svbool_t pg = svwhilelt_b32((int64_t)a, (int64_t)end);
                svfloat32_t vec_a = svld1_f32(pg, &ptr1[a]);
                svfloat32_t vec_b = svld1_f32(pg, &b[a]);
                local_sum_vec = svmla_f32_m(pg, local_sum_vec, vec_a, vec_b);
            }
            tmpbuf[tid].values[j+0] = svaddv_f32(svptrue_b32(), local_sum_vec);
        }
        #pragma omp barrier

#pragma omp master
    {
        for(int k=0; k<m; k++){
            for(int l=0; l<nth; l++){
                local_sum1[k] += tmpbuf[l].values[k];
            }
        }
#ifdef MPICH
            use_MPI(local_sum1, global_sum1, m, MPI_SUM);
#else
            memcpy(global_sum1, local_sum1, m * sizeof(MATRIXTYPE));
#endif
            for (int k = 0; k < m; ++k){
                Hsbg[k][i] = global_sum1[k];
            } 
    }
    #pragma omp barrier

        j = start;
        for (; j + svcntw() <= end; j += svcntw()) {
            svbool_t pg = svptrue_b32();                // 全真谓词
            svfloat32_t acc = svld1_f32(pg, &ptr1[j]);  // 加载当前块到累加器

            for (int k = 0; k < m; ++k) {
                svfloat32_t vec_k = svld1_f32(pg, &w[k * stride + j]); 
                acc = svmla_f32_x(pg, acc, svdup_f32(-global_sum1[k]), vec_k); 
            }
            svst1_f32(pg, &ptr1[j], acc);           
        }
        if (j < end) {
            svbool_t pg = svwhilelt_b32(j, end);        
            svfloat32_t acc = svld1_f32(pg, &ptr1[j]);
            for (int k = 0; k < m; ++k) {
                svfloat32_t vec_k = svld1_f32(pg, &w[k * stride + j]);
                acc = svmla_f32_m(pg, acc, svdup_f32(-global_sum1[k]), vec_k);
            }
            svst1_f32(pg, &ptr1[j], acc);
        }
        #pragma omp barrier
 
        j=0;
        for(; j+4 <= m; j+=4){
            int a = start;
            const float* wptr0 = &w[j    *stride];
            const float* wptr1 = &w[(j+1)*stride];
            const float* wptr2 = &w[(j+2)*stride];
            const float* wptr3 = &w[(j+3)*stride];
            
            svfloat32_t vacc0 = svdup_f32(0.0f);
            svfloat32_t vacc1 = svdup_f32(0.0f);
            svfloat32_t vacc2 = svdup_f32(0.0f);
            svfloat32_t vacc3 = svdup_f32(0.0f);
            for ( ; a + svcntw() <= end; a += svcntw()) {
                svbool_t pg = svptrue_b32();                 
                svfloat32_t a_vec = svld1_f32(pg, &ptr1[a]);     
            
                vacc0 = svmla_f32_x(pg, vacc0, a_vec, svld1_f32(pg, &wptr0[a]));
                vacc1 = svmla_f32_x(pg, vacc1, a_vec, svld1_f32(pg, &wptr1[a]));
                vacc2 = svmla_f32_x(pg, vacc2, a_vec, svld1_f32(pg, &wptr2[a]));
                vacc3 = svmla_f32_x(pg, vacc3, a_vec, svld1_f32(pg, &wptr3[a]));
            }
            if (a < end) {
                svbool_t pg       = svwhilelt_b32((int64_t)a, (int64_t)end);
                svfloat32_t a_vec = svld1_f32(pg, &ptr1[a]);

                vacc0 = svmla_f32_m(pg, vacc0, a_vec, svld1_f32(pg, &wptr0[a]));
                vacc1 = svmla_f32_m(pg, vacc1, a_vec, svld1_f32(pg, &wptr1[a]));
                vacc2 = svmla_f32_m(pg, vacc2, a_vec, svld1_f32(pg, &wptr2[a]));
                vacc3 = svmla_f32_m(pg, vacc3, a_vec, svld1_f32(pg, &wptr3[a]));
            }
            tmpbuf[tid].values[j+0] = svaddv_f32(svptrue_b32(), vacc0);
            tmpbuf[tid].values[j+1] = svaddv_f32(svptrue_b32(), vacc1);
            tmpbuf[tid].values[j+2] = svaddv_f32(svptrue_b32(), vacc2);
            tmpbuf[tid].values[j+3] = svaddv_f32(svptrue_b32(), vacc3);
        }
        for (; j < m; j++) {
            const float *b = &w[j*stride];
            svfloat32_t local_sum_vec = svdup_n_f32(0.0f);
            int a=start;
            for (; a + svcntw() <= end; a += svcntw()) {
                svbool_t pg = svptrue_b32();
                svfloat32_t vec_a = svld1_f32(pg, &ptr1[a]);
                svfloat32_t vec_b = svld1_f32(pg, &b[a]);
                local_sum_vec = svmla_f32_x(pg, local_sum_vec, vec_a, vec_b);
            }
            if ( a < end) {
                svbool_t pg = svwhilelt_b32((int64_t)a, (int64_t)end);
                svfloat32_t vec_a = svld1_f32(pg, &ptr1[a]);
                svfloat32_t vec_b = svld1_f32(pg, &b[a]);
                local_sum_vec = svmla_f32_m(pg, local_sum_vec, vec_a, vec_b);
            }
            tmpbuf[tid].values[j+0] = svaddv_f32(svptrue_b32(), local_sum_vec);
        }
        j=start;
        svfloat32_t sum_vec1 = svdup_n_f32(0.0f);
        for (; j + svcntw() <= end; j += svcntw()) {
            svbool_t pg = svptrue_b32();
            svfloat32_t vec_a = svld1_f32(pg, &ptr1[j]);
            //svfloat32_t vec_b = svld1_f32(pg, &b[j]);
            sum_vec1 = svmla_f32_x(pg, sum_vec1, vec_a, vec_a);
        }
        if ( j < end) {
            svbool_t pg = svwhilelt_b32((int64_t)j, (int64_t)end);
            svfloat32_t vec_a = svld1_f32(pg, &ptr1[j]);
            //svfloat32_t vec_b = svld1_f32(pg, &b[j]);
            sum_vec1 = svmla_f32_m(pg, sum_vec1, vec_a, vec_a);
        }
        tmpbuf[tid].values[m] = svaddv_f32(svptrue_b32(), sum_vec1);
        #pragma omp barrier

#pragma omp master
{
        for(int k=0; k<m+1; k++){
            for(int l=0; l<nth; l++){
                local_buf[k] += tmpbuf[l].values[k];
            }
        }
#ifdef MPICH
        use_MPI(local_buf, global_buf, bufsize, MPI_SUM);
#else
        memcpy(global_buf, local_buf, bufsize * sizeof(MATRIXTYPE));
#endif
        for (int k = 0; k < m; ++k) {
            // Hsbg[k][i] += global_buf[k]; //global_sum1[k];
        }
        nrm = sqrt(global_buf[m]);
        Hsbg[i + 1][i] = nrm;

        if (nrm <= 0.0) {
            printf("Warning: zero norm in orthogonalization step %d.\n", i);
            exit(-1);
        }
        nrm = 1.0f / nrm;
} 
#pragma omp barrier

    svfloat32_t inv_vec = svdup_f32(nrm);
    j = start;
    for (; j + svcntw() <= end; j += svcntw()) {
        svbool_t pg = svptrue_b32();               
        svfloat32_t b_vec = svld1_f32(pg, &ptr1[j]);  
        b_vec = svmul_f32_x(pg, b_vec, inv_vec);   
        svst1_f32(pg, &ptr1[j], b_vec);               
    }
    if (j < end) {
        svbool_t pg = svwhilelt_b32(j, end);       
        svfloat32_t b_vec = svld1_f32(pg, &ptr1[j]);
        b_vec = svmul_f32_m(pg, b_vec, inv_vec);
        svst1_f32(pg, &ptr1[j], b_vec);
    }
}
    delete[] local_sum1;
    delete[] global_sum1;
    delete[] local_buf;
    delete[] global_buf;
    delete[] tmpbuf;
}

void ClassicalGramSchmidt_hybrid_sve_onceSyn(IntType i, MATRIXTYPE **&Hsbg, MATRIXTYPE* __restrict w,
    IntType nTCell, IntType nBFace, IntType nvar, IntType matrixN, const IntType MAX_THREADS, int iter_done) {
    const IntType nT5 = nTCell * nvar;
    const IntType stride = matrixN * nvar;
    MATRIXTYPE *ptr1 = &w[(i + 1) * stride];
    const IntType m = i + 1;
    const IntType bufsize = m + 1;
    MATRIXTYPE *local_sum1 = new MATRIXTYPE[bufsize]();
    MATRIXTYPE *global_sum1 = new MATRIXTYPE[bufsize]();
    MATRIXTYPE nrm = 0.0;
    thread_private *tmpbuf = new thread_private[ MAX_THREADS ];
    #pragma omp parallel shared(nrm) 
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        int chunk = (nT5 + nth - 1) / nth;
        int start = tid * chunk;
        int end = (start + chunk < nT5) ? start + chunk : nT5;
        for(int l=0; l<MAXK; l++) tmpbuf[tid].values[l] = 0.0;

        int j=0;
    
        for(; j+4 <= m; j+=4){
            int a = start;
            const float* wptr0 = &w[j    *stride];
            const float* wptr1 = &w[(j+1)*stride];
            const float* wptr2 = &w[(j+2)*stride];
            const float* wptr3 = &w[(j+3)*stride];
            
            svfloat32_t vacc0 = svdup_f32(0.0f);
            svfloat32_t vacc1 = svdup_f32(0.0f);
            svfloat32_t vacc2 = svdup_f32(0.0f);
            svfloat32_t vacc3 = svdup_f32(0.0f);
            for ( ; a + svcntw() <= end; a += svcntw()) {
                svbool_t pg = svptrue_b32();                 
                svfloat32_t a_vec = svld1_f32(pg, &ptr1[a]);     
            
                vacc0 = svmla_f32_x(pg, vacc0, a_vec, svld1_f32(pg, &wptr0[a]));
                vacc1 = svmla_f32_x(pg, vacc1, a_vec, svld1_f32(pg, &wptr1[a]));
                vacc2 = svmla_f32_x(pg, vacc2, a_vec, svld1_f32(pg, &wptr2[a]));
                vacc3 = svmla_f32_x(pg, vacc3, a_vec, svld1_f32(pg, &wptr3[a]));
            }
            if (a < end) {
                svbool_t pg       = svwhilelt_b32((int64_t)a, (int64_t)end);
                svfloat32_t a_vec = svld1_f32(pg, &ptr1[a]);

                vacc0 = svmla_f32_m(pg, vacc0, a_vec, svld1_f32(pg, &wptr0[a]));
                vacc1 = svmla_f32_m(pg, vacc1, a_vec, svld1_f32(pg, &wptr1[a]));
                vacc2 = svmla_f32_m(pg, vacc2, a_vec, svld1_f32(pg, &wptr2[a]));
                vacc3 = svmla_f32_m(pg, vacc3, a_vec, svld1_f32(pg, &wptr3[a]));
            }
            tmpbuf[tid].values[j+0] = svaddv_f32(svptrue_b32(), vacc0);
            tmpbuf[tid].values[j+1] = svaddv_f32(svptrue_b32(), vacc1);
            tmpbuf[tid].values[j+2] = svaddv_f32(svptrue_b32(), vacc2);
            tmpbuf[tid].values[j+3] = svaddv_f32(svptrue_b32(), vacc3);
        }

        for (; j < m; j++) {
            const float *b = &w[j*stride];
            svfloat32_t local_sum_vec = svdup_n_f32(0.0f);
            int a=start;
            for (; a + svcntw() <= end; a += svcntw()) {
                svbool_t pg = svptrue_b32();
                svfloat32_t vec_a = svld1_f32(pg, &ptr1[a]);
                svfloat32_t vec_b = svld1_f32(pg, &b[a]);
                local_sum_vec = svmla_f32_x(pg, local_sum_vec, vec_a, vec_b);
            }
            if ( a < end) {
                svbool_t pg = svwhilelt_b32((int64_t)a, (int64_t)end);
                svfloat32_t vec_a = svld1_f32(pg, &ptr1[a]);
                svfloat32_t vec_b = svld1_f32(pg, &b[a]);
                local_sum_vec = svmla_f32_m(pg, local_sum_vec, vec_a, vec_b);
            }
            tmpbuf[tid].values[j+0] = svaddv_f32(svptrue_b32(), local_sum_vec);
        }

        for (; j < m; j++) {
            const float *b = &w[j*stride];
            svfloat32_t local_sum_vec = svdup_n_f32(0.0f);
            int a=start;
            for (; a + svcntw() <= end; a += svcntw()) {
                svbool_t pg = svptrue_b32();
                svfloat32_t vec_a = svld1_f32(pg, &ptr1[a]);
                svfloat32_t vec_b = svld1_f32(pg, &b[a]);
                local_sum_vec = svmla_f32_x(pg, local_sum_vec, vec_a, vec_b);
            }
            if ( a < end) {
                svbool_t pg = svwhilelt_b32((int64_t)a, (int64_t)end);
                svfloat32_t vec_a = svld1_f32(pg, &ptr1[a]);
                svfloat32_t vec_b = svld1_f32(pg, &b[a]);
                local_sum_vec = svmla_f32_m(pg, local_sum_vec, vec_a, vec_b);
            }
            tmpbuf[tid].values[j+0] = svaddv_f32(svptrue_b32(), local_sum_vec);
        }
        j=start;
        svfloat32_t sum_vec1 = svdup_n_f32(0.0f);
        for (; j + svcntw() <= end; j += svcntw()) {
            svbool_t pg = svptrue_b32();
            svfloat32_t vec_a = svld1_f32(pg, &ptr1[j]);
            //svfloat32_t vec_b = svld1_f32(pg, &b[j]);
            sum_vec1 = svmla_f32_x(pg, sum_vec1, vec_a, vec_a);
        }
        if ( j < end) {
            svbool_t pg = svwhilelt_b32((int64_t)j, (int64_t)end);
            svfloat32_t vec_a = svld1_f32(pg, &ptr1[j]);
            //svfloat32_t vec_b = svld1_f32(pg, &b[j]);
            sum_vec1 = svmla_f32_m(pg, sum_vec1, vec_a, vec_a);
        }
        tmpbuf[tid].values[m] = svaddv_f32(svptrue_b32(), sum_vec1);
        #pragma omp barrier

#pragma omp master
    {
        for(int k=0; k<m+1; k++){
            for(int l=0; l<nth; l++){
                local_sum1[k] += tmpbuf[l].values[k];
            }
        }
#ifdef MPICH
            use_MPI(local_sum1, global_sum1, m+1, MPI_SUM);
#else
            memcpy(global_sum1, local_sum1, (m+1) * sizeof(MATRIXTYPE));
#endif
            for (int k = 0; k < m; ++k){
                Hsbg[k][i] = global_sum1[k];
            } 
    }
    #pragma omp barrier

        j = start;
        for (; j + svcntw() <= end; j += svcntw()) {
            svbool_t pg = svptrue_b32();                // 全真谓词
            svfloat32_t acc = svld1_f32(pg, &ptr1[j]);  // 加载当前块到累加器

            for (int k = 0; k < m; ++k) {
                svfloat32_t vec_k = svld1_f32(pg, &w[k * stride + j]); 
                acc = svmla_f32_x(pg, acc, svdup_f32(-global_sum1[k]), vec_k); 
            }
            svst1_f32(pg, &ptr1[j], acc);           
        }
        if (j < end) {
            svbool_t pg = svwhilelt_b32(j, end);        
            svfloat32_t acc = svld1_f32(pg, &ptr1[j]);
            for (int k = 0; k < m; ++k) {
                svfloat32_t vec_k = svld1_f32(pg, &w[k * stride + j]);
                acc = svmla_f32_m(pg, acc, svdup_f32(-global_sum1[k]), vec_k);
            }
            svst1_f32(pg, &ptr1[j], acc);
        }
        #pragma omp barrier

#pragma omp master
{
    for (int k = 0; k < m; ++k) {
        Hsbg[k][i] += global_sum1[k];
    }
    nrm = sqrt(global_sum1[m]);
    Hsbg[i + 1][i] = nrm;
    if (nrm <= 0.0) {
        printf("Warning: zero norm in orthogonalization step %d.\n", i);
        exit(-1);
    }
    nrm = 1.0f / nrm;
} 
#pragma omp barrier

        j = start;
        for (; j + svcntw() <= end; j += svcntw()) {
            svbool_t pg = svptrue_b32();                // 全真谓词
            svfloat32_t acc = svld1_f32(pg, &ptr1[j]);  // 加载当前块到累加器

            for (int k = 0; k < m; ++k) {
                svfloat32_t vec_k = svld1_f32(pg, &w[k * stride + j]); 
                acc = svmla_f32_x(pg, acc, svdup_f32(-global_sum1[k]), vec_k); 
            }
            svst1_f32(pg, &ptr1[j], acc);           
        }
        if (j < end) {
            svbool_t pg = svwhilelt_b32(j, end);        
            svfloat32_t acc = svld1_f32(pg, &ptr1[j]);
            for (int k = 0; k < m; ++k) {
                svfloat32_t vec_k = svld1_f32(pg, &w[k * stride + j]);
                acc = svmla_f32_m(pg, acc, svdup_f32(-global_sum1[k]), vec_k);
            }
            svst1_f32(pg, &ptr1[j], acc);
        }
        #pragma omp barrier
        
    svfloat32_t inv_vec = svdup_f32(nrm);
    j = start;
    for (; j + svcntw() <= end; j += svcntw()) {
        svbool_t pg = svptrue_b32();               
        svfloat32_t b_vec = svld1_f32(pg, &ptr1[j]);  
        b_vec = svmul_f32_x(pg, b_vec, inv_vec);   
        svst1_f32(pg, &ptr1[j], b_vec);               
    }
    if (j < end) {
        svbool_t pg = svwhilelt_b32(j, end);       
        svfloat32_t b_vec = svld1_f32(pg, &ptr1[j]);
        b_vec = svmul_f32_m(pg, b_vec, inv_vec);
        svst1_f32(pg, &ptr1[j], b_vec);
    }
}
    delete[] local_sum1;
    delete[] global_sum1;
    //delete[] local_buf;
    //delete[] global_buf;
    delete[] tmpbuf;
}


void ClassicalGramSchmidt_sve_float(IntType i, float **&Hsbg, float *&w,
    IntType nTCell, IntType nBFace, IntType nvar, IntType matrixN, IntType NUM_OF_THREADS) {
    IntType nT5 = nTCell * nvar;
    float *ptr1 = &w[(i + 1) * matrixN * nvar];
    IntType m = i + 1;

    // fisrt projection
    float *local_sum1 = new float[m]();
    projection_sve_omp_float(ptr1, w, m, matrixN*nvar, nT5, local_sum1, NUM_OF_THREADS);
    
    //first MPI allreduce
    float *global_sum1 = new float[m]();
#ifdef MPICH
    use_MPI(local_sum1, global_sum1, m, MPI_SUM);
#else
    memcpy(global_sum1, local_sum1, m * sizeof(MATRIXTYPE));
#endif
    delete[] local_sum1;

    //vector substraction & update
    for (int k = 0; k < m; ++k) {
        Hsbg[k][i] = global_sum1[k]; 
    }

    vector_sub_sve_omp_float(ptr1, w, m, matrixN*nvar, nT5, global_sum1);
    delete[] global_sum1;

    // second projection & norm calculation ----------
    int bufsize = m + 1;                       
    float *local_buf = new float[bufsize]();
    projection_sve_omp_float(ptr1, w, m, matrixN*nvar, nT5, local_buf, NUM_OF_THREADS);

    // partial norm
    float local_nrm2 = AdotA_self_sve_omp_float(ptr1, nT5, NUM_OF_THREADS);
    local_buf[m] = local_nrm2;

    // MPI Allreduce 
    float *global_buf = new float[bufsize]();
#ifdef MPICH
    use_MPI(local_buf, global_buf, bufsize, MPI_SUM);
#else
    memcpy(global_buf, local_buf, bufsize * sizeof(MATRIXTYPE));
#endif
    delete[] local_buf;

    // Hessenberg matrix
    for (int k = 0; k < m; ++k) {
        Hsbg[k][i] += global_buf[k];
    }
    float nrm = sqrt(global_buf[m]);
    Hsbg[i + 1][i] = nrm;

    if (nrm > 0.0) {
        vector_div_scalar_sve_omp_float(ptr1, nrm, nT5);
    } else {
        printf("Warning: zero norm in orthogonalization step %d.\n", i);
        exit(-1);
    }

    delete[] global_buf;
}


void vector_initial_omp_sve_float(
    float* b, 
    const float* res_tmp, 
    IntType nT5,
    float* norm0, 
    float* beta,
    const int MAX_THREADS)
{
    float local_sums[2] = {0.0f, 0.0f};
    float global_sums[2];
    thread_private *tmpbuf = new thread_private[ MAX_THREADS ];
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        tmpbuf[tid].values[0] = 0.0f; tmpbuf[tid].values[1] = 0.0f;
        IntType chunk = (nT5 + nth - 1) / nth;
        IntType start = tid * chunk;
        IntType end = (start + chunk < nT5) ? start + chunk : nT5;

        svfloat32_t sum_res = svdup_f32(0.0f);
        svfloat32_t sum_b   = svdup_f32(0.0f);
        IntType i = start;
        for (; i + svcntw() <= end; i += svcntw()) {
            svbool_t pg = svptrue_b32();
            svfloat32_t b_vec   = svld1_f32(pg, &b[i]);
            svfloat32_t res_vec = svld1_f32(pg, &res_tmp[i]);

            svfloat32_t new_b = svsub_f32_x(pg, b_vec, res_vec);
            svst1_f32(pg, &b[i], new_b);

            svfloat32_t res_sq = svmul_f32_x(pg, res_vec, res_vec);
            sum_res = svadd_f32_x(pg, sum_res, res_sq);

            svfloat32_t b_sq = svmul_f32_x(pg, new_b, new_b);
            sum_b = svadd_f32_x(pg, sum_b, b_sq);
        }
        if (i < end) {
            svbool_t pg = svwhilelt_b32(i, end);
            svfloat32_t b_vec   = svld1_f32(pg, &b[i]);
            svfloat32_t res_vec = svld1_f32(pg, &res_tmp[i]);

            svfloat32_t new_b = svsub_f32_m(pg, b_vec, res_vec);
            svst1_f32(pg, &b[i], new_b);

            svfloat32_t res_sq = svmul_f32_x(pg, res_vec, res_vec);
            sum_res = svadd_f32_m(pg, sum_res, res_sq);

            svfloat32_t b_sq = svmul_f32_x(pg, new_b, new_b);
            sum_b = svadd_f32_m(pg, sum_b, b_sq);
        }
        tmpbuf[tid].values[0] = svaddv_f32(svptrue_b32(), sum_res);
        tmpbuf[tid].values[1] = svaddv_f32(svptrue_b32(), sum_b); 
    #pragma omp barrier

#pragma omp master
{
    for( int j=0; j<MAX_THREADS; j++){
        local_sums[0] += tmpbuf[j].values[0];
        local_sums[1] += tmpbuf[j].values[1];
    }
#ifdef MPICH
        use_MPI(local_sums, global_sums, 2, MPI_SUM);
#else
        global_sums[0] = local_sums[0];
        global_sums[1] = local_sums[1];
#endif
        *norm0 = sqrt(global_sums[0]);
        *beta  = sqrt(global_sums[1]);
}
#pragma omp barrier

        //vector_div_scalar_sve_omp_float(b, -*beta, nT5);
        svfloat32_t inv_vec = svdup_f32((1.0f / (0 - *beta)));
        i = start;
        for (; i + svcntw() <= end; i += svcntw()) {
            svbool_t pg = svptrue_b32();               
            svfloat32_t b_vec = svld1_f32(pg, &b[i]);  
            b_vec = svmul_f32_x(pg, b_vec, inv_vec); 
            svst1_f32(pg, &b[i], b_vec);               
        }
        if (i < end) {
            svbool_t pg = svwhilelt_b32(i, end);      
            svfloat32_t b_vec = svld1_f32(pg, &b[i]);
            b_vec = svmul_f32_m(pg, b_vec, inv_vec);
            svst1_f32(pg, &b[i], b_vec);
        }
    }
    delete[] tmpbuf;
}


void vector_initial_omp_sve_double(
    double* b,
    const double* res_tmp,
    IntType nT5,
    double* norm0,
    double* beta)
{
    double local_sums[2] = {0.0, 0.0};
    double global_sums[2];

    #pragma omp parallel reduction(+:local_sums[:2])
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        IntType chunk = (nT5 + nth - 1) / nth;
        IntType start = tid * chunk;
        IntType end = (start + chunk < nT5) ? start + chunk : nT5;

        svfloat64_t sum_res = svdup_f64(0.0);
        svfloat64_t sum_b   = svdup_f64(0.0);

        IntType i = start;
        for (; i + svcntd() <= end; i += svcntd()) {
            svbool_t pg = svptrue_b64();
            svfloat64_t b_vec   = svld1_f64(pg, &b[i]);
            svfloat64_t res_vec = svld1_f64(pg, &res_tmp[i]);

            svfloat64_t new_b = svsub_f64_x(pg, b_vec, res_vec);
            svst1_f64(pg, &b[i], new_b);

            svfloat64_t res_sq = svmul_f64_x(pg, res_vec, res_vec);
            sum_res = svadd_f64_x(pg, sum_res, res_sq);

            svfloat64_t b_sq = svmul_f64_x(pg, new_b, new_b);
            sum_b = svadd_f64_x(pg, sum_b, b_sq);
        }

        if (i < end) {
            svbool_t pg = svwhilelt_b64((int64_t)i, (int64_t)end);
            svfloat64_t b_vec   = svld1_f64(pg, &b[i]);
            svfloat64_t res_vec = svld1_f64(pg, &res_tmp[i]);

            svfloat64_t new_b = svsub_f64_m(pg, b_vec, res_vec);
            svst1_f64(pg, &b[i], new_b);

            svfloat64_t res_sq = svmul_f64_x(pg, res_vec, res_vec);
            sum_res = svadd_f64_m(pg, sum_res, res_sq);

            svfloat64_t b_sq = svmul_f64_x(pg, new_b, new_b);
            sum_b = svadd_f64_m(pg, sum_b, b_sq);
        }

        local_sums[0] += svaddv_f64(svptrue_b64(), sum_res);
        local_sums[1] += svaddv_f64(svptrue_b64(), sum_b);
    }

#ifdef MPICH
    MPI_Allreduce(local_sums, global_sums, 2, MATRIXMPITYPE, MPI_SUM, MPI_COMM_WORLD);
    //use_MPI(local_sums, global_sums, 2, MPI_SUM);
#else
    global_sums[0] = local_sums[0];
    global_sums[1] = local_sums[1];
#endif
    
    *norm0 = sqrt(global_sums[0]);
    *beta  = sqrt(global_sums[1]);

    vector_div_scalar_sve_omp_double(b, -*beta, nT5);
}


void cholesky_solve(MATRIXTYPE *G, MATRIXTYPE *b, int s) {
    // 列主序 Cholesky 分解（只修改下三角部分）
    for (int j = 0; j < s; j++) {
        // 计算 L[j][j]
        MATRIXTYPE sum = 0.0f;
        for (int k = 0; k < j; k++) {
            MATRIXTYPE L_jk = G[j + k * s];   // L[j][k] 位于 (j,k)，列主序
            sum += L_jk * L_jk;
        }
        MATRIXTYPE diag = G[j + j * s] - sum;
        if (diag <= 0.0f) {
            // 矩阵可能非正定，添加微小扰动或报错
            diag = 1e-12f;
        }
        MATRIXTYPE L_jj = sqrtf(diag);
        G[j + j * s] = L_jj;   // 存储 L[j][j]

        // 计算 L[i][j] for i > j
        for (int i = j+1; i < s; i++) {
            sum = 0.0f;
            for (int k = 0; k < j; k++) {
                MATRIXTYPE L_ik = G[i + k * s];
                MATRIXTYPE L_jk = G[j + k * s];
                sum += L_ik * L_jk;
            }
            MATRIXTYPE val = G[i + j * s] - sum;   // 原始 G[i][j]
            G[i + j * s] = val / L_jj;
        }
    }

    // 前代：解 L y = b
    for (int i = 0; i < s; i++) {
        MATRIXTYPE sum = 0.0f;
        for (int k = 0; k < i; k++) {
            sum += G[i + k * s] * b[k];   // L[i][k]
        }
        b[i] = (b[i] - sum) / G[i + i * s];
    }

    // 回代：解 L^T x = y
    for (int i = s-1; i >= 0; i--) {
        MATRIXTYPE sum = 0.0f;
        for (int k = i+1; k < s; k++) {
            sum += G[k + i * s] * b[k];   // L[k][i] 位于 (k,i)
        }
        b[i] = (b[i] - sum) / G[i + i * s];
    }
}

void cholesky_decompose(double *G, int s) {
    for (int j = 0; j < s; j++) {
        for (int i = j; i < s; i++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += G[i * s + k] * G[j * s + k];
            }
            if (i == j) {
                double diag = G[i * s + i] - sum;
                if (diag <= 0.0) diag = 1e-14f;   // 数值保护
                G[i * s + i] = sqrt(diag); //sqrtf(diag);
            } else {
                G[i * s + j] = (G[i * s + j] - sum) / G[j * s + j];
            }
        }
    }

// double a = G[0];     // G[0][0]
// double b = G[1];     // G[0][1] (对称)
// double c = G[3];     // G[1][1]
// double L11 = sqrt(a);
// double L21 = b / L11;
// double L22 = sqrt(c - L21*L21);
// // 存储 L 矩阵：下三角
// G[0] = L11;
// G[1*s + 0] = L21;   // G[1][0]
// G[1*s + 1] = L22;
}

// ==================== 主函数 ==================== //
void s_step_orthogonalization(IntType m,                  // 已有基向量个数
                              MATRIXTYPE **&Hsbg,        // Hessenberg 矩阵 (m+s+1) x (m+s)
                              MATRIXTYPE *&w,            // 所有基向量（列主序，每列 stride 个元素）
                              IntType nTCell,            // 每个向量的长度
                              IntType nvar,
                              IntType stride,            // 相邻基向量的地址偏移（= matrixN * nvar）
                              const MATRIXTYPE* __restrict A,       // 稀疏矩阵 A（用于 apply_A）
                              const IntType* __restrict row_ptr, 
                              const IntType* __restrict col_ind, 
                              const IntType* __restrict dia_ptr,
                              const IntType MAX_THREADS,
                              int s) {   // 本次生成的向量个数（s 通常取 2~5）

    IntType nT5 = nTCell * nvar;
    // ---------- 1. 生成 s 个 Krylov 向量（无通信） ----------
    // 分配临时存储 V_new（nT5 × s，列主序）
    MATRIXTYPE *V_new = (MATRIXTYPE*)calloc( nT5 * s, sizeof(MATRIXTYPE));

    // 最后一个已有基向量为 w[(m-1)*stride]
    const MATRIXTYPE *v_last = &w[(m-1) * stride];
    spmv_bsr5_sve_omp_float(row_ptr, col_ind, A, V_new, v_last, nTCell, nvar);

    for (int i = 0; i < nT5; i++) {
    if (isnan(V_new[i]) || isinf(V_new[i])) {
        printf("V_new NAN detected, rank %d: V_new[%d] = %f (NaN/Inf)\n", 0, i, V_new[i]);
        exit(-1);
    }
    }

    // 计算后续向量 v_t = A * v_{t-1}（t=2..s） wrong!!!!
    for (int t = 1; t < s; t++) {
        const MATRIXTYPE *v_prev = &V_new[(t-1) * nT5];
        MATRIXTYPE *v_curr = &V_new[t * nT5];
        spmv_bsr5_sve_omp_float(row_ptr, col_ind, A, v_curr, v_prev, nTCell, nvar); 
    }

    // ---------- 2. 批量计算所有需要的内积（稠密矩阵乘法） ----------
    //    H_old_new = Q^T * V_new（大小 m × s）
    //    G = V_new^T * V_new（大小 s × s，对称）
    MATRIXTYPE *H_old_new = (MATRIXTYPE*)calloc( m * s, sizeof(MATRIXTYPE));
    MATRIXTYPE *G = (MATRIXTYPE*)malloc(s * s * sizeof(MATRIXTYPE));

    // 调用稠密矩阵乘法计算 H_old_new = w^T * V_new（注意 w 是列主序，行数 nT5，列数 m）
    // 函数接口假设：dense_matmul(A, rowsA, colsA, B, rowsB, colsB, C, rowsC, colsC, alpha, beta, transA, transB)
    // 这里计算 C = alpha * op(A) * op(B) + beta * C
    // 我们需要 C = w^T * V_new，即 A = w（nT5×m），B = V_new（nT5×s），转置A，不转置B。
    // 设 alpha=1, beta=0，则 C = A^T * B
    cblas_sgemm(CblasColMajor,               // 列主序
                CblasTrans,                  // transA: A^T
                CblasNoTrans,                // transB: B
                m, s, nT5,                   // M = m, N = s, K = nT5
                1.0f,                        // alpha
                w, nT5,                      // A: 矩阵 w，lda = nT5
                V_new, nT5,                  // B: 矩阵 V_new，ldb = nT5
                0.0f,                        // beta
                H_old_new, m);               // C: 结果矩阵 H_old_new，ldc = m

    // 计算 G = V_new^T * V_new（对称，只需上三角）
    cblas_sgemm(CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                s, s, nT5,
                1.0f,
                V_new, nT5,
                V_new, nT5,
                0.0f,
                G, s);

    // ---------- 3. 全局归约（一次 MPI_Allreduce）----------
    // int total_ops = m * s + s * s;
    // MATRIXTYPE *local_pack = (MATRIXTYPE*)malloc(total_ops * sizeof(MATRIXTYPE));
    // MATRIXTYPE *global_pack = (MATRIXTYPE*)malloc(total_ops * sizeof(MATRIXTYPE));
    // memcpy(local_pack, H_old_new, m * s * sizeof(MATRIXTYPE));
    // memcpy(&local_pack[m*s], G, s * s * sizeof(MATRIXTYPE));
    // use_MPI(local_pack, global_pack, total_ops, MPI_SUM);
    // memcpy(H_old_new, global_pack, m * s * sizeof(MATRIXTYPE));
    // memcpy(G, &global_pack[m*s], s * s * sizeof(MATRIXTYPE));
    // cholesky_decompose(G, s);

    int total_ops = m * s + s * s;
    double *local_pack = (double*)malloc(total_ops * sizeof(double));
    double *global_pack = (double*)malloc(total_ops * sizeof(double));    
    for(int i=0; i<m*s; i++) {local_pack[i] = H_old_new[i];}
    for(int i=0; i<s*s; i++) {local_pack[m*s+i] = G[i];}
    MPI_Allreduce(local_pack, global_pack, total_ops, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double* G_copy = (double*)malloc(s * s * sizeof(double));
    for(int i=0; i<m*s; i++) {H_old_new[i] = global_pack[i];}
    for(int i=0; i<s*s; i++) {G_copy[i]         = global_pack[m*s+i];}
    cholesky_decompose(G_copy, s);   // 只分解不求解
    for(int i=0; i<s*s; i++) {G[i] = G_copy[i];}
    free(G_copy);

    for (int t = 0; t < s; t++) {
        for (int j = 0; j < t; j++) {
            float dot = cblas_sdot(nT5, &V_new[t*nT5], 1, &V_new[j*nT5], 1);
            //double dot = cblas_sdot(nT5, &V_new[t*nT5], 1, &V_new[j*nT5], 1);
            for (int i = 0; i < nT5; i++) V_new[t*nT5 + i] -= dot * V_new[j*nT5 + i];
        }
        // 重新归一化
        float norm = cblas_snrm2(nT5, &V_new[t*nT5], 1);
        if (norm > 0) cblas_sscal(nT5, 1.0/norm, &V_new[t*nT5], 1);
    }

    // 计算 Q_new = V_new * L^{-T}  （即解 L^T * X = V_new^T 的转置）
    // 使用 cblas_strsm，右侧乘 L^{-T}
    cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
            nT5, s, 1.0f, G, s, V_new, nT5);

    // 计算 R = H_old_new * L^{-T}
    cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
            m, s, 1.0f, G, s, H_old_new, m);

    // 将 H_old_new 的列存入 Hsbg 的对应列（列索引 m 到 m+s-1）
    for (int t = 0; t < s; t++) {
        for (int i = 0; i < m; i++) {
            Hsbg[i][m + t] = H_old_new[t * m + i];   // 注意列主序索引
        }
        // 子对角线元素（s-step Arnoldi 中只有最后一个子对角线非零）
        // 对于新生成的基向量，其范数为 1，但 Hessenberg 矩阵的子对角线元素应为 1 或由算法决定
        // 这里简单置为 1，更严格的实现需参考 s-step Arnoldi 论文
        Hsbg[m + t + 1][m + t] = 1.0f;
    }

    // 将 V_new 的列追加到全局基向量数组 w 中（已正交归一）
    double aa = 0.0;
    for (int t = 0; t < s; t++) {
        MATRIXTYPE *new_col = &w[(m + t) * stride];
        memcpy(new_col, &V_new[t * nT5], nT5 * sizeof(MATRIXTYPE));
    }

    // 释放内存
    free(V_new);
    free(H_old_new);
    free(G);
    free(local_pack);
    free(global_pack);
}

}
#endif