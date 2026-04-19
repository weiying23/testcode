#include "temporal_discretisation_implicit.h"

// C++ build-in head files
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cassert>
#include <string>
#include <iostream>
#include <algorithm>
#include <math.h>

#include <stdexcept>
#include <unistd.h>
#include <sys/mman.h>
#include <numa.h>
#include <numaif.h>

#ifdef ROOFLINE_EVENTS
    #include "roofline_events.h"
#endif
// adding eigen 
//#define EIGEN_USE_BLAS
#include </home/share/daizhe/daizhe/soft//eigen-5.0.0/Eigen/Dense>    
#include </home/share/daizhe/daizhe/soft//eigen-5.0.0/Eigen/LU>      

#include "kupl.h"
using namespace std;

// other user defined head files
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
#include "gmres_ilu.h"
#include "sve_optimization.h"

#include <stdint.h>
#include "memory_util.h"
#include "kblas.h"

//#include "reorder/reorder.hpp"
//#include "mtxops/read_mtx.h"

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif

#ifdef MPICH
#include <mpi.h>
#endif

#ifdef FS_OPENMP
#include "omp.h"
#endif

extern double ILUbuild, ILUexe, Matrixbuild, GMRESexe, MPIexe, GMRES_Schmidt;
extern int ite;
MATRIXTYPE *x_global = nullptr, *b_global =nullptr;
GETYPE *diagmatrix_global = nullptr;

#define STEP 2
namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 19001  // define file id


#ifdef MPICH
extern MPI_Comm comm_L2, comm_L3, comm_L4;
extern int myZone;  // zhnc add
#endif

const int NVAR = 5;
MATRIXTYPE DotProductMPI(MATRIXTYPE *a, IntType n)
{
    IntType i;
    MATRIXTYPE sum = 0.0, sum_glb=0.0;

#ifdef FS_OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(i=0; i<n; i++) {
        sum += a[i]*a[i];
    }
#ifdef MPICH
    MPI_Allreduce(&sum, &sum_glb, 1, MATRIXMPITYPE, MPI_SUM, MPI_COMM_WORLD);
    sum = sum_glb;
#endif
    //printf("sum:%.8f  sum_glb:%.8f\n", sum, sum_glb);
    return sum;
}

MATRIXTYPE GMRESDotProduct(MATRIXTYPE *a, MATRIXTYPE *b, IntType n)
{
    IntType  i;
    MATRIXTYPE sum = 0.0, sum_glb=0.0;

#ifdef FS_OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(i=0; i<n; i++) {
        sum += a[i]*b[i];
    }

#ifdef MPICH
    MPI_Allreduce(&sum, &sum_glb, 1, MATRIXMPITYPE, MPI_SUM, MPI_COMM_WORLD);
    sum = sum_glb;
#endif
    return sum;
}

void GmresMinResB(MATRIXTYPE *&b, MATRIXTYPE *res, IntType n){
#ifdef FS_OPENMP
#pragma omp parallel for
#endif  
    for(IntType i=0; i<n; i++){
        b[i] -= res[i];
    }
}

void bDivScalar(MATRIXTYPE*& b, MATRIXTYPE scalar, IntType n){
#ifdef FS_OPENMP
#pragma omp parallel for
#endif  
    for(IntType i=0; i<n; i++){
        b[i] /= scalar;
    }
}

void SubScalarMul(MATRIXTYPE *&a, MATRIXTYPE *b, MATRIXTYPE scalar, IntType n){
#ifdef FS_OPENMP
#pragma omp parallel for
#endif  
    for(IntType i=0; i<n; i++){
        a[i] -= scalar*b[i];
    }
}

static inline MATRIXTYPE Sign(MATRIXTYPE x, MATRIXTYPE y) {
    if (y == 0.0) return 0.0;
    return fabs(x) * (y < 0.0 ? -1.0 : 1.0);
}

void ApplyGivens(MATRIXTYPE s, MATRIXTYPE c, MATRIXTYPE & h1, MATRIXTYPE & h2){

  MATRIXTYPE temp = c*h1 + s*h2;
  h2 = c*h2 - s*h1;
  h1 = temp;

}

void GenerateGivens(MATRIXTYPE & dx, MATRIXTYPE & dy, MATRIXTYPE & s, MATRIXTYPE & c){

  if ( (dx == 0.0) && (dy == 0.0) ) {
    c = 1.0;
    s = 0.0;
  }
  else if ( fabs(dy) > fabs(dx) ) {
    MATRIXTYPE tmp = dx/dy;
    dx = sqrt(1.0 + tmp*tmp);
    s = Sign(1.0/dx, dy);
    c = tmp*s;
  }
  else if ( fabs(dy) <= fabs(dx) ) {
    MATRIXTYPE tmp = dy/dx;
    dy = sqrt(1.0 + tmp*tmp);
    c = Sign(1.0/dy, dx);
    s = tmp*c;
  }
  else {
    // dx and/or dy must be invalid
    dx = 0.0;
    dy = 0.0;
    c = 1.0;
    s = 0.0;
  }
  dx = fabs(dx*dy);
  dy = 0.0;
}


void SolveReduced(int n, MATRIXTYPE** Hsbg, MATRIXTYPE* rhs, MATRIXTYPE *&x){
  // initialize...
  for (int i = 0; i < n; i++)
    x[i] = rhs[i];
  // ... and backsolve
  for (int i = n-1; i >= 0; i--) {
    x[i] /= Hsbg[i][i];
    for (int j = i-1; j >= 0; j--) {
      x[j] -= Hsbg[j][i]*x[i];
    }
  }
}

void Addx( MATRIXTYPE* __restrict x_final, MATRIXTYPE* __restrict x, MATRIXTYPE factor, IntType n){
#ifdef FS_OPENMP
#pragma omp parallel for
#endif  
    for(IntType i=0; i<n; i++){
        x_final[i] += factor*x[i];
    }
}

void ClassicalGramSchmidt_hybrid(IntType i, MATRIXTYPE **&Hsbg, MATRIXTYPE *&w,
                          IntType nTCell, IntType nBFace, IntType nvar, IntType matrixN) {
    IntType nT5 = nTCell * nvar;
    MATRIXTYPE *ptr1 = &w[(i + 1) * matrixN * nvar];
    IntType m = i + 1;

    MATRIXTYPE *local_sum1 = new MATRIXTYPE[m]();
    MATRIXTYPE *global_sum1 = new MATRIXTYPE[m]();
    IntType bufsize = m + 1;
    MATRIXTYPE *local_buf = new MATRIXTYPE[bufsize]();
    MATRIXTYPE *global_buf = new MATRIXTYPE[bufsize]();
    MATRIXTYPE local_nrm2 = 0.0, nrm = 0.0;

    #pragma omp parallel shared(local_nrm2,nrm,local_sum1,local_buf)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();

        #pragma omp for
        for (int k = 0; k < m; ++k) {
            MATRIXTYPE *ptr2 = &w[k * matrixN * nvar];
            MATRIXTYPE sumscalar = 0.0;
            #pragma omp simd reduction(+:sumscalar) aligned(ptr1, ptr2:16)
            for (int l = 0; l < nT5; ++l) {
                sumscalar += ptr1[l] * ptr2[l];
            }
            local_sum1[k] = sumscalar;
        }

        #pragma omp master
        {
#ifdef MPICH
            MPI_Allreduce(local_sum1, global_sum1, m, MATRIXMPITYPE, MPI_SUM, MPI_COMM_WORLD);
#else
            memcpy(global_sum1, local_sum1, m * sizeof(MATRIXTYPE));
#endif
            for (int k = 0; k < m; ++k){
                Hsbg[k][i] = global_sum1[k];
            } 
        }
        #pragma omp barrier
        
        #pragma omp for
        for (IntType idx = 0; idx < nT5; ++idx) {
            MATRIXTYPE delta = 0.0;
            for (int k = 0; k < m; ++k) {
                delta += global_sum1[k] * w[k * matrixN * nvar + idx];
            }
            ptr1[idx] -= delta;
        }

        #pragma omp for
        for (int k = 0; k < m; ++k) {
            MATRIXTYPE *ptr2 = &w[k * matrixN * nvar];
            MATRIXTYPE sumscalar = 0.0;
            #pragma omp simd reduction(+:sumscalar) aligned(ptr1, ptr2:16)
            for (int l = 0; l < nT5; ++l) {
                sumscalar += ptr1[l] * ptr2[l];
            }
            local_buf[k] = sumscalar;
        }

        #pragma omp for reduction(+:local_nrm2)
        for (IntType idx = 0; idx < nT5; ++idx) {
            local_nrm2 += ptr1[idx] * ptr1[idx];
        }

        #pragma omp master
        {
            local_buf[m] = local_nrm2;
#ifdef MPICH
            MPI_Allreduce(local_buf, global_buf, bufsize, MATRIXMPITYPE, MPI_SUM, MPI_COMM_WORLD);
#else
            memcpy(global_buf, local_buf, bufsize * sizeof(MATRIXTYPE));
#endif
            for (int k = 0; k < m; ++k) {
                Hsbg[k][i] += global_buf[k];
            }
            nrm = sqrt(global_buf[m]);
            Hsbg[i + 1][i] = nrm;

            if (nrm <= 0.0) {
                printf("Warning: zero norm in orthogonalization step %d.\n", i);
                exit(-1);
            }
        } 
        #pragma omp barrier

        #pragma omp for
        for (IntType idx = 0; idx < nT5; ++idx) {
            ptr1[idx] /= nrm;
        }
    }
    delete[] local_sum1;
    delete[] global_sum1;
    delete[] local_buf;
    delete[] global_buf;
}

void ClassicalGramSchmidt(IntType i, MATRIXTYPE **&Hsbg, MATRIXTYPE *&w,
                          IntType nTCell, IntType nBFace, IntType nvar, IntType matrixN) {
    IntType nT5 = nTCell * nvar;
    MATRIXTYPE* ptr1 = &w[(i + 1) * matrixN * nvar];
    IntType m = i + 1;

    // ---------- 第一次投影 ----------
    MATRIXTYPE *local_sum1 = new MATRIXTYPE[m]();
    #pragma omp parallel for
    for (int k = 0; k < m; ++k) {
        MATRIXTYPE* __restrict ptr2 = &w[k * matrixN * nvar];
        MATRIXTYPE sumscalar = 0.0;
        #pragma omp simd reduction(+:sumscalar) aligned(ptr1, ptr2:16)
        for (int l = 0; l < nT5; ++l) {
            sumscalar += ptr1[l] * ptr2[l];
        }
        local_sum1[k] = sumscalar;
        //local_sum1[k] = cblas_sdot(nT5, ptr1, 1, ptr2, 1);
    }
    
    MATRIXTYPE *global_sum1 = new MATRIXTYPE[m]();
#ifdef MPICH
    MPI_Allreduce(local_sum1, global_sum1, m, MATRIXMPITYPE, MPI_SUM, MPI_COMM_WORLD);
#else
    memcpy(global_sum1, local_sum1, m * sizeof(MATRIXTYPE));
#endif
    delete[] local_sum1;
    // 第一次更新：减去所有投影
    for (int k = 0; k < m; ++k) {
        Hsbg[k][i] = global_sum1[k];           // 存储第一次投影系数
        MATRIXTYPE* __restrict ptr2 = &w[k * matrixN * nvar];
        SubScalarMul(ptr1, ptr2, global_sum1[k], nT5);
        //cblas_saxpy(nT5, -global_sum1[k], ptr2, 1, ptr1, 1);
    }
    delete[] global_sum1;

    // ---------- 第二次投影 & 范数计算 ----------
    int bufsize = m + 1;                       
    MATRIXTYPE *local_buf = new MATRIXTYPE[bufsize]();
    #pragma omp parallel for
    for (int k = 0; k < m; ++k) {
        MATRIXTYPE* __restrict ptr2 = &w[k * matrixN * nvar];
        MATRIXTYPE sumscalar = 0.0;
        #pragma omp simd reduction(+:sumscalar) aligned(ptr1, ptr2:16)
        for (int l = 0; l < nT5; ++l) {
            sumscalar += ptr1[l] * ptr2[l];
        }
        local_buf[k] = sumscalar;
        //local_buf[k] = cblas_sdot(nT5, ptr1, 1, ptr2, 1);
    }

    // 计算局部范数平方
    MATRIXTYPE local_nrm2 = 0.0;
    #pragma omp simd reduction(+:local_nrm2) aligned(ptr1:16)
    for (int l = 0; l < nT5; ++l) {
        local_nrm2 += ptr1[l] * ptr1[l];
    }
    local_buf[m] = local_nrm2;
    //local_buf[m] = cblas_sdot(nT5, ptr1, 1, ptr1, 1);

    // 全局归约：一次通信得到所有结果
    MATRIXTYPE *global_buf = new MATRIXTYPE[bufsize]();
#ifdef MPICH
    MPI_Allreduce(local_buf, global_buf, bufsize, MATRIXMPITYPE, MPI_SUM, MPI_COMM_WORLD);
#else
    memcpy(global_buf, local_buf, bufsize * sizeof(MATRIXTYPE));
#endif
    delete[] local_buf;

    // 更新Hessenberg矩阵：第二次投影系数累加到第一次系数上（double CGS）
    for (int k = 0; k < m; ++k) {
        Hsbg[k][i] += global_buf[k];
    }
    MATRIXTYPE nrm = sqrt(global_buf[m]);
    Hsbg[i + 1][i] = nrm;


    if (nrm > 0.0) {
        bDivScalar(ptr1, nrm, nT5);
    } else {
        printf("Warning: zero norm in orthogonalization step %d.\n", i);
        exit(-1);
    }

    delete[] global_buf;
}

void ClassicalGramSchmidt_blas(IntType i, MATRIXTYPE **&Hsbg, MATRIXTYPE *&w,
                          IntType nTCell, IntType nBFace, IntType nvar, IntType matrixN) {
    IntType nT5 = nTCell * nvar;
    MATRIXTYPE* ptr1 = &w[(i + 1) * matrixN * nvar];
    IntType m = i + 1;

    // ---------- 第一次投影 ----------
    MATRIXTYPE *local_sum1 = new MATRIXTYPE[m]();
    #pragma omp parallel for
    for (int k = 0; k < m; ++k) {
        MATRIXTYPE* __restrict ptr2 = &w[k * matrixN * nvar];
        MATRIXTYPE sumscalar = 0.0;
        // #pragma omp simd reduction(+:sumscalar) aligned(ptr1, ptr2:16)
        // for (int l = 0; l < nT5; ++l) {
        //     sumscalar += ptr1[l] * ptr2[l];
        // }
        // local_sum1[k] = sumscalar;
        local_sum1[k] = cblas_sdot(nT5, ptr1, 1, ptr2, 1);
    }
    
    MATRIXTYPE *global_sum1 = new MATRIXTYPE[m]();
#ifdef MPICH
    MPI_Allreduce(local_sum1, global_sum1, m, MATRIXMPITYPE, MPI_SUM, MPI_COMM_WORLD);
#else
    memcpy(global_sum1, local_sum1, m * sizeof(MATRIXTYPE));
#endif
    delete[] local_sum1;
    // 第一次更新：减去所有投影
    for (int k = 0; k < m; ++k) {
        Hsbg[k][i] = global_sum1[k];           // 存储第一次投影系数
        MATRIXTYPE* __restrict ptr2 = &w[k * matrixN * nvar];
        // SubScalarMul(ptr1, ptr2, global_sum1[k], nT5);
        cblas_saxpy(nT5, -global_sum1[k], ptr2, 1, ptr1, 1);
    }
    delete[] global_sum1;

    // ---------- 第二次投影 & 范数计算 ----------
    int bufsize = m + 1;                       
    MATRIXTYPE *local_buf = new MATRIXTYPE[bufsize]();
    #pragma omp parallel for
    for (int k = 0; k < m; ++k) {
        MATRIXTYPE* __restrict ptr2 = &w[k * matrixN * nvar];
        MATRIXTYPE sumscalar = 0.0;
        // #pragma omp simd reduction(+:sumscalar) aligned(ptr1, ptr2:16)
        // for (int l = 0; l < nT5; ++l) {
        //     sumscalar += ptr1[l] * ptr2[l];
        // }
        // local_buf[k] = sumscalar;
        local_buf[k] = cblas_sdot(nT5, ptr1, 1, ptr2, 1);
    }

    // 计算局部范数平方
    MATRIXTYPE local_nrm2 = 0.0;
    // #pragma omp simd reduction(+:local_nrm2) aligned(ptr1:16)
    // for (int l = 0; l < nT5; ++l) {
    //     local_nrm2 += ptr1[l] * ptr1[l];
    // }
    // local_buf[m] = local_nrm2;
    local_buf[m] = cblas_sdot(nT5, ptr1, 1, ptr1, 1);

    // 全局归约：一次通信得到所有结果
    MATRIXTYPE *global_buf = new MATRIXTYPE[bufsize]();
#ifdef MPICH
    MPI_Allreduce(local_buf, global_buf, bufsize, MATRIXMPITYPE, MPI_SUM, MPI_COMM_WORLD);
#else
    memcpy(global_buf, local_buf, bufsize * sizeof(MATRIXTYPE));
#endif
    delete[] local_buf;

    // 更新Hessenberg矩阵：第二次投影系数累加到第一次系数上（double CGS）
    for (int k = 0; k < m; ++k) {
        Hsbg[k][i] += global_buf[k];
    }
    MATRIXTYPE nrm = sqrt(global_buf[m]);
    Hsbg[i + 1][i] = nrm;


    if (nrm > 0.0) {
        bDivScalar(ptr1, nrm, nT5);
    } else {
        printf("Warning: zero norm in orthogonalization step %d.\n", i);
        exit(-1);
    }

    delete[] global_buf;
}

void ModGramSchmidt(IntType i, MATRIXTYPE **&Hsbg, MATRIXTYPE *&w, \
    IntType nTCell, IntType nBFace, IntType nvar, IntType matrixN){
      
    /*--- Parameter for reorthonormalization ---*/
  const MATRIXTYPE reorth = 0.98;
  IntType nT5 = nTCell*nvar;
  //IntType n5 = (nTCell+nBFace)*nvar;

  MATRIXTYPE *ptr1 = &w[(i+1)*matrixN*nvar];
  
  /*--- Get the norm of the vector being orthogonalized, and find the
  threshold for re-orthogonalization ---*/
  //MATRIXTYPE nrm = w[i+1].squaredNorm();
  MATRIXTYPE nrm = DotProductMPI( ptr1, nT5);
  MATRIXTYPE thr = nrm * reorth;

  //printf("rank:%d ModGramSchmidt 1 nrm:%.15f \n", mpirank, nrm);

  /*--- The norm of w[i+1] < 0.0 or w[i+1] = NaN ---*/
  if ((nrm <= 0.0) || (nrm != nrm)) {
    /*--- nrm is the result of a dot product, communications are implicitly handled. ---*/
    printf("FGMRES orthogonalization failed, linear solver diverged with nrm:%.6f.\n",nrm);
    exit(0);
  }

  /*--- Begin main Gram-Schmidt loop ---*/

  for (int k = 0; k < i + 1; k++) {
    MATRIXTYPE *ptr2 = &w[k*matrixN*nvar];

    //MATRIXTYPE prod = w[i+1].dot(w[k]);
    MATRIXTYPE prod = GMRESDotProduct( ptr1, ptr2, nT5 );
    Hsbg[k][i] = prod;
    //w[i+1] -= prod * w[k];
    SubScalarMul( ptr1, ptr2, prod, nT5 );

    /*--- Check if reorthogonalization is necessary ---*/
    if (prod * prod > thr) {
      //prod = w[i+1].dot(w[k]);
      prod = GMRESDotProduct( ptr1, ptr2, nT5 );
      Hsbg[k][i] += prod;
      //w[i+1] -= prod * w[k];
      SubScalarMul( ptr1, ptr2, prod, nT5 );
    }

    /*--- Update the norm and check its size ---*/
    nrm -= pow(Hsbg[k][i], 2);
    nrm = max<MATRIXTYPE>(nrm, 0.0);
    thr = nrm * reorth;

  }

  /*--- Test the resulting vector ---*/
  //nrm = w[i+1].norm();
  nrm = DotProductMPI( ptr1, nT5);
  nrm = sqrt(nrm);
  Hsbg[i + 1][i] = nrm;

  /*--- Scale the resulting vector ---*/
  //w[i+1] /= nrm;
  bDivScalar( ptr1, nrm, nT5);

}

void GmresMatrixVectorMult(PolyGrid *grid, IntType *row_ptr, IntType *col_ind, MATRIXTYPE *matrix, 
    MATRIXTYPE *&b, MATRIXTYPE *x, IntType nTCell, IntType nBFace, IntType nIFace, IntType nvar){
    
    IntType nT = nTCell+nBFace;
    #pragma omp parallel for schedule(static)
    for(IntType i=0; i<nTCell; i++){
        MATRIXTYPE *b_ptr = &b[i*nvar];
        for(IntType index=row_ptr[i]; index<row_ptr[i+1];index++){
            IntType col_j = col_ind[index];
            IntType len1 = index*nvar*nvar;
            for(IntType m=0; m<nvar; m++)
              for(IntType n=0; n<nvar; n++)
                b_ptr[m] += matrix[len1+m*nvar+n] * x[col_j*nvar + n];
        }
    }

    #ifdef MPICH
        //grid->RecvSendVarMatrixNeighbor_Togeth( nvar, tmp_x );
        //grid->RecvSendVarMatrixNeighbor_Togeth2( nvar, tmp_x );
    #endif

}

void InverseDiagonalBlock_ILUMatrix(IntType block_i, MATRIXTYPE *&invBlock, IntType nvar, 
    MATRIXTYPE *ILU_matrix, IntType *dia_ptr){

    /*--- Copy block, as the algorithm modifies the matrix ---*/
    MATRIXTYPE block[NVAR*NVAR];
    //MatrixCopy(&ILU_matrix[dia_ptr_ilu[block_i]*nvar*nvar], block);
    for(IntType iVar = 0; iVar < nvar*nvar; ++iVar)
        block[iVar] = ILU_matrix[dia_ptr[block_i]*nvar*nvar + iVar];

    //MatrixInverse(block, invBlock);

#define M(I,J) invBlock[(I)*nvar+(J)]

  /*--- Initialize the inverse with the identity. ---*/
  for (IntType iVar = 0; iVar < nvar; iVar++)
    for (IntType jVar = 0; jVar < nvar; jVar++)
      M(iVar,jVar) = MATRIXTYPE(iVar==jVar);

  /*--- Inversion ---*/
#define A(I,J) block[(I)*nvar+(J)]

  /*--- Transform system in Upper Matrix ---*/
  for (IntType iVar = 1ul; iVar < nvar; iVar++) {
    for (IntType jVar = 0; jVar < iVar; jVar++)
    {
      MATRIXTYPE weight = A(iVar,jVar) / A(jVar,jVar);

      for (IntType kVar = jVar; kVar < nvar; kVar++)
        A(iVar,kVar) -= weight * A(jVar,kVar);

      /*--- at this stage M is lower triangular so not all cols need updating ---*/
      for (IntType kVar = 0; kVar <= jVar; kVar++)
        M(iVar,kVar) -= weight * M(jVar,kVar);
    }
  }

  /*--- Backwards substitution ---*/
  for (IntType iVar = nvar; iVar > 0;) {
    iVar--; // unsigned type
    for (IntType jVar = iVar+1; jVar < nvar; jVar++)
      for (IntType kVar = 0; kVar < nvar; kVar++)
        M(iVar,kVar) -= A(iVar,jVar) * M(jVar,kVar);

    for (IntType kVar = 0; kVar < nvar; kVar++)
      M(iVar,kVar) /= A(iVar,iVar);
  }
#undef A
#undef M
}

void MatrixMatrixProduct( MATRIXTYPE *a, MATRIXTYPE *b, MATRIXTYPE *&c, IntType n){
  IntType i, j, k;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      c[i*n+j] = 0.0;
      for (k = 0; k < n; k++)
        c[i*n+j] += a[i*n+k] * b[k*n+j];
    }
  }
}

MATRIXTYPE* GetBlock_ILUMatrix( IntType block_i, IntType block_j, IntType nvar, MATRIXTYPE *ILU_matrix, 
    IntType *row_ptr, IntType *col_ind, IntType *dia_ptr) {
  /*--- The position of the diagonal block is known which allows halving the search space. ---*/
  const IntType end = (block_j<block_i)? dia_ptr[block_i] : row_ptr[block_i+1];
  for (IntType index = (block_j<block_i)? row_ptr[block_i] : dia_ptr[block_i]; index < end; ++index)
    if (col_ind[index] == block_j)
      return &ILU_matrix[index*nvar*nvar];
  return nullptr;
}

void PrecondILU_decomp0(PolyGrid *grid, IntType *row_ptr, IntType *col_ind, IntType *dia_ptr, MATRIXTYPE *matrix, MATRIXTYPE *&invM,
    IntType nTCell, IntType nvar, IntType nnz, MATRIXTYPE *&ILU_matrix){

    /*--- ILU0, direct copy. ---*/
    for (IntType iVar = 0; iVar < nnz*nvar*nvar; ++iVar)
      ILU_matrix[iVar] = matrix[iVar];

  /*--- OpenMP Parallelization, a loop construct is used to ensure
   *    the preconditioner is computed correctly even if called
   *    outside of a parallel section. ---*/

    const IntType begin = 0;
    const IntType end = nTCell;

    MATRIXTYPE weight[NVAR*NVAR];
    MATRIXTYPE aux_block[NVAR*NVAR];

    for (IntType iPoint = begin+1; iPoint < end; iPoint++) {

      /*--- Invert and store the previous diagonal block to later compute the weight. ---*/
      MATRIXTYPE *invm_iPtr = &invM[(iPoint-1)*nvar*nvar];
      InverseDiagonalBlock_ILUMatrix(iPoint-1, invm_iPtr, nvar, ILU_matrix, dia_ptr);

      /*--- For this row (unknown), loop over its lower diagonal entries. ---*/
      for (IntType index = row_ptr[iPoint]; index < dia_ptr[iPoint]; index++) {

        /*--- jPoint is the column index (jPoint < iPoint). ---*/
        IntType jPoint = col_ind[index];

        /*--- We only care about the sub matrix within "begin" and "end-1". ---*/
        if (jPoint < begin) continue;

        /*--- Multiply the block by the inverse of the corresponding diagonal block. ---*/
        MATRIXTYPE *Block_ij = &ILU_matrix[index*nvar*nvar];
        MATRIXTYPE *w_ptr = &weight[0];

        MatrixMatrixProduct(Block_ij, &invM[jPoint*nvar*nvar], w_ptr, nvar);

        /*--- "weight" holds Aij*inv(Ajj). Jump to the upper part of the jPoint row. ---*/
        for (IntType index_ = dia_ptr[jPoint]+1; index_ < row_ptr[jPoint+1]; index_++) {

          /*--- Get the column index (kPoint > jPoint). ---*/
          IntType kPoint = col_ind[index_];

          if (kPoint >= end) break;

          /*--- If Aik exists, update it: Aik -= Aij*inv(Ajj)*Ajk ---*/
          MATRIXTYPE *Block_ik = GetBlock_ILUMatrix(iPoint, kPoint, nvar, ILU_matrix, row_ptr, col_ind, dia_ptr);

          if (Block_ik != nullptr) {
            MATRIXTYPE *Block_jk = &ILU_matrix[index_*nvar*nvar];
            MATRIXTYPE *a_ptr = &aux_block[0];
            MatrixMatrixProduct(weight, Block_jk, a_ptr, nvar);
            //MatrixSubtraction(Block_ik, aux_block, Block_ik);
            for( IntType iVar = 0; iVar < nvar*nvar; iVar++)
                Block_ik[iVar] = Block_ik[iVar] - aux_block[iVar];
          }
        }

        /*--- Lastly, store "weight" in the lower triangular part, which
         will be reused during the forward solve in the precon/smoother. ---*/
        for (IntType iVar = 0; iVar < nvar*nvar; ++iVar)
          Block_ij[iVar] = weight[iVar];
      }
    }
    MATRIXTYPE *invmPtr = &invM[(end-1)*nvar*nvar];
    InverseDiagonalBlock_ILUMatrix(end-1, invmPtr, nvar, ILU_matrix, dia_ptr);
}

void computeILUPositions(IntType *row_ptr, IntType *col_ind, IntType *dia_ptr, IntType nTCell, \
    IntType *&matrixInfo, IntType *&Aposptr, IntType *&Alpos, IntType *&Aupos){
    
    std::vector<IntType> stdposptr; 
    std::vector<IntType> stdupos;
    std::vector<IntType> stdlpos;

    stdposptr.push_back(0);
    IntType num_counts = 0;
    for(int i = 0; i < nTCell; i++){
        for(IntType j = row_ptr[i]; j < row_ptr[i+1]; j++){
            IntType col = col_ind[j];
            IntType count = 0;
            for(IntType ii = row_ptr[i]; ii < dia_ptr[i] && ii < j; ii++){
                IntType col_p = col_ind[ii];
                for(IntType k = row_ptr[col_p]; k < row_ptr[col_p+1]; k++){
                    if(col_ind[k] >= col){
                        if(col_ind[k] == col){
                            stdupos.emplace_back(k);
                            stdlpos.emplace_back(ii);
                            count++;
                        }
                        break;
                    }
                }
            }
            num_counts += count;
            stdposptr.emplace_back(num_counts);
        }
    }
        
    IntType posptr_length = stdposptr.size();
    IntType stdlpos_length = stdlpos.size();
    IntType stdupos_length = stdupos.size();
    mfmem::snew_array_1D(Aposptr, posptr_length, dmrfl);
    mfmem::snew_array_1D(Alpos, stdlpos_length, dmrfl);
    mfmem::snew_array_1D(Aupos, stdupos_length, dmrfl);
    for(IntType i=0;i<posptr_length;i++){
        Aposptr[i] = stdposptr[i];
    }
    for(IntType i=0;i<stdlpos_length;i++){
        Alpos[i] = stdlpos[i];
    }
    for(IntType i=0;i<stdupos_length;i++){
        Aupos[i] = stdupos[i];
    }
    matrixInfo[2] = posptr_length;
    matrixInfo[3] = stdlpos_length;
    matrixInfo[4] = stdupos_length;
    //printf("posptr_length:%d stdlpos_length:%d stdupos_length:%d\n",\
        posptr_length,stdlpos_length,stdupos_length);

}
/**/
void PrecondILU_decomp1(PolyGrid *grid, IntType *row_ptr, IntType *col_ind, IntType *dia_ptr, MATRIXTYPE *matrix, MATRIXTYPE *&invM,
    IntType nTCell, IntType nvar, IntType nnz, MATRIXTYPE *&ILU_matrix){

    for (IntType iVar = 0; iVar < nnz*nvar*nvar; ++iVar)
    ILU_matrix[iVar] = matrix[iVar];

    const IntType begin = 0;
    const IntType end = nTCell;

    //  The dependency array for decomposition
    IntType *matrixInfo  = (IntType *)(grid->GetDataPtr(INT, 16, "matrixInfo"));
    IntType *posptr = (IntType *)(grid->GetDataPtr(INT, matrixInfo[2], "posptr"));
    IntType *upos = (IntType *)(grid->GetDataPtr(INT, matrixInfo[3], "upos"));
    IntType *lpos     = (IntType *)(grid->GetDataPtr(INT, matrixInfo[4], "lpos"));

    bool * isReady = (bool *)malloc(nnz * sizeof(bool));
    memset(isReady, false, nnz * sizeof(bool)); 

    IntType bs2 = NVAR * NVAR;
#pragma omp parallel for schedule(static) shared(isReady)
    for (IntType irow = begin; irow < end; irow++) {
        for (IntType jPos = row_ptr[irow]; jPos < row_ptr[irow + 1]; jPos++) {
            IntType col = col_ind[jPos];
            if(col < 0) continue;
            MATRIXTYPE *Block_ij = &ILU_matrix[jPos * nvar * nvar];
            IntType dep_start = posptr[jPos];
            IntType dep_end = posptr[jPos + 1];
            IntType num_dep = dep_end - dep_start;
            bool idep_finish[num_dep];// = {false};
            for (int i = 0; i < num_dep; i++) {
                idep_finish[i] = false;
            }
            bool diag_div_finish = false;
            IntType num_dep_completed = 0;
            while(num_dep_completed < num_dep || !diag_div_finish){
                if(num_dep_completed < num_dep){
                    for (IntType idep = 0; idep < num_dep; idep++){
                        if(idep_finish[idep])continue;
                        if(!isReady[upos[dep_start + idep]]) continue;
                        MATRIXTYPE *l_data = &ILU_matrix[lpos[dep_start + idep] * bs2];
                        MATRIXTYPE *u_data = &ILU_matrix[upos[dep_start + idep] * bs2];
                        #pragma omp flush
                        for(IntType ii = 0; ii < NVAR; ii++){
                            for(IntType jj = 0; jj < NVAR; jj++){
                                Block_ij[ii * nvar + jj] -= 
                                    l_data[ii * nvar    ] * u_data[           jj] +
                                    l_data[ii * nvar + 1] * u_data[    nvar + jj] +
                                    l_data[ii * nvar + 2] * u_data[2 * nvar + jj] +
                                    l_data[ii * nvar + 3] * u_data[3 * nvar + jj] +
                                    l_data[ii * nvar + 4] * u_data[4 * nvar + jj];
                            }   
                        }
                        num_dep_completed++;
                        idep_finish[idep] = true;
                    }
                }

                if(!diag_div_finish && num_dep_completed == num_dep){
                    if(irow < col){
                        diag_div_finish = true;
                    } else if(irow == col){
                        MATRIXTYPE *invM_ptr = &invM[irow * bs2];
                        InverseDiagonalBlock_ILUMatrix(irow, invM_ptr, NVAR, ILU_matrix, dia_ptr);
                        diag_div_finish = true;
                        #pragma omp flush
                    } else {
                        if(isReady[dia_ptr[col]]){
                            MATRIXTYPE *invM_ptr = &invM[col * bs2];
                            #pragma omp flush
                            MATRIXTYPE tmp[bs2];
                            for (IntType ii = 0; ii < NVAR; ii++){
                                for(IntType jj = 0; jj < NVAR; jj++){
                                    tmp[ii * nvar + jj] = 
                                        Block_ij[ii * nvar    ] * invM_ptr[           jj]+
                                        Block_ij[ii * nvar + 1] * invM_ptr[    nvar + jj]+
                                        Block_ij[ii * nvar + 2] * invM_ptr[2 * nvar + jj]+
                                        Block_ij[ii * nvar + 3] * invM_ptr[3 * nvar + jj]+
                                        Block_ij[ii * nvar + 4] * invM_ptr[4 * nvar + jj];
                                }
                            }

                            for (IntType ii = 0; ii < NVAR; ii++){
                                for(IntType jj = 0; jj < NVAR; jj++){
                                    Block_ij[ii * NVAR + jj] = tmp[ii * NVAR + jj]; 
                                }
                            }

                            diag_div_finish = true;
                        }
                    }
                }

                #pragma omp flush(Block_ij)
                if(diag_div_finish && num_dep_completed == num_dep)
                    isReady[jPos] = true;
                #pragma omp flush(isReady)
                    
            }   
        }
    }

    free(isReady);
}

void MatrixVectorProductSub( IntType n, MATRIXTYPE *a, MATRIXTYPE *b, MATRIXTYPE *&c ) {
  /*---
   This is a templated version of GEMV with the constants as boolean
   template parameters so that they can be optimized away at compilation.
   This is still the traditional "row dot vector" method.
  ---*/
    for (auto i = 0; i < n; i++)
      for (auto j = 0; j < n; j++)
        c[i] -= a[i*n+j] * b[j];
}

void MatrixVectorProduct( IntType n, MATRIXTYPE *a, MATRIXTYPE *b, MATRIXTYPE *&c ){
    for (auto i = 0; i < n; i++) {
      c[i] = 0.0;
      for (auto j = 0; j < n; j++)
        c[i] += a[i*n+j] * b[j];
    }
   
}

void PrecondILU_solve0( IntType *row_ptr, IntType *col_ind, IntType *dia_ptr, MATRIXTYPE *ILU_matrix, MATRIXTYPE *invM, 
    MATRIXTYPE *vec, MATRIXTYPE *&prod, IntType nTCell, IntType nvar ){ //b x

    IntType begin = 0;
    IntType end = nTCell;

    MATRIXTYPE aux_vec[NVAR*NVAR];
    MATRIXTYPE *a_ptr = &aux_vec[0];

    /*--- Copy vector to then work on prod in place ---*/

#pragma omp parallel for 
    for (IntType iVar = begin*nvar; iVar < end*nvar; iVar++)
      prod[iVar] = vec[iVar];

    /*--- Forward solve the system using the lower matrix entries that
     were computed and stored during the ILU preprocessing. Note
     that we are overwriting the residual vector as we go. ---*/
    for (IntType iPoint = begin+1; iPoint < end; iPoint++) {
      for (IntType index = row_ptr[iPoint]; index < dia_ptr[iPoint]; index++) {
        IntType jPoint = col_ind[index];
        if (jPoint < begin) {
          continue;
        }
        MATRIXTYPE *Block_ij = &ILU_matrix[index*nvar*nvar];
        MATRIXTYPE *i_ptr = &prod[iPoint*nvar];
        MatrixVectorProductSub( nvar, Block_ij, &prod[jPoint*nvar], i_ptr );
      }
    }

    /*--- Backwards substitution (starts at the last row) ---*/
    for (IntType iPoint = end; iPoint > begin;) {
      iPoint--; // unsigned int type
      for (IntType iVar = 0; iVar < nvar; iVar++)
        aux_vec[iVar] = prod[iPoint*nvar+iVar];

      for (IntType index = dia_ptr[iPoint]+1; index < row_ptr[iPoint+1]; index++) {
        IntType jPoint = col_ind[index];
        if (jPoint >= end) {
          break;
        }
        MATRIXTYPE *Block_ij = &ILU_matrix[index*nvar*nvar];
        MATRIXTYPE *j_ptr = &prod[jPoint*nvar];
        MatrixVectorProductSub( nvar, Block_ij, j_ptr, a_ptr );
      }
      MATRIXTYPE *i_ptr = &prod[iPoint*nvar];
      MatrixVectorProduct(nvar, &invM[iPoint*nvar*nvar], aux_vec, i_ptr);
    }
}

void PrecondILU_solve1( IntType *row_ptr, IntType *col_ind, IntType *dia_ptr, MATRIXTYPE *ILU_matrix, MATRIXTYPE *invM, 
    MATRIXTYPE *vec, MATRIXTYPE *&prod, IntType nTCell, IntType nvar ){ //b x

    IntType begin = 0;
    IntType end = nTCell;

    /*--- Copy vector to then work on prod in place ---*/

    #pragma omp parallel for
    for (IntType iVar = begin*nvar; iVar < end*nvar; iVar++)
      prod[iVar] = vec[iVar];
    //memcpy(prod, vec, (end-begin)*nvar*sizeof(MATRIXTYPE));

    /*--- Forward solve the system using the lower matrix entries that
     were computed and stored during the ILU preprocessing. Note
     that we are overwriting the residual vector as we go. ---*/
    bool *isReady = (bool *)malloc(nTCell * sizeof(bool));
    memset(isReady, false, nTCell * sizeof(bool));
    
    #pragma omp parallel for schedule(static) shared(isReady)
    for (IntType iPoint = begin; iPoint < end; iPoint++) {
        IntType start_idx = row_ptr[iPoint];
        IntType end_idx = dia_ptr[iPoint];
        bool Isdone = false;

        for (IntType index = start_idx; index < end_idx; index++) {
            IntType jPoint = col_ind[index];
            if( jPoint < begin ) {continue;}
            while( !isReady[jPoint] ){
                #pragma omp flush(isReady) 
            }
            MATRIXTYPE *Block_ij = &ILU_matrix[index*nvar*nvar];
            MATRIXTYPE *i_ptr = &prod[iPoint*nvar];
            #pragma omp flush(prod)
            MatrixVectorProductSub( nvar, Block_ij, &prod[jPoint*nvar], i_ptr );
        }
        isReady[iPoint] = true;
        #pragma omp flush(isReady)
    }

    memset(isReady, false, nTCell * sizeof(bool));
    #pragma omp parallel for schedule(static) shared(isReady)
    for (IntType iPoint = end-1; iPoint >= begin; iPoint--) { 
        MATRIXTYPE aux_vec[NVAR];
        MATRIXTYPE *a_ptr = &aux_vec[0];
        IntType start_idx = dia_ptr[iPoint] + 1;
        IntType end_idx = row_ptr[iPoint + 1];
        MATRIXTYPE *i_ptr = &prod[iPoint*nvar];

        for (IntType index = start_idx; index < end_idx; index++) {
            IntType jPoint = col_ind[index];
            if ( jPoint >= end ){continue;}
            MATRIXTYPE *Block_ij = &ILU_matrix[index*nvar*nvar];
            MATRIXTYPE *j_ptr = &prod[jPoint*nvar];
            while ( !isReady[jPoint] ){
                #pragma omp flush(isReady)
                // use acquire(read) or release(write) to avoid extra synchronization cost 
            }
            #pragma omp flush(prod)
            MatrixVectorProductSub( nvar, Block_ij, j_ptr, i_ptr);
        }

        MatrixVectorProduct(nvar, &invM[iPoint*nvar*nvar], i_ptr, a_ptr);
        for(IntType ii=0; ii<NVAR; ii++){i_ptr[ii] = a_ptr[ii];}
        isReady[iPoint] = true;  
        #pragma omp flush(isReady)
    }

    free(isReady);
}

void CalConvectiveFluxJacobian1(RealFlow Matrix[NVAR][NVAR], RealFlow nx, RealFlow ny, RealFlow nz,
                                     RealFlow rho, RealFlow u, RealFlow v, RealFlow w, RealFlow p, RealFlow gam)
{
    RealFlow a1, a2, a3, Vn, phi;
    RealFlow E, vv, gamm1;

    gamm1 = gam-1.0;
    vv = 0.5*(u*u+v*v+w*w);
    phi = gamm1*vv;
    E = p/(rho*gamm1)+vv;
    a1 = gam*E-phi;
    a2 = gamm1;
    a3 = gam-2.0;
    Vn = nx*u+ny*v+nz*w;

    Matrix[0][0] = 0.0;
    Matrix[0][1] = nx;
    Matrix[0][2] = ny;
    Matrix[0][3] = nz;
    Matrix[0][4] = 0.0;

    Matrix[1][0] = nx*phi - u*Vn;
    Matrix[1][1] = Vn - a3*nx*u;
    Matrix[1][2] = ny*u - a2*nx*v;
    Matrix[1][3] = nz*u - a2*nx*w;
    Matrix[1][4] = a2*nx;

    Matrix[2][0] = ny*phi - v*Vn;
    Matrix[2][1] = nx*v - a2*ny*u;
    Matrix[2][2] = Vn - a3*ny*v;
    Matrix[2][3] = nz*v - a2*ny*w;
    Matrix[2][4] = a2*ny;

    Matrix[3][0] = nz*phi - w*Vn;
    Matrix[3][1] = nx*w - a2*nz*u;
    Matrix[3][2] = ny*w - a2*nz*v;
    Matrix[3][3] = Vn - a3*nz*w;
    Matrix[3][4] = a2*nz;

    Matrix[4][0] = Vn*(phi-a1);
    Matrix[4][1] = nx*a1 - a2*u*Vn;
    Matrix[4][2] = ny*a1 - a2*v*Vn;
    Matrix[4][3] = nz*a1 - a2*w*Vn;
    Matrix[4][4] = gam*Vn;
}

void SetupMatrix( PolyGrid *grid, IntType level, MATRIXTYPE *&val, GETYPE *diagmatrix, IntType* bsrIndex ) 
{
    //set up left hand matrix:
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nIFace = grid->GetNIFace();
    IntType n      = nTCell + nBFace;

    RealFlow *q[NVAR], *precond;
    IntType iprec, vis_mode, vis_run=0;
    RealFlow gam, p_bar, lhs_omga;
    RealFlow matrix_jacobi_fc[NVAR][NVAR];
    RealFlow matrix_jacobi_d[NVAR][NVAR];
    RealFlow matrix_temp[NVAR][NVAR];
    RealFlow q_l[NVAR], q_r[NVAR];
    IntType *f2c   = grid->Getf2c();
    IntType *nFPC  = CalnFPC(grid);
    IntType **C2F  = CalC2F(grid); 
    RealGeom *xfn  = grid->GetXfn();
    RealGeom *yfn  = grid->GetYfn();
    RealGeom *zfn  = grid->GetZfn();
    RealGeom *area = grid->GetFaceArea();

    RealGeom dist;
    RealGeom face_n[3];
    RealFlow *vis_l, *vis_t = NULL;
    RealGeom *xcc, *ycc, *zcc;
    RealFlow tmparea;

    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    precond = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "precond");
    grid->GetData(&iprec,   INT, 1, "iprec");
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&lhs_omga,   REAL_FLOW, 1, "lhs_omga");
    RealFlow alf_l = 0.1;
    grid->GetData(&alf_l,  REAL_FLOW, 1, "alf_l");

    //diagnose elements:
    RealGeom *vol  =  grid->GetCellVol();
    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
    IntType nTFace = grid->GetNTFace();
    RealGeom *norm_dist_c2c = NULL;
    norm_dist_c2c = (RealGeom *)grid->GetDataPtr(REAL_GEOM, nTFace, "norm_dist_c2c");
    if(vis_mode != INVISCID){
        vis_run = 1;
        if(iprec) vis_run = 0;
        // if coarse grid doesn't want to run the viscous flux, turn it off
        if(level != 0){
            IntType cg_vis = 1;
            grid->GetData(&cg_vis, INT, 1, "cg_vis");
            if(cg_vis == 0) vis_run = 0;
        }
    }
    if(vis_run){
        xcc = grid->GetXcc();
        ycc = grid->GetYcc();
        zcc = grid->GetZcc();
        vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
        vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    }

    for(IntType i=0; i<NVAR*NVAR; i++) 
    {
        matrix_jacobi_fc[0][i] = 0.0;
        matrix_jacobi_d[0][i] = 0.0;
        matrix_temp[0][i] = 0.0;
    }
    IntType ifStart = nBFace - nIFace;
    //IntType *coo = grid->GetCOO();
    IntType *matrixInfo  = (IntType *)(grid->GetDataPtr(INT, 16, "matrixInfo"));
    IntType *bsr_row_ptr = (IntType *)(grid->GetDataPtr(INT, matrixInfo[0]+1, "bsr_row_ptr"));

#ifdef FS_OPENMP
#pragma omp parallel for private(matrix_jacobi_d, matrix_jacobi_fc, face_n, q_r, q_l, tmparea, matrix_temp, dist)
#endif
    for(int iCell = 0;iCell<nTCell;iCell++){
        IntType cell = iCell;
        MATRIXTYPE *vStart = &val[bsr_row_ptr[iCell]*NVAR*NVAR];
        IntType *bsrIndexStart = &bsrIndex[bsr_row_ptr[iCell]*NVAR*NVAR];

        IntType count = 0;
        for(IntType  m=0;m<NVAR;m++){
            for(IntType n=0;n<NVAR;n++){
                if(m==n){
                    matrix_jacobi_d[m][n] = vol[cell]/dt[cell];;
                }
                else{
                    matrix_jacobi_d[m][n] = 0.0;
                }
            }
        }
        for(IntType iFace=0; iFace<nFPC[cell]; iFace++){
            IntType face  = C2F[cell][iFace];
            IntType c1    = f2c[face+face];
            IntType c2    = f2c[face+face+1];
            face_n[0] = xfn[face];
            face_n[1] = yfn[face];
            face_n[2] = zfn[face];
            if(c2 == cell){
                IntType c_tmp = c1;
                c1    = c2;
                c2    = c_tmp;
                face_n[0] = -face_n[0];
                face_n[1] = -face_n[1];
                face_n[2] = -face_n[2];
            }
            assert(c1 == cell);

            //reorder:
            q_r[0]  = q[0][c2];
            q_r[1]  = q[1][c2];
            q_r[2]  = q[2][c2];
            q_r[3]  = q[3][c2];
            q_r[4]  = q[4][c2]+p_bar;
            //Jacobian of convective flux
            CalConvectiveFluxJacobian1(matrix_jacobi_fc, face_n[0], face_n[1], face_n[2],
                q_r[0], q_r[1], q_r[2], q_r[3], q_r[4], gam);

            // Roe matrix: 
            q_l[0]  = q[0][c1];
            q_l[1]  = q[1][c1];
            q_l[2]  = q[2][c1];
            q_l[3]  = q[3][c1];
            q_l[4]  = q[4][c1]+p_bar;
            tmparea = 0.5*area[face];
            CalJacobian_ConvectiveFlux_Roe(matrix_temp, q_l, q_r, face_n[0], face_n[1], face_n[2], gam, alf_l);     
            for(int m = 0; m < NVAR; m++)
            {
                for(int n = 0; n < NVAR; n++)
                {
                    matrix_jacobi_d[m][n]  += matrix_temp[m][n]*tmparea; 
                    matrix_jacobi_fc[m][n] -= matrix_temp[m][n];
                }
            }
  
            //viscous term for off-diag element 
            RealFlow visc_c1 = 0.0;
            RealFlow visc_c2 = 0.0;
            RealFlow eig_v_c1 = 0.0;
            RealFlow eig_v_c2 = 0.0;
            if(vis_run){
                // Eigenvalues of viscous flux
                dist = norm_dist_c2c[face];
                visc_c1 = vis_l[c1];
                visc_c2 = vis_l[c2];
                if(vis_t)
                {
                    visc_c1 += vis_t[c1];
                    visc_c2 += vis_t[c2];
                }
                eig_v_c1 = 2.0*visc_c1/(q_l[0]*dist + TINY);
                eig_v_c2 = 2.0*visc_c2/(q_r[0]*dist + TINY); 
            }
            if(vis_run){
                for(int k=0;k<NVAR;k++){
                    matrix_jacobi_d[k][k]  += eig_v_c1*tmparea;  
                    matrix_jacobi_fc[k][k] -= eig_v_c2;
                }
            }
            /// ������ͳһ�ˣ���Ϊ���ʼ���ʱ��û��
            for(int k=0;k<NVAR;k++){
                for(int l=0;l<NVAR;l++){
                    matrix_jacobi_fc[k][l] *= tmparea;
                }
            }
            if(face < ifStart)
            {
                continue;
            }
            for(int k=0;k<NVAR;k++){
                for(int l=0;l<NVAR;l++){
                    val[bsrIndexStart[count++]] = matrix_jacobi_fc[k][l];
                }
            }
            //sysMatrix->SetBlockValue(Brow, Bcol, matrix_jacobi_fc[0]);
        }
        
        for(int k=0;k<NVAR;k++){
            for(int l=0;l<NVAR;l++){
                val[bsrIndexStart[count++]] = matrix_jacobi_d[k][l];
                diagmatrix[iCell*NVAR*NVAR+k*NVAR+l] = matrix_jacobi_d[k][l];
            }
        }
        //sysMatrix->SetBlockValue(Brow, Bcol, matrix_jacobi_d[0]);
    }

    // IntType start = bsr_row_ptr[nTCell]*NVAR*NVAR;
    // IntType end   = bsr_row_ptr[matrixInfo[0]]*NVAR*NVAR;
    // // for(IntType i=start; i<end; i++){
    // //     val[i] = 0.0;
    // // }
}

void Gauss_Elimination_block( RealFlow *prod, MATRIXTYPE block[], IntType nvar, IntType idx){

    #define A(I,J) block[(I)*nvar+(J)]
    /*--- Transform system in Upper Matrix ---*/
    for (auto iVar = 1ul; iVar < nvar; iVar++) {
        for (auto jVar = 0ul; jVar < iVar; jVar++) {
            MATRIXTYPE weight = A(iVar,jVar) / A(jVar,jVar);
            for (auto kVar = jVar; kVar < nvar; kVar++){
                A(iVar,kVar) -= weight * A(jVar,kVar);
            }
            prod[ iVar ] -= weight * prod[ jVar ];
        }
    }

    /*--- Backwards substitution ---*/
    for (auto iVar = nvar; iVar > 0ul;) {
        iVar--; // unsigned type
        for (auto jVar = iVar+1; jVar < nvar; jVar++){
            prod[ iVar ] -= A(iVar,jVar) * prod[ jVar ];
        }
        prod[ iVar ] /= A(iVar,iVar);
    }
    #undef A
}

//vec should be reordered, prod is 2*n long
void LusgsFusedReorder( const IntType *fused_row_ptr, const IntType *fused_col_ind, const MATRIXTYPE *fused_matrix,\
    const MATRIXTYPE *matrix, const IntType *dia_ptr, const IntType *perm, const IntType *inv_perm, \
    RealFlow *vec, RealFlow *&prod, IntType n, IntType nvar ){

    IntType NVAR2 = nvar*nvar;
    IntType start = 0;
    IntType end   = n+n;

    bool *isReady = (bool *)malloc(2*n * sizeof(bool));
    memset(isReady, false, 2*n * sizeof(bool));
    RealFlow *tmp_prod = (RealFlow *)malloc(n*nvar * sizeof(RealFlow));
    memcpy(tmp_prod, prod, n*nvar * sizeof(RealFlow));

    #pragma omp parallel for schedule(static,1) shared(isReady)
    for (IntType iPoint = start; iPoint < end; ++iPoint) {
        RealFlow *ptr1;
        IntType idx;
        MATRIXTYPE block[NVAR*NVAR];
        IntType start_idx = fused_row_ptr[iPoint];
        IntType end_idx   = fused_row_ptr[iPoint+1];

        IntType originIdx = perm[iPoint]; //access index of x and b, perm or inv_perm??
        IntType backwardIdx; 

        // b is not reversed, x should be reversed after computations
        RealFlow tp_prod[NVAR];
        memset(tp_prod, 0.0,  NVAR * sizeof(RealFlow));
        RealFlow dia_prod[NVAR];
        memset(dia_prod, 0.0,  NVAR * sizeof(RealFlow));

        if( originIdx < n ){
            backwardIdx = iPoint;
            idx = iPoint * nvar;
            ptr1 = &vec[originIdx*nvar]; //original order to access b
            IntType tmp = fused_row_ptr[inv_perm[n + n - 1 - originIdx]]; //verified
            memcpy( block, &fused_matrix[tmp*NVAR2], NVAR2*sizeof(RealFlow));
            //memcpy( block, &matrix[dia_ptr[originIdx]*NVAR2], NVAR2*sizeof(RealFlow));
            //printf("i:%d start:%d end:%d origin:%d %d - 1\n",iPoint,start_idx,end_idx,originIdx, tmp);
        }
        else{
            backwardIdx = fused_col_ind[start_idx];  //backwardIdx in [0,n)
            idx =  backwardIdx * nvar;               //index of x and b
            ptr1 = dia_prod;
            memcpy( block, &fused_matrix[start_idx*NVAR2], NVAR2*sizeof(RealFlow));
            //memcpy( block, &matrix[dia_ptr[n+n-1-originIdx]*NVAR2], NVAR2*sizeof(RealFlow));
            start_idx++;    // avoid the virtual node adjcent with upper substitution
            //printf("i:%d start:%d end:%d origin:%d %d - 1x\n",iPoint,start_idx,end_idx,originIdx,backwardIdx);

            while( !isReady[ backwardIdx ] ){
                #pragma omp flush(isReady) 
            }
            #pragma omp flush(tmp_prod)
            //backward only, executed when isReady[col_j] are all true
            for (IntType iVar = 0ul; iVar < nvar; iVar++) 
                for (IntType jVar = 0ul; jVar < nvar; jVar++)
                    ptr1[iVar] += block[iVar*nvar + jVar] * tmp_prod[idx + jVar];
        }
        //if( (iPoint >= n) && (iPoint < (n+1000)) )
        //    printf("i:%d %d  %.8f %.8f - 1\n",iPoint,backwardIdx,ptr1[1],ptr1[3]);

        for (IntType index = start_idx; index < end_idx; index++) {     //exclude diag element
            IntType col_j = fused_col_ind[index];
            while( !isReady[col_j] ){
                #pragma omp flush(isReady) 
            }
            #pragma omp flush(tmp_prod)
            IntType col_jj = col_j;     // choose the column index with another indirect access
            if( originIdx >= n ) { 
                //col_jj = fused_col_ind[fused_row_ptr[col_j]];
                col_jj = inv_perm[n + n - 1 - perm[col_j]]; // how to make more efficient access
            }
            for (IntType iVar = 0ul; iVar < nvar; iVar++){
                for (IntType jVar = 0ul; jVar < nvar; jVar++){
                    tp_prod[iVar] += fused_matrix[index*NVAR2 + iVar*nvar + jVar] \
                                    * tmp_prod[col_jj*nvar + jVar];
                }
            }
        }
        //if( (iPoint >= n) && (iPoint < (n+1000)) )
        //    printf("i:%d %d  %.8f %.8f - 2\n",iPoint,backwardIdx,tp_prod[1],tp_prod[3]);

        for( IntType iVar = 0; iVar < nvar; iVar++ )
            tmp_prod[idx+iVar] = ptr1[iVar] - tp_prod[iVar];

        //if( (iPoint >= n) && (iPoint < (n+1000)) )
        //    printf("i:%d %d  %.8f %.8f - 3\n",iPoint,backwardIdx,tmp_prod[idx+1],tmp_prod[idx+3]);

        Gauss_Elimination_block( &tmp_prod[idx], block, nvar, iPoint );
        isReady[iPoint] = true;

        //if( (iPoint >= n) && (iPoint < (n+1000)) )
        //    printf("i:%d %d  %.8f %.8f - 4\n",iPoint,backwardIdx,tmp_prod[idx+1],tmp_prod[idx+3]);
    }
    #pragma omp parallel for
    for( IntType i=0; i<n; i++){
        IntType index = inv_perm[i];
        prod[i*nvar + 0] = tmp_prod[index*nvar + 0];
        prod[i*nvar + 1] = tmp_prod[index*nvar + 1];
        prod[i*nvar + 2] = tmp_prod[index*nvar + 2];
        prod[i*nvar + 3] = tmp_prod[index*nvar + 3];
        prod[i*nvar + 4] = tmp_prod[index*nvar + 4];
    }
    free(tmp_prod);
    free(isReady);
}

void FusedMatrixTrans( const IntType *row_ptr, const IntType *col_ind, const IntType *dia_ptr, const RealFlow *matrix, \
    IntType n, IntType nnz, IntType nvar, \
    IntType *&tmp_row_ptr, IntType *&tmp_col_ind){

    tmp_row_ptr[0] = 0;
    IntType count = 0;
    for( IntType i=0;i<n;i++){
        tmp_row_ptr[1+i] = tmp_row_ptr[i] + dia_ptr[i] - row_ptr[i];
        for( IntType j=row_ptr[i]; j<dia_ptr[i]; j++){
            tmp_col_ind[count++] = col_ind[j];

        }
    }
    for( IntType i=n-1;i>=0;i--){
        tmp_row_ptr[2*n-i] = tmp_row_ptr[2*n-i-1] + row_ptr[i+1] - dia_ptr[i];
        tmp_col_ind[count++] = i;
        for( IntType j=dia_ptr[i]+1; j<row_ptr[i+1]; j++ ){  
            tmp_col_ind[count++] = n + n - col_ind[j] - 1;
        }
    }
}
/*
void FusedReorderMethod( IntType *tmp_row_ptr, IntType *tmp_col_ind, IntType *&perm, \
    IntType *&inv_perm, IntType n_cells, IntType nvar, IntType OMPTHREADS){

    IntType methodsReorder = 4;
    IntType coreCount = (n_cells) < OMPTHREADS ? n_cells : OMPTHREADS;
    int *row_ptrs_sym=NULL, *col_idxs_sym=NULL;

    switch (methodsReorder)
    {
    case 0:
		Graph_partition_reorder(n_cells, tmp_row_ptr, tmp_col_ind, perm, inv_perm, coreCount);
        break;
    case 1:
        gray_reorder(n_cells, tmp_row_ptr, tmp_col_ind, perm, inv_perm);
        break;
    case 2:
        Hyper_graph_partition_reorder(n_cells, tmp_row_ptr, tmp_col_ind, perm, inv_perm, coreCount);
        break;
    case 3:
        run_SymmetrizeCSRMatrix(tmp_row_ptr,tmp_col_ind,n_cells,n_cells,&row_ptrs_sym,&col_idxs_sym);
        reverse_cuthill_mckee(n_cells,row_ptrs_sym,col_idxs_sym,perm,inv_perm,StartingStrategy::MinDegree);
        break;
    default:
        #pragma omp parallel for
        for(int i=0; i<n_cells; i++){
            perm[i] = i;
            inv_perm[i] = i;
        }
    }
    if(methodsReorder == 3){
        free(row_ptrs_sym);
        free(col_idxs_sym);
    }
}

void fusedReorderTrans( PolyGrid *grid, const IntType *row_ptr, const IntType *col_ind, const IntType *dia_ptr, 
    IntType *&row_ptrt, IntType *&col_indt, IntType *&perm, IntType *&inv_perm, \
    IntType n, IntType nnz, IntType nvar ){
    IntType n_cells = 2 * n;
    IntType NVAR2 = nvar*nvar;
    IntType *tmp_row_ptr = new IntType[n_cells+1];
    IntType *tmp_col_ind = new IntType[nnz];
    RealFlow *matrix = NULL;
    FusedMatrixTrans( row_ptr, col_ind, dia_ptr, matrix, n, nnz, nvar, \
        tmp_row_ptr, tmp_col_ind );

    IntType OMPTHREADS = 1;
    grid->GetData(&OMPTHREADS,   INT, 1, "OMP_THREADS");
    FusedReorderMethod( tmp_row_ptr, tmp_col_ind, perm, inv_perm, n_cells, nvar, OMPTHREADS);

    row_ptrt[0] = 0;
    #pragma omp parallel for
    for ( IntType i = 0; i < n_cells; ++i) {
        row_ptrt[inv_perm[i] + 1] = tmp_row_ptr[i + 1] - tmp_row_ptr[i];
    }
    for ( IntType i = 1; i < n_cells + 1; ++i) {
        row_ptrt[i] += row_ptrt[i - 1]; 
    }

    #pragma omp parallel for
    for( IntType i=0; i<n_cells; i++){
        IntType counter = tmp_row_ptr[i + 1] - tmp_row_ptr[i];
        // sort the adjcent element
        for ( IntType j = 0; j < counter; ++j ) {
            col_indt[row_ptrt[inv_perm[i]] + j] = inv_perm[tmp_col_ind[tmp_row_ptr[i] + j]];
        }
    }
    delete[] tmp_row_ptr;
	delete[] tmp_col_ind;
}
*/
void fused_lusgs_loop( PolyGrid *grid, const IntType *row_ptr, const IntType *col_ind, const IntType *dia_ptr, 
    const MATRIXTYPE *matrix, RealFlow *vec, RealFlow *&prod, IntType n, IntType nnz, IntType nvar ){

    IntType *row_ptrt = (IntType *)(grid->GetDataPtr(INT, (n*2+1), "fusedRowPtr"));
    IntType *col_indt = (IntType *)(grid->GetDataPtr(INT, nnz, "fusedColInd"));
    IntType *perm = (IntType *)(grid->GetDataPtr(INT, n*2, "perm"));
    IntType *inv_perm = (IntType *)(grid->GetDataPtr(INT, n*2, "inv_perm"));
    
    IntType NVAR2 = nvar*nvar;
    MATRIXTYPE *matrixt = new MATRIXTYPE[nnz*NVAR2];
    IntType n_cells = 2 * n;

    /*for(int i=0; i<n_cells;i++){
        IntType counter, tmp;
        if( i>=n ){
            int j = n+n-1-i;
            counter = row_ptr[j + 1] - dia_ptr[j];
            tmp = dia_ptr[j];
        }
        else{
            counter = dia_ptr[i] - row_ptr[i];
            tmp = row_ptr[i];
        }
        printf("i:%d c:%d   %d %d\n",i,counter,row_ptrt[inv_perm[i]],tmp );
    }*/
    #pragma omp parallel for
    for( IntType i=0; i<n_cells; i++){
        IntType counter;
        if( i>=n ){
            int j = n+n-1-i;
            counter = row_ptr[j + 1] - dia_ptr[j];
            //copy upper matrix with diag
            memcpy(&matrixt[row_ptrt[inv_perm[i]]*NVAR2], \
                &matrix[ dia_ptr[j]*NVAR2 ], counter*NVAR2*sizeof(RealFlow));
        }
        else{
            counter = dia_ptr[i] - row_ptr[i];
            //copy lower matrix without diag
            memcpy(&matrixt[row_ptrt[inv_perm[i]]*NVAR2], \
                &matrix[ row_ptr[i]*NVAR2 ], counter*NVAR2*sizeof(RealFlow));
        }
    }
    LusgsFusedReorder(  row_ptrt, col_indt, matrixt, matrix, dia_ptr, perm, inv_perm, vec, prod, n, nvar );
    delete[] matrixt;

}


// fused LUSGS loop without changing data structure
void LusgsFusedOrigin( IntType *fused_row_ptr, IntType *fused_col_ind, IntType *dia_ptr, MATRIXTYPE *fused_matrix, 
    RealFlow *vec, RealFlow *&prod, IntType n, IntType nvar ){

    IntType NVAR2 = nvar*nvar;
    IntType start = 0;
    IntType end = n+n;

    bool *isReady = (bool *)malloc(2*n * sizeof(bool));
    memset(isReady, false, 2*n * sizeof(bool));

    #pragma omp parallel for schedule(static) shared(isReady)
    for (IntType iPoint = start; iPoint < end; ++iPoint) {

        RealFlow *ptr1;
        IntType idx;
        MATRIXTYPE block[NVAR*NVAR];
        IntType start_idx;
        IntType end_idx;

        RealFlow tmp_prod[NVAR];
        memset(tmp_prod, 0.0,  NVAR * sizeof(RealFlow));
        RealFlow dia_prod[NVAR];
        memset(dia_prod, 0.0,  NVAR * sizeof(RealFlow));

        if(iPoint < n){
            idx = iPoint;
            start_idx = fused_row_ptr[idx];
            end_idx = dia_ptr[idx];
            ptr1 = &vec[idx*nvar];
            memcpy( block, &fused_matrix[dia_ptr[idx] * NVAR2], NVAR2 * sizeof(RealFlow) );
        }
        else{
            idx = n + n - iPoint - 1;
            start_idx = dia_ptr[idx]+1;
            end_idx = fused_row_ptr[idx+1];
            ptr1 = dia_prod;
            memcpy( block, &fused_matrix[dia_ptr[idx] * NVAR2], NVAR2 * sizeof(RealFlow) ); 

            while( !isReady[ idx ] ){
                #pragma omp flush(isReady) 
            } 
            #pragma omp flush(prod)

            //backward only, executed when isReady[col_j] are all true
            for (IntType iVar = 0ul; iVar < nvar; iVar++) 
                for (IntType jVar = 0ul; jVar < nvar; jVar++)
                    ptr1[iVar] += block[iVar*nvar + jVar] * prod[idx*nvar + jVar];
        }

        for (IntType index = start_idx; index < end_idx; index++) { //exclude diag element
            IntType col_j = fused_col_ind[index];
            IntType col_jj = col_j;
            if(iPoint >= n) { col_jj = n + n - col_j - 1; }
            while( !isReady[col_jj] ){
                #pragma omp flush(isReady) 
            }
            #pragma omp flush(prod)

            for (IntType iVar = 0ul; iVar < nvar; iVar++){
                for (IntType jVar = 0ul; jVar < nvar; jVar++){
                    tmp_prod[iVar] += fused_matrix[index*NVAR2 + iVar*nvar + jVar] * prod[col_j*nvar + jVar];
                }

            }
        }
        //if( (iPoint >= n) && (iPoint < (n+1000)) )
        //    printf("i:%d %d  %.8f %.8f - 2\n",iPoint,idx,tmp_prod[1],tmp_prod[3]);

        for( IntType iVar = 0; iVar < nvar; iVar++ )
            prod[idx*nvar+iVar] = ptr1[iVar] - tmp_prod[iVar];

        //if( (iPoint >= n) && (iPoint < (n+1000)) )
        //    printf("i:%d %d  %.8f %.8f - 3\n",iPoint,idx,prod[idx*nvar+1],prod[idx*nvar+3]);

        Gauss_Elimination_block( &prod[idx*nvar], block, nvar, iPoint );
        isReady[iPoint] = true;

        //if( (iPoint >= n) && (iPoint < (n+1000)) )
        //    printf("i:%d %d  %.8f %.8f - 4\n",iPoint,idx,prod[idx*nvar+1],prod[idx*nvar+3]);
    }

    free(isReady);
}

void diagblockDecomp(
    const IntType* __restrict row_ptr, 
    const IntType* __restrict col_ind, 
    const IntType* __restrict diag_ptr, 
    const MATRIXTYPE* __restrict matrix,
    GETYPE* __restrict diagmatrix,
    IntType n, IntType nvar){
//     int mpirank=0;
// #ifdef MPICH
//     MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
// #endif
    IntType n2 = NVAR * NVAR;
    #pragma omp parallel for 
    for(IntType i=0; i<n; i++){
        const GETYPE *ptr = &matrix[ diag_ptr[i] * n2];
        GETYPE *diagptr   = &diagmatrix[i * n2];

        Eigen::Map<const Eigen::Matrix<GETYPE, NVAR, NVAR, Eigen::RowMajor>> mat(ptr, NVAR, NVAR);
        Eigen::Map<Eigen::Matrix<GETYPE, NVAR, NVAR, Eigen::RowMajor>> mati(diagptr, NVAR, NVAR);
        //Eigen::Dynamic
        mati = mat.inverse();
    }
    //printf("max res:%.16f min res:%.16f\n",res, res2);
}

void for_back_LUSGS_3_1(PolyGrid *grid, 
    const IntType* __restrict row_ptr, 
    const IntType* __restrict col_ind, 
    const IntType* __restrict dia_ptr, 
    const MATRIXTYPE* __restrict matrix, 
    const GETYPE* __restrict diagmatrix, 
    const MATRIXTYPE* __restrict vec, 
    MATRIXTYPE* __restrict prod, 
    IntType n, 
    IntType nvar){

    IntType *luorder = (IntType *)grid->GetDataPtr(INT, n, "LUSGSCellOrder");
    IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, n, "LUSGScellsPerlayer");
    
    int num_threads = omp_get_max_threads();
    MATRIXTYPE *base_buffer = (MATRIXTYPE *)kupl_hbw_malloc(nvar*nvar*num_threads*2*sizeof(MATRIXTYPE));

#pragma omp parallel
{
    IntType tid = omp_get_thread_num();
    MATRIXTYPE *buffer_1 = &base_buffer[tid*nvar*nvar];
    MATRIXTYPE *buffer_2 = &base_buffer[(tid+num_threads)*nvar*nvar];
    kupl_event_h event = kupl_event_create();
    kupl_queue_h q = kupl_queue_create();

    IntType maxlayer = cellsPerlayer[0];
    for(IntType laynum=0; laynum<maxlayer; laynum++ ){
        IntType start = cellsPerlayer[laynum+1];
        IntType end   = cellsPerlayer[laynum+2];
        #pragma omp for schedule(dynamic, 1)
        for (IntType ilu = start; ilu < end; ilu++) {
            IntType iPoint = luorder[ilu];
            IntType idx = iPoint*nvar;
            IntType iVar, col_j;
            MATRIXTYPE low_prod[nvar];

            const GETYPE *ptrdiag  = &diagmatrix[idx*nvar];
            memcpy(low_prod, &vec[idx], nvar*sizeof(MATRIXTYPE));

            for (iVar = row_ptr[iPoint]; iVar < dia_ptr[iPoint]; iVar++) {
                col_j = col_ind[iVar];
                if (iVar == row_ptr[iPoint]) {
                    if (iVar + 1 < dia_ptr[iPoint]) {
                        kupl_memcpy_async(buffer_1,
                                        &matrix[(iVar+1)*nvar*nvar],
                                        nvar*nvar*sizeof(MATRIXTYPE),
                                        q, event);
                    }

                    matrix_vector_sub_sve_unroll_float(low_prod,
                                                    &matrix[iVar*nvar*nvar],
                                                    &prod[col_j*nvar],
                                                    nvar);
                    if (iVar + 1 < dia_ptr[iPoint]) {
                        kupl_event_wait(event);
                    }
                } else {
                    if (iVar + 1 < dia_ptr[iPoint]) {
                        kupl_memcpy_async(buffer_2,
                                        &matrix[(iVar+1)*nvar*nvar],
                                        nvar*nvar*sizeof(MATRIXTYPE),
                                        q, event);
                    }

                    matrix_vector_sub_sve_unroll_float(low_prod,
                                                    buffer_1,
                                                    &prod[col_j*nvar],
                                                    nvar);

                    if (iVar + 1 < dia_ptr[iPoint]) {
                        kupl_event_wait(event);
                        MATRIXTYPE *tmp = buffer_1;
                        buffer_1 = buffer_2;
                        buffer_2 = tmp;
                    }
                }
            }
            matrix_vector_sve_unroll_float(&prod[idx], ptrdiag, low_prod, nvar);
        }
    }

    for(IntType laynum=maxlayer-1; laynum>=0; laynum-- ){
        IntType start = cellsPerlayer[laynum+1];
        IntType end   = cellsPerlayer[laynum+2];
        #pragma omp for schedule(dynamic, 1)
        for(IntType ilu = start; ilu < end; ilu++){
            IntType iPoint = luorder[ilu];
            MATRIXTYPE dia_prod[nvar];
            memset(dia_prod, 0, nvar*sizeof(MATRIXTYPE));
            IntType idx = iPoint*nvar;
            IntType iVar, jVar, kVar, col_j;
            const GETYPE *ptrdiag  = &diagmatrix[idx*nvar];
            const MATRIXTYPE* ptr2 = &matrix[dia_ptr[iPoint]*nvar*nvar];
            matrix_vector_sve_unroll_float(dia_prod, ptr2, &prod[idx], nvar);
            for ( iVar = dia_ptr[iPoint]+1; iVar < row_ptr[iPoint+1]; iVar++) {
                col_j = col_ind[iVar];
                if(iVar == dia_ptr[iPoint]+1){
                    if(iVar + 1 < row_ptr[iPoint+1]){
                        kupl_memcpy_async(buffer_1,
                                        &matrix[(iVar+1)*nvar*nvar],
                                        nvar*nvar*sizeof(MATRIXTYPE),
                                        q, event);
                    }
                    matrix_vector_sub_sve_unroll_float(dia_prod,
                                                    &matrix[iVar*nvar*nvar],
                                                    &prod[col_j*nvar],
                                                    nvar);
                    if(iVar + 1 < row_ptr[iPoint+1]){
                        kupl_event_wait(event);
                    }
                } else {
                    if(iVar + 1 < row_ptr[iPoint+1]){
                        kupl_memcpy_async(buffer_2,
                                        &matrix[(iVar+1)*nvar*nvar],
                                        nvar*nvar*sizeof(MATRIXTYPE),
                                        q, event);
                    }
                    matrix_vector_sub_sve_unroll_float( \
                        dia_prod, buffer_1, &prod[col_j*nvar], nvar);
                    
                    if(iVar + 1 < row_ptr[iPoint+1]){
                        kupl_event_wait(event);
                        MATRIXTYPE *tmp = buffer_1;
                        buffer_1 = buffer_2;
                        buffer_2 = tmp;
                    }
                }
            }
            matrix_vector_sve_unroll_float(&prod[idx], ptrdiag, dia_prod, nvar);
        }
    }
    
    kupl_queue_destroy(q);
    kupl_event_destroy(event);
}
    kupl_hbw_free(base_buffer);
}



void for_back_LUSGS_3(PolyGrid *grid, 
    const IntType* __restrict row_ptr, 
    const IntType* __restrict col_ind, 
    const IntType* __restrict dia_ptr, 
    const MATRIXTYPE* __restrict matrix, 
    const GETYPE* __restrict diagmatrix, 
    const MATRIXTYPE* __restrict vec, 
    MATRIXTYPE* __restrict prod, 
    IntType n, 
    IntType nvar){

    IntType *luorder = (IntType *)grid->GetDataPtr(INT, n, "LUSGSCellOrder");
    IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, n, "LUSGScellsPerlayer");
#pragma omp parallel
{
    IntType laynum;
    IntType start = 0, end = n, ilu;
    IntType maxlayer = cellsPerlayer[0];
    for(laynum=0; laynum<maxlayer; laynum++ ){
        start = cellsPerlayer[laynum+1];
        end   = cellsPerlayer[laynum+2];
        #pragma omp for private(ilu)
        for ( ilu = start; ilu < end; ilu++) {
            IntType iPoint = luorder[ilu];
            IntType idx = iPoint*nvar;
            IntType iVar, col_j;
            MATRIXTYPE low_prod[nvar];

            const GETYPE *ptrdiag  = &diagmatrix[idx*nvar];
            memcpy(low_prod, &vec[idx], nvar*sizeof(MATRIXTYPE));

            for ( iVar = row_ptr[iPoint]; iVar < dia_ptr[iPoint]; iVar++) {
                col_j = col_ind[iVar];

                matrix_vector_sub_sve_unroll_float( \
                    low_prod, &matrix[iVar*nvar*nvar], &prod[col_j*nvar], nvar);
            }
            matrix_vector_sve_unroll_float(&prod[idx], ptrdiag, low_prod, nvar);
        }
    }

    for( laynum=maxlayer-1; laynum>=0; laynum-- ){
        start = cellsPerlayer[laynum+2];
        end   = cellsPerlayer[laynum+1];
#pragma omp for private(ilu)
        for(ilu=start-1; ilu>=end; ilu--){
            IntType iPoint = luorder[ilu];
            MATRIXTYPE dia_prod[nvar];
            memset(dia_prod, 0, nvar*sizeof(MATRIXTYPE));
            IntType idx = iPoint*nvar;
            IntType iVar, jVar, kVar, col_j;
            const GETYPE *ptrdiag  = &diagmatrix[idx*nvar];
            const MATRIXTYPE* ptr2 = &matrix[dia_ptr[iPoint]*nvar*nvar];

            matrix_vector_sve_unroll_float(dia_prod, ptr2, &prod[idx], nvar);
            for ( iVar = dia_ptr[iPoint]+1; iVar < row_ptr[iPoint+1]; iVar++) {
                col_j = col_ind[iVar];
                matrix_vector_sub_sve_unroll_float( \
                    dia_prod, &matrix[iVar*nvar*nvar], &prod[col_j*nvar], nvar);
            }
            matrix_vector_sve_unroll_float(&prod[idx], ptrdiag, dia_prod, nvar);
        }
    }
}
}
// --- diagonal-outside matrix-format LUSGS --- //
void forward_LUSGS_2(
    PolyGrid *grid, 
    const IntType* __restrict row_ptr, 
    const IntType* __restrict col_ind, 
    const IntType* __restrict dia_ptr, 
    const MATRIXTYPE* __restrict matrix, 
    const GETYPE* __restrict diagmatrix, 
    const MATRIXTYPE* __restrict vec, 
    MATRIXTYPE* __restrict prod, 
    IntType n, 
    IntType nvar){

    IntType *luorder = (IntType *)grid->GetDataPtr(INT, n, "LUSGSCellOrder");
    IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, n, "LUSGScellsPerlayer");
    
    IntType laynum;
    IntType start = 0, end = n, ilu;
    IntType maxlayer = cellsPerlayer[0];
    //MATRIXTYPE *all_prod = new MATRIXTYPE[ n * nvar ];

    for(laynum=0; laynum<maxlayer; laynum++ ){
        start = cellsPerlayer[laynum+1];
        end   = cellsPerlayer[laynum+2];
        #pragma omp parallel for private(ilu)
        for ( ilu = start; ilu < end; ilu++) {
            IntType iPoint = luorder[ilu];
            IntType idx = iPoint*nvar;
            IntType iVar, jVar, kVar, col_j;
            MATRIXTYPE low_prod[nvar];
            //MATRIXTYPE* low_prod = &all_prod[ilu*nvar];
            const GETYPE *ptrdiag  = &diagmatrix[idx*nvar];
            memcpy(low_prod, &vec[idx], nvar*sizeof(MATRIXTYPE));

            for ( iVar = row_ptr[iPoint]; iVar < dia_ptr[iPoint]; iVar++) {
                col_j = col_ind[iVar];
                // for ( jVar = 0ul; jVar < nvar; jVar++) {
                //     for ( kVar = 0ul; kVar < nvar; kVar++)
                //         low_prod[jVar] -= matrix[iVar*nvar*nvar+jVar*nvar+kVar] * 
                //                           prod[col_j*nvar+kVar];
                // }
                matrix_vector_sub_sve_unroll_float( \
                    low_prod, &matrix[iVar*nvar*nvar], &prod[col_j*nvar], nvar);
            }

            // for ( jVar = 0ul; jVar < nvar; jVar++) {
            //     MATRIXTYPE tmp = 0.0;
            //     for ( kVar = 0ul; kVar < nvar; kVar++){
            //         tmp += ptrdiag[ jVar*nvar + kVar] * low_prod[kVar];
            //     }
            //     prod[idx+jVar] = tmp;
            // }
            matrix_vector_sve_unroll_float(&prod[idx], ptrdiag, low_prod, nvar);
        }
    }
    //delete[] all_prod;
}
void backward_LUSGS_2( 
    PolyGrid *grid, 
    const IntType* __restrict row_ptr, 
    const IntType* __restrict col_ind, 
    const IntType* __restrict dia_ptr, 
    const MATRIXTYPE* __restrict matrix, 
    const GETYPE* __restrict diagmatrix, 
    const MATRIXTYPE* __restrict vec, 
    MATRIXTYPE* __restrict prod, 
    IntType n, 
    IntType nTCell,
    IntType nvar){

    IntType *luorder = (IntType *)grid->GetDataPtr(INT, n, "LUSGSCellOrder");
    IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, n, "LUSGScellsPerlayer");
    IntType laynum;
    IntType start, end, ilu;

    for( laynum=cellsPerlayer[0]-1; laynum>=0; laynum-- ){
        start = cellsPerlayer[laynum+2];
        end   = cellsPerlayer[laynum+1];
#pragma omp parallel for private(ilu)
        for(ilu=start-1; ilu>=end; ilu--){
            IntType iPoint = luorder[ilu];
            MATRIXTYPE dia_prod[nvar];
            memset(dia_prod, 0, nvar*sizeof(MATRIXTYPE));
            IntType idx = iPoint*nvar;
            IntType iVar, jVar, kVar, col_j;
            const GETYPE *ptrdiag  = &diagmatrix[idx*nvar];
            const MATRIXTYPE* ptr2 = &matrix[dia_ptr[iPoint]*nvar*nvar];
            // for ( iVar = 0ul; iVar < nvar; iVar++) {
            //     dia_prod[iVar] = 0.0;
            //     for ( jVar = 0ul; jVar < nvar; jVar++){
            //         dia_prod[iVar] += ptr2[iVar*nvar + jVar] * \
            //                           prod[idx + jVar];
            //     }
            // }
            matrix_vector_sve_unroll_float(dia_prod, ptr2, &prod[idx], nvar);
            for ( iVar = dia_ptr[iPoint]+1; iVar < row_ptr[iPoint+1]; iVar++) {
                col_j = col_ind[iVar];
                // for ( jVar = 0ul; jVar < nvar; jVar++) {
                //     for ( kVar = 0ul; kVar < nvar; kVar++)
                //         dia_prod[jVar] -= matrix[iVar*nvar*nvar + jVar*nvar + kVar] * \
                //                          prod[col_j*nvar + kVar];
                // }
                matrix_vector_sub_sve_unroll_float( \
                    dia_prod, &matrix[iVar*nvar*nvar], &prod[col_j*nvar], nvar);
            }

            // for ( jVar = 0ul; jVar < nvar; jVar++) {
            //     MATRIXTYPE tmp = 0.0;
            //     for ( kVar = 0ul; kVar < nvar; kVar++){
            //         tmp += ptrdiag[ jVar*nvar + kVar] * dia_prod[kVar];
            //     }
            //     prod[idx+jVar] = tmp;
            // }
            matrix_vector_sve_unroll_float(&prod[idx], ptrdiag, dia_prod, nvar);
        }
    }
}
// --- diagonal-outside matrix-format LUSGS --- //

void forward_LUSGS_coloring( PolyGrid *grid, const IntType *row_ptr, const IntType *col_ind, const IntType *dia_ptr, 
    const MATRIXTYPE *matrix, const MATRIXTYPE *vec, MATRIXTYPE *&prod, IntType n, IntType nvar ){

    const int NNUMBER=5;

    IntType *luorder = (IntType *)grid->GetDataPtr(INT, n, "LUSGSCellOrder");
    IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, n, "LUSGScellsPerlayer");

    IntType laynum;
    IntType start, end, ilu;
    for(laynum=0; laynum<cellsPerlayer[0]; laynum++ ){
        start = cellsPerlayer[laynum+1];
        end   = cellsPerlayer[laynum+2];
    #pragma omp parallel for private(ilu)
    for ( ilu = start; ilu < end; ilu++) {
        IntType iPoint = luorder[ilu];

        IntType idx = iPoint*nvar;
        IntType iVar, jVar, kVar, col_j;
        MATRIXTYPE low_prod[NNUMBER], block[NNUMBER*NNUMBER], weight;
        for ( iVar = 0ul; iVar < nvar; iVar++)
            low_prod[iVar] = 0.0;
        for ( iVar = row_ptr[iPoint]; iVar < dia_ptr[iPoint]; iVar++) {
            col_j = col_ind[iVar];

            for ( jVar = 0ul; jVar < nvar; jVar++) {
                for ( kVar = 0ul; kVar < nvar; kVar++)
                    low_prod[jVar] += matrix[iVar*nvar*nvar+jVar*nvar+kVar] * prod[col_j*nvar+kVar];
            }
        }
        
        for( iVar = 0; iVar < nvar; iVar++ )
            low_prod[iVar] = vec[idx+iVar] - low_prod[iVar];

        for( iVar = 0ul; iVar < nvar*nvar; ++iVar)
            block[iVar] = matrix[dia_ptr[iPoint]*nvar*nvar + iVar];
        
        #define A(I,J) block[(I)*nvar+(J)]
        //--- Transform system in Upper Matrix ---
        for ( iVar = 1ul; iVar < nvar; iVar++) {
            for ( jVar = 0ul; jVar < iVar; jVar++) {
            weight = A(iVar,jVar) / A(jVar,jVar);
            for ( kVar = jVar; kVar < nvar; kVar++)
                A(iVar,kVar) -= weight * A(jVar,kVar);
                low_prod[ iVar] -= weight * low_prod[ jVar];
            }
        }

        //--- Backwards substitution ---
        for ( iVar = nvar; iVar > 0ul;) {
            iVar--; // unsigned type
            for ( jVar = iVar+1; jVar < nvar; jVar++)
                low_prod[ iVar] -= A(iVar,jVar) * low_prod[ jVar];
                low_prod[ iVar] /= A(iVar,iVar);
        }
        #undef A

        for( iVar = 0; iVar < nvar; iVar++) 
            prod[ idx+iVar ] = low_prod[ iVar ];

    }
}
}

void forward_LUSGS( 
    const IntType* __restrict row_ptr, 
    const IntType* __restrict col_ind, 
    const IntType* __restrict dia_ptr, 
    const MATRIXTYPE* __restrict matrix, 
    const MATRIXTYPE* __restrict vec, 
    MATRIXTYPE *&prod, 
    IntType n, 
    IntType nvar ){
    
    IntType start = 0;
    IntType end = n;
    const int NNUMBER=5;
    bool *isReady = (bool *)malloc(n * sizeof(bool));
    memset(isReady, false, n * sizeof(bool));

    //IntType  iter_done;// current iterate steps
    //grid->GetData(&iter_done, INT, 1 ,"iter_done");

    //#pragma omp parallel for schedule(dynamic) shared(isReady)
    for (IntType iPoint = start; iPoint < end; iPoint++) {
        int thread = omp_get_thread_num();

        IntType idx = iPoint*nvar;
        IntType iVar, jVar, kVar, col_j;
        MATRIXTYPE low_prod[NNUMBER], block[NNUMBER*NNUMBER], weight;
        //LowerProduct(prod, iPoint, begin, low_prod);        // Compute L.x*
        memset(low_prod, 0, NNUMBER*sizeof(MATRIXTYPE));
        for ( iVar = row_ptr[iPoint]; iVar < dia_ptr[iPoint]; iVar++) {
            col_j = col_ind[iVar];
        
            // while( !isReady[col_j] ){
            //     #pragma omp flush(isReady)
            // }
            // #pragma omp flush(prod)

            for ( jVar = 0ul; jVar < nvar; jVar++) {
                for ( kVar = 0ul; kVar < nvar; kVar++)
                    low_prod[jVar] += matrix[iVar*nvar*nvar+jVar*nvar+kVar] * prod[col_j*nvar+kVar];
            }
        }
        
        //VectorSubtraction(&vec[idx], low_prod, &prod[idx]); // Compute y = b - L.x*
        for( iVar = 0; iVar < nvar; iVar++ )
            low_prod[iVar] = vec[idx+iVar] - low_prod[iVar];

        //Gauss_Elimination(iPoint, &prod[idx]);              // Solve D.x* = y
        //MatrixCopy(&matrix[dia_ptr[block_i]*nvar*nvar], block);
        for( iVar = 0ul; iVar < nvar*nvar; ++iVar)
            block[iVar] = matrix[dia_ptr[iPoint]*nvar*nvar + iVar];
        
        #define A(I,J) block[(I)*nvar+(J)]
        //--- Transform system in Upper Matrix ---
        for ( iVar = 1ul; iVar < nvar; iVar++) {
            for ( jVar = 0ul; jVar < iVar; jVar++) {
            weight = A(iVar,jVar) / A(jVar,jVar);
            for ( kVar = jVar; kVar < nvar; kVar++)
                A(iVar,kVar) -= weight * A(jVar,kVar);
                low_prod[ iVar] -= weight * low_prod[ jVar];
            }
        }

        //--- Backwards substitution ---
        for ( iVar = nvar; iVar > 0ul;) {
            iVar--; // unsigned type
            for ( jVar = iVar+1; jVar < nvar; jVar++)
                low_prod[ iVar] -= A(iVar,jVar) * low_prod[ jVar];
                low_prod[ iVar] /= A(iVar,iVar);
        }
        #undef A

        for( iVar = 0; iVar < nvar; iVar++) 
            prod[ idx+iVar ] = low_prod[ iVar ];
        //isReady[iPoint] = true;
    }
    free(isReady);
}

void backward_LUSGS(
    const IntType* __restrict row_ptr, 
    const IntType* __restrict col_ind, 
    const IntType* __restrict dia_ptr, 
    const MATRIXTYPE* __restrict matrix, 
    const MATRIXTYPE* __restrict vec, 
    MATRIXTYPE *&prod, 
    IntType n, 
    IntType nTCell,
    IntType nvar )
{
    IntType begin = 0;
    IntType end = n;
    bool *isReady = (bool *)malloc(nTCell * sizeof(bool));
    memset(isReady, false, nTCell * sizeof(bool));
    const int NNUMBER=5;

   //#pragma omp parallel for schedule(dynamic) shared(isReady)
    for (IntType iPoint = end - 1; iPoint >= begin; iPoint--) {
        MATRIXTYPE up_prod[nvar], dia_prod[nvar];
        MATRIXTYPE block[nvar*nvar];
        IntType idx = iPoint*nvar;
        IntType iVar, jVar, kVar, col_j;

        for ( iVar = 0ul; iVar < nvar; iVar++) {
            dia_prod[iVar] = 0.0;
            for ( jVar = 0ul; jVar < nvar; jVar++){
                dia_prod[iVar] += matrix[dia_ptr[iPoint]*nvar*nvar + iVar*nvar + jVar] * prod[idx + jVar];
            }
        }

        //UpperProduct(prod, iPoint, row_end, up_prod);     // Compute U.x_(n+1)
        //vec, unsigned long row_i, unsigned long col_ub, ScalarType *prod
        for ( iVar = 0ul; iVar < nvar; iVar++)
            up_prod[iVar] = 0.0;
        for ( iVar = dia_ptr[iPoint]+1; iVar < row_ptr[iPoint+1]; iVar++) {
            col_j = col_ind[iVar];

            // while( !isReady[col_j] ){
            //     #pragma omp flush
            // }
            // #pragma omp flush(prod)

            for ( jVar = 0ul; jVar < nvar; jVar++) {
                for ( kVar = 0ul; kVar < nvar; kVar++)
                    up_prod[jVar] += matrix[iVar*nvar*nvar + jVar*nvar + kVar] * prod[col_j*nvar + kVar];
            }
        }

        //VectorSubtraction(dia_prod, up_prod, &prod[idx]); // Compute y = D.x*-U.x_(n+1)
        for( iVar = 0; iVar < nvar; iVar++ )
            up_prod[iVar] = dia_prod[iVar] - up_prod[iVar];

        //Gauss_Elimination(iPoint, &prod[idx]);  // Solve D.x* = y
        //MatrixCopy(&matrix[dia_ptr[block_i]*nvar*nvar], block);
        for( iVar = 0ul; iVar < nvar*nvar; ++iVar)
            block[iVar] = matrix[dia_ptr[iPoint]*nvar*nvar + iVar];

        #define A(I,J) block[(I)*nvar+(J)]
        //--- Transform system in Upper Matrix ---
        for ( iVar = 1ul; iVar < nvar; iVar++) {
            for ( jVar = 0ul; jVar < iVar; jVar++) {
            MATRIXTYPE weight = A(iVar,jVar) / A(jVar,jVar);
            for ( kVar = jVar; kVar < nvar; kVar++)
                A(iVar,kVar) -= weight * A(jVar,kVar);
                up_prod[ iVar ] -= weight * up_prod[ jVar ];
            }
        }

        //--- Backwards substitution ---
        for ( iVar = nvar; iVar > 0ul;) {
            iVar--; // unsigned type
            for ( jVar = iVar+1; jVar < nvar; jVar++)
                up_prod[ iVar ] -= A(iVar,jVar) * up_prod[ jVar ];
                up_prod[ iVar ] /= A(iVar,iVar);
        }
        #undef A

        for(iVar = 0; iVar < nvar; iVar++) 
            prod[ idx+iVar ] = up_prod[ iVar ];

        //isReady[iPoint] = true;
    }
    free(isReady);
}
void backward_LUSGS_coloring( PolyGrid *grid, IntType *row_ptr, IntType *col_ind, IntType *dia_ptr, MATRIXTYPE *matrix, 
    MATRIXTYPE *vec, MATRIXTYPE *&prod, IntType n, IntType nTCell, IntType nvar ){

    IntType *luorder = (IntType *)grid->GetDataPtr(INT, n, "LUSGSCellOrder");
    IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, n, "LUSGScellsPerlayer");
    IntType laynum;
    IntType start, end, ilu;
    const int NNUMBER=5;

    for( laynum=cellsPerlayer[0]-1; laynum>=0; laynum-- ){
        start = cellsPerlayer[laynum+2];
        end   = cellsPerlayer[laynum+1];
#pragma omp parallel for private(ilu)
        for(ilu=start-1; ilu>=end; ilu--){
            IntType iPoint = luorder[ilu];
        MATRIXTYPE up_prod[nvar], dia_prod[nvar];
        MATRIXTYPE block[nvar*nvar];
        IntType idx = iPoint*nvar;
        IntType iVar, jVar, kVar, col_j;
        for ( iVar = 0ul; iVar < nvar; iVar++) {
            dia_prod[iVar] = 0.0;
            for ( jVar = 0ul; jVar < nvar; jVar++){
                dia_prod[iVar] += matrix[dia_ptr[iPoint]*nvar*nvar + iVar*nvar + jVar] * prod[idx + jVar];
            }
        }

        for ( iVar = 0ul; iVar < nvar; iVar++)
            up_prod[iVar] = 0.0;
        for ( iVar = dia_ptr[iPoint]+1; iVar < row_ptr[iPoint+1]; iVar++) {
            col_j = col_ind[iVar];
                for ( jVar = 0ul; jVar < nvar; jVar++) {
                    for ( kVar = 0ul; kVar < nvar; kVar++)
                        up_prod[jVar] += matrix[iVar*nvar*nvar + jVar*nvar + kVar] * prod[col_j*nvar + kVar];
                }
        }

        for( iVar = 0; iVar < nvar; iVar++ )
            up_prod[iVar] = dia_prod[iVar] - up_prod[iVar];
        for( iVar = 0ul; iVar < nvar*nvar; ++iVar)
            block[iVar] = matrix[dia_ptr[iPoint]*nvar*nvar + iVar];

        #define A(I,J) block[(I)*nvar+(J)]
        //--- Transform system in Upper Matrix ---
        for ( iVar = 1ul; iVar < nvar; iVar++) {
            for ( jVar = 0ul; jVar < iVar; jVar++) {
            MATRIXTYPE weight = A(iVar,jVar) / A(jVar,jVar);
            for ( kVar = jVar; kVar < nvar; kVar++)
                A(iVar,kVar) -= weight * A(jVar,kVar);
                up_prod[ iVar ] -= weight * up_prod[ jVar ];
            }
        }
        //--- Backwards substitution ---
        for ( iVar = nvar; iVar > 0ul;) {
            iVar--; // unsigned type
            for ( jVar = iVar+1; jVar < nvar; jVar++)
                up_prod[ iVar ] -= A(iVar,jVar) * up_prod[ jVar ];
                up_prod[ iVar ] /= A(iVar,iVar);
        }
        #undef A
        for(iVar = 0; iVar < nvar; iVar++) 
            prod[ idx+iVar ] = up_prod[ iVar ];
    }
}
}

void MatrixLUSGS( PolyGrid *grid, IntType level ){
    IntType matrixformat = 1;
    RealFlow tol = 0.01;
    //grid->GetData(&kspan, INT, 1, "kspan");

    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nIFace = grid->GetNIFace();
    IntType nvar = NVAR;
    IntType nT5    = nvar*nTCell;
    IntType n = nTCell+nBFace;

    // residual array of x in flowstar
    RealFlow *DQ[NVAR];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nvar*n, "DQ");
    if(!DQ[0]){
        mfmem::snew_array_1D(DQ[0], nvar*n, dmrfl);
        grid->UpdateDataPtr(DQ[0], REAL_FLOW, nvar*n, "DQ");
    }
    assert(DQ[0] != 0);
    for(int i=1; i<NVAR; i++) DQ[i] = &DQ[i-1][n];

    int mpirank = 0;
#ifdef MPICH
    int size = 0;      
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    IntType *matrixInfo  = (IntType *)(grid->GetDataPtr(INT, 16, "matrixInfo"));
    IntType matrixN = matrixInfo[0];  //n for csr format
    IntType nnz     = matrixInfo[1];  //nnz for csr fomart

    //allocate tmp memory during GMRES iteration
    RealFlow *x = NULL;
    mfmem::snew_array_1D(x, nvar*matrixN,dmrfl);
    assert(x != 0); 
    RealFlow *b = NULL;
    mfmem::snew_array_1D(b, nvar*matrixN,dmrfl);
    assert(b != 0); 

    //get b during solving A*x=b
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nT5, "res");
    RealFlow *rhs[nvar];
    rhs[0] = res;
    for(IntType iVar=1; iVar<nvar; iVar++) rhs[iVar] = &rhs[iVar-1][nTCell];
    IntType idx = 0;
    for(IntType iCell = 0; iCell < nTCell; iCell++){
        for(IntType iVar = 0; iVar < nvar; iVar++){
            b[idx++] = rhs[iVar][iCell];
        }
    }

    //BSR format for the matrix
    IntType *bsr_col_ind = (IntType *)(grid->GetDataPtr(INT, nnz, "bsr_col_ind"));
    IntType *bsr_row_ptr = (IntType *)(grid->GetDataPtr(INT, matrixN+1, "bsr_row_ptr"));
    IntType *dia_ptr     = (IntType *)(grid->GetDataPtr(INT, matrixN, "bsr_dia_ptr"));
    MATRIXTYPE *matrix     = (MATRIXTYPE *)(grid->GetDataPtr(MATRIXINTTYPE, nnz*nvar*nvar, "matrix"));

    timeval t2, t3;
    double time_t2;

    IntType fusedMethodFlag = 0;
    grid->GetData(&fusedMethodFlag,   INT, 1, "fusedImplicitMethod");
    gettimeofday(&t2, NULL);
    time_t2 = (double)t2.tv_sec + (double)t2.tv_usec/1000000;

    if(fusedMethodFlag == 1){
        LusgsFusedOrigin( bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, b, x, matrixN, nvar );
    }
    else if(fusedMethodFlag == 2){
        fused_lusgs_loop( grid, bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, \
            b, x, matrixN, nnz, nvar );
    }
    else if(fusedMethodFlag == 0){
        //forward_LUSGS( bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, b, x, nTCell, nvar );

#ifdef MPICH
    //communications for x
    RealFlow *MPItmp[NVAR];
    MPItmp[0] = new RealFlow[nvar*n];
    for(int i=1; i<NVAR; i++) MPItmp[i] = &MPItmp[i-1][n];

    IntType index = 0;
    for(IntType iCell = 0; iCell < nTCell; iCell++){
        for(IntType iVar = 0; iVar < nvar; iVar++){
            MPItmp[iVar][iCell] = x[index++];
        }
    }

    grid->RecvSendVarNeighbor_Togeth( nvar, MPItmp );

    index = nTCell*nvar;
    for(IntType iCell = nTCell+nBFace-nIFace; iCell < nTCell+nBFace; iCell++){
        for(IntType iVar = 0; iVar < nvar; iVar++){
            x[index++] = MPItmp[iVar][iCell];
        }
    }
    delete[] MPItmp[0];
#endif

        //backward_LUSGS( bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, b, x, nTCell, nTCell, nvar );
    }
    else{
        printf("No available implicit LU-SGS method found!\n");
        exit(-1);
    }
    ite++;
    gettimeofday(&t3, NULL);
    double time_t3 = (double)t3.tv_sec + (double)t3.tv_usec/1000000;
    ILUexe += (time_t3 - time_t2);

    IntType id = 0;
    for(IntType iCell = 0; iCell < nTCell; iCell++){
        for(IntType iVar = 0; iVar < nvar; iVar++){
            DQ[iVar][iCell] = x[id++];
        }
    }

    mfmem::sdel_array_1D(b);
    mfmem::sdel_array_1D(x);

    UpdateFlowField3D_CFL3d(grid, DQ);
}

void GmresLUSGS( PolyGrid *grid, IntType level ){
    
    IntType kspan = 25, gmresmaxits = 1000;
    RealFlow tol = 0.1;
    grid->GetData(&kspan, INT, 1, "kspan");
    grid->GetData(&tol, REAL_FLOW, 1, "gmresepsilon");
    grid->GetData(&gmresmaxits, INT, 1, "gmresmaxits");

    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nIFace = grid->GetNIFace();
    IntType nvar = NVAR;
    IntType nT5    = nvar*nTCell;
    IntType n = nTCell+nBFace;

    int mpirank = 0;
#ifdef MPICH
    int size = 0;      
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    IntType *matrixInfo  = (IntType *)(grid->GetDataPtr(INT, 16, "matrixInfo"));
    IntType matrixN = matrixInfo[0];  //n for csr format
    IntType nnz     = matrixInfo[1];  //nnz for csr fomart
    
    int iter_done;
     grid->GetData(&iter_done, INT, 1 ,"iter_done");

    //get b during solving A*x=b
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nT5, "res");
    // Arrange the storage order of res to res_tmp
    MATRIXTYPE *res_tmp = NULL;
    mfmem::snew_array_1D(res_tmp, nT5,dmrfl);
    assert(res_tmp != 0);
    RealFlow *rhs[nvar];
    rhs[0] = res;
    for(IntType iVar=1; iVar<nvar; iVar++) rhs[iVar] = &rhs[iVar-1][nTCell];
#ifdef FS_OPENMP
#pragma omp parallel for 
#endif
    for(IntType iCell = 0; iCell < nTCell; iCell++){
        for(IntType iVar = 0; iVar < nvar; iVar++){
            res_tmp[iCell*nvar+iVar] = rhs[iVar][iCell];
        }
    }

    //allocate memory of final solution
    MATRIXTYPE *x_final = NULL;
    mfmem::snew_array_1D(x_final, nvar*matrixN, dmrfl);
    assert(x_final != 0);
    memset(x_final, 0, nvar*matrixN*sizeof(MATRIXTYPE));

    //allocate tmp memory during GMRES iteration
    // MATRIXTYPE *x = NULL;
    // mfmem::snew_array_1D(x, nvar*matrixN*(kspan+1),dmrfl);
    // assert(x != 0); 
    // MATRIXTYPE *b = NULL;
    // mfmem::snew_array_1D(b, nvar*matrixN*(kspan+1),dmrfl);
    // assert(b != 0); 

    IntType NUM_OF_THREADS = 1;
#ifdef FS_OPENMP
	    #pragma omp parallel shared(NUM_OF_THREADS)
	    {
		    #pragma omp single
		    {
                NUM_OF_THREADS = omp_get_max_threads();
            }
        }
#endif

    MATRIXTYPE *x = x_global;
    MATRIXTYPE *b = b_global;
    GETYPE *diagmatrix = diagmatrix_global;
    //BSR format for the matrix
    IntType *bsr_col_ind = (IntType *)(grid->GetDataPtr(INT, nnz, "bsr_col_ind"));
    IntType *bsr_row_ptr = (IntType *)(grid->GetDataPtr(INT, matrixN+1, "bsr_row_ptr"));
    IntType *dia_ptr     = (IntType *)(grid->GetDataPtr(INT, matrixN, "bsr_dia_ptr"));
    MATRIXTYPE *matrix     = (MATRIXTYPE *)(grid->GetDataPtr(MATRIXINTTYPE, nnz*nvar*nvar, "matrix"));
    // GETYPE *diagmatrix = (GETYPE *)(grid->GetDataPtr(MATRIXINTTYPE, nTCell*nvar*nvar, "diagmatrix"));
    
    diagblockDecomp( bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, diagmatrix, nTCell, nvar );

    bool converge = false;
    IntType re = 0, i = 0, j;
    //start GMRES iteration
    for( re=0; re<gmresmaxits; re++){

        IntType m = kspan;
        // MATRIXTYPE **H = new MATRIXTYPE*[m+1];
        // for( j=0; j<m+1; j++) H[j] = new MATRIXTYPE[m];

        MATRIXTYPE **H = new MATRIXTYPE*[m+STEP+1];
        for( j=0; j<m+1; j++) H[j] = new MATRIXTYPE[m+STEP];

        MATRIXTYPE *y = new MATRIXTYPE[m];
        MATRIXTYPE *sn = new MATRIXTYPE[m+1];
        MATRIXTYPE *cs = new MATRIXTYPE[m+1];
        MATRIXTYPE *g = new MATRIXTYPE[m+1];
        for( j=0; j<m+1; j++){
            sn[j] = 0.0;
            cs[j] = 0.0;
            g[j]  = 0.0;
        }

        memset(b, 0, nvar*matrixN*(kspan+1) * sizeof(MATRIXTYPE));
        memset(x, 0, nvar*matrixN*(kspan+1) * sizeof(MATRIXTYPE));
        MATRIXTYPE norm0, beta;
        // GmresMinResB( b, res_tmp, nT5 );
        // MATRIXTYPE norm0 = DotProductMPI( res_tmp, nT5 );
        // norm0 = sqrt(norm0);
        // MATRIXTYPE tmp_norm = DotProductMPI( b, nT5 );
        // tmp_norm = sqrt(tmp_norm);
        // MATRIXTYPE beta = tmp_norm;
        // bDivScalar( b, -beta, nT5);

        //GmresMatrixVectorMult( grid, bsr_row_ptr, bsr_col_ind, matrix, b, x_final, nTCell, nBFace, nIFace, nvar );
#ifdef MIXEDPRECISION
        spmv_bsr5_sve_omp_float(bsr_row_ptr, bsr_col_ind, matrix, b, x_final, nTCell, nvar);
        vector_initial_omp_sve_float( b, res_tmp, nT5, &norm0, &beta, NUM_OF_THREADS);
#else
        spmv_bsr5_sve_omp_double(bsr_row_ptr, bsr_col_ind, matrix, b, x_final, nTCell, nvar);
        vector_initial_omp_sve_double( b, res_tmp, nT5, &norm0, &beta); 
#endif

        /*--- Initialize the RHS of the reduced system ---*/
        g[0] = beta;
        for( i=0; i<kspan; i++ ){

            MATRIXTYPE *tmp_b = &b[i*nvar*matrixN];
            MATRIXTYPE *tmp_b1 = &b[(i+1)*nvar*matrixN];
            MATRIXTYPE *tmp_x = &x[i*nvar*matrixN];

            timeval t2, t3, t4, t5;
            gettimeofday(&t2, NULL);
            double time_t2 = (double)t2.tv_sec + (double)t2.tv_usec/1000000;

            //forward_LUSGS( bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, tmp_b, tmp_x, nTCell, nvar );
            //backward_LUSGS( bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, tmp_b, tmp_x, nTCell, nTCell, nvar );

            //forward_LUSGS_2( grid, bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, diagmatrix, tmp_b, tmp_x, nTCell, nvar );
            //backward_LUSGS_2( grid, bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, diagmatrix, tmp_b, tmp_x, nTCell, nTCell, nvar );
            for_back_LUSGS_3( grid, bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, diagmatrix, tmp_b, tmp_x, nTCell, nvar );
            //for_back_LUSGS_3_1( grid, bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, diagmatrix, tmp_b, tmp_x, nTCell, nvar );
            
            //forward_LUSGS_coloring( grid, bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, tmp_b, tmp_x, nTCell, nvar );
            //backward_LUSGS_coloring( grid, bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, tmp_b, tmp_x, nTCell, nTCell, nvar );

            //LusgsFusedOrigin( bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, tmp_b, tmp_x, nTCell, nTCell, nvar );
            ite++;
            gettimeofday(&t3, NULL);
            double time_t3 = (double)t3.tv_sec + (double)t3.tv_usec/1000000;
            ILUexe += (time_t3 - time_t2);

            /*--- MPI Communication for solution x ---*/
        #ifdef MPICH
            //grid->RecvSendVarMatrixNeighbor_Togeth( nvar, tmp_x );
            //grid->RecvSendVarMatrixNeighbor_Togeth2( nvar, tmp_x );
        #endif

            //GmresMatrixVectorMult( grid, bsr_row_ptr, bsr_col_ind, matrix, tmp_b1, tmp_x, nTCell, nBFace, nIFace, nvar );
#ifdef MIXEDPRECISION
            spmv_bsr5_sve_omp_float(bsr_row_ptr, bsr_col_ind, matrix, tmp_b1, tmp_x, nTCell, nvar);
#else
            spmv_bsr5_sve_omp_double(bsr_row_ptr, bsr_col_ind, matrix, tmp_b1, tmp_x, nTCell, nvar);
#endif            

            gettimeofday(&t4, NULL);
            double time_t4 = (double)t4.tv_sec + (double)t4.tv_usec/1000000;
            MPIexe += (time_t4 - time_t3);

            //ModGramSchmidt(i, H, b, nTCell, nBFace, nvar, matrixN); // Third para should be gx on CPU memory if used. 
            MPI_Barrier(MPI_COMM_WORLD);
            double time_t44 = MPI_Wtime();

            if( ((i+1) % STEP == 0) && ( (i+STEP) < kspan) ){
                s_step_orthogonalization(i, H, b, nTCell, nvar, matrixN*nvar, \
                    matrix, bsr_row_ptr, bsr_col_ind, dia_ptr, NUM_OF_THREADS, STEP);
                for ( j = i; j < i+STEP; j++) {
                    for (int k = 0; k < j; k++) {
                        ApplyGivens(sn[k], cs[k], H[k][j], H[k+1][j]);
                    }
                    GenerateGivens(H[j][j], H[j+1][j], sn[j], cs[j]);
                    ApplyGivens(sn[j], cs[j], g[j], g[j+1]);
                    beta = fabs(g[j+1]);
                }
            }
            if (beta < tol * norm0) {
                converge = true;
                break;
            }
// #ifdef MIXEDPRECISION
//             //ClassicalGramSchmidt(i, H, b, nTCell, nBFace, nvar, matrixN);
//             //ClassicalGramSchmidt_blas(i, H, b, nTCell, nBFace, nvar, matrixN);
//             //ClassicalGramSchmidt_sve_float(i, H, b, nTCell, nBFace, nvar, matrixN, NUM_OF_THREADS);
//             //ClassicalGramSchmidt_hybrid(i, H, b, nTCell, nBFace, nvar, matrixN); 
//             //ClassicalGramSchmidt_hybrid_sve(i, H, b, nTCell, nBFace, nvar, matrixN, NUM_OF_THREADS, iter_done);
//             ClassicalGramSchmidt_hybrid_sve_onceSyn(i, H, b, nTCell, nBFace, nvar, matrixN, NUM_OF_THREADS, iter_done); 
// #else
//             ClassicalGramSchmidt(i, H, b, nTCell, nBFace, nvar, matrixN);
// #endif
//             // gettimeofday(&t5, NULL);
//             // double time_t5 = (double)t5.tv_sec + (double)t5.tv_usec/1000000;
//             // GMRES_Schmidt += (time_t5 - time_t4);
//             /*---  Apply old Givens rotations to new column of the Hessenberg matrix then generate the
//             new Givens rotation matrix and apply it to the last two elements of H[:][i] and g ---*/

//             for (unsigned long k = 0; k < i; k++){
//                 ApplyGivens(sn[k], cs[k], H[k][i], H[k + 1][i]);
//             }
//             GenerateGivens(H[i][i], H[i + 1][i], sn[i], cs[i]);
//             ApplyGivens(sn[i], cs[i], g[i], g[i + 1]);
//             beta = fabs(g[i + 1]);
//            if( beta < tol * norm0 ) {
//                if(i>1){
//                    converge = true;
//                }
//            }

           if( converge ) {break;}
            double time_t5 = MPI_Wtime();
            GMRES_Schmidt += time_t5 - time_t44;
        }

        SolveReduced(i, H, g, y);
        for (IntType k = 0; k < i; k++) {
            MATRIXTYPE factor = y[k];
            //add x to final solution, ghost cells are included
            //Addx( x_final, &x[k*nvar*matrixN], factor, nvar*matrixN);
#ifdef MIXEDPRECISION
            vector_sub_scaled_sve_omp_float( x_final, &x[k*nvar*matrixN], factor, nvar*matrixN);
#else
            vector_sub_scaled_sve_omp_double( x_final, &x[k*nvar*matrixN], factor, nvar*matrixN);
#endif
        }

        delete[] g;
        delete[] sn;
        delete[] cs;
        delete[] y;
        for(int ii=0; ii<m+1; ii++) {delete[] H[ii];}

        if( converge ) {break;}
        gmresmaxits -= (i+1);
        kspan = min<IntType>(kspan, gmresmaxits);
    }

    // residual array of x in flowstar
    RealFlow *DQ[NVAR];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nvar*n, "DQ");
    if(!DQ[0]){
        mfmem::snew_array_1D(DQ[0], nvar*n, dmrfl);
        grid->UpdateDataPtr(DQ[0], REAL_FLOW, nvar*n, "DQ");
    }
    assert(DQ[0] != 0);
    for( j=1; j<NVAR; j++) DQ[j] = &DQ[j-1][n];

#ifdef FS_OPENMP
#pragma omp parallel for 
#endif
    for(IntType iCell = 0; iCell < nTCell; iCell++){
        for(IntType iVar = 0; iVar < nvar; iVar++){
            DQ[iVar][iCell] = x_final[iCell*nvar+iVar];
        }
    }
    mfmem::sdel_array_1D(res_tmp);
    //mfmem::sdel_array_1D(b);
    //mfmem::sdel_array_1D(x);
    mfmem::sdel_array_1D(x_final);

    if(!mpirank)
        printf( "rank:%d GMRES + LUSGS sptrsv iteration:%d \n", mpirank, ite );
    UpdateFlowField3D_CFL3d(grid, DQ);
}

void GmresILU( PolyGrid *grid, IntType level ){
    
    IntType kspan = 25, gmresmaxits = 1000, matrixformat = 1;
    RealFlow tol = 0.1;
    grid->GetData(&kspan, INT, 1, "kspan");
    grid->GetData(&tol, REAL_FLOW, 1, "gmresepsilon");
    grid->GetData(&gmresmaxits, INT, 1, "gmresmaxits");
    //grid->GetData(&matrixformat, INT, 1, "matrixformat"); // bsr format defualt

    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nIFace = grid->GetNIFace();
    IntType nvar = NVAR;
    IntType nT5    = nvar*nTCell;
    IntType n = nTCell+nBFace;

    int mpirank = 0;
#ifdef MPICH
    int size = 0;      
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    IntType *matrixInfo  = (IntType *)(grid->GetDataPtr(INT, 16, "matrixInfo"));
    IntType matrixN = matrixInfo[0];  //n for csr format
    IntType nnz     = matrixInfo[1];  //nnz for csr fomart

    //get b during solving A*x=b
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nT5, "res");
    // Arrange the storage order of res to res_tmp
    MATRIXTYPE *res_tmp = NULL;
    mfmem::snew_array_1D(res_tmp, nT5,dmrfl);
    assert(res_tmp != 0);
    RealFlow *rhs[nvar];
    rhs[0] = res;
    for(IntType iVar=1; iVar<nvar; iVar++) rhs[iVar] = &rhs[iVar-1][nTCell];
#ifdef FS_OPENMP
#pragma omp parallel for 
#endif
    for(IntType iCell = 0; iCell < nTCell; iCell++){
        for(IntType iVar = 0; iVar < nvar; iVar++){
            res_tmp[iCell*nvar+iVar] = rhs[iVar][iCell];
        }
    }

    //allocate memory of final solution
    MATRIXTYPE *x_final = NULL;
    mfmem::snew_array_1D(x_final, nvar*matrixN, dmrfl);
    assert(x_final != 0);
    memset(x_final, 0, nvar*matrixN*sizeof(MATRIXTYPE));

    //allocate tmp memory during GMRES iteration
    MATRIXTYPE *x = NULL;
    mfmem::snew_array_1D(x, nvar*matrixN*(kspan+1),dmrfl);
    assert(x != 0); 
    MATRIXTYPE *b = NULL;
    mfmem::snew_array_1D(b, nvar*matrixN*(kspan+1),dmrfl);
    assert(b != 0); 

    //BSR format for the matrix
    IntType *bsr_col_ind = (IntType *)(grid->GetDataPtr(INT, nnz, "bsr_col_ind"));
    IntType *bsr_row_ptr = (IntType *)(grid->GetDataPtr(INT, matrixN+1, "bsr_row_ptr"));
    IntType *dia_ptr     = (IntType *)(grid->GetDataPtr(INT, matrixN, "bsr_dia_ptr"));
    MATRIXTYPE *matrix     = (MATRIXTYPE *)(grid->GetDataPtr(MATRIXINTTYPE, nnz*nvar*nvar, "matrix"));

    //allocate tmp memory during ILU precond
    MATRIXTYPE *ILU_matrix = NULL;
    mfmem::snew_array_1D(ILU_matrix, nnz*nvar*nvar, dmrfl);
    assert(ILU_matrix != 0);
    MATRIXTYPE *invM = NULL;
    mfmem::snew_array_1D(invM, nnz*nvar*nvar, dmrfl);
    assert(invM != 0);
   
    //ILU decomposition, only bsr format is available currently
    timeval t0, t1;
    gettimeofday(&t0, NULL);
    double time_t0 = (double)t0.tv_sec + (double)t0.tv_usec/1000000;

    #ifdef FS_OPENMP
        PrecondILU_decomp1( grid, bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, invM, nTCell, nvar, nnz, ILU_matrix );
    #else
        PrecondILU_decomp0( grid, bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, invM, nTCell, nvar, nnz, ILU_matrix );
    #endif

    gettimeofday(&t1, NULL);
    double time_t1 = (double)t1.tv_sec + (double)t1.tv_usec/1000000;
    ILUbuild += (time_t1 - time_t0);

    bool converge = false;
    IntType re = 0, i = 0, j;

    //start GMRES iteration
    for( re=0; re<gmresmaxits; re++){

        IntType m = kspan;
        MATRIXTYPE **H = new MATRIXTYPE*[m+1];
        for( j=0; j<m+1; j++) 
            H[j] = new MATRIXTYPE[m];

        MATRIXTYPE *y = new MATRIXTYPE[m];
        MATRIXTYPE *sn = new MATRIXTYPE[m+1];
        MATRIXTYPE *cs = new MATRIXTYPE[m+1];
        MATRIXTYPE *g = new MATRIXTYPE[m+1];
        for( j=0; j<m+1; j++){
            sn[j] = 0.0;
            cs[j] = 0.0;
            g[j]  = 0.0;
        }

        memset(b, 0, nvar*matrixN*(kspan+1) * sizeof(MATRIXTYPE));
        memset(x, 0, nvar*matrixN*(kspan+1) * sizeof(MATRIXTYPE));
        GmresMatrixVectorMult( grid, bsr_row_ptr, bsr_col_ind, matrix, b, x_final, nTCell, nBFace, nIFace, nvar );
// #ifdef MIXEDPRECISION
//         spmv_bsr5_sve_omp_float(bsr_row_ptr, bsr_col_ind, matrix, b, x_final, nTCell, nvar);
// #else
//         spmv_bsr5_sve_omp_double(bsr_row_ptr, bsr_col_ind, matrix, b, x_final, nTCell, nvar);
// #endif
        GmresMinResB( b, res_tmp, nT5 );
        MATRIXTYPE norm0 = DotProductMPI( res_tmp, nT5 );
        norm0 = sqrt(norm0);
        MATRIXTYPE tmp_norm = DotProductMPI( b, nT5 );
        tmp_norm = sqrt(tmp_norm);
        MATRIXTYPE beta = tmp_norm;
        bDivScalar( b, -beta, nT5);

//         MATRIXTYPE norm0, beta;
// #ifdef MIXEDPRECISION
//         vector_initial_omp_sve_float( b, res_tmp, nT5, &norm0, &beta);
// #else
//         vector_initial_omp_sve_double( b, res_tmp, nT5, &norm0, &beta);
// #endif

        /*--- Initialize the RHS of the reduced system ---*/
        g[0] = beta;
        for( i=0; i<kspan; i++ ){

            MATRIXTYPE *tmp_b = &b[i*nvar*matrixN];
            MATRIXTYPE *tmp_b1 = &b[(i+1)*nvar*matrixN];
            MATRIXTYPE *tmp_x = &x[i*nvar*matrixN];

            timeval t2, t3, t4, t5;
            gettimeofday(&t2, NULL);
            double time_t2 = (double)t2.tv_sec + (double)t2.tv_usec/1000000;

            #ifdef FS_OPENMP
                PrecondILU_solve0( bsr_row_ptr, bsr_col_ind, dia_ptr, ILU_matrix, invM, tmp_b, tmp_x, nTCell, nvar );
            #else
                PrecondILU_solve0( bsr_row_ptr, bsr_col_ind, dia_ptr, ILU_matrix, invM, tmp_b, tmp_x, nTCell, nvar );
            #endif

            //forward_LUSGS( bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, tmp_b, tmp_x, nTCell, nvar );
            //backward_LUSGS( bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, tmp_b, tmp_x, nTCell, nTCell, nvar );

            //forward_LUSGS_coloring( grid, bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, tmp_b, tmp_x, nTCell, nvar );
            //backward_LUSGS_coloring( grid, bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, tmp_b, tmp_x, nTCell, nTCell, nvar );

            //LusgsFusedOrigin( bsr_row_ptr, bsr_col_ind, dia_ptr, matrix, tmp_b, tmp_x, nTCell, nTCell, nvar );
            ite++;
            gettimeofday(&t3, NULL);
            double time_t3 = (double)t3.tv_sec + (double)t3.tv_usec/1000000;
            ILUexe += (time_t3 - time_t2);

            /*--- MPI Communication for solution x ---*/
        #ifdef MPICH
            //grid->RecvSendVarMatrixNeighbor_Togeth( nvar, tmp_x );
            grid->RecvSendVarMatrixNeighbor_Togeth2( nvar, tmp_x );
        #endif

            GmresMatrixVectorMult( grid, bsr_row_ptr, bsr_col_ind, matrix, tmp_b1, tmp_x, nTCell, nBFace, nIFace, nvar );
// #ifdef MIXEDPRECISION
//             spmv_bsr5_sve_omp_float(bsr_row_ptr, bsr_col_ind, matrix, tmp_b1, tmp_x, nTCell, nvar);
// #else
//             spmv_bsr5_sve_omp_double(bsr_row_ptr, bsr_col_ind, matrix, tmp_b1, tmp_x, nTCell, nvar);
// #endif            

            gettimeofday(&t4, NULL);
            double time_t4 = (double)t4.tv_sec + (double)t4.tv_usec/1000000;
            MPIexe += (time_t4 - time_t3);

            ModGramSchmidt(i, H, b, nTCell, nBFace, nvar, matrixN); // Third para should be gx on CPU memory if used. 

// #ifdef MIXEDPRECISION
//             ClassicalGramSchmidt_sve_float(i, H, b, nTCell, nBFace, nvar, matrixN);
// #else
//             ClassicalGramSchmidt(i, H, b, nTCell, nBFace, nvar, matrixN);
// #endif

            gettimeofday(&t5, NULL);
            double time_t5 = (double)t5.tv_sec + (double)t5.tv_usec/1000000;
            GMRES_Schmidt += (time_t5 - time_t4);
            
            /*---  Apply old Givens rotations to new column of the Hessenberg matrix then generate the
            new Givens rotation matrix and apply it to the last two elements of H[:][i] and g ---*/
    
            for (unsigned long k = 0; k < i; k++){
                ApplyGivens(sn[k], cs[k], H[k][i], H[k + 1][i]);
            }
            GenerateGivens(H[i][i], H[i + 1][i], sn[i], cs[i]);

            ApplyGivens(sn[i], cs[i], g[i], g[i + 1]);

            //---  Set L2 norm of residual and check if solution has converged ---

            beta = fabs(g[i + 1]);

            if( beta < tol * norm0 ) {
                if(i>1){
                    converge = true;
                }
            }
            if( converge ) {break;}
        }

        SolveReduced(i, H, g, y);
        for (IntType k = 0; k < i; k++) {
            MATRIXTYPE factor = y[k];
            //add x to final solution, ghost cells are included
            Addx( x_final, &x[k*nvar*matrixN], factor, nvar*matrixN);
// #ifdef MIXEDPRECISION
//             vector_sub_scaled_sve_omp_float( x_final, &x[k*nvar*matrixN], factor, nvar*matrixN);
// #else
//             vector_sub_scaled_sve_omp_double( x_final, &x[k*nvar*matrixN], factor, nvar*matrixN);
// #endif
        }

        delete[] g;
        delete[] sn;
        delete[] cs;
        delete[] y;
        for(int ii=0; ii<m+1; ii++) {delete[] H[ii];}

        if( converge ) {break;}
        gmresmaxits -= (i+1);
        kspan = min<IntType>(kspan, gmresmaxits);
    }

    // residual array of x in flowstar
    RealFlow *DQ[NVAR];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nvar*n, "DQ");
    if(!DQ[0]){
        mfmem::snew_array_1D(DQ[0], nvar*n, dmrfl);
        grid->UpdateDataPtr(DQ[0], REAL_FLOW, nvar*n, "DQ");
    }
    assert(DQ[0] != 0);
    for( j=1; j<NVAR; j++) DQ[j] = &DQ[j-1][n];

#ifdef FS_OPENMP
#pragma omp parallel for 
#endif
    for(IntType iCell = 0; iCell < nTCell; iCell++){
        for(IntType iVar = 0; iVar < nvar; iVar++){
            DQ[iVar][iCell] = x_final[iCell*nvar+iVar];
        }
    }

    mfmem::sdel_array_1D(res_tmp);
    mfmem::sdel_array_1D(b);
    mfmem::sdel_array_1D(x);
    mfmem::sdel_array_1D(x_final);
    //mfmem::sdel_array_1D(ILU_matrix);
    //mfmem::sdel_array_1D(invM);

    if(!mpirank)
        printf( "rank:%d GMRES + ILU sptrsv iteration:%d \n", mpirank, ite );
    UpdateFlowField3D_CFL3d(grid, DQ);
}

void ComputeBsrIndex( PolyGrid *grid, IntType *&row_ptr, \
    IntType *&col_ind, IntType *&dia_ptr, IntType nnz, IntType MatrixN ){

    int mpirank = 0;
#ifdef MPICH
    int size = 0;      
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    IntType Bstart = 0;
    IntType nBFace = grid->GetNBFace();
    IntType nIFace = grid->GetNIFace();
    IntType nTCell = grid->GetNTCell();
    IntType *f2c   = grid->Getf2c();
    IntType *nFPC  = CalnFPC(grid);
    IntType **C2F  = CalC2F(grid); 
    IntType ifStart = nBFace - nIFace;
    IntType count = 0;
   
    IntType *ghost2global = grid->CalGhost2Global(Bstart);
    IntType *col_ind_origin = new IntType[nnz];
    IntType *GhostCell = new IntType[2*nTCell];
    IntType ghostnnz = 0;
    for(int i=0;i<MatrixN+1;i++) {row_ptr[i] = 0;}

    for(int iCell = 0;iCell<nTCell;iCell++){
        IntType cell = iCell;
        for(IntType iFace=0; iFace<nFPC[cell]; iFace++){
            IntType face  = C2F[cell][iFace];
            IntType c2    = f2c[face+face]+f2c[face+face+1]-cell;
  
            IntType Brow = Bstart+cell;
            IntType Bcol = 0;
            if(face >= nBFace){
                Bcol = Bstart+c2;
            }
            else if(face < ifStart){
                continue;
            }
            else{
                Bcol = Bstart + c2 - (nBFace - nIFace);
                IntType index = c2 - (nBFace - nIFace);
                GhostCell[ghostnnz*2] = index;
                GhostCell[ghostnnz*2 + 1] = cell;
                ghostnnz++;

                //Bcol = ghost2global[face-ifStart];  //global index for ghost cells among ranks
                //printf("rank:%d cell:%d neighborCell:%d  right:%d  wrong:%d\n",mpirank,iCell,c2,Bstart+c2,ghost2global[face-ifStart]);

            }

            IntType Irow = Brow;
            IntType Icol = Bcol;

            row_ptr[Irow+1]++;
            col_ind_origin[count++] = Icol;

        }
        IntType Brow = Bstart+cell;
        IntType Bcol = Brow;
        IntType Irow = Brow;
        IntType Icol = Bcol;

        row_ptr[Irow+1]++;
        col_ind_origin[count++] = Icol;
    }

    for(IntType ind=0; ind<nIFace; ind++){
        for(IntType i=0; i<ghostnnz; i++){
            IntType Brow = Bstart + GhostCell[i*2];
            IntType Bcol = GhostCell[i*2+1];
            if( (nTCell+ind) == GhostCell[i*2] ){
                row_ptr[nTCell+ind+1]++;
                col_ind_origin[count++] = Bcol;
            }
            else{
                continue;
            }
        }
        IntType Brow = Bstart + nTCell + ind;
        IntType Bcol = Brow;
        row_ptr[nTCell+ind+1]++;
        col_ind_origin[count++] = Bcol;
    }

    for(int i=1; i<MatrixN+1; i++){
        row_ptr[i] += row_ptr[i-1];
    }

    for(int i=0;i<nnz;i++){
        col_ind[i] = col_ind_origin[i];
    }

    std::vector<IntType> vec = {0,0,0,0,0,0,0,0,0,0};
    for(int i=0; i<MatrixN; i++){
        int length = row_ptr[i+1] - row_ptr[i];
        for(int j=row_ptr[i]; j<row_ptr[i+1]; j++){
            vec[j - row_ptr[i]] = col_ind_origin[ j ];
        }
        vector<IntType>::iterator it = vec.begin();
        std::sort( vec.begin(), (it+length) );

        for(int j=row_ptr[i]; j<row_ptr[i+1]; j++){
            col_ind[ j ] = *( it + j - row_ptr[i] );
        }  
    }
    delete[] col_ind_origin;
    delete[] GhostCell;

    for(int i=0; i<MatrixN; i++){
        for(int j=row_ptr[i]; j<row_ptr[i+1]; j++){
            if(col_ind[j] == i){
                dia_ptr[i] = j;
                break;
            }
        }
    }
    //printf("Grid nTcell:%d  nnz:%d row_ptr:%d col_ind:%d\n",nTCell,nnz,row_ptr[nTCell],col_ind[row_ptr[nTCell]-1]);
    //grid->printcsrMatrix( row_ptr, col_ind, nnz, nTCell, nvar );
}

void ReorderedIndex( PolyGrid *grid, IntType *oor, IntType *ooc, 
    IntType *bsr_row_ptr, IntType *bsr_col_ind, IntType *&bsrIndex, IntType nnzb, IntType MatrixN ){

    IntType bs = NVAR;
    IntType bs2 = bs * bs;

    int mpirank = 0;
#ifdef MPICH
    int size = 0;      
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    for(IntType i = 0; i < nnzb * bs2; i++){
        IntType br = oor[i] / bs;
        IntType bc = ooc[i] / bs;

        if (br >= MatrixN || bc >= MatrixN) {
            printf("Error: Block row/col out of range\n");
            printf("mpirank:%d row:%d, col:%d,  MatrixN:%d\n", mpirank, br, bc, MatrixN);
            exit(-1);
        }

        IntType ir = oor[i] % bs;
        IntType ic = ooc[i] % bs;

        IntType b_pos;
        for(b_pos = bsr_row_ptr[br]; b_pos < bsr_row_ptr[br + 1]; b_pos++){
            if(bsr_col_ind[b_pos] == bc) break;
        }
        
        if(b_pos == bsr_row_ptr[br + 1]){
            printf("Error: Block not found\n");
            exit(-1);
        }

        // bsrIndex[b_pos * bs2 + ir * bs + ic] = i;
        bsrIndex[i] = b_pos * bs2 + ir * bs + ic;
    }
}

void ComputeCsrIndex( PolyGrid *grid, IntType *oor, IntType *ooc, IntType *&csr_row_ptr, IntType *&csr_col_ind,
 IntType *csrIndex, IntType MatrixN, IntType nvar, IntType nnz ){

    IntType *oor_new = new IntType[nvar*nvar*nnz];

    for(int i=0;i<nvar*nvar*nnz;i++){
        oor_new[i] = oor[i];
        csr_col_ind[i] = ooc[csrIndex[i]];
    }

    csr_row_ptr[0] = 0;
    int count = 0;

    for(int i=0; i<MatrixN*nvar; i++){
        bool flag = true;
        while(flag){
            if(oor_new[count] == i){
                count++;
            }
            else{
                csr_row_ptr[i+1] = count;
                flag = false;
            }
        }
    }	
	delete[] oor_new;
}

IntType CalCOOInfoMPI( PolyGrid *grid, IntType *&oor, IntType *&ooc, \
    IntType *MatrixN, IntType *MatrixNNZ)
{
    IntType Bstart = 0;
    IntType nVar = 5;
    IntType blockSize = nVar*nVar;
    IntType nBFace = grid->GetNBFace();
    IntType nIFace = grid->GetNIFace();
    IntType nTCell = grid->GetNTCell();
    IntType *f2c   = grid->Getf2c();
    IntType *nFPC  = CalnFPC(grid);
    IntType **C2F  = CalC2F(grid); 
    IntType ifStart = nBFace - nIFace;

    IntType count = 0;
    MatrixN[0] = 0;
    MatrixNNZ[0] = 0;

int mpirank = 0;
#ifdef MPICH
    int size = 0;      
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    IntType *nIFaceCounter = new IntType[nIFace];
    IntType *GhostCell = new IntType[2*nTCell];
    //Initialize with diag cell
    for(IntType i=0;i<nIFace;i++){
        nIFaceCounter[i] = 1;
    }
    IntType ghostnnz = 0;
    for(IntType cell = 0; cell<nTCell; cell++){
        for(IntType iFace=0; iFace<nFPC[cell]; iFace++){
            IntType face = C2F[cell][iFace];
            IntType c2   = f2c[face+face]+f2c[face+face+1]-cell;
            if(face < ifStart){
                continue;
            }
            else if((face >= ifStart) && (face < nBFace)){ //ghost cells
                //Add the nnz of ghost cells
                IntType index = c2 - (nBFace - nIFace) - nTCell;
                nIFaceCounter[ index ]++;
                GhostCell[ghostnnz*2] = index + nTCell;
                GhostCell[ghostnnz*2 + 1] = cell;
                ghostnnz++;
            }
            count ++;
        }
        count ++;
    }
    for(IntType i=0;i<nIFace;i++){
        count += nIFaceCounter[i];
    }

    // for(IntType iCell = 0; iCell<nTCell; iCell++){
    //     IntType cell = iCell;
    
    //     for(IntType iFace=0; iFace<nFPC[cell]; iFace++){
    //         IntType face  = C2F[cell][iFace];
    //         if(face < ifStart)
    //         {
    //             continue;
    //         }
    //         count ++;
    //         // if(face >= nBFace){ //only the border face are counted as nnz
    //         //     count++; //seclection 2
    //         // }
    //     }
    //     /// this cell
    //     count ++;
    // }

    MatrixNNZ[0] = count;
    MatrixN[0]   = nTCell + nIFace;

    mfmem::snew_array_1D(oor, blockSize*count, dmrfl);
    mfmem::snew_array_1D(ooc, blockSize*count, dmrfl);
   
    IntType * ghost2global = grid->CalGhost2Global(Bstart);
    count = 0;
    for(int iCell = 0;iCell<nTCell;iCell++){
        IntType cell = iCell;
        for(IntType iFace=0; iFace<nFPC[cell]; iFace++){
            IntType face  = C2F[cell][iFace];
            IntType c2    = f2c[face+face]+f2c[face+face+1]-cell;
  
            IntType Brow = Bstart+cell;
            IntType Bcol = 0;
            if(face >= nBFace){
                Bcol = Bstart+c2;
            }
            else if(face < ifStart){
                continue;
            }
            else{
                //Renumber the ghost cells with start of nTCell
                Bcol = Bstart + c2 - (nBFace - nIFace);

                //Bcol = ghost2global[face-ifStart];
            }

            IntType Irow = Brow*nVar;
            IntType Icol = Bcol*nVar;
            for(int m = 0; m < nVar; m++){
                for(int n = 0; n < nVar; n++){
                    oor[count] = Irow + m;
                    ooc[count++] = Icol + n;
                }
            }
        }
        IntType Brow = Bstart+cell;
        IntType Bcol = Brow;
        IntType Irow = Brow*nVar;
        IntType Icol = Bcol*nVar;
        for(int m = 0; m < nVar; m++){
            for(int n = 0; n < nVar; n++){
                oor[count] = Irow + m;
                ooc[count++] = Icol + n;
            }
        }
    }

    for(IntType ind=0; ind<nIFace; ind++){
        for(IntType i=0; i<ghostnnz; i++){
            IntType Brow = Bstart + GhostCell[i*2];
            IntType Bcol = GhostCell[i*2+1];
            if( (nTCell+ind) == GhostCell[i*2] ){
                IntType Irow = Brow*nVar;
                IntType Icol = Bcol*nVar;
                for(int m = 0; m < nVar; m++){
                    for(int n = 0; n < nVar; n++){
                        oor[count] = Irow + m;
                        ooc[count++] = Icol + n;
                    }
                }
            }
            else{
                continue;
            }
        }
        IntType Brow = Bstart + nTCell + ind;
        IntType Bcol = Brow;
        IntType Irow = Brow*nVar;
        IntType Icol = Bcol*nVar;
        for(int m = 0; m < nVar; m++){
            for(int n = 0; n < nVar; n++){
                oor[count] = Irow + m;
                ooc[count++] = Icol + n;
            }
        }
    }

    grid->UpdateDataPtr(oor, INT, count, "oor");
    grid->UpdateDataPtr(ooc, INT, count, "ooc");

    delete[] nIFaceCounter;
    delete[] GhostCell;
    return count;

}

/*
void printMTX( PolyGrid *grid, RealFlow *&matrix, IntType nvar, IntType mtxNum){

    IntType  iter_done;// current iterate steps
    grid->GetData(&iter_done, INT, 1 ,"iter_done");
    int iter_output[mtxNum] = {10, 60, 110, 160, 210};
    for(int i=0; i<mtxNum; i++){
        if(iter_done == iter_output[i]){break;}
        if(i == mtxNum-1) {return;}
    }

    //get the matrix information 
    IntType* matrixInfo = (IntType *)grid->GetDataPtr(INT, 16, "matrixInfo");
    IntType  bsrnnz = matrixInfo[1];
    IntType  bsrn = matrixInfo[0];
    IntType *bcol = (IntType *)(grid->GetDataPtr(INT, bsrnnz, "bsr_col_ind"));
    IntType *brow = (IntType *)(grid->GetDataPtr(INT, bsrn+1, "bsr_row_ptr"));

    //get b
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, bsrn*nvar, "res");
    RealFlow *rhs[nvar];
    RealFlow *b = NULL;
    mfmem::snew_array_1D(b, bsrn*nvar,dmrfl);
    assert(b != 0); 
    rhs[0] = res;
    for(IntType iVar=1; iVar<nvar; iVar++) rhs[iVar] = &rhs[iVar-1][matrixInfo[0]];
    IntType idx = 0;
    for(IntType iCell = 0; iCell < matrixInfo[0]; iCell++){
        for(IntType iVar = 0; iVar < nvar; iVar++){
            b[idx++] = rhs[iVar][iCell];
        }
    }

    int retcodeM = write_mtx( grid, bsrn, bsrn, bsrnnz, nvar, brow, bcol, matrix, b);
    if(retcodeM != 0){
        printf("Errors occur in matrix mtx file output\n");
        exit(0);
    }
    // int *bptr, *bco;
    // RealFlow *val, *bb;
    // IntType mm, nn, nnzz, blocksize;
    // bool ISBSR = true;

    // read_from_mtx( &mm, &nn, &nnzz, &blocksize, &bptr, &bco, &val, &bb, ISBSR );
    // #ifdef FS_OPENMP
    // #pragma omp parallel for
    // #endif 
    //     for(int i=0; i<bsrnnz*nvar*nvar; i++){
    //         matrix[i] = val[i];
    //     }
    // #ifdef FS_OPENMP
    // #pragma omp parallel for
    // #endif 
    //     for(int i=0; i<bsrn+1; i++){
    //         brow[i] = bptr[i];
    //     }
    // #ifdef FS_OPENMP
    // #pragma omp parallel for
    // #endif 
    //     for(int i=0; i<bsrnnz; i++){
    //         bcol[i] = bco[i];
    //     }

    //     free(bptr);
    //     free(bco);
    //     free(val);
    //     free(bb);
    
}
*/
void InitialMatrix( PolyGrid *grid, IntType level ){

    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nIFace = grid->GetNIFace();
    IntType nTFace = grid->GetNTFace();
    IntType nvar = NVAR;

    IntType hasCopyConstData2Device = 0;
    grid->GetData(&hasCopyConstData2Device,   INT, 1, "hasCopyConstData2Device");

    IntType *matrixInfo=NULL;
    MATRIXTYPE *matrix=NULL;
    IntType *bsr_row_ptr=NULL;
    IntType *bsr_col_ind=NULL;
    IntType *bsr_dia_ptr=NULL;
    IntType *bsrIndex=NULL;
    GETYPE *diagmatrix = NULL;
    IntType *oor = NULL;
    IntType *ooc = NULL;
    IntType MatrixN, nnz;
    IntType Bstart = 0;
    IntType *posptr=NULL, *lpos=NULL, *upos=NULL;

    int mpirank = 0;
    #ifdef MPICH
        int size = 0;      
        MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    #endif

    if(!hasCopyConstData2Device){
        CalCOOInfoMPI( grid, oor, ooc, &MatrixN, &nnz );

        mfmem::snew_array_1D(matrixInfo, 16, dmrfl);
        mfmem::snew_array_1D(matrix, nnz*nvar*nvar, dmrfl);
        //mfmem::snew_array_1D(diagmatrix, nTCell*nvar*nvar, dmrfl);
        mfmem::snew_array_1D(bsr_row_ptr, MatrixN+1, dmrfl);
        mfmem::snew_array_1D(bsr_col_ind, nnz, dmrfl);
        mfmem::snew_array_1D(bsr_dia_ptr, MatrixN, dmrfl);
        mfmem::snew_array_1D(bsrIndex, nnz*nvar*nvar, dmrfl);

        IntType *perm=NULL, *inv_perm=NULL, *fusedRowPtr=NULL, *fusedColInd=NULL;
        mfmem::snew_array_1D(perm, MatrixN*2, dmrfl);
        mfmem::snew_array_1D(inv_perm, MatrixN*2, dmrfl);
        mfmem::snew_array_1D(fusedRowPtr, (MatrixN*2+1), dmrfl);
        mfmem::snew_array_1D(fusedColInd, nnz, dmrfl);

        matrixInfo[0] = MatrixN;
        matrixInfo[1] = nnz;

        //Compute the row_ptr and col_ind for BSR format
        ComputeBsrIndex( grid, bsr_row_ptr, bsr_col_ind, bsr_dia_ptr, nnz, MatrixN );
        //Compute the index array to convert the original matrix order to bsr & csr format
        ReorderedIndex( grid, oor, ooc, bsr_row_ptr, bsr_col_ind, bsrIndex, nnz, MatrixN );

        grid->UpdateDataPtr(matrixInfo, INT, 16, "matrixInfo");
        grid->UpdateDataPtr(matrix, MATRIXINTTYPE, nnz*nvar*nvar, "matrix");
        //grid->UpdateDataPtr(diagmatrix, MATRIXINTTYPE, nTCell*nvar*nvar, "diagmatrix");
        grid->UpdateDataPtr(bsr_row_ptr, INT, MatrixN+1, "bsr_row_ptr");
        grid->UpdateDataPtr(bsr_col_ind, INT, nnz, "bsr_col_ind");
        grid->UpdateDataPtr(bsr_dia_ptr, INT, MatrixN, "bsr_dia_ptr");
        grid->UpdateDataPtr(bsrIndex, INT, nnz*nvar*nvar, "bsrIndex");


        // ------- HBM allocation in ARM by using kupel ----------//
        // 获取当前的HBW内存分配策略，并将其设置为KUPL_HBW_POLICY_BIND模式 
        const int kspan = 30;
        if (kupl_hbw_check_available() == 0) {
            printf("HBW memory undetected, skipping this example\n");
        }
        else{
            //printf("Current HBW policy is %d.\n", kupl_hbw_get_policy());
            kupl_hbw_set_policy(KUPL_HBW_POLICY_BIND);
            diagmatrix_global = (GETYPE *) kupl_hbw_malloc(nTCell*nvar*nvar*sizeof(GETYPE)); 
            x_global = (MATRIXTYPE *) kupl_hbw_malloc(nvar*MatrixN*(kspan+1)*sizeof(MATRIXTYPE));
            b_global = (MATRIXTYPE *) kupl_hbw_malloc(nvar*MatrixN*(kspan+1)*sizeof(MATRIXTYPE));
        
            //printf("%p %p %p", diagmatrix, x_global, b_global);
            int result0 = kupl_hbw_verify(diagmatrix_global, nTCell*nvar*nvar*sizeof(GETYPE), KUPL_HBW_TOUCH_PAGES);
            int result1 = kupl_hbw_verify(x_global, nTCell*nvar*nvar*sizeof(MATRIXTYPE), KUPL_HBW_TOUCH_PAGES);
            int result2 = kupl_hbw_verify(b_global, nTCell*nvar*nvar*sizeof(MATRIXTYPE), KUPL_HBW_TOUCH_PAGES);
            //printf("local test verify result is %d %d %d\n", result0 ,result1, result2);
            diagmatrix = diagmatrix_global;
        }

        // computeILUPositions(bsr_row_ptr, bsr_col_ind, bsr_dia_ptr, MatrixN, matrixInfo, posptr, lpos, upos);
        // grid->UpdateDataPtr(posptr, INT, matrixInfo[2], "posptr");
        // grid->UpdateDataPtr(lpos, INT, matrixInfo[3], "lpos");
        // grid->UpdateDataPtr(upos, INT, matrixInfo[4], "upos");

        hasCopyConstData2Device = 1;
        grid->UpdateData(&hasCopyConstData2Device, INT, 1, "hasCopyConstData2Device");
        if(!mpirank)
            printf("rank:%d nTCell:%d nTFace:%d nBFace:%d nIFace:%d n:%d nnz:%d nvar:%d\n",mpirank,nTCell,nTFace,nBFace,nIFace,MatrixN,nnz,nvar);

        #ifdef MPICH    
            init_hierarchy(MPI_COMM_WORLD);
        #endif
        //fusedReorderTrans( grid, bsr_row_ptr, bsr_col_ind, bsr_dia_ptr, \
            fusedRowPtr, fusedColInd, perm, inv_perm, MatrixN, nnz, nvar );
        //grid->UpdateDataPtr(perm, INT, MatrixN*2, "perm");
        //grid->UpdateDataPtr(inv_perm, INT, MatrixN*2, "inv_perm");
        //grid->UpdateDataPtr(fusedRowPtr, INT, (MatrixN*2+1), "fusedRowPtr");
        //grid->UpdateDataPtr(fusedColInd, INT,nnz, "fusedColInd");

        // for(int i=0; i<MatrixN; i++){
        //     printf("i:%d %d %d: ",i,bsr_row_ptr[i],bsr_row_ptr[i+1]);
        //     for(int j=bsr_row_ptr[i]; j<bsr_row_ptr[i+1]; j++){
        //         printf("%d %.4f ",bsr_col_ind[j],matrix[j*nvar*nvar+1]);
        //     }
        //     printf("\n");
        // }
    }
    else{
        matrixInfo = (IntType *)grid->GetDataPtr(INT, 16, "matrixInfo"); 
        nnz = matrixInfo[1];

        matrix = (MATRIXTYPE *)grid->GetDataPtr(MATRIXINTTYPE, nnz*nvar*nvar, "matrix");

        //diagmatrix = (GETYPE *)grid->GetDataPtr(MATRIXINTTYPE, nTCell*nvar*nvar, "diagmatrix");
        diagmatrix = diagmatrix_global;
        
        bsrIndex = (IntType *)grid->GetDataPtr(INT, nnz*nvar*nvar, "bsrIndex");
    }

    //Fill the matrix
    SetupMatrix( grid, level, matrix, diagmatrix, bsrIndex );

    //print the mtx files
    // IntType isMtx = 1;
    // grid->GetData(&isMtx,   INT, 1, "isMtx");
    // if(isMtx){
    //     const int MTXFILENUM = 5;
    //     printMTX( grid, matrix, nvar, MTXFILENUM);
    // }
}



void GMRESSolver( PolyGrid *grid, IntType level ){

    timeval t0, t1;
    gettimeofday(&t0, NULL);
    double time_t0 = (double)t0.tv_sec + (double)t0.tv_usec/1000000;

    //Initial compressed storage index and matrix
    InitialMatrix(  grid, level );

    gettimeofday(&t1, NULL);
    double time_t1 = (double)t1.tv_sec + (double)t1.tv_usec/1000000;
    Matrixbuild += (time_t1 - time_t0);

    // timeval t2, t3;
    // gettimeofday(&t2, NULL);
    // double time_t2 = (double)t2.tv_sec + (double)t2.tv_usec/1000000;

    MPI_Barrier(MPI_COMM_WORLD);
    double time_t4 = MPI_Wtime();

    IntType fusedMethodFlag = 2;
    grid->GetData(&fusedMethodFlag,   INT, 1, "fusedImplicitMethod");

#ifdef ROOFLINE_EVENTS
    ROOFLINE_EVENTS_START_REGION("NSSolver LUSGS");
#endif

    if(fusedMethodFlag <= 2){
        //LU-SGS iteration
        MatrixLUSGS(grid, level);
    }
    else if(fusedMethodFlag == 3){
        //GMRES + LUSGS precondition
        GmresLUSGS(  grid, level );
    }
    else{
        //GMRES + ILU precondition
        GmresILU(  grid, level );
    }
#ifdef ROOFLINE_EVENTS
    ROOFLINE_EVENTS_STOP_REGION("NSSolver LUSGS");
#endif    
    double time_t5 = MPI_Wtime();
    GMRESexe += time_t5 - time_t4;

    // gettimeofday(&t3, NULL);
    // double time_t3 = (double)t3.tv_sec + (double)t3.tv_usec/1000000;
    // GMRESexe += (time_t3 - time_t2);
}

}