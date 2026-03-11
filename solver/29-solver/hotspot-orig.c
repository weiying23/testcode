
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cassert>
#include <string>
#include <iostream>
#include <algorithm>
#include <arm_sve.h>
#include "omp.h"

#define MATRIXTYPE float
#define IntType int

void forward_compute( const IntType *row_ptr, const IntType *col_ind, const IntType *dia_ptr, 
    const MATRIXTYPE *matrix, const MATRIXTYPE *vec, MATRIXTYPE *&prod, IntType n, IntType nvar ){
    
    IntType start = 0;
    IntType end = n;
    const int NNUMBER=5;
    bool *isReady = (bool *)malloc(n * sizeof(bool));
    memset(isReady, false, n * sizeof(bool));

    #pragma omp parallel for schedule(dynamic) shared(isReady)
    for (IntType iPoint = start; iPoint < end; iPoint++) {
        int thread = 0;
        thread = omp_get_thread_num();

        IntType idx = iPoint*nvar;
        IntType iVar, jVar, kVar, col_j;
        MATRIXTYPE low_prod[NNUMBER], block[NNUMBER*NNUMBER], weight;
        for ( iVar = 0ul; iVar < nvar; iVar++)
            low_prod[iVar] = 0.0;
        for ( iVar = row_ptr[iPoint]; iVar < dia_ptr[iPoint]; iVar++) {
            col_j = col_ind[iVar];
        
            while( !isReady[col_j] ){
                #pragma omp flush(isReady)
            }
            #pragma omp flush(prod)

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

        isReady[iPoint] = true;
    }
    free(isReady);
}

void backward_compute( IntType *row_ptr, IntType *col_ind, IntType *dia_ptr, MATRIXTYPE *matrix, 
    MATRIXTYPE *vec, MATRIXTYPE *&prod, IntType n, IntType nTCell, IntType nvar ){

    IntType begin = 0;
    IntType end = n;
    bool *isReady = (bool *)malloc(nTCell * sizeof(bool));
    memset(isReady, false, nTCell * sizeof(bool));
    const int NNUMBER=5;

   #pragma omp parallel for schedule(dynamic) 
   //shared(isReady)
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

        for ( iVar = 0ul; iVar < nvar; iVar++)
            up_prod[iVar] = 0.0;
        for ( iVar = dia_ptr[iPoint]+1; iVar < row_ptr[iPoint+1]; iVar++) {
            col_j = col_ind[iVar];

            while( !isReady[col_j] ){
                #pragma omp flush
            }
            #pragma omp flush(prod)

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

        isReady[iPoint] = true;
    }
    free(isReady);
}