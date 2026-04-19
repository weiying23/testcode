#ifdef FS_CUSPARSE
#include "cuFillMatCOO.cuh"
#include "ILUcuSparse.cuh"
#include "grid_polyhedra.h"
#include "number_type.h"
#include <iostream>
#include <algorithm>
#include "cusparse_v2.h"
#include "grid_polyhedra.h"
#include "cuData.cuh"
#include "cuGMRES.cuh"
#include "gmres_ilu.h"
#include <ctime>
#include <cassert>

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif

using namespace mflow;
using namespace gpuData;

extern double ILUbuild, ILUexe;
extern int ite;

__global__ void minResB(RealFlow *b, RealFlow *res, IntType size){
    int iPoint = blockDim.x*blockIdx.x + threadIdx.x;
	if(iPoint < size){
        b[iPoint] -= res[iPoint];
    }
}

__global__ void dataMov(RealFlow *dest, RealFlow *src, IntType length){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < length) {
        dest[i] = src[i];
    }
}
__device__ void gpuCalConvectiveFluxJacobian1(RealFlow *Matrix, RealFlow nx, RealFlow ny, RealFlow nz,
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

    Matrix[0] = 0.0;
    Matrix[1] = nx;
    Matrix[2] = ny;
    Matrix[3] = nz;
    Matrix[4] = 0.0;

    Matrix[5] = nx*phi - u*Vn;
    Matrix[6] = Vn - a3*nx*u;
    Matrix[7] = ny*u - a2*nx*v;
    Matrix[8] = nz*u - a2*nx*w;
    Matrix[9] = a2*nx;

    Matrix[10] = ny*phi - v*Vn;
    Matrix[11] = nx*v - a2*ny*u;
    Matrix[12] = Vn - a3*ny*v;
    Matrix[13] = nz*v - a2*ny*w;
    Matrix[14] = a2*ny;

    Matrix[15] = nz*phi - w*Vn;
    Matrix[16] = nx*w - a2*nz*u;
    Matrix[17] = ny*w - a2*nz*v;
    Matrix[18] = Vn - a3*nz*w;
    Matrix[19] = a2*nz;

    Matrix[20] = Vn*(phi-a1);
    Matrix[21] = nx*a1 - a2*u*Vn;
    Matrix[22] = ny*a1 - a2*v*Vn;
    Matrix[23] = nz*a1 - a2*w*Vn;
    Matrix[24] = gam*Vn;
}
__device__ RealFlow gpu_max1(RealFlow a, RealFlow b)
{
	return a > b ? a:b;
}

__device__ void gpuCalJacobian_ConvectiveFlux_Roe1(RealFlow *matrix, RealFlow q_L[5], RealFlow q_R[5], RealFlow nx, RealFlow ny, RealFlow nz, RealFlow gam, RealFlow alf_l)
{
    RealFlow rho = sqrt(q_R[0]*q_L[0]);
    RealFlow tmp0 = rho/q_L[0];
    RealFlow tmp1 = 1.0/(1.0 + tmp0);

    RealFlow u  = (q_L[1] + q_R[1]*tmp0)*tmp1;
    RealFlow v  = (q_L[2] + q_R[2]*tmp0)*tmp1;
    RealFlow w  = (q_L[3] + q_R[3]*tmp0)*tmp1;
    RealFlow qn = u*nx + v*ny + w*nz;

    RealFlow gamm1 = gam - 1.0;
    RealFlow e_L  = q_L[4]/gamm1 + 0.5*q_L[0]*(q_L[1]*q_L[1] + q_L[2]*q_L[2] + q_L[3]*q_L[3]);
    RealFlow e_R  = q_R[4]/gamm1 + 0.5*q_R[0]*(q_R[1]*q_R[1] + q_R[2]*q_R[2] + q_R[3]*q_R[3]);

    RealFlow h_L  = e_L + q_L[4];
    RealFlow h_R  = e_R + q_R[4];

    RealFlow h = (h_L/q_L[0] + h_R/q_R[0]*tmp0)*tmp1;

    RealFlow q2 = 0.5*(u*u + v*v + w*w);
    RealFlow c2 = gamm1*(h - q2);
    c2 = fabs(c2);
    RealFlow c = sqrt(c2);
    RealFlow lamda_0, lamda_p, lamda_m;

    //if(steady){
    lamda_0 = fabs(qn);
    lamda_p = fabs(qn + c);
    lamda_m = fabs(qn - c);
    //}else{   //unsteady
        //lamda0 = fabs(qn - vgn[ns+i]);
        //lamdap = fabs(qn - vgn[ns+i] + c);
        //lamdan = fabs(qn - vgn[ns+i] - c);
    //}

    RealFlow epsa_r;
    //if(EntropyCorType == 4){
    //    if(IsNormalFace[ns+i] && IsShockFace[i]==0){
    //                epsa_r = 0.01*alf_l;
    //                //epsa_r = 0.0002;
    //            }else{
    epsa_r = alf_l;
    //            }
    //}
    
    //cfl3d form
    //if(steady){
    RealFlow spectral = abs(u)+abs(v)+abs(w) + c; //lamda_0;
    //}else{
    //    RealFlow u_vgn,v_vgn,w_vgn;
    //    u_vgn = vgn[ns+i]*xfn[i];
    //    v_vgn = vgn[ns+i]*yfn[i];
    //    w_vgn = vgn[ns+i]*zfn[i];
    //    spectral = fabs(u_a-u_vgn)+fabs(v_a-v_vgn)+fabs(w_a-w_vgn)+c_a;
    //}

    RealFlow epsaa = epsa_r*spectral;
	
    RealFlow epsbb = 0.25/gpu_max1(epsaa,TINY);

    epsaa = fabs(qn) + c; epsaa *= 0.2;
    lamda_0 += epsaa;
    lamda_p += epsaa;
    lamda_m += epsaa;
    
    {
        RealFlow lamda_pm2t = 0.5 * (lamda_p - lamda_m) / c;
        RealFlow lamda_pm02 = 0.5 * (lamda_p + lamda_m) - lamda_0;
        RealFlow lamda_pm02t = lamda_pm02 * gamm1/c2;

        RealFlow t0 = -lamda_pm2t * qn + lamda_pm02t * q2;
        RealFlow t1 =  lamda_pm2t * nx - lamda_pm02t * u;
        RealFlow t2 =  lamda_pm2t * ny - lamda_pm02t * v;
        RealFlow t3 =  lamda_pm2t * nz - lamda_pm02t * w;
        RealFlow t4 =                    lamda_pm02t;

        lamda_pm2t *= gamm1;
        RealFlow s0 =  lamda_pm2t * q2 - lamda_pm02 * qn;
        RealFlow s1 = -lamda_pm2t * u  + lamda_pm02 * nx;
        RealFlow s2 = -lamda_pm2t * v  + lamda_pm02 * ny;
        RealFlow s3 = -lamda_pm2t * w  + lamda_pm02 * nz;
        RealFlow s4 =  lamda_pm2t;

        matrix[0] = t0 + lamda_0;
        matrix[1] = t1;
        matrix[2] = t2;
        matrix[3] = t3;
        matrix[4] = t4;

        matrix[5] = u * t0 + nx * s0;
        matrix[6] = u * t1 + nx * s1 + lamda_0;
        matrix[7] = u * t2 + nx * s2;
        matrix[8] = u * t3 + nx * s3;
        matrix[9] = u * t4 + nx * s4;

        matrix[10] = v * t0 + ny * s0;
        matrix[11] = v * t1 + ny * s1;
        matrix[12] = v * t2 + ny * s2 + lamda_0;
        matrix[13] = v * t3 + ny * s3;
        matrix[14] = v * t4 + ny * s4;

        matrix[15] = w * t0 + nz * s0;
        matrix[16] = w * t1 + nz * s1;
        matrix[17] = w * t2 + nz * s2;
        matrix[18] = w * t3 + nz * s3 + lamda_0;
        matrix[19] = w * t4 + nz * s4;

        matrix[20] = h * t0 + qn * s0;
        matrix[21] = h * t1 + qn * s1;
        matrix[22] = h * t2 + qn * s2;
        matrix[23] = h * t3 + qn * s3;
        matrix[24] = h * t4 + qn * s4 + lamda_0;
    }
}

__global__ void MatrixFillBsr(RealFlow *val, IntType *gbsr_row_ptr, IntType gnvar, IntType gnTCell, IntType  gvis_run, IntType gifStart, RealFlow ggam, RealFlow gp_bar, RealFlow galf_l, IntType  *gf2c, IntType  *gnFPC, IntType  *gC2F, 
IntType *gIndexC2F, RealGeom *gxfn, RealGeom *gyfn, RealGeom *gzfn, RealGeom *garea, RealGeom *gvol, RealFlow *gdt, RealGeom *gnorm_dist_c2c, RealFlow *gvis_l, RealFlow *gvis_t, RealFlow *gq0,RealFlow *gq1,RealFlow *gq2,RealFlow *gq3,RealFlow *gq4, RealFlow * gmatrix_jacobi_d, RealFlow * gmatrix_jacobi_fc, RealFlow * gmatrix_temp)
{
	IntType cell = blockDim.x*blockIdx.x + threadIdx.x;
	if(cell < gnTCell)
	{
		RealFlow *vStart = &val[gbsr_row_ptr[cell]*gnvar*gnvar];
        RealFlow * matrix_jacobi_d = &gmatrix_jacobi_d[25*cell];
        RealFlow * matrix_jacobi_fc = &gmatrix_jacobi_fc[25*cell];
        RealFlow * matrix_temp = &gmatrix_temp[25*cell];
		RealFlow q_r[5];
		RealFlow q_l[5];
		RealFlow face_n[3];
		
		IntType count = 0;
		for(IntType  m=0;m<5;m++){
			for(IntType n=0;n<5;n++){
				if(m==n){
					matrix_jacobi_d[m*5+n] = gvol[cell]/gdt[cell];;
				}
				else{
					matrix_jacobi_d[m*5+n] = 0.0;
				}
			}
		}
		for(IntType iFace=0; iFace<gnFPC[cell]; iFace++){
			IntType cellC2FIndex = gIndexC2F[cell];
			IntType face  = gC2F[cellC2FIndex + iFace];
			IntType c1    = gf2c[face+face];
			IntType c2    = gf2c[face+face+1];
			face_n[0] = gxfn[face];
			face_n[1] = gyfn[face];
			face_n[2] = gzfn[face];
			if(c2 == cell){
				IntType c_tmp = c1;
				c1    = c2;
				c2    = c_tmp;
				face_n[0] = -face_n[0];
				face_n[1] = -face_n[1];
				face_n[2] = -face_n[2];
			}
			q_r[0]  = gq0[c2];
			q_r[1]  = gq1[c2];
			q_r[2]  = gq2[c2];
			q_r[3]  = gq3[c2];
			q_r[4]  = gq4[c2]+gp_bar;
			//Jacobian of convective flux
			gpuCalConvectiveFluxJacobian1(matrix_jacobi_fc, face_n[0], face_n[1], face_n[2], q_r[0], q_r[1], q_r[2], q_r[3], q_r[4], ggam);

			// Roe matrix: 
			q_l[0]  = gq0[c1];
			q_l[1]  = gq1[c1];
			q_l[2]  = gq2[c1];
			q_l[3]  = gq3[c1];
			q_l[4]  = gq4[c1]+gp_bar;
			RealFlow tmparea = 0.5*garea[face];
			gpuCalJacobian_ConvectiveFlux_Roe1(matrix_temp, q_l, q_r, face_n[0], face_n[1], face_n[2], ggam, galf_l); 
			for(int m = 0; m < 5; m++)
			{
				for(int n = 0; n < 5; n++)
				{
					matrix_jacobi_d[m*5+n]  += matrix_temp[m*5+n] * tmparea; 
					matrix_jacobi_fc[m*5+n] -= matrix_temp[m*5+n];
				}
			}

			//viscous term for off-diag element 
			RealFlow visc_c1 = 0.0;
			RealFlow visc_c2 = 0.0;
			RealFlow eig_v_c1 = 0.0;
			RealFlow eig_v_c2 = 0.0;
			if(gvis_run){
				// Eigenvalues of viscous flux
				RealFlow dist = gnorm_dist_c2c[face];
				visc_c1 = gvis_l[c1];
				visc_c2 = gvis_l[c2];
				if(gvis_t)
				{
					visc_c1 += gvis_t[c1];
					visc_c2 += gvis_t[c2];
				}
				eig_v_c1 = 2.0*visc_c1/(q_l[0]*dist + 1.e-40);
				eig_v_c2 = 2.0*visc_c2/(q_r[0]*dist + 1.e-40); 
			}
			if(gvis_run)
			{
				for(int k=0;k<5;k++)
				{
					matrix_jacobi_d[k*5+k]  += eig_v_c1*tmparea;  
					matrix_jacobi_fc[k*5+k] -= eig_v_c2;
				}
			}
			/// 在这里统一乘，因为最后开始算的时候没乘
			for(int k=0;k<5;k++){
				for(int l=0;l<5;l++){
					matrix_jacobi_fc[k*5+l] *= tmparea;
				}
			}
			if(face >= gifStart)
			{
			    for(int k=0;k<5;k++){
				    for(int l=0;l<5;l++){
					    vStart[count++] = matrix_jacobi_fc[k*5+l];
						//vStart[count++] = 1.0;
				    }
			    }
			}
		}
		
		for(int k=0;k<5;k++){
			for(int l=0;l<5;l++){
				vStart[count++] = matrix_jacobi_d[k*5+l];
				//vStart[count++] = 1.0;
			}
		}
        
        //delete [] matrix_jacobi_d;
        //delete [] matrix_jacobi_fc;
        //delete [] matrix_temp;
	}
}

__global__ void gGhost(IntType start, IntType end, RealFlow *matrix){
    int i = start + blockDim.x * blockIdx.x + threadIdx.x;
    if(i < end){
        matrix[i] = (i - start + 1) * 1.0;
    }
}

__global__ void gMatrixConvert(RealFlow* Matrix, const RealFlow* oriMatrix, const IntType* indx, int length, int size){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < length){
        for(int j=0;j<size;j++)
            Matrix[indx[i]*size+j] = oriMatrix[i*size+j];
    }
}

void matrix_set( PolyGrid *grid ){

    IntType blocksPerGrid = ( gnnz*gnvar*gnvar + threadsPerBlock - 1 ) / threadsPerBlock;
    //gDataZeros<<< blocksPerGrid, threadsPerBlock >>>( gval, gnnz*gnvar*gnvar );
    CHECKCUDA(cudaMemset( gval, 0.0, gnnz*gnvar*gnvar * sizeof(RealFlow) ));

    blocksPerGrid = ( gnTCell + threadsPerBlock - 1 ) / threadsPerBlock;
    MatrixFillBsr<<< blocksPerGrid, threadsPerBlock >>>(gval, gBSRrow_ptr, gnvar, gnTCell, gvis_run, gifStart, ggam, gp_bar, galf_l, \
        gf2c, gnFPC, gC2F, gIndexC2F, gxfn, gyfn, gzfn, garea, gvol, gdt, gnorm_dist_c2c, gvis_l, \
        gvis_t, gq0, gq1, gq2, gq3, gq4, gmatrix_jacobi_d, gmatrix_jacobi_fc, gmatrix_temp);
        
    IntType *csrIndex = (IntType *)grid->GetDataPtr(INT, gnnz*gnvar*gnvar, "csrIndex");
    IntType *bsrIndex = (IntType *)grid->GetDataPtr(INT, gnnz*gnvar*gnvar, "bsrIndex");
#ifdef MPICH
    IntType *bsr_row_ptr = (IntType *)(grid->GetDataPtr(INT, gn+1, "bsr_row_ptr"));
    IntType start = bsr_row_ptr[gnTCell]*gnvar*gnvar;
    IntType end   = bsr_row_ptr[gn]*gnvar*gnvar;
    blocksPerGrid = ( (end-start) + threadsPerBlock - 1 ) / threadsPerBlock;
    gGhost<<< blocksPerGrid, threadsPerBlock >>>( start, end, gval);
#endif

    //blocksPerGrid = ( gnnz*gnvar*gnvar + threadsPerBlock - 1 ) / threadsPerBlock;
    //gMatrixConvert<<< blocksPerGrid, threadsPerBlock >>>(gILU_matrix, gval, gcol_ind_index, gnnz, gnvar*gnvar);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error occurred during MatrixFillBsr execution: %s\n", cudaGetErrorString(err));
    }

if(USE_BSR){

    RealFlow *matrix = new RealFlow[gnnz*gnvar*gnvar];
	CHECKCUDA(cudaMemcpy(matrix, gval, gnnz*gnvar*gnvar * sizeof(RealFlow), cudaMemcpyDeviceToHost));
    RealFlow *matrix2 = new RealFlow[gnnz*gnvar*gnvar];
    RealFlow *matrix3 = new RealFlow[gnnz*gnvar*gnvar];
	
    for(int i=0;i<gnvar*gnvar*gnnz;i++){
        matrix2[i] = matrix[csrIndex[i]];
    }
	
	for(int i=0;i<gnvar*gnvar*gnnz;i++){
        matrix3[i] = matrix2[bsrIndex[i]];
    }

    CHECKCUDA(cudaMemcpy(gILU_matrix, matrix3, gnnz*gnvar*gnvar * sizeof(RealFlow), cudaMemcpyHostToDevice));
    delete[] matrix;
    delete[] matrix2;
    delete[] matrix3;
    blocksPerGrid = ( gnnz*gnvar*gnvar + threadsPerBlock - 1 ) / threadsPerBlock;
    dataMov<<< blocksPerGrid, threadsPerBlock >>>(gval, gILU_matrix, gnnz*gnvar*gnvar);
}
else{
    RealFlow *matrix = new RealFlow[gnnz*gnvar*gnvar];
	CHECKCUDA(cudaMemcpy(matrix, gval, gnnz*gnvar*gnvar * sizeof(RealFlow), cudaMemcpyDeviceToHost));
    RealFlow *matrix2 = new RealFlow[gnnz*gnvar*gnvar];
    //IntType  size = gnvar*gnvar;
    
    for(int i=0;i<gnvar*gnvar*gnnz;i++){
        matrix2[i] = matrix[csrIndex[i]];
    }

    CHECKCUDA(cudaMemcpy(gILU_matrix, matrix2, gnnz*gnvar*gnvar * sizeof(RealFlow), cudaMemcpyHostToDevice));
    delete[] matrix;
    delete[] matrix2;
    blocksPerGrid = ( gnnz*gnvar*gnvar + threadsPerBlock - 1 ) / threadsPerBlock;
    dataMov<<< blocksPerGrid, threadsPerBlock >>>(gval, gILU_matrix, gnnz*gnvar*gnvar);

}
}

void cuSparseInitialbsr( ){

    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);  // CUSPARSE_INDEX_BASE_ZERO   CUSPARSE_INDEX_BASE_ONE
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT); //CUSPARSE_DIAG_TYPE_NON_UNIT  CUSPARSE_DIAG_TYPE_UNIT

    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create a empty info structure
    // we need one info for csrilu02 and two info's for csrsv2
    cusparseCreateBsrsv2Info(&info_L);
    cusparseCreateBsrsv2Info(&info_U);
    cusparseCreateBsrilu02Info(&info_M);

}
void cuSparseLUfactorizationBSR( ) {
    cusparseHandle_t handle = 0;
    cusparseStatus_t CUIFSUCCESS = cusparseCreate(&handle);
    int numerical_zero;

    void* pBuffer = 0;
    int pBufferSize_M = 0;

    // step 3: query how much memory used in bsrilu02 and bsrsv2, and allocate the buffer
    cusparseStatus_t status0 = cusparseDbsrilu02_bufferSize(handle, dirA, gn, gnnz, descr_M, \
        gILU_matrix, gBSRrow_ptr, gBSRcol_ind, gnvar, info_M, &pBufferSize_M);
    //printf("CUDA status during cuSparseLUfactorizationBSR decomposition bufferSize: %d with pBufferSize_M:%d\n", status0, pBufferSize_M);
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    CHECKCUDA(cudaMalloc((void**)&pBuffer, pBufferSize_M));
    
    int structural_zero;
    cusparseStatus_t status1 = cusparseDbsrilu02_analysis(handle, dirA, gn, gnnz, descr_M, \
        gILU_matrix, gBSRrow_ptr, gBSRcol_ind, gnvar, info_M, policy_ML, pBuffer);
    //printf("CUDA status during cuSparseLUfactorizationBSR decomposition analysis: %d \n", status1);

    cusparseStatus_t status2 = cusparseXbsrilu02_zeroPivot(handle, info_M, &structural_zero);
    //printf("CUDA status during cuSparseLUfactorizationBSR decomposition zeroPivot: %d \n", status2);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status2) {
        //printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    // step 5: M = L * U
    cusparseStatus_t status3 = cusparseDbsrilu02(handle, dirA, gn, gnnz, descr_M, \
        gILU_matrix, gBSRrow_ptr, gBSRcol_ind, gnvar, info_M, policy_ML, pBuffer);
    //printf("CUDA status during cuSparseLUfactorizationBSR decomposition execution: %d \n", status3);

    cusparseStatus_t status4 = cusparseXbsrilu02_zeroPivot(handle, info_M, &numerical_zero);
    //printf("CUDA status during cuSparseLUfactorizationBSR decomposition zeroPivot: %d \n", status4);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status4) {
        //printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
    }

    cudaFree(pBuffer);
    cusparseStatus_t cusparseDestroy(cusparseHandle_t handle);

}
void cuSparseILUBSR(  ) {
    const RealFlow alpha = 1.;
    const cusparseOperation_t trans_LU = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseHandle_t handle = 0;
    cusparseStatus_t CUIFSUCCESS = cusparseCreate(&handle);
    
    // step 6: solve L*z = x  //replace with cusparseSpSV
    void *pBuffer = 0;
    int pBufferSize_L = 0;
    int pBufferSize_U = 0;

    // step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
    cusparseStatus_t statusa = cusparseDbsrsv2_bufferSize(handle, dirA, trans_LU, gn, \
        gnnz, descr_L, gILU_matrix, gBSRrow_ptr, gBSRcol_ind, gnvar, info_L, &pBufferSize_L);
    //printf("CUDA status during cuSparseILUBSR solve buffersize L execution: %d \n", statusa);
    
    cusparseStatus_t statusb = cusparseDbsrsv2_bufferSize(handle, dirA, trans_LU, gn, \
        gnnz, descr_U, gILU_matrix, gBSRrow_ptr, gBSRcol_ind, gnvar, info_U, &pBufferSize_U);
    //printf("CUDA status during cuSparseILUBSR solve buffersize U execution: %d \n", statusb);

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    if (pBufferSize_L > pBufferSize_U) {
        CHECKCUDA(cudaMalloc((void**)&pBuffer, pBufferSize_L));
    }
    else{
        CHECKCUDA(cudaMalloc((void**)&pBuffer, pBufferSize_U));
    }

    cusparseStatus_t status0 = cusparseDbsrsv2_analysis(handle, dirA, trans_LU, gn, gnnz, descr_L, \
        gILU_matrix, gBSRrow_ptr, gBSRcol_ind, gnvar, info_L, policy_ML, pBuffer);
    //printf("CUDA status during cuSparseILUBSR solve analysis 1 execution: %d \n", status0);

    cusparseStatus_t status1 = cusparseDbsrsv2_analysis(handle, dirA, trans_LU, gn, gnnz, descr_U, \
        gILU_matrix, gBSRrow_ptr, gBSRcol_ind, gnvar, info_U, policy_U, pBuffer);
    //printf("CUDA status during cuSparseILUBSR solve analysis 2 execution: %d \n", status1);

    cusparseStatus_t status2 = cusparseDbsrsv2_solve(handle, dirA, trans_LU, gn, gnnz, &alpha, descr_L, \
        gILU_matrix, gBSRrow_ptr, gBSRcol_ind, gnvar, info_L, &gb[g_i*gn*gnvar], gaux_vec, policy_ML, pBuffer);
    //printf("CUDA status during cuSparseILUBSR solve 3 execution: %d \n", status2);

    // step 7: solve U*y = z
    cusparseStatus_t status3 = cusparseDbsrsv2_solve(handle, dirA, trans_LU, gn, gnnz, &alpha, descr_U, \
        gILU_matrix, gBSRrow_ptr, gBSRcol_ind, gnvar, info_U, gaux_vec, &gx[g_i*gn*gnvar], policy_U, pBuffer);
    //printf("CUDA status during cuSparseILUBSR solve 4 execution: %d \n", status3);

    cudaFree(pBuffer);
    cusparseStatus_t cusparseDestroy(cusparseHandle_t handle);
}

void fusedILU(){
    CHECKCUDA(cudaMemset( gIsReady, 0,   sizeof(int) * 2*gn ));

    const IntType warpnum = 16;
    IntType num_threads = 32*warpnum;
    IntType blocksPerGrid = ceil((double)(2*gn)/(double)warpnum);
    gpu_bsr_syncfree_fusedspTrsv<<<blocksPerGrid, num_threads>>>( \
        gBSRrow_ptr, gBSRcol_ind, gILU_matrix, gIsReady, \
        gn, gnvar*gn, &gb[g_i*gn*gnvar], &gx[g_i*gn*gnvar], gnvar); //gaux_vec is not used
    cudaDeviceSynchronize();
}

void syncfree_spTrsv(){
    CHECKCUDA(cudaMemset( gIsReady, 0,   sizeof(int) * gn ));

    const IntType warpnum = 16;
    IntType num_threads = 32*warpnum;
    IntType num_blocks = ceil((double)gn/(double)warpnum);
    bsr_syncfree_spTRsolveL_dim5<<<num_blocks,num_threads>>>( \
        gBSRrow_ptr, gBSRcol_ind, gILU_matrix, gIsReady, \
        gn, gnvar*gn, &gb[g_i*gn*gnvar], gaux_vec, gnvar);
    cudaDeviceSynchronize();

    CHECKCUDA(cudaMemset( gIsReady, 0,   sizeof(int) * gn ));
    bsr_syncfree_spTRsolveU_dim5<<<num_blocks,num_threads>>>( \
        gBSRrow_ptr, gBSRcol_ind, gILU_matrix, gIsReady, \
        gn, gnvar*gn, gaux_vec, &gx[g_i*gn*gnvar], gnvar);
    cudaDeviceSynchronize();
}

void cuSparseInitialcsr( ){

    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);  // CUSPARSE_INDEX_BASE_ZERO   CUSPARSE_INDEX_BASE_ONE
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT); //CUSPARSE_DIAG_TYPE_NON_UNIT  CUSPARSE_DIAG_TYPE_UNIT

    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create a empty info structure
    // we need one info for csrilu02 and two info's for csrsv2
    cusparseCreateCsrsv2Info(&infocsr_L);
    cusparseCreateCsrsv2Info(&infocsr_U);
    cusparseCreateCsrilu02Info(&infocsr_M);

}
void cuSparseLUfactorizationCSR( ) {
    cusparseHandle_t handle = 0;
    cusparseStatus_t CUIFSUCCESS = cusparseCreate(&handle);
    int numerical_zero;

    void* pBuffer = 0;
    int pBufferSize_M = 0;

    // step 3: query how much memory used in bsrilu02 and bsrsv2, and allocate the buffer
    cusparseStatus_t status0 = cusparseDcsrilu02_bufferSize(handle, gn*gnvar, gnnz*gnvar*gnvar, descr_M, \
        gILU_matrix, gCSRrow_ptr, gCSRcol_ind, infocsr_M, &pBufferSize_M);
    //printf("CUDA status during cuSparseILUCSR decomposition bufferSize: %d with pBufferSize_M:%d\n", status0, pBufferSize_M);
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    CHECKCUDA(cudaMalloc((void**)&pBuffer, pBufferSize_M));
    
    int structural_zero;
    cusparseStatus_t status1 = cusparseDcsrilu02_analysis(handle, gn*gnvar, gnnz*gnvar*gnvar, descr_M, \
        gILU_matrix, gCSRrow_ptr, gCSRcol_ind, infocsr_M, policy_ML, pBuffer);
    //printf("CUDA status during cuSparseILUCSR decomposition analysis: %d \n", status1);

    cusparseStatus_t status2 = cusparseXcsrilu02_zeroPivot(handle, infocsr_M, &structural_zero);
    //printf("CUDA status during cuSparseILUCSR decomposition zeroPivot: %d \n", status2);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status2) {
        //printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    // step 5: M = L * U
    cusparseStatus_t status3 = cusparseDcsrilu02(handle, gn*gnvar, gnnz*gnvar*gnvar, descr_M, \
        gILU_matrix, gCSRrow_ptr, gCSRcol_ind, infocsr_M, policy_ML, pBuffer);
    //printf("CUDA status during cuSparseILUCSR decomposition execution: %d \n", status3);

    cusparseStatus_t status4 = cusparseXcsrilu02_zeroPivot(handle, infocsr_M, &numerical_zero);
    //printf("CUDA status during cuSparseILUCSR decomposition zeroPivot: %d \n", status4);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status4) {
        //printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
    }

    cudaFree(pBuffer);
    cusparseStatus_t cusparseDestroy(cusparseHandle_t handle);
}

void cuSparseILUCSR(  ) {
    const RealFlow alpha = 1.;
    const cusparseOperation_t trans_LU = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseHandle_t handle = 0;
    cusparseStatus_t CUIFSUCCESS = cusparseCreate(&handle);

    // step 6: solve L*z = x  //replace with cusparseSpSV
    void *pBuffer = 0;
    int pBufferSize_L = 0;
    int pBufferSize_U = 0;

    // step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
    cusparseStatus_t statusa = cusparseDcsrsv2_bufferSize(handle, trans_LU, gn*gnvar, \
        gnnz*gnvar*gnvar, descr_L, gILU_matrix, gCSRrow_ptr, gCSRcol_ind, infocsr_L, &pBufferSize_L);
    //printf("CUDA status during cuSparseILUCSR solve buffersize L execution: %d \n", statusa);
    
    cusparseStatus_t statusb = cusparseDcsrsv2_bufferSize(handle, trans_LU, gn*gnvar, \
        gnnz*gnvar*gnvar, descr_U, gILU_matrix, gCSRrow_ptr, gCSRcol_ind, infocsr_U, &pBufferSize_U);
    //printf("CUDA status during cuSparseILUCSR solve buffersize U execution: %d \n", statusb);

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    if (pBufferSize_L > pBufferSize_U) {
        CHECKCUDA(cudaMalloc((void**)&pBuffer, pBufferSize_L));
    }
    else{
        CHECKCUDA(cudaMalloc((void**)&pBuffer, pBufferSize_U));
    }

    cusparseStatus_t status0 = cusparseDcsrsv2_analysis(handle, trans_LU, gn*gnvar, gnnz*gnvar*gnvar, descr_L, \
        gILU_matrix, gCSRrow_ptr, gCSRcol_ind, infocsr_L, policy_ML, pBuffer);
    //printf("CUDA status during cuSparseILUCSR solve analysis 1 execution: %d \n", status0);

    cusparseStatus_t status1 = cusparseDcsrsv2_analysis(handle, trans_LU, gn*gnvar, gnnz*gnvar*gnvar, descr_U, \
        gILU_matrix, gCSRrow_ptr, gCSRcol_ind, infocsr_U, policy_U, pBuffer);
    //printf("CUDA status during cuSparseILUCSR solve analysis 2 execution: %d \n", status1);

    cusparseStatus_t status2 = cusparseDcsrsv2_solve(handle, trans_LU, gn*gnvar, gnnz*gnvar*gnvar, &alpha, descr_L, \
        gILU_matrix, gCSRrow_ptr, gCSRcol_ind, infocsr_L, &gb[g_i*gn*gnvar], gaux_vec, policy_ML, pBuffer);
    //printf("CUDA status during cuSparseILUCSR solve 3 execution: %d \n", status2);

    // step 7: solve U*y = z
    cusparseStatus_t status3 = cusparseDcsrsv2_solve(handle, trans_LU, gn*gnvar, gnnz*gnvar*gnvar, &alpha, descr_U, \
        gILU_matrix, gCSRrow_ptr, gCSRcol_ind, infocsr_U, gaux_vec, &gx[g_i*gn*gnvar], policy_U, pBuffer);
    //printf("CUDA status during cuSparseILUCSR solve 4 execution: %d \n", status3);

    cudaFree(pBuffer);
    cusparseStatus_t cusparseDestroy(cusparseHandle_t handle);
}

void cuMatrixInitial( PolyGrid *grid, const IntType matrixN, const IntType nnz, const IntType nvar ){
    IntType *csr_col_ind = (IntType *)(grid->GetDataPtr(INT, nnz*nvar*nvar, "csr_col_ind"));
    IntType *csr_row_ptr = (IntType *)(grid->GetDataPtr(INT, matrixN*nvar+1, "csr_row_ptr"));
    IntType *bsr_col_ind = (IntType *)(grid->GetDataPtr(INT, nnz, "bsr_col_ind"));
    IntType *bsr_row_ptr = (IntType *)(grid->GetDataPtr(INT, matrixN+1, "bsr_row_ptr"));
    IntType *csrIndex = (IntType *)grid->GetDataPtr(INT, nnz*nvar*nvar, "csrIndex");
    IntType *bsrIndex = (IntType *)grid->GetDataPtr(INT, nnz*nvar*nvar, "bsrIndex");

    gnvar = nvar;
    grid->GetData(&gkspan, INT, 1, "kspan");
    gn = matrixN;
    gnnz = nnz;
    gnnz2 = bsr_row_ptr[gnTCell];
    IntType nTCell = grid->GetNTCell();
    gnTCell = nTCell;

    IntType nIFace = grid->GetNIFace();
    gnIFace = nIFace;
	
    CHECKCUDA(cudaMalloc((void**)&gILU_matrix, gnnz*gnvar*gnvar*sizeof(RealFlow)));
    CHECKCUDA(cudaMalloc((void**)&gval, gnnz*gnvar*gnvar*sizeof(RealFlow)));
    CHECKCUDA(cudaMalloc((void**)&gb, (gkspan+1)*gn*gnvar*sizeof(RealFlow)));
    CHECKCUDA(cudaMalloc((void**)&gx, (gkspan+1)*gn*gnvar*sizeof(RealFlow)));
    CHECKCUDA(cudaMalloc((void**)&gres, gnTCell*gnvar*sizeof(RealFlow)));
    CHECKCUDA(cudaMalloc((void**)&gx_final, gn*gnvar*sizeof(RealFlow)));
    CHECKCUDA(cudaMalloc((void**)&gaux_vec, gnvar*gn*gnvar*sizeof(RealFlow)));

    CHECKCUDA(cudaMalloc((void**)&gBSRrow_ptr, (gn+1)*sizeof(IntType)));
    CHECKCUDA(cudaMemcpy(gBSRrow_ptr, bsr_row_ptr, (gn+1)*sizeof(IntType), cudaMemcpyHostToDevice));

    CHECKCUDA(cudaMalloc((void**)&gBSRcol_ind, gnnz*sizeof(IntType)));
    CHECKCUDA(cudaMemcpy(gBSRcol_ind, bsr_col_ind, gnnz*sizeof(IntType), cudaMemcpyHostToDevice));

    CHECKCUDA(cudaMalloc((void**)&gCSRrow_ptr, (gn*gnvar+1)*sizeof(IntType)));
    CHECKCUDA(cudaMemcpy(gCSRrow_ptr, csr_row_ptr, (gn*gnvar+1)*sizeof(IntType), cudaMemcpyHostToDevice));

    CHECKCUDA(cudaMalloc((void**)&gCSRcol_ind, gnnz*gnvar*gnvar*sizeof(IntType))); 
    CHECKCUDA(cudaMemcpy(gCSRcol_ind, csr_col_ind, gnnz*gnvar*gnvar*sizeof(IntType), cudaMemcpyHostToDevice));
    //cudaMalloc((void**)&gcol_ind_index, gnnz*sizeof(IntType)); 
	
    CHECKCUDA(cudaMalloc((void**)&gbsrIndex, gnnz*gnvar*gnvar*sizeof(IntType))); 
    CHECKCUDA(cudaMemcpy(gbsrIndex, bsrIndex, gnnz*gnvar*gnvar*sizeof(IntType), cudaMemcpyHostToDevice));
    CHECKCUDA(cudaMalloc((void**)&gcsrIndex, gnnz*gnvar*gnvar*sizeof(IntType))); 
    CHECKCUDA(cudaMemcpy(gcsrIndex, csrIndex, gnnz*gnvar*gnvar*sizeof(IntType), cudaMemcpyHostToDevice));
	
    CHECKCUDA(cudaMallocManaged((void **)&val_Reduction, 10*sizeof(RealFlow)));
    gnsum2 = (gnvar*gnTCell + 2*threadsPerBlock - 1) / (2*threadsPerBlock);
    gnodata2 = gnsum2;
    gnsum2 *= 2*threadsPerBlock;
    CHECKCUDA(cudaMalloc((void **)&gsumv2, gnsum2*sizeof(RealFlow)));

    CHECKCUDA(cudaMalloc((void**)&gIsReady, 2*gn*sizeof(IntType))); 

    IntType blocksPerGrid = gnodata2;
    //gDataZeros <<< blocksPerGrid, threadsPerBlock >>> (gsumv2, gnsum2);
    CHECKCUDA(cudaMemset( gsumv2, 0.0, gnsum2 * sizeof(RealFlow) ));

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error occurred during cuMatrixInitial gDataZeros execution: %s\n", cudaGetErrorString(err));
    }

    CHECKCUDA(cudaMalloc((void **)&godata2, blocksPerGrid*sizeof(RealFlow)));
}

void bForcusparse( PolyGrid *grid, RealFlow *DQ[5] ){

    g_i = 0;

    IntType size = gnTCell * gnvar;
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, size, "res");
    RealFlow *rhs[gnvar];
    rhs[0] = res;
    for(IntType iVar=1; iVar<gnvar; iVar++) rhs[iVar] = &rhs[iVar-1][gnTCell];

    IntType idx = 0;
    RealFlow *vecbLocal = new RealFlow[size];
    for(IntType iCell = 0; iCell < gnTCell; iCell++){
        for(IntType iVar = 0; iVar < gnvar; iVar++){
            vecbLocal[idx++] = rhs[iVar][iCell];
        }
    }

    CHECKCUDA(cudaMemcpy(gres, vecbLocal, size * sizeof(RealFlow), cudaMemcpyHostToDevice));

    //int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    //gDataZeros<<<blocksPerGrid, threadsPerBlock >>>( gx_final, size );
    CHECKCUDA(cudaMemset( gx_final, 0.0, size * sizeof(RealFlow) ));

    delete[] vecbLocal;
}

void mat_vec( PolyGrid *grid){
    IntType size = gnTCell * gnvar;
    int blocksPerGrid;
    if(USE_BSR){
        blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
        gpuMatrixVectorProduct <<< blocksPerGrid, threadsPerBlock >>> ( 
            gval, gx_final, gb, gBSRrow_ptr, gBSRcol_ind, gnTCell, gnvar);
    }
    else{
        blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;	
        gpuMatrixVectorProductcsr <<< blocksPerGrid, threadsPerBlock >>>(
            gval, gx_final, gb, gCSRrow_ptr, gCSRcol_ind, gnTCell, gnvar);
    }
    cudaDeviceSynchronize();
    
#ifdef MPICH
    //communications for b
    RealFlow *b = new RealFlow[gn*gnvar];
    IntType nT = gnTCell *gnvar;
    CHECKCUDA(cudaMemcpy(b, gb, gn*gnvar*sizeof(RealFlow), cudaMemcpyDeviceToHost));

    RealFlow *MPItmp[gnvar];
    MPItmp[0] = new RealFlow[gnvar*nT];
    for(int i=1; i<gnvar; i++) MPItmp[i] = &MPItmp[i-1][nT];

    IntType index = 0;
    for(IntType iCell = 0; iCell < gnTCell; iCell++){
        for(IntType iVar = 0; iVar < gnvar; iVar++){
            MPItmp[iVar][iCell] = b[index++];
        }
    }

    grid->RecvSendVarNeighbor_Togeth( gnvar, MPItmp );

    index = gnTCell*gnvar;
    for(IntType iCell = gnTCell+gnBFace-gnIFace; iCell < gnTCell+gnBFace; iCell++){
        for(IntType iVar = 0; iVar < gnvar; iVar++){
            b[index++] = MPItmp[iVar][iCell];
        }
    }
    delete[] MPItmp[0];
    CHECKCUDA(cudaMemcpy(&gb[gnTCell*gnvar], &b[gnTCell*gnvar], \
        gnIFace*gnvar*sizeof(RealFlow), cudaMemcpyHostToDevice));

#endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error occurred during mat_vec execution: %s\n", cudaGetErrorString(err));
    }
    blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    minResB<<< blocksPerGrid, threadsPerBlock >>>( gb, gres, size);
}

__global__ void gpuTransx(const double *x, double* DQ, const int nTPoint, const int nTotal, const int nvar){
	int iPoint = blockDim.x*blockIdx.x + threadIdx.x;
	if(iPoint < nTPoint){
        DQ[iPoint] = x[iPoint*nvar+0];		
        DQ[nTotal+iPoint] = x[iPoint*nvar+1];
        DQ[nTotal*2+iPoint] = x[iPoint*nvar+2];
        DQ[nTotal*3+iPoint] = x[iPoint*nvar+3];
        DQ[nTotal*4+iPoint] = x[iPoint*nvar+4];
	}
}

void xForcusparse( PolyGrid *grid, RealFlow *&vecxLocal ){
    CHECKCUDA(cudaMemcpy(vecxLocal, gx_final, gn*gnvar*sizeof(RealFlow), cudaMemcpyDeviceToHost));
}

void cuSetWvec_Zvec_index(int i){
    g_i = i;
}

void cuWvecDotWvec(RealFlow *dot_vec, const int i, const int k){
    
    int blocksPerGrid = (gnvar*gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
    gpuWvecDotWvec <<< blocksPerGrid, threadsPerBlock >>> (gsumv2, &gb[i*gn*gnvar], &gb[k*gn*gnvar], gnvar*gnTCell);
    
    blocksPerGrid = gnodata2;
    Reducekernel6 <<< blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(RealFlow) >>> (gsumv2, godata2, gnsum2);
    
    int blocksPerGrid2 = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;
    Reducekernel_sum <<< blocksPerGrid2, threadsPerBlock >>> (val_Reduction, godata2, blocksPerGrid);
    
    cudaDeviceSynchronize();
    *dot_vec = val_Reduction[0];

#ifdef MPICH
    RealFlow sum_glb=0.0;
    MPI_Allreduce(dot_vec, &sum_glb, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    *dot_vec = sum_glb;
#endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error occurred during cuWvecDotWvec execution: %s\n", cudaGetErrorString(err));
    }
}

void cuWvecSubScalarMul(RealFlow *tmp_vec, const RealFlow Scalar, const int i, const int k){
    int blocksPerGrid = (gnvar*gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
    gpuWvecSubScalarMul <<< blocksPerGrid, threadsPerBlock >>> (&gb[i*gn*gnvar], &gb[k*gn*gnvar], Scalar, gnvar*gnTCell);
                                                    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error occurred during gpuWvecSubScalarMul execution: %s\n", cudaGetErrorString(err));
    }
}

void cuMatrixVectorProduct( PolyGrid *grid, RealFlow *tmp_vec, const int i){	

    if(USE_BSR){
        int blocksPerGrid = (gn + threadsPerBlock - 1) / threadsPerBlock;
        gpuMatrixVectorProduct <<< blocksPerGrid, threadsPerBlock >>> ( 
            gval, &gx[i*gn*gnvar], &gb[(i+1)*gn*gnvar], 
            gBSRrow_ptr, gBSRcol_ind, gnTCell, gnvar);
    }
    else{
        int blocksPerGrid = (gn*gnvar + threadsPerBlock - 1) / threadsPerBlock;
        gpuMatrixVectorProductcsr <<< blocksPerGrid, threadsPerBlock >>>(
            gval, &gx[i*gn*gnvar], &gb[(i+1)*gn*gnvar], 
            gCSRrow_ptr, gCSRcol_ind, gnTCell, gnvar);
    }
    cudaDeviceSynchronize();
    
#ifdef MPICH
    //communications for b
    RealFlow *b = new RealFlow[gn*gnvar];
    CHECKCUDA(cudaMemcpy(b, &gb[(i+1)*gn*gnvar], gn*gnvar*sizeof(RealFlow), cudaMemcpyDeviceToHost));
    IntType nT = gnTCell *gnvar;

    RealFlow *MPItmp[gnvar];
    MPItmp[0] = new RealFlow[gnvar*nT];
    for(int i=1; i<gnvar; i++) MPItmp[i] = &MPItmp[i-1][nT];

    IntType index = 0;
    for(IntType iCell = 0; iCell < gnTCell; iCell++){
        for(IntType iVar = 0; iVar < gnvar; iVar++){
            MPItmp[iVar][iCell] = b[index++];
        }
    }

    grid->RecvSendVarNeighbor_Togeth( gnvar, MPItmp );

    index = gnTCell*gnvar;
    for(IntType iCell = gnTCell+gnBFace-gnIFace; iCell < gnTCell+gnBFace; iCell++){
        for(IntType iVar = 0; iVar < gnvar; iVar++){
            b[index++] = MPItmp[iVar][iCell];
        }
    }
    delete[] MPItmp[0];
    CHECKCUDA(cudaMemcpy(&gb[(i+1)*gn*gnvar+gnTCell*gnvar], &b[gnTCell*gnvar], \
        gnIFace*gnvar*sizeof(RealFlow), cudaMemcpyHostToDevice));

#endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error occurred during gpuMatrixVectorProduct execution: %s\n", cudaGetErrorString(err));
    }
}

void cuWvecDivScalar(RealFlow *tmp_vec, const RealFlow Scalar, const int i){
    int blocksPerGrid = (gnvar*gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
    gpuWvecDivScalar <<< blocksPerGrid, threadsPerBlock >>> (&gb[i*gn*gnvar], Scalar, gnvar*gnTCell);
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error occurred during gpuWvecDivScalar execution: %s\n", cudaGetErrorString(err));
    }
}

void cuWvecNorm( double *norm, const int i){
    int blocksPerGrid = (gnvar*gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
    gpuDotProductMPI <<< blocksPerGrid, threadsPerBlock >>> (gsumv2, &gb[i*gn*gnvar], gnvar*gnTCell);
    
    blocksPerGrid = gnodata2;
    Reducekernel6 <<< blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double) >>> (gsumv2, godata2, gnsum2);
    
    int blocksPerGrid2 = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;	
    Reducekernel_sum <<< blocksPerGrid2, threadsPerBlock >>> (val_Reduction, godata2, blocksPerGrid);
    
    cudaDeviceSynchronize();
    *norm = val_Reduction[0];

#ifdef MPICH
    RealFlow sum_glb=0.0;
    MPI_Allreduce(norm, &sum_glb, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    *norm = sum_glb;
#endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error occurred during cuWvecNorm execution: %s\n", cudaGetErrorString(err));
    }
}

void cuResNorm( RealFlow *norm ){

    int blocksPerGrid = (gnvar*gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
    gpuDotProductMPI <<< blocksPerGrid, threadsPerBlock >>> (gsumv2, gres, gnvar*gnTCell);

    blocksPerGrid = gnodata2;
    Reducekernel6 <<< blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(RealFlow) >>> (gsumv2, godata2, gnsum2);
    
    int blocksPerGrid2 = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;	
    Reducekernel_sum <<< blocksPerGrid2, threadsPerBlock >>> (val_Reduction, godata2, blocksPerGrid);
    
    cudaDeviceSynchronize();
    *norm = val_Reduction[0];

#ifdef MPICH
    RealFlow sum_glb=0.0;
    MPI_Allreduce(norm, &sum_glb, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    *norm = sum_glb;
#endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error occurred during cuResNorm execution: %s\n", cudaGetErrorString(err));
    }
}

void cuAddx( const RealFlow factor, const int k, const int icount ){
		
    int blocksPerGrid = (gnvar*gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
    gpuAddx<<< blocksPerGrid, threadsPerBlock >>>(gx_final, &gx[k*gn*gnvar], factor, gnvar*gn);	

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error occurred during gpuAddx execution: %s\n", cudaGetErrorString(err));
    }
}


void cuModGramSchmidt(int i, RealFlow** Hsbg, RealFlow* w) {

  /*--- Parameter for reorthonormalization ---*/
  RealFlow reorth = 0.98;//0.98;

  /*--- Get the norm of the vector being orthogonalized, and find the
  threshold for re-orthogonalization ---*/

  RealFlow nrm = 0;
  cuWvecNorm( &nrm, i + 1 );

  RealFlow thr = nrm * reorth;

  /*--- The norm of w[i+1] < 0.0 or w[i+1] = NaN ---*/
  if ((nrm <= 0.0) || (nrm != nrm)) {
    /*--- nrm is the result of a dot product, communications are implicitly handled. ---*/
    printf("FGMRES orthogonalization failed, linear solver diverged with nrm:%.6f.\n",nrm);
    //exit(0);
  }

  /*--- Begin main Gram-Schmidt loop ---*/
  for (int k = 0; k < i + 1; k++) {
    RealFlow prod = 0;
    cuWvecDotWvec(&prod, i + 1, k);
    Hsbg[k][i] = prod;
    cuWvecSubScalarMul(NULL, prod, i + 1, k);

    /*--- Check if reorthogonalization is necessary ---*/
    if (prod * prod > thr) {
      cuWvecDotWvec(&prod, i + 1, k);
      Hsbg[k][i] += prod;
      cuWvecSubScalarMul(NULL, prod, i + 1, k);
    }
    /*--- Update the norm and check its size ---*/
    nrm -= pow(Hsbg[k][i], 2);
    nrm = max<double>(nrm, 0.0);
    //nrm = nrm > 0.0 ? nrm:0.0;
    thr = nrm * reorth;
  }

  /*--- Test the resulting vector ---*/
  cuWvecNorm( &nrm, i + 1 );
  nrm = sqrt(nrm);

  Hsbg[i + 1][i] = nrm;

  /*--- Scale the resulting vector ---*/
  cuWvecDivScalar(NULL, nrm, i + 1);

}

void cuMatrixLUSGS( PolyGrid *grid, IntType level, IntType iter_done, RealFlow *DQ[5] ){

    iter = iter_done;
    // Tranport res & matrix to GPU memory
    bForcusparse( grid, NULL ); // A gILU_matrix, x gx, b gb

    IntType fusedMethodFlag = 2;
    grid->GetData(&fusedMethodFlag,   INT, 1, "fusedImplicitMethod");

    timeval t0, t1;
    gettimeofday(&t0, NULL);
    double time_t0 = (double)t0.tv_sec + (double)t0.tv_usec/1000000; 
    const IntType warpnum = 8;
    g_i = 0;

    if( fusedMethodFlag == 1 ){
        CHECKCUDA(cudaMemset( gIsReady, 0, sizeof(int) * 2*gn ));
        IntType num_threads = 32*warpnum;
        IntType blocksPerGrid = ceil((double)(2*gn)/(double)warpnum);

        fusedMatrixLusgs<<<blocksPerGrid, num_threads>>>( \
            gBSRrow_ptr, gBSRcol_ind, gILU_matrix, gaux_vec, gIsReady, \
            gn, gnvar*gn, gres, &gx[g_i*gn*gnvar], gnvar );
        cudaDeviceSynchronize();
    }
    else if( fusedMethodFlag == 2 ){
        CHECKCUDA(cudaMemset( gIsReady, 0, sizeof(int) * gn ));
        IntType num_threads = 32*warpnum;
        IntType blocksPerGrid = ceil((double)(gn)/(double)warpnum);

        MatrixLusgsForward<<<blocksPerGrid, num_threads>>>( \
            gBSRrow_ptr, gBSRcol_ind, gILU_matrix, gaux_vec, gIsReady, \
            gn, gnvar*gn, gres, &gx[g_i*gn*gnvar], gnvar );
        cudaDeviceSynchronize();

        CHECKCUDA(cudaMemset( gIsReady, 0, sizeof(int) * gn ));
        MatrixLusgsBackward<<<blocksPerGrid, num_threads>>>( \
            gBSRrow_ptr, gBSRcol_ind, gILU_matrix, gaux_vec, gIsReady, \
            gn, gnvar*gn, gres, &gx[g_i*gn*gnvar], gnvar );
        cudaDeviceSynchronize();
    }
    else{
        printf("Wrong choice of fusedImplicitMethod in input parameters\n");
        exit(0);
    }

    gettimeofday(&t1, NULL);
    double time_t1 = (double)t1.tv_sec + (double)t1.tv_usec/1000000;
    ILUexe += (time_t1 - time_t0);
    ite++;

    // Fill back x to dq
    RealFlow *vecxLocal = new RealFlow[gn*gnvar];
    CHECKCUDA(cudaMemcpy(vecxLocal, &gx[g_i*gn*gnvar], gn*gnvar*sizeof(RealFlow), cudaMemcpyDeviceToHost));

    IntType idx = 0;
    for(IntType iCell = 0; iCell < gnTCell; iCell++){
        for(IntType iVar = 0; iVar < gnvar; iVar++){
            DQ[iVar][iCell] = vecxLocal[idx++];
        }
    }
    delete[] vecxLocal;
}

void cuGMRESILU( PolyGrid *grid, IntType level, IntType iter_done, RealFlow *DQ[5] ){

    iter = iter_done;
    IntType kspan = 15, gmresmaxits = 100;
    RealFlow tol = 0.01;
    grid->GetData(&kspan, INT, 1, "kspan");
    grid->GetData(&tol, REAL_FLOW, 1, "gmresepsilon");
    grid->GetData(&gmresmaxits, INT, 1, "gmresmaxits");
    IntType m = kspan;
    IntType n = gnTCell + gnBFace;
    IntType fusedFlag = 3; //defualt using cusparse
    grid->GetData(&fusedFlag,   INT, 1, "fusedImplicitMethod");

    bool converge = false;
    int i = 0, re = 0;

    // Tranport res & matrix to GPU memory
    bForcusparse( grid, NULL ); // A gILU_matrix, x gx, b gb

    timeval t0, t1;
    gettimeofday(&t0, NULL);
    double time_t0 = (double)t0.tv_sec + (double)t0.tv_usec/1000000; 
    // ILU decomposition using cusparse
    if(USE_BSR){
        cuSparseLUfactorizationBSR( );
    }else{
        cuSparseLUfactorizationCSR( );
    }
    gettimeofday(&t1, NULL);
    double time_t1 = (double)t1.tv_sec + (double)t1.tv_usec/1000000;
    ILUbuild += (time_t1 - time_t0);

    //tol to compute
    for( re=0; re<gmresmaxits; re++){

        RealFlow **H = new RealFlow*[m+1];
        for(int ii=0;ii<m+1;ii++) {H[ii] = new RealFlow[m];}
        RealFlow *y = new RealFlow[m];
        RealFlow *sn = new RealFlow[m+1];
        RealFlow *cs = new RealFlow[m+1];
        RealFlow *g = new RealFlow[m+1];
        for( i=0;i<m+1;i++){
            sn[i] = 0.0;
            cs[i] = 0.0;
            g[i]  = 0.0;
        }

        mat_vec( grid );

        RealFlow norm0 = 0;
        cuResNorm( &norm0 );
        norm0 = sqrt(norm0);
    
        RealFlow tmp_norm = 0;
        cuWvecNorm( &tmp_norm, 0 );
        tmp_norm = sqrt(tmp_norm);
        RealFlow beta = tmp_norm;

        //printf("cusparse cuGMRESILU start beta:%.15f beta/norm0:%.15f tol:%.15f\n", beta, beta/norm0, tol);
        cuWvecDivScalar(NULL, -beta, 0);
    
        /*--- Initialize the RHS of the reduced system ---*/
        g[0] = beta;
        for( i=0; i<kspan; i++ ){
            
            cuSetWvec_Zvec_index(i);

            timeval t2, t3;
            gettimeofday(&t2, NULL);
            double time_t2 = (double)t2.tv_sec + (double)t2.tv_usec/1000000;
            if(USE_BSR){
                if(fusedFlag == 4){
                    fusedILU();
                }
                else if(fusedFlag == 3){
                    cuSparseILUBSR( );
                }
                else{
                    syncfree_spTrsv();
                }
            }else{
                cuSparseILUCSR( );
            }
            ite++;
            gettimeofday(&t3, NULL);
            double time_t3 = (double)t3.tv_sec + (double)t3.tv_usec/1000000;
            ILUexe += (time_t3 - time_t2);

        #ifdef MPICH 
            RealFlow *vecxLocal = new RealFlow[gn*gnvar];
            RealFlow *MPItmp[gnvar];
            MPItmp[0] = new RealFlow[gnvar*n];
            for(int j=1; j<gnvar; j++) MPItmp[j] = &MPItmp[j-1][n];

            CHECKCUDA(cudaMemcpy(vecxLocal, &gx[g_i*gn*gnvar], gn*gnvar*sizeof(RealFlow), cudaMemcpyDeviceToHost)); 
            IntType index = 0;
            for(IntType iCell = 0; iCell < gnTCell; iCell++){
                for(IntType iVar = 0; iVar < gnvar; iVar++){
                    MPItmp[iVar][iCell] = vecxLocal[index++];
                }
            }

            grid->RecvSendVarNeighbor_Togeth( gnvar, MPItmp );

            index = gnTCell*gnvar;
            for(IntType iCell = gnTCell+gnBFace-gnIFace; iCell < gnTCell+gnBFace; iCell++){
                for(IntType iVar = 0; iVar < gnvar; iVar++){
                    vecxLocal[index++] = MPItmp[iVar][iCell];
                }
            }
            CHECKCUDA(cudaMemcpy(&gx[g_i*gn*gnvar+gnTCell*gnvar], &vecxLocal[gnTCell*gnvar], \
                gnIFace*gnvar*sizeof(RealFlow), cudaMemcpyHostToDevice));
            delete[] MPItmp[0];
            delete[] vecxLocal;
        #endif

            cuMatrixVectorProduct( grid, NULL, i );

            cuModGramSchmidt(i, H, NULL); // Third para should be gx on CPU memory if used. 

            /*---  Apply old Givens rotations to new column of the Hessenberg matrix then generate the
            new Givens rotation matrix and apply it to the last two elements of H[:][i] and g ---*/
    
            for (unsigned long k = 0; k < i; k++){
                ApplyGivens(sn[k], cs[k], H[k][i], H[k + 1][i]);
            }
            GenerateGivens(H[i][i], H[i + 1][i], sn[i], cs[i]);
            ApplyGivens(sn[i], cs[i], g[i], g[i + 1]);
    
            /*---  Set L2 norm of residual and check if solution has converged ---*/

            beta = fabs(g[i + 1]);

            //printf("cuGMRESILU re:%d kspan:%d i:%d btea:%.15f beta/norm0:%.15f tol:%.15f\n", re, kspan, i, beta, beta/norm0, tol);

            if( beta < tol * norm0 ) {
                if(i>1){
                    converge = true;
                }
            }
            if( converge ) {break;}
        }

        SolveReduced(i, H, g, y);
        for (unsigned long k = 0; k < i; k++) {
            RealFlow factor = y[k];
            cuAddx( factor, k, i );
        }

        delete[] g;
        delete[] sn;
        delete[] cs;
        delete[] y;
        for(int ii=0;ii<m+1;ii++) {delete[] H[ii];}
        delete[] H;

        if( converge ) {break;}
        gmresmaxits -= (i+1);
        kspan = min<IntType>(kspan, gmresmaxits);
    }
 
    // Fill back x to dq
    RealFlow *vecxLocal = new RealFlow[gn*gnvar];
    xForcusparse( grid, vecxLocal );

    IntType idx = 0;
    for(IntType iCell = 0; iCell < gnTCell; iCell++){
        for(IntType iVar = 0; iVar < gnvar; iVar++){
            DQ[iVar][iCell] = vecxLocal[idx++];
        }
    }
    delete[] vecxLocal;
}

void cuInitialMatrix( PolyGrid *grid, IntType level ){

    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nIFace = grid->GetNIFace();
    IntType nTFace = grid->GetNTFace();
    IntType n = nTCell + nBFace;
    IntType nvar = 5;
    RealFlow *vis_l, *vis_t = NULL;
    RealGeom *xcc, *ycc, *zcc;

    RealFlow gam, p_bar, lhs_omga;
    IntType *f2c   = grid->Getf2c();
    IntType *nFPC  = CalnFPC(grid);
    IntType **C2F  = CalC2F(grid); 
    RealGeom *xfn  = grid->GetXfn();
    RealGeom *yfn  = grid->GetYfn();
    RealGeom *zfn  = grid->GetZfn();
    RealGeom *area = grid->GetFaceArea();

    IntType iprec, vis_mode, vis_run=0;
    grid->GetData(&iprec,   INT, 1, "iprec");
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&lhs_omga,   REAL_FLOW, 1, "lhs_omga");
    RealFlow alf_l = 0.1;
    grid->GetData(&alf_l,  REAL_FLOW, 1, "alf_l");

    if(vis_mode != INVISCID){
        vis_run = 1;
        //�����Ԥ����ϵͳ,���װ뾶������ճ��
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

    RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");

    IntType hasCopyConstData2Device = 0;
    grid->GetData(&hasCopyConstData2Device,   INT, 1, "hasCopyConstData2Device");

    IntType *matrixInfo=NULL;
    RealFlow *matrix=NULL;
    IntType *bsr_row_ptr=NULL;
    IntType *bsr_col_ind=NULL;
    IntType *bsr_dia_ptr=NULL;
    IntType *csr_row_ptr=NULL;
    IntType *csr_col_ind=NULL;
    IntType *csrIndex=NULL;
    IntType *bsrIndex=NULL;
    IntType *oor = NULL;
    IntType *ooc = NULL;
    IntType MatrixN, nnz;
    //IntType Bstart = 0;

int mpirank = 0;
#ifdef MPICH
    int size = 0;      
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    if(!hasCopyConstData2Device){
        CalCOOInfoMPI( grid, oor, ooc, &MatrixN, &nnz );

        mfmem::snew_array_1D(matrixInfo, 2, dmrfl);
        mfmem::snew_array_1D(matrix, nnz*nvar*nvar, dmrfl);
        mfmem::snew_array_1D(bsr_row_ptr, MatrixN+1, dmrfl);
        mfmem::snew_array_1D(bsr_col_ind, nnz, dmrfl);
        mfmem::snew_array_1D(bsr_dia_ptr, MatrixN, dmrfl);
        mfmem::snew_array_1D(csr_row_ptr, MatrixN*nvar+1, dmrfl);
        mfmem::snew_array_1D(csr_col_ind, nnz*nvar*nvar, dmrfl);
        mfmem::snew_array_1D(csrIndex, nnz*nvar*nvar, dmrfl);
        mfmem::snew_array_1D(bsrIndex, nnz*nvar*nvar, dmrfl);

        matrixInfo[0] = MatrixN;
        matrixInfo[1] = nnz;

        //Compute the row_ptr and col_ind for BSR format
        ComputeBsrIndex( grid, bsr_row_ptr, bsr_col_ind, bsr_dia_ptr, nnz, MatrixN );

        //Compute the index array to convert the original matrix order to bsr & csr format
        ReorderedIndex( grid, oor, ooc, bsr_row_ptr, csrIndex, bsrIndex, nnz, MatrixN );

        //Compute the row_ptr and col_ind for CSR format, not used currently
        ComputeCsrIndex( grid, oor, ooc, csr_row_ptr, csr_col_ind, csrIndex, MatrixN, nvar, nnz );
        
        grid->UpdateDataPtr(matrixInfo, INT, 2, "matrixInfo");
        grid->UpdateDataPtr(matrix, REAL_FLOW, nnz*nvar*nvar, "matrix");
        grid->UpdateDataPtr(bsr_row_ptr, INT, MatrixN+1, "bsr_row_ptr");
        grid->UpdateDataPtr(bsr_col_ind, INT, nnz, "bsr_col_ind");
        grid->UpdateDataPtr(bsr_dia_ptr, INT, MatrixN, "bsr_dia_ptr");
        grid->UpdateDataPtr(csr_row_ptr, INT, MatrixN*nvar+1, "csr_row_ptr");
        grid->UpdateDataPtr(csr_col_ind, INT, nnz*nvar*nvar, "csr_col_ind");
        grid->UpdateDataPtr(csrIndex, INT, nnz*nvar*nvar, "csrIndex");
        grid->UpdateDataPtr(bsrIndex, INT, nnz*nvar*nvar, "bsrIndex");

        hasCopyConstData2Device = 1;
        grid->UpdateData(&hasCopyConstData2Device, INT, 1, "hasCopyConstData2Device");
        printf("rank:%d nTCell:%d nTFace:%d nBFace:%d nIFace:%d nnz:%d nvar:%d\n",mpirank,nTCell,nTFace,nBFace,nIFace,nnz,nvar);
        
        //copy data to GPU memory
        cuMatrixInitial( grid, MatrixN, nnz, nvar );

        //diagnose elements:
        RealGeom *vol  =  grid->GetCellVol();
        RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
        IntType nTFace = grid->GetNTFace();
        RealGeom *norm_dist_c2c = NULL;
        norm_dist_c2c = (RealGeom *)grid->GetDataPtr(REAL_GEOM, nTFace, "norm_dist_c2c");

        IntType implicitmat = 0;
        grid->GetData(&implicitmat, INT, 1, "implicitmat", 0);
        IntType pctype = 0;
        grid->GetData(&pctype, INT, 1, "pctype", 0);

        if(implicitmat != 1 || pctype != 4){
            printf("error implicitmat in InitialGPUMem!\n");
            exit(0);
        }

        IntType ifStart = nBFace - nIFace;

        IntType *IndexC2F = NULL;
        mfmem::snew_array_1D(IndexC2F, nTCell + 1, dmrfl);
        IndexC2F[0] = 0;
        for(IntType i = 1; i < nTCell + 1; i++){
            IndexC2F[i] = IndexC2F[i - 1] + nFPC[i - 1];
        }
        cout<<"copy const data to device..."<<endl;
        CopyConstData2Device(nTCell, nBFace, nTFace, ifStart, vis_run, gam, p_bar, \
            alf_l, bsr_row_ptr, f2c, C2F, IndexC2F, nFPC, vol, xfn, yfn, zfn, dt, norm_dist_c2c, area);
        mfmem::sdel_array_1D(IndexC2F);
        hasCopyConstData2Device = 1;
        grid->UpdateData(&hasCopyConstData2Device, INT, 1, "hasCopyConstData2Device");

        if(USE_BSR){
            cuSparseInitialbsr( );
        }
        else{
            cuSparseInitialcsr( );
        }
    }

    CopyNonConstData2Device(nTCell, nBFace, vis_l,  vis_t, q);

    //Fill the matrix
    matrix_set( grid );
}

#endif