#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <limits>
using namespace std;

// user defined head files
#include "grid_polyhedra.h"
#include "io_log.h"
#include "utility_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"

#include <cuData.cuh>
#include <cuErrorReturn.cuh>
#include <cuMPI.cuh>
#include <cuLUSGS.cuh>
#include "cuGradientQ_Gauss.cuh"
#include <cuInviscidFlux.cuh>
#include <cuSAsolver.cuh>
#include <cuTurbulenceFlux.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef MPICH
#include <mpi.h>
#endif

#ifdef MultiStream
#include <cuLimit.cuh>
#endif

using namespace mflow;

using namespace gpuData;

#ifdef MPICH
void NSSolver::cuTransferInterfaceData(PolyGrid *grid) {
    
	gMPI = gq;
	//grid->cuRecvSendVarNeighbor_Togeth_q5(kNVar);
#if defined MultiStream
	grid->cuRecvSendVarNeighbor_Togeth_q5ForInterfaceData_unfold(kNVar);
#else
	grid->cuRecvSendVarNeighbor_Togeth_q5(kNVar);
#endif
}

void PolyGrid::cuGetIndex_RecvSend(IntType *Indexbqsr, IntType nvar)
{	
	IntType count = 0;
	IntType i, j ,k;
	for(i = 0; i < nNeighbor; i++) {
		for(j = 0; j < nvar; j++){
			for(k = 0; k < nZIFace[i]; k++) {
				Indexbqsr[count] = j*(nTCell + nBFace) + bCNo[i][k];				
				count++;
			}
		}
	}
	
}

void PolyGrid::cuGetIndex_RecvSend2(IntType *Indexbqsr, IntType nvar)
{	
	IntType count = 0;
	IntType i, j ,k;
	for(i = 0; i < nNeighbor; i++) {
		for(j = 0; j < nvar; j++){
			for(k = 0; k < nZIFace[i]; k++) {
				Indexbqsr[count] = j*(nTCell + nBFace) + nTCell + bFNo[i][k];				
				count++;
			}
		}
	}
	
}

/* void PolyGrid::cuGetIndex_RecvSend_Node(IntType *Indexbqsr, IntType nvar)
{	
	IntType count = 0;
	IntType i, j ,k;
	for(i = 0; i < nNeighborN; i++) {
		for(j = 0; j < nvar; j++){
			for(k = 0; k < nZINode[i]; k++) {
				Indexbqsr[count] = j*nTNode + bNSNo[i][k];				
				count++;
			}
		}
	}
	
}

void PolyGrid::cuGetIndex_RecvSend2_Node(IntType *Indexbqsr, IntType nvar)
{	
	IntType count = 0;
	IntType i, j ,k;
	for(i = 0; i < nNeighborN; i++) {
		for(j = 0; j < nvar; j++){
			for(k = 0; k < nZINode[i]; k++) {
				Indexbqsr[count] = j*nTNode + bNRNo[i][k];				
				count++;
			}
		}
	}
	
} */

void PolyGrid::cuGetLength_RecvSend_Node(){
	
    IntType i, temp_nZIFace=0;

    for(i=0; i<nNeighborN; i++) temp_nZIFace += nZINode[i];
    
	glenbqsr_Node = temp_nZIFace;

}

void PolyGrid::cuGetLength_RecvSend(){
	
    IntType i, temp_nZIFace=0;

    for(i=0; i<nNeighbor; i++) temp_nZIFace += nZIFace[i];
    
	glenbqsr = 5*temp_nZIFace;
	glenbqsrSA = temp_nZIFace;
	glenbqsrGrad = 3*5*temp_nZIFace;
	glenbqsrGradSA = temp_nZIFace;
}

__global__ void gpuSet_RecvSend(RealFlow *bqs, RealFlow *bqr, IntType lenbqsr){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < lenbqsr){
		bqs[i] = 0.0;
		bqr[i] = 0.0;
	}
	
}

void PolyGrid::cuSet_RecvSend(RealFlow ***bqs, RealFlow ***bqr, IntType nvar)
{
    IntType i, j, temp_nZIFace=0;	
	
    for(i=0; i<nNeighbor; i++) temp_nZIFace += nZIFace[i];
    bqs[0] = NULL;
    bqr[0] = NULL;
    mfmem::snew_array_1D(bqs[0], nvar*nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqr[0], nvar*nNeighbor,dmrfl);
	
    for(i=1; i<nNeighbor; i++){
        bqs[i] = &bqs[i-1][nvar];
        bqr[i] = &bqr[i-1][nvar];
    }
    bqs[0][0] = NULL;
    bqr[0][0] = NULL;
    //mfmem::snew_array_1D(bqs[0][0], nNeighbor*nvar*temp_nZIFace,dmrfl);
    //mfmem::snew_array_1D(bqr[0][0], nNeighbor*nvar*temp_nZIFace,dmrfl);
	mfmem::snew_array_1D(bqs[0][0], nvar*temp_nZIFace,dmrfl);
    mfmem::snew_array_1D(bqr[0][0], nvar*temp_nZIFace,dmrfl);
    for(i=1; i<nNeighbor; i++){
        bqs[i][0] =&bqs[i-1][0][nvar*nZIFace[i-1]]; //保证每一个面与下一个面连续				
        bqr[i][0] =&bqr[i-1][0][nvar*nZIFace[i-1]];
    }
    for(i=0; i<nNeighbor; i++){
        for(j=1; j<nvar; j++){
            bqs[i][j] = &bqs[i][j-1][nZIFace[i]];   //保证每一条线与下一条线连续			
            bqr[i][j] = &bqr[i][j-1][nZIFace[i]];
        }
    }
	//IntType blocksPerGrid = (glenbqsr + threadsPerBlock - 1) / threadsPerBlock;
	//gpuSet_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gbqs, glenbqsr);
	
	//HANDLE_API_ERR(cudaMemcpy(&bqs[0][0][0], gbqs, glenbqsr*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(&bqr[0][0][0], gbqr, glenbqsr*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	/*
    for(i=0; i<nNeighbor; i++){
        for(j=0; j<nvar; j++){
            for(k=0; k<nZIFace[i]; k++){
                bqs[i][j][k]=0.0;
                bqr[i][j][k]=0.0;
            }
        }
    }
	*/
}

void PolyGrid::cuAdd_RecvSend(RealFlow ***bqs, RealFlow *q, IntType i, IntType k)
{	
	
    for(IntType j=0; j<nZIFace[i]; j++) {
		bqs[i][k][j] = q[bCNo[i][j]];		
	}
	
}

__global__ void gpuAdd_RecvSend(RealFlow *bqs, const RealFlow *q, const IntType *Indexbqsr, const IntType lenbqsr){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;									
	if(i < lenbqsr){
		bqs[i] = q[Indexbqsr[i]];	
	}												
}

__global__ void gpuAdd_RecvSend2(RealFlow *q, const RealFlow *bqr, const IntType *Indexbqsr, const IntType lenbqsr){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;									
	if(i < lenbqsr){
		q[Indexbqsr[i]] = bqr[i];	
	}												
}

__global__ void gpuAdd_RecvSend_Gradient(RealFlow *bqs, const RealFlow *dqdx, const RealFlow *dqdy, const RealFlow *dqdz, 
								const IntType *Indexbqsr, const IntType lenbqsr){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;									
	if(i < lenbqsr){
		bqs[i] = dqdx[Indexbqsr[i]];	
		bqs[lenbqsr + i] = dqdy[Indexbqsr[i]];	
		bqs[2*lenbqsr + i] = dqdz[Indexbqsr[i]];	
	}												
}

__global__ void gpuAdd_RecvSend2_Gradient(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz,
								const RealFlow *bqr, const IntType *Indexbqsr, const IntType lenbqsr){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;									
	if(i < lenbqsr){
		dqdx[Indexbqsr[i]] = bqr[i];	
		dqdy[Indexbqsr[i]] = bqr[lenbqsr + i];	
		dqdz[Indexbqsr[i]] = bqr[2*lenbqsr + i];	
	}												
}

void PolyGrid::cuRecvSendVarNeighbor_Togeth(IntType nvar, RealFlow **q, IntType type){
	
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;
    IntType i;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
	
    cuSet_RecvSend(bqs, bqr, nvar);
	
    //for(i=0; i<nvar; i++)
    //    cuAdd_RecvSend(bqs, q[i], i);
	if (nvar == 5){	// 5*(nTCell + nBFace) variables
		// for q(type=0), dq(type=1), limit(type=2), dqdx(type=3), dqdy(type=4), dqdz(type=5):		
		IntType blocksPerGrid = (glenbqsr + threadsPerBlock - 1) / threadsPerBlock;
		if (type == 0){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gq, gIndexbqsr, glenbqsr);
		}
		else if(type == 1){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gDQ, gIndexbqsr, glenbqsr);
		}
		else if(type == 2){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, glimit, gIndexbqsr, glenbqsr);
		}
		else if(type == 3){ 
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gdqdx, gIndexbqsr, glenbqsr);
		}
		else if(type == 4){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gdqdy, gIndexbqsr, glenbqsr);
		}
		else if(type == 5){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gdqdz, gIndexbqsr, glenbqsr);
		}
		HANDLE_API_ERR(cudaMemcpy(&bqs[0][0][0], gbqs, glenbqsr*sizeof(RealFlow), cudaMemcpyDeviceToHost));		
	}
	else if(nvar == 1){ // (nTCell + nBFace) variables
		// for sa_nu(type=0), vis_l(type=1), vis_t(type=2), dq(sa_nu)(type=3):
		IntType blocksPerGrid = (glenbqsrSA + threadsPerBlock - 1) / threadsPerBlock;
		if (type == 0){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gsa_nu, gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 1){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gvis_l, gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 2){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gvis_t, gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 3){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gDQ, gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 4){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gdnutdx, gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 5){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gdnutdy, gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 6){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gdnutdz, gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 7){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gdtdx, gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 8){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gdtdy, gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 9){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gdtdz, gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 10){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gdqadu, gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 11){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gDQ, gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 12){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, &gDQ[1*(gnTCell + gnBFace)], gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 13){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, &gDQ[2*(gnTCell + gnBFace)], gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 14){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, &gDQ[3*(gnTCell + gnBFace)], gIndexbqsrSA, glenbqsrSA);
		}
		else if(type == 15){
			gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, &gDQ[4*(gnTCell + gnBFace)], gIndexbqsrSA, glenbqsrSA);
		}
		HANDLE_API_ERR(cudaMemcpy(&bqs[0][0][0], gbqs, glenbqsrSA*sizeof(RealFlow), cudaMemcpyDeviceToHost));		
	}
	else if(nvar == 3){
		// for for dnutdx, dnutdy, dnutdz, dtdx, dtdy, dtdz:
		IntType blocksPerGrid = (glenbqsr + threadsPerBlock - 1) / threadsPerBlock;
		gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gq, gIndexbqsr, glenbqsr);
		HANDLE_API_ERR(cudaMemcpy(&bqs[0][0][0], gbqs, glenbqsr*sizeof(RealFlow), cudaMemcpyDeviceToHost));		
	}
	/*	
	for(IntType i=0; i<nNeighbor; i++) {	
		for(IntType k=0; k<nvar; k++){		
			for(IntType j=0; j<nZIFace[i]; j++) {
				bqs[i][k][j] = q[k][bCNo[i][j]];
			}
			//cuAdd_RecvSend(bqs, q[k], i, k);
		}
	} 			
	*/
    RecvSendVarNeighbor_Over(bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);
		
    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
    for(i=0; i<nvar; i++)
        Read_RecvSend(bqr, q[i], i);

    mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);
	
}

void PolyGrid::cuRecvSendVarNeighbor_Togeth_q5(IntType nvar){
	
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
	
	bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
	
    cuSet_RecvSend(bqs, bqr, nvar);
	
	IntType blocksPerGrid = (glenbqsr + threadsPerBlock - 1) / threadsPerBlock;	
	
	gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gMPI, gIndexbqsr, glenbqsr);		
	
	HANDLE_API_ERR(cudaMemcpy(&bqs[0][0][0], gbqs, glenbqsr*sizeof(RealFlow), cudaMemcpyDeviceToHost));				
	
    RecvSendVarNeighbor_Over(bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);
	
	mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
	
	HANDLE_API_ERR(cudaMemcpy(gbqr, &bqr[0][0][0], glenbqsr*sizeof(RealFlow), cudaMemcpyHostToDevice));	
		   
    blocksPerGrid = (glenbqsr + threadsPerBlock - 1) / threadsPerBlock;	
	gpuAdd_RecvSend2 <<< blocksPerGrid, threadsPerBlock >>> (gMPI, gbqr, gIndexbqsr2, glenbqsr);		
	
	mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);
	
}

void PolyGrid::cuRecvSendVarNeighbor_Togeth_SA(IntType nvar){
	
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
	
	bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
	
    cuSet_RecvSend(bqs, bqr, nvar);
	
	IntType blocksPerGrid = (glenbqsrSA + threadsPerBlock - 1) / threadsPerBlock;		
	gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gMPI, gIndexbqsrSA, glenbqsrSA);		
	
	HANDLE_API_ERR(cudaMemcpy(&bqs[0][0][0], gbqs, glenbqsrSA*sizeof(RealFlow), cudaMemcpyDeviceToHost));				
	
    RecvSendVarNeighbor_Over(bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);
	
	mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
	
	HANDLE_API_ERR(cudaMemcpy(gbqr, &bqr[0][0][0], glenbqsrSA*sizeof(RealFlow), cudaMemcpyHostToDevice));	
		   
	blocksPerGrid = (glenbqsrSA + threadsPerBlock - 1) / threadsPerBlock;	
	gpuAdd_RecvSend2 <<< blocksPerGrid, threadsPerBlock >>> (gMPI, gbqr, gIndexbqsr2SA, glenbqsrSA);		
	
	mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);
	
}

#ifdef MultiStream
void PolyGrid::cuRecvSendVarNeighbor_TogethForGradient_unfold_MergedLimit(IntType nvar, RealFlow **dqdx, RealFlow **dqdy, RealFlow **dqdz){
	
	IntType blocksPerGrid, name;
	//cuVencatLimiter_MultiStream_espcell(0);
	// Find the maximum and minimum in the neighbor of each cell
	name = 0;
    blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuMaxMinDiffInit_Merged <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);

	// Manual reduction	
	gpuMaxMinDiff_Merged  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gC2F, gIndexC2F, gnFPC, gf2c, 
														gtype_bcr, gnTCell, gnBFace, name);	
	
	gpuMaxMinDiffReduceQ_Merged  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);
		 			
	
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
	
	MPI_Request *req_send2=0, *req_recv2=0;
    MPI_Status *status_array2=0;

    status_array2 = NULL;
    req_send2     = NULL;
    req_recv2     = NULL;
    mfmem::snew_array_1D(status_array2, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send2,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv2,     nNeighbor,dmrfl);
	
	MPI_Request *req_send3=0, *req_recv3=0;
    MPI_Status *status_array3=0;

    status_array3 = NULL;
    req_send3     = NULL;
    req_recv3     = NULL;
    mfmem::snew_array_1D(status_array3, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send3,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv3,     nNeighbor,dmrfl);
	
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
	
    cuSet_RecvSend(bqs, bqr, nvar);

	// for dqdx:
	blocksPerGrid = (glenbqsr + threadsPerBlock - 1) / threadsPerBlock;		
	
	gpuAdd_RecvSend_Gradient <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (gbqs, gdqdx, gdqdy, gdqdz, gIndexbqsr, glenbqsr);	
	
	HANDLE_API_ERR(cudaMemcpyAsync(hostbqs, gbqs, 3*glenbqsr*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[4]));				
	
	blocksPerGrid = (gnTCell + gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuLimitInit_Merged <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (glimit, gnTCell, gnBFace, name);
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuLimitespcell <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gespcell_MStream[name*gnTCell], gvol, gq, geps_tmp, gnTCell, gnBFace, name);  
	
	//cuVencatLimiter_MultiStream_espcell(1);
	name = 1;
    //blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuLimitespcell3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gespcell_MStream[name*gnTCell], gvol, gq, geps_tmp, ggam, gp_bar, gnTCell, gnBFace, name);
	
	//cuVencatLimiter_MultiStream_espcell(2);
	name = 2;
    //blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuLimitespcell3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gespcell_MStream[name*gnTCell], gvol, gq, geps_tmp, ggam, gp_bar, gnTCell, gnBFace, name);
		
	cudaStreamSynchronize(flowstream[4]);			
	
    RecvSendVarNeighbor_Over_Gradient(hostbqs, hostbqr, bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);
	
	mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
	
	// for dqdy:				
    RecvSendVarNeighbor_Over_Gradient(&hostbqs[glenbqsr], &hostbqr[glenbqsr], bqs, bqr, req_send2, req_recv2, status_array2, nvar);

    MPI_Waitall(nNeighbor,req_recv2,status_array2);
    MPI_Waitall(nNeighbor,req_send2,status_array2);
	
	mfmem::sdel_array_1D(req_send2);
    mfmem::sdel_array_1D(req_recv2);
    mfmem::sdel_array_1D(status_array2);
	
	// for dqdz:			
    RecvSendVarNeighbor_Over_Gradient(&hostbqs[2*glenbqsr], &hostbqr[2*glenbqsr], bqs, bqr, req_send3, req_recv3, status_array3, nvar);

    MPI_Waitall(nNeighbor,req_recv3,status_array3);
    MPI_Waitall(nNeighbor,req_send3,status_array3);
		
    mfmem::sdel_array_1D(req_send3);
    mfmem::sdel_array_1D(req_recv3);
    mfmem::sdel_array_1D(status_array3);
	
	HANDLE_API_ERR(cudaMemcpyAsync(gbqr, hostbqr, 3*glenbqsr*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[0]));			
	
	//cuVencatLimiter_MultiStream_espcell(3);
	name = 3;
    blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuLimitespcell3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gespcell_MStream[name*gnTCell], gvol, gq, geps_tmp, ggam, gp_bar, gnTCell, gnBFace, name);
	
	//cuVencatLimiter_MultiStream_espcell(4);
	name = 4;
    blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuLimitespcell4 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gespcell_MStream[name*gnTCell], gvol, gq, geps_tmp, gp_bar, gnTCell, gnBFace, name);
	
	cudaStreamSynchronize(flowstream[0]);
	blocksPerGrid = (glenbqsr + threadsPerBlock - 1) / threadsPerBlock;		
	gpuAdd_RecvSend2_Gradient <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0]>>> (gdqdx, gdqdy, gdqdz, gbqr, gIndexbqsr2, glenbqsr);   	
	
    mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);
	
	cudaStreamSynchronize(flowstream[0]);
	cudaStreamSynchronize(flowstream[1]);
	cudaStreamSynchronize(flowstream[2]);
	cudaStreamSynchronize(flowstream[3]);
	cudaStreamSynchronize(flowstream[4]);
}

void PolyGrid::cuRecvSendVarNeighbor_Togeth_q5ForInterfaceData_unfold(IntType nvar){
	
	//GPUGrad_Limit_Init();
	IntType blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gu_n, gnTNode);
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gv_n, gnTNode);
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gw_n, gnTNode);
			
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);		
	
	bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
	
    cuSet_RecvSend(bqs, bqr, nvar);
	
	blocksPerGrid = (glenbqsr + threadsPerBlock - 1) / threadsPerBlock;	
	
	gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gbqs, gMPI, gIndexbqsr, glenbqsr);		
	
	HANDLE_API_ERR(cudaMemcpyAsync(hostbqs, gbqs, glenbqsr*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[1]));
	
	blocksPerGrid = (5*(gnTCell + gnBFace) + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradientInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdqdx, gdqdy, gdqdz, 5*(gnTCell + gnBFace));
	
	blocksPerGrid = ((gnTCell + gnBFace) + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradientInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdtdx, gdtdy, gdtdz, (gnTCell + gnBFace));
	
	blocksPerGrid = (5*gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gmsq_n, 5*gnTNode);

	blocksPerGrid = (gnTCell + gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (glimit, gnTCell, gnBFace, 0);
	gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (glimit, gnTCell, gnBFace, 1);
	gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (glimit, gnTCell, gnBFace, 2);
	gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (glimit, gnTCell, gnBFace, 3);
	gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (glimit, gnTCell, gnBFace, 4);		
														
	cudaStreamSynchronize(flowstream[1]);
    RecvSendVarNeighbor_Over_Gradient(hostbqs, hostbqr, bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);
	
	mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
	
	HANDLE_API_ERR(cudaMemcpyAsync(gbqr, hostbqr, glenbqsr*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[1]));		
	cudaStreamSynchronize(flowstream[1]);	   
    
	blocksPerGrid = (glenbqsr + threadsPerBlock - 1) / threadsPerBlock;	
	gpuAdd_RecvSend2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gMPI, gbqr, gIndexbqsr2, glenbqsr);		
	
	mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);
	
	cudaStreamSynchronize(flowstream[0]);	
	cudaStreamSynchronize(flowstream[1]);	
	
}

void PolyGrid::cuRecvSendVarNeighbor_Togeth_q5ForLimit_unfold(IntType nvar){
	
	RealFlow stind = 0.1; 
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuCellIsMG <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdet, gnTCell);
	
	// Manual Reduction: 	
	blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuCellIsMG2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gtmpvar, &gq[4*(gnTCell + gnBFace)], gf2c, gp_bar, gnTFace);
	
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
	
	bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
	
    cuSet_RecvSend(bqs, bqr, nvar);
	
	blocksPerGrid = (glenbqsr + threadsPerBlock - 1) / threadsPerBlock;	
	
	gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gbqs, gMPI, gIndexbqsr, glenbqsr);		
	
	HANDLE_API_ERR(cudaMemcpyAsync(hostbqs, gbqs, glenbqsr*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[1]));	
		
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuCellIsMG3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdet, gtmpvar, gf2c, gC2F, gIndexC2F, 
														gnFPC, stind, gnTCell, gnBFace, gnTFace);
														
	IntType level  = this->GetLevel();
    
    IntType vis_mode;
    this->GetData(&vis_mode,  INT, 1, "vis_mode");  
    
    //If count viscous or not
    IntType vis_run;  //0--inviscid   1--laminar   2--turbulence
    if(vis_mode == INVISCID){
        vis_run = 0;
    }else if(vis_mode == LAMINAR){
        vis_run = 1;
    }else{
        vis_run = 2;
    }
    if((level != 0) && (vis_mode != INVISCID)){  // if coarse grid doesn't want to run the viscous flux, turn it off
        IntType cg_vis = 1;
        this->GetData(&cg_vis, INT, 1, "cg_vis");
        if(cg_vis == 0) vis_run = 0;
    }

	RealFlow C = 4.0;
    RealFlow prl, prt;
    if(vis_run){
        this->GetData(&prl, REAL_FLOW, 1, "prl");  
        this->GetData(&prt, REAL_FLOW, 1, "prt");
    }
	
	// Set dt to BIG
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuTimeStepNormal_new <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdt, gnTCell);	
	
	//Manual reduction	
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuTimeStepNormal_new2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gtmpvar, gq, gvis_l, gvis_t, gxfc, gyfc, gzfc, 
																gxcc, gycc, gzcc, gxfn, gyfn, gzfn, garea, gvol, gf2c, 
																gvgn, gsteady, gp_bar, ggam, prl, prt, C, vis_run, gnTCell, gnBFace);
		
    // For interior faces
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuTimeStepNormal_new3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gtmpvar, gq, gvis_l, gvis_t, gxfc, gyfc, gzfc, 
																gxcc, gycc, gzcc, gxfn, gyfn, gzfn, garea, gvol, gf2c, 
																gvgn, gsteady, gp_bar, ggam, prl, prt, C, vis_run,
																gnTFace, gnTCell, gnBFace);
																
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuTimeStepNormal_newReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdt, gtmpvar, gf2c, gC2F, gIndexC2F, 
															gnFPC, gnTCell);	
	
	//cuLimitTimeStep(grid);
	RealFlow cfl;
    
    IntType  iter_done, cfl_nstep;
    RealFlow cfl_start, cfl_end, cfl_coeff, cfl_ratio;
    this->GetData(&iter_done, INT, 1, "iter_done");
    this->GetData(&cfl_nstep, INT, 1, "cfl_nstep");
    this->GetData(&cfl_start, REAL_FLOW, 1, "cfl_start");
    this->GetData(&cfl_end,   REAL_FLOW, 1, "cfl_end");
    this->GetData(&cfl_coeff, REAL_FLOW, 1, "cfl_coeff");
    this->GetData(&cfl_ratio, REAL_FLOW, 1, "cfl_ratio");
    
    //for coarse grid, reduce cfl number using cfl_coeff
    if(level>0){
        cfl_start *= cfl_coeff;
        cfl_end   *= cfl_coeff;
    }
    
    //compute current step's cfl number
    if(iter_done < 0){  //粗网格迭代
        cfl = cfl_start;
    }else if(iter_done > cfl_nstep){
        cfl = cfl_end;
    }else{
        //zhyb20190620: modified from CFL3D, the ramping now is nonlinear, occurring slowerly at first
        //              and then increasing in rate.
        cfl = cfl_start*pow(cfl_ratio, (RealFlow)iter_done/cfl_nstep);
    }
    
    //limit cfl using p, come from USM3D
    RealFlow p_min, p_break, cfl_min;
    this->GetData(&p_min,   REAL_FLOW, 1, "p_min");
    this->GetData(&p_break, REAL_FLOW, 1, "p_break");
    this->GetData(&cfl_min, REAL_FLOW, 1, "cfl_min");
    //limit cfl using gradient of p, decrease cfl in big gradient of p
    cfl_min = 0.5*cfl;  //在此处将cfl_min设为当前步cfl数乘以0.5		
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuLimitTimeStep_dtmindtmax <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdtmaxsumv2, gdtminsumv2, gdt, &gq[4*(gnTCell + gnBFace)], gdet, cfl, cfl_min,  p_min, p_break, gnTCell);

	
	cudaStreamSynchronize(flowstream[1]);
    RecvSendVarNeighbor_Over_Gradient(hostbqs, hostbqr, bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);
	
	mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
	
	HANDLE_API_ERR(cudaMemcpyAsync(gbqr, hostbqr, glenbqsr*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[1]));		

	// Print out the maximum and minimun dt
    RealFlow dt_max = 0.0, dt_min = BIG;
	
	blocksPerGrid = gdtmaxnodata2;
	Reducekernel6_Max <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow), flowstream[0] >>> (gdtmaxsumv2, gdtmaxodata2, gdtmaxnsum2);
	Reducekernel6_Min <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow), flowstream[0] >>> (gdtminsumv2, gdtminodata2, gdtmaxnsum2);
	
	IntType blocksPerGrid2 = (gdtmaxnodata2 + threadsPerBlock - 1) / threadsPerBlock;	
	Reducekernel_Max <<< blocksPerGrid2, threadsPerBlock, 0, flowstream[0] >>> (val_Reduction, gdtmaxodata2, blocksPerGrid);
	Reducekernel_Min <<< blocksPerGrid2, threadsPerBlock, 0, flowstream[0] >>> (val_Reduction, gdtminodata2, blocksPerGrid); 	
	
	cudaStreamSynchronize(flowstream[1]);	   
    /* for(i=0; i<nvar; i++)
        Read_RecvSend(bqr, dqdx[i], i); */
	blocksPerGrid = (glenbqsr + threadsPerBlock - 1) / threadsPerBlock;	
	gpuAdd_RecvSend2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gMPI, gbqr, gIndexbqsr2, glenbqsr);	

	cudaDeviceSynchronize();
	dt_max = val_Reduction[0];
	dt_min = val_Reduction[1];	
	
	#ifdef MPICH
		RealFlow dt_max_glb, dt_min_glb;
		MPI_Allreduce(&dt_max, &dt_max_glb, 1, MPIReal, MPI_MAX, MPI_COMM_WORLD);
		MPI_Allreduce(&dt_min, &dt_min_glb, 1, MPIReal, MPI_MIN, MPI_COMM_WORLD);
		dt_max = dt_max_glb;
		dt_min = dt_min_glb;
	#endif

    this->UpdateData(&dt_max, REAL_FLOW, 1, "dt_max");
    this->UpdateData(&dt_min, REAL_FLOW, 1, "dt_min");
    
    //Now limit the dt to ratio_dtmax*dt_min 
    RealFlow ratio_dtmax = 1.0e20;
    this->GetData(&ratio_dtmax, REAL_FLOW, 1, "ratio_dtmax");
    RealFlow ratio_max = dt_max/dt_min;
	
    if(ratio_max > ratio_dtmax){
        RealFlow dt_max_lim = ratio_dtmax*dt_min;
		blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
        gpuLimitTimeStep2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdt, dt_max_lim, gnTCell);
		//HANDLE_API_ERR(cudaMemcpy(dt, gdt, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));		
    }
	
	mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);
	
	cudaStreamSynchronize(flowstream[0]);	
	cudaStreamSynchronize(flowstream[1]);	
	
}

void PolyGrid::cuRecvSendVarNeighbor_TogethForGradient_unfold(IntType nvar, RealFlow **dqdx, RealFlow **dqdy, RealFlow **dqdz){
	
	IntType blocksPerGrid, name;
	//cuVencatLimiter_MultiStream_espcell(0);
	// Find the maximum and minimum in the neighbor of each cell
	name = 0;
    blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuMaxMinDiffInit_Merged <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);

	// Manual reduction	
	gpuMaxMinDiff  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gC2F, gIndexC2F, gnFPC, gf2c, 
														gtype_bcr, gnTCell, gnBFace, name);	
	
	gpuMaxMinDiffReduceQ  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);	
	
	/* blocksPerGrid = (gnTCell + gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (glimit, gnTCell, gnBFace, name); */
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuLimitespcell <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gespcell_MStream[name*gnTCell], gvol, gq, geps_tmp, gnTCell, gnBFace, name);   	
	
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
	
	MPI_Request *req_send2=0, *req_recv2=0;
    MPI_Status *status_array2=0;

    status_array2 = NULL;
    req_send2     = NULL;
    req_recv2     = NULL;
    mfmem::snew_array_1D(status_array2, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send2,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv2,     nNeighbor,dmrfl);
	
	MPI_Request *req_send3=0, *req_recv3=0;
    MPI_Status *status_array3=0;

    status_array3 = NULL;
    req_send3     = NULL;
    req_recv3     = NULL;
    mfmem::snew_array_1D(status_array3, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send3,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv3,     nNeighbor,dmrfl);
	
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
	
    cuSet_RecvSend(bqs, bqr, nvar);

	// for dqdx:
	blocksPerGrid = (glenbqsr + threadsPerBlock - 1) / threadsPerBlock;		
	
	gpuAdd_RecvSend_Gradient <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (gbqs, gdqdx, gdqdy, gdqdz, gIndexbqsr, glenbqsr);	
	
	HANDLE_API_ERR(cudaMemcpyAsync(hostbqs, gbqs, 3*glenbqsr*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[4]));				
	
	//cuVencatLimiter_MultiStream_espcell(1);
	name = 1;
    blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	//gpuMaxMinDiffInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);
	
	// Manual reduction	
	gpuMaxMinDiff  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gC2F, gIndexC2F, gnFPC, gf2c, 
														gtype_bcr, gnTCell, gnBFace, name);
	
	gpuMaxMinDiffReduceQ  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);	
	
	/* blocksPerGrid = (gnTCell + gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (glimit, gnTCell, gnBFace, name); */
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuLimitespcell3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gespcell_MStream[name*gnTCell], gvol, gq, geps_tmp, ggam, gp_bar, gnTCell, gnBFace, name);
	
	//cuVencatLimiter_MultiStream_espcell(2);
	name = 2;
    blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	//gpuMaxMinDiffInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);
	
	// Manual reduction	
	gpuMaxMinDiff  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gC2F, gIndexC2F, gnFPC, gf2c, 
														gtype_bcr, gnTCell, gnBFace, name);	
	
	gpuMaxMinDiffReduceQ  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);	
	
	/* blocksPerGrid = (gnTCell + gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (glimit, gnTCell, gnBFace, name); */
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuLimitespcell3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gespcell_MStream[name*gnTCell], gvol, gq, geps_tmp, ggam, gp_bar, gnTCell, gnBFace, name);
	
	cudaStreamSynchronize(flowstream[4]);			
	
    RecvSendVarNeighbor_Over_Gradient(hostbqs, hostbqr, bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);
	
	mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
	
	// for dqdy:				
    RecvSendVarNeighbor_Over_Gradient(&hostbqs[glenbqsr], &hostbqr[glenbqsr], bqs, bqr, req_send2, req_recv2, status_array2, nvar);

    MPI_Waitall(nNeighbor,req_recv2,status_array2);
    MPI_Waitall(nNeighbor,req_send2,status_array2);
	
	mfmem::sdel_array_1D(req_send2);
    mfmem::sdel_array_1D(req_recv2);
    mfmem::sdel_array_1D(status_array2);
	
	// for dqdz:			
    RecvSendVarNeighbor_Over_Gradient(&hostbqs[2*glenbqsr], &hostbqr[2*glenbqsr], bqs, bqr, req_send3, req_recv3, status_array3, nvar);

    MPI_Waitall(nNeighbor,req_recv3,status_array3);
    MPI_Waitall(nNeighbor,req_send3,status_array3);
		
    mfmem::sdel_array_1D(req_send3);
    mfmem::sdel_array_1D(req_recv3);
    mfmem::sdel_array_1D(status_array3);
	
	HANDLE_API_ERR(cudaMemcpyAsync(gbqr, hostbqr, 3*glenbqsr*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[0]));			
	
	//cuVencatLimiter_MultiStream_espcell(3);
	name = 3;
    blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	//gpuMaxMinDiffInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);
	
	// Manual reduction	
	gpuMaxMinDiff  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gC2F, gIndexC2F, gnFPC, gf2c, 
														gtype_bcr, gnTCell, gnBFace, name);
	
	gpuMaxMinDiffReduceQ  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);	
	
	/* blocksPerGrid = (gnTCell + gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (glimit, gnTCell, gnBFace, name); */
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuLimitespcell3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gespcell_MStream[name*gnTCell], gvol, gq, geps_tmp, ggam, gp_bar, gnTCell, gnBFace, name);
	
	//cuVencatLimiter_MultiStream_espcell(4);
	name = 4;
    blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	//gpuMaxMinDiffInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);
	
	// Manual reduction	
	gpuMaxMinDiff  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gC2F, gIndexC2F, gnFPC, gf2c, 
														gtype_bcr, gnTCell, gnBFace, name);
	
	gpuMaxMinDiffReduceQ  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);	
	
	/* blocksPerGrid = (gnTCell + gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (glimit, gnTCell, gnBFace, name); */
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuLimitespcell4 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gespcell_MStream[name*gnTCell], gvol, gq, geps_tmp, gp_bar, gnTCell, gnBFace, name);
	
	cudaStreamSynchronize(flowstream[0]);
	blocksPerGrid = (glenbqsr + threadsPerBlock - 1) / threadsPerBlock;		
	gpuAdd_RecvSend2_Gradient <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0]>>> (gdqdx, gdqdy, gdqdz, gbqr, gIndexbqsr2, glenbqsr);   	
	
    mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);
	
	cudaStreamSynchronize(flowstream[0]);
	cudaStreamSynchronize(flowstream[1]);
	cudaStreamSynchronize(flowstream[2]);
	cudaStreamSynchronize(flowstream[3]);
	cudaStreamSynchronize(flowstream[4]);
}

void PolyGrid::cuRecvSendVarNeighbor_TogethForGradient_T_InVis(IntType nvar){
	
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;
	
    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
	
	MPI_Request *req_send2=0, *req_recv2=0;
    MPI_Status *status_array2=0;

    status_array2 = NULL;
    req_send2     = NULL;
    req_recv2     = NULL;
    mfmem::snew_array_1D(status_array2, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send2,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv2,     nNeighbor,dmrfl);
	
	MPI_Request *req_send3=0, *req_recv3=0;
    MPI_Status *status_array3=0;

    status_array3 = NULL;
    req_send3     = NULL;
    req_recv3     = NULL;
    mfmem::snew_array_1D(status_array3, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send3,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv3,     nNeighbor,dmrfl);
	
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
	
    cuSet_RecvSend(bqs, bqr, nvar);

	// for dqdx:
	IntType blocksPerGrid = (glenbqsrSA + threadsPerBlock - 1) / threadsPerBlock;			
	gpuAdd_RecvSend_Gradient <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gbqs, gdtdx, gdtdy, gdtdz, gIndexbqsrSA, glenbqsrSA);	
	
	HANDLE_API_ERR(cudaMemcpyAsync(hostbqs, gbqs, 3*glenbqsrSA*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[1]));
	
	// Get parameters   
    RealFlow gam, alf_l, alf_n;
    this->GetData(&gam,    REAL_FLOW, 1, "gam");
    this->GetData(&alf_l,  REAL_FLOW, 1, "alf_l");
    this->GetData(&alf_n,  REAL_FLOW, 1, "alf_n");    

	RealFlow gamm1;
	IntType EntropyCorType = 4;

	this->GetData(&EntropyCorType, INT, 1, "EntropyCorType");
	gamm1 = gam - 1.0;
	if (EntropyCorType == 4) {
		// shock face or not
		//cuCalIsShockFace(this, IsShockFace);
		RealFlow mach00;
		this->GetData(&mach00, REAL_FLOW, 1, "mach");
		RealFlow pref = gp_bar*(1.0 + 0.5*(gam - 1.0)*mach00*mach00);
		
		RealFlow ThdShock = 0.5;    // threshold for shock face
		IntType blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
		gpuCalIsShockFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gIsShockFace, gq, gxfn, gyfn, gzfn, gf2c, 
															pref, ThdShock, gnTFace, gnTCell, gnBFace);
	}
#if (defined LOOPMERGE)	
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuInviscidFlux_merge_bface <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq, gf2c, gxfn, gyfn, gzfn,
								gvgn, gtype_bcr, gsteady, gnTFace, gnBFace, gnTCell, glimit, gdqdx, gdqdy, gdqdz,
								gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gflux, garea, gIsShockFace, gIsNormalFace,
								gamm1, gp_bar, alf_l, alf_n, EntropyCorType);

	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuInviscidFlux_merge_iface <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq, gf2c, gxfn, gyfn, gzfn,
								gvgn, gsteady, gnTFace, gnBFace, gnTCell, glimit, gdqdx, gdqdy, gdqdz,
								gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gflux, garea, gIsShockFace, gIsNormalFace,
								gamm1, gp_bar, alf_l, alf_n, EntropyCorType);
#else
	blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuSetQlQrWithQ <<< blocksPerGrid, threadsPerBlock >>> (gq, gql, gqr, gf2c, gnTFace, gnBFace, gnTCell);
	
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCalcuQlQrBFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0]  >>> (gql, gqr, gf2c, glimit, 
								gdqdx, gdqdy, gdqdz, gtype_bcr, gxfc, gyfc, gzfc, 
								gxcc, gycc, gzcc, gnTFace, gnBFace, gnTCell, gp_bar); 
	// Interior face cycle:							
	blocksPerGrid = ((gnTFace - gnBFace) + threadsPerBlock - 1) / threadsPerBlock;
	gpuCalcuQlQrInFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0]  >>> (gql, gqr, gf2c, glimit, 
								gdqdx, gdqdy, gdqdz, gxfc, gyfc, gzfc, 
								gxcc, gycc, gzcc, gnTFace, gnBFace, gnTCell, gp_bar); 
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuModQlQrBou <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0]  >>> (gq, gql, gqr, gxfn, gyfn, gzfn,
								gf2c, gvgn, gtype_bcr, gsteady, gnTFace, gnBFace, gnTCell);
								
	blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuRoeFlux <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0]  >>> (gql, gqr, gflux, garea, gxfn, gyfn, gzfn, gvgn,
							gIsShockFace, gIsNormalFace, gamm1, gp_bar, alf_l, alf_n, gnTFace, gsteady, EntropyCorType);

#endif
	// Load the fluxes to residuals	
#if (defined FaceColoring)	
	cout << "Add cuLoadFluxColor for MultiStream" << endl;
	exit(0);
	//cuLoadFluxColor(this, res, flux);				
#elif (defined Atomic)
	cout << "Add cuLoadFluxAtomic for MultiStream" << endl;
	exit(0);
	//cuLoadFluxAtomic(res, flux);
#elif (defined GroupColor)
	cout << "Add cuLoadFluxGroupColor for MultiStream" << endl;
	exit(0);
	/* if (this->GroupColorSuccess) {
		//cout << "grid->GroupColorSuccess: " << endl; 
		//exit(0);
		cuLoadFluxGroupColor(this, res, flux);
	}
	else{
		cout << "grid->GroupColorFail: " << endl; 
		cuLoadFluxAtomic(res, flux);
	} */
#else
	// Reduction:
	// Reduction ShareMemory will be included here.
	// cuLoadFlux(res, flux);
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	#if (defined ShareMemory)
		gpuLoadFluxShareMemory2 <<< blocksPerGrid, threadsPerBlock, 5*threadsPerBlock*sizeof(RealFlow), flowstream[0] >>> (
													gres, gflux, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell, threadsPerBlock);
	#else
		gpuLoadFlux <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gres, gflux, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell);
	#endif
#endif
	
	cudaStreamSynchronize(flowstream[1]);
	
    RecvSendVarNeighbor_Over_Gradient(hostbqs, hostbqr, bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);
	
	mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
	
	// for dqdy:				
    RecvSendVarNeighbor_Over_Gradient(&hostbqs[glenbqsrSA], &hostbqr[glenbqsrSA], bqs, bqr, req_send2, req_recv2, status_array2, nvar);

    MPI_Waitall(nNeighbor,req_recv2,status_array2);
    MPI_Waitall(nNeighbor,req_send2,status_array2);
	
	mfmem::sdel_array_1D(req_send2);
    mfmem::sdel_array_1D(req_recv2);
    mfmem::sdel_array_1D(status_array2);
	
	// for dqdz:			
    RecvSendVarNeighbor_Over_Gradient(&hostbqs[2*glenbqsrSA], &hostbqr[2*glenbqsrSA], bqs, bqr, req_send3, req_recv3, status_array3, nvar);

    MPI_Waitall(nNeighbor,req_recv3,status_array3);
    MPI_Waitall(nNeighbor,req_send3,status_array3);
		
    mfmem::sdel_array_1D(req_send3);
    mfmem::sdel_array_1D(req_recv3);
    mfmem::sdel_array_1D(status_array3);
	
	HANDLE_API_ERR(cudaMemcpyAsync(gbqr, hostbqr, 3*glenbqsrSA*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[1]));			

	cudaStreamSynchronize(flowstream[1]);
	
	blocksPerGrid = (glenbqsrSA + threadsPerBlock - 1) / threadsPerBlock;	
	gpuAdd_RecvSend2_Gradient <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gdtdx, gdtdy, gdtdz, gbqr, gIndexbqsr2SA, glenbqsrSA);
    
    mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);
	
	cudaStreamSynchronize(flowstream[0]);
	cudaStreamSynchronize(flowstream[1]);
}

void PolyGrid::cuRecvSendVarNeighbor_Togeth_SAForInterfaceData_unfold(IntType nvar){
		
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
	
	bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
	
    cuSet_RecvSend(bqs, bqr, nvar);
	
	IntType blocksPerGrid = (glenbqsrSA + threadsPerBlock - 1) / threadsPerBlock;		
	gpuAdd_RecvSend <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gbqs, gMPI, gIndexbqsrSA, glenbqsrSA);		
	
	HANDLE_API_ERR(cudaMemcpyAsync(&bqs[0][0][0], gbqs, glenbqsrSA*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[0]));

	//cuComputeTurbGeneration_SA(this);
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuComputeTurbGeneration_SA <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gomaga, gdqdx, gdqdy, gdqdz, gnTCell + gnBFace, gnTCell);
    //cuZeroGridResiduals(this, "res", 1); 
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuZeroGridResiduals <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gres, gnTCell);		
	
	RealFlow turb_cfl_times = 2.0;
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuSAdtlhsmat <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (glhsmat, gdt, gq, gvol, gnCPC, gIndexC2C, turb_cfl_times, gnTCell);
	
	cudaStreamSynchronize(flowstream[0]);
    RecvSendVarNeighbor_Over(bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);
	
	mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
	
	HANDLE_API_ERR(cudaMemcpyAsync(gbqr, &bqr[0][0][0], glenbqsrSA*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[0]));			
		   
    /* for(i=0; i<nvar; i++)
        Read_RecvSend(bqr, dqdx[i], i); */
	cudaStreamSynchronize(flowstream[0]);
	blocksPerGrid = (glenbqsrSA + threadsPerBlock - 1) / threadsPerBlock;	
	gpuAdd_RecvSend2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gMPI, gbqr, gIndexbqsr2SA, glenbqsrSA);		
	
	mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);
	
	cudaStreamSynchronize(flowstream[0]);
	cudaStreamSynchronize(flowstream[1]);
}

void PolyGrid::cuRecvSendVarNeighbor_TogethForGradient_SA_MultiStream(IntType nvar){
	
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
	
	MPI_Request *req_send2=0, *req_recv2=0;
    MPI_Status *status_array2=0;

    status_array2 = NULL;
    req_send2     = NULL;
    req_recv2     = NULL;
    mfmem::snew_array_1D(status_array2, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send2,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv2,     nNeighbor,dmrfl);
	
	MPI_Request *req_send3=0, *req_recv3=0;
    MPI_Status *status_array3=0;

    status_array3 = NULL;
    req_send3     = NULL;
    req_recv3     = NULL;
    mfmem::snew_array_1D(status_array3, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send3,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv3,     nNeighbor,dmrfl);
	
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
	
    cuSet_RecvSend(bqs, bqr, nvar);

	// for dqdx:
	IntType blocksPerGrid = (glenbqsrSA + threadsPerBlock - 1) / threadsPerBlock;		
	
	gpuAdd_RecvSend_Gradient <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gbqs, gdnutdx, gdnutdy, gdnutdz, gIndexbqsrSA, glenbqsrSA);	
	
	HANDLE_API_ERR(cudaMemcpyAsync(hostbqs, gbqs, 3*glenbqsrSA*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[0]));		
	
	int iexp = 15;
    RealFlow xminn;
    this->GetData(&iexp, INT, 1, "iexp", 0);
    //Note: (10.**(-iexp) is machine zero)
    xminn = pow(10.0, -iexp+1);
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuAddSourceSA <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gres, glhsmat, gq, gsa_nu, gvis_l, gomaga, gdist2wall,
														gvol, gIndexC2C, gf2c, xminn, gnTCell);
	
	cudaStreamSynchronize(flowstream[0]);
    RecvSendVarNeighbor_Over_Gradient(hostbqs, hostbqr, bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);
	
	mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
	
	// for dqdy:				
    RecvSendVarNeighbor_Over_Gradient(&hostbqs[glenbqsrSA], &hostbqr[glenbqsrSA], bqs, bqr, req_send2, req_recv2, status_array2, nvar);

    MPI_Waitall(nNeighbor,req_recv2,status_array2);
    MPI_Waitall(nNeighbor,req_send2,status_array2);
	
	mfmem::sdel_array_1D(req_send2);
    mfmem::sdel_array_1D(req_recv2);
    mfmem::sdel_array_1D(status_array2);
	
	// for dqdz:			
    RecvSendVarNeighbor_Over_Gradient(&hostbqs[2*glenbqsrSA], &hostbqr[2*glenbqsrSA], bqs, bqr, req_send3, req_recv3, status_array3, nvar);

    MPI_Waitall(nNeighbor,req_recv3,status_array3);
    MPI_Waitall(nNeighbor,req_send3,status_array3);
		
    mfmem::sdel_array_1D(req_send3);
    mfmem::sdel_array_1D(req_recv3);
    mfmem::sdel_array_1D(status_array3);
	
	HANDLE_API_ERR(cudaMemcpyAsync(gbqr, hostbqr, 3*glenbqsrSA*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[0]));	
	
	cudaStreamSynchronize(flowstream[0]);
	blocksPerGrid = (glenbqsrSA + threadsPerBlock - 1) / threadsPerBlock;		
	gpuAdd_RecvSend2_Gradient <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdtdx, gdnutdx, gdnutdy, gdnutdz, gIndexbqsr2SA, glenbqsrSA);
    
    mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);
	
	cudaStreamSynchronize(flowstream[0]);
	cudaStreamSynchronize(flowstream[1]);
}



#endif

void PolyGrid::cuRecvSendVarNeighbor_TogethForGradient_T(IntType nvar){
	
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
	
	MPI_Request *req_send2=0, *req_recv2=0;
    MPI_Status *status_array2=0;

    status_array2 = NULL;
    req_send2     = NULL;
    req_recv2     = NULL;
    mfmem::snew_array_1D(status_array2, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send2,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv2,     nNeighbor,dmrfl);
	
	MPI_Request *req_send3=0, *req_recv3=0;
    MPI_Status *status_array3=0;

    status_array3 = NULL;
    req_send3     = NULL;
    req_recv3     = NULL;
    mfmem::snew_array_1D(status_array3, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send3,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv3,     nNeighbor,dmrfl);
	
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
	
    cuSet_RecvSend(bqs, bqr, nvar);

	// for dqdx:
	IntType blocksPerGrid = (glenbqsrSA + threadsPerBlock - 1) / threadsPerBlock;			
	gpuAdd_RecvSend_Gradient <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gdtdx, gdtdy, gdtdz, gIndexbqsrSA, glenbqsrSA);	
	
	HANDLE_API_ERR(cudaMemcpy(hostbqs, gbqs, 3*glenbqsrSA*sizeof(RealFlow), cudaMemcpyDeviceToHost));				
	
    RecvSendVarNeighbor_Over_Gradient(hostbqs, hostbqr, bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);
	
	mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
	
	// for dqdy:				
    RecvSendVarNeighbor_Over_Gradient(&hostbqs[glenbqsrSA], &hostbqr[glenbqsrSA], bqs, bqr, req_send2, req_recv2, status_array2, nvar);

    MPI_Waitall(nNeighbor,req_recv2,status_array2);
    MPI_Waitall(nNeighbor,req_send2,status_array2);
	
	mfmem::sdel_array_1D(req_send2);
    mfmem::sdel_array_1D(req_recv2);
    mfmem::sdel_array_1D(status_array2);
	
	// for dqdz:			
    RecvSendVarNeighbor_Over_Gradient(&hostbqs[2*glenbqsrSA], &hostbqr[2*glenbqsrSA], bqs, bqr, req_send3, req_recv3, status_array3, nvar);

    MPI_Waitall(nNeighbor,req_recv3,status_array3);
    MPI_Waitall(nNeighbor,req_send3,status_array3);
		
    mfmem::sdel_array_1D(req_send3);
    mfmem::sdel_array_1D(req_recv3);
    mfmem::sdel_array_1D(status_array3);
	
	HANDLE_API_ERR(cudaMemcpy(gbqr, hostbqr, 3*glenbqsrSA*sizeof(RealFlow), cudaMemcpyHostToDevice));	
	
	blocksPerGrid = (glenbqsrSA + threadsPerBlock - 1) / threadsPerBlock;	
	gpuAdd_RecvSend2_Gradient <<< blocksPerGrid, threadsPerBlock >>> (gdtdx, gdtdy, gdtdz, gbqr, gIndexbqsr2SA, glenbqsrSA);
    
    mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);
	
}

void PolyGrid::cuRecvSendVarNeighbor_TogethForGradient_SA(IntType nvar){
	
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
	
	MPI_Request *req_send2=0, *req_recv2=0;
    MPI_Status *status_array2=0;

    status_array2 = NULL;
    req_send2     = NULL;
    req_recv2     = NULL;
    mfmem::snew_array_1D(status_array2, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send2,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv2,     nNeighbor,dmrfl);
	
	MPI_Request *req_send3=0, *req_recv3=0;
    MPI_Status *status_array3=0;

    status_array3 = NULL;
    req_send3     = NULL;
    req_recv3     = NULL;
    mfmem::snew_array_1D(status_array3, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send3,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv3,     nNeighbor,dmrfl);
	
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
	
    cuSet_RecvSend(bqs, bqr, nvar);

	// for dqdx:
	IntType blocksPerGrid = (glenbqsrSA + threadsPerBlock - 1) / threadsPerBlock;		
	
	gpuAdd_RecvSend_Gradient <<< blocksPerGrid, threadsPerBlock >>> (gbqs, gdnutdx, gdnutdy, gdnutdz, gIndexbqsrSA, glenbqsrSA);	
	
	HANDLE_API_ERR(cudaMemcpy(hostbqs, gbqs, 3*glenbqsrSA*sizeof(RealFlow), cudaMemcpyDeviceToHost));				
	
    RecvSendVarNeighbor_Over_Gradient(hostbqs, hostbqr, bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);
	
	mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
	
	// for dqdy:				
    RecvSendVarNeighbor_Over_Gradient(&hostbqs[glenbqsrSA], &hostbqr[glenbqsrSA], bqs, bqr, req_send2, req_recv2, status_array2, nvar);

    MPI_Waitall(nNeighbor,req_recv2,status_array2);
    MPI_Waitall(nNeighbor,req_send2,status_array2);
	
	mfmem::sdel_array_1D(req_send2);
    mfmem::sdel_array_1D(req_recv2);
    mfmem::sdel_array_1D(status_array2);
	
	// for dqdz:			
    RecvSendVarNeighbor_Over_Gradient(&hostbqs[2*glenbqsrSA], &hostbqr[2*glenbqsrSA], bqs, bqr, req_send3, req_recv3, status_array3, nvar);

    MPI_Waitall(nNeighbor,req_recv3,status_array3);
    MPI_Waitall(nNeighbor,req_send3,status_array3);
		
    mfmem::sdel_array_1D(req_send3);
    mfmem::sdel_array_1D(req_recv3);
    mfmem::sdel_array_1D(status_array3);
	
	HANDLE_API_ERR(cudaMemcpy(gbqr, hostbqr, 3*glenbqsrSA*sizeof(RealFlow), cudaMemcpyHostToDevice));	

	gpuAdd_RecvSend2_Gradient <<< blocksPerGrid, threadsPerBlock >>> (gdtdx, gdnutdx, gdnutdy, gdnutdz, gIndexbqsr2SA, glenbqsrSA);
    
    mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);
	
}



#endif