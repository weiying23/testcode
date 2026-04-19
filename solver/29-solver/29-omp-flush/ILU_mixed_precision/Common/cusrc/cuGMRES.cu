#include <stdio.h>
#include <iostream>
#include <fstream>

#include "temporal_discretisation_implicit.h"
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
#include "solver_turb_sa.h"
#include "turbulence.h"

// head files relying on condition-compiling
#ifdef MPICH
#include <mpi.h>
#endif

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif

#include <cuLUSGS.cuh>
#include <cuData.cuh>
#include <cuErrorReturn.cuh>
#include <cuViscidFlux.cuh>
//#include <cuLimit.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>

//dingxin
#ifdef TIMECOST
extern double* timecost;
extern double  time_flux, time_invis, time_roe, time_vis, time_calvis;
extern double  time_limiter;
extern double  time_gradient;
extern double  time_lusgs;
#endif
//TIMECOST

using namespace mflow;

using namespace gpuData;

#ifdef MPICH
    extern int myZone;
    extern int numprocs;
    extern MPI_Comm GridComm;  //for each grid, tangj
#endif

void cuMemoryPrepara(PolyGrid *grid){
	
	IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nT5    = 5*nTCell;
	IntType n = nTCell + nBFace;
	RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nT5, "res");
	RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
	
	HANDLE_API_ERR(cudaMemcpy(res, gres, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(dt, gdt, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	IntType vis_mode, vis_run=0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
	if(vis_mode != INVISCID){
        vis_run = 1;
	} 
	RealFlow *vis_l = NULL, *vis_t = NULL;
	if(vis_run){
		vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
		vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
	}
	
	HANDLE_API_ERR(cudaMemcpy(vis_l, gvis_l, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(vis_t, gvis_t, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
	
	HANDLE_API_ERR(cudaMemcpy(q[0], gq, n*sizeof(RealFlow), cudaMemcpyDeviceToHost));			
	HANDLE_API_ERR(cudaMemcpy(q[1], &gq[1*n], n*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	HANDLE_API_ERR(cudaMemcpy(q[2], &gq[2*n], n*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(q[3], &gq[3*n], n*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(q[4], &gq[4*n], n*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

void cuMemoryPreparaGMRESDebug(PolyGrid *grid){
	
	IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nT5    = 5*nTCell;
	IntType n = nTCell + nBFace;
	RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nT5, "res");
	RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
	
	HANDLE_API_ERR(cudaMemcpy(gres, res, 5*gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(gdt, dt, gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	IntType vis_mode, vis_run=0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
	if(vis_mode != INVISCID){
        vis_run = 1;
	} 
	RealFlow *vis_l = NULL, *vis_t = NULL;
	if(vis_run){
		vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
		vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
	}
	
	HANDLE_API_ERR(cudaMemcpy(gvis_l, vis_l, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(gvis_t, vis_t, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
	
	HANDLE_API_ERR(cudaMemcpy(gq, q[0], n*sizeof(RealFlow), cudaMemcpyHostToDevice));			
	HANDLE_API_ERR(cudaMemcpy(&gq[1*n], q[1], n*sizeof(RealFlow), cudaMemcpyHostToDevice));	
	HANDLE_API_ERR(cudaMemcpy(&gq[2*n], q[2], n*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[3*n], q[3], n*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[4*n], q[4], n*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
}

void cuMemoryPreparaGMRESDebug2(PolyGrid *grid){
	
	IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
	IntType n = nTCell + nBFace;	
	
	RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
	
	HANDLE_API_ERR(cudaMemcpy(q[0], gq, n*sizeof(RealFlow), cudaMemcpyDeviceToHost));			
	HANDLE_API_ERR(cudaMemcpy(q[1], &gq[1*n], n*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	HANDLE_API_ERR(cudaMemcpy(q[2], &gq[2*n], n*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(q[3], &gq[3*n], n*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(q[4], &gq[4*n], n*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

void cuGMRESdqInit(RealFlow *dq, IntType nT){ 
  	
	//IntType nT5 = 5*gnTCell;
	IntType blocksPerGrid = (nT + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDQInit <<< blocksPerGrid, threadsPerBlock >>> (gdq, nT);
	//HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, nT5*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpures2reso(RealFlow *reso, RealFlow *res, IntType nT){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nT){
		reso[i] = res[i];
	}
	
}

void cures2reso(RealFlow *res, RealFlow *reso, IntType nT){  	
	
	//IntType nT5 = 5*gnTCell;
	IntType blocksPerGrid = (nT + threadsPerBlock - 1) / threadsPerBlock;	
	gpures2reso <<< blocksPerGrid, threadsPerBlock >>> (greso, gres, nT);
	/*if (name == 4)
		HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, 5*Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	*/
}

void cuPreconditLUSGS(PolyGrid *grid, RealFlow *Diag, IntType level)
{
    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  nTotal = nTCell + nBFace;
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");        
    
    IntType sweeps = 1;
    grid->GetData(&sweeps, INT, 1, "sweeps");
    RealFlow epsilon = 0.1;
    grid->GetData(&epsilon, REAL_FLOW, 1, "epsilon"); 
    if(epsilon < TINY) epsilon = 0.1;
    
    // Get number of faces for each cell
    IntType *nFPC = CalnFPC(grid);
    // Get cell to face connectivity
    IntType **C2F = CalC2F(grid); 
      
    IntType  i, j;

    // Allocate memories for RHS or DQ
    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTotal, "DQ");
    assert(DQ[0] != 0);
    for(i=1; i<5; i++) DQ[i] = &DQ[i-1][nTotal];
    for(j=0; j<5*nTotal; j++) DQ[0][j] = 0.; 

    if(sweeps == 1){
        // Copy the residual to DQ
		for(i=0; i<5; i++){
			cures2DQ(DQ, res, i);
        }	
        // Now the LU-SGS part
        cuSolveLUSGS3D(grid, Diag, DQ, nFPC, C2F, level);
    }else{
        RealFlow *rhs[5];
        rhs[0] = res;
        for(j=1; j<5; j++) rhs[j] = &rhs[j-1][nTCell];
        // Now the LU-SGS part ,   DQ conservative variable
        SolveLUSGS3D(grid, Diag, DQ, rhs, nFPC, C2F, sweeps, epsilon, level);
    }
}

__global__ void gpuvDQoInit(RealFlow *v, RealFlow *DQo, const RealFlow *DQ, const IntType j, 
						const IntType nBFace, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		RealFlow tmp = DQ[j*(nTCell + nBFace) + i];
		DQo[j*nTCell + i] = tmp;
		v[j*nTCell + i] = tmp;		
	}
	
}

void cuvDQoInit(IntType nvar){  	
		
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	for(IntType j=0; j<nvar; j++){
		gpuvDQoInit <<< blocksPerGrid, threadsPerBlock >>> (gv, gDQo, gDQ, j, gnBFace, gnTCell);
	}
}

__global__ void gpuCalv(RealFlow *v, const RealFlow norm, const IntType len){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < len){
		v[i] /= norm;		
	}
	
}

void cuCalv(RealFlow norm){  	
		
	IntType blocksPerGrid = (5*gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	
	gpuCalv <<< blocksPerGrid, threadsPerBlock >>> (gv, norm, 5*gnTCell);
	
}



__global__ void gpuADUDQrhs(RealFlow *DQ, RealFlow *rhs, const RealFlow *v, const RealFlow *Diag, 
						const IntType j, const IntType nBFace, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		RealFlow tmp = v[j*nTCell + i];
		DQ[j*(nTCell + nBFace) + i] = tmp;
		rhs[j*nTCell + i] = Diag[i]*tmp;
	}
	
}

void cuADUDQrhs(IntType k, IntType type){  	
		
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	if (type == 0){
		for(IntType j = 0; j < 5; j++){
			gpuADUDQrhs <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gres, &gv[k*5*gnTCell], gDiag, j, gnBFace, gnTCell);
		}
	}
	else if (type == 1) {
		for(IntType j = 0; j < 5; j++){
			gpuADUDQrhs <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gres, gdq, gDiag, j, gnBFace, gnTCell);
		}
	}
		
}

void cuComputeADU(PolyGrid *grid, RealFlow *Diag, RealFlow *v, RealFlow *res, IntType level, IntType k, IntType type)
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nTotal = nTCell + nBFace;   
    
    IntType sweeps = 1;
    grid->GetData(&sweeps, INT, 1, "sweeps");
    RealFlow epsilon = 0.1;
    grid->GetData(&epsilon, REAL_FLOW, 1, "epsilon"); 
    if(epsilon < TINY) epsilon = 0.1;
    
    // Get number of faces for each cell
    IntType *nFPC = CalnFPC(grid);
    // Get cell to face connectivity
    IntType **C2F = CalC2F(grid); 
      
    IntType i;
    
    RealFlow *rhs[5];
    rhs[0] = res;
    for(i=1; i<5; i++) rhs[i] = &rhs[i-1][nTCell];
    
    // Allocate memories for RHS or DQ
    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTotal, "DQ");
    assert(DQ[0] != 0);
    for(i=1; i<5; i++) DQ[i] = &DQ[i-1][nTotal]; 
	
    //for(j=0; j<5*nTotal; j++) DQ[0][j] = 0.; 

    cuADUDQrhs(k, type);
	HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(rhs[0], gres, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	/*
    for(j=0; j<5; j++){
        for(i=0; i<nTCell; i++){
            // Copy the v to DQ
            DQ[j][i] = v[j*nTCell + i];
            //计算当前单元对ADU的贡献
            rhs[j][i] = Diag[i]*DQ[j][i];
        }
    }
	*/
	
    // IF MPICH, we could exchange DQ here for INTERFACES
#ifdef MPICH
        IntType nvar = 5;
        RealFlow *q_mpi[5];
        for(IntType j=0; j<5; j++)
            q_mpi[j] = DQ[j];
        grid->RecvSendVarNeighbor_Togeth(nvar, q_mpi);
#endif

    if(sweeps == 1){
        //计算相邻单元对ADU的贡献
		for (IntType i = 0; i < 5; i++){
			HANDLE_API_ERR(cudaMemcpy(&gDQ[i*(nTCell + nBFace) + nTCell], &DQ[i][nTCell], nBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
		}
        cuSolveADU3D(grid, rhs, DQ, nFPC, C2F, level);
		//HANDLE_API_ERR(cudaMemcpy(rhs[0], gres, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
    }
	
}

__global__ void gpuDQ2w(RealFlow *w, const RealFlow *DQ, const IntType j, 
						const IntType nBFace, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		w[j*nTCell + i] = DQ[j*(nTCell + nBFace) + i];	
	}
	
}

void cuDQ2w(IntType nvar){  	
		
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	for(IntType j=0; j<nvar; j++){
		gpuDQ2w <<< blocksPerGrid, threadsPerBlock >>> (gw, gDQ, j, gnBFace, gnTCell);
	}
}

__global__ void gpuwSubtractHv(RealFlow *w, const RealFlow *v, const RealFlow Hjk, 
						const IntType j, const IntType nvar, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nvar*nTCell){
		w[i] -= Hjk*v[j*nvar*nTCell + i];	
	}
	
}

void cuwSubtractHv(RealFlow **H, IntType k, IntType nvar){  	
		
	IntType blocksPerGrid = (nvar*gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	for(IntType j=0; j<=k; j++){
		gpuwSubtractHv <<< blocksPerGrid, threadsPerBlock >>> (gw, gv, H[j][k], j, nvar, gnTCell);
	}
}

__global__ void gpuUpdateNewvk(RealFlow *v, const RealFlow *w, const RealFlow norm, 
						const IntType k, IntType nvar, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nvar*nTCell){
		v[(k + 1)*nvar*nTCell + i] = norm*w[i];	
	}
	
}

void cuUpdateNewvk(RealFlow norm, IntType k, IntType nvar){  	
		
	IntType blocksPerGrid = (nvar*gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	for(IntType j=0; j<=k; j++){
		gpuUpdateNewvk <<< blocksPerGrid, threadsPerBlock >>> (gv, gw, norm, k, nvar, gnTCell);
	}
}

__global__ void gpuCaldq(RealFlow *dq, const RealFlow *v, const RealFlow sk, 
						const IntType k, IntType nvar, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nvar*nTCell){
		dq[i] += sk*v[k*nvar*nTCell + i];	
	}
	
}

void cuCaldq(RealFlow *s, IntType kspan, IntType nvar){  	
		
	IntType blocksPerGrid = (nvar*gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	for(IntType k=0; k<kspan; k++){
		gpuCaldq <<< blocksPerGrid, threadsPerBlock >>> (gdq, gv, s[k], k, nvar, gnTCell);
	}
}

__global__ void gpuUpdatev(RealFlow *v, const RealFlow *DQ, const RealFlow *DQo, 
						const IntType j, const IntType nBFace, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		v[j*nTCell + i] = DQo[j*nTCell + i] - DQ[j*(nTCell + nBFace) + i];	
	}
	
}

void cuUpdatev(IntType nvar){  	
		
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	for(IntType j=0; j<nvar; j++){
		gpuUpdatev <<< blocksPerGrid, threadsPerBlock >>> (gv, gDQ, gDQo, j, gnBFace, gnTCell);
	}
}

__global__ void gpuUpdateDQ(RealFlow *DQ, const RealFlow *dq, 
						const IntType j, const IntType nBFace, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		DQ[j*(nTCell + nBFace) + i] = dq[j*nTCell + i];	
	}
	
}

void cuUpdateDQ(IntType nvar){  	
		
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	for(IntType j=0; j<nvar; j++){
		gpuUpdateDQ <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gdq, j, gnBFace, gnTCell);
	}
}

void cureso2res(IntType nvar){  	
	
	IntType nT5 = nvar*gnTCell;
	IntType blocksPerGrid = (nT5 + threadsPerBlock - 1) / threadsPerBlock;	
	gpures2reso <<< blocksPerGrid, threadsPerBlock >>> (gres, greso, nT5);

}

template <unsigned int blockSize>
__device__ void warpReduce(volatile double *sdata, unsigned int tid){
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void ReduceGMRES(double *g_idata, double *g_odata, unsigned int n){
	
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	
	while (i < n){
		sdata[tid] += g_idata[i] + g_idata[i + blockSize];
		i += gridSize;
	}
	__syncthreads();
	
	if(blockSize >= 512){
		if(tid < 256){
			sdata[tid] += sdata[tid + 256];
		}
		__syncthreads();
	}
	if(blockSize >= 256){
		if(tid < 128){
			sdata[tid] += sdata[tid + 128];
		}
		__syncthreads();
	}
	if(blockSize >= 128){
		if(tid < 64){
			sdata[tid] += sdata[tid + 64];
		}
		__syncthreads();
	}
	
	if(tid < 32) 
		warpReduce<512>(sdata, tid);
	if(tid == 0)
		g_odata[blockIdx.x] = sdata[0];
	
	
}

/******************************************************************************\
   Calculate the dot product of two vectors with length n
\******************************************************************************/
RealFlow cuDotProduct(RealFlow *a, RealFlow *b, IntType n)
{
    // a=w, b=v[k]
	IntType  i;
    RealFlow sum = 0.;
	
	
    for(i=0; i<n; i++) sum += a[i]*b[i];
    return sum;
}


/******************************************************************************\
   Calculate the dot product of two vectors with length n
\******************************************************************************/
__global__ void gpuDotProductMPI(RealFlow *sumv, const RealFlow *v, 
						const IntType n){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < n){
		sumv[i] = v[i]*v[i];	
	}
	
}

RealFlow cuDotProductMPI(RealFlow *a, RealFlow *b, IntType n)
{
    IntType  i;
    RealFlow sum = 0.0, sum_glb=0.0;
	
	IntType blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDotProductMPI <<< blocksPerGrid, threadsPerBlock >>> (gsumv, gv, n);
	RealFlow *sumv = NULL;
    mfmem::snew_array_1D(sumv, n, dmrfl);
	RealFlow *odata = NULL;
    mfmem::snew_array_1D(odata, gnodata, dmrfl);
	HANDLE_API_ERR(cudaMemcpy(sumv, gsumv, n*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	//const IntType blockSIze = threadsPerBlock;
	blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	ReduceGMRES <512> <<< gnodata, threadsPerBlock, threadsPerBlock*sizeof(RealFlow) >>> (gsumv, godata, gnsum);
	HANDLE_API_ERR(cudaMemcpy(odata, godata, gnodata*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	for(i=0; i<gnodata; i++) sum += odata[i];
	ofstream outfile;
	outfile.open("odata.dat");
	outfile << setprecision(32);
	for(IntType i = 0; i < gnodata; i++){
		outfile << i << ": " << odata[i] << endl;
	}
	outfile.close();
    //for(i=0; i<n; i++) sum += sumv[i];
#ifdef MPICH
    MPI_Allreduce(&sum, &sum_glb, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    sum = sum_glb;
#endif
    return sum;
}

__global__ void Reducekernel(double *g_idata, double *g_odata){
	
	__shared__ double sdata[512];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;

	sdata[tid] = g_idata[i];	
	__syncthreads();
	
	
	for(unsigned int s = 1; s < blockDim.x; s *= 2){
		if ((tid%(2*s)) == 0){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	if(tid == 0)
		g_odata[blockIdx.x] = sdata[0];
	
	
}

__global__ void Reducekernel2(double *g_idata, double *g_odata){
	
	//__shared__ double sdata[512];
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;

	sdata[tid] = g_idata[i];	
	__syncthreads();
	
	for(unsigned int s = 1; s < blockDim.x; s *= 2){
		int index = 2*s*tid;
		if (index < blockDim.x){
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}
	
	if(tid == 0)
		g_odata[blockIdx.x] = sdata[0];
	
	
}

__global__ void Reducekernel3(double *g_idata, double *g_odata){
	
	//__shared__ double sdata[512];
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;

	sdata[tid] = g_idata[i];	
	__syncthreads();
	
	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
		if (tid < s){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	if(tid == 0)
		g_odata[blockIdx.x] = sdata[0];
	
	
}

__global__ void Reducekernel4(double *g_idata, double *g_odata, int n){
	
	//__shared__ double sdata[512];
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + tid;

	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];		
	__syncthreads();
	
	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
		if (tid < s){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	if(tid == 0)
		g_odata[blockIdx.x] = sdata[0];
	
	
}

__device__ void warpReduce5(volatile double *sdata, unsigned int tid){
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void Reducekernel5(double *g_idata, double *g_odata, int n){
	
	//__shared__ double sdata[512];
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + tid;

	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();
	
	for(unsigned int s = blockDim.x/2; s > 32; s >>= 1){
		if (tid < s){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	if (tid < 32) warpReduce5(sdata, tid);
	
	if(tid == 0)
		g_odata[blockIdx.x] = sdata[0];	
	
}

__global__ void Reducekernel6(double *g_idata, double *g_odata, int n){
	
	//__shared__ double sdata[512];
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + tid;

	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();
	
	IntType blockSize = blockDim.x;
	if(blockSize >= 512){
		if(tid < 256){
			sdata[tid] += sdata[tid + 256];
		}
		__syncthreads();
	}
	if(blockSize >= 256){
		if(tid < 128){
			sdata[tid] += sdata[tid + 128];
		}
		__syncthreads();
	}
	if(blockSize >= 128){
		if(tid < 64){
			sdata[tid] += sdata[tid + 64];
		}
		__syncthreads();
	}
	
	if(tid < 32) warpReduce<512>(sdata, tid);	
	
	if(tid == 0)
		g_odata[blockIdx.x] = sdata[0];	
	
}

__global__ void Reducekernel_sum(double *val_Reduction, double *g_odata, int n){
	
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < 1){
		for (int j = 1; j < n; j++){
			g_odata[0] += g_odata[j];
		}
		val_Reduction[0] = g_odata[0];
	}
	
}

RealFlow cuDotProductMPIkernel4567(RealFlow *odata, RealFlow *b, IntType n, IntType k)
{
    RealFlow sum = 0.0, sum_glb=0.0;
	
	IntType blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDotProductMPI <<< blocksPerGrid, threadsPerBlock >>> (gsumv2, &gv[k*n], n);
	
	blocksPerGrid = gnodata2;
	Reducekernel6 <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gsumv2, godata2, gnsum2);
	/* HANDLE_API_ERR(cudaMemcpy(odata, godata2, blocksPerGrid*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	for(i=0; i<blocksPerGrid; i++) sum += odata[i]; */
	
	IntType blocksPerGrid2 = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;	
	Reducekernel_sum <<< blocksPerGrid2, threadsPerBlock >>> (val_Reduction, godata2, blocksPerGrid);
	//HANDLE_API_ERR(cudaMemcpy(odata, godata2, 1*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//sum = odata[0];
	cudaDeviceSynchronize();
	sum = val_Reduction[0];
	
#ifdef MPICH
    MPI_Allreduce(&sum, &sum_glb, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    sum = sum_glb;
#endif
	
    return sum;
}

RealFlow cuDotProductMPIkernel4567ww(RealFlow *odata, RealFlow *b, IntType n, IntType k)
{

    RealFlow sum = 0.0, sum_glb=0.0;
	
	IntType blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDotProductMPI <<< blocksPerGrid, threadsPerBlock >>> (gsumv2, gw, n);
	
	blocksPerGrid = gnodata2;
	Reducekernel6 <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gsumv2, godata2, gnsum2);
	//HANDLE_API_ERR(cudaMemcpy(odata, godata2, blocksPerGrid*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	//for(i=0; i<blocksPerGrid; i++) sum += odata[i];
	
	IntType blocksPerGrid2 = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;	
	Reducekernel_sum <<< blocksPerGrid2, threadsPerBlock >>> (val_Reduction, godata2, blocksPerGrid);
	//HANDLE_API_ERR(cudaMemcpy(odata, godata2, 1*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//sum = odata[0];
	cudaDeviceSynchronize();
	sum = val_Reduction[0];
	
#ifdef MPICH
    MPI_Allreduce(&sum, &sum_glb, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    sum = sum_glb;
#endif
    return sum;
}

RealFlow cuDotProductMPIkernel(RealFlow *odata, RealFlow *b, IntType n)
{
    IntType  i;
    RealFlow sum = 0.0, sum_glb=0.0;
	
	IntType blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDotProductMPI <<< blocksPerGrid, threadsPerBlock >>> (gsumv, gv, n);
    
	blocksPerGrid = (gnsum + threadsPerBlock - 1) / threadsPerBlock;
	
	Reducekernel3 <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gsumv, godata);
	HANDLE_API_ERR(cudaMemcpy(odata, godata, blocksPerGrid*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	for(i=0; i<blocksPerGrid; i++) sum += odata[i];
	
#ifdef MPICH
    MPI_Allreduce(&sum, &sum_glb, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    sum = sum_glb;
#endif

    return sum;
}

__global__ void gpuDotProduct(RealFlow *sumv, const RealFlow *w, const RealFlow *v, 
						const IntType n){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < n){
		sumv[i] = w[i]*v[i];	
	}
	
}

RealFlow cuDotProductkernel4567(RealFlow *odata, RealFlow *b, IntType n, IntType k)
{
    RealFlow sum = 0.0;
	
	IntType blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDotProduct <<< blocksPerGrid, threadsPerBlock >>> (gsumv2, gw, &gv[k*n], n);
		
	blocksPerGrid = gnodata2;
	Reducekernel6 <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gsumv2, godata2, gnsum2);
	/* HANDLE_API_ERR(cudaMemcpy(odata, godata2, blocksPerGrid*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	for(i=0; i<blocksPerGrid; i++) sum += odata[i]; */
	
	IntType blocksPerGrid2 = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;	
	Reducekernel_sum <<< blocksPerGrid2, threadsPerBlock >>> (val_Reduction, godata2, blocksPerGrid);
	//HANDLE_API_ERR(cudaMemcpy(odata, godata2, 1*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//sum = odata[0];
	cudaDeviceSynchronize();
	sum = val_Reduction[0];
	
    return sum;
}

void cuGMRESSolverOrig(PolyGrid *grid, IntType level){
	
	//cuMemoryPrepara(grid);
#ifdef FS_CUDA_DEBUG_NS_GMRES
	cuMemoryPreparaGMRESDebug(grid);
#endif
	
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nT5    = 5*nTCell;

    // We haven't consider turbulence model yet.
    IntType nvar = 5;
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nT5, "res");
    RealFlow *reso  = NULL;
    mfmem::snew_array_1D(reso, nT5,dmrfl);
    assert(reso != 0);
 
    // Control parameters
    IntType i, j, k, sweep, Adu=1, kspan = 10, Nsweeps = 5;
    grid->GetData(&Adu, INT, 1, "ADU");
    grid->GetData(&kspan, INT, 1, "kspan");
    grid->GetData(&Nsweeps, INT, 1, "gmresweeps");
    RealFlow Error = 0.;
    grid->GetData(&Error, REAL_FLOW, 1, "gmresepsilon");
    if(Error < TINY) Error = 1.0e-2;
 
    // Temporary memories
    IntType len = nvar*nTCell;
    RealFlow *dq = NULL;
    mfmem::snew_array_1D(dq, len,dmrfl);
    assert(dq != 0);

    RealFlow **H = NULL;
    mfmem::snew_array_2D(H, kspan+1,kspan,dmrfl,true);
#ifdef MPICH
    RealFlow *Htmp   = NULL;
    RealFlow *Htotal = NULL;
    mfmem::snew_array_1D(Htmp, kspan,dmrfl);
    mfmem::snew_array_1D(Htotal, kspan,dmrfl);
    for(i=1; i<kspan; i++) {
        Htmp[i] = 0.0;
        Htotal[i] = 0.0;
    }
#endif
 
    RealFlow **v = NULL;
    mfmem::snew_array_2D(v, kspan+1,len,dmrfl,true);

    RealFlow *w  = NULL;
    RealFlow *cs = NULL;
    RealFlow *sn = NULL;
    RealFlow *s  = NULL;
    mfmem::snew_array_1D(w, len,dmrfl);
    mfmem::snew_array_1D(cs, kspan,dmrfl);
    mfmem::snew_array_1D(sn, kspan,dmrfl);
    mfmem::snew_array_1D(s, kspan+1,dmrfl);
	
	RealFlow *odata = NULL;    
	mfmem::snew_array_1D(odata, gnodata2, dmrfl);
 
    RealFlow norm0, norm, dmax;
 
    // Save the beginning flow variables
    RealGeom *vol  =  grid->GetCellVol();

    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*(nTCell+nBFace), "DQ");
    if(!DQ[0]) {
        mfmem::snew_array_1D(DQ[0], 5*(nTCell+nBFace),dmrfl);
        grid->UpdateDataPtr(DQ[0], REAL_FLOW, 5*(nTCell+nBFace), "DQ");
    }
    assert(DQ[0] != 0);
    for(i=1; i<nvar; i++) DQ[i] = &DQ[i-1][nTCell+nBFace];
	
	cuDQInit(nvar);	
	
    RealFlow *DQo[5];
    DQo[0] = NULL;
    mfmem::snew_array_1D(DQo[0], 5*nTCell,dmrfl);
    for(i=1; i<nvar; i++) DQo[i] = &DQo[i-1][nTCell];
	/*
	for(i=0; i<len; i++){
        dq[i]   = 0.;
    }
	*/
	cuGMRESdqInit(dq, len);
	//HANDLE_API_ERR(cudaMemcpy(dq, gdq, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
    // Save the residuals and Initialize Matrix*(Delta q) and p[0]
	/*
	for(i=0; i<nT5; i++) 
        reso[i] = res[i];
	*/
	cures2reso(res, reso, len);
	//HANDLE_API_ERR(cudaMemcpy(reso, greso, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
    // Now diagonal term in LU-SGS, here we need information of time steps
    //在一个GMERES的子迭代中,该值保持不变
    RealFlow *Diag = NULL;
    mfmem::snew_array_1D(Diag, nTCell,dmrfl);
    assert(Diag != 0);
    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
	/*
	for(i=0; i<nTCell; i++) 
        Diag[i] = vol[i]/dt[i];
    CalDiagLUSGS(grid, Diag, level);
	*/
	cuDiagInit(dt);
	cuCalDiagLUSGS(grid, level);
	//HANDLE_API_ERR(cudaMemcpy(Diag, gDiag, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));		
    
    //PreconditLUSGS(grid, Diag, level);
	cuPreconditLUSGS(grid, Diag, level);
	//HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	
	ofstream outfile;
	
	cuvDQoInit(nvar);
	//HANDLE_API_ERR(cudaMemcpy(DQo[0], gDQo, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	////HANDLE_API_ERR(cudaMemcpy(v[0], gv, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	
	/*
	for(j=0; j<nvar; j++){
        for(i=0; i<nTCell; i++){
            DQo[j][i] = DQ[j][i];
            v[0][j*nTCell + i] = DQo[j][i];
        }
    }
    */
	
    norm0 = cuDotProductMPIkernel4567(odata, v[0], len, 0);
	//setprecision(32);
	//cout << "norm0: " << setprecision(32) << norm0 << endl;
	//exit(0);
    norm0 = sqrt(norm0);
#ifdef MPICH
    //if(myZone==1) printf("Norm = %.5e\n", norm0);
#else
    //printf("Norm = %.5e\n", norm0);
#endif    
 
    if(norm0 > 1.0e-10){
        // loop over GMRES sweeps
        //Nsweeps stands for the loop times before restarting.
        norm = norm0;
        for(sweep=0; sweep<Nsweeps; sweep++){
			//HANDLE_API_ERR(cudaMemcpy(gv, v[0], 5*gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));	
            for(k=0; k<kspan+1; k++) s[k]=0.0;
            s[0] = norm; 
			/*
			for(i=0; i<len; i++){ 
                v[0][i] /= norm;               //v=gamma/beta
			}
			*/

			cuCalv(norm);
			////HANDLE_API_ERR(cudaMemcpy(v[0], gv, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	

			// Loop over the search directions
            for(k=0; k<kspan; k++){
				
                // Calculate the epsilon in evaluating matrix * vector
				// tmp memory transfer:
				
                //选择不同的A*V的方法,1--luo的简化,2--原始的矩阵直接求解,3--差分近似
                if(Adu == 1){
					//HANDLE_API_ERR(cudaMemcpy(&gv[k*5*gnTCell], v[k], 5*gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));		
                    cuComputeADU(grid, Diag, v[k], res, level, k, 0);
				}
                else if(Adu == 2)
                    ComputeADU2(grid, Diag, v[k], res, level);
                else if(Adu == 3)
                    ComputeADU3(grid, v[k], res, reso, level);
                else if(Adu == 4)
                    ResLUSGS(grid, v[k], level);

                cuPreconditLUSGS(grid, Diag, level);
				//HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
                
                cuDQ2w(nvar);
				//HANDLE_API_ERR(cudaMemcpy(w, gw, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
				/*
                for(j=0; j<nvar; j++){
                    for(i=0; i<nTCell; i++){
                        w[j*nTCell + i] = DQ[j][i];                        
                    }
                }
                */     
                // Calculate H
                for(j=0; j<=k; j++){
                    //H[j][k] = DotProduct(w, v[j], len, j);
					H[j][k] = cuDotProductkernel4567(odata, v[j], len, j);
                }
#ifdef MPICH
                //需要并行传递H的值
                for(j=0; j<=k; j++) Htmp[j] = H[j][k];
                for(j=0; j<kspan; j++) Htotal[j] = 0.;
                MPI_Allreduce(Htmp, Htotal, kspan, MPIReal, MPI_SUM, MPI_COMM_WORLD);
                for(j=0; j<=k; j++) H[j][k] = Htotal[j];
#endif
                cuwSubtractHv(H, k, nvar);
				//HANDLE_API_ERR(cudaMemcpy(w, gw, nvar*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
				/*
				for(j=0; j<=k; j++){
                    for(i=0; i<len; i++) 
                        w[i] -= H[j][k]*v[j][i];
                }
				*/
				
                //norm  = sqrt(DotProductMPI(w, w, len));
				norm  = sqrt(cuDotProductMPIkernel4567ww(odata, w, len, 0));
                H[k+1][k] = norm;
                norm  = 1.0/norm;
				
				cuUpdateNewvk(norm, k, nvar);
				////HANDLE_API_ERR(cudaMemcpy(v[k + 1], &gv[(k + 1)*nvar*gnTCell], nvar*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
				/*
                for(i=0; i<len; i++) 
                    v[k+1][i] = w[i]*norm;
				*/
				
                // Solve the linear least square problems
                for (j=0; j<k; j++)
                    ApplyPlaneRotation(H[j][k], H[j+1][k], cs[j], sn[j]);
                
                GeneratePlaneRotation(H[k][k], H[k+1][k], cs[k], sn[k]);
                ApplyPlaneRotation(H[k][k], H[k+1][k], cs[k], sn[k]);
                ApplyPlaneRotation(s[k], s[k+1], cs[k], sn[k]);

                //在该循环内,可以用V[k+1]的空间来储存W变量,节省内存使用量
            }
                        
            //完成Updata计算后，s[0]-s[k-1]中储存的数值为y[0]-y[k-1]
            ComputeY(H, s, kspan);
		
            // Calculate the Delta q
			cuCaldq(s, kspan, nvar);
			//HANDLE_API_ERR(cudaMemcpy(dq, gdq, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
			/*
            for(k=0; k<kspan; k++)
                for(i=0; i<len; i++) 
                    dq[i] += v[k][i]*s[k];
            */
			
            // Calculate Matrix*(Delta q) and P[0] for next sweep
            
            //选择不同的A*V的方法,1--luo的简化,2--原始的矩阵直接求解,3--差分近似                
            if(Adu == 1)
                cuComputeADU(grid, Diag, dq, res, level, k, 1);
            else if(Adu == 2)
                ComputeADU2(grid, Diag, dq, res, level);
            else if(Adu == 3)
                ComputeADU3(grid, dq, res, reso, level);
            else if(Adu == 4)
                ResLUSGS(grid, dq, level);

            // Calculate Matrix*(Delta q) and P[0] for next sweep
            cuPreconditLUSGS(grid, Diag, level);
			//HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));	

			cuUpdatev(nvar);
			////HANDLE_API_ERR(cudaMemcpy(v[0], gv, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
			/*
            for(j=0; j<nvar; j++){
                for(i=0; i<nTCell; i++){
                    v[0][j*nTCell + i] = DQo[j][i] - DQ[j][i];
                    //count++;
                }
            }
			*/
			

            // Check if the solution of linear eqs has been obtained within the scope 
            norm = cuDotProductMPIkernel4567(odata, v[0], len, 0);
            norm = sqrt(norm);
//          if(sweep==0) norm0=norm;
            dmax = norm/norm0;
#ifdef MPICH
            //if(myZone==1) printf("Resi reduced by %.4e with %d sweeps\n", dmax, (int)(sweep+1));
#else
            //printf("Resi reduced by %.4e with %d sweeps\n", dmax, (int)(sweep+1));
#endif
            
            if(dmax < Error){
                sweep++;
                break;
            }
        }
    
        // update the solution
		cuUpdateDQ(nvar);
		//HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		/*
        count = 0;
        for(j=0; j<nvar; j++){
            for(i=0; i<nTCell; i++){
                DQ[j][i] = dq[j*nTCell + i];
                //count++;
            }
        }
		*/
        cuUpdateFlowField3D_CFL3d(grid, DQ);
		
		cureso2res(nvar);
		/*
        for(i=0; i<nT5; i++) 
            res[i] = reso[i];
		*/
		
    }
	/*
	RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
	HANDLE_API_ERR(cudaMemcpy(gq, q[0], n*sizeof(RealFlow), cudaMemcpyHostToDevice));			
	HANDLE_API_ERR(cudaMemcpy(&gq[1*n], q[1], n*sizeof(RealFlow), cudaMemcpyHostToDevice));	
	HANDLE_API_ERR(cudaMemcpy(&gq[2*n], q[2], n*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[3*n], q[3], n*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[4*n], q[4], n*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(gres, res, 5*gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));	
	*/
	
#ifdef FS_CUDA_DEBUG_NS_GMRES
	cuMemoryPreparaGMRESDebug2(grid);
#endif
	
	mfmem::sdel_array_1D(odata);
    // Delete temporary memories
    mfmem::sdel_array_1D(dq);
    mfmem::sdel_array_1D(Diag);
    mfmem::sdel_array_1D(reso);
    mfmem::sdel_array_1D(DQo[0]);
    mfmem::sdel_array_2D(H);
    mfmem::sdel_array_2D(v);
    mfmem::sdel_array_1D(w);
    mfmem::sdel_array_1D(cs);
    mfmem::sdel_array_1D(sn);
    mfmem::sdel_array_1D(s);
#ifdef MPICH
    mfmem::sdel_array_1D(Htmp);
    mfmem::sdel_array_1D(Htotal);
#endif
}

void cuGMRESSolverOrigUpdate(PolyGrid *grid, IntType level){
	
	//cuMemoryPrepara(grid);
#ifdef GMRES_DEGUG_FS_CUDA
	cuMemoryPreparaGMRESDebug(grid);
#endif
	
#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemLusgs, endtimeTemLusgs;
    double timeuseTemLusgs;
    gettimeofday(&starttimeTemLusgs, 0); 
#endif
#endif
	
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nT5    = 5*nTCell;

    // We haven't consider turbulence model yet.
    IntType nvar = 5;
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nT5, "res");
    RealFlow *reso  = NULL;
    mfmem::snew_array_1D(reso, nT5,dmrfl);
    assert(reso != 0);
 
    // Control parameters
    IntType i, j, k, Adu=1, kspan = 10, maxits = 10000;
    grid->GetData(&Adu, INT, 1, "ADU");
    grid->GetData(&kspan, INT, 1, "kspan");
    grid->GetData(&maxits, INT, 1, "gmresmaxits");
    RealFlow Error = 0.;
    grid->GetData(&Error, REAL_FLOW, 1, "gmresepsilon");
    if(Error < TINY) Error = 1.0e-2;
 
    // Temporary memories
    IntType len = nvar*nTCell;
    RealFlow *dq = NULL;
    mfmem::snew_array_1D(dq, len,dmrfl);
    assert(dq != 0);

    RealFlow **H = NULL;
    mfmem::snew_array_2D(H, kspan+1,kspan,dmrfl,true);
#ifdef MPICH
    RealFlow *Htmp   = NULL;
    RealFlow *Htotal = NULL;
    mfmem::snew_array_1D(Htmp, kspan,dmrfl);
    mfmem::snew_array_1D(Htotal, kspan,dmrfl);
    for(i=1; i<kspan; i++) {
        Htmp[i] = 0.0;
        Htotal[i] = 0.0;
    }
#endif
 
    RealFlow **v = NULL;
    mfmem::snew_array_2D(v, kspan+1,len,dmrfl,true);

    RealFlow *w  = NULL;
    RealFlow *cs = NULL;
    RealFlow *sn = NULL;
    RealFlow *s  = NULL;
    mfmem::snew_array_1D(w, len,dmrfl);
    mfmem::snew_array_1D(cs, kspan,dmrfl);
    mfmem::snew_array_1D(sn, kspan,dmrfl);
    mfmem::snew_array_1D(s, kspan+1,dmrfl);
	
	RealFlow *odata = NULL;    
	mfmem::snew_array_1D(odata, gnodata2, dmrfl);
 
    RealFlow norm0, norm, dmax;
 
    // Save the beginning flow variables
    RealGeom *vol  =  grid->GetCellVol();

    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*(nTCell+nBFace), "DQ");
    if(!DQ[0]) {
        mfmem::snew_array_1D(DQ[0], 5*(nTCell+nBFace),dmrfl);
        grid->UpdateDataPtr(DQ[0], REAL_FLOW, 5*(nTCell+nBFace), "DQ");
    }
    assert(DQ[0] != 0);
    for(i=1; i<nvar; i++) DQ[i] = &DQ[i-1][nTCell+nBFace];
	
	cuDQInit(nvar);	
	
    RealFlow *DQo[5];
    DQo[0] = NULL;
    mfmem::snew_array_1D(DQo[0], 5*nTCell,dmrfl);
    for(i=1; i<nvar; i++) DQo[i] = &DQo[i-1][nTCell];
	/*
	for(i=0; i<len; i++){
        dq[i]   = 0.;
    }
	*/
	cuGMRESdqInit(dq, len);
	//HANDLE_API_ERR(cudaMemcpy(dq, gdq, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
    // Save the residuals and Initialize Matrix*(Delta q) and p[0]
	/*
	for(i=0; i<nT5; i++) 
        reso[i] = res[i];
	*/
	cures2reso(res, reso, len);
	//HANDLE_API_ERR(cudaMemcpy(reso, greso, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
    // Now diagonal term in LU-SGS, here we need information of time steps
    //在一个GMERES的子迭代中,该值保持不变
    RealFlow *Diag = NULL;
    mfmem::snew_array_1D(Diag, nTCell,dmrfl);
    assert(Diag != 0);
    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
	/*
	for(i=0; i<nTCell; i++) 
        Diag[i] = vol[i]/dt[i];
    CalDiagLUSGS(grid, Diag, level);
	*/
	cuDiagInit(dt);
	cuCalDiagLUSGS(grid, level);
	//HANDLE_API_ERR(cudaMemcpy(Diag, gDiag, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));		
    
    //PreconditLUSGS(grid, Diag, level);
	cuPreconditLUSGS(grid, Diag, level);
	//HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	
	ofstream outfile;
	
	cuvDQoInit(nvar);
	//HANDLE_API_ERR(cudaMemcpy(DQo[0], gDQo, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	////HANDLE_API_ERR(cudaMemcpy(v[0], gv, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	
	/*
	for(j=0; j<nvar; j++){
        for(i=0; i<nTCell; i++){
            DQo[j][i] = DQ[j][i];
            v[0][j*nTCell + i] = DQo[j][i];
        }
    }
    */
	
    norm0 = cuDotProductMPIkernel4567(odata, v[0], len, 0);
    norm0 = sqrt(norm0);
#ifdef MPICH
    //if(myZone==1) printf("Norm = %.5e\n", norm0);
#else
    //printf("Norm = %.5e\n", norm0);
#endif    
 
    if(norm0 > 1.0e-10){
        // loop over GMRES sweeps
        //Nsweeps stands for the loop times before restarting.
        norm = norm0;
		
		bool converge = false;
        IntType its = 0;
        while (!converge )
        {
			//HANDLE_API_ERR(cudaMemcpy(gv, v[0], 5*gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));	
            for(k=0; k<kspan+1; k++) s[k]=0.0;
            s[0] = norm; 
			/*
			for(i=0; i<len; i++){ 
                v[0][i] /= norm;               //v=gamma/beta
			}
			*/
			k = 0;
			
			cuCalv(norm);
			////HANDLE_API_ERR(cudaMemcpy(v[0], gv, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	

			// Loop over the search directions
            while(!converge && k < kspan)
            {
				
                // Calculate the epsilon in evaluating matrix * vector
				// tmp memory transfer:
				
                //选择不同的A*V的方法,1--luo的简化,2--原始的矩阵直接求解,3--差分近似
                if(Adu == 1){
					//HANDLE_API_ERR(cudaMemcpy(&gv[k*5*gnTCell], v[k], 5*gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));		
                    cuComputeADU(grid, Diag, v[k], res, level, k, 0);
				}
                else if(Adu == 2)
                    ComputeADU2(grid, Diag, v[k], res, level);
                else if(Adu == 3)
                    ComputeADU3(grid, v[k], res, reso, level);
                else if(Adu == 4)
                    ResLUSGS(grid, v[k], level);

                cuPreconditLUSGS(grid, Diag, level);
				//HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
                
                cuDQ2w(nvar);
				//HANDLE_API_ERR(cudaMemcpy(w, gw, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
				/*
                for(j=0; j<nvar; j++){
                    for(i=0; i<nTCell; i++){
                        w[j*nTCell + i] = DQ[j][i];                        
                    }
                }
                */     
                // Calculate H
                for(j=0; j<=k; j++){
                    //H[j][k] = DotProduct(w, v[j], len, j);
					H[j][k] = cuDotProductkernel4567(odata, v[j], len, j);
                }
#ifdef MPICH
                //需要并行传递H的值
                for(j=0; j<=k; j++) Htmp[j] = H[j][k];
                for(j=0; j<kspan; j++) Htotal[j] = 0.;
                MPI_Allreduce(Htmp, Htotal, kspan, MPIReal, MPI_SUM, MPI_COMM_WORLD);
                for(j=0; j<=k; j++) H[j][k] = Htotal[j];
#endif
                cuwSubtractHv(H, k, nvar);
				//HANDLE_API_ERR(cudaMemcpy(w, gw, nvar*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
				/*
				for(j=0; j<=k; j++){
                    for(i=0; i<len; i++) 
                        w[i] -= H[j][k]*v[j][i];
                }
				*/
				
                //norm  = sqrt(DotProductMPI(w, w, len));
				norm  = sqrt(cuDotProductMPIkernel4567ww(odata, w, len, 0));
                H[k+1][k] = norm;
                norm  = 1.0/norm;
				
				cuUpdateNewvk(norm, k, nvar);
				////HANDLE_API_ERR(cudaMemcpy(v[k + 1], &gv[(k + 1)*nvar*gnTCell], nvar*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
				/*
                for(i=0; i<len; i++) 
                    v[k+1][i] = w[i]*norm;
				*/
				
                // Solve the linear least square problems
                for (j=0; j<k; j++)
                    ApplyPlaneRotation(H[j][k], H[j+1][k], cs[j], sn[j]);
                
                GeneratePlaneRotation(H[k][k], H[k+1][k], cs[k], sn[k]);
                ApplyPlaneRotation(H[k][k], H[k+1][k], cs[k], sn[k]);
                ApplyPlaneRotation(s[k], s[k+1], cs[k], sn[k]);

                dmax = fabs(s[k+1]/norm0);
                if(dmax < Error) converge = true;
                //在该循环内,可以用V[k+1]的空间来储存W变量,节省内存使用量
                its ++;
                k++;
				
            }
                        
            //完成Updata计算后，s[0]-s[k-1]中储存的数值为y[0]-y[k-1]
            ComputeY(H, s, k);
		
            // Calculate the Delta q
			cuCaldq(s, k, nvar);
			//HANDLE_API_ERR(cudaMemcpy(dq, gdq, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
			/*
            for(k=0; k<kspan; k++)
                for(i=0; i<len; i++) 
                    dq[i] += v[k][i]*s[k];
            */
			
			if(converge == true || its >= maxits) break;
            // Calculate Matrix*(Delta q) and P[0] for next sweep
            
            //选择不同的A*V的方法,1--luo的简化,2--原始的矩阵直接求解,3--差分近似                
            if(Adu == 1)
                cuComputeADU(grid, Diag, dq, res, level, k, 1);
            else if(Adu == 2)
                ComputeADU2(grid, Diag, dq, res, level);
            else if(Adu == 3)
                ComputeADU3(grid, dq, res, reso, level);
            else if(Adu == 4)
                ResLUSGS(grid, dq, level);

            // Calculate Matrix*(Delta q) and P[0] for next sweep
            cuPreconditLUSGS(grid, Diag, level);
			//HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));	

			cuUpdatev(nvar);
			////HANDLE_API_ERR(cudaMemcpy(v[0], gv, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
			/*
            for(j=0; j<nvar; j++){
                for(i=0; i<nTCell; i++){
                    v[0][j*nTCell + i] = DQo[j][i] - DQ[j][i];
                    //count++;
                }
            }
			*/
			

            // Check if the solution of linear eqs has been obtained within the scope 
            //norm = cuDotProductMPIkernel4567(odata, v[0], len, 0);
            //norm = sqrt(norm);
//          if(sweep==0) norm0=norm;
            //dmax = norm/norm0;
#ifdef MPICH
            //if(myZone==1) printf("Resi reduced by %.4e with %d sweeps\n", dmax, (int)(sweep+1));
#else
            //printf("Resi reduced by %.4e with %d sweeps\n", dmax, (int)(sweep+1));
#endif
            
            
        }
    
        // update the solution
		cuUpdateDQ(nvar);
		//HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		/*
        count = 0;
        for(j=0; j<nvar; j++){
            for(i=0; i<nTCell; i++){
                DQ[j][i] = dq[j*nTCell + i];
                //count++;
            }
        }
		*/
        cuUpdateFlowField3D_CFL3d(grid, DQ);
		
		cureso2res(nvar);
		/*
        for(i=0; i<nT5; i++) 
            res[i] = reso[i];
		*/
		
    }
	/*IntType n = nTCell + nBFace;
	
	RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
	HANDLE_API_ERR(cudaMemcpy(gq, q[0], n*sizeof(RealFlow), cudaMemcpyHostToDevice));			
	HANDLE_API_ERR(cudaMemcpy(&gq[1*n], q[1], n*sizeof(RealFlow), cudaMemcpyHostToDevice));	
	HANDLE_API_ERR(cudaMemcpy(&gq[2*n], q[2], n*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[3*n], q[3], n*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[4*n], q[4], n*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(gres, res, 5*gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));	
	*/
	
#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    timecost[2] = timecost[2] + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemLusgs, 0); 
    timeuseTemLusgs = (RealGeom) 1000000*(endtimeTemLusgs.tv_sec - starttimeTemLusgs.tv_sec) + endtimeTemLusgs.tv_usec - starttimeTemLusgs.tv_usec;
    timecost[2] += timeuseTemLusgs;
    timeuseTemLusgs /= 1000000.0;
    time_lusgs += timeuseTemLusgs;
#endif
#endif
	
#ifdef GMRES_DEGUG_FS_CUDA
	cuMemoryPreparaGMRESDebug2(grid);
#endif

	mfmem::sdel_array_1D(odata);
    // Delete temporary memories
    mfmem::sdel_array_1D(dq);
    mfmem::sdel_array_1D(Diag);
    mfmem::sdel_array_1D(reso);
    mfmem::sdel_array_1D(DQo[0]);
    mfmem::sdel_array_2D(H);
    mfmem::sdel_array_2D(v);
    mfmem::sdel_array_1D(w);
    mfmem::sdel_array_1D(cs);
    mfmem::sdel_array_1D(sn);
    mfmem::sdel_array_1D(s);
#ifdef MPICH
    mfmem::sdel_array_1D(Htmp);
    mfmem::sdel_array_1D(Htotal);
#endif
}

__global__ void gpures2turburhs(RealFlow *odata, RealFlow *idata, IntType nT){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nT){
		odata[i] = idata[i];
	}
	
}

void cures2turburhs(IntType nT){  	
	
	//IntType nT5 = 5*gnTCell;
	IntType blocksPerGrid = (nT + threadsPerBlock - 1) / threadsPerBlock;	
	gpures2turburhs <<< blocksPerGrid, threadsPerBlock >>> (gturburhs, gres, nT);
	
}

__global__ void gpuPreconditScalarLUSGSForward(RealFlow *dq, const RealFlow *lhsmat, const RealFlow *q, const IntType *luorder, 
								const IntType *layer, const IntType *C2C, const IntType *IndexC2C, const IntType *nCPC, 
								const RealFlow q_min, const IntType DQ_limit){
	IntType ilu = blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < 1){														
		IntType cell = luorder[ilu];

        for(IntType i=0; i < nCPC[cell]; i++){
            //cell2 = c2c[cell][i];
			IntType cell2 = C2C[IndexC2C[cell] + i];
            if(!(layer[cell2]>layer[cell])){
				dq[cell] -= lhsmat[cell + IndexC2C[cell] + i + 1]*dq[cell2];
			}
        }
        //dq[cell] /= lhsmat[cell][0];
		dq[cell] /= lhsmat[cell + IndexC2C[cell] + 0];
        
        
	}		
															
}

void tmpPreconditScalarLUSGS(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, RealFlow *dq, IntType *nCPC, IntType **c2c)
{
    IntType i, cell, cell2;
    IntType nTCell = grid->GetNTCell();
    IntType n=nTCell + grid->GetNBFace();
  
    RealFlow *turburhs = NULL;
    mfmem::snew_array_1D(turburhs,nTCell,dmrfl);
    for(cell=0;cell<nTCell;cell++){
        turburhs[cell]=res[cell];
    }
    for(cell=0;cell<n;cell++){
        dq[cell]=0.0;
    }
  
    //the Forward Sweep
    /*
    for(cell=0;cell<nTCell;cell++){
        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            if(cell2>cell) continue;

            turburhs[cell] -= lhsmat[cell][i+1]*dq[cell2];
        }
        dq[cell] = turburhs[cell]/lhsmat[cell][0];
    }
	*/
	IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer");
	IntType *luorder = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGSCellOrder");
    IntType *layer = (IntType *)grid->GetDataPtr(INT, n, "LUSGSLayer");
	
	//the Forward Sweep: first step
    for(IntType ilu=0;ilu<1;ilu++){
        cell = luorder[ilu];

        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            if(layer[cell2]>layer[cell]) continue;

            //dq[cell] -= lhsmat[cell][i+1]*dq[cell2];
			turburhs[cell] -= lhsmat[cell][i+1]*dq[cell2];
        }
        //dq[cell] /= lhsmat[cell][0];
		dq[cell] = turburhs[cell]/lhsmat[cell][0];
        
    }

    for(IntType laynum=0; laynum<cellsPerlayer[0]; laynum++ ){
        IntType start = cellsPerlayer[laynum+1];
        IntType end   = cellsPerlayer[laynum+2];
        if(laynum == 0) {start++;}
#pragma omp parallel for
        for(IntType ilu=start; ilu<end; ilu++){
            IntType cell = luorder[ilu];
            for(IntType i=0;i<nCPC[cell];i++){
                IntType cell2 = c2c[cell][i];
                if(layer[cell2]>layer[cell]) continue;

                //dq[cell] -= lhsmat[cell][i+1]*dq[cell2];
				turburhs[cell] -= lhsmat[cell][i+1]*dq[cell2];
			}
			//dq[cell] /= lhsmat[cell][0];
			dq[cell] = turburhs[cell]/lhsmat[cell][0];
        
        }

    }
	
 
#ifdef MPICH
    grid->CommInterfaceDataMPI(dq);
#endif
	
    //the Backward Sweep
	
	/*
    for(cell=nTCell-1;cell>-1;cell--){
        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];

            if(cell2<cell) continue;
            turburhs[cell] -= lhsmat[cell][i+1]*dq[cell2];
        }
        dq[cell] = turburhs[cell]/lhsmat[cell][0];
    }
	*/
	for(IntType laynum=cellsPerlayer[0]-1; laynum>=0; laynum-- ){
        IntType start = cellsPerlayer[laynum+2];
        IntType end   = cellsPerlayer[laynum+1];
#pragma omp parallel for
        for(IntType ilu=start-1; ilu>=end; ilu--){
            IntType cell = luorder[ilu];

            for(IntType i=0;i<nCPC[cell];i++){
                IntType cell2 = c2c[cell][i];
                if(layer[cell2]<layer[cell]) continue;

                //flux += lhsmat[cell][i+1]*dq[cell2];
				turburhs[cell] -= lhsmat[cell][i+1]*dq[cell2];
            }
            //dq[cell] -= flux/lhsmat[cell][0];
			dq[cell] = turburhs[cell]/lhsmat[cell][0];
        
        }
    }
    mfmem::sdel_array_1D(turburhs);
} 

__global__ void gpuGMRESScalarLUSGSForward(RealFlow *dq, const RealFlow *lhsmat, RealFlow *turburhs, const IntType *luorder, 
								const IntType *layer, const IntType *C2C, const IntType *IndexC2C, const IntType *nCPC, 
								const IntType start, const IntType end){
	IntType ilu = start + blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < end){	
		
		IntType cell = luorder[ilu];
		for(IntType i = 0; i < nCPC[cell]; i++){
			//IntType cell2 = c2c[cell][i];
			IntType cell2 = C2C[IndexC2C[cell] + i];
			if(!(layer[cell2]>layer[cell])){
				//dq[cell] -= lhsmat[cell + IndexC2C[cell] + i + 1]*dq[cell2];
				turburhs[cell] -= lhsmat[cell + IndexC2C[cell] + i + 1]*dq[cell2];
			}
		}
		//dq[cell] /= lhsmat[cell + IndexC2C[cell] + 0];
		dq[cell] = turburhs[cell]/lhsmat[cell + IndexC2C[cell] + 0];
	}		
															
}

__global__ void gpuGMRESScalarLUSGSBackward(RealFlow *dq, const RealFlow *lhsmat, RealFlow *turburhs, const IntType *luorder, 
								const IntType *layer, const IntType *C2C, const IntType *IndexC2C, const IntType *nCPC, 
								const IntType start, const IntType end){
	IntType ilu = start + blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < end){	
		IntType cell = luorder[ilu];
		for(IntType i = 0; i < nCPC[cell]; i++){
			IntType cell2 = C2C[IndexC2C[cell] + i];
			if(!(layer[cell2] < layer[cell])){
				turburhs[cell] -= lhsmat[cell + IndexC2C[cell] + i + 1]*dq[cell2];
			}
		}
		dq[cell] = turburhs[cell]/lhsmat[cell + IndexC2C[cell] + 0];
	
	}		
															
}

void cuPreconditScalarLUSGS(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, RealFlow *dq, IntType *nCPC, IntType **c2c)
{
    IntType nTCell = grid->GetNTCell();
    IntType n=nTCell + grid->GetNBFace();
  
    RealFlow *turburhs = NULL;
    mfmem::snew_array_1D(turburhs,nTCell,dmrfl);
	
	cures2turburhs(nTCell);
	// HANDLE_API_ERR(cudaMemcpy(turburhs, gturburhs, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	/*
    for(cell=0;cell<nTCell;cell++){
        turburhs[cell]=res[cell];
    }
	*/
	IntType nvar = 1;
	cuDQInit(nvar);	
	// HANDLE_API_ERR(cudaMemcpy(dq, gDQ, nvar*n*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	/*
    for(cell=0;cell<n;cell++){
        dq[cell]=0.0;
    }
	*/
	
	IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer");
	IntType *luorder = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGSCellOrder");
    IntType *layer = (IntType *)grid->GetDataPtr(INT, n, "LUSGSLayer");
  
    //the Forward Sweep
	//the Forward Sweep: first step
	IntType blocksPerGrid = (1 + threadsPerBlock - 1) / threadsPerBlock;
	IntType start = 0;
    IntType end   = 1;
	gpuGMRESScalarLUSGSForward <<< blocksPerGrid, threadsPerBlock >>> (gDQ, glhsmat, gturburhs, gluorder, glayer, gC2C, 
																gIndexC2C, gnCPC, start, end);

    for(IntType laynum = 0; laynum < cellsPerlayer[0]; laynum++){
        start = cellsPerlayer[laynum+1];
        end   = cellsPerlayer[laynum+2];
        if(laynum == 0) {start++;}
		blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;
		gpuGMRESScalarLUSGSForward <<< blocksPerGrid, threadsPerBlock >>> (gDQ, glhsmat, gturburhs, gluorder, glayer, gC2C, 
																gIndexC2C, gnCPC, start, end);		
    }
	//HANDLE_API_ERR(cudaMemcpy(dq, gDQ, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(dq, gDQ, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//HANDLE_API_ERR(cudaMemcpy(turburhs, gturburhs, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	/*
    for(cell=0;cell<nTCell;cell++){
        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            if(cell2>cell) continue;

            turburhs[cell] -= lhsmat[cell][i+1]*dq[cell2];
        }
        dq[cell] = turburhs[cell]/lhsmat[cell][0];
    }
	*/	
	
	/*
	//the Forward Sweep: first step
    for(IntType ilu=0;ilu<1;ilu++){
        cell = luorder[ilu];

        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            if(layer[cell2]>layer[cell]) continue;

            //dq[cell] -= lhsmat[cell][i+1]*dq[cell2];
			turburhs[cell] -= lhsmat[cell][i+1]*dq[cell2];
        }
        //dq[cell] /= lhsmat[cell][0];
		dq[cell] = turburhs[cell]/lhsmat[cell][0];
        
    }

    for(IntType laynum=0; laynum<cellsPerlayer[0]; laynum++ ){
        IntType start = cellsPerlayer[laynum+1];
        IntType end   = cellsPerlayer[laynum+2];
        if(laynum == 0) {start++;}
#pragma omp parallel for
        for(IntType ilu=start; ilu<end; ilu++){
            IntType cell = luorder[ilu];
            for(IntType i=0;i<nCPC[cell];i++){
                IntType cell2 = c2c[cell][i];
                if(layer[cell2]>layer[cell]) continue;

                //dq[cell] -= lhsmat[cell][i+1]*dq[cell2];
				turburhs[cell] -= lhsmat[cell][i+1]*dq[cell2];
			}
			//dq[cell] /= lhsmat[cell][0];
			dq[cell] = turburhs[cell]/lhsmat[cell][0];
        
        }

    }
	*/

#ifdef MPICH
    RealFlow *q_mpi[1];
    q_mpi[0] = dq;    
	grid->cuRecvSendVarNeighbor_Togeth(1, q_mpi, 3);
    //grid->CommInterfaceDataMPI(dq);
#endif
	
    //the Backward Sweep
	HANDLE_API_ERR(cudaMemcpy(&gDQ[nTCell], &dq[nTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	for(IntType laynum = cellsPerlayer[0] - 1; laynum >= 0; laynum--){
        IntType end = cellsPerlayer[laynum+2];
        IntType start   = cellsPerlayer[laynum+1];
		blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;
		gpuGMRESScalarLUSGSBackward <<< blocksPerGrid, threadsPerBlock >>> (gDQ, glhsmat, gturburhs, gluorder, glayer, gC2C, 
																gIndexC2C, gnCPC, start, end);
        
    }
	//HANDLE_API_ERR(cudaMemcpy(dq, gDQ, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	
	/*
    for(cell=nTCell-1;cell>-1;cell--){
        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];

            if(cell2<cell) continue;
            turburhs[cell] -= lhsmat[cell][i+1]*dq[cell2];
        }
        dq[cell] = turburhs[cell]/lhsmat[cell][0];
    }
	*/
	/*
	for(IntType laynum=cellsPerlayer[0]-1; laynum>=0; laynum-- ){
        IntType start = cellsPerlayer[laynum+2];
        IntType end   = cellsPerlayer[laynum+1];
#pragma omp parallel for
        for(IntType ilu=start-1; ilu>=end; ilu--){
            IntType cell = luorder[ilu];

            RealFlow flux = 0.0;
            for(IntType i=0;i<nCPC[cell];i++){
                IntType cell2 = c2c[cell][i];
                if(layer[cell2]<layer[cell]) continue;

                //flux += lhsmat[cell][i+1]*dq[cell2];
				turburhs[cell] -= lhsmat[cell][i+1]*dq[cell2];
            }
            //dq[cell] -= flux/lhsmat[cell][0];
			dq[cell] = turburhs[cell]/lhsmat[cell][0];
        
        }
    }
	*/
    mfmem::sdel_array_1D(turburhs);
} 

__global__ void gpuScalarv2dq(RealFlow *dq, const RealFlow *v, 
						const IntType nBFace, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		dq[i] = v[i];	
	}
	
}

__global__ void gpuScalarv2dq2(RealFlow *dq, 
						const IntType nBFace, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nBFace){
		dq[nTCell + i] = 0.0;	
	}
	
}

void cuScalarv2dq(IntType k, IntType type){  	
		
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	if (type == 0){
		gpuScalarv2dq <<< blocksPerGrid, threadsPerBlock >>> (gdqadu, &gv[k*gnTCell], gnBFace, gnTCell);
	}
	else if(type == 1) {
		gpuScalarv2dq <<< blocksPerGrid, threadsPerBlock >>> (gdqadu, gdq, gnBFace, gnTCell);
	}
	
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuScalarv2dq2 <<< blocksPerGrid, threadsPerBlock >>> (gdqadu, gnBFace, gnTCell);
	
}

__global__ void gpuScalarADUForward(RealFlow *res, const RealFlow *dq, const RealFlow *lhsmat, 
								const IntType *C2C, const IntType *IndexC2C, const IntType *nCPC, 
								const IntType nTCell){
	
	IntType cell = blockDim.x*blockIdx.x + threadIdx.x;
	if(cell < nTCell){
		res[cell] = lhsmat[cell + IndexC2C[cell] + 0]*dq[cell];
        for(IntType i=0;i<nCPC[cell];i++){
            IntType cell2 = C2C[IndexC2C[cell] + i];           
            res[cell] += lhsmat[cell + IndexC2C[cell] + i + 1]*dq[cell2];
        }
	}
	
}

void cuScalarADUForward(){  	
		
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuScalarADUForward <<< blocksPerGrid, threadsPerBlock >>> (gres, gdqadu, glhsmat, gC2C, gIndexC2C, gnCPC, gnTCell);
	
}

void cuComputeScalarADU(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, RealFlow *v, IntType *nCPC, IntType **c2c, IntType k, IntType type)
{

    IntType nTCell = grid->GetNTCell();
    IntType n=nTCell + grid->GetNBFace();
    
    RealFlow *dq = NULL;
    mfmem::snew_array_1D(dq,n,dmrfl);
	
	cuScalarv2dq(k, type);
	//HANDLE_API_ERR(cudaMemcpy(dq, gdqadu, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	/*
    for(cell=0;cell<nTCell;cell++) dq[cell]=v[cell];
    for(cell=nTCell; cell<n; cell++) dq[cell]=0.0;
	*/
	
#ifdef MPICH
	RealFlow *q_mpi[1];
    q_mpi[0] = dq;    
	grid->cuRecvSendVarNeighbor_Togeth(1, q_mpi, 10);
    //grid->CommInterfaceDataMPI(dq);
#endif
	
	HANDLE_API_ERR(cudaMemcpy(&gdqadu[nTCell], &dq[nTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
    //the Forward Sweep
	cuScalarADUForward();
	// HANDLE_API_ERR(cudaMemcpy(res, gres, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	/*
    for(cell=0;cell<nTCell;cell++){
        res[cell] = lhsmat[cell][0]*dq[cell];
        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            
            res[cell] += lhsmat[cell][i+1]*dq[cell2];
        }
    }
	*/

    mfmem::sdel_array_1D(dq);
} 

RealFlow cuSADotProductMPIkernel4567(RealFlow *odata, RealFlow *b, IntType n, IntType k)
{
    IntType  i;
    RealFlow sum = 0.0, sum_glb=0.0;
	
	IntType blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDotProductMPI <<< blocksPerGrid, threadsPerBlock >>> (gSAsumv2, &gv[k*n], n);
	
	blocksPerGrid = gSAnodata2;
	Reducekernel6 <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gSAsumv2, gSAodata2, gSAnsum2);
	HANDLE_API_ERR(cudaMemcpy(odata, gSAodata2, blocksPerGrid*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	for(i=0; i<blocksPerGrid; i++) sum += odata[i];
	
#ifdef MPICH
    MPI_Allreduce(&sum, &sum_glb, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    sum = sum_glb;
#endif
	
    return sum;
}

RealFlow cuSADotProductMPIkernel4567wv(RealFlow *odata, RealFlow *b, IntType n, IntType k)
{
    IntType  i;
    RealFlow sum = 0.0, sum_glb=0.0;
	
	IntType blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDotProduct <<< blocksPerGrid, threadsPerBlock >>> (gSAsumv2, gw, &gv[k*n], n);
		
	blocksPerGrid = gSAnodata2;
	Reducekernel6 <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gSAsumv2, gSAodata2, gSAnsum2);
	HANDLE_API_ERR(cudaMemcpy(odata, gSAodata2, blocksPerGrid*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	for(i=0; i<blocksPerGrid; i++) sum += odata[i];
	
#ifdef MPICH
    MPI_Allreduce(&sum, &sum_glb, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    sum = sum_glb;
#endif	
    return sum;
}

RealFlow cuSADotProductMPIkernel4567ww(RealFlow *odata, RealFlow *b, IntType n, IntType k)
{
    IntType  i;
    RealFlow sum = 0.0, sum_glb=0.0;
	
	IntType blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDotProduct <<< blocksPerGrid, threadsPerBlock >>> (gSAsumv2, gw, gw, n);
		
	blocksPerGrid = gSAnodata2;
	Reducekernel6 <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gSAsumv2, gSAodata2, gSAnsum2);
	HANDLE_API_ERR(cudaMemcpy(odata, gSAodata2, blocksPerGrid*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	for(i=0; i<blocksPerGrid; i++) sum += odata[i];
	
#ifdef MPICH
    MPI_Allreduce(&sum, &sum_glb, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    sum = sum_glb;
#endif	
    return sum;
}

void cuSolveScalarGMRES(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, RealFlow *dq, IntType *nCPC, IntType **c2c, const char *name, IntType level)  
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    IntType j, k, sweep;
  
    // Control parameters
    IntType Adu=1, kspan = 10, Nsweeps = 5;
    grid->GetData(&Adu, INT, 1, "ADU");
    grid->GetData(&kspan, INT, 1, "kspan");
    grid->GetData(&Nsweeps, INT, 1, "gmresweeps");
    RealFlow Error = 0.;
    grid->GetData(&Error, REAL_FLOW, 1, "gmresepsilon");
    if(Error < TINY) Error = 1.0e-2;

    IntType nvar = 1;
    IntType len = nvar*nTCell;

    // Temporary memories
	RealFlow *odata = NULL;    
	mfmem::snew_array_1D(odata, gSAnodata2, dmrfl);
	
    RealFlow *reso  = NULL;
    mfmem::snew_array_1D(reso,len,dmrfl);
    assert(reso != 0);

    RealFlow **H = NULL;
    mfmem::snew_array_2D(H,kspan+1,kspan,dmrfl,true);
    RealFlow **v = NULL;
    mfmem::snew_array_2D(v,kspan+1,len,dmrfl,true);
    RealFlow *w  = NULL;
    mfmem::snew_array_1D(w,len,dmrfl);
    RealFlow *cs = NULL;
    mfmem::snew_array_1D(cs,kspan,dmrfl);
    RealFlow *sn = NULL;
    mfmem::snew_array_1D(sn,kspan,dmrfl);
    RealFlow *s  = NULL;
    mfmem::snew_array_1D(s,kspan+1,dmrfl);
 
    RealFlow norm0, norm, dmax;
 
    // Save the beginning flow variables
    RealFlow *DQTurb = NULL;
    mfmem::snew_array_1D(DQTurb,n,dmrfl);
	
	cuDQInit(nvar);	
	//HANDLE_API_ERR(cudaMemcpy(DQTurb, gDQ, nvar*n*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	/*
    for(j=0; j<n; j++) 
		DQTurb[j] = 0.0;
	*/
	
    RealFlow *DQoTurb = NULL;
    mfmem::snew_array_1D(DQoTurb,n,dmrfl);
    
	cuGMRESdqInit(dq, len);
	//HANDLE_API_ERR(cudaMemcpy(dq, gdq, len*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	/*
	for(i=0; i<len; i++){
        dq[i]   = 0.;
    }
	*/

    // Save the residuals and Initialize Matrix*(Delta q) and p[0]
	cures2reso(res, reso, len);
	//HANDLE_API_ERR(cudaMemcpy(reso, greso, len*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	/*
    for(i=0; i<len; i++) 
        reso[i] = res[i];
	*/
    
    cuPreconditScalarLUSGS(grid, lhsmat, res, DQTurb, nCPC, c2c);
	
	cuvDQoInit(nvar);
	//HANDLE_API_ERR(cudaMemcpy(DQoTurb, gDQo, nvar*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(v[0], gv, nvar*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	/*
    for(i=0; i<nTCell; i++){
        DQoTurb[i] = DQTurb[i];
        v[0][i] = DQoTurb[i];
    }
	*/

	norm0 = cuSADotProductMPIkernel4567(odata, v[0], len, 0);
    norm0 = sqrt(norm0);
	/*
#ifdef MPICH
    if(myZone==1) printf("Norm = %.5e\n", norm0);
#else
    printf("Norm = %.5e\n", norm0);
#endif
    */
    if(norm0 > 1.0e-10){
        // loop over GMRES sweeps
        //Nsweeps stands for the loop times before restarting.
        for(sweep=0; sweep<Nsweeps; sweep++){
			//HANDLE_API_ERR(cudaMemcpy(gv, v[0], nvar*gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));	
            norm = cuSADotProductMPIkernel4567(odata, v[0], len, 0);
            norm = sqrt(norm);

            for(k=0; k<kspan+1; k++) s[k]=0.0;
            s[0] = norm; 
			
			cuCalv(norm);
			//HANDLE_API_ERR(cudaMemcpy(v[0], gv, nvar*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
			/*
            for(i=0; i<len; i++) 
                v[0][i] /= norm;               //v=gamma/beta
			*/
			
            // Loop over the search directions
            for(k=0; k<kspan; k++){
				// HANDLE_API_ERR(cudaMemcpy(&gv[k*gnTCell], v[k], nvar*gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));
                // Calculate the epsilon in evaluating matrix * vector
                cuComputeScalarADU(grid, lhsmat, res, v[k], nCPC, c2c, k, 0);

                cuPreconditScalarLUSGS(grid, lhsmat, res, DQTurb, nCPC, c2c);
                
				cuDQ2w(nvar);
				//HANDLE_API_ERR(cudaMemcpy(w, gw, nvar*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
				/*
                for(i=0; i<nTCell; i++){
                    w[i] = DQTurb[i];
                }
				*/
                        
                // Calculate H
                for(j=0; j<=k; j++){
                    //H[j][k] = DotProductMPI(w, v[j], len);
					H[j][k] = cuSADotProductMPIkernel4567wv(odata, v[j], len, j);
                    //for(i=0; i<len; i++) 
                        //w[i] -= H[j][k]*v[j][i];
                }
				
				cuwSubtractHv(H, k, nvar);
				//HANDLE_API_ERR(cudaMemcpy(w, gw, nvar*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
				
                norm  = sqrt(cuSADotProductMPIkernel4567ww(odata, v[j], len, j));
                H[k+1][k] = norm;
                norm  = 1.0/norm;
				
				cuUpdateNewvk(norm, k, nvar);
				//HANDLE_API_ERR(cudaMemcpy(v[k + 1], &gv[(k + 1)*nvar*gnTCell], nvar*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
				/*
                for(i=0; i<len; i++) 
                    v[k+1][i] = w[i]*norm;
				*/
				
                // Solve the linear least square problems
                for (j=0; j<k; j++)
                    ApplyPlaneRotation(H[j][k], H[j+1][k], cs[j], sn[j]);
                
                GeneratePlaneRotation(H[k][k], H[k+1][k], cs[k], sn[k]);
                ApplyPlaneRotation(H[k][k], H[k+1][k], cs[k], sn[k]);
                ApplyPlaneRotation(s[k], s[k+1], cs[k], sn[k]);

                //在该循环内,可以用V[k+1]的空间来储存W变量,节省内存使用量
            }
                        
            //完成Updata计算后，s[0]-s[k-1]中储存的数值为y[0]-y[k-1]
            ComputeY(H, s, kspan);

            // Calculate the Delta q
			cuCaldq(s, kspan, nvar);
			// HANDLE_API_ERR(cudaMemcpy(dq, gdq, nvar*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
			/*
            for(k=0; k<kspan; k++)
                for(i=0; i<len; i++) 
                    dq[i] += v[k][i]*s[k];
            */
			
            // Calculate Matrix*(Delta q) and P[0] for next sweep
            cuComputeScalarADU(grid, lhsmat, res, dq, nCPC, c2c, 0, 1);

            // Calculate Matrix*(Delta q) and P[0] for next sweep
            cuPreconditScalarLUSGS(grid, lhsmat, res, DQTurb, nCPC, c2c);
			
			cuUpdatev(nvar);
			//HANDLE_API_ERR(cudaMemcpy(v[0], gv, nvar*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
			/*
            for(i=0; i<nTCell; i++){
                v[0][i] = DQoTurb[i] - DQTurb[i];
            }
			*/

            // Check if the solution of linear eqs has been obtained within the scope 
            norm = cuSADotProductMPIkernel4567(odata, v[0], len, 0);
            norm = sqrt(norm);
//          if(sweep==0) norm0=norm;
            dmax = norm/norm0;
			/*
#ifdef MPICH
            if(myZone==1) printf("Resi reduced by %.4e with %d sweeps\n", dmax, (int)(sweep+1));
#else
            printf("Resi reduced by %.4e with %d sweeps\n", dmax, (int)(sweep+1));
#endif
			*/
            if(dmax < Error){
                sweep++;
                break;
            }
        }
		
		cureso2res(nvar);
		/*
        for(i=0; i<len; i++) 
            res[i] = reso[i];
		*/

        //对dq进行限制修改
        IntType DQ_limit = 1;
        RealFlow rhoP, amu, q_min;
        grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
        grid->GetData(&amu,   REAL_FLOW, 1, "amu");
        //grid->GetData(&ainf,  REAL_FLOW, 1, "ainf");
        grid->GetData(&DQ_limit, INT, 1, "DQ_limit");
        //RealFlow *q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);

        if(strcmp(name,"sa_nu") == 0){
            q_min = MIN_SA_NU;
            q_min *= (amu/rhoP);
        }
		
		//limit dq
        cuScalarGMRESlimitdq(DQ_limit, q_min);
		cuUpdateDQ(nvar);
		/**/
		// HANDLE_API_ERR(cudaMemcpy(dq, gdq, nvar*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		/*
        for(cell=0; cell<n; cell++){
            if(DQ_limit == 1){
                // do nothing!
            }else if(DQ_limit == 2){  
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] *= 0.1;
                }
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] *= 0.1;
                }
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] = 0.0;
                }
            }else if(DQ_limit == 3){
                dq[cell] = MAX(dq[cell],q_min-q[cell]);
            }else if(DQ_limit == 4){
                alph = q[cell]/(q[cell]+MAX(0.0,-dq[cell]));
                dq[cell] *= alph;
            }
        }
		*/
    }

    mfmem::sdel_array_1D(odata);
	// Delete temporary memories
    mfmem::sdel_array_1D(reso);
    mfmem::sdel_array_1D(DQTurb);
    mfmem::sdel_array_1D(DQoTurb);
    mfmem::sdel_array_2D(H);
    mfmem::sdel_array_2D(v);
    mfmem::sdel_array_1D(w);
    mfmem::sdel_array_1D(cs);
    mfmem::sdel_array_1D(sn);
    mfmem::sdel_array_1D(s);
}




