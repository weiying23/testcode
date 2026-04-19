#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>

#include <utility_functions.h>
#include <number_type.h>
#include <grid_patch_type.h>
#include <data_pool.h>
#include <zone.h>
#include <constant.h>
#include <algm.h>

#include <cuGradientQ_Gauss.cuh>
#include <cuData.cuh>
#include <cuErrorReturn.cuh>
#include "cuLUSGS.cuh"
#include <cuLimit.cuh>
#include <cuTurbulenceFlux.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef MPICH
#include <mpi.h>
#endif

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif

#ifdef TIMECOST
extern double* timecost;
extern double  time_flux, time_invis, time_roe, time_vis, time_calvis;
extern double  time_limiter;
extern double  time_gradient;
extern double  time_lusgs;
#endif

using namespace mflow;

using namespace gpuData;

void cuUpdateGhostGradSA(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz){
						
	//HANDLE_API_ERR(cudaMemcpy(gdqdx, dqdx[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(gdqdy, dqdy[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(gdqdz, dqdz[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
		
	HANDLE_API_ERR(cudaMemcpy(&gdnutdx[gnTCell], &dqdx[gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gdnutdy[gnTCell], &dqdy[gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gdnutdz[gnTCell], &dqdz[gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
		
}

void cuUpdateGhostGradT(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz){
						
	//HANDLE_API_ERR(cudaMemcpy(gdqdx, dqdx[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(gdqdy, dqdy[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(gdqdz, dqdz[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
		
	HANDLE_API_ERR(cudaMemcpy(&gdtdx[gnTCell], &dqdx[gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gdtdy[gnTCell], &dqdy[gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gdtdz[gnTCell], &dqdz[gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
		
}

void cuUpdateGhostGrad(RealFlow **dqdx, RealFlow **dqdy, RealFlow **dqdz){
						
	//HANDLE_API_ERR(cudaMemcpy(gdqdx, dqdx[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(gdqdy, dqdy[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(gdqdz, dqdz[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	for(IntType i = 0; i < 5; i++){
		HANDLE_API_ERR(cudaMemcpy(&gdqdx[i*(gnTCell + gnBFace) + gnTCell], &dqdx[i][gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(&gdqdy[i*(gnTCell + gnBFace) + gnTCell], &dqdy[i][gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(&gdqdz[i*(gnTCell + gnBFace) + gnTCell], &dqdz[i][gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
	}

}

__global__ void gpuGradientBoundary2(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, 
								const RealFlow *vol, const IntType nTCell){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;									
	if(i < nTCell){
		dqdx[i] /= vol[i];
        dqdy[i] /= vol[i];
        dqdz[i] /= vol[i];
	}												
}

void cuGradientvolaver(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, IntType name){
	
	IntType Cell = gnTCell + gnBFace;
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	if (name < 5){
		gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock >>> (&gdqdx[name*Cell], &gdqdy[name*Cell], &gdqdz[name*Cell], 
																gvol, gnTCell);
	}
	else if (name == 5){
		gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock >>> (gdtdx, gdtdy, gdtdz, 
																gvol, gnTCell);
	}
	else if (name == 6){
		gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock >>> (gdnutdx, gdnutdy, gdnutdz, 
																gvol, gnTCell);
	}
	
	if (name < 5){
		//HANDLE_API_ERR(cudaMemcpy(dqdx, &gdqdx[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
		//HANDLE_API_ERR(cudaMemcpy(dqdy, &gdqdy[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		//HANDLE_API_ERR(cudaMemcpy(dqdz, &gdqdz[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));

		//HANDLE_API_ERR(cudaMemcpy(dqdx, &gdqdx[name*Cell], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
		//HANDLE_API_ERR(cudaMemcpy(dqdy, &gdqdy[name*Cell], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		//HANDLE_API_ERR(cudaMemcpy(dqdz, &gdqdz[name*Cell], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	}
	else if (name == 5){
		//HANDLE_API_ERR(cudaMemcpy(dqdx, gdtdx, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
		//HANDLE_API_ERR(cudaMemcpy(dqdy, gdtdy, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		//HANDLE_API_ERR(cudaMemcpy(dqdz, gdtdz, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	}
	else if (name == 6){
		//HANDLE_API_ERR(cudaMemcpy(dqdx, gdnutdx, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
		//HANDLE_API_ERR(cudaMemcpy(dqdy, gdnutdy, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		//HANDLE_API_ERR(cudaMemcpy(dqdz, gdnutdz, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	}
	
}

__global__ void gpuGradientBoundary2(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, const RealFlow *q, const IntType *C2F, 
									const IntType *IndexC2F, const IntType *f2c, const IntType *nFPC, const IntType  *CellLayerNo,
									const RealFlow *area, const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, 
									const IntType GaussLayer, const IntType nTCell, const IntType Cell){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face;
	RealGeom qsum, tmpx, tmpy, tmpz;
	if(i < nTCell){
		if(!((CellLayerNo[i]==-1) || (CellLayerNo[i]>=GaussLayer))) {
			dqdx[i] = 0.0;
			dqdy[i] = 0.0;
			dqdz[i] = 0.0;
			
			for(IntType j=0;j<nFPC[i];j++){
				face = C2F[IndexC2F[i] + j];
				c1   = f2c[face+face];
				c2   = f2c[face+face+1];
						
				qsum = 0.5*(q[c1]+q[c2])*area[face];
				tmpx = qsum*xfn[face];
				tmpy = qsum*yfn[face];
				tmpz = qsum*zfn[face];
						
				if(i == c1){  
					dqdx[i] += tmpx;
					dqdy[i] += tmpy;
					dqdz[i] += tmpz;
				}else if(i == c2){
					dqdx[i] -= tmpx;
					dqdy[i] -= tmpy;
					dqdz[i] -= tmpz;
				}
			}
			
		}
	}	
}

void cuGradientBoundary2(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, IntType name){
	
	IntType Cell = gnTCell + gnBFace;
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	if (name < 5){
		gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock >>> (&gdqdx[name*Cell], &gdqdy[name*Cell], &gdqdz[name*Cell], 
																&gq[name*Cell], gC2F, gIndexC2F, gf2c, gnFPC, gCellLayerNo, garea, 
																gxfn, gyfn, gzfn, gGaussLayer, gnTCell, Cell);
	}
	else if (name == 5){
		gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock >>> (gdtdx, gdtdy, gdtdz, 
																gt, gC2F, gIndexC2F, gf2c, gnFPC, gCellLayerNo, garea, 
																gxfn, gyfn, gzfn, gGaussLayer, gnTCell, Cell);
	}
	else if (name == 6){
		gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock >>> (gdnutdx, gdnutdy, gdnutdz, 
																gsa_nu, gC2F, gIndexC2F, gf2c, gnFPC, gCellLayerNo, garea, 
																gxfn, gyfn, gzfn, gGaussLayer, gnTCell, Cell);
	}
	//HANDLE_API_ERR(cudaMemcpy(dqdx, &gdqdx[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(dqdy, &gdqdy[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//HANDLE_API_ERR(cudaMemcpy(dqdz, &gdqdz[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpuGradientBoundary(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, const RealFlow *q, const IntType *C2F, 
									const IntType *IndexC2F, const IntType *f2c, const IntType *nFPC, const IntType  *cellwallnumber,
									const RealFlow *area, const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, 
									const IntType nTCell, const IntType Cell){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face;
	RealGeom qsum, tmpx, tmpy, tmpz;
	if(i < nTCell){
		if(cellwallnumber[i] >= 2) {
			dqdx[i] = 0.0;
			dqdy[i] = 0.0;
			dqdz[i] = 0.0;
			
			for(IntType j=0;j<nFPC[i];j++){
				face = C2F[IndexC2F[i] + j];
				c1   = f2c[face+face];
				c2   = f2c[face+face+1];
						
				qsum = 0.5*(q[c1]+q[c2])*area[face];
				tmpx = qsum*xfn[face];
				tmpy = qsum*yfn[face];
				tmpz = qsum*zfn[face];
						
				if(i == c1){  
					dqdx[i] += tmpx;
					dqdy[i] += tmpy;
					dqdz[i] += tmpz;
				}else if(i == c2){
					dqdx[i] -= tmpx;
					dqdy[i] -= tmpy;
					dqdz[i] -= tmpz;
				}
			}
			
		}
	}	
}

void cuGradientBoundary(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, IntType name){
	
	IntType Cell = gnTCell + gnBFace;
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	if (name < 5){
		gpuGradientBoundary <<< blocksPerGrid, threadsPerBlock >>> (&gdqdx[name*Cell], &gdqdy[name*Cell], &gdqdz[name*Cell], 
																&gq[name*Cell], gC2F, gIndexC2F, gf2c, gnFPC, gcellwallnumber, 
																garea, gxfn, gyfn, gzfn, gnTCell, Cell);
	}
	else if (name == 5){
		gpuGradientBoundary <<< blocksPerGrid, threadsPerBlock >>> (gdtdx, gdtdy, gdtdz, 
																gt, gC2F, gIndexC2F, gf2c, gnFPC, gcellwallnumber, 
																garea, gxfn, gyfn, gzfn, gnTCell, Cell);
	}
	else if (name == 6){
		gpuGradientBoundary <<< blocksPerGrid, threadsPerBlock >>> (gdnutdx, gdnutdy, gdnutdz, 
																gsa_nu, gC2F, gIndexC2F, gf2c, gnFPC, gcellwallnumber, 
																garea, gxfn, gyfn, gzfn, gnTCell, Cell);
	}
	//HANDLE_API_ERR(cudaMemcpy(dqdx, &gdqdx[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(dqdy, &gdqdy[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//HANDLE_API_ERR(cudaMemcpy(dqdz, &gdqdz[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpuGradienttmpxyznBFace(RealFlow *tmpxyz, const RealFlow *q_n, const RealFlow *q, 
										const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, 
										const IntType *f2c, const IntType *type_bcr, const IntType *nNPF,
										const IntType *F2N, const IntType *IndexF2N, const RealFlow *area,
										const IntType nBFace, const IntType nTCell){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType    j, c1, c2, count, type;
    RealFlow   qsum;
	if(i < nBFace){
		count = 2 * i;
        c1 = f2c[count];
        c2 = f2c[count + 1];
        type = type_bcr[i];
        qsum = 0.0;

        if (type == INTERFACE || type == SYMM) {
            for (j = 0; j < nNPF[i]; j++)
                qsum += q_n[F2N[IndexF2N[i] + j]];
            qsum /= RealFlow(nNPF[i]);
        }
        else {
            qsum = 0.5 * (q[c1] + q[c2]);
        }
        j = 3 * i;
        qsum *= area[i];
        tmpxyz[j] = qsum * xfn[i];
        tmpxyz[j + 1] = qsum * yfn[i];
        tmpxyz[j + 2] = qsum * zfn[i];		
	}		
																				
}

__global__ void gpuGradienttmpxyz(RealFlow *tmpxyz, const RealFlow *q_n,
								const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, 
								const IntType *f2c, const IntType *nNPF,
								const IntType *F2N, const IntType *IndexF2N, const RealFlow *area,
								const IntType nBFace, const IntType nTCell, const IntType nTFace){
	IntType i = nBFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType    j;
    RealFlow   qsum;
	if(i < nTFace){										
        qsum = 0.0;

        for (j = 0; j < nNPF[i]; j++)
            qsum += q_n[F2N[IndexF2N[i] + j]];
        qsum /= RealFlow(nNPF[i]);
        j = 3 * i;
        qsum *= area[i];
        tmpxyz[j] = qsum * xfn[i];
        tmpxyz[j + 1] = qsum * yfn[i];
        tmpxyz[j + 2] = qsum * zfn[i];																				
	}										
}

#if (defined ShareMemory)
	
__global__ void gpuGradientReductionShareMemory(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, const RealFlow *tmpxyz, const IntType *f2c, 
									const IntType* C2F, const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell,
									const IntType nBFace){
	extern __shared__ double sdata[];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;
	
	for(IntType j = 0; j < 3; j++){
		sdata[tid*3 + j] = 0.0;
	}	
	__syncthreads();
	
	IntType c1, c2, face, count;
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			
			face = C2F[IndexC2F[i] + j];
            count = 3 * face;
            c1 = f2c[2*face];
            c2 = f2c[2*face + 1];
			
            if (i == c1) {
                sdata[tid*3 + 0] += tmpxyz[count];
                sdata[tid*3 + 1] += tmpxyz[count + 1];
                sdata[tid*3 + 2] += tmpxyz[count + 2];
            }
            else if (i == c2) {
                sdata[tid*3 + 0] -= tmpxyz[count];
                sdata[tid*3 + 1] -= tmpxyz[count + 1];
                sdata[tid*3 + 2] -= tmpxyz[count + 2];
            }
		}
	}
	__syncthreads();
	
	dqdx[i] = sdata[tid*3 + 0];
	dqdy[i] = sdata[tid*3 + 1];
	dqdz[i] = sdata[tid*3 + 2];	
}	

__global__ void gpuGradientReductionShareMemory2(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, const RealFlow *tmpxyz, const IntType *f2c, 
									const IntType* C2F, const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell,
									const IntType nBFace, const IntType threadsnum){
	extern __shared__ double sdata[];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;
	
	for(IntType j = 0; j < 3; j++){
		sdata[j*threadsnum + tid] = 0.0;
	}
	__syncthreads();
	
	IntType c1, c2, face, count;
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			
			face = C2F[IndexC2F[i] + j];
            count = 3 * face;
            c1 = f2c[2*face];
            c2 = f2c[2*face + 1];
			
            if (i == c1) {
                sdata[0*threadsnum + tid] += tmpxyz[count];
                sdata[1*threadsnum + tid] += tmpxyz[count + 1];
                sdata[2*threadsnum + tid] += tmpxyz[count + 2];
            }
            else if (i == c2) {
                sdata[0*threadsnum + tid] -= tmpxyz[count];
                sdata[1*threadsnum + tid] -= tmpxyz[count + 1];
                sdata[2*threadsnum + tid] -= tmpxyz[count + 2];
            }
		}
	}
	__syncthreads();
	
	dqdx[i] = sdata[0*threadsnum + tid];
	dqdy[i] = sdata[1*threadsnum + tid];
	dqdz[i] = sdata[2*threadsnum + tid];	
}	

#endif

__global__ void gpuGradientReduction(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, const RealFlow *tmpxyz, const IntType *f2c, 
									const IntType* C2F, const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell,
									const IntType nBFace){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face, count;

	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			
			face = C2F[IndexC2F[i] + j];
            count = 3 * face;
            c1 = f2c[2*face];
            c2 = f2c[2*face + 1];
			
            if (i == c1) {
                dqdx[i] += tmpxyz[count];
                dqdy[i] += tmpxyz[count + 1];
                dqdz[i] += tmpxyz[count + 2];
            }
            else if (i == c2) {
                dqdx[i] -= tmpxyz[count];
                dqdy[i] -= tmpxyz[count + 1];
                dqdz[i] -= tmpxyz[count + 2];
            }
		}
	}		
}

#if (defined FaceColoring)

__global__ void gpuGradientFaceColor(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, 
										const RealFlow *q_n, const RealFlow *q, 
										const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, 
										const IntType *f2c, const IntType *type_bcr, const IntType *nNPF,
										const IntType *F2N, const IntType *IndexF2N, const RealFlow *area,
										const IntType startFace, const IntType endFace, const IntType nTCell){
	IntType i = startFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType    j, c1, c2, count, type;
    RealFlow   qsum;
	RealGeom   tmpx, tmpy, tmpz;
	if(i < endFace){
		count = 2 * i;
        c1 = f2c[count];
        c2 = f2c[count + 1];
        type = type_bcr[i];
        qsum = 0.0;

        if (type == INTERFACE || type == SYMM) {
            for (j = 0; j < nNPF[i]; j++)
                qsum += q_n[F2N[IndexF2N[i] + j]];
            qsum /= RealFlow(nNPF[i]);
        }
        else {
            qsum = 0.5 * (q[c1] + q[c2]);
        }
        j = 3 * i;
        qsum *= area[i];
        tmpx = qsum * xfn[i];
        tmpy = qsum * yfn[i];
        tmpz = qsum * zfn[i];	

		dqdx[c1] += tmpx;
		dqdy[c1] += tmpy;
		dqdz[c1] += tmpz;
	}		
																				
}


__global__ void gpuGradientFaceColor2(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, 
										const RealFlow *q_n, const RealFlow *q, 
										const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, 
										const IntType *f2c, const IntType *type_bcr, const IntType *nNPF,
										const IntType *F2N, const IntType *IndexF2N, const RealFlow *area,
										const IntType startFace, const IntType endFace, const IntType nTCell){
	IntType i = startFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType    j, c1, c2, count, type;
    RealFlow   qsum;
	RealGeom   tmpx, tmpy, tmpz;
	if(i < endFace){
		count = 2 * i;
        c1 = f2c[count];
        c2 = f2c[count + 1];
        type = type_bcr[i];
        qsum = 0.0;

        if (type == INTERFACE || type == SYMM) {
            for (j = 0; j < nNPF[i]; j++)
                qsum += q_n[F2N[IndexF2N[i] + j]];
            qsum /= RealFlow(nNPF[i]);
        }
        else {
            qsum = 0.5 * (q[c1] + q[c2]);
        }
        j = 3 * i;
        qsum *= area[i];
        tmpx = qsum * xfn[i];
        tmpy = qsum * yfn[i];
        tmpz = qsum * zfn[i];	

		//dqdx[c1] += tmpx;
		//dqdy[c1] += tmpy;
		//dqdz[c1] += tmpz;
		atomicAddSM35(dqdx + c1, tmpx);
		atomicAddSM35(dqdy + c1, tmpy);
		atomicAddSM35(dqdz + c1, tmpz);
	}		
																				
}

__global__ void gpuGradientFaceColor3(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, 
										const RealFlow *q_n,
										const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, 
										const IntType *f2c, const IntType *nNPF,
										const IntType *F2N, const IntType *IndexF2N, const RealFlow *area,
										const IntType startFace, const IntType endFace, const IntType nTCell){
	IntType i = startFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType    j, c1, c2, count;
    RealFlow   qsum;
	RealGeom   tmpx, tmpy, tmpz;
	if(i < endFace){
		count = 2*i;
		c1 = f2c[count];
        c2 = f2c[count + 1];
		qsum = 0.0;

        for (j = 0; j < nNPF[i]; j++)
            qsum += q_n[F2N[IndexF2N[i] + j]];
        qsum /= RealFlow(nNPF[i]);

        qsum *= area[i];
        tmpx = qsum * xfn[i];
        tmpy = qsum * yfn[i];
        tmpz = qsum * zfn[i];	
		
		dqdx[c1] += tmpx;
		dqdy[c1] += tmpy;
		dqdz[c1] += tmpz;
		
		dqdx[c2] -= tmpx;
		dqdy[c2] -= tmpy;
		dqdz[c2] -= tmpz;
	}		
																				
}
	

void cuGradientFaceColor(PolyGrid *grid, RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, RealFlow *q_n, IntType name){
	
	//HANDLE_API_ERR(cudaMemcpy(gq_n, q_n, gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	IntType Cell = gnTCell + gnBFace;
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	if (name < 5){	
		//gpuGradienttmpxyznBFace <<< blocksPerGrid, threadsPerBlock >>> (gtmpxyz, gq_n, &gq[name*Cell], gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
		//															gnNPF, gF2N, gIndexF2N, garea, gnBFace, gnTCell);
	}
	else if (name == 5){
		gpuGradienttmpxyznBFace <<< blocksPerGrid, threadsPerBlock >>> (gtmpxyz, gq_n, gt, gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, gnBFace, gnTCell);
	}
	else if (name == 6){
		gpuGradienttmpxyznBFace <<< blocksPerGrid, threadsPerBlock >>> (gtmpxyz, gq_n, gsa_nu, gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, gnBFace, gnTCell);
	}
	
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
		
	gpuGradienttmpxyz <<< blocksPerGrid, threadsPerBlock >>> (gtmpxyz, gq_n, gxfn, gyfn, gzfn, gf2c, gnNPF, 
															gF2N, gIndexF2N, garea, gnBFace, gnTCell, gnTFace);	
	
	//HANDLE_API_ERR(cudaMemcpy(tmpxyz, gtmpxyz, 3*gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	// Reduction:
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	
	if (name < 5){
		//HANDLE_API_ERR(cudaMemcpy(q_n, gq_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		
		
		RealFlow* q;
		mfmem::snew_array_1D(q, (gnTCell + gnBFace), dmrfl);
		HANDLE_API_ERR(cudaMemcpy(q, &gq[name*(gnTCell + gnBFace)], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		
		IntType    nTCell = grid->GetNTCell();
		IntType    nTFace = grid->GetNTFace();
		IntType    nBFace = grid->GetNBFace();
		IntType* f2c = grid->Getf2c();
		BCRecord** bcr = grid->Getbcr();
		IntType    n = nTCell + nBFace;
		RealGeom* area = grid->GetFaceArea();
		RealGeom* xfn = grid->GetXfn();
		RealGeom* yfn = grid->GetYfn();
		RealGeom* zfn = grid->GetZfn();
		IntType* nNPF = grid->GetnNPF();
		IntType** F2N = CalF2N(grid);
		IntType* nFPC = CalnFPC(grid);
		IntType** C2F = CalC2F(grid);
		IntType    nIFace = grid->GetNIFace();
		IntType pfacenum = nBFace - nIFace;
		
		IntType    bfacegroup_num, ifacegroup_num;
		IntType* grid_bfacegroup, * grid_ifacegroup;
		ifacegroup_num = (*grid).ifacegroup.size();
		bfacegroup_num = (*grid).bfacegroup.size();
		grid_bfacegroup = NULL;
		grid_ifacegroup = NULL;
		mfmem::snew_array_1D(grid_bfacegroup, bfacegroup_num, dmrfl);
		mfmem::snew_array_1D(grid_ifacegroup, ifacegroup_num, dmrfl);
		for (int i = 0; i < bfacegroup_num; i++) {
			grid_bfacegroup[i] = (*grid).bfacegroup[i];
		}
		for (int i = 0; i < ifacegroup_num; i++) {
			grid_ifacegroup[i] = (*grid).ifacegroup[i];
		}
		//Boundary faces:
		for (IntType fcolor = 0; fcolor < bfacegroup_num; fcolor++) {
			IntType startFace, endFace;
			if (fcolor == 0) {
				startFace = 0;
			}
			else {
				startFace = grid_bfacegroup[fcolor - 1];
			}
			endFace = grid_bfacegroup[fcolor];
			
			IntType blocksPerGrid = (endFace - startFace + threadsPerBlock - 1) / threadsPerBlock;
			gpuGradientFaceColor <<< blocksPerGrid, threadsPerBlock >>> (&gdqdx[name*Cell], &gdqdy[name*Cell], &gdqdz[name*Cell], 
																	gq_n, &gq[name*Cell], gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, startFace, endFace, gnTCell);	
			
		}

#if (defined MPICH)			
			
		IntType blocksPerGrid = (nIFace + threadsPerBlock - 1) / threadsPerBlock;
		gpuGradientFaceColor2 <<< blocksPerGrid, threadsPerBlock >>> (&gdqdx[name*Cell], &gdqdy[name*Cell], &gdqdz[name*Cell], 
																	gq_n, &gq[name*Cell], gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, nBFace - nIFace, gnBFace, gnTCell);

		
		/*
		HANDLE_API_ERR(cudaMemcpy(dqdx, &gdqdx[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
		HANDLE_API_ERR(cudaMemcpy(dqdy, &gdqdy[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		HANDLE_API_ERR(cudaMemcpy(dqdz, &gdqdz[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		for (IntType i = pfacenum; i < nBFace; i++) {
			IntType count = 2 * i;
			IntType c1 = f2c[count];
			IntType c2 = f2c[count + 1];
			IntType type = bcr[i]->GetType();
			RealFlow qsum = 0.0;
			RealGeom   tmpx, tmpy, tmpz;

			if (type == INTERFACE || type == SYMM) {
				for (IntType j = 0; j < nNPF[i]; j++)
					qsum += q_n[F2N[i][j]];
				qsum /= RealFlow(nNPF[i]);
			}
			else {
				qsum = 0.5 * (q[c1] + q[c2]);
			}

			qsum *= area[i];
			tmpx = qsum * xfn[i];
			tmpy = qsum * yfn[i];
			tmpz = qsum * zfn[i];
			dqdx[c1] += tmpx;
			dqdy[c1] += tmpy;
			dqdz[c1] += tmpz;
		}
		
		HANDLE_API_ERR(cudaMemcpy(&gdqdx[name*Cell], dqdx, Cell*sizeof(RealFlow), cudaMemcpyHostToDevice));	
		HANDLE_API_ERR(cudaMemcpy(&gdqdy[name*Cell], dqdy, Cell*sizeof(RealFlow), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(&gdqdz[name*Cell], dqdz, Cell*sizeof(RealFlow), cudaMemcpyHostToDevice));
		*/
		
#endif		
		// Interior faces:
		for (IntType fcolor = 0; fcolor < ifacegroup_num; fcolor++) {
			IntType startFace, endFace;
			if (fcolor == 0) {
				startFace = nBFace;
			}
			else {
				startFace = grid_ifacegroup[fcolor - 1];
			}
			endFace = grid_ifacegroup[fcolor];
			
			IntType blocksPerGrid = (endFace - startFace + threadsPerBlock - 1) / threadsPerBlock;
			gpuGradientFaceColor3 <<< blocksPerGrid, threadsPerBlock >>> (&gdqdx[name*Cell], &gdqdy[name*Cell], &gdqdz[name*Cell], 
																	gq_n, gxfn, gyfn, gzfn, gf2c,  
																	gnNPF, gF2N, gIndexF2N, garea, startFace, endFace, gnTCell);

		}
		mfmem::sdel_array_1D(grid_bfacegroup);
		mfmem::sdel_array_1D(grid_ifacegroup);
		
		mfmem::sdel_array_1D(q);
		
		
		
		//gpuGradientReduction <<< blocksPerGrid, threadsPerBlock >>> (&gdqdx[name*Cell], &gdqdy[name*Cell], &gdqdz[name*Cell], 
		//															gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace);
	}
	else if (name == 5){
		gpuGradientReduction <<< blocksPerGrid, threadsPerBlock >>> (gdtdx, gdtdy, gdtdz, 
																	gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace);
	}
	else if (name == 6){
		gpuGradientReduction <<< blocksPerGrid, threadsPerBlock >>> (gdnutdx, gdnutdy, gdnutdz, 
																	gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace);
	}															
	//HANDLE_API_ERR(cudaMemcpy(dqdx, &gdqdx[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(dqdy, &gdqdy[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//HANDLE_API_ERR(cudaMemcpy(dqdz, &gdqdz[name*Cell], Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
}
#endif

void cuGradientReduction(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, RealFlow *q_n, IntType name){
	
	IntType Cell = gnTCell + gnBFace;
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	if (name < 5){	
		gpuGradienttmpxyznBFace <<< blocksPerGrid, threadsPerBlock >>> (gtmpxyz, gq_n, &gq[name*Cell], gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, gnBFace, gnTCell);
	}
	else if (name == 5){
		gpuGradienttmpxyznBFace <<< blocksPerGrid, threadsPerBlock >>> (gtmpxyz, gq_n, gt, gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, gnBFace, gnTCell);
	}
	else if (name == 6){
		gpuGradienttmpxyznBFace <<< blocksPerGrid, threadsPerBlock >>> (gtmpxyz, gq_n, gsa_nu, gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, gnBFace, gnTCell);
	}
	
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
		
	gpuGradienttmpxyz <<< blocksPerGrid, threadsPerBlock >>> (gtmpxyz, gq_n, gxfn, gyfn, gzfn, gf2c, gnNPF, 
															gF2N, gIndexF2N, garea, gnBFace, gnTCell, gnTFace);	
	
	// Reduction:
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
#if (defined ShareMemory)	
	if (name < 5){
		gpuGradientReductionShareMemory2 <<< blocksPerGrid, threadsPerBlock, 3*threadsPerBlock*sizeof(RealFlow) >>> (
																	&gdqdx[name*Cell], &gdqdy[name*Cell], &gdqdz[name*Cell], 
																	gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace, threadsPerBlock);
	}
	else if (name == 5){
		gpuGradientReductionShareMemory2 <<< blocksPerGrid, threadsPerBlock, 3*threadsPerBlock*sizeof(RealFlow) >>> (gdtdx, gdtdy, gdtdz, 
																	gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace, threadsPerBlock);
	}
	else if (name == 6){
		gpuGradientReductionShareMemory2 <<< blocksPerGrid, threadsPerBlock, 3*threadsPerBlock*sizeof(RealFlow) >>> (gdnutdx, gdnutdy, gdnutdz, 
																	gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace, threadsPerBlock);
	}
#else
	if (name < 5){
		gpuGradientReduction <<< blocksPerGrid, threadsPerBlock >>> (&gdqdx[name*Cell], &gdqdy[name*Cell], &gdqdz[name*Cell], 
																	gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace);
	}
	else if (name == 5){
		gpuGradientReduction <<< blocksPerGrid, threadsPerBlock >>> (gdtdx, gdtdy, gdtdz, 
																	gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace);
	}
	else if (name == 6){
		gpuGradientReduction <<< blocksPerGrid, threadsPerBlock >>> (gdnutdx, gdnutdy, gdnutdz, 
																	gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace);
	}
#endif	

}

__global__ void gpuGradientInit(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, const IntType Cell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < Cell){
		dqdx[i] = 0.0;
		dqdy[i] = 0.0;
		dqdz[i] = 0.0;
	}
}

void cuGradientInit(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, IntType name){
	
	IntType Cell = gnTCell + gnBFace;
	IntType blocksPerGrid = (Cell + threadsPerBlock - 1) / threadsPerBlock;
	if (name < 5){
		gpuGradientInit <<< blocksPerGrid, threadsPerBlock >>> (&gdqdx[name*Cell], &gdqdy[name*Cell], &gdqdz[name*Cell], Cell);
	}
	else if (name == 5){
		gpuGradientInit <<< blocksPerGrid, threadsPerBlock >>> (gdtdx, gdtdy, gdtdz, Cell);
	}
	else if (name == 6){
		gpuGradientInit <<< blocksPerGrid, threadsPerBlock >>> (gdnutdx, gdnutdy, gdnutdz, Cell);
	}	
	
}

__global__ void gpuCompNodeq_nWeight(RealFlow *q_n, const RealGeom *WeightNode, const IntType nTNode){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTNode){
		q_n[i] /= (WeightNode[i] + TINY);
	}
	
}

void cuCompNodeq_nWeight(RealFlow *q_n){
	
	IntType blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeq_nWeight <<< blocksPerGrid, threadsPerBlock >>> (gq_n, gWeightNode, gnTNode);
	
}

__global__ void gpuCompNodeq2q_n(RealFlow *q_n, const RealFlow *q, const RealGeom *WeightNodeN2C,
								const IntType *nCPN, const IntType *N2C, const IntType *IndexN2C, 
								const IntType *Nmark, const IntType nTNode){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTNode){
		
		if (Nmark[i] == 0) {
			for (IntType j = 0; j < nCPN[i]; j++) {
				IntType cellx = N2C[IndexN2C[i] + j];
				q_n[i] += q[cellx] * WeightNodeN2C[IndexN2C[i] + j];
			}
		}
		
	}
	
}

void cuCompNodeq2q_n(RealFlow *q_n, IntType name){
	
	IntType Cell = gnTCell + gnBFace;
	IntType blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	
	if (name < 5){
		gpuCompNodeq2q_n <<< blocksPerGrid, threadsPerBlock >>> (gq_n, &gq[name*Cell], gWeightNodeN2C, gnCPN, 
															gN2C, gIndexN2C, gNmark, gnTNode);
	}
	else if (name == 5){
		gpuCompNodeq2q_n <<< blocksPerGrid, threadsPerBlock >>> (gq_n, gt, gWeightNodeN2C, gnCPN, 
															gN2C, gIndexN2C, gNmark, gnTNode);
	}
	else if (name == 6){
		gpuCompNodeq2q_n <<< blocksPerGrid, threadsPerBlock >>> (gq_n, gsa_nu, gWeightNodeN2C, gnCPN, 
															gN2C, gIndexN2C, gNmark, gnTNode);
	}

}

__global__ void gpuCompNodefacq2q_n3(RealFlow *q_n, const RealFlow *facq, const RealGeom *WeightNodeBFace2C,
								const IntType *type_bcr, const IntType *F2N, const IntType *IndexF2N,
								const IntType *nNPF, const IntType *Nmark, const IntType nBFace){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nBFace){
		IntType type = type_bcr[i];
        if (!(type == WALL || type == SYMM || type == FAR_FIELD || type == INTERFACE)){
			for (IntType j = 0; j < nNPF[i]; j++) {
				IntType p1 = F2N[IndexF2N[i] + j];
				if (!(Nmark[p1] == WALL || Nmark[p1] == FAR_FIELD)){
					atomicAddSM35(q_n + p1, facq[i] * WeightNodeBFace2C[IndexF2N[i] + j]);
				}
			}
		}
	}
	
}

void cuCompNodefacq2q_n3(RealFlow *q_n){
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq2q_n3 <<< blocksPerGrid, threadsPerBlock >>> (gq_n, gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
	//HANDLE_API_ERR(cudaMemcpy(q_n, gq_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpuCompNodefacq2q_n2(RealFlow *q_n, const RealFlow *facq, const RealGeom *WeightNodeBFace2C,
								const IntType *type_bcr, const IntType *F2N, const IntType *IndexF2N,
								const IntType *nNPF, const IntType *Nmark, const IntType nBFace){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nBFace){
		IntType type = type_bcr[i];
        if (type == FAR_FIELD){
			for (IntType j = 0; j < nNPF[i]; j++) {
				IntType p1 = F2N[IndexF2N[i] + j];
				if (Nmark[p1] != WALL){
					atomicAddSM35(q_n + p1, facq[i] * WeightNodeBFace2C[IndexF2N[i] + j]);
				}
			}
		}
	}
	
}

void cuCompNodefacq2q_n2(RealFlow *q_n){
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq2q_n2 <<< blocksPerGrid, threadsPerBlock >>> (gq_n, gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
	//HANDLE_API_ERR(cudaMemcpy(q_n, gq_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpuCompNodefacq2q_n(RealFlow *q_n, const RealFlow *facq, const RealGeom *WeightNodeBFace2C,
								const IntType *type_bcr, const IntType *F2N, const IntType *IndexF2N,
								const IntType *nNPF, const IntType nBFace){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nBFace){
		IntType type = type_bcr[i];
        if (type == WALL){
			for (IntType j = 0; j < nNPF[i]; j++) {
				IntType p1 = F2N[IndexF2N[i] + j];
				atomicAddSM35(q_n + p1, facq[i] * WeightNodeBFace2C[IndexF2N[i] + j]);
			}	
		}
	}
	
}

void cuCompNodefacq2q_n(RealFlow *q_n){
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq2q_n <<< blocksPerGrid, threadsPerBlock >>> (gq_n, gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gnBFace);
	//HANDLE_API_ERR(cudaMemcpy(q_n, gq_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpuCompNodefacq(RealFlow *facq, const RealFlow *q, const IntType *f2c, const IntType *type_bcr, const IntType nBFace){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nBFace){
		facq[i] = 0.0;
        IntType type = type_bcr[i];
        if(!((type == INTERFACE || type == SYMM))){
			IntType c1 = f2c[2 * i];
			IntType c2 = f2c[2 * i + 1];
			facq[i] = q[c1] + q[c2];
			facq[i] *= 0.5;
		}
	}
}

void cuCompNodefacq(IntType name){
	
	IntType Cell = gnTCell + gnBFace;
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	if (name < 5){
		gpuCompNodefacq <<< blocksPerGrid, threadsPerBlock >>> (gfacq, &gq[name*Cell], gf2c, gtype_bcr, gnBFace);
	}
	else if (name == 5){
		gpuCompNodefacq <<< blocksPerGrid, threadsPerBlock >>> (gfacq, gt, gf2c, gtype_bcr, gnBFace);
	}
	else if (name == 6){
		gpuCompNodefacq <<< blocksPerGrid, threadsPerBlock >>> (gfacq, gsa_nu, gf2c, gtype_bcr, gnBFace);
	}
	
	//HANDLE_API_ERR(cudaMemcpy(facq, gfacq, gnBFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	
}

__global__ void gpuCompNodeInit(RealFlow *q_n, IntType nTNode){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTNode){
		q_n[i] = 0.0;
	}
	
}

void cuCompNodeInit(RealFlow* q_n){
	
	IntType blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock >>> (gq_n, gnTNode);
	
	//HANDLE_API_ERR(cudaMemcpy(q_n, gq_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
}

__global__ void gpuCompNodeuvw_n2q_n(RealFlow *q_n, const RealFlow *uvw_n, const RealFlow *vn, 
									const RealFlow *xyzfn_n_symm, const IntType *nodesymm, IntType nTNode){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTNode){
		if(nodesymm[i] == 1){
			q_n[i] = uvw_n[i] - vn[i]*xyzfn_n_symm[i];
		}
	}
	
}

void cuCompNodeuvw_n2q_n(RealFlow *q_n, IntType name){
	
	IntType blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	
	if(name == 1){
		gpuCompNodeuvw_n2q_n <<< blocksPerGrid, threadsPerBlock >>> (gq_n, gu_n, gvn, gxfn_n_symm, gnodesymm, gnTNode);
	}
	else if(name == 2){
		gpuCompNodeuvw_n2q_n <<< blocksPerGrid, threadsPerBlock >>> (gq_n, gv_n, gvn, gyfn_n_symm, gnodesymm, gnTNode);
	}
    else if(name == 3){
        gpuCompNodeuvw_n2q_n <<< blocksPerGrid, threadsPerBlock >>> (gq_n, gw_n, gvn, gzfn_n_symm, gnodesymm, gnTNode);
	}
	//HANDLE_API_ERR(cudaMemcpy(q_n, gq_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));
}

__global__ void gpuCompNodeuvw_n2vn(RealFlow *vn, const RealFlow *u_n, const RealFlow *v_n, 
									const RealFlow *w_n, const RealFlow *xfn_n_symm, const RealFlow *yfn_n_symm, 
									const RealFlow *zfn_n_symm, const IntType *nodesymm, IntType nTNode){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTNode){
		if(nodesymm[i] == 1){
			vn[i] = u_n[i]*xfn_n_symm[i] + v_n[i]*yfn_n_symm[i]+ w_n[i]*zfn_n_symm[i];
		}
	}
	
}

void cuCompNodeuvw_n2vn(){
	
	IntType blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeuvw_n2vn <<< blocksPerGrid, threadsPerBlock >>> (gvn, gu_n, gv_n, gw_n, gxfn_n_symm, gyfn_n_symm, 
																gzfn_n_symm, gnodesymm, gnTNode);
	
}

__global__ void gpuCompNodeuvw_nWeight(RealFlow *u_n, RealFlow *v_n, RealFlow *w_n, 
									const RealGeom *WeightNode, const IntType nTNode){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTNode){
		RealGeom wr = WeightNode[i];
		u_n[i] /= (wr + TINY);
		v_n[i] /= (wr + TINY);
		w_n[i] /= (wr + TINY);
	}
	
}

void cuCompNodeuvw_nWeight(RealFlow *u_n, RealFlow *v_n, RealFlow *w_n){

	IntType blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuCompNodeuvw_nWeight <<< blocksPerGrid, threadsPerBlock >>> (gu_n, gv_n, gw_n, gWeightNode, gnTNode);
	
	//HANDLE_API_ERR(cudaMemcpy(u_n, gu_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//HANDLE_API_ERR(cudaMemcpy(v_n, gv_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//HANDLE_API_ERR(cudaMemcpy(w_n, gw_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));
}

__global__ void gpuCompNodeuvw2uvw_n(RealFlow *u_n, RealFlow *v_n, RealFlow *w_n, const RealFlow *q, 
								const RealGeom *WeightNodeN2C, const IntType *nCPN, const IntType *N2C, 
								const IntType *IndexN2C, const IntType *Nmark, const IntType *nodesymm,
								const IntType nTNode, const IntType Cell){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTNode){		
		if (Nmark[i] == 0) {
			for (IntType j = 0; j < nCPN[i]; j++) {
				if (nodesymm[i] == 1) {
					IntType cellx = N2C[IndexN2C[i] + j];
					RealGeom wr = WeightNodeN2C[IndexN2C[i] + j];
					u_n[i] += q[1*Cell + cellx] * wr;
					v_n[i] += q[2*Cell + cellx] * wr;
					w_n[i] += q[3*Cell + cellx] * wr;
				}
			}
		}		
	}	
}

void cuCompNodeuvw2uvw_n(RealFlow *u_n, RealFlow *v_n, RealFlow *w_n){
	
	IntType Cell = gnTCell + gnBFace;
	IntType blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeuvw2uvw_n <<< blocksPerGrid, threadsPerBlock >>> (gu_n, gv_n, gw_n, gq, gWeightNodeN2C, gnCPN, 
														gN2C, gIndexN2C, gNmark, gnodesymm, gnTNode, Cell);

}

__global__ void gpuCompNodefacuvw2uvw_n3(RealFlow *u_n, RealFlow *v_n, RealFlow *w_n, const RealFlow *facu, 
									const RealFlow *facv, const RealFlow *facw, const RealGeom *WeightNodeBFace2C,
									const IntType *type_bcr, const IntType *F2N, const IntType *IndexF2N,
									const IntType *nNPF, const IntType *Nmark, const IntType *nodesymm, const IntType nBFace){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nBFace){
		IntType type = type_bcr[i];
        if (!(type == WALL || type == SYMM || type == FAR_FIELD || type == INTERFACE)){
			for (IntType j = 0; j < nNPF[i]; j++) {
				IntType p1 = F2N[IndexF2N[i] + j];
				if (!(Nmark[p1] == WALL || Nmark[p1] == FAR_FIELD)){
					if (nodesymm[p1] == 1) {
						RealGeom wr = WeightNodeBFace2C[IndexF2N[i] + j];
						atomicAddSM35(u_n + p1, facu[i] * wr);
						atomicAddSM35(v_n + p1, facv[i] * wr);
						atomicAddSM35(w_n + p1, facw[i] * wr);
					}
				}
			}
		}
	}
	
}

void cuCompNodefacuvw2uvw_n3(RealFlow *u_n, RealFlow *v_n, RealFlow *w_n){
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacuvw2uvw_n3 <<< blocksPerGrid, threadsPerBlock >>> (gu_n, gv_n, gw_n, gfacu, gfacv, gfacw, 
																gWeightNodeBFace2C, gtype_bcr, gF2N, gIndexF2N, 
																gnNPF, gNmark, gnodesymm, gnBFace);
}

__global__ void gpuCompNodefacuvw2uvw_n2(RealFlow *u_n, RealFlow *v_n, RealFlow *w_n, const RealFlow *facu, 
									const RealFlow *facv, const RealFlow *facw, const RealGeom *WeightNodeBFace2C,
									const IntType *type_bcr, const IntType *F2N, const IntType *IndexF2N,
									const IntType *nNPF, const IntType *Nmark, const IntType *nodesymm, const IntType nBFace){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nBFace){
		IntType type = type_bcr[i];
        if (type == FAR_FIELD){
			for (IntType j = 0; j < nNPF[i]; j++) {
				IntType p1 = F2N[IndexF2N[i] + j];
				if (Nmark[p1] != WALL){
					if (nodesymm[p1] == 1) {
						RealGeom wr = WeightNodeBFace2C[IndexF2N[i] + j];
						atomicAddSM35(u_n + p1, facu[i] * wr);
						atomicAddSM35(v_n + p1, facv[i] * wr);
						atomicAddSM35(w_n + p1, facw[i] * wr);
					}
				}
			}
		}
	}
	
}

void cuCompNodefacuvw2uvw_n2(RealFlow *u_n, RealFlow *v_n, RealFlow *w_n){
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacuvw2uvw_n2 <<< blocksPerGrid, threadsPerBlock >>> (gu_n, gv_n, gw_n, gfacu, gfacv, gfacw, 
																gWeightNodeBFace2C, gtype_bcr, gF2N, gIndexF2N, 
																gnNPF, gNmark, gnodesymm, gnBFace);

}

__global__ void gpuCompNodefacuvw2uvw_n(RealFlow *u_n, RealFlow *v_n, RealFlow *w_n, const RealFlow *facu, 
										const RealFlow *facv, const RealFlow *facw, const RealFlow *WeightNodeBFace2C, 
										const IntType *type_bcr, const IntType *F2N, const IntType *IndexF2N, 
										const IntType *nNPF, const IntType *nodesymm, const IntType nBFace){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nBFace){
		IntType type = type_bcr[i];
        if (type == WALL){
			for (IntType j = 0; j < nNPF[i]; j++) {
				IntType p1 = F2N[IndexF2N[i] + j];
				if (nodesymm[p1] == 1) {
					RealGeom wr = WeightNodeBFace2C[IndexF2N[i] + j];
					atomicAddSM35(u_n + p1, facu[i] * wr);
					atomicAddSM35(v_n + p1, facv[i] * wr);
					atomicAddSM35(w_n + p1, facw[i] * wr);
				}
			}	
		}
	}										
}

void cuCompNodefacuvw2uvw_n(RealFlow *u_n, RealFlow *v_n, RealFlow *w_n){
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacuvw2uvw_n <<< blocksPerGrid, threadsPerBlock >>> (gu_n, gv_n, gw_n, gfacu, gfacv, gfacw, 
																gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gnodesymm, gnBFace);

}

__global__ void gpuCompNodefacuvw(RealFlow *facu, RealFlow *facv, RealFlow *facw, const RealFlow *q, 
								const IntType *f2c, const IntType *type_bcr, const IntType nBFace, const IntType Cell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nBFace){
        IntType type = type_bcr[i];
        if(!((type == INTERFACE || type == SYMM))){
			IntType c1 = f2c[2 * i];
			IntType c2 = f2c[2 * i + 1];
			facu[i] = q[1*Cell + c1] + q[1*Cell + c2];
			facu[i] *= 0.5;
			facv[i] = q[2*Cell + c1] + q[2*Cell + c2];
			facv[i] *= 0.5;
			facw[i] = q[3*Cell + c1] + q[3*Cell + c2];
			facw[i] *= 0.5;
		}
	}
		
}

void cuCompNodefacuvw(){
	
	IntType Cell = gnTCell + gnBFace;
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuCompNodefacuvw <<< blocksPerGrid, threadsPerBlock >>> (gfacu, gfacv, gfacw, gq, gf2c, gtype_bcr, gnBFace, Cell);
	
}

void cuCompNodeuvw_nInit(RealFlow* u_n, RealFlow* v_n, RealFlow* w_n){
	
	IntType blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock >>> (gu_n, gnTNode);
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock >>> (gv_n, gnTNode);
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock >>> (gw_n, gnTNode);
	
	//HANDLE_API_ERR(cudaMemcpy(u_n, gu_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(v_n, gv_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(w_n, gw_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
}

void cuCompNodeVar3D_dist(PolyGrid *grid, RealFlow *q_n, RealFlow *q, IntType name, RealFlow* u_n, RealFlow* v_n, RealFlow* w_n){

    //at u gradient calculation, work out u_n, v_n and w_n for the v and w gradient calculation
    if (name == 1) {
		cuCompNodeuvw_nInit(u_n, v_n, w_n);
		
        //计算物理边界面心的物理值
		cuCompNodefacuvw();

        //利用物理边界面心的值，计算物理边界点
		cuCompNodefacuvw2uvw_n(u_n, v_n, w_n);
		cuCompNodefacuvw2uvw_n2(u_n, v_n, w_n);
		cuCompNodefacuvw2uvw_n3(u_n, v_n, w_n);
		
		cuCompNodeuvw2uvw_n(u_n, v_n, w_n);
		
#ifdef MPICH
		HANDLE_API_ERR(cudaMemcpy(u_n, gu_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		HANDLE_API_ERR(cudaMemcpy(v_n, gv_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		HANDLE_API_ERR(cudaMemcpy(w_n, gw_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		
        grid->CommInternodeDataMPI(u_n);
        grid->CommInternodeDataMPI(v_n);
        grid->CommInternodeDataMPI(w_n);

		HANDLE_API_ERR(cudaMemcpy(gu_n, u_n, gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gv_n, v_n, gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gw_n, w_n, gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice));
#endif
		
		cuCompNodeuvw_nWeight(u_n, v_n, w_n);
		
    }  
        
	cuCompNodeInit(q_n);
    
    //计算物理边界点的值，使用物理面的值进行加权计算，不包括并行边界和对称面边界
    //计算物理边界面心的物理值	
	cuCompNodefacq(name);
    
    //利用物理边界面心的值，计算物理边界点
	cuCompNodefacq2q_n(q_n);	
	cuCompNodefacq2q_n2(q_n);	
	cuCompNodefacq2q_n3(q_n);
	
    //计算其他点的物理值，使用与其相相邻的控制体体心值
	cuCompNodeq2q_n(q_n, name);
	
    //传递并行边界点的加权值
#ifdef MPICH	
	HANDLE_API_ERR(cudaMemcpy(q_n, gq_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));

    grid->CommInternodeDataMPI(q_n);
	
	HANDLE_API_ERR(cudaMemcpy(gq_n, q_n, gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice));
#endif
	
	cuCompNodeq_nWeight(q_n);
    
    //修正对称面顶点的速度
	if (name == 1){
		cuCompNodeuvw_n2vn();
	}
	cuCompNodeuvw_n2q_n(q_n, name);
}

void cuCompNodeVar3D_dist(PolyGrid* grid, RealFlow* q_n, RealFlow* q, IntType name){
    
    RealGeom** WeightNodeBFace2C = grid->GetWeightNodeBFace2C();

    RealGeom** WeightNodeN2C = grid->GetWeightNodeN2C();   
	
	cuCompNodeInit(q_n);
	
    //计算物理边界点的值，使用物理面的值进行加权计算，不包括并行边界和对称面边界
    //计算物理边界面心的物理值
	cuCompNodefacq(name);
	
    //利用物理边界面心的值，计算物理边界点
	cuCompNodefacq2q_n(q_n);
			
	cuCompNodefacq2q_n2(q_n);
	
	cuCompNodefacq2q_n3(q_n);	
	
    //计算其他点的物理值，使用与其相相邻的控制体体心值
	cuCompNodeq2q_n(q_n, name);	

    //传递并行边界点的加权值
#ifdef MPICH
	HANDLE_API_ERR(cudaMemcpy(q_n, gq_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
    grid->CommInternodeDataMPI(q_n);
	
	HANDLE_API_ERR(cudaMemcpy(gq_n, q_n, gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice));
#endif
	
	cuCompNodeq_nWeight(q_n);

}

void cuCompGradientQ_Gauss_Node(PolyGrid *grid, RealFlow *q, RealFlow *dqdx, RealFlow *dqdy, 
								RealFlow *dqdz, IntType name, RealFlow* u_n, RealFlow* v_n, RealFlow* w_n){
    
	IntType    nTNode = grid->GetNTNode();
    IntType    nTCell = grid->GetNTCell();
    IntType    nTFace = grid->GetNTFace();
    IntType    nBFace = grid->GetNBFace();
    IntType    n      = nTCell + nBFace;
    IntType    nIFace = grid->GetNIFace(); 
	
    // Initialize dq
	cuGradientInit(dqdx, dqdy, dqdz, name);

    RealFlow *q_n = NULL;
    mfmem::snew_array_1D(q_n, nTNode, dmrfl);
    if (name > 0 && name < 4)
        cuCompNodeVar3D_dist(grid, q_n, q, name, u_n, v_n, w_n);
    else
        cuCompNodeVar3D_dist(grid, q_n, q, name);
#if (defined FaceColoring)
	cuGradientFaceColor(grid, dqdx, dqdy, dqdz, q_n, name);
#else
	cuGradientReduction(dqdx, dqdy, dqdz, q_n, name);
#endif
	
    //如果单元含有一个以上的物面，该单元梯度采用Gauss求解
    IntType vis_mode,level;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    level = grid->GetLevel();
    if(vis_mode != INVISCID && level == 0){ 
        IntType *cellwallnumber = grid->GetGridQualityCellWallNumber();
        cuGradientBoundary(dqdx, dqdy, dqdz, name);
    }      

    //边界层前n层采用Gauss方法
    IntType GaussLayer = -1;
    grid->GetData(&GaussLayer, INT, 1, "GaussLayer");
    IntType *CellLayerNo = (IntType *)grid->GetDataPtr(INT, n, "CellLayerNo");
    if(level == 0 && GaussLayer>0){
		cuGradientBoundary2(dqdx, dqdy, dqdz, name);		
    }   
	
	cuGradientvolaver(dqdx, dqdy, dqdz, name);
	
    mfmem::sdel_array_1D(q_n);
}

void cuGradientMemoryTrans(const RealFlow *rho, const RealFlow *u, const RealFlow *v, const RealFlow *w, const RealFlow *p){
	/*
	HANDLE_API_ERR(cudaMemcpy(gq, rho, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[(gnTCell + gnBFace)], u, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[2*(gnTCell + gnBFace)], v, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[3*(gnTCell + gnBFace)], w, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[4*(gnTCell + gnBFace)], p, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	*/
	// Since UpdateFlowField3D_CFL3d has updated the 0 - nTCell values, here only the ghost cells needed to be updated
	HANDLE_API_ERR(cudaMemcpy(&gq[gnTCell], &rho[gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[(gnTCell + gnBFace) + gnTCell], &u[gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[2*(gnTCell + gnBFace) + gnTCell], &v[gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[3*(gnTCell + gnBFace) + gnTCell], &w[gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[4*(gnTCell + gnBFace) + gnTCell], &p[gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
}

/* void cuGradientMemoryTrans(const RealFlow *q, const IntType name){
	
	if (name == 5){
		// temperature:
		 HANDLE_API_ERR(cudaMemcpy(gt, q, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	}
	else 
	if (name == 6){
		// SA_NU:
		 HANDLE_API_ERR(cudaMemcpy(gsa_nu, q, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(&gsa_nu[gnTCell], &q[gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
	}
} */


void cuCompGradientQ(PolyGrid *grid, RealFlow *q, RealFlow *dqdx, RealFlow *dqdy, 
					RealFlow *dqdz, IntType name, RealFlow* u_n, RealFlow* v_n, RealFlow* w_n){
						
	// name was used to comput. gradient of different variables
	// name = 0, 1, 2, 3, 4 stands for rho, u, v, w, p
	// name = 5 stands for temperature
	// name = 6 stands for SA model
	// gradient of rho, u, v, w, p was stored in GPU memory gdqdx, gdqdy, gdqdz
	// gradient of t and SA stored gdtdx, gdtdy, gdtdz and gdnutdx, gdnutdy, gdnutdz separately
	
#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemGradient, endtimeTemGradient;
    double timeuseTemGradient;
    gettimeofday(&starttimeTemGradient, 0); 
#endif
#endif
	
	// temp memory transfer into gq (rho, u, v, w, p): 
	/*	
	if(name == 0){
		IntType    n = gnTCell + gnBFace;
		// Get flow variables
		RealFlow *rho = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
		RealFlow *u   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
		RealFlow *v   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
		RealFlow *w   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
		RealFlow *p   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
		cuGradientMemoryTrans(rho, u, v, w, p);
	}	
	*/
	/*
	else if(name == 5 || name == 6) { 
		// t or SA_NU:
		cuGradientMemoryTrans(q, name);
	}
	*/
    cuCompGradientQ_Gauss_Node(grid, q,  dqdx,  dqdy, dqdz, name, u_n, v_n, w_n);

#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    timecost[0] = timecost[0] + time_tmp + MPI_Wtime();
    time_gradient = time_gradient + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemGradient, 0); 
    timeuseTemGradient = (RealGeom) 1000000*(endtimeTemGradient.tv_sec - starttimeTemGradient.tv_sec) + endtimeTemGradient.tv_usec - starttimeTemGradient.tv_usec;
    timecost[0] += timeuseTemGradient;
    timeuseTemGradient /= 1000000.0;
    time_gradient += timeuseTemGradient;
#endif
#endif

}

#ifdef MultiStream 
void cuCompGradientQ_SA_MultiStream(PolyGrid *grid){

	// Initialize dq
	//cuGradientInit(NULL, NULL, NULL, name);
	IntType n = gnTCell + gnBFace;
	IntType blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradientInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdnutdx, gdnutdy, gdnutdz, n);	
	
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq_n, gnTNode);
	
	RealFlow rhoP,amu,ainf;
    grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
    grid->GetData(&amu,   REAL_FLOW, 1, "amu");
    grid->GetData(&ainf,  REAL_FLOW, 1, "ainf");
 
    RealFlow q_min;
    RealFlow sigma;
 
    if(strcmp("sa_nu","sa_nu") == 0){
        sigma = 1.0/SIGMA_SA; 
        q_min = MIN_SA_NU;
        q_min *= (amu/rhoP);
    }
	IntType TurM = 0;
	if(strcmp("sa_nu","sa_nu") == 0) TurM = 1;	
	
    //计算物理边界点的值，使用物理面的值进行加权计算，不包括并行边界和对称面边界
    //计算物理边界面心的物理值
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gfacq, gsa_nu, gf2c, gtype_bcr, gnBFace);
	
    //利用物理边界面心的值，计算物理边界点
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq_n, gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gnBFace);
			
	gpuCompNodefacq2q_n2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq_n, gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
	
	gpuCompNodefacq2q_n3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq_n, gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
	
    //计算其他点的物理值，使用与其相相邻的控制体体心值
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq_n, gsa_nu, gWeightNodeN2C, gnCPN, 
															gN2C, gIndexN2C, gNmark, gnTNode);
	
	HANDLE_API_ERR(cudaMemcpyAsync(hostmsq_n, gq_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[0]));
	
	//cuViscousFluxScalar(grid, "sa_nu");
	//cuViscousFluxScalar3D_New3(grid, "sa_nu");
	
	
	blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuViscousFluxScalar <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gflux, gtem, gtem_c2, gsa_nu, gq, gvis_l, 
														gxcc, gycc, gzcc, gxfc, gyfc, gzfc, gxfn, gyfn, gzfn, 
														gf2c, gtype_bcr, garea, gangle_h, sigma, TurM, gnBFace, gnTFace);																	
	
    if(strcmp("sa_nu","sa_nu") == 0){
        gpuViscousFluxScalar2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gtem, gtem_c2, gsa_nu, gq, gf2c, 
														sigma, gnBFace, gnTFace);			
    }

	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuViscousFluxScalar3Reduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gres, gflux, gtem, gtem_c2, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell);
  

    //传递并行边界点的加权值
#ifdef MPICH
    grid->CommInternodeDataMPI(hostmsq_n);
#endif
	
	HANDLE_API_ERR(cudaMemcpyAsync(gq_n, hostmsq_n, gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[0]));
		
    // Matrices from the viscous flux 	
	//cuViscousMatsScalar(grid, "sa_nu");
	// Calculate Dfdq
	//cuViscousDqScalar(grid, "sa_nu", NULL, NULL, 0, gnTFace);   
	blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuViscousDqScalar <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gdqdl, gdqdr, gq, gsa_nu, gvis_l, gxfc, gyfc, gzfc, 
														gxcc, gycc, gzcc, gxfn, gyfn, gzfn, garea, gf2c, sigma, 
														TurM, gnTFace);	
		
	// Put Dq to the LHS matrices
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuPutScalarDqToLhsReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (glhsmat, gdqdl, gdqdr, gC2F, gIndexC2F, gnFPC, 
																gnCPC, gIndexC2C, gf2c, gfcptr, gnTFace, gnTCell);
	
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuCompNodeq_nWeight <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq_n, gWeightNode, gnTNode);
	
#if (defined FaceColoring)
	cuGradientFaceColor(grid, NULL, NULL, NULL, hostmsq_n, name);
#else
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradienttmpxyznBFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gtmpxyz, gq_n, gsa_nu, gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, gnBFace, gnTCell);
	
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;		
	gpuGradienttmpxyz <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gtmpxyz, gq_n, gxfn, gyfn, gzfn, gf2c, gnNPF, 
															gF2N, gIndexF2N, garea, gnBFace, gnTCell, gnTFace);	
	
	// Reduction:
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
#if (defined ShareMemory)	
	gpuGradientReductionShareMemory2 <<< blocksPerGrid, threadsPerBlock, 3*threadsPerBlock*sizeof(RealFlow), flowstream[0] >>> (gdnutdx, gdnutdy, gdnutdz, 
																	gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace, threadsPerBlock);
#else
	gpuGradientReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdnutdx, gdnutdy, gdnutdz, 
																	gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace);
#endif	
#endif
	
    //如果单元含有一个以上的物面，该单元梯度采用Gauss求解
    IntType vis_mode,level;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    level = grid->GetLevel();
    if(vis_mode != INVISCID && level == 0){ 
        blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
		gpuGradientBoundary <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdnutdx, gdnutdy, gdnutdz, 
																gsa_nu, gC2F, gIndexC2F, gf2c, gnFPC, gcellwallnumber, 
																garea, gxfn, gyfn, gzfn, gnTCell, n);
    }      

    //边界层前n层采用Gauss方法
    IntType GaussLayer = -1;
    grid->GetData(&GaussLayer, INT, 1, "GaussLayer");
    /* IntType *CellLayerNo = (IntType *)grid->GetDataPtr(INT, n, "CellLayerNo"); */
    if(level == 0 && GaussLayer>0){
		blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
		gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdnutdx, gdnutdy, gdnutdz, 
																	gsa_nu, gC2F, gIndexC2F, gf2c, gnFPC, gCellLayerNo, garea, 
																	gxfn, gyfn, gzfn, gGaussLayer, gnTCell, n);
    }   
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdnutdx, gdnutdy, gdnutdz, 
																gvol, gnTCell);
	cudaStreamSynchronize(flowstream[0]);
	cudaStreamSynchronize(flowstream[1]);
}

void cuCompGradientQ_ComputeTimeStep_MultiStream(PolyGrid *grid){
	
	
#ifdef MPICH  
	IntType nvar = 1;
	grid->cuRecvSendVarNeighbor_TogethForGradient_T(nvar); 
#endif 
	
}

void cuCompGradientQ_MultiStream(PolyGrid *grid, RealFlow **q, RealFlow **dqdx, RealFlow **dqdy, 
					RealFlow **dqdz, RealFlow* u_n, RealFlow* v_n, RealFlow* w_n){
	
#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemGradient, endtimeTemGradient;
    double timeuseTemGradient;
    gettimeofday(&starttimeTemGradient, 0); 
#endif
#endif
	
	IntType    nTNode = grid->GetNTNode();
	IntType    nTCell = grid->GetNTCell();
    //IntType    nTFace = grid->GetNTFace();
    IntType    nBFace = grid->GetNBFace();
    IntType    n      = nTCell + nBFace;
    //mfmem::snew_array_1D(q_n, nTNode, dmrfl);
	
	//cuCompNodeuvw_nInit(u_n, v_n, w_n);
	/* IntType blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gu_n, gnTNode);
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gv_n, gnTNode);
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gw_n, gnTNode); */
		
	//计算物理边界面心的物理值
	//cuCompNodefacuvw();
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuCompNodefacuvw <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gfacu, gfacv, gfacw, gq, gf2c, gtype_bcr, gnBFace, n);

	//利用物理边界面心的值，计算物理边界点
	//cuCompNodefacuvw2uvw_n(u_n, v_n, w_n);
	gpuCompNodefacuvw2uvw_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gu_n, gv_n, gw_n, gfacu, gfacv, gfacw, 
																gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gnodesymm, gnBFace);
	//cuCompNodefacuvw2uvw_n2(u_n, v_n, w_n);
	gpuCompNodefacuvw2uvw_n2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gu_n, gv_n, gw_n, gfacu, gfacv, gfacw, 
																gWeightNodeBFace2C, gtype_bcr, gF2N, gIndexF2N, 
																gnNPF, gNmark, gnodesymm, gnBFace);
	//cuCompNodefacuvw2uvw_n3(u_n, v_n, w_n);
	gpuCompNodefacuvw2uvw_n3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gu_n, gv_n, gw_n, gfacu, gfacv, gfacw, 
																gWeightNodeBFace2C, gtype_bcr, gF2N, gIndexF2N, 
																gnNPF, gNmark, gnodesymm, gnBFace);
																
	//cuCompNodeuvw2uvw_n(u_n, v_n, w_n);
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeuvw2uvw_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gu_n, gv_n, gw_n, gq, gWeightNodeN2C, gnCPN, 
														gN2C, gIndexN2C, gNmark, gnodesymm, gnTNode, n);
	// transfer back for MPI communication:
	HANDLE_API_ERR(cudaMemcpyAsync(hostu_n, gu_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[1]));
	HANDLE_API_ERR(cudaMemcpyAsync(hostv_n, gv_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[1]));
	HANDLE_API_ERR(cudaMemcpyAsync(hostw_n, gw_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[1]));	
	
	// name = 0:
	/* blocksPerGrid = (5*n + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradientInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (&gdqdx[0*n], &gdqdy[0*n], &gdqdz[0*n], 5*n);
	
	blocksPerGrid = (5*gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gmsq_n, 5*gnTNode); */
	
    //计算物理边界点的值，使用物理面的值进行加权计算，不包括并行边界和对称面边界
    //计算物理边界面心的物理值
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gfacq, &gq[0*n], gf2c, gtype_bcr, gnBFace);
	
    //利用物理边界面心的值，计算物理边界点
	gpuCompNodefacq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gmsq_n, gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gnBFace);

	gpuCompNodefacq2q_n2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gmsq_n, gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
			
	// the last part of u_n, v_n, w_n:
	cudaStreamSynchronize(flowstream[1]);
	
#ifdef MPICH
	grid->CommInternodeDataMPI(hostu_n);
	grid->CommInternodeDataMPI(hostv_n);
	grid->CommInternodeDataMPI(hostw_n);
#endif
	
	//cuCompNodeuvw_nWeight(u_n, v_n, w_n);
	HANDLE_API_ERR(cudaMemcpyAsync(gu_n, hostu_n, gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[1]));
	HANDLE_API_ERR(cudaMemcpyAsync(gv_n, hostv_n, gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[1]));
	HANDLE_API_ERR(cudaMemcpyAsync(gw_n, hostw_n, gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[1]));
	
	gpuCompNodefacq2q_n3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gmsq_n, gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
																
	//计算其他点的物理值，使用与其相相邻的控制体体心值
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gmsq_n, &gq[0*n], gWeightNodeN2C, gnCPN, 
															gN2C, gIndexN2C, gNmark, gnTNode);
															
	HANDLE_API_ERR(cudaMemcpyAsync(hostmsq_n, gmsq_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[0]));
	
	//cuCompGradientQ_Gauss_Node(grid, q[4],  dqdx[4],  dqdy[4], dqdz[4], 4, u_n, v_n, w_n);	
	/* blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradientInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (&gdqdx[4*n], &gdqdy[4*n], &gdqdz[4*n], n);
	
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (&gmsq_n[4*gnTNode], gnTNode); */
	
    //计算物理边界点的值，使用物理面的值进行加权计算，不包括并行边界和对称面边界
    //计算物理边界面心的物理值
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (gfacq, &gq[4*n], gf2c, gtype_bcr, gnBFace);
	
    //利用物理边界面心的值，计算物理边界点
	gpuCompNodefacq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (&gmsq_n[4*gnTNode], gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gnBFace);

	gpuCompNodefacq2q_n2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (&gmsq_n[4*gnTNode], gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
	
	gpuCompNodefacq2q_n3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (&gmsq_n[4*gnTNode], gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
	
    //计算其他点的物理值，使用与其相相邻的控制体体心值
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (&gmsq_n[4*gnTNode], &gq[4*n], gWeightNodeN2C, gnCPN, 
															gN2C, gIndexN2C, gNmark, gnTNode);
															
	HANDLE_API_ERR(cudaMemcpyAsync(&hostmsq_n[4*gnTNode], &gmsq_n[4*gnTNode], gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[4]));
	
	cudaStreamSynchronize(flowstream[1]);
	
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;	
	gpuCompNodeuvw_nWeight <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gu_n, gv_n, gw_n, gWeightNode, gnTNode);	
	
	//修正对称面顶点的速度
	//cuCompNodeuvw_n2vn();
	gpuCompNodeuvw_n2vn <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gvn, gu_n, gv_n, gw_n, gxfn_n_symm, gyfn_n_symm, 
																gzfn_n_symm, gnodesymm, gnTNode);	
	
	
    //传递并行边界点的加权值
	cudaStreamSynchronize(flowstream[0]);
#ifdef MPICH
    grid->CommInternodeDataMPI(hostmsq_n);
#endif

	HANDLE_API_ERR(cudaMemcpyAsync(gmsq_n, hostmsq_n, gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[0]));
	
	
	//cuCompGradientQ_Gauss_Node(grid, q[1],  dqdx[1],  dqdy[1], dqdz[1], 1, u_n, v_n, w_n);
	// name = 1:			
	/* blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradientInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (&gdqdx[1*n], &gdqdy[1*n], &gdqdz[1*n], n);
		
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (&gmsq_n[1*gnTNode], gnTNode); */
    
    //计算物理边界点的值，使用物理面的值进行加权计算，不包括并行边界和对称面边界
    //计算物理边界面心的物理值	
	//cuCompNodefacq(1);
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gfacq, &gq[1*n], gf2c, gtype_bcr, gnBFace);
    
    //利用物理边界面心的值，计算物理边界点
	//cuCompNodefacq2q_n(q_n);	
	//cuCompNodefacq2q_n2(q_n);	
	//cuCompNodefacq2q_n3(q_n);
	gpuCompNodefacq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (&gmsq_n[1*gnTNode], gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gnBFace);

	gpuCompNodefacq2q_n2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (&gmsq_n[1*gnTNode], gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
	
	gpuCompNodefacq2q_n3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (&gmsq_n[1*gnTNode], gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
	
    //计算其他点的物理值，使用与其相相邻的控制体体心值
	//cuCompNodeq2q_n(q_n, 1);
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (&gmsq_n[1*gnTNode], &gq[1*n], gWeightNodeN2C, gnCPN, 
															gN2C, gIndexN2C, gNmark, gnTNode);
															
	HANDLE_API_ERR(cudaMemcpyAsync(&hostmsq_n[1*gnTNode], &gmsq_n[1*gnTNode], gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[1]));	
	
	cudaStreamSynchronize(flowstream[0]);	
	gpuCompNodeq_nWeight <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gmsq_n, gWeightNode, gnTNode);
	
#if (defined FaceColoring)
	cuGradientFaceColor(grid, dqdx[0], dqdy[0], dqdz[0], hostmsq_n, 0);
#else
	//cuGradientReduction(dqdx[0], dqdy[0], dqdz[0], q_n, 0);
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradienttmpxyznBFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gtmpxyz, gmsq_n, &gq[0*n], gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, gnBFace, gnTCell);
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuGradienttmpxyz <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gtmpxyz, gmsq_n, gxfn, gyfn, gzfn, gf2c, gnNPF, 
															gF2N, gIndexF2N, garea, gnBFace, gnTCell, gnTFace);	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
#if (defined ShareMemory)	
	gpuGradientReductionShareMemory2 <<< blocksPerGrid, threadsPerBlock, 3*threadsPerBlock*sizeof(RealFlow), flowstream[0] >>> (
																	&gdqdx[0*n], &gdqdy[0*n], &gdqdz[0*n], 
																	gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace, threadsPerBlock);
#else
	gpuGradientReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (&gdqdx[0*n], &gdqdy[0*n], &gdqdz[0*n], 
																	gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace);
#endif	
#endif
	
	cudaStreamSynchronize(flowstream[4]);
    //传递并行边界点的加权值
#ifdef MPICH
    grid->CommInternodeDataMPI(&hostmsq_n[4*gnTNode]);
#endif
	HANDLE_API_ERR(cudaMemcpyAsync(&gmsq_n[4*gnTNode], &hostmsq_n[4*gnTNode], gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[4]));				
	
	//如果单元含有一个以上的物面，该单元梯度采用Gauss求解
    IntType vis_mode,level;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    level = grid->GetLevel();
    if(vis_mode != INVISCID && level == 0){ 
        //IntType *cellwallnumber = grid->GetGridQualityCellWallNumber();
        //cuGradientBoundary(dqdx[0], dqdy[0], dqdz[0], 0);
		blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
		gpuGradientBoundary <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (&gdqdx[0*n], &gdqdy[0*n], &gdqdz[0*n], 
																&gq[0*n], gC2F, gIndexC2F, gf2c, gnFPC, gcellwallnumber, 
																garea, gxfn, gyfn, gzfn, gnTCell, n);
    }      

    //边界层前n层采用Gauss方法
    IntType GaussLayer = -1;
    grid->GetData(&GaussLayer, INT, 1, "GaussLayer");
    //IntType *CellLayerNo = (IntType *)grid->GetDataPtr(INT, n, "CellLayerNo");
    if(level == 0 && GaussLayer>0){
		gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (&gdqdx[0*n], &gdqdy[0*n], &gdqdz[0*n], 
																&gq[0*n], gC2F, gIndexC2F, gf2c, gnFPC, gCellLayerNo, garea, 
																gxfn, gyfn, gzfn, gGaussLayer, gnTCell, n);	
    }   
	
	
	gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (&gdqdx[0*n], &gdqdy[0*n], &gdqdz[0*n], 
																gvol, gnTCell);	
	
	cudaStreamSynchronize(flowstream[4]);
	gpuCompNodeq_nWeight <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (&gmsq_n[4*gnTNode], gWeightNode, gnTNode);
	
#if (defined FaceColoring)
	cuGradientFaceColor(grid, dqdx[4], dqdy[4], dqdz[4], &hostmsq_n[4*gnTNode], 0);
#else
	//cuGradientReduction(dqdx[0], dqdy[0], dqdz[0], q_n, 0);
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradienttmpxyznBFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (gtmpxyz_p, &gmsq_n[4*gnTNode], &gq[4*n], gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, gnBFace, gnTCell);
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuGradienttmpxyz <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (gtmpxyz_p, &gmsq_n[4*gnTNode], gxfn, gyfn, gzfn, gf2c, gnNPF, 
															gF2N, gIndexF2N, garea, gnBFace, gnTCell, gnTFace);	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
#if (defined ShareMemory)	
	gpuGradientReductionShareMemory2 <<< blocksPerGrid, threadsPerBlock, 3*threadsPerBlock*sizeof(RealFlow), flowstream[4] >>> (
																	&gdqdx[4*n], &gdqdy[4*n], &gdqdz[4*n], 
																	gtmpxyz_p, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace, threadsPerBlock);
#else
	gpuGradientReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (&gdqdx[4*n], &gdqdy[4*n], &gdqdz[4*n], 
																	gtmpxyz_p, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace);
#endif	
#endif
	
	//如果单元含有一个以上的物面，该单元梯度采用Gauss求解
    if(vis_mode != INVISCID && level == 0){ 
		blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
		gpuGradientBoundary <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (&gdqdx[4*n], &gdqdy[4*n], &gdqdz[4*n], 
																&gq[4*n], gC2F, gIndexC2F, gf2c, gnFPC, gcellwallnumber, 
																garea, gxfn, gyfn, gzfn, gnTCell, n);
    }      

    //边界层前n层采用Gauss方法
    if(level == 0 && GaussLayer>0){
		gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (&gdqdx[4*n], &gdqdy[4*n], &gdqdz[4*n], 
																&gq[4*n], gC2F, gIndexC2F, gf2c, gnFPC, gCellLayerNo, garea, 
																gxfn, gyfn, gzfn, gGaussLayer, gnTCell, n);	
    }   
	
	//cuGradientvolaver(dqdx[0], dqdy[0], dqdz[0], 0);
	gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[4] >>> (&gdqdx[4*n], &gdqdy[4*n], &gdqdz[4*n], 
																gvol, gnTCell);
	
	
	//cudaStreamSynchronize(flowstream[1]);
	
	cudaStreamSynchronize(flowstream[1]);
    //传递并行边界点的加权值
#ifdef MPICH
    grid->CommInternodeDataMPI(&hostmsq_n[1*gnTNode]);
#endif
	
	//cuCompNodeq_nWeight(q_n);
	HANDLE_API_ERR(cudaMemcpyAsync(&gmsq_n[1*gnTNode], &hostmsq_n[1*gnTNode], gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[1]));

	// cuCompGradientQ_Gauss_Node(grid, q[2],  dqdx[2],  dqdy[2], dqdz[2], 2, u_n, v_n, w_n);
	/* blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradientInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (&gdqdx[2*n], &gdqdy[2*n], &gdqdz[2*n], n);
	
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (&gmsq_n[2*gnTNode], gnTNode); */
	
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (gfacq, &gq[2*n], gf2c, gtype_bcr, gnBFace);
	
	gpuCompNodefacq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (&gmsq_n[2*gnTNode], gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gnBFace);

	gpuCompNodefacq2q_n2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (&gmsq_n[2*gnTNode], gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
	
	gpuCompNodefacq2q_n3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (&gmsq_n[2*gnTNode], gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
																
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (&gmsq_n[2*gnTNode], &gq[2*n], gWeightNodeN2C, gnCPN, 
															gN2C, gIndexN2C, gNmark, gnTNode);
															
	HANDLE_API_ERR(cudaMemcpyAsync(&hostmsq_n[2*gnTNode], &gmsq_n[2*gnTNode], gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[2]));
	
	
	cudaStreamSynchronize(flowstream[1]);
	gpuCompNodeq_nWeight <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (&gmsq_n[1*gnTNode], gWeightNode, gnTNode);      
																
	//cuCompNodeuvw_n2q_n(q_n, 1);
	gpuCompNodeuvw_n2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (&gmsq_n[1*gnTNode], gu_n, gvn, gxfn_n_symm, gnodesymm, gnTNode);
	
#if (defined FaceColoring)
	cuGradientFaceColor(grid, dqdx[1], dqdy[1], dqdz[1], &hostmsq_n[1*gnTNode], 0);
#else
	//cuGradientReduction(dqdx[0], dqdy[0], dqdz[0], q_n, 0);
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradienttmpxyznBFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gtmpxyz_u, &gmsq_n[1*gnTNode], &gq[1*n], gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, gnBFace, gnTCell);
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuGradienttmpxyz <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gtmpxyz_u, &gmsq_n[1*gnTNode], gxfn, gyfn, gzfn, gf2c, gnNPF, 
															gF2N, gIndexF2N, garea, gnBFace, gnTCell, gnTFace);	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
#if (defined ShareMemory)	
	gpuGradientReductionShareMemory2 <<< blocksPerGrid, threadsPerBlock, 3*threadsPerBlock*sizeof(RealFlow), flowstream[1] >>> (
																	&gdqdx[1*n], &gdqdy[1*n], &gdqdz[1*n], 
																	gtmpxyz_u, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace, threadsPerBlock);
#else
	gpuGradientReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (&gdqdx[1*n], &gdqdy[1*n], &gdqdz[1*n], 
																	gtmpxyz_u, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace);
#endif	
#endif

	cudaStreamSynchronize(flowstream[2]);
    //传递并行边界点的加权值
#ifdef MPICH
    grid->CommInternodeDataMPI(&hostmsq_n[2*gnTNode]);
#endif
	
	//cuCompNodeq_nWeight(q_n);
	HANDLE_API_ERR(cudaMemcpyAsync(&gmsq_n[2*gnTNode], &hostmsq_n[2*gnTNode], gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[2]));	
	
	//如果单元含有一个以上的物面，该单元梯度采用Gauss求解
    if(vis_mode != INVISCID && level == 0){ 
        //IntType *cellwallnumber = grid->GetGridQualityCellWallNumber();
        //cuGradientBoundary(dqdx[0], dqdy[0], dqdz[0], 0);
		blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
		gpuGradientBoundary <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (&gdqdx[1*n], &gdqdy[1*n], &gdqdz[1*n], 
																&gq[1*n], gC2F, gIndexC2F, gf2c, gnFPC, gcellwallnumber, 
																garea, gxfn, gyfn, gzfn, gnTCell, n);
    }      

    //边界层前n层采用Gauss方法
    if(level == 0 && GaussLayer>0){
		gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (&gdqdx[1*n], &gdqdy[1*n], &gdqdz[1*n], 
																&gq[1*n], gC2F, gIndexC2F, gf2c, gnFPC, gCellLayerNo, garea, 
																gxfn, gyfn, gzfn, gGaussLayer, gnTCell, n);	
    }   
	
	//cuGradientvolaver(dqdx[0], dqdy[0], dqdz[0], 0);
	gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (&gdqdx[1*n], &gdqdy[1*n], &gdqdz[1*n], 
																gvol, gnTCell);					
	
	//cuCompGradientQ_Gauss_Node(grid, q[3],  dqdx[3],  dqdy[3], dqdz[3], 3, u_n, v_n, w_n);
	/* blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradientInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (&gdqdx[3*n], &gdqdy[3*n], &gdqdz[3*n], n);
	
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (&gmsq_n[3*gnTNode], gnTNode); */
	
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (gfacq, &gq[3*n], gf2c, gtype_bcr, gnBFace);
	
	gpuCompNodefacq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (&gmsq_n[3*gnTNode], gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gnBFace);

	gpuCompNodefacq2q_n2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (&gmsq_n[3*gnTNode], gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
	
	gpuCompNodefacq2q_n3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (&gmsq_n[3*gnTNode], gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
																
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (&gmsq_n[3*gnTNode], &gq[3*n], gWeightNodeN2C, gnCPN, 
															gN2C, gIndexN2C, gNmark, gnTNode);
															
	HANDLE_API_ERR(cudaMemcpyAsync(&hostmsq_n[3*gnTNode], &gmsq_n[3*gnTNode], gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[3]));
	
	cudaStreamSynchronize(flowstream[2]);
	gpuCompNodeq_nWeight <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (&gmsq_n[2*gnTNode], gWeightNode, gnTNode);
    
    //修正对称面顶点的速度
	//cuCompNodeuvw_n2vn();
	//gpuCompNodeuvw_n2vn <<< blocksPerGrid, threadsPerBlock >>> (gvn, gu_n, gv_n, gw_n, gxfn_n_symm, gyfn_n_symm, 
	//															gzfn_n_symm, gnodesymm, gnTNode);
																
	//cuCompNodeuvw_n2q_n(q_n, 1);
	gpuCompNodeuvw_n2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (&gmsq_n[2*gnTNode], gv_n, gvn, gyfn_n_symm, gnodesymm, gnTNode);
	
#if (defined FaceColoring)
	cuGradientFaceColor(grid, dqdx[2], dqdy[2], dqdz[2], &hostmsq_n[2*gnTNode], 0);
#else
	//cuGradientReduction(dqdx[0], dqdy[0], dqdz[0], q_n, 0);
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradienttmpxyznBFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (gtmpxyz_v, &gmsq_n[2*gnTNode], &gq[2*n], gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, gnBFace, gnTCell);
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuGradienttmpxyz <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (gtmpxyz_v, &gmsq_n[2*gnTNode], gxfn, gyfn, gzfn, gf2c, gnNPF, 
															gF2N, gIndexF2N, garea, gnBFace, gnTCell, gnTFace);	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
#if (defined ShareMemory)	
	gpuGradientReductionShareMemory2 <<< blocksPerGrid, threadsPerBlock, 3*threadsPerBlock*sizeof(RealFlow), flowstream[2] >>> (
																	&gdqdx[2*n], &gdqdy[2*n], &gdqdz[2*n], 
																	gtmpxyz_v, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace, threadsPerBlock);
#else
	gpuGradientReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (&gdqdx[2*n], &gdqdy[2*n], &gdqdz[2*n], 
																	gtmpxyz_v, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace);
#endif	
#endif

	cudaStreamSynchronize(flowstream[3]);
    //传递并行边界点的加权值
#ifdef MPICH
    grid->CommInternodeDataMPI(&hostmsq_n[3*gnTNode]);
#endif
	
	//cuCompNodeq_nWeight(q_n);
	HANDLE_API_ERR(cudaMemcpyAsync(&gmsq_n[3*gnTNode], &hostmsq_n[3*gnTNode], gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[3]));	
	
	//如果单元含有一个以上的物面，该单元梯度采用Gauss求解
    if(vis_mode != INVISCID && level == 0){ 
        //IntType *cellwallnumber = grid->GetGridQualityCellWallNumber();
        //cuGradientBoundary(dqdx[0], dqdy[0], dqdz[0], 0);
		blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
		gpuGradientBoundary <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (&gdqdx[2*n], &gdqdy[2*n], &gdqdz[2*n], 
																&gq[2*n], gC2F, gIndexC2F, gf2c, gnFPC, gcellwallnumber, 
																garea, gxfn, gyfn, gzfn, gnTCell, n);
    }      

    //边界层前n层采用Gauss方法
    if(level == 0 && GaussLayer>0){
		gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (&gdqdx[2*n], &gdqdy[2*n], &gdqdz[2*n], 
																&gq[2*n], gC2F, gIndexC2F, gf2c, gnFPC, gCellLayerNo, garea, 
																gxfn, gyfn, gzfn, gGaussLayer, gnTCell, n);	
    }   
	
	//cuGradientvolaver(dqdx[0], dqdy[0], dqdz[0], 0);
	gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[2] >>> (&gdqdx[2*n], &gdqdy[2*n], &gdqdz[2*n], 
																gvol, gnTCell);					
	
		
	
	cudaStreamSynchronize(flowstream[3]);
	gpuCompNodeq_nWeight <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (&gmsq_n[3*gnTNode], gWeightNode, gnTNode);
    
    //修正对称面顶点的速度
	//cuCompNodeuvw_n2vn();
	//gpuCompNodeuvw_n2vn <<< blocksPerGrid, threadsPerBlock >>> (gvn, gu_n, gv_n, gw_n, gxfn_n_symm, gyfn_n_symm, 
	//															gzfn_n_symm, gnodesymm, gnTNode);
																
	//cuCompNodeuvw_n2q_n(q_n, 1);
	gpuCompNodeuvw_n2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (&gmsq_n[3*gnTNode], gw_n, gvn, gzfn_n_symm, gnodesymm, gnTNode);
	
#if (defined FaceColoring)
	cuGradientFaceColor(grid, dqdx[3], dqdy[3], dqdz[3], &hostmsq_n[3*gnTNode], 0);
#else
	//cuGradientReduction(dqdx[0], dqdy[0], dqdz[0], q_n, 0);
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradienttmpxyznBFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (gtmpxyz_w, &gmsq_n[3*gnTNode], &gq[3*n], gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, gnBFace, gnTCell);
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuGradienttmpxyz <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (gtmpxyz_w, &gmsq_n[3*gnTNode], gxfn, gyfn, gzfn, gf2c, gnNPF, 
															gF2N, gIndexF2N, garea, gnBFace, gnTCell, gnTFace);	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
#if (defined ShareMemory)	
	gpuGradientReductionShareMemory2 <<< blocksPerGrid, threadsPerBlock, 3*threadsPerBlock*sizeof(RealFlow), flowstream[3] >>> (
																	&gdqdx[3*n], &gdqdy[3*n], &gdqdz[3*n], 
																	gtmpxyz_w, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace, threadsPerBlock);
#else
	gpuGradientReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (&gdqdx[3*n], &gdqdy[3*n], &gdqdz[3*n], 
																	gtmpxyz_w, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace);
#endif	
#endif
	
	//如果单元含有一个以上的物面，该单元梯度采用Gauss求解
    if(vis_mode != INVISCID && level == 0){ 
        //IntType *cellwallnumber = grid->GetGridQualityCellWallNumber();
        //cuGradientBoundary(dqdx[0], dqdy[0], dqdz[0], 0);
		blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
		gpuGradientBoundary <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (&gdqdx[3*n], &gdqdy[3*n], &gdqdz[3*n], 
																&gq[3*n], gC2F, gIndexC2F, gf2c, gnFPC, gcellwallnumber, 
																garea, gxfn, gyfn, gzfn, gnTCell, n);
    }      

    //边界层前n层采用Gauss方法
    if(level == 0 && GaussLayer>0){
		gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (&gdqdx[3*n], &gdqdy[3*n], &gdqdz[3*n], 
																&gq[3*n], gC2F, gIndexC2F, gf2c, gnFPC, gCellLayerNo, garea, 
																gxfn, gyfn, gzfn, gGaussLayer, gnTCell, n);	
    }   
	
	//cuGradientvolaver(dqdx[0], dqdy[0], dqdz[0], 0);
	gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[3] >>> (&gdqdx[3*n], &gdqdy[3*n], &gdqdz[3*n], 
																gvol, gnTCell);									
	
#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    timecost[0] = timecost[0] + time_tmp + MPI_Wtime();
    time_gradient = time_gradient + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemGradient, 0); 
    timeuseTemGradient = (RealGeom) 1000000*(endtimeTemGradient.tv_sec - starttimeTemGradient.tv_sec) + endtimeTemGradient.tv_usec - starttimeTemGradient.tv_usec;
    timecost[0] += timeuseTemGradient;
    timeuseTemGradient /= 1000000.0;
    time_gradient += timeuseTemGradient;
#endif
#endif
	cudaStreamSynchronize(flowstream[0]);
	cudaStreamSynchronize(flowstream[1]);
	cudaStreamSynchronize(flowstream[2]);
	cudaStreamSynchronize(flowstream[3]);
	cudaStreamSynchronize(flowstream[4]);
	
}
#endif



__device__ double atomicAddSM35(double* address, double val)
{
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
}


