#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include <utility_functions.h>
#include <number_type.h>
#include "zone.h"
#include "solver_ns.h"
#include "temporal_discretisation_implicit.h"
#include "io_base_format.h"
#include "io_log.h"
#include "io_field.h"
#include "parallel_base_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"

#include <cuLimit.cuh>
#include <cuData.cuh>
#include <cuErrorReturn.cuh>
#include <cuGradientQ_Gauss.cuh>
#include <cuViscidFlux.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

using namespace gpuData;

__global__ void gpuMaxMinDiffInit(RealFlow *dmax, RealFlow *dmin, const RealFlow *q, 
								const IntType nTCell, const IntType nBFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType Cell = nTCell + nBFace;
	if(i < nTCell){
		dmax[i] =  q[name*Cell + i];
        dmin[i] =  q[name*Cell + i];
	}	
}

__global__ void gpuMaxMinDiff(RealFlow *dmax, RealFlow *dmin, const RealFlow *q, const IntType* C2F,
							const IntType* IndexC2F, const IntType* nFPC, const IntType* f2c, 
							const IntType* type_bcr, const IntType nTCell, const IntType nBFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face, type;
	IntType continueflag = 0;
	IntType Cell = nTCell + nBFace;
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			continueflag = 0;
			face = C2F[IndexC2F[i] + j];
			c1 = f2c[2*face];
			c2 = f2c[2*face + 1];
			if (face < nBFace) {
                type = type_bcr[face];
                if (type != INTERFACE) {
					continueflag = 1;
				}
            }
			if (continueflag == 0){
				if (i == c1) {
					dmax[c1] = GPUMAX(dmax[c1], q[name*Cell + c2]);
					dmin[c1] = GPUMIN(dmin[c1], q[name*Cell + c2]);
				}
				else if (i == c2){
					dmax[c2] = GPUMAX(dmax[c2], q[name*Cell + c1]);
					dmin[c2] = GPUMIN(dmin[c2], q[name*Cell + c1]);
				}
			}
			
		}
	}																		
								
}

#if (defined ShareMemory)
__global__ void gpuMaxMinDiffShareMemory(RealFlow *dmax, RealFlow *dmin, const RealFlow *q, const IntType* C2F,
							const IntType* IndexC2F, const IntType* nFPC, const IntType* f2c, 
							const IntType* type_bcr, const IntType nTCell, const IntType nBFace, const IntType name, const IntType threadsnum){
	extern __shared__ double sdata[];							
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face, type;
	IntType continueflag = 0;
	IntType Cell = nTCell + nBFace;
	
	if(i < nTCell){
		sdata[threadIdx.x] = dmax[i];
		sdata[threadsnum + threadIdx.x] = dmin[i];
	}
	__syncthreads();
	
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			continueflag = 0;
			face = C2F[IndexC2F[i] + j];
			c1 = f2c[2*face];
			c2 = f2c[2*face + 1];
			if (face < nBFace) {
                type = type_bcr[face];
                if (type != INTERFACE) {
					continueflag = 1;
				}
            }
			if (continueflag == 0){
				if (i == c1) {
					sdata[threadIdx.x] = GPUMAX(sdata[threadIdx.x], q[name*Cell + c2]);
					sdata[threadsnum + threadIdx.x] = GPUMIN(sdata[threadsnum + threadIdx.x], q[name*Cell + c2]);
				}
				else if (i == c2){
					sdata[threadIdx.x] = GPUMAX(sdata[threadIdx.x], q[name*Cell + c1]);
					sdata[threadsnum + threadIdx.x] = GPUMIN(sdata[threadsnum + threadIdx.x], q[name*Cell + c1]);
				}
			}
			
		}
	}
	__syncthreads();

	if(i < nTCell){
		dmax[i] = sdata[threadIdx.x];
		dmin[i] = sdata[threadsnum + threadIdx.x];
	}
								
}
#endif

__global__ void gpuMaxMinDiffReduceQ(RealFlow *dmax, RealFlow *dmin, const RealFlow *q, 
								const IntType nTCell, const IntType nBFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType Cell = nTCell + nBFace;
	if(i < nTCell){
		dmax[i] -=  q[name*Cell + i];
        dmin[i] -=  q[name*Cell + i];
	}		
									
}

void cuMaxMinDiff(PolyGrid *grid, IntType name){

    // Find the maximum and minimum in the neighbor of each cell
    int blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuMaxMinDiffInit <<< blocksPerGrid, threadsPerBlock >>> (gdmax, gdmin, gq, gnTCell, gnBFace, name);

#if (defined ShareMemory)	
	// Manual reduction	
	gpuMaxMinDiffShareMemory  <<< blocksPerGrid, threadsPerBlock, 2*threadsPerBlock*sizeof(RealFlow) >>> (gdmax, gdmin, gq, gC2F, gIndexC2F, gnFPC, gf2c, 
														gtype_bcr, gnTCell, gnBFace, name, threadsPerBlock);	
#else
	gpuMaxMinDiff  <<< blocksPerGrid, threadsPerBlock >>> (gdmax, gdmin, gq, gC2F, gIndexC2F, gnFPC, gf2c, 
														gtype_bcr, gnTCell, gnBFace, name);	
#endif	
	gpuMaxMinDiffReduceQ  <<< blocksPerGrid, threadsPerBlock >>> (gdmax, gdmin, gq, gnTCell, gnBFace, name);
		
	
}

__global__ void gpuLimitInit(RealFlow *limit, const IntType nTCell, const IntType nBFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType Cell = nTCell + nBFace;
	if(i < Cell){
		if (i < nTCell){
			limit[name*Cell + i] =  BIG;
		}
		else{
			limit[name*Cell + i] =  1.0;
		}
	}		
									
}

__global__ void gpuLimitespcell(RealFlow *espcell, const RealGeom *vol, const RealFlow *q, RealGeom eps_tmp,
								const IntType nTCell, const IntType nBFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType Cell = nTCell + nBFace;
	if(i < nTCell){
		espcell[i] = eps_tmp*vol[i]*q[name*Cell + i]*q[name*Cell + i];
	}
	
}

__global__ void gpuLimitespcell3(RealFlow *espcell, const RealGeom *vol, const RealFlow *q, RealGeom eps_tmp,
								const RealFlow gam, const RealFlow p_bar, 
								const IntType nTCell, const IntType nBFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType Cell = nTCell + nBFace;
	RealFlow tmp;
	if(i < nTCell){
		tmp  = gam*(q[4*Cell + i] + p_bar)/q[i];
        espcell[i] = eps_tmp*vol[i]*tmp;
	}	
}

__global__ void gpuLimitespcell4(RealFlow *espcell, const RealGeom *vol, const RealFlow *q, const RealGeom eps_tmp, 
								const RealFlow p_bar, const IntType nTCell, const IntType nBFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType Cell = nTCell + nBFace;
	if(i < nTCell){
		espcell[i] = eps_tmp*vol[i]*(q[name*Cell + i] + p_bar)*(q[name*Cell + i] + p_bar);
	}	
}

__global__ void gpuVencatLimitnBFace(RealFlow *tmp_limit, const RealFlow *dmax, const RealFlow *dmin, const RealFlow *espcell, 
									const RealGeom eps_tmp, const RealFlow *dqdx, const RealFlow *dqdy, const RealFlow *dqdz,
									const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc,
									const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, const IntType *f2c, const IntType nTCell, 
									const IntType nBFace, const IntType nTFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType Cell = nTCell + nBFace;
	IntType c1, count;
	RealGeom dx, dy, dz, eps;
	RealFlow dq_face, tmp;
	
	if(i < nBFace){
		count = 2 * i;
        c1 = f2c[count];

        dx = xfc[i] - xcc[c1];
        dy = yfc[i] - ycc[c1];
        dz = zfc[i] - zcc[c1];
        dq_face = dqdx[name*Cell + c1] * dx + dqdy[name*Cell + c1] * dy + dqdz[name*Cell + c1] * dz;

        eps = espcell[c1];
        //if ((dq_face > -TINY) && (dq_face < TINY))
		if (GPUEqualZero(dq_face))
            tmp = 1.0;
        else {
            if (dq_face > 0.0) {
                tmp = gpuVenFun(dmax[c1], dq_face, eps);
            }
            else {
                tmp = gpuVenFun(dmin[c1], dq_face, eps);
            }
            tmp /= dq_face;
        }
        tmp_limit[count] = tmp;
	}																			
}

__global__ void gpuVencatLimit(RealFlow *tmp_limit, const RealFlow *dmax, const RealFlow *dmin, const RealFlow *espcell, 
								const RealGeom eps_tmp, const RealFlow *dqdx, const RealFlow *dqdy, const RealFlow *dqdz,
								const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc,
								const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, const IntType *f2c, const IntType nTCell, 
								const IntType nBFace, const IntType nTFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x + nBFace;
	IntType Cell = nTCell + nBFace;
	IntType c1, c2, count;
	RealGeom dx, dy, dz, eps;
	RealFlow dq_face, tmp;
	
	if(i < nTFace){
		count = 2 * i;
        c1 = f2c[count];
        c2 = f2c[count + 1];

        dx = xfc[i] - xcc[c1];
        dy = yfc[i] - ycc[c1];
        dz = zfc[i] - zcc[c1];
        dq_face = dqdx[name*Cell + c1] * dx + dqdy[name*Cell + c1] * dy + dqdz[name*Cell + c1] * dz;

        eps = espcell[c1];
        if (GPUEqualZero(dq_face))
            tmp = 1.0;
        else {
            if (dq_face > 0.0) {
                tmp = gpuVenFun(dmax[c1], dq_face, eps);
            }
            else {
                tmp = gpuVenFun(dmin[c1], dq_face, eps);
            }
            tmp /= dq_face;
        }
        tmp_limit[count] = tmp;

        dx = xfc[i] - xcc[c2];
        dy = yfc[i] - ycc[c2];
        dz = zfc[i] - zcc[c2];
        dq_face = dqdx[name*Cell + c2] * dx + dqdy[name*Cell + c2] * dy + dqdz[name*Cell + c2] * dz;

        eps = espcell[c2];

        if (GPUEqualZero(dq_face))
            tmp = 1.0;
        else {
            if (dq_face > 0.0) {
                tmp = gpuVenFun(dmax[c2], dq_face, eps);
            }
            else {
                tmp = gpuVenFun(dmin[c2], dq_face, eps);
            }
            tmp /= dq_face;
        }
        tmp_limit[count + 1] = tmp;		
	}
	
}

__global__ void gpuVencatLimitAtomicnBFace(RealFlow *limit, const RealFlow *dmax, const RealFlow *dmin, const RealFlow *espcell, 
									const RealGeom eps_tmp, const RealFlow *dqdx, const RealFlow *dqdy, const RealFlow *dqdz,
									const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc,
									const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, const IntType *f2c, const IntType nTCell, 
									const IntType nBFace, const IntType nTFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType Cell = nTCell + nBFace;
	IntType c1;
	RealGeom dx, dy, dz, eps;
    RealFlow dq_face, tmp;
	if(i < nBFace){
		c1 = f2c[2*i];
        
        dx       = xfc[i] - xcc[c1];
        dy       = yfc[i] - ycc[c1];
        dz       = zfc[i] - zcc[c1];
        dq_face  = dqdx[name*Cell + c1]*dx + dqdy[name*Cell + c1]*dy + dqdz[name*Cell + c1]*dz;
        
        eps = espcell[c1];
        if ((dq_face > -TINY) && (dq_face < TINY))
            tmp = 1.0;
        else{ 
            if(dq_face > 0.0){
                tmp = ((dmax[c1] * dmax[c1] + eps + (dq_face + dq_face) * dmax[c1]) * dq_face / 
					(dmax[c1] * dmax[c1] + (dq_face + dq_face + dmax[c1]) * dq_face + eps));
            }else{
                tmp = ((dmin[c1] * dmin[c1] + eps + (dq_face + dq_face) * dmin[c1]) * dq_face / 
					(dmin[c1] * dmin[c1] + (dq_face + dq_face + dmin[c1]) * dq_face + eps));
            }
            tmp /= dq_face; 
        }
        atomicMin(limit+name*Cell + c1, tmp);
		
	}
	
}

__global__ void gpuVencatLimitReduction(RealFlow *limit, const RealFlow *tmp_limit, const IntType *f2c, const IntType* C2F,
										const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell, const IntType nBFace, 
										const IntType nTFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face, count;
	IntType Cell = nTCell + nBFace;
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			
			face = C2F[IndexC2F[i] + j];
            count = 2 * face;
			
            c1 = f2c[count];
            c2 = f2c[count + 1];
			
            if (i == c1) {
                limit[name*Cell + c1] = GPUMIN(limit[name*Cell + c1], tmp_limit[count]);
            }
            else if (i == c2) {
                limit[name*Cell + c2] = GPUMIN(limit[name*Cell + c2], tmp_limit[count + 1]);
            }					
								
		}
	}		
}

#if (defined ShareMemory)
__global__ void gpuVencatLimitReductionShareMemory(RealFlow *limit, const RealFlow *tmp_limit, const IntType *f2c, const IntType* C2F,
										const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell, const IntType nBFace, 
										const IntType nTFace, const IntType name){
	extern __shared__ double sdata[];
	
	IntType tid = threadIdx.x;
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face, count;
	IntType Cell = nTCell + nBFace;		
	
	if(i < nTCell){
		sdata[tid] = limit[name*Cell + i];
	}
	__syncthreads();
	
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			
			face = C2F[IndexC2F[i] + j];
            count = 2 * face;
			
            c1 = f2c[count];
            c2 = f2c[count + 1];
			
            if (i == c1) {
                sdata[tid] = GPUMIN(sdata[tid], tmp_limit[count]);
            }
            else if (i == c2) {
                sdata[tid] = GPUMIN(sdata[tid], tmp_limit[count + 1]);
            }					
								
		}
	}
	__syncthreads();

	if(i < nTCell){
		limit[name*Cell + i] = sdata[tid];
	}
	
}
#endif

#ifdef MultiStream

__global__ void gpuMaxMinDiffInit_Merged(RealFlow *dmax, RealFlow *dmin, const RealFlow *q, 
								const IntType nTCell, const IntType nBFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType Cell = nTCell + nBFace;
	if(i < nTCell){
		dmax[i] =  q[i];
        dmin[i] =  q[i];
		dmax[nTCell + i] =  q[Cell + i];
        dmin[nTCell + i] =  q[Cell + i];
		dmax[2*nTCell + i] =  q[2*Cell + i];
        dmin[2*nTCell + i] =  q[2*Cell + i];
		dmax[3*nTCell + i] =  q[3*Cell + i];
        dmin[3*nTCell + i] =  q[3*Cell + i];
		dmax[4*nTCell + i] =  q[4*Cell + i];
        dmin[4*nTCell + i] =  q[4*Cell + i]; 
		/* for (IntType j = 0; j < 5; j++){
			dmax[j*nTCell + i] =  q[j*Cell + i];
			dmin[j*nTCell + i] =  q[j*Cell + i];
		}*/
	}	
}

__global__ void gpuMaxMinDiff_Merged(RealFlow *dmax, RealFlow *dmin, const RealFlow *q, const IntType* C2F,
							const IntType* IndexC2F, const IntType* nFPC, const IntType* f2c, 
							const IntType* type_bcr, const IntType nTCell, const IntType nBFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face, type;
	IntType continueflag = 0;
	IntType Cell = nTCell + nBFace;
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			continueflag = 0;
			face = C2F[IndexC2F[i] + j];
			c1 = f2c[2*face];
			c2 = f2c[2*face + 1];
			if (face < nBFace) {
                type = type_bcr[face];
                if (type != INTERFACE) {
					continueflag = 1;
				}
            }
			if (continueflag == 0){
				if (i == c1) {
					dmax[c1] = GPUMAX(dmax[c1], q[c2]);
					dmin[c1] = GPUMIN(dmin[c1], q[c2]);
					dmax[nTCell + c1] = GPUMAX(dmax[nTCell + c1], q[Cell + c2]);
					dmin[nTCell + c1] = GPUMIN(dmin[nTCell + c1], q[Cell + c2]);		
					dmax[2*nTCell + c1] = GPUMAX(dmax[2*nTCell + c1], q[2*Cell + c2]);
					dmin[2*nTCell + c1] = GPUMIN(dmin[2*nTCell + c1], q[2*Cell + c2]);					
					dmax[3*nTCell + c1] = GPUMAX(dmax[3*nTCell + c1], q[3*Cell + c2]);
					dmin[3*nTCell + c1] = GPUMIN(dmin[3*nTCell + c1], q[3*Cell + c2]);
					dmax[4*nTCell + c1] = GPUMAX(dmax[4*nTCell + c1], q[4*Cell + c2]);
					dmin[4*nTCell + c1] = GPUMIN(dmin[4*nTCell + c1], q[4*Cell + c2]);
				}
				else if (i == c2){
					dmax[c2] = GPUMAX(dmax[c2], q[c1]);
					dmin[c2] = GPUMIN(dmin[c2], q[c1]);
					dmax[nTCell + c2] = GPUMAX(dmax[nTCell + c2], q[Cell + c1]);
					dmin[nTCell + c2] = GPUMIN(dmin[nTCell + c2], q[Cell + c1]);
					dmax[2*nTCell + c2] = GPUMAX(dmax[2*nTCell + c2], q[2*Cell + c1]);
					dmin[2*nTCell + c2] = GPUMIN(dmin[2*nTCell + c2], q[2*Cell + c1]);
					dmax[3*nTCell + c2] = GPUMAX(dmax[3*nTCell + c2], q[3*Cell + c1]);
					dmin[3*nTCell + c2] = GPUMIN(dmin[3*nTCell + c2], q[3*Cell + c1]);
					dmax[4*nTCell + c2] = GPUMAX(dmax[4*nTCell + c2], q[4*Cell + c1]);
					dmin[4*nTCell + c2] = GPUMIN(dmin[4*nTCell + c2], q[4*Cell + c1]);
				}
			}						
			
		}
	}																		
								
}

__global__ void gpuMaxMinDiffReduceQ_Merged(RealFlow *dmax, RealFlow *dmin, const RealFlow *q, 
								const IntType nTCell, const IntType nBFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType Cell = nTCell + nBFace;
	if(i < nTCell){
		dmax[i] -=  q[i];
        dmin[i] -=  q[i];
		dmax[nTCell + i] -=  q[Cell + i];
        dmin[nTCell + i] -=  q[Cell + i];
		dmax[2*nTCell + i] -=  q[2*Cell + i];
        dmin[2*nTCell + i] -=  q[2*Cell + i];
		dmax[3*nTCell + i] -=  q[3*Cell + i];
        dmin[3*nTCell + i] -=  q[3*Cell + i];
		dmax[4*nTCell + i] -=  q[4*Cell + i];
        dmin[4*nTCell + i] -=  q[4*Cell + i]; 
		/* for (IntType j = 0; j < 5; j++){
			dmax[j*nTCell + i] -=  q[j*Cell + i];
			dmin[j*nTCell + i] -=  q[j*Cell + i];
		} */
	}		
									
}

__global__ void gpuLimitInit_Merged(RealFlow *limit, const IntType nTCell, const IntType nBFace, const IntType name){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType Cell = nTCell + nBFace;
	if(i < Cell){
		if (i < nTCell){
			limit[i] =  BIG;
			limit[Cell + i] =  BIG;
			limit[2*Cell + i] =  BIG;
			limit[3*Cell + i] =  BIG;
			limit[4*Cell + i] =  BIG;
		}
		else{
			limit[i] =  1.0;
			limit[Cell + i] =  1.0;
			limit[2*Cell + i] =  1.0;
			limit[3*Cell + i] =  1.0;
			limit[4*Cell + i] =  1.0;
		}
	}		
									
}

void cuMaxMinDiff_MultiStream(IntType name){
	
    // Find the maximum and minimum in the neighbor of each cell
    int blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuMaxMinDiffInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);
	
	// Manual reduction	
	gpuMaxMinDiff  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gC2F, gIndexC2F, gnFPC, gf2c, 
														gtype_bcr, gnTCell, gnBFace, name);	
	
	gpuMaxMinDiffReduceQ  <<< blocksPerGrid, threadsPerBlock, 0, flowstream[name] >>> (&gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], gq, gnTCell, gnBFace, name);	
	
}

void cuVencatLimiter_MultiStream(PolyGrid *grid, IntType name){  
    
    RealGeom eps_tmp;    
    RealFlow gam;
    
	if(name>0 && name<4){
        grid->GetData(&gam, REAL_FLOW, 1, "gam");    
    }    
    
    RealFlow vol_avg = grid->GetVolAvg();
    assert(vol_avg > 0.0); //volumn average must exist
	
	RealFlow eps_vencat=1.0;
    grid->GetData(&eps_vencat, REAL_FLOW, 1, "eps_vencat",0);
    eps_tmp = eps_vencat*eps_vencat*eps_vencat/vol_avg;  	     
	
	//cuVencatLimiter_MultiStream_espcell(grid, name);
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
		
	// Atomic:
	//gpuVencatLimitAtomicnBFace <<< blocksPerGrid, threadsPerBlock >>> (gtmp_limit, gdmax, gdmin, gespcell, eps_tmp, gdqdx, gdqdy, gdqdz,
	//															gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	//HANDLE_API_ERR(cudaMemcpy(limit, &glimit[name*(gnTCell + gnBFace)], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
				
	gpuVencatLimitnBFace <<< blocksPerGrid, threadsPerBlock >>> (gtmp_limit, &gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], &gespcell_MStream[name*gnTCell], eps_tmp, gdqdx, gdqdy, gdqdz,
																gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuVencatLimit <<< blocksPerGrid, threadsPerBlock >>> (gtmp_limit, &gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], &gespcell_MStream[name*gnTCell], eps_tmp, gdqdx, gdqdy, gdqdz,
														gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuVencatLimitReduction <<< blocksPerGrid, threadsPerBlock >>> (glimit, gtmp_limit, gf2c, gC2F, gIndexC2F, 
																	gnFPC, gnTCell, gnBFace, gnTFace, name);
	//HANDLE_API_ERR(cudaMemcpy(limit, &glimit[name*(gnTCell + gnBFace)], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//HANDLE_API_ERR(cudaMemcpy(limit, &glimit[name*(gnTCell + gnBFace)], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	    
}

void cuVencatLimiter_MultiStream_Grad_T(PolyGrid *grid){
	
	RealFlow gascon;
    grid->GetData(&gascon, REAL_FLOW, 1, "gascon"); 
	// Get Temperature:
	IntType blocksPerGrid = (gnTCell + gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuGetTemperature <<< blocksPerGrid, threadsPerBlock >>> (gt, gq, gascon, gp_bar, gnBFace, gnTCell);	
    
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq_n, gnTNode);
	
    //计算物理边界点的值，使用物理面的值进行加权计算，不包括并行边界和对称面边界
    //计算物理边界面心的物理值
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gfacq, gt, gf2c, gtype_bcr, gnBFace);
	
    //利用物理边界面心的值，计算物理边界点
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq_n, gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gnBFace);
			
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq2q_n2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq_n, gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
	
	//cuCompNodefacq2q_n3(q_n);	
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodefacq2q_n3 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq_n, gfacq, gWeightNodeBFace2C, gtype_bcr, 
																gF2N, gIndexF2N, gnNPF, gNmark, gnBFace);
	
	// Limit para:
	RealGeom eps_tmp;    
    RealFlow gam;
    	
    grid->GetData(&gam, REAL_FLOW, 1, "gam");                
    RealFlow vol_avg = grid->GetVolAvg();
    assert(vol_avg > 0.0); //volumn average must exist
	
	RealFlow eps_vencat=1.0;
    grid->GetData(&eps_vencat, REAL_FLOW, 1, "eps_vencat",0);
    eps_tmp = eps_vencat*eps_vencat*eps_vencat/vol_avg;   
    
	
    //计算其他点的物理值，使用与其相相邻的控制体体心值	
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	gpuCompNodeq2q_n <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq_n, gt, gWeightNodeN2C, gnCPN, 
															gN2C, gIndexN2C, gNmark, gnTNode);
	HANDLE_API_ERR(cudaMemcpyAsync(hostmsq_n, gq_n, gnTNode*sizeof(RealFlow), cudaMemcpyDeviceToHost, flowstream[0]));
	
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	IntType name = 0;
	gpuVencatLimitnBFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gtmp_limit, &gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], &gespcell_MStream[name*gnTCell], eps_tmp, gdqdx, gdqdy, gdqdz,
																gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuVencatLimit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gtmp_limit, &gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], &gespcell_MStream[name*gnTCell], eps_tmp, gdqdx, gdqdy, gdqdz,
														gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuVencatLimitReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (glimit, gtmp_limit, gf2c, gC2F, gIndexC2F, 
																	gnFPC, gnTCell, gnBFace, gnTFace, name);
																	
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;		
	name = 1;
	gpuVencatLimitnBFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gtmp_limit, &gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], &gespcell_MStream[name*gnTCell], eps_tmp, gdqdx, gdqdy, gdqdz,
																gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuVencatLimit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gtmp_limit, &gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], &gespcell_MStream[name*gnTCell], eps_tmp, gdqdx, gdqdy, gdqdz,
														gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuVencatLimitReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (glimit, gtmp_limit, gf2c, gC2F, gIndexC2F, 
																	gnFPC, gnTCell, gnBFace, gnTFace, name);																																	
	
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;		
	name = 2;
	gpuVencatLimitnBFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gtmp_limit, &gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], &gespcell_MStream[name*gnTCell], eps_tmp, gdqdx, gdqdy, gdqdz,
																gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuVencatLimit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gtmp_limit, &gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], &gespcell_MStream[name*gnTCell], eps_tmp, gdqdx, gdqdy, gdqdz,
														gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuVencatLimitReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (glimit, gtmp_limit, gf2c, gC2F, gIndexC2F, 
																	gnFPC, gnTCell, gnBFace, gnTFace, name);	
	
	cudaStreamSynchronize(flowstream[0]);
    //传递并行边界点的加权值
#ifdef MPICH
    grid->CommInternodeDataMPI(hostmsq_n);
#endif			
	
	HANDLE_API_ERR(cudaMemcpyAsync(gq_n, hostmsq_n, gnTNode*sizeof(RealFlow), cudaMemcpyHostToDevice, flowstream[0]));
	
	//cudaStreamSynchronize(flowstream[1]);		
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;		
	name = 3;
	gpuVencatLimitnBFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gtmp_limit, &gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], &gespcell_MStream[name*gnTCell], eps_tmp, gdqdx, gdqdy, gdqdz,
																gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuVencatLimit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gtmp_limit, &gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], &gespcell_MStream[name*gnTCell], eps_tmp, gdqdx, gdqdy, gdqdz,
														gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuVencatLimitReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (glimit, gtmp_limit, gf2c, gC2F, gIndexC2F, 
																	gnFPC, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;		
	name = 4;
	gpuVencatLimitnBFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gtmp_limit, &gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], &gespcell_MStream[name*gnTCell], eps_tmp, gdqdx, gdqdy, gdqdz,
																gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuVencatLimit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (gtmp_limit, &gdmax_MStream[name*gnTCell], &gdmin_MStream[name*gnTCell], &gespcell_MStream[name*gnTCell], eps_tmp, gdqdx, gdqdy, gdqdz,
														gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuVencatLimitReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[1] >>> (glimit, gtmp_limit, gf2c, gC2F, gIndexC2F, 
																	gnFPC, gnTCell, gnBFace, gnTFace, name);
    
	
	cudaStreamSynchronize(flowstream[0]);
	blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuCompNodeq_nWeight <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gq_n, gWeightNode, gnTNode);	
	
#if (defined FaceColoring)
	cuGradientFaceColor(grid, dtdx, dtdy, dtdz, hostmsq_n, name);
#else
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradienttmpxyznBFace <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gtmpxyz, gq_n, gt, gxfn, gyfn, gzfn, gf2c, gtype_bcr, 
																	gnNPF, gF2N, gIndexF2N, garea, gnBFace, gnTCell);
		
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
		
	gpuGradienttmpxyz <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gtmpxyz, gq_n, gxfn, gyfn, gzfn, gf2c, gnNPF, 
															gF2N, gIndexF2N, garea, gnBFace, gnTCell, gnTFace);	
	
	// Reduction:
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	#if (defined ShareMemory)	
		gpuGradientReductionShareMemory2 <<< blocksPerGrid, threadsPerBlock, 3*threadsPerBlock*sizeof(RealFlow), flowstream[0] >>> (gdtdx, gdtdy, gdtdz, 
																		gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace, threadsPerBlock);
	#else
		gpuGradientReduction <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdtdx, gdtdy, gdtdz, 
																		gtmpxyz, gf2c, gC2F, gIndexC2F, gnFPC, gnTCell, gnBFace);
	#endif	
#endif		
	
	IntType level  = grid->GetLevel();    
    IntType vis_mode;
    grid->GetData(&vis_mode,  INT, 1, "vis_mode");    
    //如果单元含有一个以上的物面，该单元梯度采用Gauss求解
    if(vis_mode != INVISCID && level == 0){ 
        //cuGradientBoundary(dtdx, dtdy, dtdz, name);
		blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
		gpuGradientBoundary <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdtdx, gdtdy, gdtdz, 
																gt, gC2F, gIndexC2F, gf2c, gnFPC, gcellwallnumber, 
																garea, gxfn, gyfn, gzfn, gnTCell, gnTCell + gnBFace);
    }      

    //边界层前n层采用Gauss方法
    IntType GaussLayer = -1;
    grid->GetData(&GaussLayer, INT, 1, "GaussLayer");
    if(level == 0 && GaussLayer>0){
		blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
		gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdtdx, gdtdy, gdtdz, 
																gt, gC2F, gIndexC2F, gf2c, gnFPC, gCellLayerNo, garea, 
																gxfn, gyfn, gzfn, gGaussLayer, gnTCell, gnTCell + gnBFace);
    }   
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuGradientBoundary2 <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdtdx, gdtdy, gdtdz, 
																gvol, gnTCell);					
	
	cudaStreamSynchronize(flowstream[0]);		
	cudaStreamSynchronize(flowstream[1]);	
	/* cuVencatLimiter_MultiStream(grid, 0);
	cuVencatLimiter_MultiStream(grid, 1);
	cuVencatLimiter_MultiStream(grid, 2);
	cuVencatLimiter_MultiStream(grid, 3);
	cuVencatLimiter_MultiStream(grid, 4); */
}

#endif

void cuVencatLimiter(PolyGrid *grid, IntType name){  
    
    RealGeom eps_tmp;    
    RealFlow gam;
    
	if(name>0 && name<4){
        grid->GetData(&gam, REAL_FLOW, 1, "gam");    
    }    
    
    RealFlow vol_avg = grid->GetVolAvg();
    assert(vol_avg > 0.0); //volumn average must exist
	 
    // Find the the differences for q
    cuMaxMinDiff(grid, name);

    RealFlow eps_vencat=1.0;
    grid->GetData(&eps_vencat, REAL_FLOW, 1, "eps_vencat",0);
    eps_tmp = eps_vencat*eps_vencat*eps_vencat/vol_avg;  
	
    //initial limit[i]
	int blocksPerGrid = (gnTCell + gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuLimitInit <<< blocksPerGrid, threadsPerBlock >>> (glimit, gnTCell, gnBFace, name);
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	
    switch(name){
        case 0:
			gpuLimitespcell <<< blocksPerGrid, threadsPerBlock >>> (gespcell, gvol, gq, eps_tmp, gnTCell, gnBFace, name);			
			break;
        case 1:
        case 2:
        case 3: 
			gpuLimitespcell3 <<< blocksPerGrid, threadsPerBlock >>> (gespcell, gvol, gq, eps_tmp, gam, gp_bar, gnTCell, gnBFace, name);		
            break;
        case 4:
			gpuLimitespcell4 <<< blocksPerGrid, threadsPerBlock >>> (gespcell, gvol, gq, eps_tmp, gp_bar, gnTCell, gnBFace, name);				
            break;
    }
	
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
		
	// Atomic:
	//gpuVencatLimitAtomicnBFace <<< blocksPerGrid, threadsPerBlock >>> (gtmp_limit, gdmax, gdmin, gespcell, eps_tmp, gdqdx, gdqdy, gdqdz,
	//															gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	//HANDLE_API_ERR(cudaMemcpy(limit, &glimit[name*(gnTCell + gnBFace)], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
				
	gpuVencatLimitnBFace <<< blocksPerGrid, threadsPerBlock >>> (gtmp_limit, gdmax, gdmin, gespcell, eps_tmp, gdqdx, gdqdy, gdqdz,
																gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuVencatLimit <<< blocksPerGrid, threadsPerBlock >>> (gtmp_limit, gdmax, gdmin, gespcell, eps_tmp, gdqdx, gdqdy, gdqdz,
														gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gf2c, gnTCell, gnBFace, gnTFace, name);
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
#if (defined ShareMemory)
	gpuVencatLimitReductionShareMemory <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow) >>> (glimit, gtmp_limit, gf2c, gC2F, gIndexC2F, 
																	gnFPC, gnTCell, gnBFace, gnTFace, name);
#else 
	gpuVencatLimitReduction <<< blocksPerGrid, threadsPerBlock >>> (glimit, gtmp_limit, gf2c, gC2F, gIndexC2F, 
																	gnFPC, gnTCell, gnBFace, gnTFace, name);
#endif
	//HANDLE_API_ERR(cudaMemcpy(limit, &glimit[name*(gnTCell + gnBFace)], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//HANDLE_API_ERR(cudaMemcpy(limit, &glimit[name*(gnTCell + gnBFace)], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	    
}

__global__ void gpuLimitInit(RealFlow *limit, const IntType Cell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < Cell){
		for(IntType j=0; j<5; j++) {
			limit[j*Cell + i] = 1.0;
		}
	}
}

void cuLimitInit(RealFlow **limit){
	
	IntType Cell = gnTCell + gnBFace;
	IntType blocksPerGrid = (Cell + threadsPerBlock - 1) / threadsPerBlock;
	gpuLimitInit <<< blocksPerGrid, threadsPerBlock >>> (glimit, Cell);
	//HANDLE_API_ERR(cudaMemcpy(limit[0], glimit, 5*Cell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
}

void cuLimitMemoryTrans(RealFlow **dqdx, RealFlow **dqdy, RealFlow **dqdz, const RealFlow *rho, 
					const RealFlow *u, const RealFlow *v, const RealFlow *w, const RealFlow *p){
						
	HANDLE_API_ERR(cudaMemcpy(gdqdx, dqdx[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(gdqdy, dqdy[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(gdqdz, dqdz[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	HANDLE_API_ERR(cudaMemcpy(gq, rho, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[(gnTCell + gnBFace)], u, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[2*(gnTCell + gnBFace)], v, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[3*(gnTCell + gnBFace)], w, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[4*(gnTCell + gnBFace)], p, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
}


RealFlow **cuGetLimiters_resp(PolyGrid *grid)
{
    IntType nTCell = grid->GetNTCell();
    IntType n      = nTCell + grid->GetNBFace();
    IntType i,j;
    
    // Allocate memories and initialize limiters with value one
    RealFlow **limit = NULL;
    
    const IntType kNVar = 5;
	
	IntType vis_mode;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    
    IntType order;
    grid->GetData(&order, INT, 1, "order"); 
	
	// temp memory transfer: 
	// cuLimitMemoryTrans(dqdx, dqdy, dqdz, rho, u, v, w, p);
	
    switch(order) {
        case LIMITED_VENCAT:
#ifdef MultiStream
			if(vis_mode != INVISCID){
				cuVencatLimiter_MultiStream_Grad_T(grid);
			}
			else{
				cuVencatLimiter_MultiStream(grid, 0);
				cuVencatLimiter_MultiStream(grid, 1);
				cuVencatLimiter_MultiStream(grid, 2);
				cuVencatLimiter_MultiStream(grid, 3);
				cuVencatLimiter_MultiStream(grid, 4); 
			}
#else  
			cuVencatLimiter(grid, 0);
            cuVencatLimiter(grid, 1);
            cuVencatLimiter(grid, 2);
            cuVencatLimiter(grid, 3);
            cuVencatLimiter(grid, 4);
#endif
             break;

        case SECOND_ORDER:
             // Doing nothing, because limiters have been set to one
             break;

        // Other cases?
        default:
             printf("Warning:\n");
             printf("Limiter for order %d has not implemented yet\n", (int)order);
             printf("Error in calling limiters\n");
             printf("Set values of limiter to one everywhere except boundary\n");
             break;
    } 	
	
	IntType  iter_done;// current iterate steps
    grid->GetData(&iter_done, INT, 1 ,"iter_done");
	IntType n_wconverg = 20;
    grid->GetData(&n_wconverg,  INT, 1, "n_wconverg");
    // for convergence log file to output the first step of each physical time step
    IntType iter_step_physic_time = 1;
    grid->GetData(&iter_step_physic_time,  INT, 1, "iter_step_physic_time", 0);
	if(((iter_done + 1)%n_wconverg==0) || (iter_done==1) || iter_step_physic_time==0){
		mfmem::snew_array_2D(limit, 5,n,dmrfl,true);
		for(IntType ii = 0; ii < 5; ii++){
			HANDLE_API_ERR(cudaMemcpy(limit[ii], &glimit[ii*(gnTCell + gnBFace)], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		}
		RealFlow limitaver = 0.;
		IntType  count=0;
		for(i=0; i<5; i++) {
			for(j=0; j<nTCell; j++) {
				limitaver += limit[i][j];
				count++;
			}
		}
#ifdef MPICH
		RealFlow limitaver_total;
		IntType count_total;
		MPI_Allreduce(&limitaver, &limitaver_total, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&count, &count_total, 1, MPIIntType, MPI_SUM, MPI_COMM_WORLD);
		limitaver = limitaver_total;
		count     = count_total;
#endif  
		limitaver /= (RealFlow)count;
		grid->UpdateData(&limitaver, REAL_FLOW, 1, "limitaver");
		mfmem::sdel_array_2D(limit);
		
	}
	
#ifdef MPICH
#ifdef MultiStream
		gMPI = glimit;
		grid->cuRecvSendVarNeighbor_Togeth_q5ForLimit_unfold(kNVar);    
#else
		gMPI = glimit;
		grid->cuRecvSendVarNeighbor_Togeth_q5(kNVar);    
#endif  
#endif

    return limit;
}

__device__ RealFlow gpuVenFun(RealFlow d, RealFlow dq, RealFlow eps)
{
    return((d*d+eps+(dq+dq)*d)*dq/(d*d + (dq+dq+d)*dq +eps));
}

__device__ void atomicMax(double *addr, double val){
                unsigned long long int * addr_as_ull = (unsigned long long int *)(addr);
                unsigned long long int old = *addr_as_ull, assumed;
                do {
                        assumed = old;
                        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(kernelMAXDOUBLE(val, __longlong_as_double(assumed))));
                } while (assumed != old);

}

__device__ void atomicMin(double *addr, double val){
                unsigned long long int * addr_as_ull = (unsigned long long int *)(addr);
                unsigned long long int old = *addr_as_ull, assumed;
                do {
                        assumed = old;
                        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(kernelMINDOUBLE(val, __longlong_as_double(assumed))));
                } while (assumed != old);

}


__device__ double kernelMAXDOUBLE(double a, double b){
        return ((a>b)?a:b);
}

__device__ double kernelMINDOUBLE(double a, double b){
        return ((a<b)?a:b);
}

__device__ double GPUMIN(double a, double b)
{
        return(a>b?b:a);
}
__device__ double GPUMAX(double a, double b)
{
        return(a>b?a:b);
}
__device__ double GPUABS(double a)
{
        return(a<0.0? -a:a);
}

__device__ bool GPUEqualZero(RealFlow x) 
{ 
	return (x > -TINY) && (x < TINY); 
}