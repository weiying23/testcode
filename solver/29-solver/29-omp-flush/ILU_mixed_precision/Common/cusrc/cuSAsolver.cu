#include <stdio.h>
#include <iostream>
#include <fstream>

#include "solver_turb_sa.h"
#include "zone.h"
#include "utility_functions.h"
#include "solver_ns.h"
#include "temporal_discretisation_implicit.h"
#include "io_base_format.h"
#include "io_log.h"
#include "io_field.h"
#include "parallel_base_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"

// head files relying on condition-compiling
#ifdef MPICH
#include <mpi.h>
#endif

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif

#include <cuTurbulenceFlux.cuh>
#include <cuGradientQ_Gauss.cuh>
#include <cuSAsolver.cuh>
#include <cuData.cuh>
#include <cuErrorReturn.cuh>
#include <cuGMRES.cuh>
#include <cuLUSGS.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

__global__ void gpuSAdtlhsmat(RealFlow *lhsmat, RealFlow *dt, const RealFlow *rho, const RealFlow *vol, 
						const IntType *nCPC, const IntType *IndexC2C, 
						const RealFlow turb_cfl_times, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){	
		dt[i] *= turb_cfl_times;
		for(IntType j = 1; j < nCPC[i] + 1; j++){
            lhsmat[i + IndexC2C[i] + j] = 0.;
        }
        lhsmat[i + IndexC2C[i] + 0] = vol[i]*rho[i] / dt[i];
	}
	
}

void cuInitLHSMatScalar(PolyGrid *grid){	
   
    //turbulence time step = ns time step *turb_cfl_times
    RealFlow turb_cfl_times = 2.0;
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuSAdtlhsmat <<< blocksPerGrid, threadsPerBlock >>> (glhsmat, gdt, gq, gvol, gnCPC, gIndexC2C, turb_cfl_times, gnTCell);
	
}

__global__ void gpulimitSA_nu(RealFlow *q, const RealFlow nu_min, const RealFlow nu_max, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){	
		q[i] = GPUMAX3(q[i],nu_min);
        q[i] = GPUMIN3(q[i],nu_max);
	}
	
}

void culimitSA_nu(PolyGrid *grid){

    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell+nBFace;    
    
    RealFlow amu,rho00,nu_max,nu_min,nu_tmp;
    grid->GetData(&amu,   REAL_FLOW, 1, "amu");
    grid->GetData(&rho00, REAL_FLOW, 1, "rho");
    RealFlow max_muet;
    grid->GetData(&max_muet, REAL_FLOW, 1, "max_muet", 0);
    //湍流前1000步加强对nu_max的限制
    IntType level = grid->GetLevel();
    IntType iter_done,n_steps_coarse,step_count;
    step_count = 0;
    if(level == 0){
        grid->GetData(&iter_done, INT, 1 ,"iter_done");
        grid->GetData(&n_steps_coarse, INT, 1 ,"n_steps_coarse");
        step_count = iter_done-n_steps_coarse;
    }
    if(step_count<1000){
        max_muet = 1.0e5;
    }
    if(max_muet<0.0) max_muet=MAX_MUET_SA;
    nu_tmp = amu/rho00;
    nu_min = MIN_SA_NU*nu_tmp;
    nu_max = max_muet*nu_tmp;
    
    RealFlow *q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "sa_nu"); 
    
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpulimitSA_nu <<< blocksPerGrid, threadsPerBlock >>> (gsa_nu, nu_min, nu_max, gnTCell);
	
	//HANDLE_API_ERR(cudaMemcpy(q, gsa_nu, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));

} 

__global__ void gpuUpdateSolutionScalar_TAO(RealFlow *q, RealFlow *dq, const RealFlow dqmax_turb, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){	
		if(fabs(dq[i]/q[i]) > dqmax_turb)
            dq[i] *= dqmax_turb*q[i]/fabs(dq[i]);
        q[i] += dq[i]; 
	}
	
}

void cuUpdateSolutionScalar_TAO(PolyGrid *grid, RealFlow *dq, const char *name){
	
    IntType nBFace = grid->GetNBFace();
    IntType nTCell = grid->GetNTCell();
    IntType n      = nTCell + nBFace;
    
    RealFlow dqmax_turb = 0.25;
    grid->GetData(&dqmax_turb, REAL_FLOW, 1, "dqmax_turb");
    
    RealFlow *q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
    
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuUpdateSolutionScalar_TAO <<< blocksPerGrid, threadsPerBlock >>> (gsa_nu, gDQ, dqmax_turb, gnTCell);
	
	//HANDLE_API_ERR(cudaMemcpy(q, gsa_nu, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	
	if(strcmp(name,"sa_nu") == 0){
        culimitSA_nu(grid);
    }
}

__global__ void gpuSABackwardSweep(RealFlow *dq, const RealFlow *lhsmat, const RealFlow *q, const IntType *luorder, 
								const IntType *layer, const IntType *C2C, const IntType *IndexC2C, const IntType *nCPC, 
								const RealFlow q_min, const IntType start, const IntType end, const IntType DQ_limit){
	IntType ilu = start + blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < end){	
		IntType cell = luorder[ilu];
		RealFlow flux = 0.0;
		for(IntType i = 0; i < nCPC[cell]; i++){
			IntType cell2 = C2C[IndexC2C[cell] + i];
			if(!(layer[cell2] < layer[cell])){
				flux += lhsmat[cell + IndexC2C[cell] + i + 1]*dq[cell2];
			}
		}
		dq[cell] -= flux/lhsmat[cell + IndexC2C[cell] + 0];
	
		//limit dq
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
			dq[cell] = GPUMAX3(dq[cell],q_min-q[cell]);
		}else if(DQ_limit == 4){
			RealFlow alph = q[cell]/(q[cell]+GPUMAX3(0.0,-dq[cell]));
			dq[cell] *= alph;
		}
	}		
															
}

__global__ void gpuSAForwardSweep2(RealFlow *dq, const RealFlow *lhsmat, const RealFlow *q, const IntType *luorder, 
								const IntType *layer, const IntType *C2C, const IntType *IndexC2C, const IntType *nCPC, 
								const RealFlow q_min, const IntType start, const IntType end, const IntType DQ_limit){
	IntType ilu = start + blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < end){	
		
		IntType cell = luorder[ilu];
		for(IntType i = 0; i < nCPC[cell]; i++){
			//IntType cell2 = c2c[cell][i];
			IntType cell2 = C2C[IndexC2C[cell] + i];
			if(!(layer[cell2]>layer[cell])){
				dq[cell] -= lhsmat[cell + IndexC2C[cell] + i + 1]*dq[cell2];
			}
		}
		dq[cell] /= lhsmat[cell + IndexC2C[cell] + 0];
	
		//limit dq
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
			dq[cell] = GPUMAX3(dq[cell],q_min-q[cell]);
		}else if(DQ_limit == 4){
			RealFlow alph = q[cell]/(q[cell]+GPUMAX3(0.0,-dq[cell]));
			dq[cell] *= alph;
		}
	}		
															
}

__global__ void gpuSAForwardSweep(RealFlow *dq, const RealFlow *lhsmat, const RealFlow *q, const IntType *luorder, 
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
        
        //limit dq
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
            dq[cell] = GPUMAX3(dq[cell],q_min-q[cell]);
        }else if(DQ_limit == 4){
            RealFlow alph = q[cell]/(q[cell]+GPUMAX3(0.0,-dq[cell]));
            dq[cell] *= alph;
        }
	}		
															
}

void cuSolveScalarLUSGS(PolyGrid *grid, RealFlow **lhsmat, RealFlow *dq, IntType nTCell, const char *name){
							
    IntType n = nTCell+grid->GetNBFace();
    RealFlow rhoP,amu,ainf,q_min;
    grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
    grid->GetData(&amu,   REAL_FLOW, 1, "amu");
    grid->GetData(&ainf,  REAL_FLOW, 1, "ainf");

    IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer");
    
    if(strcmp(name,"sa_nu") == 0){
        q_min = MIN_SA_NU;
        q_min *= (amu/rhoP);
    }
    
    IntType DQ_limit = 1;
    grid->GetData(&DQ_limit, INT, 1, "DQ_limit");
    
    RealFlow *q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
	// transfer sa_nu[n] and lhsmat into GPU:
	
    //the Forward Sweep: first step
	IntType blocksPerGrid = (1 + threadsPerBlock - 1) / threadsPerBlock;
	gpuSAForwardSweep <<< blocksPerGrid, threadsPerBlock >>> (gDQ, glhsmat, gsa_nu, gluorder, glayer, gC2C, gIndexC2C, gnCPC, 
														q_min, DQ_limit);

    for(IntType laynum = 0; laynum < cellsPerlayer[0]; laynum++){
        IntType start = cellsPerlayer[laynum+1];
        IntType end   = cellsPerlayer[laynum+2];
        if(laynum == 0) {start++;}
		blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;
		gpuSAForwardSweep2 <<< blocksPerGrid, threadsPerBlock >>> (gDQ, glhsmat, gsa_nu, gluorder, glayer, gC2C, 
																gIndexC2C, gnCPC, q_min, start, end, DQ_limit);		
    }
 
#ifdef MPICH
	gMPI = gDQ;
	grid->cuRecvSendVarNeighbor_Togeth_SA(1);
#endif
	
    for(IntType laynum = cellsPerlayer[0] - 1; laynum >= 0; laynum--){
        IntType end = cellsPerlayer[laynum+2];
        IntType start   = cellsPerlayer[laynum+1];
		blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;
		gpuSABackwardSweep <<< blocksPerGrid, threadsPerBlock >>> (gDQ, glhsmat, gsa_nu, gluorder, glayer, gC2C, 
																gIndexC2C, gnCPC, q_min, start, end, DQ_limit);
        
    }

} 

__global__ void gpures2dq(RealFlow *DQ, RealFlow *res, IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		DQ[i] = res[i];
	}
	
}

void cures2dq(){ 
  	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpures2dq <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gres, gnTCell);

}

__global__ void gpuSABackwardSweepMany(RealFlow *dq, RealFlow *dqo, RealFlow *norm, const RealFlow *res, 
								const RealFlow *lhsmat, const RealFlow *q, const IntType *luorder, 
								const IntType *layer, const IntType *C2C, const IntType *IndexC2C, const IntType *nCPC, 
								const RealFlow q_min, const IntType start, const IntType end, const IntType DQ_limit){
	IntType ilu = start + blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < end){	
		IntType cell = luorder[ilu];
		RealFlow DQO, tmp;
		
		DQO = dq[cell];
		dq[cell]  = res[cell]-dqo[cell];
		dqo[cell] = 0.0;
		
		for(IntType i = 0; i < nCPC[cell]; i++){
			IntType cell2 = C2C[IndexC2C[cell] + i];
			if(!(layer[cell2] < layer[cell])){
				RealFlow flux_tmp = lhsmat[cell + IndexC2C[cell] + i + 1]*dq[cell2];
				dq[cell] -= flux_tmp;
				dqo[cell] += flux_tmp;
			}
		}
		
		dq[cell] /= lhsmat[cell + IndexC2C[cell] + 0];
		tmp = dq[cell] - DQO;
		norm[cell] += tmp*tmp;
	
		//limit dq
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
			dq[cell] = GPUMAX3(dq[cell],q_min-q[cell]);
		}else if(DQ_limit == 4){
			RealFlow alph = q[cell]/(q[cell]+GPUMAX3(0.0,-dq[cell]));
			dq[cell] *= alph;
		}
	}		
															
}

__global__ void gpuSAForwardSweep2Many(RealFlow *dq, RealFlow *dqo, RealFlow *norm, const RealFlow *res, 
								const RealFlow *lhsmat, const RealFlow *q, const IntType *luorder, 
								const IntType *layer, const IntType *C2C, const IntType *IndexC2C, const IntType *nCPC, 
								const RealFlow q_min, const IntType start, const IntType end, const IntType DQ_limit){
	IntType ilu = start + blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < end){		
		IntType cell = luorder[ilu];
		RealFlow DQO, tmp;
		
		DQO = dq[cell];
		dq[cell]  = res[cell] - dqo[cell];
        dqo[cell] = 0.0;
		
		for(IntType i = 0; i < nCPC[cell]; i++){
			IntType cell2 = C2C[IndexC2C[cell] + i];
			if(!(layer[cell2]>layer[cell])){
				RealFlow flux_tmp = lhsmat[cell + IndexC2C[cell] + i + 1]*dq[cell2];				
				dq[cell] -= flux_tmp;
				dqo[cell] += flux_tmp;
			}
		}
		
		dq[cell] /= lhsmat[cell + IndexC2C[cell] + 0];
		tmp      = dq[cell] - DQO;
        norm[cell]   = tmp*tmp;
	
		//limit dq
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
			dq[cell] = GPUMAX3(dq[cell],q_min-q[cell]);
		}else if(DQ_limit == 4){
			RealFlow alph = q[cell]/(q[cell]+GPUMAX3(0.0,-dq[cell]));
			dq[cell] *= alph;
		}
	}		
															
}

__global__ void gpuSAForwardSweepMany(RealFlow *dq, RealFlow *dqo, RealFlow *norm, const RealFlow *res, 
								const RealFlow *lhsmat, const RealFlow *q, const IntType *luorder, 
								const IntType *layer, const IntType *C2C, const IntType *IndexC2C, const IntType *nCPC, 
								const RealFlow q_min, const IntType DQ_limit){
	IntType ilu = blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < 1){														
		IntType cell = luorder[ilu];
		RealFlow DQO, tmp;
		
		DQO = dq[cell];
		dq[cell]  = res[cell] - dqo[cell];
        dqo[cell] = 0.0;

        /* for(IntType i=0; i < nCPC[cell]; i++){
            //cell2 = c2c[cell][i];
			IntType cell2 = C2C[IndexC2C[cell] + i];
            if(!(layer[cell2]>layer[cell])){
				dq[cell] -= lhsmat[cell + IndexC2C[cell] + i + 1]*dq[cell2];
			}
        } */
        //dq[cell] /= lhsmat[cell][0];
		dq[cell] /= lhsmat[cell + IndexC2C[cell] + 0];
		tmp      = dq[cell] - DQO;
        norm[cell]    = tmp*tmp;
        
        //limit dq
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
            dq[cell] = GPUMAX3(dq[cell],q_min-q[cell]);
        }else if(DQ_limit == 4){
            RealFlow alph = q[cell]/(q[cell]+GPUMAX3(0.0,-dq[cell]));
            dq[cell] *= alph;
        }
	}		
															
}


void cuSolveScalarLUSGS(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, 
                      RealFlow *dq, IntType *nCPC, IntType **c2c, IntType nTCell, const char *name, 
                      IntType Nsweep, RealFlow epsilon){
						  
    IntType sweep;
    IntType n = nTCell+grid->GetNBFace();
    
    RealFlow rhoP,amu,ainf,q_min;
    grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
    grid->GetData(&amu,   REAL_FLOW, 1, "amu");
    grid->GetData(&ainf,  REAL_FLOW, 1, "ainf");
    
    if(strcmp(name,"sa_nu") == 0){
        q_min = MIN_SA_NU;
        q_min *= (amu/rhoP);
    }
    
    IntType DQ_limit = 1;
    grid->GetData(&DQ_limit, INT, 1, "DQ_limit");
    
    RealFlow *q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
    
    RealFlow norm0, norm, dmax = 1.0, tmp;

	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDQInit <<< blocksPerGrid, threadsPerBlock >>> (gdqo, gnTCell); 
    
	IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer");
	
	RealFlow *normtmp;
	mfmem::snew_array_1D(normtmp, nTCell,dmrfl);
	
    for(sweep=0; sweep<Nsweep; sweep++){
        norm = 0.0;
		IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
		gpuDQInit <<< blocksPerGrid, threadsPerBlock >>> (gSAsumv2, gnTCell); 
        //the Forward Sweep
		blocksPerGrid = (1 + threadsPerBlock - 1) / threadsPerBlock;
		gpuSAForwardSweepMany <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gdqo, gSAsumv2, gres, glhsmat, gq, gluorder, 
							glayer, gC2C, gIndexC2C, gnCPC, q_min, DQ_limit);
		
        for(IntType laynum=0; laynum<cellsPerlayer[0]; laynum++ ){
			IntType start = cellsPerlayer[laynum+1];
			IntType end   = cellsPerlayer[laynum+2];
			if(laynum == 0) {start++;}
			
			blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;	
			gpuSAForwardSweep2Many <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gdqo, gSAsumv2, gres, glhsmat, gq, gluorder, 
							glayer, gC2C, gIndexC2C, gnCPC, q_min, start, end, DQ_limit); 
																		
			
		}
		
#ifdef MPICH
		gMPI = gDQ;
		grid->cuRecvSendVarNeighbor_Togeth_SA(1); 
		//grid->CommInterfaceDataMPI(dq);
#endif		
	
		//the Backward Sweep
		for(IntType laynum = cellsPerlayer[0] - 1; laynum >= 0; laynum--){
			IntType end = cellsPerlayer[laynum+2];
			IntType start   = cellsPerlayer[laynum+1];
			blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;
			gpuSABackwardSweepMany <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gdqo, gSAsumv2, gres, glhsmat, gq, gluorder, 
															glayer, gC2C, gIndexC2C, gnCPC, q_min, start, end, DQ_limit);
			
		}
		
		blocksPerGrid = gSAnodata2;
		Reducekernel6 <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gSAsumv2, gSAodata2, gSAnsum2);
		HANDLE_API_ERR(cudaMemcpy(normtmp, gSAodata2, blocksPerGrid*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		
		for(IntType i=0; i<blocksPerGrid; i++) norm += normtmp[i];
		
#ifdef MPICH
        MPI_Allreduce(&norm, &tmp, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
        norm = tmp;
#endif
    
        if(sweep == 0) norm0 = norm;
        else dmax = sqrt(norm/norm0);
        if(dmax < epsilon){
            sweep++;
            break;
        }
    }

/* #ifdef DEBUG
#ifdef MPICH
    if(myZone == 1) printf("Turb resi reduced by %.5e with %d sweeps\n", dmax, (int)sweep);
#else   
    printf("Turb resi reduced by %.5e with %d sweeps\n", dmax, (int)sweep);
#endif
#endif */
	//HANDLE_API_ERR(cudaMemcpy(gDQ, dq, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
    
    /* mfmem::sdel_array_1D(dqo); */
	
	mfmem::sdel_array_1D(normtmp);

}

__global__ void deResLhsmat(RealFlow* dq, RealFlow* res, RealFlow* lhsmat, const IntType *IndexC2C, IntType nTCell){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		dq[i] = res[i] / lhsmat[i + IndexC2C[i] + 0]; //lhsmat[i];
	}
}

__global__ void gpuScalarDPLUR(RealFlow *dq, RealFlow *dqo, 
	const RealFlow *lhsmat, const RealFlow *q, const IntType *luorder, 
	const IntType *layer, const IntType *C2C, const IntType *IndexC2C, const IntType *nCPC, 
	const RealFlow q_min, const IntType DQ_limit, const IntType nTCell){
	IntType ilu =  blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < nTCell){
		IntType cell = ilu; //luorder[ilu];

		for(IntType i = 0; i < nCPC[cell]; i++){
			IntType cell2 = C2C[IndexC2C[cell] + i];
			RealFlow flux_tmp = lhsmat[cell + IndexC2C[cell] + i + 1]*dqo[cell2];				
			dq[cell] -= flux_tmp;
			//dqo[cell] += flux_tmp;
		}
		
		dq[cell] /= lhsmat[cell + IndexC2C[cell] + 0];
	
		//limit dq
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
			dq[cell] = GPUMAX3(dq[cell],q_min-q[cell]);
		}else if(DQ_limit == 4){
			RealFlow alph = q[cell]/(q[cell]+GPUMAX3(0.0,-dq[cell]));
			dq[cell] *= alph;
		}
	}
}
void cuSolveScalarDPLUR(PolyGrid *grid, const char *name, IntType level){
	IntType nTCell = grid->GetNTCell();
    IntType n = nTCell+grid->GetNBFace();
    
    RealFlow rhoP, amu, ainf, q_min;
    grid->GetData(&rhoP, REAL_FLOW, 1, "rho");
    grid->GetData(&amu, REAL_FLOW, 1, "amu");
    grid->GetData(&ainf, REAL_FLOW, 1, "ainf");

    if (strcmp(name, "sa_nu") == 0) {
        q_min = MIN_SA_NU;
        q_min *= (amu / rhoP);
    }

    IntType DQ_limit = 1;
    grid->GetData(&DQ_limit, INT, 1, "DQ_limit");
    IntType sweeps = 1;
	grid->GetData(&sweeps, INT, 1, "sweeps");

	IntType blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDQInit <<< blocksPerGrid, threadsPerBlock >>> (gdqo, n); 
    
	IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer");

	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	deResLhsmat <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gres, glhsmat, gIndexC2C, gnTCell);

	for (IntType idx_sweep = 0; idx_sweep < sweeps; ++idx_sweep) {
		blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
		gpures2dq <<< blocksPerGrid, threadsPerBlock >>> (gdqo, gDQ, gnTCell);
		gpures2dq <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gres, gnTCell);

#ifdef MPICH
		gMPI = gdqo;
		grid->cuRecvSendVarNeighbor_Togeth_SA(1);
        //grid->CommInterfaceDataMPI(DQ0);
#endif

		blocksPerGrid = ( gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
		gpuScalarDPLUR <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gdqo, glhsmat, gsa_nu, gluorder, 
				glayer, gC2C, gIndexC2C, gnCPC, q_min, DQ_limit, gnTCell); 

	}
}

void cuTimeIntegrationScalar(PolyGrid *grid, const char *name){
	
    IntType  nTCell   = grid->GetNTCell();   	
    // Implicit
    IntType sweeps = 1;
    grid->GetData(&sweeps, INT, 1, "sweeps");
    RealFlow epsilon = 0.1;
    grid->GetData(&epsilon, REAL_FLOW, 1, "epsilon");
    if(epsilon < TINY) epsilon = 1.0e-1;

	cuDQInit(1);	
	IntType tScheme;
	grid->GetData(&tScheme, INT, 1, "tScheme");
	if (tScheme == DPLUR) {
		cuSolveScalarDPLUR(grid, name, 0);
	}
	else if (tScheme == LU_SGS){
		if(sweeps == -1){
		
		}else if(sweeps == 1){
			cures2dq();			
			cuSolveScalarLUSGS(grid, NULL, NULL, nTCell, name);
		}else{
			cuSolveScalarLUSGS(grid, NULL, NULL, NULL, NULL, NULL, nTCell, name, sweeps, epsilon); 
		}   
	}

	// sweep = 1:  
    cuUpdateSolutionScalar_TAO(grid, NULL, name);

}

__global__ void gpuComputeTurbInf_SA(RealFlow *gradnue2, const RealFlow *dnutdx, const RealFlow *dnutdy, 
									const RealFlow *dnutdz, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){		
        gradnue2[i]  = dnutdx[i]*dnutdx[i] + dnutdy[i]*dnutdy[i] + dnutdz[i]*dnutdz[i]; ;
	}
	
}

void cuComputeTurbInf_SA(RealFlow *gradnue2){
	
    IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuComputeTurbInf_SA <<< blocksPerGrid, threadsPerBlock >>> (ggradnue2, gdnutdx, gdnutdy, gdnutdz, gnTCell);
	
	HANDLE_API_ERR(cudaMemcpy(gradnue2, ggradnue2, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
}

__global__ void gpuGhostVariablesScalar_SA(RealFlow *sa_nu, RealFlow *rho, RealFlow *u, RealFlow *v, RealFlow *w, RealFlow *p, 
							const RealGeom *xfn, const RealGeom *yfn, const RealGeom *zfn, const RealGeom *vgn, 
							const IntType *f2c, const IntType *type_bcr, const RealFlow sa_nu00,
							const RealFlow rhoP, const RealFlow uP, const RealFlow vP, const RealFlow wP, const RealFlow pP,
							const RealFlow gam, const IntType steady, const RealFlow p_bar, const IntType nBFace){																					
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType type, c1, c2;
	RealFlow vnf, vn, cf, cc;
	RealFlow sa_nu_c2;
	RealFlow riemp, riemm;
	RealFlow vnb;
	if(i < nBFace){
		type  = type_bcr[i];
        c1    = f2c[2*i];
        c2    = f2c[2*i + 1];
        // wmark = 0;
		if(type != INTERFACE){
			if(type == WALL){
				sa_nu_c2 = -sa_nu[c1];				
			} 
			else if (type == SYMM){
				sa_nu_c2 = sa_nu[c1];								
			}
			else{	// FAR_FIELD:
				vnf   = xfn[i]*uP + yfn[i]*vP + zfn[i]*wP;
                vn    = xfn[i]*u[c1] + yfn[i]*v[c1] + zfn[i]*w[c1];
                cf    = sqrt(gam*(pP + p_bar)/rhoP);
                cc    = sqrt(gam*(p[c1] + p_bar)/rho[c1]);
                riemp = vn+2.*cc/(gam-1.);
                riemm = vnf-2.*cf/(gam-1.);
                vnb   = 0.5*(riemp+riemm);
                
                if(!steady) vnb -= vgn[i];

                if(vnb>0)  sa_nu_c2 = sa_nu[c1];
                else       sa_nu_c2 = sa_nu00; 
			}
			atomicExchSM35SA(sa_nu + c2, sa_nu_c2);
		}
	}
	
}

void cuGhostVariablesScalar_SA(PolyGrid *grid)
{  
  
    RealFlow sa_nu00;
    RealFlow rhoP,uP,vP,wP,pP;
    grid->GetData(&pP,    REAL_FLOW, 1, "p");
    grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
    grid->GetData(&uP,    REAL_FLOW, 1, "u");
    grid->GetData(&vP,    REAL_FLOW, 1, "v");
    grid->GetData(&wP,    REAL_FLOW, 1, "w");
    
    grid->GetData(&sa_nu00, REAL_FLOW, 1, "sa_nu00");
    
    IntType steady;
    grid->GetData(&steady, INT, 1, "steady");
    RealGeom *vgn = grid->GetFaceNormalVelocity();
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuGhostVariablesScalar_SA <<< blocksPerGrid, threadsPerBlock >>> (gsa_nu, gq, &gq[gnTCell + gnBFace], &gq[2*(gnTCell + gnBFace)], 
														&gq[3*(gnTCell + gnBFace)], &gq[4*(gnTCell + gnBFace)],
														gxfn, gyfn, gzfn, gvgn, gf2c, gtype_bcr, sa_nu00, 														
														rhoP, uP, vP, wP, pP, ggam, gsteady, gp_bar, gnBFace);														
	
}

void cuGhostVariablesScalar(PolyGrid *grid, const char *name)
{
    if(strcmp(name,"sa_nu") == 0)
        cuGhostVariablesScalar_SA(grid);
    
}

void cuUpdateResidualScalar(PolyGrid *grid, const char *name){
	
#ifdef MPICH
	#if defined MultiStream
		IntType nvar = 1;
		gMPI = gsa_nu;
		grid->cuRecvSendVarNeighbor_Togeth_SAForInterfaceData_unfold(nvar);
	#else
		IntType nvar = 1;
		gMPI = gsa_nu;
		grid->cuRecvSendVarNeighbor_Togeth_SA(nvar);
	#endif
#else
	IntType  n     = grid->GetNTCell() + grid->GetNBFace();
	RealFlow *turb = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);	
#endif
	
	cuGhostVariablesScalar(grid,name);
	
	// res initial zero:
	/* IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDQInit <<< blocksPerGrid, threadsPerBlock >>> (gres, gnTCell); */
	
    ComputeTurbInf(grid, name); // CUDA executes this function in solver_turb_sa.cpp: cuComputeTurbInf_SA(gradnue2)
								// for SA Gradient Computation
								
    cuInviscidFluxScalar(grid, name);
#if !(defined MultiStream)
	cuViscousFluxScalar(grid, name);
	cuAddSourceScalar(grid, name);
#endif
    
    IntType steady = 1;
    grid->GetData(&steady, INT, 1, "steady");
    if(!steady) AddSourceScalarUnst(grid, name);
   
}

void cuScalarRelaxation(PolyGrid *grid, const char *name, IntType steps){
	
    IntType n;

    for(n=0; n<steps; n++){
        grid->UpdateData(&n, INT, 1, "turb_step");
        // zero lhs matrix
#if !(defined MultiStream)
        cuInitLHSMatScalar(grid);
#endif
        // load variable "res" of grid with the array rhs
		// useless function: PutResInGrid 
        //PutResInGrid(grid, rhs, nTCell, "res");

        cuUpdateResidualScalar(grid, name);
    
        cuTimeIntegrationScalar(grid,name);
    }
    FreeLHSMatScalar(grid);
}

void cuSolveScalarOnGrid(PolyGrid *grid, const char *name){
    
    IntType turb_substeps=1;
    grid->GetData(&turb_substeps, INT, 1, "turb_substeps", 0);
    cuScalarRelaxation(grid, name, turb_substeps);     
}

__global__ void gpuComputeTurbGeneration_SA(RealFlow *omaga, const RealFlow *dqdx, const RealFlow *dqdy, 
										const RealFlow *dqdz, const IntType n, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		RealFlow dudy,dudz,dvdx,dvdz,dwdx,dwdy;
		
		dvdx = dqdx[2*n + i];
        dwdx = dqdx[3*n + i];
        dudy = dqdy[1*n + i];
        dwdy = dqdy[3*n + i];
        dudz = dqdz[1*n + i];
        dvdz = dqdz[2*n + i];
        
        omaga[i]  = sqrt((dwdy-dvdz)*(dwdy-dvdz)+(dudz-dwdx)*(dudz-dwdx)+(dvdx-dudy)*(dvdx-dudy));
	}
	
}

void cuComputeTurbGeneration_SA(PolyGrid *grid){
	
    /* IntType  nTCell = grid->GetNTCell();
    
    RealFlow *omaga = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "omaga");
    if(omaga == 0){
        mfmem::snew_array_1D(omaga, nTCell,dmrfl);
        grid->UpdateDataPtr(omaga, REAL_FLOW, nTCell, "omaga");
    } */
    
    //RealFlow *dqdx[3], *dqdy[3], *dqdz[3];
    //GetVelocityGradient(grid, dqdx, dqdy, dqdz);
    
	IntType NumCell = gnTCell + gnBFace;
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuComputeTurbGeneration_SA <<< blocksPerGrid, threadsPerBlock >>> (gomaga, gdqdx, gdqdy, gdqdz, NumCell, gnTCell);
	
	//HANDLE_API_ERR(cudaMemcpy(omaga, gomaga, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpuZeroGridResiduals(RealFlow *res, IntType n){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < n){
		res[i] = 0.0;
	}
	
}

void cuZeroGridResiduals(PolyGrid *grid, const char *name, IntType nVar){
	
	IntType blocksPerGrid = (nVar*gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuZeroGridResiduals <<< blocksPerGrid, threadsPerBlock >>> (gres, nVar*gnTCell);
	
	//HANDLE_API_ERR(cudaMemcpy(res, gres, nT*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpuComputeTurbViscosity_SA(RealFlow *vis_t, const RealFlow *rho, const RealFlow *sa_nu, 
										const RealFlow *vis_l, const RealFlow max_muet, 
										const RealFlow amu, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		RealFlow fv1, xkai, xkaip3, nue;
        nue = vis_l[i]/rho[i];
        xkai = sa_nu[i]/nue;
        xkaip3 = xkai*xkai*xkai;
        fv1 = xkaip3/(xkaip3 + CV1P3);
        vis_t[i] = rho[i]*sa_nu[i]*fv1;
        
        vis_t[i]  = GPUMAX3(vis_t[i],MIN_MUET_SA*amu);
        vis_t[i]  = GPUMIN3(vis_t[i],max_muet*amu);
	}
	
}

__global__ void gpuSetGhostvis_t(RealFlow *vis_t, RealFlow *rho, RealFlow *u, RealFlow *v, RealFlow *w, RealFlow *p, 
							const RealGeom *xfn, const RealGeom *yfn, const RealGeom *zfn, const RealGeom *vgn, 
							const IntType *f2c, const IntType *type_bcr, const RealFlow vis_t00,
							const RealFlow rhoP, const RealFlow uP, const RealFlow vP, const RealFlow wP, const RealFlow pP,
							const RealFlow gam, const IntType steady, const RealFlow p_bar, const IntType nBFace){																					
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType type, c1, c2;
	RealFlow vnf, vn, cf, cc;
	RealFlow vis_t_c2;
	RealFlow riemp, riemm;
	RealFlow vnb;
	if(i < nBFace){
		type  = type_bcr[i];
        c1    = f2c[2*i];
        c2    = f2c[2*i + 1];
        // wmark = 0;
		if(type != INTERFACE){
			if(type == WALL){
				vis_t_c2 = -vis_t[c1];			
			} 
			else if (type == SYMM){
				vis_t_c2 = vis_t[c1];							
			}
			else{	// FAR_FIELD:
				vnf   = xfn[i]*uP + yfn[i]*vP + zfn[i]*wP;
				vn    = xfn[i]*u[c1] + yfn[i]*v[c1] + zfn[i]*w[c1];
				cf    = sqrt(gam*(pP + p_bar)/rhoP);
				cc    = sqrt(gam*(p[c1] + p_bar)/rho[c1]);
				riemp = vn+2.*cc/(gam-1.);
				riemm = vnf-2.*cf/(gam-1.);
				vnb   = 0.5*(riemp+riemm);
				 
				if(!steady) vnb -= vgn[i];
					
				if(vnb>0)  vis_t_c2 = vis_t[c1];
				else       vis_t_c2 = vis_t00;    
			}
			atomicExchSM35SA(vis_t + c2, vis_t_c2);
		}
	}
	
}

void cuSetGhostvis_t(PolyGrid *grid, const char *name){
    
    RealFlow vis_t00; 
    RealFlow rhoP,uP,vP,wP,pP; 

    grid->GetData(&pP,    REAL_FLOW, 1, "p");
    grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
    grid->GetData(&uP,    REAL_FLOW, 1, "u");
    grid->GetData(&vP,    REAL_FLOW, 1, "v");
    grid->GetData(&wP,    REAL_FLOW, 1, "w");
    grid->GetData(&vis_t00, REAL_FLOW, 1, "vis_t00");
    
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuSetGhostvis_t <<< blocksPerGrid, threadsPerBlock >>> (gvis_t, gq, &gq[gnTCell + gnBFace], &gq[2*(gnTCell + gnBFace)], 
														&gq[3*(gnTCell + gnBFace)], &gq[4*(gnTCell + gnBFace)],
														gxfn, gyfn, gzfn, gvgn, gf2c, gtype_bcr, vis_t00, 														
														rhoP, uP, vP, wP, pP, ggam, gsteady, gp_bar, gnBFace);
}

void cuComputeTurbViscosity_SA(PolyGrid *grid){
	
    RealFlow amu;
    grid->GetData(&amu,  REAL_FLOW, 1, "amu");
    RealFlow max_muet;
    grid->GetData(&max_muet, REAL_FLOW, 1, "max_muet",0);
    if(max_muet<0.0) max_muet=MAX_MUET_SA;
 
    IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuComputeTurbViscosity_SA <<< blocksPerGrid, threadsPerBlock >>> (gvis_t, gq, gsa_nu, gvis_l, max_muet, amu, gnTCell);
	  
#ifdef MPICH
	gMPI = gvis_t;
	grid->cuRecvSendVarNeighbor_Togeth_SA(1);
#endif  
	
	cuSetGhostvis_t(grid,"SA");
}

void cuSAsolve(PolyGrid *grid){
#if !(defined MultiStream)	
	cuComputeTurbGeneration_SA(grid);
    cuZeroGridResiduals(grid, "res", 1); 
#endif
    cuSolveScalarOnGrid(grid, "sa_nu"); 
    cuComputeTurbViscosity_SA(grid);

}	

__device__ double GPUMIN3(double a, double b){
        return(a>b?b:a);
}
__device__ double GPUMAX3(double a, double b){
        return(a>b?a:b);
}	

__device__ double atomicExchSM35SA(double* address, double val){
	
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
        } while (assumed != old);
        return __longlong_as_double(old);
}