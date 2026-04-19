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
#include <cuGMRES.cuh>
//#include <cuLimit.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//dingxin
#ifdef TIMECOST
extern double* timecost;
extern double  time_flux, time_invis, time_roe, time_vis, time_calvis;
extern double  time_limiter;
extern double  time_gradient;
extern double  time_lusgs, time_RK;
#endif
//TIMECOST

using namespace mflow;

using namespace gpuData;

__global__ void gpuUpdateFlowField3D_CFL3d(RealFlow *rho, RealFlow *u, RealFlow *v, RealFlow *w,RealFlow *p,  
										const RealFlow *DQ, const RealFlow gam1, const RealFlow rho00, 
										const RealFlow p00, const RealFlow rho_min, const RealFlow rho_max, 
										const RealFlow p_min, const RealFlow p_max, const RealFlow alpq, const RealFlow phiq, 
										const RealFlow betq, const IntType NumCell, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	RealFlow rhot, rhotr, ru, rv, rw, re;
    RealFlow rho_del, p_del, rho_rat, p_rat, ptmp;
	if(i < nTCell){
		rhot = rho[i];
		ru   = rhot*u[i];
		rv   = rhot*v[i];
		rw   = rhot*w[i];
		re   = p[i]/gam1 + 0.5*rhot*(u[i]*u[i] + v[i]*v[i] + w[i]*w[i]);
		  
		rhot += DQ[0*NumCell + i];
		ru   += DQ[1*NumCell + i];
		rv   += DQ[2*NumCell + i];
		rw   += DQ[3*NumCell + i];
		re   += DQ[4*NumCell + i];

		rhotr = 1./(rhot + TINY);
		u[i] = ru * rhotr;
		v[i] = rv * rhotr;
		w[i] = rw * rhotr;

		rho_del = DQ[0*NumCell + i];
		rho_rat = rho_del/rho[i];
		if(rho_rat < alpq) {
			rho_del /= betq + fabs(rho_rat)*phiq;
		}
		rho[i]+= rho_del;
		rho[i] = GPUMAX2(rho[i], rho_min);
		rho[i] = GPUMIN2(rho[i], rho_max);

		ptmp    = gam1*(re - 0.5*(u[i]*u[i] + v[i]*v[i] + w[i]*w[i])*rho[i]);
		p_del   = ptmp - p[i];
		p_rat   = p_del/(p[i] + p00);
		if(p_rat < alpq) {
			p_del /= betq + fabs(p_rat)*phiq;
		}
		p[i]+= p_del;
		p[i] = GPUMAX2(p[i], p_min);
		p[i] = GPUMIN2(p[i], p_max);
	}
	
}

void cuUpdateFlowField3D_CFL3d(PolyGrid *grid, RealFlow *DQ[5]){
	
	IntType nTCell  = grid->GetNTCell();
    IntType nBFace  = grid->GetNBFace();
    IntType n       = nTCell + nBFace;
    RealFlow *rho   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    RealFlow *u     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    RealFlow *v     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    RealFlow *w     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    RealFlow *p     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    
    RealFlow alpq, phiq, betq;
    RealFlow gam, gam1, rho00, p00;
    grid->GetData(&gam, REAL_FLOW, 1, "gam");
    grid->GetData(&rho00, REAL_FLOW, 1, "rho");
    grid->GetData(&p00, REAL_FLOW, 1, "p_bar");
    gam1 = gam - 1.;
  
    RealFlow rho_min, rho_max, p_min, p_max;
    grid->GetData(&rho_min, REAL_FLOW, 1, "rho_min");
    grid->GetData(&rho_max, REAL_FLOW, 1, "rho_max");
    grid->GetData(&p_min,   REAL_FLOW, 1, "p_min");
    grid->GetData(&p_max,   REAL_FLOW, 1, "p_max");
    
    //the const value is came from CFL3D
    alpq = -0.2;
    phiq = 1./0.5;
    betq = 1.0 + alpq*phiq;
    
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuUpdateFlowField3D_CFL3d <<< blocksPerGrid, threadsPerBlock >>> (gq, &gq[1*n], &gq[2*n], &gq[3*n], &gq[4*n], 
																	gDQ, gam1, rho00, p00, rho_min, rho_max, 
																	p_min, p_max, alpq, phiq, betq, n, gnTCell);
	/*
	HANDLE_API_ERR(cudaMemcpy(rho, gq, nTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));			
	HANDLE_API_ERR(cudaMemcpy(u, &gq[1*n], nTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	HANDLE_API_ERR(cudaMemcpy(v, &gq[2*n], nTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(w, &gq[3*n], nTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(p, &gq[4*n], nTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	*/
	// void NSSolver::ProcessAfterNewQuantity would refresh the ghost cells' values, then transfer the new value of 
	// the ghost cells into GPU
}

__global__ void gpures2DQ(RealFlow *DQ, RealFlow *res, IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		DQ[i] = res[i];
	}
	
}

void cures2DQ(RealFlow *DQ[5], RealFlow *res, IntType name){ 
  	
	
	IntType Cell = gnTCell + gnBFace;
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpures2DQ <<< blocksPerGrid, threadsPerBlock >>> (&gDQ[name*Cell], &gres[name*gnTCell], gnTCell);
	
}

__global__ void gpuDQInit(RealFlow *DQ, IntType nT5){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nT5){
		DQ[i] = 0.0;
	}
	
}

void cuDQInit(IntType nvar){ 
  	
	IntType nT5 = nvar*(gnTCell + gnBFace);
	IntType blocksPerGrid = (nT5 + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDQInit <<< blocksPerGrid, threadsPerBlock >>> (gDQ, nT5);
	//HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, nT5*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpuDiag_v2Diag(RealFlow *Diag, RealFlow *Diag_v, const RealFlow *rho, 
							const RealFlow *vis_l, const RealFlow *vis_t, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		Diag[i] += (vis_l[i] + vis_t[i])*Diag_v[i]/rho[i];
	}
	
}

void cuDiag_v2Diag(){
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDiag_v2Diag <<< blocksPerGrid, threadsPerBlock >>> (gDiag, gDiag_v, gq, gvis_l, gvis_t, gnTCell);
	
}

__global__ void gpuReductionDiag_v(RealFlow *tmpvar, const RealFlow *norm_dist_c2c, const RealFlow *area, 
								const IntType nTFace){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTFace){
		tmpvar[2*i] = area[i]/(norm_dist_c2c[i] + TINY);
		tmpvar[2*i + 1] = tmpvar[2*i];
	}
	
}

__global__ void gpuReductionDiag_vInit(RealFlow *Diag_v, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		Diag_v[i] = 0.0;
	}
	
}

void cuDiag_v(){
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuReductionDiag_vInit <<< blocksPerGrid, threadsPerBlock >>> (gDiag_v, gnTCell);
															
	blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuReductionDiag_v <<< blocksPerGrid, threadsPerBlock >>> (gtmpvar, gnorm_dist_c2c, garea, gnTFace);														
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
#if (defined ShareMemory)
	gpuReductionDiag3ShareMemory <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gDiag_v, gtmpvar, gf2c, gC2F, gIndexC2F, 
															gnFPC, gnTCell, gnBFace, gnTFace);
#else
	gpuReductionDiag3 <<< blocksPerGrid, threadsPerBlock >>> (gDiag_v, gtmpvar, gf2c, gC2F, gIndexC2F, 
															gnFPC, gnTCell, gnBFace, gnTFace);
#endif
	//HANDLE_API_ERR(cudaMemcpy(Diag_v, gDiag_v, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpuReductionDiag3(RealFlow *Diag, const RealFlow *tmpvar, const IntType *f2c, const IntType* C2F,
								const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell, const IntType nBFace, 
								const IntType nTFace){
									
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType j, c1, c2, face, count;

	if(i < nTCell){		
		for (j = 0; j < nFPC[i]; j++) {
            face = C2F[IndexC2F[i] + j];
            count = 2 * face;
			c1 = f2c[count];
            c2 = f2c[count + 1];
			if (i == c1) {
                Diag[c1] += tmpvar[count];
            }
            else if (i == c2) {
                Diag[c2] += tmpvar[count + 1];
            }
		}	
		
	}
	
}
#if (defined ShareMemory)
__global__ void gpuReductionDiag3ShareMemory(RealFlow *Diag, const RealFlow *tmpvar, const IntType *f2c, const IntType* C2F,
								const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell, const IntType nBFace, 
								const IntType nTFace){
	extern __shared__ double sdata[];										
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType j, c1, c2, face, count;
	
	if(i < nTCell){		
		sdata[threadIdx.x] = Diag[i];
	}
	__syncthreads();
	
	if(i < nTCell){		
		for (j = 0; j < nFPC[i]; j++) {
            face = C2F[IndexC2F[i] + j];
            count = 2 * face;
			c1 = f2c[count];
            c2 = f2c[count + 1];
			if (i == c1) {
                sdata[threadIdx.x] += tmpvar[count];
            }
            else if (i == c2) {
                sdata[threadIdx.x] += tmpvar[count + 1];
            }
		}	
		
	}
	__syncthreads();
	
	if(i < nTCell){		
		Diag[i] = sdata[threadIdx.x];
	}	
}
#endif
void cuReductionDiag3(){
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
#if (defined ShareMemory)	
	gpuReductionDiag3ShareMemory <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gDiag, gtmpvar, gf2c, gC2F, gIndexC2F, 
															gnFPC, gnTCell, gnBFace, gnTFace);								
#else
	gpuReductionDiag3 <<< blocksPerGrid, threadsPerBlock >>> (gDiag, gtmpvar, gf2c, gC2F, gIndexC2F, 
															gnFPC, gnTCell, gnBFace, gnTFace);		
#endif	
	//HANDLE_API_ERR(cudaMemcpy(Diag, gDiag, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpuReductionDiag2(RealFlow *tmpvar, const RealFlow *q, const RealGeom *xfn, 
								const RealGeom *yfn, const RealGeom *zfn, const RealGeom *vgn, 
								const RealGeom *area, const IntType *f2c, const IntType steady, 
								const RealFlow lhs_omga, const RealFlow gam, const RealFlow p_bar, 
								const IntType nTCell, const IntType nBFace, const IntType nTFace){
	IntType i = nBFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, count;
    RealFlow vn_1, vn_2, ss_1, ss_2;
    RealFlow eig;
	IntType Cell = nTCell + nBFace;
	if(i < nTFace){		
		c1 = f2c[2*i];
        c2 = f2c[2*i + 1];
        count = 2 * i;
        // Cell c1
        vn_1 = q[1*Cell + c1]*xfn[i] + q[2*Cell + c1]*yfn[i] + q[3*Cell + c1]*zfn[i];
        if(!steady) vn_1 -= vgn[i];
        vn_1 = fabs(vn_1);
        ss_1 = gam*(q[4*Cell + c1] + p_bar)/q[c1];
        
        eig  = vn_1 + sqrt(ss_1);
		tmpvar[count] = 0.5*area[i] * eig*lhs_omga;
        
        // Cell c2
        vn_2 = q[1*Cell + c2]*xfn[i] + q[2*Cell + c2]*yfn[i] + q[3*Cell + c2]*zfn[i];
        if(!steady) vn_2 -= vgn[i];
        vn_2 = fabs(vn_2);
        ss_2 = gam*(q[4*Cell + c2] + p_bar)/q[c2];
        
        eig  = vn_2 + sqrt(ss_2);
        tmpvar[count + 1] = 0.5*area[i] * eig*lhs_omga;
	}		
																
}

void cuReductionDiag2(){
	
	IntType blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuReductionDiag2 <<< blocksPerGrid, threadsPerBlock >>> (gtmpvar, gq, gxfn, gyfn, gzfn, gvgn, garea, 
															gf2c, gsteady, glhs_omga, ggam, gp_bar, 
															gnTCell, gnBFace, gnTFace);
															
	//HANDLE_API_ERR(cudaMemcpy(tmpvar, gtmpvar, 2*gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpuReductionDiag(RealFlow *tmpvar, const RealFlow *q, const RealGeom *xfn, 
								const RealGeom *yfn, const RealGeom *zfn, const RealGeom *vgn, 
								const RealGeom *area, const IntType *f2c, const IntType steady, 
								const RealFlow lhs_omga, const RealFlow gam, const RealFlow p_bar, 
								const IntType nTCell, const IntType nBFace){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, count;
    RealFlow vn_1, ss_1;
    RealFlow eig;
	IntType Cell = nTCell + nBFace;
	if(i < nBFace){
		count = 2 * i;
        c1 = f2c[2*i];
        vn_1 = q[1*Cell + c1]*xfn[i] + q[2*Cell + c1]*yfn[i] + q[3*Cell + c1]*zfn[i];
        if(!steady) vn_1 -= vgn[i];
        vn_1 = fabs(vn_1);
        ss_1 = gam*(q[4*Cell + c1] + p_bar)/q[c1];
       
        eig  = vn_1 + sqrt(ss_1);
        
		tmpvar[count] = 0.5*area[i] * eig*lhs_omga;
	}		
																
}

void cuReductionDiag(){
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuReductionDiag <<< blocksPerGrid, threadsPerBlock >>> (gtmpvar, gq, gxfn, gyfn, gzfn, gvgn, garea, 
															gf2c, gsteady, glhs_omga, ggam, gp_bar, gnTCell, gnBFace);
															
	//HANDLE_API_ERR(cudaMemcpy(tmpvar, gtmpvar, 2*gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

void cuCalDiagLUSGS(PolyGrid *grid, IntType level){	      
   
    // Boundary faces first	
	cuReductionDiag(); 
    // Interior faces
	cuReductionDiag2(); 
	// Reduction:
	cuReductionDiag3(); 
	
    /* if (!steady){  
        for (IntType i = 0; i < nTCell; i++) Diag[i] += (1.0 + time_accuracy)*vol[i] / real_dt; 
    }*/
    
    // If flow is viscous, need to count the contribution from viscosity
    // 该程序的粘性增加预处理的需要修改,然后进行测试
    IntType vis_mode, vis_run=0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    if(vis_mode != INVISCID){
        vis_run = 1;
        // if coarse grid doesn't want to run the viscous flux, turn it off
        if(level != 0){
            IntType cg_vis = 1;
            grid->GetData(&cg_vis, INT, 1, "cg_vis");
            if(cg_vis == 0) vis_run = 0;
        }
    }

    if(vis_run){        		
		cuDiag_v();				
		cuDiag_v2Diag();
    }
}

__global__ void gpuDiagInit(RealFlow *Diag, const RealFlow *dt, const RealGeom *vol, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		Diag[i] = vol[i] / dt[i];
	}
	
}

void cuDiagInit(RealFlow *dt){
	
	//HANDLE_API_ERR(cudaMemcpy(gdt, dt, gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDiagInit <<< blocksPerGrid, threadsPerBlock >>> (gDiag, gdt, gvol, gnTCell);
	
	//HANDLE_API_ERR(cudaMemcpy(Diag, gDiag, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	
}

__global__ void gpuBackwardSweepMany(RealFlow *DQ, const RealFlow *Diag, const RealFlow *q, 
								const RealFlow *rhs, RealFlow *norm, RealFlow *dqo, 
								const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, const RealFlow *vgn, 
								const RealFlow *area, const RealFlow *vis_l, const RealFlow *vis_t, 
								const RealGeom *norm_dist_c2c, const IntType *luorder, const IntType *layer, 
								const IntType *C2F, const IntType *IndexC2F, const IntType *f2c, const IntType *nFPC, 
								const RealFlow gam, const RealFlow p_bar, const RealFlow lhs_omga, 
								const IntType vis_run, 
								const IntType start, const IntType end, 
								const IntType steady, const IntType NumCell, const IntType nTCell
								){
	
	IntType ilu = start + blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < end){
		IntType cell = luorder[ilu];
		IntType face, c1, c2, c_tmp, count;
		RealFlow flux[5], q_loc[5], DQ_loc[5], visc;
		RealGeom face_n[3], dist;
		RealFlow  DQO[5], tmp;
                   
		for(IntType i = 0; i < 5; i++) {
			DQO[i]       = DQ[i*NumCell + cell];
			DQ[i*NumCell + cell]  = rhs[i*nTCell + cell] - dqo[i*nTCell + cell];
			dqo[i*nTCell + cell] = 0.0;
		}
		for(IntType j = 0; j < nFPC[cell]; j++){
			face  = C2F[IndexC2F[cell] + j];
			count = face + face;
			c1    = f2c[count++];
			c2    = f2c[count];
			// One of c1 and c2 must be cell itself. 
			if(!(layer[c1] < layer[cell] || layer[c2] < layer[cell])){
				// Now its neighboring cell belongs to upper triangular
				face_n[0] = xfn[face];
				face_n[1] = yfn[face];
				face_n[2] = zfn[face];
				//if(!steady) vgn_tmp = vgn[face];
				if(c2 == cell){
					c_tmp = c1;
					c1 = c2;
					c2 = c_tmp;
					face_n[0] = -face_n[0];
					face_n[1] = -face_n[1];
					face_n[2] = -face_n[2];
					//if(!steady) vgn_tmp = -vgn[face];
				}
				//assert(c1 == cell);
				for(IntType i=0; i<5; i++){
					q_loc[i]  = q[i*NumCell + c2];
					DQ_loc[i] = DQ[i*NumCell + c2];
				}
				// Calculate everything (I call it Flux) in upper triangular
				if(steady){
					GPUFluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
				}else{
					//FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
				}
			   
				if(vis_run){
					dist = norm_dist_c2c[face];
					visc = vis_l[c2] + vis_t[c2];
					tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
					for(IntType i = 0; i < 5; i++) flux[i] -= tmp*DQ_loc[i];
				}
				// Add Flux together
				tmp = 0.5*area[face];
				for(IntType i = 0; i < 5; i++) {
					flux[i] *= tmp;
					dqo[i*nTCell + cell] += flux[i];
					DQ[i*NumCell + cell] -= flux[i];
				}
			}			
		}
		
		for(IntType i=0; i<5; i++) {
			DQ[i*NumCell + cell] /= Diag[cell];
			tmp = DQ[i*NumCell + cell] - DQO[i];
			norm[cell] = tmp*tmp;
		}       			        

	}	
	
}


__global__ void gpuForwardSweepMany(RealFlow *DQ, const RealFlow *Diag, const RealFlow *rhs, RealFlow *norm, 
							RealFlow *dqo, const IntType *luorder, IntType nTCell, IntType Cell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	RealFlow DQO[5];
	RealFlow tmp;
	
	if(i < 1){	
		IntType cellx = luorder[0]; 
		for (IntType ii = 0; ii < 5; ii++){
			DQO[ii] = DQ[ii*Cell + cellx];
			DQ[ii*Cell + cellx] = rhs[ii*nTCell + cellx] - dqo[ii*nTCell + cellx];
			dqo[ii*nTCell + cellx] = 0.0;
			DQ[ii*Cell + cellx] /= Diag[cellx];	
			tmp = DQ[ii*Cell + cellx] - DQO[ii];
			norm[cellx] = tmp*tmp;
		}
	}
} 

__global__ void gpuForwardSweep2Many(RealFlow *DQ, const RealFlow *Diag, const RealFlow *q, 
								const RealFlow *rhs, RealFlow *norm, RealFlow *dqo, 
								const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, const RealFlow *vgn, 
								const RealFlow *area, const RealFlow *vis_l, const RealFlow *vis_t, 
								const RealGeom *norm_dist_c2c, const IntType *luorder, const IntType *layer, 
								const IntType *C2F, const IntType *IndexC2F, const IntType *f2c, const IntType *nFPC, 
								const RealFlow gam, const RealFlow p_bar, const RealFlow lhs_omga, 
								const IntType vis_run,
								const IntType start, const IntType end, 
								const IntType steady, const IntType NumCell, const IntType nTCell
								){
	
	IntType ilu = start + blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < end){
		IntType cell = luorder[ilu];
		
		RealFlow  DQO[5], tmp;
		
		for(IntType i=0; i<5; i++) {
			DQO[i]       = DQ[i*NumCell + cell];
			DQ[i*NumCell + cell]  = rhs[i*nTCell + cell] - dqo[i*nTCell + cell];
			dqo[i*nTCell + cell] = 0.0;
		}
		
		for(IntType j=0; j<nFPC[cell]; j++){
			IntType   face, c1, c2, c_tmp, count;
			RealFlow  flux[5], q_loc[5], DQ_loc[5], visc, tmp;
			RealGeom  face_n[3], dist;
			face  = C2F[IndexC2F[cell] + j];
			count = face + face;
			c1    = f2c[count++];
			c2    = f2c[count];
			// One of c1 and c2 must be cell itself. 
			if(!(layer[c1]>layer[cell] || layer[c2]>layer[cell])){
				// Now its neighboring cell belongs to lower triangular
				face_n[0] = xfn[face];
				face_n[1] = yfn[face];
				face_n[2] = zfn[face];
				//if(!steady) vgn_tmp = vgn[face];
				if(c2 == cell){
					c_tmp = c1;
					c1    = c2;
					c2    = c_tmp;
					face_n[0] = -face_n[0];
					face_n[1] = -face_n[1];
					face_n[2] = -face_n[2];
					//if(!steady) vgn_tmp = -vgn[face];
				}
				//assert(c1 == cell);
				
				for(IntType i=0; i<5; i++){
					q_loc[i]  = q[i*NumCell + c2];
					DQ_loc[i] = DQ[i*NumCell + c2];
				}
				// Calculate everything (I call it Flux) in lower triangular
				if(steady){
					GPUFluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
				}else{
					//FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
				}
				
				if(vis_run){
					dist = norm_dist_c2c[face];
					visc = vis_l[c2] + vis_t[c2];
					tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
					for(IntType i=0; i<5; i++) flux[i] -= tmp*DQ_loc[i];
				}

				// Add Flux together
				tmp = 0.5*area[face];
				for(IntType i=0; i<5; i++) {
					flux[i] *= tmp;
					dqo[i*nTCell + cell] += flux[i];
					DQ[i*NumCell + cell] -= flux[i];
				}
			}
		}
		for(IntType i=0; i<5; i++) {
			DQ[i*NumCell + cell] /= Diag[cell];
			tmp = DQ[i*NumCell + cell] - DQO[i];
			norm[cell] = tmp*tmp;
		}
	}
	
}


void cuSolveLUSGS3D(PolyGrid *grid, RealFlow *Diag, RealFlow *DQ[5],
                  RealFlow *rhs[5], IntType *nFPC, IntType **C2F, IntType Nsweep, 
                  RealFlow epsilon, IntType level){
					  
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
   
    IntType steady=1;
    grid->GetData(&steady,  INT, 1, "steady");
    RealFlow gam, p_bar, lhs_omga;
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&lhs_omga,   REAL_FLOW, 1, "lhs_omga");
    
    IntType vis_mode, vis_run = 0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    if(vis_mode != INVISCID){
        vis_run = 1;
     
        // if coarse grid doesn't want to run the viscous flux, turn it off
        if(level != 0){
            IntType cg_vis = 1;
            grid->GetData(&cg_vis, INT, 1, "cg_vis");
            if(cg_vis == 0) vis_run = 0;
        }
    }

	RealFlow tmp;
    RealFlow norm0, norm, dmax = 1.0;

	RealFlow *normtmp;
	mfmem::snew_array_1D(normtmp, nTCell,dmrfl);
	
	IntType blocksPerGrid = (5*gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDQInit <<< blocksPerGrid, threadsPerBlock >>> (gdqo, 5*gnTCell); 

    IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer"); 
    
    IntType nTFace = grid->GetNTFace();   

    for(IntType sweep=0; sweep<Nsweep; sweep++){
        norm = 0.0;
		blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
		gpuDQInit <<< blocksPerGrid, threadsPerBlock >>> (gSAsumv2, gnTCell); 
		
        // Now the Forward Sweep
		blocksPerGrid = (1 + threadsPerBlock - 1) / threadsPerBlock;	
		gpuForwardSweepMany <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gDiag, gres, gSAsumv2, 
							gdqo, gluorder, gnTCell, gnTCell + gnBFace);		
		
        IntType laynum;
		IntType start, end;
		for(laynum=0; laynum<cellsPerlayer[0]; laynum++ ){
			start = cellsPerlayer[laynum+1];
			end   = cellsPerlayer[laynum+2];
			if(laynum == 0) {start++;} 
			
			blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;	
			gpuForwardSweep2Many <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gDiag, gq, gres, gSAsumv2, gdqo, gxfn, gyfn, gzfn, gvgn, 
								garea, gvis_l, gvis_t, gnorm_dist_c2c, gluorder, glayer, 
								gC2F, gIndexC2F, gf2c, gnFPC, gam, gp_bar, lhs_omga, 
								vis_run, start, end, steady, gnTCell + gnBFace, gnTCell);
								
		}

#ifdef MPICH
        IntType nvar = 5;
		gMPI = gDQ;
		grid->cuRecvSendVarNeighbor_Togeth_q5(nvar);
#endif        
		
		// Backward Sweep
		for(IntType laynum = cellsPerlayer[0] - 1; laynum >= 0; laynum--){
			IntType end = cellsPerlayer[laynum + 2];
			IntType start = cellsPerlayer[laynum + 1];
			// Note ensure that the value of start was smaller than the end 
			blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;	
			gpuBackwardSweepMany <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gDiag, gq, gres, gSAsumv2, gdqo, gxfn, gyfn, gzfn, gvgn, 
								garea, gvis_l, gvis_t, gnorm_dist_c2c, gluorder, glayer, 
								gC2F, gIndexC2F, gf2c, gnFPC, gam, gp_bar, lhs_omga, 
								vis_run, start, end, steady, gnTCell + gnBFace, gnTCell);		
	
		} 
		
		blocksPerGrid = gSAnodata2;
		Reducekernel6 <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gSAsumv2, gSAodata2, gSAnsum2);
		HANDLE_API_ERR(cudaMemcpy(normtmp, gSAodata2, blocksPerGrid*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		
		for(IntType i=0; i<blocksPerGrid; i++) norm += normtmp[i];  
		
		/* IntType blocksPerGrid2 = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;	
		Reducekernel_sum <<< blocksPerGrid2, threadsPerBlock >>> (val_Reduction, gSAodata2, blocksPerGrid);
		//HANDLE_API_ERR(cudaMemcpy(odata, godata2, 1*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		//sum = odata[0];
		cudaDeviceSynchronize();
		norm = val_Reduction[0];*/
		
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
	mfmem::sdel_array_1D(normtmp);

/* #ifdef MPICH
    if(myZone == 1) printf("Resi reduced by %.5e with %d sweeps\n", dmax, (int)sweep);
#else   
    printf("Resi reduced by %.5e with %d sweeps\n", dmax, (int)sweep);
#endif */

}

__global__ void deResDiag( RealFlow* DQ, RealFlow* res, RealFlow* Diag, IntType nTCell, IntType n){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		DQ[0*n + i] = res[0*nTCell + i] / Diag[i];
		DQ[1*n + i] = res[1*nTCell + i] / Diag[i];
		DQ[2*n + i] = res[2*nTCell + i] / Diag[i];
		DQ[3*n + i] = res[3*nTCell + i] / Diag[i];
		DQ[4*n + i] = res[4*nTCell + i] / Diag[i];
	}
}
__global__ void gpuDPLUR(RealFlow *DQ, RealFlow *dqo, const RealFlow *Diag, const RealFlow *q, 
								const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, const RealFlow *vgn, 
								const RealFlow *area, const RealFlow *vis_l, const RealFlow *vis_t, 
								const RealGeom *norm_dist_c2c, const IntType *luorder, const IntType *layer, 
								const IntType *C2F, const IntType *IndexC2F, const IntType *f2c, const IntType *nFPC, 
								const RealFlow gam, const RealFlow gamm1, const RealFlow p_bar, const RealFlow lhs_omga, 
								const IntType vis_run, const RealFlow rho_min, const RealFlow rho_max, const RealFlow p_min,
								const RealFlow p_max, const RealFlow rho00, const RealFlow e_stag, 
								const IntType DQ_limit, const IntType nTCell,
								const IntType steady, const IntType NumCell, const IntType NVar
								){
	
	IntType ilu = blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < nTCell){
		IntType cell = ilu;//luorder[ilu];
		
		for(IntType idx_C2F=0; idx_C2F<nFPC[cell]; idx_C2F++){
			IntType   face, c1, c2, count;
			RealFlow  flux[5], q_loc[5], DQ_loc[5];
			RealGeom  face_n[3], dist;
			face  = C2F[IndexC2F[cell] + idx_C2F];
			count = face + face;
			c1    = f2c[count];
			c2    = f2c[count++];

			face_n[0] = xfn[face];
			face_n[1] = yfn[face];
			face_n[2] = zfn[face];
			//if(!steady) vgn_tmp = vgn[face];
			if(c2 == cell){
				IntType c_tmp = c1;
				c1    = c2;
				c2    = c_tmp;
				face_n[0] = -face_n[0];
				face_n[1] = -face_n[1];
				face_n[2] = -face_n[2];
				//if(!steady) vgn_tmp = -vgn[face];
			}
				//assert(c1 == cell);
				
			for(IntType i=0; i<5; ++i){
				q_loc[i]  = q[i*NumCell + c2];
				DQ_loc[i] = dqo[i*NumCell + c2];
			}
			// Calculate everything (I call it Flux) in lower triangular
			if(steady){
				GPUFluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
			}else{
				//FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
			}
				
			if(vis_run){
				dist = norm_dist_c2c[face];
				RealFlow visc = vis_l[c2] + vis_t[c2];
				RealFlow tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
				for(IntType i=0; i<5; ++i) flux[i] -= tmp*DQ_loc[i];
			}

			// Add Flux together
			RealFlow tmp = 0.5*area[face];
			for(IntType i=0; i<5; ++i) DQ[i*NumCell + cell] -= tmp*flux[i];
			
		}
		for(IntType i=0; i<5; ++i) DQ[i*NumCell + cell] /= Diag[cell];			
		//limit for rho>0
		if(DQ_limit == 1){
			// do nothing!
		}else if(DQ_limit == 2){  
			RealFlow dp,vv; 
			vv = q[1*NumCell + cell]*q[1*NumCell + cell] + q[2*NumCell + cell]*q[2*NumCell + cell] + 
				q[3*NumCell + cell]*q[3*NumCell + cell];
			dp = DQ[4*NumCell + cell]+0.5*DQ[0*NumCell + cell]*vv - (DQ[1*NumCell + cell]*q[1*NumCell + cell] + 
				DQ[2*NumCell + cell]*q[2*NumCell + cell] + DQ[3*NumCell + cell]*q[3*NumCell + cell]);
			dp *= gamm1; 
			if((q[0*NumCell + cell] + DQ[0*NumCell + cell]) < rho_min || (q[0*NumCell + cell] + DQ[0*NumCell + cell]) > rho_max ||
			   (q[4*NumCell + cell] + dp) < p_min || (q[4*NumCell + cell] + dp) > p_max){
				DQ[0*NumCell + cell] *= 0.1;
				DQ[1*NumCell + cell] *= 0.1;
				DQ[2*NumCell + cell] *= 0.1;
				DQ[3*NumCell + cell] *= 0.1;
				DQ[4*NumCell + cell] *= 0.1;
			}
			dp = DQ[4*NumCell + cell] + 0.5*DQ[0*NumCell + cell]*vv - (DQ[1*NumCell + cell]*q[1*NumCell + cell] + 
				DQ[2*NumCell + cell]*q[2*NumCell + cell] + DQ[3*NumCell + cell]*q[3*NumCell + cell]);
			dp  *= gamm1;
			if((q[0*NumCell + cell] + DQ[0*NumCell + cell]) < rho_min || (q[0*NumCell + cell] + DQ[0*NumCell + cell]) > rho_max ||
			   (q[4*NumCell + cell] + dp) < p_min || (q[4*NumCell + cell] + dp) > p_max){
				DQ[0*NumCell + cell] *= 0.1;
				DQ[1*NumCell + cell] *= 0.1;
				DQ[2*NumCell + cell] *= 0.1;
				DQ[3*NumCell + cell] *= 0.1;
				DQ[4*NumCell + cell] *= 0.1;
			}
			dp = DQ[4*NumCell + cell] + 0.5*DQ[0*NumCell + cell]*vv - (DQ[1*NumCell + cell]*q[1*NumCell + cell] + 
				DQ[2*NumCell + cell]*q[2*NumCell + cell] + DQ[3*NumCell + cell]*q[3*NumCell + cell]);
			dp *= gamm1;
			if((q[0*NumCell + cell] + DQ[0*NumCell + cell]) < rho_min || (q[0*NumCell + cell]+DQ[0*NumCell + cell]) > rho_max ||
			   (q[4*NumCell + cell] + dp) < p_min || (q[4*NumCell + cell] + dp) > p_max){
				DQ[0*NumCell + cell] = 0.0;
				DQ[1*NumCell + cell] = 0.0;
				DQ[2*NumCell + cell] = 0.0;
				DQ[3*NumCell + cell] = 0.0;
				DQ[4*NumCell + cell] = 0.0;
			}
		}else if(DQ_limit == 3){
			DQ[0*NumCell + cell] = GPUMAX2(DQ[0*NumCell + cell], rho_min - q[0*NumCell + cell]);
			DQ[0*NumCell + cell] = GPUMIN2(DQ[0*NumCell + cell], rho_max - q[0*NumCell + cell]);     
		}else if(DQ_limit == 4){
			RealFlow alph,alph_rho,alph_rhoe,alph_p,dp,vv,rhoe;
			vv   = q[1*NumCell + cell]*q[1*NumCell + cell] + q[2*NumCell + cell]*q[2*NumCell + cell] + q[3*NumCell + cell]*q[3*NumCell + cell];
			rhoe = 0.5*q[0*NumCell + cell]*vv + (q[4*NumCell + cell] + p_bar)/(gam - 1.0);
			dp   = DQ[4*NumCell + cell] + 0.5*DQ[0*NumCell + cell]*vv - (DQ[1*NumCell + cell]*q[1*NumCell + cell] + 
				DQ[2*NumCell + cell]*q[2*NumCell + cell] + DQ[3*NumCell + cell]*q[3*NumCell + cell]);
			dp  *= gamm1; 

			alph_rho  = q[0*NumCell + cell]/(GPUMAX2(q[0*NumCell + cell], 0.05*rho00) + GPUMAX2(0.0, -DQ[0*NumCell + cell]));
			alph_rhoe = rhoe/(GPUMAX2(rhoe, 0.05*e_stag) + GPUMAX2(0.0, -DQ[4*NumCell + cell]));
			alph_p    = (q[4*NumCell + cell] + p_bar)/(GPUMAX2((q[4*NumCell + cell] + p_bar), 0.05*p_bar) + GPUMAX2(0.0, -dp));
			alph      = GPUMIN2(alph_rho, alph_rhoe);
			alph      = GPUMIN2(alph, alph_p);
			for(IntType i=0;i<5;i++) DQ[i*NumCell + cell] *= alph;
		}	
	}
}

__global__ void dplurTrans(RealFlow *dqo, RealFlow *DQ, RealFlow *res, IntType nTCell, IntType n){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		dqo[0*n + i] = DQ[0*n + i];
		DQ[0*n + i] = res[0*nTCell + i];
		dqo[1*n + i] = DQ[1*n + i];
		DQ[1*n + i] = res[1*nTCell + i];
		dqo[2*n + i] = DQ[2*n + i];
		DQ[2*n + i] = res[2*nTCell + i];
		dqo[3*n + i] = DQ[3*n + i];
		DQ[3*n + i] = res[3*nTCell + i];
		dqo[4*n + i] = DQ[4*n + i];
		DQ[4*n + i] = res[4*nTCell + i];
	}
}

void cuForwardDPLUR( PolyGrid *grid, IntType level ){
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
	IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
	IntType  n      = nTCell + nBFace;
	
	RealFlow gam, gamm1;
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    gamm1 = gam - 1.0;
	
    RealFlow rho00,u00,v00,w00,e_stag;
    grid->GetData(&rho00, REAL_FLOW, 1, "rho");
    grid->GetData(&u00, REAL_FLOW, 1, "u");
    grid->GetData(&v00, REAL_FLOW, 1, "v");
    grid->GetData(&w00, REAL_FLOW, 1, "w");
    grid->GetData(&e_stag,   REAL_FLOW, 1, "e_stag");

    RealFlow rho_min,rho_max,p_min,p_max,e_stag_max, p_bar, lhs_omga;
    grid->GetData(&rho_min, REAL_FLOW, 1, "rho_min");
    grid->GetData(&rho_max, REAL_FLOW, 1, "rho_max");
    grid->GetData(&p_min,   REAL_FLOW, 1, "p_min");
    grid->GetData(&p_max,   REAL_FLOW, 1, "p_max");
	grid->GetData(&e_stag_max, REAL_FLOW, 1, "e_stag_max");
	grid->GetData(&lhs_omga, REAL_FLOW, 1, "lhs_omga");
	grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
	
    IntType DQ_limit = 1;
    grid->GetData(&DQ_limit, INT, 1, "DQ_limit");
    
    IntType vis_mode, vis_run = 0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    
    IntType sweeps = 1;
    grid->GetData(&sweeps, INT, 1, "sweeps");
    
    // Get number of faces for each cell
    IntType *nFPC = CalnFPC(grid);
    // Get cell to face conections
    IntType **C2F = CalC2F(grid);
    
    IntType i;
    // Now diagonal term in LU-SGS, here we need information of time steps
    RealFlow *Diag = NULL;
    //mfmem::snew_array_1D(Diag, nTCell,dmrfl);
    //assert(Diag != 0);
    //未修改overlap
	
	IntType blocksPerGrid = (5*n + threadsPerBlock - 1) / threadsPerBlock;	
	gpuDQInit <<< blocksPerGrid, threadsPerBlock >>> (gdqo, 5*n);
	gpuDQInit <<< blocksPerGrid, threadsPerBlock >>> (gDQ, 5*n);

    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
	
	cuDiagInit(dt);

    // Note: As it has been shown, Diag = CFL/2*Vol/Dt.
    //       If function CalDiagLUSGS is not called, make sure CFL <= 2.
    //未修改overlap
	cuCalDiagLUSGS(grid, level);

	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	deResDiag<<< blocksPerGrid, threadsPerBlock >>>( gDQ, gres, gDiag, gnTCell, n);

	for ( IntType idx_sweeps = 0; idx_sweeps < sweeps; ++idx_sweeps ) {
		blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
		dplurTrans<<< blocksPerGrid, threadsPerBlock >>>( gdqo, gDQ, gres, gnTCell, n);

#ifdef MPICH
    IntType nvar = 5;
	gMPI = gdqo;
    grid->cuRecvSendVarNeighbor_Togeth_q5(nvar);
#endif

		blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;		
		gpuDPLUR <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gdqo, gDiag, gq, gxfn, gyfn, gzfn, gvgn, 
			garea, gvis_l, gvis_t, gnorm_dist_c2c, gluorder, glayer, gC2F, gIndexC2F, gf2c, gnFPC, 
			gam, gamm1, p_bar, lhs_omga, vis_run, rho_min, rho_max, p_min, p_max, rho00, e_stag,
			DQ_limit, gnTCell, gsteady, n, 5);
	}

	RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*n, "DQ");
    if(!DQ[0]){
        mfmem::snew_array_1D(DQ[0],5*n,dmrfl);
        grid->UpdateDataPtr(DQ[0], REAL_FLOW, 5*n, "DQ");
    }
    assert(DQ[0] != 0);
	for(i=1; i<5; i++) DQ[i] = &DQ[i-1][n];
	
	cuUpdateFlowField3D_CFL3d(grid, DQ);

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

}

void cuForwardLUSGS(PolyGrid *grid, IntType level){
	
#ifdef FS_CUDA_DEBUG_NS_LUSGS
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
	
	IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  n      = nTCell + nBFace;
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");
    RealGeom *vol   = grid->GetCellVol();
    
    IntType sweeps = 1;
    grid->GetData(&sweeps, INT, 1, "sweeps");
    RealFlow epsilon = 0.1;
    grid->GetData(&epsilon, REAL_FLOW, 1, "epsilon");
    if(epsilon < TINY) epsilon = 0.1;
    
    // Get number of faces for each cell
    IntType *nFPC = CalnFPC(grid);
    // Get cell to face conections
    IntType **C2F = CalC2F(grid);
    
    IntType i;
    // Now diagonal term in LU-SGS, here we need information of time steps
    RealFlow *Diag = NULL;
    //mfmem::snew_array_1D(Diag, nTCell,dmrfl);
    //assert(Diag != 0);
    //未修改overlap
    
    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
	
	cuDiagInit(dt);
	
    // Note: As it has been shown, Diag = CFL/2*Vol/Dt.
    //       If function CalDiagLUSGS is not called, make sure CFL <= 2.
    //未修改overlap
    cuCalDiagLUSGS(grid, level);
    
    // Allocate memories for RHS or DQ
    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*n, "DQ");
    if(!DQ[0]){
        mfmem::snew_array_1D(DQ[0],5*n,dmrfl);
        grid->UpdateDataPtr(DQ[0], REAL_FLOW, 5*n, "DQ");
    }
    //assert(DQ[0] != 0);
    for(i=1; i<5; i++) DQ[i] = &DQ[i-1][n];
	
	cuDQInit(5);
    
    if(sweeps == 1){  //单步
        // Copy the residual to DQ		
        for(i=0; i<5; i++){
			cures2DQ(DQ, res, i);
        }
		
        // Now the LU-SGS part
        cuSolveLUSGS3D(grid, Diag, DQ, nFPC, C2F, level);
    }
	else if (sweeps > 1){  //多步		
		RealFlow *rhs[5];
        rhs[0] = res;
        for(IntType j=1; j<5; j++) rhs[j] = &rhs[j-1][nTCell];
        // Now the LU-SGS part ,   DQ conservative variable
        cuSolveLUSGS3D(grid, Diag, DQ, rhs, nFPC, C2F, sweeps, epsilon, level);		 		
	}
    // Update flow field
    cuUpdateFlowField3D_CFL3d(grid, DQ);
    
    // delete temporary memories
    //mfmem::sdel_array_1D(Diag);
	
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
	
#ifdef FS_CUDA_DEBUG_NS_LUSGS
	cuMemoryPreparaGMRESDebug2(grid);
#endif
}

__global__ void gpuTimeMarch(RealFlow *nq, const RealFlow *q, const RealFlow *dt, const RealFlow *res, const RealGeom *vol, 
						const RealFlow lamda, const RealFlow gamm1, const RealFlow p_bar, const IntType nTCell, const IntType nBFace){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType n = nTCell + nBFace;
	
	RealFlow rho, mx, my, mz, et, p;
	
	if(i < nTCell){
		
		RealFlow dtbv = dt[i] / (vol[i] + TINY) * lamda;

        rho  = q[0*nTCell + i];
        mx   = q[1*nTCell + i];
        my   = q[2*nTCell + i];
        mz   = q[3*nTCell + i];
        et   = q[4*nTCell + i];

        rho += dtbv*res[0*nTCell + i];
        mx  += dtbv*res[1*nTCell + i];
        my  += dtbv*res[2*nTCell + i];
        mz  += dtbv*res[3*nTCell + i];
        et  += dtbv*res[4*nTCell + i];
        p    = gamm1*(et - 0.5*(mx*mx + my*my + mz*mz)/rho);

        // Check if density or pressure is less than 0. If they are, make correction.
        if(p <= -p_bar || rho <= 0.){        
            // let dt be one order smaller;
            rho -= dtbv*res[0*nTCell + i]*0.9;
            mx  -= dtbv*res[1*nTCell + i]*0.9;
            my  -= dtbv*res[2*nTCell + i]*0.9;
            mz  -= dtbv*res[3*nTCell + i]*0.9;
            et  -= dtbv*res[4*nTCell + i]*0.9;
            p    = gamm1*(et - 0.5*(mx*mx + my*my + mz*mz)/rho);
           
            if(p <= -p_bar || rho <= 0.){
                // let dt be one order smaller once more;
                rho -= dtbv*res[0*nTCell + i]*0.09;
                mx  -= dtbv*res[1*nTCell + i]*0.09;
                my  -= dtbv*res[2*nTCell + i]*0.09;
                mz  -= dtbv*res[3*nTCell + i]*0.09;
                et  -= dtbv*res[4*nTCell + i]*0.09;
                p    = gamm1*(et - 0.5*(mx*mx + my*my + mz*mz)/rho);
            }
        }
        
        if(p > -p_bar && rho > 0.){
            nq[0*n + i] = rho;
            nq[1*n + i] = mx/rho;
            nq[2*n + i] = my/rho;
            nq[3*n + i] = mz/rho;
            nq[4*n + i] = p;
        }
	}
	
}

void cuTimeMarch(PolyGrid *grid, RealFlow **q, RealFlow *dt, RealFlow lamda) {  
   
    const RealFlow gam = 1.4;
    const RealFlow gamm1 = gam - 1.0;
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuTimeMarch <<< blocksPerGrid, threadsPerBlock >>> (gq, goldq, gdt, gres, gvol, lamda, gamm1, gp_bar, gnTCell, gnBFace);

}

__global__ void gpuLoadQandTransQtoW(RealFlow *q, const RealFlow *q0, const RealFlow gamm1, const IntType nTCell, const IntType nBFace){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType n = nTCell + nBFace;
	if(i < nTCell){
		for (IntType k = 0; k < 5; ++k) {
            q[k*nTCell + i] = q0[k*n + i];
        }
		q[1*nTCell + i] *= q[0*nTCell + i];
        q[2*nTCell + i] *= q[0*nTCell + i];
        q[3*nTCell + i] *= q[0*nTCell + i];
        q[4*nTCell + i]  = q[4*nTCell + i]/gamm1 + 
						0.5*(q[1*nTCell + i]*q[1*nTCell + i]+q[2*nTCell + i]*q[2*nTCell + i]+q[3*nTCell + i]*q[3*nTCell + i])/q[0*nTCell + i];
	}
	
}

void LoadQandTransQtoW(PolyGrid *grid, RealFlow **q) {
    const IntType nTCell = grid->GetNTCell();
    const IntType n = nTCell + grid->GetNBFace();
	
	const RealFlow gam = 1.4;
    const RealFlow gamm1 = gam -1.0;    
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuLoadQandTransQtoW <<< blocksPerGrid, threadsPerBlock >>> (goldq, gq, gamm1, gnTCell, gnBFace);
	
	/* RealFlow* q0[5];
    q0[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q0[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q0[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q0[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q0[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    for (IntType i = 0; i < nTCell; i++) {
        for (IntType k = 0; k < 5; ++k) {
            q[k][i] = q0[k][i];
        }
		q[1][i] *= q[0][i];
        q[2][i] *= q[0][i];
        q[3][i] *= q[0][i];
        q[4][i]  = q[4][i]/gamm1 + 0.5*(q[1][i]*q[1][i]+q[2][i]*q[2][i]+q[3][i]*q[3][i])/q[0][i];
    } */
	
/* 	HANDLE_API_ERR(cudaMemcpy(q[0], goldq, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	HANDLE_API_ERR(cudaMemcpy(q[1], &goldq[gnTCell], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	HANDLE_API_ERR(cudaMemcpy(q[2], &goldq[2*gnTCell], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	HANDLE_API_ERR(cudaMemcpy(q[3], &goldq[3*gnTCell], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	HANDLE_API_ERR(cudaMemcpy(q[4], &goldq[4*gnTCell], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	 */
}

void cuExplicitStep(PolyGrid* grid) {
    const IntType nTCell = grid->GetNTCell();	
	
    IntType n_stage;
    RealFlow lamda[10];
    grid->GetData(&n_stage, INT, 1, "n_stage");
    grid->GetData(lamda, REAL_FLOW, n_stage, "lamda");

    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");

    RealFlow **q = NULL;
	mfmem::snew_array_2D(q, 5, nTCell, dmrfl, true);


    LoadQandTransQtoW(grid, q);
    //TransQtoW(grid, q);
	
    for (IntType i = 0; i < n_stage; i++) {
#if (defined FS_CUDA_DEBUG_NS_RK)		
        RealFlow *res = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5 * nTCell, "res");
        for (IntType icell = 0; icell < 5 * nTCell; icell++) {
            res[icell] = 0.;
        }
#else
		cuZeroResiduals(grid);
#endif

        UpdateResiduals(grid, 0);
		
#if (defined FS_CUDA_DEBUG_NS_RK)
		cuMemoryPreparaGMRESDebug(grid);
#endif	
	
#ifdef TIMECOST
	cudaDeviceSynchronize();
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemRK, endtimeTemRK;
    double timeuseTemRK;
    gettimeofday(&starttimeTemRK, 0); 
#endif
#endif

		cuTimeMarch(grid, q, dt, lamda[i]);
		
#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    timecost[2] = timecost[2] + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemRK, 0); 
    timeuseTemRK = (RealGeom) 1000000*(endtimeTemRK.tv_sec - starttimeTemRK.tv_sec) + endtimeTemRK.tv_usec - starttimeTemRK.tv_usec;
    timecost[2] += timeuseTemRK;
    timeuseTemRK /= 1000000.0;
    time_lusgs += timeuseTemRK;
#endif
#endif

#ifdef FS_CUDA_DEBUG_NS_RK
		cuMemoryPreparaGMRESDebug2(grid);
#endif
	}
	


    mfmem::sdel_array_2D(q); // clear, CHF, 20220324	
}

__global__ void gpuBackwardSweep(RealFlow *DQ, const RealFlow *Diag, const RealFlow *q, 
								const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, const RealFlow *vgn, 
								const RealFlow *area, const RealFlow *vis_l, const RealFlow *vis_t, 
								const RealGeom *norm_dist_c2c, const IntType *luorder, const IntType *layer, 
								const IntType *C2F, const IntType *IndexC2F, const IntType *f2c, const IntType *nFPC, 
								const RealFlow gam, const RealFlow gamm1, const RealFlow p_bar, const RealFlow lhs_omga, 
								const IntType vis_run, const RealFlow rho_min, const RealFlow rho_max, const RealFlow p_min,
								const RealFlow p_max, const RealFlow rho00, const RealFlow e_stag, 
								const IntType DQ_limit, const IntType start, const IntType end, 
								const IntType steady, const IntType NumCell, const IntType NVar
								){
	
	IntType ilu = start + blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < end){
		IntType cell = luorder[ilu];
		IntType face, c1, c2, c_tmp, count;
		RealFlow flux_s[5], flux[5], q_loc[5], DQ_loc[5], visc, tmp;
		RealGeom face_n[3], dist;
                   
		for(IntType i = 0; i < 5; i++) flux_s[i] = 0.;
		for(IntType j = 0; j < nFPC[cell]; j++){
			face  = C2F[IndexC2F[cell] + j];
			count = face + face;
			c1    = f2c[count++];
			c2    = f2c[count];
			// One of c1 and c2 must be cell itself. 
			if(!(layer[c1] < layer[cell] || layer[c2] < layer[cell])){
				// Now its neighboring cell belongs to upper triangular
				face_n[0] = xfn[face];
				face_n[1] = yfn[face];
				face_n[2] = zfn[face];
				//if(!steady) vgn_tmp = vgn[face];
				if(c2 == cell){
					c_tmp = c1;
					c1 = c2;
					c2 = c_tmp;
					face_n[0] = -face_n[0];
					face_n[1] = -face_n[1];
					face_n[2] = -face_n[2];
					//if(!steady) vgn_tmp = -vgn[face];
				}
				//assert(c1 == cell);
				for(IntType i=0; i<5; i++){
					q_loc[i]  = q[i*NumCell + c2];
					DQ_loc[i] = DQ[i*NumCell + c2];
				}
				// Calculate everything (I call it Flux) in upper triangular
				if(steady){
					GPUFluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
				}else{
					//FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
				}
			   
				if(vis_run){
					dist = norm_dist_c2c[face];
					visc = vis_l[c2] + vis_t[c2];
					tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
					for(IntType i = 0; i < 5; i++) flux[i] -= tmp*DQ_loc[i];
				}
				// Add Flux together
				tmp = area[face];
				for(IntType i = 0; i < 5; i++) flux_s[i] += tmp*flux[i];
			}			
		}
		tmp = 2.0*Diag[cell];
		for(IntType i=0; i<5; i++) DQ[i*NumCell + cell] -= flux_s[i]/tmp;             			        

		//limit for rho>0
		if(DQ_limit == 1){
			// do nothing!
		}else if(DQ_limit == 2){  
			RealFlow dp,vv; 
			vv = q[1*NumCell + cell]*q[1*NumCell + cell] + q[2*NumCell + cell]*q[2*NumCell + cell] + q[3*NumCell + cell]*q[3*NumCell + cell];
			dp = DQ[4*NumCell + cell] + 0.5*DQ[0*NumCell + cell]*vv - (DQ[1*NumCell + cell]*q[1*NumCell + cell] + 
				DQ[2*NumCell + cell]*q[2*NumCell + cell] + DQ[3*NumCell + cell]*q[3*NumCell + cell]);
			dp *= gamm1; 
			if((q[0*NumCell + cell] + DQ[0*NumCell + cell]) < rho_min || (q[0*NumCell + cell] + DQ[0*NumCell + cell]) > rho_max ||
			   (q[4*NumCell + cell] + dp) < p_min || (q[4*NumCell + cell] + dp) > p_max){
				DQ[0*NumCell + cell] *= 0.1;
				DQ[1*NumCell + cell] *= 0.1;
				DQ[2*NumCell + cell] *= 0.1;
				DQ[3*NumCell + cell] *= 0.1;
				DQ[4*NumCell + cell] *= 0.1;
			}
			dp = DQ[4*NumCell + cell] + 0.5*DQ[0*NumCell + cell]*vv - (DQ[1*NumCell + cell]*q[1*NumCell + cell] + 
				DQ[2*NumCell + cell]*q[2*NumCell + cell] + DQ[3*NumCell + cell]*q[3*NumCell + cell]);
			dp *= gamm1;
			if((q[0*NumCell + cell] + DQ[0*NumCell + cell]) < rho_min || (q[0*NumCell + cell] + DQ[0*NumCell + cell]) > rho_max ||
			   (q[4*NumCell + cell] + dp) < p_min || (q[4*NumCell + cell] + dp) > p_max){
				DQ[0*NumCell + cell] *= 0.1;
				DQ[1*NumCell + cell] *= 0.1;
				DQ[2*NumCell + cell] *= 0.1;
				DQ[3*NumCell + cell] *= 0.1;
				DQ[4*NumCell + cell] *= 0.1;
			}
			dp = DQ[4*NumCell + cell] + 0.5*DQ[0*NumCell + cell]*vv - (DQ[1*NumCell + cell]*q[1*NumCell + cell] + 
				DQ[2*NumCell + cell]*q[2*NumCell + cell] + DQ[3*NumCell + cell]*q[3*NumCell + cell]);
			dp *= gamm1;
			if((q[0*NumCell + cell] + DQ[0*NumCell + cell]) < rho_min || (q[0*NumCell + cell] + DQ[0*NumCell + cell]) > rho_max ||
			   (q[4*NumCell + cell] + dp) < p_min || (q[4*NumCell + cell] + dp) > p_max){
				DQ[0*NumCell + cell] = 0.0;
				DQ[1*NumCell + cell] = 0.0;
				DQ[2*NumCell + cell] = 0.0;
				DQ[3*NumCell + cell] = 0.0;
				DQ[4*NumCell + cell] = 0.0;
			}
		}else if(DQ_limit == 3){
			DQ[0*NumCell + cell] = GPUMAX2(DQ[0*NumCell + cell], rho_min - q[0*NumCell + cell]);
			DQ[0*NumCell + cell] = GPUMIN2(DQ[0*NumCell + cell], rho_max - q[0*NumCell + cell]);
		}else if(DQ_limit == 4){
			RealFlow alph, alph_rho, alph_rhoe, alph_p, dp, vv, rhoe;
			vv = q[1*NumCell + cell]*q[1*NumCell + cell] + q[2*NumCell + cell]*q[2*NumCell + cell] + q[3*NumCell + cell]*q[3*NumCell + cell];
			rhoe = 0.5*q[0*NumCell + cell]*vv + (q[4*NumCell + cell] + p_bar)/(gam - 1.0);
			dp = DQ[4*NumCell + cell] + 0.5*DQ[0*NumCell + cell]*vv - (DQ[1*NumCell + cell]*q[1*NumCell + cell] + 
				DQ[2*NumCell + cell]*q[2*NumCell + cell] + DQ[3*NumCell + cell]*q[3*NumCell + cell]);
			dp *= gamm1; 

			alph_rho = q[0*NumCell + cell]/(GPUMAX2(q[0*NumCell + cell], 0.05*rho00) + GPUMAX2(0.0, -DQ[0*NumCell + cell]));
			alph_rhoe = rhoe/(GPUMAX2(rhoe, 0.05*e_stag) + GPUMAX2(0.0, -DQ[4*NumCell + cell]));
			alph_p = (q[4*NumCell + cell] + p_bar)/(GPUMAX2((q[4*NumCell + cell] + p_bar), 0.05*p_bar) + GPUMAX2(0.0, -dp));
			alph = GPUMIN2(alph_rho, alph_rhoe);
			alph = GPUMIN2(alph, alph_p);
			for(IntType i=0; i < 5; i++) DQ[i*NumCell + cell] *= alph;
		}
	}
	
	
}

void cuBackwardSweep(PolyGrid *grid, RealFlow *DQ[5], RealFlow *Diag, IntType level){
	
	IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
	
	RealFlow gam, gamm1;
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    gamm1 = gam - 1.0;
	
    RealFlow rho00,u00,v00,w00,e_stag;
    grid->GetData(&rho00, REAL_FLOW, 1, "rho");
    grid->GetData(&u00, REAL_FLOW, 1, "u");
    grid->GetData(&v00, REAL_FLOW, 1, "v");
    grid->GetData(&w00, REAL_FLOW, 1, "w");
    grid->GetData(&e_stag,   REAL_FLOW, 1, "e_stag");

    RealFlow rho_min,rho_max,p_min,p_max,e_stag_max;
    grid->GetData(&rho_min, REAL_FLOW, 1, "rho_min");
    grid->GetData(&rho_max, REAL_FLOW, 1, "rho_max");
    grid->GetData(&p_min,   REAL_FLOW, 1, "p_min");
    grid->GetData(&p_max,   REAL_FLOW, 1, "p_max");
    grid->GetData(&e_stag_max, REAL_FLOW, 1, "e_stag_max");
	
    IntType DQ_limit = 1;
    grid->GetData(&DQ_limit, INT, 1, "DQ_limit");
    
    IntType vis_mode, vis_run = 0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
	
	IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer");
	
	if(vis_mode != INVISCID){
        vis_run = 1;        
        // if coarse grid doesn't want to run the viscous flux, turn it off
        if(level != 0){
            IntType cg_vis = 1;
            grid->GetData(&cg_vis, INT, 1, "cg_vis");
            if(cg_vis == 0) vis_run = 0;
        }
    }	     
	
	IntType NumCell = nTCell + nBFace;
	
	for(IntType laynum = cellsPerlayer[0] - 1; laynum >= 0; laynum--){
        IntType end = cellsPerlayer[laynum + 2];
        IntType start = cellsPerlayer[laynum + 1];
		// Note ensure that the value of start was smaller than the end 
		IntType blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;		
		gpuBackwardSweep <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gDiag, gq, gxfn, gyfn, gzfn, gvgn, 
																garea, gvis_l, gvis_t, gnorm_dist_c2c, 
																gluorder, glayer, gC2F, gIndexC2F, gf2c, gnFPC, 
																ggam, gamm1, gp_bar, glhs_omga, vis_run, 
																rho_min, rho_max, p_min, p_max, rho00, e_stag,
																DQ_limit, start, end, gsteady, NumCell, 5);	
	}
	//HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, 5*(nTCell + nBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost)); // for deltaQ_glb.out output
}

__global__ void gpuForwardSweep(RealFlow *DQ, const RealFlow *Diag, const IntType *luorder, IntType Cell, IntType NVar){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < NVar){
		DQ[i*Cell + luorder[0]] /= Diag[luorder[0]];		
	}
}

__global__ void gpuForwardSweep2(RealFlow *DQ, const RealFlow *Diag, const RealFlow *q, 
								const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, const RealFlow *vgn, 
								const RealFlow *area, const RealFlow *vis_l, const RealFlow *vis_t, 
								const RealGeom *norm_dist_c2c, const IntType *luorder, const IntType *layer, 
								const IntType *C2F, const IntType *IndexC2F, const IntType *f2c, const IntType *nFPC, 
								const RealFlow gam, const RealFlow gamm1, const RealFlow p_bar, const RealFlow lhs_omga, 
								const IntType vis_run, const RealFlow rho_min, const RealFlow rho_max, const RealFlow p_min,
								const RealFlow p_max, const RealFlow rho00, const RealFlow e_stag, 
								const IntType DQ_limit, const IntType start, const IntType end, 
								const IntType steady, const IntType NumCell, const IntType NVar
								){
	
	IntType ilu = start + blockDim.x*blockIdx.x + threadIdx.x;
	if(ilu < end){
		IntType cell = luorder[ilu];
		
		for(IntType j=0; j<nFPC[cell]; j++){
			IntType   face, c1, c2, c_tmp, count;
			RealFlow  flux[5], q_loc[5], DQ_loc[5], visc, tmp;
			RealGeom  face_n[3], dist;
			face  = C2F[IndexC2F[cell] + j];
			count = face + face;
			c1    = f2c[count++];
			c2    = f2c[count];
			// One of c1 and c2 must be cell itself. 
			if(!(layer[c1]>layer[cell] || layer[c2]>layer[cell])){
				// Now its neighboring cell belongs to lower triangular
				face_n[0] = xfn[face];
				face_n[1] = yfn[face];
				face_n[2] = zfn[face];
				//if(!steady) vgn_tmp = vgn[face];
				if(c2 == cell){
					c_tmp = c1;
					c1    = c2;
					c2    = c_tmp;
					face_n[0] = -face_n[0];
					face_n[1] = -face_n[1];
					face_n[2] = -face_n[2];
					//if(!steady) vgn_tmp = -vgn[face];
				}
				//assert(c1 == cell);
				
				for(IntType i=0; i<5; i++){
					q_loc[i]  = q[i*NumCell + c2];
					DQ_loc[i] = DQ[i*NumCell + c2];
				}
				// Calculate everything (I call it Flux) in lower triangular
				if(steady){
					GPUFluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
				}else{
					//FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
				}
				
				if(vis_run){
					dist = norm_dist_c2c[face];
					visc = vis_l[c2] + vis_t[c2];
					tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
					for(IntType i=0; i<5; i++) flux[i] -= tmp*DQ_loc[i];
				}

				// Add Flux together
				tmp = 0.5*area[face];
				for(IntType i=0; i<5; i++) DQ[i*NumCell + cell] -= tmp*flux[i];
			}
		}
		for(IntType i=0; i<5; i++) DQ[i*NumCell + cell] /= Diag[cell];			
		//limit for rho>0
		if(DQ_limit == 1){
			// do nothing!
		}else if(DQ_limit == 2){  
			RealFlow dp,vv; 
			vv = q[1*NumCell + cell]*q[1*NumCell + cell] + q[2*NumCell + cell]*q[2*NumCell + cell] + 
				q[3*NumCell + cell]*q[3*NumCell + cell];
			dp = DQ[4*NumCell + cell]+0.5*DQ[0*NumCell + cell]*vv - (DQ[1*NumCell + cell]*q[1*NumCell + cell] + 
				DQ[2*NumCell + cell]*q[2*NumCell + cell] + DQ[3*NumCell + cell]*q[3*NumCell + cell]);
			dp *= gamm1; 
			if((q[0*NumCell + cell] + DQ[0*NumCell + cell]) < rho_min || (q[0*NumCell + cell] + DQ[0*NumCell + cell]) > rho_max ||
			   (q[4*NumCell + cell] + dp) < p_min || (q[4*NumCell + cell] + dp) > p_max){
				DQ[0*NumCell + cell] *= 0.1;
				DQ[1*NumCell + cell] *= 0.1;
				DQ[2*NumCell + cell] *= 0.1;
				DQ[3*NumCell + cell] *= 0.1;
				DQ[4*NumCell + cell] *= 0.1;
			}
			dp = DQ[4*NumCell + cell] + 0.5*DQ[0*NumCell + cell]*vv - (DQ[1*NumCell + cell]*q[1*NumCell + cell] + 
				DQ[2*NumCell + cell]*q[2*NumCell + cell] + DQ[3*NumCell + cell]*q[3*NumCell + cell]);
			dp  *= gamm1;
			if((q[0*NumCell + cell] + DQ[0*NumCell + cell]) < rho_min || (q[0*NumCell + cell] + DQ[0*NumCell + cell]) > rho_max ||
			   (q[4*NumCell + cell] + dp) < p_min || (q[4*NumCell + cell] + dp) > p_max){
				DQ[0*NumCell + cell] *= 0.1;
				DQ[1*NumCell + cell] *= 0.1;
				DQ[2*NumCell + cell] *= 0.1;
				DQ[3*NumCell + cell] *= 0.1;
				DQ[4*NumCell + cell] *= 0.1;
			}
			dp = DQ[4*NumCell + cell] + 0.5*DQ[0*NumCell + cell]*vv - (DQ[1*NumCell + cell]*q[1*NumCell + cell] + 
				DQ[2*NumCell + cell]*q[2*NumCell + cell] + DQ[3*NumCell + cell]*q[3*NumCell + cell]);
			dp *= gamm1;
			if((q[0*NumCell + cell] + DQ[0*NumCell + cell]) < rho_min || (q[0*NumCell + cell]+DQ[0*NumCell + cell]) > rho_max ||
			   (q[4*NumCell + cell] + dp) < p_min || (q[4*NumCell + cell] + dp) > p_max){
				DQ[0*NumCell + cell] = 0.0;
				DQ[1*NumCell + cell] = 0.0;
				DQ[2*NumCell + cell] = 0.0;
				DQ[3*NumCell + cell] = 0.0;
				DQ[4*NumCell + cell] = 0.0;
			}
		}else if(DQ_limit == 3){
			DQ[0*NumCell + cell] = GPUMAX2(DQ[0*NumCell + cell], rho_min - q[0*NumCell + cell]);
			DQ[0*NumCell + cell] = GPUMIN2(DQ[0*NumCell + cell], rho_max - q[0*NumCell + cell]);     
		}else if(DQ_limit == 4){
			RealFlow alph,alph_rho,alph_rhoe,alph_p,dp,vv,rhoe;
			vv   = q[1*NumCell + cell]*q[1*NumCell + cell] + q[2*NumCell + cell]*q[2*NumCell + cell] + q[3*NumCell + cell]*q[3*NumCell + cell];
			rhoe = 0.5*q[0*NumCell + cell]*vv + (q[4*NumCell + cell] + p_bar)/(gam - 1.0);
			dp   = DQ[4*NumCell + cell] + 0.5*DQ[0*NumCell + cell]*vv - (DQ[1*NumCell + cell]*q[1*NumCell + cell] + 
				DQ[2*NumCell + cell]*q[2*NumCell + cell] + DQ[3*NumCell + cell]*q[3*NumCell + cell]);
			dp  *= gamm1; 

			alph_rho  = q[0*NumCell + cell]/(GPUMAX2(q[0*NumCell + cell], 0.05*rho00) + GPUMAX2(0.0, -DQ[0*NumCell + cell]));
			alph_rhoe = rhoe/(GPUMAX2(rhoe, 0.05*e_stag) + GPUMAX2(0.0, -DQ[4*NumCell + cell]));
			alph_p    = (q[4*NumCell + cell] + p_bar)/(GPUMAX2((q[4*NumCell + cell] + p_bar), 0.05*p_bar) + GPUMAX2(0.0, -dp));
			alph      = GPUMIN2(alph_rho, alph_rhoe);
			alph      = GPUMIN2(alph, alph_p);
			for(IntType i=0;i<5;i++) DQ[i*NumCell + cell] *= alph;
		}	
			
	}
	
}

void cuForwardSweep(PolyGrid *grid, RealFlow *DQ[5], RealFlow *Diag, IntType level){
	
	IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    
    RealFlow gam, gamm1;
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    gamm1 = gam-1.0;
	
    RealFlow rho00,u00,v00,w00,e_stag;
    grid->GetData(&rho00, REAL_FLOW, 1, "rho");
    grid->GetData(&u00, REAL_FLOW, 1, "u");
    grid->GetData(&v00, REAL_FLOW, 1, "v");
    grid->GetData(&w00, REAL_FLOW, 1, "w");
    grid->GetData(&e_stag,   REAL_FLOW, 1, "e_stag");

    RealFlow rho_min,rho_max,p_min,p_max,e_stag_max;
    grid->GetData(&rho_min, REAL_FLOW, 1, "rho_min");
    grid->GetData(&rho_max, REAL_FLOW, 1, "rho_max");
    grid->GetData(&p_min,   REAL_FLOW, 1, "p_min");
    grid->GetData(&p_max,   REAL_FLOW, 1, "p_max");
    grid->GetData(&e_stag_max, REAL_FLOW, 1, "e_stag_max");
	
    IntType DQ_limit = 1;
    grid->GetData(&DQ_limit, INT, 1, "DQ_limit");
    
    IntType vis_mode, vis_run = 0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
	
	if(vis_mode != INVISCID){
        vis_run = 1;        
        // if coarse grid doesn't want to run the viscous flux, turn it off
        if(level != 0){
            IntType cg_vis = 1;
            grid->GetData(&cg_vis, INT, 1, "cg_vis");
            if(cg_vis == 0) vis_run = 0;
        }
    }
	
    IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer");
	
	IntType NumCell = nTCell + nBFace;
	IntType blocksPerGrid = (5 + threadsPerBlock - 1) / threadsPerBlock;
	gpuForwardSweep <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gDiag, gluorder, NumCell, 5);
	//HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, 5*(nTCell + nBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//for(IntType i=0; i<5; i++) DQ[i][luorder[0]] /= Diag[luorder[0]];

    IntType start, end;
    for(IntType laynum=0; laynum<cellsPerlayer[0]; laynum++ ){
        start = cellsPerlayer[laynum + 1];
        end   = cellsPerlayer[laynum + 2];
        if(laynum == 0) {start++;} 
		
		blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;		
		gpuForwardSweep2 <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gDiag, gq, gxfn, gyfn, gzfn, gvgn, 
																garea, gvis_l, gvis_t, gnorm_dist_c2c, 
																gluorder, glayer, gC2F, gIndexC2F, gf2c, gnFPC, 
																ggam, gamm1, gp_bar, glhs_omga, vis_run, 
																rho_min, rho_max, p_min, p_max, rho00, e_stag,
																DQ_limit, start, end, gsteady, NumCell, 5);		
    }
	/*
	for (IntType i = 0; i < 5; i++){
		HANDLE_API_ERR(cudaMemcpy(&DQ[i][0], &gDQ[i*(gnTCell + gnBFace) + gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		//HANDLE_API_ERR(cudaMemcpy(&DQ[i][0], &gDQ[i*(gnTCell + gnBFace) + 0], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	}
	*/
	
	
}

void cuSolveLUSGS3D(PolyGrid *grid, RealFlow *Diag, RealFlow *DQ[5], IntType *nFPC, IntType **C2F, IntType level){
       
    // Now the Forward Sweep
	cuForwardSweep(grid, DQ, Diag, level);

#ifdef MPICH	
	IntType nvar = 5;
	gMPI = gDQ;
	grid->cuRecvSendVarNeighbor_Togeth_q5(nvar);
#endif
	
	// Backward Sweep 
	cuBackwardSweep(grid, DQ, Diag, level);
}

__global__ void gpuZeroResiduals (RealFlow *res, IntType nT5){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nT5){
		res[i] = 0.0;
	}
	
}

void cuZeroResiduals(PolyGrid *grid){ 
  	
	IntType nT5 = 5*gnTCell;
	IntType blocksPerGrid = (nT5 + threadsPerBlock - 1) / threadsPerBlock;	
	gpuZeroResiduals <<< blocksPerGrid, threadsPerBlock >>> (gres, nT5);
	
}

void cuForwardStep(PolyGrid *grid, RealFlow *rhs, IntType level, IntType steps){
    
	cuZeroResiduals(grid); //Flux Comput. no need to transfer res into GPU again. 
	
    UpdateResiduals(grid, level); // Including Grad. Limit and Flux Comput.

    IntType gmres = 0;
    grid->GetData(&gmres, INT, 1, "GMRES", 0);
#if defined FS_CUDA_DEBUG_NS_GMRES
	if(gmres != 1){
		cout << "input.para gmres should set to 1 to run GMRES. " << endl;
		exit(0);
	}
#endif
#if defined FS_CUDA_DEBUG_NS_LUSGS
	if(gmres == 1){
		cout << "input.para gmres should set to 0 to run LUSGS. " << endl;
		exit(0);
	}
#endif

    if(gmres == 1){
        //cuGMRESSolverOrig(grid, level);
		cuGMRESSolverOrigUpdate(grid, level);
    }else if(gmres == 0) {
		IntType tScheme;
		grid->GetData(&tScheme, INT, 1, "tScheme");
		if (tScheme == LU_SGS) {
			cuForwardLUSGS(grid, level);
        }
        else if (tScheme == DPLUR) {
			cuForwardDPLUR(grid, level);
			//cuForwardLUSGS(grid, level);
        }
    }
}

__global__ void gpuTimeStepNormal_new (RealFlow *dt, IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		dt[i] = BIG;
	}
	
}


__global__ void gpuTimeStepNormal_new2 (RealFlow *tmpvar, const RealFlow *q, const RealFlow *vis_l, const RealFlow *vis_t, 
									const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc, 
									const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, 
									const RealGeom *xfn, const RealGeom *yfn, const RealGeom *zfn, const RealGeom *area, 
									const RealGeom *vol, 
									const IntType *f2c, const RealGeom *vgn, const IntType steady, const RealFlow p_bar, 
									const RealFlow gam, const RealFlow prl, const RealFlow prt, const RealFlow C,
									const IntType vis_run, const IntType nTCell, const IntType nBFace){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType count, c1;
	RealFlow eigv, dn, vn, c2tmp, gam_tmp;
	RealFlow muoopr;
	IntType Cell = nTCell + nBFace;
	if(i < nBFace){
		count = 2 * i;
        c1 = f2c[count];

        c2tmp = gam * (q[4*Cell + c1] + p_bar) / q[c1];
        dn = fabs((xfc[i] - xcc[c1]) * xfn[i] + (yfc[i] - ycc[c1]) * yfn[i] + (zfc[i] - zcc[c1]) * zfn[i]);

        vn = q[1*Cell + c1] * xfn[i] + q[2*Cell + c1] * yfn[i] + q[3*Cell + c1] * zfn[i];
        if (!steady) vn -= vgn[i];
        vn = fabs(vn);
        eigv = vn + sqrt(c2tmp);

        if (vis_run) {
            muoopr = vis_l[c1] / prl + vis_t[c1] / prt;
            gam_tmp = gam;

            //eigv += C*gam_tmp/rho[c1]*muoopr/(dn+TINY);
            eigv += C * gam_tmp / q[c1] * muoopr * area[i] / vol[c1];
        }
        tmpvar[count] = dn / eigv;
	}
	
}

__global__ void gpuTimeStepNormal_new3 (RealFlow *tmpvar, const RealFlow *q, const RealFlow *vis_l, const RealFlow *vis_t, 
									const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc, 
									const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, 
									const RealGeom *xfn, const RealGeom *yfn, const RealGeom *zfn, const RealGeom *area, 
									const RealGeom *vol, 
									const IntType *f2c, const RealGeom *vgn, const IntType steady, const RealFlow p_bar, 
									const RealFlow gam, const RealFlow prl, const RealFlow prt, const RealFlow C,
									const IntType vis_run, const IntType nTFace, const IntType nTCell, const IntType nBFace){
	
	IntType i = nBFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType count, c1, c2;
	RealFlow eigv, dn, vn, c2tmp, gam_tmp;
	RealFlow muoopr;
	IntType Cell = nTCell + nBFace;
	if(i < nTFace){
		count = 2 * i;
        c1 = f2c[count];
        c2 = f2c[count + 1];

        c2tmp = gam * (q[4*Cell + c1] + p_bar) / q[c1];
        dn = fabs((xfc[i] - xcc[c1]) * xfn[i] + (yfc[i] - ycc[c1]) * yfn[i] + (zfc[i] - zcc[c1]) * zfn[i]);

        vn = q[1*Cell + c1] * xfn[i] + q[2*Cell + c1] * yfn[i] + q[3*Cell + c1] * zfn[i];
        if (!steady) vn -= vgn[i];
        vn = fabs(vn);
        eigv = vn + sqrt(c2tmp);

        if (vis_run) {
            muoopr = vis_l[c1] / prl + vis_t[c1] / prt;

            gam_tmp = gam;

            //eigv += C*gam_tmp/rho[c1]*muoopr/dn;
            eigv += C * gam_tmp / q[c1] * muoopr * area[i] / vol[c1];
        }
        tmpvar[count] = dn / eigv;

        c2tmp = gam * (q[4*Cell + c2] + p_bar) / q[c2];
        dn = fabs((xfc[i] - xcc[c2]) * xfn[i] + (yfc[i] - ycc[c2]) * yfn[i] + (zfc[i] - zcc[c2]) * zfn[i]);

        vn = q[1*Cell + c2] * xfn[i] + q[2*Cell + c2] * yfn[i] + q[3*Cell + c2] * zfn[i];
        if (!steady) vn -= vgn[i];
        vn = fabs(vn);
        eigv = vn + sqrt(c2tmp);

        if (vis_run) {
            muoopr = vis_l[c2] / prl + vis_t[c2] / prt;
            gam_tmp = gam;

            //eigv += C*gam_tmp/rho[c2]*muoopr/dn;
            eigv += C * gam_tmp / q[c2] * muoopr * area[i] / vol[c2];
        }
        tmpvar[count + 1] = dn / eigv;
	}
	
}

__global__ void gpuTimeStepNormal_newReduction(RealFlow *dt, const RealFlow *tmpvar, const IntType *f2c, const IntType* C2F,
								const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell){
									
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType j, c1, c2, face, count;

	if(i < nTCell){		
		for (j = 0; j < nFPC[i]; j++) {
            face = C2F[IndexC2F[i] + j];
            count = 2 * face;
			c1 = f2c[count];
            c2 = f2c[count + 1];
			if (i == c1) {
                dt[c1] = GPUMIN2(dt[c1], tmpvar[count]);
            }
            else{
                dt[c2] = GPUMIN2(dt[c2], tmpvar[count + 1]);
            }
		}	
		
	}
	
}

#if (defined ShareMemory)
__global__ void gpuTimeStepNormal_newReductionShareMemory(RealFlow *dt, const RealFlow *tmpvar, const IntType *f2c, const IntType* C2F,
								const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell){
	extern __shared__ double sdata[];		
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType j, c1, c2, face, count;
	
	if(i < nTCell){
		sdata[threadIdx.x] = dt[i];
	}
	__syncthreads();

	if(i < nTCell){		
		for (j = 0; j < nFPC[i]; j++) {
            face = C2F[IndexC2F[i] + j];
            count = 2 * face;
			c1 = f2c[count];
            c2 = f2c[count + 1];
			if (i == c1) {
                sdata[threadIdx.x] = GPUMIN2(sdata[threadIdx.x], tmpvar[count]);
            }
            else{
                sdata[threadIdx.x] = GPUMIN2(sdata[threadIdx.x], tmpvar[count + 1]);
            }
		}	
		
	}
	__syncthreads();
	
	if(i < nTCell){
		dt[i] = sdata[threadIdx.x];
	}
}
#endif

void cuTimeStepNormal_new(PolyGrid *grid, IntType vis_run){
    
    RealFlow C = 4.0;
    RealFlow prl, prt;
    if(vis_run){
        grid->GetData(&prl, REAL_FLOW, 1, "prl");  
        grid->GetData(&prt, REAL_FLOW, 1, "prt");
    }
	// Set dt to BIG
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuTimeStepNormal_new <<< blocksPerGrid, threadsPerBlock >>> (gdt, gnTCell);	
	
	//Manual reduction    		
	blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuTimeStepNormal_new2 <<< blocksPerGrid, threadsPerBlock >>> (gtmpvar, gq, gvis_l, gvis_t, gxfc, gyfc, gzfc, 
																gxcc, gycc, gzcc, gxfn, gyfn, gzfn, garea, gvol, gf2c, 
																gvgn, gsteady, gp_bar, ggam, prl, prt, C, vis_run, gnTCell, gnBFace);
		
    // For interior faces
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuTimeStepNormal_new3 <<< blocksPerGrid, threadsPerBlock >>> (gtmpvar, gq, gvis_l, gvis_t, gxfc, gyfc, gzfc, 
																gxcc, gycc, gzcc, gxfn, gyfn, gzfn, garea, gvol, gf2c, 
																gvgn, gsteady, gp_bar, ggam, prl, prt, C, vis_run,
																gnTFace, gnTCell, gnBFace);
    //HANDLE_API_ERR(cudaMemcpy(tmp_dt, gtmpvar, 2 * gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
#if (defined ShareMemory)
	gpuTimeStepNormal_newReductionShareMemory <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gdt, gtmpvar, gf2c, gC2F, gIndexC2F, 
															gnFPC, gnTCell);	
#else
	gpuTimeStepNormal_newReduction <<< blocksPerGrid, threadsPerBlock >>> (gdt, gtmpvar, gf2c, gC2F, gIndexC2F, 
															gnFPC, gnTCell);	
#endif	
	//HANDLE_API_ERR(cudaMemcpy(dt, gdt, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));

}

__global__ void gpuLimitTimeStep (RealFlow *dt, const RealFlow *p, const IntType *det, const RealFlow cfl, 
								const RealFlow cfl_min, const RealFlow p_min, const RealFlow p_break, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		RealFlow cfl_tmp;
        //根据压力场来确定当地cfl数。
        if(p[i] > p_break){
            cfl_tmp = cfl;
        }else if(p[i] < p_min){
            cfl_tmp = cfl_min;
        }else{
            cfl_tmp = (p[i] - p_min)/(p_break - p_min)*(cfl - cfl_min) + cfl_min;
        }
        //根据压力梯度的极值来限制当地cfl数
        if(!det[i]){
            cfl_tmp = cfl_min;
        }
        
        dt[i] *= cfl_tmp;
	}
	
}

__global__ void gpuLimitTimeStep_dtmindtmax (RealFlow *dtmax, RealFlow *dtmin, RealFlow *dt, const RealFlow *p, const IntType *det, const RealFlow cfl, 
								const RealFlow cfl_min, const RealFlow p_min, const RealFlow p_break, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		RealFlow cfl_tmp;
        //根据压力场来确定当地cfl数。
        if(p[i] > p_break){
            cfl_tmp = cfl;
        }else if(p[i] < p_min){
            cfl_tmp = cfl_min;
        }else{
            cfl_tmp = (p[i] - p_min)/(p_break - p_min)*(cfl - cfl_min) + cfl_min;
        }
        //根据压力梯度的极值来限制当地cfl数
        if(!det[i]){
            cfl_tmp = cfl_min;
        }
        
        dt[i] *= cfl_tmp;
		dtmax[i] = dt[i];
		dtmin[i] = dt[i];
	}
	
}

__global__ void gpuLimitTimeStep2 (RealFlow *dt, const RealFlow dt_max_lim, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		if(dt[i] > dt_max_lim){
            dt[i] = dt_max_lim;
        }
	}
	
}

__global__ void gpuCellIsMG(IntType *det, IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTCell){
		det[i] = 1;
	}
	
}

__global__ void gpuCellIsMG2(RealFlow *tmpvar, const RealFlow *p, const IntType *f2c, const RealFlow p_bar, IntType nTFace){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2;
	if(i < nTFace){
		c1 = f2c[i+i];
        c2 = f2c[i+i+1];
        tmpvar[i] = fabs(p[c2] - p[c1])/(p[c2] + p[c1] + p_bar + p_bar);
	}
	
}

__global__ void gpuCellIsMG3(IntType *det, const RealFlow *tmpvar, const IntType *f2c, const IntType* C2F,
							const IntType* IndexC2F, const IntType* nFPC, const RealFlow stind, 
							const IntType nTCell, const IntType nBFace, const IntType nTFace){
									
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType j, c1, c2, face, count;

	if(i < nTCell){		
		for (j = 0; j < nFPC[i]; j++) {
            face = C2F[IndexC2F[i] + j];
            count = 2 * face;
			c1 = f2c[count];
            c2 = f2c[count + 1];
			if (i == c1) {
                if (tmpvar[face] > stind){
					det[c1] = 0;
				}
            }
            else{ // i == c2
                if (tmpvar[face] > stind){
					det[c2] = 0;
				}
            }
		}	
		
	}
	
}

#if (defined ShareMemory)
__global__ void gpuCellIsMG3ShareMemory(IntType *det, const RealFlow *tmpvar, const IntType *f2c, const IntType* C2F,
							const IntType* IndexC2F, const IntType* nFPC, const RealFlow stind, 
							const IntType nTCell, const IntType nBFace, const IntType nTFace){
	extern __shared__ double sdata[];	
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType j, c1, c2, face, count;
	
	if(i < nTCell){
		sdata[threadIdx.x] = det[i];
	}
	__syncthreads();

	if(i < nTCell){		
		for (j = 0; j < nFPC[i]; j++) {
            face = C2F[IndexC2F[i] + j];
            count = 2 * face;
			c1 = f2c[count];
            c2 = f2c[count + 1];
			if (i == c1) {
                if (tmpvar[face] > stind){
					sdata[threadIdx.x] = 0;
				}
            }
            else{ // i == c2
                if (tmpvar[face] > stind){
					sdata[threadIdx.x] = 0;
				}
            }
		}	
		
	}
	__syncthreads();
	
	if(i < nTCell){
		det[i] = sdata[threadIdx.x];
	}
}		
#endif

void cuCellIsMG(PolyGrid *grid){

    // pressure threshod, default value is 0.001 in file "input.par"
    //RealFlow stind = 0.0001;
    //grid->GetData(&stind,REAL_FLOW,1,"stind",0);
    //stind *= 20.0;  // 0.02 as default
    //zhyb20200615: 由于修改了激波探测的规则，由压力差比来流总压修改为压力差比压力和，stind参数需要调整。
    //zhyb20200615: 根据喷流数值试验，将stind参数固化为0.1，这个值越大，判断出的激波单元越少，越小，越容易误判激波单元
    RealFlow stind = 0.1; 
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuCellIsMG <<< blocksPerGrid, threadsPerBlock >>> (gdet, gnTCell);
	
	// Manual Reduction: 	
	blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuCellIsMG2 <<< blocksPerGrid, threadsPerBlock >>> (gtmpvar, &gq[4*(gnTCell + gnBFace)], gf2c, gp_bar, gnTFace);
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
#if (defined ShareMemory)
	gpuCellIsMG3ShareMemory <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gdet, gtmpvar, gf2c, gC2F, gIndexC2F, 
														gnFPC, stind, gnTCell, gnBFace, gnTFace);
#else
	gpuCellIsMG3 <<< blocksPerGrid, threadsPerBlock >>> (gdet, gtmpvar, gf2c, gC2F, gIndexC2F, 
														gnFPC, stind, gnTCell, gnBFace, gnTFace);
#endif
}

template <unsigned int blockSize>
__device__ void warpReduce_Max(volatile double *sdata, unsigned int tid){
	if (blockSize >= 64) sdata[tid] = fmax(sdata[tid], sdata[tid + 32]);
	if (blockSize >= 32) sdata[tid] = fmax(sdata[tid], sdata[tid + 16]);
	if (blockSize >= 16) sdata[tid] = fmax(sdata[tid], sdata[tid + 8]);
	if (blockSize >= 8)  sdata[tid] = fmax(sdata[tid], sdata[tid + 4]);
	if (blockSize >= 4)  sdata[tid] = fmax(sdata[tid], sdata[tid + 2]);
	if (blockSize >= 2)  sdata[tid] = fmax(sdata[tid], sdata[tid + 1]);
}

__global__ void Reducekernel6_Max(double *g_idata, double *g_odata, int n){
	
	//__shared__ double sdata[512];
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + tid;

	sdata[tid] = fmax(g_idata[i], g_idata[i + blockDim.x]);
	__syncthreads();
	
	IntType blockSize = blockDim.x;
	if(blockSize >= 512){
		if(tid < 256){
			sdata[tid] = fmax(sdata[tid], sdata[tid + 256]);
		}
		__syncthreads();
	}
	if(blockSize >= 256){
		if(tid < 128){
			sdata[tid] = fmax(sdata[tid], sdata[tid + 128]);
		}
		__syncthreads();
	}
	if(blockSize >= 128){
		if(tid < 64){
			sdata[tid] = fmax(sdata[tid], sdata[tid + 64]);
		}
		__syncthreads();
	}
	
	if(tid < 32) warpReduce_Max<512>(sdata, tid);	
	
	if(tid == 0)
		g_odata[blockIdx.x] = sdata[0];	
	
}

__global__ void Reducekernel_Max(double *val_Reduction, double *g_odata, int n){
	
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < 1){
		for (int j = 1; j < n; j++){
			g_odata[0] = fmax(g_odata[0], g_odata[j]);
		}
		val_Reduction[0] = g_odata[0];
	}
	
}

template <unsigned int blockSize>
__device__ void warpReduce_Min(volatile double *sdata, unsigned int tid){
	if (blockSize >= 64) sdata[tid] = fmin(sdata[tid], sdata[tid + 32]);
	if (blockSize >= 32) sdata[tid] = fmin(sdata[tid], sdata[tid + 16]);
	if (blockSize >= 16) sdata[tid] = fmin(sdata[tid], sdata[tid + 8]);
	if (blockSize >= 8)  sdata[tid] = fmin(sdata[tid], sdata[tid + 4]);
	if (blockSize >= 4)  sdata[tid] = fmin(sdata[tid], sdata[tid + 2]);
	if (blockSize >= 2)  sdata[tid] = fmin(sdata[tid], sdata[tid + 1]);
}

__global__ void Reducekernel6_Min(double *g_idata, double *g_odata, int n){
	
	//__shared__ double sdata[512];
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + tid;

	sdata[tid] = fmin(g_idata[i], g_idata[i + blockDim.x]);
	__syncthreads();
	
	IntType blockSize = blockDim.x;
	if(blockSize >= 512){
		if(tid < 256){
			sdata[tid] = fmin(sdata[tid], sdata[tid + 256]);
		}
		__syncthreads();
	}
	if(blockSize >= 256){
		if(tid < 128){
			sdata[tid] = fmin(sdata[tid], sdata[tid + 128]);
		}
		__syncthreads();
	}
	if(blockSize >= 128){
		if(tid < 64){
			sdata[tid] = fmin(sdata[tid], sdata[tid + 64]);
		}
		__syncthreads();
	}
	
	if(tid < 32) warpReduce_Min<512>(sdata, tid);	
	
	if(tid == 0)
		g_odata[blockIdx.x] = sdata[0];	
	
}

__global__ void Reducekernel_Min(double *val_Reduction, double *g_odata, int n){
	
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < 1){
		for (int j = 1; j < n; j++){
			g_odata[0] = fmin(g_odata[0], g_odata[j]);
		}
		val_Reduction[1] = g_odata[0];
	}
	
}

__global__ void Reducekernel6_MaxAndMin(double *g_idataMax, double *g_odataMax, double *g_idataMin, double *g_odataMin, int n){
	
	//__shared__ double sdata[512];
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + tid;

	sdata[tid] = fmax(g_idataMax[i], g_idataMax[i + blockDim.x]);
	sdata[tid + n] = fmin(g_idataMin[i], g_idataMin[i + blockDim.x]);
	__syncthreads();
	
	IntType blockSize = blockDim.x;
	if(blockSize >= 512){
		if(tid < 256){
			sdata[tid] = fmax(sdata[tid], sdata[tid + 256]);
			sdata[tid + n] = fmin(sdata[tid], sdata[tid + 256]);
		}
		__syncthreads();
	}
	if(blockSize >= 256){
		if(tid < 128){
			sdata[tid] = fmax(sdata[tid], sdata[tid + 128]);
			sdata[tid + n] = fmin(sdata[tid], sdata[tid + 128]);
		}
		__syncthreads();
	}
	if(blockSize >= 128){
		if(tid < 64){
			sdata[tid] = fmax(sdata[tid], sdata[tid + 64]);
			sdata[tid + n] = fmin(sdata[tid], sdata[tid + 64]);
		}
		__syncthreads();
	}
	
	if(tid < 32) {
		warpReduce_Max<512>(sdata, tid);
		warpReduce_Min<512>(sdata + n, tid);
	}		
	
	if(tid == 0){
		g_odataMax[blockIdx.x] = sdata[0];	
		g_odataMin[blockIdx.x] = sdata[n];	
	}
}

__global__ void Reducekernel_MaxAndMin(double *val_Reduction, double *g_odataMax, double *g_odataMin, int n){
	
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i == 0){
		for (int j = 1; j < n; j++){
			g_odataMax[0] = fmax(g_odataMax[0], g_odataMax[j]);
		}
		val_Reduction[0] = g_odataMax[0];
	}
	else if (i == 1){
		for (int j = 1; j < n; j++){
			g_odataMin[0] = fmin(g_odataMin[0], g_odataMin[j]);
		}
		val_Reduction[1] = g_odataMin[0];
	} 	
	
}

void cuLimitTimeStep(PolyGrid *grid){
	
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    IntType level  = grid->GetLevel();
    
    RealFlow *p    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    
    RealFlow cfl;
    
    IntType  iter_done, cfl_nstep;
    RealFlow cfl_start, cfl_end, cfl_coeff, cfl_ratio;
    grid->GetData(&iter_done, INT, 1, "iter_done");
    grid->GetData(&cfl_nstep, INT, 1, "cfl_nstep");
    grid->GetData(&cfl_start, REAL_FLOW, 1, "cfl_start");
    grid->GetData(&cfl_end,   REAL_FLOW, 1, "cfl_end");
    grid->GetData(&cfl_coeff, REAL_FLOW, 1, "cfl_coeff");
    grid->GetData(&cfl_ratio, REAL_FLOW, 1, "cfl_ratio");
    
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
    grid->GetData(&p_min,   REAL_FLOW, 1, "p_min");
    grid->GetData(&p_break, REAL_FLOW, 1, "p_break");
    grid->GetData(&cfl_min, REAL_FLOW, 1, "cfl_min");
    //limit cfl using gradient of p, decrease cfl in big gradient of p
#if !(defined MultiStream)
    cuCellIsMG(grid);	
#endif   

    cfl_min = 0.5*cfl;  //在此处将cfl_min设为当前步cfl数乘以0.5	
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuLimitTimeStep_dtmindtmax <<< blocksPerGrid, threadsPerBlock >>> (gdtmaxsumv2, gdtminsumv2, gdt, &gq[4*(gnTCell + gnBFace)], gdet, cfl, cfl_min,  p_min, p_break, gnTCell);

    // Print out the maximum and minimun dt
    RealFlow dt_max = 0.0, dt_min = BIG;
	
	blocksPerGrid = gdtmaxnodata2;
	Reducekernel6_Max <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gdtmaxsumv2, gdtmaxodata2, gdtmaxnsum2);
	Reducekernel6_Min <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gdtminsumv2, gdtminodata2, gdtmaxnsum2);
	
	IntType blocksPerGrid2 = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;	
	Reducekernel_Max <<< blocksPerGrid2, threadsPerBlock >>> (val_Reduction, gdtmaxodata2, blocksPerGrid);
	Reducekernel_Min <<< blocksPerGrid2, threadsPerBlock >>> (val_Reduction, gdtminodata2, blocksPerGrid); 
	
	/* blocksPerGrid = gdtmaxnodata2;
	Reducekernel6_MaxAndMin <<< blocksPerGrid, threadsPerBlock, 2*threadsPerBlock*sizeof(RealFlow)>>> (gdtmaxsumv2, gdtmaxodata2, gdtminsumv2, gdtminodata2, threadsPerBlock);
	IntType blocksPerGrid2 = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;	
	Reducekernel_MaxAndMin <<< blocksPerGrid2, threadsPerBlock >>> (val_Reduction, gdtmaxodata2, gdtminodata2, blocksPerGrid);*/
	
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

    grid->UpdateData(&dt_max, REAL_FLOW, 1, "dt_max");
    grid->UpdateData(&dt_min, REAL_FLOW, 1, "dt_min");
    
    //Now limit the dt to ratio_dtmax*dt_min 
    RealFlow ratio_dtmax = 1.0e20;
    grid->GetData(&ratio_dtmax, REAL_FLOW, 1, "ratio_dtmax");
    RealFlow ratio_max = dt_max/dt_min;
	
    if(ratio_max > ratio_dtmax){
        RealFlow dt_max_lim = ratio_dtmax*dt_min;
		
        gpuLimitTimeStep2 <<< blocksPerGrid, threadsPerBlock >>> (gdt, dt_max_lim, gnTCell);
		//HANDLE_API_ERR(cudaMemcpy(dt, gdt, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));		
    }

}

void cuComputeTimeStep(PolyGrid *grid){
	
    IntType nTCell = grid->GetNTCell();
    IntType level  = grid->GetLevel();
    
    IntType vis_mode;
    grid->GetData(&vis_mode,  INT, 1, "vis_mode");  
    
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
        grid->GetData(&cg_vis, INT, 1, "cg_vis");
        if(cg_vis == 0) vis_run = 0;
    }

    cuTimeStepNormal_new(grid, vis_run);
    
    cuLimitTimeStep(grid);  //note: cfl number in this function
	
	//HANDLE_API_ERR(cudaMemcpy(gdt, dt, gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));
}

__global__ void gpuComputeVis_l(RealFlow *vis_l, const RealFlow *t, const RealFlow tref, const RealFlow sref,
							const RealFlow amuref, const IntType n){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < n){
		RealFlow temp = t[i]/tref;
        vis_l[i] = amuref*(temp*sqrt(temp)*(tref + sref)/(t[i] + sref));
	}
	
}

__global__ void gpuComputeVis_l2(RealFlow *vis_l, const IntType *f2c, const IntType *type_bcr, const RealGeom *tw_bcr, 
							const RealFlow tref, const RealFlow sref, const RealFlow amuref, const IntType nBFace){							
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType type, c1, c2;
	RealFlow temp, vis_wall, tmp;
	if(i < nBFace){
		type = type_bcr[i]; 
		c1 = f2c[2*i];
		c2 = f2c[2*i + 1];
		if(type == WALL){
			RealFlow tw = -1.0;
			tw = tw_bcr[i];
			if(tw > 0){
				temp = tw/tref;
				vis_wall = amuref*(temp*sqrt(temp)*(tref + sref)/(tw + sref));
				tmp = 2.0*vis_wall - vis_l[c1];
				atomicExchSM35(vis_l + c2, tmp);
			}
		}
	}
	
}


void cuComputeVis_l(PolyGrid *grid){
	
    IntType n = grid->GetNTCell() + grid->GetNBFace();
    IntType nBFace = grid->GetNBFace();
    RealFlow *vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
    if (vis_l == 0){
        mfmem::snew_array_1D(vis_l,n,dmrfl);
        grid->UpdateDataPtr(vis_l, REAL_FLOW, n,"vis_l");
    }

    IntType vis_mode, i;
    grid->GetData(&vis_mode,INT,1,"vis_mode");
    if( vis_mode==INVISCID ) {
        for(i=0; i<n; i++) vis_l[i] = 0.;
    }else{
        //RealFlow tref=273.0, sref=110.4, amuref=1.71e-5, temp;
        RealFlow tref=288.15, sref=110.4, amuref=1.78938e-5;
        // viscosity function -- Sutherland's Law
        RealFlow *t = cuGetTemperature(grid);
		
		//HANDLE_API_ERR(cudaMemcpy(t, gt, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
		
		IntType blocksPerGrid = (gnTCell + gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
		gpuComputeVis_l <<< blocksPerGrid, threadsPerBlock >>> (gvis_l, gt, tref, sref, amuref, gnTCell + gnBFace);
		
		blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
		gpuComputeVis_l2 <<< blocksPerGrid, threadsPerBlock >>> (gvis_l, gf2c, gtype_bcr, gtw_bcr, tref, sref, amuref, gnBFace);
		
		// HANDLE_API_ERR(cudaMemcpy(vis_l, gvis_l, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
        mfmem::sdel_array_1D(t);
    }
}

__global__ void gpuSetGhostVariables(RealFlow *rho, RealFlow *u, RealFlow *v, RealFlow *w, RealFlow *p, 
							const RealGeom *xfn, const RealGeom *yfn, const RealGeom *zfn, const RealGeom *vgn, 
							const IntType *f2c, const IntType *type_bcr, const RealGeom *tw_bcr, 
							const RealFlow rho_min, const RealFlow rho_max, const RealFlow p_min, const RealFlow p_max,
							const RealFlow rho00, const RealFlow u00, const RealFlow v00, const RealFlow w00, const RealFlow p00,
							const RealFlow norm_of_uvw, const RealFlow eps_of_farfield_vn, const RealFlow gam,
							const IntType vis_mode, const IntType steady, const RealFlow p_bar, const RealFlow gascon, 
							const IntType nBFace){							
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType type, c1, c2;
	RealFlow tw;
	RealFlow rho_c2, u_c2, v_c2, w_c2, p_c2;
	RealFlow vn, rhow;
	RealFlow vtx, vty, vtz, riemp, riemm;
	RealFlow rhom, um, vm, wm, pm, rhop, up, vp, wp, pp;
	RealFlow vnp, vnm, cp, cm, entr, vnb, cb;
	if(i < nBFace){
		type  = type_bcr[i];
        c1    = f2c[2*i];
        c2    = f2c[2*i + 1];
        // wmark = 0;
		if(type != INTERFACE){
			if(type == WALL){
				p_c2   = p[c1];
                rho_c2 = rho[c1];
                
                if(vis_mode == INVISCID){
                    vn    = 2.*(xfn[i]*u[c1] + yfn[i]*v[c1] + zfn[i]*w[c1]);
                    u_c2 = u[c1] - vn*xfn[i];
                    v_c2 = v[c1] - vn*yfn[i];
                    w_c2 = w[c1] - vn*zfn[i];
                }else{
                    if(steady){
                        u_c2 = -u[c1];
                        v_c2 = -v[c1];
                        w_c2 = -w[c1];
                    }else{
						/*
                        u[c2] = -u[c1] + 2.*BFacevgx[i];
                        v[c2] = -v[c1] + 2.*BFacevgy[i];
                        w[c2] = -w[c1] + 2.*BFacevgz[i];
						*/
                    }                   
                    //viscous iso-thermal wall
                    tw = -1.0;
                    tw = tw_bcr[i];
                    if(tw > 0.0){
                        rhow = (p_c2 + p_bar)/gascon/tw;
                        rho_c2 = 2.0*rhow-rho[c1];
                        if(rho_c2<0.0){
                            rho_c2 = rhow;
                        }
                    }
                }				
			} 
			else if (type == SYMM){
				rho_c2 = rho[c1];
                p_c2 = p[c1];
                vn = 2.*(xfn[i]*u[c1] + yfn[i]*v[c1] + zfn[i]*w[c1]);
                if(!steady){        //zhyb:对称面vgn为0，此处本来可以不考虑。但是在粘性计算时，有时可能会采用对称边界条件表示无粘的物面，
                    vn -= 2*vgn[i]; //因此在此需要加上非定常的情况
                }
                u_c2 = u[c1] - vn*xfn[i];
                v_c2 = v[c1] - vn*yfn[i];
                w_c2 = w[c1] - vn*zfn[i];
								
			}
			else{	// FAR_FIELD:
				um = u00;
                vm = v00;
                wm = w00;
                up = u[c1];
                vp = v[c1];
                wp = w[c1];
				/*
                if(!steady){
                    um -= BFacevgx[i];
                    vm -= BFacevgy[i];
                    wm -= BFacevgz[i];
                    up -= BFacevgx[i];
                    vp -= BFacevgy[i];
                    wp -= BFacevgz[i];
                }
				*/
                rhom = rho00;
                pm = p00 + p_bar;
                rhop = rho[c1];
                pp = p[c1]+p_bar;
                
                vnm = xfn[i]*um + yfn[i]*vm + zfn[i]*wm;
                vnp = xfn[i]*up + yfn[i]*vp + zfn[i]*wp;
                cm  = sqrt(gam*pm/rhom);
                cp  = sqrt(gam*pp/rhop);
                riemm = vnm - 2.*cm/(gam - 1.);
                riemp = vnp + 2.*cp/(gam - 1.);
                
                vnb = 0.5*(riemp + riemm);
                cb  = 0.25*(riemp - riemm)*(gam - 1.);
				
                if(fabs(vnb/cb)>1.){  //supersonic
                    if(vnb<=0.0){  //inlet
                        rho_c2 = rhom;
                        u_c2   = um;
                        v_c2   = vm;
                        w_c2   = wm;
                        p_c2   = pm;
                    }else{   //exit
                        rho_c2 = rhop;
                        u_c2   = up;
                        v_c2   = vp;
                        w_c2   = wp;
                        p_c2   = pp;
                    }
                }else{ //subsonic
                    RealFlow rela_vnb = vnb / norm_of_uvw;
                    if(rela_vnb <= -eps_of_farfield_vn) {  //inlet
                        vtx = um - vnm*xfn[i];
                        vty = vm - vnm*yfn[i];
                        vtz = wm - vnm*zfn[i];
                        entr = pm/pow(rhom, gam);

                        rho_c2 = pow((cb*cb/(entr*gam)), RealFlow(1./(gam-1.))); 
                        u_c2 = vtx + vnb*xfn[i];
                        v_c2 = vty + vnb*yfn[i];
                        w_c2 = vtz + vnb*zfn[i];
                        p_c2 = cb*cb*rho_c2/gam;
                    } else if(rela_vnb > eps_of_farfield_vn) {  //exit
                        vtx = up - vnp*xfn[i];
                        vty = vp - vnp*yfn[i];
                        vtz = wp - vnp*zfn[i];
                        entr = pp/pow(rhop, gam);

                        rho_c2 = pow((cb*cb/(entr*gam)),RealFlow(1./(gam-1.))); 
                        u_c2   = vtx + vnb*xfn[i];
                        v_c2   = vty + vnb*yfn[i];
                        w_c2   = vtz + vnb*zfn[i];
                        p_c2   = cb*cb*rho_c2/gam;
                    } else {
                        rho_c2 = 0.5*(rhop + rhom);
                        u_c2   = 0.5*(up + um);
                        v_c2   = 0.5*(vp + vm);
                        w_c2   = 0.5*(wp + wm);
                        p_c2   = 0.5*(pp + pm);
                    }
                    
                }
                
                rho_c2 = 2*rho_c2 - rhop;
                u_c2   = 2*u_c2 - up;
                v_c2   = 2*v_c2 - vp;
                w_c2   = 2*w_c2 - wp;
                p_c2   = 2*p_c2 - pp;
                p_c2  -= p_bar;
                /*
                if(!steady){
                    u[c2] += BFacevgx[i];
                    v[c2] += BFacevgy[i];
                    w[c2] += BFacevgz[i];
                }
				*/			
			}
			//ZHYB:对c2单元的rho和p做限制，不能为负，不能大于10倍的驻点值
			rho_c2 = GPUMAX2(rho_c2, rho_min);
			rho_c2 = GPUMIN2(rho_c2, rho_max);
			p_c2 = GPUMAX2(p_c2, p_min);
			p_c2 = GPUMIN2(p_c2, p_max);
			atomicExchSM35(rho + c2, rho_c2);
			atomicExchSM35(u + c2, u_c2);
			atomicExchSM35(v + c2, v_c2);
			atomicExchSM35(w + c2, w_c2);
			atomicExchSM35(p + c2, p_c2);
		}
	}
	
}

void cuSetGhostVariables(PolyGrid *grid){
	
    IntType  steady, vis_mode;

    RealFlow rho00, u00, v00, w00, p00;
    RealFlow gascon;  	
  
    grid->GetData(&rho00, REAL_FLOW, 1, "rho");
    grid->GetData(&u00, REAL_FLOW, 1, "u");
    grid->GetData(&v00, REAL_FLOW, 1, "v");
    grid->GetData(&w00, REAL_FLOW, 1, "w");
    grid->GetData(&p00, REAL_FLOW, 1, "p");
   
    RealFlow norm_of_uvw = sqrt( u00*u00 + v00*v00 + w00*w00 );
    RealFlow eps_of_farfield_vn = 0.0;
    grid->GetData(&eps_of_farfield_vn, REAL_FLOW, 1, "eps_of_farfield_vn",0);

    RealFlow rho_min,rho_max,p_min,p_max;
    grid->GetData(&rho_min, REAL_FLOW, 1, "rho_min");
    grid->GetData(&rho_max, REAL_FLOW, 1, "rho_max");
    grid->GetData(&p_min,   REAL_FLOW, 1, "p_min");
    grid->GetData(&p_max,   REAL_FLOW, 1, "p_max");
    
    grid->GetData(&steady,  INT, 1, "steady");
    grid->GetData(&vis_mode,INT, 1, "vis_mode");
    grid->GetData(&gascon,  REAL_FLOW, 1, "gascon"); 
    
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuSetGhostVariables <<< blocksPerGrid, threadsPerBlock >>> (gq, &gq[gnTCell + gnBFace], &gq[2*(gnTCell + gnBFace)], 
														&gq[3*(gnTCell + gnBFace)], &gq[4*(gnTCell + gnBFace)], 
														gxfn, gyfn, gzfn, gvgn, 
														gf2c, gtype_bcr, gtw_bcr, rho_min, rho_max, p_min, p_max, 
														rho00, u00, v00, w00, p00, norm_of_uvw, eps_of_farfield_vn, ggam, 
														vis_mode, gsteady, gp_bar, gascon, gnBFace);
    
    
}

__global__ void gpuSetGhostQuantityGradients(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, 
							const RealGeom *xfn, const RealGeom *yfn, const RealGeom *zfn,
							const IntType *f2c, const IntType *type_bcr, const IntType nTCell, const IntType nBFace){							
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType type, c1, c2;
	IntType Cell = nTCell + nBFace;	
	    
	RealFlow dqdx_1_c2, dqdx_2_c2, dqdx_3_c2;
	RealFlow dqdy_1_c2, dqdy_2_c2, dqdy_3_c2;
	RealFlow dqdz_1_c2, dqdz_2_c2, dqdz_3_c2;

	if(i < nBFace){
		type = type_bcr[i];
        c1 = f2c[2*i];
        c2 = f2c[2*i + 1];
        // wmark = 0;
		if(type != INTERFACE){
			if(type == WALL){
				RealFlow dta[3], dnn[3], dnnn;
				dnnn = 
                    dqdx[1*Cell + c1] * xfn[i]
                  + dqdy[1*Cell + c1] * yfn[i]
                  + dqdz[1*Cell + c1] * zfn[i];
                dnn[0] = dnnn * xfn[i];
                dnn[1] = dnnn * yfn[i];
                dnn[2] = dnnn * zfn[i];
                dta[0] = dqdx[1*Cell + c1] - dnn[0];
                dta[1] = dqdy[1*Cell + c1] - dnn[1];
                dta[2] = dqdz[1*Cell + c1] - dnn[2];
                dqdx_1_c2 = dnn[0] - dta[0];
                dqdy_1_c2 = dnn[1] - dta[1];
                dqdz_1_c2 = dnn[2] - dta[2];
                dnnn = 
                    dqdx[2*Cell + c1] * xfn[i]
                  + dqdy[2*Cell + c1] * yfn[i]
                  + dqdz[2*Cell + c1] * zfn[i];
                dnn[0] = dnnn * xfn[i];
                dnn[1] = dnnn * yfn[i];
                dnn[2] = dnnn * zfn[i];
                dta[0] = dqdx[2*Cell + c1] - dnn[0];
                dta[1] = dqdy[2*Cell + c1] - dnn[1];
                dta[2] = dqdz[2*Cell + c1] - dnn[2];
                dqdx_2_c2 = dnn[0] - dta[0];
                dqdy_2_c2 = dnn[1] - dta[1];
                dqdz_2_c2 = dnn[2] - dta[2];
                dnnn = 
                    dqdx[3*Cell + c1] * xfn[i]
                  + dqdy[3*Cell + c1] * yfn[i]
                  + dqdz[3*Cell + c1] * zfn[i];
                dnn[0] = dnnn * xfn[i];
                dnn[1] = dnnn * yfn[i];
                dnn[2] = dnnn * zfn[i];
                dta[0] = dqdx[3*Cell + c1] - dnn[0];
                dta[1] = dqdy[3*Cell + c1] - dnn[1];
                dta[2] = dqdz[3*Cell + c1] - dnn[2];
                dqdx_3_c2 = dnn[0] - dta[0];
                dqdy_3_c2 = dnn[1] - dta[1];
                dqdz_3_c2 = dnn[2] - dta[2];	
			} 
			else if (type == SYMM){
				RealFlow gv1[9], gv2[9];
				gv1[0*3 + 0] = dqdx[1*Cell + c1];
                gv1[1*3 + 0] = dqdx[2*Cell + c1];
                gv1[2*3 + 0] = dqdx[3*Cell + c1];
                gv1[0*3 + 1] = dqdy[1*Cell + c1];
                gv1[1*3 + 1] = dqdy[2*Cell + c1];
                gv1[2*3 + 1] = dqdy[3*Cell + c1];
                gv1[0*3 + 2] = dqdz[1*Cell + c1];
                gv1[1*3 + 2] = dqdz[2*Cell + c1];
                gv1[2*3 + 2] = dqdz[3*Cell + c1];
                GPUSolveEquationforGradSYMM(gv1, gv2, xfn[i], yfn[i], zfn[i]);
                dqdx_1_c2 = gv2[0*3 + 0];
                dqdx_2_c2 = gv2[1*3 + 0];
                dqdx_3_c2 = gv2[2*3 + 0];
                dqdy_1_c2 = gv2[0*3 + 1];
                dqdy_2_c2 = gv2[1*3 + 1];
                dqdy_3_c2 = gv2[2*3 + 1];
                dqdz_1_c2 = gv2[0*3 + 2];
                dqdz_2_c2 = gv2[1*3 + 2];
                dqdz_3_c2 = gv2[2*3 + 2];				
			}
			else if (type == FAR_FIELD){
				dqdx_1_c2 = 0.0;
                dqdx_2_c2 = 0.0;
                dqdx_3_c2 = 0.0;
                dqdy_1_c2 = 0.0;
                dqdy_2_c2 = 0.0;
                dqdy_3_c2 = 0.0;
                dqdz_1_c2 = 0.0;
                dqdz_2_c2 = 0.0;
                dqdz_3_c2 = 0.0;
			}
			else{	// default:
				dqdx_1_c2 = 0.0;
                dqdx_2_c2 = 0.0;
                dqdx_3_c2 = 0.0;
                dqdy_1_c2 = 0.0;
                dqdy_2_c2 = 0.0;
                dqdy_3_c2 = 0.0;
                dqdz_1_c2 = 0.0;
                dqdz_2_c2 = 0.0;
                dqdz_3_c2 = 0.0;
			}						
			atomicExchSM35(dqdx + 1*Cell + c2, dqdx_1_c2);
			atomicExchSM35(dqdy + 1*Cell + c2, dqdy_1_c2);
			atomicExchSM35(dqdz + 1*Cell + c2, dqdz_1_c2);
			atomicExchSM35(dqdx + 2*Cell + c2, dqdx_2_c2);
			atomicExchSM35(dqdy + 2*Cell + c2, dqdy_2_c2);
			atomicExchSM35(dqdz + 2*Cell + c2, dqdz_2_c2);
			atomicExchSM35(dqdx + 3*Cell + c2, dqdx_3_c2);
			atomicExchSM35(dqdy + 3*Cell + c2, dqdy_3_c2);
			atomicExchSM35(dqdz + 3*Cell + c2, dqdz_3_c2);
		}
	}
	
}

__global__ void gpuSolveADU3D(RealFlow *rhs, const RealFlow *DQ, const RealFlow *q, 
								const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, const RealFlow *vgn, 
								const RealFlow *area, const RealFlow *vis_l, const RealFlow *vis_t, 
								const RealGeom *norm_dist_c2c, 
								const IntType *C2F, const IntType *IndexC2F, const IntType *f2c, const IntType *nFPC, 
								const RealFlow gam, const RealFlow p_bar, const RealFlow lhs_omga, 
								const IntType vis_run, 
								const IntType steady, const IntType NumCell, const IntType NVar, const IntType nTCell
								){
	
	IntType cell = blockDim.x*blockIdx.x + threadIdx.x;
	if(cell < nTCell){
		for(IntType j=0; j<nFPC[cell]; j++){
			IntType   face, c1, c2, c_tmp, count;
			RealFlow  flux[5], q_loc[5], DQ_loc[5], visc, tmp;
			RealGeom  face_n[3], dist;
			face  = C2F[IndexC2F[cell] + j];
			count = face + face;
			c1    = f2c[count++];
			c2    = f2c[count];

			face_n[0] = xfn[face];
			face_n[1] = yfn[face];
			face_n[2] = zfn[face];
			if(c2 == cell){
				c_tmp = c1;
				c1    = c2;
				c2    = c_tmp;
				face_n[0] = -face_n[0];
				face_n[1] = -face_n[1];
				face_n[2] = -face_n[2];
			}
			//assert(c1 == cell);
			
			for(IntType i=0; i<5; i++){
				q_loc[i]  = q[i*NumCell + c2];
				DQ_loc[i] = DQ[i*NumCell + c2];
			}
			// Calculate everything (I call it Flux) in lower triangular
			if(steady){
				GPUFluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
			}else{
				//FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
			}
			
			if(vis_run){
				dist = norm_dist_c2c[face];
				visc = vis_l[c2] + vis_t[c2];
				tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
				for(IntType i=0; i<5; i++) flux[i] -= tmp*DQ_loc[i];
			}

			// Add Flux together
			tmp = 0.5*area[face];
			for(IntType i=0; i<5; i++) rhs[i*nTCell + cell] += tmp*flux[i];
			
		}								
	}
	
}

void cuSolveADU3D(PolyGrid *grid, RealFlow **rhs, RealFlow *DQ[5], IntType *nFPC, IntType **C2F, IntType level){
	
    RealFlow lhs_omga;
    grid->GetData(&lhs_omga,   REAL_FLOW, 1, "lhs_omga");
    IntType vis_mode, vis_run = 0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    if(vis_mode != INVISCID) 
    {
        vis_run = 1;
    }

	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;	
	gpuSolveADU3D <<< blocksPerGrid, threadsPerBlock >>> (gres, gDQ, gq, gxfn, gyfn, gzfn, gvgn, 
														garea, gvis_l, gvis_t, gnorm_dist_c2c, 
														gC2F, gIndexC2F, gf2c, gnFPC, 
														ggam, gp_bar, glhs_omga, vis_run, 
														gsteady, gnTCell + gnBFace, 5, gnTCell);	
    
}

__global__ void gpuScalarGMRESlimitdq(RealFlow *dq, const RealFlow *q, const RealFlow q_min, 
								const RealFlow DQ_limit, const IntType nT
								){
	
	IntType cell = blockDim.x*blockIdx.x + threadIdx.x;
	if(cell < nT){
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
			dq[cell] = GPUMAX2(dq[cell],q_min-q[cell]);
		}else if(DQ_limit == 4){
			RealFlow alph = q[cell]/(q[cell]+GPUMAX2(0.0,-dq[cell]));
			dq[cell] *= alph;
		}
		
	}
	
}

void cuScalarGMRESlimitdq(IntType DQ_limit, RealFlow q_min){
	
	IntType blocksPerGrid = (gnTCell + gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuScalarGMRESlimitdq <<< blocksPerGrid, threadsPerBlock >>> (gDQ, gsa_nu, q_min, DQ_limit, gnTCell + gnBFace);	
	
}

void cuSetGhostQuantityGradients(const PolyGrid *grid, RealFlow **dqdx, RealFlow **dqdy, RealFlow **dqdz){
	
    const IntType nTCell = grid->GetNTCell();
    const IntType nBFace = grid->GetNBFace();

    const IntType *f2c = grid->Getf2c();
    const RealGeom *xfn = grid->GetXfn();
    const RealGeom *yfn = grid->GetYfn();
    const RealGeom *zfn = grid->GetZfn();
    const RealGeom *xcc = grid->GetXcc();
    const RealGeom *ycc = grid->GetYcc();
    const RealGeom *zcc = grid->GetZcc();
    const BCRecord **bcr = const_cast<const BCRecord **>(grid->Getbcr());
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuSetGhostQuantityGradients <<< blocksPerGrid, threadsPerBlock >>> (gdqdx, gdqdy, gdqdz, gxfn, gyfn, gzfn, 														
														gf2c, gtype_bcr, gnTCell, gnBFace);
	
}

__device__ void GPUSolveEquationforGradSYMM(RealFlow gv1[9], RealFlow gv2[9], RealGeom xfn, RealGeom yfn, RealGeom zfn){
	
    RealGeom dtmp;
    RealGeom xft1,yft1,zft1,xft2,yft2,zft2;
    RealFlow gradvn1[3],gradvt11[3],gradvt21[3],gradvn2[3],gradvt12[3],gradvt22[3];         
    // Get first tangential vector on the face
    if(xfn != 0.) {
        xft1 =  yfn;
        yft1 = -xfn;
        zft1 =  0.;
    } else if(yfn != 0.) {
        xft1 = -yfn;
        yft1 =  xfn;
        zft1 =  0.;
    } else if(zfn != 0.) {
        xft1 =  0.;
        yft1 = -zfn;
        zft1 =  yfn;
    } 
    // normalize the tangential vector
    dtmp = sqrt(xft1*xft1 + yft1*yft1 + zft1*zft1);
    xft1 /= dtmp;
    yft1 /= dtmp;
    zft1 /= dtmp;
    
    // Get second tangential vector by cross dot t1 to normal
    xft2 = yfn*zft1 - zfn*yft1;
    yft2 = zfn*xft1 - xfn*zft1;
    zft2 = xfn*yft1 - yfn*xft1;
    
    gradvn1[0]  = gv1[0*3 + 0]*xfn+gv1[1*3 + 0]*yfn+gv1[2*3 + 0]*zfn;
    gradvn1[1]  = gv1[0*3 + 1]*xfn+gv1[1*3 + 1]*yfn+gv1[2*3 + 1]*zfn;
    gradvn1[2]  = gv1[0*3 + 2]*xfn+gv1[1*3 + 2]*yfn+gv1[2*3 + 2]*zfn;
    gradvt11[0] = gv1[0*3 + 0]*xft1+gv1[1*3 + 0]*yft1+gv1[2*3 + 0]*zft1;
    gradvt11[1] = gv1[0*3 + 1]*xft1+gv1[1*3 + 1]*yft1+gv1[2*3 + 1]*zft1;
    gradvt11[2] = gv1[0*3 + 2]*xft1+gv1[1*3 + 2]*yft1+gv1[2*3 + 2]*zft1;
    gradvt21[0] = gv1[0*3 + 0]*xft2+gv1[1*3 + 0]*yft2+gv1[2*3 + 0]*zft2;
    gradvt21[1] = gv1[0*3 + 1]*xft2+gv1[1*3 + 1]*yft2+gv1[2*3 + 1]*zft2;
    gradvt21[2] = gv1[0*3 + 2]*xft2+gv1[1*3 + 2]*yft2+gv1[2*3 + 2]*zft2;
    dtmp = gradvn1[0]*xfn+gradvn1[1]*yfn+gradvn1[2]*zfn;
    gradvn2[0]  = 2.0*dtmp*xfn-gradvn1[0];
    gradvn2[1]  = 2.0*dtmp*yfn-gradvn1[1];
    gradvn2[2]  = 2.0*dtmp*zfn-gradvn1[2];
    dtmp = gradvt11[0]*xfn+gradvt11[1]*yfn+gradvt11[2]*zfn;
    gradvt12[0]  = gradvt11[0]-2.0*dtmp*xfn;
    gradvt12[1]  = gradvt11[1]-2.0*dtmp*yfn;
    gradvt12[2]  = gradvt11[2]-2.0*dtmp*zfn;
    dtmp = gradvt21[0]*xfn+gradvt21[1]*yfn+gradvt21[2]*zfn;
    gradvt22[0]  = gradvt21[0]-2.0*dtmp*xfn;
    gradvt22[1]  = gradvt21[1]-2.0*dtmp*yfn;
    gradvt22[2]  = gradvt21[2]-2.0*dtmp*zfn;
    
    gv2[0*3 + 0] = xfn*gradvn2[0]+xft1*gradvt12[0]+xft2*gradvt22[0];
    gv2[1*3 + 0] = yfn*gradvn2[0]+yft1*gradvt12[0]+yft2*gradvt22[0];
    gv2[2*3 + 0] = zfn*gradvn2[0]+zft1*gradvt12[0]+zft2*gradvt22[0];
    gv2[0*3 + 1] = xfn*gradvn2[1]+xft1*gradvt12[1]+xft2*gradvt22[1];
    gv2[1*3 + 1] = yfn*gradvn2[1]+yft1*gradvt12[1]+yft2*gradvt22[1];
    gv2[2*3 + 1] = zfn*gradvn2[1]+zft1*gradvt12[1]+zft2*gradvt22[1];
    gv2[0*3 + 2] = xfn*gradvn2[2]+xft1*gradvt12[2]+xft2*gradvt22[2];
    gv2[1*3 + 2] = yfn*gradvn2[2]+yft1*gradvt12[2]+yft2*gradvt22[2];
    gv2[2*3 + 2] = zfn*gradvn2[2]+zft1*gradvt12[2]+zft2*gradvt22[2];

}

__device__ double atomicExchSM35(double* address, double val){
	
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
        } while (assumed != old);
        return __longlong_as_double(old);
}

__device__ void GPUFluxLUSGS3D(RealFlow flux[5], RealFlow q[5], RealFlow DQ[5], RealGeom fa_n[3], 
							RealFlow gam, RealFlow p_bar, RealFlow lhs_omga){
								
	IntType i;
    RealFlow Q[5], rv2, v_n, p, peff, c2, eig, gam1 = gam - 1.;
    RealGeom nx, ny, nz;

    ///
    //RealFlow norm_of_dq = -1;
    //for(i=0; i<5; i++) if( abs(DQ[i]) > norm_of_dq ) norm_of_dq = abs(DQ[i]);

    //for(i=0; i<5; i++) DQ[i] *= 1.0e-5 / (norm_of_dq + TINY);
    ///

    nx   = fa_n[0];
    ny   = fa_n[1];
    nz   = fa_n[2];
        
    Q[0] = q[0];
    Q[1] = q[0]*q[1];
    Q[2] = q[0]*q[2];
    Q[3] = q[0]*q[3];
    rv2  = 0.5*q[0]*(q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    p    = q[4];
    Q[4] = p/gam1 + rv2;
    
    // Normal velocity and Eigenvalues
    v_n  = q[1]*nx + q[2]*ny + q[3]*nz;
    c2   = gam*(p + p_bar)/q[0];
    eig  = fabs(v_n) + sqrt(c2);
    eig *= lhs_omga;
    
    // Need to find out the fluxes on level n
    peff = gam*p_bar/gam1;
    flux[1] = -Q[1]*v_n - p*nx;
    flux[2] = -Q[2]*v_n - p*ny;
    flux[3] = -Q[3]*v_n - p*nz;
    flux[4] = -(Q[4] + p + peff)*v_n;
    
    // Conservative variable on level n+1
    for(i=0; i<5; i++) Q[i] += DQ[i];
    rv2 = 0.5*(Q[1]*Q[1] + Q[2]*Q[2] + Q[3]*Q[3])/Q[0];
    p   = gam1*(Q[4] - rv2);
    
    // Now the flux difference due to DQ
    flux[0]  = DQ[1]*nx + DQ[2]*ny + DQ[3]*nz;
    v_n     *= q[0];
    v_n     += flux[0];
    v_n     /= Q[0];
    flux[1] += Q[1]*v_n + p*nx;
    flux[2] += Q[2]*v_n + p*ny;
    flux[3] += Q[3]*v_n + p*nz;
    flux[4] +=(Q[4] + p + peff)*v_n;
    
    ///
    //for(i=0; i<5; i++) DQ[i] /= 1.0e-5 / (norm_of_dq + TINY);
    //for(i=0; i<5; i++) flux[i] /= 1.0e-5 / (norm_of_dq + TINY);
    ///

    // Subtract eigenvalue terms from the flux difference
    for(i=0; i<5; i++) flux[i] -= eig*DQ[i];
    
    // Note: We do not check Q[0] and p here, because they are used to
    // calculate sound speed in LUSGS. Check them later in UpdateFlowField							
								
}			

__device__ double GPUMIN2(double a, double b)
{
        return(a>b?b:a);
}
__device__ double GPUMAX2(double a, double b)
{
        return(a>b?a:b);
}					
