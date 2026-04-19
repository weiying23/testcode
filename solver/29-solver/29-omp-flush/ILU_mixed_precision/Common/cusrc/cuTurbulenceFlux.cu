#include <stdio.h>
#include <iostream>
#include <fstream>

#include "turbulence.h"

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

#include <cuGradientQ_Gauss.cuh>
#include <cuTurbulenceFlux.cuh>
#include <cuData.cuh>
#include <cuErrorReturn.cuh>

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

__global__ void gpuSetQlQrUseQ(const RealFlow* q, RealFlow* ql, RealFlow* qr, const IntType* f2c, 
							const IntType nTFace){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nTFace){
		IntType  c1, c2, count;
		count = 2 * i;
        c1 = f2c[count++];
        c2 = f2c[count];
		ql[i] = q[c1]; //ql[0][i] ql[1][i]
		qr[i] = q[c2];
	}	
}

void cuSetQlQrUseQ(PolyGrid *grid, IntType name){
	
	IntType blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
    if (name < 4){
		gpuSetQlQrUseQ <<< blocksPerGrid, threadsPerBlock >>> (&gq[name*(gnTCell + gnBFace)], &gql[name*gnTFace], &gqr[name*gnTFace], 
															gf2c, gnTFace);				
	}
	else{	
		//HANDLE_API_ERR(cudaMemcpy(gsa_nu, q, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
		gpuSetQlQrUseQ <<< blocksPerGrid, threadsPerBlock >>> (gsa_nu, &gql[name*gnTFace], &gqr[name*gnTFace], 
															gf2c, gnTFace);
	}
	// tmp transfer:
	//HANDLE_API_ERR(cudaMemcpy(ql, &gql[name*gnTFace], gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(qr, &gqr[name*gnTFace], gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));		    
		
}

__global__ void gpuCalcuQlQr_turb(RealFlow* ql, RealFlow* qr, const RealFlow *dnutdx, const RealFlow *dnutdy, 
							const RealFlow *dnutdz, const RealFlow *xfc, const RealFlow *yfc, const RealFlow *zfc, 
							const RealFlow *xcc, const RealFlow *ycc, const RealFlow *zcc, const IntType *type_bcr,
							const IntType *f2c, const IntType nBFace){
								
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nBFace){
		IntType  c1, c2, count, type;
        RealGeom dx, dy, dz;
        RealFlow tk;
        type = type_bcr[i];
        count = 2*i;
        c1     = f2c[count];
        c2     = f2c[count + 1];
        
        // Left one
        dx = xfc[i] - xcc[c1];
        dy = yfc[i] - ycc[c1];
        dz = zfc[i] - zcc[c1];
        tk = ql[i] + (dnutdx[c1]*dx + dnutdy[c1]*dy + dnutdz[c1]*dz);
        if(tk > TINY) ql[i]  = tk;
        
        if (type == INTERFACE){
            // Right one
            dx = xfc[i] - xcc[c2];
            dy = yfc[i] - ycc[c2];
            dz = zfc[i] - zcc[c2];
            tk = qr[i] + (dnutdx[c2]*dx + dnutdy[c2]*dy + dnutdz[c2]*dz);
            if(tk > TINY) qr[i]  = tk;
        }
	}	
}

__global__ void gpuCalcuQlQr_turb2(RealFlow* ql, RealFlow* qr, const RealFlow *dnutdx, const RealFlow *dnutdy, 
							const RealFlow *dnutdz, const RealFlow *xfc, const RealFlow *yfc, const RealFlow *zfc, 
							const RealFlow *xcc, const RealFlow *ycc, const RealFlow *zcc,
							const IntType *f2c, const IntType nBFace, const IntType nTFace){
								
	IntType i = nBFace + blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nTFace){
		IntType  c1, c2, count;
        RealGeom dx, dy, dz;
        RealFlow tk;
        count = 2*i;
        c1     = f2c[count];
        c2     = f2c[count + 1];
        
        // Left one
        dx     = xfc[i] - xcc[c1];
        dy     = yfc[i] - ycc[c1];
        dz     = zfc[i] - zcc[c1];
        tk     = ql[i] + (dnutdx[c1]*dx + dnutdy[c1]*dy + dnutdz[c1]*dz);
        if(tk > TINY) ql[i]  = tk;
        
        // Right one
        dx     = xfc[i] - xcc[c2];
        dy     = yfc[i] - ycc[c2];
        dz     = zfc[i] - zcc[c2];
        tk     = qr[i] + (dnutdx[c2]*dx + dnutdy[c2]*dy + dnutdz[c2]*dz);
        if(tk > TINY) qr[i]  = tk;
	}	
}

void cuCalcuQlQr_turb(PolyGrid *grid, IntType ns, IntType ne, const char *name){
        
    // boundary faces:
    IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCalcuQlQr_turb <<< blocksPerGrid, threadsPerBlock >>> (&gql[4*gnTFace], &gqr[4*gnTFace], gdnutdx, gdnutdy, gdnutdz,
															gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gtype_bcr, 
															gf2c, gnBFace);		
    
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCalcuQlQr_turb2 <<< blocksPerGrid, threadsPerBlock >>> (&gql[4*gnTFace], &gqr[4*gnTFace], gdnutdx, gdnutdy, gdnutdz,
															gxfc, gyfc, gzfc, gxcc, gycc, gzcc, 
															gf2c, gnBFace, gnTFace);	
	//HANDLE_API_ERR(cudaMemcpy(ql, &gql[4*gnTFace], gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(qr, &gqr[4*gnTFace], gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpuModQlQrBou_turb(const RealFlow* q, RealFlow* ql, RealFlow* qr, const IntType* type_bcr, 
								const IntType* f2c, const IntType nBFace, const IntType name){
									
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nBFace){
		IntType c1, c2, type, count;
        RealFlow temm;
        type = type_bcr[i];
        count = 2*i;
        if (type != INTERFACE){ 
			c1 = f2c[count];
			c2 = f2c[count+1];
						
			temm = (q[c1] + q[c2])*0.5;
			ql[i] = temm;
			qr[i] = temm;

		}
	}	
}

__global__ void gpuModQlQrBou_turb2(const RealFlow* q, RealFlow* ql, RealFlow* qr, const IntType* type_bcr, 
								const IntType* f2c, const IntType nBFace, const IntType name){
									
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nBFace){
		IntType c1, c2, type, count;
        RealFlow temm;
		type = type_bcr[i];
        count = 2*i;
        type = type_bcr[i];
        if (type != INTERFACE){ 
			if(type == SYMM){
				qr[i] = ql[i];
			}
			else{
				c1 = f2c[count];
				c2 = f2c[count+1];
							
				temm = (q[c1] + q[c2])*0.5;
				ql[i] = temm;
				qr[i] = temm;
			}
		}
	}		
}

void cuModQlQrBou_turb(PolyGrid *grid, IntType ns, IntType ne, IntType name){
        
    // boundary faces:
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	if (name < 4){
		gpuModQlQrBou_turb <<< blocksPerGrid, threadsPerBlock >>> (&gq[name*(gnTCell + gnBFace)], &gql[name*gnTFace], &gqr[name*gnTFace], 
															gtype_bcr, gf2c, gnBFace, name);				
	}
	else{	// name = 4 :
		gpuModQlQrBou_turb2 <<< blocksPerGrid, threadsPerBlock >>> (gsa_nu, &gql[name*gnTFace], &gqr[name*gnTFace], 
															gtype_bcr, gf2c, gnBFace, name);
	}
	//HANDLE_API_ERR(cudaMemcpy(ql, &gql[name*gnTFace], gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(qr, &gqr[name*gnTFace], gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

__global__ void gpuScalarFlux(RealFlow* flux, RealFlow* dqdl, RealFlow* dqdr, const RealFlow* ql, const RealFlow* qr,
							const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, const RealGeom *area,
							const RealGeom *vgn, const IntType steady, const IntType nTFace){							
								
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nTFace){
		RealFlow vnl, vnr;
        vnl = ql[1*nTFace + i]*xfn[i] + ql[2*nTFace + i]*yfn[i] + ql[3*nTFace + i]*zfn[i];
        if(!steady) vnl -= vgn[i];

        if(vnl > 0.) {
            dqdl[i] = ql[i]*vnl*area[i];
            flux[i] = ql[4*nTFace + i]*dqdl[i];
        } else {
            dqdl[i] = 0.;
            flux[i] = 0.;
        }
 
        vnr = qr[1*nTFace + i]*xfn[i] + qr[2*nTFace + i]*yfn[i] + qr[3*nTFace + i]*zfn[i];
        if(!steady) vnr -= vgn[i];
        if(vnr < 0.) {
            dqdr[i]  = qr[i]*vnr*area[i];
            flux[i] += qr[4*nTFace + i]*dqdr[i];
        } else {
            dqdr[i] = 0.;
        }
	}	
}

void cuScalarFlux(){
     
    // velocity based upwind
	IntType blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuScalarFlux <<< blocksPerGrid, threadsPerBlock >>> (gflux, gdqdl, gdqdr, gql, gqr, gxfn, gyfn, gzfn, 
														garea, gvgn, gsteady, gnTFace);	
	//HANDLE_API_ERR(cudaMemcpy(dqdl, gdqdl, gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(dqdr, gdqdr, gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//HANDLE_API_ERR(cudaMemcpy(flux, gflux, gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	
}

__global__ void gpuSALoadFluxReduction(RealFlow* res, const RealFlow* flux, const IntType* C2F, const IntType* IndexC2F, 
						const IntType* nFPC, const IntType* f2c, const IntType nTFace, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face;
	if(i < nTCell){
		//res[i] = 0;
		for(IntType j = 0; j < nFPC[i]; j++){
			face = C2F[IndexC2F[i] + j];
			c1 = f2c[2*face];
			c2 = f2c[2*face + 1];
			if (i == c1) {
                res[i] -= flux[face];
            }
            else if (i == c2) {
                res[i] += flux[face];
            }			
		}		
	}
	
}

#if (defined ShareMemory)
__global__ void gpuSALoadFluxReductionShareMemory(RealFlow* res, const RealFlow* flux, const IntType* C2F, const IntType* IndexC2F, 
						const IntType* nFPC, const IntType* f2c, const IntType nTFace, const IntType nTCell){
	extern __shared__ double sdata[];
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face;
	
	if(i < nTCell){
		sdata[threadIdx.x] = res[i];
	}
	__syncthreads();
	
	if(i < nTCell){
		//res[i] = 0;
		for(IntType j = 0; j < nFPC[i]; j++){
			face = C2F[IndexC2F[i] + j];
			c1 = f2c[2*face];
			c2 = f2c[2*face + 1];
			if (i == c1) {
                sdata[threadIdx.x] -= flux[face];
            }
            else if (i == c2) {
                sdata[threadIdx.x] += flux[face];
            }			
		}		
	}
	__syncthreads();
	
	if(i < nTCell){
		res[i] = sdata[threadIdx.x];
	}
	
}	
#endif

void cuLoadFlux(PolyGrid *grid, IntType nVar, IntType ns, IntType ne){
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
#if (defined ShareMemory)
	gpuSALoadFluxReductionShareMemory <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gres, gflux, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell);
#else
	gpuSALoadFluxReduction <<< blocksPerGrid, threadsPerBlock >>> (gres, gflux, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell);
#endif

	//HANDLE_API_ERR(cudaMemcpy(res[0], gres, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));

}

__global__ void gpuPutScalarDqToLhsReduction(RealFlow* lhsmat, const RealFlow* dqdl, const RealFlow* dqdr, const IntType* C2F, 
									const IntType* IndexC2F, const IntType* nFPC, const IntType* nCPC, const IntType* IndexC2C, 
									const IntType* f2c, const IntType* fcptr, const IntType nTFace, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face, count, nc1, nc2;
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			face = C2F[IndexC2F[i] + j];
			count = 2 * face;
			nc1 = fcptr[count];
			c1 = f2c[count++];
			nc2 = fcptr[count];
			c2 = f2c[count];
			if (i == c1) {
				lhsmat[i + IndexC2C[i] + 0] += dqdl[face];
				if (nc1 > 0) lhsmat[i + IndexC2C[i] + nc1] += dqdr[face];
			}
			else if (i == c2) {
				lhsmat[i + IndexC2C[i] + 0] -= dqdr[face];
				if (nc2 > 0) lhsmat[i + IndexC2C[i] + nc2] -= dqdl[face];
			}
		}		
	}
	
}

/* #if (defined ShareMemory)
__global__ void gpuPutScalarDqToLhsReductionShareMemory(RealFlow* lhsmat, const RealFlow* dqdl, const RealFlow* dqdr, const IntType* C2F, 
									const IntType* IndexC2F, const IntType* nFPC, const IntType* nCPC, const IntType* IndexC2C, 
									const IntType* f2c, const IntType* fcptr, const IntType nTFace, const IntType nTCell){
	extern __shared__ double sdata[];
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face, count, nc1, nc2;
	
	if(i < nTCell){
		sdata[threadIdx.x] = lhsmat[i + IndexC2C[i] + 0];
	}
	__syncthreads();
	
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			face = C2F[IndexC2F[i] + j];
			count = 2 * face;
			nc1 = fcptr[count];
			c1 = f2c[count++];
			nc2 = fcptr[count];
			c2 = f2c[count];
			if (i == c1) {
				sdata[threadIdx.x] += dqdl[face];
				if (nc1 > 0) lhsmat[i + IndexC2C[i] + nc1] += dqdr[face];
			}
			else if (i == c2) {
				lhsmat[i + IndexC2C[i] + 0] -= dqdr[face];
				if (nc2 > 0) lhsmat[i + IndexC2C[i] + nc2] -= dqdl[face];
			}
		}		
	}
	__syncthreads();
	
	if(i < nTCell){
		sdata[threadIdx.x] = lhsmat[i + IndexC2C[i] + 0];
	}
	
}	
#endif */

void cuPutScalarDqToLhs(PolyGrid *grid, IntType ns, IntType ne){
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
/* #if (defined ShareMemory)
	gpuPutScalarDqToLhsReductionShareMemory <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (glhsmat, gdqdl, gdqdr, gC2F, gIndexC2F, gnFPC, 
																gnCPC, gIndexC2C, gf2c, gfcptr, gnTFace, gnTCell);
#else */
	gpuPutScalarDqToLhsReduction <<< blocksPerGrid, threadsPerBlock >>> (glhsmat, gdqdl, gdqdr, gC2F, gIndexC2F, gnFPC, 
																gnCPC, gIndexC2C, gf2c, gfcptr, gnTFace, gnTCell);
/* #endif */
	//HANDLE_API_ERR(cudaMemcpy(lhsmat[0], glhsmat, (gnTCell + glenC2C)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

void cuInviscidFluxScalar(PolyGrid *grid, const char *name){
	
    IntType i, ns, ne, nVar;
    
    IntType turb_order = 1;
    grid->GetData(&turb_order, INT, 1, "turb_order");
    if(turb_order ==2){
        IntType iter_done=1;
        grid->GetData(&iter_done, INT, 1 ,"iter_done");
        if(iter_done<2000) turb_order = 1;       //前2000步用一阶
    }
 
    nVar=5;
    
    ns = 0;
    do {
        ne   = ns + gnTFace;
        // Get left variables and right variables
        for(i=0; i<nVar; i++){
            cuSetQlQrUseQ(grid, i);			
            if(turb_order==2 && i==nVar-1){
                cuCalcuQlQr_turb(grid, ns, ne, name);
            }
            cuModQlQrBou_turb(grid, ns, ne, i);
        }
                
        cuScalarFlux();
        
        // Load the fluxes to residuals
        cuLoadFlux(grid, 1, ns, ne);
        
        // Put Dq to the LHS matrices
        cuPutScalarDqToLhs(grid, ns, ne);
        
        ns  = ne;
    } while (ns < gnTFace);

} 

__global__ void gpuViscousFluxScalar(RealFlow *flux, RealFlow *tem, RealFlow *tem_c2, const RealFlow *k, const RealFlow *rho,
							const RealFlow *vis_l, const RealFlow *xcc, const RealFlow *ycc, const RealFlow *zcc, 
							const RealFlow *xfc, const RealFlow *yfc, const RealFlow *zfc, 
							const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, const IntType *f2c, 
							const IntType *type_bcr, const RealGeom *area, RealGeom *angle_h, const RealFlow sigma,
							const IntType TurM, const IntType nBFace, const IntType nTFace){	
																				
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nTFace){
		IntType  type;  
        RealFlow k_vis;
        IntType c1 = f2c[2*i];
        IntType c2 = f2c[2*i+1];
        RealGeom dtmp, d1, d2, x1, x2, y1, y2, z1, z2, angle1, angle2;  
        RealFlow k1, k2, kmid, dkd1, dkd2, dkdn;
        // positions
        x1 = xcc[c1]  - xfc[i];
        y1 = ycc[c1]  - yfc[i];
        z1 = zcc[c1]  - zfc[i];
        x2 = xcc[c2]  - xfc[i];
        y2 = ycc[c2]  - yfc[i];
        z2 = zcc[c2]  - zfc[i];
        d1 = x1*xfn[i] + y1*yfn[i] + z1*zfn[i];
        d2 = x2*xfn[i] + y2*yfn[i] + z2*zfn[i];
		
		/*
        dtmp = -d1/(sqrt(x1*x1+ y1*y1 + z1*z1) + TINY);
        if(dtmp >  1.0) dtmp =  1.0;
        if(dtmp < -1.0) dtmp = -1.0;
        angle1 = asin(dtmp)*180.0/PI;
		*/
		angle1 = angle_h[2*i];	
		
		/*
        dtmp = d2/(sqrt(x2*x2+ y2*y2 + z2*z2) + TINY);
        if(dtmp >  1.0) dtmp =  1.0;
        if(dtmp < -1.0) dtmp = -1.0;
        angle2 = asin(dtmp)*180.0/PI; 
		*/
		angle2 = angle_h[2*i + 1];	
      
        // quantities at points 1 and 2
        k1   = k[c1];  
        k2   = k[c2];
        kmid = 0.5*(k1 + k2);   
          
        dkdn  = 0.0;
        if(angle1 > 0.0 && angle2 > 0.0 && fabs(d1) > TINY && fabs(d2) > TINY) {
            dkd1 = (k1 - kmid)/d1;
            dkd2 = (k2 - kmid)/d2;

            dtmp = d1*d1 + d2*d2;
            d1   = d1*d1/dtmp;
            d2   = d2*d2/dtmp;
            dkdn = dkd1*d1 + dkd2*d2; 
        }
		

        if(i<nBFace){
            type = type_bcr[i];
            if (type!=INTERFACE && type!=SYMM) {
                RealFlow dn = (xcc[c2]-xcc[c1])*xfn[i]+(ycc[c2]-ycc[c1])*yfn[i]
                + (zcc[c2]-zcc[c1])*zfn[i];
                dkdn = (k[c2]-k[c1])/dn;
            } 
        }  
		
		//RealFlow tmpvar, tmpvarc1, tmpvarc2;
        if(TurM == 1){
            //k_vis = 0.5*sigma*(vis_l[c1]+vis_l[c2]+rho[c1]*k[c1]+rho[c2]*k[c2]);
			//tmpvarc1 = rho[c1]*k[c1];
			//tmpvarc2 = rho[c2]*k[c2];
			//tmpvar = rho[c1]*k[c1]+rho[c2]*k[c2];
            k_vis = 0.5*sigma*(vis_l[c1]+vis_l[c2]+(1.0+CB2)*(rho[c1]*k[c1]+rho[c2]*k[c2]));
		}
        tem[i] = dkdn*area[i];
        flux[i]     = k_vis*tem[i]; 
		//outputdkdn[i] = flux[i];		outputdkdnc1[i] = tmpvarc1;		outputdkdnc2[i] = tmpvarc2;
	}
}


__global__ void gpuViscousFluxScalar2(RealFlow *tem, RealFlow *tem_c2, const RealFlow *k, const RealFlow *rho,
							const IntType *f2c, const RealFlow sigma,
							const IntType nBFace, const IntType nTFace){	
																				
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nTFace){
		RealFlow factor_c1, factor_c2;
            IntType  c1 = f2c[i+i];
            IntType  c2 = f2c[i+i+1];
            tem[i] = -CB2*sigma*tem[i];
            factor_c1 = rho[c1]*k[c1];            
            if(i >= nBFace) {
                factor_c2 = rho[c2]*k[c2];
                tem_c2[i] = tem[i]*factor_c2;
            }
        tem[i] = tem[i]*factor_c1;				
	}
}	

__global__ void gpuViscousFluxScalar3Reduction(RealFlow* res, const RealFlow* flux, const RealFlow* tem, const RealFlow* tem_c2, 
											const IntType* C2F, const IntType* IndexC2F, const IntType* nFPC, 
											const IntType* f2c, const IntType nTFace, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face;
	RealFlow factor_c1, factor_c2;
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			face = C2F[IndexC2F[i] + j];
			c1 = f2c[2*face];
			c2 = f2c[2*face + 1];
			if (i == c1) {
                factor_c1 = flux[face] + tem[face];
                res[i] += factor_c1;
            }
            else if (i == c2) {
                factor_c2 = flux[face] + tem_c2[face];
                res[i] -= factor_c2;
            }			
		}		
	}
	
}

#if (defined ShareMemory)
__global__ void gpuViscousFluxScalar3ReductionShareMemory(RealFlow* res, const RealFlow* flux, const RealFlow* tem, const RealFlow* tem_c2, 
											const IntType* C2F, const IntType* IndexC2F, const IntType* nFPC, 
											const IntType* f2c, const IntType nTFace, const IntType nTCell){
	extern __shared__ double sdata[];		
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face;
	RealFlow factor_c1, factor_c2;
	
	if(i < nTCell){
		sdata[threadIdx.x] = res[i];
	}
	__syncthreads();
	
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			face = C2F[IndexC2F[i] + j];
			c1 = f2c[2*face];
			c2 = f2c[2*face + 1];
			if (i == c1) {
                factor_c1 = flux[face] + tem[face];
                sdata[threadIdx.x] += factor_c1;
            }
            else if (i == c2) {
                factor_c2 = flux[face] + tem_c2[face];
                sdata[threadIdx.x] -= factor_c2;
            }			
		}		
	}
	__syncthreads();	
	
	if(i < nTCell){
		res[i] = sdata[threadIdx.x];
	}	
}
#endif

void cuViscousFluxScalar3D_New3(PolyGrid *grid, const char *name){	
  
    RealFlow rhoP,amu,ainf;
    grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
    grid->GetData(&amu,   REAL_FLOW, 1, "amu");
    grid->GetData(&ainf,  REAL_FLOW, 1, "ainf");
 
    RealFlow q_min;
    RealFlow sigma;
 
    if(strcmp(name,"sa_nu") == 0){
        sigma = 1.0/SIGMA_SA; 
        q_min = MIN_SA_NU;
        q_min *= (amu/rhoP);
    }
	IntType TurM = 0;
	if(strcmp(name,"sa_nu") == 0) TurM = 1;
	
	// HANDLE_API_ERR(cudaMemcpy(&gq[gnTCell], &rho[gnTCell], gnBFace*sizeof(RealFlow), cudaMemcpyHostToDevice));
	// HANDLE_API_ERR(cudaMemcpy(gvis_l, vis_l, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	IntType blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
	
	gpuViscousFluxScalar <<< blocksPerGrid, threadsPerBlock >>> (gflux, gtem, gtem_c2, gsa_nu, gq, gvis_l, 
														gxcc, gycc, gzcc, gxfc, gyfc, gzfc, gxfn, gyfn, gzfn, 
														gf2c, gtype_bcr, garea, gangle_h, sigma, TurM, gnBFace, gnTFace);																	
	
    if(strcmp(name,"sa_nu") == 0){
        gpuViscousFluxScalar2 <<< blocksPerGrid, threadsPerBlock >>> (gtem, gtem_c2, gsa_nu, gq, gf2c, 
														sigma, gnBFace, gnTFace);			
    }	
	
	blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
#if (defined ShareMemory)
	gpuViscousFluxScalar3ReductionShareMemory <<< blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(RealFlow)>>> (gres, gflux, gtem, gtem_c2, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell);
#else
	gpuViscousFluxScalar3Reduction <<< blocksPerGrid, threadsPerBlock >>> (gres, gflux, gtem, gtem_c2, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell);
#endif

	//HANDLE_API_ERR(cudaMemcpy(res, gres, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
    
}

__global__ void gpuViscousDqScalar(RealFlow* dqdl, RealFlow* dqdr, const RealFlow *rho, const RealFlow *k, const RealFlow *vis_l, 
							const RealFlow *xfc, const RealFlow *yfc, const RealFlow *zfc,
							const RealFlow *xcc, const RealFlow *ycc, const RealFlow *zcc,
							const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, const RealGeom *area,
							const IntType *f2c, const RealFlow sigma, const IntType TurM, const IntType nTFace){							
								
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nTFace){
		RealGeom d1, d2, dtmp;
        RealFlow k_vis;
        IntType  c1, c2, count;
        count = 2*i;  
        c1      = f2c[count];
        c2      = f2c[count+1];
 
        d1 = (xcc[c1] - xfc[i])*xfn[i] + (ycc[c1] - yfc[i])*yfn[i] + (zcc[c1] - zfc[i])*zfn[i];
        d2 = (xcc[c2] - xfc[i])*xfn[i] + (ycc[c2] - yfc[i])*yfn[i] + (zcc[c2] - zfc[i])*zfn[i];
      
        dtmp = 0.0;
        if(d1*d2 < 0 && fabs(d1) > TINY && fabs(d2) > TINY) {
            dtmp = 0.5*fabs(d2 - d1)/(d1*d1 + d2*d2);
        }
     
        if(TurM == 1){
            k_vis = 0.5*sigma*(vis_l[c1] + vis_l[c2] + (1.0 + CB2)*(rho[c1]*k[c1] + rho[c2]*k[c2]));
        }
		
        dqdl[i] = k_vis*area[i]*dtmp;
        dqdr[i] = -dqdl[i];
      
        if(TurM == 1){
            RealFlow tem;
            tem = -CB2*sigma*dtmp*area[i];
            dqdl[i] += tem*rho[c1]*k[c1];
            dqdr[i] -= tem*rho[c2]*k[c2];
        }
	}	
}

void cuViscousDqScalar(PolyGrid *grid, const char *name, RealFlow *dqdl,   RealFlow *dqdr, IntType ns, IntType ne){
    
    RealFlow sigma;   

    if(strcmp(name,"sa_nu") == 0)
        sigma = 1.0/SIGMA_SA;     
	
	IntType TurM = 0;
	if(strcmp(name,"sa_nu") == 0) TurM = 1;	
	IntType blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuViscousDqScalar <<< blocksPerGrid, threadsPerBlock >>> (gdqdl, gdqdr, gq, gsa_nu, gvis_l, gxfc, gyfc, gzfc, 
														gxcc, gycc, gzcc, gxfn, gyfn, gzfn, garea, gf2c, sigma, 
														TurM, gnTFace);	
	
}

void cuViscousMatsScalar(PolyGrid *grid, const char *name){
	
    IntType ns, ne;
    IntType nTFace = grid->GetNTFace();
   
    RealFlow *dqdl = NULL;
    RealFlow *dqdr = NULL;
    ns = 0;
	
	/* RealFlow **lhsmat = (RealFlow **)grid->GetDataPtr(REAL_FLOW, gnTCell, "lhsmat"); */
	
    do {
        ne   = ns + nTFace;
        if(ne > nTFace) ne = nTFace;
 
        // Calculate Dfdq
        cuViscousDqScalar(grid, name, dqdl, dqdr, ns, ne);
		    
        // Put Dq to the LHS matrices
        cuPutScalarDqToLhs(grid, ns, ne);
		//HANDLE_API_ERR(cudaMemcpy(lhsmat[0], glhsmat, (gnTCell + glenC2C)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
        ns  = ne;
    } while (ns < nTFace);
    
}

void cuViscousFluxScalar(PolyGrid *grid, const char *name){
	
    cuViscousFluxScalar3D_New3(grid, name);
  
    // Matrices from the viscous flux 	
	cuViscousMatsScalar(grid, name);
	
}

__global__ void gpuAddSourceSA(RealFlow* res, RealFlow* lhsmat, const RealFlow *rho, const RealFlow *sa_nu, const RealFlow *vis_l, 
							const RealFlow *omaga, const RealGeom *dist2wall, const RealGeom *vol, const IntType *IndexC2C, 
							const IntType *f2c, const RealFlow xminn, const IntType nTCell){
													
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nTCell){
		RealFlow omaga_cur, S_bar, xkai, xkaip3, fv1, fv2, d, dp2, odp2;
        RealFlow nue, rr, gg, ft2, fw, term1, term2, source, fsim;
        RealFlow dfv1, dfv2, dft2, drr, dgg, dfw;
        nue    = vis_l[i]/rho[i];             
        xkai   = sa_nu[i]/nue; 
        xkaip3 = xkai*xkai*xkai;
        fv1    = xkaip3/(xkaip3+CV1P3); 
        fv2    = 1.0-xkai/(1.0+xkai*fv1);
      
        d      = dist2wall[i];
        dp2    = d*d;
      
        omaga_cur = omaga[i];
       
        S_bar = omaga_cur+sa_nu[i]*fv2/(KAIP2*dp2);
        S_bar = GPUMAXSA(S_bar,xminn);  
       
        rr     = sa_nu[i]/(S_bar*KAIP2*dp2); 
        rr     = GPUMINSA(rr,10.0);           
    
        gg     = rr+CW2*(P6(rr)-rr);
        gg     = GPUMAXSA(gg,xminn);  
    
        fw     = gg*SQRT_SIX((1.0+CW3P6)/(P6(gg)+CW3P6));
        ft2    = CT3*exp(-CT4*xkai*xkai);
      
        term1  = CB1*(1.0-ft2)*omaga_cur;
        term2  = CB1*((1.0-ft2)*fv2+ft2)/KAIP2-CW1*fw;
      
        odp2   = 1.0/dp2;      
        source = term1*sa_nu[i]+term2*sa_nu[i]*sa_nu[i]*odp2;
      
        fsim   = 2.0*term2*sa_nu[i]*odp2;
        dfv1   = 3.0*(fv1-fv1*fv1)/sa_nu[i];
        dfv2   = (fv2-1.0)/sa_nu[i]+(1.0-fv2)*(1.0-fv2)*(fv1/sa_nu[i]+dfv1);
        dft2   = -(2.0*CT4*sa_nu[i]/(nue*nue))*ft2;
        drr    = rr/sa_nu[i]-rr*rr*(fv2/sa_nu[i]+dfv2); 
        dgg    = (1.0-CW2+6.0*CW2*(rr*rr*rr*rr*rr))*drr;
        gg     = GPUMAXSA(gg,10.0*xminn);
        dfw    = SQRT_SIXSA((1.0+CW3P6)/(P6(gg)+CW3P6))
               -(SQRT_SIXSA(1.0+CW3P6)/(pow((P6(gg)+CW3P6),(7.0/6.0))))*P6(gg);      
        dfw   *= dgg;  
        fsim  += odp2*sa_nu[i]*sa_nu[i]*(CB1/KAIP2*(dfv2-ft2*dfv2-fv2*dft2+dft2)-CW1*dfw);
    
        res[i]+= source*rho[i]*vol[i];
        if(fsim<0.0)  lhsmat[i + IndexC2C[i] + 0]  -= fsim*rho[i]*vol[i];
		
	}
}

void cuAddSourceSA(PolyGrid *grid){
   
    IntType nTCell = grid->GetNTCell();
    IntType n      = nTCell + grid->GetNBFace();   

    RealFlow *res     = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "res");
    RealFlow **lhsmat = (RealFlow **)grid->GetDataPtr(REAL_FLOW, nTCell, "lhsmat");
    	
    int iexp = 15;
    RealFlow xminn;
    grid->GetData(&iexp, INT, 1, "iexp", 0);
    //Note: (10.**(-iexp) is machine zero)
    xminn = pow(10.0, -iexp+1);
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	gpuAddSourceSA <<< blocksPerGrid, threadsPerBlock >>> (gres, glhsmat, gq, gsa_nu, gvis_l, gomaga, gdist2wall,
														gvol, gIndexC2C, gf2c, xminn, gnTCell);
															
	//HANDLE_API_ERR(cudaMemcpy(res, gres, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));													
	//HANDLE_API_ERR(cudaMemcpy(lhsmat[0], glhsmat, (gnTCell + glenC2C)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

void cuAddSourceScalar(PolyGrid *grid, const char *name)
{
    if(strcmp(name, "sa_nu") == 0)    
        cuAddSourceSA(grid);
    
}

void cuLoadBackResSA(PolyGrid *grid){
	
	RealFlow *res;
    res = (RealFlow *)grid->GetDataPtr(REAL_FLOW, gnTCell, "res");

    if(!res) {
        mfmem::snew_array_1D(res,gnTCell,dmrfl);
        assert(res != 0);
        grid->UpdateDataPtr(res, REAL_FLOW, gnTCell, "res");
    }
    //for(IntType i=1; i<nVar; i++) res[i] = &res[i-1][gnTCell];
	
	HANDLE_API_ERR(cudaMemcpy(res, gres, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));

}

__device__ double SQRT_SIXSA(double x)
{
	x = exp(0.166666666666666 * log((x)));
	return x;
}

__device__ double GPUMINSA(double a, double b)
{
        return(a>b?b:a);
}

__device__ double GPUMAXSA(double a, double b)
{
        return(a>b?a:b);
}	

RealFlow* CalAngle(PolyGrid* grid) {
	IntType nTFace = grid->GetNTFace();
	IntType nBFace = grid->GetNBFace();
	IntType* f2c = grid->Getf2c();
	RealGeom* xfn = grid->GetXfn();
	RealGeom* yfn = grid->GetYfn();
	RealGeom* zfn = grid->GetZfn();
	RealGeom* xfc = grid->GetXfc();
	RealGeom* yfc = grid->GetYfc();
	RealGeom* zfc = grid->GetZfc();
	RealGeom* xcc = grid->GetXcc();
	RealGeom* ycc = grid->GetYcc();
	RealGeom* zcc = grid->GetZcc();
	RealFlow* angle_h = NULL;

	IntType face, count, c1, c2;
	RealGeom areax, areay, areaz;
	RealFlow dtmp, d1, d2, x1, x2, y1, y2, z1, z2;

	mfmem::snew_array_1D(angle_h, 2 * nTFace, dmrfl);
	for (face = 0; face < nTFace; face++) {
		count = 2 * face;
		c1 = f2c[count];
		c2 = f2c[count + 1];
		areax = xfn[face];
		areay = yfn[face];
		areaz = zfn[face];

		if (areax == 0. && areay == 0. && areaz == 0.) {
			angle_h[count] = -0.;
			angle_h[count + 1] = 0.;
			continue;
		}

		// positions
		x1 = xcc[c1] - xfc[face];
		y1 = ycc[c1] - yfc[face];
		z1 = zcc[c1] - zfc[face];
		x2 = xcc[c2] - xfc[face];
		y2 = ycc[c2] - yfc[face];
		z2 = zcc[c2] - zfc[face];
		d1 = x1 * areax + y1 * areay + z1 * areaz;
		d2 = x2 * areax + y2 * areay + z2 * areaz;

		dtmp = -d1 / (sqrt(x1 * x1 + y1 * y1 + z1 * z1) + TINY);
		if (dtmp > 1.0) dtmp = 1.0;
		if (dtmp < -1.0) dtmp = -1.0;
		angle_h[count] = asin(dtmp) * 180.0 / PI;

		dtmp = d2 / (sqrt(x2 * x2 + y2 * y2 + z2 * z2) + TINY);
		if (dtmp > 1.0) dtmp = 1.0;
		if (dtmp < -1.0) dtmp = -1.0;
		angle_h[count + 1] = asin(dtmp) * 180.0 / PI;
	}
	return angle_h;
}