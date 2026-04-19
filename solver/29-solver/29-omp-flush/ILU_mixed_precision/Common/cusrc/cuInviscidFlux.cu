#include <stdio.h>
#include <iostream>
#include <math.h>

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
#include "solver_ns.h"

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif

#include <cuInviscidFlux.cuh>
#include <cuData.cuh>
#include <cuErrorReturn.cuh>
//#include <cuLUSGS.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

using namespace gpuData;

//dingxin
#ifdef TIMECOST
extern double* timecost;
extern double  time_flux, time_invis, time_roe, time_vis, time_calvis;
extern double  time_limiter;
extern double  time_gradient;
extern double  time_lusgs;
#endif
//TIMECOST

__global__ void gpuRoeFlux(const double* ql, const double* qr, double* flux, const double* area, 
						const double* xfn, const double* yfn, const double* zfn, 
						const double* vgn,
						const int* IsShockFace, const int* IsNormalFace,
						const double gamm1, const double p_bar, const double alf_l, const double alf_n, 
						const int nTFace, const int steady, const int EntropyCorType){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i<nTFace){
		int  ni;
        double rho_a, u_a, v_a, w_a, h_a, c_a, c2_a, vn_a, q2;
        double vn_l, et_l, ht_l, vn_r, et_r, ht_r;
        double tmp0, tmp1, tmp2, alpha1, alpha2, alpha3, eigv1, eigv2, eigv3;
        double drho, du, dv, dw, dp, dvn, dq2;
        double areax, areay, areaz;
        double spectral, epsaa, epsbb, epscc, epsa_r;
        double u_vgn, v_vgn, w_vgn;
        ni = i;
        areax = xfn[i];
        areay = yfn[i];
        areaz = zfn[i];

        // Total energy
        et_l = (ql[4*nTFace + i] + p_bar) / gamm1 + 0.5 * ql[0*nTFace + i] *
            (ql[1*nTFace + i] * ql[1*nTFace + i] + ql[2*nTFace + i] * ql[2*nTFace + i] + ql[3*nTFace + i] * ql[3*nTFace + i]);
        et_r = (qr[4*nTFace + i] + p_bar) / gamm1 + 0.5 * qr[0*nTFace + i] *
            (qr[1*nTFace + i] * qr[1*nTFace + i] + qr[2*nTFace + i] * qr[2*nTFace + i] + qr[3*nTFace + i] * qr[3*nTFace + i]);
        ht_l = et_l + ql[4*nTFace + i] + p_bar;
        ht_r = et_r + qr[4*nTFace + i] + p_bar;

        // Full flux
        vn_l = areax * ql[1*nTFace + i] + areay * ql[2*nTFace + i] + areaz * ql[3*nTFace + i];
        vn_r = areax * qr[1*nTFace + i] + areay * qr[2*nTFace + i] + areaz * qr[3*nTFace + i];
        if (!steady) {   //unsteady
            vn_l -= vgn[ni];
            vn_r -= vgn[ni];
        }

        tmp0 = vn_l * ql[0*nTFace + i];
        tmp1 = vn_r * qr[0*nTFace + i];
        flux[0*nTFace + i] = tmp0 + tmp1;
        flux[1*nTFace + i] = tmp0 * ql[1*nTFace + i] + areax * ql[4*nTFace + i]
            + tmp1 * qr[1*nTFace + i] + areax * qr[4*nTFace + i];
        flux[2*nTFace + i] = tmp0 * ql[2*nTFace + i] + areay * ql[4*nTFace + i]
            + tmp1 * qr[2*nTFace + i] + areay * qr[4*nTFace + i];
        flux[3*nTFace + i] = tmp0 * ql[3*nTFace + i] + areaz * ql[4*nTFace + i]
            + tmp1 * qr[3*nTFace + i] + areaz * qr[4*nTFace + i];
        flux[4*nTFace + i] = ht_l * vn_l + ht_r * vn_r;
        if (!steady) flux[4*nTFace + i] += (ql[4*nTFace + i] + qr[4*nTFace + i] + 2.0 * p_bar) * vgn[ni];   //unsteady, 0.5在最后乘面积的地方

        //采用roe平均计算单元面上的物理量
        tmp0 = sqrt(qr[0*nTFace + i] / ql[0*nTFace + i]);
        tmp1 = 1.0 / (1.0 + tmp0);
        rho_a = sqrt(qr[0*nTFace + i] * ql[0*nTFace + i]);
        u_a = (ql[1*nTFace + i] + qr[1*nTFace + i] * tmp0) * tmp1;
        v_a = (ql[2*nTFace + i] + qr[2*nTFace + i] * tmp0) * tmp1;
        w_a = (ql[3*nTFace + i] + qr[3*nTFace + i] * tmp0) * tmp1;
        vn_a = u_a * areax + v_a * areay + w_a * areaz;
        h_a = (ht_l / ql[0*nTFace + i] + ht_r / qr[0*nTFace + i] * tmp0) * tmp1;

        q2 = 0.5 * (u_a * u_a + v_a * v_a + w_a * w_a);
        c2_a = gamm1 * (h_a - q2);
        c2_a = fabs(c2_a);
        c_a = sqrt(c2_a);

        if (steady) {
            eigv1 = fabs(vn_a);
            eigv2 = fabs(vn_a + c_a);
            eigv3 = fabs(vn_a - c_a);
        }
        else {   //unsteady
            eigv1 = fabs(vn_a - vgn[i]);
            eigv2 = fabs(vn_a - vgn[i] + c_a);
            eigv3 = fabs(vn_a - vgn[i] - c_a);
        }

        //Entropy fix          
        if (EntropyCorType == 3) {
            epsa_r = alf_l;
        }
        else if (EntropyCorType == 4) {
            if (IsNormalFace[ni] && IsShockFace[i] == 0) {
                epsa_r = 0.01 * alf_l;
                //epsa_r = 0.0002;
            }
            else {
                epsa_r = alf_l;
            }
        }
        else {
			//exit(0);
            //(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }

        //cfl3d form
        if (steady) {
            spectral = fabs(u_a) + fabs(v_a) + fabs(w_a) + c_a;
        }
        else {
            u_vgn = vgn[ni] * xfn[i];
            v_vgn = vgn[ni] * yfn[i];
            w_vgn = vgn[ni] * zfn[i];
            spectral = fabs(u_a - u_vgn) + fabs(v_a - v_vgn) + fabs(w_a - w_vgn) + c_a;
        }
        epsaa = epsa_r * spectral;
        epsbb = 0.25 / max(epsaa, TINY);
        epscc = 2.0 * epsaa;
        if (eigv1 < epscc) eigv1 = eigv1 * eigv1 * epsbb + epsaa;
        if (eigv2 < epscc) eigv2 = eigv2 * eigv2 * epsbb + epsaa;
        if (eigv3 < epscc) eigv3 = eigv3 * eigv3 * epsbb + epsaa;

        drho = qr[0*nTFace + i] - ql[0*nTFace + i];
        du = qr[1*nTFace + i] - ql[1*nTFace + i];
        dv = qr[2*nTFace + i] - ql[2*nTFace + i];
        dw = qr[3*nTFace + i] - ql[3*nTFace + i];
        dp = qr[4*nTFace + i] - ql[4*nTFace + i];
        dvn = vn_r - vn_l;

        dq2 = u_a * du + v_a * dv + w_a * dw;

        tmp0 = dp / c2_a;
        tmp1 = rho_a * dvn / c_a;
        alpha1 = (drho - tmp0) * eigv1;
        alpha2 = 0.5 * (tmp0 + tmp1) * eigv2;
        alpha3 = 0.5 * (tmp0 - tmp1) * eigv3;

        tmp0 = alpha1 + alpha2 + alpha3;
        tmp1 = eigv1 * rho_a;
        tmp2 = -tmp1 * dvn + (alpha2 - alpha3) * c_a;
        flux[0*nTFace + i] -= tmp0;
        flux[1*nTFace + i] -= tmp0 * u_a + tmp1 * du + tmp2 * areax;
        flux[2*nTFace + i] -= tmp0 * v_a + tmp1 * dv + tmp2 * areay;
        flux[3*nTFace + i] -= tmp0 * w_a + tmp1 * dw + tmp2 * areaz;
        flux[4*nTFace + i] -= alpha1 * q2 + (alpha2 + alpha3) * h_a + tmp1 * dq2 + tmp2 * vn_a;

        tmp0 = 0.5 * area[i];
        flux[0*nTFace + i] *= tmp0;
        flux[1*nTFace + i] *= tmp0;
        flux[2*nTFace + i] *= tmp0;
        flux[3*nTFace + i] *= tmp0;
        flux[4*nTFace + i] *= tmp0;
	}

}

void cuRoeFlux(double* ql[5], double* qr[5], double* flux[5], 
			double* area, int* face_act, double* vgn, int* IsNormalFace,  int* IsShockFace,
			double gamm1, double p_bar, double alf_l, double alf_n,
			int steady, int EntropyCorType){	

	int blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;	
	
	//Transfer host data into device *IsShockFace:
	//HANDLE_API_ERR(cudaMemcpy(gIsShockFace, IsShockFace, gnTFace*sizeof(int), cudaMemcpyHostToDevice));
	
	//Transfer host data into device *IsNormalFace:
	// HANDLE_API_ERR(cudaMemcpy(gIsNormalFace, IsNormalFace, gnTFace*sizeof(int), cudaMemcpyHostToDevice));
	
	gpuRoeFlux <<< blocksPerGrid, threadsPerBlock >>> (gql, gqr, gflux, garea, gxfn, gyfn, gzfn, gvgn,
							gIsShockFace, gIsNormalFace, gamm1, p_bar, alf_l, alf_n, gnTFace, gsteady, EntropyCorType);
		
}

__global__ void gpuSetQlQrWithQ(const double* q, double* ql, double* qr, const int* f2c, 
							const int nTFace, const int nBFace, const int nTCell){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nTFace){
		IntType  c1, c2, count;
		IntType nvar = 5;
		IntType Cell = nTCell + nBFace;
		count = 2 * i;
        c1 = f2c[count++];
        c2 = f2c[count];
		for (IntType n = 0; n < nvar; n++) {
            ql[n*nTFace + i] = q[n*Cell + c1];//ql[0][i] ql[1][i]
            qr[n*nTFace + i] = q[n*Cell + c2];
        }
	}	
}

void cuSetQlQrWithQ(RealFlow* q[5]){
	
	IntType blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
	
	//Transfer host data into device q[5]:
	//HANDLE_API_ERR(cudaMemcpy(gq, q[0], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(&gq[(gnTCell + gnBFace)], q[1], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(&gq[2*(gnTCell + gnBFace)], q[2], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(&gq[3*(gnTCell + gnBFace)], q[3], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(&gq[4*(gnTCell + gnBFace)], q[4], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	//SetQlQrWithQ:
	gpuSetQlQrWithQ <<< blocksPerGrid, threadsPerBlock >>> (gq, gql, gqr, gf2c, gnTFace, gnBFace, gnTCell);
	
	
}

__global__ void gpuCalcuQlQrBFace(RealFlow* ql, RealFlow* qr, const IntType* f2c, const RealFlow* limit, 
								const RealFlow* dqdx, const RealFlow* dqdy, const RealFlow* dqdz, const IntType* type_bcr,
								const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc, 
								const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, 
								const IntType nTFace, const IntType nBFace, const IntType nTCell, const RealFlow p_bar){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, count, type;
	RealGeom dx, dy, dz;
	RealFlow trho, tpre;
	IntType Cell = nTCell + nBFace;
	if (i < nBFace){
		type = type_bcr[i];
		if(type!=INTERFACE && type!=SYMM) {
			
		}
		else{
			count = 2*i;
			c1 = f2c[count++];
			c2 = f2c[count];
		
			// Left one
			dx     = xfc[i] - xcc[c1];
			dy     = yfc[i] - ycc[c1];
			dz     = zfc[i] - zcc[c1];
        
			trho   = ql[0*nTFace + i] + limit[0*Cell + c1]*(dqdx[0*Cell + c1]*dx + dqdy[0*Cell + c1]*dy + dqdz[0*Cell + c1]*dz);
			tpre   = ql[4*nTFace + i] + limit[4*Cell + c1]*(dqdx[4*Cell + c1]*dx + dqdy[4*Cell + c1]*dy + dqdz[4*Cell + c1]*dz);
		
			if(trho > 0 && tpre > -p_bar){
				ql[0*nTFace + i]  = trho;
				ql[1*nTFace + i] += limit[1*Cell + c1]*(dqdx[1*Cell + c1]*dx + dqdy[1*Cell + c1]*dy + dqdz[1*Cell + c1]*dz);
				ql[2*nTFace + i] += limit[2*Cell + c1]*(dqdx[2*Cell + c1]*dx + dqdy[2*Cell + c1]*dy + dqdz[2*Cell + c1]*dz);
				ql[3*nTFace + i] += limit[3*Cell + c1]*(dqdx[3*Cell + c1]*dx + dqdy[3*Cell + c1]*dy + dqdz[3*Cell + c1]*dz);
				ql[4*nTFace + i]  = tpre;
			}
		
			if (type == INTERFACE){
				// Right one
				dx     = xfc[i] - xcc[c2];
				dy     = yfc[i] - ycc[c2];
				dz     = zfc[i] - zcc[c2];
    
				trho   = qr[0*nTFace + i] + limit[0*Cell + c2]*(dqdx[0*Cell + c2]*dx + dqdy[0*Cell + c2]*dy + dqdz[0*Cell + c2]*dz);
				tpre   = qr[4*nTFace + i] + limit[4*Cell + c2]*(dqdx[4*Cell + c2]*dx + dqdy[4*Cell + c2]*dy + dqdz[4*Cell + c2]*dz);
				if(trho > 0 && tpre > -p_bar){
					qr[0*nTFace + i]  = trho;
					qr[1*nTFace + i] += limit[1*Cell + c2]*(dqdx[1*Cell + c2]*dx + dqdy[1*Cell + c2]*dy + dqdz[1*Cell + c2]*dz);
					qr[2*nTFace + i] += limit[2*Cell + c2]*(dqdx[2*Cell + c2]*dx + dqdy[2*Cell + c2]*dy + dqdz[2*Cell + c2]*dz);
					qr[3*nTFace + i] += limit[3*Cell + c2]*(dqdx[3*Cell + c2]*dx + dqdy[3*Cell + c2]*dy + dqdz[3*Cell + c2]*dz);
					qr[4*nTFace + i]  = tpre;
				}
			}
		}
						
	}
										
}

__global__ void gpuCalcuQlQrInFace(RealFlow* ql, RealFlow* qr, const IntType* f2c, const RealFlow* limit, 
								const RealFlow* dqdx, const RealFlow* dqdy, const RealFlow* dqdz,
								const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc, 
								const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, 
								const IntType nTFace, const IntType nBFace, const IntType nTCell, const RealFlow p_bar){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x + nBFace;
	IntType c1, c2, count;
	RealGeom dx, dy, dz;
	RealFlow trho, tpre;
	IntType Cell = nTCell + nBFace;
	if (i < nTFace){
		count = 2*i;
		c1 = f2c[count++];
		c2 = f2c[count];
		
		// Left one
        dx     = xfc[i] - xcc[c1];
        dy     = yfc[i] - ycc[c1];
        dz     = zfc[i] - zcc[c1];
      
        trho   = ql[0*nTFace + i] + limit[0*Cell + c1]*(dqdx[0*Cell + c1]*dx + dqdy[0*Cell + c1]*dy + dqdz[0*Cell + c1]*dz);
        tpre   = ql[4*nTFace + i] + limit[4*Cell + c1]*(dqdx[4*Cell + c1]*dx + dqdy[4*Cell + c1]*dy + dqdz[4*Cell + c1]*dz);
        if(trho > 0 && tpre > -p_bar){
            ql[0*nTFace + i]  = trho;
            ql[1*nTFace + i] += limit[1*Cell + c1]*(dqdx[1*Cell + c1]*dx + dqdy[1*Cell + c1]*dy + dqdz[1*Cell + c1]*dz);
            ql[2*nTFace + i] += limit[2*Cell + c1]*(dqdx[2*Cell + c1]*dx + dqdy[2*Cell + c1]*dy + dqdz[2*Cell + c1]*dz);
            ql[3*nTFace + i] += limit[3*Cell + c1]*(dqdx[3*Cell + c1]*dx + dqdy[3*Cell + c1]*dy + dqdz[3*Cell + c1]*dz);
            ql[4*nTFace + i]  = tpre;
        }
        
        // Right one
        dx     = xfc[i] - xcc[c2];
        dy     = yfc[i] - ycc[c2];
        dz     = zfc[i] - zcc[c2];
        
        trho   = qr[0*nTFace + i] + limit[0*Cell + c2]*(dqdx[0*Cell + c2]*dx + dqdy[0*Cell + c2]*dy + dqdz[0*Cell + c2]*dz);
        tpre   = qr[4*nTFace + i] + limit[4*Cell + c2]*(dqdx[4*Cell + c2]*dx + dqdy[4*Cell + c2]*dy + dqdz[4*Cell + c2]*dz);
        if(trho > 0 && tpre > -p_bar){
            qr[0*nTFace + i]  = trho;
            qr[1*nTFace + i] += limit[1*Cell + c2]*(dqdx[1*Cell + c2]*dx + dqdy[1*Cell + c2]*dy + dqdz[1*Cell + c2]*dz);
            qr[2*nTFace + i] += limit[2*Cell + c2]*(dqdx[2*Cell + c2]*dx + dqdy[2*Cell + c2]*dy + dqdz[2*Cell + c2]*dz);
            qr[3*nTFace + i] += limit[3*Cell + c2]*(dqdx[3*Cell + c2]*dx + dqdy[3*Cell + c2]*dy + dqdz[3*Cell + c2]*dz);
            qr[4*nTFace + i]  = tpre;
        }      
		
		
	}
}

void cuCalcuQlQr(RealFlow* ql[5], RealFlow* qr[5], RealFlow **limit, RealFlow *dqdx[5], RealFlow *dqdy[5], RealFlow *dqdz[5]){
	
	
	// Transfer host data into device dqdx[5]:
	// MPI had updated the value of dqdx, dqdy and dqdz, thus another transfer of dqdx and etc was needed. 
	// move this module into cuLimit.cu
	
	// HANDLE_API_ERR(cudaMemcpy(glimit, limit[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	// Boundary face cycle:
	int blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCalcuQlQrBFace <<< blocksPerGrid, threadsPerBlock >>> (gql, gqr, gf2c, glimit, 
								gdqdx, gdqdy, gdqdz, gtype_bcr, gxfc, gyfc, gzfc, 
								gxcc, gycc, gzcc, gnTFace, gnBFace, gnTCell, gp_bar); 
	// Interior face cycle:							
	blocksPerGrid = ((gnTFace - gnBFace) + threadsPerBlock - 1) / threadsPerBlock;
	gpuCalcuQlQrInFace <<< blocksPerGrid, threadsPerBlock >>> (gql, gqr, gf2c, glimit, 
								gdqdx, gdqdy, gdqdz, gxfc, gyfc, gzfc, 
								gxcc, gycc, gzcc, gnTFace, gnBFace, gnTCell, gp_bar); 
	
}

__global__ void gpuModQlQrBou(const RealFlow* q, RealFlow* ql, RealFlow* qr, const RealGeom* xfn, const RealGeom* yfn, const RealGeom* zfn,
								const IntType* f2c, const RealGeom* vgn, const IntType* type_bcr, const IntType steady,
								const IntType nTFace, const IntType nBFace, const IntType nTCell){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, type;
	RealFlow vn, tem;
	IntType Cell = nTCell + nBFace;
	if(i < nBFace){
		type = type_bcr[i];
		if (type == INTERFACE){
			//continue;
		}
		else if(type == SYMM){
            //rho
            qr[0*nTFace + i] = ql[0*nTFace + i];
            //u,v,w
            vn = ql[1*nTFace + i]*xfn[i] + ql[2*nTFace + i]*yfn[i] + ql[3*nTFace + i]*zfn[i];
            if(!steady){         //zhyb:对称面vgn为0，此处本来可以不考虑。但是在粘性计算时，有时可能会采用对称边界条件表示无粘的物面，
                vn -= vgn[i];    //因此在此需要加上非定常的情况
            }
            qr[1*nTFace + i] = ql[1*nTFace + i] - 2.0*vn*xfn[i];
            qr[2*nTFace + i] = ql[2*nTFace + i] - 2.0*vn*yfn[i];
            qr[3*nTFace + i] = ql[3*nTFace + i] - 2.0*vn*zfn[i];
            //p
            qr[4*nTFace + i] = ql[4*nTFace + i];
        }
        else{
			c1 = f2c[2*i];
			c2 = f2c[2*i + 1];
            for (IntType j = 0; j < 5; j++) {
                tem = 0.5 * (q[j*Cell + c1] + q[j*Cell + c2]);
                ql[j*nTFace + i] = tem;
                qr[j*nTFace + i] = tem;
            }
        }
		
	}
	
}

void cuModQlQrBou(RealFlow* ql[5], RealFlow* qr[5]){
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuModQlQrBou <<< blocksPerGrid, threadsPerBlock >>> (gq, gql, gqr, gxfn, gyfn, gzfn,
								gf2c, gvgn, gtype_bcr, gsteady, gnTFace, gnBFace, gnTCell);
}

__global__ void gpuLoadFlux(RealFlow* res, const RealFlow* flux, const IntType* C2F, const IntType* IndexC2F, 
						const IntType* nFPC, const IntType* f2c, const IntType nTFace, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face;
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			face = C2F[IndexC2F[i] + j];
			c1 = f2c[2*face];
			c2 = f2c[2*face + 1];
			if (i == c1) {
                res[0*nTCell + i] -= flux[0*nTFace + face];
                res[1*nTCell + i] -= flux[1*nTFace + face];
                res[2*nTCell + i] -= flux[2*nTFace + face];
                res[3*nTCell + i] -= flux[3*nTFace + face];
                res[4*nTCell + i] -= flux[4*nTFace + face];
            }
            else if (i == c2) {
                res[0*nTCell + i] += flux[0*nTFace + face];
                res[1*nTCell + i] += flux[1*nTFace + face];
                res[2*nTCell + i] += flux[2*nTFace + face];
                res[3*nTCell + i] += flux[3*nTFace + face];
                res[4*nTCell + i] += flux[4*nTFace + face];
            }
			
		}		
	}
	
}

#if (defined ShareMemory)
__global__ void gpuLoadFluxShareMemory(RealFlow* res, const RealFlow* flux, const IntType* C2F, const IntType* IndexC2F, 
						const IntType* nFPC, const IntType* f2c, const IntType nTFace, const IntType nTCell){
	
	extern __shared__ double sdata[];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;
	
	for(IntType j = 0; j < 5; j++){
		sdata[tid*5 + j] = 0.0;
	}
	__syncthreads();
	
	IntType c1, c2, face;
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			face = C2F[IndexC2F[i] + j];
			c1 = f2c[2*face];
			c2 = f2c[2*face + 1];
			if (i == c1) {
                sdata[tid*5 + 0] -= flux[0*nTFace + face];
                sdata[tid*5 + 1] -= flux[1*nTFace + face];
                sdata[tid*5 + 2] -= flux[2*nTFace + face];
                sdata[tid*5 + 3] -= flux[3*nTFace + face];
                sdata[tid*5 + 4] -= flux[4*nTFace + face];
            }
            else if (i == c2) {
                sdata[tid*5 + 0] += flux[0*nTFace + face];
                sdata[tid*5 + 1] += flux[1*nTFace + face];
                sdata[tid*5 + 2] += flux[2*nTFace + face];
                sdata[tid*5 + 3] += flux[3*nTFace + face];
                sdata[tid*5 + 4] += flux[4*nTFace + face];
            }
			
		}				
		
	}
	__syncthreads();
	if(i < nTCell){
		res[0*nTCell + i] = sdata[tid*5 + 0];
		res[1*nTCell + i] = sdata[tid*5 + 1];
		res[2*nTCell + i] = sdata[tid*5 + 2];
		res[3*nTCell + i] = sdata[tid*5 + 3];
		res[4*nTCell + i] = sdata[tid*5 + 4];
	}
	
}


__global__ void gpuLoadFluxShareMemory2(RealFlow* res, const RealFlow* flux, const IntType* C2F, const IntType* IndexC2F, 
						const IntType* nFPC, const IntType* f2c, const IntType nTFace, const IntType nTCell, const IntType threadsnum){
	
	extern __shared__ double sdata[];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;
	
	for(IntType j = 0; j < 5; j++){
		sdata[j*threadsnum + tid] = 0.0;
	}
	__syncthreads();
	
	IntType c1, c2, face;
	if(i < nTCell){
		for(IntType j = 0; j < nFPC[i]; j++){
			face = C2F[IndexC2F[i] + j];
			c1 = f2c[2*face];
			c2 = f2c[2*face + 1];
			if (i == c1) {
                sdata[0*threadsnum + tid] -= flux[0*nTFace + face];
                sdata[1*threadsnum + tid] -= flux[1*nTFace + face];
                sdata[2*threadsnum + tid] -= flux[2*nTFace + face];
                sdata[3*threadsnum + tid] -= flux[3*nTFace + face];
                sdata[4*threadsnum + tid] -= flux[4*nTFace + face];
            }
            else if (i == c2) {
                sdata[0*threadsnum + tid] += flux[0*nTFace + face];
                sdata[1*threadsnum + tid] += flux[1*nTFace + face];
                sdata[2*threadsnum + tid] += flux[2*nTFace + face];
                sdata[3*threadsnum + tid] += flux[3*nTFace + face];
                sdata[4*threadsnum + tid] += flux[4*nTFace + face];
            }
			
		}				
		
	}
	__syncthreads();
	if(i < nTCell){
		res[0*nTCell + i] = sdata[0*threadsnum + tid];
		res[1*nTCell + i] = sdata[1*threadsnum + tid];
		res[2*nTCell + i] = sdata[2*threadsnum + tid];
		res[3*nTCell + i] = sdata[3*threadsnum + tid];
		res[4*nTCell + i] = sdata[4*threadsnum + tid];
	}
	
}
#endif

void cuLoadFlux(RealFlow* res[5], RealFlow* flux[5]){

	//Transfer host data into device res[5]:
	//HANDLE_API_ERR(cudaMemcpy(gres, res[0], 5*gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(&gq[(gnTCell + gnBFace)], q[1], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	
#if (defined ShareMemory)
	/*
	gpuLoadFluxShareMemory <<< blocksPerGrid, threadsPerBlock, 5*threadsPerBlock*sizeof(RealFlow)>>> (
												gres, gflux, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell);
	*/
	gpuLoadFluxShareMemory2 <<< blocksPerGrid, threadsPerBlock, 5*threadsPerBlock*sizeof(RealFlow)>>> (
												gres, gflux, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell, threadsPerBlock);
#else
	gpuLoadFlux <<< blocksPerGrid, threadsPerBlock >>> (gres, gflux, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell);
#endif

	// HANDLE_API_ERR(cudaMemcpy(res[0], gres, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	// no need to trans back to CPU
	
}

#if (defined Atomic) || (defined GroupColor)
__global__ void gpuLoadFluxAtomic(RealFlow* res, const RealFlow* flux, const IntType* f2c, const IntType nBFace, 
						const IntType nTFace, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1;
	if(i < nBFace){
		c1  = f2c[2*i];

		atomicAddSM35LoadFlux(res + (0*nTCell + c1), -1.0*flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (1*nTCell + c1), -1.0*flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(res + (2*nTCell + c1), -1.0*flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(res + (3*nTCell + c1), -1.0*flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(res + (4*nTCell + c1), -1.0*flux[4*nTFace + i]);			

	}
	
}

__global__ void gpuLoadFluxAtomic2_groupsize(RealFlow* res, const RealFlow* flux, const IntType* f2c, const IntType nBFace, 
						const IntType nTFace, const IntType nTCell, const IntType groupSize){
	
	IntType i = nBFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2;
	if(i < (nBFace + groupSize)){
		c1  = f2c[2*i];
		c2  = f2c[2*i + 1];

		//atomicAdd(res + (0*nTCell + c1), -1.0*flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (0*nTCell + c1), -1.0*flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (1*nTCell + c1), -1.0*flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(res + (2*nTCell + c1), -1.0*flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(res + (3*nTCell + c1), -1.0*flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(res + (4*nTCell + c1), -1.0*flux[4*nTFace + i]);	

		atomicAddSM35LoadFlux(res + (0*nTCell + c2), flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (1*nTCell + c2), flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(res + (2*nTCell + c2), flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(res + (3*nTCell + c2), flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(res + (4*nTCell + c2), flux[4*nTFace + i]);	

	}
	
}

__global__ void gpuLoadFluxAtomic2_groupsize_c1(RealFlow* res, const RealFlow* flux, const IntType* f2c, const IntType nBFace, 
						const IntType nTFace, const IntType nTCell, const IntType groupSize){
	
	IntType i = nBFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1;
	if(i < (nBFace + groupSize)){
		c1  = f2c[2*i];

		//atomicAdd(res + (0*nTCell + c1), -1.0*flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (0*nTCell + c1), -1.0*flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (1*nTCell + c1), -1.0*flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(res + (2*nTCell + c1), -1.0*flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(res + (3*nTCell + c1), -1.0*flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(res + (4*nTCell + c1), -1.0*flux[4*nTFace + i]);	

	}
	
}

__global__ void gpuLoadFluxAtomic2_groupsize_c2(RealFlow* res, const RealFlow* flux, const IntType* f2c, const IntType nBFace, 
						const IntType nTFace, const IntType nTCell, const IntType groupSize){
	
	IntType i = nBFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType c2;
	if(i < (nBFace + groupSize)){
		c2  = f2c[2*i + 1];

		atomicAddSM35LoadFlux(res + (0*nTCell + c2), flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (1*nTCell + c2), flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(res + (2*nTCell + c2), flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(res + (3*nTCell + c2), flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(res + (4*nTCell + c2), flux[4*nTFace + i]);	

	}
	
}

__global__ void gpuLoadFluxAtomic2_grid_ifacegroup(RealFlow* res, const RealFlow* flux, const IntType* f2c, const IntType nBFace, 
						const IntType nTFace, const IntType nTCell, const IntType groupSize, const IntType grid_ifacegroup){
	
	IntType i = nBFace + groupSize + blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2;
	if(i < (nBFace + grid_ifacegroup)){
		c1  = f2c[2*i];
		c2  = f2c[2*i + 1];

		//atomicAdd(res + (0*nTCell + c1), -1.0*flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (0*nTCell + c1), -1.0*flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (1*nTCell + c1), -1.0*flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(res + (2*nTCell + c1), -1.0*flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(res + (3*nTCell + c1), -1.0*flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(res + (4*nTCell + c1), -1.0*flux[4*nTFace + i]);	

		atomicAddSM35LoadFlux(res + (0*nTCell + c2), flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (1*nTCell + c2), flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(res + (2*nTCell + c2), flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(res + (3*nTCell + c2), flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(res + (4*nTCell + c2), flux[4*nTFace + i]);	

	}
	
}

__global__ void gpuLoadFluxAtomic2_2(RealFlow* res, const RealFlow* flux, const IntType* f2c, const IntType nBFace, 
						const IntType nTFace, const IntType nTCell, const IntType groupSize){
	
	IntType i = nBFace + blockDim.x*blockIdx.x + threadIdx.x + groupSize;
	IntType c1, c2;
	if(i < nTFace){
		c1  = f2c[2*i];
		c2  = f2c[2*i + 1];

		//atomicAdd(res + (0*nTCell + c1), -1.0*flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (0*nTCell + c1), -1.0*flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (1*nTCell + c1), -1.0*flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(res + (2*nTCell + c1), -1.0*flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(res + (3*nTCell + c1), -1.0*flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(res + (4*nTCell + c1), -1.0*flux[4*nTFace + i]);	

		atomicAddSM35LoadFlux(res + (0*nTCell + c2), flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (1*nTCell + c2), flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(res + (2*nTCell + c2), flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(res + (3*nTCell + c2), flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(res + (4*nTCell + c2), flux[4*nTFace + i]);	

	}
	
}

__global__ void gpuLoadFluxAtomic2(RealFlow* res, const RealFlow* flux, const IntType* f2c, const IntType nBFace, 
						const IntType nTFace, const IntType nTCell){
	
	IntType i = nBFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2;
	if(i < nTFace){
		c1  = f2c[2*i];
		c2  = f2c[2*i + 1];

		//atomicAdd(res + (0*nTCell + c1), -1.0*flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (0*nTCell + c1), -1.0*flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (1*nTCell + c1), -1.0*flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(res + (2*nTCell + c1), -1.0*flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(res + (3*nTCell + c1), -1.0*flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(res + (4*nTCell + c1), -1.0*flux[4*nTFace + i]);	

		atomicAddSM35LoadFlux(res + (0*nTCell + c2), flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (1*nTCell + c2), flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(res + (2*nTCell + c2), flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(res + (3*nTCell + c2), flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(res + (4*nTCell + c2), flux[4*nTFace + i]);	

	}
	
}

void cuLoadFluxAtomic(RealFlow* res[5], RealFlow* flux[5]){

	//Transfer host data into device res[5]:
	//HANDLE_API_ERR(cudaMemcpy(gres, res[0], 5*gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(&gq[(gnTCell + gnBFace)], q[1], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuLoadFluxAtomic <<< blocksPerGrid, threadsPerBlock >>> (gres, gflux, gf2c, gnBFace, gnTFace, gnTCell);
	
	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuLoadFluxAtomic2 <<< blocksPerGrid, threadsPerBlock >>> (gres, gflux, gf2c, gnBFace, gnTFace, gnTCell);

	// HANDLE_API_ERR(cudaMemcpy(res[0], gres, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	
}
#endif

#if (defined FaceColoring) || (defined GroupColor)
	
__global__ void gpuLoadFluxColorShareMemory(RealFlow* res, const RealFlow* flux, const IntType* f2c, const IntType startFace, const IntType endFace, 
						const IntType nTFace, const IntType nTCell, const IntType threadsnum){
	
	extern __shared__ double sdata[];
	
	unsigned int tid = threadIdx.x;
	IntType i = startFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1;
	
	if(i < endFace){
		c1  = f2c[2*i];
		for(IntType j = 0; j < 5; j++){
			sdata[j*threadsnum + tid] = res[j*nTCell + c1];
		}
	}
	__syncthreads();
	
	
	if(i < endFace){		
	
		sdata[0*threadsnum + tid] -= flux[0*nTFace + i];
		sdata[1*threadsnum + tid] -= flux[1*nTFace + i];
		sdata[2*threadsnum + tid] -= flux[2*nTFace + i];
		sdata[3*threadsnum + tid] -= flux[3*nTFace + i];
		sdata[4*threadsnum + tid] -= flux[4*nTFace + i];
				
	}
	__syncthreads();
	
	if(i < endFace){
		res[0*nTCell + c1] = sdata[0*threadsnum + tid];
		res[1*nTCell + c1] = sdata[1*threadsnum + tid];
		res[2*nTCell + c1] = sdata[2*threadsnum + tid];
		res[3*nTCell + c1] = sdata[3*threadsnum + tid];
		res[4*nTCell + c1] = sdata[4*threadsnum + tid];
	}
}

__global__ void gpuLoadFluxColor3ShareMemory(RealFlow* res, const RealFlow* flux, const IntType* f2c, const IntType startFace, const IntType endFace, 
						const IntType nTFace, const IntType nTCell, const IntType threadsnum){
	
	extern __shared__ double sdata[];
	
	unsigned int tid = threadIdx.x;	
	IntType i = startFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2;
	
	if(i < endFace){
		c1  = f2c[2*i];
		c2  = f2c[2*i + 1];
		for(IntType j = 0; j < 5; j++){
			sdata[j*threadsnum + tid] = res[j*nTCell + c1];
			sdata[(j + 5)*threadsnum + tid] = res[j*nTCell + c2];
		}
	}
	__syncthreads();

	if(i < endFace){		
	
		sdata[0*threadsnum + tid] -= flux[0*nTFace + i];
		sdata[1*threadsnum + tid] -= flux[1*nTFace + i];
		sdata[2*threadsnum + tid] -= flux[2*nTFace + i];
		sdata[3*threadsnum + tid] -= flux[3*nTFace + i];
		sdata[4*threadsnum + tid] -= flux[4*nTFace + i];
		
		sdata[5*threadsnum + tid] += flux[0*nTFace + i];
		sdata[6*threadsnum + tid] += flux[1*nTFace + i];
		sdata[7*threadsnum + tid] += flux[2*nTFace + i];
		sdata[8*threadsnum + tid] += flux[3*nTFace + i];
		sdata[9*threadsnum + tid] += flux[4*nTFace + i];		
	}
	
	__syncthreads();
	
	if(i < endFace){
		res[0*nTCell + c1] = sdata[0*threadsnum + tid];
		res[1*nTCell + c1] = sdata[1*threadsnum + tid];
		res[2*nTCell + c1] = sdata[2*threadsnum + tid];
		res[3*nTCell + c1] = sdata[3*threadsnum + tid];
		res[4*nTCell + c1] = sdata[4*threadsnum + tid];
		
		res[0*nTCell + c2] = sdata[5*threadsnum + tid];
		res[1*nTCell + c2] = sdata[6*threadsnum + tid];
		res[2*nTCell + c2] = sdata[7*threadsnum + tid];
		res[3*nTCell + c2] = sdata[8*threadsnum + tid];
		res[4*nTCell + c2] = sdata[9*threadsnum + tid];
	}
	
}

__global__ void gpuLoadFluxColor(RealFlow* res, const RealFlow* flux, const IntType* f2c, const IntType startFace, const IntType endFace, 
						const IntType nTFace, const IntType nTCell){
	
	IntType i = startFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1;
	if(i < endFace){
		c1  = f2c[2*i];
	
		res[0*nTCell + c1] -= flux[0*nTFace + i];
		res[1*nTCell + c1] -= flux[1*nTFace + i];
		res[2*nTCell + c1] -= flux[2*nTFace + i];
		res[3*nTCell + c1] -= flux[3*nTFace + i];
		res[4*nTCell + c1] -= flux[4*nTFace + i];
				
	}
	
}

__global__ void gpuLoadFluxColor2(RealFlow* res, const RealFlow* flux, const IntType* f2c, const IntType nIFace, const IntType nBFace, 
						const IntType nTFace, const IntType nTCell){
	
	IntType pfacenum = nBFace - nIFace;
	IntType i = pfacenum + blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1;
	if(i < nBFace){
		c1  = f2c[2*i];

		atomicAddSM35LoadFlux(res + (0*nTCell + c1), -1.0*flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(res + (1*nTCell + c1), -1.0*flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(res + (2*nTCell + c1), -1.0*flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(res + (3*nTCell + c1), -1.0*flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(res + (4*nTCell + c1), -1.0*flux[4*nTFace + i]);			

	}
	
}

__global__ void gpuLoadFluxColor3(RealFlow* res, const RealFlow* flux, const IntType* f2c, const IntType startFace, const IntType endFace, 
						const IntType nTFace, const IntType nTCell){
	
	IntType i = startFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2;
	if(i < endFace){
		c1  = f2c[2*i];
		c2  = f2c[2*i + 1];
	
		res[0*nTCell + c1] -= flux[0*nTFace + i];
		res[1*nTCell + c1] -= flux[1*nTFace + i];
		res[2*nTCell + c1] -= flux[2*nTFace + i];
		res[3*nTCell + c1] -= flux[3*nTFace + i];
		res[4*nTCell + c1] -= flux[4*nTFace + i];
		
		res[0*nTCell + c2] += flux[0*nTFace + i];
		res[1*nTCell + c2] += flux[1*nTFace + i];
		res[2*nTCell + c2] += flux[2*nTFace + i];
		res[3*nTCell + c2] += flux[3*nTFace + i];
		res[4*nTCell + c2] += flux[4*nTFace + i];		
	}
	
}

void cuLoadFluxColor(PolyGrid *grid, RealFlow* res[5], RealFlow* flux[5]){
				
	IntType  *f2c   = grid->Getf2c();
	IntType    nIFace = grid->GetNIFace();
	IntType    nBFace = grid->GetNBFace();
	IntType     pfacenum = nBFace - nIFace;
	IntType    bfacegroup_num, ifacegroup_num;
	IntType* grid_bfacegroup, * grid_ifacegroup;
	ifacegroup_num = (*grid).ifacegroup.size();
	bfacegroup_num = (*grid).bfacegroup.size();
	grid_bfacegroup = NULL;
	grid_ifacegroup = NULL;
	mfmem::snew_array_1D(grid_bfacegroup, bfacegroup_num, dmrfl);
	mfmem::snew_array_1D(grid_ifacegroup, ifacegroup_num, dmrfl);
	for (IntType i = 0; i < bfacegroup_num; i++) {
		grid_bfacegroup[i] = (*grid).bfacegroup[i];
	}
	for (IntType i = 0; i < ifacegroup_num; i++) {
		grid_ifacegroup[i] = (*grid).ifacegroup[i];
	}		
	
	for (IntType fcolor = 0; fcolor < bfacegroup_num; fcolor++) {
        IntType startFace, endFace;
        if (fcolor == 0) {
            startFace = 0; //for ns>0 && ns<grid_bfacegroup[0]
        }
        else {
            startFace = grid_bfacegroup[fcolor - 1];
        }
        endFace = grid_bfacegroup[fcolor];
		
		IntType blocksPerGrid = (endFace - startFace + threadsPerBlock - 1) / threadsPerBlock;
#if	(defined ShareMemory)
		gpuLoadFluxColorShareMemory <<< blocksPerGrid, threadsPerBlock, 5*threadsPerBlock*sizeof(RealFlow) >>> 
											(gres, gflux, gf2c, startFace, endFace, gnTFace, gnTCell, threadsPerBlock);
#else
		gpuLoadFluxColor <<< blocksPerGrid, threadsPerBlock >>> (gres, gflux, gf2c, startFace, endFace, gnTFace, gnTCell);
#endif
	}	
			
#ifdef MPICH 	
	IntType blocksPerGrid2 = (gnIFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuLoadFluxColor2 <<< blocksPerGrid2, threadsPerBlock >>> (gres, gflux, gf2c, gnIFace, gnBFace, gnTFace, gnTCell);
#endif	
	
	for (IntType fcolor = 0; fcolor < ifacegroup_num; fcolor++) {
        IntType startFace, endFace;
        if (fcolor == 0) {
            startFace = gnBFace;
        }
        else {
            startFace = grid_ifacegroup[fcolor - 1];
        }
        endFace = grid_ifacegroup[fcolor];
		
		IntType blocksPerGrid = (endFace - startFace + threadsPerBlock - 1) / threadsPerBlock;
#if	(defined ShareMemory)
		gpuLoadFluxColor3ShareMemory <<< blocksPerGrid, threadsPerBlock, 10*threadsPerBlock*sizeof(RealFlow) >>> 
											(gres, gflux, gf2c, startFace, endFace, gnTFace, gnTCell, threadsPerBlock);
#else
		gpuLoadFluxColor3 <<< blocksPerGrid, threadsPerBlock >>> (gres, gflux, gf2c, startFace, endFace, gnTFace, gnTCell);		
#endif		
	}
	
	mfmem::sdel_array_1D(grid_bfacegroup);
	mfmem::sdel_array_1D(grid_ifacegroup);

}

#endif

#if (defined GroupColor)
__global__ void gpuLoadFluxGroupColor(RealFlow* res, const RealFlow* flux, const IntType* f2c, 
						const IntType* b_SMc2c, const IntType* b_f2SMc, const IntType* b_SM_index,
						const IntType startFace, const IntType endFace, 
						const IntType nTFace, const IntType nTCell, const IntType offset_group, const IntType threadsnum){
	
	extern __shared__ double sdata[];
	
	unsigned int tid = threadIdx.x;
	IntType i = startFace + blockDim.x*blockIdx.x + threadIdx.x;
	IntType cell;
	
	IntType startCellIndex = b_SM_index[offset_group + blockIdx.x];
	IntType endCellIndex = b_SM_index[offset_group + blockIdx.x + 1];
	IntType numCell = (endCellIndex - startCellIndex);
	// (endCellIndex - startCellIndex) <= blockDim.x
	// load the cell data on global memory into the cell data on share memory:
	if(tid < numCell){
		cell  = b_SMc2c[startCellIndex + tid];
		for(IntType j = 0; j < 5; j++){
			sdata[j*threadsnum + tid] = res[j*nTCell + cell];
		}
	}
	__syncthreads();	
	
	// add/subtract the cell data on share memory with the flux data on global face:
	// atomic add/subtract:
	if(i < endFace){				
		IntType c1 = b_f2SMc[i];
		atomicAddSM35LoadFlux(sdata + 0*threadsnum + c1, -flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(sdata + 1*threadsnum + c1, -flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(sdata + 2*threadsnum + c1, -flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(sdata + 3*threadsnum + c1, -flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(sdata + 4*threadsnum + c1, -flux[4*nTFace + i]);
		/* sdata[0*threadsnum + cell] -= flux[0*nTFace + i];
		sdata[1*threadsnum + cell] -= flux[1*nTFace + i];
		sdata[2*threadsnum + cell] -= flux[2*nTFace + i];
		sdata[3*threadsnum + cell] -= flux[3*nTFace + i];
		sdata[4*threadsnum + cell] -= flux[4*nTFace + i];	 */			
	}
	__syncthreads();
	
	// load the cell data on share memory back to the cell data on global memory:
	if(tid < numCell){
		/*atomicExchSM35res(res + 0*nTCell + c1, sdata[0*threadsnum + tid]);
		atomicExchSM35res(res + 1*nTCell + c1, sdata[1*threadsnum + tid]);
		atomicExchSM35res(res + 2*nTCell + c1, sdata[2*threadsnum + tid]);
		atomicExchSM35res(res + 3*nTCell + c1, sdata[3*threadsnum + tid]);
		atomicExchSM35res(res + 4*nTCell + c1, sdata[4*threadsnum + tid]);*/
		res[0*nTCell + cell] = sdata[0*threadsnum + tid];
		res[1*nTCell + cell] = sdata[1*threadsnum + tid];
		res[2*nTCell + cell] = sdata[2*threadsnum + tid];
		res[3*nTCell + cell] = sdata[3*threadsnum + tid];
		res[4*nTCell + cell] = sdata[4*threadsnum + tid];
	}
}

__global__ void gpuLoadFluxGroupColor3(RealFlow* res, const RealFlow* flux, const IntType* f2c, 
						const IntType* i_SMc2c, const IntType* i_f2SMc, const IntType* i_SM_index,
						const IntType startFace, const IntType endFace, 
						const IntType nTFace, const IntType nTCell, const IntType offset_group, const IntType threadsnum){
	
	extern __shared__ double sdata[];
	
	unsigned int tid = threadIdx.x;	
	IntType i = startFace + blockDim.x*blockIdx.x + threadIdx.x;
	
	IntType startCellIndex = i_SM_index[offset_group + blockIdx.x];
	IntType endCellIndex = i_SM_index[offset_group + blockIdx.x + 1];
	IntType numCell = (endCellIndex - startCellIndex);
	/* if((blockDim.x*blockIdx.x + threadIdx.x) == 0){
		printf("startCellIndex: %d. ", startCellIndex);
		printf("endCellIndex: %d. ", endCellIndex);
	} */
	
	// load threadsnum first:
	/* if(numCell >= threadsnum){ //this group has cell number larger than threadsnum
		cell  = i_SMc2c[startCellIndex + tid];
		for(IntType j = 0; j < 5; j++){
			sdata[2*j*threadsnum + tid] = res[j*nTCell + cell];
		}
		// load (numCell - threadsnum):
		if((tid + threadsnum) < numCell){
			cell2  = i_SMc2c[startCellIndex + tid + threadsnum];
			for(IntType j = 0; j < 5; j++){
				//RealFlow x = res[j*nTCell + cell2];
				//sdata[(2*j + 1)*threadsnum + tid] = x;
				sdata[(2*j + 1)*threadsnum + tid] = res[j*nTCell + cell2];
			}
		}
	}
	else{ //this group has cell number smaller than threadsnum, numCell < threadsnum
		if(tid < numCell){
			cell  = i_SMc2c[startCellIndex + tid];
			for(IntType j = 0; j < 5; j++){
				sdata[2*j*threadsnum + tid] = res[j*nTCell + cell];
			}
		}
	} */
	if(tid < numCell){
		IntType cell  = i_SMc2c[startCellIndex + tid];
		sdata[0*threadsnum + tid] = res[0*nTCell + cell];
		sdata[2*threadsnum + tid] = res[1*nTCell + cell];
		sdata[4*threadsnum + tid] = res[2*nTCell + cell];
		sdata[6*threadsnum + tid] = res[3*nTCell + cell];
		sdata[8*threadsnum + tid] = res[4*nTCell + cell];
	}
	
	if ((tid + threadsnum) < numCell){
		IntType cell  = i_SMc2c[startCellIndex + threadsnum + tid];		
		sdata[1*threadsnum + tid] = res[0*nTCell + cell];
		sdata[3*threadsnum + tid] = res[1*nTCell + cell];
		sdata[5*threadsnum + tid] = res[2*nTCell + cell];
		sdata[7*threadsnum + tid] = res[3*nTCell + cell];
		sdata[9*threadsnum + tid] = res[4*nTCell + cell];
	}
	
	__syncthreads();

	if(i < endFace){				
		IntType c1 = i_f2SMc[2*i];
		IntType c2 = i_f2SMc[2*i + 1];
		/* if((blockDim.x*blockIdx.x + threadIdx.x) == 0){
			printf("c1: %d. ", c1);
			printf("c2: %d. ", c2);
		} */
		
		atomicAddSM35LoadFlux(sdata + 0*threadsnum + c1, -flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(sdata + 2*threadsnum + c1, -flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(sdata + 4*threadsnum + c1, -flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(sdata + 6*threadsnum + c1, -flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(sdata + 8*threadsnum + c1, -flux[4*nTFace + i]);
		
		atomicAddSM35LoadFlux(sdata + 0*threadsnum + c2, flux[0*nTFace + i]);
		atomicAddSM35LoadFlux(sdata + 2*threadsnum + c2, flux[1*nTFace + i]);
		atomicAddSM35LoadFlux(sdata + 4*threadsnum + c2, flux[2*nTFace + i]);
		atomicAddSM35LoadFlux(sdata + 6*threadsnum + c2, flux[3*nTFace + i]);
		atomicAddSM35LoadFlux(sdata + 8*threadsnum + c2, flux[4*nTFace + i]);
		
	}
	
	__syncthreads();
	
	// load back threadsnum first:
	/* if(numCell >= threadsnum){ //this group has cell number larger than threadsnum
		cell  = i_SMc2c[startCellIndex + tid];
		for(IntType j = 0; j < 5; j++){
			res[j*nTCell + cell] = sdata[2*j*threadsnum + tid];
		}
		// load (numCell - threadsnum):
		if((tid + threadsnum) < numCell){
			cell2  = i_SMc2c[startCellIndex + tid + threadsnum];
			for(IntType j = 0; j < 5; j++){
				//RealFlow x = res[j*nTCell + cell2];
				//sdata[(2*j + 1)*threadsnum + tid] = x;
				res[j*nTCell + cell2] = sdata[(2*j + 1)*threadsnum + tid];
			}
		}
	}
	else{ //this group has cell number smaller than threadsnum, numCell < threadsnum
		if(tid < numCell){
			cell  = i_SMc2c[startCellIndex + tid];
			for(IntType j = 0; j < 5; j++){
				res[j*nTCell + cell] = sdata[2*j*threadsnum + tid];
			}
		}
	} */
	if(tid < numCell){
		IntType cell  = i_SMc2c[startCellIndex + tid];
		res[0*nTCell + cell] = sdata[0*threadsnum + tid];
		res[1*nTCell + cell] = sdata[2*threadsnum + tid];
		res[2*nTCell + cell] = sdata[4*threadsnum + tid];
		res[3*nTCell + cell] = sdata[6*threadsnum + tid];
		res[4*nTCell + cell] = sdata[8*threadsnum + tid];
	}
	if((tid + threadsnum) < numCell){
		IntType cell  = i_SMc2c[startCellIndex + threadsnum + tid];
		res[0*nTCell + cell] = sdata[1*threadsnum + tid];
		res[1*nTCell + cell] = sdata[3*threadsnum + tid];
		res[2*nTCell + cell] = sdata[5*threadsnum + tid];
		res[3*nTCell + cell] = sdata[7*threadsnum + tid];
		res[4*nTCell + cell] = sdata[9*threadsnum + tid];
	}
	__syncthreads(); 
	
}

void cuLoadFluxGroupColor(PolyGrid *grid, RealFlow* res[5], RealFlow* flux[5]){
				
	IntType  *f2c   = grid->Getf2c();
	IntType    nIFace = grid->GetNIFace();
	IntType    nBFace = grid->GetNBFace();
	IntType     pfacenum = nBFace - nIFace;
	IntType    bfacegroup_num, ifacegroup_num;
	IntType* grid_bfacegroup, * grid_ifacegroup;
	ifacegroup_num = (*grid).ifacegroup.size();
	bfacegroup_num = (*grid).bfacegroup.size();
	grid_bfacegroup = NULL;
	grid_ifacegroup = NULL;
	mfmem::snew_array_1D(grid_bfacegroup, bfacegroup_num, dmrfl);
	mfmem::snew_array_1D(grid_ifacegroup, ifacegroup_num, dmrfl);
	for (IntType i = 0; i < bfacegroup_num; i++) {
		grid_bfacegroup[i] = (*grid).bfacegroup[i];
	}
	for (IntType i = 0; i < ifacegroup_num; i++) {
		grid_ifacegroup[i] = (*grid).ifacegroup[i];
	}		
	IntType groupSize = grid->groupSize;
	
	/* IntType blocksPerGridb = (gnBFace - gnIFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuLoadFluxAtomic <<< blocksPerGridb, threadsPerBlock >>> (gres, gflux, gf2c, gnBFace - gnIFace, gnTFace, gnTCell);  */
	
#ifdef MPICH   
	IntType mpirank = 0; //when mpirank = 0, mpi was off. 
    MPI_Comm_rank(MPI_COMM_WORLD, & mpirank);	
#endif
	//cout << "rankid: " << mpirank << "-" << "bfacegroup_num:" << bfacegroup_num << endl;
	for (IntType fcolor = 0; fcolor < bfacegroup_num; fcolor++) {
        IntType startFace, endFace;
		IntType num_Group_this_color;
		IntType offset_group;
        if (fcolor == 0) {
            startFace = 0; //for ns>0 && ns<grid_bfacegroup[0]
			num_Group_this_color = grid->group_b_SM_color_index[fcolor];
			offset_group = 0;
        }
        else {
            startFace = grid_bfacegroup[fcolor - 1];
			num_Group_this_color = grid->group_b_SM_color_index[fcolor] - grid->group_b_SM_color_index[fcolor - 1];
			offset_group = grid->group_b_SM_color_index[fcolor - 1]; // the group offset in this color
        }
        endFace = grid_bfacegroup[fcolor];
		//cout << "rankid: " << mpirank << "-" << num_Group_this_color << ", " << offset_group << endl;
		//IntType blocksPerGrid = num_Group_this_color; // the num of group in single color
		
		gpuLoadFluxGroupColor <<< num_Group_this_color, groupSize, 5*groupSize*sizeof(RealFlow) >>> 
											(gres, gflux, gf2c, g_b_SMc2c, g_b_f2SMc, g_b_SM_index,
											startFace, endFace, gnTFace, gnTCell, offset_group, groupSize);
		
	}
			
#ifdef MPICH 	
	IntType blocksPerGrid2 = (gnIFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuLoadFluxColor2 <<< blocksPerGrid2, threadsPerBlock >>> (gres, gflux, gf2c, gnIFace, gnBFace, gnTFace, gnTCell);
#endif	
	
	/*for (IntType fcolor = 0; fcolor < 1; fcolor++) {
        IntType startFace, endFace;
		IntType num_Group_this_color;
		IntType offset_group;
        if (fcolor == 0) {
            startFace = gnBFace;
			num_Group_this_color = grid->group_i_SM_color_index[fcolor];
			offset_group = 0;
        }
        else {
            startFace = grid_ifacegroup[fcolor - 1];
			num_Group_this_color = grid->group_i_SM_color_index[fcolor] - grid->group_i_SM_color_index[fcolor - 1];
			offset_group = grid->group_i_SM_color_index[fcolor - 1]; // the group offset in this color
        }
        endFace = grid_ifacegroup[fcolor];
		
		//IntType blocksPerGrid = (endFace - startFace + threadsPerBlock - 1) / threadsPerBlock;

		gpuLoadFluxGroupColor3 <<< num_Group_this_color, groupSize, 10*groupSize*sizeof(RealFlow) >>> 
											(gres, gflux, gf2c, g_i_SMc2c, g_i_f2SMc, g_i_SM_index,
											startFace, endFace, gnTFace, gnTCell, offset_group, groupSize);	
	
	}*/
	
	for (IntType fcolor = 0; fcolor < ifacegroup_num; fcolor++) {
		IntType startFace, endFace;
		IntType num_Group_this_color;
		IntType offset_group;
		if (fcolor == 0) {
			startFace = gnBFace;
			num_Group_this_color = grid->group_i_SM_color_index[fcolor];
			offset_group = 0;
		}
		else {
			startFace = grid_ifacegroup[fcolor - 1];
			num_Group_this_color = grid->group_i_SM_color_index[fcolor] - grid->group_i_SM_color_index[fcolor - 1];
			offset_group = grid->group_i_SM_color_index[fcolor - 1]; // the group offset in this color
		}
		endFace = grid_ifacegroup[fcolor];
		
		//IntType blocksPerGrid = (endFace - startFace + threadsPerBlock - 1) / threadsPerBlock;

		/* gpuLoadFluxGroupColor3 <<< 1, groupSize, 10*groupSize*sizeof(RealFlow) >>> 
											(gres, gflux, gf2c, g_i_SMc2c, g_i_f2SMc, g_i_SM_index,
											startFace, startFace + groupSize, gnTFace, gnTCell, offset_group, groupSize); */
		/* cout << "num_Group_this_color: " << num_Group_this_color << endl;
		cout << "endFace - startFace: " << endFace - startFace << endl;	
		exit(0); */
		gpuLoadFluxGroupColor3 <<< num_Group_this_color, groupSize, 10*groupSize*sizeof(RealFlow) >>> 
											(gres, gflux, gf2c, g_i_SMc2c, g_i_f2SMc, g_i_SM_index,
											startFace, endFace, gnTFace, gnTCell, offset_group, groupSize);
		
		/* for (IntType i = startFace; i < startFace + groupSize; i++){
			IntType c1 = f2c[2*i];
			IntType c2 = f2c[2*i + 1];
			cout << "face id: " << i  << ", c1: " << c1 << ", c2: "  << c2 << "; " << endl;
		}
		exit(0); */
	}
	
	/* IntType blocksPerGridtmp = (groupSize + threadsPerBlock - 1) / threadsPerBlock;	
	gpuLoadFluxAtomic2_groupsize_c1 <<< blocksPerGridtmp, threadsPerBlock >>> (gres, gflux, gf2c, gnBFace, gnTFace, gnTCell, groupSize); 
	gpuLoadFluxAtomic2_groupsize_c2 <<< blocksPerGridtmp, threadsPerBlock >>> (gres, gflux, gf2c, gnBFace, gnTFace, gnTCell, groupSize); */
	
	/* IntType blocksPerGridtmp = (groupSize + threadsPerBlock - 1) / threadsPerBlock;	
	gpuLoadFluxAtomic2_groupsize <<< blocksPerGridtmp, threadsPerBlock >>> (gres, gflux, gf2c, gnBFace, gnTFace, gnTCell, groupSize); */
	
	/* IntType blocksPerGridtmp2 = (grid_ifacegroup[0] - groupSize + threadsPerBlock - 1) / threadsPerBlock;	
	gpuLoadFluxAtomic2_grid_ifacegroup <<< blocksPerGridtmp2, threadsPerBlock >>> (gres, gflux, gf2c, gnBFace, gnTFace, gnTCell, groupSize, grid_ifacegroup[0]); */
	
	/* IntType blocksPerGrid = (gnTFace - gnBFace - grid_ifacegroup[0] + threadsPerBlock - 1) / threadsPerBlock;	
	gpuLoadFluxAtomic2_2 <<< blocksPerGrid, threadsPerBlock >>> (gres, gflux, gf2c, gnBFace, gnTFace, gnTCell, grid_ifacegroup[0]); */
	
	/* IntType blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuLoadFluxAtomic2 <<< blocksPerGrid, threadsPerBlock >>> (gres, gflux, gf2c, gnBFace, gnTFace, gnTCell); */
	
	mfmem::sdel_array_1D(grid_bfacegroup);
	mfmem::sdel_array_1D(grid_ifacegroup);

}

#endif

__global__ void gpuCalIsShockFace(IntType *IsShockFace, const RealFlow *q, const RealFlow *xfn, 
								const RealFlow *yfn, const RealFlow *zfn, const IntType *f2c, 
								const RealFlow pref, const RealFlow ThdShock, const IntType nTFace, 
								const IntType nTCell, const IntType nBFace){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType  c1, c2;
	RealFlow dp, t, vx, vy, vz;
	IntType Cell = nTCell + nBFace;
	if(i < nTFace){
		IsShockFace[i] = 0;
		
		c1 = f2c[2*i];
        c2 = f2c[2*i + 1];

        // average velocity
        vx = q[1*Cell + c1] + q[1*Cell + c2];
        vy = q[2*Cell + c1] + q[2*Cell + c2];
        vz = q[3*Cell + c1] + q[3*Cell + c2];
        
        t = vx*xfn[i] + vy*yfn[i] + vz*zfn[i];
        if(t > 0){
			dp = q[4*Cell + c2] - q[4*Cell + c1];
		}
        else{
			dp = q[4*Cell + c1] - q[4*Cell + c2];
		}
        dp /= pref;
        if(dp > ThdShock) IsShockFace[i] = 1; 
	}
	
}

void cuCalIsShockFace(PolyGrid *grid, IntType *IsShockFace){

    RealFlow mach00, gam, p_bar;
    grid->GetData(&mach00, REAL_FLOW, 1, "mach");
    grid->GetData(&gam, REAL_FLOW, 1, "gam");
    grid->GetData(&p_bar,   REAL_FLOW, 1, "p_bar");
    RealFlow pref = p_bar*(1.0 + 0.5*(gam - 1.0)*mach00*mach00);
    
    RealFlow ThdShock = 0.5;    // threshold for shock face
	IntType blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCalIsShockFace <<< blocksPerGrid, threadsPerBlock >>> (gIsShockFace, gq, gxfn, gyfn, gzfn, gf2c, 
														pref, ThdShock, gnTFace, gnTCell, gnBFace);
	
}

void cuRoeFlux_noprec(PolyGrid* grid, RealFlow* ql[5], RealFlow* qr[5], RealFlow* flux[5],
					RealGeom* xfn, RealGeom* yfn, RealGeom* zfn, RealGeom* area, IntType* face_act,
					RealFlow gam, RealFlow p_bar, RealFlow alf_l, RealFlow alf_n,
					IntType ns, IntType ne){
    
    RealFlow gamm1;

    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  nTFace = grid->GetNTFace();
    //IntType  n = nTCell + nBFace;
    //IntType* f2c = grid->Getf2c();

    RealGeom* vgn = grid->GetFaceNormalVelocity();

    IntType steady = 1;
    grid->GetData(&steady, INT, 1, "steady");
    RealFlow gascon;
    grid->GetData(&gascon, REAL_FLOW, 1, "gascon");
    IntType EntropyCorType = 4;
    grid->GetData(&EntropyCorType, INT, 1, "EntropyCorType");

    IntType* IsNormalFace = 0;
    IntType* IsShockFace = 0;
    if (EntropyCorType == 4) {
        IsNormalFace = (IntType*)grid->GetDataPtr(INT, nTFace, "IsNormalFace");
        if (!IsNormalFace) {
            grid->FindNormalFace();
            IsNormalFace = (IntType*)grid->GetDataPtr(INT, nTFace, "IsNormalFace");
        }

        // shock face or not
        IsShockFace = NULL;
        mfmem::snew_array_1D(IsShockFace, nTFace, dmrfl);
        cuCalIsShockFace(grid, IsShockFace);
    }

    gamm1 = gam - 1.0;

	//Work out Roe function on GPU with CUDA, ruitian, 2022.3.21
	//len must equal to nTFace!
	cuRoeFlux(ql, qr, flux, area, face_act, vgn, IsNormalFace, IsShockFace, gamm1, p_bar, alf_l, alf_n, steady, EntropyCorType);
    
    if (EntropyCorType == 4) { mfmem::sdel_array_1D(IsShockFace); }
}


void cuCompInvFlux(PolyGrid *grid, RealFlow *ql[5],   RealFlow *qr[5], RealFlow *flux[5],
                 RealGeom *xfn,  RealGeom *yfn,     RealGeom *zfn,   RealGeom *area, 
                 RealGeom *vgn,  IntType *face_act, RealFlow gam,    RealFlow p_bar,  
                 RealFlow alf_l, RealFlow alf_n,    IntType type_flux,   
                 IntType ns,     IntType ne){
					 

    cuRoeFlux_noprec(grid, ql, qr, flux, xfn, yfn, zfn,
                   area, face_act, gam, p_bar, alf_l, alf_n, ns, ne);

}

void cuMemoryPreparaInVisFluxDebug(PolyGrid *grid, RealFlow **limit){
	
	IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nTFace = grid->GetNTFace();
    IntType n      = nTCell + nBFace;
	
	// Allocate temporary memories for ql, qr and flux
    RealFlow *q[5];

    // Get flow variables
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
	
	HANDLE_API_ERR(cudaMemcpy(gq, q[0], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[(gnTCell + gnBFace)], q[1], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[2*(gnTCell + gnBFace)], q[2], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[3*(gnTCell + gnBFace)], q[3], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(&gq[4*(gnTCell + gnBFace)], q[4], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
    
    const IntType kNVar = 5;
    RealFlow **dqdx = NULL, **dqdy = NULL, **dqdz = NULL;
    mfmem::snew_array_1D(dqdx, kNVar, dmrfl);
    mfmem::snew_array_1D(dqdy, kNVar, dmrfl);
    mfmem::snew_array_1D(dqdz, kNVar, dmrfl);
    dqdx[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdx"));
    dqdy[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdy"));
    dqdz[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdz"));
    for (IntType i = 1; i < kNVar; ++i) {
        dqdx[i] = &dqdx[i - 1][n];
        dqdy[i] = &dqdy[i - 1][n];
        dqdz[i] = &dqdz[i - 1][n];
    }
	
	HANDLE_API_ERR(cudaMemcpy(gdqdx, dqdx[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(gdqdy, dqdy[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	HANDLE_API_ERR(cudaMemcpy(gdqdz, dqdz[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	HANDLE_API_ERR(cudaMemcpy(glimit, limit[0], 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));

	RealFlow *res[5];
	res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");
	res[1] = &res[0][nTCell];
    res[2] = &res[1][nTCell];
    res[3] = &res[2][nTCell];
    res[4] = &res[3][nTCell];
	   
	HANDLE_API_ERR(cudaMemcpy(gres, res[0], 5*gnTCell*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
}

void cuMemoryPreparaInVisFluxDebug2(PolyGrid *grid){
	
	IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
	
	RealFlow *res[5];
	res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");
	res[1] = &res[0][nTCell];
    res[2] = &res[1][nTCell];
    res[3] = &res[2][nTCell];
    res[4] = &res[3][nTCell];
	   
	HANDLE_API_ERR(cudaMemcpy(res[0], gres, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
}

void cuInviscidFlux(PolyGrid *grid, RealFlow **limit, IntType level){
	
#ifdef FS_CUDA_DEBUG_NS_Flux
	cuMemoryPreparaInVisFluxDebug(grid, limit);
#endif
	
#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemInvis, endtimeTemInvis;
    double timeuseTemInvis;
    gettimeofday(&starttimeTemInvis, 0); 
#endif
#endif
    IntType ns, ne; 
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nTFace = grid->GetNTFace();

    // Get parameters
    IntType  steady;
    RealFlow gam, p_bar, alf_l, alf_n, disFact=1.;
    grid->GetData(&steady, INT, 1, "steady");
    grid->GetData(&gam,    REAL_FLOW, 1, "gam");
    grid->GetData(&p_bar,  REAL_FLOW, 1, "p_bar");
    grid->GetData(&alf_l,  REAL_FLOW, 1, "alf_l");
    grid->GetData(&alf_n,  REAL_FLOW, 1, "alf_n");
    grid->GetData(&disFact,  REAL_FLOW, 1, "disFact",0);
    
    // Get metrics
    RealGeom *xfn  = grid->GetXfn();
    RealGeom *yfn  = grid->GetYfn();
    RealGeom *zfn  = grid->GetZfn();
    RealGeom *area = grid->GetFaceArea();
    RealGeom *vgn  = grid->GetFaceNormalVelocity();
    // for overlap
    IntType *face_act = NULL;
       
    ns = 0;    
	ne = nTFace;
	
	/* cudaEvent_t cu_start, cu_stop;
	float cu_esp;
	cudaEventCreate(&cu_start);
	cudaEventCreate(&cu_stop);
	cudaEventRecord(cu_start, 0); */
	
	// Get left variables and right variables	   
	cuSetQlQrWithQ(NULL);	   
	
	//if (limit != NULL)
	cuCalcuQlQr(NULL, NULL, limit, NULL, NULL, NULL);  
	cuModQlQrBou(NULL, NULL);       
	cuCompInvFlux(grid, NULL, NULL, NULL, &xfn[ns], &yfn[ns], &zfn[ns], &area[ns], &vgn[ns],
			   &face_act[ns], gam, p_bar, alf_l, alf_n, 0, ns, ne);
			   
	/* cudaEventRecord(cu_stop, 0);
	cudaEventSynchronize(cu_stop);
	
	cudaEventElapsedTime(&cu_esp, cu_start, cu_stop);	
	
#ifdef TIMECOST//dingxin
    timecost[6] += (RealGeom)cu_esp;
#endif */
	
#if (defined FaceColoring)		
		cuLoadFluxColor(grid, NULL, NULL);				
#elif (defined Atomic)
		cuLoadFluxAtomic(NULL, NULL);
#elif (defined GroupColor)
		if (grid->GroupColorSuccess) {
			//cout << "grid->GroupColorSuccess: " << endl; 
			//exit(0);
			cuLoadFluxGroupColor(grid, NULL, NULL);
		}
		else{
			cout << "grid->GroupColorFail: " << endl;
			cuLoadFluxAtomic(NULL, NULL);
		}
#else
		// Reduction:
		// Reduction ShareMemory will be included here.
		cuLoadFlux(NULL, NULL);
#endif
		ns  = ne;   
#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    timecost[1] = timecost[1] + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemInvis, 0); 
    timeuseTemInvis = (RealGeom) 1000000*(endtimeTemInvis.tv_sec - starttimeTemInvis.tv_sec) + endtimeTemInvis.tv_usec - starttimeTemInvis.tv_usec;
    timecost[1] += timeuseTemInvis;
    timeuseTemInvis /= 1000000.0;
    time_invis += timeuseTemInvis;
#endif
#endif

#ifdef FS_CUDA_DEBUG_NS_Flux
	cuMemoryPreparaInVisFluxDebug2(grid);
#endif
}

#ifdef LOOPMERGE
__global__ void gpuInviscidFlux_merge_bface(const double* q, const int* f2c,
		const double* xfn, const double* yfn, const double* zfn, const double* vgn, const IntType* type_bcr,
		const int steady, const int nTFace, const int nBFace, const int nTCell, const RealFlow* limit, 
		const RealFlow* dqdx, const RealFlow* dqdy, const RealFlow* dqdz, const RealGeom *xfc,
		const RealGeom *yfc, const RealGeom *zfc, const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, 
		double* flux, const double* area, const int* IsShockFace, const int* IsNormalFace,
		const double gamm1, const double p_bar, const double alf_l, const double alf_n,
		const int EntropyCorType){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i >= nBFace)
		return;
	IntType  j, c1, c2, count, type;
	RealGeom dx, dy, dz;
	RealFlow trho, tpre;
	RealFlow vn, tem;
	IntType nvar = 5;
	IntType Cell = nTCell + nBFace;
	RealFlow ql[5], qr[5];
	count = 2 * i;
    c1 = f2c[count++];
    c2 = f2c[count];
	for (IntType n = 0; n < nvar; n++) {
        ql[n] = q[n*Cell + c1];
        qr[n] = q[n*Cell + c2];
    }

	type = type_bcr[i];
	if (limit != NULL) {
		if (type == INTERFACE || type == SYMM){
			// Left one
			dx     = xfc[i] - xcc[c1];
			dy     = yfc[i] - ycc[c1];
			dz     = zfc[i] - zcc[c1];
        
			trho   = ql[0] + limit[0*Cell + c1]*(dqdx[0*Cell + c1]*dx + dqdy[0*Cell + c1]*dy + dqdz[0*Cell + c1]*dz);
			tpre   = ql[4] + limit[4*Cell + c1]*(dqdx[4*Cell + c1]*dx + dqdy[4*Cell + c1]*dy + dqdz[4*Cell + c1]*dz);
		
			if(trho > 0 && tpre > -p_bar){
				ql[0]  = trho;
				ql[1] += limit[1*Cell + c1]*(dqdx[1*Cell + c1]*dx + dqdy[1*Cell + c1]*dy + dqdz[1*Cell + c1]*dz);
				ql[2] += limit[2*Cell + c1]*(dqdx[2*Cell + c1]*dx + dqdy[2*Cell + c1]*dy + dqdz[2*Cell + c1]*dz);
				ql[3] += limit[3*Cell + c1]*(dqdx[3*Cell + c1]*dx + dqdy[3*Cell + c1]*dy + dqdz[3*Cell + c1]*dz);
				ql[4]  = tpre;
			}
		
			if (type == INTERFACE){
				// Right one
				dx     = xfc[i] - xcc[c2];
				dy     = yfc[i] - ycc[c2];
				dz     = zfc[i] - zcc[c2];
    
				trho   = qr[0] + limit[0*Cell + c2]*(dqdx[0*Cell + c2]*dx + dqdy[0*Cell + c2]*dy + dqdz[0*Cell + c2]*dz);
				tpre   = qr[4] + limit[4*Cell + c2]*(dqdx[4*Cell + c2]*dx + dqdy[4*Cell + c2]*dy + dqdz[4*Cell + c2]*dz);
				if(trho > 0 && tpre > -p_bar){
					qr[0]  = trho;
					qr[1] += limit[1*Cell + c2]*(dqdx[1*Cell + c2]*dx + dqdy[1*Cell + c2]*dy + dqdz[1*Cell + c2]*dz);
					qr[2] += limit[2*Cell + c2]*(dqdx[2*Cell + c2]*dx + dqdy[2*Cell + c2]*dy + dqdz[2*Cell + c2]*dz);
					qr[3] += limit[3*Cell + c2]*(dqdx[3*Cell + c2]*dx + dqdy[3*Cell + c2]*dy + dqdz[3*Cell + c2]*dz);
					qr[4]  = tpre;
				}
			}
		}
	} // limit != NULL
	
	if (type == INTERFACE){
		//continue;
	}
	else if(type == SYMM){
        //rho
        qr[0] = ql[0];
        //u,v,w
        vn = ql[1] * xfn[i] + ql[2] * yfn[i] + ql[3] * zfn[i];
        if(!steady){         //zhyb:对称面vgn为0，此处本来可以不考虑。但是在粘性计算时，有时可能会采用对称边界条件表示无粘的物面，
            vn -= vgn[i];    //因此在此需要加上非定常的情况
        }
        qr[1] = ql[1] - 2.0*vn*xfn[i];
        qr[2] = ql[2] - 2.0*vn*yfn[i];
        qr[3] = ql[3] - 2.0*vn*zfn[i];
        //p
        qr[4] = ql[4];
    }
    else{
        for (j = 0; j < 5; j++) {
            tem = 0.5 * (q[j*Cell + c1] + q[j*Cell + c2]);
            ql[j] = tem;
            qr[j] = tem;
        }
    }
		
	int  ni;
    double rho_a, u_a, v_a, w_a, h_a, c_a, c2_a, vn_a, q2;
    double vn_l, et_l, ht_l, vn_r, et_r, ht_r;
    double tmp0, tmp1, tmp2, alpha1, alpha2, alpha3, eigv1, eigv2, eigv3;
    double drho, du, dv, dw, dp, dvn, dq2;
    double areax, areay, areaz;
    double spectral, epsaa, epsbb, epscc, epsa_r;
    double u_vgn, v_vgn, w_vgn;
    ni = i;
    areax = xfn[i];
    areay = yfn[i];
    areaz = zfn[i];

    // Total energy
    et_l = (ql[4] + p_bar) / gamm1 + 0.5 * ql[0] *
        (ql[1] * ql[1] + ql[2] * ql[2] + ql[3] * ql[3]);
    et_r = (qr[4] + p_bar) / gamm1 + 0.5 * qr[0] *
        (qr[1] * qr[1] + qr[2] * qr[2] + qr[3] * qr[3]);
    ht_l = et_l + ql[4] + p_bar;
    ht_r = et_r + qr[4] + p_bar;

    // Full flux
    vn_l = areax * ql[1] + areay * ql[2] + areaz * ql[3];
    vn_r = areax * qr[1] + areay * qr[2] + areaz * qr[3];
    if (!steady) {   //unsteady
        vn_l -= vgn[ni];
        vn_r -= vgn[ni];
    }

    tmp0 = vn_l * ql[0];
    tmp1 = vn_r * qr[0];
    flux[0*nTFace + i] = tmp0 + tmp1;
    flux[1*nTFace + i] = tmp0 * ql[1] + areax * ql[4]
        + tmp1 * qr[1] + areax * qr[4];
    flux[2*nTFace + i] = tmp0 * ql[2] + areay * ql[4]
        + tmp1 * qr[2] + areay * qr[4];
    flux[3*nTFace + i] = tmp0 * ql[3] + areaz * ql[4]
        + tmp1 * qr[3] + areaz * qr[4];
    flux[4*nTFace + i] = ht_l * vn_l + ht_r * vn_r;
    if (!steady) flux[4*nTFace + i] += (ql[4] + qr[4] + 2.0 * p_bar) * vgn[ni];   //unsteady, 0.5在最后乘面积的地方

    //采用roe平均计算单元面上的物理量
    tmp0 = sqrt(qr[0] / ql[0]);
    tmp1 = 1.0 / (1.0 + tmp0);
    rho_a = sqrt(qr[0] * ql[0]);
    u_a = (ql[1] + qr[1] * tmp0) * tmp1;
    v_a = (ql[2] + qr[2] * tmp0) * tmp1;
    w_a = (ql[3] + qr[3] * tmp0) * tmp1;
    vn_a = u_a * areax + v_a * areay + w_a * areaz;
    h_a = (ht_l / ql[0] + ht_r / qr[0] * tmp0) * tmp1;

    q2 = 0.5 * (u_a * u_a + v_a * v_a + w_a * w_a);
    c2_a = gamm1 * (h_a - q2);
    c2_a = fabs(c2_a);
    c_a = sqrt(c2_a);

    if (steady) {
        eigv1 = fabs(vn_a);
        eigv2 = fabs(vn_a + c_a);
        eigv3 = fabs(vn_a - c_a);
    }
    else {   //unsteady
        eigv1 = fabs(vn_a - vgn[i]);
        eigv2 = fabs(vn_a - vgn[i] + c_a);
        eigv3 = fabs(vn_a - vgn[i] - c_a);
    }

    //Entropy fix          
    if (EntropyCorType == 3) {
        epsa_r = alf_l;
    }
    else if (EntropyCorType == 4) {
        if (IsNormalFace[ni] && IsShockFace[i] == 0) {
            epsa_r = 0.01 * alf_l;
            //epsa_r = 0.0002;
        }
        else {
            epsa_r = alf_l;
        }
    }
    else {
		//exit(0);
        //(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }

    //cfl3d form
    if (steady) {
        spectral = fabs(u_a) + fabs(v_a) + fabs(w_a) + c_a;
    }
    else {
        u_vgn = vgn[ni] * xfn[i];
        v_vgn = vgn[ni] * yfn[i];
        w_vgn = vgn[ni] * zfn[i];
        spectral = fabs(u_a - u_vgn) + fabs(v_a - v_vgn) + fabs(w_a - w_vgn) + c_a;
    }
    epsaa = epsa_r * spectral;
    epsbb = 0.25 / max(epsaa, TINY);
    epscc = 2.0 * epsaa;
    if (eigv1 < epscc) eigv1 = eigv1 * eigv1 * epsbb + epsaa;
    if (eigv2 < epscc) eigv2 = eigv2 * eigv2 * epsbb + epsaa;
    if (eigv3 < epscc) eigv3 = eigv3 * eigv3 * epsbb + epsaa;

    drho = qr[0] - ql[0];
    du = qr[1] - ql[1];
    dv = qr[2] - ql[2];
    dw = qr[3] - ql[3];
    dp = qr[4] - ql[4];
    dvn = vn_r - vn_l;

    dq2 = u_a * du + v_a * dv + w_a * dw;

    tmp0 = dp / c2_a;
    tmp1 = rho_a * dvn / c_a;
    alpha1 = (drho - tmp0) * eigv1;
    alpha2 = 0.5 * (tmp0 + tmp1) * eigv2;
    alpha3 = 0.5 * (tmp0 - tmp1) * eigv3;

    tmp0 = alpha1 + alpha2 + alpha3;
    tmp1 = eigv1 * rho_a;
    tmp2 = -tmp1 * dvn + (alpha2 - alpha3) * c_a;
    flux[0*nTFace + i] -= tmp0;
    flux[1*nTFace + i] -= tmp0 * u_a + tmp1 * du + tmp2 * areax;
    flux[2*nTFace + i] -= tmp0 * v_a + tmp1 * dv + tmp2 * areay;
    flux[3*nTFace + i] -= tmp0 * w_a + tmp1 * dw + tmp2 * areaz;
    flux[4*nTFace + i] -= alpha1 * q2 + (alpha2 + alpha3) * h_a + tmp1 * dq2 + tmp2 * vn_a;

    tmp0 = 0.5 * area[i];
    flux[0*nTFace + i] *= tmp0;
    flux[1*nTFace + i] *= tmp0;
    flux[2*nTFace + i] *= tmp0;
    flux[3*nTFace + i] *= tmp0;
    flux[4*nTFace + i] *= tmp0;
	
}

__global__ void gpuInviscidFlux_merge_iface(const double* q, const int* f2c,
		const double* xfn, const double* yfn, const double* zfn, const double* vgn,
		const int steady, const int nTFace, const int nBFace, const int nTCell, const RealFlow* limit, 
		const RealFlow* dqdx, const RealFlow* dqdy, const RealFlow* dqdz, const RealGeom *xfc,
		const RealGeom *yfc, const RealGeom *zfc, const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, 
		double* flux, const double* area, const int* IsShockFace, const int* IsNormalFace,
		const double gamm1, const double p_bar, const double alf_l, const double alf_n,
		const int EntropyCorType){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x + nBFace;
	if (i >= nTFace)
		return;
	IntType  c1, c2, count;
	RealGeom dx, dy, dz;
	RealFlow trho, tpre;
	IntType nvar = 5;
	IntType Cell = nTCell + nBFace;
	RealFlow ql[5], qr[5];
	count = 2 * i;
    c1 = f2c[count++];
    c2 = f2c[count];
	for (IntType n = 0; n < nvar; n++) {
        ql[n] = q[n*Cell + c1];
        qr[n] = q[n*Cell + c2];
    }

	if (limit != NULL) {
		// Left one
		dx     = xfc[i] - xcc[c1];
		dy     = yfc[i] - ycc[c1];
		dz     = zfc[i] - zcc[c1];
        
		trho   = ql[0] + limit[0*Cell + c1]*(dqdx[0*Cell + c1]*dx + dqdy[0*Cell + c1]*dy + dqdz[0*Cell + c1]*dz);
		tpre   = ql[4] + limit[4*Cell + c1]*(dqdx[4*Cell + c1]*dx + dqdy[4*Cell + c1]*dy + dqdz[4*Cell + c1]*dz);
		
		if(trho > 0 && tpre > -p_bar){
			ql[0]  = trho;
			ql[1] += limit[1*Cell + c1]*(dqdx[1*Cell + c1]*dx + dqdy[1*Cell + c1]*dy + dqdz[1*Cell + c1]*dz);
			ql[2] += limit[2*Cell + c1]*(dqdx[2*Cell + c1]*dx + dqdy[2*Cell + c1]*dy + dqdz[2*Cell + c1]*dz);
			ql[3] += limit[3*Cell + c1]*(dqdx[3*Cell + c1]*dx + dqdy[3*Cell + c1]*dy + dqdz[3*Cell + c1]*dz);
			ql[4]  = tpre;
		}
		// Right one
		dx     = xfc[i] - xcc[c2];
		dy     = yfc[i] - ycc[c2];
		dz     = zfc[i] - zcc[c2];
    
		trho   = qr[0] + limit[0*Cell + c2]*(dqdx[0*Cell + c2]*dx + dqdy[0*Cell + c2]*dy + dqdz[0*Cell + c2]*dz);
		tpre   = qr[4] + limit[4*Cell + c2]*(dqdx[4*Cell + c2]*dx + dqdy[4*Cell + c2]*dy + dqdz[4*Cell + c2]*dz);
		if(trho > 0 && tpre > -p_bar){
			qr[0]  = trho;
			qr[1] += limit[1*Cell + c2]*(dqdx[1*Cell + c2]*dx + dqdy[1*Cell + c2]*dy + dqdz[1*Cell + c2]*dz);
			qr[2] += limit[2*Cell + c2]*(dqdx[2*Cell + c2]*dx + dqdy[2*Cell + c2]*dy + dqdz[2*Cell + c2]*dz);
			qr[3] += limit[3*Cell + c2]*(dqdx[3*Cell + c2]*dx + dqdy[3*Cell + c2]*dy + dqdz[3*Cell + c2]*dz);
			qr[4]  = tpre;
		}
	} // limit != NULL
		
	int  ni;
    double rho_a, u_a, v_a, w_a, h_a, c_a, c2_a, vn_a, q2;
    double vn_l, et_l, ht_l, vn_r, et_r, ht_r;
    double tmp0, tmp1, tmp2, alpha1, alpha2, alpha3, eigv1, eigv2, eigv3;
    double drho, du, dv, dw, dp, dvn, dq2;
    double areax, areay, areaz;
    double spectral, epsaa, epsbb, epscc, epsa_r;
    double u_vgn, v_vgn, w_vgn;
    ni = i;
    areax = xfn[i];
    areay = yfn[i];
    areaz = zfn[i];

    // Total energy
    et_l = (ql[4] + p_bar) / gamm1 + 0.5 * ql[0] *
        (ql[1] * ql[1] + ql[2] * ql[2] + ql[3] * ql[3]);
    et_r = (qr[4] + p_bar) / gamm1 + 0.5 * qr[0] *
        (qr[1] * qr[1] + qr[2] * qr[2] + qr[3] * qr[3]);
    ht_l = et_l + ql[4] + p_bar;
    ht_r = et_r + qr[4] + p_bar;

    // Full flux
    vn_l = areax * ql[1] + areay * ql[2] + areaz * ql[3];
    vn_r = areax * qr[1] + areay * qr[2] + areaz * qr[3];
    if (!steady) {   //unsteady
        vn_l -= vgn[ni];
        vn_r -= vgn[ni];
    }

    tmp0 = vn_l * ql[0];
    tmp1 = vn_r * qr[0];
    flux[0*nTFace + i] = tmp0 + tmp1;
    flux[1*nTFace + i] = tmp0 * ql[1] + areax * ql[4]
        + tmp1 * qr[1] + areax * qr[4];
    flux[2*nTFace + i] = tmp0 * ql[2] + areay * ql[4]
        + tmp1 * qr[2] + areay * qr[4];
    flux[3*nTFace + i] = tmp0 * ql[3] + areaz * ql[4]
        + tmp1 * qr[3] + areaz * qr[4];
    flux[4*nTFace + i] = ht_l * vn_l + ht_r * vn_r;
    if (!steady) flux[4*nTFace + i] += (ql[4] + qr[4] + 2.0 * p_bar) * vgn[ni];   //unsteady, 0.5在最后乘面积的地方

    //采用roe平均计算单元面上的物理量
    tmp0 = sqrt(qr[0] / ql[0]);
    tmp1 = 1.0 / (1.0 + tmp0);
    rho_a = sqrt(qr[0] * ql[0]);
    u_a = (ql[1] + qr[1] * tmp0) * tmp1;
    v_a = (ql[2] + qr[2] * tmp0) * tmp1;
    w_a = (ql[3] + qr[3] * tmp0) * tmp1;
    vn_a = u_a * areax + v_a * areay + w_a * areaz;
    h_a = (ht_l / ql[0] + ht_r / qr[0] * tmp0) * tmp1;

    q2 = 0.5 * (u_a * u_a + v_a * v_a + w_a * w_a);
    c2_a = gamm1 * (h_a - q2);
    c2_a = fabs(c2_a);
    c_a = sqrt(c2_a);

    if (steady) {
        eigv1 = fabs(vn_a);
        eigv2 = fabs(vn_a + c_a);
        eigv3 = fabs(vn_a - c_a);
    }
    else {   //unsteady
        eigv1 = fabs(vn_a - vgn[i]);
        eigv2 = fabs(vn_a - vgn[i] + c_a);
        eigv3 = fabs(vn_a - vgn[i] - c_a);
    }

    //Entropy fix          
    if (EntropyCorType == 3) {
        epsa_r = alf_l;
    }
    else if (EntropyCorType == 4) {
        if (IsNormalFace[ni] && IsShockFace[i] == 0) {
            epsa_r = 0.01 * alf_l;
            //epsa_r = 0.0002;
        }
        else {
            epsa_r = alf_l;
        }
    }
    else {
		//exit(0);
        //(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }

    //cfl3d form
    if (steady) {
        spectral = fabs(u_a) + fabs(v_a) + fabs(w_a) + c_a;
    }
    else {
        u_vgn = vgn[ni] * xfn[i];
        v_vgn = vgn[ni] * yfn[i];
        w_vgn = vgn[ni] * zfn[i];
        spectral = fabs(u_a - u_vgn) + fabs(v_a - v_vgn) + fabs(w_a - w_vgn) + c_a;
    }
    epsaa = epsa_r * spectral;
    epsbb = 0.25 / max(epsaa, TINY);
    epscc = 2.0 * epsaa;
    if (eigv1 < epscc) eigv1 = eigv1 * eigv1 * epsbb + epsaa;
    if (eigv2 < epscc) eigv2 = eigv2 * eigv2 * epsbb + epsaa;
    if (eigv3 < epscc) eigv3 = eigv3 * eigv3 * epsbb + epsaa;

    drho = qr[0] - ql[0];
    du = qr[1] - ql[1];
    dv = qr[2] - ql[2];
    dw = qr[3] - ql[3];
    dp = qr[4] - ql[4];
    dvn = vn_r - vn_l;

    dq2 = u_a * du + v_a * dv + w_a * dw;

    tmp0 = dp / c2_a;
    tmp1 = rho_a * dvn / c_a;
    alpha1 = (drho - tmp0) * eigv1;
    alpha2 = 0.5 * (tmp0 + tmp1) * eigv2;
    alpha3 = 0.5 * (tmp0 - tmp1) * eigv3;

    tmp0 = alpha1 + alpha2 + alpha3;
    tmp1 = eigv1 * rho_a;
    tmp2 = -tmp1 * dvn + (alpha2 - alpha3) * c_a;
    flux[0*nTFace + i] -= tmp0;
    flux[1*nTFace + i] -= tmp0 * u_a + tmp1 * du + tmp2 * areax;
    flux[2*nTFace + i] -= tmp0 * v_a + tmp1 * dv + tmp2 * areay;
    flux[3*nTFace + i] -= tmp0 * w_a + tmp1 * dw + tmp2 * areaz;
    flux[4*nTFace + i] -= alpha1 * q2 + (alpha2 + alpha3) * h_a + tmp1 * dq2 + tmp2 * vn_a;

    tmp0 = 0.5 * area[i];
    flux[0*nTFace + i] *= tmp0;
    flux[1*nTFace + i] *= tmp0;
    flux[2*nTFace + i] *= tmp0;
    flux[3*nTFace + i] *= tmp0;
    flux[4*nTFace + i] *= tmp0;
	
}

void cuInviscidFlux_merge(PolyGrid *grid, RealFlow **limit, IntType level){
#ifdef FS_CUDA_DEBUG_NS_Flux
	cuMemoryPreparaInVisFluxDebug(grid, limit);
#endif
	
#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemInvis, endtimeTemInvis;
    double timeuseTemInvis;
    gettimeofday(&starttimeTemInvis, 0); 
#endif
#endif

    // Get parameters   
    RealFlow gam, alf_l, alf_n;
    grid->GetData(&gam,    REAL_FLOW, 1, "gam");
    grid->GetData(&alf_l,  REAL_FLOW, 1, "alf_l");
    grid->GetData(&alf_n,  REAL_FLOW, 1, "alf_n");    
	
    // Allocate temporary memories for flux
    RealFlow *flux[5];

	RealFlow gamm1;
	IntType* IsShockFace = 0;
	IntType EntropyCorType = 4;

	grid->GetData(&EntropyCorType, INT, 1, "EntropyCorType");
	gamm1 = gam - 1.0;
	if (EntropyCorType == 4) {
		// shock face or not
		cuCalIsShockFace(grid, IsShockFace);
	}
	
	/* cudaEvent_t cu_start, cu_stop;
	float cu_esp;
	cudaEventCreate(&cu_start);
	cudaEventCreate(&cu_stop);
	cudaEventRecord(cu_start, 0); */
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuInviscidFlux_merge_bface <<< blocksPerGrid, threadsPerBlock >>> (gq, gf2c, gxfn, gyfn, gzfn,
								gvgn, gtype_bcr, gsteady, gnTFace, gnBFace, gnTCell, glimit, gdqdx, gdqdy, gdqdz,
								gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gflux, garea, gIsShockFace, gIsNormalFace,
								gamm1, gp_bar, alf_l, alf_n, EntropyCorType);

	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuInviscidFlux_merge_iface <<< blocksPerGrid, threadsPerBlock >>> (gq, gf2c, gxfn, gyfn, gzfn,
								gvgn, gsteady, gnTFace, gnBFace, gnTCell, glimit, gdqdx, gdqdy, gdqdz,
								gxfc, gyfc, gzfc, gxcc, gycc, gzcc, gflux, garea, gIsShockFace, gIsNormalFace,
								gamm1, gp_bar, alf_l, alf_n, EntropyCorType);
								
	/* cudaEventRecord(cu_stop, 0);
	cudaEventSynchronize(cu_stop);
	
	cudaEventElapsedTime(&cu_esp, cu_start, cu_stop);							

#ifdef TIMECOST//dingxin
    timecost[8] += (RealGeom)cu_esp;
#endif */

	// Load the fluxes to residuals
	RealFlow *res[5];
	
#if (defined FaceColoring)		
	cuLoadFluxColor(grid, res, flux);				
#elif (defined Atomic)
	cuLoadFluxAtomic(res, flux);
#elif (defined GroupColor)
	if (grid->GroupColorSuccess) {
		//cout << "grid->GroupColorSuccess: " << endl; 
		//exit(0);
		cuLoadFluxGroupColor(grid, res, flux);
	}
	else{
		cout << "grid->GroupColorFail: " << endl; 
		cuLoadFluxAtomic(res, flux);
	}
#else
	// Reduction:
	// Reduction ShareMemory will be included here.
	cuLoadFlux(res, flux);
#endif


#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    timecost[1] = timecost[1] + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemInvis, 0); 
    timeuseTemInvis = (RealGeom) 1000000*(endtimeTemInvis.tv_sec - starttimeTemInvis.tv_sec) + endtimeTemInvis.tv_usec - starttimeTemInvis.tv_usec;
    timecost[1] += timeuseTemInvis;
    timeuseTemInvis /= 1000000.0;
    time_invis += timeuseTemInvis;
#endif
#endif

#ifdef FS_CUDA_DEBUG_NS_Flux
	cuMemoryPreparaInVisFluxDebug2(grid);
#endif
}
#endif // ~LOOPMERGE

__device__ double atomicAddSM35LoadFlux(double* address, double val)
{
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
}

__device__ double atomicExchSM35res(double* address, double val){
	
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
        } while (assumed != old);
        return __longlong_as_double(old);
}




















