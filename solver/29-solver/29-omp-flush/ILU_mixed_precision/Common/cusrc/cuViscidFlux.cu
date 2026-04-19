#include <stdio.h>
#include <iostream>
#include <cmath>

#include <number_type.h>
#include <grid_patch_type.h>
#include <data_pool.h>
#include <zone.h>
#include <constant.h>

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif

#include "cuGradientQ_Gauss.cuh"
#include <cuInviscidFlux.cuh>
#include <cuViscidFlux.cuh>
#include <cuData.cuh>
#include <cuErrorReturn.cuh>
#include "cuLUSGS.cuh"

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

__global__ void gpuCalDeriWeight(const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, const IntType *f2c, 
							const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc, const RealGeom *xfn, 
							const RealGeom *yfn, const RealGeom *zfn, const RealGeom *vol, RealGeom *deltl, 
							RealGeom *deltr, const IntType nTFace, const IntType key){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType c1, c2, face;
	RealGeom delt1, delt2, delta;
	if(i < nTFace){
		c1 = f2c[2*i];
        c2 = f2c[2*i + 1];
		face = i;
 
        //Left
        if(key == 1){  // distance weight
            delt1 = sqrt((xcc[c1] - xfc[face])*(xcc[c1] - xfc[face])
                  +      (ycc[c1] - yfc[face])*(ycc[c1] - yfc[face])
                  +      (zcc[c1] - zfc[face])*(zcc[c1] - zfc[face]));        
        }else if(key == 2){  //normal distance weight
            delt1 = fabs((xcc[c1] - xfc[face])*xfn[face]
                  +      (ycc[c1] - yfc[face])*yfn[face]
                  +      (zcc[c1] - zfc[face])*zfn[face]);
        }else if(key == 3){  //volume weight
            delt1 = vol[c1];
        }
 
        // Right
        if(key == 1){   // distance weight
            delt2 = sqrt((xcc[c2] - xfc[face])*(xcc[c2] - xfc[face])
                  +      (ycc[c2] - yfc[face])*(ycc[c2] - yfc[face])
                  +      (zcc[c2] - zfc[face])*(zcc[c2] - zfc[face]));
        }else if(key == 2){   //normal distance weight
            delt2 = fabs((xcc[c2] - xfc[face])*xfn[face]
                  +      (ycc[c2] - yfc[face])*yfn[face]
                  +      (zcc[c2] - zfc[face])*zfn[face]);
        }else if(key == 3){  //volume weight
            delt2 = vol[c2];
        }
 
        delta    = 1./(delt1 + delt2 + TINY);
        deltl[i] = delt2*delta;
        deltr[i] = delt1*delta;
	}
	
}

void cuCalDeriWeight(RealGeom *deltl, RealGeom *deltr, IntType key){

	//Transfer host data into device:
	
	IntType blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuCalDeriWeight <<< blocksPerGrid, threadsPerBlock >>> (gxcc, gycc, gzcc, gf2c, gxfc, gyfc, gzfc, 
														gxfn, gyfn, gzfn, gvol, gdeltl, gdeltr, gnTFace, key);
		
	
}

__global__ void gpuCalVisHeatFace_averageLaminar(RealFlow *visc_f, RealFlow *heat_f, const RealFlow *vis_l, const IntType *f2c, 
												const RealFlow heat, const IntType nTFace){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTFace){
		IntType      c1, c2;
		c1 = f2c[2*i];
		c2 = f2c[2*i + 1];
		visc_f[i] = 0.5*(vis_l[c1]+vis_l[c2]);
		heat_f[i] = heat*visc_f[i];
	}
}

__global__ void gpuCalVisHeatFace_averageTurbulentSA(RealFlow *visc_f, RealFlow *heat_f, const RealFlow *vis_t, const IntType *f2c, 
												const RealFlow heat, const IntType nTFace){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTFace){
		IntType      c1, c2;
		RealFlow     tmp;
		c1 = f2c[2*i];
		c2 = f2c[2*i + 1];
 
		tmp        = 0.5*(vis_t[c1] + vis_t[c2]);
		visc_f[i] += tmp;
		heat_f[i] += heat*tmp;

	}
																																			
}

void cuCalVisHeatFace_average(PolyGrid *grid, RealFlow *vis_l, RealFlow *visc_f, RealFlow *heat_f){
	
	// Get parameters
    IntType  vis_mode, cond_comp = 1;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    grid->GetData(&cond_comp, INT, 1, "comp",0);

    // Get specific heat ratio, gas constant, cp
    RealFlow gam, gascon, cp;
    grid->GetData(&gam, REAL_FLOW, 1, "gam");
    grid->GetData(&gascon, REAL_FLOW, 1, "gascon");
    cp = gascon*gam/(gam - 1.);
    if(cond_comp == 0)grid->GetData(&cp, REAL_FLOW, 1, "cp");

    // Get viscosity, Prandtl number
    RealFlow prl, heat;
    grid->GetData(&prl, REAL_FLOW, 1, "prl");
    heat = cp/prl;
	
	IntType *f2c = grid->Getf2c();
    // Laminar Flows
	IntType blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
	
	//HANDLE_API_ERR(cudaMemcpy(gvis_l, vis_l, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	gpuCalVisHeatFace_averageLaminar <<< blocksPerGrid, threadsPerBlock >>> (gvisc_f, gheat_f, gvis_l, gf2c, heat, gnTFace);
		
    //Turbulent viscosity (Eddy viscosity?)
    if(vis_mode == S_A_MODEL) {
		IntType      n      = grid->GetNTCell() + grid->GetNBFace();
		
		// Note: the size of vis_t
		RealFlow *vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
		RealFlow prt;
		
		grid->GetData(&prt, REAL_FLOW, 1, "prt");
		heat  = cp/prt; 
		
		//HANDLE_API_ERR(cudaMemcpy(gvis_t, vis_t, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
		
		gpuCalVisHeatFace_averageTurbulentSA <<< blocksPerGrid, threadsPerBlock >>> (gvisc_f, gheat_f, gvis_t, gf2c, heat, gnTFace);
		
    } 
	//HANDLE_API_ERR(cudaMemcpy(visc_f, gvisc_f, gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	//HANDLE_API_ERR(cudaMemcpy(heat_f, gheat_f, gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
}

__global__ void gpuCalVeloandTFace_average(const RealFlow *q, const RealFlow *t, const IntType *f2c, RealFlow *vel_f, 
										 RealFlow *t_f,	const IntType nTFace, const int nBFace, const int nTCell){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTFace){
		IntType c1, c2;
		IntType Cell = nTCell + nBFace;
		c1 = f2c[2*i];
        c2 = f2c[2*i + 1];
		
		vel_f[0*nTFace + i] = 0.5*(q[1*Cell + c1] + q[1*Cell + c2]);
        vel_f[1*nTFace + i] = 0.5*(q[2*Cell + c1] + q[2*Cell + c2]);
        vel_f[2*nTFace + i] = 0.5*(q[3*Cell + c1] + q[3*Cell + c2]);
		
		t_f[i] = 0.5*(t[c1] + t[c2]);
	}		
																																							
}

void cuCalVeloandTFace_average(PolyGrid *grid, RealFlow *vel_f[3], RealFlow *vel[3],
                        RealFlow *t_f, RealFlow *t){
    
	IntType blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
	
	// HANDLE_API_ERR(cudaMemcpy(gt, t, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	gpuCalVeloandTFace_average <<< blocksPerGrid, threadsPerBlock >>> (gq, gt, gf2c, gvel_f, gt_f, gnTFace, gnBFace, gnTCell);
	
	//HANDLE_API_ERR(cudaMemcpy(vel_f[0], gvel_f, 3*gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	//HANDLE_API_ERR(cudaMemcpy(t_f, gt_f, gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));		
}

__global__ void gpuCalVisFluxTest(const RealFlow *q, const RealFlow *t, const RealFlow *dqdx, const RealFlow *dqdy, const RealFlow *dqdz, 
								const RealFlow *dtdx, const RealFlow *dtdy, const RealFlow *dtdz, const RealFlow *vel_f, 
								const RealFlow *t_f, const RealFlow *visc_f, const RealFlow *heat_f, const RealFlow *deltl, 
								const RealFlow *deltr, RealFlow *flux, const IntType *f2c, const RealGeom *area, 
								const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc, const RealGeom *xcc, const RealGeom *ycc, 
								const RealGeom *zcc, const RealGeom *xfn, const RealGeom *yfn, const RealGeom *zfn, 
								const RealGeom *facecentroidskewness, const IntType *type_bcr, const RealGeom *tw_bcr,
								const IntType level, IntType warn, const RealGeom two3, const RealGeom BadFaceAngle, 
								const IntType nTFace, const IntType nBFace, const IntType nTCell
								){
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < nTFace){
		IntType c1, c2, type;
		RealGeom areax, areay, areaz;
		RealFlow umid, vmid, wmid, tmid, d_vis, heat_con, tw;
		RealFlow t1x, t1y, t1z, t2x, t2y, t2z;
		RealFlow dtmp, d1, d2, u1, u2, v1, v2, w1, w2, t1, t2, x1, x2, y1, y2, z1, z2;
		RealFlow dud1, dud2, dvd1, dvd2, dwd1, dwd2, dtd1, dtd2;
		RealFlow dudn, dvdn, dwdn, dtdn;
		RealFlow dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz;
		RealFlow txx, tyy, tzz, txy, txz, tyz;
		RealFlow dudt1, dvdt1, dwdt1, dudt2, dvdt2, dwdt2;
		RealFlow angle1, angle2;
		RealFlow delta;
		IntType Cell = nBFace + nTCell;
		
		IntType fluxzeroflag = 0;
		
		c1 = f2c[2*i];
        c2 = f2c[2*i + 1];
		
		areax = xfn[i];
        areay = yfn[i];
        areaz = zfn[i];
		
		// Get first tangential vector on the face
        if(areax != 0.) {
            t1x =  areay;
            t1y = -areax;
            t1z =  0.;
        } else if(areay != 0.) {
            t1x = -areay;
            t1y =  areax;
            t1z =  0.;
        } else if(areaz != 0.) {
            t1x =  0.;
            t1y = -areaz;
            t1z =  areay;
        } else {
			fluxzeroflag = 1;
			// fluxzeroflag was designed to execute following code:
			/*
            if(warn) printf("Warninng: %ldth Face is singular\n", (long)face);
            flux[0][i] = 0.0;
            flux[1][i] = 0.0;
            flux[2][i] = 0.0;
            flux[3][i] = 0.0;
            flux[4][i] = 0.0;
            if(warn !=0 && ne == nTFace) warn = 0;
            continue;
			*/
        }
		
		if(fluxzeroflag == 0){
		// normalize the tangential vector
        dtmp = sqrt(t1x*t1x + t1y*t1y + t1z*t1z);
        t1x /= dtmp;
        t1y /= dtmp;
        t1z /= dtmp;
        
        // Get second tangential vector by cross dot t1 to normal
        t2x = areay*t1z - areaz*t1y;
        t2y = areaz*t1x - areax*t1z;
        t2z = areax*t1y - areay*t1x;
        
        // positions
        x1 = xcc[c1]  - xfc[i];
        y1 = ycc[c1]  - yfc[i];
        z1 = zcc[c1]  - zfc[i];
        x2 = xcc[c2]  - xfc[i];
        y2 = ycc[c2]  - yfc[i];
        z2 = zcc[c2]  - zfc[i];
        d1 = x1*areax + y1*areay + z1*areaz;
        d2 = x2*areax + y2*areay + z2*areaz;
        
        dtmp = -d1/(sqrt(x1*x1+ y1*y1 + z1*z1) + TINY);
        if(dtmp >  1.0) dtmp =  1.0;
        if(dtmp < -1.0) dtmp = -1.0;
        angle1 = asin(dtmp)*180.0/PI;
        
        dtmp = d2/(sqrt(x2*x2+ y2*y2 + z2*z2) + TINY);
        if(dtmp >  1.0) dtmp =  1.0;
        if(dtmp < -1.0) dtmp = -1.0;
        angle2 = asin(dtmp)*180.0/PI;
        
        // quentities at points 1 and 2
        u1   = q[1*Cell + c1];
        v1   = q[2*Cell + c1];
        w1   = q[3*Cell + c1];
        t1   = t[c1];
        u2   = q[1*Cell + c2];
        v2   = q[2*Cell + c2];
        w2   = q[3*Cell + c2];
        t2   = t[c2];
        umid = 0.5*(u1 + u2);
        vmid = 0.5*(v1 + v2);
        wmid = 0.5*(w1 + w2);
        tmid = 0.5*(t1 + t2);
        
        // Theroretically, more accurate to include the following terms
        if(angle1 > 10.0 && angle2 > 10.0) {
            u1 += dqdx[0*Cell + c1]*(d1*areax - x1) + dqdy[0*Cell + c1]*(d1*areay - y1) + dqdz[0*Cell + c1]*(d1*areaz - z1);
            v1 += dqdx[1*Cell + c1]*(d1*areax - x1) + dqdy[1*Cell + c1]*(d1*areay - y1) + dqdz[1*Cell + c1]*(d1*areaz - z1);
            w1 += dqdx[2*Cell + c1]*(d1*areax - x1) + dqdy[2*Cell + c1]*(d1*areay - y1) + dqdz[2*Cell + c1]*(d1*areaz - z1);
            
            u2 += dqdx[0*Cell + c2]*(d2*areax - x2) + dqdy[0*Cell + c2]*(d2*areay - y2) + dqdz[0*Cell + c2]*(d2*areaz - z2);
            v2 += dqdx[1*Cell + c2]*(d2*areax - x2) + dqdy[1*Cell + c2]*(d2*areay - y2) + dqdz[1*Cell + c2]*(d2*areaz - z2);
            w2 += dqdx[2*Cell + c2]*(d2*areax - x2) + dqdy[2*Cell + c2]*(d2*areay - y2) + dqdz[2*Cell + c2]*(d2*areaz - z2);
            
            t1 += dtdx[c1]*(d1*areax - x1) + dtdy[c1]*(d1*areay - y1) + dtdz[c1]*(d1*areaz - z1);
            t2 += dtdx[c2]*(d2*areax - x2) + dtdy[c2]*(d2*areay - y2) + dtdz[c2]*(d2*areaz - z2);
            if(t1 < TINY) t1  = t[c1];
            if(t2 < TINY) t2  = t[c2];
            
            // quantities at the face
            umid = vel_f[0*nTFace + i];
            vmid = vel_f[1*nTFace + i];
            wmid = vel_f[2*nTFace + i];
            tmid = t_f[i];
        }
        
        dudx  = dqdx[0*Cell + c1]*deltl[i] + dqdx[0*Cell + c2]*deltr[i];
        dudy  = dqdy[0*Cell + c1]*deltl[i] + dqdy[0*Cell + c2]*deltr[i];
        dudz  = dqdz[0*Cell + c1]*deltl[i] + dqdz[0*Cell + c2]*deltr[i];
        dvdx  = dqdx[1*Cell + c1]*deltl[i] + dqdx[1*Cell + c2]*deltr[i];
        dvdy  = dqdy[1*Cell + c1]*deltl[i] + dqdy[1*Cell + c2]*deltr[i];
        dvdz  = dqdz[1*Cell + c1]*deltl[i] + dqdz[1*Cell + c2]*deltr[i];
        dwdx  = dqdx[2*Cell + c1]*deltl[i] + dqdx[2*Cell + c2]*deltr[i];
        dwdy  = dqdy[2*Cell + c1]*deltl[i] + dqdy[2*Cell + c2]*deltr[i];
        dwdz  = dqdz[2*Cell + c1]*deltl[i] + dqdz[2*Cell + c2]*deltr[i];
        
        dudn  = 0.0;
        dvdn  = 0.0;
        dwdn  = 0.0;
        dtdn  = 0.0;
        
        if(angle1 > 0.0 && angle2 > 0.0 && fabs(d1) > TINY && fabs(d2) > TINY) {
            dud1 = (u1 - umid)/d1;
            dvd1 = (v1 - vmid)/d1;
            dwd1 = (w1 - wmid)/d1;
            dtd1 = (t1 - tmid)/d1;
            dud2 = (u2 - umid)/d2;
            dvd2 = (v2 - vmid)/d2;
            dwd2 = (w2 - wmid)/d2;
            dtd2 = (t2 - tmid)/d2;
            dtmp = d1*d1 + d2*d2;
            d1   = d1*d1/dtmp;
            d2   = d2*d2/dtmp;
            dudn = dud1*d1 + dud2*d2;
            dvdn = dvd1*d1 + dvd2*d2;
            dwdn = dwd1*d1 + dwd2*d2;
            dtdn = dtd1*d1 + dtd2*d2;
        }
        
        // dqdt, does not matter too much
        dudt1 = dudx*t1x + dudy*t1y + dudz*t1z;
        dvdt1 = dvdx*t1x + dvdy*t1y + dvdz*t1z;
        dwdt1 = dwdx*t1x + dwdy*t1y + dwdz*t1z;
        dudt2 = dudx*t2x + dudy*t2y + dudz*t2z;
        dvdt2 = dvdx*t2x + dvdy*t2y + dvdz*t2z;
        dwdt2 = dwdx*t2x + dwdy*t2y + dwdz*t2z;
        
        // now true gradients
        dudx  = dudn*areax + dudt1*t1x + dudt2*t2x;
        dudy  = dudn*areay + dudt1*t1y + dudt2*t2y;
        dudz  = dudn*areaz + dudt1*t1z + dudt2*t2z;
        dvdx  = dvdn*areax + dvdt1*t1x + dvdt2*t2x;
        dvdy  = dvdn*areay + dvdt1*t1y + dvdt2*t2y;
        dvdz  = dvdn*areaz + dvdt1*t1z + dvdt2*t2z;
        dwdx  = dwdn*areax + dwdt1*t1x + dwdt2*t2x;
        dwdy  = dwdn*areay + dwdt1*t1y + dwdt2*t2y;
        dwdz  = dwdn*areaz + dwdt1*t1z + dwdt2*t2z;
        if(level==0 && BadFaceAngle>0.0 && facecentroidskewness[i]<BadFaceAngle){
            dudx  = dudn*areax;
            dudy  = dudn*areay;
            dudz  = dudn*areaz;
            dvdx  = dvdn*areax;
            dvdy  = dvdn*areay;
            dvdz  = dvdn*areaz;
            dwdx  = dwdn*areax;
            dwdy  = dwdn*areay;
            dwdz  = dwdn*areaz;
        }
         
        if(i < nBFace){
            type = type_bcr[i];
            if(type!=WALL && type!=SYMM && type!=FAR_FIELD && type!=INTERFACE){  
                delta = sqrt((xcc[c1]-xcc[c2])*(xcc[c1]-xcc[c2]) +
                             (ycc[c1]-ycc[c2])*(ycc[c1]-ycc[c2]) +
                             (zcc[c1]-zcc[c2])*(zcc[c1]-zcc[c2]));
                
                dvdn  = (q[1*Cell + c2]-q[1*Cell + c1])/delta;
                dudx  = dvdn*areax;
                dudy  = dvdn*areay;
                dudz  = dvdn*areaz;
                
                dvdn  = (q[2*Cell + c2]-q[2*Cell + c1])/delta;
                dvdx  = dvdn*areax;
                dvdy  = dvdn*areay;
                dvdz  = dvdn*areaz;
                
                dvdn  = (q[3*Cell + c2]-q[3*Cell + c1])/delta;
                dwdx  = dvdn*areax;
                dwdy  = dvdn*areay;
                dwdz  = dvdn*areaz;
                
                dtdn  = (t[c2]-t[c1])/delta;
            }
            
            //for aerodynamic heating!
            if(type == WALL){
                tw = -1.0;
				tw = tw_bcr[i];
                //bcr[face]->GetBCVar(&tw, REAL_FLOW, "tw",0);
                if(tw>0.0){
                    delta =(xfc[i]-xcc[c1])*areax+
                           (yfc[i]-ycc[c1])*areay+
                           (zfc[i]-zcc[c1])*areaz;
                    dtdn = (tw-t[c1])/delta;
                }
            }
        }
		
		// Get velocity at the face
        d_vis    = visc_f[i];
        heat_con = heat_f[i];
        
        txx = (2.*dudx - dvdy - dwdz)*two3;
        tyy = (2.*dvdy - dudx - dwdz)*two3;
        tzz = (2.*dwdz - dudx - dvdy)*two3;
        txy = dudy + dvdx;
        txz = dudz + dwdx;
        tyz = dwdy + dvdz;
        
        flux[0*nTFace + i] =  0.;
        flux[1*nTFace + i] = -d_vis*(txx*areax + txy*areay + txz*areaz)*area[i];
        flux[2*nTFace + i] = -d_vis*(txy*areax + tyy*areay + tyz*areaz)*area[i];
        flux[3*nTFace + i] = -d_vis*(txz*areax + tyz*areay + tzz*areaz)*area[i];
        flux[4*nTFace + i] =  umid*flux[1*nTFace + i] + vmid*flux[2*nTFace + i] + wmid*flux[3*nTFace + i]
            -  dtdn*heat_con*area[i];
			
		}
		else{
			flux[0*nTFace + i] = 0.0;
            flux[1*nTFace + i] = 0.0;
            flux[2*nTFace + i] = 0.0;
            flux[3*nTFace + i] = 0.0;
            flux[4*nTFace + i] = 0.0;
		}
		
	}	
	
}


void cuCalVisFluxTest(PolyGrid *grid, RealFlow *vel[3], RealFlow *t, RealFlow *vel_f[3],
                    RealFlow *visc_f, RealFlow *heat_f, RealFlow *t_f,
                    RealFlow *dqdx[3], RealFlow *dqdy[3], RealFlow *dqdz[3],
                    RealFlow *dtdx, RealFlow *dtdy, RealFlow *dtdz,
                    RealGeom *deltl, RealGeom *deltr, RealFlow *flux[5])
{
    IntType level  = grid->GetLevel();
	
	RealGeom BadFaceAngle = -1.0;
    grid->GetData(&BadFaceAngle, REAL_GEOM, 1, "BadFaceAngle");  
    RealGeom *facecentroidskewness = grid->GetGridQualityFaceCentroidSkewness();    
               
    RealGeom two3;
    static IntType warn = 1;
    two3 = 2.0/3.0;
	
	IntType blocksPerGrid = (gnTFace + threadsPerBlock - 1) / threadsPerBlock;
	
	//HANDLE_API_ERR(cudaMemcpy(gdtdx, dtdx, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(gdtdy, dtdy, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	//HANDLE_API_ERR(cudaMemcpy(gdtdz, dtdz, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
	
	gpuCalVisFluxTest <<< blocksPerGrid, threadsPerBlock >>> (gq, gt, &gdqdx[gnTCell + gnBFace], &gdqdy[gnTCell + gnBFace], &gdqdz[gnTCell + gnBFace], 
															gdtdx, gdtdy, gdtdz, 
															gvel_f, gt_f, gvisc_f, gheat_f, gdeltl, gdeltr, gflux, 
															gf2c, garea, gxfc, gyfc, gzfc, gxcc, gycc, gzcc, 
															gxfn, gyfn, gzfn, gfacecentroidskewness, gtype_bcr, gtw_bcr,
															level, warn, two3, BadFaceAngle, gnTFace, gnBFace, gnTCell);		
		
	// Turn off warning
    if(warn !=0) warn = 0;
	//HANDLE_API_ERR(cudaMemcpy(flux[0], gflux, 5*gnTFace*sizeof(RealFlow), cudaMemcpyDeviceToHost));			
}

#if (defined ShareMemory)
__global__ void gpuLoadVisFluxShareMemory(RealFlow* res, const RealFlow* flux, const IntType* C2F, const IntType* IndexC2F, 
						const IntType* nFPC, const IntType* f2c, const IntType nTFace, const IntType nTCell){
	
	extern __shared__ double sdata[];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;
	
	for(IntType j = 0; j < 5; j++){
		sdata[tid*5 + j] = res[j*nTCell + i];
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

__global__ void gpuLoadVisFluxShareMemory2(RealFlow* res, const RealFlow* flux, const IntType* C2F, const IntType* IndexC2F, 
						const IntType* nFPC, const IntType* f2c, const IntType nTFace, const IntType nTCell, const IntType threadsnum){
	
	extern __shared__ double sdata[];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;
	
	for(IntType j = 0; j < 5; j++){
		sdata[j*threadsnum + tid] = res[j*nTCell + i];
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


void cuLoadFluxVis(PolyGrid *grid, RealFlow* flux[5]){
	
	RealFlow *res[5];
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*gnTCell, "res");
    res[1] = &res[0][gnTCell];
    res[2] = &res[1][gnTCell];
    res[3] = &res[2][gnTCell];
    res[4] = &res[3][gnTCell];
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	//gpuLoadFlux <<< blocksPerGrid, threadsPerBlock >>> (gres, gflux, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell);
#if (defined ShareMemory)
	/*
	gpuLoadVisFluxShareMemory <<< blocksPerGrid, threadsPerBlock, 5*threadsPerBlock*sizeof(RealFlow)>>> (
												gres, gflux, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell);
	*/
	gpuLoadVisFluxShareMemory2 <<< blocksPerGrid, threadsPerBlock, 5*threadsPerBlock*sizeof(RealFlow)>>> (
												gres, gflux, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell, threadsPerBlock);	
#else
	gpuLoadFlux <<< blocksPerGrid, threadsPerBlock >>> (gres, gflux, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell);
#endif
	//HANDLE_API_ERR(cudaMemcpy(res[0], gres, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	
}

#if (defined FaceColoring)
void cuLoadFluxVisFaceColor(PolyGrid *grid, RealFlow* flux[5]){
	
	RealFlow *res[5];
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*gnTCell, "res");
    res[1] = &res[0][gnTCell];
    res[2] = &res[1][gnTCell];
    res[3] = &res[2][gnTCell];
    res[4] = &res[3][gnTCell];
	
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	//gpuLoadFlux <<< blocksPerGrid, threadsPerBlock >>> (gres, gflux, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell);
#if (defined ShareMemory)
	gpuLoadVisFluxShareMemory <<< blocksPerGrid, threadsPerBlock, 5*threadsPerBlock*sizeof(RealFlow)>>> (
												gres, gflux, gC2F, gIndexC2F, gnFPC, gf2c, gnTFace, gnTCell);
#else	
	cuLoadFluxColor(grid, res, flux);				
#endif
	// HANDLE_API_ERR(cudaMemcpy(res[0], gres, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	
}

#endif

void cuLoadBackRes(PolyGrid *grid){
	
	// res:
	RealFlow *res[5];
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*gnTCell, "res");
    res[1] = &res[0][gnTCell];
    res[2] = &res[1][gnTCell];
    res[3] = &res[2][gnTCell];
    res[4] = &res[3][gnTCell];
	
	HANDLE_API_ERR(cudaMemcpy(res[0], gres, 5*gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	RealFlow *vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, gnTCell + gnBFace, "vis_l");
	HANDLE_API_ERR(cudaMemcpy(vis_l, gvis_l, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	// rho, u, v, w, p:
	IntType n = grid->GetNTCell()+grid->GetNBFace();
	RealFlow *rho = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
	RealFlow *u   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
	RealFlow *v   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
	RealFlow *w   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
	RealFlow *p   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
	
	HANDLE_API_ERR(cudaMemcpy(rho, gq, gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));			
	HANDLE_API_ERR(cudaMemcpy(u, &gq[1*n], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	HANDLE_API_ERR(cudaMemcpy(v, &gq[2*n], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(w, &gq[3*n], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(p, &gq[4*n], gnTCell*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	// Grad:
	IntType kNVar = 5;
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
	
	HANDLE_API_ERR(cudaMemcpy(dqdx[0], gdqdx, 5*n*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	HANDLE_API_ERR(cudaMemcpy(dqdy[0], gdqdy, 5*n*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	HANDLE_API_ERR(cudaMemcpy(dqdz[0], gdqdz, 5*n*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	
	// DQ:
	RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*n, "DQ");
    if(!DQ[0]){
        mfmem::snew_array_1D(DQ[0],5*n,dmrfl);
        grid->UpdateDataPtr(DQ[0], REAL_FLOW, 5*n, "DQ");
    }
    
    for(IntType i=1; i<5; i++) DQ[i] = &DQ[i-1][n];
	HANDLE_API_ERR(cudaMemcpy(DQ[0], gDQ, 5*n*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	mfmem::sdel_array_1D(dqdx); 
	mfmem::sdel_array_1D(dqdy); 
	mfmem::sdel_array_1D(dqdz);

}

__global__ void gpuGetTemperature(RealFlow *t, const RealFlow *q, const RealFlow gascon, const RealFlow p_bar, 
								const IntType nBFace, const IntType nTCell){
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType Cell = nBFace + nTCell;
	if(i < Cell){
		t[i] = (q[4*Cell + i] + p_bar)/(q[i]*gascon);
	}
	
}

RealFlow *cuGetTemperature(PolyGrid *grid){
	
    IntType nBFace = grid->GetNBFace();
    IntType n      = grid->GetNTCell() + nBFace;
    
    RealFlow p_bar, gascon;
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&gascon, REAL_FLOW, 1, "gascon"); 
    
    RealFlow *rho = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho"); 
    RealFlow *p   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p"); 
    
    RealFlow *t = NULL;
    //mfmem::snew_array_1D(t,n,dmrfl);
    //assert(t != 0);
    IntType blocksPerGrid = (gnTCell + gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuGetTemperature <<< blocksPerGrid, threadsPerBlock >>> (gt, gq, gascon, gp_bar, gnBFace, gnTCell);	
    
    return t;
}

__global__ void gpuSetGhostTemperatureGradient(RealFlow *dtdx, RealFlow *dtdy, RealFlow *dtdz, 
							const RealGeom *xfn, const RealGeom *yfn, const RealGeom *zfn, 
							const IntType *f2c, const IntType *type_bcr, const RealFlow *tw_bcr,
							const IntType nBFace){																					
	
	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	IntType type, c1, c2;
	RealFlow tw = -1.0;
    RealFlow dta[3], dnn[3], dnnn;
	RealFlow dtdx_c2, dtdy_c2, dtdz_c2;
	if(i < nBFace){
		type  = type_bcr[i];
        c1    = f2c[2*i];
        c2    = f2c[2*i + 1];
        // wmark = 0;
		if(type != INTERFACE){
			if(type == WALL){
				dnnn = 
                    dtdx[c1] * xfn[i] + dtdy[c1] * yfn[i] + dtdz[c1] * zfn[i];
                dnn[0] = dnnn * xfn[i];
                dnn[1] = dnnn * yfn[i];
                dnn[2] = dnnn * zfn[i];
                dta[0] = dtdx[c1] - dnn[0];
                dta[1] = dtdy[c1] - dnn[1];
                dta[2] = dtdz[c1] - dnn[2];
                dtdx_c2 = dta[0] - dnn[0];
                dtdy_c2 = dta[1] - dnn[1];
                dtdz_c2 = dta[2] - dnn[2];
                
                tw = tw_bcr[i];
                if (tw > 0.0) {
                    dtdx_c2 = -dta[0] + dnn[0];
                    dtdy_c2 = -dta[1] + dnn[1];
                    dtdz_c2 = -dta[2] + dnn[2];
                }
			} 
			else if (type == SYMM){
				dnnn = 
                    dtdx[c1] * xfn[i] + dtdy[c1] * yfn[i] + dtdz[c1] * zfn[i];
                dnn[0] = dnnn * xfn[i];
                dnn[1] = dnnn * yfn[i];
                dnn[2] = dnnn * zfn[i];
                dta[0] = dtdx[c1] - dnn[0];
                dta[1] = dtdy[c1] - dnn[1];
                dta[2] = dtdz[c1] - dnn[2];
                dtdx_c2 = dta[0] - dnn[0];
                dtdy_c2 = dta[1] - dnn[1];
                dtdz_c2 = dta[2] - dnn[2];	
			}
			else{	// FAR_FIELD:
				dtdx_c2 = 0.0;
                dtdy_c2 = 0.0;
                dtdz_c2 = 0.0;
			}
			atomicExchSM35T(dtdx + c2, dtdx_c2);
			atomicExchSM35T(dtdy + c2, dtdy_c2);
			atomicExchSM35T(dtdz + c2, dtdz_c2);
		}
	}
	
}

void cuSetGhostTemperatureGradient(const PolyGrid *grid, RealFlow *dtdx, RealFlow *dtdy, RealFlow *dtdz) {
	
	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;	
	gpuSetGhostTemperatureGradient <<< blocksPerGrid, threadsPerBlock >>> (gdtdx, gdtdy, gdtdz, 
														gxfn, gyfn, gzfn, gf2c, gtype_bcr, gtw_bcr, gnBFace);
	
}

void cuViscousFlux(PolyGrid *grid, IntType level){
	
#if !(defined MultiStream)	
    // Get temperature
    //未修改overlap
	//RealFlow *t=NULL;
    RealFlow *t = cuGetTemperature(grid); // this is necessary for many times UpdateResiduals, such as for RK method.    

	cuCompGradientQ(grid, NULL, NULL, NULL, NULL, 5, NULL, NULL, NULL);
	#ifdef MPICH  
		IntType nvar = 1;
		grid->cuRecvSendVarNeighbor_TogethForGradient_T(nvar); 
	#endif  
#endif
	cuSetGhostTemperatureGradient(grid, NULL, NULL, NULL);

#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemVis, endtimeTemVis;
    double timeuseTemVis;
    gettimeofday(&starttimeTemVis, 0); 
#endif
#endif
	
	/* cudaEvent_t cu_start, cu_stop;
	float cu_esp;
	cudaEventCreate(&cu_start);
	cudaEventCreate(&cu_stop);
	cudaEventRecord(cu_start, 0); */
	
	cuCalDeriWeight(NULL, NULL, 1);
	//average of value in cell centroid
	cuCalVisHeatFace_average(grid, NULL, NULL, NULL);
	cuCalVeloandTFace_average(grid, NULL, NULL, NULL, NULL); 
	
	cuCalVisFluxTest(grid, NULL, NULL, NULL, NULL, NULL, NULL,
				  NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	
/* 	cudaEventRecord(cu_stop, 0);
	cudaEventSynchronize(cu_stop);
	
	cudaEventElapsedTime(&cu_esp, cu_start, cu_stop);	
	
#ifdef TIMECOST//dingxin
    timecost[7] += (RealGeom)cu_esp;
#endif */
	
#if (defined FaceColoring)
	cuLoadFluxVisFaceColor(grid, NULL);
#else
	cuLoadFluxVis(grid, NULL);
#endif	

#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    timecost[1] = timecost[1] + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemVis, 0); 
    timeuseTemVis = (RealGeom) 1000000*(endtimeTemVis.tv_sec - starttimeTemVis.tv_sec) + endtimeTemVis.tv_usec - starttimeTemVis.tv_usec;
    timecost[1] += timeuseTemVis;
    timeuseTemVis /= 1000000.0;
    time_vis += timeuseTemVis;
#endif
#endif
    
}

#ifdef LOOPMERGE
__global__ void gpuViscousFlux_merge_bface(const RealFlow heat, const RealFlow cp, const RealFlow prt,
		const RealGeom two3, IntType warn, const IntType level, const RealGeom BadFaceAngle, IntType vis_mode,
		const RealGeom *facecentroidskewness, const RealFlow *dtdx, const RealFlow *dtdy, const RealFlow *dtdz,
		const RealFlow *vel_f, const RealFlow *t_f, RealFlow *flux, const RealGeom *area, const RealGeom *xfc,
		const RealGeom *yfc, const RealGeom *zfc, const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc,
		const RealGeom *xfn, const RealGeom *yfn, const RealGeom *zfn, const IntType *type_bcr, const RealGeom *tw_bcr,
		const RealFlow *dqdx, const RealFlow *dqdy, const RealFlow *dqdz, const RealFlow *vis_t, const RealFlow *vis_l, 
		const IntType *f2c, const RealGeom *vol, const RealFlow *q, const RealFlow *t, const IntType nTFace,
		const IntType nBFace, const IntType nTCell, const IntType key){

	IntType i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i >= nBFace)
		return;
	IntType c1, c2, type, face;
	RealGeom delt1, delt2, delta;
	RealGeom deltl, deltr;
	RealFlow visc_f, heat_f, tmp;

	c1 = f2c[2*i];
    c2 = f2c[2*i + 1];
	face = i;
 
    //Left
    if(key == 1){  // distance weight
        delt1 = sqrt((xcc[c1] - xfc[face])*(xcc[c1] - xfc[face])
                +      (ycc[c1] - yfc[face])*(ycc[c1] - yfc[face])
                +      (zcc[c1] - zfc[face])*(zcc[c1] - zfc[face]));        
    }else if(key == 2){  //normal distance weight
        delt1 = fabs((xcc[c1] - xfc[face])*xfn[face]
                +      (ycc[c1] - yfc[face])*yfn[face]
                +      (zcc[c1] - zfc[face])*zfn[face]);
    }else if(key == 3){  //volume weight
        delt1 = vol[c1];
    }
 
    // Right
    if(key == 1){   // distance weight
        delt2 = sqrt((xcc[c2] - xfc[face])*(xcc[c2] - xfc[face])
                +      (ycc[c2] - yfc[face])*(ycc[c2] - yfc[face])
                +      (zcc[c2] - zfc[face])*(zcc[c2] - zfc[face]));
    }else if(key == 2){   //normal distance weight
        delt2 = fabs((xcc[c2] - xfc[face])*xfn[face]
                +      (ycc[c2] - yfc[face])*yfn[face]
                +      (zcc[c2] - zfc[face])*zfn[face]);
    }else if(key == 3){  //volume weight
        delt2 = vol[c2];
    }

    delta    = 1./(delt1 + delt2 + TINY);
    deltl = delt2*delta;
    deltr = delt1*delta;

	visc_f = 0.5 * (vis_l[c1] + vis_l[c2]);
	heat_f = heat * visc_f;

	if (vis_mode == S_A_MODEL) {
		tmp = 0.5 * (vis_t[c1] + vis_t[c2]);
		visc_f += tmp;
		heat_f += cp / prt * tmp;
	}

	RealGeom areax, areay, areaz;
	RealFlow umid, vmid, wmid, tmid, d_vis, heat_con, tw;
	RealFlow t1x, t1y, t1z, t2x, t2y, t2z;
	RealFlow dtmp, d1, d2, u1, u2, v1, v2, w1, w2, t1, t2, x1, x2, y1, y2, z1, z2;
	RealFlow dud1, dud2, dvd1, dvd2, dwd1, dwd2, dtd1, dtd2;
	RealFlow dudn, dvdn, dwdn, dtdn;
	RealFlow dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz;
	RealFlow txx, tyy, tzz, txy, txz, tyz;
	RealFlow dudt1, dvdt1, dwdt1, dudt2, dvdt2, dwdt2;
	RealFlow angle1, angle2;
	IntType Cell = nBFace + nTCell;
		
	areax = xfn[i];
    areay = yfn[i];
    areaz = zfn[i];
		
	// Get first tangential vector on the face
    if(areax != 0.) {
        t1x =  areay;
        t1y = -areax;
        t1z =  0.;
    } else if(areay != 0.) {
        t1x = -areay;
        t1y =  areax;
        t1z =  0.;
    } else if(areaz != 0.) {
        t1x =  0.;
        t1y = -areaz;
        t1z =  areay;
    } else {
		flux[0*nTFace + i] = 0.0;
        flux[1*nTFace + i] = 0.0;
        flux[2*nTFace + i] = 0.0;
        flux[3*nTFace + i] = 0.0;
        flux[4*nTFace + i] = 0.0;
		return;
    }

	// normalize the tangential vector
    dtmp = sqrt(t1x*t1x + t1y*t1y + t1z*t1z);
    t1x /= dtmp;
    t1y /= dtmp;
    t1z /= dtmp;
        
    // Get second tangential vector by cross dot t1 to normal
    t2x = areay*t1z - areaz*t1y;
    t2y = areaz*t1x - areax*t1z;
    t2z = areax*t1y - areay*t1x;
        
    // positions
    x1 = xcc[c1]  - xfc[i];
    y1 = ycc[c1]  - yfc[i];
    z1 = zcc[c1]  - zfc[i];
    x2 = xcc[c2]  - xfc[i];
    y2 = ycc[c2]  - yfc[i];
    z2 = zcc[c2]  - zfc[i];
    d1 = x1*areax + y1*areay + z1*areaz;
    d2 = x2*areax + y2*areay + z2*areaz;
        
    dtmp = -d1/(sqrt(x1*x1+ y1*y1 + z1*z1) + TINY);
    if(dtmp >  1.0) dtmp =  1.0;
    if(dtmp < -1.0) dtmp = -1.0;
    angle1 = asin(dtmp)*180.0/PI;
        
    dtmp = d2/(sqrt(x2*x2+ y2*y2 + z2*z2) + TINY);
    if(dtmp >  1.0) dtmp =  1.0;
    if(dtmp < -1.0) dtmp = -1.0;
    angle2 = asin(dtmp)*180.0/PI;
        
    // quentities at points 1 and 2
    u1   = q[1*Cell + c1];
    v1   = q[2*Cell + c1];
    w1   = q[3*Cell + c1];
    t1   = t[c1];
    u2   = q[1*Cell + c2];
    v2   = q[2*Cell + c2];
    w2   = q[3*Cell + c2];
    t2   = t[c2];
    umid = 0.5*(u1 + u2);
    vmid = 0.5*(v1 + v2);
    wmid = 0.5*(w1 + w2);
    tmid = 0.5*(t1 + t2);
        
    // Theroretically, more accurate to include the following terms
    if(angle1 > 10.0 && angle2 > 10.0) {
        u1 += dqdx[0*Cell + c1]*(d1*areax - x1) + dqdy[0*Cell + c1]*(d1*areay - y1) + dqdz[0*Cell + c1]*(d1*areaz - z1);
        v1 += dqdx[1*Cell + c1]*(d1*areax - x1) + dqdy[1*Cell + c1]*(d1*areay - y1) + dqdz[1*Cell + c1]*(d1*areaz - z1);
        w1 += dqdx[2*Cell + c1]*(d1*areax - x1) + dqdy[2*Cell + c1]*(d1*areay - y1) + dqdz[2*Cell + c1]*(d1*areaz - z1);
            
        u2 += dqdx[0*Cell + c2]*(d2*areax - x2) + dqdy[0*Cell + c2]*(d2*areay - y2) + dqdz[0*Cell + c2]*(d2*areaz - z2);
        v2 += dqdx[1*Cell + c2]*(d2*areax - x2) + dqdy[1*Cell + c2]*(d2*areay - y2) + dqdz[1*Cell + c2]*(d2*areaz - z2);
        w2 += dqdx[2*Cell + c2]*(d2*areax - x2) + dqdy[2*Cell + c2]*(d2*areay - y2) + dqdz[2*Cell + c2]*(d2*areaz - z2);
            
        t1 += dtdx[c1]*(d1*areax - x1) + dtdy[c1]*(d1*areay - y1) + dtdz[c1]*(d1*areaz - z1);
        t2 += dtdx[c2]*(d2*areax - x2) + dtdy[c2]*(d2*areay - y2) + dtdz[c2]*(d2*areaz - z2);
        if(t1 < TINY) t1  = t[c1];
        if(t2 < TINY) t2  = t[c2];
            
        // quantities at the face
        //umid = vel_f[0*nTFace + i];
        //vmid = vel_f[1*nTFace + i];
        //wmid = vel_f[2*nTFace + i];
        //tmid = t_f[i];
    }
        
    dudx  = dqdx[0*Cell + c1]*deltl + dqdx[0*Cell + c2]*deltr;
    dudy  = dqdy[0*Cell + c1]*deltl + dqdy[0*Cell + c2]*deltr;
    dudz  = dqdz[0*Cell + c1]*deltl + dqdz[0*Cell + c2]*deltr;
    dvdx  = dqdx[1*Cell + c1]*deltl + dqdx[1*Cell + c2]*deltr;
    dvdy  = dqdy[1*Cell + c1]*deltl + dqdy[1*Cell + c2]*deltr;
    dvdz  = dqdz[1*Cell + c1]*deltl + dqdz[1*Cell + c2]*deltr;
    dwdx  = dqdx[2*Cell + c1]*deltl + dqdx[2*Cell + c2]*deltr;
    dwdy  = dqdy[2*Cell + c1]*deltl + dqdy[2*Cell + c2]*deltr;
    dwdz  = dqdz[2*Cell + c1]*deltl + dqdz[2*Cell + c2]*deltr;
        
    dudn  = 0.0;
    dvdn  = 0.0;
    dwdn  = 0.0;
    dtdn  = 0.0;
        
    if(angle1 > 0.0 && angle2 > 0.0 && fabs(d1) > TINY && fabs(d2) > TINY) {
        dud1 = (u1 - umid)/d1;
        dvd1 = (v1 - vmid)/d1;
        dwd1 = (w1 - wmid)/d1;
        dtd1 = (t1 - tmid)/d1;
        dud2 = (u2 - umid)/d2;
        dvd2 = (v2 - vmid)/d2;
        dwd2 = (w2 - wmid)/d2;
        dtd2 = (t2 - tmid)/d2;
        dtmp = d1*d1 + d2*d2;
        d1   = d1*d1/dtmp;
        d2   = d2*d2/dtmp;
        dudn = dud1*d1 + dud2*d2;
        dvdn = dvd1*d1 + dvd2*d2;
        dwdn = dwd1*d1 + dwd2*d2;
        dtdn = dtd1*d1 + dtd2*d2;
    }
        
    // dqdt, does not matter too much
    dudt1 = dudx*t1x + dudy*t1y + dudz*t1z;
    dvdt1 = dvdx*t1x + dvdy*t1y + dvdz*t1z;
    dwdt1 = dwdx*t1x + dwdy*t1y + dwdz*t1z;
    dudt2 = dudx*t2x + dudy*t2y + dudz*t2z;
    dvdt2 = dvdx*t2x + dvdy*t2y + dvdz*t2z;
    dwdt2 = dwdx*t2x + dwdy*t2y + dwdz*t2z;
        
    // now true gradients
    dudx  = dudn*areax + dudt1*t1x + dudt2*t2x;
    dudy  = dudn*areay + dudt1*t1y + dudt2*t2y;
    dudz  = dudn*areaz + dudt1*t1z + dudt2*t2z;
    dvdx  = dvdn*areax + dvdt1*t1x + dvdt2*t2x;
    dvdy  = dvdn*areay + dvdt1*t1y + dvdt2*t2y;
    dvdz  = dvdn*areaz + dvdt1*t1z + dvdt2*t2z;
    dwdx  = dwdn*areax + dwdt1*t1x + dwdt2*t2x;
    dwdy  = dwdn*areay + dwdt1*t1y + dwdt2*t2y;
    dwdz  = dwdn*areaz + dwdt1*t1z + dwdt2*t2z;
    if(level==0 && BadFaceAngle>0.0 && facecentroidskewness[i]<BadFaceAngle){
        dudx  = dudn*areax;
        dudy  = dudn*areay;
        dudz  = dudn*areaz;
        dvdx  = dvdn*areax;
        dvdy  = dvdn*areay;
        dvdz  = dvdn*areaz;
        dwdx  = dwdn*areax;
        dwdy  = dwdn*areay;
        dwdz  = dwdn*areaz;
    }
         
    type = type_bcr[i];
    if(type!=WALL && type!=SYMM && type!=FAR_FIELD && type!=INTERFACE){  
        delta = sqrt((xcc[c1]-xcc[c2])*(xcc[c1]-xcc[c2]) +
                        (ycc[c1]-ycc[c2])*(ycc[c1]-ycc[c2]) +
                        (zcc[c1]-zcc[c2])*(zcc[c1]-zcc[c2]));
                
        dvdn  = (q[1*Cell + c2]-q[1*Cell + c1])/delta;
        dudx  = dvdn*areax;
        dudy  = dvdn*areay;
        dudz  = dvdn*areaz;
                
        dvdn  = (q[2*Cell + c2]-q[2*Cell + c1])/delta;
        dvdx  = dvdn*areax;
        dvdy  = dvdn*areay;
        dvdz  = dvdn*areaz;
                
        dvdn  = (q[3*Cell + c2]-q[3*Cell + c1])/delta;
        dwdx  = dvdn*areax;
        dwdy  = dvdn*areay;
        dwdz  = dvdn*areaz;
                
        dtdn  = (t[c2]-t[c1])/delta;
    }
            
    //for aerodynamic heating!
    if(type == WALL){
        tw = -1.0;
		tw = tw_bcr[i];
        //bcr[face]->GetBCVar(&tw, REAL_FLOW, "tw",0);
        if(tw>0.0){
            delta =(xfc[i]-xcc[c1])*areax+
                    (yfc[i]-ycc[c1])*areay+
                    (zfc[i]-zcc[c1])*areaz;
            dtdn = (tw-t[c1])/delta;
        }
    }
		
	// Get velocity at the face
    d_vis    = visc_f;
    heat_con = heat_f;
        
    txx = (2.*dudx - dvdy - dwdz)*two3;
    tyy = (2.*dvdy - dudx - dwdz)*two3;
    tzz = (2.*dwdz - dudx - dvdy)*two3;
    txy = dudy + dvdx;
    txz = dudz + dwdx;
    tyz = dwdy + dvdz;
        
    flux[0*nTFace + i] =  0.;
    flux[1*nTFace + i] = -d_vis*(txx*areax + txy*areay + txz*areaz)*area[i];
    flux[2*nTFace + i] = -d_vis*(txy*areax + tyy*areay + tyz*areaz)*area[i];
    flux[3*nTFace + i] = -d_vis*(txz*areax + tyz*areay + tzz*areaz)*area[i];
    flux[4*nTFace + i] =  umid*flux[1*nTFace + i] + vmid*flux[2*nTFace + i] + wmid*flux[3*nTFace + i]
        -  dtdn*heat_con*area[i];
}

__global__ void gpuViscousFlux_merge_iface(const RealFlow heat, const RealFlow cp, const RealFlow prt,
		const RealGeom two3, IntType warn, const IntType level, const RealGeom BadFaceAngle, IntType vis_mode,
		const RealGeom *facecentroidskewness, const RealFlow *dtdx, const RealFlow *dtdy, const RealFlow *dtdz,
		const RealFlow *vel_f, const RealFlow *t_f, RealFlow *flux, const RealGeom *area, const RealGeom *xfc,
		const RealGeom *yfc, const RealGeom *zfc, const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc,
		const RealGeom *xfn, const RealGeom *yfn, const RealGeom *zfn, const RealFlow *dqdx, const RealFlow *dqdy,
		const RealFlow *dqdz, const RealFlow *vis_t, const RealFlow *vis_l, const IntType *f2c, const RealGeom *vol,
		const RealFlow *q, const RealFlow *t, const IntType nTFace, const IntType nBFace, const IntType nTCell,
		const IntType key){

	IntType i = blockDim.x*blockIdx.x + threadIdx.x + nBFace;
	if (i >= nTFace)
		return;
	IntType c1, c2, face;
	RealGeom delt1, delt2, delta;
	RealGeom deltl, deltr;
	RealFlow visc_f, heat_f, tmp;

	c1 = f2c[2*i];
    c2 = f2c[2*i + 1];
	face = i;
 
    //Left
    if(key == 1){  // distance weight
        delt1 = sqrt((xcc[c1] - xfc[face])*(xcc[c1] - xfc[face])
                +      (ycc[c1] - yfc[face])*(ycc[c1] - yfc[face])
                +      (zcc[c1] - zfc[face])*(zcc[c1] - zfc[face]));        
    }else if(key == 2){  //normal distance weight
        delt1 = fabs((xcc[c1] - xfc[face])*xfn[face]
                +      (ycc[c1] - yfc[face])*yfn[face]
                +      (zcc[c1] - zfc[face])*zfn[face]);
    }else if(key == 3){  //volume weight
        delt1 = vol[c1];
    }
 
    // Right
    if(key == 1){   // distance weight
        delt2 = sqrt((xcc[c2] - xfc[face])*(xcc[c2] - xfc[face])
                +      (ycc[c2] - yfc[face])*(ycc[c2] - yfc[face])
                +      (zcc[c2] - zfc[face])*(zcc[c2] - zfc[face]));
    }else if(key == 2){   //normal distance weight
        delt2 = fabs((xcc[c2] - xfc[face])*xfn[face]
                +      (ycc[c2] - yfc[face])*yfn[face]
                +      (zcc[c2] - zfc[face])*zfn[face]);
    }else if(key == 3){  //volume weight
        delt2 = vol[c2];
    }

    delta    = 1./(delt1 + delt2 + TINY);
    deltl = delt2*delta;
    deltr = delt1*delta;

	visc_f = 0.5 * (vis_l[c1] + vis_l[c2]);
	heat_f = heat * visc_f;

	if (vis_mode == S_A_MODEL) {
		tmp = 0.5 * (vis_t[c1] + vis_t[c2]);
		visc_f += tmp;
		heat_f += cp / prt * tmp;
	}

	RealGeom areax, areay, areaz;
	RealFlow umid, vmid, wmid, tmid, d_vis, heat_con;
	RealFlow t1x, t1y, t1z, t2x, t2y, t2z;
	RealFlow dtmp, d1, d2, u1, u2, v1, v2, w1, w2, t1, t2, x1, x2, y1, y2, z1, z2;
	RealFlow dud1, dud2, dvd1, dvd2, dwd1, dwd2, dtd1, dtd2;
	RealFlow dudn, dvdn, dwdn, dtdn;
	RealFlow dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz;
	RealFlow txx, tyy, tzz, txy, txz, tyz;
	RealFlow dudt1, dvdt1, dwdt1, dudt2, dvdt2, dwdt2;
	RealFlow angle1, angle2;
	IntType Cell = nBFace + nTCell;
		
	areax = xfn[i];
    areay = yfn[i];
    areaz = zfn[i];
		
	// Get first tangential vector on the face
    if(areax != 0.) {
        t1x =  areay;
        t1y = -areax;
        t1z =  0.;
    } else if(areay != 0.) {
        t1x = -areay;
        t1y =  areax;
        t1z =  0.;
    } else if(areaz != 0.) {
        t1x =  0.;
        t1y = -areaz;
        t1z =  areay;
    } else {
		flux[0*nTFace + i] = 0.0;
        flux[1*nTFace + i] = 0.0;
        flux[2*nTFace + i] = 0.0;
        flux[3*nTFace + i] = 0.0;
        flux[4*nTFace + i] = 0.0;
		return;
    }

	// normalize the tangential vector
    dtmp = sqrt(t1x*t1x + t1y*t1y + t1z*t1z);
    t1x /= dtmp;
    t1y /= dtmp;
    t1z /= dtmp;
        
    // Get second tangential vector by cross dot t1 to normal
    t2x = areay*t1z - areaz*t1y;
    t2y = areaz*t1x - areax*t1z;
    t2z = areax*t1y - areay*t1x;
        
    // positions
    x1 = xcc[c1]  - xfc[i];
    y1 = ycc[c1]  - yfc[i];
    z1 = zcc[c1]  - zfc[i];
    x2 = xcc[c2]  - xfc[i];
    y2 = ycc[c2]  - yfc[i];
    z2 = zcc[c2]  - zfc[i];
    d1 = x1*areax + y1*areay + z1*areaz;
    d2 = x2*areax + y2*areay + z2*areaz;
        
    dtmp = -d1/(sqrt(x1*x1+ y1*y1 + z1*z1) + TINY);
    if(dtmp >  1.0) dtmp =  1.0;
    if(dtmp < -1.0) dtmp = -1.0;
    angle1 = asin(dtmp)*180.0/PI;
        
    dtmp = d2/(sqrt(x2*x2+ y2*y2 + z2*z2) + TINY);
    if(dtmp >  1.0) dtmp =  1.0;
    if(dtmp < -1.0) dtmp = -1.0;
    angle2 = asin(dtmp)*180.0/PI;
        
    // quentities at points 1 and 2
    u1   = q[1*Cell + c1];
    v1   = q[2*Cell + c1];
    w1   = q[3*Cell + c1];
    t1   = t[c1];
    u2   = q[1*Cell + c2];
    v2   = q[2*Cell + c2];
    w2   = q[3*Cell + c2];
    t2   = t[c2];
    umid = 0.5*(u1 + u2);
    vmid = 0.5*(v1 + v2);
    wmid = 0.5*(w1 + w2);
    tmid = 0.5*(t1 + t2);
        
    // Theroretically, more accurate to include the following terms
    if(angle1 > 10.0 && angle2 > 10.0) {
        u1 += dqdx[0*Cell + c1]*(d1*areax - x1) + dqdy[0*Cell + c1]*(d1*areay - y1) + dqdz[0*Cell + c1]*(d1*areaz - z1);
        v1 += dqdx[1*Cell + c1]*(d1*areax - x1) + dqdy[1*Cell + c1]*(d1*areay - y1) + dqdz[1*Cell + c1]*(d1*areaz - z1);
        w1 += dqdx[2*Cell + c1]*(d1*areax - x1) + dqdy[2*Cell + c1]*(d1*areay - y1) + dqdz[2*Cell + c1]*(d1*areaz - z1);
            
        u2 += dqdx[0*Cell + c2]*(d2*areax - x2) + dqdy[0*Cell + c2]*(d2*areay - y2) + dqdz[0*Cell + c2]*(d2*areaz - z2);
        v2 += dqdx[1*Cell + c2]*(d2*areax - x2) + dqdy[1*Cell + c2]*(d2*areay - y2) + dqdz[1*Cell + c2]*(d2*areaz - z2);
        w2 += dqdx[2*Cell + c2]*(d2*areax - x2) + dqdy[2*Cell + c2]*(d2*areay - y2) + dqdz[2*Cell + c2]*(d2*areaz - z2);
            
        t1 += dtdx[c1]*(d1*areax - x1) + dtdy[c1]*(d1*areay - y1) + dtdz[c1]*(d1*areaz - z1);
        t2 += dtdx[c2]*(d2*areax - x2) + dtdy[c2]*(d2*areay - y2) + dtdz[c2]*(d2*areaz - z2);
        if(t1 < TINY) t1  = t[c1];
        if(t2 < TINY) t2  = t[c2];
            
        // quantities at the face
        //umid = vel_f[0*nTFace + i];
        //vmid = vel_f[1*nTFace + i];
        //wmid = vel_f[2*nTFace + i];
        //tmid = t_f[i];
    }
        
    dudx  = dqdx[0*Cell + c1]*deltl + dqdx[0*Cell + c2]*deltr;
    dudy  = dqdy[0*Cell + c1]*deltl + dqdy[0*Cell + c2]*deltr;
    dudz  = dqdz[0*Cell + c1]*deltl + dqdz[0*Cell + c2]*deltr;
    dvdx  = dqdx[1*Cell + c1]*deltl + dqdx[1*Cell + c2]*deltr;
    dvdy  = dqdy[1*Cell + c1]*deltl + dqdy[1*Cell + c2]*deltr;
    dvdz  = dqdz[1*Cell + c1]*deltl + dqdz[1*Cell + c2]*deltr;
    dwdx  = dqdx[2*Cell + c1]*deltl + dqdx[2*Cell + c2]*deltr;
    dwdy  = dqdy[2*Cell + c1]*deltl + dqdy[2*Cell + c2]*deltr;
    dwdz  = dqdz[2*Cell + c1]*deltl + dqdz[2*Cell + c2]*deltr;
        
    dudn  = 0.0;
    dvdn  = 0.0;
    dwdn  = 0.0;
    dtdn  = 0.0;
        
    if(angle1 > 0.0 && angle2 > 0.0 && fabs(d1) > TINY && fabs(d2) > TINY) {
        dud1 = (u1 - umid)/d1;
        dvd1 = (v1 - vmid)/d1;
        dwd1 = (w1 - wmid)/d1;
        dtd1 = (t1 - tmid)/d1;
        dud2 = (u2 - umid)/d2;
        dvd2 = (v2 - vmid)/d2;
        dwd2 = (w2 - wmid)/d2;
        dtd2 = (t2 - tmid)/d2;
        dtmp = d1*d1 + d2*d2;
        d1   = d1*d1/dtmp;
        d2   = d2*d2/dtmp;
        dudn = dud1*d1 + dud2*d2;
        dvdn = dvd1*d1 + dvd2*d2;
        dwdn = dwd1*d1 + dwd2*d2;
        dtdn = dtd1*d1 + dtd2*d2;
    }
        
    // dqdt, does not matter too much
    dudt1 = dudx*t1x + dudy*t1y + dudz*t1z;
    dvdt1 = dvdx*t1x + dvdy*t1y + dvdz*t1z;
    dwdt1 = dwdx*t1x + dwdy*t1y + dwdz*t1z;
    dudt2 = dudx*t2x + dudy*t2y + dudz*t2z;
    dvdt2 = dvdx*t2x + dvdy*t2y + dvdz*t2z;
    dwdt2 = dwdx*t2x + dwdy*t2y + dwdz*t2z;
        
    // now true gradients
    dudx  = dudn*areax + dudt1*t1x + dudt2*t2x;
    dudy  = dudn*areay + dudt1*t1y + dudt2*t2y;
    dudz  = dudn*areaz + dudt1*t1z + dudt2*t2z;
    dvdx  = dvdn*areax + dvdt1*t1x + dvdt2*t2x;
    dvdy  = dvdn*areay + dvdt1*t1y + dvdt2*t2y;
    dvdz  = dvdn*areaz + dvdt1*t1z + dvdt2*t2z;
    dwdx  = dwdn*areax + dwdt1*t1x + dwdt2*t2x;
    dwdy  = dwdn*areay + dwdt1*t1y + dwdt2*t2y;
    dwdz  = dwdn*areaz + dwdt1*t1z + dwdt2*t2z;
    if(level==0 && BadFaceAngle>0.0 && facecentroidskewness[i]<BadFaceAngle){
        dudx  = dudn*areax;
        dudy  = dudn*areay;
        dudz  = dudn*areaz;
        dvdx  = dvdn*areax;
        dvdy  = dvdn*areay;
        dvdz  = dvdn*areaz;
        dwdx  = dwdn*areax;
        dwdy  = dwdn*areay;
        dwdz  = dwdn*areaz;
    }
		
	// Get velocity at the face
    d_vis    = visc_f;
    heat_con = heat_f;
        
    txx = (2.*dudx - dvdy - dwdz)*two3;
    tyy = (2.*dvdy - dudx - dwdz)*two3;
    tzz = (2.*dwdz - dudx - dvdy)*two3;
    txy = dudy + dvdx;
    txz = dudz + dwdx;
    tyz = dwdy + dvdz;
        
    flux[0*nTFace + i] =  0.;
    flux[1*nTFace + i] = -d_vis*(txx*areax + txy*areay + txz*areaz)*area[i];
    flux[2*nTFace + i] = -d_vis*(txy*areax + tyy*areay + tyz*areaz)*area[i];
    flux[3*nTFace + i] = -d_vis*(txz*areax + tyz*areay + tzz*areaz)*area[i];
    flux[4*nTFace + i] =  umid*flux[1*nTFace + i] + vmid*flux[2*nTFace + i] + wmid*flux[3*nTFace + i]
        -  dtdn*heat_con*area[i];
}

void cuViscousFlux_merge(PolyGrid *grid, IntType level){
	
    IntType  nBFace = grid->GetNBFace();
    IntType  nTCell = grid->GetNTCell();
    IntType  n      = nTCell + nBFace;
    IntType  nTFace = grid->GetNTFace(); 

    // Get temperature
    //未修改overlap
	//RealFlow *t=NULL;
    RealFlow *t = cuGetTemperature(grid); // this is necessary for many times UpdateResiduals, such as for RK method.
    RealFlow *dtdx = NULL;
    RealFlow *dtdy = NULL;
    RealFlow *dtdz = NULL;
	
#if !(defined MultiStream)
	cuCompGradientQ(grid, t, dtdx, dtdy, dtdz, 5, NULL, NULL, NULL);
	#ifdef MPICH  
		IntType nvar = 1;
		grid->cuRecvSendVarNeighbor_TogethForGradient_T(nvar); 
	#endif  
#endif
	//cuUpdateGhostGradT(dtdx, dtdy, dtdz);
	
	cuSetGhostTemperatureGradient(grid, dtdx, dtdy, dtdz);
	
    //Get viscosity coefficients in each control volume
    RealFlow *vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
    if (vis_l == 0){
        printf("Should not come here! ViscousFlux!\n");
        //mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    // Allocate temporary memories for fluxes
    RealFlow  *flux[5];

#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemVis, endtimeTemVis;
    double timeuseTemVis;
    gettimeofday(&starttimeTemVis, 0); 
#endif
#endif

	IntType  vis_mode, cond_comp = 1;
	grid->GetData(&vis_mode, INT, 1, "vis_mode");
	grid->GetData(&cond_comp, INT, 1, "comp", 0);

	// Get specific heat ratio, gas constant, cp
	RealFlow gam, gascon, cp;
	grid->GetData(&gam, REAL_FLOW, 1, "gam");
	grid->GetData(&gascon, REAL_FLOW, 1, "gascon");
	cp = gascon * gam / (gam - 1.);
	if (cond_comp == 0)grid->GetData(&cp, REAL_FLOW, 1, "cp");
	// Get viscosity, Prandtl number
	RealFlow prl, heat;
	grid->GetData(&prl, REAL_FLOW, 1, "prl");
	heat = cp / prl;

	RealGeom BadFaceAngle = -1.0;
	grid->GetData(&BadFaceAngle, REAL_GEOM, 1, "BadFaceAngle");
	RealFlow prt;
	grid->GetData(&prt, REAL_FLOW, 1, "prt");
               
    RealGeom two3;
    static IntType warn = 1;
    two3 = 2.0/3.0;
	
	/* cudaEvent_t cu_start, cu_stop;
	float cu_esp;
	cudaEventCreate(&cu_start);
	cudaEventCreate(&cu_stop);
	cudaEventRecord(cu_start, 0); */

	IntType blocksPerGrid = (gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuViscousFlux_merge_bface <<< blocksPerGrid, threadsPerBlock >>> (heat, cp, prt, two3, warn, level, BadFaceAngle, vis_mode,
			gfacecentroidskewness, gdtdx, gdtdy, gdtdz, gvel_f, gt_f, gflux, garea, gxfc, gyfc, gzfc, gxcc, gycc, gzcc,
			gxfn, gyfn, gzfn, gtype_bcr, gtw_bcr, &gdqdx[gnTCell + gnBFace], &gdqdy[gnTCell + gnBFace], &gdqdz[gnTCell + gnBFace], 
			gvis_t, gvis_l, gf2c, gvol, gq, gt, gnTFace, gnBFace, gnTCell, 1);

	blocksPerGrid = (gnTFace - gnBFace + threadsPerBlock - 1) / threadsPerBlock;
	gpuViscousFlux_merge_iface <<< blocksPerGrid, threadsPerBlock >>> (heat, cp, prt, two3, warn, level, BadFaceAngle, vis_mode,
			gfacecentroidskewness, gdtdx, gdtdy, gdtdz, gvel_f, gt_f, gflux, garea, gxfc, gyfc, gzfc, gxcc, gycc, gzcc,
			gxfn, gyfn, gzfn, &gdqdx[gnTCell + gnBFace], &gdqdy[gnTCell + gnBFace], &gdqdz[gnTCell + gnBFace], 
			gvis_t, gvis_l, gf2c, gvol, gq, gt, gnTFace, gnBFace, gnTCell, 1);
	
/* 	cudaEventRecord(cu_stop, 0);
	cudaEventSynchronize(cu_stop);
	
	cudaEventElapsedTime(&cu_esp, cu_start, cu_stop);	
	
#ifdef TIMECOST//dingxin
    timecost[9] += (RealGeom)cu_esp;
#endif */
	
#if (defined FaceColoring)
	cuLoadFluxVisFaceColor(grid, flux);
#else
	cuLoadFluxVis(grid, flux);
#endif	

#ifdef TIMECOST//dingxin
	cudaDeviceSynchronize();
#ifdef MPICH
    timecost[1] = timecost[1] + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemVis, 0); 
    timeuseTemVis = (RealGeom) 1000000*(endtimeTemVis.tv_sec - starttimeTemVis.tv_sec) + endtimeTemVis.tv_usec - starttimeTemVis.tv_usec;
    timecost[1] += timeuseTemVis;
    timeuseTemVis /= 1000000.0;
    time_vis += timeuseTemVis;
#endif
#endif

    //mfmem::sdel_array_1D(t);
    /* mfmem::sdel_array_1D(dtdx);
    mfmem::sdel_array_1D(dtdy);
    mfmem::sdel_array_1D(dtdz); */
    
}
#endif // ~LOOPMERGE

__device__ double atomicExchSM35T(double* address, double val){
	
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
        } while (assumed != old);
        return __longlong_as_double(old);
}












