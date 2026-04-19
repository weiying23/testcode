#include "cuFillMatCOO.cuh"
#include "number_type.h"
//#include "petscdevice.h"
#include <iostream>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuData.cuh"

using namespace mflow;
using namespace gpuData;

__device__ void gpuCalConvectiveFluxJacobian(RealFlow *Matrix, RealFlow nx, RealFlow ny, RealFlow nz,
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
__device__ double gpu_max(double a, double b)
{
	return a > b ? a:b;
}

__device__ void gpuCalJacobian_ConvectiveFlux_Roe(RealFlow *matrix, RealFlow q_L[5], RealFlow q_R[5], RealFlow nx, RealFlow ny, RealFlow nz, RealFlow gam, RealFlow alf_l)
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
	
    RealFlow epsbb = 0.25/gpu_max(epsaa,TINY);

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

__global__ void FillValues(RealFlow *val, IntType gnTCell, IntType  gvis_run, IntType gifStart, RealFlow ggam, RealFlow gp_bar, RealFlow galf_l, IntType  *gcoo, IntType  *gf2c, IntType  *gnFPC, IntType  *gC2F, 
IntType *gIndexC2F, RealGeom *gxfn, RealGeom *gyfn, RealGeom *gzfn, RealGeom *garea, RealGeom *gvol, RealFlow *gdt, RealGeom *gnorm_dist_c2c, RealFlow *gvis_l, RealFlow *gvis_t, RealFlow *gq0,RealFlow *gq1,RealFlow *gq2,RealFlow *gq3,RealFlow *gq4, RealFlow * gmatrix_jacobi_d, RealFlow * gmatrix_jacobi_fc, RealFlow * gmatrix_temp)
{
	IntType cell = blockDim.x*blockIdx.x + threadIdx.x;
	if(cell < gnTCell)
	{
		RealFlow * vStart = &val[gcoo[cell]];
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
			IntType face  = gC2F[cellC2FIndex + iFace]; ///!!!!!!!!!!!
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
			gpuCalConvectiveFluxJacobian(matrix_jacobi_fc, face_n[0], face_n[1], face_n[2], q_r[0], q_r[1], q_r[2], q_r[3], q_r[4], ggam);

			// Roe matrix: 
			q_l[0]  = gq0[c1];
			q_l[1]  = gq1[c1];
			q_l[2]  = gq2[c1];
			q_l[3]  = gq3[c1];
			q_l[4]  = gq4[c1]+gp_bar;
			RealFlow tmparea = 0.5*garea[face];
			gpuCalJacobian_ConvectiveFlux_Roe(matrix_temp, q_l, q_r, face_n[0], face_n[1], face_n[2], ggam, galf_l); 
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

#ifdef USING_PETSC
PetscErrorCode FillMatrixCUDACOO( Mat A, IntType nnz )
{
    RealFlow *val;
	IntType blocksPerGrid = (gnTCell + threadsPerBlock - 1) / threadsPerBlock;
	cudaMalloc((void**)&val, nnz*sizeof(RealFlow));
	
	FillValues<<< blocksPerGrid, threadsPerBlock >>>(val, gnTCell, gvis_run, gifStart, ggam, gp_bar, 
		galf_l, gcoo, gf2c, gnFPC, gC2F, gIndexC2F, gxfn, gyfn, gzfn, garea, gvol, gdt, gnorm_dist_c2c, 
		gvis_l, gvis_t, gq0, gq1, gq2, gq3, gq4, gmatrix_jacobi_d, gmatrix_jacobi_fc, gmatrix_temp);
	

		/*int gnvar = 5;
		int gnnz = nnz;
		RealFlow *mat = new RealFlow[gnnz];
		cudaMemcpy(mat, val, gnnz*sizeof(RealFlow), cudaMemcpyDeviceToHost);
		RealFlow *matrix2 = new RealFlow[gnnz];
		IntType  size = gnvar*gnvar;
		for(int i = 0; i < gnTCell; i++){
			//printf("n:%d ",i);
			for(int j=row_ptr[i]; j<row_ptr[i+1]; j++){
				RealFlow *m2_ptr = &matrix2[ j*size ];
				if( i > col_ind[j] ){
					//printf("from %d to %d, ", j,j);
					RealFlow *m_ptr = &mat[ j * size ];
					for(int k=0; k<size; k++)
						m2_ptr[ k ] = m_ptr[ k ];
				}
				else if( i == col_ind[j]){
					//printf("from %d to %d, ", j, (row_ptr[i+1]-1));
					RealFlow *m_ptr = &mat[ (row_ptr[i+1]-1) * size ];
					for(int k=0; k<size; k++)
						m2_ptr[ k ] = m_ptr[ k ];
				}
				else{
					//printf("from %d to %d, ", j, (j-1));
					RealFlow *m_ptr = &mat[ (j-1) * size ];
					for(int k=0; k<size; k++)
						m2_ptr[ k ] = m_ptr[ k ];
				}
			}
			//printf("\n");
		}
		CHECKCUDA(cudaMemcpy(val, matrix2, gnnz*sizeof(RealFlow), cudaMemcpyHostToDevice));
		delete[] mat;
		delete[] matrix2;*/


	PetscCall(MatSetValuesCOO(A, val, INSERT_VALUES));
	cudaFree(val);
	
	return 0;
}
#endif

void CopyConstData2Device(IntType nTCell, IntType nBFace, IntType nTFace, IntType ifStart, IntType vis_run, RealFlow gam, RealFlow p_bar, RealFlow alf_l, IntType *coo, IntType *f2c, IntType **C2F, IntType *IndexC2F, IntType *nFPC, 
RealGeom *vol, RealGeom *xfn, RealGeom *yfn, RealGeom *zfn, RealFlow *dt, RealGeom *norm_dist_c2c, RealGeom *area)
{
	gnTCell = nTCell;
	gnBFace = nBFace;
	gnTFace = nTFace;
	gvis_run = vis_run;
	gifStart = ifStart;
	ggam = gam;
	gp_bar = p_bar;
	galf_l = alf_l;
	
	IntType coosize = nTCell+1;
	cudaMalloc((void **)&gcoo, coosize*sizeof(IntType));
	cudaMemcpy(gcoo, coo, coosize*sizeof(IntType), cudaMemcpyHostToDevice);
	
	cudaMalloc((void **)&gf2c, 2*nTFace*sizeof(IntType));
	cudaMemcpy(gf2c, f2c, 2*nTFace*sizeof(IntType), cudaMemcpyHostToDevice);
	
	cudaMalloc((void **)&gC2F, IndexC2F[nTCell]*sizeof(IntType));
	cudaMalloc((void **)&gIndexC2F, nTCell*sizeof(IntType));
	cudaMemcpy(gC2F, C2F[0], IndexC2F[nTCell]*sizeof(IntType), cudaMemcpyHostToDevice);
	cudaMemcpy(gIndexC2F, IndexC2F, nTCell*sizeof(IntType), cudaMemcpyHostToDevice);
	
	cudaMalloc((void **)&gnFPC, nTCell*sizeof(IntType));
	cudaMemcpy(gnFPC, nFPC, nTCell*sizeof(IntType), cudaMemcpyHostToDevice);
	
	size_t sizecell = (nTCell + nBFace)*sizeof(RealFlow);
	cudaMalloc((void **)&gvol, sizecell);
	cudaMemcpy(gvol, vol, sizecell, cudaMemcpyHostToDevice);
	
	size_t sizeface = nTFace*sizeof(RealGeom);
	cudaMalloc((void **)&gxfn, sizeface);
	cudaMalloc((void **)&gyfn, sizeface);
	cudaMalloc((void **)&gzfn, sizeface);
	cudaMemcpy(gxfn, xfn, sizeface, cudaMemcpyHostToDevice);
	cudaMemcpy(gyfn, yfn, sizeface, cudaMemcpyHostToDevice);
	cudaMemcpy(gzfn, zfn, sizeface, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **)&garea, sizeface);
	cudaMemcpy(garea, area, sizeface, cudaMemcpyHostToDevice);

	cudaMalloc((void **)&gdt, nTCell*sizeof(RealFlow));
	cudaMemcpy(gdt, dt, nTCell*sizeof(RealFlow), cudaMemcpyHostToDevice);
	
	cudaMalloc((void **)&gnorm_dist_c2c, nTFace*sizeof(RealGeom));
	cudaMemcpy(gnorm_dist_c2c, norm_dist_c2c, nTFace*sizeof(RealGeom), cudaMemcpyHostToDevice);
    
    if(gvis_run){
		cudaMalloc((void **)&gvis_l, sizecell);
		cudaMalloc((void **)&gvis_t, sizecell);
	}
    cudaMalloc((void **)&gq0, sizecell);
	cudaMalloc((void **)&gq1, sizecell);
	cudaMalloc((void **)&gq2, sizecell);
	cudaMalloc((void **)&gq3, sizecell);
	cudaMalloc((void **)&gq4, sizecell);
    
    cudaMalloc((void **)&gmatrix_jacobi_d, nTCell*25*sizeof(RealFlow));
    cudaMalloc((void **)&gmatrix_jacobi_fc, nTCell*25*sizeof(RealFlow));
    cudaMalloc((void **)&gmatrix_temp, nTCell*25*sizeof(RealFlow));
}

void CopyNonConstData2Device(IntType nTCell, IntType nBFace, RealFlow *vis_l,  RealFlow *vis_t, RealFlow *q[5])
{
	size_t sizecell = (nTCell + nBFace)*sizeof(RealFlow);
	
	if(gvis_run){
		cudaMemcpy(gvis_l, vis_l, sizecell, cudaMemcpyHostToDevice);
		cudaMemcpy(gvis_t, vis_t, sizecell, cudaMemcpyHostToDevice);
	}
	
	cudaMemcpy(gq0, q[0], sizecell, cudaMemcpyHostToDevice);
	cudaMemcpy(gq1, q[1], sizecell, cudaMemcpyHostToDevice);
	cudaMemcpy(gq2, q[2], sizecell, cudaMemcpyHostToDevice);
	cudaMemcpy(gq3, q[3], sizecell, cudaMemcpyHostToDevice);
	cudaMemcpy(gq4, q[4], sizecell, cudaMemcpyHostToDevice);
}