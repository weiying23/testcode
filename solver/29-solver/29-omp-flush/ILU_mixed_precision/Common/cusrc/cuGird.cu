#include <stdio.h>
#include <iostream>

#include <number_type.h>

#include <cuErrorReturn.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

namespace gpuGrid{
	
	//GPU Device Parameter:
	IntType   threadsPerBlock = 512;	//thread size per block 
	
	//Grid Info:
	IntType   gnTNode;					// no. of total nodes
	IntType   gnSP;						// no. of nodes on solid surfaces
	IntType   gnSF;						// Number of the Solid-Face 壁面上Face数量
	
	RealGeom *gxSf, *gySf, *gzSf;		// the coordinate of nodes on solid surfaces
	IntType	 *gindices;
	IntType  *gnSfP, *gSfP, *gnPntS, *gPntS;
	RealGeom *gdistP;
	RealGeom *gx, *gy, *gz;				// the coordinate of nodes
	
	// bounding box accelerating method
	IntType   gnSurfBox;   
    IntType  *gnPt_SurfBox;
    IntType  *gPt_SurfBox;
    RealGeom *gbnd_SurfBox;
	
	void GPUGridDataTrans(IntType nTNode, IntType nSP, IntType nSF, IntType *nSfP, IntType *SfP, 
						IntType *nPntS, IntType *PntS,
						RealGeom *x, RealGeom *y, RealGeom *z, RealGeom *xSf, RealGeom *ySf, RealGeom *zSf
						){
		gnTNode = nTNode;
		gnSP = nSP;
		gnSF = nSF;
		
		HANDLE_API_ERR(cudaMalloc((void **)&gindices, gnTNode*sizeof(IntType)));	
		HANDLE_API_ERR(cudaMalloc((void **)&gdistP, gnTNode*sizeof(RealGeom)));	

		HANDLE_API_ERR(cudaMalloc((void **)&gx, gnTNode*sizeof(RealGeom)));		
		HANDLE_API_ERR(cudaMalloc((void **)&gy, gnTNode*sizeof(RealGeom)));		
		HANDLE_API_ERR(cudaMalloc((void **)&gz, gnTNode*sizeof(RealGeom)));						

		HANDLE_API_ERR(cudaMalloc((void **)&gxSf, gnSP*sizeof(RealGeom)));		
		HANDLE_API_ERR(cudaMalloc((void **)&gySf, gnSP*sizeof(RealGeom)));		
		HANDLE_API_ERR(cudaMalloc((void **)&gzSf, gnSP*sizeof(RealGeom)));
		
		HANDLE_API_ERR(cudaMemcpy(gx, x, gnTNode*sizeof(RealGeom), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gy, y, gnTNode*sizeof(RealGeom), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gz, z, gnTNode*sizeof(RealGeom), cudaMemcpyHostToDevice));
		
		HANDLE_API_ERR(cudaMemcpy(gxSf, xSf, gnSP*sizeof(RealGeom), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gySf, ySf, gnSP*sizeof(RealGeom), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gzSf, zSf, gnSP*sizeof(RealGeom), cudaMemcpyHostToDevice));
		
		HANDLE_API_ERR(cudaMalloc((void **)&gnSfP, (gnSP + 1)*sizeof(IntType)));
		HANDLE_API_ERR(cudaMalloc((void **)&gSfP, nSfP[nSP]*sizeof(IntType)));
		
		HANDLE_API_ERR(cudaMemcpy(gnSfP, nSfP, (gnSP + 1)*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gSfP, SfP, nSfP[nSP]*sizeof(IntType), cudaMemcpyHostToDevice));
		
		HANDLE_API_ERR(cudaMalloc((void **)&gnPntS, (gnSF + 1)*sizeof(IntType)));
		HANDLE_API_ERR(cudaMalloc((void **)&gPntS, nPntS[nSF]*sizeof(IntType)));
		
		HANDLE_API_ERR(cudaMemcpy(gnPntS, nPntS, (gnSF + 1)*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gPntS, PntS, nPntS[nSF]*sizeof(IntType), cudaMemcpyHostToDevice));
		
	}
	
	void GPUGridDataTrans2(IntType nSurfBox, IntType *nPt_SurfBox, IntType *Pt_SurfBox, RealGeom *bnd_SurfBox
						){
		gnSurfBox = nSurfBox;			
		
		HANDLE_API_ERR(cudaMalloc((void **)&gnPt_SurfBox, (gnSurfBox + 1)*sizeof(IntType)));	
		HANDLE_API_ERR(cudaMalloc((void **)&gPt_SurfBox, gnSP*sizeof(IntType)));	
		HANDLE_API_ERR(cudaMalloc((void **)&gbnd_SurfBox, gnSurfBox*6*sizeof(RealGeom)));
		
		HANDLE_API_ERR(cudaMemcpy(gnPt_SurfBox, nPt_SurfBox, (gnSurfBox + 1)*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gPt_SurfBox, Pt_SurfBox, gnSP*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gbnd_SurfBox, bnd_SurfBox, gnSurfBox*6*sizeof(RealGeom), cudaMemcpyHostToDevice));
	}
	
	void GPUGridDataInit(RealGeom *distP, IntType *indices){
		
		HANDLE_API_ERR(cudaMemcpy(gdistP, distP, gnTNode*sizeof(RealGeom), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gindices, indices, gnTNode*sizeof(IntType), cudaMemcpyHostToDevice));
		
	}
	
	void GPUGridDataTransBack(RealGeom *distP){
		
		HANDLE_API_ERR(cudaMemcpy(distP, gdistP, gnTNode*sizeof(RealGeom), cudaMemcpyDeviceToHost));	
		
		HANDLE_API_ERR(cudaFree(gx));
		HANDLE_API_ERR(cudaFree(gy));
		HANDLE_API_ERR(cudaFree(gz));
		
		HANDLE_API_ERR(cudaFree(gxSf));
		HANDLE_API_ERR(cudaFree(gySf));
		HANDLE_API_ERR(cudaFree(gzSf));
		
		HANDLE_API_ERR(cudaFree(gnSfP));
		HANDLE_API_ERR(cudaFree(gSfP));
		
		HANDLE_API_ERR(cudaFree(gnPntS));
		HANDLE_API_ERR(cudaFree(gPntS));
		
		HANDLE_API_ERR(cudaFree(gnPt_SurfBox));
		HANDLE_API_ERR(cudaFree(gPt_SurfBox));
		HANDLE_API_ERR(cudaFree(gbnd_SurfBox));
		
		HANDLE_API_ERR(cudaFree(gindices));
		HANDLE_API_ERR(cudaFree(gdistP));
		
	}
	
	__device__ bool GPUGridEqualZero(RealFlow x) 
	{ 
		return (x > -TINY) && (x < TINY); 
	}
	
	__device__ void gpuFindRp2tri(RealGeom &dist, RealGeom xp, RealGeom yp, RealGeom zp, 
							RealGeom xa, RealGeom ya, RealGeom za, RealGeom xb, RealGeom yb, RealGeom zb,
							RealGeom xc, RealGeom yc, RealGeom zc
							){
		RealGeom pp[3], aa[3], bb[3], cc[3];
		pp[0]=xp; pp[1]=yp; pp[2]=zp;
		aa[0]=xa; aa[1]=ya; aa[2]=za;
		bb[0]=xb; bb[1]=yb; bb[2]=zb;
		cc[0]=xc; cc[1]=yc; cc[2]=zc;
	  
		RealGeom p[3],a[3],b[3],r[3], rr;
		RealGeom daa=0., dbb=0., dab=0., den, dap=0., dbp=0., s, t, dsq;
		for(IntType i=0; i<3; i++)
		{
			p[i] = pp[i] - aa[i];
			a[i] = bb[i] - aa[i];
			b[i] = cc[i] - aa[i];
			daa += a[i]*a[i];
			dbb += b[i]*b[i];
			dab += a[i]*b[i];
		}
		den = dab*dab - daa*dbb;

		if(GPUGridEqualZero(den)) ;   //zhyb: 面积不为零则den不为零
		else
		{
			for(IntType i=0; i<3; i++)
			{
				dap += a[i]*p[i];
				dbp += b[i]*p[i];
			}
			s = (dab*dbp-dbb*dap)/den;
			t = (dab*dap-daa*dbp)/den;
			if( s<0. || t<0. || (t+s)>1. ) ;   //zhyb: 这三种情况垂足落在三角形外边
			else
			{
				for(IntType i=0; i<3; i++) r[i] = p[i]-s*a[i]-t*b[i];
				rr = 0.;
				for(IntType i=0; i<3; i++) rr += r[i]*r[i];
				if ( rr<dist ) dist = rr;
				return;
			}
		}

		dsq = dist;
		if(GPUGridEqualZero(daa)) ;  //zhyb: bb点和aa点重合
		else
		{
			dap = 0.;
			for(IntType i=0; i<3; i++) dap += a[i]*p[i];
			t = dap/daa;
			if( t<0. || t>1.) ;     //zhyb: 这两种情况下垂足落在线段aabb外
			else
			{
				for(IntType i=0; i<3; i++) r[i] = p[i]-t*a[i];
				rr = 0.;
				for(IntType i=0; i<3; i++) rr += r[i]*r[i];
				if ( rr<dsq ) dsq = rr;
			}
		}

		if(GPUGridEqualZero(dbb)) ;   //zhyb: cc点和aa点重合
		else
		{
			dbp = 0.;
			for(IntType i=0; i<3; i++) dbp += b[i]*p[i];
			t = dbp/dbb;
			if( t<0. || t>1. ) ;    //zhyb: 这两种情况下垂足落在线段aacc外
			else
			{
				for(IntType i=0; i<3; i++) r[i] = p[i]-t*b[i];
				rr = 0.;
				for(IntType i=0; i<3; i++) rr += r[i]*r[i];
				if ( rr<dsq ) dsq = rr;
			}
		}
	  
		daa = 0.;
		for(IntType i=0; i<3; i++)
		{
			p[i] = pp[i]-bb[i];
			a[i] = cc[i]-bb[i];
			daa += a[i]*a[i];
		}
	   
		if(GPUGridEqualZero(daa)) ;   //zhyb: cc点和bb点重合
		else
		{
			dap = 0;
			for(IntType i=0; i<3; i++) dap += a[i]*p[i];
			t = dap/daa;
			if( t<0. || t>1. ) ;    //zhyb: 这两种情况下垂足落在线段bbcc外
			else
			{
				for(IntType i=0; i<3; i++) r[i] = p[i]-t*a[i];
				rr = 0.;
				for(IntType i=0; i<3; i++) rr += r[i]*r[i];
				if ( rr<dsq ) dsq = rr;
			}
		}

		if( dsq<dist ) dist = dsq;							
									
									
	}
	
	__global__ void gpuDistP(RealGeom *distP, const IntType *indices, const IntType *nSfP, 
							const IntType *SfP, const IntType *nPntS, const IntType *PntS, 
							const RealGeom *x, const RealGeom *y, const RealGeom *z, 
							const RealGeom *xSf, const RealGeom *ySf, const RealGeom *zSf, 
							IntType nTNode){
		IntType i = blockDim.x*blockIdx.x + threadIdx.x;
		if (i < nTNode){
			IntType face_pnts[20];
			IntType pntmin = indices[i];
			RealGeom distP2P = BIG;
			for(IntType k = nSfP[pntmin]; k < nSfP[pntmin+1]; k++)
			{
				IntType sface = SfP[k];
				IntType tri_pnt[3];

				// real points start from 1
				//face_pnts.resize(1);
				IntType count = 1;
				for(IntType jj = nPntS[sface]; jj < nPntS[sface+1]; jj++)
				{
					face_pnts[count] = PntS[jj];
					count++;
				}
				face_pnts[0] = face_pnts[count - 1];
				face_pnts[count] = face_pnts[1];

				if((nPntS[sface+1] - nPntS[sface]) == 3)
				{
					tri_pnt[0] = face_pnts[1];
					tri_pnt[1] = face_pnts[2];
					tri_pnt[2] = face_pnts[3];
				}
				else if((nPntS[sface+1]-nPntS[sface]) == 4)
				{
					tri_pnt[0] = face_pnts[1];
					tri_pnt[1] = face_pnts[2];
					tri_pnt[2] = face_pnts[3];
					if     ( pntmin==face_pnts[1] ) tri_pnt[2] = face_pnts[4];
					else if( pntmin==face_pnts[3] ) tri_pnt[0] = face_pnts[4];
					else if( pntmin==face_pnts[4] ) tri_pnt[1] = face_pnts[4];
				}
				else if((nPntS[sface+1]-nPntS[sface]) > 4)
				{
					// find the anchor point in the face nodes
					IntType j;
					for(j = 1;j <= nPntS[sface+1]-nPntS[sface]; ++j)
					{
						if(pntmin == face_pnts[j]) 
						{
							count = j;
							j = nPntS[sface+1]-nPntS[sface];
							//break;
						}
					}
					j = count;
					tri_pnt[0] = face_pnts[j-1];
					tri_pnt[1] = face_pnts[j];
					tri_pnt[2] = face_pnts[j+1];
				}
				gpuFindRp2tri( distP2P, x[i], y[i], z[i], xSf[tri_pnt[0]], ySf[tri_pnt[0]], zSf[tri_pnt[0]],
					xSf[tri_pnt[1]], ySf[tri_pnt[1]], zSf[tri_pnt[1]], xSf[tri_pnt[2]], ySf[tri_pnt[2]], zSf[tri_pnt[2]] );
			}
			if(distP2P < distP[i]*distP[i]) distP[i] = sqrt(distP2P);
		}
	}
	
	__global__ void gpuDistPInit(RealGeom *distP, IntType nTNode){

		IntType i = blockDim.x*blockIdx.x + threadIdx.x;
		if (i < nTNode){
			distP[i] = BIG;
		}

	}	
	
	__device__ RealGeom gpuFindRminbox( const RealGeom xp, const RealGeom yp, const RealGeom zp, const RealGeom *bnd){
		
		RealGeom rr, rx, ry, rz;
		if( xp>=bnd[0] && xp<=bnd[3]) rx = 0;
		else rx = ( xp<bnd[0] ) ? bnd[0]-xp : xp-bnd[3];
		if( yp>=bnd[1] && yp<=bnd[4]) ry = 0;
		else ry = ( yp<bnd[1] ) ? bnd[1]-yp : yp-bnd[4];
		if( zp>=bnd[2] && zp<=bnd[5]) rz = 0;
		else rz = ( zp<bnd[2] ) ? bnd[2]-zp : zp-bnd[5];
		
		rr = rx*rx + ry*ry + rz*rz;
	 
		return(rr);
	}
	
	__device__ void gpuquick_sort_OddEven(RealGeom *a, IntType n, IntType *ib){
	
		if(n <= 1) return;
		
		IntType sorted = 0;
		while (!sorted){
			sorted = 1;
			for (IntType i = 1; i < n - 1; i += 2){
				if(a[i] > a[i + 1]){
					RealGeom temp = a[i];
					a[i] = a[i + 1];
					a[i + 1] = temp;
					IntType temp_ib = ib[i];
					ib[i] = ib[i + 1];
					ib[i + 1] = temp_ib;
					sorted = 0;
				}
			}
			
			for (IntType i = 0; i < n - 1; i += 2){
				if(a[i] > a[i + 1]){
					RealGeom temp = a[i];
					a[i] = a[i + 1];
					a[i + 1] = temp;
					IntType temp_ib = ib[i];
					ib[i] = ib[i + 1];
					ib[i + 1] = temp_ib;
					sorted = 0;
				}
			}
		}		
	}
	
	__global__ void gpuNodeSearchIndexBox(RealGeom *lmin, IntType *indices, 
										const IntType *nPt_SurfBox, const IntType *Pt_SurfBox, const RealGeom *bnd_SurfBox,
										const RealGeom *xs, const RealGeom *ys, const RealGeom *zs, 
										const RealGeom *xin, const RealGeom *yin, const RealGeom *zin, 
										const IntType nSurfBox, const IntType nTNode){

		IntType ip = blockDim.x*blockIdx.x + threadIdx.x;
		if (ip < nTNode){
			RealGeom error = 1e-6;
			RealGeom rr[40];               	// MinDist::Init(void): MAX_NODES_BOX = 20
			IntType BSort[40];				// the length of rr[] and BSort[] was set as MAX_NODES_BOX*2
			
			for(IntType ibox = 0; ibox < nSurfBox; ++ibox)
			{
				rr[ibox] = gpuFindRminbox(xin[ip], yin[ip], zin[ip], &bnd_SurfBox[ibox*6]); // square of distance
				BSort[ibox] = ibox;
			}

			gpuquick_sort_OddEven(rr, nSurfBox, BSort);
			
			for(IntType ibox = 0; ibox < nSurfBox; ++ibox)
			{
				// point in box which distance is lager than lmin is ignored
				if(rr[ibox] < lmin[ip]+error)
				{
					IntType sorted_box = BSort[ibox];
					for(IntType k = nPt_SurfBox[sorted_box]; k < nPt_SurfBox[sorted_box+1]; ++k)
					{
						IntType pnt = Pt_SurfBox[k];
						RealGeom dx = xin[ip] - xs[pnt];
						RealGeom dy = yin[ip] - ys[pnt];
						RealGeom dz = zin[ip] - zs[pnt];
						RealGeom len = dx*dx + dy*dy + dz*dz;
						if(len < lmin[ip])
						{
							lmin[ip] = len;
							indices[ip] = pnt;
						}
					}
				}
			}
		
		}

	}	
	
	__global__ void gpuDistPSqrt(RealGeom *distP, IntType nTNode){

		IntType i = blockDim.x*blockIdx.x + threadIdx.x;
		if (i < nTNode){
			distP[i] = sqrt(distP[i]);
		}

	}

	void cuSearchIndex(RealGeom *distP, IntType *indices){	
		
		IntType blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;	
		gpuDistPInit <<< blocksPerGrid, threadsPerBlock >>> (gdistP, gnTNode);
		
		gpuNodeSearchIndexBox <<< blocksPerGrid, threadsPerBlock >>> (gdistP, gindices, 
										gnPt_SurfBox, gPt_SurfBox, gbnd_SurfBox, gxSf, gySf, gzSf, 
										gx, gy, gz, gnSurfBox, gnTNode);
										
		gpuDistPSqrt <<< blocksPerGrid, threadsPerBlock >>> (gdistP, gnTNode);
	}
	
	void cuComputeDist2Wall(RealGeom *distP, IntType *indices){
		
		//GPUGridDataInit(distP, indices);
		
		IntType blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;	
		gpuDistP <<< blocksPerGrid, threadsPerBlock >>> (gdistP, gindices, gnSfP, gSfP, gnPntS, gPntS, 
							gx, gy, gz, gxSf, gySf, gzSf, gnTNode);
		
		GPUGridDataTransBack(distP);
	}
	
}

