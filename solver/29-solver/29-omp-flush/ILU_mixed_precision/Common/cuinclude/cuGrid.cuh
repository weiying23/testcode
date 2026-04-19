#include <stdio.h>
#include <iostream>

#include <number_type.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

namespace gpuGrid{
	
	//GPU Device Parameter:
	extern IntType   threadsPerBlock;	//thread size per block 
	
	//Grid Info:
	extern IntType   gnTNode;					// no. of total nodes
	extern IntType   gnSP;						// no. of nodes on solid surfaces
	extern IntType   gnSF;						// Number of the Solid-Face 壁面上Face数量
	
	extern RealGeom *gxSf, *gySf, *gzSf;		// the coordinate of nodes on solid surfaces
	extern IntType	*gindices;
	extern IntType  *gnSfP, *gSfP, *gnPntS, *gPntS;
	extern RealGeom *gdistP;	
	extern RealGeom *gx, *gy, *gz;					// the coordinate of nodes
	
	// bounding box accelerating method
	extern IntType   gnSurfBox;   
    extern IntType  *gnPt_SurfBox;
    extern IntType  *gPt_SurfBox;
    extern RealGeom *gbnd_SurfBox;
	
	void GPUGridDataTrans(IntType nTNode, IntType nSP, IntType nSF, IntType *nSfP, IntType *SfP, 
						IntType *nPntS, IntType *PntS,
						RealGeom *x, RealGeom *y, RealGeom *z, RealGeom *xSf, RealGeom *ySf, RealGeom *zSf);
	void GPUGridDataTrans2(IntType nSurfBox, IntType *nPt_SurfBox, IntType *Pt_SurfBox, RealGeom *bnd_SurfBox);
	
	void GPUGridDataInit(RealGeom *distP, IntType *indices);
	
	void GPUGridDataTransBack(RealGeom *distP);
	
	__global__ void gpuDistP(RealGeom *distP, const IntType *indices, const IntType *nSfP, 
							const IntType *SfP, const IntType *nPntS, const IntType *PntS, 
							const RealGeom *x, const RealGeom *y, const RealGeom *z, 
							const RealGeom *xSf, const RealGeom *ySf, const RealGeom *zSf, 
							IntType nTNode);
							
	__device__ void gpuFindRp2tri(RealGeom &dist, RealGeom xp, RealGeom yp, RealGeom zp, 
							RealGeom xa, RealGeom ya, RealGeom za, RealGeom xb, RealGeom yb, RealGeom zb,
							RealGeom xc, RealGeom yc, RealGeom zc);
							
	void cuSearchIndex(RealGeom *distP, IntType *indices);		
	
	void cuComputeDist2Wall(RealGeom *distP, IntType *indices);
}

