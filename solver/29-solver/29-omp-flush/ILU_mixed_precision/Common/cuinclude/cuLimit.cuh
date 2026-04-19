#include <stdio.h>

#include <number_type.h>
#include <grid_patch_type.h>
#include <data_pool.h>
#include <zone.h>
#include <utility_functions.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

__global__ void gpuMaxMinDiffInit(RealFlow *dmax, RealFlow *dmin, const RealFlow *q, 
								const IntType nTCell, const IntType nBFace, const IntType name);

__global__ void gpuMaxMinDiff(RealFlow *dmax, RealFlow *dmin, const RealFlow *q, const IntType* C2F,
							const IntType* IndexC2F, const IntType* nFPC, const IntType* f2c, 
							const IntType* type_bcr, const IntType nTCell, const IntType nBFace, const IntType name);

__global__ void gpuMaxMinDiffReduceQ(RealFlow *dmax, RealFlow *dmin, const RealFlow *q, 
								const IntType nTCell, const IntType nBFace, const IntType name);

void cuMaxMinDiff(RealFlow *q, PolyGrid *grid, IntType name);

__global__ void gpuLimitInit(RealFlow *limit, const IntType nTCell, const IntType nBFace, const IntType name);

__global__ void gpuLimitespcell(RealFlow *espcell, const RealGeom *vol, const RealFlow *q, RealGeom eps_tmp,
								const IntType nTCell, const IntType nBFace, const IntType name);

__global__ void gpuLimitespcell3(RealFlow *espcell, const RealGeom *vol, const RealFlow *q, RealGeom eps_tmp,
								const RealFlow gam, const RealFlow p_bar, 
								const IntType nTCell, const IntType nBFace, const IntType name);
								
__global__ void gpuLimitespcell4(RealFlow *espcell, const RealGeom *vol, const RealFlow *q, const RealGeom eps_tmp, 
								const RealFlow p_bar, const IntType nTCell, const IntType nBFace, const IntType name);

__global__ void gpuVencatLimitnBFace(RealFlow *tmp_limit, const RealFlow *dmax, const RealFlow *dmin, const RealFlow *espcell, 
									const RealGeom eps_tmp, const RealFlow *dqdx, const RealFlow *dqdy, const RealFlow *dqdz,
									const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc,
									const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, const IntType *f2c, const IntType nTCell, 
									const IntType nBFace, const IntType nTFace, const IntType name);
									
__global__ void gpuVencatLimit(RealFlow *tmp_limit, const RealFlow *dmax, const RealFlow *dmin, const RealFlow *espcell, 
								const RealGeom eps_tmp, const RealFlow *dqdx, const RealFlow *dqdy, const RealFlow *dqdz,
								const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc,
								const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, const IntType *f2c, const IntType nTCell, 
								const IntType nBFace, const IntType nTFace, const IntType name);							

__global__ void gpuVencatLimitAtomicnBFace(RealFlow *limit, const RealFlow *dmax, const RealFlow *dmin, const RealFlow *espcell, 
									const RealGeom eps_tmp, const RealFlow *dqdx, const RealFlow *dqdy, const RealFlow *dqdz,
									const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc,
									const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, const IntType *f2c, const IntType nTCell, 
									const IntType nBFace, const IntType nTFace, const IntType name);

__global__ void gpuVencatLimitReduction(RealFlow *limit, const RealFlow *tmp_limit, const IntType *f2c, const IntType* C2F,
										const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell, const IntType nBFace, 
										const IntType nTFace, const IntType name);

void cuVencatLimiter(PolyGrid *grid, RealFlow *limit, RealFlow *q, RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, IntType name);

__global__ void gpuLimitInit(RealFlow *limit, const IntType Cell);

void cuLimitInit(RealFlow **limit);

#ifdef MultiStream
__global__ void gpuMaxMinDiffInit_Merged(RealFlow *dmax, RealFlow *dmin, const RealFlow *q, 
								const IntType nTCell, const IntType nBFace, const IntType name);

__global__ void gpuMaxMinDiff_Merged(RealFlow *dmax, RealFlow *dmin, const RealFlow *q, const IntType* C2F,
							const IntType* IndexC2F, const IntType* nFPC, const IntType* f2c, 
							const IntType* type_bcr, const IntType nTCell, const IntType nBFace, const IntType name);

__global__ void gpuMaxMinDiffReduceQ_Merged(RealFlow *dmax, RealFlow *dmin, const RealFlow *q, 
								const IntType nTCell, const IntType nBFace, const IntType name);

__global__ void gpuLimitInit_Merged(RealFlow *limit, const IntType nTCell, const IntType nBFace, const IntType name);
								
void cuVencatLimiter_MultiStream_espcell(IntType name);
#endif

void cuLimitMemoryTrans(RealFlow **dqdx, RealFlow **dqdy, RealFlow **dqdz, const RealFlow *rho, 
					const RealFlow *u, const RealFlow *v, const RealFlow *w, const RealFlow *p);

RealFlow **cuGetLimiters_resp(PolyGrid *grid);

__device__ RealFlow gpuVenFun(RealFlow d, RealFlow dq, RealFlow eps);

__device__ void atomicMax(double *addr, double val);

__device__ void atomicMin(double *addr, double val);


__device__ double kernelMAXDOUBLE(double a, double b);

__device__ double kernelMINDOUBLE(double a, double b);

__device__ double GPUMIN(double a, double b);
__device__ double GPUMAX(double a, double b);
__device__ double GPUABS(double a);
__device__ bool GPUEqualZero(RealFlow x) ;

