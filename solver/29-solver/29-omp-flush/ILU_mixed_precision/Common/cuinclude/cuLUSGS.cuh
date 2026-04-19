#include <cuData.cuh>
#include <cuErrorReturn.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

using namespace gpuData;

__global__ void gpuCellIsMG(IntType *det, IntType nTCell);

__global__ void gpuCellIsMG2(RealFlow *tmpvar, const RealFlow *p, const IntType *f2c, const RealFlow p_bar, IntType nTFace);

__global__ void gpuCellIsMG3(IntType *det, const RealFlow *tmpvar, const IntType *f2c, const IntType* C2F,
							const IntType* IndexC2F, const IntType* nFPC, const RealFlow stind, 
							const IntType nTCell, const IntType nBFace, const IntType nTFace);

void cures2DQ(RealFlow *DQ[5], RealFlow *res, IntType name);

void cuCalDiagLUSGS(PolyGrid *grid, IntType level);

void cuDiagInit(RealFlow *dt);

__global__ void gpuDQInit(RealFlow *DQ, IntType nT5);

void cuDQInit(IntType nvar);

__global__ void gpuReductionDiag3(RealFlow *Diag, const RealFlow *tmpvar, const IntType *f2c, const IntType* C2F,
								const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell, const IntType nBFace, 
								const IntType nTFace);
								
#if (defined ShareMemory)
__global__ void gpuReductionDiag3ShareMemory(RealFlow *Diag, const RealFlow *tmpvar, const IntType *f2c, const IntType* C2F,
								const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell, const IntType nBFace, 
								const IntType nTFace);
#endif	

void cuUpdateFlowField3D_CFL3d(PolyGrid *grid, RealFlow *DQ[5]);

void cuTimeStepNormal_new(PolyGrid *grid, IntType vis_run);

void cuLimitTimeStep(PolyGrid *grid);

__global__ void gpuTimeStepNormal_new (RealFlow *dt, IntType nTCell);
__global__ void gpuTimeStepNormal_new2 (RealFlow *tmpvar, const RealFlow *q, const RealFlow *vis_l, const RealFlow *vis_t, 
									const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc, 
									const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, 
									const RealGeom *xfn, const RealGeom *yfn, const RealGeom *zfn, const RealGeom *area, 
									const RealGeom *vol, 
									const IntType *f2c, const RealGeom *vgn, const IntType steady, const RealFlow p_bar, 
									const RealFlow gam, const RealFlow prl, const RealFlow prt, const RealFlow C,
									const IntType vis_run, const IntType nTCell, const IntType nBFace);
__global__ void gpuTimeStepNormal_new3 (RealFlow *tmpvar, const RealFlow *q, const RealFlow *vis_l, const RealFlow *vis_t, 
									const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc, 
									const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, 
									const RealGeom *xfn, const RealGeom *yfn, const RealGeom *zfn, const RealGeom *area, 
									const RealGeom *vol, 
									const IntType *f2c, const RealGeom *vgn, const IntType steady, const RealFlow p_bar, 
									const RealFlow gam, const RealFlow prl, const RealFlow prt, const RealFlow C,
									const IntType vis_run, const IntType nTFace, const IntType nTCell, const IntType nBFace);
__global__ void gpuTimeStepNormal_newReduction(RealFlow *dt, const RealFlow *tmpvar, const IntType *f2c, const IntType* C2F,
								const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell);									

__global__ void gpuLimitTimeStep_dtmindtmax (RealFlow *dtmax, RealFlow *dtmin, RealFlow *dt, const RealFlow *p, const IntType *det, const RealFlow cfl, 
								const RealFlow cfl_min, const RealFlow p_min, const RealFlow p_break, const IntType nTCell);
__global__ void Reducekernel6_Max(double *g_idata, double *g_odata, int n);
__global__ void Reducekernel6_Min(double *g_idata, double *g_odata, int n);
__global__ void Reducekernel_Max(double *val_Reduction, double *g_odata, int n);
__global__ void Reducekernel_Min(double *val_Reduction, double *g_odata, int n);
__global__ void gpuLimitTimeStep2 (RealFlow *dt, const RealFlow dt_max_lim, const IntType nTCell);

void cuForwardStep(PolyGrid *grid, RealFlow *rhs, IntType level, IntType steps);

void cuTimeMarch(PolyGrid *grid, RealFlow **q, RealFlow *dt, RealFlow lamda);

void cuExplicitStep(PolyGrid* grid);

void cuSolveLUSGS3D(PolyGrid *grid, RealFlow *Diag, RealFlow *DQ[5], IntType *nFPC, IntType **C2F, IntType level);

void cuZeroResiduals(PolyGrid *grid);

void cuComputeTimeStep(PolyGrid *grid);

void cuComputeVis_l(PolyGrid *grid);

void cuSetGhostVariables(PolyGrid *grid);

void cuSetGhostQuantityGradients(const PolyGrid *grid, RealFlow **dqdx, RealFlow **dqdy, RealFlow **dqdz);

void cuSolveADU3D(PolyGrid *grid, RealFlow **rhs, RealFlow *DQ[5], IntType *nFPC, IntType **C2F, IntType level);

void cuScalarGMRESlimitdq(IntType DQ_limit, RealFlow q_min);

__device__ void GPUSolveEquationforGradSYMM(RealFlow gv1[9], RealFlow gv2[9], RealGeom xfn, RealGeom yfn, RealGeom zfn);

__device__ double atomicExchSM35(double* address, double val);

__device__ void GPUFluxLUSGS3D(RealFlow flux[5], RealFlow q[5], RealFlow DQ[5], RealGeom fa_n[3], 
							RealFlow gam, RealFlow p_bar, RealFlow lhs_omga);

__device__ double GPUMIN2(double a, double b);

__device__ double GPUMAX2(double a, double b);