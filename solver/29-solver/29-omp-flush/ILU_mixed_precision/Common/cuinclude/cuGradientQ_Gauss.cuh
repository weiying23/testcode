#include <cuData.cuh>
#include <cuErrorReturn.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

using namespace gpuData;

__global__ void gpuCompNodefacq(RealFlow *facq, const RealFlow *q, const IntType *f2c, const IntType *type_bcr, const IntType nBFace);
__global__ void gpuCompNodefacq2q_n(RealFlow *q_n, const RealFlow *facq, const RealGeom *WeightNodeBFace2C,
								const IntType *type_bcr, const IntType *F2N, const IntType *IndexF2N,
								const IntType *nNPF, const IntType nBFace);
__global__ void gpuCompNodefacq2q_n2(RealFlow *q_n, const RealFlow *facq, const RealGeom *WeightNodeBFace2C,
								const IntType *type_bcr, const IntType *F2N, const IntType *IndexF2N,
								const IntType *nNPF, const IntType *Nmark, const IntType nBFace);
__global__ void gpuCompNodefacq2q_n3(RealFlow *q_n, const RealFlow *facq, const RealGeom *WeightNodeBFace2C,
								const IntType *type_bcr, const IntType *F2N, const IntType *IndexF2N,
								const IntType *nNPF, const IntType *Nmark, const IntType nBFace);
__global__ void gpuCompNodeq2q_n(RealFlow *q_n, const RealFlow *q, const RealGeom *WeightNodeN2C,
								const IntType *nCPN, const IntType *N2C, const IntType *IndexN2C, 
								const IntType *Nmark, const IntType nTNode);
__global__ void gpuCompNodeq_nWeight(RealFlow *q_n, const RealGeom *WeightNode, const IntType nTNode);
__global__ void gpuGradienttmpxyz(RealFlow *tmpxyz, const RealFlow *q_n,
								const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, 
								const IntType *f2c, const IntType *nNPF,
								const IntType *F2N, const IntType *IndexF2N, const RealFlow *area,
								const IntType nBFace, const IntType nTCell, const IntType nTFace);
__global__ void gpuGradienttmpxyznBFace(RealFlow *tmpxyz, const RealFlow *q_n, const RealFlow *q, 
										const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, 
										const IntType *f2c, const IntType *type_bcr, const IntType *nNPF,
										const IntType *F2N, const IntType *IndexF2N, const RealFlow *area,
										const IntType nBFace, const IntType nTCell);
__global__ void gpuGradientReductionShareMemory2(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, const RealFlow *tmpxyz, const IntType *f2c, 
									const IntType* C2F, const IntType* IndexC2F, const IntType* nFPC, const IntType nTCell,
									const IntType nBFace, const IntType threadsnum);
__global__ void gpuGradientBoundary(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, const RealFlow *q, const IntType *C2F, 
									const IntType *IndexC2F, const IntType *f2c, const IntType *nFPC, const IntType  *cellwallnumber,
									const RealFlow *area, const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, 
									const IntType nTCell, const IntType Cell);
__global__ void gpuGradientBoundary2(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, const RealFlow *q, const IntType *C2F, 
									const IntType *IndexC2F, const IntType *f2c, const IntType *nFPC, const IntType  *CellLayerNo,
									const RealFlow *area, const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, 
									const IntType GaussLayer, const IntType nTCell, const IntType Cell);
__global__ void gpuGradientBoundary2(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, 
								const RealFlow *vol, const IntType nTCell);									

__global__ void gpuCompNodeInit(RealFlow *q_n, IntType nTNode);
__global__ void gpuGradientInit(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, const IntType Cell);

void cuUpdateGhostGradSA(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz);

void cuUpdateGhostGradT(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz);

void cuUpdateGhostGrad(RealFlow **dqdx, RealFlow **dqdy, RealFlow **dqdz);

__global__ void gpuGradienttmpxyznBFace(RealFlow *tmpxyz, const RealFlow *q_n, const RealFlow *q, 
										const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, 
										const IntType *f2c, const IntType *type_bcr, const IntType *nNPF,
										const IntType *F2N, const IntType *IndexF2N, const RealFlow *area,
										const IntType nBFace, const IntType nTCell, const IntType name);

void cuGradientReduction(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, RealFlow *q_n, IntType name);

__global__ void gpuGradientInit(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, const IntType name, const IntType Cell);

void cuGradientInit(RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, IntType name);

void cuCompGradientQ_Gauss_Node(PolyGrid *grid, RealFlow *q, RealFlow *dqdx, RealFlow *dqdy, 
								RealFlow *dqdz, IntType name, RealFlow* u_n, RealFlow* v_n, RealFlow* w_n);

void cuGradientMemoryTrans(const RealFlow *rho, const RealFlow *u, const RealFlow *v, const RealFlow *w, const RealFlow *p);

void cuCompGradientQ(PolyGrid *grid, RealFlow *q, RealFlow *dqdx, RealFlow *dqdy, 
					RealFlow *dqdz, IntType name, RealFlow* u_n, RealFlow* v_n, RealFlow* w_n);

void cuCompGradientQ_SA_MultiStream(PolyGrid *grid);

void cuCompGradientQ_ComputeTimeStep_MultiStream(PolyGrid *grid);

void cuCompGradientQ_MultiStream(PolyGrid *grid, RealFlow **q, RealFlow **dqdx, RealFlow **dqdy, 
					RealFlow **dqdz, RealFlow* u_n, RealFlow* v_n, RealFlow* w_n);
					
__device__ double atomicAddSM35(double* address, double val);


