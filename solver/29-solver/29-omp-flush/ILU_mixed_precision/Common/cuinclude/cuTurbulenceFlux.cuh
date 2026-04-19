#include "number_type.h"
#include "grid_polyhedra.h"
#include "data_pool.h"
#include "solver_base.h"
#include "solver_turb_sa.h"

#include <cuData.cuh>
#include <cuErrorReturn.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

using namespace gpuData;

__global__ void gpuAddSourceSA(RealFlow* res, RealFlow* lhsmat, const RealFlow *rho, const RealFlow *sa_nu, const RealFlow *vis_l, 
							const RealFlow *omaga, const RealGeom *dist2wall, const RealGeom *vol, const IntType *IndexC2C, 
							const IntType *f2c, const RealFlow xminn, const IntType nTCell);

void cuInviscidFluxScalar(PolyGrid *grid, const char *name);

void cuViscousFluxScalar(PolyGrid *grid, const char *name);

void cuAddSourceScalar(PolyGrid *grid, const char *name);

void cuLoadBackResSA(PolyGrid *grid);

void cuViscousFluxScalar3D_New3(PolyGrid *grid, const char *name);
void cuViscousMatsScalar(PolyGrid *grid, const char *name);
void cuViscousDqScalar(PolyGrid *grid, const char *name, RealFlow *dqdl,   RealFlow *dqdr, IntType ns, IntType ne);
void cuPutScalarDqToLhs(PolyGrid *grid, IntType ns, IntType ne);

__global__ void gpuViscousFluxScalar(RealFlow *flux, RealFlow *tem, RealFlow *tem_c2, const RealFlow *k, const RealFlow *rho,
							const RealFlow *vis_l, const RealFlow *xcc, const RealFlow *ycc, const RealFlow *zcc, 
							const RealFlow *xfc, const RealFlow *yfc, const RealFlow *zfc, 
							const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, const IntType *f2c, 
							const IntType *type_bcr, const RealGeom *area, RealGeom *angle_h, const RealFlow sigma,
							const IntType TurM, const IntType nBFace, const IntType nTFace);
__global__ void gpuViscousFluxScalar2(RealFlow *tem, RealFlow *tem_c2, const RealFlow *k, const RealFlow *rho,
							const IntType *f2c, const RealFlow sigma,
							const IntType nBFace, const IntType nTFace);
__global__ void gpuViscousFluxScalar3Reduction(RealFlow* res, const RealFlow* flux, const RealFlow* tem, const RealFlow* tem_c2, 
											const IntType* C2F, const IntType* IndexC2F, const IntType* nFPC, 
											const IntType* f2c, const IntType nTFace, const IntType nTCell);
__global__ void gpuViscousDqScalar(RealFlow* dqdl, RealFlow* dqdr, const RealFlow *rho, const RealFlow *k, const RealFlow *vis_l, 
							const RealFlow *xfc, const RealFlow *yfc, const RealFlow *zfc,
							const RealFlow *xcc, const RealFlow *ycc, const RealFlow *zcc,
							const RealFlow *xfn, const RealFlow *yfn, const RealFlow *zfn, const RealGeom *area,
							const IntType *f2c, const RealFlow sigma, const IntType TurM, const IntType nTFace);
__global__ void gpuPutScalarDqToLhsReduction(RealFlow* lhsmat, const RealFlow* dqdl, const RealFlow* dqdr, const IntType* C2F, 
									const IntType* IndexC2F, const IntType* nFPC, const IntType* nCPC, const IntType* IndexC2C, 
									const IntType* f2c, const IntType* fcptr, const IntType nTFace, const IntType nTCell);
									
__device__ double SQRT_SIXSA(double x);

__device__ double GPUMINSA(double a, double b);

__device__ double GPUMAXSA(double a, double b);

RealFlow* CalAngle(PolyGrid* grid);