#include "number_type.h"
#include "grid_polyhedra.h"
#include "data_pool.h"
#include "solver_base.h"
#include "turbulence.h"
#include "zone.h"

#include <cuData.cuh>
#include <cuErrorReturn.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

using namespace gpuData;

__global__ void gpuComputeTurbGeneration_SA(RealFlow *omaga, const RealFlow *dqdx, const RealFlow *dqdy, 
										const RealFlow *dqdz, const IntType n, const IntType nTCell);
__global__ void gpuZeroGridResiduals(RealFlow *res, IntType n);
__global__ void gpuSAdtlhsmat(RealFlow *lhsmat, RealFlow *dt, const RealFlow *rho, const RealFlow *vol, 
						const IntType *nCPC, const IntType *IndexC2C, 
						const RealFlow turb_cfl_times, const IntType nTCell);

void cuComputeTurbInf_SA(RealFlow *gradnue2);

void cuSAsolve(PolyGrid *grid);								

__device__ double GPUMIN3(double a, double b);

__device__ double GPUMAX3(double a, double b);

__device__ double atomicExchSM35SA(double* address, double val);