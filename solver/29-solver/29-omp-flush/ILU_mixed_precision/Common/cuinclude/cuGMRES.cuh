#include <cuData.cuh>
#include <cuErrorReturn.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

using namespace gpuData;

__global__ void Reducekernel_sum(double *val_Reduction, double *g_odata, int n);
__global__ void Reducekernel6(double *g_idata, double *g_odata, int n);

void cuMemoryPreparaGMRESDebug(PolyGrid *grid);

void cuMemoryPreparaGMRESDebug2(PolyGrid *grid);

void cuGMRESSolverOrig(PolyGrid *grid, IntType level);

void cuGMRESSolverOrigUpdate(PolyGrid *grid, IntType level);

void cuSolveScalarGMRES(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, RealFlow *dq, IntType *nCPC, IntType **c2c, const char *name, IntType level);  

