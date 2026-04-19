#include <stdio.h>

#include <number_type.h>
#include <grid_patch_type.h>
#include <data_pool.h>
#include <zone.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

__global__ void gpuGetTemperature(RealFlow *t, const RealFlow *q, const RealFlow gascon, const RealFlow p_bar, 
								const IntType nBFace, const IntType nTCell);

//..................................................................................................................................//
__global__ void gpuCalDeriWeight(const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, const IntType *f2c, 
							const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc, const RealGeom *xfn, 
							const RealGeom *yfn, const RealGeom *zfn, const RealGeom *vol, RealGeom *deltl, 
							RealGeom *deltr, const IntType nTFace, const IntType key);
void cuCalDeriWeight(RealGeom *deltl, RealGeom *deltr, IntType key);
//..................................................................................................................................//


//..................................................................................................................................//
__global__ void gpuCalVisHeatFace_averageLaminar(RealFlow *visc_f, RealFlow *heat_f, const RealFlow *vis_l, const IntType *f2c, 
												const RealFlow heat, const IntType nTFace);
__global__ void gpuCalVisHeatFace_averageTurbulentSA(RealFlow *visc_f, RealFlow *heat_f, const RealFlow *vis_t, const IntType *f2c, 
												const RealFlow heat, const IntType nTFace);
void cuCalVisHeatFace_average(PolyGrid *grid, RealFlow *vis_l, RealFlow *visc_f, RealFlow *heat_f);
//..................................................................................................................................//


//..................................................................................................................................//
__global__ void gpuCalVeloandTFace_average(const RealFlow *q, const RealFlow *t, const IntType *f2c, RealFlow *vel_f, 
										 RealFlow *t_f, const IntType nTFace, const int nBFace, const int nTCell);
void cuCalVeloandTFace_average(PolyGrid *grid, RealFlow *vel_f[3], RealFlow *vel[3],
                        RealFlow *t_f, RealFlow *t);
//..................................................................................................................................//

//..................................................................................................................................//
__global__ void gpuCalVisFluxTest(const RealFlow *q, const RealFlow *t, const RealFlow *dqdx, const RealFlow *dqdy, const RealFlow *dqdz, 
								const RealFlow *dtdx, const RealFlow *dtdy, const RealFlow *dtdz, const RealFlow *vel_f, 
								const RealFlow *t_f, const RealFlow *visc_f, const RealFlow *heat_f, const RealFlow *deltl, 
								const RealFlow *deltr, RealFlow *flux, const IntType *f2c, const RealGeom *area, 
								const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc, const RealGeom *xcc, const RealGeom *ycc, 
								const RealGeom *zcc, const RealGeom *xfn, const RealGeom *yfn, const RealGeom *zfn, 
								const RealGeom *facecentroidskewness, const IntType *type_bcr, const RealGeom *tw_bcr,
								const IntType level, IntType warn, const RealGeom two3, const RealGeom BadFaceAngle, 
								const IntType nTFace, const IntType nBFace, const IntType nTCell);
void cuCalVisFluxTest(PolyGrid *grid, RealFlow *vel[3], RealFlow *t, RealFlow *vel_f[3],
                    RealFlow *visc_f, RealFlow *heat_f, RealFlow *t_f,
                    RealFlow *dqdx[3], RealFlow *dqdy[3], RealFlow *dqdz[3],
                    RealFlow *dtdx, RealFlow *dtdy, RealFlow *dtdz,
                    RealGeom *deltl, RealGeom *deltr, RealFlow *flux[5]);
//..................................................................................................................................//

//..................................................................................................................................//

void cuLoadFluxVis(PolyGrid *grid, RealFlow* flux[5]);
//..................................................................................................................................//

void cuViscousFlux(PolyGrid *grid, IntType level);

RealFlow *cuGetTemperature(PolyGrid *grid);

#if (defined LOOPMERGE)
void cuViscousFlux_merge(PolyGrid *grid, IntType level);
#endif

__device__ double atomicExchSM35T(double* address, double val);
