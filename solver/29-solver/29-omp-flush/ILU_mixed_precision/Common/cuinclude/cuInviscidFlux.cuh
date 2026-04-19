#include <stdio.h>
#include <number_type.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

__global__ void gpuLoadFlux(RealFlow* res, const RealFlow* flux, const IntType* C2F, const IntType* IndexC2F, 
						const IntType* nFPC, const IntType* f2c, const IntType nTFace, const IntType nTCell);
__global__ void gpuLoadFluxShareMemory2(RealFlow* res, const RealFlow* flux, const IntType* C2F, const IntType* IndexC2F, 
						const IntType* nFPC, const IntType* f2c, const IntType nTFace, const IntType nTCell, const IntType threadsnum);
__global__ void gpuCalIsShockFace(IntType *IsShockFace, const RealFlow *q, const RealFlow *xfn, 
								const RealFlow *yfn, const RealFlow *zfn, const IntType *f2c, 
								const RealFlow pref, const RealFlow ThdShock, const IntType nTFace, 
								const IntType nTCell, const IntType nBFace);
__global__ void gpuInviscidFlux_merge_bface(const double* q, const int* f2c,
		const double* xfn, const double* yfn, const double* zfn, const double* vgn, const IntType* type_bcr,
		const int steady, const int nTFace, const int nBFace, const int nTCell, const RealFlow* limit, 
		const RealFlow* dqdx, const RealFlow* dqdy, const RealFlow* dqdz, const RealGeom *xfc,
		const RealGeom *yfc, const RealGeom *zfc, const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, 
		double* flux, const double* area, const int* IsShockFace, const int* IsNormalFace,
		const double gamm1, const double p_bar, const double alf_l, const double alf_n,
		const int EntropyCorType);
__global__ void gpuInviscidFlux_merge_iface(const double* q, const int* f2c,
		const double* xfn, const double* yfn, const double* zfn, const double* vgn,
		const int steady, const int nTFace, const int nBFace, const int nTCell, const RealFlow* limit, 
		const RealFlow* dqdx, const RealFlow* dqdy, const RealFlow* dqdz, const RealGeom *xfc,
		const RealGeom *yfc, const RealGeom *zfc, const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, 
		double* flux, const double* area, const int* IsShockFace, const int* IsNormalFace,
		const double gamm1, const double p_bar, const double alf_l, const double alf_n,
		const int EntropyCorType);		

__global__ void gpuRoeFlux(const RealFlow* ql, const RealFlow* qr, double* flux, const double* area, 
						const double* xfn, const double* yfn, const double* zfn, 
						const double* vgn,
						const int* IsShockFace, const int* IsNormalFace,
						const double gamm1, const double p_bar, const double alf_l, const double alf_n, 
						const int nTFace, const int steady, const int EntropyCorType);
void cuRoeFlux(double* ql[5], double* qr[5], double* flux[5], 
			double* area, int* face_act, double* vgn, int* IsNormalFace,  int* IsShockFace,
			double gamm1, double p_bar, double alf_l, double alf_n, 
			int steady, int EntropyCorType);


__global__ void gpuSetQlQrWithQ(const double* q, double* ql, double* qr, const int* f2c, const int nTFace, const int nBFace, const int nTCell);			
//void cuSetQlQrWithQ(double* q[5], double* ql[5], double* qr[5]);
void cuSetQlQrWithQ(double* q[5]);


__global__ void gpuCalcuQlQrBFace(RealFlow* ql, RealFlow* qr, const IntType* f2c, const RealFlow* limit, 
								const RealFlow* dqdx, const RealFlow* dqdy, const RealFlow* dqdz, const IntType* type_bcr,
								const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc, 
								const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, 
								const IntType nTFace, const IntType nBFace, const IntType nTCell, const RealFlow p_bar);
__global__ void gpuCalcuQlQrInFace(RealFlow* ql, RealFlow* qr, const IntType* f2c, const RealFlow* limit, 
								const RealFlow* dqdx, const RealFlow* dqdy, const RealFlow* dqdz,
								const RealGeom *xfc, const RealGeom *yfc, const RealGeom *zfc, 
								const RealGeom *xcc, const RealGeom *ycc, const RealGeom *zcc, 
								const IntType nTFace, const IntType nBFace, const IntType nTCell, const RealFlow p_bar);
void cuCalcuQlQr(RealFlow* ql[5], RealFlow* qr[5], RealFlow **limit, RealFlow *dqdx[5], RealFlow *dqdy[5], RealFlow *dqdz[5]);

__global__ void gpuModQlQrBou(const RealFlow* q, RealFlow* ql, RealFlow* qr, const RealGeom* xfn, const RealGeom* yfn, const RealGeom* zfn,
								const IntType* f2c, const RealGeom* vgn, const IntType* type_bcr, const IntType steady,
								const IntType nTFace, const IntType nBFace, const IntType nTCell);

void cuModQlQrBou(RealFlow* ql[5], RealFlow* qr[5]);

#if (defined FaceColoring)
void cuLoadFluxColor(PolyGrid *grid, RealFlow* res[5], RealFlow* flux[5]);
#endif

__global__ void gpuLoadFlux(RealFlow* res, const RealFlow* flux, const IntType* C2F, const IntType* IndexC2F, 
						const IntType* nFPC, const IntType* f2c, const IntType nTFace, const IntType nTCell);

void cuLoadFlux(RealFlow* res[5], RealFlow* flux[5]);

void cuLoadBackRes(PolyGrid *grid);

void cuInviscidFlux(PolyGrid *grid, RealFlow **limit, IntType level);

#if (defined LOOPMERGE)
void cuInviscidFlux_merge(PolyGrid *grid, RealFlow **limit, IntType level);
#endif

__device__ double atomicAddSM35LoadFlux(double* address, double val);

__device__ double atomicExchSM35res(double* address, double val);
