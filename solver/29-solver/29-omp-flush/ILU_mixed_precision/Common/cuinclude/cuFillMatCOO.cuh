#ifdef USING_PETSC
#include <petscmat.h>
#endif
#include <number_type.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuData.cuh"
#include "grid_polyhedra.h"
using namespace mflow;

#ifdef USING_PETSC
	PetscErrorCode FillMatrixCUDACOO(Mat A, IntType nnz);
#endif

void CopyConstData2Device(IntType nTCell, IntType nBFace, IntType nTFace, IntType ifStart, IntType vis_run, RealFlow gam, RealFlow p_bar, RealFlow alf_l, IntType *coo, IntType *f2c, IntType **C2F, IntType *IndexC2F, IntType *nFPC, 
RealGeom *vol, RealGeom *xfn, RealGeom *yfn, RealGeom *zfn, RealFlow *dt, RealGeom *norm_dist_c2c, RealGeom *area);

void CopyNonConstData2Device(IntType nTCell, IntType nBFace, RealFlow *vis_l,  RealFlow *vis_t, RealFlow *q[5]);

void CalCOOInfo_new( PolyGrid *grid, IntType * & oor, IntType * &ooc, IntType Bstart);
