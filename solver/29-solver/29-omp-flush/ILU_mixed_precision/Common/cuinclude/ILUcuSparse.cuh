#ifdef FS_CUSPARSE

//#include "cuFillMatCOO.cuh"
#include "number_type.h"
#include "grid_polyhedra.h"
//#include "petscdevice.h"
#include <iostream>
#include "cusparse_v2.h"
#include "grid_polyhedra.h"
#include "cuData.cuh"

using namespace mflow;
using namespace gpuData;

void cuGMRESILU( PolyGrid *grid, IntType level, IntType iter, RealFlow *DQ[5] );
void cuInitialMatrix( PolyGrid *grid, IntType level );
void cuMatrixLUSGS( PolyGrid *grid, IntType level, IntType iter_done, RealFlow *DQ[5] );
#endif