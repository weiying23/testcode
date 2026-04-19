#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>

// user defined head files
#include "grid_polyhedra.h"
#include "io_log.h"
#include "utility_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"

#include <cuData.cuh>
#include <cuErrorReturn.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef MPICH
#include <mpi.h>
#endif

using namespace mflow;

using namespace gpuData;
