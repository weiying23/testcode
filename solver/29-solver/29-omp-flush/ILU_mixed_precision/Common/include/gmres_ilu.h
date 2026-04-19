#include "temporal_discretisation_implicit.h"

// C++ build-in head files
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cassert>
#include <cstring>
#include <iostream>
using namespace std;

// other user defined head files
#include "number_type.h"
#include "zone.h"
#include "grid_polyhedra.h"
#include "utility_functions.h"
#include "solver_ns.h"
#include "io_base_format.h"
#include "io_log.h"
#include "parallel_base_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"
#include "linsys_solver.h"
// head files relying on condition-compiling
#ifdef MPICH
#include <mpi.h>
#endif

namespace mflow
{
    static inline MATRIXTYPE Sign(MATRIXTYPE x, MATRIXTYPE y);
    void ApplyGivens(MATRIXTYPE s, MATRIXTYPE c, MATRIXTYPE & h1, MATRIXTYPE & h2);
    void GenerateGivens(MATRIXTYPE & dx, MATRIXTYPE & dy, RealFlow & s, MATRIXTYPE & c);
    void SolveReduced(int n, MATRIXTYPE** Hsbg, MATRIXTYPE* rhs, MATRIXTYPE *&x);
    void GmresILU( PolyGrid *grid, IntType level );
    void GMRESSolver( PolyGrid *grid, IntType level );
    void SetupMatrix( PolyGrid *grid, IntType level, MATRIXTYPE *&val );
    IntType CalCOOInfoMPI( PolyGrid *grid, IntType *&oor, \
        IntType *&ooc, IntType *MatrixN, IntType *MatrixNNZ);
    void ComputeBsrIndex( PolyGrid *grid, IntType *&row_ptr, \
        IntType *&col_ind, IntType *&dia_ptr, IntType nnz, IntType MatrixN );
    void ComputeCsrIndex( PolyGrid *grid, IntType *oor, IntType *ooc, \
        IntType *&csr_row_ptr, IntType *&csr_col_ind, IntType *csrIndex, \
        IntType MatrixN, IntType nvar, IntType nnz );
    void ReorderedIndex( PolyGrid *grid, IntType *oor, IntType *ooc, \
        IntType *row_ptr, IntType *bsr_col_ind, IntType *&bsrIndex, IntType nnz, IntType MatrixN );  

}