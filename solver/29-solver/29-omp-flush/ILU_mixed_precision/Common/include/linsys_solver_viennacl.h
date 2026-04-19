#ifdef USING_VIENNACL
#ifndef FS_CUDA
#ifndef LINSYS_SOLVER_VIENNACL_H
#define LINSYS_SOLVER_VIENNACL_H

#include "number_type.h"
#include <vector>
#include <map>
namespace mflow
{
void SetViennaclGMRESPara(RealFlow tol, IntType maxits, IntType kspan, IntType pcType);
void SetViennaclChowPatelILUPara(IntType sweeps, IntType jacobi_iters);
void SetViennaclBlockILUSize(IntType size);
void viennaclGMRESSolve(std::vector<std::map<unsigned int, RealFlow> > & stl_A, std::vector<RealFlow> & stl_b, std::vector<RealFlow> & stl_x );
void SetViennaclILULevelScheduling(IntType flag);
}
#endif
#endif
#endif