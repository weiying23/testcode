//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   number_type.h
/// \brief  define some types and numbers according to the precision
/// \author 
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_NUMBER_TYPE_H
#define MFL_NUMBER_TYPE_H

namespace mflow
{

#ifdef  SINGLE_PRECISION
typedef float   RealGeom;
typedef float   RealFlow;
#define BIG     1.e30
#define TINY    1.e-30
#define MPIReal  MPI_FLOAT
#else   //DOUBLE_PRECISION
typedef double  RealGeom;
typedef double  RealFlow;
#define BIG     1.e40
#define TINY    1.e-40
#define MPIReal  MPI_DOUBLE
#endif

//#define LARGE_GRID
#ifdef LARGE_GRID
typedef long IntType;
#define MPIIntType MPI_LONG
#define INTBIG  9223372036854775807
#else
typedef int IntType;
#define MPIIntType MPI_INT
#define INTBIG 2147483647
#endif

// Define a char array of size MAX_STRING as String.
// MAX_STRING is usually used as a string of long size.
const IntType MAX_STRING = 256;
typedef char String[MAX_STRING];

// Define a char array of size MAX_SHORT_STRING as ShortString.
// ShortString is usually used as the name of parameters 
const IntType MAX_SHORT_STRING = 64;
typedef char ShortString[MAX_SHORT_STRING];

//add by dingxin
#define VEC_SIZE 8
#define MAX_ELEM_PER_PART 256
typedef struct tree_s { // D&C tree structure
    IntType* bfaceID, * ifaceID, * ifaceType;
    IntType firstElem, lastElem, lastSep, vecOffset;
    IntType n_bface, n_iface;
    bool isSep;
    struct tree_s* left, * right, * sep;
} tree_t;
typedef unsigned int fvm_morton_int_t;
typedef struct {
    fvm_morton_int_t   L;     /* Level in the tree structure */
    fvm_morton_int_t   X[3];  /* X, Y, Z coordinates in Cartesian grid */
} fvm_morton_code_t;
//

} //~namespace mflow

#endif //~MFL_NUMBER_TYPE_H
