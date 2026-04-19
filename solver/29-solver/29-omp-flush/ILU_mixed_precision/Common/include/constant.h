//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   constant.h
/// \brief  define some constants used in FlowStar
/// \author 
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_CONSTANT_H
#define MFL_CONSTANT_H
#include "number_type.h"

#ifdef MIXEDPRECISION
#define MATRIXTYPE float
#define MATRIXINTTYPE FLOAT
#define MATRIXMPITYPE MPI_FLOAT 
#else
#define MATRIXTYPE double
#define MATRIXINTTYPE DOUBLE
#define MATRIXMPITYPE MPI_DOUBLE
#endif

namespace mflow
{

// Constant type
const RealGeom PI =  3.14159265358979323846;
const IntType  MAXLINE      =   1024;

// Face variable reconstruction order
const IntType FIRST_ORDER             = 1;
const IntType SECOND_ORDER            = 2;
const IntType LIMITED_VENCAT          = 4;

// Flow type
const IntType INVISCID                = 0;
const IntType LAMINAR                 = 1;
const IntType S_A_MODEL               = 2;

// Temporal discretization method
const IntType LU_SGS                  = 1;
const IntType MULTI_STAGE             = 2;
const IntType DPLUR                   = 3;
const IntType MATRIX_FORMAT           = 4;

// Attribute of cell/node for overlap grid
const IntType INACTIVE                = 0;
const IntType CHANGACTIVE             = 3;

const IntType SEG_LEN                 = 100240;

// Simple function
#ifndef MIN
#define         MIN(X,Y)        std::min((X),(Y))
#endif
#ifndef MAX
#define         MAX(X,Y)        std::max((X),(Y))
#endif

#ifndef SIN
#define         SIN(X)        (sin(X*PI/180.0))
#endif
#ifndef COS
#define         COS(X)        (cos(X*PI/180.0))
#endif

} //~namespace mflow

#endif //~MFL_CONSTANT_H
