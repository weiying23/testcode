//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   grid_patch_type.h
/// \brief  Create patch name according patch type and/or patch id
/// \author tangj
/// \date   2020-3-16 
/// \copyright  C.All rights reserved. 2020-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_GRID_PATCH_TYPE_TO_NAME_H
#define MFL_GRID_PATCH_TYPE_TO_NAME_H

#include "number_type.h"
#include "constant.h"

namespace mflow
{

// Boundary condition type
const IntType WALL                    = 3;
const IntType SYMM                    = 4;
const IntType FAR_FIELD               = 6;
const IntType INTERFACE               = 10;


// Return name according to boundary condition type.
// Note: user is required to delete the returned name with mfmem::sdel_array_1D().
ShortString *fromTypeToName(const IntType type);


} // ~namespace mflow

#endif //~MFL_GRID_PATCH_TYPE_TO_NAME_H
