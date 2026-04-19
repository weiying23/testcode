//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   grid_patch_type.cpp
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

// direct head file
#include "grid_patch_type.h"

// C++ build-in head files
#include <cstring>   // strcpy()
#include <stdio.h>   // printf()
#include <iostream>

// other user defined head files
#include "memory_util.h"


namespace mflow
{
// waring: the length of name must be less than MAX_SHORT_STRING
ShortString *fromTypeToName(const IntType type)
{
    ShortString *name = NULL;
    mfmem::snew_array_1D(name, 1, dmrfl);

    if(type == WALL) {
        strcpy(*name,"wall");
    } else if(type == SYMM) {
        strcpy(*name,"symm");
    } else if(type == FAR_FIELD) {
        strcpy(*name,"far_field");
    } else if(type == INTERFACE) {
        strcpy(*name,"interface");
    } else {
        std::cerr << "wrong bc type" << std::endl;
    }

    return name;
}

} // ~namespace mflow

