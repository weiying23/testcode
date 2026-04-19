//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   grid_base.cpp
/// \brief  Abstract base grid object
/// \author 
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

/// direct head file
#include "grid_base.h"

using namespace std;

namespace mflow
{

Grid::Grid() : nTNode(0)
{
    x = NULL;
    y = NULL;
    z = NULL;
    gPara  = NULL;
    gField = NULL;
}


Grid::Grid(IntType in) : nTNode(0), zn(in) 
{
    x = NULL;
    y = NULL;
    z = NULL;
    gPara  = NULL;
    gField = NULL;
}


Grid::~Grid()
{
    mfmem::sdel_array_1D(x);
    mfmem::sdel_array_1D(y);
    mfmem::sdel_array_1D(z);
}




} // ~namespace mflow
