//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   io_base_format.h
/// \brief  macro definition for text format 
/// \author tangj
/// \date   2020-02-22
/// \copyright  C.All rights reserved. 2020-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_IO_BASE_FORMAT_H
#define MFL_IO_BASE_FORMAT_H

#include <iostream>
#include <iomanip>

namespace mflow
{
// define separation of a blank between two words
#define IOS_SEP  " "

// define scientific format of real number with precision 'p'
#define IOS_EP(p) std::resetiosflags(std::ios::fixed) << std::setiosflags(std::ios::scientific) << std::setprecision(p) << std::setw(p+8)

// define fixed format of real number with precision 'p'
#define IOS_FP(p) std::resetiosflags(std::ios::scientific) << std::setiosflags(std::ios::fixed) << std::setprecision(p) << std::setw(p+8)

// define scientific format of real number with width 'w' and precision 'p'
#define IOS_EWP(w,p) std::resetiosflags(std::ios::fixed) << std::setiosflags(std::ios::scientific) << std::setprecision(p) << std::setw(w)

// define fixed format of real number with  width 'w' and precision 'p'
#define IOS_FWP(w,p) std::resetiosflags(std::ios::scientific) << std::setiosflags(std::ios::fixed) << std::setprecision(p) << std::setw(w)

// define auto format of real number with precision 8
#define IOS_AUTO std::resetiosflags(std::ios::scientific) << std::resetiosflags(std::ios::fixed) << std::setprecision(15)


// define separation-line of 80 '=' characters
#define SEP_LINE "================================================================================"

} // ~namespace mflow

#endif  //~MFL_IO_BASE_FORMAT_H
