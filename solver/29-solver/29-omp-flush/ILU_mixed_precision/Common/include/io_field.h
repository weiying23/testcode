//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   io_field.h
/// \brief  functions for field output
/// \author tangj
/// \date   2020-02-24
/// \copyright  C.All rights reserved. 2020-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_IO_FIELD_H
#define MFL_IO_FIELD_H

// C++ build-in head files
#include <string>

// other user defined head files
#include "number_type.h"
#include "algm.h"
#include "constant.h"


namespace mflow
{

namespace FieldIO
{

// flow file name used to restart simulation
const std::string RESTART_FILE("RESTART");

// folder the RESTART* files saved in
const std::string RESTART_FOLDER("Restart");

// the path of restart file relative to the current working directory
const std::string RESTART_PATH = RESTART_FOLDER + "/" + RESTART_FILE;

// Return the path of restart file with file id relative to the current
// working directory
inline std::string restart_file_with_id(const IntType file_id)
{
    return RESTART_PATH + int2str(file_id);
}

} // ~namespace FieldIO


} // ~namespace mflow

#endif  //~MFL_IO_FIELD_H
