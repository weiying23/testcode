//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   parallel_base_functions.h
/// \brief  Base functions for parallel
/// \author tangj
/// \date   2020-02-21
/// \copyright  C.All rights reserved. 2020-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_PARALLEL_BASE_FUNCTIONS_H
#define MFL_PARALLEL_BASE_FUNCTIONS_H

// user defined head files
#include "number_type.h"
#include "parameter_reader.h"

#ifdef MPICH
#include <mpi.h>
#endif

namespace mflow
{

namespace Parallel
{

#ifdef MPICH
    /// \brief Initialize MPI parallel environment
    void InitMpi(int argc, char *argv[]);

    /// \brief Exit MPI parallel environment
    void ExitMpi();

    // Find the maximum data among all parallel processors in global_comm_world
    void parallel_max(IntType &max_data, const MPI_Comm global_comm_world);
    void parallel_max(RealFlow &max_data, const MPI_Comm global_comm_world);

    // Find the maximum data among all parallel processors in global_comm_world
    void parallel_max(RealFlow *data, IntType n_data, const MPI_Comm global_comm_world);

    // Find the minimum and maximum data among all parallel processors in global_comm_world
    void parallel_min_max(RealFlow &min_data, RealFlow &max_data, const MPI_Comm global_comm_world);

    // Find the minimum and maximum data among all parallel processors in global_comm_world
    void parallel_min_max(IntType &min_data, IntType &max_data, const MPI_Comm global_comm_world);

    // Sum the data among all parallel processors in global_comm_world    
    void parallel_sum(IntType &data, const MPI_Comm global_comm_world);

    // Sum each the item of data among all parallel processors in global_comm_world
    void parallel_sum(IntType *data, IntType n_data, const MPI_Comm global_comm_world);

    // Sum the data among all parallel processors in global_comm_world
    void parallel_sum(RealFlow &data, const MPI_Comm global_comm_world);

    // Sum each the item of data among all parallel processors in global_comm_world
    void parallel_sum(RealFlow *data, IntType n_data, const MPI_Comm global_comm_world);

#endif //~MPICH

} // ~namespace parallel

} // ~namespace mflow

#endif // MFL_PARALLEL_BASE_FUNCTIONS_H
