//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   parallel_base_functions.cpp
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

// direct head file
#include "parallel_base_functions.h"

// user defined head file
#include "memory_util.h"
#include "io_log.h"
#include "utility_functions.h"

namespace mflow
{

#ifdef MPICH
// The variables bellow should belong to namespace Parallel.
// We put these here in order to keep compatibility.
// TODO: improve it in future.
int myZone = 1;   // start from 1
int numprocs = 1;
MPI_Comm GridComm;  //for each grid, tangj
IntType rePartition = 0;
#endif

namespace Parallel
{

#ifdef MPICH
// Initialize MPI parallel environment
void InitMpi(int argc, char *argv[])
{
    int  myid, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];         

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(processor_name, &namelen);

    myZone = myid + 1;
    
#ifdef DEBUG
    std::cout << "Process " << myid << " on " << processor_name << std::endl;
#endif

    // Initialize to global communication world, tangj add
    // For overlap case, GridComm will be reset to correct 
    // value, so the ranks used for one same grid will be
    // assigned to one same sub-communication.
    GridComm = MPI_COMM_WORLD;

    // initialize mflog::log.
    // Here we have not known the grid number, so we assume only one
    // grid exists.
    // From now on, we can use mflog::logset_one_processor_out() to 
    // output log by only one parallel rank.
    mflog::log.mpi_init(MPI_COMM_WORLD);
}


void ExitMpi()
{
    MPI_Finalize();
}

// Find the maximum data among all parallel processors in global_comm_world
void parallel_max(IntType &max_data, const MPI_Comm global_comm_world)
{
    IntType maxt = max_data;
    MPI_Allreduce(&maxt, &max_data, 1, MPIIntType, MPI_MAX, global_comm_world);
}

// Find the maximum data among all parallel processors in global_comm_world
void parallel_max(RealFlow &max_data, const MPI_Comm global_comm_world)
{
    RealFlow maxt = max_data;
    MPI_Allreduce(&maxt, &max_data, 1, MPIReal, MPI_MAX, global_comm_world);
}

// Find the maximum data among all parallel processors in global_comm_world
void parallel_max(RealFlow *data, IntType n_data, const MPI_Comm global_comm_world)
{
    RealFlow *data_init = NULL;
    mfmem::snew_array_1D(data_init, n_data, dmrfl);
    for (IntType n = 0; n < n_data; ++n) data_init[n] = data[n];
    MPI_Allreduce(data_init, data, n_data, MPIReal, MPI_MAX, global_comm_world);
    mfmem::sdel_array_1D(data_init);
}

// Find the minimum and maximum data among all parallel processors in global_comm_world
void parallel_min_max(RealFlow &min_data, RealFlow &max_data, const MPI_Comm global_comm_world)
{
    RealFlow min_max[2] = {min_data, -max_data};
    RealFlow min_max_glb[2] = {min_max[0], min_max[1]};
    MPI_Allreduce(min_max, min_max_glb, 2, MPIReal, MPI_MIN, global_comm_world);
    min_data =  min_max_glb[0];
    max_data = -min_max_glb[1];
}


// Find the minimum and maximum data among all parallel processors in global_comm_world
void parallel_min_max(IntType &min_data, IntType &max_data, const MPI_Comm global_comm_world)
{
    IntType min_max[2] = {min_data, -max_data};
    IntType min_max_glb[2] = {min_max[0], min_max[1]};
    MPI_Allreduce(min_max, min_max_glb, 2, MPIIntType, MPI_MIN, global_comm_world);
    min_data =  min_max_glb[0];
    max_data = -min_max_glb[1];
}


// Sum the data among all parallel processors in global_comm_world
void parallel_sum(IntType &data, const MPI_Comm global_comm_world)
{
    IntType data_init = data;
    MPI_Allreduce(&data_init, &data, 1, MPIIntType, MPI_SUM, global_comm_world);
}


// Sum each the item of data among all parallel processors in global_comm_world
void parallel_sum(IntType *data, IntType n_data, const MPI_Comm global_comm_world)
{
    IntType *data_init = NULL;
    mfmem::snew_array_1D(data_init, n_data, dmrfl);
    for (IntType n = 0; n < n_data; ++n) data_init[n] = data[n];
    MPI_Allreduce(data_init, data, n_data, MPIIntType, MPI_SUM, global_comm_world);
    mfmem::sdel_array_1D(data_init);
}


// Sum the data among all parallel processors in global_comm_world
void parallel_sum(RealFlow &data, const MPI_Comm global_comm_world)
{
    RealFlow data_init = data;
    MPI_Allreduce(&data_init, &data, 1, MPIReal, MPI_SUM, global_comm_world);
}


// Sum each the item of data among all parallel processors in global_comm_world
void parallel_sum(RealFlow *data, IntType n_data, const MPI_Comm global_comm_world)
{
    RealFlow *data_init = NULL;
    mfmem::snew_array_1D(data_init, n_data, dmrfl);
    for (IntType n = 0; n < n_data; ++n) data_init[n] = data[n];
    MPI_Allreduce(data_init, data, n_data, MPIReal, MPI_SUM, global_comm_world);
    mfmem::sdel_array_1D(data_init);
}


#endif //~MPICH

} // ~ namespace parallel

} // namespace mflow
