//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   io_log.cpp
/// \brief  A wrapper class for std::cout with more controlling for MPI parallel
///         and overlap grid.
/// \author tangj
/// \date   2020-02-18
/// \copyright  C.All rights reserved. 2020-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#include "io_log.h"

#include <vector>
#include <sstream>
#include <iomanip>

namespace mflog
{

// the real object of log
Logout log(std::cout);


// constructor
Logout::Logout (streamT& target) : 
    OStreamProxy   (target),
    need_out_      (true),
    root_rank_out_ (true),
    each_grid_out_ (true),
    rank_id_       (1),
    grid_id_       (1)
{

}

// Output object of type ostream to the target.
Logout& Logout::operator<< (streamT& (*in)(streamT&)) 
{
    if (need_out_)
    {
        (*OStreamProxy::get()) << in; 
    }
    return *this;
}

// Pass any ios manipulators into the target.
Logout& Logout::operator<< (os_ios_type & (*in)(os_ios_type&)) 
{
    if (need_out_)
    {
        (*OStreamProxy::get()) << in; 
    }
    return *this;
}

/**
* Pass any ios_base manipulators into the target.
*/
Logout& Logout::operator<< (std::ios_base& (*in)(std::ios_base&)) 
{
    if (need_out_)
    {
        (*OStreamProxy::get()) << in; 
    }
    return *this;
}


#ifdef MPICH
// initialize for MPI parallel 

void Logout::mpi_init(const MPI_Comm global_comm_world)
{
    int my_rank = 0;
    MPI_Comm_rank(global_comm_world, &my_rank);
    if (my_rank == 0)
    {
        root_rank_out_ = true;
    } 
    else
    {
        root_rank_out_ = false;
    }

    this->rank_id_ = my_rank + 1;
}


void Logout::mpi_init(const unsigned int grid_id, const MPI_Comm global_comm_world)
{
    //
    // set rank information
    this->mpi_init(global_comm_world);

    //
    // set grid id information

    int n_procs = 1;
    MPI_Comm_size(global_comm_world, &n_procs);

    std::vector<unsigned int> grid_id_on_procs(n_procs);
    unsigned int this_grid_id = grid_id;
    MPI_Allgather(&this_grid_id, 1, MPI_INT, &(grid_id_on_procs[0]), 1, MPI_INT, global_comm_world);
 
    // Let the first rank of each grid to output message
    int p = 0;
    for (; p < n_procs; ++p)
    {        
        if (grid_id_on_procs[p] == this_grid_id) break;
    }

    if (this->rank_id_ == p+1)
    {
        each_grid_out_ = true;
    }
    else
    {
        each_grid_out_ = false;
    }

    this->grid_id_ = this_grid_id;
}
#endif


} // ~namespace mflog

