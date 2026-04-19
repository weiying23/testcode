//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   memory_util.cpp
/// \brief  A library for dynamic memory management in C++.
/// \author tangj
/// \date   2018-4-5
/// \copyright  C.All rights reserved. 2018-2020, CAI/CARDC
/// 
/// \note
///    This library provides several wrap functions for C++ operator
///    'delete' and 'new', and supports statistics for dynamic memory use.
///    Before calling the subroutines provided here, the user must declare
///    the container before and outside the function 'main' in the main.cpp,
///    e.g.
///
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 2020-07-04  tangj      Normalize notation according coding guideline of FlowStar
/// </pre>

#include "memory_util.h"

namespace mfmem
{

// Specific function to safe delete dynamically allocated 
// one-dimensional array of type void in order to replace
// the operator "delete []ptr". 
// The "ptr" will be set to NULL after calling this function.
void sdel_void_array_1D(void* &ptr)
{
    if(ptr != NULL) 
    {
        // The data type is not important when 1D array is deallocated
        // using delete[].
        // In order to eliminate the warning that delete[] void* is undefined
        // with gcc, here we converted it to char *. tangj
        delete[] static_cast<char *> (ptr);
    }
    else
    {
        return;
    }

#ifdef DynMemReg
    if(dmreg.find(ptr) != dmreg.end()) 
    {
        dmreg.erase(ptr);
    }
    else
    {
        std::cout << "the varibale allocated without snew()" << std::endl;
    }
#endif

    ptr = NULL;
}

#ifdef DynMemReg
// Print statistics information of dynamic objects or arrays
void print_dynmem_info()
{
    std::cout << "--------Statistics information of memory leak--------" << std::endl;

    std::map<const void *, std::pair<std::string, int> > ::iterator
        iter = dmreg.begin();
    for(std::size_t i=0 ; iter != dmreg.end(); ++iter, ++i)
    {
        std::cout << i 
            <<" : file " << iter->second.first
            <<" line " << iter->second.second << std::endl;
    }
}
#endif

} // namespace mfmesh

