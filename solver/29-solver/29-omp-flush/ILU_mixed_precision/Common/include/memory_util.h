//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   memory_util.h
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
/// \code {.cpp}
///    // in main.cpp
///    
///    // head file include
///    ...
///    #include "memory_util.h"
///    ...
///    // declare the container, just copy the three lines code   
///    // 'DynMemRegContainor' is the container type defined in this library.
///    #ifdef DynMemReg    // macro definition provided in this library     
///    DynMemRegContainor dmreg; 
///    #endif
///    ...
///    
///    int main()
///    {
///        // do what you need.
///        // Use the functions snew_*() and sdel_*() provided
///        // in this library to replace the 'new' and 'delete'
///        // operator for dynamic memory management
///        int *p = NULL;   // Initialising to NULL is required before calling
///                         // snew_*() function
///        // allocate array and n is the size of array
///        // 'dmrfl' is the macro definition provided by this library, just copy it.
///        mfmem::snew_array_1D(p, n, dmrfl);  
///       
///        // use the array                  
///        for(int i=0; i<n; ++i) p[i] = i;  
///        //...
///    
///        // free the array
///        mfmem::sdel_array_1D(p);                 
///    
///        // print out the information about dynamic memory leak.
///        // just copy the three lines code
///    #ifdef DynMemReg    // defined in this library     
///        mfmem::print_dynmem_info();
///    #endif            
///    
///        return 0;
///    }
/// \endcode
/// 
/// \par Usage
/// <pre>        
///   <1> For dynamic object: snew_object() and sdel_object()
///       e.g.
///         A *p = NULL;  // A is a pre-defined class
///         mfmem::snew_object(p, dmrfl);  // allocate memory
///         mfmem::sdel_object(p);         // free memory
///       NOTE: dmrfl is the macro definition provided by this library, 
///       which is used to pass in the file and line information where 
///       the function 'snew_*' is called hereinafter.
///   <2> For dynamic 1D array: snew_array_1D() and sdel_array_1D()
///       e.g.
///         int *p = NULL;  
///         mfmem::snew_array_1D(p, n, dmrfl);  // n is the size of array
///         mfmem::sdel_array_1D(p);                
///   <3> For dynamic 2D array: snew_array_2D() and sdel_array_2D().  
///       Make sure the contiguous mode for the same array are same, viz, 
///       both the parameter 'contiguous' in snew_array_2D() and 
///       sdel_array_2D() are same.
///       e.g.
///         int **p = NULL;   // the size of array is n*m
///         mfmem::snew_array_2D(p, n, m, dmrfl, true); 
///         mfmem::sdel_array_2D(p, n, true); 
///   <4> 2D array or nD array can be created using snew_array_1D() and 
///       deleted using coresponding sdel_array_1D() in a nested procedure.
///       e.g.
///         int **p = NULL;  
///         mfmem::snew_array_1D(p, n,dmrfl);  // n*m is the size of array
///         mfmem::snew_array_1D(p[0], n*m, dmrfl);  //
///         for(i=1; i<n; ++i) p[i] = &(p[i-1][m]);   // construct manually
///         mfmem::sdel_array_1D(p[0]); 
///         mfmem::sdel_array_1D(p); 
///   <5> Assign an array with another array: sSet()
///       e.g.
///         int *p1 = NULL; 
///         mfmem::snew_array_1D(p1, n, dmrfl);
///         int *p2 = NULL; 
///         mfmem::sSet(p1, p2);
///       Here p1 will be deleted firstly and then set to p2(NULL);
///       This function will be very useful if p1 is a member variable
///       of some user defined class, which will be deleted in the 
///       destructor function of that class. 
///   <6> print out statistics information: print_dynmem_info()
///       This subroutine at best is located before 'return' in the 
///       function 'main()'
/// </pre>
///
///
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 2020-07-04  tangj      Normalize notation according coding guideline of FlowStar
/// </pre>

#ifndef MFL_MEMORY_UTIL_H
#define MFL_MEMORY_UTIL_H

#include <string>

#if defined(DEBUG) || defined(_DEBUG)
#define DynMemReg
#endif

#ifdef DynMemReg
#include <iostream>
#include <map>
typedef std::map<const void *, std::pair<std::string, int> > DynMemRegContainor;
extern DynMemRegContainor dmreg;
#endif

/// \namespace mfmem namespace for MFlow memory management
namespace mfmem
{
//#define LARGE_GRID
#ifdef LARGE_GRID
    typedef long IntType;
#define MPIIntType MPI_LONG
#define INTBIG  9223372036854775807
#else
    typedef int IntType;
#define MPIIntType MPI_INT
#define INTBIG 2147483647
#endif

#ifdef DynMemReg
#define dmrfl __FILE__,__LINE__
#else
#define dmrfl "0",0
#endif

/// \brief Template function to safe delete dynamically allocated
/// object in order to replace the operator "delete ptr".
/// \param[in,out] ptr the pointer of object to be deallocated.
/// \attention The "ptr" will be set to NULL after calling this function.
template<typename T> void sdel_object(T* &ptr)
{
    if(ptr != NULL) 
    {
        delete ptr;
    }
    else
    {
        return;
    }

#ifdef DynMemReg
    if(dmreg.find(static_cast<const void *>(ptr)) != dmreg.end()) 
    {
        dmreg.erase(static_cast<const void *>(ptr));
    }
    else
    {
        std::cout << "the varibale allocated without snew()" 
                  << std::endl;
    }
#endif

    ptr = NULL;
}


/// \brief Template function to safe new dynamic object in order
/// to replace the operator "new T".
/// \param[in,out] ptr the pointer after memory allocated.
/// \param[in] file the file name in which this function has been called.
/// \param[in] line the line at which this function has been called.
/// \note the parameters 'file' and 'line' can be replaced by macro definition 'dmrfl'.
/// \attention The "ptr" must equal to NULL, otherwise "ptr" will be 
/// deleted before allocated and that may be error if the
/// "ptr" is a wild pointer.
template<typename T> void snew_object(T* &ptr, const std::string &file, const int line)
{
    if(ptr != NULL) 
    {
        sdel_object<T>(ptr);
    }

    ptr = new T;

#ifdef DynMemReg
    dmreg.insert (std::make_pair(static_cast<const void *>(ptr), std::make_pair(file, line)));
#endif
}


/// \brief Template function to safe delete dynamically allocated 
/// one-dimensional array in order to replace the operator
/// "delete []ptr". 
/// \param[in,out] ptr the pointer of object to be deallocated.
/// \attention The "ptr" will be set to NULL after calling this function.
template<typename T> void sdel_array_1D(T* &ptr)
{
    if(ptr != NULL) 
    {
        delete[] ptr;
    }
    else
    {
        return;
    }

#ifdef DynMemReg
    if(dmreg.find(static_cast<const void *>(ptr)) != dmreg.end()) 
    {
        dmreg.erase(static_cast<const void *>(ptr));
    }
    else
    {
        std::cout << "the varibale allocated without snew()" << std::endl;
    }
#endif

    ptr = NULL;
}


/// \brief Specific function to safe delete dynamically allocated 
/// one-dimensional array of type void in order to replace
/// the operator "delete []ptr". 
/// \param[in,out] ptr the pointer of object to be deallocated.
/// \attention The "ptr" will be set to NULL after calling this function.
void sdel_void_array_1D(void* &ptr);


/// \brief Template function to safe delete dynamically allocated 
/// two-dimensional array whose dimension is n*m in order
/// to replace the operator "delete **ptr".
/// \param[in,out] ptr the pointer of object to be deallocated.
/// \param[in] n the first dimension of the 2D array.
/// \param[in] contiguous the memory mode.
/// \note If the 2D array is allocated in contiguous mode, the
/// parameter 'n' which indicates the size of the first
/// dimension is not required, otherwise, 
/// the parameter 'n' is necessary.
/// \attention The "ptr" will be set to NULL after calling this function.
template<typename T> void sdel_array_2D(T** &ptr, const std::size_t n = 1, const bool contiguous = true)
{
    if(ptr != NULL) 
    {
        // first delete the dynamic memory of the second dimension
        if(contiguous)
        {
            //if(ptr[0] != NULL) 
            sdel_array_1D(ptr[0]);
        }
        else
        {
            for(std::size_t i=0; i<n; ++i)
            {
                //if(ptr[i] != NULL) 
                sdel_array_1D(ptr[i]);
            }
        }

        // delete the 'ptr' and erase it from the map 
        sdel_array_1D(ptr);
    }
}


/// \brief Template function to safe new one-dimensional array 
/// whose dimension is n in order to replace the operator
/// "new T[n]".
/// \param[in,out] ptr the pointer after memory allocated.
/// \param[in] n the dimension of the 1D array.
/// \param[in] file the file name in which this function has been called.
/// \param[in] line the line at which this function has been called.
/// \note the parameters 'file' and 'line' can be replaced by macro definition 'dmrfl'.
/// \attention The "ptr" must equal to NULL, otherwise "ptr" will be
/// deleted before allocated and that may be error if the  
/// "ptr" is a wild pointer.
template<typename T> void snew_array_1D(T* &ptr, const std::size_t n, const std::string &file, const int line)
{
    if(ptr != NULL) 
    {
        sdel_array_1D<T>(ptr);
    }

    ptr = new T[n];

#ifdef DynMemReg
    dmreg.insert(std::make_pair(static_cast<const void *>(ptr), std::make_pair(file, line)));
#endif
}


/// \brief Template function to safe new two-dimensional array
/// whose dimension is n*m in order to replace the operator
/// "new T[n][m]"
///
/// \param[in,out] ptr the pointer after memory allocated.
/// \param[in] n the first dimension of the 2D array.
/// \param[in] m the second dimension of the 2D array.
/// \param[in] file the file name in which this function has been called.
/// \param[in] line the line at which this function has been called.
/// \param[in] contiguous the memory mode.
/// \note the parameters 'file' and 'line' can be replaced by macro definition 'dmrfl'.
///
/// \attention The "ptr" must equal to NULL, otherwise "ptr" will be
/// deleted before allocated and that may be error if the
/// "ptr" is a wild pointer.
///
/// \note If contiguous is true(default), the 2D array will be
/// allocated in contiguous block, i.e. 
/// \code {.cpp}
///   ptr = new int*[n]; 
///   ptr[0] = new int[n*m]; 
///   for each i=1:n ptr[i] = &(ptr[i-1][m]);
/// \endcode
///
/// \note Otherwise, the second dimension of 2D array will be
/// allocated one by one, i.e.
/// \code {.cpp}
///   ptr = new int*[n];
///   for each i=1:n ptr[i] = new int[m]; 
/// \endcode
template<typename T> void snew_array_2D(T** &ptr, const std::size_t n, const std::size_t m, const std::string &file, const int line, const bool contiguous = true)
{
    if(ptr != NULL) 
    {
        sdel_array_2D(ptr, n, contiguous);
    }

    snew_array_1D(ptr, n, file, line);
    
    if(contiguous)
    {
        ptr[0] = NULL;
        snew_array_1D(ptr[0], n*m, file, line);
        for(std::size_t i=1; i<n; ++i)
        {
            ptr[i] = &(ptr[i-1][m]);
        }
    }
    else
    {
        for(std::size_t i=0; i<n; ++i)
        {
            ptr[i] = NULL;
            snew_array_1D(ptr[i], m, file, line);
        }
    }    
}


/// \brief Template function to safe new two-dimensional array
/// whose first dimension is n and second dimension is 
/// variable and  not equal to each other, in order to 
/// replace the operator operator "new T[n][size[i=0:n]]".
///
/// \param[in,out] ptr the pointer after memory allocated.
/// \param[in] n the first dimension of the 2D array.
/// \param[in] size the second dimension array of the 2D array.
/// \param[in] file the file name in which this function has been called.
/// \param[in] line the line at which this function has been called.
/// \param[in] contiguous the memory mode.
/// \note the parameters 'file' and 'line' can be replaced by macro definition 'dmrfl'.
///
/// \attention the user must guarantee the array 'size' for the
/// second dimension is usable.
///
/// \attention The "ptr" must equal to NULL, otherwise "ptr" will be
/// deleted before allocated and that may be error if the
/// "ptr" is a wild pointer.
///
/// \note If contiguous is true(default), the 2D array will be
/// allocated in contiguous block, i.e. 
/// \code {.cpp}
///   ptr = new int*[n];
///   for each i=0:n  sum += size[i];
///   ptr[0] = new int[sum]; 
///   for each i=1:n ptr[i] = &(ptr[i-1][m]);
/// \endcode
///
/// \note Otherwise, the second dimension of 2D array will be
/// allocated one by one, i.e.
/// \code {.cpp}
///   ptr = new int*[n]; 
///   for each i=1:n  ptr[i] = new int[size[i]];
/// \endcode
template<typename T> void snew_array_2D(T** &ptr, const std::size_t n, const IntType *size, const std::string &file, const int line, const bool contiguous = true)
{
    if(ptr != NULL) 
    {
        sdel_array_2D(ptr, n, contiguous);
    }

    snew_array_1D(ptr, n, file, line);
    
    if(contiguous)
    {
        std::size_t sum = 0;
        for(std::size_t i=0; i<n; ++i) sum += size[i];

        ptr[0] = NULL;
        snew_array_1D(ptr[0], sum, file, line);

        for(std::size_t i=1; i<n; ++i)
        {
            ptr[i] = &(ptr[i-1][size[i-1]]);
        }
    }
    else
    {
        for(std::size_t i=0; i<n; ++i)
        {
            ptr[i] = NULL;
            snew_array_1D(ptr[i], size[i],  file, line);
        }
    }   
}


#ifdef DynMemReg
/// \brief Print statistics information of dynamic objects or arrays
void print_dynmem_info();
#endif


/// \brief Template function to safe operator "set ptr".
/// \attention The "ptr_old" will be deleted firstly if it is not NULL,
/// so there will be an error if the "ptr" is a wild pointer. 
/// \warning This function is designed for 1D array, not for 
///          an object pointer.
template<typename T> void sSet(T* &ptr_old, T* ptr_new)
{
    if((ptr_old != NULL) && (ptr_old != ptr_new))
    {
         sdel_array_1D(ptr_old); 
    }
    ptr_old = ptr_new;
}


/// \brief Template function to safe operator "set **ptr" to
/// set 2D array in contiguous block. i.e. 
/// \code {.cpp}
///   ptr = new int*[n]; 
///   ptr[0] = new int[n*m];
///   for each i=1:n ptr[i] = &(ptr[i-1][m]);
/// \endcode
/// \attention The "ptr_old" will be deleted firstly if it is not NULL,
/// so there will be an error if the "ptr" is a wild pointer. 
/// \author lihuan
/// \warning This function is designed for 2D array, not for 
///          1D array filled with object pointers.
template<typename T> void sSet(T** &ptr_old, T** ptr_new)
{
    if((ptr_old != NULL) && (ptr_old != ptr_new)) 
    {
        sdel_array_2D(ptr_old);  // contiguous mode
    }
    ptr_old = ptr_new;
}


/// \brief Template function to safe operator to set **ptr" for
/// allocating 2D array in non-contiguous block. i.e. 
/// \code {.cpp}
///   ptr = new int*[n]; 
///   for each i=0:n ptr[i] = new int[m]
/// \endcode
/// \attention The "ptr_old" will be deleted firstly if it is not NULL,
/// so there will be an error if the "ptr" is a wild pointer.
/// \author lihuan
/// \warning This function is designed for 2D array, not for 
///          1D array filled with object pointers.
template<typename T> void sSet(T** &ptr_old, T** ptr_new, std::size_t n)
{
    if((ptr_old != NULL) && (ptr_old != ptr_new)) 
    { 
        sdel_array_2D(ptr_old, n, false);  // non-contiguous mode
    }
    ptr_old = ptr_new;
}

} // namespace mfmem
#endif //~MFL_MEMORY_UTIL_H
