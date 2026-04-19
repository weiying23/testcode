//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   data_pool.h
/// \brief  Data pool to save parameters or flow field
/// \author 
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_DATA_POOL_H
#define MFL_DATA_POOL_H

#include "constant.h"
#include "number_type.h"

namespace mflow
{
// define some constants for data type
const IntType INT    = 1;
const IntType FLOAT  = 2;
const IntType DOUBLE = 3;
const IntType STRING = 4;
const IntType CHAR   = 5;
const IntType LONG   = 6;
#ifdef SINGLE_PRECISION
const IntType REAL_FLOW = 2;
const IntType REAL_GEOM = 2;
#else
const IntType REAL_FLOW = 3;
const IntType REAL_GEOM = 3;
#endif

// forward declaration
class DataNode;


/// \brief A data-pool to save pointer of one-dimensional arrays
class DataStore
{

private:
    IntType   nData;
    DataNode *top;

public:
    /// \brief Save an array in data pool, DO NOT deallocate the array use 
    /// 'delete []' once you give it to our data pool, instead, you can 
    /// use DeleteDataByName() to deallocate it. 
    void UpdateDataStore(void *data, IntType type, IntType size, const ShortString name);

    /// \brief Return the array by name. A correct array will return if you have saved
    /// it in our data pool, otherwise, return NULL.
    void *GetDataPtrByName(IntType type, IntType size, const ShortString name) const;

    /// \brief Deallocate an array by its name.
    void DeleteDataByName(const ShortString name);

    /// \brief Deallocate all arrays in the data pool.
    void DeleteAllData();

    DataStore();
   ~DataStore();
};


/// \brief A data-pool to save parameters
class DataSafe 
{

private:
    IntType   nData;
    DataNode *top;

public:
    /// \brief Save parameter in data pool. 
    void UpdateDataSafe(void * data, IntType type, IntType size, const ShortString name);

    /// \brief Return the value of a parameter by name.
    /// \note The value of 'data' will remain unchanged if not exist.
    void GetDataByName(void * data, IntType type, IntType size, const ShortString name) const;

    /// \brief Return the value of a parameter by name.
    /// \note If the inquired parameter do not exist, the value of 'data' will remain unchanged, and
    /// a warning information will show if 'messageOn' do not equal 0.
    void GetDataByName(void * data, IntType type, IntType size, const ShortString name, IntType messageOn) const;

    /// \brief Erase a parameter from the data pool by name.    
    void DeleteDataByName(const ShortString name);

    /// \brief Erase all the parameters in the data pool.
    void DeleteAllData();

    /// \brief Print brief information of all the parameters saved in the data pool.
    void ListAllData() const;

    /// \brief Add or update all the data of DataSafe object src
    void CopyDataFrom(const DataSafe *src);

    DataSafe();
   ~DataSafe();
};

} // ~namespace mflow

#endif  //~MFL_DATA_POOL_H
