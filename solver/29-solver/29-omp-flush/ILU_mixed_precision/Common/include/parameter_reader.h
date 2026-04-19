//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   parameter_reader.h
/// \brief  A class to read mflow parameters
/// \author tangj
/// \date   2020-06-10
/// \copyright  C.All rights reserved. 2020, CAI/CARDC
/// 
/// 
/// \par Usage
/// <pre>        
///   <1> declare an object.
///      ParameterReader param_reader;
///   <2> call member function to read parameter file
///      param_reader.read_parameter();
///   <3> call member function to return parameter or parameter list 
/// </pre>
///
///
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 2020-07-06  tangj      Normalize notation according coding guideline of FlowStar
/// </pre>

#ifndef MFL_PARAMETER_READER_H
#define MFL_PARAMETER_READER_H

#include <vector>
#include <string>
#include <cassert>   /// assert()
#include <iostream>
#include <stdlib.h>  /// atoi()
#include <map>

#include "boundary_condition.h"
#include "number_type.h"
#include "data_pool.h"
#include "memory_util.h"


namespace mflow
{

/// \brief  A class to read mflow parameters
/// \attention There are two member functions to read parameter file, including 
///  read_parameter() and read_parameter_directory(). But one ParameterReader
///  object can only call one function.
/// \detail When read parameter file, ParameterReader judges the file version
/// according the postfix, such as .par for old version and .para for new version.
/// What's more, ParameterReader preferentially seeks new version file.
class ParameterReader
{
public:

    /// \brief Constructor
    ParameterReader(void);

    /// \brief Read parameter files
    /// \param[in] app_case 1->preprocessor, 2->solver, 3->postprocessor
    /// \param[in] is_overset true for overset grids, false otherwise
    void read_parameter(const IntType app_case, const bool is_overset);

    /// \brief Get the simulation parameters pointer
    const DataSafe* get_simulation_parameters() const;

    /// \brief Get the i{th} zone parameters pointer
    /// \param[in] izone the zone id (start from 0).
    const DataSafe* get_zone_parameters(const IntType izone) const;

    /// \brief Get the i{th} BCond pointer
    /// \param[in] izone the zone id (start from 0).
    const BCond* get_zone_bcond(const IntType izone) const;

    /// \brief Destructor
    ~ParameterReader();

    /// \brief Return title
    std::string GetTitle() const;

    /// \brief Return steady flag
    IntType GetSteadyFlag() const;

    /// \brief Return dynamic flag
    IntType GetDynamicFlag() const;

    /// \brief Get the number of grids
    IntType GetNumberOfGrids() const;

    /// \brief Get number of parallel cores for each grid
    const IntType * GetnCoresForGrids() const;

    /// \brief Get a const reference of post directories
    const std::vector<std::string> & GetPostDirectories() const;

    /// \brief Get a const reference of post zones
    const std::vector<IntType> & GetPostZones() const;

    /// \brief Set grid index for preprocessor
    /// \param[in] index the grid id (start from 1).
    void SetGridIndex(const IntType index);

    /// \brief Get grid index for preprocessor
    IntType GetGridIndex(void) const;

    // private functions
private:

    /// \brief Read parameter files of old version, such as input.par and chmrinput.par
    /// \param[in] app_case 1->preprocessor, 2->solver, 3->postprocessor
    void read_parameter_v1(const IntType app_case);

    /// read input.par of old format
    void read_file_input_v1(const std::string & file, const IntType zone_id);

    /// \brief Read parameter files of old version, such as input.para and chmrinput.para
    /// \param[in] app_case 1->preprocessor, 2->solver, 3->postprocessor
    void read_parameter_v2(const IntType app_case);

    /// read parameter files of new version, such as input.para
    void read_file_input_v2(const std::string & file);

    /// read forceget.par of new format
    void read_file_forceget_v2(const std::string & file);

    // member variables
private:

    /// some key words
    std::string titile_;  // software title
    IntType steady_;          // steady or not
    IntType dynamic_;         // dynamic or not

    /// number of grids for overlap 
    IntType n_grids_;
    /// number of parallel cores for each grid
    std::vector<IntType> n_cores_for_grid_;

    /// parameters for simulation
    DataSafe *simu_parameters_;

    /// parameters for common items for all zones
    DataSafe *zones_common_parameters_;

    /// parameters for all zones
    std::vector<DataSafe *> zones_parameters_;

    /// boundary conditions for all zones
    std::vector<BCond *> zones_bc_records_;

    /// directories for post-processors
    std::vector<std::string> post_directories_;

    /// zone id for post-processors
    std::vector<IntType> post_zones_;

    /// grid index for preprocessor
    IntType grid_index_;

    /// whether multi-grids for overset
    bool is_overset_;
};


/// inline functions of class BoundaryGrid

/// Return title
inline std::string ParameterReader::GetTitle() const
{
    return titile_;
}


/// Return steady flag
inline IntType ParameterReader::GetSteadyFlag() const
{
    return steady_;
}


/// Return dynamic flag
inline IntType ParameterReader::GetDynamicFlag() const
{
    return dynamic_;
}


/// Get the number of grids
inline IntType ParameterReader::GetNumberOfGrids() const
{
    return n_grids_;
}

/// Get number of parallel cores for each grid
inline const IntType * ParameterReader::GetnCoresForGrids() const 
{
    return &(n_cores_for_grid_[0]);
}


/// Get the simulation parameters pointer
inline const DataSafe* ParameterReader::get_simulation_parameters() const 
{
    return simu_parameters_;
}

/// Get the i{th} zone parameters pointer
inline const DataSafe* ParameterReader::get_zone_parameters(const IntType izone) const 
{
    assert(izone < n_grids_);
    return zones_parameters_[izone];
}

/// Get the i{th} BCond pointer
inline const BCond* ParameterReader::get_zone_bcond(const IntType izone) const 
{
    assert(izone < n_grids_);
    return zones_bc_records_[izone];
}

/// Get a const reference of post directories
inline const std::vector<std::string> & ParameterReader::GetPostDirectories() const
{
    return post_directories_;
}

/// Get a const reference of post zones
inline const std::vector<IntType> & ParameterReader::GetPostZones() const
{
    return post_zones_;
}


/// Set grid index for preprocessor
inline void ParameterReader::SetGridIndex(const IntType index)
{
    grid_index_ = index;
}

/// Get grid index for preprocessor
inline IntType ParameterReader::GetGridIndex(void) const
{
    return grid_index_;
}


///-------------------------------------------------------------------
/// Modify the value of parameter in the input file
///-------------------------------------------------------------------

/// \brief Modify the value of parameter in the input file
/// \param[in] file path of file
/// \param[in] params parameters to be modified including key-value pair
/// \warning If used for input file of old version (input.par), only parameter restart can be modified.
/// \attention If the parameter is an array for new version file, the multi-values are separated by ',', 
///  such as "-3.5, 25.0";
/// \par Usage
/// <pre>
/// ...
/// std::map<std::string, std::string> params;
/// params.insert(std::make_pair("steady", "1"));
/// params.insert(std::make_pair("TotalGrid", "10, 6"));
/// modify_parameter_file("input.para", params);
/// </pre>
/// 
void modify_parameter_file(const std::string & file, std::map<std::string, std::string> & params);


///-------------------------------------------------------------------
/// Auxiliary functions declaration for interpreting parameter file
///-------------------------------------------------------------------

template<class T> IntType ProcessGeneric(T *item, ShortString word, char *line, IntType *p);


/// \brief Count the number of words in a line, the words is separated by blank character,
///   including space(' '), horizontal tab(\t), newline(\n), vertical tab(\v), feed(\f)
///   and carriage return(\r).
IntType NumberOfWords(char *line);

/// \brief Get the next word from line, starting at position p.
///   The words is separated by blank character,including space(' '), horizontal tab(\t),
///   newline(\n), vertical tab(\v), feed(\f) and carriage return(\r).
IntType GetNextWord(ShortString word, char *line, IntType *p);

/// implementation of ProcessGeneric()
template<class T>
inline IntType ProcessGeneric(T *item, ShortString word, char *line, IntType *p)
{
    IntType type, i, num=1, c;
    IntType error = 0;
    String snum;
    ShortString name;
    float *fn;
    double *dn;
    IntType *in;
    String *sn;

    if(word[0] != '$') return 1;
    if(word[1] == 'I' || word[1] == 'i') type = INT;
    else if(word[1] == 'F' || word[1] == 'f') type = FLOAT;
    else if(word[1] == 'D' || word[1] == 'd') type = DOUBLE;
#ifdef SINGLE_PRECISION
    else if(word[1] == 'R' || word[1] == 'r') type = FLOAT;
#else
    else if(word[1] == 'R' || word[1] == 'r') type = DOUBLE;
#endif
    else if(word[1] == 'S' || word[1] == 's') type = STRING;
    else
    {
        std::cerr << "Wrong variable type in ProcessGeneric" << std::endl;
        return 2;
    }

    // check how many to read
    c=2;
    if(word[c] =='[')
    {
        c++;
        while(word[c] != ']')
        {
            snum[c-3] = word[c];
            c++;
        }
        snum[c-3] = '\0';
        num = atoi(snum);
        c++;
    }
    assert(word[c] == '-');
    c++;
    strcpy(name, &word[c]);

    if(type ==  FLOAT)
    {
        fn = NULL;
        mfmem::snew_array_1D(fn, num,dmrfl);
    }
    else if(type ==  DOUBLE)
    {
        dn = NULL;
        mfmem::snew_array_1D(dn, num,dmrfl);
    }
    else if(type ==  INT)
    {
        in = NULL;
        mfmem::snew_array_1D(in, num,dmrfl);
    }
    else if(type ==  STRING)
    {
        sn = NULL;
        mfmem::snew_array_1D(sn, num,dmrfl);
    }

    for(i=0; i<num; i++)
    {
        GetNextWord(snum, line, p);
        if(type ==  FLOAT)
        {
            fn[i] = (float) atof(snum);
        }
        else if(type ==  DOUBLE)
        {
            dn[i] = atof(snum);
        }
        else if(type ==  INT)
        {
            in[i] = atoi(snum);
        }
        else if(type ==  STRING)
        {
            strcpy(sn[i], snum);
        }
    }

    if(type ==  FLOAT)
    {
        item->UpdateData(fn, type, num, name);   
        mfmem::sdel_array_1D(fn); 
    }
    else if(type ==  DOUBLE)
    {
        item->UpdateData(dn, type, num, name);     
        mfmem::sdel_array_1D(dn); 
    }
    else if(type ==  INT)
    {
        item->UpdateData(in, type, num, name);     
        mfmem::sdel_array_1D(in);
    }
    else if(type ==  STRING)
    {
        item->UpdateData(sn, type, num, name);    
        mfmem::sdel_array_1D(sn); 
    }

    return error;
}

} /// ~namespace mflow

#endif ///~MFL_PARAMETER_READER_H
