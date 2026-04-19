//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   simulation.h
/// \brief  A class for simulation
/// \author 
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_SIMULATION_H
#define MFL_SIMULATION_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cctype>
using namespace std;

#include "constant.h"
#include "zone.h"
#include "parameter_reader.h"
#include "grid_polyhedra.h"


namespace mflow
{

class Simulation 
{

private:
    IntType     steady;     // steady or unsteady simulation
    IntType     dynamic;    // dynamic derivatives computing
    IntType     nZones;     // no. of zones
    Zone      **zones;      // the zones
    DataSafe    sPara;      // store the actual data, safe
    DataStore   sData;      // store data pointers, not safe, but efficient

public:
    /// \brief Constructor
    Simulation();

    /// \brief Destroy
    virtual ~Simulation();

    /// \brief Initialize object as the constructor do
    virtual void Construct(const ParameterReader *params);
    
    /// \brief Set steady flag of the simulation
    void SetSteadyFlag(const IntType steady_flag);

    /// \brief Return 1 if the simulation is steady, otherwise return 0
    IntType IsSteady() const;

    /// \brief Set dynamic derivatives flag of the simulation
    void SetDynamicFlag(const IntType dynamic_flag);

    /// \brief Add a Zone into the simulation
    void AddZone( Zone *zone);

    /// \brief Return the z-th Zone
    Zone *GetZone(IntType z) const;

    /// \brief Add or update parameters from source par
    void CopyParameters(const DataSafe* par);

    /// \brief Return the number of Zones
    IntType GetNoOfZones() const
    {
        return nZones;
    }

    /// \brief Return all the Zones
    Zone **GetZones() const
    {
        return zones;
    }

    /// \brief Update a simulation parameter
    void UpdateData(void *data, IntType type, IntType size, const ShortString name);

    /// \brief Get a simulation parameter
    void GetData(void *data, IntType type, IntType size, const ShortString name) const;
    void GetData(void *data, IntType type, IntType size, const ShortString name, IntType messageOn) const;

    /// \brief start application
    void Start(int argc, char *argv[]);

    /// \brief Run simulation
    void RunSimu();

    /// \brief Post simulation
    void PostSimu();

    /// \brief Initialize
    void Init();

protected:

    /// \brief Clear grid-related data in all zones
    void ClearGridRelatedDataForAllZones();

    /// \brief Update the Zone parameters based on the input file
    /// \note Fix, deduce and check parameters
    void UpdateParameter();

    /// \brief Create Solvers for Zones
    virtual void CreateSolversForZone(Zone *zone);

private:

    /// 
    void ReLoadGridDataForZones(vector<string> &grid_dir_of_zone);

    /// Load grid information for all zones
    void LoadGrid(vector<string> &grid_dir_of_zone);

    /// Generate the communication graph about interfaces
    void CommGraph();

    /// Generate the communication graph about inter-nodes
    void CommGraph_node();

    /// Get initial grid directory for each zone
    void GetInitialGridDir(vector<string> &grid_dir_of_zone);

    /// Update time accuracy for unsteady iteration
    void UpdateTimeAccuracy(const IntType time_step);

    /// Do inner iteration
    /// Return true if residual satisfies convergence criteria or user wants to stop simulation
    bool DoInnerIteration(const IntType inner_step, RealFlow &res_rho_max);

    /// determine iteration steps for initial time step
    IntType AssignInitalIterationSteps();

    /// load grid for all zones
    void LoadGridDataForZones();

};

// inline functions

/// Set steady flag of the simulation
inline void Simulation::SetSteadyFlag(const IntType steady_flag) 
{
    steady = steady_flag;
}

inline IntType Simulation::IsSteady() const
{
    return steady;
}


/// Set dynamic derivatives flag of the simulation
inline void Simulation::SetDynamicFlag(const IntType dynamic_flag) 
{ 
    dynamic = dynamic_flag;
}

inline Zone * Simulation::GetZone(IntType z) const 
{
    return zones[z];
}

/// Pass in parameter pool
inline void Simulation::CopyParameters(const DataSafe* par)
{
    sPara.CopyDataFrom(par);
}

inline void Simulation::UpdateData(void *data, IntType type, IntType size, const ShortString name) 
{
    sPara.UpdateDataSafe(data,type,size,name);
}

inline void Simulation::GetData(void *data, IntType type, IntType size, const ShortString name) const
{
    sPara.GetDataByName(data,type,size,name);
}

inline void Simulation::GetData(void *data, IntType type, IntType size, const ShortString name, IntType messageOn) const
{
    sPara.GetDataByName(data,type,size,name,messageOn);
}

} //~namespace mflow

#endif //~MFL_SIMULATION_H
