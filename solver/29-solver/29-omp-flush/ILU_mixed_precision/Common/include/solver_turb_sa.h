//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   solver_turb_sa.h
/// \brief  the Spalart-Allmaras turbulence model solver
/// \author 
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_SOLVER_TURB_SA_H
#define MFL_SOLVER_TURB_SA_H

#include <stdlib.h>
#include "number_type.h"
#include "grid_polyhedra.h"
#include "data_pool.h"
#include "solver_base.h"
#include "turbulence.h"
#include "zone.h"


namespace mflow
{

#define CB1         0.1355
#define SIGMA_SA    0.6666666666666667  //(2./3.) 
#define CB2         0.622
#define KAI_SA      0.41
#define KAIP2       0.1681  //(KAI_SA*KAI_SA)
#define CW1         3.2390678167757287  //(CB1/KAIP2+(1.0+CB2)/SIGMA_SA)
#define CW2         0.3   
#define CW3         2.0   
#define CW3P6       64.0  //(CW3*CW3*CW3*CW3*CW3*CW3)
#define CV1         7.1   
#define CV1P3       357.911  //(CV1*CV1*CV1)    
#define CT3         1.2           
#define CT4         0.5
#define SADES       0.65

#define MAX_MUET_SA 1.0e5
#define MIN_MUET_SA 1.0e-5
#define MIN_SA_NU   1.0e-12
#define SQRT_SIX(x) (exp(0.166666666666666 * log((x))))
#define P6(x)       ((x) * (x) * (x) * (x) * (x) * (x))


class SASolver : public Solver 
{
  
private:
    IntType     nGrids;         // no. of grids
    PolyGrid    **grids;        // the grids for the solver
    BCond       *bc;            // physical boundary conditions only.
                                // grid related bc is stored in Grid object    
    DataStore   **fields;       // the fields associated with each grid
    DataSafe    *cPara;         // the control parameters for the solver
    Zone        *zone;          // the zone pointer

public:
  
    SASolver(IntType ng, PolyGrid **gridsin, DataStore **fieldsin, 
             DataSafe *cParain, BCond *bcin, Zone *zonein);

    ~SASolver(){};
  
    PolyGrid   *GetGrid(IntType n) const;

    void UpdateData(void *data, IntType type, IntType size, const ShortString name);
    void GetData(void *data, IntType type, IntType size, const ShortString name) const;
    void GetData(void *data, IntType type, IntType size, const ShortString name, IntType messageOn) const;

    void Init();
    void Solve();
    void Post();
    void UpdateInterfaceData();
    void UpdataUnstVolData();
    void AllocateFlowfieldMemory(PolyGrid *grid); // allocate memory  for flow field
    void ReadRestartFromFile(PolyGrid *grid);
    void CommInterfaceData(const char *name);
    void InitGridVar(PolyGrid *grid);
    void DumpRestart(PolyGrid *grid, IntType iter, IntType zn, RealFlow t_now);
};

// inline functions

inline PolyGrid * SASolver::GetGrid(IntType n) const 
{
    return grids[n];
}

inline void SASolver::UpdateData(void *data, IntType type, IntType size, const ShortString name)
{
    cPara->UpdateDataSafe(data,type,size,name);
}

inline void SASolver::GetData(void *data, IntType type, IntType size, const ShortString name) const
{
    cPara->GetDataByName(data,type,size,name);
}

inline void SASolver::GetData(void *data, IntType type, IntType size, const ShortString name, IntType messageOn) const
{
    cPara->GetDataByName(data,type,size,name,messageOn);
}


// Auxiliary functions

void GhostVariablesScalar_SA(PolyGrid *grid);
void AddSourceSA(PolyGrid *grid);
void AddSourceUnstSA(PolyGrid *grid);
void ComputeTurbViscosity_SA(PolyGrid *grid);
void limitSA_nu(PolyGrid *grid);
void ComputeTurbGeneration_SA(PolyGrid *grid);
void ComputeTurbInf_SA(PolyGrid *grid, const char *name);

} //~namespace mflow

#endif //~MFL_SOLVER_TURB_SA_H
