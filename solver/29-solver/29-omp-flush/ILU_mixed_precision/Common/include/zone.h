//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   zone.h
/// \brief  A class of Zone
/// \author 
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_ZONE_H
#define MFL_ZONE_H

#include "constant.h"
#include "number_type.h"
#include "grid_polyhedra.h"
#include "boundary_condition.h"
#include "data_pool.h"
#include "solver_base.h"

namespace mflow
{

class Simulation;

class Zone 
{ 

private:
    IntType     zn;          // the current zone number
    IntType     nGrids;      // No. of grids in the zone
    Grid      **grids;       // the grids
    DataStore **fields;      // the fields associated with each grid
      
    BCond       bc;         // physical boundary conditions only. Grid related bc is stored in Grid object
    DataSafe    zPara;      // zonal control parameters
    DataStore   zData;      // zonal data pointers
    
    IntType     nSolvers;
    Solver    **solvers;

    IntType     nNeighbor,  *nb;  // No. of neighbors, for multi-zone problems, for interfaces
    IntType     nNeighborN, *nbN; // No. of neighbors, for multi-zone problems, for inter-nodes

    // For overlap grid problems, for Chimera-zones
    // nOverZone: number of processors that other grids locate on except current grid.
    // nCh[0:nOverZone-1]: the id of processor where other grids exist
    // OverZoneMark[0:nOverZone-1]: the OverGridMark of other grids on processors
    IntType     nOverZone,  *nCh, *OverZoneMark;

    // The mapping from the whole grid to serial partitioned grids
    // It is used to assemble partitioned flow-field to a whole for post-processing.
    IntType     NumZone;     // Number of partitioned grids
    IntType    *NumCell;     // Number of cells for each partitioned grid
    IntType    *partnerZone; // Which partitioned zone the cell of whole grid belongs to. 
    IntType    *partnerCell; // The mapping from the cell of whole grid to the cell of partitioned grid.

public:

    static Simulation *simu;   // ??? used to go back from Zone to Simulation

public:

    IntType     GetZoneNo() const;
    void        AddGridAndField(Grid *grid);
    IntType     GetNoOfGrids() const;
    Grid       *GetGrid(IntType g) const;   
    Grid      **GetGrids() const;
    DataStore **GetFields() const;
    DataSafe   *GetZonePara();
    BCond      *GetZoneBc(); 

    /// \brief Add or update parameters from the source zpar
    void CopyParameters(const DataSafe *zpar);

    /// \brief Add or update BCRecord from the source zbc
    void CopyZoneBCs(const BCond *zbc);

    void        SetnNeighbor (IntType in);
    void        SetnNeighborN(IntType in);
    void        SetZone(IntType znin);
    void        Setnb (IntType *in);
    void        SetnbN(IntType *in);

    // Clear all grids and corresponding fields in the zone
    void ClearAllGridsAndFields(void);

    // Clear neighbors information of zones 
    void ClearNeighborZonesInformation(void);

    // Clear overlap information of zones
    void ClearOverlapZonesInformation(void);

    void UpdateData(void *data, IntType type, IntType size, const ShortString name);
    void GetData(void *data, IntType type, IntType size, const ShortString name) const;
    void GetData(void *data, IntType type, IntType size, const ShortString name, IntType messageOn) const;

    void UpdateDataPtr(void *data, IntType type, IntType size, ShortString name);
    void *GetDataPtr(IntType type, IntType size, const ShortString name) const;
    void ListAllData();

    void UpdateParameter();
    void FixParameter();
    void CheckParameter();
    void UpdateVariableLimit();

    void AddBCRecord(BCRecord *bcr);

    BCRecord *GetBCRecord(IntType n);
    BCRecord *GetVCRecord(IntType n);
    IntType   GetNoBCRecord();

    // The mapping from the whole grid to serial partitioned grids
    // The mapping is used to assemble partitioned flowfield to a whole.
    // Clear the mapping information from parallel multi-parts grids to a single grid
    void ClearPartGridsToSingleGridInformation(void);

    void AddSolver(Solver *solver);
    // Clear all solvers in the zone
    void ClearAllSolvers(void); 
    
    void InitZone();
    void InitZoneTranGrid();
    void InitSolvers();
    void SolveZone();
    void PostZone();

    void SpecifyBC();
    
    void UpdateInterfaceData();
    void UpdataUnstVolData();
    void CommCellCenterData();
    void CommInterfaceData(char *name);
    void TestReconstruction();
    void SpecifyNeighbors();
    void ComputeMachineZero();
    void GetGridDir();
    IntType BreakZone();

    void output_info_c2c(IntType *CellToZone, IntType npart, IntType mg);
    void GetFineGrids(IntType *CellToZone, PolyGrid **z_Grids, IntType n_part, IntType mg);
    void GetFineGrids_ff2pf(IntType *CellToZone, PolyGrid **z_Grids, IntType n_part, IntType **ff2pf, IntType **pf2ff, IntType mg);  
    void GetFineGrids_fc2pc(IntType *CellToZone, PolyGrid **z_Grids, IntType n_part, IntType *fc2pc, IntType **pc2fc, IntType mg);  
    void GetFineGrids_pn2fn(PolyGrid **z_Grids, IntType n_part, IntType **pf2ff, IntType **pn2fn, IntType mg);  
    void GetFineGrids_coord(PolyGrid *grid, PolyGrid **z_Grids, IntType n_part, IntType **pn2fn);
    void GetFineGrids_f2c(PolyGrid *grid, PolyGrid **z_Grids, IntType n_part, IntType **pf2ff, IntType *fc2pc, IntType *CellToZone);
    void GetFineGrids_F2N(IntType nTNode, PolyGrid **z_Grids, IntType n_part, IntType **pn2fn);
    void GetFineGrids_nbBF(IntType *CellToZone, PolyGrid **z_Grids, IntType n_part, IntType **ff2pf, IntType mg);
    void GetFineGrids_nbRN(PolyGrid **z_Grids, IntType n_part, IntType **pn2fn, IntType mg);
    //Reorder cell for LUSGS
    void ReorderCellforLUSGS();  
    
    void PartitionGrids(IntType *CellToZone,IntType n_zone);
#ifdef MPICH
    void SetUpComm(IntType nbZone, PolyGrid *grid);
#endif
    
    explicit Zone(IntType znin);
    Zone();

   ~Zone();
#if (defined FS_CUDA)||(defined FS_CUDA_DEBUG)
   Solver* GetNSSolver() {
	   return solvers[0];
   }
#endif

};

// inline functions

inline IntType Zone::GetZoneNo() const 
{
    return zn;
}

inline IntType Zone::GetNoOfGrids()  const 
{
    return nGrids;
}

inline Grid * Zone::GetGrid(IntType g)  const 
{
    if(grids) return grids[g];
    else return 0;
}   

inline Grid ** Zone::GetGrids() const
{
    return grids;
}

inline DataStore **Zone::GetFields() const
{
    return fields;
}

inline DataSafe * Zone::GetZonePara()
{
    return &zPara;
}

inline BCond   *Zone::GetZoneBc()
{
    return &bc;
}

inline void Zone::CopyParameters(const DataSafe *zpar)
{
    zPara.CopyDataFrom(zpar);
}

inline void Zone::CopyZoneBCs(const BCond *zbc)
{
    bc.CopyFrom(zbc);
}

inline void Zone::SetnNeighbor (IntType in) 
{
    nNeighbor = in;
}

inline void Zone::SetnNeighborN(IntType in) 
{
    nNeighborN = in;
}

inline void Zone::SetZone(IntType znin) 
{ 
    zn = znin;
}

inline void Zone::Setnb (IntType *in) 
{
    nb = in;
}

inline void Zone::SetnbN(IntType *in) 
{
    nbN = in;
}

inline void Zone::UpdateData(void *data, IntType type, IntType size, const ShortString name)
{
    zPara.UpdateDataSafe(data,type,size,name);
}

inline void Zone::GetData(void *data, IntType type, IntType size, const ShortString name) const
{
    zPara.GetDataByName(data,type,size,name);
}

inline void Zone::GetData(void *data, IntType type, IntType size, const ShortString name, IntType messageOn) const
{
    zPara.GetDataByName(data,type,size,name,messageOn);
}

inline void Zone::UpdateDataPtr(void *data, IntType type, IntType size, ShortString name)
{
    zData.UpdateDataStore(data,type,size,name);
}

inline void *Zone::GetDataPtr(IntType type, IntType size, const ShortString name) const
{
    return zData.GetDataPtrByName(type,size,name);
}

inline void Zone::ListAllData()
{
    zPara.ListAllData();
}

inline void Zone::AddBCRecord(BCRecord *bcr) 
{
    bc.AddBCRecord(bcr);
}

inline BCRecord *Zone::GetBCRecord(IntType n) 
{
    return bc.GetBCRecord(n);
}

inline IntType Zone::GetNoBCRecord() 
{ 
    return bc.GetNoBCR(); 
}

} //~namespace mflow

#endif //~MFL_ZONE_H
