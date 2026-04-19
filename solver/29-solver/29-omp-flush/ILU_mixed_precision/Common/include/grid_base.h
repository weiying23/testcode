//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   grid_base.h
/// \brief  Abstract base grid object
/// \author 
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_GRID_BASE_H
#define MFL_GRID_BASE_H

#include "number_type.h"
#include "data_pool.h"
#include "memory_util.h"

namespace mflow
{

//=============================================================================
//                            Base Grid class
//=============================================================================
class Grid 
{

private:
    IntType   nTNode;    // no of total nodes
    RealGeom  *x,*y,*z;  // coordinates
    DataStore *gField;   // store the field variables, anything
    DataSafe  *gPara;    // store the control parameters
    IntType   zn;        // the zone number

public:
    virtual void ComputeMetrics() = 0;

    IntType   GetZone() const;
    void      SetZone(const IntType znin);
    IntType   GetNTNode() const;
    void      SetNTNode(const IntType ntn);
    void      SetX(RealGeom *xin);
    void      SetY(RealGeom *yin);
    void      SetZ(RealGeom *zin);
    RealGeom *GetX() const;
    RealGeom *GetY() const;
    RealGeom *GetZ() const;    
    
    void CopyDataFrom(DataSafe *in);

    void CopyFieldFrom(DataStore *in);

    void UpdateDataPtr(void *data,IntType type,IntType size,const ShortString name);
    void *GetDataPtr(IntType type, IntType size, const ShortString name) const;
    void DeleteDataPtr(const ShortString name);

    void UpdateData(void *data, IntType type, IntType size, const ShortString name);
    void GetData(void *data, IntType type, IntType size, const ShortString name) const;
    void GetData(void *data, IntType type, IntType size, const ShortString name, IntType messageOn) const;

    Grid();
    explicit Grid(IntType in);
    virtual ~Grid();
};
// inline functions of class Grid
#include "grid_base.inl"

} // ~namespace mflow

#endif //~MFL_GRID_BASE_H
