//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   utility_functions.h
/// \brief  define some utility functions
/// \author 
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 2020-07-18  tangj      Add function CaseNameFreeStream()
/// </pre>

#ifndef MFL_UTILITY_FUNCTIONS_H
#define MFL_UTILITY_FUNCTIONS_H

#include <string>

#include "data_pool.h"
#include "simulation.h"
#include "zone.h"
#include "grid_polyhedra.h"


namespace mflow
{
/// \brief Print version information on screen
void PrintVersion();

/// \brief Zero the residuals. Also allocate memory for the residuals if they had not been allocated
void ZeroGridResiduals(PolyGrid *grid, const char *name, IntType nVar);

/// \brief Copy the value of rhs to grid's variable named name and switch the sign
void PutResInGrid(PolyGrid *grid, RealFlow *rhs, IntType n, const char *name);

/// \brief Update residuals in cell with the fluxes at cell faces
void LoadFlux(PolyGrid *grid, RealFlow *flux[], IntType nVar, IntType ns, IntType ne); 

/// \brief Free memories of the residuals for all grids. Here 'grid' must be the finest one.
void FreeGridResi(PolyGrid *grid, IntType nVar);

/// \brief Set ql and qr using the values of q
void SetQlQrUseQ(PolyGrid *grid, RealFlow *q, RealFlow *ql, RealFlow *qr, IntType ns, IntType ne);

/// \brief Copy a vector
void VectCopyFrom(RealFlow *a, RealFlow *b, IntType size, IntType sign);

/// \brief Calculate Gradients
//void CompGradientQ(PolyGrid *grid, RealFlow *q, RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, IntType name);
void CompGradientQ(PolyGrid* grid, RealFlow* q, RealFlow* dqdx, RealFlow* dqdy, RealFlow* dqdz, IntType name, RealFlow* u_n, RealFlow* v_n, RealFlow* w_n);

/// \brief Calculate the gradients of flow variable q in 3D use Node-Green-Gauss Approach
//void CompGradientQ_Gauss_Node(PolyGrid *grid, RealFlow *q, RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, IntType name);
void CompGradientQ_Gauss_Node(PolyGrid* grid, RealFlow* q, RealFlow* dqdx, RealFlow* dqdy, RealFlow* dqdz, IntType name, RealFlow* u_n, RealFlow* v_n, RealFlow* w_n);

/// \brief Compute the node variable use the distance weight
//void CompNodeVar3D_dist(PolyGrid *grid, RealFlow *q_n, RealFlow *q, IntType name);
void CompNodeVar3D_dist(PolyGrid* grid, RealFlow* q_n, RealFlow* q, IntType name, RealFlow* u_n, RealFlow* v_n, RealFlow* w_n);
void CompNodeVar3D_dist(PolyGrid* grid, RealFlow* q_n, RealFlow* q);//dingxin-add

/// \brief Calculate limiter of type Vencat for a variable
RealFlow VenFun(RealFlow d, RealFlow dq, RealFlow eps);
void MaxMinDiff(RealFlow *dmax, RealFlow *dmin, RealFlow *q, BCRecord **bcr, IntType *f2c, IntType nTCell, IntType nBFace, IntType nTFace);
void MaxMinDiff(RealFlow *dmax, RealFlow *dmin, RealFlow *q, PolyGrid *grid);
void VencatLimiter(PolyGrid *grid, RealFlow *limit, RealFlow *q, RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, IntType name);

/// \brief Check nan in string 'str', if nan program exit 
void IfNAN(char *str);

/// \brief 由美国标准大气参数表根据高度H获取温度T和压力p
void CalpandTfromAltitude(RealGeom Altitude, RealFlow &p_bar, RealFlow &T);

} //~namespace mflow

#endif //~MFL_UTILITY_FUNCTIONS_H
