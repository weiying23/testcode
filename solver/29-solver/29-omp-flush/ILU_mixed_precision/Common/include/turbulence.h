//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   turbulence.h
/// \brief  the common program for turbulence model solver
/// \author zhangyb
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_TURBULENCE_H
#define MFL_TURBULENCE_H

#include "number_type.h"
#include "grid_polyhedra.h"
#include "data_pool.h"
#include "solver_base.h"
#include "solver_turb_sa.h"


namespace mflow
{

void AddSourceScalar(PolyGrid *grid, const char *name);
void AddSourceScalarUnst(PolyGrid *grid, const char *name);
void ComputeTurbInf(PolyGrid *grid, const char *name);
void CommUnstVolData(PolyGrid **grids, const char *name, const char *name_cur, const char *name_old);
void ComTubSourceUnst(PolyGrid *grid, const char *name, const char *name_cur, const char *name_old);
void FreeLHSMatScalar(PolyGrid *grid);
void GhostVariablesScalar(PolyGrid *grid, const char *name);
void InitLHSMatScalar(PolyGrid *grid);
void InviscidFluxScalar(PolyGrid *grid, const char *name);
void CalcuQlQr_turb(PolyGrid *grid, RealFlow *q, RealFlow *ql, RealFlow *qr, IntType ns, IntType ne, const char *name);
void ModQlQrBou_turb(PolyGrid *grid, RealFlow *q, RealFlow *ql, RealFlow *qr, IntType ns, IntType ne, IntType n);
void PutScalarDqToLhs(PolyGrid *grid, RealFlow *dqdl, RealFlow *dqdr, IntType ns, IntType ne);
void ScalarFlux(RealFlow *ql[], RealFlow *qr[], RealFlow *flux, RealGeom *xfn, 
                RealGeom *yfn, RealGeom *zfn, RealGeom *area, RealGeom *vgn, 
                RealFlow *dqdl, RealFlow *dqdr,IntType len, IntType ns, IntType ne, IntType steady);
void ScalarRelaxation(PolyGrid *grid, const char *name, RealFlow *rhs, IntType steps);
void SetGhostvis_t(PolyGrid *grid, const char *name);
void Setturb00(PolyGrid *grid, const char *name);
void SolveScalarLUSGS(PolyGrid *grid, RealFlow **lhsmat, RealFlow *dq, IntType *nCPC, IntType **c2c, IntType nTCell, const char *name);
void AdditionTermforScalarLUSGS(PolyGrid *grid, RealFlow *AddTerm, RealFlow **lhsmat, RealFlow *dq, 
                                IntType *nCPC, IntType **c2c, IntType nTCell, const char *name);
void SolveScalarLUSGS(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, 
                      RealFlow *dq, IntType *nCPC, IntType **c2c, IntType nTCell, const char *name, 
                      IntType Nsweep, RealFlow epsilon);
void SolveScalarGMRES(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, 
                      RealFlow *dq, IntType *nCPC, IntType **c2c, const char *name, IntType level);
void PreconditScalarLUSGS(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, RealFlow *dq, IntType *nCPC, IntType **c2c);
void ComputeScalarADU(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, RealFlow *v, IntType *nCPC, IntType **c2c);
void SolveScalarOnGrid(PolyGrid *grid, const char *name);
void SolveScalarDPLUR(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res,
                      RealFlow *dq, IntType *nCPC, IntType **c2c, const char *name, IntType level);
void TimeIntegrationScalar(PolyGrid *grid, const char *name);
void UpdateSolutionScalar_TAO(PolyGrid *grid, RealFlow *dq, const char *name);
void UpdateResidualScalar(PolyGrid *grid, const char *name);
void ViscousFluxScalar(PolyGrid *grid, const char *name);
void ViscousFluxScalar3D_New3(PolyGrid *grid, const char *name);
void ViscousDqScalar(PolyGrid *grid, const char *name, RealFlow *dqdl,   RealFlow *dqdr, IntType ns, IntType ne);
void ViscousMatsScalar(PolyGrid *grid, const char *name);
void GetTurbGrad(PolyGrid *grid, const char *name, RealFlow *qgrad[3]);
void DumpTurbNormResi(PolyGrid *grid, IntType iter, IntType zn, RealFlow t_now, const char *name, IntType mark);

} //~namespace mflow

#endif //~MFL_TURBULENCE_H
