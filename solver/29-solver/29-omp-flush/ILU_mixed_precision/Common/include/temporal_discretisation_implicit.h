//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   temporal_discretisation_implicit.h
/// \brief  implicit temporal discretisation
/// \author zhangyb
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_TEMPORAL_DISCRTEISATION_IMPLICIT_H
#define MFL_TEMPORAL_DISCRTEISATION_IMPLICIT_H

#include "number_type.h"
#include "grid_polyhedra.h"

namespace mflow
{

void ForwardStep(PolyGrid *grid, RealFlow *rhs, IntType level, IntType steps);
void ForwardLUSGS(PolyGrid *grid, IntType level);
void ForwardDPLUR(PolyGrid *grid, IntType level);
void AdditionTermforLUSGS(PolyGrid *grid, RealFlow *AddTerm[5], RealFlow *Diag, RealFlow *DQ[5], IntType level);
void CalJacobian_ConvectiveFlux(RealFlow Matrix[5][5], RealFlow nx, RealFlow ny, RealFlow nz,
                                RealFlow rho, RealFlow u, RealFlow v, RealFlow w, RealFlow p, RealFlow gam);
void CalJacobian_ConvectiveFlux_Roe(RealFlow matrix[5][5], RealFlow q_L[5], RealFlow q_R[5], RealFlow nx, RealFlow ny, RealFlow nz,
                                    RealFlow gam, RealFlow alf_l);
void CalJacobian_ViscousFlux(RealFlow matrix[5][5], RealFlow nx, RealFlow ny, RealFlow nz,
                             RealFlow rho, RealFlow u, RealFlow v, RealFlow w, RealFlow p, RealFlow gam, RealFlow miu, RealFlow k_mod, RealFlow sd1);

void CalDiagLUSGS(PolyGrid *grid, RealFlow *Diag, IntType level);
void SolveLUSGS3D(PolyGrid *grid, RealFlow *Diag, RealFlow *DQ[5], IntType *nFPC, IntType **C2F, IntType level);
void SolveLUSGS3D(PolyGrid *grid, RealFlow *Diag, RealFlow *DQ[5], RealFlow *rhs[5], IntType *nFPC, IntType **C2F, 
                  IntType Nsweep, RealFlow epsilon, IntType level);
void FluxLUSGS3D(RealFlow flux[5], RealFlow q[5], RealFlow DQ[5], RealGeom fa_n[3], RealFlow gam,  RealFlow p_bar, RealFlow lhs_omga);
void FluxLUSGS3D_unsteady(RealFlow flux[5], RealFlow q[5], RealFlow DQ[5], RealGeom fa_n[3], RealFlow gam,  RealFlow p_bar, RealFlow lhs_omga, RealFlow vgn);

void UpdateFlowField3D_CFL3d(PolyGrid *grid, RealFlow *DQ[5]);

void GeneratePlaneRotation(RealFlow &dx, RealFlow &dy, RealFlow &cs, RealFlow &sn);
void ApplyPlaneRotation(RealFlow &dx, RealFlow &dy, RealFlow &cs, RealFlow &sn);
void ComputeY(RealFlow **h, RealFlow *s, IntType k);

RealFlow DotProduct(RealFlow *a, RealFlow *b, IntType n);
RealFlow DotProductMPI(RealFlow *a, RealFlow *b, IntType n);
RealFlow DotProduct(RealFlow *u[], RealFlow *v[], IntType nvar, IntType nTCell);

void PreconditLUSGS(PolyGrid *grid, RealFlow *Diag, IntType level);
void ResLUSGS(PolyGrid *grid, RealFlow *dq, IntType level);
void ResLUSGS3D(PolyGrid *grid, RealFlow *Diag, RealFlow *DQ[5], IntType *nFPC, IntType **C2F, IntType level);

void SolveADU3D(PolyGrid *grid, RealFlow *Diag, RealFlow *DQ[5], IntType *nFPC, IntType **C2F, IntType level);
void SolveADU3D2(PolyGrid *grid, RealFlow **rhs, RealFlow *DQ[5], IntType *nFPC, IntType **C2F, IntType level);
void ComputeADU(PolyGrid *grid, RealFlow *Diag, RealFlow *v, RealFlow *res, IntType level);
void ComputeADU2(PolyGrid *grid, RealFlow *Diag, RealFlow *v, RealFlow *res, IntType level);
void ComputeADU3(PolyGrid *grid, RealFlow *v, RealFlow *res, RealFlow *reso, IntType level);

void GMRESSolverOrig(PolyGrid *grid, IntType level);
void GMRESSolverOrigUpdate( PolyGrid *grid, IntType level );
} // ~namespace mflow

#endif //~MFL_TEMPORAL_DISCRTEISATION_IMPLICIT_H
