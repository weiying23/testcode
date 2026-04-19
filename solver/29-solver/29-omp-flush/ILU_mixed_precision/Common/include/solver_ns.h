//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   solver_ns.h
/// \brief  the flow solver for NS equations
/// \author 
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_SOLVER_NS_H
#define MFL_SOLVER_NS_H

#include <stdlib.h>
#include "number_type.h"
#include "grid_polyhedra.h"
#include "data_pool.h"
#include "zone.h"
#include "solver_base.h"


namespace mflow
{

class NSSolver : public Solver 
{
  
private:
    IntType   nGrids;   // no. of grids
    PolyGrid  **grids;  // the grids for the solver
    BCond     *bc;      // physical boundary conditions only.
                        // grid related bc is stored in Grid object    
    DataStore **fields; // the fields associated with each grid
    DataSafe  *cPara;   // the control parameters for the solver
    
    RealFlow  dt;       // global dt for unsteady simulation
    Zone      *zone;    // the zone pointer
    
    const IntType kNVar;
    vector<string> var_name_;
    IntType grad_method_;
public:
    NSSolver(IntType ng, PolyGrid **gridsin, DataStore **fieldsin, 
             DataSafe *cParain, BCond *bcin, Zone *zonein);
    ~NSSolver(){};
  
    PolyGrid   *GetGrid(IntType n) const;

    void UpdateData(void *data, IntType type, IntType size, const ShortString name);
    void GetData(void *data, IntType type, IntType size, const ShortString name) const;
    void GetData(void *data, IntType type, IntType size, const ShortString name, IntType messageOn) const;

    void Init();
    void Solve();
    void Post();

    /// \brief  �ڵõ��µ����������󣬲��д�ֵ�����ñ߽�ֵ�������ݶȡ�Ԥ�������㣬
    ///         ����ճ��
    virtual void ProcessAfterNewQuantity(PolyGrid *grid);

    void UpdateInterfaceData();
    void UpdataUnstVolData();
    void CommInterfaceData(const char *name);
    void CommUnstVolData(const char *name, const char *name_cur, const char *name_old);
    void DumpRestart(PolyGrid *grid, IntType iter, IntType zn, RealFlow t_now);
    void ReadStepInfoFromFile(PolyGrid *grid); // read step information from RESTART file
    void AllocateFlowfieldMemory(PolyGrid *grid); // allocate memory  for flow field
    void ReadRestartFromFile(PolyGrid *grid);  
    void InitGridVar(PolyGrid *grid);
    void InitGridVarUnst(PolyGrid *grid);
    void DumpPressureForce(PolyGrid *grid, IntType iter, IntType zn,
                           RealFlow &pfx, RealFlow &pfy, RealFlow &pfz, RealFlow &total,
                           RealFlow &pmx, RealFlow &pmy, RealFlow &pmz);
    void DumpViscousForce(PolyGrid *grid, IntType iter, IntType zn, 
                          RealFlow &vfx, RealFlow &vfy, RealFlow &vfz, RealFlow &total,
                          RealFlow &vmx, RealFlow &vmy, RealFlow &vmz);
    void DumpForce(PolyGrid *grid, IntType iter, IntType zn);
#if (defined FS_CUDA)||(defined FS_CUDA_DEBUG)
	void QuantityGradient_Init(PolyGrid* grid);
	void cuTransferInterfaceData(PolyGrid *grid);
#endif

private:
    /// \brief  Ϊ���б߽����������ֵ
    void TransferInterfaceData(PolyGrid *grid);

    /// \brief  ����������������ݶȼ��㷽ʽ
    void set_grad_method(const PolyGrid *grid);
    /// \brief  Ϊrho, u, v, w, p�ݶȷ����ڴ棬�����µ�gField��
    void AllocateQuantityGradientMemory(PolyGrid *grid);
    /// \brief  ����rho, u, v, w, p�ݶȣ������µ�gField��
    /// \note   ��ҪԤ�ȷ����ڴ�
    void CalculateQuantityGradient(PolyGrid *grid);
	void InitCalculateQuantityGradient(PolyGrid *grid);

    /// \brief  ������������ݶ�ֵ��ע�⣺ֻ�������ٶ��ݶȵ�ֵ����֧�ֵı߽�����
    ///         ���٣�����û�з���������ڱ߽�
    void SetGhostQuantityGradients(
        const PolyGrid *grid, 
        RealFlow **dqdx, RealFlow **dqdy, RealFlow **dqdz
    );
};

// inline functions

inline PolyGrid * NSSolver::GetGrid(IntType n) const 
{
    return grids[n];
}

inline void NSSolver::UpdateData(void *data, IntType type, IntType size, const ShortString name)
{
    cPara->UpdateDataSafe(data,type,size,name);
}

inline void NSSolver::GetData(void *data, IntType type, IntType size, const ShortString name) const
{
    cPara->GetDataByName(data,type,size,name);
}

inline void NSSolver::GetData(void *data, IntType type, IntType size, const ShortString name, IntType messageOn) const
{
    cPara->GetDataByName(data,type,size,name,messageOn);
}

// Auxiliary functions
void SetGhostVariables(PolyGrid *grid);
/// \brief  ������������¶��ݶ�ֵ��ע�⣺֧�ֵı߽����ͽ��٣�����û�з�����
///         ����ڱ߽�
void SetGhostTemperatureGradient(
    const PolyGrid *grid, RealFlow *dtdx, RealFlow *dtdy, RealFlow *dtdz
);
/// \brief  ���Ѿ������ݶȵ�����»�ȡ������ٶ��ݶ�
void GetVelocityGradient(PolyGrid *grid, RealFlow *dvdxout[3], RealFlow *dvdyout[3], RealFlow *dvdzout[3]);
void SolveNSOnGrid(PolyGrid *grid, IntType level);
void UpdateResiduals(PolyGrid *grid, IntType level);
void Relaxation(PolyGrid *grid, IntType level, RealFlow *rhs, IntType steps);
void ZeroResiduals(PolyGrid *grid);
void CellIsMG(PolyGrid *grid, IntType *det);
void FreeAllGridResi(PolyGrid *grid);

void ExplicitStep(PolyGrid* grid);
void LoadQ(PolyGrid *grid, RealFlow **q);
void TransQtoW(PolyGrid *grid, RealFlow **q);
void TimeMarch(PolyGrid *grid, RealFlow **q, RealFlow *dt, RealFlow lamda);

void ComputeTimeStep(PolyGrid *grid);
void LimitTimeStep(PolyGrid *grid, RealFlow *dt);
void TimeStepNormal_new(PolyGrid *grid, RealFlow *dt, IntType vis_run);
void CompInvFlux(PolyGrid *grid,  RealFlow *ql[5], RealFlow *qr[5], RealFlow *flux[5], RealGeom *xfn, RealGeom *yfn, RealGeom *zfn,   
                 RealGeom *area,  RealGeom *vgn,   IntType *face_act, RealFlow gam, RealFlow p_bar,  
                 RealFlow alf_l, RealFlow alf_n, IntType type_flux, 
#ifdef DC
                 RealFlow gascon, IntType EntropyCorType,
                 IntType steady, IntType *IsShockFace, 
                 IntType *IsNormalFace,
#endif
                IntType ns, IntType ne);
void SetQlQrWithQ(PolyGrid *grid, RealFlow *q[], RealFlow *ql[], RealFlow *qr[], IntType ns, IntType ne);
RealFlow **GetLimiters_resp(PolyGrid *grid);
void InviscidFlux(PolyGrid *grid, RealFlow **limit, IntType level);
void CalcuQlQr(PolyGrid *grid, RealFlow *ql[5], RealFlow *qr[5], RealFlow **limit,
               RealFlow *dqdx[5], RealFlow *dqdy[5], RealFlow *dqdz[5], 
#ifdef DC
               RealFlow p_bar,
#endif
                IntType ns, IntType ne);
void ModQlQrBou(PolyGrid *grid, RealFlow *ql[], RealFlow *qr[], 
#ifdef DC
                RealFlow *q[],
                IntType steady,
#endif
                IntType ns, IntType ne);
void RoeFlux_noprec(PolyGrid *grid, RealFlow *ql[5], RealFlow *qr[5], RealFlow *flux[5], RealGeom *xfn, RealGeom *yfn, RealGeom *zfn, 
                    RealGeom *area, IntType *face_act, RealFlow gam, RealFlow p_bar, RealFlow alf_l, RealFlow alf_n, 
#ifdef DC
                    RealFlow gascon, IntType EntropyCorType,
                    IntType steady, IntType *IsShockFace, 
                    IntType *IsNormalFace,
#endif
                    IntType ns, IntType ne);

void LoadFlux_DC(PolyGrid *grid, RealFlow *flux[], 
                IntType ns, IntType ne
#ifdef DC
                ,RealFlow **res
#endif
);
void LoadFlux(PolyGrid *grid, RealFlow *flux[], IntType ns, IntType ne);
void LoadFlux(PolyGrid *grid, RealFlow *fluxl[], RealFlow *fluxr[], IntType ns, IntType ne);
void ViscousFlux(PolyGrid *grid, IntType level);
RealFlow *GetTemperature(PolyGrid *grid);
void ComputeVis_l(PolyGrid *grid);
RealFlow Sutherland_classic(RealFlow T);
void InitVis_t(PolyGrid *grid);
void CalDeriWeight(PolyGrid *grid, RealGeom *deltl, RealGeom *deltr, IntType ns, IntType ne, IntType key);
void CalVeloandTFace_average(PolyGrid *grid, RealFlow *vel_f[3], RealFlow *vel[3], RealFlow *t_f, RealFlow *t, IntType ns, IntType ne);
void CalVisHeatFace_average(PolyGrid *grid, RealFlow *vis_l, RealFlow *visc_f, RealFlow *heat_f, 
#ifdef DC
                                IntType vis_mode, IntType cond_comp, RealFlow gam, RealFlow gascon, 
                                RealFlow cp, RealFlow prl, 
#endif
                                IntType ns, IntType ne);
void CalVisFluxTest(PolyGrid *grid, RealFlow *vel[3], RealFlow *t, RealFlow *vel_f[3],
                    RealFlow *visc_f, RealFlow *heat_f, RealFlow *t_f,
                    RealFlow *dqdx[3], RealFlow *dqdy[3], RealFlow *dqdz[3],
                    RealFlow *dtdx, RealFlow *dtdy, RealFlow *dtdz,
                    RealGeom *deltl, RealGeom *deltr, RealFlow *flux[5],
#ifdef DC
                    IntType vis_mode, RealGeom BadFaceAngle,
#endif
                    IntType ns, IntType ne);
void DumpNormResi(PolyGrid *grid, IntType iter, IntType zn, RealFlow t_now);

void ModifyRestartInputFile(PolyGrid *grid);
void AddUnstSource(PolyGrid *grid);
void SolveEquationforGradSYMM(RealFlow gv1[3][3], RealFlow gv2[3][3], RealGeom xfn, RealGeom yfn, RealGeom zfn);
void CalIsShockFace(PolyGrid *grid, IntType *IsShockFace, IntType ns, IntType ne);
void TellDivergence(PolyGrid *grid, IntType iter, RealFlow norm);
} //~namespace mflow

#endif //~MFL_SOLVER_NS_H
