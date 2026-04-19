//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   temporal_discretisation_implicit.cpp
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

// direct head file
#include "temporal_discretisation_implicit.h"

// C++ build-in head files
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cassert>
#include <cstring>
#include <iostream>
using namespace std;

// other user defined head files
#include "number_type.h"
#include "zone.h"
#include "grid_polyhedra.h"
#include "utility_functions.h"
#include "solver_ns.h"
#include "io_base_format.h"
#include "io_log.h"
#include "parallel_base_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"
#include "gmres_ilu.h"
// head files relying on condition-compiling
#ifdef MPICH
#include <mpi.h>
#endif

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif

//dingxin
#ifdef TIMECOST
extern double* timecost;
extern double  time_flux, time_invis, time_roe, time_vis, time_calvis;
extern double  time_limiter;
extern double  time_gradient;
extern double  time_lusgs;
#endif
//TIMECOST

const static int VEC = 8; 
#define ALIGN 64

namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 12001  // define file id


#ifdef MPICH
extern int myZone;  // zhnc add
#endif


/*******************************************************************************\
   Forward the flow field One Step
\*******************************************************************************/
void ForwardStep(PolyGrid *grid, RealFlow *rhs, IntType level, IntType steps)
{
    ZeroResiduals(grid);
    UpdateResiduals(grid, level);

#ifdef TIMECOST//dingxin
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemLusgs, endtimeTemLusgs;
    double timeuseTemLusgs;
    gettimeofday(&starttimeTemLusgs, 0); 
#endif
#endif

    IntType gmres = 0;
    grid->GetData(&gmres, INT, 1, "GMRES", 0);
    IntType tScheme;
    grid->GetData(&tScheme, INT, 1, "tScheme");
    if(tScheme == MATRIX_FORMAT){
        // CPU execution of GMRES + ILU precond
        GMRESSolver(grid, level);
    }
    else if(tScheme == LU_SGS){
        ForwardLUSGS(grid, level);
        //printf("tScheme:%d ForwardLUSGS\n",tScheme);
    }
    else if(tScheme == DPLUR){
        ForwardDPLUR(grid, level);
        //printf("tScheme:%d ForwardDPLUR\n",tScheme);
    }
    else{
        GMRESSolverOrigUpdate(grid, level);
        //printf("tScheme:%d GMRESSolverOrigUpdate\n",tScheme);
    }

#ifdef TIMECOST//dingxin
#ifdef MPICH
    timecost[2] = timecost[2] + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemLusgs, 0); 
    timeuseTemLusgs = (RealGeom) 1000000*(endtimeTemLusgs.tv_sec - starttimeTemLusgs.tv_sec) + endtimeTemLusgs.tv_usec - starttimeTemLusgs.tv_usec;
    timecost[2] += timeuseTemLusgs;
    timeuseTemLusgs /= 1000000.0;
    time_lusgs += timeuseTemLusgs;
#endif
#endif
}


/*******************************************************************************\
  Forward the solution one time step using LU-SGS method.
  zhyb: 包含LU-SGS的改进型，单重网格单sweep的预估-校正型LU-SGS比原始的LU-SGS计算效率明显要高，
        别的情况下，效率提高不明显，甚至效率下降。
  参考文献：赵信文，预估-校正LU-SGS的隐式算法，航空计算技术第42卷第4期，2012年
  方法：在原始LU-SGS的基础上，增加一个校正步，将原始LU-SGS省略的高阶项L*(D-1)*U*(DQ)加进来
  zhyb20190301: 目前的经验是，多步LUSGS迭代收敛速度最快，在无多重网格时，建议sweeps取4，有多
                重网格时，建议sweeps取3收敛最快，且残差下降最好。epsilon建议取0.01~0.1之间，
                一般情况下取0.05收敛更快。
\*******************************************************************************/
void ForwardLUSGS(PolyGrid *grid, IntType level)
{
    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  n      = nTCell + nBFace;
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");
    RealGeom *vol   = grid->GetCellVol();
    
    IntType sweeps = 1;
    grid->GetData(&sweeps, INT, 1, "sweeps");
    RealFlow epsilon = 0.1;
    grid->GetData(&epsilon, REAL_FLOW, 1, "epsilon");
    if(epsilon < TINY) epsilon = 0.1;
    
    // Get number of faces for each cell
    IntType *nFPC = CalnFPC(grid);
    // Get cell to face conections
    IntType **C2F = CalC2F(grid);
    
    IntType i, j, ntemp;
    // Now diagonal term in LU-SGS, here we need information of time steps
    RealFlow *Diag = NULL;
    mfmem::snew_array_1D(Diag, nTCell,dmrfl);
    assert(Diag != 0);
    //未修改overlap
    
    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
    for(i=0; i<nTCell; i++) 
        Diag[i] = vol[i]/dt[i];
    
    // Note: As it has been shown, Diag = CFL/2*Vol/Dt.
    //       If function CalDiagLUSGS is not called, make sure CFL <= 2.
    //未修改overlap
    CalDiagLUSGS(grid, Diag, level);
    
    // Allocate memories for RHS or DQ
    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*n, "DQ");
    if(!DQ[0]){
        mfmem::snew_array_1D(DQ[0],5*n,dmrfl);
        grid->UpdateDataPtr(DQ[0], REAL_FLOW, 5*n, "DQ");
    }
    assert(DQ[0] != 0);
    for(i=1; i<5; i++) DQ[i] = &DQ[i-1][n];
    for(j=0; j<5*n; j++) DQ[0][j] = 0.; 
    
    if(sweeps == -1){
        //预估步
        // Copy the residual to DQ
        ntemp = 0;
        for(i=0; i<5; i++){
            for(j=0; j<nTCell; j++){
                DQ[i][j] = res[ntemp++];
            }
        }
        // Now the LU-SGS part
        SolveLUSGS3D(grid, Diag, DQ, nFPC, C2F, level);
        
        //校正步
        //计算高阶项L(D-1)U(DQ)
        //需要先更新虚拟网格的值，因为在求高阶项中会用到虚拟网格的值，而在SolveLUSGS3D函数中
        //后扫描后没有更新虚拟网格的DQ的值
#ifdef MPICH
        IntType nvar = 5;
        RealFlow *q_mpi[5];
        for(j=0; j<5; j++)
            q_mpi[j] = DQ[j];
        grid->RecvSendVarNeighbor_Togeth(nvar, q_mpi);
#endif
        //高阶项
        RealFlow *AddTerm[5];
        AddTerm[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "AddTerm");
        if(!AddTerm[0]){
            mfmem::snew_array_1D(AddTerm[0],5*nTCell,dmrfl);
            for(i=0;i<5*nTCell;i++){
                AddTerm[0][i] = 0.0;
            }
            grid->UpdateDataPtr(AddTerm[0], REAL_FLOW, 5*nTCell, "AddTerm");
        }
        assert(AddTerm[0] != 0);
        for(i=1; i<5; i++) AddTerm[i] = &AddTerm[i-1][nTCell];
        
        AdditionTermforLUSGS(grid,AddTerm,Diag,DQ,level);
        
        ntemp = 0;
        for(i=0; i<5; i++){
            for(j=0; j<nTCell; j++){
                DQ[i][j] = res[ntemp++]+AddTerm[i][j];
            }
        }
        
        // Now the LU-SGS part
        SolveLUSGS3D(grid, Diag, DQ, nFPC, C2F, level);
    }else if(sweeps == -2){
        //预估步
        // Copy the residual to DQ
        ntemp = 0;
        for(i=0; i<5; i++){
            for(j=0; j<nTCell; j++){
                DQ[i][j] = res[ntemp++];
            }
        }
        // Now the LU-SGS part
        SolveLUSGS3D(grid, Diag, DQ, nFPC, C2F, level);
        
        //校正步
        //计算高阶项L(D-1)U(DQ)
        //需要先更新虚拟网格的值，因为在求高阶项中会用到虚拟网格的值，而在SolveLUSGS3D函数中
        //后扫描后没有更新虚拟网格的DQ的值
#ifdef MPICH
        IntType nvar = 5;
        RealFlow *q_mpi[5];
        for(j=0; j<5; j++)
            q_mpi[j] = DQ[j];
        grid->RecvSendVarNeighbor_Togeth(nvar, q_mpi);
#endif
        RealFlow *AddTerm[5];
        AddTerm[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "AddTerm");
        if(!AddTerm[0]){
            mfmem::snew_array_1D(AddTerm[0],5*nTCell,dmrfl);
            for(i=0;i<5*nTCell;i++){
                AddTerm[0][i] = 0.0;
            }
            grid->UpdateDataPtr(AddTerm[0], REAL_FLOW, 5*nTCell, "AddTerm");
        }
        assert(AddTerm[0] != 0);
        for(i=1; i<5; i++) AddTerm[i] = &AddTerm[i-1][nTCell];
        AdditionTermforLUSGS(grid,AddTerm,Diag,DQ,level);
        
        ntemp = 0;
        for(i=0; i<5; i++){
            for(j=0; j<nTCell; j++){
                DQ[i][j] = res[ntemp++]+AddTerm[i][j];
            }
        }
        
        // Now the LU-SGS part
        SolveLUSGS3D(grid, Diag, DQ, nFPC, C2F, level);
            
#ifdef MPICH
        grid->RecvSendVarNeighbor_Togeth(nvar, q_mpi);
#endif
        AdditionTermforLUSGS(grid,AddTerm,Diag,DQ,level);
        
        ntemp = 0;
        for(i=0; i<5; i++){
            for(j=0; j<nTCell; j++){
                DQ[i][j] = res[ntemp++]+AddTerm[i][j];
            }
        }
        
        // Now the LU-SGS part
        SolveLUSGS3D(grid, Diag, DQ, nFPC, C2F, level);
    }else if(sweeps == 1){  //单步
        // Copy the residual to DQ
        ntemp = 0;
        for(i=0; i<5; i++){
            for(j=0; j<nTCell; j++){
                DQ[i][j] = res[ntemp++];
            }
        }
        // Now the LU-SGS part
        SolveLUSGS3D(grid, Diag, DQ, nFPC, C2F, level);
    }else{  //多步
        RealFlow *rhs[5];
        rhs[0] = res;
        for(j=1; j<5; j++) rhs[j] = &rhs[j-1][nTCell];
        // Now the LU-SGS part ,   DQ conservative variable
        SolveLUSGS3D(grid, Diag, DQ, rhs, nFPC, C2F, sweeps, epsilon, level);
    }
    // Update flow field
    UpdateFlowField3D_CFL3d(grid, DQ);
    
    // delete temporary memories
    mfmem::sdel_array_1D(Diag);
}

void ForwardDPLUR(PolyGrid *grid, IntType level) {
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n = nTCell + nBFace;
    IntType nTFace = grid->GetNTFace();
    IntType* f2c = grid->Getf2c();
    RealFlow* xfn = grid->GetXfn();
    RealFlow* yfn = grid->GetYfn();
    RealFlow* zfn = grid->GetZfn();
    RealFlow* area = grid->GetFaceArea();
    RealFlow* vol = grid->GetCellVol();

    RealFlow* vgn = grid->GetFaceNormalVelocity();

    // Get number of faces for each cell
    IntType* nFPC = CalnFPC(grid);
    // Get cell to face conections
    IntType** C2F = CalC2F(grid);

    RealFlow *norm_dist_c2c = (RealGeom *)grid->GetDataPtr(REAL_GEOM, nTFace, "norm_dist_c2c");

    IntType steady = 1;
    grid->GetData(&steady, INT, 1, "steady");
    IntType DQ_limit = 1;
    grid->GetData(&DQ_limit, INT, 1, "DQ_limit");

    RealFlow gam, gamm1, p_bar, lhs_omga;
    grid->GetData(&gam, REAL_FLOW, 1, "gam");
    gamm1 = gam - 1.0;
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&lhs_omga, REAL_FLOW, 1, "lhs_omga");

    RealFlow rho00, u00, v00, w00, e_stag;
    grid->GetData(&rho00, REAL_FLOW, 1, "rho");
    grid->GetData(&u00, REAL_FLOW, 1, "u");
    grid->GetData(&v00, REAL_FLOW, 1, "v");
    grid->GetData(&w00, REAL_FLOW, 1, "w");
    grid->GetData(&e_stag, REAL_FLOW, 1, "e_stag");
    RealFlow rho_min, rho_max, p_min, p_max, e_stag_max;
    grid->GetData(&rho_min, REAL_FLOW, 1, "rho_min");
    grid->GetData(&rho_max, REAL_FLOW, 1, "rho_max");
    grid->GetData(&p_min, REAL_FLOW, 1, "p_min");
    grid->GetData(&p_max, REAL_FLOW, 1, "p_max");
    grid->GetData(&e_stag_max, REAL_FLOW, 1, "e_stag_max");

    RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");

    IntType vis_mode, vis_run = 0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    if (vis_mode != INVISCID) {
        vis_run = 1;
        // if coarse grid doesn't want to run the viscous flux, turn it off
        if (level != 0) {
            IntType cg_vis = 1;
            grid->GetData(&cg_vis, INT, 1, "cg_vis");
            if (cg_vis == 0) vis_run = 0;
        }
    }
    RealFlow *vis_l = NULL, *vis_t = NULL;
    if (vis_run) {
        vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
        vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    }

    IntType sweeps = 1;
    grid->GetData(&sweeps, INT, 1, "sweeps");
    RealFlow *Diag = NULL;
    mfmem::snew_array_1D(Diag, nTCell, dmrfl);

    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");

    RealFlow* res[5];
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5 * nTCell, "res");
    for (IntType i = 1; i < 5; i++) res[i] = &res[i - 1][nTCell];

    // Allocate memories for RHS or DQ
    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5 * n, "DQ");
    if (!DQ[0]) {
        mfmem::snew_array_1D(DQ[0], 5 * n, dmrfl);
        grid->UpdateDataPtr(DQ[0], REAL_FLOW, 5 * n, "DQ");
    }
    assert(DQ[0] != 0);
    for (IntType i = 1; i < 5; i++) DQ[i] = &DQ[i - 1][n];
    for (IntType j = 0; j < 5 * n; j++) DQ[0][j] = 0.;
    RealFlow** DQ0 = NULL;
    mfmem::snew_array_2D(DQ0, 5, n, dmrfl);
    for (IntType k = 0; k < 5; ++k) {
    	for (IntType i = 0; i < n; ++i) {
            DQ0[k][i] = 0;
    	}
    }

    IntType *luorder = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGSCellOrder"); //Ϊ�˶Խ�ռ�ţ��������������

    // ����Խ��ߣ�δ�޸�overlap
    for (IntType i = 0; i < nTCell; i++)
        Diag[i] = vol[i] / dt[i];

    CalDiagLUSGS(grid, Diag, level);

    // �����ֵ
    for (IntType i = 0; i < nTCell; ++i) {
    	for (IntType k = 0; k < 5; ++k) {
            DQ[k][i] = res[k][i] / Diag[i];
    	}
    }

    // Jacobi ����
    for (IntType idx_sweeps = 0; idx_sweeps < sweeps; ++idx_sweeps) {
        for (IntType i = 0; i < nTCell; ++i) {
            for (IntType k = 0; k < 5; ++k) {
                DQ0[k][i] = DQ[k][i];
                DQ[k][i]  = res[k][i];
            }
        }
#ifdef MPICH
        IntType nvar = 5;
        RealFlow *q_mpi[5];
        for (IntType j = 0; j < 5; j++)
            q_mpi[j] = DQ0[j];
        grid->RecvSendVarNeighbor_Togeth(nvar, q_mpi);
#endif

#if (defined FS_OPENMP)
#pragma omp parallel for //private(ilu, cell)
#endif
    	for (IntType ilu = 0; ilu < nTCell; ++ilu) {
    		IntType cell = ilu;//luorder[ilu];

            for (IntType idx_C2F = 0; idx_C2F < nFPC[cell]; ++idx_C2F) {                
                IntType face = C2F[cell][idx_C2F];
                IntType c1 = f2c[2 * face];
                IntType c2 = f2c[2 * face + 1];

                RealFlow face_n[3];
                RealFlow vgn_tmp;

                face_n[0] = xfn[face];
                face_n[1] = yfn[face];
                face_n[2] = zfn[face];
                if (!steady) vgn_tmp = vgn[face];
                if (c2 == cell) {
                    IntType c_tmp = c1;
                    c1 = c2;
                    c2 = c_tmp;
                    face_n[0] = -face_n[0];
                    face_n[1] = -face_n[1];
                    face_n[2] = -face_n[2];
                    if (!steady) vgn_tmp = -vgn[face];
                }
                assert(c1 == cell);

                RealFlow q_loc[5], DQ_loc[5], flux[5];
                for (IntType k = 0; k < 5; ++k) {
                    q_loc[k] = q[k][c2];
                    DQ_loc[k] = DQ0[k][c2];
                }
                // Calculate everything (I call it Flux) in lower triangular
                if (steady) {
                    FluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
                }
                else {
                    FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
                }
                if (vis_run) {
                    RealFlow dist = norm_dist_c2c[face];
                    RealFlow visc = vis_l[c2] + vis_t[c2];
                    RealFlow tmp = 2.0 * visc / (q_loc[0] * dist + TINY);
                    for (IntType i = 0; i < 5; ++i) flux[i] -= tmp*DQ_loc[i];
                }

                // Add Flux together
                RealFlow tmp = 0.5 * area[face];
                for (IntType i = 0; i < 5; ++i) DQ[i][cell] -= tmp * flux[i];
            }

            for (IntType k = 0; k < 5; ++k) DQ[k][cell] /= Diag[cell];
    	

            mflog::log.set_all_processors_out();
            if (fabs(DQ[0][cell])>1.0e3*rho00) {
                mflog::log << "Forward sweep: drho>1.0e3*rho00!  " << IOS_EP(2) << DQ[0][cell] << IOS_SEP
                    << cell << IOS_SEP << mflog::log.rank_id() << endl;
            }
            if (fabs(DQ[4][cell])>1.0e5*e_stag) {
                mflog::log << "Forward sweep: de>1.0e5*e_stag!  " << IOS_EP(2) << DQ[4][cell] << IOS_SEP
                    << cell << IOS_SEP << mflog::log.rank_id() << endl;
            }
            if (fabs(DQ[0][cell])>rho_max || DQ[4][cell] > e_stag_max) {
                mflog::log << "Simu_01_01_0001" << std::endl;
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }

            //limit for rho>0
            if (DQ_limit == 1) {
                // do nothing!
            }
            else if (DQ_limit == 2) {
                RealFlow dp, vv;
                vv = q[1][cell] * q[1][cell] + q[2][cell] * q[2][cell] + q[3][cell] * q[3][cell];
                dp = DQ[4][cell] + 0.5*DQ[0][cell] * vv - (DQ[1][cell] * q[1][cell] + DQ[2][cell] * q[2][cell] + DQ[3][cell] * q[3][cell]);
                dp *= gamm1;
                if ((q[0][cell] + DQ[0][cell]) < rho_min || (q[0][cell] + DQ[0][cell]) > rho_max ||
                    (q[4][cell] + dp) < p_min || (q[4][cell] + dp) > p_max) {
                    DQ[0][cell] *= 0.1;
                    DQ[1][cell] *= 0.1;
                    DQ[2][cell] *= 0.1;
                    DQ[3][cell] *= 0.1;
                    DQ[4][cell] *= 0.1;
                }
                dp = DQ[4][cell] + 0.5*DQ[0][cell] * vv - (DQ[1][cell] * q[1][cell] + DQ[2][cell] * q[2][cell] + DQ[3][cell] * q[3][cell]);
                dp *= gamm1;
                if ((q[0][cell] + DQ[0][cell]) < rho_min || (q[0][cell] + DQ[0][cell]) > rho_max ||
                    (q[4][cell] + dp) < p_min || (q[4][cell] + dp) > p_max) {
                    DQ[0][cell] *= 0.1;
                    DQ[1][cell] *= 0.1;
                    DQ[2][cell] *= 0.1;
                    DQ[3][cell] *= 0.1;
                    DQ[4][cell] *= 0.1;
                }
                dp = DQ[4][cell] + 0.5*DQ[0][cell] * vv - (DQ[1][cell] * q[1][cell] + DQ[2][cell] * q[2][cell] + DQ[3][cell] * q[3][cell]);
                dp *= gamm1;
                if ((q[0][cell] + DQ[0][cell]) < rho_min || (q[0][cell] + DQ[0][cell]) > rho_max ||
                    (q[4][cell] + dp) < p_min || (q[4][cell] + dp) > p_max) {
                    DQ[0][cell] = 0.0;
                    DQ[1][cell] = 0.0;
                    DQ[2][cell] = 0.0;
                    DQ[3][cell] = 0.0;
                    DQ[4][cell] = 0.0;
                }
            }
            else if (DQ_limit == 3) {
                DQ[0][cell] = MAX(DQ[0][cell], rho_min - q[0][cell]);
                DQ[0][cell] = MIN(DQ[0][cell], rho_max - q[0][cell]);
            }
            else if (DQ_limit == 4) {
                //come from NSMB code
                RealFlow alph, alph_rho, alph_rhoe, alph_p, dp, vv, rhoe;
                vv = q[1][cell] * q[1][cell] + q[2][cell] * q[2][cell] + q[3][cell] * q[3][cell];
                rhoe = 0.5*q[0][cell] * vv + (q[4][cell] + p_bar) / (gam - 1.0);
                dp = DQ[4][cell] + 0.5*DQ[0][cell] * vv - (DQ[1][cell] * q[1][cell] + DQ[2][cell] * q[2][cell] + DQ[3][cell] * q[3][cell]);
                dp *= gamm1;

                alph_rho = q[0][cell] / (MAX(q[0][cell], 0.05*rho00) + MAX(0.0, -DQ[0][cell]));
                alph_rhoe = rhoe / (MAX(rhoe, 0.05*e_stag) + MAX(0.0, -DQ[4][cell]));
                alph_p = (q[4][cell] + p_bar) / (MAX((q[4][cell] + p_bar), 0.05*p_bar) + MAX(0.0, -dp));
                alph = MIN(alph_rho, alph_rhoe);
                alph = MIN(alph, alph_p);
                for (IntType i = 0; i < 5; i++) DQ[i][cell] *= alph;
            }
            else {
                mflog::log.set_one_processor_out();
                mflog::log << endl << "DQ_limit is greater to 4! Now only have 4 methods." << endl;
                mflog::log << "Then we will use the first method, i.e. do nothing!" << endl;
            }
        }
    }


    // Update flow field
    UpdateFlowField3D_CFL3d(grid, DQ);

    // delete temporary memories
    mfmem::sdel_array_1D(Diag);
    mfmem::sdel_array_2D(DQ0);
}

/*******************************************************************************\
    计算LUSGS的附加项: L*(D-1)*U*DQ
\*******************************************************************************/
void AdditionTermforLUSGS(PolyGrid *grid, RealFlow *AddTerm[5], RealFlow *Diag, RealFlow *DQ[5], IntType level) 
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    IntType *f2c   = grid->Getf2c();
    IntType *nFPC  = CalnFPC(grid);
    IntType **C2F  = CalC2F(grid); 
    RealGeom *xfn  = grid->GetXfn();
    RealGeom *yfn  = grid->GetYfn();
    RealGeom *zfn  = grid->GetZfn();
    RealGeom *area = grid->GetFaceArea();
    
    // Get flow variables
    RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    
    IntType vis_mode, vis_run=0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    
    RealFlow gam, p_bar, lhs_omga;
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&lhs_omga,   REAL_FLOW, 1, "lhs_omga");
    
    // Some itermidiate variables
    IntType i, j, k, l, ilu, cell, face, count, c1, c2, c_tmp;
    RealFlow v_n, eig_c, eig_v, visc, cc;
    RealFlow matrix_jacobi_fc[5][5],q_loc[5],matrix_tmp;
    RealGeom dist;
    RealGeom face_n[3];
    RealFlow **D_1UDQ = NULL;
    mfmem::snew_array_2D(D_1UDQ,5,nTCell,dmrfl,false);
    for(i=0;i<5;i++){
        for(j=0;j<nTCell;j++){
            D_1UDQ[i][j] = 0.0;
        }
    }
        
    if(vis_mode != INVISCID){
        vis_run = 1;
        
        // if coarse grid doesn't want to run the viscous flux, turn it off
        if(level != 0){
            IntType cg_vis = 1;
            grid->GetData(&cg_vis, INT, 1, "cg_vis");
            if(cg_vis == 0) vis_run = 0;
        }
    }
    RealFlow *vis_l = NULL, *vis_t = NULL;
    if(vis_run){
        vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
        vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    }
    
    IntType *luorder = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGSCellOrder");
    IntType *layer = (IntType *)grid->GetDataPtr(INT, n, "LUSGSLayer");
    IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer");
    
    IntType nTFace = grid->GetNTFace();
    RealGeom *norm_dist_c2c = NULL;
    norm_dist_c2c = (RealGeom *)grid->GetDataPtr(REAL_GEOM, nTFace, "norm_dist_c2c");
    assert(norm_dist_c2c);  //must exist
    
    //赋初值
    for(i=0;i<5;i++){
        for(j=0;j<nTCell;j++){
            AddTerm[i][j] = 0.0;
        }
    }
#if (defined FS_OPENMP) && (defined CellColoring) 
    IntType laynum;
    IntType start, end;
    for(laynum=0; laynum<cellsPerlayer[0]; laynum++ ){
        start = cellsPerlayer[laynum+1];
        end   = cellsPerlayer[laynum+2];
        if(laynum == 0) {start++;}
#pragma omp parallel for private(ilu, cell)
    for(ilu=start; ilu<end; ilu++){
        cell = luorder[ilu];
#else
    for(ilu=1;ilu<nTCell;ilu++){
        cell = luorder[ilu];
#endif    
    //后扫描，求D-1*U*DQ
    //for(ilu=nTCell-1;ilu>=0;ilu--){
    //   cell = luorder[ilu];

        for(IntType j=0; j<nFPC[cell]; j++){
            IntType face, count, c1, c2;
            IntType c_tmp;
            RealFlow v_n, eig_c, eig_v, visc, cc;
            RealFlow matrix_jacobi_fc[5][5],q_loc[5],matrix_tmp;
            RealGeom dist;
            RealGeom face_n[3];
            face  = C2F[cell][j];
            count = 2*face;
            c1    = f2c[count++];
            c2    = f2c[count];
            // One of c1 and c2 must be cell itself. 
            if(layer[c1]<layer[cell] || layer[c2]<layer[cell]) continue;

            // Now its neighboring cell belongs to upper triangular
            face_n[0] = xfn[face];
            face_n[1] = yfn[face];
            face_n[2] = zfn[face];
            if(c2 == cell){
                c_tmp = c1;
                c1    = c2;
                c2    = c_tmp;
                face_n[0] = -face_n[0];
                face_n[1] = -face_n[1];
                face_n[2] = -face_n[2];
            }
            assert(c1 == cell);
            q_loc[0]  = q[0][c2];
            q_loc[1]  = q[1][c2];
            q_loc[2]  = q[2][c2];
            q_loc[3]  = q[3][c2];
            q_loc[4]  = q[4][c2]+p_bar;
            
            //Jacobian of convective flux
            CalJacobian_ConvectiveFlux(matrix_jacobi_fc, face_n[0], face_n[1], face_n[2],
                                       q_loc[0], q_loc[1], q_loc[2], q_loc[3], q_loc[4], gam);
            
            // Eigenvalues of convective flux
            v_n    = q_loc[1]*face_n[0] + q_loc[2]*face_n[1] + q_loc[3]*face_n[2];
            cc     = gam*q_loc[4]/q_loc[0];
            eig_c  = fabs(v_n) + sqrt(cc);
            eig_c *= lhs_omga;
            
            if(vis_run){
                // Eigenvalues of viscous flux
                dist = norm_dist_c2c[face];
                visc = vis_l[c2] + vis_t[c2];
                eig_v = 2.0*visc/(q_loc[0]*dist + TINY);
            }
            
            for(IntType k=0;k<5;k++){
                matrix_jacobi_fc[k][k] -= eig_c;
                if(vis_run){
                    matrix_jacobi_fc[k][k] -= eig_v;
                }
            }
            for(IntType k=0;k<5;k++){
                matrix_tmp = 0.0;
                for(IntType l=0;l<5;l++){
                    matrix_tmp += matrix_jacobi_fc[k][l]*DQ[l][c2];
                }
                matrix_tmp *= 0.5*area[face];
                D_1UDQ[k][c1] += matrix_tmp;
            }
        }
    }
#if (defined FS_OPENMP) && (defined CellColoring) 
    }
#endif    
    //D-1*(U*DQ)
    for(i=0;i<5;i++){
        for(cell=0;cell<nTCell;cell++){
            D_1UDQ[i][cell] /= Diag[cell];
        }
    }
    
    //前扫描，求L*(D-1*(U*DQ))
#if (defined FS_OPENMP) && (defined CellColoring)
    for(laynum=cellsPerlayer[0]-1; laynum>=0; laynum-- ){
        start = cellsPerlayer[laynum+2];
        end   = cellsPerlayer[laynum+1];
#pragma omp parallel for private(ilu, cell)
    for(ilu=start-1; ilu>=end; ilu--){
        cell = luorder[ilu];
#else
    for(ilu=nTCell-1;ilu>=0;ilu--){
        cell = luorder[ilu];
#endif
    //for(ilu=0;ilu<nTCell;ilu++){
    //   cell = luorder[ilu];

        for(j=0; j<nFPC[cell]; j++){
            IntType face, count, c1, c2;
            IntType c_tmp;
            RealFlow v_n, eig_c, eig_v, visc, cc;
            RealFlow matrix_jacobi_fc[5][5],q_loc[5],matrix_tmp;
            RealGeom dist;
            RealGeom face_n[3];
            face  = C2F[cell][j];
            count = 2*face;
            c1    = f2c[count++];
            c2    = f2c[count];
            // One of c1 and c2 must be cell itself. 
            if(layer[c1]>layer[cell] || layer[c2]>layer[cell]) continue;

            // Now its neighboring cell belongs to lower triangular
            face_n[0] = xfn[face];
            face_n[1] = yfn[face];
            face_n[2] = zfn[face];
            if(c2 == cell){
                c_tmp = c1;
                c1    = c2;
                c2    = c_tmp;
                face_n[0] = -face_n[0];
                face_n[1] = -face_n[1];
                face_n[2] = -face_n[2];
            }
            assert(c1 == cell);
            q_loc[0]  = q[0][c2];
            q_loc[1]  = q[1][c2];
            q_loc[2]  = q[2][c2];
            q_loc[3]  = q[3][c2];
            q_loc[4]  = q[4][c2]+p_bar;
            
            //Jacobian of convective flux
            CalJacobian_ConvectiveFlux(matrix_jacobi_fc, face_n[0], face_n[1], face_n[2],
                                       q_loc[0], q_loc[1], q_loc[2], q_loc[3], q_loc[4], gam);
            
            // Eigenvalues of convective flux
            v_n    = q_loc[1]*face_n[0] + q_loc[2]*face_n[1] + q_loc[3]*face_n[2];
            cc     = gam*q_loc[4]/q_loc[0];
            eig_c  = fabs(v_n) + sqrt(cc);
            eig_c *= lhs_omga;
            
            if(vis_run){
                // Eigenvalues of viscous flux
                dist = norm_dist_c2c[face];
                visc = vis_l[c2] + vis_t[c2];
                eig_v = 2.0*visc/(q_loc[0]*dist + TINY);
            }
            
            for(IntType k=0;k<5;k++){
                matrix_jacobi_fc[k][k] -= eig_c;
                if(vis_run){
                    matrix_jacobi_fc[k][k] -= eig_v;
                }
            }
            
            for(IntType k=0;k<5;k++){
                matrix_tmp = 0.0;
                for(IntType l=0;l<5;l++){
                    matrix_tmp += matrix_jacobi_fc[k][l]*D_1UDQ[l][c2];
                }
                matrix_tmp *= 0.5*area[face];
                AddTerm[k][c1] += matrix_tmp;
            }
        }
    }
#if (defined FS_OPENMP) && (defined CellColoring) 
    }
#endif    
    mfmem::sdel_array_2D(D_1UDQ,5,false);

}


/*******************************************************************************\
  Calculate Convective flux Jacobian
\*******************************************************************************/
void CalJacobian_ConvectiveFlux(RealFlow Matrix[5][5], RealFlow nx, RealFlow ny, RealFlow nz,
                                RealFlow rho, RealFlow u, RealFlow v, RealFlow w, RealFlow p, RealFlow gam)
{
    RealFlow a1, a2, a3, Vn, phi;
    RealFlow E, vv, gamm1;
    
    gamm1 = gam-1.0;
    vv = 0.5*(u*u+v*v+w*w);
    phi = gamm1*vv;
    E  = p/(rho*gamm1) + vv;
    a1 = gam*E-phi;
    a2 = gamm1;
    a3 = gam-2.0;
    Vn = nx*u+ny*v+nz*w;
    
    Matrix[0][0] = 0.0;
    Matrix[0][1] = nx;
    Matrix[0][2] = ny;
    Matrix[0][3] = nz;
    Matrix[0][4] = 0.0;
    
    Matrix[1][0] = nx*phi - u*Vn;
    Matrix[1][1] = Vn - a3*nx*u;
    Matrix[1][2] = ny*u - a2*nx*v;
    Matrix[1][3] = nz*u - a2*nx*w;
    Matrix[1][4] = a2*nx;
    
    Matrix[2][0] = ny*phi - v*Vn;
    Matrix[2][1] = nx*v - a2*ny*u;
    Matrix[2][2] = Vn - a3*ny*v;
    Matrix[2][3] = nz*v - a2*ny*w;
    Matrix[2][4] = a2*ny;
    
    Matrix[3][0] = nz*phi - w*Vn;
    Matrix[3][1] = nx*w - a2*nz*u;
    Matrix[3][2] = ny*w - a2*nz*v;
    Matrix[3][3] = Vn - a3*nz*w;
    Matrix[3][4] = a2*nz;
    
    Matrix[4][0] = Vn*(phi-a1);
    Matrix[4][1] = nx*a1 - a2*u*Vn;
    Matrix[4][2] = ny*a1 - a2*v*Vn;
    Matrix[4][3] = nz*a1 - a2*w*Vn;
    Matrix[4][4] = gam*Vn;
}


/*******************************************************************************\
  Forward the solution one time step using LU-SGS method.
\*******************************************************************************/
void PreconditLUSGS(PolyGrid *grid, RealFlow *Diag, IntType level)
{
    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  nTotal = nTCell + nBFace;
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");        
    
    IntType sweeps = 1;
    grid->GetData(&sweeps, INT, 1, "sweeps");
    RealFlow epsilon = 0.1;
    grid->GetData(&epsilon, REAL_FLOW, 1, "epsilon"); 
    if(epsilon < TINY) epsilon = 0.1;
    
    // Get number of faces for each cell
    IntType *nFPC = CalnFPC(grid);
    // Get cell to face connectivity
    IntType **C2F = CalC2F(grid); 
      
    IntType  i, j, ntemp;

    // Allocate memories for RHS or DQ
    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTotal, "DQ");
    assert(DQ[0] != 0);
    for(i=1; i<5; i++) DQ[i] = &DQ[i-1][nTotal];
    for(j=0; j<5*nTotal; j++) DQ[0][j] = 0.; 

    if(sweeps == 1){
        // Copy the residual to DQ
        ntemp = 0;
        for(i=0; i<5; i++)
            for(j=0; j<nTCell; j++) DQ[i][j] = res[ntemp++];
        // Now the LU-SGS part
        SolveLUSGS3D(grid, Diag, DQ, nFPC, C2F, level);
    }else{
        RealFlow *rhs[5];
        rhs[0] = res;
        for(j=1; j<5; j++) rhs[j] = &rhs[j-1][nTCell];
        // Now the LU-SGS part ,   DQ conservative variable
        SolveLUSGS3D(grid, Diag, DQ, rhs, nFPC, C2F, sweeps, epsilon, level);
    }
}


/*******************************************************************************\
  Forward the solution one time step using LU-SGS method.
\*******************************************************************************/
void ResLUSGS(PolyGrid *grid, RealFlow *dq, IntType level)
{
    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  n      = nTCell + nBFace;
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");        
    RealGeom *vol = grid->GetCellVol();
    
    IntType sweeps = 1;
    grid->GetData(&sweeps, INT, 1, "sweeps");
    RealFlow epsilon = 0.1;
    grid->GetData(&epsilon, REAL_FLOW, 1, "epsilon"); 
    if(epsilon < TINY) epsilon = 0.1;
    
    // Get number of faces for each cell
    IntType *nFPC = CalnFPC(grid);
    // Get cell to face connectivity
    IntType **C2F = CalC2F(grid); 
      
    IntType  i, j, ntemp;
    // Now diagonal term in LU-SGS, here we need information of time steps
    RealFlow *Diag = NULL;
    mfmem::snew_array_1D(Diag, nTCell,dmrfl);
    assert(Diag != 0);
    //未修改overlap
    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
    for(i=0; i<nTCell; i++) 
        Diag[i] = vol[i]/dt[i];
    
    // Note: As it has been shown, Diag = CFL/2*Vol/Dt.
    //       If function CalDiagLUSGS is not called, make sure CFL <= 2.
    //未修改overlap
    CalDiagLUSGS(grid, Diag, level);
    
    // Allocate memories for RHS or DQ
    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*n, "DQ");
    assert(DQ[0] != 0);
    for(i=1; i<5; i++) DQ[i] = &DQ[i-1][n];
    for(j=0; j<5*n; j++) DQ[0][j] = 0.; 

    if(sweeps == 1){
        // Copy the residual to DQ
        ntemp = 0;
        for(i=0; i<5; i++)
            for(j=0; j<nTCell; j++) DQ[i][j] = dq[ntemp++];
        // Now the LU-SGS part
        ResLUSGS3D(grid, Diag, DQ, nFPC, C2F, level);
    }else{
        std::cerr<<"Need new code!"<<endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        RealFlow *rhs[5];
        rhs[0] = res;
        for(j=1; j<5; j++) rhs[j] = &rhs[j-1][nTCell];
        // Now the LU-SGS part ,   DQ conservative variable
        SolveLUSGS3D(grid, Diag, DQ, rhs, nFPC, C2F, sweeps, epsilon, level);
    }
    // Update flow field

    //Update the Res
    ntemp = 0;
    for(i=0; i<5; i++)
        for(j=0; j<nTCell; j++) res[ntemp++] = DQ[i][j];

    // delete temporary memories
   mfmem::sdel_array_1D(Diag);
}


/*******************************************************************************\
 Solve linear systems using the LU-SGS in 3D ~~~ORIGINAL LU-SGS ONE SWEEP~~~
\*******************************************************************************/
void ResLUSGS3D(PolyGrid *grid, RealFlow *Diag, RealFlow *DQ[5], IntType *nFPC, IntType **C2F, IntType level)
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    IntType *f2c   = grid->Getf2c();
    // Get grid metrics
    RealGeom *xfn = grid->GetXfn();
    RealGeom *yfn = grid->GetYfn();
    RealGeom *zfn = grid->GetZfn();
    RealGeom *area= grid->GetFaceArea();
    // Get flow variables
    RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    
    RealFlow p_bar,gam,lhs_omga;
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    grid->GetData(&lhs_omga,   REAL_FLOW, 1, "lhs_omga");
    
    IntType vis_mode, vis_run = 0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    if(vis_mode != INVISCID) 
    {
        vis_run = 1;
    
        // if coarse grid doesn't want to run the viscous flux, turn it off
        if(level != 0) {
            IntType cg_vis = 1;
            grid->GetData(&cg_vis, INT, 1, "cg_vis");
            if(cg_vis == 0) vis_run = 0;
        }
    }
    RealFlow  *vis_l = NULL, *vis_t = NULL;
    if(vis_run){
        vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
        vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    }
    // Some temporary variables
    IntType  i, j, face, cell, c1, c2, c_tmp, count;
    RealFlow flux_s[5], flux[5], q_loc[5], DQ_loc[5], visc, tmp;
    RealGeom face_n[3], dist;
    
    IntType nTFace = grid->GetNTFace();
    RealGeom *norm_dist_c2c = NULL;
    norm_dist_c2c = (RealGeom *)grid->GetDataPtr(REAL_GEOM, nTFace, "norm_dist_c2c");
    assert(norm_dist_c2c);  //must exist
    
    // Backward Sweep 
    for(cell=0; cell<nTCell; cell++){
        for(i=0; i<5; i++) flux_s[i] = 0.;
        for(j=0; j<nFPC[cell]; j++)
        {
            face  = C2F[cell][j];
            count = 2*face;
            c1    = f2c[count++];
            c2    = f2c[count];
            // One of c1 and c2 must be cell itself. 
            if(c1<cell || c2<cell) continue;

            // Now its neighboring cell belongs to upper triangular
            face_n[0] = xfn[face];
            face_n[1] = yfn[face];
            face_n[2] = zfn[face];
            if(c2 == cell)
            {
                c_tmp = c1;
                c1    = c2;
                c2    = c_tmp;
                face_n[0] = -face_n[0];
                face_n[1] = -face_n[1];
                face_n[2] = -face_n[2];
            }
            assert(c1 == cell);
            for(i=0; i<5; i++) 
            {
                q_loc[i]  = q[i][c2];
                DQ_loc[i] = DQ[i][c2];
            }
            // Calculate everything (I call it Flux) in upper triangular
            FluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
        
            if(vis_run)
            {
                dist = norm_dist_c2c[face]; 
                visc = vis_l[c2] + vis_t[c2];
                tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
                for(i=0; i<5; i++) flux[i] -= tmp*DQ_loc[i];
            }
        
            // Add Flux together
            tmp = area[face];
            for(i=0; i<5; i++) flux_s[i] += tmp*flux[i];
        }
        tmp = 2.0*Diag[cell];
        for(i=0; i<5; i++) DQ[i][cell] += flux_s[i]/tmp;
    }

#ifdef MPICH
    IntType nvar = 5;
    RealFlow *q_mpi[5];
    for(IntType j=0; j<5; j++)
        q_mpi[j] = DQ[j];
    grid->RecvSendVarNeighbor_Togeth(nvar, q_mpi);
#endif
        

    // Now the Forward Sweep
    for(cell=nTCell-1; cell>=0; cell--){
        for(i=0; i<5; i++) flux_s[i] = 0.;

        for(j=0; j<nFPC[cell]; j++)
        {
            face  = C2F[cell][j];
            count = 2*face;
            c1    = f2c[count++];
            c2    = f2c[count];
            // One of c1 and c2 must be cell itself. 
            if(c1>cell || c2>cell) continue;

            // Now its neighboring cell belongs to lower triangular
            face_n[0] = xfn[face];
            face_n[1] = yfn[face];
            face_n[2] = zfn[face];
            if(c2 == cell)
            {
                c_tmp = c1;
                c1    = c2;
                c2    = c_tmp;
                face_n[0] = -face_n[0];
                face_n[1] = -face_n[1];
                face_n[2] = -face_n[2];
            }
            assert(c1 == cell);
            for(i=0; i<5; i++) {
                q_loc[i]  = q[i][c2];
                DQ_loc[i] = DQ[i][c2];
            }
            // Calculate everything (I call it Flux) in lower triangular
            FluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);

            if(vis_run)
            {
                dist = norm_dist_c2c[face];
                visc = vis_l[c2] + vis_t[c2];
                tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
                for(i=0; i<5; i++) flux[i] -= tmp*DQ_loc[i];
            }
        
            // Add Flux together
            tmp = 0.5*area[face];
            for(i=0; i<5; i++) flux_s[i] += tmp*flux[i];
        }

        for(i=0; i<5; i++) {
            DQ[i][cell] *= Diag[cell];
            DQ[i][cell] += flux_s[i];
        }
    }
}


/*******************************************************************************\
    Calculate diagonal term in LU-SGS.
\*******************************************************************************/
void CalDiagLUSGS(PolyGrid *grid, RealFlow *Diag, IntType level)
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nTFace = grid->GetNTFace();
    IntType n      = nTCell + nBFace;
    IntType *f2c   = grid->Getf2c();
    RealGeom *xfn  = grid->GetXfn();
    RealGeom *yfn  = grid->GetYfn();
    RealGeom *zfn  = grid->GetZfn();
    RealGeom *area = grid->GetFaceArea();
    RealGeom *vgn  = grid->GetFaceNormalVelocity();
    RealGeom *vol  = grid->GetCellVol();
    RealFlow *rho  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    RealFlow *u    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    RealFlow *v    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    RealFlow *w    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    RealFlow *p    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
  
    IntType steady;
    RealFlow gam, p_bar, time_accuracy, real_dt, lhs_omga;
    grid->GetData(&steady,  INT, 1, "steady");
    grid->GetData(&gam,    REAL_FLOW, 1, "gam");
    grid->GetData(&p_bar,  REAL_FLOW, 1, "p_bar");
    grid->GetData(&time_accuracy,  REAL_FLOW, 1, "time_accuracy");
    grid->GetData(&real_dt,  REAL_FLOW, 1, "real_dt");
    grid->GetData(&lhs_omga,  REAL_FLOW, 1, "lhs_omga");
    
    IntType i, c1, c2, count;
    RealFlow vn_1, vn_2, ss_1, ss_2;
    RealFlow eig, temp;
    
    RealGeom *norm_dist_c2c = NULL;
    norm_dist_c2c = (RealGeom *)grid->GetDataPtr(REAL_GEOM, nTFace, "norm_dist_c2c");
    assert(norm_dist_c2c);  //must exist
   
    // Boundary faces first
    for(i=0; i<nBFace; i++){
        c1   = f2c[2*i];
        vn_1 = u[c1]*xfn[i] + v[c1]*yfn[i] + w[c1]*zfn[i];
        if(!steady) vn_1 -= vgn[i];
        vn_1 = fabs(vn_1);
        ss_1 = gam*(p[c1] + p_bar)/rho[c1];
       
        eig  = vn_1 + sqrt(ss_1);
        
        Diag[c1] += 0.5*area[i] * eig*lhs_omga;//对角线的表达式
    }
    // Interior faces
    count = 2*nBFace;
    for(i=nBFace; i<nTFace; i++){
        c1 = f2c[count++];
        c2 = f2c[count++];
        
        // Cell c1
        vn_1 = u[c1]*xfn[i] + v[c1]*yfn[i] + w[c1]*zfn[i];
        if(!steady) vn_1 -= vgn[i];
        vn_1 = fabs(vn_1);
        ss_1 = gam*(p[c1] + p_bar)/rho[c1];
        
        eig  = vn_1 + sqrt(ss_1);
       
        Diag[c1] += 0.5*area[i] * eig*lhs_omga;//对角线的表达式
        
        // Cell c2
        vn_2 = u[c2]*xfn[i] + v[c2]*yfn[i] + w[c2]*zfn[i];
        if(!steady) vn_2 -= vgn[i];
        vn_2 = fabs(vn_2);
        ss_2 = gam*(p[c2] + p_bar)/rho[c2];
        
        eig  = vn_2 + sqrt(ss_2);
        
        Diag[c2] += 0.5*area[i] * eig*lhs_omga;//对角线的表达式
    }
  
    if (!steady){  
        for (i = 0; i < nTCell; i++) Diag[i] += (1.0 + time_accuracy)*vol[i] / real_dt;
    }
    
    // If flow is viscous, need to count the contribution from viscosity
    // 该程序的粘性增加预处理的需要修改,然后进行测试
    IntType vis_mode, vis_run=0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    if(vis_mode != INVISCID){
        vis_run = 1;
        // if coarse grid doesn't want to run the viscous flux, turn it off
        if(level != 0){
            IntType cg_vis = 1;
            grid->GetData(&cg_vis, INT, 1, "cg_vis");
            if(cg_vis == 0) vis_run = 0;
        }
    }

    if(vis_run){
        RealFlow *Diag_v = NULL;
        mfmem::snew_array_1D(Diag_v, nTCell,dmrfl);
        assert(Diag_v != 0);
        for(i=0; i<nTCell; i++) Diag_v[i] = 0.;
      
        RealFlow dot;

        count = 0;
        for(i=0; i<nTFace; i++){
            c1  = f2c[count++];
            c2  = f2c[count++];
            dot = norm_dist_c2c[i];
            temp = area[i]/(dot + TINY);
            // Cell c1
            Diag_v[c1] += temp;
            // Cell c2
            if(c2 < nTCell) Diag_v[c2] += temp;
        }
      
        RealFlow *vis_l= (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
        RealFlow *vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
        for(i=0; i<nTCell; i++) Diag[i] += (vis_l[i] + vis_t[i])*Diag_v[i]/rho[i];
     
        mfmem::sdel_array_1D(Diag_v);
    }
}


/*******************************************************************************\
 Solve linear systems using the LU-SGS in 3D ~~~ORIGINAL LU-SGS ONE SWEEP~~~
 zhyb: 增加网格排序，在单重网格时有明显的加速效果，但是多重网格时加速不明显，甚至收敛变慢
\*******************************************************************************/
void SolveLUSGS3D(PolyGrid *grid, RealFlow *Diag, RealFlow *DQ[5], IntType *nFPC, IntType **C2F, IntType level)
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    IntType *f2c   = grid->Getf2c();
    // Get grid metrics
    RealGeom *xfn = grid->GetXfn();
    RealGeom *yfn = grid->GetYfn();
    RealGeom *zfn = grid->GetZfn();
    RealGeom *area= grid->GetFaceArea();
    RealGeom *vgn = grid->GetFaceNormalVelocity();
    // Get flow variables
    RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    
    RealFlow gam, gamm1, p_bar, lhs_omga;
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    gamm1 = gam-1.0;
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&lhs_omga,   REAL_FLOW, 1, "lhs_omga");
    IntType steady=1;
    grid->GetData(&steady,  INT, 1, "steady");
   
    RealFlow rho00,u00,v00,w00,e_stag;
    grid->GetData(&rho00, REAL_FLOW, 1, "rho");
    grid->GetData(&u00, REAL_FLOW, 1, "u");
    grid->GetData(&v00, REAL_FLOW, 1, "v");
    grid->GetData(&w00, REAL_FLOW, 1, "w");
    grid->GetData(&e_stag,   REAL_FLOW, 1, "e_stag");

    RealFlow rho_min,rho_max,p_min,p_max,e_stag_max;
    grid->GetData(&rho_min, REAL_FLOW, 1, "rho_min");
    grid->GetData(&rho_max, REAL_FLOW, 1, "rho_max");
    grid->GetData(&p_min,   REAL_FLOW, 1, "p_min");
    grid->GetData(&p_max,   REAL_FLOW, 1, "p_max");
    grid->GetData(&e_stag_max, REAL_FLOW, 1, "e_stag_max");
   
    IntType DQ_limit = 1;
    grid->GetData(&DQ_limit, INT, 1, "DQ_limit");
    
    IntType vis_mode, vis_run = 0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");

    if(vis_mode != INVISCID){
        vis_run = 1;
        
        // if coarse grid doesn't want to run the viscous flux, turn it off
        if(level != 0){
            IntType cg_vis = 1;
            grid->GetData(&cg_vis, INT, 1, "cg_vis");
            if(cg_vis == 0) vis_run = 0;
        }
    }
    RealFlow *vis_l = NULL, *vis_t = NULL;
    if(vis_run){
        vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
        vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    }
    // Some temporary variables
    IntType i, j, ilu, face, cell, c1, c2, c_tmp, count;
    RealFlow flux_s[5], flux[5], q_loc[5], DQ_loc[5], visc, tmp, vgn_tmp;
    RealGeom face_n[3], dist;
        
    IntType *luorder = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGSCellOrder"); //为了对角占优，网格排序后的序号
    IntType *layer = (IntType *)grid->GetDataPtr(INT, n, "LUSGSLayer"); //LUSGS迭代的层号，层号小为下三角，层号大为上三角
    IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer");
    
    IntType nTFace = grid->GetNTFace();
    RealGeom *norm_dist_c2c = NULL;
    norm_dist_c2c = (RealGeom *)grid->GetDataPtr(REAL_GEOM, nTFace, "norm_dist_c2c");
    assert(norm_dist_c2c);  //must exist
    
    // Now the Forward Sweep
    for(i=0; i<5; i++) DQ[i][luorder[0]] /= Diag[luorder[0]];
/*
#ifdef FS_SIMD 
//containing SIMD
        
        IntType laynum;
        IntType start, end;
        RealFlow peff = gam*p_bar/gamm1;
    for(laynum=0; laynum<cellsPerlayer[0]; laynum++ ){
        start = cellsPerlayer[laynum+1];
        end   = cellsPerlayer[laynum+2];
        IntType yushu,ilu;
        if(laynum == 0) {
            start++;
        }
        yushu = (end-start) % VEC;
        end -= yushu;
#ifdef FS_OPENMP        
#pragma omp parallel for private(ilu)
#endif
        for(ilu=start; ilu<end; ilu+=VEC){
            IntType c_tmp[VEC], count[VEC], face[VEC], cell[VEC], c1[VEC], c2[VEC];
            RealFlow flux_0[VEC], flux_1[VEC], flux_2[VEC], flux_3[VEC], flux_4[VEC]; 
            RealFlow q_loc_0[VEC], q_loc_1[VEC], q_loc_2[VEC], q_loc_3[VEC], q_loc_4[VEC];
            RealFlow DQ_loc_0[VEC], DQ_loc_1[VEC], DQ_loc_2[VEC], DQ_loc_3[VEC], DQ_loc_4[VEC];
            RealFlow visc[VEC], tmp[VEC], vgn_tmp[VEC], Diag_cell[VEC];
            RealGeom face_n_0[VEC], face_n_1[VEC], face_n_2[VEC], dist[VEC];
            RealFlow DQ_0[VEC], DQ_1[VEC], DQ_2[VEC], DQ_3[VEC], DQ_4[VEC], q_0[VEC], q_1[VEC], q_2[VEC], q_3[VEC], q_4[VEC];
            RealFlow Q_0[VEC], Q_1[VEC], Q_2[VEC], Q_3[VEC], Q_4[VEC], rv2[VEC], v_n[VEC], p[VEC], eig[VEC];
            RealFlow alph[VEC], alph_rho[VEC], alph_rhoe[VEC], alph_p[VEC], dp[VEC], vv[VEC], rhoe[VEC];

            #pragma omp simd safelen(VEC)
            for(IntType i=0; i<VEC; i++){
                cell[i] = luorder[ilu+i];        //闈炶繛缁�鍙栧潃
                Diag_cell[i] = Diag[cell[i]];
                DQ_0[i] = DQ[0][cell[i]]; DQ_1[i] = DQ[1][cell[i]]; DQ_2[i] = DQ[2][cell[i]]; DQ_3[i] = DQ[3][cell[i]]; DQ_4[i] = DQ[4][cell[i]];  //闈炶繛缁�璁垮瓨鍚戦噺鍖�
                q_0[i] = q[0][cell[i]];   q_1[i] = q[1][cell[i]];   q_2[i] = q[2][cell[i]];   q_3[i] = q[3][cell[i]];   q_4[i] = q[4][cell[i]];
            }

            #pragma omp simd safelen(VEC)
            for(IntType i=0; i<VEC; i++){
                for(IntType j=0; j<nFPC[cell[i]]; j++){
                    face[i] = C2F[cell[i]][j];
                    count[i] = face[i] + face[i];
                    c1[i] = f2c[count[i]++];
                    c2[i] = f2c[count[i]];
                    if(layer[c1[i]]>layer[cell[i]] || layer[c2[i]]>layer[cell[i]]) { continue;}

                    face_n_0[i] = xfn[face[i]];  //闈炶繛缁�鍙栧潃
                    face_n_1[i] = yfn[face[i]];
                    face_n_2[i] = zfn[face[i]];
                    if(!steady) vgn_tmp[i] = vgn[face[i]];
                    if(c2[i] == cell[i]){
                        c_tmp[i] = c1[i];
                        c1[i]    = c2[i];
                        c2[i]    = c_tmp[i];
                        face_n_0[i] = -face_n_0[i];
                        face_n_1[i] = -face_n_1[i];
                        face_n_2[i] = -face_n_2[i];
                        if(!steady) vgn_tmp[i] = -vgn[face[i]];
                    }
                    assert(c1[i] == cell[i]);

                    q_loc_0[i]   = q[0][c2[i]];  q_loc_1[i]  = q[1][c2[i]];  
                    q_loc_2[i]   = q[2][c2[i]];  q_loc_3[i]  = q[3][c2[i]];   q_loc_4[i]  = q[4][c2[i]]; // DQ鍜宷闈炶繛缁�鍙栧潃
                    DQ_loc_0[i]  = DQ[0][c2[i]]; DQ_loc_1[i] = DQ[1][c2[i]];
                    DQ_loc_2[i]  = DQ[2][c2[i]]; DQ_loc_3[i] = DQ[3][c2[i]]; DQ_loc_4[i] = DQ[4][c2[i]];

                    //FluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
                    //FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
                    Q_0[i] = q_loc_0[i]; 
                    Q_1[i] = q_loc_0[i]*q_loc_1[i]; 
                    Q_2[i] = q_loc_0[i]*q_loc_2[i]; 
                    Q_3[i] = q_loc_0[i]*q_loc_3[i];
                    rv2[i] = 0.5*q_loc_0[i] * (q_loc_1[i]*q_loc_1[i] + q_loc_2[i]*q_loc_2[i] + q_loc_3[i]*q_loc_3[i]);
                    p[i] = q_loc_4[i];
                    Q_4[i] = p[i] / gamm1 + rv2[i];

                    v_n[i] = q_loc_1[i]*face_n_0[i] + q_loc_2[i]*face_n_1[i] + q_loc_3[i]*face_n_2[i];
                    if(steady){
                        eig[i] = fabs(v_n[i]) + sqrt( gam*(p[i]+p_bar) / q_loc_0[i] );
                    }
                    else{
                        eig[i] = fabs(v_n[i] - vgn_tmp[i]) + sqrt( gam*(p[i]+p_bar) / q_loc_0[i] );
                    }
                    eig[i] *= lhs_omga;

                    flux_1[i] = -Q_1[i]*v_n[i] - p[i]*face_n_0[i]; 
                    flux_2[i] = -Q_2[i]*v_n[i] - p[i]*face_n_1[i]; 
                    flux_3[i] = -Q_3[i]*v_n[i] - p[i]*face_n_2[i]; 
                    flux_4[i] = -(Q_4[i]+p[i]+peff) * v_n[i];

                    Q_0[i] +=DQ_loc_0[i]; Q_1[i] +=DQ_loc_1[i]; Q_2[i] +=DQ_loc_2[i]; 
                    Q_3[i] +=DQ_loc_3[i]; Q_4[i] +=DQ_loc_4[i];
                    rv2[i] = 0.5 * (Q_1[i]*Q_1[i]+Q_2[i]*Q_2[i]+Q_3[i]*Q_3[i]) / Q_0[i];
                    p[i] = gamm1*(Q_4[i] - rv2[i]);

                    flux_0[i] = DQ_loc_1[i]*face_n_0[i] + DQ_loc_2[i]*face_n_1[i] + DQ_loc_3[i]*face_n_2[i];
                    //v_n[i] *= q_loc_0[i]; v_n[i] += flux_0[i]; v_n[i] /= Q_0[i];
                    v_n[i] = ( v_n[i]*q_loc_0[i] + flux_0[i] ) / Q_0[i];
                    flux_1[i] += Q_1[i]*v_n[i] + p[i]*face_n_0[i]; 
                    flux_2[i] += Q_2[i]*v_n[i] + p[i]*face_n_1[i]; 
                    flux_3[i] += Q_3[i]*v_n[i] + p[i]*face_n_2[i];
                    flux_4[i] += (Q_4[i] + p[i] + peff)*v_n[i];

                    flux_0[i] -= eig[i]*DQ_loc_0[i]; flux_1[i] -= eig[i]*DQ_loc_1[i]; flux_2[i] -= eig[i]*DQ_loc_2[i];
                    flux_3[i] -= eig[i]*DQ_loc_3[i]; flux_4[i] -= eig[i]*DQ_loc_4[i];
                    if(!steady){
                        flux_0[i] -= vgn_tmp[i]*DQ_loc_0[i]; flux_1[i] -= vgn_tmp[i]*DQ_loc_1[i]; flux_2[i] -= vgn_tmp[i]*DQ_loc_2[i];
                        flux_3[i] -= vgn_tmp[i]*DQ_loc_3[i]; flux_4[i] -= vgn_tmp[i]*DQ_loc_4[i];
                    }

                    if(vis_run){
                        dist[i] = norm_dist_c2c[face[i]];
                        visc[i] = vis_l[c2[i]] + vis_t[c2[i]];
                        tmp[i]  = 2.0*visc[i]/(q_loc_0[i]*dist[i] + TINY);
                        flux_0[i] -= tmp[i]*DQ_loc_0[i]; flux_1[i] -= tmp[i]*DQ_loc_1[i]; flux_2[i] -= tmp[i]*DQ_loc_2[i]; 
                        flux_3[i] -= tmp[i]*DQ_loc_3[i]; flux_4[i] -= tmp[i]*DQ_loc_4[i];
                    }
                    tmp[i] = 0.5*area[face[i]];
                    DQ_0[i] -= tmp[i]*flux_0[i]; DQ_1[i] -= tmp[i]*flux_1[i]; DQ_2[i] -= tmp[i]*flux_2[i]; 
                    DQ_3[i] -= tmp[i]*flux_3[i]; DQ_4[i] -= tmp[i]*flux_4[i]; //闈炶繛缁�鍐欏�?            
                }
                DQ_0[i] /= Diag_cell[i]; DQ_1[i] /= Diag_cell[i]; DQ_2[i] /= Diag_cell[i]; 
                DQ_3[i] /= Diag_cell[i]; DQ_4[i] /= Diag_cell[i];

                mflog::log.set_all_processors_out();
                if(fabs(DQ_0[i])>1.0e3*rho00){
                    mflog::log << "Forward sweep: drho>1.0e3*rho00!  " << IOS_EP(2) << DQ_0[i] << IOS_SEP
                               << cell[i] << IOS_SEP << mflog::log.rank_id() << endl;
                }
                if(fabs(DQ_4[i])>1.0e5*e_stag){
                    mflog::log << "Forward sweep: de>1.0e5*e_stag!  " << IOS_EP(2) << DQ_4[i] << IOS_SEP
                               << cell[i] << IOS_SEP << mflog::log.rank_id() << endl;
                }
                if(fabs(DQ_0[i])>rho_max || DQ_4[i]>e_stag_max){
                    printf("Error!\nForward sweep: Maybe CFL too big or entropy correction coefficient too small!");
                    mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
                }
            }

//            #pragma omp simd safelen(VEC)//safelen(VEC)
            for(IntType i=0; i<VEC; i++){
                if(DQ_limit == 1){
                    // do nothing!
                }else if(DQ_limit == 2){
                    //RealFlow dp[VEC],vv[VEC]; 
                    vv[i]   = q_1[i]*q_1[i] + q_2[i]*q_2[i] + q_3[i]*q_3[i];
                    dp[i]   = DQ_4[i]+0.5*DQ_0[i]*vv[i] - (DQ_1[i]*q_1[i] + DQ_2[i]*q_2[i] + DQ_3[i]*q_3[i]);
                    dp[i]  *= gamm1; 
                    if( (q_0[i] + DQ_0[i])<rho_min || (q_0[i]+DQ_0[i])>rho_max ||
                        (q_4[i]+dp[i])    <p_min   || (q_4[i]+dp[i])  >p_max ){
                        DQ_0[i] *=0.1; DQ_1[i] *=0.1; DQ_2[i] *=0.1; DQ_3[i] *=0.1; DQ_4[i] *=0.1;
                    }
                    dp[i]   = DQ_4[i]+0.5*DQ_0[i]*vv[i] - (DQ_1[i]*q_1[i] + DQ_2[i]*q_2[i] + DQ_3[i]*q_3[i]);
                    dp[i]  *= gamm1;
                    if( (q_0[i] + DQ_0[i])<rho_min || (q_0[i]+DQ_0[i])>rho_max ||
                        (q_4[i]+dp[i])    <p_min   || (q_4[i]+dp[i])  >p_max ){
                        DQ_0[i] *=0.1; DQ_1[i] *=0.1; DQ_2[i] *=0.1; DQ_3[i] *=0.1; DQ_4[i] *=0.1;
                    }
                    dp[i]   = DQ_4[i]+0.5*DQ_0[i]*vv[i] - (DQ_1[i]*q_1[i] + DQ_2[i]*q_2[i] + DQ_3[i]*q_3[i]);
                    dp[i]  *= gamm1;
                    if( (q_0[i] + DQ_0[i])<rho_min || (q_0[i]+DQ_0[i])>rho_max ||
                        (q_4[i] + dp[i])  <p_min   || (q_4[i] + dp[i])>p_max ){
                        DQ_0[i] =0.0; DQ_1[i] =0.0; DQ_2[i] =0.0; DQ_3[i] =0.0; DQ_4[i] =0.0;
                    }
                }else if(DQ_limit == 3){
                    DQ_0[i] = MAX( DQ_0[i] , rho_min-q_0[i] );
                    DQ_0[i] = MIN( DQ_0[i] , rho_max-q_0[i] );
                }else if(DQ_limit == 4){
                    vv[i]   = q_1[i]*q_1[i] + q_2[i]*q_2[i] + q_3[i]*q_3[i];
                    rhoe[i] = 0.5*q_0[i]*vv[i] + (q_4[i]+p_bar)/(gam-1.0);
                    dp[i]   = DQ_4[i]+0.5*DQ_0[i]*vv[i] - (DQ_1[i]*q_1[i] + DQ_2[i]*q_2[i] + DQ_3[i]*q_3[i]);
                    dp[i]  *= gamm1; 

                    alph_rho[i]  = q_0[i] /(MAX(q_0[i],0.05*rho00)+MAX(0.0,-DQ_0[i])  );
                    alph_rhoe[i] = rhoe[i]/(MAX(rhoe[i],0.05*e_stag)+MAX(0.0,-DQ_4[i]));
                    alph_p[i]    = (q_4[i]+p_bar) / (MAX( (q_4[i]+p_bar) , 0.05*p_bar) + MAX(0.0,-dp[i]));
                    alph[i]      = MIN(alph_rho[i] , alph_rhoe[i]);
                    alph[i]      = MIN(alph[i] , alph_p[i]);
                    DQ_0[i] *=alph[i]; DQ_1[i] *=alph[i]; DQ_2[i] *=alph[i]; DQ_3[i] *=alph[i]; DQ_4[i] *=alph[i];
                }else{
                    mflog::log.set_one_processor_out();
                    mflog::log << endl<<"DQ_limit is greater to 4! Now only have 4 methods."<<endl;
                    mflog::log << "Then we will use the first method, i.e. do nothing!"<<endl;
                }
                DQ[0][cell[i]] = DQ_0[i]; DQ[1][cell[i]] = DQ_1[i]; DQ[2][cell[i]] = DQ_2[i]; DQ[3][cell[i]] = DQ_3[i]; DQ[4][cell[i]] = DQ_4[i];
            }
        }

        if(yushu != 0){
            IntType face, cell, c1, c2;
            for(ilu = end;ilu<end+yushu;ilu++){
                cell = luorder[ilu];
                IntType c_tmp, count;
                RealFlow flux[5], q_loc[5], DQ_loc[5], visc, tmp, vgn_tmp;
                RealGeom face_n[3], dist;
                for(IntType j=0; j<nFPC[cell]; j++){
                    face  = C2F[cell][j];
                    count = face + face;
                    c1    = f2c[count++];
                    c2    = f2c[count];
                    // One of c1 and c2 must be cell itself. 
                    if(layer[c1]>layer[cell] || layer[c2]>layer[cell]) {continue;}

                    // Now its neighboring cell belongs to lower triangular
                    face_n[0] = xfn[face];
                    face_n[1] = yfn[face];
                    face_n[2] = zfn[face];
                    if(!steady) vgn_tmp = vgn[face];
                    if(c2 == cell){
                        c_tmp = c1;
                        c1    = c2;
                        c2    = c_tmp;
                        face_n[0] = -face_n[0];
                        face_n[1] = -face_n[1];
                        face_n[2] = -face_n[2];
                        if(!steady) vgn_tmp = -vgn[face];
                    }
                    assert(c1 == cell);
            
                    for(IntType i=0; i<5; i++){
                        q_loc[i]  = q[i][c2];
                        DQ_loc[i] = DQ[i][c2];
                    }
                    // Calculate everything (I call it Flux) in lower triangular
                    if(steady){
                        FluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
                    }else{
                        FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
                    }
            
                    if(vis_run){
                        dist = norm_dist_c2c[face];
                        visc = vis_l[c2] + vis_t[c2];
                        tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
                        for(IntType i=0; i<5; i++) flux[i] -= tmp*DQ_loc[i];
                    }

                    // Add Flux together
                    tmp = 0.5*area[face];
                    for(IntType i=0; i<5; i++) DQ[i][cell] -= tmp*flux[i];
                }
                for(IntType i=0; i<5; i++) DQ[i][cell] /= Diag[cell];
       
                mflog::log.set_all_processors_out();
                if(fabs(DQ[0][cell])>1.0e3*rho00){
                    mflog::log << "Forward sweep: drho>1.0e3*rho00!  " << IOS_EP(2) << DQ[0][cell] << IOS_SEP
                               << cell << IOS_SEP << mflog::log.rank_id() << endl;
                }
                if(fabs(DQ[4][cell])>1.0e5*e_stag){
                    mflog::log << "Forward sweep: de>1.0e5*e_stag!  " << IOS_EP(2) << DQ[4][cell] << IOS_SEP
                               << cell << IOS_SEP << mflog::log.rank_id() << endl;
                }
                if(fabs(DQ[0][cell])>rho_max || DQ[4][cell]>e_stag_max){
                    printf("Error!\nForward sweep: Maybe CFL too big or entropy correction coefficient too small!");
                    mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
                }

                //limit for rho>0
                if(DQ_limit == 1){
                    // do nothing!
                }else if(DQ_limit == 2){
                    RealFlow dp,vv; 
                    vv   = q[1][cell]*q[1][cell]+q[2][cell]*q[2][cell]+q[3][cell]*q[3][cell];
                    dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
                    dp  *= gamm1; 
                    if((q[0][cell]+DQ[0][cell])<rho_min || (q[0][cell]+DQ[0][cell])>rho_max ||
                       (q[4][cell]+dp)<p_min || (q[4][cell]+dp)>p_max){
                        DQ[0][cell] *=0.1;
                        DQ[1][cell] *=0.1;
                        DQ[2][cell] *=0.1;
                        DQ[3][cell] *=0.1;
                        DQ[4][cell] *=0.1;
                    }
                    dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
                    dp  *= gamm1;
                    if((q[0][cell]+DQ[0][cell])<rho_min || (q[0][cell]+DQ[0][cell])>rho_max ||
                        (q[4][cell]+dp)<p_min || (q[4][cell]+dp)>p_max){
                        DQ[0][cell] *=0.1;
                        DQ[1][cell] *=0.1;
                        DQ[2][cell] *=0.1;
                        DQ[3][cell] *=0.1;
                        DQ[4][cell] *=0.1;
                    }
                    dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
                    dp  *= gamm1;
                    if((q[0][cell]+DQ[0][cell])<rho_min || (q[0][cell]+DQ[0][cell])>rho_max ||
                        (q[4][cell]+dp)<p_min || (q[4][cell]+dp)>p_max){
                        DQ[0][cell] =0.0;
                        DQ[1][cell] =0.0;
                        DQ[2][cell] =0.0;
                        DQ[3][cell] =0.0;
                        DQ[4][cell] =0.0;
                    }    
                }else if(DQ_limit == 3){
                    DQ[0][cell] = MAX(DQ[0][cell],rho_min-q[0][cell]);
                    DQ[0][cell] = MIN(DQ[0][cell], rho_max-q[0][cell]);     
                }else if(DQ_limit == 4){
                    RealFlow alph,alph_rho,alph_rhoe,alph_p,dp,vv,rhoe;
                    vv   = q[1][cell]*q[1][cell]+q[2][cell]*q[2][cell]+q[3][cell]*q[3][cell];
                    rhoe = 0.5*q[0][cell]*vv+(q[4][cell]+p_bar)/(gam-1.0);
                    dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
                    dp  *= gamm1; 

                    alph_rho  = q[0][cell]/(MAX(q[0][cell],0.05*rho00)+MAX(0.0,-DQ[0][cell]));
                    alph_rhoe = rhoe/(MAX(rhoe,0.05*e_stag)+MAX(0.0,-DQ[4][cell]));
                    alph_p    = (q[4][cell]+p_bar)/(MAX((q[4][cell]+p_bar),0.05*p_bar)+MAX(0.0,-dp));
                    alph      = MIN(alph_rho,alph_rhoe);
                    alph      = MIN(alph,alph_p);
                    for(IntType i=0;i<5;i++) DQ[i][cell] *= alph;
                }else{
                    mflog::log.set_one_processor_out();
                    mflog::log << endl<<"DQ_limit is greater to 4! Now only have 4 methods."<<endl;
                    mflog::log << "Then we will use the first method, i.e. do nothing!"<<endl;
                }
            }
        }
    }         
#else
*/    
#if (defined FS_OPENMP) && (defined CellColoring) 
//not containing SIMD
    IntType laynum;
    IntType start, end;
    for(laynum=0; laynum<cellsPerlayer[0]; laynum++ ){
        start = cellsPerlayer[laynum+1];
        end   = cellsPerlayer[laynum+2];
        if(laynum == 0) {start++;}        
#pragma omp parallel for private(ilu)
    for(ilu=start; ilu<end; ilu++){
        IntType cell;
        cell = luorder[ilu];
#else
    for(ilu=1;ilu<nTCell;ilu++){
        cell = luorder[ilu];
#endif    
        for(IntType j=0; j<nFPC[cell]; j++){
            IntType   face, c1, c2, c_tmp, count;
            RealFlow  flux_s[5], flux[5], q_loc[5], DQ_loc[5], visc, tmp, vgn_tmp;
            RealGeom  face_n[3], dist;
            face  = C2F[cell][j];
            count = face + face;
            c1    = f2c[count++];
            c2    = f2c[count];
            // One of c1 and c2 must be cell itself. 
            if(layer[c1]>layer[cell] || layer[c2]>layer[cell]) continue;

            // Now its neighboring cell belongs to lower triangular
            face_n[0] = xfn[face];
            face_n[1] = yfn[face];
            face_n[2] = zfn[face];
            if(!steady) vgn_tmp = vgn[face];
            if(c2 == cell){
                c_tmp = c1;
                c1    = c2;
                c2    = c_tmp;
                face_n[0] = -face_n[0];
                face_n[1] = -face_n[1];
                face_n[2] = -face_n[2];
                if(!steady) vgn_tmp = -vgn[face];
            }
            assert(c1 == cell);
            
            for(IntType i=0; i<5; i++){
                q_loc[i]  = q[i][c2];
                DQ_loc[i] = DQ[i][c2];
            }
            // Calculate everything (I call it Flux) in lower triangular
            if(steady){
                FluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
            }else{
                FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
            }
            
            if(vis_run){
                dist = norm_dist_c2c[face];
                visc = vis_l[c2] + vis_t[c2];
                tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
                for(IntType i=0; i<5; i++) flux[i] -= tmp*DQ_loc[i];
            }

            // Add Flux together
            tmp = 0.5*area[face];
            for(IntType i=0; i<5; i++) DQ[i][cell] -= tmp*flux[i];
        }
        for(IntType i=0; i<5; i++) DQ[i][cell] /= Diag[cell];
        
       
        mflog::log.set_all_processors_out();
        if(fabs(DQ[0][cell])>1.0e3*rho00){
            mflog::log << "Forward sweep: drho>1.0e3*rho00!  " << IOS_EP(2) << DQ[0][cell] << IOS_SEP
                       << cell << IOS_SEP << mflog::log.rank_id() << endl;
        }
        if(fabs(DQ[4][cell])>1.0e5*e_stag){
            mflog::log << "Forward sweep: de>1.0e5*e_stag!  " << IOS_EP(2) << DQ[4][cell] << IOS_SEP
                       << cell << IOS_SEP << mflog::log.rank_id() << endl;
        }
        if(fabs(DQ[0][cell])>rho_max || DQ[4][cell]>e_stag_max){
            printf("Error!\n Maybe CFL too big or entropy correction coefficient too small!");
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
        
    
        //limit for rho>0
        if(DQ_limit == 1){
            // do nothing!
        }else if(DQ_limit == 2){  
            RealFlow dp,vv; 
            vv   = q[1][cell]*q[1][cell]+q[2][cell]*q[2][cell]+q[3][cell]*q[3][cell];
            dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
            dp  *= gamm1; 
            if((q[0][cell]+DQ[0][cell])<rho_min || (q[0][cell]+DQ[0][cell])>rho_max ||
               (q[4][cell]+dp)<p_min || (q[4][cell]+dp)>p_max){
                DQ[0][cell] *=0.1;
                DQ[1][cell] *=0.1;
                DQ[2][cell] *=0.1;
                DQ[3][cell] *=0.1;
                DQ[4][cell] *=0.1;
            }
            dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
            dp  *= gamm1;
            if((q[0][cell]+DQ[0][cell])<rho_min || (q[0][cell]+DQ[0][cell])>rho_max ||
               (q[4][cell]+dp)<p_min || (q[4][cell]+dp)>p_max){
                DQ[0][cell] *=0.1;
                DQ[1][cell] *=0.1;
                DQ[2][cell] *=0.1;
                DQ[3][cell] *=0.1;
                DQ[4][cell] *=0.1;
            }
            dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
            dp  *= gamm1;
            if((q[0][cell]+DQ[0][cell])<rho_min || (q[0][cell]+DQ[0][cell])>rho_max ||
               (q[4][cell]+dp)<p_min || (q[4][cell]+dp)>p_max){
                DQ[0][cell] =0.0;
                DQ[1][cell] =0.0;
                DQ[2][cell] =0.0;
                DQ[3][cell] =0.0;
                DQ[4][cell] =0.0;
            }    
        }else if(DQ_limit == 3){
            DQ[0][cell] = MAX(DQ[0][cell],rho_min-q[0][cell]);
            DQ[0][cell] = MIN(DQ[0][cell], rho_max-q[0][cell]);     
        }else if(DQ_limit == 4){
            RealFlow alph,alph_rho,alph_rhoe,alph_p,dp,vv,rhoe;
            vv   = q[1][cell]*q[1][cell]+q[2][cell]*q[2][cell]+q[3][cell]*q[3][cell];
            rhoe = 0.5*q[0][cell]*vv+(q[4][cell]+p_bar)/(gam-1.0);
            dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
            dp  *= gamm1; 

            alph_rho  = q[0][cell]/(MAX(q[0][cell],0.05*rho00)+MAX(0.0,-DQ[0][cell]));
            alph_rhoe = rhoe/(MAX(rhoe,0.05*e_stag)+MAX(0.0,-DQ[4][cell]));
            alph_p    = (q[4][cell]+p_bar)/(MAX((q[4][cell]+p_bar),0.05*p_bar)+MAX(0.0,-dp));
            alph      = MIN(alph_rho,alph_rhoe);
            alph      = MIN(alph,alph_p);
            for(IntType i=0;i<5;i++) DQ[i][cell] *= alph;
        }else{
            mflog::log.set_one_processor_out();
            mflog::log << endl<<"DQ_limit is greater to 4! Now only have 4 methods."<<endl;
            mflog::log << "Then we will use the first method, i.e. do nothing!"<<endl;
        }   
    }
#if (defined FS_OPENMP) && (defined CellColoring) 
    }
#endif
//#endif


#ifdef MPICH
    IntType nvar = 5;
    RealFlow *q_mpi[5];
    for(IntType j=0; j<5; j++)
        q_mpi[j] = DQ[j];
    grid->RecvSendVarNeighbor_Togeth(nvar, q_mpi);
#endif

    // Backward Sweep 
/*
#ifdef FS_SIMD 
//containing SIMD    
    for( laynum=cellsPerlayer[0]-1; laynum>=0; laynum-- ){
        start = cellsPerlayer[laynum+2];
        end   = cellsPerlayer[laynum+1];
        IntType yushu,ilu;
        yushu = (start-end) % VEC;
        end += yushu;
#ifdef FS_OPENMP
#pragma omp parallel for private(ilu)
#endif
    for(ilu=start-1; ilu>=end; ilu-=VEC){
        IntType c_tmp[VEC], count[VEC], face[VEC], cell[VEC], c1[VEC], c2[VEC];
        RealFlow flux_s_0[VEC], flux_s_1[VEC], flux_s_2[VEC], flux_s_3[VEC], flux_s_4[VEC];
        RealFlow flux_0[VEC], flux_1[VEC], flux_2[VEC], flux_3[VEC], flux_4[VEC]; 
        RealFlow q_loc_0[VEC], q_loc_1[VEC], q_loc_2[VEC], q_loc_3[VEC], q_loc_4[VEC];
        RealFlow DQ_loc_0[VEC], DQ_loc_1[VEC], DQ_loc_2[VEC], DQ_loc_3[VEC], DQ_loc_4[VEC];
        RealFlow visc[VEC], tmp[VEC], vgn_tmp[VEC], Diag_cell[VEC];
        RealGeom face_n_0[VEC], face_n_1[VEC], face_n_2[VEC], dist[VEC];
        RealFlow DQ_0[VEC], DQ_1[VEC], DQ_2[VEC], DQ_3[VEC], DQ_4[VEC], q_0[VEC], q_1[VEC], q_2[VEC], q_3[VEC], q_4[VEC];
        RealFlow Q_0[VEC], Q_1[VEC], Q_2[VEC], Q_3[VEC], Q_4[VEC], rv2[VEC], v_n[VEC], p[VEC], eig[VEC];
        RealFlow alph[VEC], alph_rho[VEC], alph_rhoe[VEC], alph_p[VEC], dp[VEC], vv[VEC], rhoe[VEC];

        #pragma omp simd safelen(VEC)
        for(IntType i=0; i<VEC; i++){
            cell[i] = luorder[ilu-i];        //闈炶繛缁�鍙栧潃
            Diag_cell[i] = Diag[cell[i]];
            DQ_0[i] = DQ[0][cell[i]]; DQ_1[i] = DQ[1][cell[i]]; DQ_2[i] = DQ[2][cell[i]]; DQ_3[i] = DQ[3][cell[i]]; DQ_4[i] = DQ[4][cell[i]];  //闈炶繛缁�璁垮瓨鍚戦噺鍖�
            q_0[i] = q[0][cell[i]];   q_1[i] = q[1][cell[i]];   q_2[i] = q[2][cell[i]];   q_3[i] = q[3][cell[i]];   q_4[i] = q[4][cell[i]];
            flux_s_0[i] = 0.; flux_s_1[i] = 0.; flux_s_2[i] = 0.; flux_s_3[i] = 0.; flux_s_4[i] = 0.;
        }

        #pragma omp simd safelen(VEC)
        for(IntType i=0; i<VEC; i++){
            for(IntType j=0; j<nFPC[cell[i]]; j++){
                face[i] = C2F[cell[i]][j];
                count[i] = face[i] + face[i];
                c1[i] = f2c[count[i]++];
                c2[i] = f2c[count[i]];
                if(layer[c1[i]]<layer[cell[i]] || layer[c2[i]]<layer[cell[i]]) { continue;}

                face_n_0[i] = xfn[face[i]];  //闈炶繛缁�鍙栧潃
                face_n_1[i] = yfn[face[i]];
                face_n_2[i] = zfn[face[i]];
                if(!steady) vgn_tmp[i] = vgn[face[i]];
                if(c2[i] == cell[i]){
                    c_tmp[i] = c1[i];
                    c1[i]    = c2[i];
                    c2[i]    = c_tmp[i];
                    face_n_0[i] = -face_n_0[i];
                    face_n_1[i] = -face_n_1[i];
                    face_n_2[i] = -face_n_2[i];
                    if(!steady) vgn_tmp[i] = -vgn[face[i]];
                }
                assert(c1[i] == cell[i]);

                q_loc_0[i]   = q[0][c2[i]];  q_loc_1[i]  = q[1][c2[i]];  
                q_loc_2[i]   = q[2][c2[i]];  q_loc_3[i]  = q[3][c2[i]];   q_loc_4[i]  = q[4][c2[i]]; // DQ鍜宷闈炶繛缁�鍙栧潃
                DQ_loc_0[i]  = DQ[0][c2[i]]; DQ_loc_1[i] = DQ[1][c2[i]];
                DQ_loc_2[i]  = DQ[2][c2[i]]; DQ_loc_3[i] = DQ[3][c2[i]]; DQ_loc_4[i] = DQ[4][c2[i]];

                //FluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
                //FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
                Q_0[i] = q_loc_0[i]; 
                Q_1[i] = q_loc_0[i]*q_loc_1[i]; 
                Q_2[i] = q_loc_0[i]*q_loc_2[i]; 
                Q_3[i] = q_loc_0[i]*q_loc_3[i];
                rv2[i] = 0.5*q_loc_0[i] * (q_loc_1[i]*q_loc_1[i] + q_loc_2[i]*q_loc_2[i] + q_loc_3[i]*q_loc_3[i]);
                p[i] = q_loc_4[i];
                Q_4[i] = p[i] / gamm1 + rv2[i];

                v_n[i] = q_loc_1[i]*face_n_0[i] + q_loc_2[i]*face_n_1[i] + q_loc_3[i]*face_n_2[i];
                if(steady){
                    eig[i] = fabs(v_n[i]) + sqrt( gam*(p[i]+p_bar) / q_loc_0[i] );
                }
                else{
                    eig[i] = fabs(v_n[i] - vgn_tmp[i]) + sqrt( gam*(p[i]+p_bar) / q_loc_0[i] );
                }
                eig[i] *= lhs_omga;

                flux_1[i] = -Q_1[i]*v_n[i] - p[i]*face_n_0[i]; 
                flux_2[i] = -Q_2[i]*v_n[i] - p[i]*face_n_1[i]; 
                flux_3[i] = -Q_3[i]*v_n[i] - p[i]*face_n_2[i]; 
                flux_4[i] = -(Q_4[i]+p[i]+peff) * v_n[i];

                Q_0[i] +=DQ_loc_0[i]; Q_1[i] +=DQ_loc_1[i]; Q_2[i] +=DQ_loc_2[i]; 
                Q_3[i] +=DQ_loc_3[i]; Q_4[i] +=DQ_loc_4[i];
                rv2[i] = 0.5 * (Q_1[i]*Q_1[i]+Q_2[i]*Q_2[i]+Q_3[i]*Q_3[i]) / Q_0[i];
                p[i] = gamm1*(Q_4[i] - rv2[i]);

                flux_0[i] = DQ_loc_1[i]*face_n_0[i] + DQ_loc_2[i]*face_n_1[i] + DQ_loc_3[i]*face_n_2[i];
                //v_n[i] *= q_loc_0[i]; v_n[i] += flux_0[i]; v_n[i] /= Q_0[i];
                v_n[i] = ( v_n[i]*q_loc_0[i] + flux_0[i] ) / Q_0[i];
                flux_1[i] += Q_1[i]*v_n[i] + p[i]*face_n_0[i]; 
                flux_2[i] += Q_2[i]*v_n[i] + p[i]*face_n_1[i]; 
                flux_3[i] += Q_3[i]*v_n[i] + p[i]*face_n_2[i];
                flux_4[i] += (Q_4[i] + p[i] + peff)*v_n[i];

                flux_0[i] -= eig[i]*DQ_loc_0[i]; flux_1[i] -= eig[i]*DQ_loc_1[i]; flux_2[i] -= eig[i]*DQ_loc_2[i];
                flux_3[i] -= eig[i]*DQ_loc_3[i]; flux_4[i] -= eig[i]*DQ_loc_4[i];
                if(!steady){
                    flux_0[i] -= vgn_tmp[i]*DQ_loc_0[i]; flux_1[i] -= vgn_tmp[i]*DQ_loc_1[i]; flux_2[i] -= vgn_tmp[i]*DQ_loc_2[i];
                    flux_3[i] -= vgn_tmp[i]*DQ_loc_3[i]; flux_4[i] -= vgn_tmp[i]*DQ_loc_4[i];
                }

                if(vis_run){
                    dist[i] = norm_dist_c2c[face[i]];
                    visc[i] = vis_l[c2[i]] + vis_t[c2[i]];
                    tmp[i]  = 2.0*visc[i]/(q_loc_0[i]*dist[i] + TINY);
                    flux_0[i] -= tmp[i]*DQ_loc_0[i]; flux_1[i] -= tmp[i]*DQ_loc_1[i]; flux_2[i] -= tmp[i]*DQ_loc_2[i]; 
                    flux_3[i] -= tmp[i]*DQ_loc_3[i]; flux_4[i] -= tmp[i]*DQ_loc_4[i];
                }
                tmp[i] = area[face[i]];
                flux_s_0[i] += tmp[i]*flux_0[i]; flux_s_1[i] += tmp[i]*flux_1[i]; flux_s_2[i] += tmp[i]*flux_2[i]; 
                flux_s_3[i] += tmp[i]*flux_3[i]; flux_s_4[i] += tmp[i]*flux_4[i]; //闈炶繛缁�鍐欏�?            
            }

            tmp[i] = 2.0*Diag_cell[i];
            DQ_0[i] -= flux_s_0[i]/tmp[i]; DQ_1[i] -= flux_s_1[i]/tmp[i]; DQ_2[i] -= flux_s_2[i]/tmp[i]; 
            DQ_3[i] -= flux_s_3[i]/tmp[i]; DQ_4[i] -= flux_s_4[i]/tmp[i];

            mflog::log.set_all_processors_out();
            if(fabs(DQ_0[i])>1.0e3*rho00){
                mflog::log << "Backward sweep: drho>1.0e3*rho00!  " << IOS_EP(2) << DQ_0[i] << IOS_SEP
                       << cell[i] << IOS_SEP << mflog::log.rank_id() << endl;
            }
            if(fabs(DQ_4[i])>1.0e5*e_stag){
                mflog::log << "Backward sweep: de>1.0e5*e_stag!  " << IOS_EP(2) << DQ_4[i] << IOS_SEP
                       << cell[i] << IOS_SEP << mflog::log.rank_id() << endl;
            }
            if(fabs(DQ_0[i])>rho_max || DQ_4[i]>e_stag_max){
                std::cerr << "Error!" << endl << "Backward sweep: Maybe CFL too big or entropy correction coefficient too small!" << endl;
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }

        #pragma omp simd safelen(VEC)//safelen(VEC)
        for(IntType i=0; i<VEC; i++){
            if(DQ_limit == 1){
            // do nothing!
            }else if(DQ_limit == 2){
                vv[i]   = q_1[i]*q_1[i] + q_2[i]*q_2[i] + q_3[i]*q_3[i];
                dp[i]   = DQ_4[i]+0.5*DQ_0[i]*vv[i] - (DQ_1[i]*q_1[i] + DQ_2[i]*q_2[i] + DQ_3[i]*q_3[i]);
                dp[i]  *= gamm1; 
                if( (q_0[i] + DQ_0[i])<rho_min || (q_0[i]+DQ_0[i])>rho_max ||
                    (q_4[i]+dp[i])    <p_min   || (q_4[i]+dp[i])  >p_max ){
                    DQ_0[i] *=0.1; DQ_1[i] *=0.1; DQ_2[i] *=0.1; DQ_3[i] *=0.1; DQ_4[i] *=0.1;
                }
                dp[i]   = DQ_4[i]+0.5*DQ_0[i]*vv[i] - (DQ_1[i]*q_1[i] + DQ_2[i]*q_2[i] + DQ_3[i]*q_3[i]);
                dp[i]  *= gamm1;
                if( (q_0[i] + DQ_0[i])<rho_min || (q_0[i]+DQ_0[i])>rho_max ||
                    (q_4[i]+dp[i])    <p_min   || (q_4[i]+dp[i])  >p_max ){
                    DQ_0[i] *=0.1; DQ_1[i] *=0.1; DQ_2[i] *=0.1; DQ_3[i] *=0.1; DQ_4[i] *=0.1;
                }
                dp[i]   = DQ_4[i]+0.5*DQ_0[i]*vv[i] - (DQ_1[i]*q_1[i] + DQ_2[i]*q_2[i] + DQ_3[i]*q_3[i]);
                dp[i]  *= gamm1;
                if( (q_0[i] + DQ_0[i])<rho_min || (q_0[i]+DQ_0[i])>rho_max ||
                    (q_4[i] + dp[i])  <p_min   || (q_4[i] + dp[i])>p_max ){
                        DQ_0[i] =0.0; DQ_1[i] =0.0; DQ_2[i] =0.0; DQ_3[i] =0.0; DQ_4[i] =0.0;
                }
            }else if(DQ_limit == 3){
                DQ_0[i] = MAX( DQ_0[i] , rho_min-q_0[i] );
                DQ_0[i] = MIN( DQ_0[i] , rho_max-q_0[i] );
            }else if(DQ_limit == 4){
                vv[i]   = q_1[i]*q_1[i] + q_2[i]*q_2[i] + q_3[i]*q_3[i];
                rhoe[i] = 0.5*q_0[i]*vv[i] + (q_4[i]+p_bar)/(gam-1.0);
                dp[i]   = DQ_4[i]+0.5*DQ_0[i]*vv[i] - (DQ_1[i]*q_1[i] + DQ_2[i]*q_2[i] + DQ_3[i]*q_3[i]);
                dp[i]  *= gamm1; 

                alph_rho[i]  = q_0[i] /(MAX(q_0[i],0.05*rho00)+MAX(0.0,-DQ_0[i])  );
                alph_rhoe[i] = rhoe[i]/(MAX(rhoe[i],0.05*e_stag)+MAX(0.0,-DQ_4[i]));
                alph_p[i]    = (q_4[i]+p_bar) / (MAX( (q_4[i]+p_bar) , 0.05*p_bar) + MAX(0.0,-dp[i]));
                alph[i]      = MIN(alph_rho[i] , alph_rhoe[i]);
                alph[i]      = MIN(alph[i] , alph_p[i]);
                DQ_0[i] *=alph[i]; DQ_1[i] *=alph[i]; DQ_2[i] *=alph[i]; DQ_3[i] *=alph[i]; DQ_4[i] *=alph[i];
            }
            DQ[0][cell[i]] = DQ_0[i]; DQ[1][cell[i]] = DQ_1[i]; DQ[2][cell[i]] = DQ_2[i]; DQ[3][cell[i]] = DQ_3[i]; DQ[4][cell[i]] = DQ_4[i];
        }
    }

    if(yushu != 0){
        IntType face, cell, c1, c2;
        for(ilu = end-1; ilu>=end-yushu; ilu--){
            cell = luorder[ilu];

            IntType c_tmp, count;
            RealFlow flux_s[5], flux[5], q_loc[5], DQ_loc[5], visc, tmp, vgn_tmp;
            RealGeom face_n[3], dist;
            for(IntType i=0; i<5; i++) flux_s[i] = 0.;
            for(IntType j=0; j<nFPC[cell]; j++){
                face  = C2F[cell][j];
                count = face + face;
                c1    = f2c[count++];
                c2    = f2c[count];
                // One of c1 and c2 must be cell itself. 
                if(layer[c1]<layer[cell] || layer[c2]<layer[cell]) {continue;}

                // Now its neighboring cell belongs to upper triangular
                face_n[0] = xfn[face];
                face_n[1] = yfn[face];
                face_n[2] = zfn[face];
                if(!steady) vgn_tmp   = vgn[face];
                if(c2 == cell){
                    c_tmp = c1;
                    c1    = c2;
                    c2    = c_tmp;
                    face_n[0] = -face_n[0];
                    face_n[1] = -face_n[1];
                    face_n[2] = -face_n[2];
                    if(!steady) vgn_tmp   = -vgn[face];
                }
                assert(c1 == cell);
                for(IntType i=0; i<5; i++){
                    q_loc[i]  = q[i][c2];
                    DQ_loc[i] = DQ[i][c2];
                }
                // Calculate everything (I call it Flux) in upper triangular
                if(steady){
                    FluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
                }else{
                    FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
                }
           
                if(vis_run){
                    dist = norm_dist_c2c[face];
                    visc = vis_l[c2] + vis_t[c2];
                    tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
                    for(IntType i=0; i<5; i++) flux[i] -= tmp*DQ_loc[i];
                }

                // Add Flux together
                tmp = area[face];
                for(IntType i=0; i<5; i++) flux_s[i] += tmp*flux[i];
            }
            tmp = 2.0*Diag[cell];
            for(IntType i=0; i<5; i++) DQ[i][cell] -= flux_s[i]/tmp;
        
            mflog::log.set_all_processors_out();
            if(fabs(DQ[0][cell])>1.0e3*rho00){
                mflog::log << "Backward sweep: drho>1.0e3*rho00! " << IOS_EP(2) << DQ[0][cell] << IOS_SEP
                       << cell << IOS_SEP << mflog::log.rank_id() << endl;
            }
            if(fabs(DQ[4][cell])>1.0e5*e_stag){
                mflog::log << "Backward sweep: de>1.0e5*e_stag! " << IOS_EP(2) << DQ[4][cell] << IOS_SEP
                       << cell << IOS_SEP << mflog::log.rank_id() << endl;
            }
            //if(fabs(DQ[0][cell])>1.0e4*rho00 || DQ[4][cell]>1.0e8*e_stag){
            if(fabs(DQ[0][cell])>rho_max || DQ[4][cell]>e_stag_max){
                std::cerr << "Error!" << endl << "Backward sweep: Maybe CFL too big or entropy correction coefficient too small!" << endl;
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        
            //limit for rho>0
            if(DQ_limit == 1){
                // do nothing!
            }else if(DQ_limit == 2){
                RealFlow dp,vv; 
                vv   = q[1][cell]*q[1][cell]+q[2][cell]*q[2][cell]+q[3][cell]*q[3][cell];
                dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
                dp  *= gamm1; 
                if( (q[0][cell]+DQ[0][cell])<rho_min || (q[0][cell]+DQ[0][cell])>rho_max ||
                    (q[4][cell]+dp)<p_min || (q[4][cell]+dp)>p_max){
                    DQ[0][cell] *=0.1;
                    DQ[1][cell] *=0.1;
                    DQ[2][cell] *=0.1;
                    DQ[3][cell] *=0.1;
                    DQ[4][cell] *=0.1;
                }
                dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
                dp  *= gamm1;
                if( (q[0][cell]+DQ[0][cell])<rho_min || (q[0][cell]+DQ[0][cell])>rho_max ||
                    (q[4][cell]+dp)<p_min || (q[4][cell]+dp)>p_max){
                    DQ[0][cell] *=0.1;
                    DQ[1][cell] *=0.1;
                    DQ[2][cell] *=0.1;
                    DQ[3][cell] *=0.1;
                    DQ[4][cell] *=0.1;
                }
                dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
                dp  *= gamm1;
                if( (q[0][cell]+DQ[0][cell])<rho_min || (q[0][cell]+DQ[0][cell])>rho_max ||
                    (q[4][cell]+dp)<p_min || (q[4][cell]+dp)>p_max){
                    DQ[0][cell] =0.0;
                    DQ[1][cell] =0.0;
                    DQ[2][cell] =0.0;
                    DQ[3][cell] =0.0;
                    DQ[4][cell] =0.0;
                }
            }else if(DQ_limit == 3){
                DQ[0][cell] = MAX(DQ[0][cell],rho_min-q[0][cell]);
                DQ[0][cell] = MIN(DQ[0][cell],rho_max-q[0][cell]);
            }else if(DQ_limit == 4){
                RealFlow alph,alph_rho,alph_rhoe,alph_p,dp,vv,rhoe;
                vv   = q[1][cell]*q[1][cell]+q[2][cell]*q[2][cell]+q[3][cell]*q[3][cell];
                rhoe = 0.5*q[0][cell]*vv+(q[4][cell]+p_bar)/(gam-1.0);
                dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
                dp  *= gamm1; 

                alph_rho  = q[0][cell]/(MAX(q[0][cell],0.05*rho00)+MAX(0.0,-DQ[0][cell]));
                alph_rhoe = rhoe/(MAX(rhoe,0.05*e_stag)+MAX(0.0,-DQ[4][cell]));
                alph_p    = (q[4][cell]+p_bar)/(MAX((q[4][cell]+p_bar),0.05*p_bar)+MAX(0.0,-dp));
                alph      = MIN(alph_rho,alph_rhoe);
                alph      = MIN(alph,alph_p);
                for(IntType i=0;i<5;i++) DQ[i][cell] *= alph;
            }
        }
    }
    }

#else
*/
#if (defined FS_OPENMP) && (defined CellColoring) 
//not containing SIMD
    for( laynum=cellsPerlayer[0]-1; laynum>=0; laynum-- ){
        start = cellsPerlayer[laynum+2];
        end   = cellsPerlayer[laynum+1];
        IntType ilu;
#pragma omp parallel for private(ilu, cell)
        for(ilu=start-1; ilu>=end; ilu--){
            cell = luorder[ilu];
            IntType face, c1, c2, c_tmp, count;
            RealFlow flux_s[5], flux[5], q_loc[5], DQ_loc[5], visc, tmp, vgn_tmp;
            RealGeom face_n[3], dist;
            
#else
    for(ilu=nTCell-1;ilu>=0;ilu--){
        cell = luorder[ilu];
#endif        
        for(IntType i=0; i<5; i++) flux_s[i] = 0.;
        for(IntType j=0; j<nFPC[cell]; j++){
            face  = C2F[cell][j];
            count = face + face;
            c1    = f2c[count++];
            c2    = f2c[count];
            // One of c1 and c2 must be cell itself. 
            if(layer[c1]<layer[cell] || layer[c2]<layer[cell]) continue;

            // Now its neighboring cell belongs to upper triangular
            face_n[0] = xfn[face];
            face_n[1] = yfn[face];
            face_n[2] = zfn[face];
            if(!steady) vgn_tmp   = vgn[face];
            if(c2 == cell){
                c_tmp = c1;
                c1    = c2;
                c2    = c_tmp;
                face_n[0] = -face_n[0];
                face_n[1] = -face_n[1];
                face_n[2] = -face_n[2];
                if(!steady) vgn_tmp   = -vgn[face];
            }
            assert(c1 == cell);
            for(IntType i=0; i<5; i++){
                q_loc[i]  = q[i][c2];
                DQ_loc[i] = DQ[i][c2];
            }
            // Calculate everything (I call it Flux) in upper triangular
            if(steady){
                FluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
            }else{
                FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
            }
           

            if(vis_run){
                dist = norm_dist_c2c[face];
                visc = vis_l[c2] + vis_t[c2];
                tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
                for(IntType i=0; i<5; i++) flux[i] -= tmp*DQ_loc[i];
            }

            // Add Flux together
            tmp = area[face];
            for(IntType i=0; i<5; i++) flux_s[i] += tmp*flux[i];
        }
        tmp = 2.0*Diag[cell];
        for(IntType i=0; i<5; i++) DQ[i][cell] -= flux_s[i]/tmp;
        
       
        mflog::log.set_all_processors_out();
        if(fabs(DQ[0][cell])>1.0e3*rho00){
            mflog::log << "Backward sweep: drho>1.0e3*rho00! " << IOS_EP(2) << DQ[0][cell] << IOS_SEP
                       << cell << IOS_SEP << mflog::log.rank_id() << endl;
        }
        if(fabs(DQ[4][cell])>1.0e5*e_stag){
            mflog::log << "Backward sweep: de>1.0e5*e_stag! " << IOS_EP(2) << DQ[4][cell] << IOS_SEP
                       << cell << IOS_SEP << mflog::log.rank_id() << endl;
        }
        //if(fabs(DQ[0][cell])>1.0e4*rho00 || DQ[4][cell]>1.0e8*e_stag){
        if(fabs(DQ[0][cell])>rho_max || DQ[4][cell]>e_stag_max){
            std::cerr << "Error!" << endl << "Maybe CFL too big or entropy correction coefficient too small!" << endl;
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
        

        //limit for rho>0
        if(DQ_limit == 1){
            // do nothing!
        }else if(DQ_limit == 2){  
            RealFlow dp,vv; 
            vv   = q[1][cell]*q[1][cell]+q[2][cell]*q[2][cell]+q[3][cell]*q[3][cell];
            dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
            dp  *= gamm1; 
            if((q[0][cell]+DQ[0][cell])<rho_min || (q[0][cell]+DQ[0][cell])>rho_max ||
               (q[4][cell]+dp)<p_min || (q[4][cell]+dp)>p_max){
                DQ[0][cell] *=0.1;
                DQ[1][cell] *=0.1;
                DQ[2][cell] *=0.1;
                DQ[3][cell] *=0.1;
                DQ[4][cell] *=0.1;
            }
            dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
            dp  *= gamm1;
            if((q[0][cell]+DQ[0][cell])<rho_min || (q[0][cell]+DQ[0][cell])>rho_max ||
               (q[4][cell]+dp)<p_min || (q[4][cell]+dp)>p_max){
                DQ[0][cell] *=0.1;
                DQ[1][cell] *=0.1;
                DQ[2][cell] *=0.1;
                DQ[3][cell] *=0.1;
                DQ[4][cell] *=0.1;
            }
            dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
            dp  *= gamm1;
            if((q[0][cell]+DQ[0][cell])<rho_min || (q[0][cell]+DQ[0][cell])>rho_max ||
               (q[4][cell]+dp)<p_min || (q[4][cell]+dp)>p_max){
                DQ[0][cell] =0.0;
                DQ[1][cell] =0.0;
                DQ[2][cell] =0.0;
                DQ[3][cell] =0.0;
                DQ[4][cell] =0.0;
            }
            }else if(DQ_limit == 3){
                DQ[0][cell] = MAX(DQ[0][cell],rho_min-q[0][cell]);
                DQ[0][cell] = MIN(DQ[0][cell],rho_max-q[0][cell]);
            }else if(DQ_limit == 4){
                RealFlow alph,alph_rho,alph_rhoe,alph_p,dp,vv,rhoe;
                vv   = q[1][cell]*q[1][cell]+q[2][cell]*q[2][cell]+q[3][cell]*q[3][cell];
                rhoe = 0.5*q[0][cell]*vv+(q[4][cell]+p_bar)/(gam-1.0);
                dp   = DQ[4][cell]+0.5*DQ[0][cell]*vv-(DQ[1][cell]*q[1][cell]+DQ[2][cell]*q[2][cell]+DQ[3][cell]*q[3][cell]);
                dp  *= gamm1; 

                alph_rho  = q[0][cell]/(MAX(q[0][cell],0.05*rho00)+MAX(0.0,-DQ[0][cell]));
                alph_rhoe = rhoe/(MAX(rhoe,0.05*e_stag)+MAX(0.0,-DQ[4][cell]));
                alph_p    = (q[4][cell]+p_bar)/(MAX((q[4][cell]+p_bar),0.05*p_bar)+MAX(0.0,-dp));
                alph      = MIN(alph_rho,alph_rhoe);
                alph      = MIN(alph,alph_p);
                for(IntType i=0;i<5;i++) DQ[i][cell] *= alph;
            }
    }
#if (defined FS_OPENMP) && (defined CellColoring) 
    }
#endif
//#endif

    

}


/*******************************************************************************\
 Solve linear systems using the LU-SGS in 3D ~~~ORIGINAL LU-SGS ONE SWEEP~~~
\*******************************************************************************/
void SolveADU3D(PolyGrid *grid, RealFlow **rhs, RealFlow *DQ[5], IntType *nFPC, IntType **C2F, IntType level)
{
    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  n      = nTCell + nBFace;
    IntType  *f2c   = grid->Getf2c();
    // Get grid metrics
    RealGeom *xfn = grid->GetXfn();
    RealGeom *yfn = grid->GetYfn();
    RealGeom *zfn = grid->GetZfn();
    RealGeom *area= grid->GetFaceArea();
    RealGeom *vgn = grid->GetFaceNormalVelocity();//lihuan
    // Get flow variables
    RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    RealFlow p_bar, gam, lhs_omga;
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    grid->GetData(&lhs_omga,   REAL_FLOW, 1, "lhs_omga");
    IntType steady=1;
    grid->GetData(&steady, INT, 1, "steady");
    IntType vis_mode, vis_run = 0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    if(vis_mode != INVISCID) 
    {
        vis_run = 1;
       
        // if coarse grid doesn't want to run the viscous flux, turn it off
        if(level != 0) {
            IntType cg_vis = 1;
            grid->GetData(&cg_vis, INT, 1, "cg_vis");
            if(cg_vis == 0) vis_run = 0;
        }
    }
    RealFlow *vis_l = NULL, *vis_t = NULL;
    if(vis_run) {
        vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
        vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    }
    // Some temporary variables
    IntType  i, j, face, cell, c1, c2, c_tmp, count;
    RealFlow flux[5], q_loc[5], DQ_loc[5], visc, tmp;
    RealGeom face_n[3], dist;
    
    IntType nTFace = grid->GetNTFace();
    RealGeom *norm_dist_c2c = NULL;
    norm_dist_c2c = (RealGeom *)grid->GetDataPtr(REAL_GEOM, nTFace, "norm_dist_c2c");
    assert(norm_dist_c2c);  //must exist
    
    // Now the Forward Sweep
    for(cell=0; cell<nTCell; cell++){
        for(j=0; j<nFPC[cell]; j++)
        {
            face  = C2F[cell][j];
            count = 2*face;
            c1    = f2c[count++];
            c2    = f2c[count];

            // Now its neighboring cell belongs to lower triangular
            face_n[0] = xfn[face];
            face_n[1] = yfn[face];
            face_n[2] = zfn[face];
            if(c2 == cell)
            {
                c_tmp = c1;
                c1    = c2;
                c2    = c_tmp;
                face_n[0] = -face_n[0];
                face_n[1] = -face_n[1];
                face_n[2] = -face_n[2];
            }
            assert(c1 == cell);

            for(i=0; i<5; i++) {
                q_loc[i]  = q[i][c2];
                DQ_loc[i] = DQ[i][c2];
            }

            // Calculate everything (I call it Flux) in lower triangular
            if(steady){
                FluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
            }else{
                FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn[face]);
            }
            

            if(vis_run)
            {
                dist = norm_dist_c2c[face];
                visc = vis_l[c2] + vis_t[c2];
                
                tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
                for(i=0; i<5; i++) flux[i] -= tmp*DQ_loc[i];
            }
        
            // Add Flux together
            tmp = 0.5*area[face];
            for(i=0; i<5; i++) rhs[i][cell] += tmp*flux[i];
        }
    }
}


/*******************************************************************************\
 Solve linear systems using the LU-SGS in 3D ~~~ORIGINAL LU-SGS ONE SWEEP~~~
\*******************************************************************************/
void SolveADU3D2(PolyGrid *grid, RealFlow **rhs, RealFlow *DQ[5], IntType *nFPC, IntType **C2F, IntType level)
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    IntType *f2c   = grid->Getf2c();
    // Get grid metrics
    RealGeom *xfn = grid->GetXfn();
    RealGeom *yfn = grid->GetYfn();
    RealGeom *zfn = grid->GetZfn();
    RealGeom *area= grid->GetFaceArea();
    RealGeom *vgn = grid->GetFaceNormalVelocity();//lihuan
    // Get flow variables
    RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");

    RealFlow p_bar, gam, lhs_omga, steady;
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    grid->GetData(&lhs_omga,   REAL_FLOW, 1, "lhs_omga");
    grid->GetData(&steady, INT, 1, "steady");
    IntType vis_mode, vis_run = 0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    if(vis_mode != INVISCID) 
    {
      vis_run = 1;
  
      // if coarse grid doesn't want to run the viscous flux, turn it off
      if(level != 0) {
        IntType cg_vis = 1;
        grid->GetData(&cg_vis, INT, 1, "cg_vis");
        if(cg_vis == 0) vis_run = 0;
      }
    }
    RealFlow  *vis_l = NULL, *vis_t = NULL;
    if(vis_run) {
      vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
      vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    }
    // Some temporary variables
    IntType  i, j, face, cell, c1, c2, c_tmp, count;
    RealFlow flux[5], q_loc[5], DQ_loc[5], visc, tmp;
    RealGeom face_n[3], dist;
    
    IntType nTFace = grid->GetNTFace();
    RealGeom *norm_dist_c2c = NULL;
    norm_dist_c2c = (RealGeom *)grid->GetDataPtr(REAL_GEOM, nTFace, "norm_dist_c2c");
    assert(norm_dist_c2c);  //must exist
    
    // Now the Forward Sweep
    for(cell=0; cell<nTCell; cell++){
        for(j=0; j<nFPC[cell]; j++)
        {
            face  = C2F[cell][j];
            count = 2*face;
            c1    = f2c[count++];
            c2    = f2c[count];

            // Now its neighboring cell belongs to lower triangular
            face_n[0] = xfn[face];
            face_n[1] = yfn[face];
            face_n[2] = zfn[face];
            if(c2 == cell)
            {
                c_tmp = c1;
                c1    = c2;
                c2    = c_tmp;
                face_n[0] = -face_n[0];
                face_n[1] = -face_n[1];
                face_n[2] = -face_n[2];
            }
            assert(c1 == cell);

            for(i=0; i<5; i++) {
                q_loc[i]  = q[i][c2];
                DQ_loc[i] = DQ[i][c2];
            }

            // Calculate everything (I call it Flux) in lower triangular
            if(steady){
                FluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
            }else{
                FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn[face]);
            }
            

            if(vis_run)
            {
                dist = norm_dist_c2c[face];
                visc = vis_l[c2] + vis_t[c2];
                
                tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
                for(i=0; i<5; i++) flux[i] -= tmp*DQ_loc[i];
            }
        
            // Add Flux together
            tmp = 0.5*area[face];
            for(i=0; i<5; i++) rhs[i][cell] += tmp*flux[i];
        }
    }
  
#ifdef MPICH
    IntType nvar = 5;
    RealFlow *q_mpi[5];
    for(IntType j=0; j<5; j++)
        q_mpi[j] = DQ[j];
    grid->RecvSendVarNeighbor_Togeth(nvar, q_mpi);
#endif
}


/*******************************************************************************\
 Solve linear systems using the LU-SGS in 3D ~~~ORIGINAL LU-SGS MANY SWEEPS~~~~
\*******************************************************************************/
void SolveLUSGS3D(PolyGrid *grid, RealFlow *Diag, RealFlow *DQ[5],
                  RealFlow *rhs[5], IntType *nFPC, IntType **C2F, IntType Nsweep, 
                  RealFlow epsilon, IntType level)
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    IntType *f2c   = grid->Getf2c();
    // Get grid metrics
    RealGeom *xfn  = grid->GetXfn();
    RealGeom *yfn  = grid->GetYfn();
    RealGeom *zfn  = grid->GetZfn();
    RealGeom *area = grid->GetFaceArea();
    RealGeom *vgn  = grid->GetFaceNormalVelocity();
    // Get flow variables
    RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    
    IntType steady=1;
    grid->GetData(&steady,  INT, 1, "steady");
    RealFlow gam, p_bar, lhs_omga;
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&lhs_omga,   REAL_FLOW, 1, "lhs_omga");
    
    IntType vis_mode, vis_run = 0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");

    if(vis_mode != INVISCID){
        vis_run = 1;
     
        // if coarse grid doesn't want to run the viscous flux, turn it off
        if(level != 0){
            IntType cg_vis = 1;
            grid->GetData(&cg_vis, INT, 1, "cg_vis");
            if(cg_vis == 0) vis_run = 0;
        }
    }
    RealFlow *vis_l = NULL, *vis_t = NULL;
    if(vis_run){
        vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
        vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    }
    // Some temporary variables
    IntType i, j, ilu, face, cell, c1, c2, c_tmp, count, sweep;
    RealFlow flux[5], q_loc[5], DQ_loc[5], DQO[5], visc, tmp, vgn_tmp;
    RealGeom face_n[3], dist;
    
    RealFlow norm0, norm, dmax = 1.0;
    RealFlow *dqo[5];
    dqo[0]  = NULL;
    mfmem::snew_array_1D(dqo[0], 5*nTCell,dmrfl);
    assert(dqo[0]  != 0);
    for(j=1; j<5; j++) dqo[j] = &dqo[j-1][nTCell];
    for(j=0; j<5; j++){
        for(i=0; i<nTCell; i++){
            dqo[j][i] = 0.0;
        }
    }
    
    IntType *luorder = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGSCellOrder");
    IntType *layer = (IntType *)grid->GetDataPtr(INT, n, "LUSGSLayer");
    IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer");
    
    IntType nTFace = grid->GetNTFace();
   
    RealGeom *norm_dist_c2c = NULL;

    if(vis_run){
        norm_dist_c2c = (RealGeom *)grid->GetDataPtr(REAL_GEOM, nTFace, "norm_dist_c2c");
    }

    assert(norm_dist_c2c);  //must exist
    
    for(sweep=0; sweep<Nsweep; sweep++){
        norm = 0.0;
        // Now the Forward Sweep
        for(ilu=0;ilu<nTCell;ilu++){
            cell = luorder[ilu];

            for(i=0; i<5; i++){
                DQO[i]       = DQ[i][cell];
                DQ[i][cell]  = rhs[i][cell] - dqo[i][cell];
                dqo[i][cell] = 0.0;
            }
            for(j=0; j<nFPC[cell]; j++){
                face  = C2F[cell][j];
                count = 2*face;
                c1    = f2c[count++];
                c2    = f2c[count];
                // One of c1 and c2 must be cell itself.
                if(layer[c1]>layer[cell] || layer[c2]>layer[cell]) continue;

                // Now its neighboring cell belongs to lower triangular
                face_n[0] = xfn[face];
                face_n[1] = yfn[face];
                face_n[2] = zfn[face];
                if(!steady) vgn_tmp = vgn[face];
                if(c2 == cell){
                    c_tmp = c1;
                    c1    = c2;
                    c2    = c_tmp;
                    face_n[0] = -face_n[0];
                    face_n[1] = -face_n[1];
                    face_n[2] = -face_n[2];
                    if(!steady) vgn_tmp = -vgn[face];
                }
                assert(c1 == cell);
                for(i=0; i<5; i++){
                    q_loc[i]  = q[i][c2];
                    DQ_loc[i] = DQ[i][c2];
                }
                 // Calculate everything (I call it Flux) in lower triangular
                
                if(steady){
                    FluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
                }else{
                    FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
                }

                
                if(vis_run){
                    dist = norm_dist_c2c[face];
                    visc = vis_l[c2] + vis_t[c2];

                    tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
                    for(i=0; i<5; i++) flux[i] -= tmp*DQ_loc[i];
                }

                // Add Flux together
                tmp = 0.5*area[face];
                for(i=0; i<5; i++){
                    flux[i]      *= tmp;
                    dqo[i][cell] += flux[i];
                    DQ[i][cell]  -= flux[i];
                }
            }
            for(i=0; i<5; i++){
                DQ[i][cell] /= Diag[cell];
                tmp          = DQ[i][cell] - DQO[i];
                norm        += tmp*tmp;
            }
        }
        
#ifdef MPICH
        IntType nvar = 5;
        RealFlow *q_mpi[5];
        for(IntType j=0; j<5; j++)
            q_mpi[j] = DQ[j];
        grid->RecvSendVarNeighbor_Togeth(nvar, q_mpi);
#endif
        // Backward Sweep
        for(ilu=nTCell-1;ilu>=0;ilu--){
            cell = luorder[ilu];

            for(i=0; i<5; i++){
                DQO[i]       = DQ[i][cell];
                DQ[i][cell]  = rhs[i][cell] - dqo[i][cell];
                dqo[i][cell] = 0.;
            }
            for(IntType j=0; j<nFPC[cell]; j++){
                face  = C2F[cell][j];
                count = 2*face;
                c1    = f2c[count++];
                c2    = f2c[count];
                // One of c1 and c2 must be cell itself.
                if(layer[c1]<layer[cell] || layer[c2]<layer[cell]) continue;

                // Now its neighboring cell belongs to upper triangular
                face_n[0] = xfn[face];
                face_n[1] = yfn[face];
                face_n[2] = zfn[face];
                if(!steady) vgn_tmp   = vgn[face];
                if(c2 == cell){
                    c_tmp = c1;
                    c1    = c2;
                    c2    = c_tmp;
                    face_n[0] = -face_n[0];
                    face_n[1] = -face_n[1];
                    face_n[2] = -face_n[2];
                    if(!steady) vgn_tmp   = -vgn[face];
                }
                assert(c1 == cell);
                for(i=0; i<5; i++){
                    q_loc[i]  = q[i][c2];
                    DQ_loc[i] = DQ[i][c2];
                }
                // Calculate everything (I call it Flux) in upper triangular
                if(steady){
                    FluxLUSGS3D(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga);
                }else{
                    FluxLUSGS3D_unsteady(flux, q_loc, DQ_loc, face_n, gam, p_bar, lhs_omga, vgn_tmp);
                }
                
                if(vis_run){
                    dist = norm_dist_c2c[face];
                    visc = vis_l[c2] + vis_t[c2];
                    
                    tmp  = 2.0*visc/(q_loc[0]*dist + TINY);
                    for(i=0; i<5; i++) flux[i] -= tmp*DQ_loc[i];
                }
                // Add Flux together
                tmp = 0.5*area[face];
                for(i=0; i<5; i++){
                    flux[i]      *= tmp;
                    dqo[i][cell] += flux[i];
                    DQ[i][cell]  -= flux[i];
                }
            }
            for(i=0; i<5; i++){
                DQ[i][cell] /= Diag[cell];
                tmp          = DQ[i][cell] - DQO[i];
                norm        += tmp*tmp;
            }  
        }
#ifdef MPICH
        MPI_Allreduce(&norm, &tmp, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
        norm = tmp;
#endif

        if(sweep == 0) norm0 = norm;
        else dmax = sqrt(norm/norm0);
        if(dmax < epsilon){
            sweep++;
            break;
        }
    }

/* #ifdef MPICH
    if(myZone == 1) printf("Resi reduced by %.5e with %d sweeps\n", dmax, (int)sweep);
#else   
    printf("Resi reduced by %.5e with %d sweeps\n", dmax, (int)sweep);
#endif */

    mfmem::sdel_array_1D(dqo[0]);
}

/*******************************************************************************\
  Calculate the Roe flux Jacobian |A| for Block LUSGS /Line Implicit LUSGS ,liming
\*******************************************************************************/
void CalJacobian_ConvectiveFlux_Roe(RealFlow matrix[5][5], RealFlow q_L[5], RealFlow q_R[5], RealFlow nx, RealFlow ny, RealFlow nz,
    RealFlow gam, RealFlow alf_l)
{
    RealFlow rho = sqrt(q_R[0]*q_L[0]);
    RealFlow tmp0 = rho/q_L[0];
    RealFlow tmp1 = 1.0/(1.0 + tmp0);

    RealFlow u  = (q_L[1] + q_R[1]*tmp0)*tmp1;
    RealFlow v  = (q_L[2] + q_R[2]*tmp0)*tmp1;
    RealFlow w  = (q_L[3] + q_R[3]*tmp0)*tmp1;
    RealFlow qn = u*nx + v*ny + w*nz;

    RealFlow gamm1 = gam - 1.0;
    RealFlow e_L  = q_L[4]/gamm1 + 0.5*q_L[0]*(q_L[1]*q_L[1] + q_L[2]*q_L[2] + q_L[3]*q_L[3]);
    RealFlow e_R  = q_R[4]/gamm1 + 0.5*q_R[0]*(q_R[1]*q_R[1] + q_R[2]*q_R[2] + q_R[3]*q_R[3]);

    RealFlow h_L  = e_L + q_L[4];
    RealFlow h_R  = e_R + q_R[4];

    RealFlow h = (h_L/q_L[0] + h_R/q_R[0]*tmp0)*tmp1;

    RealFlow q2 = 0.5*(u*u + v*v + w*w);
    RealFlow c2 = gamm1*(h - q2);
    c2 = fabs(c2);
    RealFlow c = sqrt(c2);
    RealFlow lamda_0, lamda_p, lamda_m;

    //if(steady){
    lamda_0 = fabs(qn);
    lamda_p = fabs(qn + c);
    lamda_m = fabs(qn - c);
    //}else{   //unsteady
        //lamda0 = fabs(qn - vgn[ns+i]);
        //lamdap = fabs(qn - vgn[ns+i] + c);
        //lamdan = fabs(qn - vgn[ns+i] - c);
    //}

    RealFlow epsa_r;
    //if(EntropyCorType == 4){
    //    if(IsNormalFace[ns+i] && IsShockFace[i]==0){
    //                epsa_r = 0.01*alf_l;
    //                //epsa_r = 0.0002;
    //            }else{
    epsa_r = alf_l;
    //            }
    //}
    
    //cfl3d form
    //if(steady){
    RealFlow spectral = abs(u)+abs(v)+abs(w) + c; //lamda_0;
    //}else{
    //    RealFlow u_vgn,v_vgn,w_vgn;
    //    u_vgn = vgn[ns+i]*xfn[i];
    //    v_vgn = vgn[ns+i]*yfn[i];
    //    w_vgn = vgn[ns+i]*zfn[i];
    //    spectral = fabs(u_a-u_vgn)+fabs(v_a-v_vgn)+fabs(w_a-w_vgn)+c_a;
    //}

    RealFlow epsaa = epsa_r*spectral;
    RealFlow epsbb = 0.25/std::max(epsaa,TINY);
    RealFlow epscc = 2.0*epsaa;

    //if(lamda_0<epscc) lamda_0 = lamda_0*lamda_0*epsbb + epsaa;
    //if(lamda_p<epscc) lamda_p = lamda_p*lamda_p*epsbb + epsaa;
    //if(lamda_m<epscc) lamda_m = lamda_m*lamda_m*epsbb + epsaa;

    epsaa = fabs(qn) + c; epsaa *= 0.2;
    lamda_0 += epsaa;
    lamda_p += epsaa;
    lamda_m += epsaa;
    
    {
        RealFlow lamda_pm2t = 0.5 * (lamda_p - lamda_m) / c;
        RealFlow lamda_pm02 = 0.5 * (lamda_p + lamda_m) - lamda_0;
        RealFlow lamda_pm02t = lamda_pm02 * gamm1/c2;

        RealFlow t0 = -lamda_pm2t * qn + lamda_pm02t * q2;
        RealFlow t1 =  lamda_pm2t * nx - lamda_pm02t * u;
        RealFlow t2 =  lamda_pm2t * ny - lamda_pm02t * v;
        RealFlow t3 =  lamda_pm2t * nz - lamda_pm02t * w;
        RealFlow t4 =                    lamda_pm02t;

        lamda_pm2t *= gamm1;
        RealFlow s0 =  lamda_pm2t * q2 - lamda_pm02 * qn;
        RealFlow s1 = -lamda_pm2t * u  + lamda_pm02 * nx;
        RealFlow s2 = -lamda_pm2t * v  + lamda_pm02 * ny;
        RealFlow s3 = -lamda_pm2t * w  + lamda_pm02 * nz;
        RealFlow s4 =  lamda_pm2t;

        matrix[0][0] = t0 + lamda_0;
        matrix[0][1] = t1;
        matrix[0][2] = t2;
        matrix[0][3] = t3;
        matrix[0][4] = t4;

        matrix[1][0] = u * t0 + nx * s0;
        matrix[1][1] = u * t1 + nx * s1 + lamda_0;
        matrix[1][2] = u * t2 + nx * s2;
        matrix[1][3] = u * t3 + nx * s3;
        matrix[1][4] = u * t4 + nx * s4;

        matrix[2][0] = v * t0 + ny * s0;
        matrix[2][1] = v * t1 + ny * s1;
        matrix[2][2] = v * t2 + ny * s2 + lamda_0;
        matrix[2][3] = v * t3 + ny * s3;
        matrix[2][4] = v * t4 + ny * s4;

        matrix[3][0] = w * t0 + nz * s0;
        matrix[3][1] = w * t1 + nz * s1;
        matrix[3][2] = w * t2 + nz * s2;
        matrix[3][3] = w * t3 + nz * s3 + lamda_0;
        matrix[3][4] = w * t4 + nz * s4;

        matrix[4][0] = h * t0 + qn * s0;
        matrix[4][1] = h * t1 + qn * s1;
        matrix[4][2] = h * t2 + qn * s2;
        matrix[4][3] = h * t3 + qn * s3;
        matrix[4][4] = h * t4 + qn * s4 + lamda_0;
    }
}

/*******************************************************************************\
  Calculate viscous flux Jacobian for Block LUSGS /Line Implicit LUSGS ,liming
\*******************************************************************************/
void CalJacobian_ViscousFlux(RealFlow matrix[5][5], RealFlow nx, RealFlow ny, RealFlow nz,
    RealFlow rho, RealFlow u, RealFlow v, RealFlow w, RealFlow p, RealFlow gam, RealFlow miu, RealFlow k_mod, RealFlow sd1)
{
    RealFlow gamm1 = gam-1.0;
    RealFlow q2 = u*u+v*v+w*w;
    RealFlow qn = nx*u+ny*v+nz*w;

    RealFlow coe1 = miu / rho * sd1;
    //RealFlow coe2 = gam * k_mod / miu;
    //RealFlow coe3 = coe1 * coe2;
    RealFlow coe3 = coe1 * gam * (k_mod / miu);
    
    const RealFlow third = 1.0 / 3.0;

    matrix[0][0] = 0.0;
    matrix[0][1] = 0.0;
    matrix[0][2] = 0.0;
    matrix[0][3] = 0.0;
    matrix[0][4] = 0.0;
    
    matrix[1][0] = coe1 * (-u - third * nx * qn);
    matrix[1][1] = coe1 * (third * nx * nx + 1.0);
    matrix[1][2] = coe1 * (third * nx * ny);
    matrix[1][3] = coe1 * (third * nx * nz);
    matrix[1][4] = 0.0;
    
    matrix[2][0] = coe1 * (-v - third * ny * qn);
    matrix[2][1] = matrix[1][2]; //coe1 * (third * ny * nx);
    matrix[2][2] = coe1 * (third * ny * ny + 1.0);
    matrix[2][3] = coe1 * (third * ny * nz);
    matrix[2][4] = 0.0;
    
    matrix[3][0] = coe1 * (-w - third * nz * qn);
    matrix[3][1] = matrix[1][3]; //coe1 * (third * nz * nx);
    matrix[3][2] = matrix[2][3]; //coe1 * (third * nz * ny);
    matrix[3][3] = coe1 * (third * nz * nz + 1.0);
    matrix[3][4] = 0.0;
    
    matrix[4][0] = coe3 * (0.5*q2 - p/rho/gamm1) - coe1 * (third * qn*qn + q2);
    matrix[4][1] = coe1 * (third * nx * qn + u) - u * coe3;
    matrix[4][2] = coe1 * (third * ny * qn + v) - v * coe3;
    matrix[4][3] = coe1 * (third * nz * qn + w) - w * coe3;
    matrix[4][4] = coe3;

    //for(int i=0; i<5; ++i){
    //    for(int j=0; j<5; ++j){
    //        matrix[i][j] *= 0.72 / gam;
    //    }
    //}
}

/*******************************************************************************\
        Calculate the Flux in LUSGS in 3D
\*******************************************************************************/
void FluxLUSGS3D(RealFlow flux[5], RealFlow q[5], RealFlow DQ[5], RealGeom fa_n[3], RealFlow gam, RealFlow p_bar, RealFlow lhs_omga)
{
    IntType i;
    RealFlow Q[5], rv2, v_n, p, peff, c2, eig, gam1 = gam - 1.;
    RealGeom nx, ny, nz;

    ///
    //RealFlow norm_of_dq = -1;
    //for(i=0; i<5; i++) if( abs(DQ[i]) > norm_of_dq ) norm_of_dq = abs(DQ[i]);

    //for(i=0; i<5; i++) DQ[i] *= 1.0e-5 / (norm_of_dq + TINY);
    ///

    nx   = fa_n[0];
    ny   = fa_n[1];
    nz   = fa_n[2];
        
    Q[0] = q[0];
    Q[1] = q[0]*q[1];
    Q[2] = q[0]*q[2];
    Q[3] = q[0]*q[3];
    rv2  = 0.5*q[0]*(q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    p    = q[4];
    Q[4] = p/gam1 + rv2;
    
    // Normal velocity and Eigenvalues
    v_n  = q[1]*nx + q[2]*ny + q[3]*nz;
    c2   = gam*(p + p_bar)/q[0];
    eig  = fabs(v_n) + sqrt(c2);
    eig *= lhs_omga;
    
    // Need to find out the fluxes on level n
    peff = gam*p_bar/gam1;
    flux[1] = -Q[1]*v_n - p*nx;
    flux[2] = -Q[2]*v_n - p*ny;
    flux[3] = -Q[3]*v_n - p*nz;
    flux[4] = -(Q[4] + p + peff)*v_n;
    
    // Conservative variable on level n+1
    for(i=0; i<5; i++) Q[i] += DQ[i];
    rv2 = 0.5*(Q[1]*Q[1] + Q[2]*Q[2] + Q[3]*Q[3])/Q[0];
    p   = gam1*(Q[4] - rv2);
    
    // Now the flux difference due to DQ
    flux[0]  = DQ[1]*nx + DQ[2]*ny + DQ[3]*nz;
    v_n     *= q[0];
    v_n     += flux[0];
    v_n     /= Q[0];
    flux[1] += Q[1]*v_n + p*nx;
    flux[2] += Q[2]*v_n + p*ny;
    flux[3] += Q[3]*v_n + p*nz;
    flux[4] +=(Q[4] + p + peff)*v_n;
    
    ///
    //for(i=0; i<5; i++) DQ[i] /= 1.0e-5 / (norm_of_dq + TINY);
    //for(i=0; i<5; i++) flux[i] /= 1.0e-5 / (norm_of_dq + TINY);
    ///

    // Subtract eigenvalue terms from the flux difference
    for(i=0; i<5; i++) flux[i] -= eig*DQ[i];
    
    // Note: We do not check Q[0] and p here, because they are used to
    // calculate sound speed in LUSGS. Check them later in UpdateFlowField
}


/*******************************************************************************\
        Calculate the Flux in LUSGS in 3D for unsteady
\*******************************************************************************/
void FluxLUSGS3D_unsteady(RealFlow flux[5], RealFlow q[5], RealFlow DQ[5],
                          RealGeom fa_n[3], RealFlow gam,  RealFlow p_bar, RealFlow lhs_omga,
                          RealFlow vgn)
{
    IntType i;
    RealFlow Q[5], rv2, v_n, p, peff, c2, eig, gam1 = gam - 1.;
    RealGeom nx, ny, nz;
    
    nx   = fa_n[0];
    ny   = fa_n[1];
    nz   = fa_n[2];
    
    Q[0] = q[0];
    Q[1] = q[0]*q[1];
    Q[2] = q[0]*q[2];
    Q[3] = q[0]*q[3];
    rv2  = 0.5*q[0]*(q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    p    = q[4];
    Q[4] = p/gam1 + rv2;
    
    // Normal velocity and Eigenvalues
    v_n  = q[1]*nx   + q[2]*ny  + q[3]*nz;
    c2   = gam*(p + p_bar)/q[0];
    eig  = fabs(v_n-vgn) + sqrt(c2);   //unsteady
    eig *= lhs_omga;
    
    // Need to find out the fluxes on level n
    peff = gam*p_bar/gam1;
    flux[1] = -Q[1]*v_n - p*nx; 
    flux[2] = -Q[2]*v_n - p*ny; 
    flux[3] = -Q[3]*v_n - p*nz; 
    flux[4] = -(Q[4] + p + peff)*v_n; 
    
    // Conservative variable on level n+1
    for(i=0; i<5; i++) Q[i] += DQ[i];
    rv2 = 0.5*(Q[1]*Q[1] + Q[2]*Q[2] + Q[3]*Q[3])/Q[0];
    p   = gam1*(Q[4] - rv2);
    
    // Now the flux difference due to DQ
    flux[0]  = DQ[1]*nx + DQ[2]*ny + DQ[3]*nz;
    v_n     *= q[0];
    v_n     += flux[0];
    v_n     /= Q[0];
    flux[1] += Q[1]*v_n + p*nx;
    flux[2] += Q[2]*v_n + p*ny;
    flux[3] += Q[3]*v_n + p*nz;
    flux[4] +=(Q[4] + p + peff)*v_n;
    
    // Subtract eigenvalue terms from the flux difference
    for(i=0; i<5; i++) flux[i] -= eig*DQ[i];
    
    for(i=0; i<5; i++) flux[i] -= vgn*DQ[i];   //unsteady
    
    //Note: We do not check Q[0] and p here, because they are used to 
    //      calculate sound speed in LUSGS. Check them later in UpdateFlowField
}


/*******************************************************************************\
       Update flow field in 3D
\*******************************************************************************/ 
void UpdateFlowField3D_CFL3d(PolyGrid *grid, RealFlow *DQ[5])
{
    IntType nTCell  = grid->GetNTCell();
    IntType nBFace  = grid->GetNBFace();
    IntType n       = nTCell + nBFace;
    RealFlow *rho   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    RealFlow *u     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    RealFlow *v     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    RealFlow *w     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    RealFlow *p     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    
    IntType  i, count[2];
    RealFlow alpq, phiq, betq;
    RealFlow gam, gam1, rho00, p00, rhot, rhotr, ru, rv, rw, re;
    RealFlow rho_del, p_del, rho_rat, p_rat, ptmp;
    grid->GetData(&gam, REAL_FLOW, 1, "gam");
    grid->GetData(&rho00, REAL_FLOW, 1, "rho");
    grid->GetData(&p00, REAL_FLOW, 1, "p_bar");
    gam1 = gam - 1.;
  
    RealFlow rho_min,rho_max,p_min,p_max;
    grid->GetData(&rho_min, REAL_FLOW, 1, "rho_min");
    grid->GetData(&rho_max, REAL_FLOW, 1, "rho_max");
    grid->GetData(&p_min,   REAL_FLOW, 1, "p_min");
    grid->GetData(&p_max,   REAL_FLOW, 1, "p_max");
    
    //the const value is came from CFL3D
    alpq = -0.2;
    phiq = 1./0.5;
    betq = 1.0 + alpq*phiq;
    
    count[0] = 0;
    count[1] = 0;
    for(i=0; i<nTCell; i++) {
      // Convert q to conservative variable
      rhot = rho[i];
      ru   = rhot*u[i];
      rv   = rhot*v[i];
      rw   = rhot*w[i];
      re   = p[i]/gam1 + 0.5*rhot*(u[i]*u[i] + v[i]*v[i] + w[i]*w[i]);
      
      rhot += DQ[0][i];
      ru   += DQ[1][i];
      rv   += DQ[2][i];
      rw   += DQ[3][i];
      re   += DQ[4][i];

      rhotr = 1./(rhot + TINY);
      u[i] = ru * rhotr;
      v[i] = rv * rhotr;
      w[i] = rw * rhotr;

      rho_del = DQ[0][i];
      rho_rat = rho_del/rho[i];
      if(rho_rat < alpq) {
          rho_del /= betq + fabs(rho_rat)*phiq;
          count[0]++;
      }
      rho[i]+= rho_del;
      rho[i] = MAX(rho[i], rho_min);
      rho[i] = MIN(rho[i], rho_max);

      ptmp    = gam1*(re - 0.5*(u[i]*u[i] + v[i]*v[i] + w[i]*w[i])*rho[i]);
      p_del   = ptmp - p[i];
      p_rat   = p_del/(p[i] + p00);
      if(p_rat < alpq) {
          p_del /= betq + fabs(p_rat)*phiq;
          count[1]++;
      }
      p[i]+= p_del;
      p[i] = MAX(p[i], p_min);
      p[i] = MIN(p[i], p_max);
    }
#ifdef DEBUG
#ifdef MPICH
    Parallel::parallel_sum(count, 2, MPI_COMM_WORLD);
#endif
    mflog::log.set_one_processor_out();
    if(count[0] != 0){
        mflog::log<<"Warning: "<<count[0]<<"Cells have been modify for rho in the UpdateFlowField3D_CFL3d!"<<endl;
    }
    if(count[1] != 0){
        mflog::log<<"Warning: "<<count[1]<<"Cells have been modify for p   in the UpdateFlowField3D_CFL3d!"<<endl;
    }
#endif
}


/*******************************************************************************\
   From now on, we start to build matrix free solvers such as:
        GMRES and CGS and others
\*******************************************************************************/ 
/*******************************************************************************\
     GMRES matrix free solver
\*******************************************************************************/ 
void GMRESSolverOrig(PolyGrid *grid, IntType level)
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nT5    = 5*nTCell;

    // We haven't consider turbulence model yet.
    IntType nvar = 5;
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nT5, "res");
    RealFlow *reso  = NULL;
    mfmem::snew_array_1D(reso, nT5,dmrfl);
    assert(reso != 0);
 
    // Control parameters
    IntType i, j, k, count, sweep, Adu=1, kspan = 10, Nsweeps = 5;
    grid->GetData(&Adu, INT, 1, "ADU");
    grid->GetData(&kspan, INT, 1, "kspan");
    grid->GetData(&Nsweeps, INT, 1, "gmresweeps");
    RealFlow Error = 0.;
    grid->GetData(&Error, REAL_FLOW, 1, "gmresepsilon");
    if(Error < TINY) Error = 1.0e-2;
 
    // Temporary memories
    IntType len = nvar*nTCell;
    RealFlow *dq = NULL;
    mfmem::snew_array_1D(dq, len,dmrfl);
    assert(dq != 0);

    RealFlow **H = NULL;
    mfmem::snew_array_2D(H, kspan+1,kspan,dmrfl,true);
#ifdef MPICH
    RealFlow *Htmp   = NULL;
    RealFlow *Htotal = NULL;
    mfmem::snew_array_1D(Htmp, kspan,dmrfl);
    mfmem::snew_array_1D(Htotal, kspan,dmrfl);
    for(i=1; i<kspan; i++) {
        Htmp[i] = 0.0;
        Htotal[i] = 0.0;
    }
#endif
 
    RealFlow **v = NULL;
    mfmem::snew_array_2D(v, kspan+1,len,dmrfl,true);

    RealFlow *w  = NULL;
    RealFlow *cs = NULL;
    RealFlow *sn = NULL;
    RealFlow *s  = NULL;
    mfmem::snew_array_1D(w, len,dmrfl);
    mfmem::snew_array_1D(cs, kspan,dmrfl);
    mfmem::snew_array_1D(sn, kspan,dmrfl);
    mfmem::snew_array_1D(s, kspan+1,dmrfl);
 
    RealFlow norm0, norm, dmax;
 
    // Save the beginning flow variables
    RealGeom *vol  =  grid->GetCellVol();

    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*(nTCell+nBFace), "DQ");
    if(!DQ[0]) {
        mfmem::snew_array_1D(DQ[0], 5*(nTCell+nBFace),dmrfl);
        grid->UpdateDataPtr(DQ[0], REAL_FLOW, 5*(nTCell+nBFace), "DQ");
    }
    assert(DQ[0] != 0);
    for(i=1; i<nvar; i++) DQ[i] = &DQ[i-1][nTCell+nBFace];
    for(i=0; i<nvar; i++)
        for(j=0; j<(nTCell+nBFace); j++)
            DQ[i][j] = 0.0;

    RealFlow *DQo[5];
    DQo[0] = NULL;
    mfmem::snew_array_1D(DQo[0], 5*nTCell,dmrfl);
    for(i=1; i<nvar; i++) DQo[i] = &DQo[i-1][nTCell];

    for(i=0; i<len; i++){
        dq[i]   = 0.;
    }

    // Save the residuals and Initialize Matrix*(Delta q) and p[0]
    for(i=0; i<nT5; i++) 
        reso[i] = res[i];

    // Now diagonal term in LU-SGS, here we need information of time steps
    //在一个GMERES的子迭代中,该值保持不变
    RealFlow *Diag = NULL;
    mfmem::snew_array_1D(Diag, nTCell,dmrfl);
    assert(Diag != 0);
    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
    for(i=0; i<nTCell; i++) 
        Diag[i] = vol[i]/dt[i];
    CalDiagLUSGS(grid, Diag, level);
    
    PreconditLUSGS(grid, Diag, level);

    count = 0;
    for(j=0; j<nvar; j++){
        for(i=0; i<nTCell; i++){
            DQo[j][i] = DQ[j][i];
            v[0][count] = DQo[j][i];
            count++;
        }
    }

    norm0 = DotProductMPI(v[0], v[0], len);
    norm0 = sqrt(norm0);
#ifdef MPICH
    //if(myZone==1) printf("Norm = %.5e\n", norm0);
#else
    //printf("Norm = %.5e\n", norm0);
#endif
    
 
    if(norm0 > 1.0e-10){
        // loop over GMRES sweeps
        //Nsweeps stands for the loop times before restarting.
        norm = norm0;
        for(sweep=0; sweep<Nsweeps; sweep++){
            for(k=0; k<kspan+1; k++) s[k]=0.0;
            s[0] = norm; 
            for(i=0; i<len; i++) 
                v[0][i] /= norm;               //v=gamma/beta
      
            // Loop over the search directions
            for(k=0; k<kspan; k++){
                // Calculate the epsilon in evaluating matrix * vector

                //选择不同的A*V的方法,1--luo的简化,2--原始的矩阵直接求解,3--差分近似
                if(Adu == 1)
                    ComputeADU(grid, Diag, v[k], res, level);
                else if(Adu == 2)
                    ComputeADU2(grid, Diag, v[k], res, level);
                else if(Adu == 3)
                    ComputeADU3(grid, v[k], res, reso, level);
                else if(Adu == 4)
                    ResLUSGS(grid, v[k], level);

                PreconditLUSGS(grid, Diag, level);
                
                count = 0;
                for(j=0; j<nvar; j++){
                    for(i=0; i<nTCell; i++){
                        w[count] = DQ[j][i];
                        count++;
                    }
                }
                        
                // Calculate H
                for(j=0; j<=k; j++){
                    H[j][k] = DotProduct(w, v[j], len);
                }
#ifdef MPICH
                //需要并行传递H的值
                for(j=0; j<=k; j++) Htmp[j] = H[j][k];
                for(j=0; j<kspan; j++) Htotal[j] = 0.;
                MPI_Allreduce(Htmp, Htotal, kspan, MPIReal, MPI_SUM, MPI_COMM_WORLD);
                for(j=0; j<=k; j++) H[j][k] = Htotal[j];
#endif
                for(j=0; j<=k; j++){
                    for(i=0; i<len; i++) 
                        w[i] -= H[j][k]*v[j][i];
                }

                norm  = sqrt(DotProductMPI(w, w, len));
                H[k+1][k] = norm;
                norm  = 1.0/norm;
                for(i=0; i<len; i++) 
                    v[k+1][i] = w[i]*norm;

                // Solve the linear least square problems
                for (j=0; j<k; j++)
                    ApplyPlaneRotation(H[j][k], H[j+1][k], cs[j], sn[j]);
                
                GeneratePlaneRotation(H[k][k], H[k+1][k], cs[k], sn[k]);
                ApplyPlaneRotation(H[k][k], H[k+1][k], cs[k], sn[k]);
                ApplyPlaneRotation(s[k], s[k+1], cs[k], sn[k]);

                //在该循环内,可以用V[k+1]的空间来储存W变量,节省内存使用量
            }
                        
            //完成Updata计算后，s[0]-s[k-1]中储存的数值为y[0]-y[k-1]
            ComputeY(H, s, kspan);

            // Calculate the Delta q
            for(k=0; k<kspan; k++)
                for(i=0; i<len; i++) 
                    dq[i] += v[k][i]*s[k];
            
            // Calculate Matrix*(Delta q) and P[0] for next sweep
            
            //选择不同的A*V的方法,1--luo的简化,2--原始的矩阵直接求解,3--差分近似                
            if(Adu == 1)
                ComputeADU(grid, Diag, dq, res, level);
            else if(Adu == 2)
                ComputeADU2(grid, Diag, dq, res, level);
            else if(Adu == 3)
                ComputeADU3(grid, dq, res, reso, level);
            else if(Adu == 4)
                ResLUSGS(grid, dq, level);

            // Calculate Matrix*(Delta q) and P[0] for next sweep
            PreconditLUSGS(grid, Diag, level);

            count = 0;
            for(j=0; j<nvar; j++){
                for(i=0; i<nTCell; i++){
                    v[0][count] = DQo[j][i] - DQ[j][i];
                    count++;
                }
            }

            // Check if the solution of linear eqs has been obtained within the scope 
            norm = DotProductMPI(v[0], v[0], len);
            norm = sqrt(norm);
//          if(sweep==0) norm0=norm;
            dmax = norm/norm0;
#ifdef MPICH
            //if(myZone==1) printf("Resi reduced by %.4e with %d sweeps\n", dmax, (int)(sweep+1));
#else
            //printf("Resi reduced by %.4e with %d sweeps\n", dmax, (int)(sweep+1));
#endif
            
            if(dmax < Error){
                sweep++;
                break;
            }
        }
    
        // update the solution
        count = 0;
        for(j=0; j<nvar; j++){
            for(i=0; i<nTCell; i++){
                DQ[j][i] = dq[count];
                count++;
            }
        }
        UpdateFlowField3D_CFL3d(grid, DQ);

        for(i=0; i<nT5; i++) 
            res[i] = reso[i];
    }

    // Delete temporary memories
    mfmem::sdel_array_1D(dq);
    mfmem::sdel_array_1D(Diag);
    mfmem::sdel_array_1D(reso);
    mfmem::sdel_array_1D(DQo[0]);
    mfmem::sdel_array_2D(H);
    mfmem::sdel_array_2D(v);
    mfmem::sdel_array_1D(w);
    mfmem::sdel_array_1D(cs);
    mfmem::sdel_array_1D(sn);
    mfmem::sdel_array_1D(s);
#ifdef MPICH
    mfmem::sdel_array_1D(Htmp);
    mfmem::sdel_array_1D(Htotal);
#endif
}

void GMRESSolverOrigUpdate( PolyGrid *grid, IntType level )
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nT5    = 5*nTCell;

    // We haven't consider turbulence model yet.
    IntType nvar = 5;
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nT5, "res");
    RealFlow *reso  = NULL;
    mfmem::snew_array_1D(reso, nT5,dmrfl);
    assert(reso != 0);

    // Control parameters
    IntType i, j, k, count, Adu=1, kspan = 10, maxits = 10000;
    grid->GetData(&Adu, INT, 1, "ADU");
    grid->GetData(&kspan, INT, 1, "kspan");
    grid->GetData(&maxits, INT, 1, "gmresmaxits");
    RealFlow Error = 0.;
    grid->GetData(&Error, REAL_FLOW, 1, "gmresepsilon");
    if(Error < TINY) Error = 1.0e-2;

    // Temporary memories
    IntType len = nvar*nTCell;
    RealFlow *dq = NULL;
    mfmem::snew_array_1D(dq, len,dmrfl);
    assert(dq != 0);

    RealFlow **H = NULL;
    mfmem::snew_array_2D(H, kspan+1,kspan,dmrfl,true);
#ifdef MPICH
    RealFlow *Htmp   = NULL;
    RealFlow *Htotal = NULL;
    mfmem::snew_array_1D(Htmp, kspan,dmrfl);
    mfmem::snew_array_1D(Htotal, kspan,dmrfl);
    for(i=1; i<kspan; i++) {
        Htmp[i] = 0.0;
        Htotal[i] = 0.0;
    }
#endif

    RealFlow **v = NULL;
    mfmem::snew_array_2D(v, kspan+1,len,dmrfl,true);

    RealFlow *w  = NULL;
    RealFlow *cs = NULL;
    RealFlow *sn = NULL;
    RealFlow *s  = NULL;
    mfmem::snew_array_1D(w, len,dmrfl);
    mfmem::snew_array_1D(cs, kspan,dmrfl);
    mfmem::snew_array_1D(sn, kspan,dmrfl);
    mfmem::snew_array_1D(s, kspan+1,dmrfl);

    RealFlow norm0, norm, dmax;

    // Save the beginning flow variables
    RealGeom *vol  =  grid->GetCellVol();

    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*(nTCell+nBFace), "DQ");
    if(!DQ[0]) {
        mfmem::snew_array_1D(DQ[0], 5*(nTCell+nBFace),dmrfl);
        grid->UpdateDataPtr(DQ[0], REAL_FLOW, 5*(nTCell+nBFace), "DQ");
    }
    assert(DQ[0] != 0);
    for(i=1; i<nvar; i++) DQ[i] = &DQ[i-1][nTCell+nBFace];
#ifdef FS_OPENMP
#pragma omp parallel for
#endif 
    for(i=0; i<nvar; i++)
        for(j=0; j<(nTCell+nBFace); j++)
            DQ[i][j] = 0.0;

    RealFlow *DQo[5];
    DQo[0] = NULL;
    mfmem::snew_array_1D(DQo[0], 5*nTCell,dmrfl);
    for(i=1; i<nvar; i++) DQo[i] = &DQo[i-1][nTCell];
#ifdef FS_OPENMP
#pragma omp parallel for
#endif 
    for(i=0; i<len; i++){
        dq[i]   = 0.;
    }

    // Save the residuals and Initialize Matrix*(Delta q) and p[0]
#ifdef FS_OPENMP
#pragma omp parallel for
#endif
    for(i=0; i<nT5; i++) 
        reso[i] = res[i];

    // Now diagonal term in LU-SGS, here we need information of time steps
    //在一个GMERES的子迭代中,该值保持不变
    RealFlow *Diag = NULL;
    mfmem::snew_array_1D(Diag, nTCell,dmrfl);
    assert(Diag != 0);
    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
#ifdef FS_OPENMP
#pragma omp parallel for
#endif
    for(i=0; i<nTCell; i++) 
        Diag[i] = vol[i]/dt[i];
    CalDiagLUSGS(grid, Diag, level);

    IntType pctype = 0;

    PreconditLUSGS(grid, Diag, level);
    
    count = 0;
    for(j=0; j<nvar; j++){
        for(i=0; i<nTCell; i++){
            DQo[j][i] = DQ[j][i];
            v[0][count] = DQo[j][i];
            count++;
        }
    }

    norm0 = DotProductMPI(v[0], v[0], len);
    norm0 = sqrt(norm0);
//#ifdef MPICH
//    if(myZone==1) printf("Norm = %.5e\n", norm0);
//#else
//    printf("Norm = %.5e\n", norm0);
//#endif


    if(norm0 > 1.0e-10){
        // loop over GMRES sweeps
        //Nsweeps stands for the loop times before restarting.
        norm = norm0;
        bool converge = false;
        IntType its = 0;
        while (!converge )
        {
            for(k=0; k<kspan+1; k++) s[k]=0.0;
            s[0] = norm; 
            for(i=0; i<len; i++) 
                v[0][i] /= norm;               //v=gamma/beta

            k = 0;
            // Loop over the search directions
            while(!converge && k < kspan)
            {
                // Calculate the epsilon in evaluating matrix * vector

                //选择不同的A*V的方法,1--luo的简化,2--原始的矩阵直接求解,3--差分近似
                if(Adu == 1)
                    ComputeADU(grid, Diag, v[k], res, level);
                else if(Adu == 2)
                    ComputeADU2(grid, Diag, v[k], res, level);
                else if(Adu == 3)
                    ComputeADU3(grid, v[k], res, reso, level);
                else if(Adu == 4)
                    ResLUSGS(grid, v[k], level);

                PreconditLUSGS(grid, Diag, level);
				
                count = 0;
                for(j=0; j<nvar; j++){
                    for(i=0; i<nTCell; i++){
                        w[count] = DQ[j][i];
                        count++;
                    }
                }

                // Calculate H
                for(j=0; j<=k; j++){
                    H[j][k] = DotProduct(w, v[j], len);
                }
#ifdef MPICH
                //需要并行传递H的值
                for(j=0; j<=k; j++) Htmp[j] = H[j][k];
                for(j=0; j<kspan; j++) Htotal[j] = 0.;
                MPI_Allreduce(Htmp, Htotal, kspan, MPIReal, MPI_SUM, MPI_COMM_WORLD);
                for(j=0; j<=k; j++) H[j][k] = Htotal[j];
#endif
                for(j=0; j<=k; j++){
                    for(i=0; i<len; i++) 
                        w[i] -= H[j][k]*v[j][i];
                }

                norm  = sqrt(DotProductMPI(w, w, len));
                H[k+1][k] = norm;
                norm  = 1.0/norm;
                for(i=0; i<len; i++) 
                    v[k+1][i] = w[i]*norm;

                // Solve the linear least square problems
                for (j=0; j<k; j++)
                    ApplyPlaneRotation(H[j][k], H[j+1][k], cs[j], sn[j]);

                GeneratePlaneRotation(H[k][k], H[k+1][k], cs[k], sn[k]);
                ApplyPlaneRotation(H[k][k], H[k+1][k], cs[k], sn[k]);
                ApplyPlaneRotation(s[k], s[k+1], cs[k], sn[k]);

                dmax = fabs(s[k+1]/norm0);
                if(dmax < Error) converge = true;
                //在该循环内,可以用V[k+1]的空间来储存W变量,节省内存使用量
                its ++;
                k++;
            }

            //完成Updata计算后，s[0]-s[k-1]中储存的数值为y[0]-y[k-1]
            ComputeY(H, s, k);

            // Calculate the Delta q
            for(j=0; j<k; j++)
                for(i=0; i<len; i++) 
                    dq[i] += v[j][i]*s[j];

            if(converge == true || its >= maxits) break;
            // Calculate Matrix*(Delta q) and P[0] for next sweep

            //选择不同的A*V的方法,1--luo的简化,2--原始的矩阵直接求解,3--差分近似                
            if(Adu == 1)
                ComputeADU(grid, Diag, dq, res, level);
            else if(Adu == 2)
                ComputeADU2(grid, Diag, dq, res, level);
            else if(Adu == 3)
                ComputeADU3(grid, dq, res, reso, level);
            else if(Adu == 4)
                ResLUSGS(grid, dq, level);

            // Calculate Matrix*(Delta q) and P[0] for next sweep
            PreconditLUSGS(grid, Diag, level);

            count = 0;
            for(j=0; j<nvar; j++){
                for(i=0; i<nTCell; i++){
                    v[0][count] = DQo[j][i] - DQ[j][i];
                    count++;
                }
            }

        }
#ifdef MPICH
       if(myZone == 1)
       {
#endif
           if(!converge)
           {
               printf("This should not happen!!!\n");
           }
           /*else
           {
               printf("Resi reduced by %.4e with %d iterations\n", dmax, (int)(its));
           }*/
#ifdef MPICH
       }
#endif
        // update the solution
        count = 0;
        for(j=0; j<nvar; j++){
            for(i=0; i<nTCell; i++){
                DQ[j][i] = dq[count];
                count++;
            }
        }
        UpdateFlowField3D_CFL3d(grid, DQ);

        for(i=0; i<nT5; i++) 
            res[i] = reso[i];
    }

    // Delete temporary memories
    mfmem::sdel_array_1D(dq);
    mfmem::sdel_array_1D(Diag);
    mfmem::sdel_array_1D(DQo[0]);
    mfmem::sdel_array_1D(reso);
    mfmem::sdel_array_2D(H);
    mfmem::sdel_array_2D(v);
    mfmem::sdel_array_1D(w);
    mfmem::sdel_array_1D(cs);
    mfmem::sdel_array_1D(sn);
    mfmem::sdel_array_1D(s);

#ifdef MPICH
    mfmem::sdel_array_1D(Htmp);
    mfmem::sdel_array_1D(Htotal);
#endif
}



/*******************************************************************************\
     GMRES matrix free solver
\*******************************************************************************/ 
void ComputeY(RealFlow **h, RealFlow *y, IntType k)
{
    // Backsolve:  
    for (IntType i = k-1; i >= 0; i--) {
        y[i] /= h[i][i];
        for (IntType j = k-1; j >i; j--)
            y[i] -= h[i][j] * y[j] / h[i][i];
    }
}


/*******************************************************************************\
     GMRES matrix free solver
\*******************************************************************************/ 
void GeneratePlaneRotation(RealFlow &dx, RealFlow &dy, RealFlow &cs, RealFlow &sn)
{    
    if (dy == 0.0) {
        cs = 1.0;
        sn = 0.0;
    } else if (abs(dy) > abs(dx)) {
        RealFlow temp = dx / dy;
        sn = 1.0 / sqrt( 1.0 + temp*temp );
        cs = temp * sn;
    } else {
        RealFlow temp = dy / dx;
        cs = 1.0 / sqrt( 1.0 + temp*temp );
        sn = temp * cs;
    }
}
/*******************************************************************************\
     GMRES matrix free solver
\*******************************************************************************/ 
void ApplyPlaneRotation(RealFlow &dx, RealFlow &dy, RealFlow &cs, RealFlow &sn)
{
    RealFlow temp  =  cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = temp;
}

/******************************************************************************\
   Calculate the dot product of two vectors with length n
\******************************************************************************/
RealFlow DotProduct(RealFlow *a, RealFlow *b, IntType n)
{
    IntType  i;
    RealFlow sum = 0.;
 
    for(i=0; i<n; i++) sum += a[i]*b[i];
    return sum;
}


/******************************************************************************\
   Calculate the dot product of two vectors with length n
\******************************************************************************/
RealFlow DotProductMPI(RealFlow *a, RealFlow *b, IntType n)
{
    IntType  i;
    RealFlow sum = 0.0, sum_glb=0.0;

    for(i=0; i<n; i++) sum += a[i]*b[i];
#ifdef MPICH
    MPI_Allreduce(&sum, &sum_glb, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    sum = sum_glb;
#endif
    return sum;
}


/*******************************************************************************\
     GMRES with LU-SGS preconditioning 
\*******************************************************************************/ 
void ComputeADU(PolyGrid *grid, RealFlow *Diag, RealFlow *v, RealFlow *res, IntType level)
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nTotal = nTCell + nBFace;   
    
    IntType sweeps = 1;
    grid->GetData(&sweeps, INT, 1, "sweeps");
    RealFlow epsilon = 0.1;
    grid->GetData(&epsilon, REAL_FLOW, 1, "epsilon"); 
    if(epsilon < TINY) epsilon = 0.1;
    
    // Get number of faces for each cell
    IntType *nFPC = CalnFPC(grid);
    // Get cell to face connectivity
    IntType **C2F = CalC2F(grid); 
      
    IntType i, j, ntemp;
    
    RealFlow *rhs[5];
    rhs[0] = res;
    for(i=1; i<5; i++) rhs[i] = &rhs[i-1][nTCell];
    
    // Allocate memories for RHS or DQ
    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTotal, "DQ");
    assert(DQ[0] != 0);
    for(i=1; i<5; i++) DQ[i] = &DQ[i-1][nTotal]; 
    for(j=0; j<5*nTotal; j++) DQ[0][j] = 0.; 

    ntemp = 0;
    for(j=0; j<5; j++){
        for(i=0; i<nTCell; i++){
            // Copy the v to DQ
            DQ[j][i] = v[ntemp++];

            //计算当前单元对ADU的贡献
            rhs[j][i] = Diag[i]*DQ[j][i];
        }
    }

    // IF MPICH, we could exchange DQ here for INTERFACES
#ifdef MPICH
        IntType nvar = 5;
        RealFlow *q_mpi[5];
        for(IntType j=0; j<5; j++)
            q_mpi[j] = DQ[j];
        grid->RecvSendVarNeighbor_Togeth(nvar, q_mpi);
#endif

    if(sweeps == 1){
        //计算相邻单元对ADU的贡献
        SolveADU3D(grid, rhs, DQ, nFPC, C2F, level);
    }
}


/*******************************************************************************\
     GMRES with LU-SGS preconditioning 
\*******************************************************************************/ 
void ComputeADU2(PolyGrid *grid, RealFlow *Diag, RealFlow *v, RealFlow *res, IntType level)
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nTotal = nTCell + nBFace;   
    
    IntType sweeps = 1;
    grid->GetData(&sweeps, INT, 1, "sweeps");
    RealFlow epsilon = 0.1;
    grid->GetData(&epsilon, REAL_FLOW, 1, "epsilon"); 
    if(epsilon < TINY) epsilon = 0.1;
    
    // Get number of faces for each cell
    IntType *nFPC = CalnFPC(grid);
    // Get cell to face connectivity
    IntType **C2F = CalC2F(grid); 
      
    IntType  i, j, ntemp;
    
    RealFlow *rhs[5];
    rhs[0] = res;
    for(i=1; i<5; i++) rhs[i] = &rhs[i-1][nTCell];
    
    // Allocate memories for RHS or DQ
    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTotal, "DQ");
    assert(DQ[0] != 0);
    for(i=1; i<5; i++) DQ[i] = &DQ[i-1][nTotal]; 
    for(j=0; j<5*nTotal; j++) DQ[0][j] = 0.; 

    ntemp = 0;
    for(j=0; j<5; j++){
        for(i=0; i<nTCell; i++){
            // Copy the v to DQ
            DQ[j][i] = v[ntemp++];

            //计算当前单元对ADU的贡献, 时间项
            rhs[j][i] = Diag[i]*DQ[j][i];
        }
    }

    // IF MPICH, we could exchange DQ here for INTERFACES
#ifdef MPICH
        IntType nvar = 5;
        RealFlow *q_mpi[5];
        for(IntType j=0; j<5; j++)
            q_mpi[j] = DQ[j];
        grid->RecvSendVarNeighbor_Togeth(nvar, q_mpi);
#endif

    if(sweeps == 1){
        //计算相邻单元对ADU的贡献,偏导项
        SolveADU3D2(grid, rhs, DQ, nFPC, C2F, level);
    }
}


/*******************************************************************************\
     GMRES with LU-SGS preconditioning 
\*******************************************************************************/ 
void ComputeADU3(PolyGrid *grid, RealFlow *v, RealFlow *res, RealFlow *reso, IntType level)
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;   
    IntType nvar   = 5;
    IntType len    = nvar*nTCell;
    IntType count, i, j;

    // Tempory memories
    RealFlow *qo  = NULL;
    mfmem::snew_array_1D(qo, len,dmrfl);
    assert(qo != 0);

    RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");

    // Save the beginning flow variables
    count = 0;
    for(j=0; j<nvar; j++)
        for(i=0; i<nTCell; i++) 
            qo[count++] = q[j][i];

    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*n, "DQ");
    assert(DQ[0] != 0);
    for(i=1; i<nvar; i++) DQ[i] = &DQ[i-1][n];
    for(j=0; j<5*n; j++) DQ[0][j] = 0.; 

    RealFlow eps, eps0 = 1.0e-7;
    eps  = sqrt(DotProduct(v, v, len));
    eps  = eps0/eps;

    count = 0;
    for(j=0; j<nvar; j++){
        for(i=0; i<nTCell; i++){
            DQ[j][i] = eps*v[count];
            count++;
        }
    }
    UpdateFlowField3D_CFL3d(grid, DQ);

    ZeroResiduals(grid);
    UpdateResiduals(grid, 0);
    for(i=0; i<len; i++){
        res[i] = reso[i]-res[i];
        res[i] /= eps;
    }

    // Save the beginning flow variables
    count = 0;
    for(j=0; j<nvar; j++)
        for(i=0; i<nTCell; i++) 
            q[j][i] = qo[count++];
    mfmem::sdel_array_1D(qo);
}

#undef CPP_FILD_ID  // clear out file id
} //~namespace mflow
