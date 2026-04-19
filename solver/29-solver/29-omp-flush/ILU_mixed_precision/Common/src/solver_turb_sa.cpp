//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   solver_turb_sa.cpp
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

// direct head file
#include "solver_turb_sa.h"

// C++ build-in head files
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cassert>
#include <iostream>
#include <fstream>

// other user defined head file
#include "zone.h"
#include "utility_functions.h"
#include "solver_ns.h"
#include "temporal_discretisation_implicit.h"
#include "io_base_format.h"
#include "io_log.h"
#include "io_field.h"
#include "parallel_base_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif
using namespace std;

#ifdef MPICH
#include "mpi.h"
#endif

#ifdef FS_CUDA
#include "cuSAsolver.cuh"
#include "cuGradientQ_Gauss.cuh"
#include <cuTurbulenceFlux.cuh>
#endif

//dingxin
#ifdef TIMECOST
extern double* timecost;
extern int num_timecost;
extern double  time_flux, time_invis, time_roe, time_vis, time_calvis;
extern double  time_limiter;
extern double  time_gradient;
extern double  time_lusgs;
extern double  time_SA;
#endif // TIMECOST
//TIMECOST

namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 11906  // define file id

#ifdef MPICH
extern int myZone;
extern int numprocs;
extern MPI_Comm GridComm;  //for each grid, tangj
#endif

// constructor
SASolver::SASolver(IntType ng, PolyGrid **gridsin, DataStore **fieldsin, 
                   DataSafe *cParain, BCond *bcin, Zone *zonein) 
{
    nGrids = ng; 
    grids  = gridsin; 
    fields = fieldsin; 
    bc     = bcin; 
    cPara  = cParain;
    zone   = zonein;
}


void SASolver::Init()
{
    PolyGrid *grid = (PolyGrid *) grids[0];
        
    IntType restart,turbRst=0;
    GetData(&restart,INT,1,"restart");
    GetData(&turbRst,INT,1,"turbRst",0);

    Setturb00(grid, "SA");

    // allocate memory for flow field
    AllocateFlowfieldMemory(grid);

    if(restart==0){
        RealFlow start_time = 0.;
        UpdateData(&start_time, REAL_FLOW, 1, "start_time");
        InitGridVar(grid);
    }     
    else if(restart==1){
        ReadRestartFromFile(grid);
    }

    SetGhostvis_t(grid,"SA");
    
    GhostVariablesScalar_SA(grid);
   
#ifdef MPICH
    IntType n;
    RealFlow *vis_t,*sa_nu;
    n     = grid->GetNTCell() + grid->GetNBFace();
    vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    sa_nu = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "sa_nu");
    grid->CommInterfaceDataMPI(vis_t);
    grid->CommInterfaceDataMPI(sa_nu);
#endif

}


/************************************************************************
  allocate memory  for flow field
  tangj, 2019-11-05
************************************************************************/
void SASolver::AllocateFlowfieldMemory(PolyGrid *grid)
{
    RealFlow   *sa_nu, *vis_t;
    RealFlow   *sa_nu_cur, *sa_nu_old;
    IntType    nTCell = grid->GetNTCell(), n = nTCell + grid->GetNBFace();
    IntType    steady = 1;
    grid->GetData(&steady, INT, 1, "steady");

    // Allocate memories for flow variables
    sa_nu = NULL;    vis_t = NULL;
    mfmem::snew_array_1D(sa_nu, n, dmrfl);
    mfmem::snew_array_1D(vis_t, n, dmrfl);

    if(!steady){
        sa_nu_cur = NULL;
        sa_nu_old = NULL;
        mfmem::snew_array_1D(sa_nu_cur, nTCell, dmrfl);
        mfmem::snew_array_1D(sa_nu_old, nTCell, dmrfl);
    }

    for(IntType i=0; i<n; i++){
        sa_nu[i]= 0.0;
        vis_t[i]= 0.0; 
    } 
      
    // attach the data to the grid now
    grid->UpdateDataPtr(sa_nu, REAL_FLOW, n, "sa_nu");
    grid->UpdateDataPtr(vis_t, REAL_FLOW, n, "vis_t");
    if(!steady){
        grid->UpdateDataPtr(sa_nu_cur, REAL_FLOW, nTCell, "sa_nu_cur");
        grid->UpdateDataPtr(sa_nu_old, REAL_FLOW, nTCell, "sa_nu_old");
    }
}


/************************************************************************
          Read flow variables from restart file
************************************************************************/
void SASolver::ReadRestartFromFile(PolyGrid *grid)
{
    FILE       *fp;
    IntType    ntmp, nstp, iter, nvar;
    RealFlow   *sa_nu, *vis_t, start_time;
    RealFlow   *sa_nu_cur, *sa_nu_old;
    IntType    nTCell = grid->GetNTCell();
    IntType    n = nTCell + grid->GetNBFace();
    IntType    zn = grid->GetZone();
    IntType    steady=1, Unst_steps_Curt;
    grid->GetData(&steady, INT, 1, "steady");
    zone->simu->GetData(&Unst_steps_Curt, INT, 1, "Unst_steps_Curt");

    IntType file_id = 0;
#ifdef MPICH
    file_id = myZone;
#else
    file_id = zn + 1;
#endif
    std::string rest_file = FieldIO::restart_file_with_id(file_id);

    fp = fopen(rest_file.c_str(), "rb");
    if(!fp) {
        std::cerr << "Failed to open file " << rest_file << " in function ReadRestartFromFile" << std::endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
  
    fread(&ntmp, sizeof(IntType), 1, fp);
    fread(&nstp, sizeof(IntType), 1, fp);
    fread(&iter, sizeof(IntType), 1, fp);
    fread(&nvar, sizeof(IntType), 1, fp);
    fread(&start_time, sizeof(RealFlow), 1, fp);

    if(ntmp != nTCell) {
        std::cerr << "Total numbers of cells in the grid and in the file are different" << std::endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }

    if(nstp != Unst_steps_Curt) {
        std::cerr << "nstp and Unst_steps_Curt are different" << std::endl;
    }
  
    // Get memories for flow variables
    sa_nu = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "sa_nu");
    vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    assert(sa_nu != NULL);
    if(!steady){
        sa_nu_cur = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "sa_nu_cur");
        sa_nu_old = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "sa_nu_old");
    }
    
    if(nvar == 8) {      
        fread(sa_nu, sizeof(RealFlow), nTCell, fp); 
        fread(sa_nu, sizeof(RealFlow), nTCell, fp);
        fread(sa_nu, sizeof(RealFlow), nTCell, fp);
        fread(sa_nu, sizeof(RealFlow), nTCell, fp);
        fread(sa_nu, sizeof(RealFlow), nTCell, fp);
        if(!steady){
            fread(sa_nu, sizeof(RealFlow), nTCell, fp); 
            fread(sa_nu, sizeof(RealFlow), nTCell, fp);
            fread(sa_nu, sizeof(RealFlow), nTCell, fp);
            fread(sa_nu, sizeof(RealFlow), nTCell, fp);
            fread(sa_nu, sizeof(RealFlow), nTCell, fp);

            fread(sa_nu, sizeof(RealFlow), nTCell, fp); 
            fread(sa_nu, sizeof(RealFlow), nTCell, fp);
            fread(sa_nu, sizeof(RealFlow), nTCell, fp);
            fread(sa_nu, sizeof(RealFlow), nTCell, fp);
            fread(sa_nu, sizeof(RealFlow), nTCell, fp);
        }
 
        fread(sa_nu, sizeof(RealFlow), nTCell, fp); 
        if(!steady){
            fread(sa_nu_cur, sizeof(RealFlow), nTCell, fp);
            fread(sa_nu_old, sizeof(RealFlow), nTCell, fp);
        }
       
        fread(vis_t, sizeof(RealFlow), nTCell, fp);
        fread(vis_t, sizeof(RealFlow), nTCell, fp); 
    } else {
        std::cerr << "Variable number in the restart file isn't 8" << std::endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    fclose(fp);
    
    UpdateData(&start_time, REAL_FLOW, 1, "start_time");
}


//*****************************************************************************\
/// \brief 求解SA方程流场 
///
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 2020-08-07  tangj      Add parameter 'iter_step_physic_time' for NS/Turb
///                        convergence log file.
/// </pre>
//*****************************************************************************/
void SASolver::Solve()
{
    PolyGrid *grid = (PolyGrid *) grids[0];
    IntType   iter_done, zn;
    RealFlow  t_now;
    double    t_onestep;

    // Find the local time before solving flow on the grid
#ifdef MPICH
    t_onestep = MPI_Wtime();
#else
#if !(defined(Windows_NT) )
    timeval    t_tmp;
    gettimeofday(&t_tmp, NULL);
    t_onestep = (double)t_tmp.tv_sec + (double)t_tmp.tv_usec/1000000;
#else
    t_onestep = 0.0;
#endif
#endif

#ifdef TIMECOST//dingxin
#ifdef FS_CUDA
	cudaDeviceSynchronize();
#endif
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemVis, endtimeTemVis;
    double timeuseTemVis;
    gettimeofday(&starttimeTemVis, 0);
#endif
#endif

    // Get iteration and zonal number
    GetData(&iter_done, INT, 1 ,"iter_done");
    zn = grid->GetZone();
    // Get the starting time and find out the recent time
    GetData(&t_now, REAL_FLOW, 1, "start_time");
	
#ifdef FS_CUDA
	cuSAsolve(grid);
#else
    ComputeTurbGeneration_SA(grid);
    ZeroGridResiduals(grid, "res", 1);
    SolveScalarOnGrid(grid, "sa_nu"); 
    ComputeTurbViscosity_SA(grid);
#endif     
    
    // Find the local time after solving flow on the grid
#ifdef MPICH
    t_onestep = MPI_Wtime() - t_onestep;
#else
#if !(defined(Windows_NT) )
    gettimeofday(&t_tmp, NULL);
    t_onestep = (double)t_tmp.tv_sec + (double)t_tmp.tv_usec/1000000 - t_onestep;
#else
    t_onestep = 0.0;
#endif
#endif
    t_now    += t_onestep;
    
    IntType n_wconverg = 20;
    GetData(&n_wconverg,  INT, 1, "n_wconverg");
    // for convergence log file to output the first step of each physical time step
    IntType iter_step_physic_time = 1;
    GetData(&iter_step_physic_time,  INT, 1, "iter_step_physic_time", 0);
    if((iter_done%n_wconverg==0) || (iter_done==1) || iter_step_physic_time==0){
#ifdef FS_CUDA
		cuLoadBackResSA(grid);
#endif
        // Print out norm of the residuals
        DumpTurbNormResi(grid, iter_done, zn, t_now, "SA", 1);
    }
    
    UpdateData(&t_now, REAL_FLOW, 1, "start_time");
    
    // Free the memories of the residuals and dq for all grids
    FreeGridResi(grid, 1); 
    
    // Print out restart file if necessary
    IntType n_wrest = 50;
    GetData(&n_wrest, INT, 1, "n_wrest", 0);
    if(iter_done%n_wrest == 0) {
        DumpRestart(grid, iter_done, zn, t_now); 
    }
#ifdef TIMECOST//dingxin
#ifdef FS_CUDA
	cudaDeviceSynchronize();
#endif
#ifdef MPICH
    timecost[4] = timecost[4] + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemVis, 0);
    timeuseTemVis = (RealGeom)1000000 * (endtimeTemVis.tv_sec - starttimeTemVis.tv_sec) + endtimeTemVis.tv_usec - starttimeTemVis.tv_usec;
    timecost[4] += timeuseTemVis;
	timeuseTemVis /= 1000000.0;
    time_SA += timeuseTemVis;
#endif
#endif
}


/*******************************************************************************\
  Post-Processing, i.e. dump data file for visualization                    
\*******************************************************************************/
void SASolver::Post()
{
    IntType iter_done, zn, n_steps;
    RealFlow t_now;
    PolyGrid *grid = (PolyGrid *)grids[0];
    
    GetData(&iter_done,INT,1,"iter_done");
    GetData(&t_now, REAL_FLOW, 1, "start_time");
    zn = zone->GetZoneNo();
    zone->simu->GetData(&n_steps, INT, 1, "n_steps");
    
    IntType n_wrest = 50;
    GetData(&n_wrest, INT, 1, "n_wrest", 0);
    if(n_steps > 0 && iter_done%n_wrest != 0) {
        DumpRestart(grid, iter_done, zn, t_now);
    }
}


/*******************************************************************************\
                     
\*******************************************************************************/
void SASolver::UpdateInterfaceData()
{
    CommInterfaceData("sa_nu");
}


/*******************************************************************************\
                     
\*******************************************************************************/
void SASolver::UpdataUnstVolData()
{
    CommUnstVolData(grids, "sa_nu", "sa_nu_cur", "sa_nu_old");
}


/*******************************************************************************\
                     
\*******************************************************************************/
void SASolver::CommInterfaceData(const char *name)
{
    IntType g;
    PolyGrid *grid;

#ifndef MPICH
    PolyGrid *grid0 = grids[0];
    IntType *nbz = grid0->GetFaceNeighborZones();

    for(IntType i=0; i<grid0->GetNumberOfFaceNeighbors(); i++) {        
        Zone *nz = zone->simu->GetZone(nbz[i]);
        for(g=0; g<nGrids; g++) {
            grid = (PolyGrid *) grids[g];
            grid->CommInterfaceData(nbz[i], (PolyGrid*)nz->GetGrid(g), name);
        }
    }
#else
    IntType n;
    RealFlow *q;
    
    for(g=0; g<nGrids; g++) {
        grid = (PolyGrid *) grids[g];
        n = grid->GetNTCell()+grid->GetNBFace();
        q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
        grid->CommInterfaceDataMPI(q);
    }

#endif
}


/************************************************************************
          Initialize flow field as a uniform flow
************************************************************************/
void SASolver::InitGridVar(PolyGrid *grid)
{
    IntType i;
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    
    IntType steady;
    RealFlow sa_nu00,vis_t00;
    grid->GetData(&steady, INT, 1, "steady");
    grid->GetData(&sa_nu00, REAL_FLOW, 1, "sa_nu00");
    grid->GetData(&vis_t00, REAL_FLOW, 1, "vis_t00");
    
    RealFlow *sa_nu = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "sa_nu");
    RealFlow *vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    RealFlow *sa_nu_cur = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "sa_nu_cur");
    RealFlow *sa_nu_old = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "sa_nu_old");
    if(sa_nu == 0){
      mfmem::snew_array_1D(sa_nu, n,dmrfl);
      grid->UpdateDataPtr(sa_nu, REAL_FLOW, n, "sa_nu");
    }
    if(vis_t == 0){
      mfmem::snew_array_1D(vis_t, n,dmrfl);
      grid->UpdateDataPtr(vis_t, REAL_FLOW, n, "vis_t");
    }

    if(!steady){
        if(sa_nu_cur == 0){
            mfmem::snew_array_1D(sa_nu_cur,nTCell,dmrfl);
            grid->UpdateDataPtr(sa_nu_cur, REAL_FLOW, nTCell, "sa_nu_cur");
        }
        if(sa_nu_old == 0){
            mfmem::snew_array_1D(sa_nu_old,nTCell,dmrfl);
            grid->UpdateDataPtr(sa_nu_old, REAL_FLOW, nTCell, "sa_nu_old");
        }
    }

    for(i=0; i<nTCell; i++){ 
        sa_nu[i] = sa_nu00;
        vis_t[i] = vis_t00;
    }
    
    if(!steady){
        for(i=0; i<nTCell; i++){
            sa_nu_cur[i] = sa_nu00;
            sa_nu_old[i] = sa_nu00;
        }
    }  
}


/******************************************************************************\
         Print out restart file 
\******************************************************************************/
void SASolver::DumpRestart(PolyGrid *grid, IntType iter, IntType zn, RealFlow t_now)
{
    FILE    *fp;
    IntType nTCell = grid->GetNTCell();
    IntType n      = nTCell + grid->GetNBFace();
    IntType steady=1;
    grid->GetData(&steady, INT, 1, "steady");
    
    IntType file_id = 0;
#ifdef MPICH
    file_id = myZone;
#else
    file_id = zn + 1;
#endif

    std::string rest_file = FieldIO::restart_file_with_id(file_id);

    fp = fopen(rest_file.c_str(),"ab");
    if(!fp) {
        std::cerr << "Failed to open file " << rest_file << " in function DumpRestart" << std::endl;
        return;
    }
  
    RealFlow *sa_nu  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "sa_nu");
    RealFlow *sa_nu_cur  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "sa_nu_cur");
    RealFlow *sa_nu_old  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "sa_nu_old");
    RealFlow *vis_l  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
    RealFlow *vis_t  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    
    fwrite(sa_nu, sizeof(RealFlow), nTCell, fp);
    if(!steady){
        fwrite(sa_nu_cur, sizeof(RealFlow), nTCell, fp);
        fwrite(sa_nu_old, sizeof(RealFlow), nTCell, fp);
    }
    fwrite(vis_l, sizeof(RealFlow), nTCell, fp);
    fwrite(vis_t, sizeof(RealFlow), nTCell, fp);
    
    fclose(fp);
}


/******************************************************************************\
|       set flow variables at ghost cells
\******************************************************************************/
void GhostVariablesScalar_SA(PolyGrid *grid)
{
    IntType i, c1, c2, type;
    IntType nBFace  = grid->GetNBFace();
    IntType nTCell  = grid->GetNTCell();
    IntType n       = nTCell + nBFace; 
    IntType *f2c    = grid->Getf2c();
    BCRecord **bcr  = grid->Getbcr();
    RealGeom *xfn   = grid->GetXfn();
    RealGeom *yfn   = grid->GetYfn();
    RealGeom *zfn   = grid->GetZfn();
    
    RealFlow *rho   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    RealFlow *u     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    RealFlow *v     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    RealFlow *w     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    RealFlow *p     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    RealFlow *sa_nu = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "sa_nu");
  
    RealFlow vn,vnb,vnf,cf,cc,gam,p_bar,sa_nu00;
    RealFlow rhoP,uP,vP,wP,pP,riemp, riemm;
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&pP,    REAL_FLOW, 1, "p");
    grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
    grid->GetData(&uP,    REAL_FLOW, 1, "u");
    grid->GetData(&vP,    REAL_FLOW, 1, "v");
    grid->GetData(&wP,    REAL_FLOW, 1, "w");
    
    grid->GetData(&sa_nu00, REAL_FLOW, 1, "sa_nu00");
    
    IntType steady;
    grid->GetData(&steady, INT, 1, "steady");
    RealGeom *vgn = grid->GetFaceNormalVelocity();

    for(i=0; i<nBFace; i++) {
        type = bcr[i]->GetType();
        
        // Do nothing for interfaces.
        if(type == INTERFACE) continue;
        
        c1   = f2c[i+i];
        c2   = f2c[i+i+1];

        // Assign the variable values for each ghost cell whose index is c2.
        switch(type) {
            case WALL:
                sa_nu[c2] = -sa_nu[c1];
                break; 

            case SYMM:
                sa_nu[c2] = sa_nu[c1];
                break;

            case FAR_FIELD:
                vnf   = xfn[i]*uP + yfn[i]*vP + zfn[i]*wP;
                vn    = xfn[i]*u[c1] + yfn[i]*v[c1] + zfn[i]*w[c1];
                cf    = sqrt(gam*(pP + p_bar)/rhoP);
                cc    = sqrt(gam*(p[c1] + p_bar)/rho[c1]);
                riemp = vn+2.*cc/(gam-1.);
                riemm = vnf-2.*cf/(gam-1.);
                vnb   = 0.5*(riemp+riemm);
                
                if(!steady) vnb -= vgn[i];

                if(vnb>0)  sa_nu[c2] = sa_nu[c1];
                else       sa_nu[c2] = sa_nu00; 
                break;
             
            default:
                printf("Error in GhostVariablesScalar_SA\n");
                break;
        }
    }
}


/******************************************************************************\
|                                  CFL3D method
\******************************************************************************/
void AddSourceSA(PolyGrid *grid)
{
   
    IntType nTCell = grid->GetNTCell();
    IntType n      = nTCell + grid->GetNBFace();
    
    RealFlow *rho     = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "rho");
    RealFlow *sa_nu   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "sa_nu");
    RealFlow *res     = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "res");
    RealFlow **lhsmat = (RealFlow **)grid->GetDataPtr(REAL_FLOW, nTCell, "lhsmat");
    RealFlow *omaga   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "omaga");
    RealFlow *vis_l   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "vis_l");
    RealGeom *vol     = grid->GetCellVol();
    
    RealFlow time_accuracy;
    grid->GetData(&time_accuracy, REAL_FLOW, 1, "time_accuracy");
    
    RealGeom *dist2wall = (RealGeom *) grid->GetDataPtr(REAL_GEOM, nTCell, "dist2wall_cell");
    RealGeom *dist2wall_temp = NULL;
    mfmem::snew_array_1D(dist2wall_temp, nTCell,dmrfl);
#ifdef FS_OPENMP
#pragma omp parallel for
#endif    
    for(IntType i=0; i<nTCell; i++){
        dist2wall_temp[i] = dist2wall[i];
    } 
    
    int iexp = 15;
    RealFlow xminn;
    grid->GetData(&iexp, INT, 1, "iexp", 0);
    //Note: (10.**(-iexp) is machine zero)
    xminn = pow(10.0, -iexp+1);
#ifdef FS_OPENMP
#pragma omp parallel for
#endif    
    for(IntType i=0; i<nTCell; i++){
        RealFlow omaga_cur,S_bar,xkai,xkaip3,fv1,fv2,d,dp2,odp2;
        RealFlow nue,rr,gg,ft2,fw,term1,term2,source,fsim;
        RealFlow dfv1,dfv2,dft2,drr,dgg,dfw;
        nue    = vis_l[i]/rho[i];             
        xkai   = sa_nu[i]/nue; 
        xkaip3 = xkai*xkai*xkai;
        fv1    = xkaip3/(xkaip3+CV1P3); 
        fv2    = 1.0-xkai/(1.0+xkai*fv1);
      
        d      = dist2wall_temp[i];
        dp2    = d*d;
      
        omaga_cur = omaga[i];
       
        S_bar = omaga_cur+sa_nu[i]*fv2/(KAIP2*dp2);
        S_bar = MAX(S_bar,xminn);  
       
        rr     = sa_nu[i]/(S_bar*KAIP2*dp2); 
        rr     = MIN(rr,10.0);           
    
        gg     = rr+CW2*(P6(rr)-rr);
        gg     = MAX(gg,xminn);  
    
        fw     = gg*SQRT_SIX((1.0+CW3P6)/(P6(gg)+CW3P6));
        ft2    = CT3*exp(-CT4*xkai*xkai);
      
        term1  = CB1*(1.0-ft2)*omaga_cur;
        term2  = CB1*((1.0-ft2)*fv2+ft2)/KAIP2-CW1*fw;
      
        odp2   = 1.0/dp2;      
        source = term1*sa_nu[i]+term2*sa_nu[i]*sa_nu[i]*odp2;
      
        fsim   = 2.0*term2*sa_nu[i]*odp2;
        dfv1   = 3.0*(fv1-fv1*fv1)/sa_nu[i];
        dfv2   = (fv2-1.0)/sa_nu[i]+(1.0-fv2)*(1.0-fv2)*(fv1/sa_nu[i]+dfv1);
        dft2   = -(2.0*CT4*sa_nu[i]/(nue*nue))*ft2;
        drr    = rr/sa_nu[i]-rr*rr*(fv2/sa_nu[i]+dfv2); 
        dgg    = (1.0-CW2+6.0*CW2*(rr*rr*rr*rr*rr))*drr;
        gg     = MAX(gg,10.0*xminn);
        dfw    = SQRT_SIX((1.0+CW3P6)/(P6(gg)+CW3P6))
               -(SQRT_SIX(1.0+CW3P6)/(pow((P6(gg)+CW3P6),(7.0/6.0))))*P6(gg);      
        dfw   *= dgg;  
        fsim  += odp2*sa_nu[i]*sa_nu[i]*(CB1/KAIP2*(dfv2-ft2*dfv2-fv2*dft2+dft2)-CW1*dfw);
    
        res[i]+= source*rho[i]*vol[i];
        if(fsim<0.0)  lhsmat[i][0]  -= fsim*rho[i]*vol[i];
    } 
    mfmem::sdel_array_1D(dist2wall_temp);
}
/******************************************************************************\
|
\******************************************************************************/
void AddSourceUnstSA(PolyGrid *grid)
{
    ComTubSourceUnst(grid, "sa_nu", "sa_nu_cur", "sa_nu_old");
}


/******************************************************************************\
| compute vis_t in SA model
\******************************************************************************/
void ComputeTurbViscosity_SA(PolyGrid *grid)
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    RealFlow *vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    RealFlow *rho   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    RealFlow *sa_nu = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "sa_nu");
    RealFlow *vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
    RealFlow amu;
    grid->GetData(&amu,  REAL_FLOW, 1, "amu");
    RealFlow max_muet;
    grid->GetData(&max_muet, REAL_FLOW, 1, "max_muet",0);
    if(max_muet<0.0) max_muet=MAX_MUET_SA;

#ifdef DEBUG
    RealFlow vis_t_min=BIG, vis_t_max=-BIG;
#endif
    
#ifdef FS_OPENMP
#pragma omp parallel for
#endif    
    for(IntType i=0; i<nTCell; i++){
        RealFlow fv1, xkai, xkaip3, nue;
        nue = vis_l[i]/rho[i];
        xkai = sa_nu[i]/nue;
        xkaip3 = xkai*xkai*xkai;
        fv1 = xkaip3/(xkaip3+CV1P3);
        vis_t[i] = rho[i]*sa_nu[i]*fv1;
        
#if (defined DEBUG) && !(defined FS_OPENMP)
        vis_t_min = MIN(vis_t_min,vis_t[i]);
        vis_t_max = MAX(vis_t_max,vis_t[i]);
#endif
        
        vis_t[i]  = MAX(vis_t[i],MIN_MUET_SA*amu);
        vis_t[i]  = MIN(vis_t[i],max_muet*amu);
    }
    
#ifdef DEBUG
#ifdef MPICH
    Parallel::parallel_min_max(vis_t_min, vis_t_max, MPI_COMM_WORLD);
#endif
    mflog::log.set_one_processor_out();
    mflog::log << "    vis_t_min = " << IOS_EP(8) << vis_t_min << "      vis_t_max = " << vis_t_max << endl;  
#endif
  
    SetGhostvis_t(grid,"SA");
  
#ifdef MPICH
    grid->CommInterfaceDataMPI(vis_t);
#endif  

}


/******************************************************************************\
                                 Limit nu 
\******************************************************************************/
void limitSA_nu(PolyGrid *grid)
{

    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell+nBFace;    
    
    RealFlow amu,rho00,nu_max,nu_min,nu_tmp;
    grid->GetData(&amu,   REAL_FLOW, 1, "amu");
    grid->GetData(&rho00, REAL_FLOW, 1, "rho");
    RealFlow max_muet;
    grid->GetData(&max_muet, REAL_FLOW, 1, "max_muet", 0);
    //湍流前1000步加强对nu_max的限制
    IntType level = grid->GetLevel();
    IntType iter_done,n_steps_coarse,step_count;
    step_count = 0;
    if(level == 0){
        grid->GetData(&iter_done, INT, 1 ,"iter_done");
        grid->GetData(&n_steps_coarse, INT, 1 ,"n_steps_coarse");
        step_count = iter_done-n_steps_coarse;
    }
    if(step_count<1000){
        max_muet = 1.0e5;
    }
    if(max_muet<0.0) max_muet=MAX_MUET_SA;
    nu_tmp = amu/rho00;
    nu_min = MIN_SA_NU*nu_tmp;
    nu_max = max_muet*nu_tmp;
    
    RealFlow *sa_nu = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "sa_nu"); 
#ifdef FS_OPENMP
#pragma omp parallel for
#endif      
    for(IntType i=0;i<nTCell;i++){
        sa_nu[i] = MAX(sa_nu[i],nu_min);
        sa_nu[i] = MIN(sa_nu[i],nu_max);
    }  
} 


/******************************************************************************\
                 Compute parameter omaga
\******************************************************************************/
void ComputeTurbGeneration_SA(PolyGrid *grid)
{
    IntType  nTCell = grid->GetNTCell();
    
    RealFlow *omaga = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "omaga");
    if(omaga == 0){
        mfmem::snew_array_1D(omaga, nTCell,dmrfl);
        grid->UpdateDataPtr(omaga, REAL_FLOW, nTCell, "omaga");
    }
    
    RealFlow *dqdx[3], *dqdy[3], *dqdz[3];
    GetVelocityGradient(grid, dqdx, dqdy, dqdz);
    
#ifdef FS_OPENMP
#pragma omp parallel for
#endif    
    for(IntType i=0; i<nTCell; i++){
        RealFlow dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz;
        dvdx = dqdx[1][i];
        dwdx = dqdx[2][i];
        dudy = dqdy[0][i];
        dwdy = dqdy[2][i];
        dudz = dqdz[0][i];
        dvdz = dqdz[1][i];
        
        omaga[i]  = sqrt((dwdy-dvdz)*(dwdy-dvdz)+(dudz-dwdx)*(dudz-dwdx)+(dvdx-dudy)*(dvdx-dudy));
    }    
}


/******************************************************************************\
                 Compute parameter gradnue2
\******************************************************************************/
void ComputeTurbInf_SA(PolyGrid *grid, const char *name)
{
    IntType   nTCell  = grid->GetNTCell();
    IntType   n       = nTCell + grid->GetNBFace();
    RealFlow  *sa_nu  = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "sa_nu");
    RealFlow  *dnutdx = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dnutdx");
    RealFlow  *dnutdy = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dnutdy");
    RealFlow  *dnutdz = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dnutdz");   
    RealFlow  *gradnue2 = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "gradnue2");
    
    if(gradnue2 == 0) {
        mfmem::snew_array_1D(gradnue2, nTCell,dmrfl);
        grid->UpdateDataPtr(gradnue2, REAL_FLOW, nTCell, "gradnue2");
    }
    if(dnutdx == 0) {
        dnutdy = NULL;
        dnutdz = NULL; 
        mfmem::snew_array_1D(dnutdx, n,dmrfl);
        mfmem::snew_array_1D(dnutdy, n,dmrfl);
        mfmem::snew_array_1D(dnutdz, n,dmrfl);
        grid->UpdateDataPtr(dnutdx, REAL_FLOW, n, "dnutdx");
        grid->UpdateDataPtr(dnutdy, REAL_FLOW, n, "dnutdy");
        grid->UpdateDataPtr(dnutdz, REAL_FLOW, n, "dnutdz");
    }

    RealFlow* u_n, * v_n, * w_n;
    IntType  nTNode = grid->GetNTNode();
    u_n = NULL;
    v_n = NULL;
    w_n = NULL;
	
#ifdef TIMECOST//dingxin
#ifdef FS_CUDA
	cudaDeviceSynchronize();
#endif
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemInvis, endtimeTemInvis;
    double timeuseTemInvis;
    gettimeofday(&starttimeTemInvis, 0); 
#endif
#endif
	
#ifdef FS_CUDA
	#if (defined MultiStream)
		// included CompGradientQ for SA and ViscousFluxScalar for SA
		cuCompGradientQ_SA_MultiStream(grid);		
	#else
		cuCompGradientQ(grid, sa_nu, dnutdx, dnutdy, dnutdz, 6, u_n, v_n, w_n);
	#endif
#else
	CompGradientQ(grid, sa_nu, dnutdx, dnutdy, dnutdz, 0, u_n, v_n, w_n);
#endif

#ifdef TIMECOST//dingxin
	#ifdef FS_CUDA
		cudaDeviceSynchronize();
	#endif
	#ifdef MPICH
		timecost[5] = timecost[5] + time_tmp + MPI_Wtime();
	#else
		gettimeofday(&endtimeTemInvis, 0); 
		timeuseTemInvis = (RealGeom) 1000000*(endtimeTemInvis.tv_sec - starttimeTemInvis.tv_sec) + endtimeTemInvis.tv_usec - starttimeTemInvis.tv_usec;
		timecost[5] += timeuseTemInvis;
		time_calvis += timeuseTemInvis / 1000000.0;
	#endif
#endif

#ifdef MPICH
	#ifdef FS_CUDA
		#if (defined MultiStream)
			IntType nvar = 1;
			grid->cuRecvSendVarNeighbor_TogethForGradient_SA_MultiStream(nvar); 
		#else
			IntType nvar = 1;
			grid->cuRecvSendVarNeighbor_TogethForGradient_SA(nvar); 
		#endif
	#else
		IntType nvar = 3;
		RealFlow *q_mpi[3];
		q_mpi[0] = dnutdx;
		q_mpi[1] = dnutdy;
		q_mpi[2] = dnutdz;
		grid->RecvSendVarNeighbor_Togeth(nvar, q_mpi);
	#endif
#endif
	
#ifdef FS_CUDA	
	//cuUpdateGhostGradSA(dnutdx, dnutdy, dnutdz);
	//cuComputeTurbInf_SA(gradnue2); // seems useless
#else
	#ifdef FS_OPENMP
	#pragma omp parallel for
	#endif	
    for(IntType i=0; i<nTCell; i++){
        gradnue2[i] = dnutdx[i]*dnutdx[i] + dnutdy[i]*dnutdy[i] + dnutdz[i]*dnutdz[i];  
    }
#endif

}

#undef CPP_FILD_ID  // clear out file id
} //~namespace mflow
