//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   turbulence.cpp
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

// direct head file
#include "turbulence.h"

// C++ build-in head files
#include <cmath>
#include <cassert>
#include <iostream>
using namespace std;

// user defined head files
#include "utility_functions.h"
#include "temporal_discretisation_implicit.h"
#include "solver_ns.h"
#include "io_base_format.h"
#include "io_log.h"
#include "parallel_base_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif
// 
#ifdef MPICH
#include "mpi.h"
#endif

//dingxin
#ifdef TIMECOST
extern double* timecost;
#endif
//TIMECOST

namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 12002  // define file id


#ifdef MPICH
    extern int myZone;
    extern int numprocs;
    extern MPI_Comm GridComm;  //for each grid, tangj
#endif


/******************************************************************************\
|   add source term
\******************************************************************************/
void AddSourceScalar(PolyGrid *grid, const char *name)
{
    if(strcmp(name, "sa_nu") == 0)    
        AddSourceSA(grid);
    
}


/******************************************************************************\
|   add source term for unsteady
\******************************************************************************/
void AddSourceScalarUnst(PolyGrid *grid, const char *name)
{
    if(strcmp(name, "sa_nu") == 0)    
        AddSourceUnstSA(grid);
    
}


/******************************************************************************\
|   
\******************************************************************************/
void ComputeTurbInf(PolyGrid *grid, const char *name)
{
    if(strcmp(name, "sa_nu") == 0)    
      ComputeTurbInf_SA(grid, name);

}


void CommUnstVolData(PolyGrid **grids, const char *name, const char *name_cur, const char *name_old)
{
    IntType i, g, n, nTCell, nBFace;
  
    for(g=0; g<1; g++) {
        PolyGrid *grid = (PolyGrid *) grids[g];
        nTCell = grid->GetNTCell();
        nBFace = grid->GetNBFace();
        n      = nTCell + nBFace;
        RealFlow *q     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
        RealFlow *q_cur = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, name_cur);
        RealFlow *q_old = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, name_old);
        for(i=0; i<nTCell; i++) q_old[i] = q_cur[i];
        for(i=0; i<nTCell; i++) q_cur[i] = q[i];
    }
}


/******************************************************************************\
|   compute source term for unsteady
\******************************************************************************/
void ComTubSourceUnst(PolyGrid *grid, const char *name, const char *name_cur, const char *name_old)
{
    IntType i, n, nTCell, nBFace;
    RealGeom *vol;
    RealFlow *res, *q, *q_cur, *q_old;

    RealFlow time_accuracy, real_dt;
    grid->GetData(&real_dt, REAL_FLOW, 1, "real_dt");
    grid->GetData(&time_accuracy, REAL_FLOW, 1, "time_accuracy");

    nTCell = grid->GetNTCell();
    nBFace = grid->GetNBFace();
    vol    = grid->GetCellVol();
    n      = nTCell + nBFace;
    q      = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
    q_cur  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, name_cur);
    q_old  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, name_old);
    res    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "res");    
    
    RealFlow *rho, *rho_cur, *rho_old;
    rho    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");  
    rho_cur= (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "rho_cur");  
    rho_old= (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "rho_old");  
  
    for(i=0; i<nTCell; i++){
        res[i] += (-(1.0+time_accuracy)*rho[i]*q[i]*vol[i] + (1.0+time_accuracy)*rho_cur[i]*q_cur[i]*vol[i]
                        + time_accuracy*(rho_cur[i]*q_cur[i]*vol[i] - rho_old[i]*q_old[i]*vol[i]))/real_dt;
    }
}


/*******************************************************************************\
  free lhs memory
\*******************************************************************************/
void FreeLHSMatScalar(PolyGrid *grid)
{
    IntType nTCell = grid->GetNTCell();
    RealFlow **lhsmat = (RealFlow **)grid->GetDataPtr(REAL_FLOW, nTCell, "lhsmat");

    if(lhsmat){
        mfmem::sdel_array_1D(lhsmat[0]);
       // delete [] lhsmat;
       grid->DeleteDataPtr("lhsmat");
    }
}


/*******************************************************************************\
  set ghost cell value for turbulent variable
\*******************************************************************************/
void GhostVariablesScalar(PolyGrid *grid, const char *name)
{
    if(strcmp(name,"sa_nu") == 0)
        GhostVariablesScalar_SA(grid);
    
}


/*******************************************************************************\
    Allocate memories for LHS matrices
\*******************************************************************************/
void InitLHSMatScalar(PolyGrid *grid)
{
    IntType nTCell = grid->GetNTCell();
    IntType n      = nTCell + grid->GetNBFace();
    IntType *nCPC  = CalnCPC(grid);
    RealGeom *vol  = grid->GetCellVol();
    
    RealFlow *rho  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    
    RealFlow **lhsmat = (RealFlow **)grid->GetDataPtr(REAL_FLOW, nTCell, "lhsmat");
    if(!lhsmat){ // Allocate memories
        mfmem::snew_array_1D(lhsmat,nTCell,dmrfl);
        assert(lhsmat != 0);
 
        IntType j = nTCell;
        for(IntType i=0; i<nTCell; i++) j += nCPC[i];
        lhsmat[0] = NULL;
        mfmem::snew_array_1D(lhsmat[0],j,dmrfl);
        for(IntType i=1; i<nTCell; i++)
            lhsmat[i] = &(lhsmat[i-1][nCPC[i-1] + 1]);
        grid->UpdateDataPtr(lhsmat, REAL_FLOW, nTCell, "lhsmat");
    }
   
    //turbulence time step = ns time step *turb_cfl_times
    RealFlow turb_cfl_times = 2.0;
    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
#ifdef FS_OPENMP
#pragma omp parallel for
#endif   
    for(IntType i=0; i<nTCell; i++){
        dt[i] *= turb_cfl_times;
    }
  
    IntType steady=1;
    RealFlow time_accuracy, real_dt;
    grid->GetData(&steady, INT, 1, "steady");
    grid->GetData(&real_dt, REAL_FLOW, 1, "real_dt");
    grid->GetData(&time_accuracy, REAL_FLOW, 1, "time_accuracy");
#ifdef FS_OPENMP
#pragma omp parallel for
#endif 
    for(IntType i=0; i<nTCell; i++) {
        for(IntType j=1; j<nCPC[i]+1; j++) {
            lhsmat[i][j] = 0.;
        }
        lhsmat[i][0] = vol[i]*rho[i]/dt[i];
        if(!steady) lhsmat[i][0] += (1.0+time_accuracy)*vol[i]*rho[i]/real_dt;
    } 
}


/******************************************************************************\
|
\******************************************************************************/
void InviscidFluxScalar(PolyGrid *grid, const char *name)
{
    IntType i, ns, ne, len, nVar;
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nTFace = grid->GetNTFace();
    IntType n      = nTCell + nBFace;
    
    //Get metrics
    RealGeom *xfn   = grid->GetXfn();
    RealGeom *yfn   = grid->GetYfn();
    RealGeom *zfn   = grid->GetZfn();
    RealGeom *area  = grid->GetFaceArea();
    RealGeom *vgn   = grid->GetFaceNormalVelocity();
    
    IntType steady = 1;
    grid->GetData(&steady, INT, 1, "steady");
    
    IntType turb_order = 1;
    grid->GetData(&turb_order, INT, 1, "turb_order");
    if(turb_order ==2){
        IntType iter_done=1;
        grid->GetData(&iter_done, INT, 1 ,"iter_done");
        if(iter_done<2000) turb_order = 1;       //前2000步用一阶
    }
    
    // Allocate temporary memories for ql, qr and flux
    RealFlow *ql[5], *qr[5], *flux, *q[5], *dqdl, *dqdr;
    nVar=5;
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
    
    ql[0]    = NULL;
    qr[0]    = NULL;
    flux     = NULL;
    dqdl     = NULL;
    dqdr     = NULL;
    mfmem::snew_array_1D(ql[0],nTFace*nVar,dmrfl);
    mfmem::snew_array_1D(qr[0],nTFace*nVar,dmrfl);
    mfmem::snew_array_1D(flux ,nTFace,dmrfl);
    mfmem::snew_array_1D(dqdl ,nTFace,dmrfl);
    mfmem::snew_array_1D(dqdr ,nTFace,dmrfl);
    
    for(i=1; i<nVar; i++) {
        ql[i]   = &ql[i-1][nTFace];
        qr[i]   = &qr[i-1][nTFace];
    }  
    
    ns = 0;
    do {
        ne   = ns + nTFace;
        if(ne > nTFace) ne = nTFace;
        len = ne - ns;
        
        // Get left variables and right variables
        for(i=0; i<nVar; i++){
            SetQlQrUseQ(grid, q[i], ql[i], qr[i], ns, ne);
            if(turb_order==2 && i==nVar-1){
                CalcuQlQr_turb(grid, q[i], ql[i], qr[i], ns, ne, name);
            }
            ModQlQrBou_turb(grid, q[i], ql[i], qr[i], ns, ne, i);
        }
                
        ScalarFlux(ql, qr, flux, &xfn[ns], &yfn[ns], &zfn[ns],            
                   &area[ns], &vgn[ns], dqdl, dqdr, len, ns, ne, steady);
        
        // Load the fluxes to residuals
        LoadFlux(grid, &flux, 1, ns, ne);
        
        // Put Dq to the LHS matrices
        PutScalarDqToLhs(grid, dqdl, dqdr, ns, ne);
        
        ns  = ne;
    } while (ns < nTFace);
    mfmem::sdel_array_1D(ql[0]);
    mfmem::sdel_array_1D(qr[0]);
    mfmem::sdel_array_1D(flux);
    mfmem::sdel_array_1D(dqdl);
    mfmem::sdel_array_1D(dqdr);
} 


/*******************************************************************************\
            calculate ql and qr in 3D using dqdx, dqdy, dqdz
\*******************************************************************************/
void CalcuQlQr_turb(PolyGrid *grid, RealFlow *q, RealFlow *ql, RealFlow *qr, IntType ns, IntType ne, const char *name)
{
    IntType  nBFace = grid->GetNBFace();
    IntType  *f2c   = grid->Getf2c();
    RealGeom *xfc  = grid->GetXfc();
    RealGeom *yfc  = grid->GetYfc();
    RealGeom *zfc  = grid->GetZfc();
    RealGeom *xcc  = grid->GetXcc();
    RealGeom *ycc  = grid->GetYcc();
    RealGeom *zcc  = grid->GetZcc();
    BCRecord **bcr  = grid->Getbcr();

    //RealFlow qmax,qmin;
    
    //get gradient at cell centers
    RealFlow *qgrad[3];
    GetTurbGrad(grid, name, qgrad);
    
    // Determine if there are boundary faces.
    IntType nMid  = ns;
    if(ne <= nBFace) {
        // If all boundary faces
        nMid = ne;
    } else if(ns < nBFace) {
        // Part of them are boundary faces
        nMid = nBFace;
    }
#ifdef FS_OPENMP
#pragma omp parallel for
#endif    
    for(IntType face=ns; face<nMid; face++) {
        IntType  c1, c2, count, type;
        RealGeom dx, dy, dz;
        RealFlow tk;
        type = bcr[face]->GetType();
        count = 2*face;
        c1     = f2c[count];
        c2     = f2c[count+1];
        
        // Left one
        dx     = xfc[face] - xcc[c1];
        dy     = yfc[face] - ycc[c1];
        dz     = zfc[face] - zcc[c1];
        tk     = ql[face] + (qgrad[0][c1]*dx + qgrad[1][c1]*dy + qgrad[2][c1]*dz);
        if(tk > TINY) ql[face]  = tk;
        
        if (type == INTERFACE){
            // Right one
            dx     = xfc[face] - xcc[c2];
            dy     = yfc[face] - ycc[c2];
            dz     = zfc[face] - zcc[c2];
            tk     = qr[face] + (qgrad[0][c2]*dx + qgrad[1][c2]*dy + qgrad[2][c2]*dz);
            if(tk > TINY) qr[face]  = tk;
        }
    }        
#ifdef FS_OPENMP
#pragma omp parallel for
#endif    
    for(IntType face=nMid; face<ne; face++) {
        IntType  c1, c2, count;
        RealGeom dx, dy, dz;
        RealFlow tk;
        count = 2*face;
        c1     = f2c[count];
        c2     = f2c[count+1];
        
        // Left one
        dx     = xfc[face] - xcc[c1];
        dy     = yfc[face] - ycc[c1];
        dz     = zfc[face] - zcc[c1];
        tk     = ql[face] + (qgrad[0][c1]*dx + qgrad[1][c1]*dy + qgrad[2][c1]*dz);
        if(tk > TINY) ql[face]  = tk;
        
        // Right one
        dx     = xfc[face] - xcc[c2];
        dy     = yfc[face] - ycc[c2];
        dz     = zfc[face] - zcc[c2];
        tk     = qr[face] + (qgrad[0][c2]*dx + qgrad[1][c2]*dy + qgrad[2][c2]*dz);
        if(tk > TINY) qr[face]  = tk;
        
    }
}


/*******************************************************************************\
modify ql and qr on boundary
\*******************************************************************************/
void ModQlQrBou_turb(PolyGrid *grid, RealFlow *q, RealFlow *ql, RealFlow *qr, IntType ns, IntType ne, IntType n) 
{
    IntType *f2c   = grid->Getf2c();
    IntType nBFace = grid->GetNBFace(); 
    BCRecord   **bcr  = grid->Getbcr();
    IntType nMid;
    
    
    //Check if there are boundary faces. If no, return.
    if(ns >= nBFace) return;
    if(ne <= nBFace) 
        // If they are all boundary faces
        nMid = ne;
    else 
        // Part of them are boundary faces
        nMid = nBFace;
#ifdef FS_OPENMP
#pragma omp parallel for
#endif    
    for(IntType face=ns; face<nMid; face++){
        IntType c1, c2, type, count;
        RealFlow temm;
        type = bcr[face]->GetType();
        count = 2*face;
        if (type == INTERFACE) continue;
        c1 = f2c[count];
        c2 = f2c[count+1];
        
        if(type == SYMM && n==4){
            qr[face] = ql[face];
        }else{
            temm = (q[c1]+q[c2])*0.5;
            ql[face] = temm;
            qr[face] = temm;
        }
    }
} 


/******************************************************************************\
| Put dqdl and dqdr to the LHS matrices
\******************************************************************************/
void PutScalarDqToLhs(PolyGrid *grid, RealFlow *dqdl, RealFlow *dqdr, IntType ns, IntType ne)
{
    IntType i, j, c1, c2, nc1, nc2, count;
    IntType *f2c   = grid->Getf2c();
    IntType nTCell = grid->GetNTCell();
    IntType nTFace = grid->GetNTFace();
    IntType *fcptr = (IntType*)grid->GetDataPtr(INT, 2*nTFace, "fcptr");
    IntType  nBFace = grid->GetNBFace();
    if(!fcptr){  // Calculate fcptr
        CalCNNCF(grid);
        fcptr = (IntType*)grid->GetDataPtr(INT, 2*nTFace, "fcptr");
    }
  
    RealFlow **lhsmat = (RealFlow **)grid->GetDataPtr(REAL_FLOW, nTCell, "lhsmat");

#if (defined FS_OPENMP) && (defined GroupColor)
    if (grid->GroupColorSuccess) {
        IntType nBFace = grid->GetNBFace();
        IntType pfacenum = nBFace - grid->GetNIFace();
        IntType groupSize = grid->groupSize;
        IntType bfacegroup_num, ifacegroup_num;
        IntType startFace, endFace;
        bfacegroup_num = grid->bfacegroup.size();
        ifacegroup_num = grid->ifacegroup.size();
        //Boundary faces:
        for (IntType fcolor = 0; fcolor < bfacegroup_num; fcolor++) {
            if (!fcolor) {
                startFace = 0;
            }
            else {
                startFace = grid->bfacegroup[fcolor - 1];
            }
            endFace = grid->bfacegroup[fcolor];
#pragma omp parallel for private(i,count,c1,nc1) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                count = 2 * i;
                c1 = f2c[count];
                nc1 = fcptr[count];
                lhsmat[c1][0] += dqdl[i];
                if (nc1 > 0) lhsmat[c1][nc1] += dqdr[i];
            }
        }
        // zone boundary face
        count = 2 * pfacenum;
        for (i = pfacenum; i < nBFace; i++) {
            c1 = f2c[count];
            nc1 = fcptr[count];
            count += 2;
            lhsmat[c1][0] += dqdl[i];
            if (nc1 > 0) lhsmat[c1][nc1] += dqdr[i];
        }
        // Interior faces
        for (IntType fcolor = 0; fcolor < ifacegroup_num; fcolor++) {
            if (!fcolor) {
                startFace = nBFace;
            }
            else {
                startFace = grid->ifacegroup[fcolor - 1];
            }
            endFace = grid->ifacegroup[fcolor];
#pragma omp parallel for private(i,count,c1,c2,nc1,nc2) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                count = 2 * i;
                c1 = f2c[count];
                c2 = f2c[count + 1];
                nc1 = fcptr[count];
                nc2 = fcptr[count + 1];
                lhsmat[c1][0] += dqdl[i];
                if (nc1 > 0) lhsmat[c1][nc1] += dqdr[i];
                lhsmat[c2][0] -= dqdr[i];
                if (nc2 > 0) lhsmat[c2][nc2] -= dqdl[i];
            }
        }
}
    else {
        count = 2 * ns;
        for (i = ns; i < ne; i++) {
            nc1 = fcptr[count];
            c1 = f2c[count++];
            nc2 = fcptr[count];
            c2 = f2c[count++];
            j = i - ns;
            // For the left
            lhsmat[c1][0] += dqdl[j];
            if (nc1 > 0) lhsmat[c1][nc1] += dqdr[j];
            // For the right, c2 may be a ghost cell
            if (c2 < nTCell) {
                lhsmat[c2][0] -= dqdr[j];
                if (nc2 > 0) lhsmat[c2][nc2] -= dqdl[j];
            }
        }
    }

#elif (defined FS_OPENMP) && (defined FaceColoring)
    //face coloring information：
    IntType    bfacegroup_num, ifacegroup_num;
    IntType    *grid_bfacegroup, *grid_ifacegroup;
    ifacegroup_num = (*grid).ifacegroup.size();
    bfacegroup_num = (*grid).bfacegroup.size();
    grid_bfacegroup = NULL;
    grid_ifacegroup = NULL;
    mfmem::snew_array_1D(grid_bfacegroup, bfacegroup_num, dmrfl);
    mfmem::snew_array_1D(grid_ifacegroup, ifacegroup_num, dmrfl);
    for (IntType i = 0; i < bfacegroup_num; i++) {
        grid_bfacegroup[i] = (*grid).bfacegroup[i];
    }
    for (IntType i = 0; i < ifacegroup_num; i++){
        grid_ifacegroup[i] = (*grid).ifacegroup[i];
    }

    // Diagnal matrix
    IntType    nIFace = grid->GetNIFace();
    IntType     pfacenum = nBFace - nIFace;
    for (IntType fcolor = 0; fcolor < bfacegroup_num; fcolor++) {
        IntType startFace, endFace;
        if (fcolor == 0) {
            startFace = 0;
        }
        else {
            startFace = grid_bfacegroup[fcolor - 1];
        }
        endFace = grid_bfacegroup[fcolor];
#pragma omp parallel for
        for (IntType i = startFace; i < endFace; i++) {
            IntType count = 2*i;
            IntType nc1 = fcptr[count];
            IntType c1  = f2c[count];
            IntType nc2 = fcptr[count+1];
            IntType c2  = f2c[count+1];
            // For the left
            lhsmat[c1][0] += dqdl[i];
            if(nc1 > 0) lhsmat[c1][nc1] += dqdr[i];
            // For the right, c2 may be a ghost cell
            if(c2 < nTCell) {      
                lhsmat[c2][0] -= dqdr[i];
                if(nc2 > 0) lhsmat[c2][nc2] -= dqdl[i];
            }
        }
    }
#ifdef MPICH    
    for (IntType i = pfacenum; i < nBFace; i++) {
        IntType count = 2*i;
        IntType nc1 = fcptr[count];
        IntType c1  = f2c[count];
        IntType nc2 = fcptr[count+1];
        IntType c2  = f2c[count+1];
        // For the left
        lhsmat[c1][0] += dqdl[i];
        if(nc1 > 0) lhsmat[c1][nc1] += dqdr[i];
        // For the right, c2 may be a ghost cell
        if(c2 < nTCell) {      
            lhsmat[c2][0] -= dqdr[i];
            if(nc2 > 0) lhsmat[c2][nc2] -= dqdl[i];
        }
    }
#endif
    // Interior faces
    for (IntType fcolor = 0; fcolor < ifacegroup_num; fcolor++) {
        IntType startFace, endFace;
        if (fcolor == 0) {
            startFace = nBFace;
        }
        else {
            startFace = grid_ifacegroup[fcolor - 1];
        }
        endFace = grid_ifacegroup[fcolor];
#pragma omp parallel for        
        for (IntType i = startFace; i < endFace; i++) {
            IntType count = 2*i;
            IntType nc1 = fcptr[count];
            IntType c1  = f2c[count];
            IntType nc2 = fcptr[count+1];
            IntType c2  = f2c[count+1];
            // For the left
            lhsmat[c1][0] += dqdl[i];
            if(nc1 > 0) lhsmat[c1][nc1] += dqdr[i];
            // For the right, c2 may be a ghost cell
            if(c2 < nTCell) {      
                lhsmat[c2][0] -= dqdr[i];
                if(nc2 > 0) lhsmat[c2][nc2] -= dqdl[i];
            }
        }
    }
    mfmem::sdel_array_1D(grid_bfacegroup);
    mfmem::sdel_array_1D(grid_ifacegroup);
#elif (defined FS_OPENMP) && (defined Reduction)//Manual reduction
    IntType* nFPC = CalnFPC(grid);
    IntType** C2F = CalC2F(grid);
    IntType face;
#pragma omp parallel for private(i,j,count,c1,c2,nc1,nc2,face)
    for (i = 0; i < nTCell; i++) {
        for (j = 0; j < nFPC[i]; j++) {
            face = C2F[i][j];
            count = 2 * face;
            nc1 = fcptr[count];
            c1 = f2c[count++];
            nc2 = fcptr[count];
            c2 = f2c[count];
            if (i == c1) {
                lhsmat[c1][0] += dqdl[face];
                if (nc1 > 0) lhsmat[c1][nc1] += dqdr[face];
            }
            else if (i == c2) {
                lhsmat[c2][0] -= dqdr[face];
                if (nc2 > 0) lhsmat[c2][nc2] -= dqdl[face];
            }
            else {
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
    }
#elif (defined FS_OPENMP) && (defined DIVREP)//Division & replication
    IntType threads = grid->threads;
    IntType startFace, endFace, t, k, face;
    if (grid->DivRepSuccess) {
#pragma omp parallel for private(t,i,k,startFace,endFace,count,c1,c2,face,nc1,nc2)
        for (t = 0; t < threads; t++) {
            //Boundary faces
            startFace = grid->idx_pthreads_bface[t];
            endFace = grid->idx_pthreads_bface[t + 1];
            for (i = startFace; i < endFace; i++) {
                face = grid->id_division_bface[i];
                count = 2 * face;
                c1 = f2c[count];
                nc1 = fcptr[count];
                lhsmat[c1][0] += dqdl[face];
                if (nc1 > 0) lhsmat[c1][nc1] += dqdr[face];
            }
            //Interior faces
            startFace = grid->idx_pthreads_iface[t];
            endFace = grid->idx_pthreads_iface[t + 1];
            for (i = startFace; i < endFace; i++) {
                k = grid->id_division_iface[i];
                if (abs(k) < nTFace)
                    face = k;
                else
                    face = abs(k) - nTFace;
                count = 2 * face;
                c1 = f2c[count];
                c2 = f2c[count + 1];
                nc1 = fcptr[count];
                nc2 = fcptr[count + 1];
                if (abs(k) < nTFace) {// write back to c1 & c2
                    lhsmat[c1][0] += dqdl[face];
                    if (nc1 > 0) lhsmat[c1][nc1] += dqdr[face];
                    lhsmat[c2][0] -= dqdr[face];
                    if (nc2 > 0) lhsmat[c2][nc2] -= dqdl[face];
                }
                else {
                    if (k > 0) {//just write back to c1
                        lhsmat[c1][0] += dqdl[face];
                        if (nc1 > 0) lhsmat[c1][nc1] += dqdr[face];
                    }
                    else {//just write back to c2
                        lhsmat[c2][0] -= dqdr[face];
                        if (nc2 > 0) lhsmat[c2][nc2] -= dqdl[face];
                    }
                }
            }
        }
    }
#elif (defined FS_OPENMP) && (defined DIVCON) //D&C TREE
#pragma omp parallel
    {
    #pragma omp single nowait
        tree_traversal(grid->treeHead, lhsmat, dqdl, dqdr, f2c, fcptr);
    }
#else
    for(IntType i=ns; i<ne; i++) {
        IntType count = 2*i;
        IntType nc1 = fcptr[count];
        IntType c1  = f2c[count];
        IntType nc2 = fcptr[count+1];
        IntType c2  = f2c[count+1];
        // For the left
        lhsmat[c1][0] += dqdl[i];
        if(nc1 > 0) lhsmat[c1][nc1] += dqdr[i];
        // For the right, c2 may be a ghost cell
        if(c2 < nTCell) {      
            lhsmat[c2][0] -= dqdr[i];
            if(nc2 > 0) lhsmat[c2][nc2] -= dqdl[i];
        }
    }
#endif
}


/******************************************************************************\
|    
\******************************************************************************/
void ScalarFlux(RealFlow *ql[], RealFlow *qr[], RealFlow *flux, RealGeom *xfn, 
                RealGeom *yfn, RealGeom *zfn, RealGeom *area, RealGeom *vgn, RealFlow *dqdl, 
                RealFlow *dqdr, IntType len, IntType ns, IntType ne, IntType steady)
{
    RealFlow  *rhol,*ul,*vl,*wl,*kl;
    RealFlow  *rhor,*ur,*vr,*wr,*kr;

    rhol = ql[0];
    ul   = ql[1];
    vl   = ql[2];
    wl   = ql[3];
    kl   = ql[4];
 
    rhor = qr[0];
    ur   = qr[1];
    vr   = qr[2];
    wr   = qr[3];
    kr   = qr[4];
 
    // velocity based upwind
#ifdef FS_OPENMP
#pragma omp parallel for
#endif    
    for(IntType i=0; i<len; i++) {
        RealFlow vnl, vnr;
        vnl = ul[i]*xfn[i] + vl[i]*yfn[i] + wl[i]*zfn[i];
        if(!steady) vnl -= vgn[i];

        if(vnl > 0.) {
            dqdl[i] = rhol[i]*vnl*area[i];
            flux[i] = kl[i]*dqdl[i];
        } else {
            dqdl[i] = 0.;
            flux[i] = 0.;
        }
 
        vnr = ur[i]*xfn[i] + vr[i]*yfn[i] + wr[i]*zfn[i];
        if(!steady) vnr -= vgn[i];
        if(vnr < 0.) {
            dqdr[i]  = rhor[i]*vnr*area[i];
            flux[i] += kr[i]*dqdr[i];
        } else {
            dqdr[i] = 0.;
        }
    }  
}


/*******************************************************************************\
          Forward solution for one time step                                    
          We assume flow variables and grid matrices are all known
\*******************************************************************************/
void ScalarRelaxation(PolyGrid *grid, const char *name, RealFlow *rhs, IntType steps)
{
    IntType n;
    IntType nTCell = grid->GetNTCell();

    for(n=0; n<steps; n++){
        grid->UpdateData(&n, INT, 1, "turb_step");
        // zero lhs matrix
        InitLHSMatScalar(grid);

        // load variable "res" of grid with the array rhs
        PutResInGrid(grid, rhs, nTCell, "res");

        UpdateResidualScalar(grid, name);
    
        TimeIntegrationScalar(grid,name);
    }
    FreeLHSMatScalar(grid);
}


/******************************************************************************\
|       set vis_t at ghost cells
\******************************************************************************/
void SetGhostvis_t(PolyGrid *grid, const char *name)
{
    IntType  i, c1, c2, type; 
    RealFlow vn,vnb,vnf,cf,cc,gam,p_bar,vis_t00; 
    RealFlow rhoP,uP,vP,wP,pP,riemp, riemm; 
    IntType  nBFace = grid->GetNBFace();
    IntType  nTCell = grid->GetNTCell();
    IntType  n      = nTCell + nBFace;
    IntType  *f2c   = grid->Getf2c();
    BCRecord **bcr  = grid->Getbcr();
    RealFlow *vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    
    RealFlow *rho   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    RealFlow *u     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    RealFlow *v     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    RealFlow *w     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    RealFlow *p     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    
    RealGeom *xfn   = grid->GetXfn();
    RealGeom *yfn   = grid->GetYfn();
    RealGeom *zfn   = grid->GetZfn();
    
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&pP,    REAL_FLOW, 1, "p");
    grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
    grid->GetData(&uP,    REAL_FLOW, 1, "u");
    grid->GetData(&vP,    REAL_FLOW, 1, "v");
    grid->GetData(&wP,    REAL_FLOW, 1, "w");
    grid->GetData(&vis_t00, REAL_FLOW, 1, "vis_t00");
    
    IntType steady;
    grid->GetData(&steady, INT, 1, "steady");
    RealGeom *vgn = grid->GetFaceNormalVelocity();
    
    for(i=0; i<nBFace; i++) {
      type = bcr[i]->GetType();
        
      // Do nothing for interfaces.
      if(type == INTERFACE) continue;
        
      c1   = f2c[i+i];
      c2   = f2c[i+i+1];
      
      switch(type){
        case WALL:
               vis_t[c2] = -vis_t[c1];
             break;
             
        case SYMM:
             vis_t[c2] = vis_t[c1];
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
                
             if(vnb>0)  vis_t[c2] = vis_t[c1];
             else       vis_t[c2] = vis_t00;      
             break;
            
        default:
             printf("Error in SetGhostvis_t\n");
             break;
      }
    }      
}


/******************************************************************************\
                    set turbulence values for free stream 
               (come from CFL3D User's Manual(Version5.0) p297)
\******************************************************************************/
void Setturb00(PolyGrid *grid, const char *name)
{
    RealFlow rho00,amu,ainf,vis_t00;
    grid->GetData(&rho00,  REAL_FLOW, 1, "rho");
    grid->GetData(&amu,   REAL_FLOW, 1, "amu");
    grid->GetData(&ainf,  REAL_FLOW, 1, "ainf");
  
    if(strcmp(name,"SA") == 0){
        RealFlow sa_nu00,sa_nu00p3,fv1;
    
        sa_nu00   = 1.341946;
    
        sa_nu00p3 = sa_nu00*sa_nu00*sa_nu00;
        fv1       = sa_nu00p3/(sa_nu00p3+CV1P3);
        vis_t00   = sa_nu00*fv1;
        sa_nu00  *= amu/rho00;
        vis_t00  *= amu;
    
        grid->UpdateData(&sa_nu00, REAL_FLOW, 1, "sa_nu00");  
        grid->UpdateData(&vis_t00, REAL_FLOW, 1, "vis_t00");
        
        mflog::log.set_one_processor_out();
        mflog::log<<endl<<endl;
        mflog::log<<"******************************"<<endl;
        mflog::log<<" sa_nu00 = "<<sa_nu00<<endl;
        mflog::log<<" vis_t00 = "<<vis_t00<<endl;
        mflog::log<<"******************************"<<endl<<endl;

    }
}  


/**********************************************************************\
         solve the linear equations using LU-SGS method
                      ~~~ONE SWEEPS~~~~
\**********************************************************************/
/*
void SolveScalarLUSGS(PolyGrid *grid, RealFlow **lhsmat, RealFlow *dq, IntType *nCPC, IntType **c2c, IntType nTCell, const char *name)      
{
    IntType i, ilu, cell, cell2;
    IntType n = nTCell+grid->GetNBFace();
    RealFlow alph,flux;
    
    RealFlow rhoP,amu,ainf,q_min;
    grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
    grid->GetData(&amu,   REAL_FLOW, 1, "amu");
    grid->GetData(&ainf,  REAL_FLOW, 1, "ainf");
    
    if(strcmp(name,"sa_nu") == 0){
        q_min = MIN_SA_NU;
        q_min *= (amu/rhoP);
    }
    
    IntType DQ_limit = 1;
    grid->GetData(&DQ_limit, INT, 1, "DQ_limit");
    
    RealFlow *q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
    
    IntType *luorder = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGSCellOrder");
    IntType *layer = (IntType *)grid->GetDataPtr(INT, n, "LUSGSLayer");
    
    //the Forward Sweep
    for(ilu=0;ilu<nTCell;ilu++){
        cell = luorder[ilu];

        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            if(layer[cell2]>layer[cell]) continue;

            dq[cell] -= lhsmat[cell][i+1]*dq[cell2];
        }
        dq[cell] /= lhsmat[cell][0];
        
        //limit dq
        if(DQ_limit == 1){
            // do nothing!
        }else if(DQ_limit == 2){  
            if(q[cell]+dq[cell]<q_min){
                dq[cell] *= 0.1;
            }
            if(q[cell]+dq[cell]<q_min){
                dq[cell] *= 0.1;
            }
            if(q[cell]+dq[cell]<q_min){
                dq[cell] = 0.0;
            }
        }else if(DQ_limit == 3){
            dq[cell] = MAX(dq[cell],q_min-q[cell]);
        }else if(DQ_limit == 4){
            alph = q[cell]/(q[cell]+MAX(0.0,-dq[cell]));
            dq[cell] *= alph;
        }
    }
    
#ifdef MPICH
    grid->CommInterfaceDataMPI(dq);
#endif
    
    //the Backward Sweep
    for(ilu=nTCell-1;ilu>-1;ilu--){
        cell = luorder[ilu];

        flux = 0.0;
        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            if(layer[cell2]<layer[cell]) continue;

            flux += lhsmat[cell][i+1]*dq[cell2];
        }
        dq[cell] -= flux/lhsmat[cell][0];
        
        //limit dq
        if(DQ_limit == 1){
            // do nothing!
        }else if(DQ_limit == 2){  
            if(q[cell]+dq[cell]<q_min){
                dq[cell] *= 0.1;
            }
            if(q[cell]+dq[cell]<q_min){
                dq[cell] *= 0.1;
            }
            if(q[cell]+dq[cell]<q_min){
                dq[cell] = 0.0;
            }
        }else if(DQ_limit == 3){
            dq[cell] = MAX(dq[cell],q_min-q[cell]);
        }else if(DQ_limit == 4){
            alph = q[cell]/(q[cell]+MAX(0.0,-dq[cell]));
            dq[cell] *= alph;
        }
    }
} 
*/
void SolveScalarLUSGS(PolyGrid *grid, RealFlow **lhsmat, RealFlow *dq, IntType *nCPC, IntType **c2c, IntType nTCell, const char *name)      
{
    IntType i, ilu, cell, cell2;
    IntType n = nTCell+grid->GetNBFace();
    RealFlow alph,flux;
    
    RealFlow rhoP,amu,ainf,q_min;
    grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
    grid->GetData(&amu,   REAL_FLOW, 1, "amu");
    grid->GetData(&ainf,  REAL_FLOW, 1, "ainf");
#if (defined FS_OPENMP) && (defined CellColoring) 
    IntType *cellsPerlayer = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer");
#endif    
    if(strcmp(name,"sa_nu") == 0){
        q_min = MIN_SA_NU;
        q_min *= (amu/rhoP);
    }
    
    IntType DQ_limit = 1;
    grid->GetData(&DQ_limit, INT, 1, "DQ_limit");
    
    RealFlow *q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
    
    IntType *luorder = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGSCellOrder");
    IntType *layer = (IntType *)grid->GetDataPtr(INT, n, "LUSGSLayer");
    
    //the Forward Sweep: first step
    for(ilu=0;ilu<1;ilu++){
        cell = luorder[ilu];

        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            if(layer[cell2]>layer[cell]) continue;

            dq[cell] -= lhsmat[cell][i+1]*dq[cell2];
        }
        dq[cell] /= lhsmat[cell][0];
        
        //limit dq
        if(DQ_limit == 1){
            // do nothing!
        }else if(DQ_limit == 2){  
            if(q[cell]+dq[cell]<q_min){
                dq[cell] *= 0.1;
            }
            if(q[cell]+dq[cell]<q_min){
                dq[cell] *= 0.1;
            }
            if(q[cell]+dq[cell]<q_min){
                dq[cell] = 0.0;
            }
        }else if(DQ_limit == 3){
            dq[cell] = MAX(dq[cell],q_min-q[cell]);
        }else if(DQ_limit == 4){
            RealFlow alph = q[cell]/(q[cell]+MAX(0.0,-dq[cell]));
            dq[cell] *= alph;
        }
    }
#if (defined FS_OPENMP) && (defined CellColoring) 
    for(IntType laynum=0; laynum<cellsPerlayer[0]; laynum++ ){
        IntType start = cellsPerlayer[laynum+1];
        IntType end   = cellsPerlayer[laynum+2];
        if(laynum == 0) {start++;}
#pragma omp parallel for
        for(IntType ilu=start; ilu<end; ilu++){
            IntType cell = luorder[ilu];
            for(IntType i=0;i<nCPC[cell];i++){
                IntType cell2 = c2c[cell][i];
                if(layer[cell2]>layer[cell]) continue;

                dq[cell] -= lhsmat[cell][i+1]*dq[cell2];
            }
            dq[cell] /= lhsmat[cell][0];
        
            //limit dq
            if(DQ_limit == 1){
                // do nothing!
            }else if(DQ_limit == 2){  
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] *= 0.1;
                }
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] *= 0.1;
                }
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] = 0.0;
                }
            }else if(DQ_limit == 3){
                dq[cell] = MAX(dq[cell],q_min-q[cell]);
            }else if(DQ_limit == 4){
                RealFlow alph = q[cell]/(q[cell]+MAX(0.0,-dq[cell]));
                dq[cell] *= alph;
            }
        }

    }
#else
    for(ilu=1;ilu<nTCell;ilu++){
        cell = luorder[ilu];

        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            if(layer[cell2]>layer[cell]) continue;

            dq[cell] -= lhsmat[cell][i+1]*dq[cell2];
        }
        dq[cell] /= lhsmat[cell][0];
        
        //limit dq
        if(DQ_limit == 1){
            // do nothing!
        }else if(DQ_limit == 2){  
            if(q[cell]+dq[cell]<q_min){
                dq[cell] *= 0.1;
            }
            if(q[cell]+dq[cell]<q_min){
                dq[cell] *= 0.1;
            }
            if(q[cell]+dq[cell]<q_min){
                dq[cell] = 0.0;
            }
        }else if(DQ_limit == 3){
            dq[cell] = MAX(dq[cell],q_min-q[cell]);
        }else if(DQ_limit == 4){
            alph = q[cell]/(q[cell]+MAX(0.0,-dq[cell]));
            dq[cell] *= alph;
        }
    }
#endif    
#ifdef MPICH
    grid->CommInterfaceDataMPI(dq);
#endif

#if (defined FS_OPENMP) && (defined CellColoring) 
    for(IntType laynum=cellsPerlayer[0]-1; laynum>=0; laynum-- ){
        IntType start = cellsPerlayer[laynum+2];
        IntType end   = cellsPerlayer[laynum+1];
#pragma omp parallel for
        for(IntType ilu=start-1; ilu>=end; ilu--){
            IntType cell = luorder[ilu];

            RealFlow flux = 0.0;
            for(IntType i=0;i<nCPC[cell];i++){
                IntType cell2 = c2c[cell][i];
                if(layer[cell2]<layer[cell]) continue;

                flux += lhsmat[cell][i+1]*dq[cell2];
            }
            dq[cell] -= flux/lhsmat[cell][0];
        
            //limit dq
            if(DQ_limit == 1){
                // do nothing!
            }else if(DQ_limit == 2){  
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] *= 0.1;
                }
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] *= 0.1;
                }
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] = 0.0;
                }
            }else if(DQ_limit == 3){
                dq[cell] = MAX(dq[cell],q_min-q[cell]);
            }else if(DQ_limit == 4){
                RealFlow alph = q[cell]/(q[cell]+MAX(0.0,-dq[cell]));
                dq[cell] *= alph;
            }
        }
    }
#else
    //the Backward Sweep
    for(ilu=nTCell-1;ilu>-1;ilu--){
        cell = luorder[ilu];

        flux = 0.0;
        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            if(layer[cell2]<layer[cell]) continue;

            flux += lhsmat[cell][i+1]*dq[cell2];
        }
        dq[cell] -= flux/lhsmat[cell][0];
        
        //limit dq
        if(DQ_limit == 1){
            // do nothing!
        }else if(DQ_limit == 2){  
            if(q[cell]+dq[cell]<q_min){
                dq[cell] *= 0.1;
            }
            if(q[cell]+dq[cell]<q_min){
                dq[cell] *= 0.1;
            }
            if(q[cell]+dq[cell]<q_min){
                dq[cell] = 0.0;
            }
        }else if(DQ_limit == 3){
            dq[cell] = MAX(dq[cell],q_min-q[cell]);
        }else if(DQ_limit == 4){
            alph = q[cell]/(q[cell]+MAX(0.0,-dq[cell]));
            dq[cell] *= alph;
        }
    }
#endif
} 

/*******************************************************************************\
    计算湍流LUSGS的附加项: L*(D-1)*U*DQ
\*******************************************************************************/
void AdditionTermforScalarLUSGS(PolyGrid *grid, RealFlow *AddTerm, RealFlow **lhsmat, RealFlow *dq, 
                                IntType *nCPC, IntType **c2c, IntType nTCell, const char *name)
{
    IntType i, ilu, cell, cell2;
    
    IntType n = nTCell+grid->GetNBFace();
    
    IntType *luorder = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGSCellOrder");
    IntType *layer = (IntType *)grid->GetDataPtr(INT, n, "LUSGSLayer");
    
    RealFlow *D_1UDQ = NULL;
    mfmem::snew_array_1D(D_1UDQ,nTCell,dmrfl);
    for(i=0;i<nTCell;i++){
        D_1UDQ[i] = 0.0;
    }
    
    //赋初值
    for(i=0;i<nTCell;i++){
        AddTerm[i] = 0.0;
    }
    
    //后扫描，求D-1*U*DQ
    for(ilu=nTCell-1;ilu>=0;ilu--){
        cell = luorder[ilu];

        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            if(layer[cell2]<layer[cell]) continue;

            D_1UDQ[cell] += lhsmat[cell][i+1]*dq[cell2];
        }
    }
        
    //D-1*(U*DQ)
    for(cell=0;cell<nTCell;cell++){
        D_1UDQ[cell] /= lhsmat[cell][0];
    }
            
    //前扫描，求L*(D-1*(U*DQ))      
    for(ilu=0;ilu<nTCell;ilu++){
        cell = luorder[ilu];

        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            if(layer[cell2]>layer[cell]) continue;
         
            AddTerm[cell] += lhsmat[cell][i+1]*D_1UDQ[cell2];
        }
    }
    
    mfmem::sdel_array_1D(D_1UDQ);
}


/**********************************************************************\
         solve the linear equations using LU-SGS method
                      ~~~MANY SWEEPS~~~~
\**********************************************************************/
void SolveScalarLUSGS(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, 
                      RealFlow *dq, IntType *nCPC, IntType **c2c, IntType nTCell, const char *name, 
                      IntType Nsweep, RealFlow epsilon)
{
    IntType i, ilu, cell, cell2, sweep;
    IntType n = nTCell+grid->GetNBFace();
    RealFlow alph;
    
    RealFlow rhoP,amu,ainf,q_min;
    grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
    grid->GetData(&amu,   REAL_FLOW, 1, "amu");
    grid->GetData(&ainf,  REAL_FLOW, 1, "ainf");
    
    if(strcmp(name,"sa_nu") == 0){
        q_min = MIN_SA_NU;
        q_min *= (amu/rhoP);
    }
    
    IntType DQ_limit = 1;
    grid->GetData(&DQ_limit, INT, 1, "DQ_limit");
    
    RealFlow *q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
    
    RealFlow norm0, norm, dmax = 1.0, DQO, flux_tmp,tmp;
    RealFlow *dqo = NULL;
    mfmem::snew_array_1D(dqo,nTCell,dmrfl);
    assert(dqo != 0);
    for(i=0; i<nTCell; i++){
        dqo[i] = 0.0;
    }
    
    IntType *luorder = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGSCellOrder");
    IntType *layer = (IntType *)grid->GetDataPtr(INT, n, "LUSGSLayer");
    
    for(sweep=0; sweep<Nsweep; sweep++){
        norm = 0.0;
        //the Forward Sweep
        for(ilu=0;ilu<nTCell;ilu++){
            cell = luorder[ilu];

            DQO = dq[cell];
            dq[cell]  = res[cell]-dqo[cell];
            dqo[cell] = 0.0;
            
            for(i=0;i<nCPC[cell];i++){
                cell2 = c2c[cell][i];
                if(layer[cell2]>layer[cell]) continue;

                flux_tmp = lhsmat[cell][i+1]*dq[cell2];
                dq[cell] -= flux_tmp;
                dqo[cell] += flux_tmp;
            }
            dq[cell] /= lhsmat[cell][0];
            tmp      = dq[cell] - DQO;
            norm    += tmp*tmp;
        
            //limit dq
            if(DQ_limit == 1){
                // do nothing!
            }else if(DQ_limit == 2){  
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] *= 0.1;
                }
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] *= 0.1;
                }
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] = 0.0;
                }
            }else if(DQ_limit == 3){
                dq[cell] = MAX(dq[cell],q_min-q[cell]);
            }else if(DQ_limit == 4){
                alph = q[cell]/(q[cell]+MAX(0.0,-dq[cell]));
                dq[cell] *= alph;
            }
        }
    
#ifdef MPICH
        grid->CommInterfaceDataMPI(dq);
#endif
    
        //the Backward Sweep
        for(ilu=nTCell-1;ilu>-1;ilu--){
            cell = luorder[ilu];

            DQO = dq[cell];
            dq[cell]  = res[cell]-dqo[cell];
            dqo[cell] = 0.0;
            
            for(i=0;i<nCPC[cell];i++){
                cell2 = c2c[cell][i];
                if(layer[cell2]<layer[cell]) continue;

                flux_tmp = lhsmat[cell][i+1]*dq[cell2]; 
                dq[cell] -= flux_tmp;
                dqo[cell] += flux_tmp;
            }
            dq[cell] /= lhsmat[cell][0];
            tmp       = dq[cell] - DQO;
            norm     += tmp*tmp;
        
            //limit dq
            if(DQ_limit == 1){
                // do nothing!
            }else if(DQ_limit == 2){  
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] *= 0.1;
                }
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] *= 0.1;
                }
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] = 0.0;
                }
            }else if(DQ_limit == 3){
                dq[cell] = MAX(dq[cell],q_min-q[cell]);
            }else if(DQ_limit == 4){
                alph = q[cell]/(q[cell]+MAX(0.0,-dq[cell]));
                dq[cell] *= alph;
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

#ifdef DEBUG
#ifdef MPICH
    if(myZone == 1) printf("Turb resi reduced by %.5e with %d sweeps\n", dmax, (int)sweep);
#else   
    printf("Turb resi reduced by %.5e with %d sweeps\n", dmax, (int)sweep);
#endif
#endif
    
    mfmem::sdel_array_1D(dqo);

}

void SolveScalarDPLUR(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res,
                      RealFlow *dq, IntType *nCPC, IntType **c2c, const char *name, IntType level) {
    IntType nTCell = grid->GetNTCell();
    IntType n = nTCell+grid->GetNBFace();
    
    RealFlow rhoP, amu, ainf, q_min;
    grid->GetData(&rhoP, REAL_FLOW, 1, "rho");
    grid->GetData(&amu, REAL_FLOW, 1, "amu");
    grid->GetData(&ainf, REAL_FLOW, 1, "ainf");

    if (strcmp(name, "sa_nu") == 0) {
        q_min = MIN_SA_NU;
        q_min *= (amu / rhoP);
    }

    IntType DQ_limit = 1;
    grid->GetData(&DQ_limit, INT, 1, "DQ_limit");
    IntType sweeps = 1;
    grid->GetData(&sweeps, INT, 1, "sweeps");

    RealFlow* q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);

    RealFlow* DQ0 = NULL;
    mfmem::snew_array_1D(DQ0, n, dmrfl);
    for (IntType i = 0; i < n; ++i) {
        DQ0[i] = 0;
    }

    IntType* luorder = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGSCellOrder");


    // �����ֵ
    for (IntType i = 0; i < nTCell; ++i) {
        dq[i] = res[i] / lhsmat[i][0];
    }


    // Jacobi ����
    for (IntType idx_sweep = 0; idx_sweep < sweeps; ++idx_sweep) {
        for (IntType i = 0; i < nTCell; ++i) {
            DQ0[i] = dq[i];
            dq[i] = res[i];
        }
#ifdef MPICH
        grid->CommInterfaceDataMPI(DQ0);
#endif

        for (IntType ilu = 0; ilu < nTCell; ++ilu) {
            IntType cell = luorder[ilu];

            for (IntType i = 0; i < nCPC[cell]; ++i) {
                IntType c2 = c2c[cell][i];

                dq[cell] -= lhsmat[cell][i + 1] * DQ0[c2];
            }
            dq[cell] /= lhsmat[cell][0];

            //limit dq
            if (DQ_limit == 1) {
                // do nothing!
            }
            else if (DQ_limit == 2) {
                if (q[cell] + dq[cell] < q_min) {
                    dq[cell] *= 0.1;
                }
                if (q[cell] + dq[cell] < q_min) {
                    dq[cell] *= 0.1;
                }
                if (q[cell] + dq[cell] < q_min) {
                    dq[cell] = 0.0;
                }
            }
            else if (DQ_limit == 3) {
                dq[cell] = MAX(dq[cell], q_min - q[cell]);
            }
            else if (DQ_limit == 4) {
                RealFlow alph = q[cell] / (q[cell] + MAX(0.0, -dq[cell]));
                dq[cell] *= alph;
            }
        }
    }

    mfmem::sdel_array_1D(DQ0);
}

void SolveScalarGMRES(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, RealFlow *dq, IntType *nCPC, IntType **c2c, const char *name, IntType level)  
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    IntType i, j, k, sweep, cell;
  
    // Control parameters
    IntType Adu=1, kspan = 10, Nsweeps = 5;
    grid->GetData(&Adu, INT, 1, "ADU");
    grid->GetData(&kspan, INT, 1, "kspan");
    grid->GetData(&Nsweeps, INT, 1, "gmresweeps");
    RealFlow Error = 0.;
    grid->GetData(&Error, REAL_FLOW, 1, "gmresepsilon");
    if(Error < TINY) Error = 1.0e-2;

    IntType nvar = 1;
    IntType len = nvar*nTCell;

    // Temporary memories
    RealFlow *reso  = NULL;
    mfmem::snew_array_1D(reso,len,dmrfl);
    assert(reso != 0);

    RealFlow **H = NULL;
    mfmem::snew_array_2D(H,kspan+1,kspan,dmrfl,true);
    RealFlow **v = NULL;
    mfmem::snew_array_2D(v,kspan+1,len,dmrfl,true);
    RealFlow *w  = NULL;
    mfmem::snew_array_1D(w,len,dmrfl);
    RealFlow *cs = NULL;
    mfmem::snew_array_1D(cs,kspan,dmrfl);
    RealFlow *sn = NULL;
    mfmem::snew_array_1D(sn,kspan,dmrfl);
    RealFlow *s  = NULL;
    mfmem::snew_array_1D(s,kspan+1,dmrfl);
 
    RealFlow norm0, norm, dmax;
 
    // Save the beginning flow variables
    RealFlow *DQTurb = NULL;
    mfmem::snew_array_1D(DQTurb,n,dmrfl);
    for(j=0; j<n; j++) DQTurb[j] = 0.0;
    RealFlow *DQoTurb = NULL;
    mfmem::snew_array_1D(DQoTurb,n,dmrfl);
    for(i=0; i<len; i++){
        dq[i]   = 0.;
    }

    // Save the residuals and Initialize Matrix*(Delta q) and p[0]
    for(i=0; i<len; i++) 
        reso[i] = res[i];
    
    PreconditScalarLUSGS(grid, lhsmat, res, DQTurb, nCPC, c2c);

    for(i=0; i<nTCell; i++){
        DQoTurb[i] = DQTurb[i];
        v[0][i] = DQoTurb[i];
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
        for(sweep=0; sweep<Nsweeps; sweep++){
            norm = DotProductMPI(v[0], v[0], len);
            norm = sqrt(norm);

            for(k=0; k<kspan+1; k++) s[k]=0.0;
            s[0] = norm; 
            for(i=0; i<len; i++) 
                v[0][i] /= norm;               //v=gamma/beta
      
            // Loop over the search directions
            for(k=0; k<kspan; k++){
                // Calculate the epsilon in evaluating matrix * vector
                ComputeScalarADU(grid, lhsmat, res, v[k], nCPC, c2c);

                PreconditScalarLUSGS(grid, lhsmat, res, DQTurb, nCPC, c2c);
                
                for(i=0; i<nTCell; i++){
                    w[i] = DQTurb[i];
                }
                        
                // Calculate H
                for(j=0; j<=k; j++){
                    H[j][k] = DotProductMPI(w, v[j], len);
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
            ComputeScalarADU(grid, lhsmat, res, dq, nCPC, c2c);

            // Calculate Matrix*(Delta q) and P[0] for next sweep
            PreconditScalarLUSGS(grid, lhsmat, res, DQTurb, nCPC, c2c);

            for(i=0; i<nTCell; i++){
                v[0][i] = DQoTurb[i] - DQTurb[i];
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

        for(i=0; i<len; i++) 
            res[i] = reso[i];

        //对dq进行限制修改
        IntType DQ_limit = 1;
        RealFlow alph, rhoP, amu, ainf, q_min;
        grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
        grid->GetData(&amu,   REAL_FLOW, 1, "amu");
        grid->GetData(&ainf,  REAL_FLOW, 1, "ainf");
        grid->GetData(&DQ_limit, INT, 1, "DQ_limit");
        RealFlow *q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);

        if(strcmp(name,"sa_nu") == 0){
            q_min = MIN_SA_NU;
            q_min *= (amu/rhoP);
        }
   
        //limit dq
        for(cell=0; cell<n; cell++){
            if(DQ_limit == 1){
                // do nothing!
            }else if(DQ_limit == 2){  
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] *= 0.1;
                }
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] *= 0.1;
                }
                if(q[cell]+dq[cell]<q_min){
                    dq[cell] = 0.0;
                }
            }else if(DQ_limit == 3){
                dq[cell] = MAX(dq[cell],q_min-q[cell]);
            }else if(DQ_limit == 4){
                alph = q[cell]/(q[cell]+MAX(0.0,-dq[cell]));
                dq[cell] *= alph;
            }
        }
    }

    // Delete temporary memories
    mfmem::sdel_array_1D(reso);
    mfmem::sdel_array_1D(DQTurb);
    mfmem::sdel_array_1D(DQoTurb);
    mfmem::sdel_array_2D(H);
    mfmem::sdel_array_2D(v);
    mfmem::sdel_array_1D(w);
    mfmem::sdel_array_1D(cs);
    mfmem::sdel_array_1D(sn);
    mfmem::sdel_array_1D(s);
}


/*******************************************************************************\
            Execute the precondition for the GMRES solver on grid level                    
\*******************************************************************************/
void PreconditScalarLUSGS(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, RealFlow *dq, IntType *nCPC, IntType **c2c)
{
    IntType i, cell, cell2;
    IntType nTCell = grid->GetNTCell();
    IntType n=nTCell + grid->GetNBFace();
  
    RealFlow *turburhs = NULL;
    mfmem::snew_array_1D(turburhs,nTCell,dmrfl);
    for(cell=0;cell<nTCell;cell++){
        turburhs[cell]=res[cell];
    }
    for(cell=0;cell<n;cell++){
        dq[cell]=0.0;
    }
  
    //the Forward Sweep
    for(cell=0;cell<nTCell;cell++){
        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            if(cell2>cell) continue;

            turburhs[cell] -= lhsmat[cell][i+1]*dq[cell2];
        }
        dq[cell] = turburhs[cell]/lhsmat[cell][0];
    }
 
#ifdef MPICH
    grid->CommInterfaceDataMPI(dq);
#endif

    //the Backward Sweep
    for(cell=nTCell-1;cell>-1;cell--){
        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];

            if(cell2<cell) continue;
            turburhs[cell] -= lhsmat[cell][i+1]*dq[cell2];
        }
        dq[cell] = turburhs[cell]/lhsmat[cell][0];
    }
    mfmem::sdel_array_1D(turburhs);
} 


/*******************************************************************************\
            Execute the A*DU for the GMRES solver on grid level                    
\*******************************************************************************/
void ComputeScalarADU(PolyGrid *grid, RealFlow **lhsmat, RealFlow *res, RealFlow *v, IntType *nCPC, IntType **c2c)
{
    IntType i, cell, cell2;
    IntType nTCell = grid->GetNTCell();
    IntType n=nTCell + grid->GetNBFace();
    
    RealFlow *dq = NULL;
    mfmem::snew_array_1D(dq,n,dmrfl);
    for(cell=0;cell<nTCell;cell++) dq[cell]=v[cell];
    for(cell=nTCell; cell<n; cell++) dq[cell]=0.0;
#ifdef MPICH
    grid->CommInterfaceDataMPI(dq);
#endif
  
    //the Forward Sweep
    for(cell=0;cell<nTCell;cell++){
        res[cell] = lhsmat[cell][0]*dq[cell];
        for(i=0;i<nCPC[cell];i++){
            cell2 = c2c[cell][i];
            
            res[cell] += lhsmat[cell][i+1]*dq[cell2];
        }
    }

    mfmem::sdel_array_1D(dq);
} 


/*******************************************************************************\
            Execute the multi-grid solver on grid level                    
\*******************************************************************************/
void SolveScalarOnGrid(PolyGrid *grid, const char *name)
{
    IntType  nTCell = grid->GetNTCell();
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "res");
    
    IntType turb_substeps=1;
    grid->GetData(&turb_substeps, INT, 1, "turb_substeps", 0);
    ScalarRelaxation(grid, name, res, turb_substeps);     
}


/*******************************************************************************\
   Implicit Time Integration       
\*******************************************************************************/
void TimeIntegrationScalar(PolyGrid *grid, const char *name)
{
    IntType  nBFace   = grid->GetNBFace();
    IntType  nTCell   = grid->GetNTCell();
    IntType  n        = nTCell + nBFace;
    IntType  *nCPC    = CalnCPC(grid);
    IntType  **c2c    = CalC2C(grid);
    RealFlow *res     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "res");
    RealFlow **lhsmat = (RealFlow **)grid->GetDataPtr(REAL_FLOW,nTCell,"lhsmat");
    
    IntType gmres = 0;
    grid->GetData(&gmres, INT, 1, "GMRES", 0);
 
    // Implicit
    IntType i;
    IntType sweeps = 1;
    grid->GetData(&sweeps, INT, 1, "sweeps");
    RealFlow epsilon = 0.1;
    grid->GetData(&epsilon, REAL_FLOW, 1, "epsilon");
    if(epsilon < TINY) epsilon = 1.0e-1;
   
    RealFlow *dq = NULL;
    mfmem::snew_array_1D(dq,n,dmrfl);
    for(i=0;i<n;i++){
        dq[i] = 0.0;
    }
	
    /* if(gmres==1) {
        SolveScalarGMRES(grid, lhsmat, res, dq, nCPC, c2c, name, 0);
    } else { */
    	IntType tScheme;
        grid->GetData(&tScheme, INT, 1, "tScheme");
        if (tScheme == DPLUR) {
            SolveScalarDPLUR(grid, lhsmat, res, dq, nCPC, c2c, name, 0);
        }
        else if (tScheme == LU_SGS){
            if(sweeps == -1){
                //预估步
                for(i=0;i<nTCell;i++){
                    dq[i] = res[i];
                }
                SolveScalarLUSGS(grid, lhsmat, dq, nCPC, c2c, nTCell, name);
                //校正步
                //计算高阶项L(D-1)U(DQ)
                //需要先更新虚拟网格的值，因为在求高阶项中会用到虚拟网格的值，而在SolveScalarLUSGS函数中
                //后扫描后没有更新虚拟网格的DQ的值
#ifdef MPICH
                grid->CommInterfaceDataMPI(dq);
#endif

                //高阶项
                RealFlow *AddTerm = NULL;
                mfmem::snew_array_1D(AddTerm, nTCell, dmrfl);
                assert(AddTerm != 0);
                            
                AdditionTermforScalarLUSGS(grid, AddTerm, lhsmat, dq, nCPC, c2c, nTCell, name);
                for(i=0;i<nTCell;i++){
                    dq[i] = res[i]+AddTerm[i];
                }
                // Now the LU-SGS part
                SolveScalarLUSGS(grid, lhsmat, dq, nCPC, c2c, nTCell, name);
            
                mfmem::sdel_array_1D(AddTerm);
            }else if(sweeps == 1){
                for(i=0;i<nTCell;i++){
                    dq[i] = res[i];
                }
                SolveScalarLUSGS(grid, lhsmat, dq, nCPC, c2c, nTCell, name);
            }else{			
                SolveScalarLUSGS(grid, lhsmat, res, dq, nCPC, c2c, nTCell, name, sweeps, epsilon);
            }
        } 
  
    UpdateSolutionScalar_TAO(grid, dq, name);
   
    mfmem::sdel_array_1D(dq);
}


/******************************************************************************\
|     compute residual
\******************************************************************************/
void UpdateResidualScalar(PolyGrid *grid, const char *name)
{
    // Ghost cells
    GhostVariablesScalar(grid,name);
  
#ifdef MPICH
    IntType  n     = grid->GetNTCell() + grid->GetNBFace();
    RealFlow *turb = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
    grid->CommInterfaceDataMPI(turb);
#endif

    ComputeTurbInf(grid,name);

    InviscidFluxScalar(grid, name);

    ViscousFluxScalar(grid, name);

    AddSourceScalar(grid, name);

    IntType steady=1;
    grid->GetData(&steady, INT, 1, "steady");
    if(!steady) AddSourceScalarUnst(grid, name);
   
}


/******************************************************************************\
|   update variable value (Come from TAO)
\******************************************************************************/
void UpdateSolutionScalar_TAO(PolyGrid *grid, RealFlow *dq, const char *name)
{
    IntType nBFace = grid->GetNBFace();
    IntType nTCell = grid->GetNTCell();
    IntType n      = nTCell + nBFace;
    
    RealFlow dqmax_turb = 0.25;
    grid->GetData(&dqmax_turb, REAL_FLOW, 1, "dqmax_turb");
    
    RealFlow *q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
#ifdef FS_OPENMP
#pragma omp parallel for
#endif    
    for(IntType i=0;i<nTCell;i++){
        if(fabs(dq[i]/q[i])>dqmax_turb)
            dq[i] *= dqmax_turb*q[i]/fabs(dq[i]);
        q[i] += dq[i]; 
    }
    
   if(strcmp(name,"sa_nu") == 0){
        limitSA_nu(grid);
    }
}
/******************************************************************************\
|       now calculate the viscous flux
\******************************************************************************/
void ViscousFluxScalar(PolyGrid *grid, const char *name)
{
    ViscousFluxScalar3D_New3(grid, name);
  
    // Matrices from the viscous flux
    ViscousMatsScalar(grid, name);
}
/******************************************************************************\
|    now calculate the viscous flux using ~METHOD TEST~ in evaluating
     the derivatives of velocity along the face normal
\******************************************************************************/
void ViscousFluxScalar3D_New3(PolyGrid *grid, const char *name)
{
    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  nTFace = grid->GetNTFace();
    IntType  n      = nTCell + nBFace;
    IntType  *f2c   = grid->Getf2c();
    RealGeom *area  = grid->GetFaceArea();
    RealGeom *xfn   = grid->GetXfn();
    RealGeom *yfn   = grid->GetYfn();
    RealGeom *zfn   = grid->GetZfn();
    RealGeom *xfc   = grid->GetXfc();
    RealGeom *yfc   = grid->GetYfc();
    RealGeom *zfc   = grid->GetZfc();
    RealGeom *xcc   = grid->GetXcc();
    RealGeom *ycc   = grid->GetYcc();
    RealGeom *zcc   = grid->GetZcc();
    BCRecord **bcr  = grid->Getbcr();
  
    RealFlow rhoP,amu,ainf;
    grid->GetData(&rhoP,  REAL_FLOW, 1, "rho");
    grid->GetData(&amu,   REAL_FLOW, 1, "amu");
    grid->GetData(&ainf,  REAL_FLOW, 1, "ainf");
  
    IntType  i, c1, c2, type;  
    RealFlow q_min;
    RealFlow sigma;
    
    RealFlow *rho   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho"); 
    RealFlow *vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l"); 
    RealFlow *res   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "res");
   
    // get variable
    RealFlow *k = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);

    //tem variables:
    RealFlow *flux = NULL;
    RealFlow *tem = NULL;
    RealFlow *tem_c2 = NULL;
    mfmem::snew_array_1D(flux,nTFace,dmrfl);
    mfmem::snew_array_1D(tem,nTFace,dmrfl);
    mfmem::snew_array_1D(tem_c2,nTFace,dmrfl);
 
    if(strcmp(name,"sa_nu") == 0){
        sigma = 1.0/SIGMA_SA; 
        q_min = MIN_SA_NU;
        q_min *= (amu/rhoP);
    }

#ifdef FS_OPENMP
#pragma omp parallel for
#endif
    for(IntType i=0; i<nTFace; i++) {
        IntType  type;  
        RealFlow k_vis;
        IntType c1    = f2c[2*i];
        IntType c2    = f2c[2*i+1];
        RealGeom dtmp, d1, d2, x1, x2, y1, y2, z1, z2, angle1, angle2;  
        RealFlow k1, k2, kmid, dkd1, dkd2, dkdn;
        // positions
        x1 = xcc[c1]  - xfc[i];
        y1 = ycc[c1]  - yfc[i];
        z1 = zcc[c1]  - zfc[i];
        x2 = xcc[c2]  - xfc[i];
        y2 = ycc[c2]  - yfc[i];
        z2 = zcc[c2]  - zfc[i];
        d1 = x1*xfn[i] + y1*yfn[i] + z1*zfn[i];
        d2 = x2*xfn[i] + y2*yfn[i] + z2*zfn[i];
      
        dtmp = -d1/(sqrt(x1*x1+ y1*y1 + z1*z1) + TINY);
        if(dtmp >  1.0) dtmp =  1.0;
        if(dtmp < -1.0) dtmp = -1.0;
        angle1 = asin(dtmp)*180.0/PI;  
 
        dtmp = d2/(sqrt(x2*x2+ y2*y2 + z2*z2) + TINY);
        if(dtmp >  1.0) dtmp =  1.0;
        if(dtmp < -1.0) dtmp = -1.0;
        angle2 = asin(dtmp)*180.0/PI;   
      
        // quantities at points 1 and 2
        k1   = k[c1];  
        k2   = k[c2];
        kmid = 0.5*(k1 + k2);   
          
        dkdn  = 0.0;
        if(angle1 > 0.0 && angle2 > 0.0 && fabs(d1) > TINY && fabs(d2) > TINY) {
            dkd1 = (k1 - kmid)/d1;
            dkd2 = (k2 - kmid)/d2;

            dtmp = d1*d1 + d2*d2;
            d1   = d1*d1/dtmp;
            d2   = d2*d2/dtmp;
            dkdn = dkd1*d1 + dkd2*d2; 
        }

        if(i<nBFace){
            type = bcr[i]->GetType();
            if (type!=INTERFACE && type!=SYMM) {
                RealFlow dn = (xcc[c2]-xcc[c1])*xfn[i]+(ycc[c2]-ycc[c1])*yfn[i]
                + (zcc[c2]-zcc[c1])*zfn[i];
                dkdn = (k[c2]-k[c1])/dn;
            } 
        }  

        if(strcmp(name,"sa_nu") == 0)
            //k_vis = 0.5*sigma*(vis_l[c1]+vis_l[c2]+rho[c1]*k[c1]+rho[c2]*k[c2]);
            k_vis = 0.5*sigma*(vis_l[c1]+vis_l[c2]+(1.0+CB2)*(rho[c1]*k[c1]+rho[c2]*k[c2]));
        tem[i] = dkdn*area[i];
        flux[i]     = k_vis*tem[i];        
    }

    if(strcmp(name,"sa_nu") == 0){
        
        //for(IntType i=0; i<nTFace; i++) {
        //    tem[i] = -CB2*sigma*tem[i];
        //}
#ifdef FS_OPENMP
#pragma omp parallel for
#endif
        for(IntType i=0; i<nTFace; i++) {
            RealFlow factor_c1, factor_c2;
            IntType  c1    = f2c[i+i];
            IntType  c2    = f2c[i+i+1];
            tem[i] = -CB2*sigma*tem[i];
            factor_c1 = rho[c1]*k[c1];            
            if(i >= nBFace) {
                factor_c2 = rho[c2]*k[c2];
                tem_c2[i] = tem[i]*factor_c2;
                //res[c2] -= tem[i]*factor_c2;
            }
            //res[c1] += tem[i]*factor_c1;
            tem[i] = tem[i]*factor_c1;
        } 
    }

#if (defined FS_OPENMP) && (defined GroupColor)
    RealFlow factor_c1, factor_c2;
    if (grid->GroupColorSuccess) {
        IntType pfacenum = nBFace - grid->GetNIFace();
        IntType groupSize = grid->groupSize;
        IntType bfacegroup_num, ifacegroup_num;
        IntType startFace, endFace, count;
        bfacegroup_num = grid->bfacegroup.size();
        ifacegroup_num = grid->ifacegroup.size();
        //Boundary faces:
        for (IntType fcolor = 0; fcolor < bfacegroup_num; fcolor++) {
            if (!fcolor) {
                startFace = 0;
            }
            else {
                startFace = grid->bfacegroup[fcolor - 1];
            }
            endFace = grid->bfacegroup[fcolor];
#pragma omp parallel for private(i,factor_c1,c1) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                c1 = f2c[2 * i];
                factor_c1 = flux[i] + tem[i];
                res[c1] += factor_c1;
            }
        }
        // zone boundary face
        count = 2 * pfacenum;
        for (i = pfacenum; i < nBFace; i++) {
            c1 = f2c[count];
            count += 2;
            factor_c1 = flux[i] + tem[i];
            res[c1] += factor_c1;
        }
        // Interior faces
        for (IntType fcolor = 0; fcolor < ifacegroup_num; fcolor++) {
            if (!fcolor) {
                startFace = nBFace;
            }
            else {
                startFace = grid->ifacegroup[fcolor - 1];
            }
            endFace = grid->ifacegroup[fcolor];
#pragma omp parallel for private(i,count,c1,c2,factor_c1,factor_c2) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                count = 2 * i;
                c1 = f2c[count];
                c2 = f2c[count + 1];
                factor_c1 = flux[i] + tem[i];
                res[c1] += factor_c1;
                factor_c2 = flux[i] + tem_c2[i];
                res[c2] -= factor_c2;
            }
        }
    }
    else {
        for (i = 0; i < nTFace; i++) {
            c1 = f2c[i + i];
            c2 = f2c[i + i + 1];
            factor_c1 = flux[i] + tem[i];
            res[c1] += factor_c1;
            if (i >= nBFace) {
                factor_c2 = flux[i] + tem_c2[i];
                res[c2] -= factor_c2;
            }
        }
    }
#elif (defined FS_OPENMP) && (defined FaceColoring)
    //face coloring information：
    IntType    bfacegroup_num, ifacegroup_num;
    IntType    *grid_bfacegroup, *grid_ifacegroup;
    ifacegroup_num = (*grid).ifacegroup.size();
    bfacegroup_num = (*grid).bfacegroup.size();
    grid_bfacegroup = NULL;
    grid_ifacegroup = NULL;
    mfmem::snew_array_1D(grid_bfacegroup, bfacegroup_num, dmrfl);
    mfmem::snew_array_1D(grid_ifacegroup, ifacegroup_num, dmrfl);
    for (IntType i = 0; i < bfacegroup_num; i++) {
        grid_bfacegroup[i] = (*grid).bfacegroup[i];
    }
    for (IntType i = 0; i < ifacegroup_num; i++){
        grid_ifacegroup[i] = (*grid).ifacegroup[i];
    }
    for (IntType fcolor = 0; fcolor < bfacegroup_num; fcolor++) {
        IntType startFace, endFace;
        if (fcolor == 0) {
            startFace = 0;
        }
        else {
            startFace = grid_bfacegroup[fcolor - 1];
        }
        endFace = grid_bfacegroup[fcolor];
#pragma omp parallel for
        for (IntType i = startFace; i < endFace; i++) {
            IntType c1    = f2c[i+i];
            RealFlow factor_c1 = flux[i] + tem[i];
            res[c1] += factor_c1;
        }
    }
    // Interior faces
    for (IntType fcolor = 0; fcolor < ifacegroup_num; fcolor++) {
        IntType startFace, endFace;
        if (fcolor == 0) {
            startFace = nBFace;
        }
        else {
            startFace = grid_ifacegroup[fcolor - 1];
        }
        endFace = grid_ifacegroup[fcolor];
#pragma omp parallel for        
        for (IntType i = startFace; i < endFace; i++) {
            RealFlow factor_c1, factor_c2;
            IntType c1    = f2c[i+i];
            IntType c2    = f2c[i+i+1];
            factor_c1 = flux[i] + tem[i];
            res[c1] += factor_c1;
            factor_c2 = flux[i] + tem_c2[i];
            res[c2] -= factor_c2;
        }
    }
    mfmem::sdel_array_1D(grid_bfacegroup);
    mfmem::sdel_array_1D(grid_ifacegroup);
#elif (defined FS_OPENMP) && (defined Reduction)//Manual reduction
    RealFlow factor_c1, factor_c2;
    IntType* nFPC = CalnFPC(grid);
    IntType** C2F = CalC2F(grid);
    IntType j, face, count;
#pragma omp parallel for private(i,j,count,c1,c2,face,factor_c1,factor_c2)
    for (i = 0; i < nTCell; i++) {
        for (j = 0; j < nFPC[i]; j++) {
            face = C2F[i][j];
            count = 2 * face;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            if (i == c1) {
                factor_c1 = flux[face] + tem[face];
                res[i] += factor_c1;
            }
            else if (i == c2) {
                factor_c2 = flux[face] + tem_c2[face];
                res[i] -= factor_c2;
            }
            else {
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
    }
#elif (defined FS_OPENMP) && (defined DIVREP)//Division & replication
    IntType threads = grid->threads;
    RealFlow factor_c1, factor_c2;
    IntType startFace, endFace, t, id_conversion, face, count;
    if (grid->DivRepSuccess) {
#pragma omp parallel for private(t,i,id_conversion,startFace,endFace,count,c1,c2,face,factor_c1,factor_c2)
        for (t = 0; t < threads; t++) {
            //Boundary faces
            startFace = grid->idx_pthreads_bface[t];
            endFace = grid->idx_pthreads_bface[t + 1];
            for (i = startFace; i < endFace; i++) {
                face = grid->id_division_bface[i];
                count = 2 * face;
                c1 = f2c[count];
                factor_c1 = flux[face] + tem[face];
                res[c1] += factor_c1;
            }
            //Interior faces
            startFace = grid->idx_pthreads_iface[t];
            endFace = grid->idx_pthreads_iface[t + 1];
            for (i = startFace; i < endFace; i++) {
                id_conversion = grid->id_division_iface[i];
                if (abs(id_conversion) < nTFace)
                    face = id_conversion;
                else
                    face = abs(id_conversion) - nTFace;
                count = 2 * face;
                c1 = f2c[count];
                c2 = f2c[count + 1];

                if (abs(id_conversion) < nTFace) {// write back to c1 & c2
                    factor_c1 = flux[face] + tem[face];
                    res[c1] += factor_c1;
                    factor_c2 = flux[face] + tem_c2[face];
                    res[c2] -= factor_c2;
                }
                else {
                    if (id_conversion > 0) {//just write back to c1
                        factor_c1 = flux[face] + tem[face];
                        res[c1] += factor_c1;
                    }
                    else {//just write back to c2
                        factor_c2 = flux[face] + tem_c2[face];
                        res[c2] -= factor_c2;
                    }
                }
            }
        }
    }
#elif (defined FS_OPENMP) && (defined DIVCON) //D&C TREE
#pragma omp parallel
    {
    #pragma omp single nowait
        tree_traversal(grid->treeHead, res, flux, tem, tem_c2, f2c);
    }
    
#else
    for(IntType i=0; i<nTFace; i++) {
        RealFlow factor_c1, factor_c2;
        IntType c1    = f2c[i+i];
        IntType c2    = f2c[i+i+1];
        factor_c1 = flux[i] + tem[i];
        res[c1] += factor_c1;
        //res[c1] += flux[i] + tem[i];
        if(i >= nBFace){
            factor_c2 = flux[i] + tem_c2[i];
            res[c2] -= factor_c2;
            //res[c2] -= flux[i];
        } 
    }
#endif    
    mfmem::sdel_array_1D(flux);
    mfmem::sdel_array_1D(tem);
    mfmem::sdel_array_1D(tem_c2);       
}


/******************************************************************************\
|       calculate contribution of viscous term to LHS
\******************************************************************************/
void ViscousDqScalar(PolyGrid *grid, const char *name, RealFlow *dqdl,   RealFlow *dqdr, IntType ns, IntType ne)
{  
    
    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  n      = nTCell + nBFace;
    IntType  *f2c   = grid->Getf2c();
    
    RealFlow sigma;
    
    
    // get variable
    RealFlow *k  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name); 

    if(strcmp(name,"sa_nu") == 0)
        sigma = 1.0/SIGMA_SA; 
    
    RealFlow *rho   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    RealFlow *vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");

    RealGeom *area = grid->GetFaceArea();
    RealGeom *xfn  = grid->GetXfn();
    RealGeom *yfn  = grid->GetYfn();
    RealGeom *zfn  = grid->GetZfn();
    RealGeom *xfc  = grid->GetXfc();
    RealGeom *yfc  = grid->GetYfc();
    RealGeom *zfc  = grid->GetZfc();
    RealGeom *xcc  = grid->GetXcc();
    RealGeom *ycc  = grid->GetYcc();
    RealGeom *zcc  = grid->GetZcc();


#ifdef FS_OPENMP
#pragma omp parallel for
#endif    
    for(IntType i=ns; i<ne; i++) {
        RealGeom d1,d2,dtmp;
        RealFlow k_vis;
        IntType  j, c1, c2, count;
        count = 2*i;  
        //j       = i - ns;
        c1      = f2c[count];
        c2      = f2c[count+1];
 
        d1 = (xcc[c1] - xfc[i])*xfn[i] + (ycc[c1] - yfc[i])*yfn[i] + (zcc[c1] - zfc[i])*zfn[i];
        d2 = (xcc[c2] - xfc[i])*xfn[i] + (ycc[c2] - yfc[i])*yfn[i] + (zcc[c2] - zfc[i])*zfn[i];
      
        dtmp = 0.0;
        if(d1*d2 < 0 && fabs(d1) > TINY && fabs(d2) > TINY) {
            dtmp  = 0.5*fabs(d2-d1)/(d1*d1 + d2*d2);
        }
     
        if(strcmp(name,"sa_nu") == 0)
            //k_vis = 0.5*sigma*(vis_l[c1]+vis_l[c2]+rho[c1]*k[c1]+rho[c2]*k[c2]);
            k_vis = 0.5*sigma*(vis_l[c1]+vis_l[c2]+(1.0+CB2)*(rho[c1]*k[c1]+rho[c2]*k[c2]));
        
        dqdl[i] = k_vis*area[i]*dtmp;
        dqdr[i] =-dqdl[i];
      
        if(strcmp(name,"sa_nu") == 0){
            RealFlow tem;
            tem = -CB2*sigma*dtmp*area[i];
            dqdl[i] += tem*rho[c1]*k[c1];
            dqdr[i] -= tem*rho[c2]*k[c2];
        }
    } 
}


/******************************************************************************\
|       calculate viscous matrices
\******************************************************************************/
void ViscousMatsScalar(PolyGrid *grid, const char *name)
{
    IntType ns, ne;
    IntType nTFace = grid->GetNTFace();
   
    RealFlow *dqdl = NULL;
    RealFlow *dqdr = NULL;
    mfmem::snew_array_1D(dqdl,nTFace,dmrfl);
    mfmem::snew_array_1D(dqdr,nTFace,dmrfl);
    ns = 0;
    do {
        ne   = ns + nTFace;
        if(ne > nTFace) ne = nTFace;
 
        // Calculate Dfdq
        ViscousDqScalar(grid, name, dqdl, dqdr, ns, ne);		
		
        // Put Dq to the LHS matrices
        PutScalarDqToLhs(grid, dqdl, dqdr, ns, ne);
 
        ns  = ne;
    } while (ns < nTFace);
    mfmem::sdel_array_1D(dqdl);
    mfmem::sdel_array_1D(dqdr);
}


/*******************************************************************************\
        Get turbulence var gradient     
\*******************************************************************************/
void GetTurbGrad(PolyGrid *grid, const char *name, RealFlow *qgrad[3])
{
    IntType n = grid->GetNTCell() + grid->GetNBFace();
    if(strcmp(name,"sst_k") == 0 || strcmp(name,"kw_k") == 0 || strcmp(name,"ke_k") == 0 ){
        qgrad[0] = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dkdx");
        qgrad[1] = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dkdy");
        qgrad[2] = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dkdz");
    }else if(strcmp(name,"sst_w") == 0 || strcmp(name,"kw_w") == 0){
        qgrad[0] = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dwwdx");
        qgrad[1] = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dwwdy");
        qgrad[2] = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dwwdz");
    }else if(strcmp(name,"sa_nu") == 0){
        qgrad[0] = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dnutdx");
        qgrad[1] = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dnutdy");
        qgrad[2] = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dnutdz");
    }else if(strcmp(name,"ke_e") == 0){
        qgrad[0] = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dedx");
        qgrad[1] = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dedy");
        qgrad[2] = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "dedz");
    }
}


/*******************************************************************************\
  输出湍流模型方程的残差
  name控制输出的湍流模型方程变量名，mark控制是否是添加的第二个方程。
\*******************************************************************************/
void DumpTurbNormResi(PolyGrid *grid, IntType iter, IntType zn, RealFlow t_now, const char *name, IntType mark)
{
    IntType i;
    FILE    *fp;
    ShortString     resid;
    String char_tmp;
    RealFlow norm,resmax;
    IntType  key_file = 0;
    
    IntType nTCell = grid->GetNTCell();
    //Get the residual
    RealFlow *res = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "res");

    norm = 0.0;
    resmax = 0.0;
    for(i=0; i<nTCell; i++){
        norm += res[i]*res[i];
        resmax = MAX(resmax,fabs(res[i]));
    }    

#ifdef MPICH
    RealFlow total, resmax_glb;
    MPI_Allreduce(&norm, &total, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&resmax, &resmax_glb, 1, MPIReal, MPI_MAX, MPI_COMM_WORLD);
    norm = total;
    resmax = resmax_glb;
#endif
    norm = sqrt(norm);
    
    //zhyb：如果SA模型的resmax大于1.0e4，则认为湍流模型发散，退出程序
    if(strcmp(name,"SA")==0 && resmax>1.0e4){
        mflog::log.set_one_processor_out();
        mflog::log<<endl<<"SA model equation is divergence! Please decrease CFL number, test again!"<<endl;  
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
  
    //Open the file for norms of the residuals
#ifdef MPICH
    if(myZone == 1){
        sprintf(resid, "resid_turb_glb.out");
        if((fp = fopen(resid, "r")) == NULL){  //文件不存在
            key_file = 1;
        }else{
            fclose(fp);
        }
        if((iter==1 && mark==1) || key_file==1)
            fp = fopen(resid, "w");
        else
            fp = fopen(resid, "a");
        if(!fp){
            printf("Can't open file %s \n", resid);
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
#else
    sprintf(resid, "resid_turb_glb.out");
    if((fp = fopen(resid, "r")) == NULL){  //文件不存在
        key_file = 1;
    }else{
        fclose(fp);
    }
    if((iter==1 && mark==1) || key_file==1)
        fp = fopen(resid, "w");
    else
        fp = fopen(resid, "a");
    if(!fp) {
        printf("Can't open file %s \n", resid);
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
#endif
    
#ifdef MPICH
    if(myZone == 1){
        if(key_file==1 || (iter==1 && mark==1)){
           if(strcmp(name,"SA") == 0){   //一方程
               fprintf(fp, "#iter turb1_res  turb1_res_max");
           }else{  //两方程    
               fprintf(fp, "#iter turb1_res  turb1_res_max  turb2_res  turb2_res_max");
           }
        }
        if(mark == 1){ //第一个方程
            sprintf(char_tmp, "\n%5d %.4e %.4e", (int)iter, norm, resmax);
        }else if(mark == 2){ //第二个方程
            sprintf(char_tmp, " %.4e %.4e", norm, resmax); 
        }else{
            std::cerr<<endl<<"Error! mark="<<mark<<" in function DumpTurbNormResi! mark must be 1 or 2!"<<endl;
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
        //printf(char_tmp);
        fprintf(fp, char_tmp);
        fclose(fp);
    }
#else
    if(key_file==1 || (iter==1 && mark==1)){
        if(strcmp(name,"SA") == 0){   //一方程
            fprintf(fp, "#iter turb1_res  turb1_res_max");
        }else{  //两方程    
            fprintf(fp, "#iter turb1_res  turb1_res_max  turb2_res  turb2_res_max");
        }
    }
    if(mark == 1){ //第一个方程
        sprintf(char_tmp, "\n%5d %.4e %.4e", (int)iter, norm, resmax);
    }else if(mark == 2){ //第二个方程
        sprintf(char_tmp, " %.4e %.4e", norm, resmax);
    }else{
        std::cerr<<endl<<"Error! mark="<<mark<<" in function DumpTurbNormResi! mark must be 1 or 2!"<<endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    //printf(char_tmp);
    fprintf(fp, char_tmp);
    fclose(fp);
#endif
    
    IfNAN(char_tmp);  //如果NAN，退出程序  
}

#undef CPP_FILD_ID  // clear out file id
} //~namespace mflow
