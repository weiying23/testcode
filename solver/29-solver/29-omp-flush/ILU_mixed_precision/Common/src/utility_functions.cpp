//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   utility_functions.cpp
/// \brief  define some utility functions
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
#include "utility_functions.h"

// C++ build-in head files
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
using namespace std;

// other user defined head files
#include "number_type.h"
#include "algm.h"
#include "io_base_format.h"
#include "io_log.h"
#include "parallel_base_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"

// head files relying on condition-compiling 
#ifdef MPICH
#include <mpi.h>
#endif

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif

//for FaceColoring SIMD, based SSE, ruitian, 2021.12.28
//#include <immintrin.h>

//dingxin
#ifdef TIMECOST
extern double* timecost;
extern double  time_flux, time_invis, time_roe, time_vis, time_calvis;
extern double  time_limiter;
extern double  time_gradient;
extern double  time_lusgs;
#endif
//TIMECOST
namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 12101  // define file id


#ifdef MPICH
extern int numprocs;
extern int myZone;
extern MPI_Comm GridComm;  //for each grid, tangj
#endif

//****************************************************************************\
/// \brief  print the version information
///
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 2020-07-15  tangj      modify version information and use full name
/// </pre> 
//****************************************************************************/
void PrintVersion()
{
    mflog::log.set_one_processor_out();
    mflog::log << std::endl;
    mflog::log << "$#============================================================================#$" << std::endl;
    mflog::log << "$#                     National Numerical Windtunnel                          #$" << std::endl;
    mflog::log << "$#          FlowStar -- Flow Simulation Tools for Aerospace Research          #$" << std::endl;
    mflog::log << "$#                Computational Aerodynamics Institute(CAI)                   #$" << std::endl;
    mflog::log << "$#            China Aerodynamics Research&Development Center(CARDC)           #$" << std::endl;
    mflog::log << "$#                        MianYang SiChuan, CHINA                             #$" << std::endl;
    mflog::log << "$#                         C.All rights reserved.                             #$" << std::endl;
    mflog::log << "$#============================================================================#$" << std::endl;
    mflog::log << std::endl;
}


/*******************************************************************************\
       Zero the residuals. Also allocate memory for the residuals               
       if they had not been allocated                                           
\*******************************************************************************/
void ZeroGridResiduals(PolyGrid *grid, const char *name, IntType nVar)
{
    IntType    nTCell=grid->GetNTCell(), nT = nVar*nTCell;
    RealFlow   *res;

    res = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nT, name);
    if(!res) {
        mfmem::snew_array_1D(res,nT,dmrfl);
        assert(res != 0);
        grid->UpdateDataPtr(res, REAL_FLOW, nT, name);
    }
#ifdef FS_OPENMP
#pragma omp parallel for
#endif
    for(IntType i=0; i<nT; i++) {
        res[i] = 0.;
    }
}

/******************************************************************************\
  Copy the value of rhs to grid's variable named name and switch the sign
\******************************************************************************/
void PutResInGrid(PolyGrid *grid, RealFlow *rhs, IntType n, const char *name)
{
    IntType i;

    RealFlow *res = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);

    i = n-1;
    do {
      res[i] = -rhs[i];
    } while (--i >= 0);

}

/*******************************************************************************\
     Update residuals in cell with the fluxes at cell faces
\*******************************************************************************/
/*
void LoadFlux(PolyGrid *grid, RealFlow *flux[], IntType nVar, IntType ns, IntType ne) 
{
    IntType   face, i, j, c1, c2, count, nMid;
    IntType   nTCell = grid->GetNTCell();
    IntType   nBFace = grid->GetNBFace();
    IntType   *f2c   = grid->Getf2c();
      
    // Determine if there are boundary faces.
    nMid  = ns;
    if(ne  <= nBFace) {
       // If all boundary faces
       nMid = ne;
    } else if(ns < nBFace) {
       // Part of them are boundary faces
       nMid = nBFace;
    }
        
    // Get Residual
    RealFlow *res[5];
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nVar*nTCell, "res");
    for(i=1; i<nVar; i++) res[i] = &res[i-1][nTCell];
 
    // For boundary faces, remember c2 is ghost cell
    for(j=0; j<nVar; j++) {
       count = 2*ns;
       i     = 0;
       for(face=ns; face<nMid; face++) {
           c1  = f2c[count++];
           count++;
 
           res[j][c1] -= flux[j][i];
           i++;
       }
 
       // Interior faces
       for(face=nMid; face<ne; face++) {
           c1 = f2c[count++];
           c2 = f2c[count++];

           res[j][c1] -= flux[j][i];
           res[j][c2] += flux[j][i];
           i++;
       }
    }
}
*/
void LoadFlux(PolyGrid *grid, RealFlow *flux[], IntType nVar, IntType ns, IntType ne) 
{
    IntType   face, i, j, c1, c2, count, nMid;
    IntType   nTCell = grid->GetNTCell();
    IntType   nBFace = grid->GetNBFace();
    IntType   *f2c   = grid->Getf2c();
      
    // Determine if there are boundary faces.
    nMid  = ns;
    if(ne  <= nBFace) {
       // If all boundary faces
       nMid = ne;
    } else if(ns < nBFace) {
       // Part of them are boundary faces
       nMid = nBFace;
    }
        
    // Get Residual
    RealFlow *res[5];
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nVar*nTCell, "res");
    for(IntType i=1; i<nVar; i++) res[i] = &res[i-1][nTCell];

#if (defined FS_OPENMP) && (defined GroupColor)
    if (grid->GroupColorSuccess) {
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
#pragma omp parallel for private(i,j,count,c1) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                count = 2 * i;
                c1 = f2c[count];
                for (j = 0; j < nVar; j++) {
                    res[j][c1] -= flux[j][i];
                }
            }
        }
        // zone boundary face
        count = 2 * pfacenum;
        for (i = pfacenum; i < nBFace; i++) {
            c1 = f2c[count];
            count += 2;
            for (j = 0; j < nVar; j++) {
                res[j][c1] -= flux[j][i];
            }
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
#pragma omp parallel for private(i,j,count,c1,c2) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                count = 2 * i;
                c1 = f2c[count];
                c2 = f2c[count + 1];
                for (j = 0; j < nVar; j++) {
                    res[j][c1] -= flux[j][i];
                    res[j][c2] += flux[j][i];
                }
            }
        }
    }
    else {
        for (j = 0; j < nVar; j++) {
            count = 2 * ns;
            i = 0;
            for (face = ns; face < nMid; face++) {
                c1 = f2c[count++];
                count++;

                res[j][c1] -= flux[j][i];
                i++;
            }

            // Interior faces
            for (face = nMid; face < ne; face++) {
                c1 = f2c[count++];
                c2 = f2c[count++];

                res[j][c1] -= flux[j][i];
                res[j][c2] += flux[j][i];
                i++;
            }
        }
    }

#elif (defined FS_OPENMP) && (defined FaceColoring) 
    IntType    nIFace = grid->GetNIFace();
    IntType     pfacenum = nBFace - nIFace;
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

    // For boundary faces, remember c2 is ghost cell
    for(IntType j=0; j<nVar; j++) {
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
            for (IntType face = startFace; face < endFace; face++) {
                IntType count = 2*face;
                IntType c1  = f2c[count]; 
                res[j][c1] -= flux[j][face];
            }
        } 
#ifdef MPICH    
        for (IntType face = pfacenum; face < nBFace; face++) {    
            IntType count = 2*face;
            IntType c1  = f2c[count]; 
            res[j][c1] -= flux[j][face];
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
            for (IntType face = startFace; face < endFace; face++) {
                IntType count = 2*face;
                IntType c1 = f2c[count];
                IntType c2 = f2c[count+1];

                res[j][c1] -= flux[j][face];
                res[j][c2] += flux[j][face];
            }
        }
    }
    mfmem::sdel_array_1D(grid_bfacegroup);
    mfmem::sdel_array_1D(grid_ifacegroup);
	
#elif (defined FS_OPENMP) && (defined Reduction)//Manual reduction
    IntType* nFPC = CalnFPC(grid);
    IntType** C2F = CalC2F(grid);
    IntType k;
#pragma omp parallel for private(i,j,k,c1,c2,face)
    for (i = 0; i < nTCell; i++) {
        for (j = 0; j < nFPC[i]; j++) {
            face = C2F[i][j];
            c1 = f2c[face + face];
            c2 = f2c[face + face + 1];
            if (i == c1) {
                for (k = 0; k < nVar; k++) {
                    res[k][c1] -= flux[k][face];
                }
            }
            else if (i == c2) {
                for (k = 0; k < nVar; k++) {
                    res[k][c2] += flux[k][face];
                }
            }
            else {
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
    }

#elif (defined FS_OPENMP) && (defined DIVREP)//Division & replication
    IntType threads = grid->threads;
    IntType startFace, endFace, t, k;
    IntType nTFace = grid->GetNTFace();
    if (grid->DivRepSuccess) {
#pragma omp parallel for private(t,i,j,k,startFace,endFace,count,c1,c2,face)
        for (t = 0; t < threads; t++) {
            //Boundary faces
            startFace = grid->idx_pthreads_bface[t];
            endFace = grid->idx_pthreads_bface[t + 1];
            for (i = startFace; i < endFace; i++) {
                face = grid->id_division_bface[i];
                c1 = f2c[2 * face];
                for (j = 0; j < nVar; j++) {
                    res[j][c1] -= flux[j][face];
                }
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

                if (abs(k) < nTFace) {//write back to c1 & c2
                    for (j = 0; j < nVar; j++) {
                        res[j][c1] -= flux[j][face];
                        res[j][c2] += flux[j][face];
                    }
                }
                else {
                    if (k > 0) {//just write back to c1
                        for (j = 0; j < nVar; j++) {
                            res[j][c1] -= flux[j][face];
                        }
                    }
                    else {//just write back to c2
                        for (j = 0; j < nVar; j++) {
                            res[j][c2] += flux[j][face];
                        }
                    }
                }
            }
        }
    }
	
#elif (defined FS_OPENMP) && (defined DIVCON) //D&C TREE
#pragma omp parallel
    {
    #pragma omp single nowait
        tree_traversal(grid->treeHead, res, flux, nVar, f2c);
    }
	
#else

    // For boundary faces, remember c2 is ghost cell
    for(IntType j=0; j<nVar; j++) {
       
       for(IntType face=ns; face<nMid; face++) {
           IntType count = 2*face;
           IntType c1  = f2c[count]; 
           res[j][c1] -= flux[j][face];
       }
 
       // Interior faces
       for(IntType face=nMid; face<ne; face++) {
           IntType count = 2*face;
           IntType c1 = f2c[count];
           IntType c2 = f2c[count+1];

           res[j][c1] -= flux[j][face];
           res[j][c2] += flux[j][face];
       }
    }
#endif
}

/*******************************************************************************\
Free memeories of the residuals for all grids. Here 'grid' must be the finest one.                             
\*******************************************************************************/
void FreeGridResi(PolyGrid *grid, IntType nVar)
{
    PolyGrid *cgrid;

    cgrid = grid;
    
    IntType nTCell;
    nTCell = cgrid->GetNTCell();
    RealFlow *res = (RealFlow *)cgrid->GetDataPtr(REAL_FLOW, nVar*nTCell, "res");
    if(res){
       // delete []res;
       cgrid->DeleteDataPtr("res");
    }
}


/*******************************************************************************\
       Set ql and qr using the values of q
\*******************************************************************************/
void SetQlQrUseQ(PolyGrid *grid, RealFlow *q, RealFlow *ql, RealFlow *qr, IntType ns, IntType ne) 
{
    IntType *f2c   = grid->Getf2c(); 
#ifdef FS_OPENMP
#pragma omp parallel for
#endif            
    for(IntType face=ns; face<ne; face++) {
        IntType c1, c2, count;
        count = 2*face;
        c1 = f2c[count];
        c2 = f2c[count+1];
 
        ql[face] = q[c1];
        qr[face] = q[c2];
    }
}
/*******************************************************************************\
            Copy the values of vector b to vector a                             
\*******************************************************************************/
void VectCopyFrom(RealFlow *a, RealFlow *b, IntType size, IntType sign)
{
    IntType i;
    if(sign > 0) {
       for(i=0; i<size; i++) a[i] = b[i];
    } else if(sign < 0) {
       for(i=0; i<size; i++) a[i] = -b[i];
    } else printf("Warning! Wrongly use VectCopyFrom! Sign cannot be zero!");
}


/*******************************************************************************\
               Calculate the gradients of flow variable q in 3D. 
\*******************************************************************************/
void CompGradientQ(PolyGrid *grid, RealFlow *q, RealFlow *dqdx,   RealFlow *dqdy, RealFlow *dqdz, IntType name, RealFlow* u_n, RealFlow* v_n, RealFlow* w_n)
{

#ifdef TIMECOST//dingxin
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemGradient, endtimeTemGradient;
    double timeuseTemGradient;
    gettimeofday(&starttimeTemGradient, 0); 
#endif
#endif

    CompGradientQ_Gauss_Node(grid, q,  dqdx,  dqdy, dqdz, name, u_n, v_n, w_n);

#ifdef TIMECOST//dingxin
#ifdef MPICH
    timecost[0] = timecost[0] + time_tmp + MPI_Wtime();
    time_gradient = time_gradient + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemGradient, 0); 
    timeuseTemGradient = (RealGeom) 1000000*(endtimeTemGradient.tv_sec - starttimeTemGradient.tv_sec) + endtimeTemGradient.tv_usec - starttimeTemGradient.tv_usec;
    timecost[0] += timeuseTemGradient;
    timeuseTemGradient /= 1000000.0;
    time_gradient += timeuseTemGradient;
#endif
#endif

}

/*******************************************************************************\
 Calculate the gradients of flow variable q in 3D use Node-Green-Gauss Approach.
\*******************************************************************************/
void CompGradientQ_Gauss_Node(PolyGrid *grid, RealFlow *q, RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, IntType name, RealFlow* u_n, RealFlow* v_n, RealFlow* w_n)
{
    IntType    nTNode = grid->GetNTNode();
    IntType    nTCell = grid->GetNTCell();
    IntType    nTFace = grid->GetNTFace();
    IntType    nBFace = grid->GetNBFace();
    IntType    *f2c   = grid->Getf2c();
    BCRecord   **bcr  = grid->Getbcr();
    IntType    n      = nTCell + nBFace;
    RealGeom   *vol   = grid->GetCellVol();  
    RealGeom   *area  = grid->GetFaceArea();
    RealGeom   *xfn   = grid->GetXfn();
    RealGeom   *yfn   = grid->GetYfn();
    RealGeom   *zfn   = grid->GetZfn();
    IntType    *nNPF  = grid->GetnNPF();
    IntType    **F2N  = CalF2N(grid);
    IntType    *nFPC  = CalnFPC(grid);
    IntType    **C2F  = CalC2F(grid); 
    IntType    nIFace = grid->GetNIFace();
    //cout<<"nIFace: "<<nIFace<<endl;
    //cout<<"nBFace: "<<nBFace<<endl;
    //cout<<"nTFace: "<<nTFace<<endl;
    
    IntType    i,  j, c1, c2, count, type, face;
    RealGeom   tmpx, tmpy, tmpz;
    RealFlow   qsum;
    IntType ifacenum = nTFace - nBFace;
    IntType pfacenum = nBFace - nIFace;
      
    // Initialize dq
#ifdef FS_OPENMP  
#pragma omp parallel for
#endif  
    for(i=0; i<n; i++) {
        dqdx[i] = 0.;
        dqdy[i] = 0.;
        dqdz[i] = 0.;
    }

    RealFlow *q_n = NULL;
    mfmem::snew_array_1D(q_n, nTNode,dmrfl);

    if (name > 0 && name < 4)
        CompNodeVar3D_dist(grid, q_n, q, name, u_n, v_n, w_n);
    else
        CompNodeVar3D_dist(grid, q_n, q);
    //Group color openmp
#if (defined FS_OPENMP) && (!defined FS_SIMD) && (defined GroupColor)
    if (grid->GroupColorSuccess) {
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
#pragma omp parallel for private(i,j,count,c1,c2,type,qsum) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                count = 2 * i;
                c1 = f2c[count];
                c2 = f2c[count + 1];
                type = bcr[i]->GetType();
                qsum = 0.0;

                if (type == INTERFACE || type == SYMM) {
                    for (j = 0; j < nNPF[i]; j++)
                        qsum += q_n[F2N[i][j]];
                    qsum /= RealFlow(nNPF[i]);
                }
                else {
                    qsum = 0.5 * (q[c1] + q[c2]);
                }

                qsum *= area[i];
                dqdx[c1] += qsum * xfn[i];
                dqdy[c1] += qsum * yfn[i];
                dqdz[c1] += qsum * zfn[i];
            }
        }
        // zone boundary face
        count = 2 * pfacenum;
        for (i = pfacenum; i < nBFace; i++) {
            c1 = f2c[count++];
            c2 = f2c[count++];
            type = bcr[i]->GetType();
            qsum = 0.0;

            if (type == INTERFACE || type == SYMM) {
                for (j = 0; j < nNPF[i]; j++)
                    qsum += q_n[F2N[i][j]];
                qsum /= RealFlow(nNPF[i]);
            }
            else {
                qsum = 0.5 * (q[c1] + q[c2]);
            }

            qsum *= area[i];
            tmpx = qsum * xfn[i];
            tmpy = qsum * yfn[i];
            tmpz = qsum * zfn[i];
            dqdx[c1] += tmpx;
            dqdy[c1] += tmpy;
            dqdz[c1] += tmpz;
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
#pragma omp parallel for private(i,j,count,c1,c2,type,qsum,tmpx,tmpy,tmpz) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                count = 2 * i;
                c1 = f2c[count];
                c2 = f2c[count + 1];
                qsum = 0.0;

                for (j = 0; j < nNPF[i]; j++)
                    qsum += q_n[F2N[i][j]];
                qsum /= RealFlow(nNPF[i]);

                qsum *= area[i];
                tmpx = qsum * xfn[i];
                tmpy = qsum * yfn[i];
                tmpz = qsum * zfn[i];
                // For cell c1
                dqdx[c1] += tmpx;
                dqdy[c1] += tmpy;
                dqdz[c1] += tmpz;

                // For cell c2
                dqdx[c2] -= tmpx;
                dqdy[c2] -= tmpy;
                dqdz[c2] -= tmpz;
            }
        }
    }
    else {
        for (i = 0; i < nBFace; i++) {
            count = 2 * i;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            type = bcr[i]->GetType();
            qsum = 0.0;

            if (type == INTERFACE || type == SYMM) {
                for (j = 0; j < nNPF[i]; j++)
                    qsum += q_n[F2N[i][j]];
                qsum /= RealFlow(nNPF[i]);
            }
            else {
                qsum = 0.5 * (q[c1] + q[c2]);
            }

            qsum *= area[i];
            tmpx = qsum * xfn[i];
            tmpy = qsum * yfn[i];
            tmpz = qsum * zfn[i];
            dqdx[c1] += tmpx;
            dqdy[c1] += tmpy;
            dqdz[c1] += tmpz;
        }

        for (i = nBFace; i < nTFace; i++) {
            count = 2 * i;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            qsum = 0.0;

            for (j = 0; j < nNPF[i]; j++)
                qsum += q_n[F2N[i][j]];
            qsum /= RealFlow(nNPF[i]);

            qsum *= area[i];
            tmpx = qsum * xfn[i];
            tmpy = qsum * yfn[i];
            tmpz = qsum * zfn[i];
            // For cell c1
            dqdx[c1] += tmpx;
            dqdy[c1] += tmpy;
            dqdz[c1] += tmpz;

            // For cell c2
            dqdx[c2] -= tmpx;
            dqdy[c2] -= tmpy;
            dqdz[c2] -= tmpz;
        }
    }

#elif (defined FS_OPENMP) && (defined FaceColoring) 
    IntType    bfacegroup_num, ifacegroup_num;
    IntType    *grid_bfacegroup, *grid_ifacegroup;
    ifacegroup_num = (*grid).ifacegroup.size();
    bfacegroup_num = (*grid).bfacegroup.size();
    grid_bfacegroup = NULL;
    grid_ifacegroup = NULL;
    mfmem::snew_array_1D(grid_bfacegroup, bfacegroup_num, dmrfl);
    mfmem::snew_array_1D(grid_ifacegroup, ifacegroup_num, dmrfl);
    for (int i = 0; i < bfacegroup_num; i++) {
        grid_bfacegroup[i] = (*grid).bfacegroup[i];
    }
    for (int i = 0; i < ifacegroup_num; i++){
        grid_ifacegroup[i] = (*grid).ifacegroup[i];
    }
    //Boundary faces:
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
            IntType    j, c1, c2, count, type;
            RealGeom   tmpx, tmpy, tmpz;
            RealFlow   qsum;
            count = 2 * i;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            type = bcr[i]->GetType();
            qsum = 0.0;

            if (type == INTERFACE || type == SYMM) {
                for (j = 0; j < nNPF[i]; j++)
                    qsum += q_n[F2N[i][j]];
                qsum /= RealFlow(nNPF[i]);
            }
            else {
                qsum = 0.5 * (q[c1] + q[c2]);
            }

            qsum *= area[i];
            tmpx = qsum * xfn[i];
            tmpy = qsum * yfn[i];
            tmpz = qsum * zfn[i];
            dqdx[c1] += tmpx;
            dqdy[c1] += tmpy;
            dqdz[c1] += tmpz;
        }
    }
    for (IntType i = pfacenum; i < nBFace; i++) {
        IntType count = 2 * i;
        IntType c1 = f2c[count];
        IntType c2 = f2c[count + 1];
        IntType type = bcr[i]->GetType();
        RealFlow qsum = 0.0;
        RealGeom   tmpx, tmpy, tmpz;

        if (type == INTERFACE || type == SYMM) {
            for (j = 0; j < nNPF[i]; j++)
                qsum += q_n[F2N[i][j]];
            qsum /= RealFlow(nNPF[i]);
        }
        else {
            qsum = 0.5 * (q[c1] + q[c2]);
        }

        qsum *= area[i];
        tmpx = qsum * xfn[i];
        tmpy = qsum * yfn[i];
        tmpz = qsum * zfn[i];
        dqdx[c1] += tmpx;
        dqdy[c1] += tmpy;
        dqdz[c1] += tmpz;
    }
    // Interior faces:
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
            IntType    j, c1, c2, count;
            RealGeom   tmpx, tmpy, tmpz;
            RealFlow   qsum;
            count = 2 * i;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            qsum = 0.0;

            for (j = 0; j < nNPF[i]; j++)
                qsum += q_n[F2N[i][j]];
            qsum /= RealFlow(nNPF[i]);

            qsum *= area[i];
            tmpx = qsum * xfn[i];
            tmpy = qsum * yfn[i];
            tmpz = qsum * zfn[i];
            // For cell c1
            dqdx[c1] += tmpx;
            dqdy[c1] += tmpy;
            dqdz[c1] += tmpz;
            // For cell c2
            dqdx[c2] -= tmpx;
            dqdy[c2] -= tmpy;
            dqdz[c2] -= tmpz;        
        }
               
    }
    mfmem::sdel_array_1D(grid_bfacegroup);
    mfmem::sdel_array_1D(grid_ifacegroup);
	
#elif (defined FS_OPENMP) && (defined Reduction)//Manual reduction
    RealGeom* tmpxyz = NULL;
    mfmem::snew_array_1D(tmpxyz, 3 * nTFace, dmrfl);
#pragma omp parallel for private(i,j,count,c1,c2,type,qsum)
    for (i = 0; i < nBFace; i++) {
        count = 2 * i;
        c1 = f2c[count];
        c2 = f2c[count + 1];
        type = bcr[i]->GetType();
        qsum = 0.0;

        if (type == INTERFACE || type == SYMM) {
            for (j = 0; j < nNPF[i]; j++)
                qsum += q_n[F2N[i][j]];
            qsum /= RealFlow(nNPF[i]);
        }
        else {
            qsum = 0.5 * (q[c1] + q[c2]);
        }
        j = 3 * i;
        qsum *= area[i];
        tmpxyz[j] = qsum * xfn[i];
        tmpxyz[j + 1] = qsum * yfn[i];
        tmpxyz[j + 2] = qsum * zfn[i];
    }
#pragma omp parallel for private(i,j,count,c1,c2,qsum)
    for (i = nBFace; i < nTFace; i++) {
        count = 2 * i;
        c1 = f2c[count];
        c2 = f2c[count + 1];
        qsum = 0.0;

        for (j = 0; j < nNPF[i]; j++)
            qsum += q_n[F2N[i][j]];
        qsum /= RealFlow(nNPF[i]);
        j = 3 * i;
        qsum *= area[i];
        tmpxyz[j] = qsum * xfn[i];
        tmpxyz[j + 1] = qsum * yfn[i];
        tmpxyz[j + 2] = qsum * zfn[i];
    }

#pragma omp parallel for private(i,j,count,c1,c2,face)
    for (i = 0; i < nTCell; i++) {
        for (j = 0; j < nFPC[i]; j++) {
            face = C2F[i][j];
            count = 3 * face;
            c1 = f2c[face + face];
            c2 = f2c[face + face + 1];
            if (i == c1) {
                dqdx[i] += tmpxyz[count];
                dqdy[i] += tmpxyz[count + 1];
                dqdz[i] += tmpxyz[count + 2];
            }
            else if (i == c2) {
                dqdx[i] -= tmpxyz[count];
                dqdy[i] -= tmpxyz[count + 1];
                dqdz[i] -= tmpxyz[count + 2];
            }
            else {
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
    }
    mfmem::sdel_array_1D(tmpxyz);
#elif (defined FS_OPENMP) && (!defined FS_SIMD) && (defined DIVREP)//Division & replication
    IntType threads = grid->threads;
    IntType startFace, endFace, t, k;
    if (grid->DivRepSuccess) {
    #pragma omp parallel for private(t,i,j,k,startFace,endFace,count,c1,c2,face,type,qsum,tmpx,tmpy,tmpz)
        for (t = 0; t < threads; t++) {
            //Boundary faces
            startFace = grid->idx_pthreads_bface[t];
            endFace = grid->idx_pthreads_bface[t + 1];
            for (i = startFace; i < endFace; i++) {
                face = grid->id_division_bface[i];
                count = 2 * face;
                c1 = f2c[count];
                c2 = f2c[count + 1];
                type = bcr[face]->GetType();
                qsum = 0.0;
                if (type == INTERFACE || type == SYMM) {
                    for (j = 0; j < nNPF[face]; j++)
                        qsum += q_n[F2N[face][j]];
                    qsum /= RealFlow(nNPF[face]);
                }
                else {
                    qsum = 0.5 * (q[c1] + q[c2]);
                }

                qsum *= area[face];
                tmpx = qsum * xfn[face];
                tmpy = qsum * yfn[face];
                tmpz = qsum * zfn[face];
                dqdx[c1] += tmpx;
                dqdy[c1] += tmpy;
                dqdz[c1] += tmpz;
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
                qsum = 0.0;
                for (j = 0; j < nNPF[face]; j++)
                    qsum += q_n[F2N[face][j]];
                qsum /= RealFlow(nNPF[face]);
                qsum *= area[face];
                tmpx = qsum * xfn[face];
                tmpy = qsum * yfn[face];
                tmpz = qsum * zfn[face];
                if (abs(k) < nTFace) {
                    dqdx[c1] += tmpx;
                    dqdy[c1] += tmpy;
                    dqdz[c1] += tmpz;
                    dqdx[c2] -= tmpx;
                    dqdy[c2] -= tmpy;
                    dqdz[c2] -= tmpz;
                }
                else{
                    if (k > 0) {
                        dqdx[c1] += tmpx;
                        dqdy[c1] += tmpy;
                        dqdz[c1] += tmpz;
                    }
                    else{
                        dqdx[c2] -= tmpx;
                        dqdy[c2] -= tmpy;
                        dqdz[c2] -= tmpz;
                    }
                }
            }
        }
    }

#elif (defined FS_OPENMP) && (defined DIVCON) //D&C TREE
    RealGeom* tmpxyz = NULL;
    mfmem::snew_array_1D(tmpxyz, 3 * (nTFace-nBFace), dmrfl);
#pragma omp parallel
    {
    #pragma omp single nowait
            tree_traversal(grid->treeHead, dqdx, dqdy, dqdz, tmpxyz,
                f2c, bcr, nNPF, F2N, q_n, q, area, xfn, yfn, zfn, nBFace); 
    }
    mfmem::sdel_array_1D(tmpxyz);
#elif (defined FS_SIMD) && (defined FaceColoring) && (!defined FS_SIMD_AVX) && (!defined Tile) //add by ruitian, 2021.11.30
    IntType    bfacegroup_num, ifacegroup_num;
    IntType    *grid_bfacegroup, *grid_ifacegroup;
    ifacegroup_num = (*grid).ifacegroup.size();
    bfacegroup_num = (*grid).bfacegroup.size();
    grid_bfacegroup = NULL;
    grid_ifacegroup = NULL;
    const IntType    Vec = 4;
    mfmem::snew_array_1D(grid_bfacegroup, bfacegroup_num, dmrfl);
    mfmem::snew_array_1D(grid_ifacegroup, ifacegroup_num, dmrfl);
    for (int i = 0; i < bfacegroup_num; i++) {
        grid_bfacegroup[i] = (*grid).bfacegroup[i];
    }
    for (int i = 0; i < ifacegroup_num; i++){
        grid_ifacegroup[i] = (*grid).ifacegroup[i];
    }
    //#ifdef _OPENMP
    //#pragma omp parallel for private(j,c1,c2,count,type,qsum,tmpx,tmpy,tmpz) \
      reduction(+:dqdx[0:n],dqdy[0:n],dqdz[0:n])// schedule(static)
    for (IntType fcolor = 0; fcolor < bfacegroup_num; fcolor++) {
        IntType startFace, endFace;
        if (fcolor == 0) {
            startFace = 0;
        }
        else {
            startFace = grid_bfacegroup[fcolor - 1];
        }
        endFace = grid_bfacegroup[fcolor];
//#pragma omp parallel for private(type,j,c1,c2,count,qsum,tmpx,tmpy,tmpz)
        for (IntType i = startFace; i < endFace; i++) {
            count = 2 * i;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            type = bcr[i]->GetType();
            qsum = 0.0;

            if (type == INTERFACE || type == SYMM) {
                for (j = 0; j < nNPF[i]; j++)
                    qsum += q_n[F2N[i][j]];
                qsum /= RealFlow(nNPF[i]);
            }
            else {
                qsum = 0.5 * (q[c1] + q[c2]);
            }

            qsum *= area[i];
            tmpx = qsum * xfn[i];
            tmpy = qsum * yfn[i];
            tmpz = qsum * zfn[i];
            dqdx[c1] += tmpx;
            dqdy[c1] += tmpy;
            dqdz[c1] += tmpz;
        }

    }
    for (IntType i = pfacenum; i < nBFace; i++) {
        IntType count = 2 * i;
        IntType c1 = f2c[count];
        IntType c2 = f2c[count + 1];
        IntType type = bcr[i]->GetType();
        RealFlow qsum = 0.0;
        RealGeom   tmpx, tmpy, tmpz;

        if (type == INTERFACE || type == SYMM) {
            for (j = 0; j < nNPF[i]; j++)
                qsum += q_n[F2N[i][j]];
            qsum /= RealFlow(nNPF[i]);
        }
        else {
            qsum = 0.5 * (q[c1] + q[c2]);
        }

        qsum *= area[i];
        tmpx = qsum * xfn[i];
        tmpy = qsum * yfn[i];
        tmpz = qsum * zfn[i];
        dqdx[c1] += tmpx;
        dqdy[c1] += tmpy;
        dqdz[c1] += tmpz;
    }
    // Interior faces
    //#pragma omp parallel for private(j,c1,c2,count,qsum,tmpx,tmpy,tmpz) \
      reduction(+:dqdx[0:n],dqdy[0:n],dqdz[0:n])// schedule(static)
    for (IntType fcolor = 0; fcolor < ifacegroup_num; fcolor++) {
        IntType startFace, endFace;
        if (fcolor == 0) {
            startFace = nBFace;
        }
        else {
            startFace = grid_ifacegroup[fcolor - 1];
        }
        endFace = grid_ifacegroup[fcolor];
//#pragma omp parallel for private(j,c1,c2,count,qsum,tmpx,tmpy,tmpz)
//#pragma omp parallel for
        IntType k;
        for (k = startFace; k + Vec < endFace; k += Vec) {
            IntType    jv[Vec], c1v[Vec], c2v[Vec], countv[Vec];
            RealGeom   xfnv[Vec], yfnv[Vec], zfnv[Vec], areav[Vec];
            RealGeom   qsumv[Vec];
            RealGeom   dqdxc1v[Vec], dqdxc2v[Vec];
            RealGeom   dqdyc1v[Vec], dqdyc2v[Vec];
            RealGeom   dqdzc1v[Vec], dqdzc2v[Vec];
            RealGeom   tmpxv[Vec], tmpyv[Vec], tmpzv[Vec];
            RealGeom   nNPFv[Vec];
            //Load:
//#pragma omp simd
            for (IntType iv = 0; iv < Vec; iv++) {
                countv[iv] = 2 * (iv + k);
                c1v[iv] = f2c[countv[iv]];
                c2v[iv] = f2c[countv[iv] + 1];
                qsumv[iv] = 0.0;
                areav[iv] = area[iv + k];
                xfnv[iv] = xfn[iv + k];
                yfnv[iv] = yfn[iv + k];
                zfnv[iv] = zfn[iv + k];
                nNPFv[iv] = nNPF[iv + k];
                //c1 cell:
                dqdxc1v[iv] = dqdx[c1v[iv]];
                dqdyc1v[iv] = dqdy[c1v[iv]];
                dqdzc1v[iv] = dqdz[c1v[iv]];
                //c2 cell:
                dqdxc2v[iv] = dqdx[c2v[iv]];
                dqdyc2v[iv] = dqdy[c2v[iv]];
                dqdzc2v[iv] = dqdz[c2v[iv]];
            }
            //Computation:
            for (IntType iv = 0; iv < Vec; iv++) {
                for (j = 0; j < nNPFv[iv]; j++) {
                    qsumv[iv] += q_n[F2N[iv + k][j]];
                }
            }
 #pragma omp simd safelen(Vec)          
            for (IntType iv = 0; iv < Vec; iv++) {
                qsumv[iv] /= nNPFv[iv];
                qsumv[iv] *= areav[iv];
                tmpxv[iv] = qsumv[iv] * xfnv[iv];
                tmpyv[iv] = qsumv[iv] * yfnv[iv];
                tmpzv[iv] = qsumv[iv] * zfnv[iv];
                //c1 cell:
                dqdxc1v[iv] += tmpxv[iv];
                dqdyc1v[iv] += tmpyv[iv];
                dqdzc1v[iv] += tmpzv[iv];
                //c2 cell:
                dqdxc2v[iv] -= tmpxv[iv];
                dqdyc2v[iv] -= tmpyv[iv];
                dqdzc2v[iv] -= tmpzv[iv];
            }
            //Load Back:
            for (IntType iv = 0; iv < Vec; iv++) {
                //c1 cell:
                dqdx[c1v[iv]] = dqdxc1v[iv];
                dqdy[c1v[iv]] = dqdyc1v[iv];
                dqdz[c1v[iv]] = dqdzc1v[iv];
                //c2 cell:
                dqdx[c2v[iv]] = dqdxc2v[iv];
                dqdy[c2v[iv]] = dqdyc2v[iv];
                dqdz[c2v[iv]] = dqdzc2v[iv];
            }
        }
        for (IntType i = k; i < endFace; i++) {
            IntType    j, c1, c2, count;
            RealGeom   tmpx, tmpy, tmpz;
            RealFlow   qsum;
            count = 2 * i;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            qsum = 0.0;

            for (j = 0; j < nNPF[i]; j++)
                qsum += q_n[F2N[i][j]];
            qsum /= RealFlow(nNPF[i]);

            qsum *= area[i];
            tmpx = qsum * xfn[i];
            tmpy = qsum * yfn[i];
            tmpz = qsum * zfn[i];
            // For cell c1
            dqdx[c1] += tmpx;
            dqdy[c1] += tmpy;
            dqdz[c1] += tmpz;
            // For cell c2
            dqdx[c2] -= tmpx;
            dqdy[c2] -= tmpy;
            dqdz[c2] -= tmpz;
        }
    }
    mfmem::sdel_array_1D(grid_bfacegroup);
    mfmem::sdel_array_1D(grid_ifacegroup);

#elif (defined FS_SIMD) && (defined Tile)  //add by ruitian, 2021.11.30, for AVX512 simd based on tile
    for (i = 0; i < nBFace; i++) {
        count = 2 * i;
        c1 = f2c[count];
        c2 = f2c[count + 1];
        type = bcr[i]->GetType();
        qsum = 0.0;

        if (type == INTERFACE || type == SYMM) {
            for (j = 0; j < nNPF[i]; j++)
                qsum += q_n[F2N[i][j]];
            qsum /= RealFlow(nNPF[i]);
        }
        else {
            qsum = 0.5 * (q[c1] + q[c2]);
        }

        qsum *= area[i];
        tmpx = qsum * xfn[i];
        tmpy = qsum * yfn[i];
        tmpz = qsum * zfn[i];
        dqdx[c1] += tmpx;
        dqdy[c1] += tmpy;
        dqdz[c1] += tmpz;
    }
    //added by ruitianSIMD
    RealFlow* qsumiface = NULL;
    mfmem::snew_array_1D(qsumiface, nTFace, dmrfl);
    //transform the grid information into the tile:
    //ruitian, 2021.12.21
    //cout<<"start tile comp."<<endl;
    for (IntType i = 0; i < nTFace; i++) {
        qsumiface[i] = 0.;
        grid->qsumtile[i] = 0.;
    }
    for (IntType i = 0; i < nTCell; i++) {
        grid->dqdxtile[i] = dqdx[i];
        grid->dqdytile[i] = dqdy[i];
        grid->dqdztile[i] = dqdz[i];
    }

    RealFlow* qsumtile = NULL;
    mfmem::snew_array_1D(qsumtile, nTFace, dmrfl);
    for (IntType i = 0; i < nTFace; i++) {
        qsumtile[i] = 0.;
    }
    for (IntType i = nBFace; i < nTFace; i++) {
        for (IntType j = 0; j < nNPF[i]; j++)
            qsumtile[i] += q_n[F2N[i][j]];
        qsumtile[i] /= RealFlow(nNPF[i]);
    }

    for (IntType ii = 0; ii < grid->iSIMDnnz; ii++) {
        IntType i = grid->iSIMDval[ii];
        grid->qsumt[ii] = qsumtile[i];
    }
    for (IntType ii = 0; ii < (*grid->ioffsets).size(); ii++) {
//#pragma omp parallel for num_threads(2)
        for (IntType jj = 0; jj < (*grid->ioffsets)[ii].size() - 1; jj++) {
            //this for cycle contains an independent tile
            IntType kk;
            __m256i vc1, vc2;
            __m512i vc1tem, vc2tem;
            __m512d vtmpx, vtmpy, vtmpz;
            __m512d vqsum, varea;// , vfacezero;
            __m512d vxfn, vyfn, vzfn;
            __m512d vdqdxc1, vdqdyc1, vdqdzc1;
            __m512d vdqdxc2, vdqdyc2, vdqdzc2;
            //for (kk = (*grid->ioffsets)[ii][jj]; kk< (*grid->ioffsets)[ii][jj + 1]; kk ++) {
            for ( kk = (*grid->ioffsets)[ii][jj]; kk< (*grid->ioffsets)[ii][jj + 1]; kk += 8) {
                ///*
                //this for cycle contains 8 times faces
                if (grid->ifacezero[kk + 7] == 1) {
                    
                    //load c1 and c2:
                    vc1tem = _mm512_load_epi64(&grid->iSIMDrow[kk]);
                    vc2tem = _mm512_load_epi64(&grid->iSIMDcol[kk]);
                    vc1 = _mm512_castsi512_si256(vc1tem);
                    vc2 = _mm512_castsi512_si256(vc2tem);
                    //load data on face:
                    vxfn = _mm512_load_pd(&grid->xfnt[kk]);
                    vyfn = _mm512_load_pd(&grid->yfnt[kk]);
                    vzfn = _mm512_load_pd(&grid->zfnt[kk]);
                    vqsum = _mm512_load_pd(&grid->qsumt[kk]);
                    varea = _mm512_load_pd(&grid->areat[kk]);

                    //comput.
                    vqsum = _mm512_mul_pd(vqsum, varea);
                    vtmpx = _mm512_mul_pd(vqsum, vxfn);
                    vtmpy = _mm512_mul_pd(vqsum, vyfn);
                    vtmpz = _mm512_mul_pd(vqsum, vzfn);

                    //gather c1:
                    vdqdxc1 = _mm512_i32gather_pd(vc1, grid->dqdxtile, 8);
                    vdqdyc1 = _mm512_i32gather_pd(vc1, grid->dqdytile, 8);
                    vdqdzc1 = _mm512_i32gather_pd(vc1, grid->dqdztile, 8);
                    
                    //comput.
                    vdqdxc1 = _mm512_add_pd(vdqdxc1, vtmpx);
                    vdqdyc1 = _mm512_add_pd(vdqdyc1, vtmpy);
                    vdqdzc1 = _mm512_add_pd(vdqdzc1, vtmpz);

                    //scatter c1:
                    _mm512_i32scatter_pd(grid->dqdxtile, vc1, vdqdxc1, 8);
                    _mm512_i32scatter_pd(grid->dqdytile, vc1, vdqdyc1, 8);
                    _mm512_i32scatter_pd(grid->dqdztile, vc1, vdqdzc1, 8);

                    //gather c2:
                    vdqdxc2 = _mm512_i32gather_pd(vc2, grid->dqdxtile, 8);
                    vdqdyc2 = _mm512_i32gather_pd(vc2, grid->dqdytile, 8);
                    vdqdzc2 = _mm512_i32gather_pd(vc2, grid->dqdztile, 8);

                    vdqdxc2 = _mm512_sub_pd(vdqdxc2, vtmpx);
                    vdqdyc2 = _mm512_sub_pd(vdqdyc2, vtmpy);
                    vdqdzc2 = _mm512_sub_pd(vdqdzc2, vtmpz);

                    //scatter c2:
                    _mm512_i32scatter_pd(grid->dqdxtile, vc2, vdqdxc2, 8);
                    _mm512_i32scatter_pd(grid->dqdytile, vc2, vdqdyc2, 8);
                    _mm512_i32scatter_pd(grid->dqdztile, vc2, vdqdzc2, 8);
                }
                else {
                    for (IntType ikk = kk; ikk < kk + 8; ikk++) {
                        if (grid->ifacezero[kk] == 0) {
                            break;
                        }
                        else {
                            IntType c1 = grid->iSIMDrow[ikk];
                            IntType c2 = grid->iSIMDcol[ikk];

                            grid->qsumt[ikk] *= grid->areat[ikk];

                            RealGeom tmpx = grid->qsumt[ikk] * grid->xfnt[ikk] * grid->ifacezero[ikk];
                            RealGeom tmpy = grid->qsumt[ikk] * grid->yfnt[ikk] * grid->ifacezero[ikk];
                            RealGeom tmpz = grid->qsumt[ikk] * grid->zfnt[ikk] * grid->ifacezero[ikk];

                            // For cell c1
                            grid->dqdxtile[c1] += tmpx;
                            grid->dqdytile[c1] += tmpy;
                            grid->dqdztile[c1] += tmpz;
                            // For cell c2
                            grid->dqdxtile[c2] -= tmpx;
                            grid->dqdytile[c2] -= tmpy;
                            grid->dqdztile[c2] -= tmpz;
                        }
                    }
                }
            }
        }
    }
    //transform back:
    for (IntType i = 0; i < nTCell; i++) {
        dqdx[i] = grid->dqdxtile[i];
        dqdy[i] = grid->dqdytile[i];
        dqdz[i] = grid->dqdztile[i];
    }
    mfmem::sdel_array_1D(qsumtile);
#elif (defined FS_SIMD) && (defined FS_SIMD_AVX) && (defined FaceColoring) //for AVX512 simd based on face coloring
    IntType    bfacegroup_num, ifacegroup_num;
    IntType    *grid_bfacegroup, *grid_ifacegroup;
    ifacegroup_num = (*grid).ifacegroup.size();
    bfacegroup_num = (*grid).bfacegroup.size();
    grid_bfacegroup = NULL;
    grid_ifacegroup = NULL;
    mfmem::snew_array_1D(grid_bfacegroup, bfacegroup_num, dmrfl);
    mfmem::snew_array_1D(grid_ifacegroup, ifacegroup_num, dmrfl);
    for (int i = 0; i < bfacegroup_num; i++) {
        grid_bfacegroup[i] = (*grid).bfacegroup[i];
    }
    for (int i = 0; i < ifacegroup_num; i++){
        grid_ifacegroup[i] = (*grid).ifacegroup[i];
    }
    //add by ruitian, for SIMD AVX facecoloring
    //#pragma omp parallel for 
    for (IntType i = 0; i < nTCell; i++) {
        grid->dqdxtile[i] = dqdx[i];
        grid->dqdytile[i] = dqdy[i];
        grid->dqdztile[i] = dqdz[i];
    }
    RealFlow* bqsumtile = NULL;
    //mfmem::snew_array_1D(qsumtile, nTFace, dmrfl);
    bqsumtile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nBFace, 64);

//#pragma omp parallel for 
    for (IntType i = 0; i < nBFace; i++) {
        bqsumtile[i] = 0.;
    }
//#pragma omp parallel for 
    for (IntType i = 0; i < nBFace; i++) {
        IntType   c1, c2, type;
        c1 = grid->f2c1[i];
        c2 = grid->f2c2[i];
        type = bcr[i]->GetType();
        if (type == INTERFACE || type == SYMM) {
            for (j = 0; j < nNPF[i]; j++)
                bqsumtile[i] += q_n[F2N[i][j]];
            bqsumtile[i] /= RealFlow(nNPF[i]);            
        }
        else {
            bqsumtile[i] = 0.5 * (q[c1] + q[c2]);
        }
        bqsumtile[i] *= area[i];
    }
    //Boundary faces:
    for (IntType fcolor = 0; fcolor < bfacegroup_num; fcolor++) {
        IntType startFace, endFace;
        if (fcolor == 0) {
            startFace = 0;
        }
        else {
            startFace = grid_bfacegroup[fcolor - 1];
        }
        endFace = grid_bfacegroup[fcolor];
        IntType i;
#ifdef FS_OPENMP  
#pragma omp parallel for
#endif  
        for (i = startFace; i < endFace-8; i += 8) {
            __m256i vc1;
            __m512i vc1tem;
            __m512d vtmpx, vtmpy, vtmpz;
            __m512d vqsum;
            __m512d vxfn, vyfn, vzfn;
            __m512d vdqdxc1, vdqdyc1, vdqdzc1;

            //load c1 and c2:
            //_mm256_load_epi32
            //vc1 = _mm256_load_epi32(&f2c[i]);
            vc1tem = _mm512_load_epi64(&grid->f2c1[i]);
            vc1 = _mm512_castsi512_si256(vc1tem);
                        
            //load data on face:
            vxfn = _mm512_load_pd(&grid->xfntile[i]);
            vyfn = _mm512_load_pd(&grid->yfntile[i]);
            vzfn = _mm512_load_pd(&grid->zfntile[i]);
            vqsum = _mm512_load_pd(&bqsumtile[i]);

            //gather:
            vdqdxc1 = _mm512_i32gather_pd (vc1, grid->dqdxtile, 8);
            vdqdyc1 = _mm512_i32gather_pd (vc1, grid->dqdytile, 8);
            vdqdzc1 = _mm512_i32gather_pd (vc1, grid->dqdztile, 8);

            //comput.
            vtmpx = _mm512_mul_pd(vqsum, vxfn);
            vtmpy = _mm512_mul_pd(vqsum, vyfn);
            vtmpz = _mm512_mul_pd(vqsum, vzfn);
            vdqdxc1 = _mm512_add_pd(vdqdxc1, vtmpx);
            vdqdyc1 = _mm512_add_pd(vdqdyc1, vtmpy);
            vdqdzc1 = _mm512_add_pd(vdqdzc1, vtmpz);

            //scatter:
            _mm512_i32scatter_pd(grid->dqdxtile, vc1, vdqdxc1, 8);
            _mm512_i32scatter_pd(grid->dqdytile, vc1, vdqdyc1, 8);
            _mm512_i32scatter_pd(grid->dqdztile, vc1, vdqdzc1, 8);
        }
        for (IntType vi = i; vi < endFace; vi++) {            
            IntType    c1, c2;
            RealGeom   tmpx, tmpy, tmpz;
            c1 = grid->f2c1[vi];
            c2 = grid->f2c2[vi];

            tmpx = bqsumtile[vi] * xfn[vi];
            tmpy = bqsumtile[vi] * yfn[vi];
            tmpz = bqsumtile[vi] * zfn[vi];
            grid->dqdxtile[c1] += tmpx;
            grid->dqdytile[c1] += tmpy;
            grid->dqdztile[c1] += tmpz;            

        }
    }
    for (IntType i = pfacenum; i < nBFace; i++) {
        IntType count = 2 * i;
        IntType c1 = f2c[count];
        IntType c2 = f2c[count + 1];
        IntType type = bcr[i]->GetType();
        RealFlow qsum = 0.0;
        RealGeom   tmpx, tmpy, tmpz;

        if (type == INTERFACE || type == SYMM) {
            for (j = 0; j < nNPF[i]; j++)
                qsum += q_n[F2N[i][j]];
            qsum /= RealFlow(nNPF[i]);
        }
        else {
            qsum = 0.5 * (q[c1] + q[c2]);
        }

        qsum *= area[i];
        tmpx = qsum * xfn[i];
        tmpy = qsum * yfn[i];
        tmpz = qsum * zfn[i];
        grid->dqdxtile[c1] += tmpx;
        grid->dqdytile[c1] += tmpy;
        grid->dqdztile[c1] += tmpz;
    }
/*************************************************************************/
    // ruitianSIMD, 2021.12.28
    //added by ruitian, for SIMD
    //transform the grid information into the tile:
    //ruitian, 2021.12.21
    RealFlow* qsumtile = NULL;
    //mfmem::snew_array_1D(qsumtile, nTFace, dmrfl);
    qsumtile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nTFace, 64);
#ifdef FS_OPENMP  
#pragma omp parallel for
#endif   
    for (IntType i = 0; i < nTFace; i++) {
        qsumtile[i] = 0.;
    }
#ifdef FS_OPENMP  
#pragma omp parallel for
#endif   
    for (IntType i = nBFace; i < nTFace; i++) {
        for (j = 0; j < nNPF[i]; j++)
            qsumtile[i] += q_n[F2N[i][j]];
        qsumtile[i] /= RealFlow(nNPF[i]);
        qsumtile[i] *= area[i];
    }
    // Interior faces:
    for (IntType fcolor = 0; fcolor < ifacegroup_num; fcolor++) {
        IntType startFace, endFace;
        if (fcolor == 0) {
            startFace = nBFace;
        }
        else {
            startFace = grid_ifacegroup[fcolor - 1];
        }
        endFace = grid_ifacegroup[fcolor];

        IntType i;
#ifdef FS_OPENMP  
#pragma omp parallel for
#endif          
        for (i = startFace; i < endFace-8; i += 8) {

            __m256i vc1, vc2;
            __m512i vc1tem, vc2tem;
            __m512d vtmpx, vtmpy, vtmpz;
            __m512d vqsum;
            __m512d vxfn, vyfn, vzfn;
            __m512d vdqdxc1, vdqdyc1, vdqdzc1;
            __m512d vdqdxc2, vdqdyc2, vdqdzc2;
            //load c1 and c2:
            //_mm256_load_epi32
            //vc1 = _mm256_load_epi32(&f2c[i]);
            vc1tem = _mm512_load_epi64(&grid->f2c1[i]);
            vc2tem = _mm512_load_epi64 (&grid->f2c2[i]);
            vc1 = _mm512_castsi512_si256(vc1tem);
            vc2 = _mm512_castsi512_si256(vc2tem);
            //load data on face:
            vxfn = _mm512_load_pd(&grid->xfntile[i]);
            vyfn = _mm512_load_pd(&grid->yfntile[i]);
            vzfn = _mm512_load_pd(&grid->zfntile[i]);
            vqsum = _mm512_load_pd(&qsumtile[i]);

            //gather:
            vdqdxc1 = _mm512_i32gather_pd (vc1, grid->dqdxtile, 8);
            vdqdyc1 = _mm512_i32gather_pd (vc1, grid->dqdytile, 8);
            vdqdzc1 = _mm512_i32gather_pd (vc1, grid->dqdztile, 8);
            vdqdxc2 = _mm512_i32gather_pd (vc2, grid->dqdxtile, 8);
            vdqdyc2 = _mm512_i32gather_pd (vc2, grid->dqdytile, 8);
            vdqdzc2 = _mm512_i32gather_pd (vc2, grid->dqdztile, 8);

            //comput.
            vtmpx = _mm512_mul_pd(vqsum, vxfn);
            vtmpy = _mm512_mul_pd(vqsum, vyfn);
            vtmpz = _mm512_mul_pd(vqsum, vzfn);
            vdqdxc1 = _mm512_add_pd(vdqdxc1, vtmpx);
            vdqdyc1 = _mm512_add_pd(vdqdyc1, vtmpy);
            vdqdzc1 = _mm512_add_pd(vdqdzc1, vtmpz);
            vdqdxc2 = _mm512_sub_pd(vdqdxc2, vtmpx);
            vdqdyc2 = _mm512_sub_pd(vdqdyc2, vtmpy);
            vdqdzc2 = _mm512_sub_pd(vdqdzc2, vtmpz);

            //scatter:
            _mm512_i32scatter_pd(grid->dqdxtile, vc1, vdqdxc1, 8);
            _mm512_i32scatter_pd(grid->dqdytile, vc1, vdqdyc1, 8);
            _mm512_i32scatter_pd(grid->dqdztile, vc1, vdqdzc1, 8);
            _mm512_i32scatter_pd(grid->dqdxtile, vc2, vdqdxc2, 8);
            _mm512_i32scatter_pd(grid->dqdytile, vc2, vdqdyc2, 8);
            _mm512_i32scatter_pd(grid->dqdztile, vc2, vdqdzc2, 8);

        }
        for (IntType vi = i; vi < endFace; vi++) {        

            IntType c1 = grid->f2c1[vi];
            IntType c2 = grid->f2c2[vi];

            RealGeom tmpx = qsumtile[vi] * grid->xfntile[vi];
            RealGeom tmpy = qsumtile[vi] * grid->yfntile[vi];
            RealGeom tmpz = qsumtile[vi] * grid->zfntile[vi];

            // For cell c1
            grid->dqdxtile[c1] += tmpx;
            grid->dqdytile[c1] += tmpy;
            grid->dqdztile[c1] += tmpz;

            // For cell c2
            grid->dqdxtile[c2] -= tmpx;
            grid->dqdytile[c2] -= tmpy;
            grid->dqdztile[c2] -= tmpz;

        }              
    }
    mfmem::sdel_array_1D(grid_bfacegroup);
    mfmem::sdel_array_1D(grid_ifacegroup);
	//transform back:
#ifdef FS_OPENMP  
#pragma omp parallel for
#endif   
    for (IntType i = 0; i < nTCell; i++) {
        dqdx[i] = grid->dqdxtile[i];
        dqdy[i] = grid->dqdytile[i];
        dqdz[i] = grid->dqdztile[i];
    }

#elif (defined FS_OPENMP) && (defined FS_SIMD) && (defined DIVREP) && (defined BoundedColoring)//add by dingxin, 2021-11-30
    IntType threads = grid->threads;
    IntType startFace, endFace, t, k, iv;
    IntType endIndex_iFace_vec;
    const IntType Vec = VEC_SIZE;
    if (grid->DivRepSuccess) {
#pragma omp parallel for private(t,i,j,startFace,endFace,count,c1,c2,face,type,qsum,tmpx,tmpy,tmpz)
        for (t = 0; t < threads; t++) { //Boundary faces
            startFace = grid->idx_pthreads_bface[t];
            endFace = grid->idx_pthreads_bface[t + 1];
            for (i = startFace; i < endFace; i++) {
                face = grid->id_division_bface[i];
                count = 2 * face;
                c1 = f2c[count];
                c2 = f2c[count + 1];
                type = bcr[face]->GetType();
                qsum = 0.0;
                if (type == INTERFACE || type == SYMM) {
                    for (j = 0; j < nNPF[face]; j++)
                        qsum += q_n[F2N[face][j]];
                    qsum /= RealFlow(nNPF[face]);
                }
                else {
                    qsum = 0.5 * (q[c1] + q[c2]);
                }

                qsum *= area[face];
                tmpx = qsum * xfn[face];
                tmpy = qsum * yfn[face];
                tmpz = qsum * zfn[face];
                dqdx[c1] += tmpx;
                dqdy[c1] += tmpy;
                dqdz[c1] += tmpz;
            }
        }
#ifdef TIMECOST//dingxin
#ifdef MPICH
        double time_tmp;
        time_tmp = -MPI_Wtime();
#else
        struct timeval starttimeTemRoe, endtimeTemRoe;
        double time_tmp;
        gettimeofday(&starttimeTemRoe, 0);
#endif
#endif
#pragma omp parallel for private(t,i,j,k,iv,startFace,endFace,endIndex_iFace_vec,count,c1,c2,face,type,qsum,tmpx,tmpy,tmpz)
        for (t = 0; t < threads; t++) {
            //Interior faces
            startFace = grid->idx_pthreads_iface[t];
            endFace = grid->idx_pthreads_iface[t + 1];
            endIndex_iFace_vec = grid->endIndex_iFace_vec[t];
            for (k = startFace; k + Vec <= endIndex_iFace_vec; k += Vec) {
                IntType    c1v[Vec], c2v[Vec], countv[Vec], tag[Vec], facev[Vec];
                RealGeom   xfnv[Vec], yfnv[Vec], zfnv[Vec], areav[Vec];
                RealGeom   qsumv[Vec];
                RealGeom   dqdxc1v[Vec], dqdxc2v[Vec];
                RealGeom   dqdyc1v[Vec], dqdyc2v[Vec];
                RealGeom   dqdzc1v[Vec], dqdzc2v[Vec];
                RealGeom   tmpxv[Vec], tmpyv[Vec], tmpzv[Vec];
                RealGeom   nNPFv[Vec];
                //Load:
                for (iv = 0; iv < Vec; iv++) {
                    tag[iv] = grid->id_division_iface[iv + k];
                    if (abs(tag[iv]) < nTFace)
                        facev[iv] = tag[iv];
                    else
                        facev[iv] = abs(tag[iv]) - nTFace;
                }
#pragma omp simd safelen(Vec)   
                for (iv = 0; iv < Vec; iv++) {
                    countv[iv] = 2 * facev[iv];
                    c1v[iv] = f2c[countv[iv]];
                    c2v[iv] = f2c[countv[iv] + 1];
                    qsumv[iv] = 0.0;
                    areav[iv] = area[facev[iv]];
                    xfnv[iv] = xfn[facev[iv]];
                    yfnv[iv] = yfn[facev[iv]];
                    zfnv[iv] = zfn[facev[iv]];
                    nNPFv[iv] = nNPF[facev[iv]];
                    if (abs(tag[iv]) < nTFace) {
                        dqdxc1v[iv] = dqdx[c1v[iv]];
                        dqdyc1v[iv] = dqdy[c1v[iv]];
                        dqdzc1v[iv] = dqdz[c1v[iv]];
                        dqdxc2v[iv] = dqdx[c2v[iv]];
                        dqdyc2v[iv] = dqdy[c2v[iv]];
                        dqdzc2v[iv] = dqdz[c2v[iv]];
                    }
                    else {
                        if (tag[iv] > 0) {
                            dqdxc1v[iv] = dqdx[c1v[iv]];
                            dqdyc1v[iv] = dqdy[c1v[iv]];
                            dqdzc1v[iv] = dqdz[c1v[iv]];
                        }
                        else {
                            dqdxc2v[iv] = dqdx[c2v[iv]];
                            dqdyc2v[iv] = dqdy[c2v[iv]];
                            dqdzc2v[iv] = dqdz[c2v[iv]];
                        }
                    }
                    
                }
                //Computation:
                for (iv = 0; iv < Vec; iv++) {
                    for (j = 0; j < nNPFv[iv]; j++) {
                        qsumv[iv] += q_n[F2N[facev[iv]][j]];
                    }
                }
#pragma omp simd safelen(Vec)          
                for (iv = 0; iv < Vec; iv++) {
                    qsumv[iv] /= RealFlow(nNPFv[iv]);
                    qsumv[iv] *= areav[iv];
                    tmpxv[iv] = qsumv[iv] * xfnv[iv];
                    tmpyv[iv] = qsumv[iv] * yfnv[iv];
                    tmpzv[iv] = qsumv[iv] * zfnv[iv];
                    if (abs(tag[iv]) < nTFace) {
                        dqdxc1v[iv] += tmpxv[iv];
                        dqdyc1v[iv] += tmpyv[iv];
                        dqdzc1v[iv] += tmpzv[iv];
                        dqdxc2v[iv] -= tmpxv[iv];
                        dqdyc2v[iv] -= tmpyv[iv];
                        dqdzc2v[iv] -= tmpzv[iv];
                    }
                    else {
                        if (tag[iv] > 0) {
                            dqdxc1v[iv] += tmpxv[iv];
                            dqdyc1v[iv] += tmpyv[iv];
                            dqdzc1v[iv] += tmpzv[iv];
                        }
                        else {
                            dqdxc2v[iv] -= tmpxv[iv];
                            dqdyc2v[iv] -= tmpyv[iv];
                            dqdzc2v[iv] -= tmpzv[iv];
                        }
                    }
                }
                //Load Back:
#pragma omp simd safelen(Vec)
                for (iv = 0; iv < Vec; iv++) {
                    if (abs(tag[iv]) < nTFace) {
                        dqdx[c1v[iv]] = dqdxc1v[iv];
                        dqdy[c1v[iv]] = dqdyc1v[iv];
                        dqdz[c1v[iv]] = dqdzc1v[iv];
                        dqdx[c2v[iv]] = dqdxc2v[iv];
                        dqdy[c2v[iv]] = dqdyc2v[iv];
                        dqdz[c2v[iv]] = dqdzc2v[iv];
                    }
                    else {
                        if (tag[iv] > 0) {
                            dqdx[c1v[iv]] = dqdxc1v[iv];
                            dqdy[c1v[iv]] = dqdyc1v[iv];
                            dqdz[c1v[iv]] = dqdzc1v[iv];
                        }
                        else {
                            dqdx[c2v[iv]] = dqdxc2v[iv];
                            dqdy[c2v[iv]] = dqdyc2v[iv];
                            dqdz[c2v[iv]] = dqdzc2v[iv];
                        }
                    }
                }
            }
            startFace = k;
            for (i = startFace; i < endFace; i++) {
                k = grid->id_division_iface[i];
                if (abs(k) < nTFace)
                    face = k;
                else
                    face = abs(k) - nTFace;
                count = 2 * face;
                c1 = f2c[count];
                c2 = f2c[count + 1];
                qsum = 0.0;
                for (j = 0; j < nNPF[face]; j++)
                    qsum += q_n[F2N[face][j]];
                qsum /= RealFlow(nNPF[face]);
                qsum *= area[face];
                tmpx = qsum * xfn[face];
                tmpy = qsum * yfn[face];
                tmpz = qsum * zfn[face];
                if (abs(k) < nTFace) {
                    dqdx[c1] += tmpx;
                    dqdy[c1] += tmpy;
                    dqdz[c1] += tmpz;
                    dqdx[c2] -= tmpx;
                    dqdy[c2] -= tmpy;
                    dqdz[c2] -= tmpz;
                }
                else {
                    if (k > 0) {
                        dqdx[c1] += tmpx;
                        dqdy[c1] += tmpy;
                        dqdz[c1] += tmpz;
                    }
                    else {
                        dqdx[c2] -= tmpx;
                        dqdy[c2] -= tmpy;
                        dqdz[c2] -= tmpz;
                    }
                }
            }
    }
#ifdef TIMECOST//dingxin
#ifdef MPICH
        timecost[5] = timecost[5] + time_tmp + MPI_Wtime();
#else
        gettimeofday(&endtimeTemRoe, 0);
        time_tmp = (RealGeom)1000000 * (endtimeTemRoe.tv_sec - starttimeTemRoe.tv_sec) + endtimeTemRoe.tv_usec - starttimeTemRoe.tv_usec;
        timecost[5] += time_tmp;
#endif
#endif
    }
#elif (defined FS_OPENMP) && (defined FS_SIMD) && (defined DIVREP) //add by dingxin, 2021-12-05
    IntType threads = grid->threads;
    IntType startFace, endFace, t, k, iv;
    const IntType Vec = VEC_SIZE;
    if (grid->DivRepSuccess) {
#pragma omp parallel for private(t,i,j,startFace,endFace,count,c1,c2,face,type,qsum,tmpx,tmpy,tmpz)
        for (t = 0; t < threads; t++) { //Boundary faces
            startFace = grid->idx_pthreads_bface[t];
            endFace = grid->idx_pthreads_bface[t + 1];
            for (i = startFace; i < endFace; i++) {
                face = grid->id_division_bface[i];
                count = 2 * face;
                c1 = f2c[count];
                c2 = f2c[count + 1];
                type = bcr[face]->GetType();
                qsum = 0.0;
                if (type == INTERFACE || type == SYMM) {
                    for (j = 0; j < nNPF[face]; j++)
                        qsum += q_n[F2N[face][j]];
                    qsum /= RealFlow(nNPF[face]);
                }
                else {
                    qsum = 0.5 * (q[c1] + q[c2]);
                }

                qsum *= area[face];
                tmpx = qsum * xfn[face];
                tmpy = qsum * yfn[face];
                tmpz = qsum * zfn[face];
                dqdx[c1] += tmpx;
                dqdy[c1] += tmpy;
                dqdz[c1] += tmpz;
            }
        }
#ifdef TIMECOST//dingxin
#ifdef MPICH
        double time_tmp;
        time_tmp = -MPI_Wtime();
#else
        struct timeval starttimeTemRoe, endtimeTemRoe;
        double time_tmp;
        gettimeofday(&starttimeTemRoe, 0);
#endif
#endif
#pragma omp parallel for private(t,i,j,k,iv,startFace,endFace,count,c1,c2,face,type,qsum,tmpx,tmpy,tmpz)
        for (t = 0; t < threads; t++) {
            //Interior faces
            IntType    c1v[Vec], c2v[Vec], countv[Vec], tag[Vec], facev[Vec];
            RealGeom   xfnv[Vec], yfnv[Vec], zfnv[Vec], areav[Vec];
            RealGeom   qsumv[Vec];
            RealGeom   tmpxv[Vec], tmpyv[Vec], tmpzv[Vec];
            RealGeom   nNPFv[Vec];
            startFace = grid->idx_pthreads_iface[t];
            endFace = grid->idx_pthreads_iface[t + 1];
            for (k = startFace; k + Vec < endFace; k += Vec) {
                //Load:
                for (iv = 0; iv < Vec; iv++) {
                    tag[iv] = grid->id_division_iface[iv + k];
                    if (abs(tag[iv]) < nTFace)
                        facev[iv] = tag[iv];
                    else
                        facev[iv] = abs(tag[iv]) - nTFace;
                }
#pragma omp simd safelen(Vec)   
                for (iv = 0; iv < Vec; iv++) {
                    countv[iv] = 2 * facev[iv];
                    c1v[iv] = f2c[countv[iv]];
                    c2v[iv] = f2c[countv[iv] + 1];
                    qsumv[iv] = 0.0;
                    areav[iv] = area[facev[iv]];
                    xfnv[iv] = xfn[facev[iv]];
                    yfnv[iv] = yfn[facev[iv]];
                    zfnv[iv] = zfn[facev[iv]];
                    nNPFv[iv] = nNPF[facev[iv]];

                }
                //Computation:
                for (iv = 0; iv < Vec; iv++) {
                    for (j = 0; j < nNPFv[iv]; j++) {
                        qsumv[iv] += q_n[F2N[facev[iv]][j]];
                    }
                }
#pragma omp simd safelen(Vec)          
                for (iv = 0; iv < Vec; iv++) {
                    qsumv[iv] /= nNPFv[iv];
                    qsumv[iv] *= areav[iv];
                    tmpxv[iv] = qsumv[iv] * xfnv[iv];
                    tmpyv[iv] = qsumv[iv] * yfnv[iv];
                    tmpzv[iv] = qsumv[iv] * zfnv[iv];
                }
                //Load Back: no color, exist conflicts
                for (iv = 0; iv < Vec; iv++) {
                    if (abs(tag[iv]) < nTFace) {
                        dqdx[c1v[iv]] += tmpxv[iv];
                        dqdy[c1v[iv]] += tmpyv[iv];
                        dqdz[c1v[iv]] += tmpzv[iv];
                        dqdx[c2v[iv]] -= tmpxv[iv];
                        dqdy[c2v[iv]] -= tmpyv[iv];
                        dqdz[c2v[iv]] -= tmpzv[iv];
                    }
                    else {
                        if (tag[iv] > 0) {
                            dqdx[c1v[iv]] += tmpxv[iv];
                            dqdy[c1v[iv]] += tmpyv[iv];
                            dqdz[c1v[iv]] += tmpzv[iv];
                        }
                        else {
                            dqdx[c2v[iv]] -= tmpxv[iv];
                            dqdy[c2v[iv]] -= tmpyv[iv];
                            dqdz[c2v[iv]] -= tmpzv[iv];
                        }
                    }
                }
            }
            startFace = k;
            for (i = startFace; i < endFace; i++) {
                k = grid->id_division_iface[i];
                if (abs(k) < nTFace)
                    face = k;
                else
                    face = abs(k) - nTFace;
                count = 2 * face;
                c1 = f2c[count];
                c2 = f2c[count + 1];
                qsum = 0.0;
                for (j = 0; j < nNPF[face]; j++)
                    qsum += q_n[F2N[face][j]];
                qsum /= RealFlow(nNPF[face]);
                qsum *= area[face];
                tmpx = qsum * xfn[face];
                tmpy = qsum * yfn[face];
                tmpz = qsum * zfn[face];
                if (abs(k) < nTFace) {
                    dqdx[c1] += tmpx;
                    dqdy[c1] += tmpy;
                    dqdz[c1] += tmpz;
                    dqdx[c2] -= tmpx;
                    dqdy[c2] -= tmpy;
                    dqdz[c2] -= tmpz;
                }
                else {
                    if (k > 0) {
                        dqdx[c1] += tmpx;
                        dqdy[c1] += tmpy;
                        dqdz[c1] += tmpz;
                    }
                    else {
                        dqdx[c2] -= tmpx;
                        dqdy[c2] -= tmpy;
                        dqdz[c2] -= tmpz;
                    }
                }
            }
        }
#ifdef TIMECOST//dingxin
#ifdef MPICH
        timecost[5] = timecost[5] + time_tmp + MPI_Wtime();
#else
        gettimeofday(&endtimeTemRoe, 0);
        time_tmp = (RealGeom)1000000 * (endtimeTemRoe.tv_sec - starttimeTemRoe.tv_sec) + endtimeTemRoe.tv_usec - starttimeTemRoe.tv_usec;
        timecost[5] += time_tmp;
#endif
#endif
    }
#else
    for (i = 0; i < nBFace; i++) {
        count = 2 * i;
        c1 = f2c[count];
        c2 = f2c[count + 1];
        type = bcr[i]->GetType();
        qsum = 0.0;

        if (type == INTERFACE || type == SYMM) {
            for (j = 0; j < nNPF[i]; j++)
                qsum += q_n[F2N[i][j]];
            qsum /= RealFlow(nNPF[i]);
        }
        else {
            qsum = 0.5 * (q[c1] + q[c2]);
        }

        qsum *= area[i];
        tmpx = qsum * xfn[i];
        tmpy = qsum * yfn[i];
        tmpz = qsum * zfn[i];
        dqdx[c1] += tmpx;
        dqdy[c1] += tmpy;
        dqdz[c1] += tmpz;
    }
    for (i = nBFace; i < nTFace; i++) {
        count = 2 * i;
        c1 = f2c[count];
        c2 = f2c[count + 1];
        qsum = 0.0;

        for (j = 0; j < nNPF[i]; j++)
            qsum += q_n[F2N[i][j]];
        qsum /= RealFlow(nNPF[i]);

        qsum *= area[i];
        tmpx = qsum * xfn[i];
        tmpy = qsum * yfn[i];
        tmpz = qsum * zfn[i];
        // For cell c1
        dqdx[c1] += tmpx;
        dqdy[c1] += tmpy;
        dqdz[c1] += tmpz;

        // For cell c2
        dqdx[c2] -= tmpx;
        dqdy[c2] -= tmpy;
        dqdz[c2] -= tmpz;
    }
#endif
   
    //如果单元含有一个以上的物面，该单元梯度采用Gauss求解
    IntType vis_mode,level;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    level = grid->GetLevel();
    if(vis_mode != INVISCID && level == 0){ 
        IntType *cellwallnumber = grid->GetGridQualityCellWallNumber();
#ifdef FS_OPENMP
#pragma omp parallel for
#endif        
        for(IntType i=0;i<nTCell;i++){
            if(cellwallnumber[i]<2) continue;
            dqdx[i] = 0.0;
            dqdy[i] = 0.0;
            dqdz[i] = 0.0;
            for(IntType j=0;j<nFPC[i];j++){
                IntType face = C2F[i][j];
                IntType c1   = f2c[face+face];
                IntType c2   = f2c[face+face+1];
                
                RealGeom qsum = 0.5*(q[c1]+q[c2])*area[face];
                RealGeom tmpx = qsum*xfn[face];
                RealGeom tmpy = qsum*yfn[face];
                RealGeom tmpz = qsum*zfn[face];
                
                if(i == c1){  
                    dqdx[i] += tmpx;
                    dqdy[i] += tmpy;
                    dqdz[i] += tmpz;
                }else if(i == c2){
                    dqdx[i] -= tmpx;
                    dqdy[i] -= tmpy;
                    dqdz[i] -= tmpz;
                }else{
                    std::cerr<<endl<<"Error in function CompGradientQ_Gauss_Node_Dist!  i is not c1 or c2! "<<i<<"  "<<c1<<"  "<<c2<<endl;
                    mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
                }
            }   
        }
    }      

    //边界层前n层采用Gauss方法
    IntType GaussLayer = -1;
    grid->GetData(&GaussLayer, INT, 1, "GaussLayer");
    IntType *CellLayerNo = (IntType *)grid->GetDataPtr(INT, n, "CellLayerNo");
    if(level == 0 && GaussLayer>0){
        for(i=0;i<nTCell;i++){
            if((CellLayerNo[i]==-1) || (CellLayerNo[i]>=GaussLayer)) continue;
            dqdx[i] = 0.0;
            dqdy[i] = 0.0;
            dqdz[i] = 0.0;
            for(j=0;j<nFPC[i];j++){
                face = C2F[i][j];
                c1   = f2c[face+face];
                c2   = f2c[face+face+1];
                
                qsum = 0.5*(q[c1]+q[c2])*area[face];
                tmpx = qsum*xfn[face];
                tmpy = qsum*yfn[face];
                tmpz = qsum*zfn[face];
                
                if(i == c1){  
                    dqdx[i] += tmpx;
                    dqdy[i] += tmpy;
                    dqdz[i] += tmpz;
                }else if(i == c2){
                    dqdx[i] -= tmpx;
                    dqdy[i] -= tmpy;
                    dqdz[i] -= tmpz;
                }else{
                    std::cerr<<endl<<"Error in function CompGradientQ_Gauss_Node_Dist!  i is not c1 or c2! "<<i<<"  "<<c1<<"  "<<c2<<endl;
                    mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
                }
            }   
        }
    }   
#ifdef FS_OPENMP
#pragma omp parallel for
#endif   
    for(IntType i=0; i<nTCell; i++) {
        dqdx[i] /= vol[i] ;
        dqdy[i] /= vol[i] ;
        dqdz[i] /= vol[i] ;
    }
    mfmem::sdel_array_1D(q_n);
}


/*******************************************************************************\
              Compute the node variable use the distance weight
\*******************************************************************************/
void CompNodeVar3D_dist(PolyGrid *grid, RealFlow *q_n, RealFlow *q, IntType name, RealFlow* u_n, RealFlow* v_n, RealFlow* w_n)
{
    IntType  i, j, c1, c2, type, p1;
    IntType  nTNode = grid->GetNTNode();
    IntType  nBFace = grid->GetNBFace();
    IntType  nTCell = grid->GetNTCell();
    IntType  n      = nTCell+nBFace;
    
    IntType  *f2c   = grid->Getf2c();
    IntType  *nNPF  = grid->GetnNPF();
    IntType  *nNPC  = CalnNPC(grid);
    IntType  **C2N  = CalC2N(grid);
    IntType  **F2N  = CalF2N(grid);
    RealGeom *x     = grid->GetX();
    RealGeom *y     = grid->GetY();
    RealGeom *z     = grid->GetZ();
    RealGeom *xfc   = grid->GetXfc();
    RealGeom *yfc   = grid->GetYfc();
    RealGeom *zfc   = grid->GetZfc();
    RealGeom dx, dy, dz, wr;
    //node to cell connection:
    IntType* nCPN = CalnCPN(grid);
    IntType** N2C = CalN2C(grid); 

    BCRecord **bcr = grid->Getbcr();
    if(grid->GetWeightNodeDist() == NULL){
        ComputeWeight3D_Node(grid);  //距离分之一权
    } 
    RealGeom *WeightNode     = grid->GetWeightNodeDist();
    IntType  *Nmark          = grid->GetNodeType();
    RealGeom **WeightNodeC2N = grid->GetWeightNodeC2N();

    RealGeom** WeightNodeN2C = grid->GetWeightNodeN2C();
    RealGeom** WeightNodeBFace2C = grid->GetWeightNodeBFace2C();
    
    //求顶点速度需要的量
    IntType *nodesymm = 0;
    RealFlow vn;
    RealGeom *xfn_n_symm,*yfn_n_symm,*zfn_n_symm;
    
    nodesymm = (IntType *)grid->GetDataPtr(INT, nTNode, "nodesymm");
    if(!nodesymm){
        FindNodeSYMM(grid);
        nodesymm = (IntType *)grid->GetDataPtr(INT, nTNode, "nodesymm");
    }

    //at u gradient calculation, work out u_n, v_n and w_n for the v and w gradient calculation
    if (name == 1) {
        RealFlow* u, * v, * w, * facu, * facv, * facw;

        u = (RealFlow*)grid->GetDataPtr(REAL_FLOW, n, "u");
        v = (RealFlow*)grid->GetDataPtr(REAL_FLOW, n, "v");
        w = (RealFlow*)grid->GetDataPtr(REAL_FLOW, n, "w");
#ifdef FS_OPENMP
#pragma omp parallel for
#endif 
        for(IntType i=0;i<nTNode;i++){
            u_n[i] = 0.0;
            v_n[i] = 0.0;
            w_n[i] = 0.0;
        }

        facu = NULL;
        facv = NULL;
        facw = NULL;
        mfmem::snew_array_1D(facu, nBFace, dmrfl);
        mfmem::snew_array_1D(facv, nBFace, dmrfl);
        mfmem::snew_array_1D(facw, nBFace, dmrfl);

        //计算物理边界面心的物理值
#ifdef FS_OPENMP
#pragma omp parallel for
#endif        
        for(IntType i=0; i<nBFace; i++){
            IntType type = bcr[i]->GetType();
            if(type==INTERFACE || type==SYMM) continue;
            IntType c1 = f2c[2*i];
            IntType c2 = f2c[2*i+1];
            facu[i] = u[c1]+u[c2];
            facu[i]*= 0.5;
            facv[i] = v[c1]+v[c2];
            facv[i]*= 0.5;
            facw[i] = w[c1]+w[c2];
            facw[i]*= 0.5;
        
        }

        //利用物理边界面心的值，计算物理边界点
        for (IntType i = 0; i < nBFace; i++) {
            type = bcr[i]->GetType();
            if (type != WALL) continue;
            for (j = 0; j < nNPF[i]; j++) {
                p1 = F2N[i][j];
                wr = WeightNodeBFace2C[i][j];

                if (nodesymm[p1] == 1) {
                    u_n[p1] += facu[i] * wr;
                    v_n[p1] += facv[i] * wr;
                    w_n[p1] += facw[i] * wr;
                }
            }
        }
        for (IntType i = 0; i < nBFace; i++) {
            type = bcr[i]->GetType();
            if (type != FAR_FIELD)  continue;
            for (j = 0; j < nNPF[i]; j++) {
                p1 = F2N[i][j];
                if (Nmark[p1] == WALL) continue;
                wr = WeightNodeBFace2C[i][j];

                if (nodesymm[p1] == 1) {
                    u_n[p1] += facu[i] * wr;
                    v_n[p1] += facv[i] * wr;
                    w_n[p1] += facw[i] * wr;
                }
            }
        }
        for (IntType i = 0; i < nBFace; i++) {
            type = bcr[i]->GetType();
            if (type == WALL || type == SYMM || type == FAR_FIELD || type == INTERFACE) continue;
            for (j = 0; j < nNPF[i]; j++) {
                p1 = F2N[i][j];
                if (Nmark[p1] == WALL || Nmark[p1] == FAR_FIELD) continue;
                wr = WeightNodeBFace2C[i][j];

                if (nodesymm[p1] == 1) {
                    u_n[p1] += facu[i] * wr;
                    v_n[p1] += facv[i] * wr;
                    w_n[p1] += facw[i] * wr;
                }
            }
        }
#ifdef FS_OPENMP
#pragma omp parallel for
#endif  
        for (IntType i = 0; i < nTNode; i++) {
            if (Nmark[i] != 0) continue;
            for (IntType j = 0; j < nCPN[i]; j++) {
                IntType cellx = N2C[i][j];
                if (nodesymm[i] == 1) {
                    u_n[i] += u[cellx] * WeightNodeN2C[i][j];
                    v_n[i] += v[cellx] * WeightNodeN2C[i][j];
                    w_n[i] += w[cellx] * WeightNodeN2C[i][j];
                }
            }
        }
#ifdef MPICH
        grid->CommInternodeDataMPI(u_n);
        grid->CommInternodeDataMPI(v_n);
        grid->CommInternodeDataMPI(w_n);
#endif
#ifdef FS_OPENMP
#pragma omp parallel for
#endif  
        for (IntType i = 0; i < nTNode; i++) {
            if (nodesymm[i] == 1) {
                u_n[i] /= (WeightNode[i] + TINY);
                v_n[i] /= (WeightNode[i] + TINY);
                w_n[i] /= (WeightNode[i] + TINY);
            }
        }

        mfmem::sdel_array_1D(facu);
        mfmem::sdel_array_1D(facv);
        mfmem::sdel_array_1D(facw);
    }  
    xfn_n_symm = (RealFlow *)grid->GetDataPtr(REAL_GEOM, nTNode, "xfn_n_symm");
    yfn_n_symm = (RealFlow *)grid->GetDataPtr(REAL_GEOM, nTNode, "yfn_n_symm");
    zfn_n_symm = (RealFlow *)grid->GetDataPtr(REAL_GEOM, nTNode, "zfn_n_symm");
    
#ifdef FS_OPENMP
#pragma omp parallel for
#endif    
    for(IntType i=0; i<nTNode; i++) 
        q_n[i] = 0.0;
    
    //计算物理边界点的值，使用物理面的值进行加权计算，不包括并行边界和对称面边界
    //计算物理边界面心的物理值
    RealFlow *facq = NULL;
    mfmem::snew_array_1D(facq, nBFace, dmrfl);
#ifdef FS_OPENMP
#pragma omp parallel for
#endif      
    for(IntType i=0; i<nBFace; i++){
        facq[i] = 0.0;
        IntType type = bcr[i]->GetType();
        if(type==INTERFACE || type==SYMM) continue;
        IntType c1 = f2c[2*i];
        IntType c2 = f2c[2*i+1];
        facq[i] =  q[c1]+q[c2];
        facq[i]*= 0.5;
        
    }
    
    //利用物理边界面心的值，计算物理边界点
    for(IntType i=0; i<nBFace; i++){
        type = bcr[i]->GetType();
        if(type != WALL) continue;
        for(j=0; j<nNPF[i]; j++){
            p1 = F2N[i][j];
            wr = WeightNodeBFace2C[i][j];
            q_n[p1] += facq[i] * wr;
        }
    }
    for(IntType i=0; i<nBFace; i++){
        type = bcr[i]->GetType();
        if(type != FAR_FIELD)  continue;
        for(j=0; j<nNPF[i]; j++){
            p1 = F2N[i][j];
            if(Nmark[p1]==WALL) continue;
            wr = WeightNodeBFace2C[i][j];
            q_n[p1] += facq[i] * wr;
        }
    }
    for(IntType i=0; i<nBFace; i++){
        type = bcr[i]->GetType();
        if(type == WALL || type == SYMM || type == FAR_FIELD || type == INTERFACE) continue;
        for(j=0; j<nNPF[i]; j++){
            p1 = F2N[i][j];
            if(Nmark[p1]==WALL || Nmark[p1]==FAR_FIELD) continue;
            wr = WeightNodeBFace2C[i][j];
            q_n[p1] += facq[i] * wr;
        }
    }
    
    //计算其他点的物理值，使用与其相相邻的控制体体心值
#ifdef FS_OPENMP
#pragma omp parallel for
    for (IntType i = 0; i < nTNode; i++) {
        if (Nmark[i] != 0) continue;
        for (IntType j = 0; j < nCPN[i]; j++) {
            IntType cellx = N2C[i][j];
            q_n[i] += q[cellx] * WeightNodeN2C[i][j];
        }
    }
#else
    for(IntType i=0; i<nTCell; i++){
        for(j=0; j<nNPC[i]; j++){
            p1 = C2N[i][j];
            if(Nmark[p1] != 0) continue; 
            q_n[p1] += q[i] * WeightNodeC2N[i][j];
        }
    }
#endif
    //传递并行边界点的加权值
#ifdef MPICH
    grid->CommInternodeDataMPI(q_n);
#endif
#ifdef FS_OPENMP
#pragma omp parallel for
#endif       
    for(IntType i=0; i<nTNode; i++){
        q_n[i] /= (WeightNode[i]+TINY);
    }
    
    //修正对称面顶点的速度
#ifdef FS_OPENMP
#pragma omp parallel for
#endif      
    for(IntType i=0;i<nTNode;i++){
        if(nodesymm[i] != 1) continue;
            
        RealGeom vn = u_n[i]*xfn_n_symm[i]+v_n[i]*yfn_n_symm[i]+w_n[i]*zfn_n_symm[i];
        if(name == 1)
            q_n[i] = u_n[i]-vn*xfn_n_symm[i];
        else if(name == 2)
            q_n[i] = v_n[i]-vn*yfn_n_symm[i];
        else if(name == 3)
            q_n[i] = w_n[i]-vn*zfn_n_symm[i];
    }        

    mfmem::sdel_array_1D(facq);    
}


void CompNodeVar3D_dist(PolyGrid* grid, RealFlow* q_n, RealFlow* q)//dingxin-add
{
    IntType  i, j, c1, c2, type, p1;
    IntType  nTNode = grid->GetNTNode();
    IntType  nBFace = grid->GetNBFace();
    IntType  nTCell = grid->GetNTCell();
    IntType  n = nTCell + nBFace;

    IntType* f2c = grid->Getf2c();
    IntType* nNPF = grid->GetnNPF();
    IntType* nNPC = CalnNPC(grid);
    IntType** C2N = CalC2N(grid);
    IntType** F2N = CalF2N(grid);
    RealGeom* x = grid->GetX();
    RealGeom* y = grid->GetY();
    RealGeom* z = grid->GetZ();
    RealGeom* xfc = grid->GetXfc();
    RealGeom* yfc = grid->GetYfc();
    RealGeom* zfc = grid->GetZfc();
    RealGeom dx, dy, dz, wr;
    //node to cell connection:
    IntType* nCPN = CalnCPN(grid);
    IntType** N2C = CalN2C(grid);

    BCRecord** bcr = grid->Getbcr();
    if (grid->GetWeightNodeDist() == NULL) {
        ComputeWeight3D_Node(grid);  //距离分之一权
    }
    RealGeom* WeightNode = grid->GetWeightNodeDist();
    IntType* Nmark = grid->GetNodeType();
    RealGeom** WeightNodeC2N = grid->GetWeightNodeC2N();
    RealGeom** WeightNodeBFace2C = grid->GetWeightNodeBFace2C();

    RealGeom** WeightNodeN2C = grid->GetWeightNodeN2C();   

    for (IntType i = 0; i < nTNode; i++)
        q_n[i] = 0.0;

    //计算物理边界点的值，使用物理面的值进行加权计算，不包括并行边界和对称面边界
    //计算物理边界面心的物理值
    RealFlow* facq = NULL;
    mfmem::snew_array_1D(facq, nBFace, dmrfl);
#ifdef FS_OPENMP
#pragma omp parallel for
#endif    
    for (IntType i = 0; i < nBFace; i++) {
        facq[i] = 0.0;
        IntType type = bcr[i]->GetType();
        if (type == INTERFACE || type == SYMM) continue;
        IntType c1 = f2c[2 * i];
        IntType c2 = f2c[2 * i + 1];
        facq[i] = q[c1] + q[c2];
        facq[i] *= 0.5;
    }

    //利用物理边界面心的值，计算物理边界点
    for (i = 0; i < nBFace; i++) {
        type = bcr[i]->GetType();
        if (type != WALL) continue;
        for (j = 0; j < nNPF[i]; j++) {
            p1 = F2N[i][j];
            q_n[p1] += facq[i] * WeightNodeBFace2C[i][j];
        }
    }
    for (i = 0; i < nBFace; i++) {
        type = bcr[i]->GetType();
        if (type != FAR_FIELD)  continue;
        for (j = 0; j < nNPF[i]; j++) {
            p1 = F2N[i][j];
            if (Nmark[p1] == WALL) continue;
            q_n[p1] += facq[i] * WeightNodeBFace2C[i][j];

        }
    }
    for (i = 0; i < nBFace; i++) {
        type = bcr[i]->GetType();
        if (type == WALL || type == SYMM || type == FAR_FIELD || type == INTERFACE) continue;
        for (j = 0; j < nNPF[i]; j++) {
            p1 = F2N[i][j];
            if (Nmark[p1] == WALL || Nmark[p1] == FAR_FIELD) continue;
            q_n[p1] += facq[i] * WeightNodeBFace2C[i][j];

        }
    }

    //计算其他点的物理值，使用与其相相邻的控制体体心值
/*
    
*/
#ifdef FS_OPENMP
#pragma omp parallel for
    for (IntType i = 0; i < nTNode; i++) {
        if (Nmark[i] != 0) continue;
        for (IntType j = 0; j < nCPN[i]; j++) {
            IntType cellx = N2C[i][j];
            q_n[i] += q[cellx] * WeightNodeN2C[i][j];
        }
    }
#else
    for (i = 0; i < nTCell; i++) {
        for (j = 0; j < nNPC[i]; j++) {
            p1 = C2N[i][j];
            if (Nmark[p1] != 0) continue;
            q_n[p1] += q[i] * WeightNodeC2N[i][j];
        }
    }
#endif

    //传递并行边界点的加权值
#ifdef MPICH
    grid->CommInternodeDataMPI(q_n);
#endif

#ifdef FS_OPENMP
#pragma omp parallel for
#endif
    for (IntType i = 0; i < nTNode; i++) {
        q_n[i] /= (WeightNode[i] + TINY);
    }

    mfmem::sdel_array_1D(facq);
}

/*******************************************************************************\
          Find differences between the value in every cell and 
          maximum/minimum in the neighboring cells
\*******************************************************************************/
void MaxMinDiff(RealFlow *dmax, RealFlow *dmin, RealFlow *q, BCRecord **bcr,
                IntType *f2c, IntType nTCell, IntType nBFace, IntType nTFace)
{
    IntType i, c1, c2, count, type;

    // Find the maximum and minimum in the neighbor of each cell
    for(i=0; i<nTCell; i++) {
        dmax[i] =  q[i];
        dmin[i] =  q[i];
    }

    // Interfaces only
    for(i=0; i<nBFace; i++) {
        count    = 2*i;
        c1       = f2c[count++];
        c2       = f2c[count]; 
               
        type = bcr[i]->GetType();
        if(type != INTERFACE) continue;
              
        dmax[c1] = MAX(dmax[c1], q[c2]);
        dmin[c1] = MIN(dmin[c1], q[c2]);
    }


    for(i=nBFace; i<nTFace; i++) {
        count = i+i;
        c1       = f2c[count];
        c2       = f2c[count+1];

        dmax[c1] = MAX(dmax[c1], q[c2]);
        dmin[c1] = MIN(dmin[c1], q[c2]);
       
        dmax[c2] = MAX(dmax[c2], q[c1]);
        dmin[c2] = MIN(dmin[c2], q[c1]);
    }
    
    // Get the maximum and the minimum difference
    for(i=0; i<nTCell; i++) {
        dmax[i] -=  q[i];
        dmin[i] -=  q[i];
    }
}


/*******************************************************************************\
          Find differences between the value in every cell and 
          maximum/minimum in the neighboring cells
\*******************************************************************************/
void MaxMinDiff(RealFlow *dmax, RealFlow *dmin, RealFlow *q, PolyGrid *grid)
{
    IntType i, c1, c2, count, type;
    IntType  *f2c   = grid->Getf2c();
    BCRecord **bcr  = grid->Getbcr();
    IntType  nTFace = grid->GetNTFace();
    IntType  nBFace = grid->GetNBFace();
    IntType  nTCell = grid->GetNTCell();
    IntType  pfacenum = nBFace - grid->GetNIFace();
    // Find the maximum and minimum in the neighbor of each cell
#ifdef FS_OPENMP
#pragma omp parallel for
#endif    
    for(i=0; i<nTCell; i++) {
        dmax[i] =  q[i];
        dmin[i] =  q[i];
    }

    //Group color openmp
#if (defined FS_OPENMP) && (defined GroupColor)
    if (grid->GroupColorSuccess) {
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
#pragma omp parallel for private(i,count,c1,c2,type) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                count = 2 * i;
                c1 = f2c[count++];
                c2 = f2c[count];

                type = bcr[i]->GetType();
                if (type != INTERFACE) continue;

                dmax[c1] = MAX(dmax[c1], q[c2]);
                dmin[c1] = MIN(dmin[c1], q[c2]);
            }
        }
        // zone boundary face
        count = 2 * pfacenum;
        for (i = pfacenum; i < nBFace; i++) {
            c1 = f2c[count++];
            c2 = f2c[count++];

            type = bcr[i]->GetType();
            if (type != INTERFACE) continue;

            dmax[c1] = MAX(dmax[c1], q[c2]);
            dmin[c1] = MIN(dmin[c1], q[c2]);
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
#pragma omp parallel for private(i,count,c1,c2) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                count = i * 2;
                c1 = f2c[count];
                c2 = f2c[count+1];

                dmax[c1] = MAX(dmax[c1], q[c2]);
                dmin[c1] = MIN(dmin[c1], q[c2]);

                dmax[c2] = MAX(dmax[c2], q[c1]);
                dmin[c2] = MIN(dmin[c2], q[c1]);
            }
        }
    }
    else {
        for (i = 0; i < nBFace; i++) {
            count = 2 * i;
            c1 = f2c[count++];
            c2 = f2c[count];

            type = bcr[i]->GetType();
            if (type != INTERFACE) continue;

            dmax[c1] = MAX(dmax[c1], q[c2]);
            dmin[c1] = MIN(dmin[c1], q[c2]);
        }

        count = 2 * nBFace;
        for (i = nBFace; i < nTFace; i++) {
            c1 = f2c[count++];
            c2 = f2c[count++];

            dmax[c1] = MAX(dmax[c1], q[c2]);
            dmin[c1] = MIN(dmin[c1], q[c2]);

            dmax[c2] = MAX(dmax[c2], q[c1]);
            dmin[c2] = MIN(dmin[c2], q[c1]);
        }
    }

#elif (defined FS_OPENMP) && (defined FaceColoring)
    IntType    bfacegroup_num, ifacegroup_num;
    IntType    *grid_bfacegroup, *grid_ifacegroup;
    ifacegroup_num = (*grid).ifacegroup.size();
    bfacegroup_num = (*grid).bfacegroup.size();
    grid_bfacegroup = NULL;
    grid_ifacegroup = NULL;
    mfmem::snew_array_1D(grid_bfacegroup, bfacegroup_num, dmrfl);
    mfmem::snew_array_1D(grid_ifacegroup, ifacegroup_num, dmrfl);
    for (int i = 0; i < bfacegroup_num; i++) {
        grid_bfacegroup[i] = (*grid).bfacegroup[i];
    }
    for (int i = 0; i < ifacegroup_num; i++){
        grid_ifacegroup[i] = (*grid).ifacegroup[i];
    }
    // Interfaces only
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
            IntType  c1, c2, count, type;
            count    = 2*i;
            c1       = f2c[count];
            c2       = f2c[count + 1];                
            type = bcr[i]->GetType();
            if(type != INTERFACE) continue;              
            dmax[c1] = MAX(dmax[c1], q[c2]);
            dmin[c1] = MIN(dmin[c1], q[c2]);
        }
    }
    for (IntType i = pfacenum; i < nBFace; i++) {
        IntType  c1, c2, count, type;
        count    = 2*i;
        c1       = f2c[count++];
        c2       = f2c[count]; 
               
        type = bcr[i]->GetType();
        if(type != INTERFACE) continue;
              
        dmax[c1] = MAX(dmax[c1], q[c2]);
        dmin[c1] = MIN(dmin[c1], q[c2]);
    }
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
            IntType  c1, c2, count;
            count    = 2*i;
            c1       = f2c[count];
            c2       = f2c[count + 1];                
            dmax[c1] = MAX(dmax[c1], q[c2]);
            dmin[c1] = MIN(dmin[c1], q[c2]);
       
            dmax[c2] = MAX(dmax[c2], q[c1]);
            dmin[c2] = MIN(dmin[c2], q[c1]);        
        }
    }
    mfmem::sdel_array_1D(grid_bfacegroup);
    mfmem::sdel_array_1D(grid_ifacegroup);

#elif (defined FS_OPENMP) && (defined Reduction)//Manual reduction
    IntType* nFPC = CalnFPC(grid);
    IntType** C2F = CalC2F(grid);
    IntType j, face;
#pragma omp parallel for private(i,j,c1,c2,face,type)
    for (i = 0; i < nTCell; i++) {
        for (j = 0; j < nFPC[i]; j++) {
            face = C2F[i][j];
            if (face < nBFace) {
                type = bcr[face]->GetType();
                if (type != INTERFACE) continue;
            }
            c1 = f2c[face + face];
            c2 = f2c[face + face + 1];
            if (i == c1) {
                dmax[c1] = MAX(dmax[c1], q[c2]);
                dmin[c1] = MIN(dmin[c1], q[c2]);
            }
            else if (i == c2) {
                dmax[c2] = MAX(dmax[c2], q[c1]);
                dmin[c2] = MIN(dmin[c2], q[c1]);
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
    #pragma omp parallel for private(t,i,k,startFace,endFace,c1,c2,face,type)
        for (t = 0; t < threads; t++) {
            //Boundary faces
            startFace = grid->idx_pthreads_bface[t];
            endFace = grid->idx_pthreads_bface[t + 1];
            for (i = startFace; i < endFace; i++) {
                face = grid->id_division_bface[i];
                c1 = f2c[2 * face];
                c2 = f2c[2 * face + 1];
                type = bcr[face]->GetType();
                if (type != INTERFACE) continue;

                dmax[c1] = MAX(dmax[c1], q[c2]);
                dmin[c1] = MIN(dmin[c1], q[c2]);
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
                c1 = f2c[2 * face];
                c2 = f2c[2 * face + 1];
                if (abs(k) < nTFace) {
                    dmax[c1] = MAX(dmax[c1], q[c2]);
                    dmin[c1] = MIN(dmin[c1], q[c2]);

                    dmax[c2] = MAX(dmax[c2], q[c1]);
                    dmin[c2] = MIN(dmin[c2], q[c1]);
                }
                else {
                    if (k > 0) {
                        dmax[c1] = MAX(dmax[c1], q[c2]);
                        dmin[c1] = MIN(dmin[c1], q[c2]);
                    }
                    else {
                        dmax[c2] = MAX(dmax[c2], q[c1]);
                        dmin[c2] = MIN(dmin[c2], q[c1]);
                    }
                }
            }
        }
    }
#elif (defined FS_OPENMP) && (defined DIVCON) //D&C TREE
#pragma omp parallel
    {
    #pragma omp single nowait
        tree_traversal(grid->treeHead, dmax, dmin, q, f2c, bcr);
    }

#else
    // Interfaces only
    for(i=0; i<nBFace; i++) {
        count    = 2*i;
        c1       = f2c[count++];
        c2       = f2c[count]; 
               
        type = bcr[i]->GetType();
        if(type != INTERFACE) continue;
              
        dmax[c1] = MAX(dmax[c1], q[c2]);
        dmin[c1] = MIN(dmin[c1], q[c2]);
    }

    count = 2*nBFace;
    for(i=nBFace; i<nTFace; i++) {
        c1       = f2c[count++];
        c2       = f2c[count++];

        dmax[c1] = MAX(dmax[c1], q[c2]);
        dmin[c1] = MIN(dmin[c1], q[c2]);
       
        dmax[c2] = MAX(dmax[c2], q[c1]);
        dmin[c2] = MIN(dmin[c2], q[c1]);
    }
#endif    
    // Get the maximum and the minimum difference
#ifdef FS_OPENMP
#pragma omp parallel for
#endif     
    for(i=0; i<nTCell; i++) {
        dmax[i] -=  q[i];
        dmin[i] -=  q[i];
    }
}


/*******************************************************************************\
              Set values for Vencat limiters in 3D
\*******************************************************************************/
void VencatLimiter(PolyGrid *grid, RealFlow *limit, RealFlow *q, RealFlow *dqdx, RealFlow *dqdy, RealFlow *dqdz, IntType name)
{
    IntType  nTFace = grid->GetNTFace();
    IntType  nBFace = grid->GetNBFace();
    IntType  nTCell = grid->GetNTCell();
    IntType  nIFace = grid->GetNIFace();
    IntType  n      = nTCell+nBFace;    
    IntType  *f2c   = grid->Getf2c();
    RealGeom *xfc   = grid->GetXfc();
    RealGeom *yfc   = grid->GetYfc();
    RealGeom *zfc   = grid->GetZfc();
    RealGeom *xcc   = grid->GetXcc();
    RealGeom *ycc   = grid->GetYcc();
    RealGeom *zcc   = grid->GetZcc();
    BCRecord **bcr  = grid->Getbcr();
    RealGeom *vol   = grid->GetCellVol();
    
    IntType  i, c1, c2, count;
    RealGeom dx, dy, dz, eps, eps_tmp;
    RealFlow dq_face, tmp;
    RealFlow *p, *rho, gam, p_bar;
    IntType ifacenum = nTFace - nBFace;
    IntType pfacenum = nBFace - nIFace;
    
    if(name>0 && name<4){
        rho = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
        p   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
        grid->GetData(&gam, REAL_FLOW, 1, "gam");    
    }
    if(name>0){
        grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar"); 
    }
    
    RealFlow vol_avg = grid->GetVolAvg();
    assert(vol_avg > 0.0); //volumn average must exist
     
    // Need tempory arrays for the differences between the value in every cell and 
    // maximum/minimum in the neighboring cells   
    RealFlow *dmax = NULL;
    RealFlow *dmin = NULL;
    mfmem::snew_array_1D(dmax,nTCell,dmrfl);
    mfmem::snew_array_1D(dmin,nTCell,dmrfl);
    assert(dmax != 0);
    assert(dmin != 0);
 
    // Find the the differences for q
#ifdef FS_OPENMP
    MaxMinDiff(dmax, dmin, q, grid);
#else
    MaxMinDiff(dmax, dmin, q, bcr, f2c, nTCell, nBFace, nTFace);
#endif

    RealFlow eps_vencat=1.0;
    grid->GetData(&eps_vencat, REAL_FLOW, 1, "eps_vencat",0);
    eps_tmp = eps_vencat*eps_vencat*eps_vencat/vol_avg;  
    //initial limit[i]
#ifdef FS_OPENMP
#pragma omp parallel for
#endif
    for(IntType i=0;i<nTCell;i++)
      limit[i] = BIG;
#ifdef FS_OPENMP
#pragma omp parallel for
#endif      
    for(IntType i=nTCell;i<n;i++)
      limit[i] = 1.0;

    RealFlow *espcell = NULL;
    mfmem::snew_array_1D(espcell,nTCell,dmrfl);
    switch(name){
          case 0:
#ifdef FS_OPENMP
#pragma omp parallel for
#endif
              for(IntType i=0;i<nTCell;i++){
                  espcell[i] = eps_tmp*vol[i]*q[i]*q[i];
              }
              break;
          case 1:
          case 2:
          case 3:
#ifdef FS_OPENMP
#pragma omp parallel for private(tmp)
#endif          
              for(IntType i=0;i<nTCell;i++){
                  tmp  = gam*(p[i]+p_bar)/rho[i];
                  espcell[i] = eps_tmp*vol[i]*tmp;
              }
              break;
          case 4:
#ifdef FS_OPENMP
#pragma omp parallel for 
#endif
              for(IntType i=0;i<nTCell;i++){
                  espcell[i] = eps_tmp*vol[i]*(q[i]+p_bar)*(q[i]+p_bar);
              }
              break;
        }
    //Group color openmp
#if (defined FS_OPENMP) && (defined GroupColor)
    if (grid->GroupColorSuccess) {
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
#pragma omp parallel for private(i,c1,dx,dy,dz,dq_face,eps,tmp) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                c1 = f2c[2 * i];

                dx = xfc[i] - xcc[c1];
                dy = yfc[i] - ycc[c1];
                dz = zfc[i] - zcc[c1];
                dq_face = dqdx[c1] * dx + dqdy[c1] * dy + dqdz[c1] * dz;

                eps = espcell[c1];
                if (EqualZero(dq_face))
                    tmp = 1.0;
                else {
                    if (dq_face > 0.0) {
                        tmp = VenFun(dmax[c1], dq_face, eps);
                    }
                    else {
                        tmp = VenFun(dmin[c1], dq_face, eps);
                    }
                    tmp /= dq_face;
                }
                limit[c1] = MIN(limit[c1], tmp);
            }
        }
        // zone boundary face
        count = 2 * pfacenum;
        for (i = pfacenum; i < nBFace; i++) {
            c1 = f2c[count++];
            count++;

            dx = xfc[i] - xcc[c1];
            dy = yfc[i] - ycc[c1];
            dz = zfc[i] - zcc[c1];
            dq_face = dqdx[c1] * dx + dqdy[c1] * dy + dqdz[c1] * dz;

            eps = espcell[c1];
            if (EqualZero(dq_face))
                tmp = 1.0;
            else {
                if (dq_face > 0.0) {
                    tmp = VenFun(dmax[c1], dq_face, eps);
                }
                else {
                    tmp = VenFun(dmin[c1], dq_face, eps);
                }
                tmp /= dq_face;
            }
            limit[c1] = MIN(limit[c1], tmp);
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
#pragma omp parallel for private(i,count,c1,c2,dx,dy,dz,dq_face,eps,tmp) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                count = 2 * i;
                c1 = f2c[count];
                c2 = f2c[count + 1];

                dx = xfc[i] - xcc[c1];
                dy = yfc[i] - ycc[c1];
                dz = zfc[i] - zcc[c1];
                dq_face = dqdx[c1] * dx + dqdy[c1] * dy + dqdz[c1] * dz;

                eps = espcell[c1];
                if (EqualZero(dq_face))
                    tmp = 1.0;
                else {
                    if (dq_face > 0.0) {
                        tmp = VenFun(dmax[c1], dq_face, eps);
                    }
                    else {
                        tmp = VenFun(dmin[c1], dq_face, eps);
                    }
                    tmp /= dq_face;
                }
                limit[c1] = MIN(limit[c1], tmp);

                dx = xfc[i] - xcc[c2];
                dy = yfc[i] - ycc[c2];
                dz = zfc[i] - zcc[c2];
                dq_face = dqdx[c2] * dx + dqdy[c2] * dy + dqdz[c2] * dz;

                eps = espcell[c2];

                if (EqualZero(dq_face))
                    tmp = 1.0;
                else {
                    if (dq_face > 0.0) {
                        tmp = VenFun(dmax[c2], dq_face, eps);
                    }
                    else {
                        tmp = VenFun(dmin[c2], dq_face, eps);
                    }
                    tmp /= dq_face;
                }

                limit[c2] = MIN(limit[c2], tmp);
            }
        }
    }
    else {
        count = 0;
        for (i = 0; i < nBFace; i++) {
            c1 = f2c[count++];
            count++;

            dx = xfc[i] - xcc[c1];
            dy = yfc[i] - ycc[c1];
            dz = zfc[i] - zcc[c1];
            dq_face = dqdx[c1] * dx + dqdy[c1] * dy + dqdz[c1] * dz;

            eps = espcell[c1];
            if (EqualZero(dq_face))
                tmp = 1.0;
            else {
                if (dq_face > 0.0) {
                    tmp = VenFun(dmax[c1], dq_face, eps);
                }
                else {
                    tmp = VenFun(dmin[c1], dq_face, eps);
                }
                tmp /= dq_face;
            }
            limit[c1] = MIN(limit[c1], tmp);
        }
        for (i = nBFace; i < nTFace; i++) {
            count = 2 * i;
            c1 = f2c[count];
            c2 = f2c[count + 1];

            dx = xfc[i] - xcc[c1];
            dy = yfc[i] - ycc[c1];
            dz = zfc[i] - zcc[c1];
            dq_face = dqdx[c1] * dx + dqdy[c1] * dy + dqdz[c1] * dz;

            eps = espcell[c1];
            if (EqualZero(dq_face))
                tmp = 1.0;
            else {
                if (dq_face > 0.0) {
                    tmp = VenFun(dmax[c1], dq_face, eps);
                }
                else {
                    tmp = VenFun(dmin[c1], dq_face, eps);
                }
                tmp /= dq_face;
            }
            limit[c1] = MIN(limit[c1], tmp);

            dx = xfc[i] - xcc[c2];
            dy = yfc[i] - ycc[c2];
            dz = zfc[i] - zcc[c2];
            dq_face = dqdx[c2] * dx + dqdy[c2] * dy + dqdz[c2] * dz;

            eps = espcell[c2];

            if (EqualZero(dq_face))
                tmp = 1.0;
            else {
                if (dq_face > 0.0) {
                    tmp = VenFun(dmax[c2], dq_face, eps);
                }
                else {
                    tmp = VenFun(dmin[c2], dq_face, eps);
                }
                tmp /= dq_face;
            }

            limit[c2] = MIN(limit[c2], tmp);
        }
    }

#elif (defined FS_OPENMP) && (defined FaceColoring)
    IntType    bfacegroup_num, ifacegroup_num;
    IntType    *grid_bfacegroup, *grid_ifacegroup;
    ifacegroup_num = (*grid).ifacegroup.size();
    bfacegroup_num = (*grid).bfacegroup.size();
    grid_bfacegroup = NULL;
    grid_ifacegroup = NULL;
    mfmem::snew_array_1D(grid_bfacegroup, bfacegroup_num, dmrfl);
    mfmem::snew_array_1D(grid_ifacegroup, ifacegroup_num, dmrfl);
    for (int i = 0; i < bfacegroup_num; i++) {
        grid_bfacegroup[i] = (*grid).bfacegroup[i];
    }
    for (int i = 0; i < ifacegroup_num; i++){
        grid_ifacegroup[i] = (*grid).ifacegroup[i];
    }    
    // Boundary faces. 
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
            IntType  c1, c2, count;
            RealGeom dx, dy, dz, eps;
            RealFlow dq_face, tmp;
            RealFlow *p, *rho, gam, p_bar;
            count    = 2*i;
            c1       = f2c[count];
            
            dx       = xfc[i] - xcc[c1];
            dy       = yfc[i] - ycc[c1];
            dz       = zfc[i] - zcc[c1];
            dq_face  = dqdx[c1]*dx + dqdy[c1]*dy + dqdz[c1]*dz;
        
            eps = espcell[c1];
            if(EqualZero(dq_face))
            tmp = 1.0;
            else{ 
                if(dq_face > 0.0){
                    tmp  = VenFun(dmax[c1], dq_face, eps);
                }else{
                    tmp  = VenFun(dmin[c1], dq_face, eps);
                }
                tmp     /= dq_face; 
            }
            limit[c1] = MIN(limit[c1], tmp);
        }
                       
    }
    for (i = pfacenum; i < nBFace; i++) {
        count = 2 * i;
        c1 = f2c[count];
        dx       = xfc[i] - xcc[c1];
        dy       = yfc[i] - ycc[c1];
        dz       = zfc[i] - zcc[c1];
        dq_face  = dqdx[c1]*dx + dqdy[c1]*dy + dqdz[c1]*dz;
        
        eps = espcell[c1];
        if(EqualZero(dq_face))
            tmp = 1.0;
        else{ 
            if(dq_face > 0.0){
                tmp  = VenFun(dmax[c1], dq_face, eps);
            }else{
                tmp  = VenFun(dmin[c1], dq_face, eps);
            }
            tmp     /= dq_face; 
        }
        limit[c1] = MIN(limit[c1], tmp);                
    }
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
            IntType  c1, c2, count;
            RealGeom dx, dy, dz, eps;
            RealFlow dq_face, tmp;
            RealFlow *p, *rho, gam, p_bar;
            count = 2 * i;
            c1      = f2c[count];
            c2      = f2c[count + 1];
            dx      = xfc[i] - xcc[c1];
            dy      = yfc[i] - ycc[c1];
            dz      = zfc[i] - zcc[c1];
            dq_face = dqdx[c1]*dx + dqdy[c1]*dy + dqdz[c1]*dz;
        
            eps = espcell[c1];
            if(EqualZero(dq_face))
                tmp = 1.0;
            else{
                if(dq_face > 0.0){
                    tmp  = VenFun(dmax[c1], dq_face, eps);
                }else{
                    tmp  = VenFun(dmin[c1], dq_face, eps);
                }
                tmp     /= dq_face; 
            }
            limit[c1] = MIN(limit[c1], tmp);
 
            dx      = xfc[i] - xcc[c2];
            dy      = yfc[i] - ycc[c2];
            dz      = zfc[i] - zcc[c2];
            dq_face = dqdx[c2]*dx + dqdy[c2]*dy + dqdz[c2]*dz;

            eps = espcell[c2];
        
            if(EqualZero(dq_face))
                tmp = 1.0;
            else{
                if(dq_face > 0.0){
                    tmp  = VenFun(dmax[c2], dq_face, eps);
                }else{
                    tmp  = VenFun(dmin[c2], dq_face, eps);
                }
                tmp     /= dq_face; 
            }

            limit[c2] = MIN(limit[c2], tmp);
        }
    }
    mfmem::sdel_array_1D(grid_bfacegroup);
    mfmem::sdel_array_1D(grid_ifacegroup);
	
#elif (defined FS_OPENMP) && (defined Reduction)//Manual reduction
    RealFlow* tmp_limit = NULL;
    IntType* nFPC = CalnFPC(grid);
    IntType** C2F = CalC2F(grid);
    IntType j, face;
    mfmem::snew_array_1D(tmp_limit, 2 * nTFace, dmrfl);
#pragma omp parallel for private(i,count,c1,dx,dy,dz,dq_face,eps,tmp)
    for (i = 0; i < nBFace; i++) {
        count = 2 * i;
        c1 = f2c[count];

        dx = xfc[i] - xcc[c1];
        dy = yfc[i] - ycc[c1];
        dz = zfc[i] - zcc[c1];
        dq_face = dqdx[c1] * dx + dqdy[c1] * dy + dqdz[c1] * dz;

        eps = espcell[c1];
        if (EqualZero(dq_face))
            tmp = 1.0;
        else {
            if (dq_face > 0.0) {
                tmp = VenFun(dmax[c1], dq_face, eps);
            }
            else {
                tmp = VenFun(dmin[c1], dq_face, eps);
            }
            tmp /= dq_face;
        }
        tmp_limit[count] = tmp;
    }
#pragma omp parallel for private(i,count,c1,c2,dx,dy,dz,dq_face,eps,tmp)
    for (i = nBFace; i < nTFace; i++) {
        count = 2 * i;
        c1 = f2c[count];
        c2 = f2c[count + 1];

        dx = xfc[i] - xcc[c1];
        dy = yfc[i] - ycc[c1];
        dz = zfc[i] - zcc[c1];
        dq_face = dqdx[c1] * dx + dqdy[c1] * dy + dqdz[c1] * dz;

        eps = espcell[c1];
        if (EqualZero(dq_face))
            tmp = 1.0;
        else {
            if (dq_face > 0.0) {
                tmp = VenFun(dmax[c1], dq_face, eps);
            }
            else {
                tmp = VenFun(dmin[c1], dq_face, eps);
            }
            tmp /= dq_face;
        }
        tmp_limit[count] = tmp;

        dx = xfc[i] - xcc[c2];
        dy = yfc[i] - ycc[c2];
        dz = zfc[i] - zcc[c2];
        dq_face = dqdx[c2] * dx + dqdy[c2] * dy + dqdz[c2] * dz;

        eps = espcell[c2];

        if (EqualZero(dq_face))
            tmp = 1.0;
        else {
            if (dq_face > 0.0) {
                tmp = VenFun(dmax[c2], dq_face, eps);
            }
            else {
                tmp = VenFun(dmin[c2], dq_face, eps);
            }
            tmp /= dq_face;
        }
        tmp_limit[count + 1] = tmp;
    }
#pragma omp parallel for private(i,j,count,c1,c2,face)
    for (i = 0; i < nTCell; i++) {
        for (j = 0; j < nFPC[i]; j++) {
            face = C2F[i][j];
            count = 2 * face;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            if (i == c1) {
                limit[c1] = MIN(limit[c1], tmp_limit[count]);
            }
            else if (i == c2) {
                limit[c2] = MIN(limit[c2], tmp_limit[count + 1]);
            }
            else {
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
    }
    mfmem::sdel_array_1D(tmp_limit);
#elif (defined FS_OPENMP) && (defined DIVREP)//Division & replication
    IntType threads = grid->threads;
    IntType startFace, endFace, t, k, face;
    if (grid->DivRepSuccess) {
    #pragma omp parallel for private(t,i,k,startFace,endFace,c1,c2,face,dx,dy,dz,dq_face,eps,tmp)
        for (t = 0; t < threads; t++) {
            //Boundary faces
            startFace = grid->idx_pthreads_bface[t];
            endFace = grid->idx_pthreads_bface[t + 1];
            for (i = startFace; i < endFace; i++) {
                face = grid->id_division_bface[i];
                c1 = f2c[2 * face];
                dx = xfc[face] - xcc[c1];
                dy = yfc[face] - ycc[c1];
                dz = zfc[face] - zcc[c1];
                dq_face = dqdx[c1] * dx + dqdy[c1] * dy + dqdz[c1] * dz;

                eps = espcell[c1];
                if (EqualZero(dq_face))
                    tmp = 1.0;
                else {
                    if (dq_face > 0.0) {
                        tmp = VenFun(dmax[c1], dq_face, eps);
                    }
                    else {
                        tmp = VenFun(dmin[c1], dq_face, eps);
                    }
                    tmp /= dq_face;
                }
                limit[c1] = MIN(limit[c1], tmp);
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
                c1 = f2c[2 * face];
                c2 = f2c[2 * face + 1];
                if (abs(k) < nTFace) {
                    dx = xfc[face] - xcc[c1];
                    dy = yfc[face] - ycc[c1];
                    dz = zfc[face] - zcc[c1];
                    dq_face = dqdx[c1] * dx + dqdy[c1] * dy + dqdz[c1] * dz;

                    eps = espcell[c1];
                    if (EqualZero(dq_face))
                        tmp = 1.0;
                    else {
                        if (dq_face > 0.0) {
                            tmp = VenFun(dmax[c1], dq_face, eps);
                        }
                        else {
                            tmp = VenFun(dmin[c1], dq_face, eps);
                        }
                        tmp /= dq_face;
                    }
                    limit[c1] = MIN(limit[c1], tmp);

                    dx = xfc[face] - xcc[c2];
                    dy = yfc[face] - ycc[c2];
                    dz = zfc[face] - zcc[c2];
                    dq_face = dqdx[c2] * dx + dqdy[c2] * dy + dqdz[c2] * dz;

                    eps = espcell[c2];

                    if (EqualZero(dq_face))
                        tmp = 1.0;
                    else {
                        if (dq_face > 0.0) {
                            tmp = VenFun(dmax[c2], dq_face, eps);
                        }
                        else {
                            tmp = VenFun(dmin[c2], dq_face, eps);
                        }
                        tmp /= dq_face;
                    }

                    limit[c2] = MIN(limit[c2], tmp);
                }
                else {
                    if (k > 0) {
                        dx = xfc[face] - xcc[c1];
                        dy = yfc[face] - ycc[c1];
                        dz = zfc[face] - zcc[c1];
                        dq_face = dqdx[c1] * dx + dqdy[c1] * dy + dqdz[c1] * dz;

                        eps = espcell[c1];
                        if (EqualZero(dq_face))
                            tmp = 1.0;
                        else {
                            if (dq_face > 0.0) {
                                tmp = VenFun(dmax[c1], dq_face, eps);
                            }
                            else {
                                tmp = VenFun(dmin[c1], dq_face, eps);
                            }
                            tmp /= dq_face;
                        }
                        limit[c1] = MIN(limit[c1], tmp);
                    }
                    else {
                        dx = xfc[face] - xcc[c2];
                        dy = yfc[face] - ycc[c2];
                        dz = zfc[face] - zcc[c2];
                        dq_face = dqdx[c2] * dx + dqdy[c2] * dy + dqdz[c2] * dz;

                        eps = espcell[c2];

                        if (EqualZero(dq_face))
                            tmp = 1.0;
                        else {
                            if (dq_face > 0.0) {
                                tmp = VenFun(dmax[c2], dq_face, eps);
                            }
                            else {
                                tmp = VenFun(dmin[c2], dq_face, eps);
                            }
                            tmp /= dq_face;
                        }

                        limit[c2] = MIN(limit[c2], tmp);
                    }
                }
            }
        }
    }
#elif (defined FS_OPENMP) && (defined DIVCON) //D&C TREE
    RealFlow* tmp_limit = NULL;
    mfmem::snew_array_1D(tmp_limit, 2 * (nTFace - nBFace), dmrfl);
#pragma omp parallel
    {
    #pragma omp single nowait
        tree_traversal(grid->treeHead, limit, tmp_limit, f2c, xfc, yfc, zfc,
            xcc, ycc, zcc, dqdx, dqdy, dqdz, espcell, dmax, dmin, nBFace);
    }
    mfmem::sdel_array_1D(tmp_limit);
	
#else
    // Boundary faces. 
    count = 0;
    for(i=0; i<nBFace; i++) {
        c1       = f2c[count++];
        count++;
        
        dx       = xfc[i] - xcc[c1];
        dy       = yfc[i] - ycc[c1];
        dz       = zfc[i] - zcc[c1];
        dq_face  = dqdx[c1]*dx + dqdy[c1]*dy + dqdz[c1]*dz;
        
        eps = espcell[c1];
        if(EqualZero(dq_face))
            tmp = 1.0;
        else{ 
            if(dq_face > 0.0){
                tmp  = VenFun(dmax[c1], dq_face, eps);
            }else{
                tmp  = VenFun(dmin[c1], dq_face, eps);
            }
            tmp     /= dq_face; 
        }
        limit[c1] = MIN(limit[c1], tmp);
    }
    for(i=nBFace; i<nTFace; i++) {
        count = 2*i;
        c1      = f2c[count];
        c2      = f2c[count+1];
        
        dx      = xfc[i] - xcc[c1];
        dy      = yfc[i] - ycc[c1];
        dz      = zfc[i] - zcc[c1];
        dq_face = dqdx[c1]*dx + dqdy[c1]*dy + dqdz[c1]*dz;
        
        eps = espcell[c1];
        if(EqualZero(dq_face))
            tmp = 1.0;
        else{
            if(dq_face > 0.0){
                tmp  = VenFun(dmax[c1], dq_face, eps);
            }else{
                tmp  = VenFun(dmin[c1], dq_face, eps);
            }
            tmp     /= dq_face; 
        }
        limit[c1] = MIN(limit[c1], tmp);
 
        dx      = xfc[i] - xcc[c2];
        dy      = yfc[i] - ycc[c2];
        dz      = zfc[i] - zcc[c2];
        dq_face = dqdx[c2]*dx + dqdy[c2]*dy + dqdz[c2]*dz;

        eps = espcell[c2];
        
        if(EqualZero(dq_face))
            tmp = 1.0;
        else{
            if(dq_face > 0.0){
                tmp  = VenFun(dmax[c2], dq_face, eps);
            }else{
                tmp  = VenFun(dmin[c2], dq_face, eps);
            }
            tmp     /= dq_face; 
        }

        limit[c2] = MIN(limit[c2], tmp);
    }
#endif
    mfmem::sdel_array_1D(espcell);
    mfmem::sdel_array_1D(dmax);
    mfmem::sdel_array_1D(dmin);
    
}


/*******************************************************************************\
       Function used in VencatLimiter 
\*******************************************************************************/
RealFlow VenFun(RealFlow d, RealFlow dq, RealFlow eps)
{
    return((d*d+eps+(dq+dq)*d)*dq/(d*d + (dq+dq+d)*dq +eps));
}


/******************************************************************************\
|  判断字符串中是否有NAN、inf，如果有，则退出程序
\******************************************************************************/
void IfNAN(char *str)
{
    IntType i,len,mark;
    
    mark = 0;
#ifdef MPICH
    IntType mark_glb;
    if(myZone == 1){
        len = static_cast<IntType>(strlen(str));
        for(i=0;i<len-2;i++){
            if(!(memcmp(&(str[i]),"NAN",3) && memcmp(&(str[i]),"Nan",3) && memcmp(&(str[i]),"nan",3) &&
                 memcmp(&(str[i]),"INF",3) && memcmp(&(str[i]),"Inf",3) && memcmp(&(str[i]),"inf",3))){
                mark = 1;
                break;
            }
        }
        if(mark){
            cout<<endl<<"NaN or inf detected after residual evaluation!"<<endl;
        }
    }
    MPI_Allreduce(&mark, &mark_glb, 1, MPIIntType, MPI_MAX, MPI_COMM_WORLD);
    if(mark_glb){
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
#else
    len = static_cast<IntType> (strlen(str));
    for(i=0;i<len-2;i++){
        if(!(memcmp(&(str[i]),"NAN",3) && memcmp(&(str[i]),"Nan",3) &&
             memcmp(&(str[i]),"inf",3) && memcmp(&(str[i]),"Inf",3))){
            mark = 1;
            break;
        }
    }
    if(mark){
        cout<<endl<<"NAN or inf detected after residual evaluation!"<<endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
#endif
}


/*******************************************************************************\
   由美国标准大气参数表根据高度H获取温度T和压力p
   参考文献：杨炳尉，标准大气参数的公式表示，宇航学报，1983年1月
   将文献中采用原始高度做判断改为采用重力位势高度判断
   zhyb, 20181012
\*******************************************************************************/    
void CalpandTfromAltitude(RealGeom h, RealFlow &p, RealFlow &T)
{
    //零高度温度和压力
    RealFlow T0 = 288.15;
    RealFlow p0 = 101325.0;
    
    RealGeom R = 6356.766;  //地球半径R，单位km
    RealFlow w;
    RealGeom H;
    
    H = h/(1.0+h/R); //重力位势高度H
    
    if(H < 0.0){
        mflog::log.set_one_processor_out();
        mflog::log <<endl<<"Error! The Altitude is < 0.0km. There is no data!"<<endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));      
    }else if(H <= 11.0){
        w = 1.0-H/44.3308;
        T = T0*w;
        p = p0*pow(w,5.2559);
    }else if(H <= 20.0){
        w = exp((14.9647-H)/6.3416);
        T = 216.65;
        p = 0.11953*p0*w;
    }else if(H <= 32.0){
        w = 1.0+(H-24.9021)/221.552;
        T = 221.552*w;
        p = 0.025158*p0*pow(w,-34.1629);
    }else if(H <= 47.0){
        w = 1.0+(H-39.7499)/89.4107;
        T = 250.35*w;
        p = 0.0028338*p0*pow(w,-12.2011);
    }else if(H <= 51.0){
        w = exp((48.6252-H)/7.9223);
        T = 270.65;
        p = 0.00089155*p0*w;
    }else if(H <= 71.0){
        w = 1.0-(H-59.439)/88.2218;
        T = 247.021*w;
        p = 0.00021671*p0*pow(w,12.2011);
    }else if(H <= 84.852){
        w = 1.0-(H-78.0303)/100.295;
        T = 200.59*w;
        p = 0.000012274*p0*pow(w,17.0816);
    }else if(H <= 89.7157){
        w = exp((87.2848-H)/5.47);
        T = 186.87;
        p = (2.273+0.001042*H)*p0*w*1.0e-6;
    }else{
        mflog::log.set_one_processor_out();
        mflog::log<<endl<<"Error! The Altitude is too high. There is no data!"<<endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    } 
}


#undef CPP_FILD_ID  // clear out file id
} //~namespace mflow
