//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   grid_polyhedra.cpp
/// \brief  A class for unstructured polyhedral grid
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
#include "grid_polyhedra.h"

// build-in head files
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <deque>
#include <queue>
#include <list>
#include <set>
#include <map>
#include <iomanip>
#include <algorithm>
using namespace std;

// user defined head files
#include "number_type.h"
#include "zone.h"
#include "solver_ns.h"
#include "utility_functions.h"
#include "algm.h"
#include "io_log.h"
#include "io_base_format.h"
#include "parallel_base_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"

// this header file is copied from cart

#ifdef MPICH
#include "mpi.h"
#endif

#ifdef FS_OPENMP
#include "omp.h"
#endif

#if FS_CUDA_Grid
#include "cuGrid.cuh"
#include "cuDeviceControl.cuh"
using namespace gpuGrid;
#endif

namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 10711  // define file id

#ifdef MPICH
extern int myZone;
extern int numprocs;
extern MPI_Comm GridComm;  //for each grid, tangj
#endif

PolyGrid::PolyGrid(IntType zin,IntType lin):
    Grid(zin), level(lin), nIFace(0), nNeighbor(0), nINode(0), nNeighborN(0), VolAvg(0.0)
{
    //指针
    nNPF=NULL;f2c=NULL;nCPC=NULL;nNPC=NULL;
    nFPC=NULL;nbZ=NULL;nbBF=NULL; nZIFace=NULL;
    nZINode=NULL;nbN=NULL; nbSN=NULL;nbZN=NULL;nbRN=NULL;
    WeightNodeDist=NULL; Nmark=NULL;
    xcc=NULL;ycc=NULL;zcc=NULL;xfn=NULL;yfn=NULL;zfn=NULL;vol=NULL;area=NULL;xfc=NULL;yfc=NULL;
    zfc=NULL;vgn=NULL;BFacevgx=NULL;BFacevgy=NULL;BFacevgz=NULL; 
    vccx=NULL; vccy=NULL; vccz=NULL;
    node_act=NULL;
    edge_act=NULL; vgn=NULL;nChCellS=NULL;nChCellR=NULL;
    nChRNoComp=NULL; 
    Cell2Zone=NULL; 
    facecentroidskewness=NULL;faceangleskewness=NULL;cellcentroidskewness=NULL;
    cellvolsmoothness=NULL;faceangle=NULL;cellwallnumber=NULL;WeightNodeProl=NULL;
    nCPN=NULL;N2C=NULL;f2n=NULL;nb=NULL;
    ghost2global = NULL;
    
    //指针的指针
    c2c=NULL;C2F=NULL;F2N=NULL;C2N=NULL;
    bcr=NULL;bNSNo=NULL;bNRNo=NULL;Seta_Center=NULL;nChSNo=NULL;nChRNo=NULL;
    nChxcc=NULL;nChycc=NULL;nChzcc=NULL;WeightNodeC2N=NULL;WeightNodeN2C = NULL; WeightNodeBFace2C = NULL;
    bCNo=NULL;
    nbg=NULL;bFNo=NULL;
    
}


PolyGrid::PolyGrid(IntType zin):
    Grid(zin), level(0), nIFace(0), nNeighbor(0), nINode(0), nNeighborN(0), VolAvg(0.0)
{
    //指针
    nNPF=NULL;f2c=NULL;nCPC=NULL;nNPC=NULL;
    nFPC=NULL;nbZ=NULL;nbBF=NULL; nZIFace=NULL;
    nZINode=NULL;nbN=NULL; nbSN=NULL;nbZN=NULL;nbRN=NULL;
    WeightNodeDist=NULL; Nmark=NULL;
    xcc=NULL;ycc=NULL;zcc=NULL;xfn=NULL;yfn=NULL;zfn=NULL;vol=NULL;area=NULL;xfc=NULL;yfc=NULL;
    zfc=NULL;vgn=NULL;BFacevgx=NULL;BFacevgy=NULL;BFacevgz=NULL; 
    vccx=NULL; vccy=NULL; vccz=NULL; 
    node_act=NULL;
    edge_act=NULL; vgn=NULL;nChCellS=NULL;nChCellR=NULL;
    nChRNoComp=NULL;
    Cell2Zone=NULL; 
    facecentroidskewness=NULL;faceangleskewness=NULL;cellcentroidskewness=NULL;
    cellvolsmoothness=NULL;faceangle=NULL;cellwallnumber=NULL;WeightNodeProl=NULL;
    nCPN=NULL;f2n=NULL;nb=NULL;
    ghost2global = NULL;

    //指针的指针;
    c2c=NULL;C2F=NULL;F2N=NULL;C2N=NULL;
    bcr=NULL;bNSNo=NULL;bNRNo=NULL;Seta_Center=NULL;nChSNo=NULL;nChRNo=NULL;
    nChxcc=NULL;nChycc=NULL;nChzcc=NULL;WeightNodeC2N=NULL;N2C=NULL;WeightNodeN2C = NULL; WeightNodeBFace2C = NULL;
    bCNo=NULL;
    nbg=NULL;bFNo=NULL;

}


PolyGrid::PolyGrid():
    Grid(), level(0), nIFace(0), nNeighbor(0), nINode(0), nNeighborN(0), VolAvg(0.0)
{
    //指针
    nNPF=NULL;f2c=NULL;nCPC=NULL;nNPC=NULL;
    nFPC=NULL;nbZ=NULL;nbBF=NULL; nZIFace=NULL;
    nZINode=NULL;nbN=NULL; nbSN=NULL;nbZN=NULL;nbRN=NULL;
    WeightNodeDist=NULL; Nmark=NULL;
    xcc=NULL;ycc=NULL;zcc=NULL;xfn=NULL;yfn=NULL;zfn=NULL;vol=NULL;area=NULL;xfc=NULL;yfc=NULL;
    zfc=NULL;vgn=NULL;BFacevgx=NULL;BFacevgy=NULL;BFacevgz=NULL; 
    vccx=NULL; vccy=NULL; vccz=NULL; 
    node_act=NULL;
    edge_act=NULL; vgn=NULL;nChCellS=NULL;nChCellR=NULL;
    nChRNoComp=NULL;
    Cell2Zone=NULL; 
    facecentroidskewness=NULL;faceangleskewness=NULL;cellcentroidskewness=NULL;
    cellvolsmoothness=NULL;faceangle=NULL;cellwallnumber=NULL;WeightNodeProl=NULL;
    nCPN=NULL;f2n=NULL;nb=NULL;
    //指针的指针;
    c2c=NULL;C2F=NULL;F2N=NULL;C2N=NULL;
    bcr=NULL;bNSNo=NULL;bNRNo=NULL;Seta_Center=NULL;nChSNo=NULL;nChRNo=NULL;
    nChxcc=NULL;nChycc=NULL;nChzcc=NULL;WeightNodeC2N=NULL;N2C=NULL;WeightNodeN2C = NULL; WeightNodeBFace2C = NULL;
    bCNo=NULL;
    nbg=NULL;bFNo=NULL;
    ghost2global = NULL;
   
}


PolyGrid::~PolyGrid()
{
    //未删除指针：nNPC/c2cc
    mfmem::sdel_array_1D(nNPF);
    mfmem::sdel_array_1D(nNPC);
    mfmem::sdel_array_1D(nFPC);
    mfmem::sdel_array_1D(nbZ);
    mfmem::sdel_array_1D(nbBF);
    mfmem::sdel_array_1D(nZINode);
    mfmem::sdel_array_1D(nb);
    mfmem::sdel_array_1D(nbN);
    mfmem::sdel_array_1D(nbSN);//nbSN可以在initzone后删除,在transgrid OutParGrid后可删去
    mfmem::sdel_array_1D(nbZN);//nbSN可以在initzone后删除,在transgrid OutParGrid后可删去
    mfmem::sdel_array_1D(nbRN);//nbSN可以在initzone后删除,在transgrid OutParGrid后可删去
    mfmem::sdel_array_1D(WeightNodeDist);//程序已删除
    mfmem::sdel_array_1D(Nmark);
    mfmem::sdel_array_1D(nCPN);
    if(vgn){
        mfmem::sdel_array_1D(vgn);
        mfmem::sdel_array_1D(BFacevgx);
        mfmem::sdel_array_1D(BFacevgy);
        mfmem::sdel_array_1D(BFacevgz);
        mfmem::sdel_array_1D(vccx);
        mfmem::sdel_array_1D(vccy);
        mfmem::sdel_array_1D(vccz);
    }
    mfmem::sdel_array_1D(node_act);
    mfmem::sdel_array_1D(edge_act);//程序已删除
    mfmem::sdel_array_1D(nChCellS);
    mfmem::sdel_array_1D(nChCellR);
    mfmem::sdel_array_1D(nChRNoComp);
    mfmem::sdel_array_1D(WeightNodeProl);
    //未删除的指针的指针
    mfmem::sdel_array_2D(C2F);

    //未调用的指针的指针：bcr（不懂）
    mfmem::sdel_array_1D(f2n);
    mfmem::sdel_array_1D(f2c);
    mfmem::sdel_array_1D(nCPC);
    mfmem::sdel_array_2D(c2c);
    mfmem::sdel_array_1D(F2N); //F2N is the reference to f2n, so use 1D array operator, tangj 
    mfmem::sdel_array_2D(C2N);
    mfmem::sdel_array_1D(bcr);
    mfmem::sdel_array_2D(bNSNo,nNeighborN,false);
    mfmem::sdel_array_2D(bNRNo,nNeighborN,false);
    mfmem::sdel_array_2D(Seta_Center);
    mfmem::sdel_array_2D(nChSNo);
    mfmem::sdel_array_2D(nChRNo);
    mfmem::sdel_array_2D(nChxcc);
    mfmem::sdel_array_2D(nChycc);
    mfmem::sdel_array_2D(nChzcc);
    mfmem::sdel_array_2D(WeightNodeC2N);
    mfmem::sdel_array_2D(WeightNodeN2C);
    mfmem::sdel_array_2D(WeightNodeBFace2C);
    IntType nTNode = this->GetNTNode();
    mfmem::sdel_array_2D(N2C,nTNode,false);
    //
//  if(nbZ)  delete []nbZ; //因为他们指向同一个数组,在grids[0]就已经删除
//  if(nbBF) delete []nbBF;
    mfmem::sdel_array_1D(nZIFace);
  // delete []bcr;
    mfmem::sdel_array_1D(xcc);
    mfmem::sdel_array_1D(ycc);
    mfmem::sdel_array_1D(zcc);
    mfmem::sdel_array_1D(xfn);
    mfmem::sdel_array_1D(yfn);
    mfmem::sdel_array_1D(zfn);
    mfmem::sdel_array_1D(vol);
    mfmem::sdel_array_1D(area);
    mfmem::sdel_array_1D(xfc);
    mfmem::sdel_array_1D(yfc);
    mfmem::sdel_array_1D(zfc);

    //add by dingxin
#ifdef REORDER
    mfmem::sdel_array_1D(order_cell_oTon);
    mfmem::sdel_array_1D(order_cell_nToo);
#endif // REORDER
#ifdef DIVREP
    if (DivRepSuccess) {
        mfmem::sdel_array_1D(idx_pthreads_bface);
        mfmem::sdel_array_1D(id_division_bface);
        mfmem::sdel_array_1D(idx_pthreads_iface);
        mfmem::sdel_array_1D(id_division_iface);
#ifdef BoundedColoring
        mfmem::sdel_array_1D(endIndex_bFace_vec);
        mfmem::sdel_array_1D(endIndex_iFace_vec);
#endif // BoundedColoring
    }
#endif // DIVREP
#ifdef DIVCON
    tree_free(this->treeHead);
    mfmem::sdel_array_1D(this->treeHead);
#endif // DIVCON

    if(facecentroidskewness){
        mfmem::sdel_array_1D(facecentroidskewness);
        mfmem::sdel_array_1D(faceangleskewness);//在EquiangleSkewnessSummary后不再调用
        mfmem::sdel_array_1D(cellcentroidskewness);
        mfmem::sdel_array_1D(cellvolsmoothness);
        mfmem::sdel_array_1D(faceangle);//在FaceAngleSummary后不再调用
        mfmem::sdel_array_1D(cellwallnumber);
    }
    mfmem::sdel_array_1D(Cell2Zone);
    mfmem::sdel_array_1D(cellwallnumber);
    if(nNeighbor > 0) {
        mfmem::sdel_array_1D(nbg);
    }
#ifdef MPICH
    mfmem::sdel_array_2D(bCNo,nNeighbor,false);
    mfmem::sdel_array_2D(bFNo,nNeighbor,false);    
#endif
    mfmem::sdel_array_1D(ghost2global);
}


void PolyGrid::ComputeMetrics()
{
    IntType n = nTCell + nBFace;   // interior+ghost cells
    
    if(vol==0) mfmem::snew_array_1D(vol, n, dmrfl); 
          
    if(xcc==0) mfmem::snew_array_1D(xcc, n, dmrfl); 
    if(ycc==0) mfmem::snew_array_1D(ycc, n, dmrfl); 
    if(zcc==0) mfmem::snew_array_1D(zcc, n, dmrfl); 

    if(area==0)mfmem::snew_array_1D(area, nTFace, dmrfl); 
  
    if(xfn==0) mfmem::snew_array_1D(xfn, nTFace, dmrfl); 
    if(yfn==0) mfmem::snew_array_1D(yfn, nTFace, dmrfl); 
    if(zfn==0) mfmem::snew_array_1D(zfn, nTFace, dmrfl); 
        
    if(xfc==0) mfmem::snew_array_1D(xfc, nTFace, dmrfl); 
    if(yfc==0) mfmem::snew_array_1D(yfc, nTFace, dmrfl); 
    if(zfc==0) mfmem::snew_array_1D(zfc, nTFace, dmrfl); 


    FaceCellCenterbyAverage(this,xfc,yfc,zfc,xcc,ycc,zcc);
    FaceAreaNormalCentroid_cycle(this,area,xfn,yfn,zfn,xfc,yfc,zfc);
    CellVolCentroid(this,vol,xcc,ycc,zcc);
    CorrectFaceNormal(this,xfn,yfn,zfn);
    CorrectCellCentroid(this, xcc, ycc, zcc, xfc, yfc, zfc, xfn, yfn, zfn);

#ifdef MPICH
    //体心，体积的并行传值
    CommInterfaceDataMPI(xcc);
    CommInterfaceDataMPI(ycc);
    CommInterfaceDataMPI(zcc);
    CommInterfaceDataMPI(vol);
#endif        
   
    ClosureCheck(this,xfn,area);
    ClosureCheck(this,yfn,area);
    ClosureCheck(this,zfn,area);
    
#ifdef DEBUG
    mflog::log.set_one_processor_out();
    mflog::log << "Exit ComputeMetrics" << std::endl;
#endif  
}


/************************************************************************\
  无网格重排序，此函数是为了编程统一
\************************************************************************/
void PolyGrid::ReorderCellforLUSGS_0()
{
    IntType i;
    IntType n = nBFace+nTCell;
    
    IntType *Layer = NULL;
    IntType *LUSGSCellOrder = NULL;
    mfmem::snew_array_1D(Layer, n,dmrfl);
    mfmem::snew_array_1D(LUSGSCellOrder, nTCell,dmrfl);
    for(i=0;i<n;i++){
        Layer[i] = i;
    }
    for(i=0;i<nTCell;i++){
        LUSGSCellOrder[i] = i;
    }
    
    this->UpdateDataPtr(LUSGSCellOrder, INT, nTCell, "LUSGSCellOrder");
    this->UpdateDataPtr(Layer, INT, n, "LUSGSLayer");
}


/************************************************************************
*  Communicate data from the current grid to neighbor grid in zone      *
************************************************************************/
void PolyGrid::CommInterfaceData(IntType nbZone, PolyGrid *grid, const char *name)
{
    IntType i,n,c1,c2;
    RealFlow *cq,*nq;
    BCRecord **bcr=Getbcr();
    IntType *f2c_n=grid->Getf2c();
    IntType nBFace = GetNBFace();
    IntType nIFace=0;

    n = GetNTCell() + nBFace;
    cq = (RealFlow *)GetDataPtr(REAL_FLOW,n,name);
    if(!cq) {
        printf("Variable %s to be communicated not found\n", name);
        return;
    }

    n = grid->GetNTCell() + grid->GetNBFace();
    nq = (RealFlow *)grid->GetDataPtr(REAL_FLOW,n,name);
    if(!nq) {
        printf("Variable %s to be communicated not found\n", name);
        return;
    }

    n = 0;
    IntType ncnb = grid->GetNTCell();
    for(i=0; i<nBFace; i++) {
        if(bcr[i]->GetType() == INTERFACE) {
            if(nbZ[nIFace] == nbZone) {
                c1 = f2c[i*2];
                c2 = nbBF[nIFace] + ncnb;
                nq[c2] = cq[c1];
                
                c2 = f2c[i*2+1];
                c1 = f2c_n[nbBF[nIFace]*2];
                cq[c2] = nq[c1];
                
                n++;
            }
            nIFace++;
        }
    }
    assert(nIFace == GetNIFace());
}

/************************************************************************
*  Communicate data from the current grid to neighbor grid in zone      *
************************************************************************/
void PolyGrid::CommCellCenterData(IntType nbZone, PolyGrid *grid)
{
    IntType i,n,c1,c2;
    BCRecord **bcr=Getbcr();
    IntType *f2c_n=grid->Getf2c();
    IntType nBFace = GetNBFace();
    IntType nIFace=0;

    RealGeom *nxcc = grid->GetXcc();
    RealGeom *nycc = grid->GetYcc();
    RealGeom *nzcc = grid->GetZcc();
    
    n = 0;
    IntType ncnb = grid->GetNTCell();
    
    for(i=0; i<nBFace; i++) {
        if(bcr[i]->GetType() == INTERFACE) {
            if(nbZ[nIFace] == nbZone) {
                c1 = f2c[i*2];
                c2 = nbBF[nIFace] + ncnb;
                nxcc[c2] = xcc[c1];
                nycc[c2] = ycc[c1];
                nzcc[c2] = zcc[c1];

                c2 = f2c[i*2+1];
                c1 = f2c_n[nbBF[nIFace]*2];
                xcc[c2] = nxcc[c1];
                ycc[c2] = nycc[c1];
                zcc[c2] = nzcc[c1];

                n++;
            }
            nIFace++;
        }
    } 
    assert(nIFace == GetNIFace());
}


/*******************************************************************************
     Purpose: To compute the least distance from point to box 
*******************************************************************************/
RealGeom FindRminbox(RealGeom xp, RealGeom yp, RealGeom zp, RealGeom x1, RealGeom x2,
                     RealGeom y1, RealGeom y2, RealGeom z1, RealGeom z2 )
{
    RealGeom rr, rx, ry, rz;
    if( xp>=x1 && xp<=x2) rx = 0;
    else rx = ( xp<x1 ) ? x1-xp : xp-x2;
    if( yp>=y1 && yp<=y2) ry = 0;
    else ry = ( yp<y1 ) ? y1-yp : yp-y2;
    if( zp>=z1 && zp<=z2) rz = 0;
    else rz = ( zp<z1 ) ? z1-zp : zp-z2;
    
    rr = rx*rx + ry*ry + rz*rz;
 
    return(rr);
}


/*******************************************************************************
     Purpose: To compute the distance from point to point 
*******************************************************************************/
RealGeom FindRp2p(RealGeom x1, RealGeom y1, RealGeom z1, RealGeom x2, RealGeom y2, RealGeom z2 )
{
    RealGeom dx, dy, dz;
    dx = x1-x2;
    dy = y1-y2;
    dz = z1-z2;
    return( dx*dx+dy*dy+dz*dz );
}


/***********************************************************************
     Purpose:  To find the closest distance from a field point to the
     actual surface (i.e. not simply the closest discrete surface
     point), using local triangulation of the surface.
***********************************************************************/
void FindRp2tri(RealGeom &dist, RealGeom xp, RealGeom yp, RealGeom zp, 
                RealGeom xa, RealGeom ya, RealGeom za, RealGeom xb, RealGeom yb, RealGeom zb,
                RealGeom xc, RealGeom yc, RealGeom zc )
{
    RealGeom pp[3], aa[3], bb[3], cc[3];
    pp[0]=xp; pp[1]=yp; pp[2]=zp;
    aa[0]=xa; aa[1]=ya; aa[2]=za;
    bb[0]=xb; bb[1]=yb; bb[2]=zb;
    cc[0]=xc; cc[1]=yc; cc[2]=zc;
  
    RealGeom p[3],a[3],b[3],r[3], rr;
    RealGeom daa=0., dbb=0., dab=0., den, dap=0., dbp=0., s, t, dsq;
    for(IntType i=0; i<3; i++)
    {
        p[i] = pp[i] - aa[i];
        a[i] = bb[i] - aa[i];
        b[i] = cc[i] - aa[i];
        daa += a[i]*a[i];
        dbb += b[i]*b[i];
        dab += a[i]*b[i];
    }
    den = dab*dab - daa*dbb;

    if(EqualZero(den)) ;   //zhyb: 面积不为零则den不为零
    else
    {
        for(IntType i=0; i<3; i++)
        {
            dap += a[i]*p[i];
            dbp += b[i]*p[i];
        }
        s = (dab*dbp-dbb*dap)/den;
        t = (dab*dap-daa*dbp)/den;
        if( s<0. || t<0. || (t+s)>1. ) ;   //zhyb: 这三种情况垂足落在三角形外边
        else
        {
            for(IntType i=0; i<3; i++) r[i] = p[i]-s*a[i]-t*b[i];
            rr = 0.;
            for(IntType i=0; i<3; i++) rr += r[i]*r[i];
            if ( rr<dist ) dist = rr;
            return;
        }
    }

    dsq = dist;
    if(EqualZero(daa)) ;  //zhyb: bb点和aa点重合
    else
    {
        dap = 0.;
        for(IntType i=0; i<3; i++) dap += a[i]*p[i];
        t = dap/daa;
        if( t<0. || t>1.) ;     //zhyb: 这两种情况下垂足落在线段aabb外
        else
        {
            for(IntType i=0; i<3; i++) r[i] = p[i]-t*a[i];
            rr = 0.;
            for(IntType i=0; i<3; i++) rr += r[i]*r[i];
            if ( rr<dsq ) dsq = rr;
        }
    }

    if(EqualZero(dbb)) ;   //zhyb: cc点和aa点重合
    else
    {
        dbp = 0.;
        for(IntType i=0; i<3; i++) dbp += b[i]*p[i];
        t = dbp/dbb;
        if( t<0. || t>1. ) ;    //zhyb: 这两种情况下垂足落在线段aacc外
        else
        {
            for(IntType i=0; i<3; i++) r[i] = p[i]-t*b[i];
            rr = 0.;
            for(IntType i=0; i<3; i++) rr += r[i]*r[i];
            if ( rr<dsq ) dsq = rr;
        }
    }
  
    daa = 0.;
    for(IntType i=0; i<3; i++)
    {
        p[i] = pp[i]-bb[i];
        a[i] = cc[i]-bb[i];
        daa += a[i]*a[i];
    }
   
    if(EqualZero(daa)) ;   //zhyb: cc点和bb点重合
    else
    {
        dap = 0;
        for(IntType i=0; i<3; i++) dap += a[i]*p[i];
        t = dap/daa;
        if( t<0. || t>1. ) ;    //zhyb: 这两种情况下垂足落在线段bbcc外
        else
        {
            for(IntType i=0; i<3; i++) r[i] = p[i]-t*a[i];
            rr = 0.;
            for(IntType i=0; i<3; i++) rr += r[i]*r[i];
            if ( rr<dsq ) dsq = rr;
        }
    }

    if( dsq<dist ) dist = dsq;
}


/************************************************************************
    Purpose:  To interchange v[i] and v[j]
************************************************************************/
void xswap(IntType *indx, RealGeom *v, IntType i, IntType j)
{
    RealGeom temp;
    temp = v[i];
    v[i] = v[j];
    v[j] = temp;
    IntType itmp;
    itmp = indx[i];
    indx[i] = indx[j];
    indx[j] = itmp;
}


/************************************************************************
    Purpose:  To sort a list of points
************************************************************************/
void quicksort(IntType s, IntType e, IntType *indx, RealGeom *v)
{
    IntType i, last;
  
    if( e-s<=0 ) // nothing to do
        return;
      
    last = s-1;
    for( i=s; i<e; i++)
    {
        if( v[i]<v[s-1] )
            xswap( indx, v, ++last, i);
    }
  
    xswap( indx, v, s-1, last);
    quicksort( s, last, indx, v );
    quicksort( last+2, e, indx, v );
}


/************************************************************************
      Compute the distance to wall(triangles) about the cell center
      Author: zm 20080810
      Modify: zm 20150423
              zhyb 20180914 考虑顶点全在物面上，而所有面均不是物面的单元
************************************************************************/
void PolyGrid::ComputeDist2WallTriang(RealGeom *dist2wall_cell, IntType mark)
{
    IntType i,j,k,count;
    FILE *fp;
    IntType nTNode=GetNTNode();
    RealGeom *x=GetX(),*y=GetY(),*z=GetZ();
    String filename, filename_tmp;
  
    mflog::log.set_one_processor_out();
    mflog::log << "Now compute the distance to wall!" << endl;
  
    GetData(filename, STRING, 1, "griddir");
    sprintf(filename_tmp,"GeominfoDist.dat");
    strcat(filename, filename_tmp);

    mflog::log.set_each_grid_out();
    fp=fopen(filename,"rb");
    if(!fp) {
        mflog::log<<"Failed to open file "<<filename<<" in function ComputeDist2WallTriang!"<<endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    IntType nSP, nSF, *nSfP, *SfP, *nPntS, *PntS;
    

    fread(&nSP,  sizeof(IntType),  1, fp);

    RealGeom *xSf = NULL;// the coordinate
    RealGeom *ySf = NULL;
    RealGeom *zSf = NULL;
    mfmem::snew_array_1D(xSf, nSP,dmrfl);
    mfmem::snew_array_1D(ySf, nSP,dmrfl);
    mfmem::snew_array_1D(zSf, nSP,dmrfl);
    fread(xSf, sizeof(RealGeom), nSP, fp);
    fread(ySf, sizeof(RealGeom), nSP, fp);
    fread(zSf, sizeof(RealGeom), nSP, fp);

    fread(&nSF,  sizeof(IntType),  1, fp);     // Number of the Solid-Face 壁面上Face数量
	
	/* cout << "nSF: " << nSF << endl;
	exit(0); */

    nSfP = NULL;
    mfmem::snew_array_1D(nSfP, nSP+1,dmrfl);
    fread(nSfP,  sizeof(IntType),  nSP+1, fp);     // Number of the Solid-Face that connected Local Point
	/* cout << "nSfP[0]: " << nSfP[0] << endl;
	cout << "nSfP[1]: " << nSfP[1] << endl;
	cout << "nSfP[2]: " << nSfP[2] << endl;
	cout << "nSfP[nSP]: " << nSfP[nSP] << endl;
	exit(0); */
    SfP = NULL;
    mfmem::snew_array_1D(SfP, nSfP[nSP],dmrfl);
    fread(SfP,  sizeof(IntType),  nSfP[nSP], fp);  // these Solid-Faces that connected Local Point

    nPntS = NULL;
    mfmem::snew_array_1D(nPntS, nSF+1,dmrfl);
    fread(nPntS,  sizeof(IntType),  nSF+1, fp);
    PntS = NULL;
    mfmem::snew_array_1D(PntS, nPntS[nSF],dmrfl);
    fread(PntS,  sizeof(IntType),  nPntS[nSF], fp);
	
#if defined FS_CUDA_Grid	
	IntType mpirank = 0; //when mpirank = 0, mpi was off. 	
#ifdef MPICH        
    MPI_Comm_rank(MPI_COMM_WORLD, & mpirank);	
#endif
	GetGPUNum(GPUNum);
	MultiGPUDevice(GPUNum, mpirank, GPUProp);
	// Transfer Grid Data to GPU:
	GPUGridDataTrans(nTNode, nSP, nSF, nSfP, SfP, 
					nPntS, PntS, x, y, z, xSf, ySf, zSf);
#endif      
    

	/* IntType nBox, *nPBox, *SP;
    RealGeom **BBox;
	
	SP = NULL;
    mfmem::snew_array_1D(SP, nSP,dmrfl);
    fread(SP,  sizeof(IntType),  nSP, fp);// Number of the boxs
    
	fread(&nBox,  sizeof(IntType),  1, fp);
    nPBox = NULL;// Number of the Points in the box
    mfmem::snew_array_1D(nPBox, nBox+1,dmrfl);
    fread(nPBox,  sizeof(IntType),  nBox+1, fp);

    BBox = NULL;//Six Bound-Core of the Box
    mfmem::snew_array_2D(BBox, nBox, 6, dmrfl,true);
    for( i=0; i<nBox; i++)
        fread(BBox[i],  sizeof(RealGeom),  6, fp); */

    fclose(fp);

    mflog::log.set_one_processor_out();
    mflog::log << "Finished to Read the coordinate of all nodes on the solid surface" << endl;
  
    /* RealGeom *distP = NULL;
    mfmem::snew_array_1D(distP, nTNode,dmrfl);
    for( i=0; i<nTNode; i++) distP[i] = BIG;

    for( i=0; i<nTNode; i++){
        IntType pntmin = -1;

        RealGeom *distB = NULL;
        mfmem::snew_array_1D(distB, nBox,dmrfl);
        for( j=0; j<nBox; j++){
            distB[j] = FindRminbox( x[i], y[i], z[i], BBox[j][0], BBox[j][1], BBox[j][2],
                                    BBox[j][3], BBox[j][4], BBox[j][5] );
        }

        IntType *Bsort = NULL;
        mfmem::snew_array_1D(Bsort, nBox,dmrfl);
        for( j=0; j<nBox; j++) Bsort[j]=j;
        quicksort( 1, nBox, Bsort, distB);

        RealFlow dd, distP2P=BIG;
        IntType B, pnt2;
        for( j=0; j<nBox; j++){
            if( distP[i]<distB[j] ) break;
            
            B=Bsort[j];
            for(k=nPBox[B]; k<nPBox[B+1]; k++ ){
                pnt2=SP[k];
                dd = FindRp2p( x[i], y[i], z[i], xSf[pnt2], ySf[pnt2], zSf[pnt2]);
                if( distP2P>dd ){
                    distP2P = dd;
                    pntmin = pnt2;
                }
            }
        }
      
        for(k=nSfP[pntmin]; k<nSfP[pntmin+1]; k++){
            IntType sface = SfP[k];
            IntType pnt[4];
            for(IntType jj=nPntS[sface]; jj<nPntS[sface+1]; jj++)
                pnt[jj-nPntS[sface]] = PntS[jj];
            if( nPntS[sface+1]-nPntS[sface]==4 ) {
                if( pntmin==pnt[0] ) pnt[2]=pnt[3];
                else if( pntmin==pnt[2] ) pnt[0]=pnt[3];
                else if( pntmin==pnt[3] ) pnt[1]=pnt[3];
            }
            FindRp2tri( distP2P, x[i], y[i], z[i], xSf[pnt[0]], ySf[pnt[0]], zSf[pnt[0]],
                        xSf[pnt[1]], ySf[pnt[1]], zSf[pnt[1]], xSf[pnt[2]], ySf[pnt[2]], zSf[pnt[2]] );
        }
        if(distP2P<distP[i]) distP[i] = distP2P;
        distP[i] = sqrt(distP[i]);
        mfmem::sdel_array_1D(distB);
        mfmem::sdel_array_1D(Bsort);
    } */
	RealGeom *distP = NULL;
    mfmem::snew_array_1D(distP, nTNode, dmrfl);
    //for( i=0; i<nTNode; i++) distP[i] = BIG;

    IntType *indices = NULL;
    mfmem::snew_array_1D(indices, nTNode,dmrfl);    

    MinDist min_dist;
    min_dist.SetPoints(nSP, xSf, ySf, zSf);
    min_dist.Init();
	
#if defined FS_CUDA_Grid
	cuSearchIndex(distP, indices);
#else
    min_dist.SearchIndex(nTNode, x, y, z, distP, indices);
#endif

    // here we don't known how many points the face has, so we
    // set 20.
    // We use face_pnts to creat a ring to make j-1 and j+1 always valid.
#if defined FS_CUDA_Grid	
	cuComputeDist2Wall(distP, indices);
#else
	for(i=0; i<nTNode; ++i)
    {
        IntType face_pnts[20];
		IntType pntmin = indices[i];
        RealGeom distP2P = BIG;
        for(k = nSfP[pntmin]; k < nSfP[pntmin+1]; k++)
        {
            IntType sface = SfP[k];
            IntType tri_pnt[3];

            // real points start from 1
            //face_pnts.resize(1);
			IntType count = 1;
            for(IntType jj = nPntS[sface]; jj < nPntS[sface+1]; jj++)
            {
                face_pnts[count] = PntS[jj];
				count++;
            }
            face_pnts[0] = face_pnts[count - 1];
            face_pnts[count] = face_pnts[1];

            if((nPntS[sface+1] - nPntS[sface]) == 3)
            {
                tri_pnt[0] = face_pnts[1];
                tri_pnt[1] = face_pnts[2];
                tri_pnt[2] = face_pnts[3];
            }
            else if((nPntS[sface+1]-nPntS[sface]) == 4)
            {
                tri_pnt[0] = face_pnts[1];
                tri_pnt[1] = face_pnts[2];
                tri_pnt[2] = face_pnts[3];
                if     ( pntmin==face_pnts[1] ) tri_pnt[2] = face_pnts[4];
                else if( pntmin==face_pnts[3] ) tri_pnt[0] = face_pnts[4];
                else if( pntmin==face_pnts[4] ) tri_pnt[1] = face_pnts[4];
            }
            else if((nPntS[sface+1]-nPntS[sface]) > 4)
            {
                // find the anchor point in the face nodes
                for(j = 1;j <= nPntS[sface+1]-nPntS[sface]; ++j)
                {
					if(pntmin == face_pnts[j]) 
                    {
                        count = j;
						j = nPntS[sface+1]-nPntS[sface];
						//break;
                    }
                }
				j = count;
                tri_pnt[0] = face_pnts[j-1];
                tri_pnt[1] = face_pnts[j];
                tri_pnt[2] = face_pnts[j+1];
            }
            FindRp2tri( distP2P, x[i], y[i], z[i], xSf[tri_pnt[0]], ySf[tri_pnt[0]], zSf[tri_pnt[0]],
                xSf[tri_pnt[1]], ySf[tri_pnt[1]], zSf[tri_pnt[1]], xSf[tri_pnt[2]], ySf[tri_pnt[2]], zSf[tri_pnt[2]] );
        }
        if(distP2P < distP[i]*distP[i]) distP[i] = sqrt(distP2P);
    }
#endif

	/* ofstream output;
	output.open("distP-revised.dat");
	for(IntType i = 0; i < nTNode; i++){
		output << i << ": " << distP[i] << endl;
	}
	output.close();
	exit(0); */
	
    mfmem::sdel_array_1D(indices);
    /* mfmem::sdel_array_2D(BBox);
    mfmem::sdel_array_1D(nPBox);
	mfmem::sdel_array_1D(SP); */

    RealGeom *distC = NULL;
    mfmem::snew_array_1D(distC, nTCell,dmrfl);
    IntType *nNPC = CalnNPC(this);
    IntType **C2N = CalC2N(this);
  
    for( i=0; i<nTCell; i++){
        distC[i] = 0.;
        for( j=0; j<nNPC[i]; j++) {
            distC[i] += distP[ C2N[i][j] ];
        }
        distC[i] /= nNPC[i];
    }
  
    IntType type,c1,c2;
    RealGeom d;
    for(i=0;i<nBFace;i++){
        type = bcr[i]->GetType();
        if(type == WALL){
            c1 = f2c[i+i];
            distC[c1] = BIG;
        }
    }
    for(i=0;i<nBFace;i++){
        type = bcr[i]->GetType();
        if(type == WALL){
            c1 = f2c[i+i];
            c2 = f2c[i+i+1];
            
            d  = FindRp2p( xcc[c2], ycc[c2], zcc[c2], xcc[c1], ycc[c1], zcc[c1] );
            d  = 0.5*sqrt(d);
            distC[c1] = MIN(distC[c1],d);  
        }
    } 
    
    //考虑顶点全在物面上，而所有面均不是物面的单元
    IntType *mark_dist0 = NULL;
    mfmem::snew_array_1D(mark_dist0, nTCell,dmrfl);
    count = 0;
    for(i=0; i<nTCell; i++){
        if(distC[i] < TINY){
            mark_dist0[i] = 1;
            count++;
        }else{
            mark_dist0[i] = 0;
        }
    }

#ifdef MPICH
    Parallel::parallel_sum(count, GridComm);
#endif    
    
    if(count){ //可能存在 
        mflog::log.set_one_processor_out();
        mflog::log << endl << "There is " << count << " cell's distC=0! Now correcting!" << endl;

        IntType p1, f1;
        IntType *nNPF = GetnNPF();
        IntType **F2N = CalF2N(this);
        IntType *nFPC = CalnFPC(this);
        IntType **C2F = CalC2F(this);
        
        IntType *mark_wallnode = NULL;  //物面点
        mfmem::snew_array_1D(mark_wallnode, nTNode,dmrfl);
        for(i=0; i<nTNode; i++){
            mark_wallnode[i] = 0;
        }
        for(i=0; i<nBFace; i++){
            type = bcr[i]->GetType();
            if(type!=WALL) continue;
                
            for(j=0; j<nNPF[i]; j++){
                p1 = F2N[i][j];
                mark_wallnode[p1] = 1;
            }
        }
#ifdef MPICH
        CommInternodeDataMPISUM(mark_wallnode);
#endif
        
        for(i=0; i<nTCell; i++){
            if(!mark_dist0[i]) continue; //不是距离为0单元，跳出
            
            mark = 0;
            for(j=0; j<nNPC[i]; j++){
                p1 = C2N[i][j];
                if(!mark_wallnode[p1]){ //这个点不是物面点
                    mark = 1;
                }
            }
            if(mark) continue;  //不是所有点均在物面上，跳出
            
            mark = 0;
            for(j=0; j<nFPC[i]; j++){
                f1 = C2F[i][j];
                if(f1 < nBFace){
                    type = bcr[f1]->GetType();
                    if(type==WALL){
                        mark = 1;
                    }
                }
            }
            if(mark) continue;  //有物面边界，跳出
            
            //求体心到单元顶点的最小距离，作为其到物面的距离
            distC[i] = BIG;
            for(j=0; j<nNPC[i]; j++){
                p1 = C2N[i][j];
          
                d  = FindRp2p( xcc[i], ycc[i], zcc[i], x[p1], y[p1], z[p1]);
                distC[i] = MIN(distC[i], d);
            }
            distC[i] = sqrt(distC[i]);
        }
        mfmem::sdel_array_1D(mark_wallnode);
    }
    mfmem::sdel_array_1D(mark_dist0);

    mflog::log.set_all_processors_out();

    count=0;
    for(i=0;i<nTCell;i++){
        if(distC[i]<TINY){
#ifdef MPICH
            mflog::log<<endl<<"Zone "<<myZone<<"  Cell "<<i<<" dist2wall<TINY"<<endl;
#else
            mflog::log<<endl<<"Cell "<<i<<" dist2wall<TINY"<<endl;
#endif
            count++;
        }
    }
#ifdef MPICH
    Parallel::parallel_sum(count, GridComm);
#endif
    if(count != 0) mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
  
    for(i=0; i<nTCell; i++) dist2wall_cell[i] = MIN(dist2wall_cell[i], distC[i]);
//  DumpPlyhedra3D_FieldView(this,0);

    mfmem::sdel_array_1D(distP);
    mfmem::sdel_array_1D(distC);
    mfmem::sdel_array_1D(xSf);
    mfmem::sdel_array_1D(ySf);
    mfmem::sdel_array_1D(zSf);
    mfmem::sdel_array_1D(nSfP);
    mfmem::sdel_array_1D(SfP);
    mfmem::sdel_array_1D(nPntS);
    mfmem::sdel_array_1D(PntS);
    

    mflog::log.set_one_processor_out();
    mflog::log << "Distance is OK !!" << endl;
}

 
/************************************************************************
    Write the information for Computing the distance to wall 
    about the cell center in the MPI.
************************************************************************/
void PolyGrid::WriteInfoDist()
{
    IntType i, j, k, count, type;
    IntType nTNode=GetNTNode();
    RealGeom *x=GetX(),*y=GetY(),*z=GetZ();
    FILE *fp;
    String filename;
  
    IntType nSF = 0; // 物面数

    IntType *nFN = NULL;;
    mfmem::snew_array_1D(nFN, nBFace+1,dmrfl);

    nFN[0] = 0;
    for(i=1; i<=nBFace; i++) nFN[i]=nFN[i-1]+nNPF[i-1];

    IntType *mrk = NULL;
    mfmem::snew_array_1D(mrk, nTNode,dmrfl);
    for(i=0; i<nTNode; i++) mrk[i]=0;
    IntType nSP=0;  // nSP 物面上点的数量
    for(i=0; i<nBFace; i++){
        type = bcr[i]->GetType();
        if(type==WALL){
            nSF++;
            for(k=nFN[i]; k<nFN[i+1]; k++){
                if( mrk[f2n[k]]==0 ) nSP++;
                mrk[f2n[k]]++;
            }
        }
    }

    IntType *PtNew = NULL;   
    RealGeom *xSf  = NULL;//物面上点的坐标
    RealGeom *ySf  = NULL;
    RealGeom *zSf  = NULL;
    mfmem::snew_array_1D(PtNew, nTNode,dmrfl);
    mfmem::snew_array_1D(xSf, nSP,dmrfl);
    mfmem::snew_array_1D(ySf, nSP,dmrfl);
    mfmem::snew_array_1D(zSf, nSP,dmrfl);
    count = 0;
    for(i=0; i<nTNode; i++){
        if(mrk[i]>0){
            PtNew[i] = count;
            xSf[count] = x[i];
            ySf[count] = y[i];
            zSf[count] = z[i];
            mrk[count] = mrk[i];
            count++;
        }else{
          PtNew[i] = -1;
        }
    }

    mflog::log.set_all_processors_out();

    IntType *nPntS = NULL;   //zhyb: 每个物面面单元对应的点数，第i个物面面单元对应的点数为nPntS[i+1]-nPntS[i]
    IntType *nSfP  = NULL;    //zhyb: 每个物面点对应的面数，第i个物面点对应的面数为nSfP[i+1]-nSfP[i]
    mfmem::snew_array_1D(nPntS, nSF+1,dmrfl);
    mfmem::snew_array_1D(nSfP, nSP+1,dmrfl);
    nPntS[0] = 0;
    nSfP[0] = 0;
    for(i=0; i<nSP; i++) nSfP[i+1] = nSfP[i] + mrk[i];
    mfmem::sdel_array_1D(mrk);
    mflog::log << std::endl << " nPntS = " << nSfP[nSP] << std::endl;
    
    IntType *ntmp = NULL;
    IntType *SfP  = NULL;     //物面点的相关面   //zhyb: 每个物面点对应的面号
    IntType *PntS = NULL;       //zhyb: 每个物面面对应的点号
    mfmem::snew_array_1D(ntmp, nSP,dmrfl);
    mfmem::snew_array_1D(SfP, nSfP[nSP],dmrfl);
    mfmem::snew_array_1D(PntS, nSfP[nSP],dmrfl);

    for(i=0; i<nSP; i++) ntmp[i] = 0;
    nSF = 0;
    count = 0;
    for(i=0; i<nBFace; i++){
        type = bcr[i]->GetType();
        if(type==WALL){
            for(k=nFN[i]; k<nFN[i+1]; k++){
                IntType pnt = PtNew[f2n[k]];
                PntS[count++] = pnt;
                SfP[nSfP[pnt]+ntmp[pnt]] = nSF;
                ntmp[pnt]++;
            }
            nPntS[++nSF] = count;
        }
    }
    mfmem::sdel_array_1D(ntmp);
    mfmem::sdel_array_1D(nFN);
    mfmem::sdel_array_1D(PtNew);
    mflog::log << endl << " nPntS = " << nPntS[0] << IOS_SEP << nPntS[nSF] << endl;

    IntType *SP    = NULL;      //物面点的序号
    RealGeom *xtmp = NULL;
    mfmem::snew_array_1D(SP, nSP,dmrfl);
    mfmem::snew_array_1D(xtmp, nSP,dmrfl);
    for(i=0; i<nSP; i++){
        SP[i] = i;
        xtmp[i] = xSf[i];
    }
    quicksort( 1, nSP, SP, xtmp);
    mfmem::sdel_array_1D(xtmp);
  
    IntType nBox;           //zhyb: 等于物面点数的开方+1
    count = IntType(sqrt(1.*nSP));
    nBox = static_cast<IntType> (nSP/(count+TINY) + 1);

    IntType *nPBox = NULL;
    mfmem::snew_array_1D(nPBox, nBox+1,dmrfl);
    for(i=0; i<nBox; i++){
        nPBox[i] = i*count;
        if(nPBox[i]>nSP){
            nBox = i;
            break;
        }
    }
    nPBox[i]=nSP;
    mflog::log << "nBox = " << nBox << " nSP = " << nSP << IOS_SEP << count << endl;

    RealGeom **BBox = NULL;  // sort: xmin, xmax, ymin, ymax, zmin, zmax
    mfmem::snew_array_2D(BBox, nBox, 6, dmrfl,true);
    for(i=0; i<nBox; i++){
        BBox[i][0] = BIG;
        BBox[i][1] = -BIG;
        BBox[i][2] = BIG;
        BBox[i][3] = -BIG;
        BBox[i][4] = BIG;
        BBox[i][5] = -BIG;
    }
    for(i=0; i<nBox; i++){
        IntType pnt;
        for(j=nPBox[i]; j<nPBox[i+1]; j++){
            pnt = SP[j];
            if( xSf[pnt]<BBox[i][0] ) BBox[i][0]=xSf[pnt];
            if( xSf[pnt]>BBox[i][1] ) BBox[i][1]=xSf[pnt];
            if( ySf[pnt]<BBox[i][2] ) BBox[i][2]=ySf[pnt];
            if( ySf[pnt]>BBox[i][3] ) BBox[i][3]=ySf[pnt];
            if( zSf[pnt]<BBox[i][4] ) BBox[i][4]=zSf[pnt];
            if( zSf[pnt]>BBox[i][5] ) BBox[i][5]=zSf[pnt];
        }
    }
    mflog::log << "Finished to compute BBox" << endl;

    String  filetmp;
    GetData(filename, STRING, 1, "griddir");
    sprintf(filetmp, "GeominfoDist.dat");
    strcat(filename, filetmp);

    fp=fopen(filename,"wb");
    //fp=fopen(filetmp,"wb");
    if(!fp){
        mflog::log<<"Failed to open file "<<filename<<" in function WriteInfoDist!"<<endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    //write the coorade of all nodes
    fwrite(&nSP,  sizeof(IntType),  1, fp);
    fwrite(xSf,   sizeof(RealGeom), nSP, fp);
    fwrite(ySf,   sizeof(RealGeom), nSP, fp);
    fwrite(zSf,   sizeof(RealGeom), nSP, fp);
    fwrite(&nSF,  sizeof(IntType),  1, fp);
    fwrite(nSfP,  sizeof(IntType),  nSP+1, fp);
    fwrite(SfP,   sizeof(IntType),  nSfP[nSP], fp);
    fwrite(nPntS, sizeof(IntType),  nSF+1, fp);
    fwrite(PntS,  sizeof(IntType),  nPntS[nSF], fp);
    fwrite(SP,    sizeof(IntType),  nSP, fp);

    fwrite(&nBox, sizeof(IntType),  1, fp);
    fwrite(nPBox, sizeof(IntType),  nBox+1, fp);
    for( i=0; i<nBox; i++)
        fwrite(BBox[i],  sizeof(RealGeom),  6, fp);

    fclose(fp);
    mfmem::sdel_array_1D(xSf);
    mfmem::sdel_array_1D(ySf);
    mfmem::sdel_array_1D(zSf);
    mfmem::sdel_array_1D(nSfP);
    mfmem::sdel_array_1D(SfP);
    mfmem::sdel_array_1D(nPntS);
    mfmem::sdel_array_1D(PntS);
    mfmem::sdel_array_1D(SP);
    mfmem::sdel_array_1D(nPBox);
    mfmem::sdel_array_2D(BBox);
  
}


/************************************************************************
    计算网格单元体心到壁面的距离in the MPI.
************************************************************************/
void PolyGrid::ComputeCellDist()
{
#ifndef MPICH
    WriteInfoDist();
#endif
   
    RealGeom *dist2wall_cell=0;
    dist2wall_cell = (RealGeom *)GetDataPtr(REAL_GEOM, nTCell, "dist2wall_cell");
    if(!dist2wall_cell){
        mfmem::snew_array_1D(dist2wall_cell, nTCell,dmrfl);
        UpdateDataPtr(dist2wall_cell,REAL_GEOM,nTCell,"dist2wall_cell");
    }
    for(IntType j=0; j<nTCell; j++) dist2wall_cell[j] = BIG;

    IntType mark = 1;
    ComputeDist2WallTriang(dist2wall_cell, mark); 
}


void PolyGrid::Set_RecvSend(RealFlow ***bqs, RealFlow ***bqr, IntType nvar)
{
    IntType i, j, k, temp_nZIFace=0;

    for(i=0; i<nNeighbor; i++) temp_nZIFace += nZIFace[i];
    bqs[0] = NULL;
    bqr[0] = NULL;
    mfmem::snew_array_1D(bqs[0], nvar*nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqr[0], nvar*nNeighbor,dmrfl);

    for(i=1; i<nNeighbor; i++){
        bqs[i] = &bqs[i-1][nvar];
        bqr[i] = &bqr[i-1][nvar];
    }
    bqs[0][0] = NULL;
    bqr[0][0] = NULL;
    mfmem::snew_array_1D(bqs[0][0], nNeighbor*nvar*temp_nZIFace,dmrfl);
    mfmem::snew_array_1D(bqr[0][0], nNeighbor*nvar*temp_nZIFace,dmrfl);
    for(i=1; i<nNeighbor; i++){
        bqs[i][0] =&bqs[i-1][0][nvar*nZIFace[i-1]];
        bqr[i][0] =&bqr[i-1][0][nvar*nZIFace[i-1]];
    }
    for(i=0; i<nNeighbor; i++){
        for(j=1; j<nvar; j++){
            bqs[i][j] = &bqs[i][j-1][nZIFace[i]];
            bqr[i][j] = &bqr[i][j-1][nZIFace[i]];
        }
    }

    for(i=0; i<nNeighbor; i++){
        for(j=0; j<nvar; j++){
            for(k=0; k<nZIFace[i]; k++){
                bqs[i][j][k]=0.0;
                bqr[i][j][k]=0.0;
            }
        }
    }
}

void PolyGrid::Set_MatrixRecvSend(MATRIXTYPE **bqs, MATRIXTYPE **bqr, IntType nvar)
{
    IntType i, j, k, temp_nZIFace = 0;

    for(i = 0; i < nNeighbor; i++) temp_nZIFace += nZIFace[i];

    bqs[0] = NULL;
    bqr[0] = NULL;

    // 为每个邻居分配一个一维数组，大小为 nvar * nZIFace[i]
    // 先分配所有数据的总内存
    mfmem::snew_array_1D(bqs[0], nNeighbor * nvar * temp_nZIFace, dmrfl);
    mfmem::snew_array_1D(bqr[0], nNeighbor * nvar * temp_nZIFace, dmrfl);

    // 设置每个 bqs[i] 指向对应邻居的数据块起始位置
    for(i = 1; i < nNeighbor; i++) {
        bqs[i] = &bqs[i-1][nvar * nZIFace[i-1]];
        bqr[i] = &bqr[i-1][nvar * nZIFace[i-1]];
    }

    for(i = 0; i < nNeighbor; i++) {
        for(j = 0; j < nZIFace[i] * nvar; j++) {
            //for(k = 0; k < nvar; k++) {
            //    IntType idx = j * nvar + k;
            bqs[i][j] = 0.0;
            bqr[i][j] = 0.0;
            //}
        }
    }
}

void PolyGrid::Add_RecvSend(RealFlow ***bqs, RealFlow *q, IntType num_var)
{
    IntType i, j;
    for(i=0; i<nNeighbor; i++){
        for(j=0; j<nZIFace[i]; j++){
            bqs[i][num_var][j] = q[bCNo[i][j]];
        }
    }
}

void PolyGrid::Add_MatrixRecvSend(RealFlow ***bqs, RealFlow *q, IntType nvar)
{
    IntType i, j, k, ghost;
    for(i=0; i<nNeighbor; i++){
        for(j=0; j<nZIFace[i]; j++){
            for(k=0; k<nvar; k++){
                ghost = bCNo[i][j]; // - (nBFace-nIFace);
                bqs[i][k][j] = q[ ghost * nvar + k];
            }
        }
    }
}     

void PolyGrid::Add_MatrixRecvSend2(MATRIXTYPE **bqs, MATRIXTYPE *q, IntType nvar)
{
    IntType i, j, k, ghost;
    for(i=0; i<nNeighbor; i++){
        for(j=0; j<nZIFace[i]; j++){
            for(k=0; k<nvar; k++){
                ghost = bCNo[i][j]; // - (nBFace-nIFace);
                bqs[i][ j * nvar + k ] = q[ ghost * nvar + k ];
            }
        }
    }
} 

void PolyGrid::Read_RecvSend(RealFlow ***bqr, RealFlow *q, IntType num_var)
{
    IntType i, j, ghost;
    for(i=0; i<nNeighbor; i++) {
        for(j=0; j<nZIFace[i]; j++) {
            ghost    = nTCell + bFNo[i][j];
            q[ghost] = bqr[i][num_var][j];
        }
    } 
}

void PolyGrid::Read_MatrixRecvSend(RealFlow ***bqr, RealFlow *q, IntType nvar)
{
    IntType i, j, k, ghost;
    for(i=0; i<nNeighbor; i++) {
        for(j=0; j<nZIFace[i]; j++) {
            ghost    = nTCell + bFNo[i][j] - (nBFace - nIFace);
            for(k=0; k<nvar; k++){
                q[ghost * nvar + k] = bqr[i][k][j];
            }
        }
    } 
}

void PolyGrid::Read_MatrixRecvSend2(MATRIXTYPE **bqr, MATRIXTYPE *q, IntType nvar)
{
    IntType i, j, k, ghost;
    for(i=0; i<nNeighbor; i++) {
        for(j=0; j<nZIFace[i]; j++) {
            ghost    = nTCell + bFNo[i][j] - (nBFace - nIFace);
            for(k=0; k<nvar; k++){
                q[ghost * nvar + k] = bqr[i][j * nvar + k];
            }
        }
    } 
}
/************************************************************************
  初始化非定常计算网格单元面心的速度和体心的速度
************************************************************************/
void PolyGrid::InitialVgn()
{
    if(vgn==0){
        IntType i;
        
        RealGeom BFacevgx_init=0.0, BFacevgy_init=0.0, BFacevgz_init=0.0;
        
        //zhyb: 保存上一步的边界面速度，用于非定常流场中流体加速度对物面边界压力的影响
        //zhyb: 初始化为第一步的速度
        RealGeom *BFacevgx_last = NULL;
        RealGeom *BFacevgy_last = NULL;
        RealGeom *BFacevgz_last = NULL;
        mfmem::snew_array_1D(BFacevgx_last, nBFace,dmrfl);
        mfmem::snew_array_1D(BFacevgy_last, nBFace,dmrfl);
        mfmem::snew_array_1D(BFacevgz_last, nBFace,dmrfl);
        for(i=0; i<nBFace; i++){
            BFacevgx_last[i] = BFacevgx_init;
            BFacevgy_last[i] = BFacevgy_init;
            BFacevgz_last[i] = BFacevgz_init;
        }
        UpdateDataPtr(BFacevgx_last, REAL_GEOM, nBFace, "BFacevgx_last");
        UpdateDataPtr(BFacevgy_last, REAL_GEOM, nBFace, "BFacevgy_last");
        UpdateDataPtr(BFacevgz_last, REAL_GEOM, nBFace, "BFacevgz_last");
     
        vgn      = NULL;
        BFacevgx = NULL;
        BFacevgy = NULL;
        BFacevgz = NULL;
        mfmem::snew_array_1D(vgn     , nTFace,dmrfl);
        mfmem::snew_array_1D(BFacevgx, nBFace,dmrfl);
        mfmem::snew_array_1D(BFacevgy, nBFace,dmrfl);
        mfmem::snew_array_1D(BFacevgz, nBFace,dmrfl);
        for(i=0; i<nBFace; i++){
            BFacevgx[i] = BFacevgx_init;
            BFacevgy[i] = BFacevgy_init;
            BFacevgz[i] = BFacevgz_init;
        }
        for(i=0; i<nTFace; i++){
            vgn[i] = BFacevgx_init*xfn[i]+BFacevgy_init*yfn[i]+BFacevgz_init*zfn[i];
        }
        
    }else{
        mflog::log.set_one_processor_out();
        mflog::log << "Error! vgn!=0 in function InitialVgn!" << endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
}


/************************************************************************
 quicksort排序法的原始子程序
************************************************************************/
void PolyGrid::quick_sort(IntType *a, IntType is, IntType ie)
{
    IntType i, last;
    if(ie-is<=1) return;
    swap(a, is, (ie+is)/2);
    last = is;
    for(i=is+1; i<=ie; i++){
        if(a[i]<a[is]) swap(a, ++last, i);
    }
    swap(a, is, last);
    quick_sort(a, is, last);
    quick_sort(a, last+1, ie);
}


void PolyGrid::swap(IntType *a, IntType i, IntType j)
{
    IntType temp; 
    temp = a[i];
    a[i] = a[j];
    a[j] = temp;
}


/************************************************************************
                         check grid quality 
************************************************************************/
void PolyGrid::CheckGridQuality()
{   
    
    if(level==0) CheckGridScale();
    
    IntType n = nTCell+nBFace;
    if(facecentroidskewness==0) mfmem::snew_array_1D(facecentroidskewness, nTFace,dmrfl);
    if(faceangleskewness==0) mfmem::snew_array_1D(faceangleskewness, nTFace,dmrfl);
    if(cellcentroidskewness==0) mfmem::snew_array_1D(cellcentroidskewness, n,dmrfl);
    if(cellvolsmoothness==0) mfmem::snew_array_1D(cellvolsmoothness, n,dmrfl);
    if(cellwallnumber==0) mfmem::snew_array_1D(cellwallnumber, nTCell,dmrfl);
    if(faceangle==0) mfmem::snew_array_1D(faceangle, nTFace,dmrfl);
    
    SkewnessSummary();
    EquiangleSkewnessSummary();
    SmoothnessSummary();
    FindIllWallCell();
    CheckSymmetryFace();
    FaceAngleSummary();
    
    //deal bad grid for robust
    DealBadGrid();
}


/******************************************************************************\
                summary of skewness of a given grid
\******************************************************************************/ 
void PolyGrid::SkewnessSummary()
{
    IntType  i, c1, c2, type, count;
    IntType  angleC[18];
    IntType  n = nTCell+nBFace;
    RealGeom dotp1, x1, y1, z1, dis1, dotp2, x2, y2, z2, dis2, angle, angle1, angle2;
    RealGeom minAng, maxAng;
    
    if(nNPC == 0) nNPC = CalnNPC(this);
    if(C2N == 0)  C2N  = CalC2N(this);
    
    //face skew check
    for(i=0; i<18; i++){
        angleC[i]=0;
    }
    
    //cell skew initialized
    for(i=0;i<n;i++){
        cellcentroidskewness[i] = 90.0;
    }
    
    minAng = 90.0;
    maxAng = -90.0;
    for(i=0; i<nTFace; i++) {
        c1 = f2c[i+i];
        c2 = f2c[i+i+1];
        
        x1 = xfc[i] - xcc[c1];
        y1 = yfc[i] - ycc[c1];
        z1 = zfc[i] - zcc[c1];
        dis1 = sqrt(x1*x1 + y1*y1 + z1*z1);
        dotp1 = (xfn[i]*x1 + yfn[i]*y1 + zfn[i]*z1)/(dis1 + TINY);
        dotp1 = MIN(dotp1,  1.0);
        dotp1 = MAX(dotp1, -1.0);
        angle1 = asin(dotp1)*180/PI; 
        cellcentroidskewness[c1] = MIN(cellcentroidskewness[c1], angle1);
        
        x2 = xcc[c2] - xfc[i];
        y2 = ycc[c2] - yfc[i];
        z2 = zcc[c2] - zfc[i];
        dis2  = sqrt(x2*x2 + y2*y2 + z2*z2);
        dotp2 = (xfn[i]*x2 + yfn[i]*y2 + zfn[i]*z2)/(dis2 + TINY);
        dotp2 = MIN(dotp2,  1.0);
        dotp2 = MAX(dotp2, -1.0);
        angle2 = asin(dotp2)*180/PI;
        if(i >= nBFace){
            cellcentroidskewness[c2] = MIN(cellcentroidskewness[c2], angle2);
        }
        
        angle  = MIN(angle1,angle2);
        facecentroidskewness[i] = angle;
        
        minAng = MIN(minAng, angle);
        maxAng = MAX(maxAng, angle);
        
        angle = MIN(angle,89.9);
        angle = MAX(angle,-89.9);
        if(angle < 0) angle -= 10;
        angleC[(IntType)(angle/10) + 9]++;
    }   
    
    mflog::log.set_one_processor_out();

    IntType nTFace_glb = nTFace;
#ifdef MPICH
    Parallel::parallel_sum(nTFace_glb, MPI_COMM_WORLD);
    Parallel::parallel_sum(angleC, 18, MPI_COMM_WORLD);
    Parallel::parallel_min_max(minAng, maxAng, MPI_COMM_WORLD);
#endif    
    
    mflog::log << SEP_LINE << endl;
    mflog::log << " Face Skewness Summary (face angle of 90 degrees being the best) " << endl;
    mflog::log << SEP_LINE << endl;  
    mflog::log << "Total faces number: " << nTFace_glb << endl;
    mflog::log << "Min skewness angle: " << IOS_EP(6) << minAng << endl;
    mflog::log << "Max skewness angle: " << IOS_EP(6) << maxAng << endl;
    for(i=0; i<18; i++){
        mflog::log << "Face angle from " << (int)(-90+i*10) << " to " << (int)(-80+i*10) << " is "
            << IOS_FWP(5,2) << angleC[i]/((float) nTFace_glb)*100 << " percent, " << (long)angleC[i]
            << std::endl;
    }
    mflog::log << SEP_LINE << endl;
    
    
    if(level == 0){
        RealGeom BadFaceAngle = -1.0;
        GetData(&BadFaceAngle, REAL_GEOM, 1, "BadFaceAngle", 0); 
        
        count = 0;
        for(i=0;i<nTFace;i++){
            if(facecentroidskewness[i]<BadFaceAngle){
                count++;
            }
        }

        IntType count_glb = count;
#ifdef MPICH
        Parallel::parallel_sum(count_glb, MPI_COMM_WORLD);
#endif
        mflog::log << "Bad Face Angle = "<< BadFaceAngle << endl
                   << "The number of face centroid skewness angle less than BadFaceAngle is: "
                   << count_glb << endl;
    }
    
    //cell skew check
    //ghost cell's value
    for(i=0;i<nBFace;i++){
        c1 = f2c[i+i];
        c2 = f2c[i+i+1];
        type = bcr[i]->GetType();
        if(type == INTERFACE) continue;
        
        cellcentroidskewness[c2] = cellcentroidskewness[c1];
    }
#ifdef MPICH
    CommInterfaceDataMPI(cellcentroidskewness);
#endif
    
    for(i=0; i<18; i++) angleC[i] = 0;    
    minAng = 90.0;
    maxAng = -90.0;
    for(i=0;i<nTCell;i++){
        minAng = MIN(minAng,cellcentroidskewness[i]);
        maxAng = MAX(maxAng,cellcentroidskewness[i]);
        
        angle = MIN(cellcentroidskewness[i],89.9);
        angle = MAX(angle,-89.9);
        if(angle < 0) angle -= 10;
        angleC[(IntType)(angle/10)+9]++;
    }

    IntType nTCell_glb = nTCell;
#ifdef MPICH
    Parallel::parallel_sum(nTCell_glb, MPI_COMM_WORLD);
    Parallel::parallel_sum(angleC, 18, MPI_COMM_WORLD);
    Parallel::parallel_min_max(minAng, maxAng, MPI_COMM_WORLD);
#endif
    
    mflog::log << SEP_LINE << endl;
    mflog::log << " Cell Skewness Summary(cell angle of 90 degrees being the best) " << endl;
    mflog::log << SEP_LINE << endl; 
    mflog::log << "Total cell number: " << nTCell_glb << endl;
    mflog::log << "Min cell centroid skewness value is " << IOS_EP(6) << minAng << endl;
    mflog::log << "Max cell centroid skewness value is " << IOS_EP(6) << maxAng << endl;
    for(i=0; i<18; i++){
        mflog::log << "Cell centroid skewness value from " << (int)(-90+i*10) << " to "
            << (int)(-80+i*10) << " is " << IOS_FWP(5,2) << angleC[i]/((float)nTCell_glb)*100 
            << " percent, " << (long)angleC[i] << std::endl;
    }
    mflog::log << SEP_LINE << endl;
        
    if(level == 0){
        RealGeom BadCellAngle = 0.0;
        GetData(&BadCellAngle, REAL_GEOM, 1, "BadCellAngle", 0); 
        
        count = 0;
        for(i=0;i<nTCell;i++){
            if(cellcentroidskewness[i]<BadCellAngle){
                count++;
            }
        }
        IntType count_glb = count;
#ifdef MPICH
        Parallel::parallel_sum(count_glb, MPI_COMM_WORLD);
#endif
        mflog::log << "Bad Cell Angle = " << BadCellAngle << endl
                   << "The number of cell centroid skewness angle less than BadCellAngle is: " 
                   << count_glb << endl;
    }
}


/******************************************************************************\
                summary of equi-angle skewness of a given grid
\******************************************************************************/ 
void PolyGrid::EquiangleSkewnessSummary()
{
    IntType i,j,p1,p2,p3;
    IntType face_skew[11];
    RealGeom xx1,yy1,zz1,l1,xx2,yy2,zz2,l2,dot;
    RealGeom angle,min_angle,max_angle,e_angle,min_faceskew,max_faceskew;
    
    if(F2N==0){
        F2N = CalF2N(this);
    }
    assert(F2N);

    RealGeom *x = GetX();
    RealGeom *y = GetY();
    RealGeom *z = GetZ();
    
    for(i=0;i<11;i++)
        face_skew[i] = 0;
    
    min_faceskew = 1.0;
    max_faceskew = 0.0;
    for(i=0;i<nTFace;i++){
        min_angle = 180.0;
        max_angle = 0.0;
        for(j=0;j<nNPF[i];j++){
            p1 = F2N[i][j];
            if(j == 0){
                p2 = F2N[i][nNPF[i]-1];
                p3 = F2N[i][j+1];
            }else if(j == nNPF[i]-1){
                p2 = F2N[i][j-1];
                p3 = F2N[i][0];
            }else{
                p2 = F2N[i][j-1];
                p3 = F2N[i][j+1];
            }
            
            xx1 = x[p2]-x[p1];
            yy1 = y[p2]-y[p1];
            zz1 = z[p2]-z[p1];
            xx2 = x[p3]-x[p1];
            yy2 = y[p3]-y[p1];
            zz2 = z[p3]-z[p1];
            l1  = sqrt(xx1*xx1+yy1*yy1+zz1*zz1);
            l2  = sqrt(xx2*xx2+yy2*yy2+zz2*zz2);
            dot = xx1*xx2+yy1*yy2+zz1*zz2;
            dot/= (l1*l2);
            dot = MIN(dot,1.0);
            dot = MAX(dot,-1.0);
            angle = acos(dot)*180.0/PI;
            max_angle = MAX(max_angle,angle);
            min_angle = MIN(min_angle,angle);
        }
        e_angle = 180.0*(1.0-2.0/(RealGeom)nNPF[i]);
        faceangleskewness[i] = MAX((max_angle-e_angle)/(180.0-e_angle),(e_angle-min_angle)/e_angle);
        
        min_faceskew = MIN(min_faceskew,faceangleskewness[i]);
        max_faceskew = MAX(max_faceskew,faceangleskewness[i]);
        face_skew[(IntType)(faceangleskewness[i]*10)]++;
    }
    //是否删除faceangleskewness;lihuan-2018-11-26
    //mfmesh::sdel_array_1D(faceangleskewness);

    mflog::log.set_one_processor_out();

    IntType nTFace_glb = nTFace;
#ifdef MPICH
    Parallel::parallel_sum(nTFace_glb, MPI_COMM_WORLD);
    Parallel::parallel_sum(face_skew, 10, MPI_COMM_WORLD);
    Parallel::parallel_min_max(min_faceskew, max_faceskew, MPI_COMM_WORLD);
#endif    

    mflog::log << SEP_LINE << endl;
    mflog::log << "          Face Equiangle Skewness Summary"<<endl;
    mflog::log << "(0.0 being the best, 1.0 is the worse, value below 0.9 is acceptable)" << endl;
    mflog::log << SEP_LINE << endl;  
    mflog::log << "Min face equiangle skewness value is " << IOS_EP(6) << min_faceskew << endl;
    mflog::log << "Max face equiangle skewness value is " << IOS_EP(6) << max_faceskew << endl;
    for(i=0; i<10; i++){
        mflog::log << "Face equiangle skewness value from " << IOS_FWP(3,1) << i*0.1 << " to "
            << IOS_FWP(3,1) << (i+1)*0.1 << " is " << IOS_FWP(5,2) << face_skew[i]/((float) nTFace_glb)*100
            << " percent, " << (long)face_skew[i] << std::endl;
    }
    mflog::log << SEP_LINE << endl;
}


/************************************************************************
    Summary of cell's smoothness of a given grid  
************************************************************************/
void PolyGrid::SmoothnessSummary()
{
    IntType  i,j,c1,c2,cell,type;
    IntType  cell_skew[10];
    RealGeom min_vol,max_vol,min_cellskew,max_cellskew;
    
    if(nCPC==0) nCPC = CalnCPC(this);
    if(c2c==0) c2c = CalC2C(this);

    for(i=0;i<10;i++)
        cell_skew[i] = 0;

    min_vol = BIG;
    max_vol = 0.0;
    for(i=0;i<nTCell;i++){
        min_vol = MIN(min_vol,vol[i]);
        max_vol = MAX(max_vol,vol[i]);

        cellvolsmoothness[i] = BIG;
    }
    
    for(i=0;i<nTCell;i++){
        for(j=0;j<nCPC[i];j++){
            cell = c2c[i][j];       
            cellvolsmoothness[i] = MIN(cellvolsmoothness[i],MIN(vol[cell],vol[i])/MAX(vol[cell],vol[i]));
        }
        cellvolsmoothness[i] = 1.0-cellvolsmoothness[i];
    }

    for(i=0;i<nBFace;i++){
      c1 = f2c[i+i]; 
      c2 = f2c[i+i+1]; 
        type = bcr[i]->GetType();
        if(type == INTERFACE) continue;
         
        cellvolsmoothness[c2] = cellvolsmoothness[c1];
    }
#ifdef MPICH
    CommInterfaceDataMPI(cellvolsmoothness);
#endif  

    min_cellskew = 1.0;
    max_cellskew = 0.0;
    for(i=0;i<nTCell;i++){
        min_cellskew = MIN(min_cellskew,cellvolsmoothness[i]);
        max_cellskew = MAX(max_cellskew,cellvolsmoothness[i]);
        cell_skew[(IntType)(cellvolsmoothness[i]*10)]++;
    }

    mflog::log.set_one_processor_out();

    IntType nTCell_glb = nTCell;
#ifdef MPICH
    Parallel::parallel_sum(nTCell_glb, MPI_COMM_WORLD);
    Parallel::parallel_sum(cell_skew, 10, MPI_COMM_WORLD);
    Parallel::parallel_min_max(min_cellskew, max_cellskew, MPI_COMM_WORLD);
    Parallel::parallel_min_max(min_vol, max_vol, MPI_COMM_WORLD);
#endif

    mflog::log << SEP_LINE << endl;
    mflog::log << "             Cell volume Smoothness Summary"<<endl;
    mflog::log << "(0.0 being the best, 1.0 is the worse, value below 0.8 is acceptable)" << endl;
    mflog::log << SEP_LINE << endl;  
    mflog::log << "Min cell volume value is " << IOS_EP(6) << min_vol << endl;
    mflog::log << "Max cell volume value is " << IOS_EP(6) << max_vol << endl;
    mflog::log << "Min cell volume smoothness value is " << IOS_EP(6) << min_cellskew << endl;
    mflog::log << "Max cell volume smoothness value is " << IOS_EP(6) << max_cellskew << endl;

    for(i=0; i<10; i++){
        mflog::log << "Cell volume smoothness value from " << IOS_FWP(3,1) << i*0.1 << " to "
            << IOS_FWP(3,1) << (i+1)*0.1 << " is " << IOS_FWP(5,2) << cell_skew[i]/((float) nTCell_glb)*100 
            << " percent, " << (long)cell_skew[i] << std::endl;
    }
    mflog::log << SEP_LINE << endl;
}


/************************************************************************
    find ill wall cell, viz. the cell that has two or more wall faces  
************************************************************************/
void PolyGrid::FindIllWallCell()
{
    IntType i,type,c1;
    IntType *nNPC = CalnNPC(this);
    
    IntType vis_mode = LAMINAR;
    GetData(&vis_mode, INT, 1, "vis_mode");

    for(i=0;i<nTCell;i++)
        cellwallnumber[i]=0; 
   
    for(i=0;i<nBFace;i++){
        type  = bcr[i]->GetType();
        c1    = f2c[i+i];
        if(type != WALL) continue;
        cellwallnumber[c1]++;
    }
    
    IntType num_tetra=0,num_pyra=0,num_wall2=0;
    for(i=0;i<nTCell;i++){
        if(cellwallnumber[i] == 0) continue;
        
        if(vis_mode != INVISCID){
            if(nNPC[i] == 4){
                num_tetra++;     
            }
            else if(nNPC[i] == 5){
                num_pyra++;     
            }
        }

        if(cellwallnumber[i]>1){
            num_wall2++;     
        }        
    }

    mflog::log.set_one_processor_out();

#ifdef MPICH
    Parallel::parallel_sum(num_wall2, MPI_COMM_WORLD);
#endif
    if(num_wall2 > 0){
        mflog::log << endl << "total have " << num_wall2 << " cell has two or more face on wall!!!" << endl;
    } 

    if(vis_mode != INVISCID){
        IntType tet_pyr_total[2] = {num_tetra, num_pyra};
#ifdef MPICH
        Parallel::parallel_sum(tet_pyr_total, 2, MPI_COMM_WORLD);
#endif
        mflog::log << endl << "total have " << tet_pyr_total[0] << " tetrahedron cell on wall!" << endl;
        mflog::log << endl << "total have " << tet_pyr_total[1] << " pyramid cell on wall!" << endl;    
    } 
}


/************************************************************************
*     check symmetry face, find the max distant of symmetry boundary    *
*                         node from symmetry plane                      *  
************************************************************************/
void PolyGrid::CheckSymmetryFace()
{
    IntType i,j,type,count,p1,nsymm;
    RealGeom dmax,maxx=0,maxy=0,maxz=0;
    
    RealGeom *x = GetX();
    RealGeom *y = GetY();
    RealGeom *z = GetZ();
    
    count = 0;
    nsymm = 0;
    dmax = 0.0;
    for(i=0;i<nBFace;i++){
        type = bcr[i]->GetType();
        if(type == SYMM){
            nsymm++;
            for(j=0;j<nNPF[i];j++){
                p1 = f2n[count++];
                if(dmax<fabs(y[p1])){
                    dmax = fabs(y[p1]);
                    maxx = x[p1];
                    maxy = y[p1];
                    maxz = z[p1];
                }                
            }
        }else{
            count += nNPF[i];
        }
    }

    mflog::log.set_all_processors_out();

#ifdef MPICH
    struct{
        RealGeom dmax;
        IntType rank;
    } in,out;
    in.dmax = dmax;
    in.rank = myZone;
#ifdef SINGLE_PRECISION
    MPI_Allreduce(&in, &out, 1, MPI_FLOAT_INT,  MPI_MINLOC, MPI_COMM_WORLD);
#else
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
#endif
    IntType nsymm_total;
    MPI_Allreduce(&nsymm, &nsymm_total, 1, MPIIntType, MPI_SUM, MPI_COMM_WORLD);

    if(myZone==out.rank && nsymm_total!=0){        
        mflog::log<<endl<<"There have "<<nsymm_total<<"symmetry boundary face."
                  <<endl<<"The max distant of symmetry boundary node from symmetry plane is: "<<IOS_EP(8)<<maxy
                  <<endl<<"The coordinate is: "<<maxx<<", "<<maxy<<", "<<maxz<<endl<<endl;
    }
#else
    if(nsymm!=0){
        mflog::log<<endl<<"There have "<<nsymm<<"symmetry boundary face."
                  <<endl<<"The max distant of symmetry boundary node from symmetry plane is: "<<IOS_EP(8)<<maxy
                  <<endl<<"The coordinate is: "<<maxx<<", "<<maxy<<", "<<maxz<<endl<<endl;
    }
#endif
}


/************************************************************************
*           check grid scale, estimate precision enough or not          *  
************************************************************************/
void PolyGrid::CheckGridScale()
{
    IntType i,j,k,type,p1,p2,c1,count;
    RealGeom len,lenscal,maxx,minx,maxy,miny,maxz,minz;
    
    RealGeom ratio;
#ifdef  SINGLE_PRECISION
    ratio = 1.0e-7;
#else   //DOUBLE_PRECISION
    ratio = 1.0e-14;
#endif
    if(nNPC==0) nNPC  = CalnNPC(this);
    if(C2N==0)  C2N  = CalC2N(this);
    if(F2N==0)  F2N  = CalF2N(this);
    
    RealGeom *x=GetX();
    RealGeom *y=GetY();
    RealGeom *z=GetZ();
    
    maxx = -BIG;
    minx = BIG;
    maxy = -BIG;
    miny = BIG;
    maxz = -BIG;
    minz = BIG;
    for(i=0;i<nBFace;i++){
        type  = bcr[i]->GetType();
        if(type != WALL) continue;
        
        for(j=0;j<nNPF[i];j++){
            p1 = F2N[i][j];
            maxx = MAX(maxx,x[p1]);
            minx = MIN(minx,x[p1]);
            maxy = MAX(maxy,y[p1]);
            miny = MIN(miny,y[p1]);
            maxz = MAX(maxz,z[p1]);
            minz = MIN(minz,z[p1]);
        }   
    }

    mflog::log.set_one_processor_out();

#ifdef MPICH
    Parallel::parallel_min_max(minx, maxx, MPI_COMM_WORLD);
    Parallel::parallel_min_max(miny, maxy, MPI_COMM_WORLD);
    Parallel::parallel_min_max(minz, maxz, MPI_COMM_WORLD);
#endif
    lenscal = MAX(MAX(maxx-minx,maxy-miny),maxz-minz);
    mflog::log<<endl<<"length scale is: "<<IOS_EP(4)<<lenscal<<endl<<endl;

    count = 0;
    for(i=0;i<nBFace;i++){
        type  = bcr[i]->GetType();
        if(type != WALL) continue;
        
        c1 = f2c[i+i];
        for(j=0;j<nNPC[c1]-1;j++){
            p1 = C2N[c1][j];
            for(k=j+1;k<nNPC[c1];k++){
                p2 = C2N[c1][k];
                len = sqrt((x[p2]-x[p1])*(x[p2]-x[p1])+
                           (y[p2]-y[p1])*(y[p2]-y[p1])+
                           (z[p2]-z[p1])*(z[p2]-z[p1]));
                if(len/lenscal<ratio){
                    mflog::log.set_all_processors_out();
                    mflog::log << endl<<"p1 and p2 is too close!"<<endl;
                    mflog::log << "p1 " << p1 << IOS_EP(10) << x[p1] << IOS_EP(10) << y[p1] << IOS_EP(10) << z[p1] << endl;
                    mflog::log << "p2 " << p2 << IOS_EP(10) << x[p2] << IOS_EP(10) << y[p2] << IOS_EP(10) << z[p2] << endl;
                    count++;
                }
            }
        }
    }
#ifdef MPICH
    Parallel::parallel_sum(count, MPI_COMM_WORLD);
#endif
    if(count!=0){
        mflog::log.set_one_processor_out();
        mflog::log<<endl<<"There has "<<count<<" node too fine!"<<endl
            <<"You should use higher precision or make grid's first layer being coarser!"<<endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }    
}


/************************************************************************
                           Find Cell Layer No.
                         from wall, No.0,1,2,3...
************************************************************************/
void PolyGrid::FindCellLayerNo()
{
    IntType i,j,c1,c2,type,mark,count;
    
    IntType n = nBFace+nTCell;
    IntType nPFace = nBFace-nIFace;
    IntType *nCPC  = CalnCPC(this);  //注意：只带并行边界的虚拟网格
    IntType **C2C  = CalC2C(this);
    
    IntType *CellLayerNo = NULL;
    mfmem::snew_array_1D(CellLayerNo, n,dmrfl);

    UpdateDataPtr(CellLayerNo, INT, n,"CellLayerNo");
    for(i=0;i<n;i++)
        CellLayerNo[i] = -1;
    
    for(i=0;i<nPFace;i++){
        type = bcr[i]->GetType();
        if (type != WALL) continue;
        c1 = f2c[i+i];
        CellLayerNo[c1] = 0;
    }
    
#ifdef MPICH
    CommInterfaceDataMPI(CellLayerNo);
#endif
    
    count = 0;
    mark = 1;
    while(mark){
        mark = 0;
        count++;
    
        for(i=nPFace;i<nBFace;i++){
            c1 = f2c[i+i];
            c2 = f2c[i+i+1];
            
            if(CellLayerNo[c2] != count-1) continue;
            
            if(CellLayerNo[c1] == -1){
                CellLayerNo[c1] = count;
                mark = 1;
            }
        }
        
        for(i=0;i<nTCell;i++){
            if(CellLayerNo[i] != count-1) continue;
                
            for(j=0; j<nCPC[i]; j++){
                c1 = C2C[i][j];
                    
                if(c1>nTCell) continue;
                    
                if(CellLayerNo[c1] == -1){
                    CellLayerNo[c1] = count;
                    mark = 1;
                }
            }
        }
#ifdef MPICH
        CommInterfaceDataMPI(CellLayerNo);
        
        //只要还有更新的，就继续循环
        IntType mark_glb;
        MPI_Allreduce(&mark, &mark_glb, 1, MPIIntType, MPI_MAX, MPI_COMM_WORLD);
        mark = mark_glb; 
#endif
    }

#ifdef DEBUG
    mflog::log.set_one_processor_out();
    mflog::log << endl << "Find cell layer no. Ok!" << endl;
#endif    
}


/************************************************************************
                   Find  face parallel to wall face
                   used for Roe's entropy fix
************************************************************************/
void PolyGrid::FindNormalFace()
{
    //IntType CellNum = 5;
    IntType CellNum = 20;
    IntType i,j,face1,face2,mark,type;
    
    IntType *nFPC = CalnFPC(this);
    IntType **C2F = CalC2F(this); 
    IntType n = nTCell+nBFace;
    
    IntType *CellLayerNo = (IntType *)this->GetDataPtr(INT, n, "CellLayerNo");
    
    IntType *IsNormalFace = NULL;
    mfmem::snew_array_1D(IsNormalFace, nTFace,dmrfl);
    this->UpdateDataPtr(IsNormalFace, INT, nTFace, "IsNormalFace");
    for(i=0;i<nTFace;i++) IsNormalFace[i] = 0;
    
    //首先物面边界面都是
    for(i=0;i<nBFace;i++){
        type = bcr[i]->GetType();
        if(type != WALL) continue;
        
        IsNormalFace[i] = 1; 
    }
    
    //附面层里面积最大的两个面一般就是垂直于物面法向的面
    for(i=0;i<nTCell;i++){
        if((CellLayerNo[i]==-1) || (CellLayerNo[i]>=CellNum)) continue;
        if(nNPC[i] < 6) continue; //排除四面体和金字塔
        
        //面积最大的面
        face1 = C2F[i][0];
        mark = 0;
        for(j=1;j<nFPC[i];j++){
            face2 = C2F[i][j];
            if(area[face1]<area[face2]){
                face1 = face2;
                mark = j;
            }
        }
        IsNormalFace[face1] = 1; 
        
        //面积第二大的面
        if(mark == 0){
            face1 = C2F[i][1];
        }else{
            face1 = C2F[i][0];
        }
        for(j=0;j<nFPC[i];j++){
            if(j == mark) continue;
            face2 = C2F[i][j];
            if(area[face1]<area[face2]){
                face1 = face2;
                mark = j;
            }
        }
        IsNormalFace[face1] = 1; 
    }
#ifdef MPICH
    IntType *mpitmp = NULL;
    mfmem::snew_array_1D(mpitmp, n,dmrfl);
    RealGeom *mpifc[3];
    for(i=0;i<3;i++){
        mpifc[i] = NULL;
        mfmem::snew_array_1D(mpifc[i], n,dmrfl);
    }
    for(i=0;i<n;i++){
        mpitmp[i] = 0;
        for(j=0;j<3;j++){
            mpifc[j][i] = 0.0;
        }
    }
    
    for(i=0;i<nBFace;i++){
        IntType c1 = f2c[i+i];
        mpitmp[c1] = IsNormalFace[i];
        mpifc[0][c1] = xfc[i];
        mpifc[1][c1] = yfc[i];
        mpifc[2][c1] = zfc[i];
    }
    
    this->CommInterfaceDataMPI(mpitmp);
    for(j=0;j<3;j++){
        this->CommInterfaceDataMPI(mpifc[j]);
    }
    
    for(i=0;i<nBFace;i++){
        IntType c2 = f2c[i+i+1];
        if(mpitmp[c2] == 1){
            if(fabs(mpifc[0][c2]-xfc[i])<TINY && 
               fabs(mpifc[1][c2]-yfc[i])<TINY &&
               fabs(mpifc[2][c2]-zfc[i])<TINY){
                IsNormalFace[i] = 1; 
            }
        }
    }
    mfmem::sdel_array_1D(mpitmp);
    for(i=0;i<3;i++){
        mfmem::sdel_array_1D(mpifc[i]);
    }
#endif      
}


/******************************************************************************\
       summary of angle of lines between face center with two center center 
\******************************************************************************/ 
void PolyGrid::FaceAngleSummary()
{
    IntType i,c1,c2;
    IntType angleC[18];
    RealGeom x1,y1,z1,x2,y2,z2,dot,dis1,dis2;
    RealGeom minang,maxang,angle;
    
    for(i=0; i<18; i++)
        angleC[i]=0;
    
    minang = 180.0;
    maxang = 0.0;
    for(i=0; i<nTFace; i++) {
        c1 = f2c[i+i];
        c2 = f2c[i+i+1];
        
        x1 = xfc[i] - xcc[c1];
        y1 = yfc[i] - ycc[c1];
        z1 = zfc[i] - zcc[c1];
        x2 = xfc[i] - xcc[c2];
        y2 = yfc[i] - ycc[c2];
        z2 = zfc[i] - zcc[c2];
        
        dot = x1*x2+y1*y2+z1*z2;
        dis1 = sqrt(x1*x1+y1*y1+z1*z1);
        dis2 = sqrt(x2*x2+y2*y2+z2*z2);
        faceangle[i] = dot/(dis1+TINY)/(dis2+TINY);
        faceangle[i] = MIN(faceangle[i],1.0);
        faceangle[i] = MAX(faceangle[i],-1.0);
        faceangle[i] = acos(faceangle[i])*180.0/PI;
    
        minang = MIN(minang, faceangle[i]);
        maxang = MAX(maxang, faceangle[i]);
        
        angle = MIN(179.0, faceangle[i]);
        angleC[(IntType)(angle/10)]++;
    }
    //是否删除faceangle;lihuan-2018-11-26
    //mfmesh::sdel_array_1D(faceangle);
    
    mflog::log.set_one_processor_out();

    IntType nTFace_glb = nTFace;
#ifdef MPICH
    Parallel::parallel_sum(nTFace_glb, MPI_COMM_WORLD);
    Parallel::parallel_sum(angleC, 18, MPI_COMM_WORLD);
    Parallel::parallel_min_max(minang, maxang, MPI_COMM_WORLD);
#endif 

    mflog::log << SEP_LINE << endl;
    mflog::log << " Face Angle Summary (angle of 180 degrees being the best) " << endl;
    mflog::log << SEP_LINE << endl;  
    mflog::log << "Total number of faces " << nTFace << endl;
    mflog::log << "Min face angle is " << IOS_EP(6) << minang << endl;
    mflog::log << "Max face angle is " << IOS_EP(6) << maxang << endl;
    for(i=0; i<18; i++){

        mflog::log << "Face angle from " << (int)(i*10) << " to " << (int)(i*10+10) 
            << " is " << IOS_FWP(5,2) << angleC[i]/((float) nTFace)*100 << " percent, " 
            << (long)angleC[i] << std::endl;
    }
    mflog::log << SEP_LINE << endl; 
} 


/******************************************************************************\
                       根据网格质量判断在做重构时是否降阶 
\******************************************************************************/ 
void PolyGrid::DealBadGrid()
{
    IntType i;
    
    IntType *IfOneOrder = NULL;
    IfOneOrder = (IntType *)this->GetDataPtr(INT, nTFace, "IfOneOrder");
    if(!IfOneOrder){
        mfmem::snew_array_1D(IfOneOrder, nTFace,dmrfl);
        UpdateDataPtr(IfOneOrder, INT, nTFace, "IfOneOrder");
    }
    
    //面心与左右两侧体心连线的夹角>100度
    for(i=0;i<nTFace;i++){
        IfOneOrder[i] = 1;
    }
} 

/************************************************************************
*  compute some additional info for geomety for efficiency              *
*  zhyb, 20200311                                                       *
************************************************************************/
void PolyGrid::AdditionalInfoForGeometry()
{
    
    CalculateVolumnAverage();  //Calculate volume average of cell
    
    CalNormalDistanceOfC2C();  //calculate normal distance of two cell
    
}


/************************************************************************
*  Calculate volume average of cell                                     *
*  used for volume reference value in venkatakrishnan's limiter          *
*  zhyb, 20190423                                                       *
************************************************************************/
RealGeom PolyGrid::CalculateVolumnAverage()
{
    IntType nTCell = this->GetNTCell();
    
    IntType i;
    
    RealGeom volume_average = 0.0;
    for(i=0;i<nTCell;i++) volume_average += vol[i];
#ifdef MPICH
    RealGeom vol_glb;
    IntType nTCell_glb;
    MPI_Allreduce(&volume_average, &vol_glb,    1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&nTCell,         &nTCell_glb, 1, MPIIntType, MPI_SUM, MPI_COMM_WORLD);
    volume_average = vol_glb/nTCell_glb;
#else
    volume_average /= nTCell;
#endif
 
    SetVolAvg(volume_average);
    return volume_average;  
}


/************************************************************************
*  calculate normal distance of two cell                                *
*  used for viscous spectral radii in LUSGS                             *
*  zhyb, 20200311                                                       *
************************************************************************/
RealGeom *PolyGrid::CalNormalDistanceOfC2C()
{
    IntType nTFace = GetNTFace();
    IntType *f2c   = Getf2c();
    RealGeom *xfn  = GetXfn();
    RealGeom *yfn  = GetYfn();
    RealGeom *zfn  = GetZfn();
    RealGeom *xcc  = GetXcc();
    RealGeom *ycc  = GetYcc();
    RealGeom *zcc  = GetZcc();
    
    IntType i, c1, c2;
    
    RealGeom *norm_dist_c2c = NULL;
    norm_dist_c2c = (RealGeom *)this->GetDataPtr(REAL_GEOM, nTFace, "norm_dist_c2c");
    if(!norm_dist_c2c){  
        mfmem::snew_array_1D(norm_dist_c2c, nTFace, dmrfl);
        UpdateDataPtr(norm_dist_c2c, REAL_GEOM, nTFace, "norm_dist_c2c");
    }
    
    for(i=0; i<nTFace; i++){
        c1 = f2c[i+i];
        c2 = f2c[i+i+1];
        
        norm_dist_c2c[i] = fabs(xfn[i]*(xcc[c2] - xcc[c1])
                          +     yfn[i]*(ycc[c2] - ycc[c1])
                          +     zfn[i]*(zcc[c2] - zcc[c1]));
    }
    
    return norm_dist_c2c;
}   

void PolyGrid::PartitionGrids( IntType *CellToZone, IntType n_zone )
{
    idx_t *xadj   = NULL;
    idx_t *adjncy = NULL;
    idx_t *adjwgt = NULL;
    mfmem::snew_array_1D(xadj,   nTCell+1,          dmrfl);
    mfmem::snew_array_1D(adjncy, 2*(nTFace-nBFace), dmrfl);
    mfmem::snew_array_1D(adjwgt, 2*(nTFace-nBFace), dmrfl);

    Getxadjadjncy(xadj, adjncy, adjwgt);

    SerialMetis(xadj, adjncy, adjwgt, n_zone, CellToZone);

    mfmem::sdel_array_1D(xadj);
    mfmem::sdel_array_1D(adjncy);
    mfmem::sdel_array_1D(adjwgt);
}

/************************************************************************
*                   get xadj, adjncy and adjwgt                         *
************************************************************************/
void PolyGrid::Getxadjadjncy(idx_t *xadj, idx_t *adjncy, idx_t *adjwgt)
{   
    IntType i, j, k;
    IntType *nCPC = CalnCPC(this); 
    IntType **c2c = CalC2C(this);

    xadj[0] = 0;
    k = 0;
    for(i=0; i<nTCell; i++){
        xadj[i+1] = xadj[i] + nCPC[i];

        for(j=0; j<nCPC[i]; j++)
            adjncy[k++] = c2c[i][j];
    }

    // Now get adjwgt
    if(level == 0){  //fine grid
        IntType c1, c2;
        RealGeom dx, dy, dz, d;
        RealGeom maxlen = -1.0;

        RealGeom *len = NULL;
        mfmem::snew_array_1D(len, k, dmrfl);

        k = 0;
        for(i=0; i<nTCell; i++){
            c1 = i;

            for(j=0; j<nCPC[i]; j++){
                c2 = c2c[i][j];

                dx = xcc[c2] - xcc[c1];
                dy = ycc[c2] - ycc[c1];
                dz = zcc[c2] - zcc[c1];
                d  = sqrt(dx*dx + dy*dy + dz*dz);

                len[k++] = d;
                maxlen = max(maxlen, d);
            }
        }
        maxlen += TINY;

        k = 0;
        for(i=0; i<nTCell; i++){
            for(j=0; j<nCPC[i]; j++){
                adjwgt[k] = (idx_t)(maxlen/len[k]);
                ++k;
            }
        }

        mfmem::sdel_array_1D(len);
    }else{  //coarse grid
        k = 0;
        for(i=0; i<nTCell; i++){
            for(j=0; j<nCPC[i]; j++){
                adjwgt[k++] = 1;
            }
        }
    }

    SetnCPC(NULL);
    Setc2c(NULL);
    c2c = NULL;
    nCPC = NULL;
}


/*************************************************************************
*                            serial metis                                *                           
**************************************************************************/
void PolyGrid::SerialMetis(idx_t *xadj, idx_t *adjncy, idx_t *adjwgt, IntType n_zone, IntType *CellToZone)
{
    IntType i, status;
    idx_t nvtxs, ncon, nparts, objval;
    idx_t *part;

    mflog::log.set_one_processor_out();
    mflog::log<<endl<<"Now beginning partition graph!"<<endl;

    nvtxs  = (idx_t)nTCell;
    ncon   = 1;
    nparts = (idx_t)n_zone;
    part   = NULL;
    mfmem::snew_array_1D(part, static_cast<size_t>(nvtxs), dmrfl);
    for(i=0; i<nvtxs; i++){
        part[i] = 0;
    }

    if(n_zone > 1){   //zhyb:n_zone==1时不划分网格，只输出！
        //choose Recursive bisection for nparts<64, k-way for nparts>=64, refer to EDGE
        if(n_zone < 64){
            mflog::log<<"Using Recursive Partitioning!"<<endl;
            status = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, adjwgt,
                &nparts, NULL, NULL, NULL, &objval, part);
        }else{
            mflog::log<<"Using k-way Partitioning!"<<endl;
            status = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, adjwgt,
                &nparts, NULL, NULL, NULL, &objval, part);
        }

        if(status != METIS_OK){
            mflog::log<<"Metis partition error! status code: "<<status<<endl;
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }

    for(i=0; i<nTCell; i++){
        CellToZone[i] = (IntType)part[i];
    }

    mflog::log<<"The interface number: "<<objval<<endl; 
    mflog::log<<"Serial Metis is OK!"<<endl;
    mfmem::sdel_array_1D(part);
}

/********************************************************************/
/*                        Add by DZ 2021-8-20                       */
/*                       Reference：AIAA94-0645                     */
/********************************************************************/

void PolyGrid::ReorderCellforLUSGS_1( )
{
    //IntType nTCell = grid->GetNTCell();
    //IntType nBFace = grid->GetNBFace();

    IntType i,c1,c2,type,count,status1,status2,status3,status4;

    IntType n = nBFace+nTCell;
    IntType BigLayer = 1000*nTCell;
    IntType NowLayer = 0;
    IntType MaxLayer = 0;
   
    IntType *Layer = NULL;
    IntType *LUSGSCellOrder = NULL;
    mfmem::snew_array_1D(Layer, n, dmrfl);
    for(i=0; i<n; i++) Layer[i] = BigLayer;
    mfmem::snew_array_1D( LUSGSCellOrder, nTCell, dmrfl );
    for(i=0; i<nTCell; i++) LUSGSCellOrder[i] = -1;
    //找任意一个物面单元作为起始层
    for(i=0;i<nBFace;i++){
        type = bcr[i]->GetType();
        if(type == WALL){
            c1 = f2c[i+i];
            Layer[c1] = NowLayer++;
            break;
        }
    }
    //如果没有物面，则找第一个边界单元作为起始层
    if(NowLayer == 0){
        c1 = f2c[0+0];
        Layer[c1] = NowLayer++;
    }
    
    mflog::log.set_all_processors_out();

    while(1){
        status1 = 0;
        count = 2*nBFace;
        for(i=nBFace; i<nTFace; i++){
            c1 = f2c[count++];
            c2 = f2c[count++];
            
            if(Layer[c1]<NowLayer && Layer[c2]<NowLayer){
                continue;
            }else if(Layer[c1]>NowLayer && Layer[c2]>NowLayer){
                continue;
            }else if(Layer[c1]==NowLayer || Layer[c2]==NowLayer){
                continue;
            }else if(Layer[c1]==BigLayer){
                Layer[c1] = NowLayer;
                status1++;
            }else if(Layer[c2]==BigLayer){
                Layer[c2] = NowLayer;
                status1++;
            }
        }
        
        //检查是否还有同层相邻的单元
        status2 = 0;
        for(i=nBFace;i<nTFace;i++){
            c1 = f2c[i+i];
            c2 = f2c[i+i+1];
            
            if(Layer[c1]==NowLayer && Layer[c1] == Layer[c2]){
                Layer[c2]++;
                status2++;
            }
        }
        //检查是否有遗漏的单元
        status3 = 0;
        if(status1==0 && status2==0){
            for(i=0;i<nTCell;i++){
                if(Layer[i] == BigLayer){
                    status3++;
                    break;
                }
            }
        }
        
        //如果没有，就跳出循环
        if(status1==0 && status2==0 && status3==0) break;
        
        //该块存在多区不联通，需要再次启动
        if(status1==0 && status2==0 &&status3!=0){
            status4 = 0;
            //找任意一个物面单元作为起始层
            for(i=0;i<nBFace;i++){
                type = bcr[i]->GetType();
                if(type == WALL){
                    c1 = f2c[i+i];
                    if(Layer[c1] == BigLayer){
                        Layer[c1] = 0;
                        status4++;
                        break;
                    }
                }
            }
            //如果没有物面，则找任意边界单元作为起始层
            if(status4 == 0){
                for(i=0;i<nBFace;i++){
                    c1 = f2c[i+i];
                    if(Layer[c1] == BigLayer){
                        Layer[c1] = 0;
                        status4++;
                        break;
                    }
                }
            }
            
            NowLayer = 0;
            //肯定有边界单元存在，否则报错退出
            if(status4 == 0){
#ifdef MPICH    
                mflog::log<<endl<<"Error! Zone "<<myZone<<" is a multi-block, but I do not find"
                    <<" the boundary cell for layer!"<<endl;
#else
                mflog::log<<endl<<"Error! This is a multi-block, but I do not find"
                    <<" the boundary cell for layer!"<<endl;    
#endif                  
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
        
        NowLayer++;
    }
    
    for(i=0;i<nTCell;i++){
        MaxLayer = MAX(MaxLayer,Layer[i]);
    }
    MaxLayer++;
    
    for(i=0;i<nTCell;i++){
        if(Layer[i]>MaxLayer){
#ifdef MPICH    
            mflog::log<<endl<<"Wrong! Zone"<<myZone<<"Layer["<<i<<"]="<<Layer[i]<<endl;
#else
            mflog::log<<endl<<"Wrong! Layer["<<i<<"]="<<Layer[i]<<endl;
#endif
        }
    }
    
    //虚拟单元排在最后一层
    for(i=0;i<nBFace;i++){
        c2 = f2c[i+i+1];
        Layer[c2] = MaxLayer;
    }
    
    IntType *cellsPerlayer = NULL;
    mfmem::snew_array_1D(cellsPerlayer, nTCell, dmrfl);
    cellsPerlayer[0] = MaxLayer;
    cellsPerlayer[1] = 0;

    IntType temp;
    //迭代顺序从低层到高层
    count = 0;
    for(NowLayer=0; NowLayer<MaxLayer; NowLayer++){
        temp = 0;
        for(i=0; i<nTCell; i++){
            if(Layer[i]==NowLayer){
                LUSGSCellOrder[count++] = i;
                temp++;
            }
        }
        cellsPerlayer[NowLayer+2] = count;
    }
    this->UpdateDataPtr(cellsPerlayer, INT, nTCell, "LUSGScellsPerlayer");

    for(i=0;i<nTCell;i++){
        Layer[LUSGSCellOrder[i]] = i;
    }
    for(i=nTCell;i<n;i++){
        Layer[i] = i;
    }    
    printf("LUSGS reordered by hyperplane approach 1!\n");
    this->UpdateDataPtr(LUSGSCellOrder, INT, nTCell, "LUSGSCellOrder");
    this->UpdateDataPtr(Layer, INT, n, "LUSGSLayer");
}


/************************************************************************\
  网格重排序
  参考文献：航空计算技术35卷第3期，基于非结构网格流场计算的网格重排序，方法1
\************************************************************************/
void PolyGrid::ReorderCellforLUSGS_2( )
{
    //IntType nTCell = grid->GetNTCell();
    //IntType nBFace = grid->GetNBFace();

    IntType i,c1,c2,type,count,status1,status2,status3;
    IntType n = nBFace+nTCell;
    IntType BigLayer=1000*nTCell, NowLayer=0, LastLayer, MaxLayer=0;
    IntType *Layer = NULL;
    IntType *LUSGSCellOrder = NULL;
    mfmem::snew_array_1D(Layer, n,dmrfl);
    for(i=0; i<n; i++) Layer[i] = BigLayer;
    mfmem::snew_array_1D(LUSGSCellOrder, nTCell,dmrfl);
    for(i=0; i<nTCell; i++) LUSGSCellOrder[i] = -1;
    //找第一个物面单元作为起始层
    for(i=0;i<nBFace;i++){
        type = bcr[i]->GetType();
        if(type == WALL){
            c1 = f2c[i+i];
            Layer[c1] = NowLayer++;
            break;
        }
    }
    //如果没有物面，则找第一个非内部边界单元作为起始层
    if(NowLayer == 0){
        for(i=0;i<nBFace;i++){
            type = bcr[i]->GetType();
            if(type != INTERFACE){
                c1 = f2c[i+i];
                Layer[c1] = NowLayer++;
                break;
            }
        }
    }
    //如果全是内部边界，则找第一个边界单元作为起始层
    if(NowLayer == 0){
        c1 = f2c[0+0];
        Layer[c1] = NowLayer++;
    }

    mflog::log.set_all_processors_out();

    while(1){
        LastLayer = NowLayer-1;
        status1 = 0;
        count = 2*nBFace;
        for(i=nBFace; i<nTFace; i++){
            c1 = f2c[count++];
            c2 = f2c[count++];
            
            if(Layer[c1]==LastLayer && Layer[c2]==BigLayer){
                Layer[c2] = NowLayer;
                status1++;
            }else if(Layer[c2]==LastLayer && Layer[c1]==BigLayer){
                Layer[c1] = NowLayer;
                status1++;
            }
        }
        
        //检查是否有遗漏的单元
        status2 = 0;
        if(status1==0){
            for(i=0;i<nTCell;i++){
                if(Layer[i] == BigLayer){
                    status2++;
                    break;
                }
            }
        }
        
        //如果没有，就跳出循环
        if(status1==0 && status2==0) break;
        
        //该块存在多区不联通，需要再次启动
        if(status1==0 && status2!=0){
            status3 = 0;
            //找第一个物面单元作为起始层
            for(i=0;i<nBFace;i++){
                type = bcr[i]->GetType();
                if(type == WALL){
                    c1 = f2c[i+i];
                    if(Layer[c1] == BigLayer){
                        Layer[c1] = 0;
                        status3++;
                        break;
                    }
                }
            }
            //如果没有物面，则找任意非内部边界单元作为起始层
            if(status3 == 0){
                for(i=0;i<nBFace;i++){
                    type = bcr[i]->GetType();
                    if(type != INTERFACE){
                        c1 = f2c[i+i];
                        if(Layer[c1] == BigLayer){
                            Layer[c1] = 0;
                            status3++;
                            break;
                        }
                    }
                }
            }
            //如果全是内部边界，则找第一个边界单元作为起始层
            if(status3 == 0){
                for(i=0;i<nBFace;i++){
                    c1 = f2c[i+i];
                    if(Layer[c1] == BigLayer){
                        Layer[c1] = 0;
                        status3++;
                        break;
                    }
                }
            }
            
            NowLayer = 0;
            //肯定有边界单元存在，否则报错退出
            if(status3 == 0){
#ifdef MPICH    
                mflog::log<<endl<<"Error! Zone "<<myZone<<" is a multi-block, but I do not find"
                    <<" the boundary cell for layer!"<<endl;
#else
                mflog::log<<endl<<"Error! This is a multi-block, but I do not find"
                    <<" the boundary cell for layer!"<<endl;    
#endif                  
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
        
        NowLayer++;
    }
    
    for(i=0;i<nTCell;i++){
        MaxLayer = MAX(MaxLayer,Layer[i]);
    }
    MaxLayer++;

    //迭代顺序从低层到高层
    count = 0;
    for(NowLayer=0; NowLayer<MaxLayer; NowLayer++){
        for(i=0; i<nTCell; i++){
            if(Layer[i]==NowLayer){
                LUSGSCellOrder[count++] = i;
            }
        }
    }
    for(i=0;i<nTCell;i++){
        Layer[LUSGSCellOrder[i]] = i;
    }
    for(i=nTCell;i<n;i++){
        Layer[i] = i;
    } 
    printf("LUSGS reordered by hyperplane approach 2!\n");
    this->UpdateDataPtr(LUSGSCellOrder, INT, nTCell, "LUSGSCellOrder");
    this->UpdateDataPtr(Layer, INT, n, "LUSGSLayer");
}


/************************************************************************\
  网格重排序
  参考文献：航空计算技术35卷第3期，基于非结构网格流场计算的网格重排序，方法2
                                Reordering of 3-D Hybrid Grids for Vectorized LU-SGS NS Computations
\************************************************************************/
void PolyGrid::ReorderCellforLUSGS_3( )
{
    IntType i,j,c1,c2,type,count,status1,status2,status3;
    IntType n = nBFace+nTCell;
    IntType BigLayer=1000*nTCell, NowLayer=0, LastLayer, MaxLayer=0;
    
    IntType *Layer = NULL;
    IntType *LUSGSCellOrder = NULL;
    mfmem::snew_array_1D(Layer, n, dmrfl);
    for(i=0; i<n; i++) Layer[i] = BigLayer;
    mfmem::snew_array_1D(LUSGSCellOrder, nTCell, dmrfl);
    for(i=0; i<nTCell; i++) LUSGSCellOrder[i] = -1;
    //找任意一个物面单元作为起始层
    for(i=0;i<nBFace;i++){
        type = bcr[i]->GetType();
        if(type == WALL){
            c1 = f2c[i+i];
            Layer[c1] = NowLayer++;
            break;
        }
    }
    //如果没有物面，则找第一个边界单元作为起始层
    if(NowLayer == 0){
        c1 = f2c[0+0];
        Layer[c1] = NowLayer++;
    }

    mflog::log.set_all_processors_out();
    
    while(1){
        LastLayer = NowLayer-1;
        status1 = 0;
        count = 2*nBFace;
        for(i=nBFace; i<nTFace; i++){
            c1 = f2c[count++];
            c2 = f2c[count++];
            
            if(Layer[c1]==LastLayer && Layer[c2]==BigLayer){
                Layer[c2] = NowLayer;
                status1++;
            }else if(Layer[c2]==LastLayer && Layer[c1]==BigLayer){
                Layer[c1] = NowLayer;
                status1++;
            }
        }
        
        //检查是否有遗漏的单元
        status2 = 0;
        if(status1==0){
            for(i=0;i<nTCell;i++){
                if(Layer[i] == BigLayer){
                    status2++;
                    break;
                }
            }
        }
        
        //如果没有，就跳出循环
        if(status1==0 && status2==0) break;
        
        //该块存在多区不联通，需要再次启动
        if(status1==0 && status2!=0){
            status3 = 0;
            //找任意一个物面单元作为起始层
            for(i=0;i<nBFace;i++){
                type = bcr[i]->GetType();
                if(type == WALL){
                    c1 = f2c[i+i];
                    if(Layer[c1] == BigLayer){
                        Layer[c1] = 0;
                        status3++;
                        break;
                    }
                }
            }
            //如果没有物面，则找任意边界单元作为起始层
            if(status3 == 0){
                for(i=0;i<nBFace;i++){
                    c1 = f2c[i+i];
                    if(Layer[c1] == BigLayer){
                        Layer[c1] = 0;
                        status3++;
                        break;
                    }
                }
            }
            
            NowLayer = 0;
            //肯定有边界单元存在，否则报错退出
            if(status3 == 0){
#ifdef MPICH    
                mflog::log<<endl<<"Error! Zone "<<myZone<<" is a multi-block, but I do not find"
                    <<" the boundary cell for layer!"<<endl;
#else
                mflog::log<<endl<<"Error! This is a multi-block, but I do not find"
                    <<" the boundary cell for layer!"<<endl;    
#endif                  
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
        
        NowLayer++;
    }
    for(i=0;i<nTCell;i++){
        MaxLayer = MAX(MaxLayer,Layer[i]);
    }
    MaxLayer++;
    
    //排列子层
    IntType *SubLayerNum = NULL;
    IntType *SubLayer = NULL;
    mfmem::snew_array_1D(SubLayerNum, MaxLayer,dmrfl);
    mfmem::snew_array_1D(SubLayer, nTCell,dmrfl);
    for(i=0;i<nTCell;i++){
        SubLayer[i] = 0;
    }
        
    for(NowLayer=0;NowLayer<MaxLayer;NowLayer++){
        SubLayerNum[NowLayer] = 1;
        while(1){
            status1 = 0;
            for(i=nBFace;i<nTFace;i++){
                c1 = f2c[i+i];
                c2 = f2c[i+i+1];
            
                if(Layer[c1]==NowLayer && Layer[c1]==Layer[c2]){
                    if(SubLayer[c1]==SubLayer[c2] && SubLayer[c1]==SubLayerNum[NowLayer]-1){
                        SubLayer[c2] = SubLayerNum[NowLayer];
                        status1++;
                    }
                }
            }
            if(status1 == 0) break;
            SubLayerNum[NowLayer]++;
        }
    }
    
    count = 0;
    for(NowLayer=0;NowLayer<MaxLayer;NowLayer++){
        count += SubLayerNum[NowLayer];
    }
    count--;
    for(NowLayer=MaxLayer-1;NowLayer>=0;NowLayer--){
        for(i=SubLayerNum[NowLayer]-1;i>=0;i--){
            for(j=0;j<nTCell;j++){
                if(Layer[j] == NowLayer && SubLayer[j]==i){
                    Layer[j] = count;
                }
            }
            count--;
        }
    }
    mfmem::sdel_array_1D(SubLayerNum);  
    mfmem::sdel_array_1D(SubLayer);
    for(i=0;i<nTCell;i++){
        MaxLayer = MAX(MaxLayer,Layer[i]);
    }
    MaxLayer++;
    
    //虚拟单元排在最后一层
    for(i=0;i<nBFace;i++){
        c2 = f2c[i+i+1];
        Layer[c2] = MaxLayer;
    }
    
    IntType *cellsPerlayer = NULL;
    mfmem::snew_array_1D(cellsPerlayer, nTCell, dmrfl);
    cellsPerlayer[0] = MaxLayer;
    cellsPerlayer[1] = 0;

    IntType temp = 0;
    //迭代顺序从低层到高层
    count = 0;
    for(NowLayer=0; NowLayer<MaxLayer; NowLayer++){
        for(i=0; i<nTCell; i++){
            if(Layer[i]==NowLayer){
                LUSGSCellOrder[count++] = i;
            }
        }
        cellsPerlayer[NowLayer+2] = count;
    }
    for(i=0;i<nTCell;i++){
        Layer[LUSGSCellOrder[i]] = i;
    }
    for(i=nTCell;i<n;i++){
        Layer[i] = i;
    }    

    //computing the details of divided layers
    
    int mincell = nTCell; 
    int maxcell = -1;
    double ave = double(nTCell) / double(cellsPerlayer[0]);
    double std = 0;
	for(IntType laynum = cellsPerlayer[0] - 1; laynum >= 0; laynum--){
        IntType cells = cellsPerlayer[laynum + 2] - cellsPerlayer[laynum + 1];
        mincell = MIN(cells, mincell);
        maxcell = MAX(cells, maxcell);
        std += (ave - cells)*(ave - cells);
    }
    std = sqrt(std/cellsPerlayer[0]);
#ifdef MPICH
	IntType mpirank = 0; //when mpirank = 0, mpi was off.       
    MPI_Comm_rank(MPI_COMM_WORLD, & mpirank);
    printf("mpirank:%d layers:%d  max:%d  min:%d  ave:%lf  std:%lf\n",mpirank,cellsPerlayer[0],maxcell,mincell,\
        ave, std);
#else
    printf("layers:%d  max:%d  min:%d  ave:%lf  std:%lf\n",cellsPerlayer[0],maxcell,mincell,\
        ave, std);
#endif
    printf("LUSGS reordered by hyperplane approach 3!\n");
    this->UpdateDataPtr(cellsPerlayer, INT, nTCell, "LUSGScellsPerlayer");
    this->UpdateDataPtr(LUSGSCellOrder, INT, nTCell, "LUSGSCellOrder");
    this->UpdateDataPtr(Layer, INT, n, "LUSGSLayer");
}

/************************************************************************\
  网格重排序
  参考文献： Reordering of 3-D Hybrid Grids for Vectorized LU-SGS NS Computations
  daizhe注：改进了子层排序算法以提升子层均衡性，提高OpenMP的算法均衡性
\************************************************************************/

/*****************LUSGS face coloring cited by computers & Fliuds 88(2013)496-509***************/
// input:coloring mode 0 or 1; output array:cellsPerlayer,LUSGSCellOrder,Layer
void PolyGrid::LUSGSGridColor(int colort) {
    IntType i, nf, j, maxcolor, colored, tempcolor, tempcolorNum, c, c1, c2;
    IntType count;
    IntType n = nTCell + nBFace;
    IntType* LUSGSCellOrder = NULL;
    IntType* Layer = NULL;
    IntType* cellsPerlayer = NULL;
    bool* color_tmp = NULL;
    mfmem::snew_array_1D(color_tmp, nTCell, dmrfl);
    mfmem::snew_array_1D(Layer, n, dmrfl);                     //the array to judge the upper or lower cell in LUSGS
    mfmem::snew_array_1D(cellsPerlayer, nTCell, dmrfl);        //the number of colors and the indce number in each color
    mfmem::snew_array_1D(LUSGSCellOrder, nTCell, dmrfl);       // the coloring information array

    for (i = 0; i < n; i++) {
        Layer[i] = -1;
    }
    maxcolor = 1;
    tempcolorNum = 1;
    tempcolor = 0;

    nCPC = CalnCPC(this);
    c2c = CalC2C(this);
    
    //create nCPC & c2c
    set<IntType> c_ng;
    for (i = 0; i < nTCell; i++) { // Get grid's maxcolor, usually the degree of the grid, N or N+1.
        c_ng.clear();
        for (j = 0; j < nCPC[i]; j++) {
            c = c2c[i][j];
            if(c < nTCell)
                c_ng.insert(c);
        }
        maxcolor = max(maxcolor, (IntType)c_ng.size());
    }
    for (i = 0; i < nTCell; i++) {
        color_tmp[i] = true;
    }
    maxcolor++;
    //maxcolor = 10;
    cellsPerlayer[0] = maxcolor;
    for (i = 1; i < nTCell; i++) {
        cellsPerlayer[i] = 0;
    }
    
    set<IntType> cell_boundry;
    for (i = 0; i < nBFace; i++) {
        cell_boundry.insert(f2c[i + i]);
    }
    set<IntType>::iterator iter_index = cell_boundry.begin();
    // 为了避免因面重排序对体着色顺序的影响，统一为体循环
    for (; iter_index != cell_boundry.end(); ++iter_index) {    //对包含壁面和边界面体的着色
        c1 = *iter_index;
        if (Layer[c1] >= 0) {
            continue;
        }
        for (j = 0; j < maxcolor; j++) {
            if (!color_tmp[j])
                color_tmp[j] = true;
        }
        for (j = 0; j < nCPC[c1]; j++) {
            c = c2c[c1][j];
            if ((c < nTCell) && (Layer[c] >= 0) && color_tmp[Layer[c]]) {
                color_tmp[Layer[c]] = false;
            }
        }
        for (j = 0; j < maxcolor; j++) {
            if (color_tmp[j])
                break;
        }
        if (j >= maxcolor) {
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            cellsPerlayer[0] = ++maxcolor;
        }
        tempcolorNum = nTCell;
        if (colort == 0) {    //imbalanced greedy algorithm for face coloring
            for (j = 0; j < maxcolor; j++) {
                if (color_tmp[j]) {   //with satisfied color number 
                    tempcolor = j;
                    break;
                }
            }
        }
        else {     //balanced algorithm for face coloring
            for (j = 0; j < maxcolor; j++) {
                if (color_tmp[j]) {             //with satisfied color number 
                    if (cellsPerlayer[j + 2] < tempcolorNum) {
                        tempcolorNum = cellsPerlayer[j + 2];
                        tempcolor = j;
                    }
                }
            }
        }
        cellsPerlayer[tempcolor + 2]++;
        Layer[c1] = tempcolor;
    }
    cell_boundry.clear();
    /* for (i = 0; i < nBFace; i++) {    //对包含壁面和边界面体的着色
        c1 = f2c[i + i];
        if (Layer[c1] >= 0) {
            continue;
        }
        for (j = 0; j < maxcolor; j++) {
            if (!color_tmp[j])
                color_tmp[j] = true;
        }
        for (j = 0; j < nCPC[c1]; j++) {
            c = c2c[c1][j];
            if ((c < nTCell) && (Layer[c] >= 0) && color_tmp[Layer[c]]) {
                color_tmp[Layer[c]] = false;
            }
        }
        for (j = 0; j < maxcolor; j++) {
            if (color_tmp[j])
                break;
        }
        if (j >= maxcolor) {
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            cellsPerlayer[0] = ++maxcolor;
        }
        tempcolorNum = nTCell;
        if (colort == 0) {    //imbalanced greedy algorithm for face coloring
            for (j = 0; j < maxcolor; j++) {
                if (color_tmp[j]) {   //with satisfied color number 
                    tempcolor = j;
                    break;
                }
            }
        }
        else {     //balanced algorithm for face coloring
            for (j = 0; j < maxcolor; j++) {
                if (color_tmp[j]) {             //with satisfied color number 
                    if (cellsPerlayer[j + 2] < tempcolorNum) {
                        tempcolorNum = cellsPerlayer[j + 2];
                        tempcolor = j;
                    }
                }
            }
        }
        cellsPerlayer[tempcolor + 2]++;
        Layer[c1] = tempcolor;
    } */
    for (c = 0; c < nTCell; c++) {    //对内部体进行着色
        if (Layer[c] >= 0) {
            continue;
        }
        for (j = 0; j < maxcolor; j++) {
            if (!color_tmp[j])
                color_tmp[j] = true;
        }
        for (j = 0; j < nCPC[c]; j++) {
            c1 = c2c[c][j];
            if ((c1 < nTCell) && (Layer[c1] >= 0) && color_tmp[Layer[c1]]) {
                color_tmp[Layer[c1]] = false;
            }
        }
        for (j = 0; j < maxcolor; j++) {
            if (color_tmp[j])
                break;
        }
        if (j >= maxcolor) {
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            cellsPerlayer[0] = ++maxcolor;
        }
        tempcolorNum = nTCell;
        if (colort == 0) {    //imbalanced greedy algorithm for face coloring
            for (j = 0; j < maxcolor; j++) {
                if (color_tmp[j]) {   //with satisfied color number 
                    tempcolor = j;
                    break;
                }
            }
        }
        else {   //balanced algorithm for face coloring
            for (j = 0; j < maxcolor; j++) {
                if (color_tmp[j]) {             //with satisfied color number 
                    if (cellsPerlayer[j + 2] < tempcolorNum) {
                        tempcolorNum = cellsPerlayer[j + 2];
                        tempcolor = j;
                    }
                }
            }
        }
        cellsPerlayer[tempcolor + 2]++;
        Layer[c] = tempcolor;
    }

    //虚拟单元排在最后一层
    maxcolor++;
    for (i = 0; i < nBFace; i++) {
        c2 = f2c[i + i + 1];
        Layer[c2] = maxcolor;
    }
    count = 0;

    for (i = 0; i < maxcolor; i++) {
        for (j = 0; j < nTCell; j++) {
            if (Layer[j] == i) {
                LUSGSCellOrder[count++] = j;// new to old
            }
        }
        cellsPerlayer[i + 2] += cellsPerlayer[i + 1];
    }

    for (i = 0; i < nTCell; i++) {
        Layer[LUSGSCellOrder[i]] = i;
    }
    for (i = nTCell; i < n; i++) {
        Layer[i] = i;
    }
    
    mfmem::sdel_array_1D(color_tmp);
    
    //computing the details of divided layers
    
    int mincell = nTCell; 
    int maxcell = -1;
    double ave = double(nTCell) / double(cellsPerlayer[0]);
    double std = 0;
	for(IntType laynum = cellsPerlayer[0] - 1; laynum >= 0; laynum--){
        IntType cells = cellsPerlayer[laynum + 2] - cellsPerlayer[laynum + 1];
        mincell = MIN(cells, mincell);
        maxcell = MAX(cells, maxcell);
        std += (ave - cells)*(ave - cells);
        //printf("%d\n",cells);
    }
    std = sqrt(std/cellsPerlayer[0]);
#ifdef MPICH
	IntType mpirank = 0; //when mpirank = 0, mpi was off.       
    MPI_Comm_rank(MPI_COMM_WORLD, & mpirank);
    printf("mpirank:%d layers:%d  max:%d  min:%d  ave:%lf  std:%lf\n",mpirank,cellsPerlayer[0],maxcell,mincell,\
        ave, std);
#else
    printf("layers:%d  max:%d  min:%d  ave:%lf  std:%lf\n",cellsPerlayer[0],maxcell,mincell,\
        ave, std);
#endif

    printf("LUSGS reordered by cell color approach 4!\n");
    this->UpdateDataPtr(cellsPerlayer, INT, nTCell, "LUSGScellsPerlayer");
    this->UpdateDataPtr(LUSGSCellOrder, INT, nTCell, "LUSGSCellOrder");
    this->UpdateDataPtr(Layer, INT, n, "LUSGSLayer");
}

/********************************************************************/
/*                        Add by DZ 2021-8-20                       */
/********************************************************************/
void PolyGrid::SparseRecurrence_globalSyn( ){
    IntType c1, c2;
    IntType n = nTCell + nBFace;
    IntType TotalLayer = 0;
    if(nCPC == 0) nCPC = CalnCPC(this);
    if(c2c  == 0) c2c = CalC2C(this);

    IntType *cellsPerlayer = NULL;
    IntType *Layer = NULL; 
    IntType *LUSGSCellOrder = NULL; 
    mfmem::snew_array_1D(cellsPerlayer, nTCell, dmrfl);
    mfmem::snew_array_1D(LUSGSCellOrder,nTCell, dmrfl);
    mfmem::snew_array_1D(Layer,   n, dmrfl);
    for(IntType i=0; i<nTCell; i++){
        Layer[i] = -1;
    }

    Layer[0] = 0;
    //Layer is the number of plane for every cell unit;
    //TotalLayer is the total planes for the whole grid mesh
    for(IntType i=1; i<nTCell; i++){
        IntType lastLayer = -1;
        for(IntType j=0; j<nCPC[i]; j++){
            c1 = c2c[i][j];
            if( c1 < nTCell ){ //&& (Layer[ c1 ] >= 0)
                if( Layer[c1] == -1 ) {
                    continue;
                }
                else{
                    lastLayer = max(lastLayer, Layer[c1]);
                }
            }
        }

        if(lastLayer == -1){
            //Layer[i] = Layer[i-1];
			Layer[i]=0;
        }
        else{
            TotalLayer = max( TotalLayer, lastLayer+1 );
            Layer[i] = lastLayer+1;
        }
    }
    TotalLayer++;
	
	// cout Layer Info:
	IntType mpirank_ = 0; //when mpirank = 0, mpi was off. 	
#ifdef MPICH        
    MPI_Comm_rank(MPI_COMM_WORLD, & mpirank_);	
#endif

	std::cout << "Total level: " << TotalLayer << std::endl; 

    cellsPerlayer[0] = TotalLayer;
    cellsPerlayer[1] = 0;
    IntType count = 0;
    for(IntType i=0; i<TotalLayer; i++){
        for(IntType j=0; j<nTCell; j++){
            if(Layer[j] == i){
                LUSGSCellOrder[count] = j;
                count++;
            }
        }
        cellsPerlayer[i+2] = count;
    }

    for(IntType i=0;i<nBFace;i++){
        c2 = f2c[i+i+1];
        Layer[c2] = TotalLayer;
    }
    for(IntType i=0;i<nTCell;i++){
        Layer[LUSGSCellOrder[i]] = i;
    }
    for(IntType i=nTCell;i<n;i++){
        Layer[i] = i;
    }

#ifdef MPICH
	IntType mpirank = 0; 
    MPI_Comm_rank(MPI_COMM_WORLD, & mpirank);

    if(count != nTCell){
        printf("MPIrank:%d Error algorithm in SparseRecurrence_globalSyn function!\n",mpirank);
        exit(-1);
    }
#endif

    //computing the details of divided layers
    /*int mincell = nTCell; 
    int maxcell = -1;
    double ave = nTCell / cellsPerlayer[0];
    double std = 0;
	for(IntType laynum = cellsPerlayer[0] - 1; laynum >= 0; laynum--){
        IntType cells = cellsPerlayer[laynum + 2] - cellsPerlayer[laynum + 1];
        mincell = MIN(cells, mincell);
        maxcell = MAX(cells, maxcell);
        std += (ave - cells)*(ave - cells);
    }
    std = sqrt(std);
    printf("layers:%d  max:%d  min:%d  ave:%lf  std:%lf\n",cellsPerlayer[0],maxcell,mincell,\
        ave, std);*/
    
    printf("LUSGS reordered by cell color approach 5!\n");
    this->UpdateDataPtr(LUSGSCellOrder, INT, nTCell, "LUSGSCellOrder");
    this->UpdateDataPtr(cellsPerlayer,  INT, nTCell, "LUSGScellsPerlayer");
    this->UpdateDataPtr(Layer, INT, n, "LUSGSLayer");
}

void PolyGrid::SparseRecurrence_NonGlobalSyn( ){
    IntType c1, c2;
    IntType n = nTCell + nBFace;
    IntType TotalLayer = 0;
    if(nCPC == 0) nCPC = CalnCPC(this);
    if(c2c  == 0) c2c = CalC2C(this);

    IntType *cellsPerlayer = NULL;
    IntType *Layer = NULL; 
    IntType *LUSGSCellOrder = NULL; 
    mfmem::snew_array_1D(cellsPerlayer, nTCell, dmrfl);
    mfmem::snew_array_1D(LUSGSCellOrder,nTCell, dmrfl);
    mfmem::snew_array_1D(Layer,   n, dmrfl);
    for(IntType i=0; i<nTCell; i++){
        Layer[i] = -1;
    }

    Layer[0] = 0;
    for(IntType i=1; i<nTCell; i++){
        IntType lastLayer = -1;
        for(IntType j=0; j<nCPC[i]; j++){
            c1 = c2c[i][j];
            if( c1 < nTCell ){ 
                if( Layer[c1] == -1 ) {
                    continue;
                }
                else{
                    lastLayer = max(lastLayer, Layer[c1]);
                }
            }
        }

        if(lastLayer == -1){
            Layer[i] = Layer[i-1];
        }
        else{
            TotalLayer = max( TotalLayer, lastLayer+1 );
            Layer[i] = lastLayer+1;
        }
    }
    TotalLayer++;


}

IntType *PolyGrid::CalGhost2Global(IntType Bstart)
{
    if(ghost2global) return ghost2global;
#ifdef MPICH
    mfmem::snew_array_1D(ghost2global, nIFace, dmrfl);
    IntType **bqs = NULL;
    IntType **bqr = NULL;

    MPI_Request *req_send = NULL;
    MPI_Request *req_recv = NULL;
    MPI_Status *status_array = NULL ;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
    mfmem::snew_array_2D(bqr,nNeighbor,nZIFace,dmrfl,false);
    mfmem::snew_array_2D(bqs,nNeighbor,nZIFace,dmrfl,false);

    for(IntType i = 0; i<nNeighbor; i++){
        for(IntType j=0; j<nZIFace[i]; j++){
            //Global_index = Local_index + Block_start
            //bqs[i][j] = bCNo[i][j] + Bstart;
            bqs[i][j] = bCNo[i][j] + Bstart;
        }
    }
    for(IntType np=1; np<=numprocs; np++){
        if(myZone==np)
        {   
            //       send   to other 	
            for(IntType g=0; g<nNeighbor; g++) {
                IntType nbZone = nb[g];
                MPI_Isend(bqs[g], nZIFace[g], MPI_INT, nbZone, level, MPI_COMM_WORLD,&req_send[g]);
            }
        }
        else{   
            //       receive   from np 	
            for(IntType g=0; g<nNeighbor; g++) {
                IntType nbZone = nb[g];
                if(nbZone ==(np-1)){
                    MPI_Irecv(bqr[g], nZIFace[g], MPI_INT, nbZone, level, MPI_COMM_WORLD, &req_recv[g]);
                }
            } 
        }
    }   // np
    MPI_Waitall(nNeighbor,req_recv,status_array);
    IntType ifStart = nBFace - nIFace;
    for(IntType g=0; g<nNeighbor; g++)
    {
        for(IntType i=0; i<nZIFace[g];i++)
        {
            IntType idx = bFNo[g][i] - ifStart;
            ghost2global[idx] = bqr[g][i];
        }
    }
    MPI_Waitall(nNeighbor,req_send,status_array);
    mfmem::sdel_array_2D(bqr,nNeighbor,false);
    mfmem::sdel_array_2D(bqs,nNeighbor,false);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(status_array);
    return ghost2global;
#endif
}

void PolyGrid::printcsrMatrix(IntType *row_ptr, IntType *col_ind, IntType nnz, IntType n, IntType nVar ){

    FILE* file;
	file = fopen("Matrix.bin", "w");
	fwrite(&n, sizeof(IntType), 1, file);
	fwrite(&nVar,   sizeof(IntType), 1, file);
	fwrite(&nnz,    sizeof(IntType), 1, file);
	fwrite(row_ptr, sizeof(IntType), n + 1, file);
	fwrite(col_ind,     sizeof(IntType), nnz, file);
	//fwrite(matrix, sizeof(double), nnz*nVar*nVar, file);

	fclose(file);
}

//export the divided layers of LU-SGS parallelizations
void PolyGrid::exportLayers(){
    IntType c1, c2;
    IntType n = nTCell + nBFace;

    IntType *luorder = (IntType *)this->GetDataPtr(INT, nTCell, "LUSGSCellOrder");
    IntType *layer = (IntType *)this->GetDataPtr(INT, n, "LUSGSLayer");
    IntType *cellsPerlayer = (IntType *)this->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer");

    IntType laynum;
    FILE* file;
	file = fopen("Layers.bin", "w");
	fwrite(&cellsPerlayer[0], sizeof(IntType), 1, file);
    for(laynum=0; laynum<cellsPerlayer[0]; laynum++ ){
	    fwrite(&cellsPerlayer[laynum+1], sizeof(IntType), 1, file);
    }
    fwrite(&cellsPerlayer[laynum+1], sizeof(IntType), 1, file);

    IntType start, end, ilu;
    for(laynum=0; laynum<cellsPerlayer[0]; laynum++ ){
        start = cellsPerlayer[laynum+1];
        end   = cellsPerlayer[laynum+2];
        for(ilu=start; ilu<end; ilu++){
            fwrite(&luorder[ilu], sizeof(IntType), 1, file);
        }
    }

	fclose(file);
}

void PolyGrid::ComputeBSRindex( ){

    printf("BSR initialized with self counting\n");
    IntType Bstart = 0;
    IntType *nFPC = CalnFPC(this);
    IntType **C2F = CalC2F(this);
    IntType ifStart = nBFace - nIFace;
    IntType gnnz = 0;
    IntType count = 0;
   
    IntType * ghost2global = this->CalGhost2Global(Bstart);

    for(IntType iCell = 0; iCell<nTCell; iCell++){
        IntType cell = iCell;
    
        for(IntType iFace=0; iFace<nFPC[cell]; iFace++){
            IntType face  = C2F[cell][iFace];
            if(face < ifStart)
            {
                continue;
            }
            gnnz ++;
        }
        gnnz ++;
    }

    IntType *row_ptr = new IntType[nTCell+1];
    IntType *col_ind = new IntType[gnnz];
    IntType *col_ind_origin = new IntType[gnnz];
    for(int i=0;i<nTCell+1;i++) {row_ptr[i] = 0;}

    for(int iCell = 0;iCell<nTCell;iCell++){
        IntType cell = iCell;
        IntType flag = true;
    
        for(IntType iFace=0; iFace<nFPC[cell]; iFace++){
            IntType face  = C2F[cell][iFace];
            IntType c2    = f2c[face+face]+f2c[face+face+1]-cell;
  
            IntType Brow = Bstart+cell;
            IntType Bcol = 0;
            if(face >= nBFace){
                Bcol = Bstart+c2;
            }
            else if(face < ifStart)
            {
                continue;
            }
            else{
                Bcol = ghost2global[face-ifStart];
            }

            IntType Irow = Brow;
            IntType Icol = Bcol;

            row_ptr[Irow+1]++ ;
            col_ind_origin[count++] = Icol;

        }
        IntType Brow = Bstart+cell;
        IntType Bcol = Brow;
        IntType Irow = Brow;
        IntType Icol = Bcol;

        row_ptr[Irow+1]++ ;
        col_ind_origin[count++] = Icol;
    }

    for(int i=1; i<nTCell+1; i++){
        row_ptr[i] += row_ptr[i-1];
    }

    std::vector<IntType> vec = {0,0,0,0,0,0,0,0,0,0};
    for(int i=0; i<nTCell; i++){
        int length = row_ptr[i+1] - row_ptr[i];
        for(int j=row_ptr[i]; j<row_ptr[i+1]; j++){
            vec[j - row_ptr[i]] = col_ind_origin[ j ] ;
        }
        vector<IntType>::iterator it = vec.begin();
        std::sort( vec.begin(), (it+length) );

        for(int j=row_ptr[i]; j<row_ptr[i+1]; j++){
            col_ind[ j ] = *( it + j - row_ptr[i] );
        }      
    }
    printf("n:%d  nnz:%d\n",nTCell,gnnz);
    this->printcsrMatrix( row_ptr, col_ind, gnnz, nTCell, 5 );
    delete[] row_ptr;
    delete[] col_ind;
    delete[] col_ind_origin;
}

#ifdef DC0
/************************************************************************
Purpose:  use DC Tree index to do face reorder
************************************************************************/

void uTaskTree_FaceReordering(PolyGrid *grid)
{	
	IntType nBFace = grid->GetNBFace();
	IntType nTFace = grid->GetNTFace();
	IntType nIFace = grid->GetNIFace();
	IntType pfacenum = nBFace - nIFace;
	IntType *f2c = grid->Getf2c();
	IntType *f2n = grid->Getf2n();
	IntType *nNPF = grid->GetnNPF();
    IntType *nbz = grid->GetnbZ();
    IntType *nbf = grid->GetnbBF();

	IntType *f2c_backup = NULL;
	IntType *f2n_backup = NULL;
	IntType *nNPF_backup = NULL;
    
	mfmem::snew_array_1D(f2c_backup, nTFace*2, dmrfl);
	mfmem::snew_array_1D(nNPF_backup, nTFace, dmrfl);
	
	IntType n = 0;
	for(IntType i = 0; i < nTFace; ++i)
	{
		f2c_backup[i+i] = f2c[i+i];
		f2c_backup[i+i+1] = f2c[i+i+1];
		nNPF_backup[i] = nNPF[i];
		n+=nNPF_backup[i];
	}
	
	mfmem::snew_array_1D(f2n_backup, n, dmrfl);
	for(IntType i = 0; i < n; ++i)
		f2n_backup[i] = f2n[i];
	
	IntType    **F2N_backup = NULL;
	mfmem::snew_array_1D(F2N_backup, nTFace, dmrfl);
	F2N_backup[0] = f2n_backup;
	for (IntType i = 1; i < nTFace; ++i)
	{
		F2N_backup[i] = &(F2N_backup[i - 1][nNPF_backup[i - 1]]);
	}
	
	n = 0;
	uTaskTree *uTaskTreeRoot = grid->GetuTaskTree();
	IntType *index = uTaskTreeRoot->uTaskTree_get_faceRev();
	IntType *Index = uTaskTreeRoot->uTaskTree_get_facePerm();

	for(IntType i = 0; i < nTFace; ++i)
	{
		f2c[i+i] = f2c_backup[index[i]+index[i]];
		f2c[i+i+1] = f2c_backup[index[i]+index[i]+1];
		nNPF[i] = nNPF_backup[index[i]];
		for(IntType j = 0; j < nNPF[i]; ++j)
			f2n[n++] = F2N_backup[index[i]][j];
	}

#ifdef MPICH

    IntType *nbz_backup = NULL;
    IntType *nbf_backup = NULL;
	mfmem::snew_array_1D(nbz_backup, nIFace, dmrfl);
	mfmem::snew_array_1D(nbf_backup, nIFace, dmrfl);

	for (IntType i = 0; i < nIFace; ++i)
    {
        nbz_backup[i] = nbz[i];
        nbf_backup[i] = nbf[i];
    }

    IntType size, id;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    IntType *nbFace = new int[size];
    IntType **facePerm = new int*[size];
    for(IntType i = 0; i < size; i++)
    {
        if (i == id)
        {
            for (IntType j = 0; j < size; j++)
                if (i != j)
                {
                    MPI_Send(&nBFace, 1, MPI_INT, j, j, MPI_COMM_WORLD);
                    MPI_Send(Index, nBFace, MPI_INT, j, j+size, MPI_COMM_WORLD);
                }
        }
        else
        {
            MPI_Recv(&nbFace[i], 1, MPI_INT, i, id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            facePerm[i] = new int[nbFace[i]];
            MPI_Recv(facePerm[i], nbFace[i], MPI_INT, i, id+size, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    for (IntType i = 0; i < nIFace; ++i)
    {
        nbz[i] = nbz_backup[index[i+pfacenum]-pfacenum];
        nbf[i] = facePerm[nbz[i]][nbf_backup[index[i+pfacenum]-pfacenum]];
    }

    for(IntType i = 0; i < size; i++)
        if (i != id)
            delete[] facePerm[i];
    delete[] facePerm;
    delete[] nbFace;
    
	mfmem::sdel_array_1D(nbz_backup);
	mfmem::sdel_array_1D(nbf_backup);
#endif

    cout << "uTaskTree Face Reordering successfully!" << endl;

	mfmem::sdel_array_1D(f2c_backup);
	mfmem::sdel_array_1D(f2n_backup);
	mfmem::sdel_array_1D(nNPF_backup);
	mfmem::sdel_array_1D(F2N_backup);
}

void uTaskTree_NodeReordering(PolyGrid *grid)
{	
	IntType nTFace = grid->GetNTFace();
	IntType nTNode = grid->GetNTNode();
	IntType nINode = grid->GetNINode();
	IntType *f2n = grid->Getf2n();
	IntType *nNPF = grid->GetnNPF();
    RealGeom *x = grid->GetX();
    RealGeom *y = grid->GetY();
    RealGeom *z = grid->GetZ();

    uTaskTree *uTaskTreeRoot = grid->GetuTaskTree();
    IntType *index = uTaskTreeRoot->uTaskTree_get_nodePerm();
    IntType *Index = uTaskTreeRoot->uTaskTree_get_nodeRev();
	
	IntType n = 0;
	for(IntType i = 0; i < nTFace; ++i) n+=nNPF[i];
	
    for (IntType i = 0; i < n; ++i)
		f2n[i] = index[f2n[i]];
	
	n = 0;
	
    RealGeom *x_backup = NULL;
    RealGeom *y_backup = NULL;
    RealGeom *z_backup = NULL;

	mfmem::snew_array_1D(x_backup, nTNode, dmrfl);
	mfmem::snew_array_1D(y_backup, nTNode, dmrfl);
	mfmem::snew_array_1D(z_backup, nTNode, dmrfl);

    for(IntType i = 0; i < nTNode; i++)
    {
        x_backup[i] = x[i];
        y_backup[i] = y[i];
        z_backup[i] = z[i];
    }

    for(IntType i = 0; i < nTNode; i++)
    {
        x[i] = x_backup[Index[i]];
        y[i] = y_backup[Index[i]];
        z[i] = z_backup[Index[i]];
    }

#ifdef MPICH

    IntType *nbsN = grid->GetnbSN();
    IntType *nbzN = grid->GetnbZN();
    IntType *nbrN = grid->GetnbRN();

    IntType size, id;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    IntType *ntNode = new int[size];
    IntType **nodePerm = new int*[size];
    for(IntType i = 0; i < size; i++)
    {
        if (i == id)
        {
            for (IntType j = 0; j < size; j++)
                if (i != j)
                {
                    MPI_Send(&nTNode, 1, MPI_INT, j, j, MPI_COMM_WORLD);
                    MPI_Send(index, nTNode, MPI_INT, j, j+size, MPI_COMM_WORLD);
                }
        }
        else
        {
            MPI_Recv(&ntNode[i], 1, MPI_INT, i, id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            nodePerm[i] = new int[ntNode[i]];
            MPI_Recv(nodePerm[i], ntNode[i], MPI_INT, i, id+size, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    for(IntType i = 0; i < nINode; i++)
    {
        nbsN[i] = index[nbsN[i]];
        nbrN[i] = nodePerm[nbzN[i]][nbrN[i]];
    }

#endif

    cout << "uTaskTree Node Reordering successfully!" << endl;

	mfmem::sdel_array_1D(x_backup);
	mfmem::sdel_array_1D(y_backup);
	mfmem::sdel_array_1D(z_backup);

    IntType *c2n = NULL;
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType *f2c = grid->Getf2c();
	unordered_map<int,bool> *C2N = new unordered_map<int,bool>[nTCell];

    n = 0;
    int dimCell3 = 0;
    for (int i = 0; i < nTFace; i++)
    {
        for(int j = 0; j < nNPF[i]; j++, n++)
        {
            C2N[f2c[i+i]][f2n[n]] = 1;
            dimCell3 = std::max(dimCell3, (int)C2N[f2c[i+i]].size());
            if (i >= nBFace) 
            {
                C2N[f2c[i+i+1]][f2n[n]] = 1;
                dimCell3 = std::max(dimCell3, (int)C2N[f2c[i+i+1]].size());
            }
        }
    }

    c2n = new int[dimCell3 * nTCell];


    for (int i = 0; i < nTCell; i++)
    {
        int j = 0;
        for (auto cell:C2N[i])
        {
            c2n[i*dimCell3+j] = cell.first;
            j++;
        }
        for (; j < dimCell3; j++)
            c2n[i*dimCell3+j] = c2n[i*dimCell3];
        C2N[i].clear();
    }

    unordered_map<int,bool> *n2n = new unordered_map<int,bool>[nTNode];
    for (int i = 0; i < nTCell; i++)
    {
        for(int j = 0; j < dimCell3; j++)
            for(int k = j+1; k < dimCell3; k++)
            {
                if (c2n[i*dimCell3+j] == c2n[i*dimCell3+k]) continue;
                n2n[c2n[i*dimCell3+j]][c2n[i*dimCell3+k]] = 1;
                n2n[c2n[i*dimCell3+k]][c2n[i*dimCell3+j]] = 1;
            }
    }
    // printGrid2(grid, n2n);
    for (int i = 0; i < nTNode; i++)
        n2n[i].clear();
    
    delete[] n2n;
}


/************************************************************************
Purpose:  create DC tree of face for DC task parallelism
************************************************************************/
void CreateTree(PolyGrid *grid)
{
	IntType i, j, n = 0;
    IntType dimCell1 = 0, dimCell2 = 0;
    IntType dimCell3 = 0, dimFace = 0;
	IntType nTFace = grid->GetNTFace();
    IntType nTCell = grid->GetNTCell();
    IntType nTNode = grid->GetNTNode();

	IntType nBFace = grid->GetNBFace();
    IntType nIFace = grid->GetNIFace();
    IntType pfacenum = nBFace - nIFace;

    IntType *f2c = grid->Getf2c();
	IntType *f2n = grid->Getf2n();
    IntType *nNPF = grid->GetnNPF();
	
	IntType *f2n_backup = NULL;
	IntType *f2c_backup = NULL;

    for(i = 0; i < nTFace; ++i)
        dimFace = std::max(dimFace, nNPF[i]);

    int nbParts = 4;
    int partSize = 256;
    while (nTCell / partSize > 1800 && partSize < 2048) partSize += 128;
    while (nTCell / partSize < 360 && partSize > 32) partSize -= 32;
	cout<< "nbParts = " << nbParts << ", partSize = "<< partSize << endl << flush;

	mfmem::snew_array_1D(f2n_backup, nTFace * dimFace, dmrfl);
	mfmem::snew_array_1D(f2c_backup, nTFace * 2, dmrfl);

    for(i = 0; i < nTFace; i++) 
    {
        for(j = 0; j < nNPF[i]; j++)
            f2n_backup[i*dimFace+j] = f2n[n++];
        for(; j < dimFace; j++)
            f2n_backup[i*dimFace+j] = f2n_backup[i*dimFace];
        f2c_backup[i<<1] = f2c[i<<1];
        f2c_backup[i<<1|1] = f2c[i<<1|1];
    }

    IntType *c2c = NULL;
    IntType *c2f = NULL;
    IntType *c2n = NULL;

	unordered_map<int,bool> *C2C = new unordered_map<int,bool>[nTCell];
	unordered_map<int,bool> *C2F = new unordered_map<int,bool>[nTCell];
	unordered_map<int,bool> *C2N = new unordered_map<int,bool>[nTCell];

    n = 0;
    for (i = 0; i < nTFace; i++)
    {
        for(j = 0; j < nNPF[i]; j++, n++)
        {
            C2N[f2c[i+i]][f2n[n]] = 1;
            dimCell3 = std::max(dimCell3, (int)C2N[f2c[i+i]].size());
            if (i >= nBFace) 
            {
                C2N[f2c[i+i+1]][f2n[n]] = 1;
                dimCell3 = std::max(dimCell3, (int)C2N[f2c[i+i+1]].size());
            }
        }

        C2F[f2c[i+i]][i] = 1;
        dimCell2 = std::max(dimCell2, (int)C2F[f2c[i+i]].size());
        if (i >= nBFace) 
        {
            C2F[f2c[i+i+1]][i] = 1;
            dimCell2 = std::max(dimCell2, (int)C2F[f2c[i+i+1]].size());

            C2C[f2c[i+i]][f2c[i+i]] = 1;
            C2C[f2c[i+i+1]][f2c[i+i+1]] = 1;
            C2C[f2c[i+i]][f2c[i+i+1]] = 1;
            C2C[f2c[i+i+1]][f2c[i+i]] = 1;
            dimCell1 = std::max(dimCell1, (int)std::max(C2C[f2c[i+i]].size(), C2C[f2c[i+i+1]].size()));
        }

    }

    c2n = new int[dimCell3 * nTCell];
    c2f = new int[dimCell2 * nTCell];
    c2c = new int[dimCell1 * nTCell];


    for (i = 0; i < nTCell; i++)
    {
        j = 0;
        for (auto cell:C2N[i])
        {
            c2n[i*dimCell3+j] = cell.first;
            j++;
        }
        for (; j < dimCell3; j++)
            c2n[i*dimCell3+j] = c2n[i*dimCell3];
        C2N[i].clear();

        j = 0;
        for (auto cell:C2F[i])
        {
            c2f[i*dimCell2+j] = cell.first;
            j++;
        }
        for (; j < dimCell2; j++)
            c2f[i*dimCell2+j] = c2f[i*dimCell2];
        C2F[i].clear();

        j = 0;
        for (auto cell:C2C[i])
        {
            c2c[i*dimCell1+j] = cell.first;
            j++;
        }
        for (; j < dimCell1; j++)
            c2c[i*dimCell1+j] = c2c[i*dimCell1];
        C2C[i].clear();
    }

    unordered_map<int,bool> *n2n = new unordered_map<int,bool>[nTNode];
    for (int i = 0; i < nTCell; i++)
    {
        for(int j = 0; j < dimCell3; j++)
            for(int k = j+1; k < dimCell3; k++)
            {
                if (c2n[i*dimCell3+j] == c2n[i*dimCell3+k]) continue;
                n2n[c2n[i*dimCell3+j]][c2n[i*dimCell3+k]] = 1;
                n2n[c2n[i*dimCell3+k]][c2n[i*dimCell3+j]] = 1;
            }
    }
    // printGrid1(grid, n2n);
    for (i = 0; i < nTNode; i++)
        n2n[i].clear();
    
    delete[] n2n;
    delete[] C2N;
    delete[] C2F;
    delete[] C2C;
		
    cout << "uTaskTree begin to create!\n" << flush;
    uTaskTree *uTaskTreeRoot = new uTaskTree(nTCell, nTFace, nTNode, nbParts, partSize);
    uTaskTreeRoot->uTaskTree_creation(c2c, c2f, c2n, f2n_backup, f2c_backup, nTCell, dimCell1, dimCell2, dimCell3, nTFace, dimFace, nTNode, pfacenum, nBFace);
    grid->SetuTaskTree(uTaskTreeRoot);
    cout << "uTaskTree created successfully!\n" << flush;
	mfmem::sdel_array_1D(f2n_backup);
	mfmem::sdel_array_1D(f2c_backup);
	delete[] c2n;
    delete[] c2f;
    delete[] c2c;
}


void uTaskTree_CellReordering(PolyGrid *grid)
{
	uTaskTree *uTaskTreeRoot = grid->GetuTaskTree();
	IntType *index = uTaskTreeRoot->uTaskTree_get_cellPerm();

	IntType i;
	IntType nTCell = grid->GetNTCell();
	IntType nTFace = grid->GetNTFace();
	IntType *f2c = grid->Getf2c();

    IntType *t = new int[nTCell]();
    for(i=0;i<nTCell;i++)
        t[index[i]]++;
    
    for(i=0;i<nTCell;i++)
        assert(t[i]==1);
	
	for(i=0;i<nTFace;i++)
	{
		if (f2c[i<<1]>=0 && f2c[i<<1] < nTCell) 
			f2c[i<<1] = index[f2c[i<<1]];
		if (f2c[i<<1|1]>=0 && f2c[i<<1|1] < nTCell) 
			f2c[i<<1|1] = index[f2c[i<<1|1]];
	}

    cout << "uTaskTree Cell Reordering successfully!" << endl;
}
#endif

#undef CPP_FILD_ID  // clear out file id
} //~namespace mflow
