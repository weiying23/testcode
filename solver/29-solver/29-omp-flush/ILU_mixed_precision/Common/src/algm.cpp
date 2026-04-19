//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   algm.cpp
/// \brief  common used algorithm
/// \author ...
/// \date   2014-8-19
/// \copyright  C.All rights reserved. 2014-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

// direct head file
#include "algm.h"

// C++ build-in head files
#include <iostream>
#include <sstream>
#include <cmath>
#include <string.h>
#include <stdlib.h>

// other user defined head files
#include "memory_util.h"

#if FS_CUDA_Grid
#include "cuGrid.cuh"
using namespace gpuGrid;
#endif

using namespace std;

namespace mflow
{

/*============================================================================\
                                排序函数
\============================================================================*/
/******************************************************************************
    利用Quick排序法实现ADT的子程序
******************************************************************************/
void adt_sort(IntType layer, IntType *idvol, IntType is, IntType ie, RealGeom* ttsort, RealGeom *x, RealGeom *y, RealGeom *z, IntType NLayer, IntType *MarkLayer)
{    
    IntType i;  // 循环计数器 

    IntType lyr = layer%3;

    if(lyr==0) {
        for(i=is; i<=ie; i++) ttsort[i] = x[idvol[i]];
    } else if(lyr==1) {
        for(i=is; i<=ie; i++) ttsort[i] = y[idvol[i]];
    } else if(lyr==2) {
        for(i=is; i<=ie; i++) ttsort[i] = z[idvol[i]];
    }

    IntType h = (is+ie)/2;
    quick_sort(ttsort, is, ie, idvol, h);
        
    if(layer == NLayer) {
        MarkLayer[idvol[h]]  = layer;
        MarkLayer[idvol[ie]] = layer;
        return;
    }

    switch(h-is){
        case 0:
            break;

        default:
            IntType ls = is;
            IntType le = h;
            adt_sort(layer+1, idvol, ls, le, ttsort, x, y, z, NLayer, MarkLayer);
    }
    
    if (h+1==ie){
        ;
    }else{
        IntType rs = h + 1;
        IntType re = ie;
        adt_sort(layer+1, idvol, rs, re, ttsort, x, y, z, NLayer, MarkLayer);
    }
}

/******************************************************************************
       quicksort排序法的子程序
NOTE：排序数组a中第is到ie个元素，h为is和ie之间。输出结果保证第is到h的元素小于
      第h到ie中的元素。ib中的元素随着a交换
******************************************************************************/
void quick_sort(RealGeom *a, IntType is, IntType ie, IntType *ib, IntType h)
{
    IntType i, last;
    if(ie-is < 1) return;

    swap(a, is, (ie+is)/2, ib);
    last = is;
    for(i=is+1; i<=ie; i++)
        if(a[i]<a[is]) swap(a, ++last, i, ib);
            swap(a, is, last, ib);

    if(last == h) return;
    if(last > h) quick_sort(a, is, last-1, ib, h);
    if(last < h) quick_sort(a, last+1, ie, ib, h);
}
/******************************************************************************
    交换数组a的第i和j个元素，同步交换数组ib的第i和j个元素
******************************************************************************/
void swap(RealGeom* a, IntType i, IntType j, IntType* ib)
{
    IntType itemp;
    RealGeom temp;
        
    temp = a[i];
    a[i] = a[j];
    a[j] = temp;
    itemp = ib[i];
    ib[i] = ib[j];
    ib[j] = itemp;
}

void swap(RealGeom* a, IntType i, IntType j)
{
    RealGeom temp;
    temp=a[i];
    a[i]=a[j];
    a[j]=temp;
}

/******************************************************************************
       quicksort排序法，对数组a中的第is个到ie个元素进行升序排序
******************************************************************************/
void quick_sort_entire(RealGeom *a, IntType is, IntType ie)
{
    IntType i, last;
    if(ie-is < 1) return;
    swap(a, is, (ie+is)/2);
    last = is;
    for(i=is+1; i<=ie; i++)
        if(a[i]<a[is]) swap(a, ++last, i);

    swap(a, is, last);
    quick_sort_entire(a, is, last-1);
    quick_sort_entire(a, last+1, ie);
}

/******************************************************************************
   quicksort排序法，对数组a中的第is个到ie个元素进行升序排序，ib随着a排列元素
******************************************************************************/
void quick_sort_entire(RealGeom *a, IntType is, IntType ie, IntType *ib)
{
    IntType i, last;
    if(ie-is<1) return;
    swap(a, is, (ie+is)/2, ib);
    last = is;
    for(i=is+1; i<=ie; i++)
        if(a[i]<a[is]) swap(a, ++last, i, ib);
    swap(a, is, last, ib);
    quick_sort_entire(a, is, last-1, ib);
    quick_sort_entire(a, last+1, ie, ib);
}




MinDist::MinDist(void)
{
    //  all surface nodes
    nnodes = 0;     // node num. of all surface nodes
    xs = NULL;      // x coordinates of all surface nodes
    ys = NULL;      // y coordinates of all surface nodes
    zs = NULL;      // z coordinates of all surface nodes

    boxexist = false;

    // surface nodes boxes
    nSurfBox = 0; 
    nPt_SurfBox = NULL; 
    Pt_SurfBox  = NULL;
    bnd_SurfBox = NULL; 
}

MinDist::~MinDist(void)
{
    // xs,ys,zs,xsd,ysd,zsd,nmove,xv,yv,zv each was not allocated in DWF
    // need not to be dealloacted.

    // delete surface boxes
    mfmem::sdel_array_1D(nPt_SurfBox);
    mfmem::sdel_array_1D(Pt_SurfBox);
    mfmem::sdel_array_2D(bnd_SurfBox);// 注意，bnd_SurfBox默认为连续内存申请方式
}

// delete bounding box
void MinDist::Reset()
{ 
    boxexist = false; 

    // delete surface boxes
    mfmem::sdel_array_1D(nPt_SurfBox);
    mfmem::sdel_array_1D(Pt_SurfBox);
    mfmem::sdel_array_2D(bnd_SurfBox);// 注意，bnd_SurfBox默认为连续内存申请方式
}

void MinDist::Init(void)
{
    if(boxexist) return;

    const IntType MAX_NODES_BOX = 20;

    //为点集合分配盒子
    IntType count = IntType(sqrt(1.*nnodes)); 
    count = std::min(count, MAX_NODES_BOX);

    IntType NLayer;
    nSurfBox = 1;
    for(NLayer=0; NLayer<count; NLayer++) {
        nSurfBox *= 2;
        if(nSurfBox >= count) break;
    }
	/* cout << "nSurfBox:" << nSurfBox << endl;
	exit(0); */
    mfmem::snew_array_1D(nPt_SurfBox, nSurfBox+1,dmrfl);
    mfmem::snew_array_1D(Pt_SurfBox, nnodes,dmrfl);
    Compute_PntBox(nSurfBox, NLayer, nPt_SurfBox, Pt_SurfBox, nnodes, xs, ys, zs);

    //计算盒子的角点
    mfmem::snew_array_2D(bnd_SurfBox, nSurfBox, 6, dmrfl, true);
    Compute_bnd_Box(bnd_SurfBox, nSurfBox, nPt_SurfBox, Pt_SurfBox, xs, ys, zs);
	
#if FS_CUDA_Grid	
	GPUGridDataTrans2(nSurfBox, nPt_SurfBox, Pt_SurfBox, &bnd_SurfBox[0][0]);
#endif

    boxexist = true;
}

IntType MinDist::SearchIndex(RealGeom xin, RealGeom yin, RealGeom zin, RealGeom &lmin)
{
    RealGeom *rr = NULL;
    IntType *BSort = NULL;
    mfmem::snew_array_1D(rr, nSurfBox,dmrfl);
    mfmem::snew_array_1D(BSort, nSurfBox,dmrfl);

    IntType j;

    for(j=0; j<nSurfBox; j++) {
        rr[j] = FindRminbox(xin, yin, zin, bnd_SurfBox[j]); // square of distance
        BSort[j] = j;
    }
    quick_sort_entire(rr, 0, nSurfBox-1, BSort);

    lmin = BIG;     // initialise
    RealGeom dx, dy, dz, len;
    IntType pnt, index = 0;
    RealGeom error = 1e-6;
    for(IntType ib=0; ib<nSurfBox; ib++) {
        if(rr[ib]<lmin+error) { // point in box which distance is lager than lenmin is ingnored
            j = BSort[ib];
            for(IntType k=nPt_SurfBox[j]; k<nPt_SurfBox[j+1]; k++) {
                pnt = Pt_SurfBox[k];
                dx = xin - xs[pnt];
                dy = yin - ys[pnt];
                dz = zin - zs[pnt];
                len = dx*dx + dy*dy + dz*dz;
                if(len<lmin){
                    lmin = len;
                    index = pnt;
                }
            }
        }
    }
    mfmem::sdel_array_1D(rr);
    mfmem::sdel_array_1D(BSort);

    lmin = sqrt(lmin);
    return index;
}


void MinDist::SearchIndex(IntType np, RealGeom *xin, RealGeom *yin, RealGeom *zin, RealGeom *lmin, IntType *indices)
{
    RealGeom error = 1e-6;

    RealGeom *rr = NULL;
    IntType *BSort = NULL;
    mfmem::snew_array_1D(rr, nSurfBox,dmrfl);
    mfmem::snew_array_1D(BSort, nSurfBox,dmrfl);

    // initialize minimum distance
    for (IntType ip = 0; ip < np; ++ip) lmin[ip] = BIG;

    for (IntType ip = 0; ip < np; ++ip)
    {
        for(IntType ibox = 0; ibox < nSurfBox; ++ibox)
        {
            rr[ibox] = FindRminbox(xin[ip], yin[ip], zin[ip], bnd_SurfBox[ibox]); // square of distance
            BSort[ibox] = ibox;
        }

        quick_sort_entire(rr, 0, nSurfBox-1, BSort);
        
        for(IntType ibox = 0; ibox < nSurfBox; ++ibox)
        {
            // point in box which distance is lager than lmin is ignored
            if(rr[ibox] < lmin[ip]+error)
            {
                IntType sorted_box = BSort[ibox];
                for(IntType k = nPt_SurfBox[sorted_box]; k < nPt_SurfBox[sorted_box+1]; ++k)
                {
                    IntType pnt = Pt_SurfBox[k];
                    RealGeom dx = xin[ip] - xs[pnt];
                    RealGeom dy = yin[ip] - ys[pnt];
                    RealGeom dz = zin[ip] - zs[pnt];
                    RealGeom len = dx*dx + dy*dy + dz*dz;
                    if(len < lmin[ip])
                    {
                        lmin[ip] = len;
                        indices[ip] = pnt;
                    }
                }
            }
        }
    }
    
    mfmem::sdel_array_1D(rr);
    mfmem::sdel_array_1D(BSort);

    for (IntType ip = 0; ip < np; ++ip) lmin[ip] = sqrt(lmin[ip]);
}


 /****************************************************************************\
                建立点集的盒子
NOTE：  创建box数组，返回nPt_Box[nBox]
        第i个box包含点序号[nPt_Box[i],nPt_Box[i+1])
        Pt_Box[nNode](与nPt_Box[nBox]对应的点的序号)
INPUT： 
OUTPUT：
Update: 2013-02-21 19:29:40 
Update: 2014-8-19 20:18:13 [开始序号,结束序号)=>nPt_Box[++j] = i + 1; tangj
\*****************************************************************************/
void MinDist::Compute_PntBox(IntType nBox, IntType NLayer, IntType *nPt_Box, IntType *Pt_Box, IntType nNode, RealGeom *x, RealGeom *y, RealGeom *z)
{
    IntType i,j;
    IntType is, ie;
    
    IntType *MarkLayer = NULL;
    mfmem::snew_array_1D(MarkLayer, nNode,dmrfl);
    for(i=0; i<nNode; i++) MarkLayer[i] = -1;

    for(i=0; i<nNode; i++) Pt_Box[i] = i;

    RealGeom *ttsort = NULL;
    mfmem::snew_array_1D(ttsort, nNode,dmrfl);
    is=0;
    ie=nNode-1;
    IntType layer=0;
    adt_sort(layer, Pt_Box, is, ie, ttsort, x, y, z, NLayer, MarkLayer);

    nPt_Box[0] = 0;
    j = 0;
    for(i=0; i<nNode; i++) {
        if(MarkLayer[Pt_Box[i]]==NLayer) {
            nPt_Box[++j] = i + 1;
        }
    }
    mfmem::sdel_array_1D(MarkLayer);
    mfmem::sdel_array_1D(ttsort);
}

/******************************************************************************
    求每个box的最小最大角点box[xmin,ymin,zmin,xmax,ymax,zmax]
******************************************************************************/
void MinDist::Compute_bnd_Box(RealGeom **bnd, IntType nBox, IntType *nPt, IntType *Pt, RealGeom *x, RealGeom *y, RealGeom *z)
{
    RealGeom GREAT = 1.0E+20;
    IntType i,j;

    for(i=1; i<nBox; i++) bnd[i] = &bnd[i-1][6];
    for(i=0; i<nBox; i++) {
        for(j=0; j<3; j++) bnd[i][j] = GREAT;
        for(j=3; j<6; j++) bnd[i][j] = -GREAT;

        for(j=nPt[i]; j<nPt[i+1]; j++) {
            if(bnd[i][0]>x[ Pt[j] ]) bnd[i][0] = x[ Pt[j] ];
            if(bnd[i][1]>y[ Pt[j] ]) bnd[i][1] = y[ Pt[j] ];
            if(bnd[i][2]>z[ Pt[j] ]) bnd[i][2] = z[ Pt[j] ];
            if(bnd[i][3]<x[ Pt[j] ]) bnd[i][3] = x[ Pt[j] ];
            if(bnd[i][4]<y[ Pt[j] ]) bnd[i][4] = y[ Pt[j] ];
            if(bnd[i][5]<z[ Pt[j] ]) bnd[i][5] = z[ Pt[j] ];
        }
    }
}
/******************************************************************************
     Purpose: To compute the least distance square from point to box 
******************************************************************************/
RealGeom MinDist::FindRminbox( RealGeom xp, RealGeom yp, RealGeom zp, RealGeom *bnd)
{
    RealGeom rr, rx, ry, rz;
    if( xp>=bnd[0] && xp<=bnd[3]) rx = 0;
    else rx = ( xp<bnd[0] ) ? bnd[0]-xp : xp-bnd[3];
    if( yp>=bnd[1] && yp<=bnd[4]) ry = 0;
    else ry = ( yp<bnd[1] ) ? bnd[1]-yp : yp-bnd[4];
    if( zp>=bnd[2] && zp<=bnd[5]) rz = 0;
    else rz = ( zp<bnd[2] ) ? bnd[2]-zp : zp-bnd[5];
    
    rr = rx*rx + ry*ry + rz*rz;
 
    return(rr);
}

/*============================================================================\
                                string functions
\============================================================================*/
 /************************************************************************
                        int转化为string类型函数
NOTE：  利用字符串流ostringstream将整型转化为string类型
        函数共有2个重载类型
        1、Int2Str(const int &source)
            将整型source转化为string，string中字符数等于int的位数，如
            Int2Str(1) = "1";   Int2Str(12) = "12"; Int2Str(333) = "333"
        2、Int2Str(const int &source, const int &width)
            指定返回string的最小宽度 width，右对齐，左侧用0补位填充
            当整数位数大于width时，以实际   位数为准，如
            Int2Str(2,3) = "002";   Int2Str(22,3) = "022";
            Int2Str(222,3) = "222"; Int2Str(2222,3) = "2222";
Update: 2012-4-9 21:49:47  tangj
************************************************************************/
string int2str(const IntType source){
    ostringstream ss("");
    ss<<source;
    return ss.str();
}

} // ~namespace mflow
