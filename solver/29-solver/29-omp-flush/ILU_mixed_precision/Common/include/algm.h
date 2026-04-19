//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   algm.h
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

#ifndef MFL_ALGOMMFLOW_H
#define MFL_ALGOMMFLOW_H

#include "number_type.h"
#include "constant.h"
#include <string>
using namespace std;


namespace mflow
{
	
/*============================================================================\
                                排序函数
\============================================================================*/

// 对数组a中的第is个到ie个元素进行升序排序
void quick_sort_entire(RealGeom *a, IntType is, IntType ie);
// 对数组a中的第is个到ie个元素进行升序排序，ib随着a排列元素
void quick_sort_entire(RealGeom *a, IntType is, IntType ie, IntType *ib);

static void adt_sort(IntType layer, IntType *idvol, IntType is, IntType ie, RealGeom* ttsort, RealGeom *x, RealGeom *y, RealGeom *z, IntType NLayer, IntType *MarkLayer);
static void quick_sort(RealGeom *a, IntType is, IntType ie, IntType *ib, IntType h);

// 交换数组a的第i和j个元素，同步交换数组ib的第i和j个元素
void swap(RealGeom* a, IntType i, IntType j, IntType* ib);
void swap(RealGeom* a, IntType i, IntType j);



/*============================================================================\
                      Class MinDist
Calculate the distance of one point to a set of points 
Update: 2021-04-07 Add search multi-points
\============================================================================*/
class MinDist
{    
private:
    //  all interpolated points
    IntType nnodes;     
    RealGeom *xs;   
    RealGeom *ys;   
    RealGeom *zs;

    bool boxexist;  // bounding box exist

    // bounding box accelerating method
    IntType nSurfBox;   
    IntType *nPt_SurfBox;
    IntType *Pt_SurfBox;
    RealGeom **bnd_SurfBox;

public:
    MinDist(void);
    ~MinDist(void);

    /// \brief Pass-in donor points
    void SetPoints(IntType npin, RealGeom *xin, RealGeom *yin, RealGeom *zin)
    { nnodes = npin; xs = xin; ys = yin; zs = zin;};

    /// \Initialization, create boxes
    void Init(void);

    /// \brief reset, delete bounding box
    void Reset(void);

    /// \brief Search minimum distance and corresponding point index
    IntType SearchIndex(RealGeom xin, RealGeom yin, RealGeom zin, RealGeom &lmin);
    void SearchIndex(IntType np, RealGeom *xin, RealGeom *yin, RealGeom *zin, RealGeom *lmin, IntType *indices);
    friend class DWF;
private:
    void Compute_PntBox(IntType nBox, IntType NLayer, IntType *nPt_Box, IntType *Pt_Box, IntType nNode, RealGeom *x, RealGeom *y, RealGeom *z);

    // 求点(xp,yp,zp)到box(由两个角点定义)的距离（到box内的点列最小距离）
    RealGeom FindRminbox( RealGeom xp, RealGeom yp, RealGeom zp, RealGeom *bnd);

    // 计算box数组中每个定义box的角点
    void Compute_bnd_Box(RealGeom **bnd, IntType nBox, IntType *nPt, IntType *Pt, RealGeom *x, RealGeom *y, RealGeom *z);    
};

/*============================================================================\
                                string functions
\============================================================================*/
string int2str(const IntType source);

// Judge a real number equal to zero
inline bool EqualZero(RealFlow x) { return (x > -TINY) && (x < TINY); }

} //~ namespace mflow

#endif // ~MFL_ALGOMMFLOW_H
