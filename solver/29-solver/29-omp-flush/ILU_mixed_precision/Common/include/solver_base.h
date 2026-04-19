//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   solver_base.h
/// \brief  A base class for solver
/// \author 
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_SOLVER_BASE_H
#define MFL_SOLVER_BASE_H


namespace mflow
{

class Solver     // the solver class, assumed zonal
{                // i.e. each zone can have different solvers                                
public:
  
    virtual     ~Solver() {} ;
          
    virtual void    Init() =0;
    virtual void    Solve() =0;
    virtual void    Post() =0;

    // 原本应该写成纯虚函数，但目前只有NSSolver重写了函数，以后其他求解器修改后，
    // 再修改本函数 -- 王新建20211011
    /// \brief  在得到新的流场变量后，并行传值、设置边界值、计算梯度、预处理计算
    virtual void ProcessAfterNewQuantity(PolyGrid *grid) {}

    virtual void    UpdateInterfaceData() = 0;  // for multi-zone problem
    virtual void    UpdataUnstVolData()   = 0;
};

} //~namespace mflow

#endif //~MFL_SOLVER_BASE_H
