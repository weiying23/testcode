//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   solver_ns.cpp
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

// direct head file
#include "solver_ns.h"

// C++ build-in head files
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <map>
//#include <immintrin.h>
using namespace std;

// other user defined head files
#include "utility_functions.h"
#include "zone.h"
#include "turbulence.h"
#include "algm.h"
#include "temporal_discretisation_implicit.h"
#include "io_base_format.h"
#include "io_log.h"
#include "parallel_base_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"
#include "io_field.h"
#include "parameter_reader.h"

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif

#ifdef MPICH
#include "mpi.h"
#endif

#ifdef FS_SIMD
#include "omp.h"
#endif

#ifdef FS_OPENMP
#include "omp.h"
#endif

#if (defined FS_CUDA)||(defined FS_CUDA_DEBUG)
#include "cuInviscidFlux.cuh"
#include "cuViscidFlux.cuh"
#include "cuLimit.cuh"
#include "cuGradientQ_Gauss.cuh"
#include "cuLUSGS.cuh"
#include "cuMPI.cuh"

using namespace gpuData;
#endif



//add by dingxin, 20211126
#ifdef TIMECOST
extern double* timecost;
extern int num_timecost;
extern double  time_flux, time_invis, time_roe, time_vis, time_calvis;
extern double  time_limiter;
extern double  time_gradient;
extern double  time_lusgs, time_RK;
extern double  time_SA;
#endif // TIMECOST

const int  Vec = 8;
#define ALIGN 64

namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 11903  // define file id


#ifdef MPICH
extern int myZone;
extern int numprocs;
extern MPI_Comm GridComm;  //for each grid, tangj
#endif



NSSolver::NSSolver(IntType ng, PolyGrid **gridsin, DataStore **fieldsin, 
                   DataSafe *cParain, BCond *bcin, Zone *zonein) : kNVar(5)
{
    nGrids = ng; 
    grids  = gridsin; 
    fields = fieldsin; 
    bc     = bcin;  
    cPara  = cParain;
    zone   = zonein;

    var_name_.reserve(kNVar);
    var_name_.resize(kNVar);
    var_name_[0] = "rho";
    var_name_[1] = "u";
    var_name_[2] = "v";
    var_name_[3] = "w";
    var_name_[4] = "p";
}


void NSSolver::Init()
{
    IntType restart,steady=1;
    PolyGrid *pgrid = (PolyGrid *)grids[0];

    int mpirank = 0;
#ifdef MPICH
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
#endif

#ifdef FS_OPENMP
    IntType OMPTHREADS = 12;
    pgrid->GetData(&OMPTHREADS,   INT, 1, "OMP_THREADS");
    omp_set_num_threads( OMPTHREADS ); 
    if(!mpirank){
	    #pragma omp parallel
	    {
		    #pragma omp single
		    {
                printf("OpenMP version: %d\n", _OPENMP);
			    printf("Number of CPU cores: %d\n", omp_get_num_procs());
			    printf("the threadID: %d\n", omp_get_thread_num());
			    printf("Number of Threads: %d\n", omp_get_num_threads());
		    }
	    }
    }
    //omp_set_dynamic( 0 );
#endif

    //add by dingxin, 20211126
#ifdef TIMECOST
    num_timecost = 10;//统计时间的模块个数
    mfmem::snew_array_1D(timecost, num_timecost, dmrfl);
    for (IntType i = 0; i < num_timecost; i++) {
        timecost[i] = 0;
    }

    time_flux = 0; //including all flux computation
    time_invis = 0; //including all invis computation
    time_roe = 0; //only including roe computation
    time_vis = 0;    //including all vis computation
    time_calvis = 0; //only including CalVisFluxTest computation
    time_limiter = 0;
    time_gradient = 0;
    time_lusgs = 0;
    time_SA = 0;
#endif // TIMECOST

    // initialize the finest grid
    GetData(&restart,INT,1,"restart");
    GetData(&steady,INT,1,"steady");

    // read unsteady steps and time information from file
    if(restart != 0) {
        ReadStepInfoFromFile((PolyGrid *) grids[0]);
    }

    // allocate memory for flow field
    AllocateFlowfieldMemory((PolyGrid *) grids[0]);

    if(restart==0) {
        IntType iter_done = 0;
        UpdateData(&iter_done, INT, 1, "iter_done");
        zone->simu->UpdateData(&iter_done, INT, 1, "iter_done");
        RealFlow start_time = 0.;
        UpdateData(&start_time, REAL_FLOW, 1, "start_time");
        RealFlow time = 0.;
        UpdateData(&time, REAL_FLOW, 1, "time");
        IntType Unst_steps_Curt=0;
        zone->simu->UpdateData(&Unst_steps_Curt, INT, 1, "Unst_steps_Curt");
 
        InitGridVar(pgrid);
        if(!steady) InitGridVarUnst(pgrid);
    }
    else if(restart==1){
        ReadRestartFromFile((PolyGrid *) grids[0]);
    }
    // 并行传值
    TransferInterfaceData(pgrid);
    // 计算边界虚网格
    SetGhostVariables(pgrid);
    // 为梯度分配内存并计算
    set_grad_method(pgrid);
#if (!defined FS_CUDA)
    if (grad_method_) {
        AllocateQuantityGradientMemory(pgrid);
        InitCalculateQuantityGradient(pgrid);
    }
#endif
    // 计算层流动力粘性
    ComputeVis_l(pgrid);

    InitVis_t(pgrid);

    
 
}


//*****************************************************************************\
/// \brief 求解NS方程流场 
///
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 2020-08-07  tangj      Add parameter 'iter_step_physic_time' for NS/Turb
///                        convergence log file.
/// </pre>
//*****************************************************************************/
void NSSolver::Solve()
{
    PolyGrid *grid = (PolyGrid *) grids[0];//zhyb: the finest grid
    RealFlow t_now;    //zhyb: total time
    double   t_onestep;//zhyb: the time of current step

    // Find the local time before solving flow on the grid
#ifdef MPICH
    t_onestep = MPI_Wtime();
#else
    
#if !(defined(Windows_NT) )
    timeval t_tmp;
    gettimeofday(&t_tmp, NULL);
    t_onestep = (double)t_tmp.tv_sec + (double)t_tmp.tv_usec/1000000;
#else
    t_onestep = ((double)clock()) / CLOCKS_PER_SEC;
#endif
#endif
     
    // Get iteration and zonal number
    IntType  iter_done;// current iterate steps
    GetData(&iter_done, INT, 1 ,"iter_done");
    // Get the starting time and find out the recent time
    GetData(&t_now, REAL_FLOW, 1, "start_time");
    
    // Zero the residuals of the finest grid
    ZeroResiduals(grid);
    
    // Solve NS for all grids starting from the finest grid
    SolveNSOnGrid(grid, 0);

    // 得到新的流场变量后，并行传值、设置边界值、计算梯度、预处理计算，计算粘性
    ProcessAfterNewQuantity(grid);

    iter_done++;
    
    // Find the local time after solving flow on the grid
#ifdef MPICH
    t_onestep = MPI_Wtime() - t_onestep;
#else
#if !(defined(Windows_NT))
    gettimeofday(&t_tmp, NULL);
    t_onestep = (double)t_tmp.tv_sec + (double)t_tmp.tv_usec/1000000 - t_onestep;
#else
    t_onestep = ((double)clock()) / CLOCKS_PER_SEC - t_onestep;
#endif
#endif
    t_now += t_onestep;
    
    IntType n_wconverg = 20;
    GetData(&n_wconverg,  INT, 1, "n_wconverg");
    // for convergence log file to output the first step of each physical time step
    IntType iter_step_physic_time = 1;
    GetData(&iter_step_physic_time,  INT, 1, "iter_step_physic_time", 0);
    if((iter_done%n_wconverg==0) || (iter_done==1) || iter_step_physic_time==0){
#ifdef FS_CUDA
		cuLoadBackRes(grid);
#endif
        // Print out norm of the residuals
        IntType zn = grid->GetZone();
        DumpNormResi(grid, iter_done, zn, t_now);
  
        //注意:此时虚拟单元的流场值未更新,严格的来说计算得到的气动力有区别
        IntType outputforce = 0;
        GetData(&outputforce, INT, 1, "output_Force", 0);
        if(outputforce) DumpForce(grid, iter_done, zn); 
    }
    
    UpdateData(&iter_done, INT, 1,"iter_done");
    UpdateData(&t_now, REAL_FLOW, 1, "start_time");
    
    // Free the memories of the residuals for all grids
    FreeAllGridResi(grid);
    
    // Print out restart file if necessary
    IntType n_wrest = 500;
    GetData(&n_wrest, INT, 1, "n_wrest");

    if(iter_done%n_wrest == 0){
        IntType zn = grid->GetZone();
        DumpRestart(grid, iter_done, zn, t_now);
    }
    
    IntType n_steps_coarse;
    grid->GetData(&n_steps_coarse, INT, 1 ,"n_steps_coarse");
    if((iter_done-n_steps_coarse) == n_wrest){
        ModifyRestartInputFile(grid);
    }
}


/*******************************************************************************\
  Post-Processing, i.e. dump data file for visualization                    
\*******************************************************************************/
void NSSolver::Post()
{
    IntType iter_done, zn, n_steps;
    RealFlow t_now;
    PolyGrid *grid = (PolyGrid *)grids[0];
    
    GetData(&iter_done,INT,1,"iter_done");
    GetData(&t_now, REAL_FLOW, 1, "start_time");
    zn = zone->GetZoneNo();
    zone->simu->GetData(&n_steps, INT, 1, "n_steps");
    
    // dump restart file
    IntType n_wrest = 50;
    GetData(&n_wrest, INT, 1, "n_wrest", 0);
    if(n_steps > 0 && iter_done%n_wrest != 0){
        DumpRestart(grid, iter_done, zn, t_now);
    }
}

/// \brief  在得到新的流场变量后，并行传值、设置边界值、计算梯度、预处理计算，
///         计算粘性
/// \par    Update records:
/// <pre>
/// Date        Author      Description
/// 2021-09-23  王新建      编写函数
/// </pre>
void NSSolver::ProcessAfterNewQuantity(PolyGrid *grid) {

#ifdef MPICH
    // 并行传值
	#ifdef FS_CUDA
		cuTransferInterfaceData(grid);
		/* IntType n = grid->GetNTCell()+grid->GetNBFace();
		RealFlow *rho = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
		RealFlow *u   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
		RealFlow *v   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
		RealFlow *w   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
		RealFlow *p   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
		cuGradientMemoryTrans(rho, u, v, w, p);   // transfer ghost cells  */
	#else	
		TransferInterfaceData(grid);
	#endif
#endif

    // 计算边界虚网格
#ifdef FS_CUDA
    cuSetGhostVariables(grid);		
#else
	SetGhostVariables(grid);
#endif

    // 计算梯度
    set_grad_method(grid);
    if (grad_method_) {
        CalculateQuantityGradient(grid);
    }

    // 计算层流动力粘性
#ifdef FS_CUDA	
    cuComputeVis_l(grid);
#else
	ComputeVis_l(grid);
#endif

}

/*******************************************************************************\
                     
\*******************************************************************************/
void NSSolver::UpdateInterfaceData()
{
    CommInterfaceData("rho");
    CommInterfaceData("u");
    CommInterfaceData("v");
    CommInterfaceData("w");
    CommInterfaceData("p");   
}


void NSSolver::UpdataUnstVolData()
{
    CommUnstVolData("rho", "rho_cur", "rho_old");
    CommUnstVolData("u", "u_cur", "u_old");
    CommUnstVolData("v", "v_cur", "v_old");
    CommUnstVolData("w", "w_cur", "w_old");
    CommUnstVolData("p", "p_cur", "p_old");
}


/*******************************************************************************\
                     
\*******************************************************************************/
void NSSolver::CommInterfaceData(const char *name)
{
    IntType g;
    PolyGrid *grid;

#ifndef MPICH
    PolyGrid *grid0 = grids[0];
    IntType  *nbz = grid0->GetFaceNeighborZones();

    for(IntType i=0; i<grid0->GetNumberOfFaceNeighbors(); i++) {        
        Zone *nz = zone->simu->GetZone(nbz[i]);
        for(g=0; g<nGrids; g++) {
            grid = (PolyGrid *) grids[g];
            grid->CommInterfaceData(nbz[i], (PolyGrid*)nz->GetGrid(g), name);
        }
    }
#else
    IntType n;

    for(g=0; g<nGrids; g++) {
        grid = (PolyGrid *) grids[g];
        n = grid->GetNTCell()+grid->GetNBFace();
        RealFlow *q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
        grid->CommInterfaceDataMPI(q);
    }

#endif
}


/*******************************************************************************\
 对非定常进行n层和n-1层物理量的准备
\*******************************************************************************/
void NSSolver::CommUnstVolData(const char *name, const char *name_cur, const char *name_old)
{
    IntType i, g, n, nTCell, nBFace;
    PolyGrid *grid;

    for(g=0; g<nGrids; g++) {
        grid = (PolyGrid *) grids[g];
        nTCell = grid->GetNTCell();
        nBFace = grid->GetNBFace();
        n      = nTCell + nBFace;
        RealFlow *q     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
        RealFlow *q_cur = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, name_cur);
        RealFlow *q_old = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, name_old);

        //将当前步洞中单元使用与其相邻的计算单元代替
//      ModyHollCellValue(grid, q);

        for(i=0; i<nTCell; i++) q_old[i] = q_cur[i];
        for(i=0; i<nTCell; i++) q_cur[i] = q[i];
    }
}

/// \brief  为并行边界面的虚网格赋值
/// \par    Update records:
/// <pre>
/// Date        Author      Description
/// 2021-09-23  王新建      编写函数
/// </pre>
void NSSolver::TransferInterfaceData(PolyGrid *grid) {
    const IntType n = grid->GetNTCell() + grid->GetNBFace();
    RealFlow **q = NULL;
    mfmem::snew_array_1D(q, kNVar, dmrfl);
    for (IntType i = 0; i < kNVar; ++i) {
        q[i] = static_cast<RealFlow *>(
            grid->GetDataPtr(REAL_FLOW, n, var_name_[i].c_str()));
    }
#ifdef MPICH
	grid->RecvSendVarNeighbor_Togeth(kNVar, q);
    /*
    for (IntType i = 0; i < kNVar; ++i) {
        grid->CommInterfaceDataMPI(q[i]);
    }
    */
#endif
    mfmem::sdel_array_1D(q);
}

void NSSolver::set_grad_method(const PolyGrid *grid) {
    grad_method_ = 0;
    IntType vis_mode = 0;
    GetData(&vis_mode, INT, 1, "vis_mode");
    IntType order = 0;
    GetData(&order, INT, 1, "order");

    const IntType level = grid->GetLevel();
    if (vis_mode != INVISCID || order != FIRST_ORDER) {
        GetData(&grad_method_, INT, 1, "GradQ");
    }
}

/// \brief  为rho, u, v, w, p梯度分配内存，并更新到gField中
/// \par    Update records:
/// <pre>
/// Date        Author      Description
/// 2021-09-23  王新建      编写函数
/// </pre>
void NSSolver::AllocateQuantityGradientMemory(PolyGrid *grid) {
    const IntType nTCell = grid->GetNTCell();
    const IntType n = nTCell + grid->GetNBFace();

    RealFlow *dqdx = NULL, *dqdy = NULL, *dqdz = NULL;
    mfmem::snew_array_1D(dqdx, kNVar * n, dmrfl);
    mfmem::snew_array_1D(dqdy, kNVar * n, dmrfl);
    mfmem::snew_array_1D(dqdz, kNVar * n, dmrfl);
    for (IntType i = 0; i < kNVar * n; ++i) {
        dqdx[i] = 0;
        dqdy[i] = 0;
        dqdz[i] = 0;
    }
    grid->UpdateDataPtr(dqdx, REAL_FLOW, kNVar * n, "dqdx");
    grid->UpdateDataPtr(dqdy, REAL_FLOW, kNVar * n, "dqdy");
    grid->UpdateDataPtr(dqdz, REAL_FLOW, kNVar * n, "dqdz");
}

/// \brief  计算rho, u, v, w, p梯度，并更新到gField中
/// \note   需要预先分配内存
/// \par    Update records:
/// <pre>
/// Date        Author      Description
/// 2021-09-17  王新建      编写函数
/// </pre>
void NSSolver::CalculateQuantityGradient(PolyGrid *grid) {
    const IntType nTCell = grid->GetNTCell();
    const IntType n = nTCell + grid->GetNBFace();

    RealFlow **dqdx = NULL, **dqdy = NULL, **dqdz = NULL;
    mfmem::snew_array_1D(dqdx, kNVar, dmrfl);
    mfmem::snew_array_1D(dqdy, kNVar, dmrfl);
    mfmem::snew_array_1D(dqdz, kNVar, dmrfl);
    dqdx[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdx"));
    dqdy[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdy"));
    dqdz[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdz"));
    for (IntType i = 1; i < kNVar; ++i) {
        dqdx[i] = &dqdx[i - 1][n];
        dqdy[i] = &dqdy[i - 1][n];
        dqdz[i] = &dqdz[i - 1][n];
    }
    RealFlow **q = NULL;
    mfmem::snew_array_1D(q, kNVar, dmrfl);
    for (IntType i = 0; i < kNVar; ++i) {
        q[i] = static_cast<RealFlow *>(
            grid->GetDataPtr(REAL_FLOW, n, var_name_[i].c_str()));
    }
    RealFlow* u_n, * v_n, * w_n;
    IntType  nTNode = grid->GetNTNode();
    
#ifdef FS_CUDA	

	#ifdef MultiStream
		cuCompGradientQ_MultiStream(grid, q, dqdx, dqdy, dqdz, u_n, v_n, w_n);	
	#else
		u_n = NULL;
		v_n = NULL;
		w_n = NULL;
		mfmem::snew_array_1D(u_n, nTNode, dmrfl);
		mfmem::snew_array_1D(v_n, nTNode, dmrfl);
		mfmem::snew_array_1D(w_n, nTNode, dmrfl);
		for (IntType i = 0; i < kNVar; ++i) {
			cuCompGradientQ(grid, q[i], dqdx[i], dqdy[i], dqdz[i], i, u_n, v_n, w_n);
		}
	#endif

	#ifdef MultiStream
		#ifdef MPICH
			grid->cuRecvSendVarNeighbor_TogethForGradient_unfold(5, dqdx, dqdy, dqdz);
		#endif	
		cuSetGhostQuantityGradients(grid, dqdx, dqdy, dqdz);
	#else
		#ifdef MPICH
			grid->cuRecvSendVarNeighbor_Togeth(kNVar, dqdx, 3);
			grid->cuRecvSendVarNeighbor_Togeth(kNVar, dqdy, 4);
			grid->cuRecvSendVarNeighbor_Togeth(kNVar, dqdz, 5);
			cuUpdateGhostGrad(dqdx, dqdy, dqdz);
		#endif		
		cuSetGhostQuantityGradients(grid, dqdx, dqdy, dqdz);
		mfmem::sdel_array_1D(u_n);
		mfmem::sdel_array_1D(v_n);
		mfmem::sdel_array_1D(w_n);
	#endif

#else
	u_n = NULL;
    v_n = NULL;
    w_n = NULL;
    mfmem::snew_array_1D(u_n, nTNode, dmrfl);
    mfmem::snew_array_1D(v_n, nTNode, dmrfl);
    mfmem::snew_array_1D(w_n, nTNode, dmrfl);
	for (IntType i = 0; i < kNVar; ++i) {
		CompGradientQ(grid, q[i], dqdx[i], dqdy[i], dqdz[i], i, u_n, v_n, w_n);
    }
	#ifdef MPICH	
		RealFlow **grad_mpi = NULL;
		mfmem::snew_array_1D(grad_mpi, 3 * kNVar, dmrfl);
		IntType count = 0;
		for (IntType i = 0; i < kNVar; ++i) {
			grad_mpi[count++] = dqdx[i];
			grad_mpi[count++] = dqdy[i];
			grad_mpi[count++] = dqdz[i];
		}
		grid->RecvSendVarNeighbor_Togeth(3 * kNVar, grad_mpi);
		mfmem::sdel_array_1D(grad_mpi);
	#endif
	SetGhostQuantityGradients(grid, dqdx, dqdy, dqdz);
	mfmem::sdel_array_1D(u_n);
    mfmem::sdel_array_1D(v_n);
	mfmem::sdel_array_1D(w_n);
#endif
    

    // 设置虚网格的梯度值
    // 注意：只计算了速度梯度的值，且支持的边界类型较少，例如没有发动机出入口边界
/* #ifdef FS_CUDA
	cuUpdateGhostGrad(dqdx, dqdy, dqdz);
#endif */

/* #ifdef FS_CUDA
    cuSetGhostQuantityGradients(grid, dqdx, dqdy, dqdz);
#else
	SetGhostQuantityGradients(grid, dqdx, dqdy, dqdz);
#endif */

    mfmem::sdel_array_1D(dqdx);
    mfmem::sdel_array_1D(dqdy);
    mfmem::sdel_array_1D(dqdz);
    mfmem::sdel_array_1D(q);
}

void NSSolver::InitCalculateQuantityGradient(PolyGrid *grid) {
    const IntType nTCell = grid->GetNTCell();
    const IntType n = nTCell + grid->GetNBFace();

    RealFlow **dqdx = NULL, **dqdy = NULL, **dqdz = NULL;
    mfmem::snew_array_1D(dqdx, kNVar, dmrfl);
    mfmem::snew_array_1D(dqdy, kNVar, dmrfl);
    mfmem::snew_array_1D(dqdz, kNVar, dmrfl);
    dqdx[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdx"));
    dqdy[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdy"));
    dqdz[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdz"));
    for (IntType i = 1; i < kNVar; ++i) {
        dqdx[i] = &dqdx[i - 1][n];
        dqdy[i] = &dqdy[i - 1][n];
        dqdz[i] = &dqdz[i - 1][n];
    }
    RealFlow **q = NULL;
    mfmem::snew_array_1D(q, kNVar, dmrfl);
    for (IntType i = 0; i < kNVar; ++i) {
        q[i] = static_cast<RealFlow *>(
            grid->GetDataPtr(REAL_FLOW, n, var_name_[i].c_str()));
    }
    RealFlow* u_n, * v_n, * w_n;
    IntType  nTNode = grid->GetNTNode();
    u_n = NULL;
    v_n = NULL;
    w_n = NULL;
    mfmem::snew_array_1D(u_n, nTNode, dmrfl);
    mfmem::snew_array_1D(v_n, nTNode, dmrfl);
    mfmem::snew_array_1D(w_n, nTNode, dmrfl);

    for (IntType i = 0; i < kNVar; ++i) {
		CompGradientQ(grid, q[i], dqdx[i], dqdy[i], dqdz[i], i, u_n, v_n, w_n);
    }

    mfmem::sdel_array_1D(u_n);
    mfmem::sdel_array_1D(v_n);
	mfmem::sdel_array_1D(w_n);

    // 并行传值
#ifdef MPICH
    RealFlow **grad_mpi = NULL;
    mfmem::snew_array_1D(grad_mpi, 3 * kNVar, dmrfl);
    IntType count = 0;
    for (IntType i = 0; i < kNVar; ++i) {
        grad_mpi[count++] = dqdx[i];
        grad_mpi[count++] = dqdy[i];
        grad_mpi[count++] = dqdz[i];
    }
    grid->RecvSendVarNeighbor_Togeth(3 * kNVar, grad_mpi);
    mfmem::sdel_array_1D(grad_mpi);
    /*
    for (IntType i = 0; i < 5; ++i) {
        grid->CommInterfaceDataMPI(dqdx[i]);
        grid->CommInterfaceDataMPI(dqdy[i]);
        grid->CommInterfaceDataMPI(dqdz[i]);
    }
    */
#endif
    // 设置虚网格的梯度值
    // 注意：只计算了速度梯度的值，且支持的边界类型较少，例如没有发动机出入口边界
    SetGhostQuantityGradients(grid, dqdx, dqdy, dqdz);

    mfmem::sdel_array_1D(dqdx);
    mfmem::sdel_array_1D(dqdy);
    mfmem::sdel_array_1D(dqdz);
    mfmem::sdel_array_1D(q);
}

/// \brief  设置虚网格的梯度值，注意：只计算了速度梯度的值，且支持的边界类型
///         较少，例如没有发动机出入口边界
void NSSolver::SetGhostQuantityGradients(
    const PolyGrid *grid, 
    RealFlow **dqdx, RealFlow **dqdy, RealFlow **dqdz
) {
    const IntType nTCell = grid->GetNTCell();
    const IntType nBFace = grid->GetNBFace();
    const IntType n = nTCell + nBFace;

    const IntType *f2c = grid->Getf2c();
    const RealGeom *xfn = grid->GetXfn();
    const RealGeom *yfn = grid->GetYfn();
    const RealGeom *zfn = grid->GetZfn();
    const RealGeom *xcc = grid->GetXcc();
    const RealGeom *ycc = grid->GetYcc();
    const RealGeom *zcc = grid->GetZcc();
    const BCRecord **bcr = const_cast<const BCRecord **>(grid->Getbcr());

    IntType count = 0;
    for (IntType i = 0; i < nBFace; ++i) {
        const IntType type = bcr[i]->GetType();
        const IntType c1 = f2c[count++];
        const IntType c2 = f2c[count++];
 
        // Assign the variable gradient values for each ghost cell whose index is c2.
        RealFlow dta[3], dnn[3], dnnn;
        RealFlow gv1[3][3], gv2[3][3];
        switch (type) {
            case INTERFACE:
                break;
            case WALL:
                dnnn = 
                    dqdx[1][c1] * xfn[i]
                  + dqdy[1][c1] * yfn[i]
                  + dqdz[1][c1] * zfn[i];
                dnn[0] = dnnn * xfn[i];
                dnn[1] = dnnn * yfn[i];
                dnn[2] = dnnn * zfn[i];
                dta[0] = dqdx[1][c1] - dnn[0];
                dta[1] = dqdy[1][c1] - dnn[1];
                dta[2] = dqdz[1][c1] - dnn[2];
                dqdx[1][c2] = dnn[0] - dta[0];
                dqdy[1][c2] = dnn[1] - dta[1];
                dqdz[1][c2] = dnn[2] - dta[2];
                dnnn = 
                    dqdx[2][c1] * xfn[i]
                  + dqdy[2][c1] * yfn[i]
                  + dqdz[2][c1] * zfn[i];
                dnn[0] = dnnn * xfn[i];
                dnn[1] = dnnn * yfn[i];
                dnn[2] = dnnn * zfn[i];
                dta[0] = dqdx[2][c1] - dnn[0];
                dta[1] = dqdy[2][c1] - dnn[1];
                dta[2] = dqdz[2][c1] - dnn[2];
                dqdx[2][c2] = dnn[0] - dta[0];
                dqdy[2][c2] = dnn[1] - dta[1];
                dqdz[2][c2] = dnn[2] - dta[2];
                dnnn = 
                    dqdx[3][c1] * xfn[i]
                  + dqdy[3][c1] * yfn[i]
                  + dqdz[3][c1] * zfn[i];
                dnn[0] = dnnn * xfn[i];
                dnn[1] = dnnn * yfn[i];
                dnn[2] = dnnn * zfn[i];
                dta[0] = dqdx[3][c1] - dnn[0];
                dta[1] = dqdy[3][c1] - dnn[1];
                dta[2] = dqdz[3][c1] - dnn[2];
                dqdx[3][c2] = dnn[0] - dta[0];
                dqdy[3][c2] = dnn[1] - dta[1];
                dqdz[3][c2] = dnn[2] - dta[2];
                break;
            case SYMM:
                gv1[0][0] = dqdx[1][c1];
                gv1[1][0] = dqdx[2][c1];
                gv1[2][0] = dqdx[3][c1];
                gv1[0][1] = dqdy[1][c1];
                gv1[1][1] = dqdy[2][c1];
                gv1[2][1] = dqdy[3][c1];
                gv1[0][2] = dqdz[1][c1];
                gv1[1][2] = dqdz[2][c1];
                gv1[2][2] = dqdz[3][c1];
                SolveEquationforGradSYMM(gv1, gv2, xfn[i], yfn[i], zfn[i]);
                dqdx[1][c2] = gv2[0][0];
                dqdx[2][c2] = gv2[1][0];
                dqdx[3][c2] = gv2[2][0];
                dqdy[1][c2] = gv2[0][1];
                dqdy[2][c2] = gv2[1][1];
                dqdy[3][c2] = gv2[2][1];
                dqdz[1][c2] = gv2[0][2];
                dqdz[2][c2] = gv2[1][2];
                dqdz[3][c2] = gv2[2][2];
                break;          
            case FAR_FIELD:
                dqdx[1][c2] = 0.0;
                dqdx[2][c2] = 0.0;
                dqdx[3][c2] = 0.0;
                dqdy[1][c2] = 0.0;
                dqdy[2][c2] = 0.0;
                dqdy[3][c2] = 0.0;
                dqdz[1][c2] = 0.0;
                dqdz[2][c2] = 0.0;
                dqdz[3][c2] = 0.0;
                break;
            default:
                dqdx[1][c2] = 0.0;
                dqdx[2][c2] = 0.0;
                dqdx[3][c2] = 0.0;
                dqdy[1][c2] = 0.0;
                dqdy[2][c2] = 0.0;
                dqdy[3][c2] = 0.0;
                dqdz[1][c2] = 0.0;
                dqdz[2][c2] = 0.0;
                dqdz[3][c2] = 0.0;
                break;
        }
    }
}




/*******************************************************************************\
            Execute the multi-grid flow solver on grid level                    
\*******************************************************************************/
void SolveNSOnGrid(PolyGrid *grid, IntType level)
{
    IntType nTCell = grid->GetNTCell(), nT5 = 5*nTCell;

    RealFlow *rhs = NULL;
    mfmem::snew_array_1D( rhs, nT5,dmrfl);
    assert(rhs != 0);
    RealFlow *res = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nT5, "res");
    //修改了rhs中储存的内容,从强迫函数变更为从细网格限制得到的残差
    //并且在n_pre的第一次迭代中,利用当前网格的残差变更为强迫函数储存起来
    //后来在限制到粗网格的时候使用
    VectCopyFrom(rhs, res, nT5, 1);

    Relaxation(grid, level, rhs, 1);
    
    mfmem::sdel_array_1D(rhs);
}


/*******************************************************************************\
           Determine one cell to use multigrid or not(based on cell)
NOTE: Shock identification based on pressure ratio across shock,
      if one element is in contact with shock, then set det[] = 0
Update:
Time:  2019-6-20 
\*******************************************************************************/
void CellIsMG(PolyGrid *grid, IntType *det)
{
    IntType nTCell = grid->GetNTCell();
    IntType n = nTCell + grid->GetNBFace();
    IntType nTFace = grid->GetNTFace();
    IntType nBFace = grid->GetNBFace();
    IntType *f2c = grid->Getf2c();
    IntType i, c1, c2;

    RealFlow dp;
    RealFlow *p = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");

    RealFlow p_bar;
    grid->GetData(&p_bar,   REAL_FLOW, 1, "p_bar");
    RealFlow p_stag;
    grid->GetData(&p_stag,   REAL_FLOW, 1, "p_stag");


    // pressure threshod, default value is 0.001 in file "input.par"
    //RealFlow stind = 0.0001;
    //grid->GetData(&stind,REAL_FLOW,1,"stind",0);
    //stind *= 20.0;  // 0.02 as default
    //zhyb20200615: 由于修改了激波探测的规则，由压力差比来流总压修改为压力差比压力和，stind参数需要调整。
    //zhyb20200615: 根据喷流数值试验，将stind参数固化为0.1，这个值越大，判断出的激波单元越少，越小，越容易误判激波单元
    RealFlow stind = 0.1;
  

    for(i=0; i<nTCell; ++i) det[i] = 1;
    for(i=0; i<nBFace; ++i){
        c1 = f2c[i+i];
        c2 = f2c[i+i+1];
        dp = abs(p[c2] - p[c1])/(p[c2] + p[c1] + p_bar +p_bar);
        if(dp > stind){      //压力变化过大，不多重计算
            det[c1] = 0;
        }
    }
    for(i=nBFace; i<nTFace; ++i){
        c1 = f2c[i+i];
        c2 = f2c[i+i+1];
        dp = abs(p[c2] - p[c1])/(p[c2] + p[c1] + p_bar +p_bar);
        if(dp > stind){      //压力变化过大，不多重计算
            det[c1] = 0;
            det[c2] = 0;
        }
    }
  
}


/*******************************************************************************\
       Zero the residuals. Also allocate memory for the residuals               
       if they had not been allocated                                           
\*******************************************************************************/
void ZeroResiduals(PolyGrid *grid)
{
    IntType nTCell=grid->GetNTCell(), nT5 = 5*nTCell;
    RealFlow *res;
    res = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nT5, "res");
    if(!res){
        mfmem::snew_array_1D(res, nT5,dmrfl);
        assert(res != 0);
        grid->UpdateDataPtr(res, REAL_FLOW, nT5, "res");
    }

#ifdef FS_OPENMP
#pragma omp parallel for
#endif
    for(IntType i=0; i<nT5; i++){
        res[i] = 0.;
    }
}


/*******************************************************************************\
   Free memeories of the residuals for all grids. Here grid is the finiest one.                             
\*******************************************************************************/
void FreeAllGridResi(PolyGrid *grid)
{
    IntType  nTCell, nBFace;       
    PolyGrid *cgrid;

    cgrid = grid;
    nTCell = cgrid->GetNTCell();
    nBFace = cgrid->GetNBFace();
    RealFlow *res = (RealFlow *)cgrid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");
    RealFlow *DQ  = (RealFlow *)cgrid->GetDataPtr(REAL_FLOW, 5*(nTCell+nBFace), "DQ");
    if(res) cgrid->DeleteDataPtr("res");
    if(DQ) cgrid->DeleteDataPtr("DQ");
      
}


/*******************************************************************************\
          Forward solution for one time step                                    
          We assume flow variables q and grid matrices are all known
\*******************************************************************************/
void Relaxation(PolyGrid *grid, IntType level, RealFlow *rhs, IntType steps)
{
    //now for all schemes compute time step
#if (defined FS_CUDA)&&!(defined MultiStream)
	cuComputeTimeStep(grid);
#endif
#if !(defined FS_CUDA)
	ComputeTimeStep(grid);		
#endif

	IntType tScheme;
    grid->GetData(&tScheme, INT, 1, "tScheme");
    if (tScheme == MULTI_STAGE) {
#if (defined FS_CUDA)||(defined FS_CUDA_DEBUG_NS_RK)
        cuExplicitStep(grid);
#else
		ExplicitStep(grid); 
#endif
    } else{ // if ( (tScheme == LU_SGS) || (tScheme == DPLUR) ) {
		// LU-SGS or other methods e.g. Newton-Krylov matrix free method
		for(IntType n=0; n<steps; n++){		
#if (defined FS_CUDA)||(defined FS_CUDA_DEBUG_NS_GMRES)||(defined FS_CUDA_DEBUG_NS_LUSGS)
			cuForwardStep(grid, rhs, level, n);
#else
			ForwardStep(grid, rhs, level, n);
#endif	
		}
	}
}

/// \brief  显式时间推进方法
/// \par    Update records:
/// <pre>
/// Date        Author      Description
/// 2022-11-13  王新建      编写函数
/// </pre>
void ExplicitStep(PolyGrid* grid) {
    const IntType nTCell = grid->GetNTCell();

    IntType n_stage;
    RealFlow lamda[10];
    grid->GetData(&n_stage, INT, 1, "n_stage");
    grid->GetData(lamda, REAL_FLOW, n_stage, "lamda");

    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");

    RealFlow **q = NULL;
    mfmem::snew_array_2D(q, 5, nTCell, dmrfl, true);
	
    LoadQ(grid, q);
    TransQtoW(grid, q);
	
    for (IntType i = 0; i < n_stage; i++) {
        RealFlow *res = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5 * nTCell, "res");
        for (IntType icell = 0; icell < 5 * nTCell; icell++) {
            res[icell] = 0.;
        }

        UpdateResiduals(grid, 0);

#ifdef TIMECOST
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemLusgs, endtimeTemLusgs;
    double timeuseTemLusgs;
    gettimeofday(&starttimeTemLusgs, 0); 
#endif
#endif

		TimeMarch(grid, q, dt, lamda[i]);
        
#ifdef TIMECOST

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
    mfmem::sdel_array_2D(q); // clear, CHF, 20220324
}

/******************************************************************************\
        Load flow variables stored in grid to q
\******************************************************************************/
void LoadQ(PolyGrid *grid, RealFlow **q) {
    const IntType nTCell = grid->GetNTCell();
    const IntType n = nTCell + grid->GetNBFace();
    
    RealFlow* q0[5];
    q0[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q0[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q0[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q0[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q0[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");

    for (IntType i = 0; i < nTCell; i++) {
        for (IntType k = 0; k < 5; ++k) {
            q[k][i] = q0[k][i];
        }
    }
}

/******************************************************************************\
                    将原始变量q变为守恒变量，并覆盖q
\******************************************************************************/
void TransQtoW(PolyGrid *grid, RealFlow **q) {
    IntType nTCell = grid->GetNTCell();
    const RealFlow gam = 1.4;
    const RealFlow gamm1 = gam -1.0;
    
    for (IntType i = 0; i < nTCell; i++) {
        q[1][i] *= q[0][i];
        q[2][i] *= q[0][i];
        q[3][i] *= q[0][i];
        q[4][i]  = q[4][i]/gamm1 + 0.5*(q[1][i]*q[1][i]+q[2][i]*q[2][i]+q[3][i]*q[3][i])/q[0][i];
    }
}

/*******************************************************************************\
             Advance the solution one step explicitly
\*******************************************************************************/
void TimeMarch(PolyGrid *grid, RealFlow **q, RealFlow *dt, RealFlow lamda) {  
    RealFlow rho, mx, my, mz, et, p;
    
    const RealFlow gam = 1.4;
    const RealFlow gamm1 = gam - 1.0;

    RealFlow p_bar;
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    // Grid informations
    const IntType nTCell = grid->GetNTCell();
    const IntType nBFace = grid->GetNBFace();
    const IntType n = nTCell + nBFace;
    const RealGeom *vol  = grid->GetCellVol();
    
    const RealFlow* res[5];
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5 * nTCell, "res");
    for (IntType i = 1; i < 5; ++i) {
        res[i] = res[i - 1] + nTCell;
    }

    RealFlow* nq[5];
    nq[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    nq[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    nq[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    nq[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    nq[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");


    for (IntType i = 0; i < nTCell; i++) {
        //assert(vol[i] > TINY);
        RealFlow dtbv = dt[i] / (vol[i] + TINY) * lamda;

        rho  = q[0][i];
        //assert(rho > TINY);
        mx   = q[1][i];
        my   = q[2][i];
        mz   = q[3][i];
        et   = q[4][i];

        rho += dtbv*res[0][i];
        mx  += dtbv*res[1][i];
        my  += dtbv*res[2][i];
        mz  += dtbv*res[3][i];
        et  += dtbv*res[4][i];
        p    = gamm1*(et - 0.5*(mx*mx + my*my + mz*mz)/rho);

        // Check if density or pressure is less than 0. If they are, make correction.
        if(p <= -p_bar || rho <= 0.){        
            // let dt be one order smaller;
            rho -= dtbv*res[0][i]*0.9;
            mx  -= dtbv*res[1][i]*0.9;
            my  -= dtbv*res[2][i]*0.9;
            mz  -= dtbv*res[3][i]*0.9;
            et  -= dtbv*res[4][i]*0.9;
            p    = gamm1*(et - 0.5*(mx*mx + my*my + mz*mz)/rho);
           
            if(p <= -p_bar || rho <= 0.){
                // let dt be one order smaller once more;
                rho -= dtbv*res[0][i]*0.09;
                mx  -= dtbv*res[1][i]*0.09;
                my  -= dtbv*res[2][i]*0.09;
                mz  -= dtbv*res[3][i]*0.09;
                et  -= dtbv*res[4][i]*0.09;
                p    = gamm1*(et - 0.5*(mx*mx + my*my + mz*mz)/rho);
            }
        }
        
        if(p > -p_bar && rho > 0.){
            nq[0][i] = rho;
            nq[1][i] = mx/rho;
            nq[2][i] = my/rho;
            nq[3][i] = mz/rho;
            nq[4][i] = p;
        }else{
            // Give warning and do not update solution
            printf("Warning: Negative pressure or density in cell %ld !!!\n", (long)(i+1));
        }
    }
}

/******************************************************************************\
        Compute time step 
\******************************************************************************/
void ComputeTimeStep(PolyGrid *grid)
{
    IntType nTCell = grid->GetNTCell();
    IntType level  = grid->GetLevel();
    
    IntType vis_mode;
    grid->GetData(&vis_mode,  INT, 1, "vis_mode");
    
    // Allocate memories for time step
    RealFlow *dt = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "dt_timestep");
    if(!dt){
        mfmem::snew_array_1D(dt, nTCell, dmrfl);
        assert(dt != 0);
        grid->UpdateDataPtr(dt, REAL_FLOW, nTCell, "dt_timestep");
    }
    
    //If count viscous or not
    IntType vis_run;  //0--inviscid   1--laminar   2--turbulence
    if(vis_mode == INVISCID){
        vis_run = 0;
    }else if(vis_mode == LAMINAR){
        vis_run = 1;
    }else{
        vis_run = 2;
    }
    if((level != 0) && (vis_mode != INVISCID)){  // if coarse grid doesn't want to run the viscous flux, turn it off
        IntType cg_vis = 1;
        grid->GetData(&cg_vis, INT, 1, "cg_vis");
        if(cg_vis == 0) vis_run = 0;
    }

    TimeStepNormal_new(grid, dt, vis_run);
    
    LimitTimeStep(grid, dt);  //note: cfl number in this function
}


/******************************************************************************\
        limit time step for robust
\******************************************************************************/
void LimitTimeStep(PolyGrid *grid, RealFlow *dt)
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    IntType level  = grid->GetLevel();
    
    RealFlow *p    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    
    IntType i, count;
    RealFlow cfl, cfl_tmp;
    
    IntType  iter_done, cfl_nstep;
    RealFlow cfl_start, cfl_end, cfl_coeff, cfl_ratio;
    grid->GetData(&iter_done, INT, 1, "iter_done");
    grid->GetData(&cfl_nstep, INT, 1, "cfl_nstep");
    grid->GetData(&cfl_start, REAL_FLOW, 1, "cfl_start");
    grid->GetData(&cfl_end,   REAL_FLOW, 1, "cfl_end");
    grid->GetData(&cfl_coeff, REAL_FLOW, 1, "cfl_coeff");
    grid->GetData(&cfl_ratio, REAL_FLOW, 1, "cfl_ratio");
    
    //for coarse grid, reduce cfl number using cfl_coeff
    if(level>0){
        cfl_start *= cfl_coeff;
        cfl_end   *= cfl_coeff;
    }
    
    //compute current step's cfl number
    if(iter_done < 0){  //粗网格迭代
        cfl = cfl_start;
    }else if(iter_done > cfl_nstep){
        cfl = cfl_end;
    }else{
        //zhyb20190620: modified from CFL3D, the ramping now is nonlinear, occurring slowerly at first
        //              and then increasing in rate.
        cfl = cfl_start*pow(cfl_ratio, (RealFlow)iter_done/cfl_nstep);
    }
    
    //limit cfl using p, come from USM3D
    RealFlow p_min, p_break, cfl_min;
    grid->GetData(&p_min,   REAL_FLOW, 1, "p_min");
    grid->GetData(&p_break, REAL_FLOW, 1, "p_break");
    grid->GetData(&cfl_min, REAL_FLOW, 1, "cfl_min");
    //limit cfl using gradient of p, decrease cfl in big gradient of p
    IntType *det = NULL;
    mfmem::snew_array_1D( det, nTCell,dmrfl);
    CellIsMG(grid, det);
   
    cfl_min = 0.5*cfl;  //在此处将cfl_min设为当前步cfl数乘以0.5
#ifdef FS_OPENMP
#pragma omp parallel for
#endif
    for(IntType i=0; i<nTCell; i++){
        RealFlow cfl_tmp;
        //根据压力场来确定当地cfl数。
        if(p[i]>p_break){
            cfl_tmp = cfl;
        }else if(p[i]<p_min){
            cfl_tmp = cfl_min;
        }else{
            cfl_tmp = (p[i]-p_min)/(p_break-p_min)*(cfl-cfl_min)+cfl_min;
        }
        //根据压力梯度的极值来限制当地cfl数
        if(!det[i]){
            cfl_tmp = cfl_min;
        }
        
        dt[i] *= cfl_tmp;
    }

    mfmem::sdel_array_1D(det);
    
    // Print out the maximum and minimun dt
    RealFlow dt_max = 0.0, dt_min = BIG;
#ifdef FS_OPENMPXX
#pragma omp parallel for
#endif
    for(IntType i=0; i<nTCell; i++){
        dt_max = std::max(dt_max, dt[i]);
        dt_min = std::min(dt_min, dt[i]);
    }
#ifdef MPICH
    RealFlow dt_max_glb, dt_min_glb;
    MPI_Allreduce(&dt_max, &dt_max_glb, 1, MPIReal, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&dt_min, &dt_min_glb, 1, MPIReal, MPI_MIN, MPI_COMM_WORLD);
    dt_max = dt_max_glb;
    dt_min = dt_min_glb;
#endif
    grid->UpdateData(&dt_max, REAL_FLOW, 1, "dt_max");
    grid->UpdateData(&dt_min, REAL_FLOW, 1, "dt_min");
    
    //Now limit the dt to ratio_dtmax*dt_min 
    RealFlow ratio_dtmax = 1.0e20;
    grid->GetData(&ratio_dtmax, REAL_FLOW, 1, "ratio_dtmax");
    RealFlow ratio_max = dt_max/dt_min;
    if(ratio_max > ratio_dtmax){
        RealFlow dt_max_lim = ratio_dtmax*dt_min;
#ifdef FS_OPENMP
#pragma omp parallel for
#endif        
        for(IntType i=0; i<nTCell; i++){
            //dt[i] = std::min(dt[i], dt_max_lim);
            if(dt[i] > dt_max_lim){
                dt[i] = dt_max_lim;
            }
        }
    }
    
#ifdef DEBUG
    count = 0;
    if(ratio_max > ratio_dtmax){
        RealFlow dt_max_lim = ratio_dtmax*dt_min;
        for(i=0; i<nTCell; i++){
            //dt[i] = std::min(dt[i], dt_max_lim);
            if(dt[i] > dt_max_lim){
                count++;
            }
        }
#ifdef MPICH
        Parallel::parallel_sum(count, MPI_COMM_WORLD);
#endif
        mflog::log.set_one_processor_out();
        mflog::log<<endl<<"dt_max/dt_min="<<IOS_EP(2)<<ratio_max<<endl;
        mflog::log<<endl<<count<<" cells are limited for dt too big."<<endl;
    }
#endif
    
    
#ifdef DEBUG
    BCRecord **bcr = grid->Getbcr();
    IntType  *f2c  = grid->Getf2c();
    IntType type, c1;
    
    count = 0;
    RealFlow dt_avg = 0.0;
    for(i=0; i<nBFace; i++){
        type  = bcr[i]->GetType();
        if(type != WALL) continue;
        c1 = f2c[i+i];
        
        count++;
        dt_avg += dt[c1];
    }
#ifdef MPICH
    Parallel::parallel_sum(dt_avg, MPI_COMM_WORLD);
    Parallel::parallel_sum(count, MPI_COMM_WORLD);
#endif
    mflog::log.set_one_processor_out();
    mflog::log << endl << "First layer's average time step is: " << IOS_EP(8) << dt_avg/count << endl;
#endif
}


#ifdef DC
void TimeStepNormal_new_Kernel1(char **userArgs, uTaskTreeArgs *treeArgs)
{
    IntType  i, c1, c2, nMid;
    RealFlow eigv, dn, vn, c2tmp, gam_tmp;
    
    RealFlow C = 4.0;
    RealFlow  muoopr;

    IntType ns = treeArgs->firstFace;
	IntType ne = treeArgs->lastFace + 1;
    if (ns >= ne) return; 


    PolyGrid *grid 		= (PolyGrid *)userArgs[0];
    RealFlow *dt        = (RealFlow *)userArgs[1];
    IntType  *f2c 		= (IntType *)userArgs[2];
    RealGeom *xfn       = (RealFlow *)userArgs[3];
    RealGeom *yfn       = (RealFlow *)userArgs[4];
    RealGeom *zfn       = (RealFlow *)userArgs[5];
    RealGeom *xfc       = (RealFlow *)userArgs[6];
    RealGeom *yfc       = (RealFlow *)userArgs[7];
    RealGeom *zfc       = (RealFlow *)userArgs[8];
    RealGeom *xcc       = (RealFlow *)userArgs[9];
    RealGeom *ycc       = (RealFlow *)userArgs[10];
    RealGeom *zcc       = (RealFlow *)userArgs[11];
    RealGeom *vgn       = (RealFlow *)userArgs[12];
    RealGeom *area      = (RealFlow *)userArgs[13];
    RealGeom *vol       = (RealFlow *)userArgs[14];
    
    RealFlow *rho       = (RealFlow *)userArgs[15];
    RealFlow *u         = (RealFlow *)userArgs[16];
    RealFlow *v         = (RealFlow *)userArgs[17];
    RealFlow *w         = (RealFlow *)userArgs[18];
    RealFlow *p         = (RealFlow *)userArgs[19];

    IntType steady      = (IntType)(*(IntType *)userArgs[20]); 
    RealFlow gam        = (RealFlow)(*(RealFlow *)userArgs[21]);
    RealFlow p_bar      = (RealFlow)(*(RealFlow *)userArgs[22]);

    RealFlow *vis_l     = (RealFlow *)userArgs[23];
    RealFlow *vis_t     = (RealFlow *)userArgs[24];
    RealFlow prl        = (RealFlow)(*(RealFlow *)userArgs[25]);
    RealFlow prt        = (RealFlow)(*(RealFlow *)userArgs[26]);

	IntType  nBFace = grid->GetNBFace();

    nMid  = ns; 
    if(ne  <= nBFace) {
        // If all boundary faces
        nMid = ne;
    } else if(ns < nBFace) {
        // Part of them are boundary faces
        nMid = nBFace;
    }
    // For boundary faces
    for(i=ns; i<nMid; i++){
        c1    = f2c[i+i];
        
        c2tmp = gam*(p[c1]+p_bar)/rho[c1];
        dn    = fabs((xfc[i]-xcc[c1])*xfn[i]+(yfc[i]-ycc[c1])*yfn[i]+(zfc[i]-zcc[c1])*zfn[i]);

        vn    = u[c1]*xfn[i]+v[c1]*yfn[i]+w[c1]*zfn[i];
        if(!steady) vn -= vgn[i];
        vn    = fabs(vn);
        eigv = vn+sqrt(c2tmp);
        
//        if(vis_run){
            muoopr = vis_l[c1]/prl + vis_t[c1]/prt;
            gam_tmp = gam;
           
            //eigv += C*gam_tmp/rho[c1]*muoopr/(dn+TINY);
            eigv   += C*gam_tmp/rho[c1]*muoopr*area[i]/vol[c1];
//        }
        dt[c1] = std::min(dt[c1],dn/eigv);
    }
    // For interior faces
    for(i=nMid; i<ne; i++){
        c1    = f2c[i+i];
        c2    = f2c[i+i+1];
        
        c2tmp = gam*(p[c1]+p_bar)/rho[c1];
        dn    = fabs((xfc[i]-xcc[c1])*xfn[i]+(yfc[i]-ycc[c1])*yfn[i]+(zfc[i]-zcc[c1])*zfn[i]);
        
        vn    = u[c1]*xfn[i]+v[c1]*yfn[i]+w[c1]*zfn[i];
        if(!steady) vn -= vgn[i];
        vn    = fabs(vn);
        eigv = vn+sqrt(c2tmp);
        
//        if(vis_run){
            muoopr = vis_l[c1]/prl + vis_t[c1]/prt;
         
            gam_tmp = gam;
            
            //eigv += C*gam_tmp/rho[c1]*muoopr/dn;
            eigv   += C*gam_tmp/rho[c1]*muoopr*area[i]/vol[c1];
//        }
        dt[c1] = std::min(dt[c1],dn/eigv);
        
        c2tmp = gam*(p[c2]+p_bar)/rho[c2];
        dn    = fabs((xfc[i]-xcc[c2])*xfn[i]+(yfc[i]-ycc[c2])*yfn[i]+(zfc[i]-zcc[c2])*zfn[i]);
        
        vn    = u[c2]*xfn[i]+v[c2]*yfn[i]+w[c2]*zfn[i];
        if(!steady) vn -= vgn[i];
        vn    = fabs(vn);
        eigv = vn+sqrt(c2tmp);
     
//        if(vis_run){
            muoopr = vis_l[c2]/prl + vis_t[c2]/prt;
            gam_tmp = gam;
            
            //eigv += C*gam_tmp/rho[c2]*muoopr/dn;
            eigv   += C*gam_tmp/rho[c2]*muoopr*area[i]/vol[c2];
//        }
        dt[c2] = std::min(dt[c2],dn/eigv);
    }
}

void TimeStepNormal_new_Kernel2(char **userArgs, uTaskTreeArgs *treeArgs)
{
    IntType  i, c1, c2, nMid;
    RealFlow eigv, dn, vn, c2tmp, gam_tmp;
    
    RealFlow C = 4.0;
    RealFlow  muoopr;

    IntType ns = treeArgs->firstFace;
	IntType ne = treeArgs->lastFace + 1;
    if (ns >= ne) return; 


    PolyGrid *grid 		= (PolyGrid *)userArgs[0];
    RealFlow *dt        = (RealFlow *)userArgs[1];
    IntType  *f2c 		= (IntType *)userArgs[2];
    RealGeom *xfn       = (RealFlow *)userArgs[3];
    RealGeom *yfn       = (RealFlow *)userArgs[4];
    RealGeom *zfn       = (RealFlow *)userArgs[5];
    RealGeom *xfc       = (RealFlow *)userArgs[6];
    RealGeom *yfc       = (RealFlow *)userArgs[7];
    RealGeom *zfc       = (RealFlow *)userArgs[8];
    RealGeom *xcc       = (RealFlow *)userArgs[9];
    RealGeom *ycc       = (RealFlow *)userArgs[10];
    RealGeom *zcc       = (RealFlow *)userArgs[11];
    RealGeom *vgn       = (RealFlow *)userArgs[12];
    RealGeom *area      = (RealFlow *)userArgs[13];
    RealGeom *vol       = (RealFlow *)userArgs[14];
    
    RealFlow *rho       = (RealFlow *)userArgs[15];
    RealFlow *u         = (RealFlow *)userArgs[16];
    RealFlow *v         = (RealFlow *)userArgs[17];
    RealFlow *w         = (RealFlow *)userArgs[18];
    RealFlow *p         = (RealFlow *)userArgs[19];

    IntType steady      = (IntType)(*(IntType *)userArgs[20]); 
    RealFlow gam        = (RealFlow)(*(RealFlow *)userArgs[21]);
    RealFlow p_bar      = (RealFlow)(*(RealFlow *)userArgs[22]);


	IntType  nBFace = grid->GetNBFace();

    nMid  = ns; 
    if(ne  <= nBFace) {
        // If all boundary faces
        nMid = ne;
    } else if(ns < nBFace) {
        // Part of them are boundary faces
        nMid = nBFace;
    }
    // For boundary faces
    for(i=ns; i<nMid; i++){
        c1    = f2c[i+i];
        
        c2tmp = gam*(p[c1]+p_bar)/rho[c1];
        dn    = fabs((xfc[i]-xcc[c1])*xfn[i]+(yfc[i]-ycc[c1])*yfn[i]+(zfc[i]-zcc[c1])*zfn[i]);

        vn    = u[c1]*xfn[i]+v[c1]*yfn[i]+w[c1]*zfn[i];
        if(!steady) vn -= vgn[i];
        vn    = fabs(vn);
        eigv = vn+sqrt(c2tmp);
        
/*       if(vis_run){
            muoopr = vis_l[c1]/prl + vis_t[c1]/prt;
            gam_tmp = gam;
           
            //eigv += C*gam_tmp/rho[c1]*muoopr/(dn+TINY);
            eigv   += C*gam_tmp/rho[c1]*muoopr*area[i]/vol[c1];
        }*/
        dt[c1] = std::min(dt[c1],dn/eigv);
    }
    // For interior faces
    for(i=nMid; i<ne; i++){
        c1    = f2c[i+i];
        c2    = f2c[i+i+1];
        
        c2tmp = gam*(p[c1]+p_bar)/rho[c1];
        dn    = fabs((xfc[i]-xcc[c1])*xfn[i]+(yfc[i]-ycc[c1])*yfn[i]+(zfc[i]-zcc[c1])*zfn[i]);
        
        vn    = u[c1]*xfn[i]+v[c1]*yfn[i]+w[c1]*zfn[i];
        if(!steady) vn -= vgn[i];
        vn    = fabs(vn);
        eigv = vn+sqrt(c2tmp);
        
/*        if(vis_run){
            muoopr = vis_l[c1]/prl + vis_t[c1]/prt;
         
            gam_tmp = gam;
            
            //eigv += C*gam_tmp/rho[c1]*muoopr/dn;
            eigv   += C*gam_tmp/rho[c1]*muoopr*area[i]/vol[c1];
        }*/
        dt[c1] = std::min(dt[c1],dn/eigv);
        
        c2tmp = gam*(p[c2]+p_bar)/rho[c2];
        dn    = fabs((xfc[i]-xcc[c2])*xfn[i]+(yfc[i]-ycc[c2])*yfn[i]+(zfc[i]-zcc[c2])*zfn[i]);
        
        vn    = u[c2]*xfn[i]+v[c2]*yfn[i]+w[c2]*zfn[i];
        if(!steady) vn -= vgn[i];
        vn    = fabs(vn);
        eigv = vn+sqrt(c2tmp);
     
/*        if(vis_run){
            muoopr = vis_l[c2]/prl + vis_t[c2]/prt;
            gam_tmp = gam;
            
            //eigv += C*gam_tmp/rho[c2]*muoopr/dn;
            eigv   += C*gam_tmp/rho[c2]*muoopr*area[i]/vol[c2];
        }*/
        dt[c2] = std::min(dt[c2],dn/eigv);
    }
}
#endif


void TimeStepNormal_new(PolyGrid *grid, RealFlow *dt, IntType vis_run)
{
    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  nTFace = grid->GetNTFace();
    IntType  n      = nTCell+nBFace;
    IntType  *f2c   = grid->Getf2c();
    RealGeom *xfn   = grid->GetXfn();
    RealGeom *yfn   = grid->GetYfn();
    RealGeom *zfn   = grid->GetZfn();
    RealGeom *xfc   = grid->GetXfc();
    RealGeom *yfc   = grid->GetYfc();
    RealGeom *zfc   = grid->GetZfc();
    RealGeom *xcc   = grid->GetXcc();
    RealGeom *ycc   = grid->GetYcc();
    RealGeom *zcc   = grid->GetZcc();
    RealGeom *vgn   = grid->GetFaceNormalVelocity();
    RealGeom *area  = grid->GetFaceArea();
    RealGeom *vol   = grid->GetCellVol();
    
    RealFlow *rho   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    RealFlow *u     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    RealFlow *v     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    RealFlow *w     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    RealFlow *p     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");

  
    IntType steady;
    RealFlow gam, p_bar;
    grid->GetData(&steady,      INT, 1, "steady");
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    
    IntType  i, c1, c2;
    RealFlow eigv, dn, vn, c2tmp, gam_tmp;
    
    RealFlow C = 4.0;
    RealFlow *vis_l, *vis_t;
    RealFlow prl, prt, muoopr;
    if(vis_run){
        vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
        grid->GetData(&prl, REAL_FLOW, 1, "prl");  
        vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
        grid->GetData(&prt, REAL_FLOW, 1, "prt");
    }

    // Set dt to BIG
#ifdef FS_OPENMP
#pragma omp parallel for
#endif
    for(IntType i=0; i<nTCell; i++){
        dt[i] = BIG;
    } 
#if (defined FS_OPENMP) && (defined GroupColor)
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
#pragma omp parallel for private(i,count,c1,eigv,dn,vn,c2tmp,gam_tmp,muoopr) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                count = 2 * i;
                c1 = f2c[count];
                c2tmp = gam * (p[c1] + p_bar) / rho[c1];
                dn = fabs((xfc[i] - xcc[c1]) * xfn[i] + (yfc[i] - ycc[c1]) * yfn[i] + (zfc[i] - zcc[c1]) * zfn[i]);

                vn = u[c1] * xfn[i] + v[c1] * yfn[i] + w[c1] * zfn[i];
                if (!steady) vn -= vgn[i];
                vn = fabs(vn);
                eigv = vn + sqrt(c2tmp);

                if (vis_run) {
                    muoopr = vis_l[c1] / prl + vis_t[c1] / prt;
                    gam_tmp = gam;

                    //eigv += C*gam_tmp/rho[c1]*muoopr/(dn+TINY);
                    eigv += C * gam_tmp / rho[c1] * muoopr * area[i] / vol[c1];
                }
                dt[c1] = std::min(dt[c1], dn / eigv);
            }
        }
        // zone boundary face
        count = 2 * pfacenum;
        for (i = pfacenum; i < nBFace; i++) {
            c1 = f2c[count];
            count += 2;
            c2tmp = gam * (p[c1] + p_bar) / rho[c1];
            dn = fabs((xfc[i] - xcc[c1]) * xfn[i] + (yfc[i] - ycc[c1]) * yfn[i] + (zfc[i] - zcc[c1]) * zfn[i]);

            vn = u[c1] * xfn[i] + v[c1] * yfn[i] + w[c1] * zfn[i];
            if (!steady) vn -= vgn[i];
            vn = fabs(vn);
            eigv = vn + sqrt(c2tmp);

            if (vis_run) {
                muoopr = vis_l[c1] / prl + vis_t[c1] / prt;
                gam_tmp = gam;

                //eigv += C*gam_tmp/rho[c1]*muoopr/(dn+TINY);
                eigv += C * gam_tmp / rho[c1] * muoopr * area[i] / vol[c1];
            }
            dt[c1] = std::min(dt[c1], dn / eigv);
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
#pragma omp parallel for private(i,count,c1,c2,eigv,dn,vn,c2tmp,gam_tmp,muoopr) schedule(static,groupSize)
            for (i = startFace; i < endFace; i++) {
                count = 2 * i;
                c1 = f2c[count];
                c2 = f2c[count + 1];
                c2tmp = gam * (p[c1] + p_bar) / rho[c1];
                dn = fabs((xfc[i] - xcc[c1]) * xfn[i] + (yfc[i] - ycc[c1]) * yfn[i] + (zfc[i] - zcc[c1]) * zfn[i]);

                vn = u[c1] * xfn[i] + v[c1] * yfn[i] + w[c1] * zfn[i];
                if (!steady) vn -= vgn[i];
                vn = fabs(vn);
                eigv = vn + sqrt(c2tmp);

                if (vis_run) {
                    muoopr = vis_l[c1] / prl + vis_t[c1] / prt;

                    gam_tmp = gam;

                    //eigv += C*gam_tmp/rho[c1]*muoopr/dn;
                    eigv += C * gam_tmp / rho[c1] * muoopr * area[i] / vol[c1];
                }
                dt[c1] = std::min(dt[c1], dn / eigv);

                c2tmp = gam * (p[c2] + p_bar) / rho[c2];
                dn = fabs((xfc[i] - xcc[c2]) * xfn[i] + (yfc[i] - ycc[c2]) * yfn[i] + (zfc[i] - zcc[c2]) * zfn[i]);

                vn = u[c2] * xfn[i] + v[c2] * yfn[i] + w[c2] * zfn[i];
                if (!steady) vn -= vgn[i];
                vn = fabs(vn);
                eigv = vn + sqrt(c2tmp);

                if (vis_run) {
                    muoopr = vis_l[c2] / prl + vis_t[c2] / prt;
                    gam_tmp = gam;

                    //eigv += C*gam_tmp/rho[c2]*muoopr/dn;
                    eigv += C * gam_tmp / rho[c2] * muoopr * area[i] / vol[c2];
                }
                dt[c2] = std::min(dt[c2], dn / eigv);
            }
        }
    }
    else {
        for (i = 0; i < nBFace; i++) {
            c1 = f2c[i + i];

            c2tmp = gam * (p[c1] + p_bar) / rho[c1];
            dn = fabs((xfc[i] - xcc[c1]) * xfn[i] + (yfc[i] - ycc[c1]) * yfn[i] + (zfc[i] - zcc[c1]) * zfn[i]);

            vn = u[c1] * xfn[i] + v[c1] * yfn[i] + w[c1] * zfn[i];
            if (!steady) vn -= vgn[i];
            vn = fabs(vn);
            eigv = vn + sqrt(c2tmp);

            if (vis_run) {
                muoopr = vis_l[c1] / prl + vis_t[c1] / prt;
                gam_tmp = gam;

                //eigv += C*gam_tmp/rho[c1]*muoopr/(dn+TINY);
                eigv += C * gam_tmp / rho[c1] * muoopr * area[i] / vol[c1];
            }
            dt[c1] = std::min(dt[c1], dn / eigv);
        }
        // For interior faces
        for (i = nBFace; i < nTFace; i++) {
            c1 = f2c[i + i];
            c2 = f2c[i + i + 1];

            c2tmp = gam * (p[c1] + p_bar) / rho[c1];
            dn = fabs((xfc[i] - xcc[c1]) * xfn[i] + (yfc[i] - ycc[c1]) * yfn[i] + (zfc[i] - zcc[c1]) * zfn[i]);

            vn = u[c1] * xfn[i] + v[c1] * yfn[i] + w[c1] * zfn[i];
            if (!steady) vn -= vgn[i];
            vn = fabs(vn);
            eigv = vn + sqrt(c2tmp);

            if (vis_run) {
                muoopr = vis_l[c1] / prl + vis_t[c1] / prt;

                gam_tmp = gam;

                //eigv += C*gam_tmp/rho[c1]*muoopr/dn;
                eigv += C * gam_tmp / rho[c1] * muoopr * area[i] / vol[c1];
            }
            dt[c1] = std::min(dt[c1], dn / eigv);

            c2tmp = gam * (p[c2] + p_bar) / rho[c2];
            dn = fabs((xfc[i] - xcc[c2]) * xfn[i] + (yfc[i] - ycc[c2]) * yfn[i] + (zfc[i] - zcc[c2]) * zfn[i]);

            vn = u[c2] * xfn[i] + v[c2] * yfn[i] + w[c2] * zfn[i];
            if (!steady) vn -= vgn[i];
            vn = fabs(vn);
            eigv = vn + sqrt(c2tmp);

            if (vis_run) {
                muoopr = vis_l[c2] / prl + vis_t[c2] / prt;
                gam_tmp = gam;

                //eigv += C*gam_tmp/rho[c2]*muoopr/dn;
                eigv += C * gam_tmp / rho[c2] * muoopr * area[i] / vol[c2];
            }
            dt[c2] = std::min(dt[c2], dn / eigv);
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
            IntType  c1, c2;
            RealFlow eigv, dn, vn, c2tmp, gam_tmp;
            c1    = f2c[i+i];
        
            c2tmp = gam*(p[c1]+p_bar)/rho[c1];
            dn    = fabs((xfc[i]-xcc[c1])*xfn[i]+(yfc[i]-ycc[c1])*yfn[i]+(zfc[i]-zcc[c1])*zfn[i]);

            vn    = u[c1]*xfn[i]+v[c1]*yfn[i]+w[c1]*zfn[i];
            if(!steady) vn -= vgn[i];
            vn    = fabs(vn);
            eigv = vn+sqrt(c2tmp);
        
            if(vis_run){
                muoopr = vis_l[c1]/prl + vis_t[c1]/prt;
                gam_tmp = gam;
           
                //eigv += C*gam_tmp/rho[c1]*muoopr/(dn+TINY);
                eigv   += C*gam_tmp/rho[c1]*muoopr*area[i]/vol[c1];
            }
            dt[c1] = std::min(dt[c1],dn/eigv);
        }
    }
    //nIFaces:
#ifdef MPICH    
    for (IntType i = pfacenum; i < nBFace; i++) {
        c1    = f2c[i+i];
        
        c2tmp = gam*(p[c1]+p_bar)/rho[c1];
        dn    = fabs((xfc[i]-xcc[c1])*xfn[i]+(yfc[i]-ycc[c1])*yfn[i]+(zfc[i]-zcc[c1])*zfn[i]);

        vn    = u[c1]*xfn[i]+v[c1]*yfn[i]+w[c1]*zfn[i];
        if(!steady) vn -= vgn[i];
        vn    = fabs(vn);
        eigv = vn+sqrt(c2tmp);
        
        if(vis_run){
            muoopr = vis_l[c1]/prl + vis_t[c1]/prt;
            gam_tmp = gam;
           
            //eigv += C*gam_tmp/rho[c1]*muoopr/(dn+TINY);
            eigv   += C*gam_tmp/rho[c1]*muoopr*area[i]/vol[c1];
        }
        dt[c1] = std::min(dt[c1],dn/eigv);

    }
#endif    
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
            IntType  c1, c2;
            RealFlow eigv, dn, vn, c2tmp, gam_tmp;
            c1    = f2c[i+i];
            c2    = f2c[i+i+1];
        
            c2tmp = gam*(p[c1]+p_bar)/rho[c1];
            dn    = fabs((xfc[i]-xcc[c1])*xfn[i]+(yfc[i]-ycc[c1])*yfn[i]+(zfc[i]-zcc[c1])*zfn[i]);
        
            vn    = u[c1]*xfn[i]+v[c1]*yfn[i]+w[c1]*zfn[i];
            if(!steady) vn -= vgn[i];
            vn    = fabs(vn);
            eigv = vn+sqrt(c2tmp);
        
            if(vis_run){
                muoopr = vis_l[c1]/prl + vis_t[c1]/prt;
         
                gam_tmp = gam;
            
                //eigv += C*gam_tmp/rho[c1]*muoopr/dn;
                eigv   += C*gam_tmp/rho[c1]*muoopr*area[i]/vol[c1];
            }
            dt[c1] = std::min(dt[c1],dn/eigv);
        
            c2tmp = gam*(p[c2]+p_bar)/rho[c2];
            dn    = fabs((xfc[i]-xcc[c2])*xfn[i]+(yfc[i]-ycc[c2])*yfn[i]+(zfc[i]-zcc[c2])*zfn[i]);
        
            vn    = u[c2]*xfn[i]+v[c2]*yfn[i]+w[c2]*zfn[i];
            if(!steady) vn -= vgn[i];
            vn    = fabs(vn);
            eigv = vn+sqrt(c2tmp);
     
            if(vis_run){
                muoopr = vis_l[c2]/prl + vis_t[c2]/prt;
                gam_tmp = gam;
            
                //eigv += C*gam_tmp/rho[c2]*muoopr/dn;
                eigv   += C*gam_tmp/rho[c2]*muoopr*area[i]/vol[c2];
            }
            dt[c2] = std::min(dt[c2],dn/eigv);
        }
    }
    mfmem::sdel_array_1D(grid_bfacegroup);
    mfmem::sdel_array_1D(grid_ifacegroup);
#elif (defined FS_OPENMP) && (defined Reduction)//Manual reduction
    RealFlow* tmp_dt = NULL;
    IntType* nFPC = CalnFPC(grid);
    IntType** C2F = CalC2F(grid);
    mfmem::snew_array_1D(tmp_dt, 2 * nTFace, dmrfl);
    IntType j, face, count;
#pragma omp parallel for private(i,count,c1,eigv,dn,vn,c2tmp,gam_tmp,muoopr)
    for (i = 0; i < nBFace; i++) {
        count = 2 * i;
        c1 = f2c[count];

        c2tmp = gam * (p[c1] + p_bar) / rho[c1];
        dn = fabs((xfc[i] - xcc[c1]) * xfn[i] + (yfc[i] - ycc[c1]) * yfn[i] + (zfc[i] - zcc[c1]) * zfn[i]);

        vn = u[c1] * xfn[i] + v[c1] * yfn[i] + w[c1] * zfn[i];
        if (!steady) vn -= vgn[i];
        vn = fabs(vn);
        eigv = vn + sqrt(c2tmp);

        if (vis_run) {
            muoopr = vis_l[c1] / prl + vis_t[c1] / prt;
            gam_tmp = gam;

            //eigv += C*gam_tmp/rho[c1]*muoopr/(dn+TINY);
            eigv += C * gam_tmp / rho[c1] * muoopr * area[i] / vol[c1];
        }
        tmp_dt[count] = dn / eigv;
    }
    // For interior faces
#pragma omp parallel for private(i,count,c1,c2,eigv,dn,vn,c2tmp,gam_tmp,muoopr)
    for (i = nBFace; i < nTFace; i++) {
        count = 2 * i;
        c1 = f2c[count];
        c2 = f2c[count + 1];

        c2tmp = gam * (p[c1] + p_bar) / rho[c1];
        dn = fabs((xfc[i] - xcc[c1]) * xfn[i] + (yfc[i] - ycc[c1]) * yfn[i] + (zfc[i] - zcc[c1]) * zfn[i]);

        vn = u[c1] * xfn[i] + v[c1] * yfn[i] + w[c1] * zfn[i];
        if (!steady) vn -= vgn[i];
        vn = fabs(vn);
        eigv = vn + sqrt(c2tmp);

        if (vis_run) {
            muoopr = vis_l[c1] / prl + vis_t[c1] / prt;

            gam_tmp = gam;

            //eigv += C*gam_tmp/rho[c1]*muoopr/dn;
            eigv += C * gam_tmp / rho[c1] * muoopr * area[i] / vol[c1];
        }
        tmp_dt[count] = dn / eigv;

        c2tmp = gam * (p[c2] + p_bar) / rho[c2];
        dn = fabs((xfc[i] - xcc[c2]) * xfn[i] + (yfc[i] - ycc[c2]) * yfn[i] + (zfc[i] - zcc[c2]) * zfn[i]);

        vn = u[c2] * xfn[i] + v[c2] * yfn[i] + w[c2] * zfn[i];
        if (!steady) vn -= vgn[i];
        vn = fabs(vn);
        eigv = vn + sqrt(c2tmp);

        if (vis_run) {
            muoopr = vis_l[c2] / prl + vis_t[c2] / prt;
            gam_tmp = gam;

            //eigv += C*gam_tmp/rho[c2]*muoopr/dn;
            eigv += C * gam_tmp / rho[c2] * muoopr * area[i] / vol[c2];
        }
        tmp_dt[count + 1] = dn / eigv;
    }
#pragma omp parallel for private(i,j,count,c1,c2,face)
    for (i = 0; i < nTCell; i++) {
        for (j = 0; j < nFPC[i]; j++) {
            face = C2F[i][j];
            count = 2 * face;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            if (i == c1) {
                dt[c1] = std::min(dt[c1], tmp_dt[count]);
            }
            else if (i == c2) {
                dt[c2] = std::min(dt[c2], tmp_dt[count + 1]);
            }
            else {
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
    }
    mfmem::sdel_array_1D(tmp_dt);
#elif (defined FS_OPENMP) && (defined DIVREP)//Division & replication
    IntType threads = grid->threads;
    IntType startFace, endFace, t, k, count, face;
    RealFlow tmp_dt1, tmp_dt2;
    if (grid->DivRepSuccess) {
    #pragma omp parallel for private(t,i,k,startFace,endFace,count,c1,c2,face,eigv,dn,vn,c2tmp,gam_tmp,muoopr,tmp_dt1,tmp_dt2)
        for (t = 0; t < threads; t++) {
            //Boundary faces
            startFace = grid->idx_pthreads_bface[t];
            endFace = grid->idx_pthreads_bface[t + 1];
            for (i = startFace; i < endFace; i++) {
                face = grid->id_division_bface[i];
                count = 2 * face;
                c1 = f2c[count];
                c2tmp = gam * (p[c1] + p_bar) / rho[c1];
                dn = fabs((xfc[face] - xcc[c1]) * xfn[face] + (yfc[face] - ycc[c1]) * yfn[face] + (zfc[face] - zcc[c1]) * zfn[face]);

                vn = u[c1] * xfn[face] + v[c1] * yfn[face] + w[c1] * zfn[face];
                if (!steady) vn -= vgn[face];
                vn = fabs(vn);
                eigv = vn + sqrt(c2tmp);

                if (vis_run) {
                    muoopr = vis_l[c1] / prl + vis_t[c1] / prt;
                    gam_tmp = gam;

                    //eigv += C*gam_tmp/rho[c1]*muoopr/(dn+TINY);
                    eigv += C * gam_tmp / rho[c1] * muoopr * area[face] / vol[c1];
                }
                dt[c1] = std::min(dt[c1], dn / eigv);
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
                c2tmp = gam * (p[c1] + p_bar) / rho[c1];
                dn = fabs((xfc[face] - xcc[c1]) * xfn[face] + (yfc[face] - ycc[c1]) * yfn[face] + (zfc[face] - zcc[c1]) * zfn[face]);

                vn = u[c1] * xfn[face] + v[c1] * yfn[face] + w[c1] * zfn[face];
                if (!steady) vn -= vgn[face];
                vn = fabs(vn);
                eigv = vn + sqrt(c2tmp);

                if (vis_run) {
                    muoopr = vis_l[c1] / prl + vis_t[c1] / prt;

                    gam_tmp = gam;

                    //eigv += C*gam_tmp/rho[c1]*muoopr/dn;
                    eigv += C * gam_tmp / rho[c1] * muoopr * area[face] / vol[c1];
                }
                tmp_dt1 = dn / eigv;

                c2tmp = gam * (p[c2] + p_bar) / rho[c2];
                dn = fabs((xfc[face] - xcc[c2]) * xfn[face] + (yfc[face] - ycc[c2]) * yfn[face] + (zfc[face] - zcc[c2]) * zfn[face]);

                vn = u[c2] * xfn[face] + v[c2] * yfn[face] + w[c2] * zfn[face];
                if (!steady) vn -= vgn[face];
                vn = fabs(vn);
                eigv = vn + sqrt(c2tmp);

                if (vis_run) {
                    muoopr = vis_l[c2] / prl + vis_t[c2] / prt;
                    gam_tmp = gam;

                    //eigv += C*gam_tmp/rho[c2]*muoopr/dn;
                    eigv += C * gam_tmp / rho[c2] * muoopr * area[face] / vol[c2];
                }
                tmp_dt2 = dn / eigv;
                
                if (abs(k) < nTFace) {// write back to c1 & c2
                    dt[c1] = MIN(dt[c1], tmp_dt1);
                    dt[c2] = MIN(dt[c2], tmp_dt2);
                }
                else {
                    if (k > 0) {//just write back to c1
                        dt[c1] = MIN(dt[c1], tmp_dt1);
                    }
                    else {//just write back to c2
                        dt[c2] = MIN(dt[c2], tmp_dt2);
                    }
                }
            }
        }
    }
#elif (defined FS_OPENMP) && (defined DIVCON) //D&C TREE
    RealFlow* tmp_dt = NULL;
    mfmem::snew_array_1D(tmp_dt, 2 * (nTFace - nBFace), dmrfl);
#pragma omp parallel
    {
    #pragma omp single nowait
        tree_traversal(grid->treeHead, dt, tmp_dt, f2c, nBFace, vis_run, p, xfn, yfn, zfn, xfc, yfc, zfc,
            xcc, ycc, zcc, rho, u, v, w, vgn, area, vol, gam, p_bar, steady, C, vis_l, vis_t, prl, prt);
    }
    mfmem::sdel_array_1D(tmp_dt);
#elif (defined FS_OPENMP) && (defined DC) //DC

    uTaskTree *uTaskTreeRoot = grid->GetuTaskTree();
  
    if(vis_run){
        char* userArgs[27] = {(char *)grid, (char *)dt, (char *)f2c, (char *)xfn,(char *)yfn, (char *)zfn, (char *)xfc, (char *)yfc, 
                            (char *)zfc, (char *)xcc, (char *)ycc, (char *)zcc, (char *)vgn, (char *)area,(char *)vol,
                            (char *)rho, (char *)u, (char *)v, (char *)w, (char *)p, (char *)&steady, (char *)&gam,
                            (char *)&p_bar, (char *)vis_l, (char *)vis_t, (char *)&prl,(char *)&prt};

        uTaskTreeRoot->task_traversal(TimeStepNormal_new_Kernel1, NULL, userArgs, Forward);  
    }else{
        char* userArgs[23] = {(char *)grid, (char *)dt, (char *)f2c, (char *)xfn,(char *)yfn, (char *)zfn, (char *)xfc, (char *)yfc, 
                            (char *)zfc, (char *)xcc, (char *)ycc, (char *)zcc, (char *)vgn, (char *)area,(char *)vol,
                            (char *)rho, (char *)u, (char *)v, (char *)w, (char *)p, (char *)&steady, (char *)&gam,
                            (char *)&p_bar};

        uTaskTreeRoot->task_traversal(TimeStepNormal_new_Kernel2, NULL, userArgs, Forward);
    }
#else

    // For boundary faces
    for(i=0; i<nBFace; i++){
        c1    = f2c[i+i];
        
        c2tmp = gam*(p[c1]+p_bar)/rho[c1];
        dn    = fabs((xfc[i]-xcc[c1])*xfn[i]+(yfc[i]-ycc[c1])*yfn[i]+(zfc[i]-zcc[c1])*zfn[i]);

        vn    = u[c1]*xfn[i]+v[c1]*yfn[i]+w[c1]*zfn[i];
        if(!steady) vn -= vgn[i];
        vn    = fabs(vn);
        eigv = vn+sqrt(c2tmp);
        
        if(vis_run){
            muoopr = vis_l[c1]/prl + vis_t[c1]/prt;
            gam_tmp = gam;
           
            //eigv += C*gam_tmp/rho[c1]*muoopr/(dn+TINY);
            eigv   += C*gam_tmp/rho[c1]*muoopr*area[i]/vol[c1];
        }
        dt[c1] = std::min(dt[c1],dn/eigv);
    }
    // For interior faces
    for(i=nBFace; i<nTFace; i++){
        c1    = f2c[i+i];
        c2    = f2c[i+i+1];
        
        c2tmp = gam*(p[c1]+p_bar)/rho[c1];
        dn    = fabs((xfc[i]-xcc[c1])*xfn[i]+(yfc[i]-ycc[c1])*yfn[i]+(zfc[i]-zcc[c1])*zfn[i]);
        
        vn    = u[c1]*xfn[i]+v[c1]*yfn[i]+w[c1]*zfn[i];
        if(!steady) vn -= vgn[i];
        vn    = fabs(vn);
        eigv = vn+sqrt(c2tmp);
        
        if(vis_run){
            muoopr = vis_l[c1]/prl + vis_t[c1]/prt;
         
            gam_tmp = gam;
            
            //eigv += C*gam_tmp/rho[c1]*muoopr/dn;
            eigv   += C*gam_tmp/rho[c1]*muoopr*area[i]/vol[c1];
        }
        dt[c1] = std::min(dt[c1],dn/eigv);
        
        c2tmp = gam*(p[c2]+p_bar)/rho[c2];
        dn    = fabs((xfc[i]-xcc[c2])*xfn[i]+(yfc[i]-ycc[c2])*yfn[i]+(zfc[i]-zcc[c2])*zfn[i]);
        
        vn    = u[c2]*xfn[i]+v[c2]*yfn[i]+w[c2]*zfn[i];
        if(!steady) vn -= vgn[i];
        vn    = fabs(vn);
        eigv = vn+sqrt(c2tmp);
     
        if(vis_run){
            muoopr = vis_l[c2]/prl + vis_t[c2]/prt;
            gam_tmp = gam;
            
            //eigv += C*gam_tmp/rho[c2]*muoopr/dn;
            eigv   += C*gam_tmp/rho[c2]*muoopr*area[i]/vol[c2];
        }
        dt[c2] = std::min(dt[c2],dn/eigv);
    }
#endif

}


/*******************************************************************************\
    Drive functions to calculate residuals for different orders and dimensions
\*******************************************************************************/
void UpdateResiduals(PolyGrid *grid, IntType level)
{
    IntType vis_mode, order, steady;
    grid->GetData(&steady, INT, 1, "steady");
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    grid->GetData(&order, INT, 1, "order");
 
    RealFlow **limit = NULL;
    
    if(level == 0 && order != FIRST_ORDER) {

#ifdef TIMECOST//dingxin
#ifdef FS_CUDA
		cudaDeviceSynchronize();
#endif
#ifdef MPICH
        double time_tmp;
        time_tmp = -MPI_Wtime();
#else
        struct timeval starttimeTemLimiter, endtimeTemLimiter;
        double timeuseTemLimiter;
        gettimeofday(&starttimeTemLimiter, 0); 
#endif
#endif
#ifdef FS_CUDA
        limit = cuGetLimiters_resp(grid);
#else
		limit = GetLimiters_resp(grid);
#endif
		
#ifdef TIMECOST//dingxin
#ifdef FS_CUDA
		cudaDeviceSynchronize();
#endif
#ifdef MPICH
        timecost[3] = timecost[3] + time_tmp + MPI_Wtime();
#else
        gettimeofday(&endtimeTemLimiter, 0); 
        timeuseTemLimiter = (RealGeom) 1000000*(endtimeTemLimiter.tv_sec - starttimeTemLimiter.tv_sec) + endtimeTemLimiter.tv_usec - starttimeTemLimiter.tv_usec;
        timecost[3] += timeuseTemLimiter;
        timeuseTemLimiter /= 1000000.0;
        time_limiter += timeuseTemLimiter;
#endif
#endif

#if (defined FS_CUDA)||(defined FS_CUDA_DEBUG_NS_Flux)
	#if (defined LOOPMERGE)
		#if defined MultiStream
			IntType vis_mode;
			grid->GetData(&vis_mode, INT, 1, "vis_mode");
			if(vis_mode != INVISCID) {
				// need calculate gradient T for vis flux
				// call gradient T MPI trans module which includes of InviscidFlux calculation
				grid->cuRecvSendVarNeighbor_TogethForGradient_T_InVis(1);
			}
			else{
				cuInviscidFlux_merge(grid, limit, level);
			}
		#else			
			cuInviscidFlux_merge(grid, limit, level);
		#endif		
	#else
		#if defined MultiStream
			IntType vis_mode;
			grid->GetData(&vis_mode, INT, 1, "vis_mode");
			if(vis_mode != INVISCID) {
				// need calculate gradient T for vis flux
				// call gradient T MPI trans module which includes of InviscidFlux calculation
				grid->cuRecvSendVarNeighbor_TogethForGradient_T_InVis(1);
			}
			else{
				cuInviscidFlux(grid, limit, level);
			}
		#else
			cuInviscidFlux(grid, limit, level);
		#endif
	#endif
#else
		InviscidFlux(grid, limit, level);
#endif

        //计算粘性通量
        if(vis_mode != INVISCID) {
            //未修该overlap
#ifdef FS_CUDA	
	#if (defined LOOPMERGE)
			cuViscousFlux_merge(grid, level);
	#else
			cuViscousFlux(grid, level);
	#endif
#else
			ViscousFlux(grid, level);
#endif		
        }
    } else {
        // First order computation
        InviscidFlux(grid, limit, level);

        IntType vis_run = 0;
        if(vis_mode != INVISCID) {
            vis_run = 1;
            // if coarse grid doesn't want to run the viscous flux, turn it off
            if(level != 0) {
                IntType cg_vis = 1, cg_visflux=1;
                grid->GetData(&cg_vis, INT, 1, "cg_vis"); 
                grid->GetData(&cg_visflux, INT, 1, "cg_visflux"); 
                if(cg_vis == 0) {
                    vis_run = 0;
                }else if((cg_vis==1) && (cg_visflux==0)){
                    vis_run = 2;
                }
            }
        }
        if(vis_run==0){
            //Don't run the viscous flux, and the vis_l
        }else if(vis_run==1){
            ComputeVis_l(grid);
            ViscousFlux(grid, level); 
        }else if(vis_run==2){
            ComputeVis_l(grid);
        } 
    }

    //在残差中增加非定常效应
    if(!steady) AddUnstSource(grid);
	
	mfmem::sdel_array_2D(limit);
	
/*  	IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nT5    = 5*nTCell;
	IntType n = nTCell + nBFace;
	RealFlow *res[5];
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");
    res[1] = &res[0][nTCell];
    res[2] = &res[1][nTCell];
    res[3] = &res[2][nTCell];
    res[4] = &res[3][nTCell];
	
	RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
	
	RealFlow **dqdx = NULL, **dqdy = NULL, **dqdz = NULL;
	IntType kNVar = 5;
    mfmem::snew_array_1D(dqdx, kNVar, dmrfl);
    mfmem::snew_array_1D(dqdy, kNVar, dmrfl);
    mfmem::snew_array_1D(dqdz, kNVar, dmrfl);
    dqdx[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdx"));
    dqdy[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdy"));
    dqdz[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdz"));
    for (IntType i = 1; i < kNVar; ++i) {
        dqdx[i] = &dqdx[i - 1][n];
        dqdy[i] = &dqdy[i - 1][n];
        dqdz[i] = &dqdz[i - 1][n];
    }

#ifdef FS_CUDA	
	HANDLE_API_ERR(cudaMemcpy(q[0], gq, n*sizeof(RealFlow), cudaMemcpyDeviceToHost));			
	HANDLE_API_ERR(cudaMemcpy(q[1], &gq[1*n], n*sizeof(RealFlow), cudaMemcpyDeviceToHost));	
	HANDLE_API_ERR(cudaMemcpy(q[2], &gq[2*n], n*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(q[3], &gq[3*n], n*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(q[4], &gq[4*n], n*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	
	HANDLE_API_ERR(cudaMemcpy(dqdx[0], gdqdx, 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(dqdy[0], gdqdy, 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
	HANDLE_API_ERR(cudaMemcpy(dqdz[0], gdqdz, 5*(gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyDeviceToHost));
#endif
	
	IntType mpirank = 0;
#ifdef MPICH        
    MPI_Comm_rank(MPI_COMM_WORLD, & mpirank);
#endif	
	if (mpirank == 0){
		ofstream output;
		output.precision(16);
		output.open("RKq.dat");
		for(IntType j = 0; j < nTCell; j++){
			for(IntType i = 0; i < 5; i++){
				output << dqdx[i][j] << ", ";
			}
			output << endl;
		}
		output.close();
		exit(0);
	}   */
	
}


/*******************************************************************************\
               Set flow variables at ghost cells                                
\*******************************************************************************/
void SetGhostVariables(PolyGrid *grid)
{
    IntType  i, c1, c2, count, steady, vis_mode, type, wmark;
    IntType  nBFace = grid->GetNBFace();
    IntType  nTCell = grid->GetNTCell();
    IntType  *f2c   = grid->Getf2c();
    IntType  n      = nTCell + nBFace;
    RealFlow *rho   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    RealFlow *u     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    RealFlow *v     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    RealFlow *w     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    RealFlow *p     = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");

    RealGeom *xfn   = grid->GetXfn();
    RealGeom *yfn   = grid->GetYfn();
    RealGeom *zfn   = grid->GetZfn();
    RealGeom *vgn   = grid->GetFaceNormalVelocity();
    RealGeom *BFacevgx   = grid->GetBoundaryFaceVelocityX();
    RealGeom *BFacevgy   = grid->GetBoundaryFaceVelocityY();
    RealGeom *BFacevgz   = grid->GetBoundaryFaceVelocityZ();
    BCRecord **bcr  = grid->Getbcr();

    
    RealFlow vn;
    RealFlow rho00, u00, v00, w00, p00, vtx, vty, vtz, riemp, riemm;
    RealFlow vnp, vnm, cp, cm, gam, p_bar, p_t, yc,entr,vnb,cb;
    RealFlow gascon;
    RealFlow rhow;
    
    
    RealFlow rhom,um,vm,wm,pm,rhop,up,vp,wp,pp;
    RealFlow tw;
    
  
    grid->GetData(&rho00, REAL_FLOW, 1, "rho");
    grid->GetData(&u00, REAL_FLOW, 1, "u");
    grid->GetData(&v00, REAL_FLOW, 1, "v");
    grid->GetData(&w00, REAL_FLOW, 1, "w");
    grid->GetData(&p00, REAL_FLOW, 1, "p");
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
   
    RealFlow norm_of_uvw = sqrt( u00*u00 + v00*v00 + w00*w00 );
    RealFlow eps_of_farfield_vn = 0.0;
    grid->GetData(&eps_of_farfield_vn, REAL_FLOW, 1, "eps_of_farfield_vn",0);

    RealFlow rho_min,rho_max,p_min,p_max;
    grid->GetData(&rho_min, REAL_FLOW, 1, "rho_min");
    grid->GetData(&rho_max, REAL_FLOW, 1, "rho_max");
    grid->GetData(&p_min,   REAL_FLOW, 1, "p_min");
    grid->GetData(&p_max,   REAL_FLOW, 1, "p_max");
    
    grid->GetData(&steady,  INT, 1, "steady");
    grid->GetData(&vis_mode,INT, 1, "vis_mode");
    grid->GetData(&gam,     REAL_FLOW, 1, "gam");
    grid->GetData(&gascon,  REAL_FLOW, 1, "gascon"); 
    
  
    
    
  
    count = 0;
    for(i=0; i<nBFace; i++){
        type  = bcr[i]->GetType();
        c1    = f2c[count++];
        c2    = f2c[count++];
        wmark = 0;

        // Do nothing for interfaces.
        if(type == INTERFACE) continue;
 
        // Assign the variable values for each ghost cell whose index is c2.
        switch(type) {
            case WALL:
                p[c2]   = p[c1];
                rho[c2] = rho[c1];
                
                if(vis_mode == INVISCID){
                    vn    = 2.*(xfn[i]*u[c1] + yfn[i]*v[c1] + zfn[i]*w[c1]);
                    if(!steady){
                        vn -= 2*vgn[i];
                    }
                    u[c2] = u[c1] - vn*xfn[i];
                    v[c2] = v[c1] - vn*yfn[i];
                    w[c2] = w[c1] - vn*zfn[i];
                }else{
                    if(steady){
                        u[c2] = -u[c1];
                        v[c2] = -v[c1];
                        w[c2] = -w[c1];
                    }else{
                        u[c2] = -u[c1] + 2.*BFacevgx[i];
                        v[c2] = -v[c1] + 2.*BFacevgy[i];
                        w[c2] = -w[c1] + 2.*BFacevgz[i];
                    }
                    
                    //viscous adiabatic wall
                    //nothing!!!
                    
                    //viscous iso-thermal wall
                    tw = -1.0;
                    bcr[i]->GetBCVar(&tw, REAL_FLOW, "tw",0);
                    if(tw>0.0){
                        rhow = (p[c2]+p_bar)/gascon/tw;
                        rho[c2] = 2.0*rhow-rho[c1];
                        if(rho[c2]<0.0){
                            rho[c2] = rhow;
                        }
                    }
                }
                break;

            case SYMM:
                rho[c2] = rho[c1];
                p[c2]   = p[c1];
                vn      = 2.*(xfn[i]*u[c1] + yfn[i]*v[c1] + zfn[i]*w[c1]);
                if(!steady){        //zhyb:对称面vgn为0，此处本来可以不考虑。但是在粘性计算时，有时可能会采用对称边界条件表示无粘的物面，
                    vn -= 2*vgn[i]; //因此在此需要加上非定常的情况
                }
                u[c2]   = u[c1] - vn*xfn[i];
                v[c2]   = v[c1] - vn*yfn[i];
                w[c2]   = w[c1] - vn*zfn[i];
                break;    

            case FAR_FIELD:
                um = u00;
                vm = v00;
                wm = w00;
                up = u[c1];
                vp = v[c1];
                wp = w[c1];
                if(!steady){
                    um -= BFacevgx[i];
                    vm -= BFacevgy[i];
                    wm -= BFacevgz[i];
                    up -= BFacevgx[i];
                    vp -= BFacevgy[i];
                    wp -= BFacevgz[i];
                }
                rhom = rho00;
                pm = p00+p_bar;
                rhop = rho[c1];
                pp = p[c1]+p_bar;
                
                vnm = xfn[i]*um+yfn[i]*vm+zfn[i]*wm;
                vnp = xfn[i]*up+yfn[i]*vp+zfn[i]*wp;
                cm  = sqrt(gam*pm/rhom);
                cp  = sqrt(gam*pp/rhop);
                riemm = vnm - 2.*cm/(gam-1.);
                riemp = vnp + 2.*cp/(gam-1.);
                
                vnb = 0.5*(riemp+riemm);
                cb  = 0.25*(riemp-riemm)*(gam-1.);
              
                if(fabs(vnb/cb)>1.){  //supersonic
                    if(vnb<=0.0){  //inlet
                        rho[c2] = rhom;
                        u[c2]   = um;
                        v[c2]   = vm;
                        w[c2]   = wm;
                        p[c2]   = pm;
                    }else{   //exit
                        rho[c2] = rhop;
                        u[c2]   = up;
                        v[c2]   = vp;
                        w[c2]   = wp;
                        p[c2]   = pp;
                    }
                }else{ //subsonic
                    RealFlow rela_vnb = vnb / norm_of_uvw;
                    if(rela_vnb <= -eps_of_farfield_vn) {  //inlet
                        vtx = um - vnm*xfn[i];
                        vty = vm - vnm*yfn[i];
                        vtz = wm - vnm*zfn[i];
                        entr = pm/pow(rhom, gam);

                        rho[c2] = pow((cb*cb/(entr*gam)),RealFlow(1./(gam-1.))); 
                        u[c2]   = vtx + vnb*xfn[i];
                        v[c2]   = vty + vnb*yfn[i];
                        w[c2]   = vtz + vnb*zfn[i];
                        p[c2]   = cb*cb*rho[c2]/gam;
                    } else if(rela_vnb>eps_of_farfield_vn) {  //exit
                        vtx = up - vnp*xfn[i];
                        vty = vp - vnp*yfn[i];
                        vtz = wp - vnp*zfn[i];
                        entr = pp/pow(rhop, gam);

                        rho[c2] = pow((cb*cb/(entr*gam)),RealFlow(1./(gam-1.))); 
                        u[c2]   = vtx + vnb*xfn[i];
                        v[c2]   = vty + vnb*yfn[i];
                        w[c2]   = vtz + vnb*zfn[i];
                        p[c2]   = cb*cb*rho[c2]/gam;
                    } else {

                        rho[c2] = 0.5*(rhop+rhom);
                        u[c2]   = 0.5*(up+um);
                        v[c2]   = 0.5*(vp+vm);
                        w[c2]   = 0.5*(wp+wm);
                        p[c2]   = 0.5*(pp+pm);
                    }

                    /*if(vnb<=0.0){  //inlet
                        vtx = um - vnm*xfn[i];
                        vty = vm - vnm*yfn[i];
                        vtz = wm - vnm*zfn[i];
                        entr = pm/pow(rhom, gam);
                    }else{  //exit
                        vtx = up - vnp*xfn[i];
                        vty = vp - vnp*yfn[i];
                        vtz = wp - vnp*zfn[i];
                        entr = pp/pow(rhop, gam); 
                    }
                    rho[c2] = pow((cb*cb/(entr*gam)),RealFlow(1./(gam-1.))); 
                    u[c2]   = vtx + vnb*xfn[i];
                    v[c2]   = vty + vnb*yfn[i];
                    w[c2]   = vtz + vnb*zfn[i];
                    p[c2]   = cb*cb*rho[c2]/gam;*/
                }
                
                rho[c2] = 2*rho[c2] - rhop;
                u[c2]   = 2*u[c2] - up;
                v[c2]   = 2*v[c2] - vp;
                w[c2]   = 2*w[c2] - wp;
                p[c2]   = 2*p[c2] - pp;
                p[c2]  -= p_bar;
                
                if(!steady){
                    u[c2] += BFacevgx[i];
                    v[c2] += BFacevgy[i];
                    w[c2] += BFacevgz[i];
                }
                break; 
                
            default:
                printf("Error in SetGhostVariables 001\n");
                break;
        }
        
        //ZHYB:对c2单元的rho和p做限制，不能为负，不能大于10倍的驻点值
        rho[c2] = std::max(rho[c2],rho_min);
        rho[c2] = std::min(rho[c2],rho_max);
        p[c2] = std::max(p[c2],p_min);
        p[c2] = std::min(p[c2],p_max);
    }

}

/// \brief  设置虚网格的温度梯度值，注意：支持的边界类型较少，例如没有发动机
///         出入口边界
/// \par    Update records:
/// <pre>
/// Date        Author      Description
/// 2021-09-30  王新建      编写函数
/// </pre>
void SetGhostTemperatureGradient(
    const PolyGrid *grid, RealFlow *dtdx, RealFlow *dtdy, RealFlow *dtdz
) {
    const IntType nTCell = grid->GetNTCell();
    const IntType nBFace = grid->GetNBFace();
    const IntType n = nTCell + nBFace;

    const IntType *f2c = grid->Getf2c();
    const RealGeom *xfn = grid->GetXfn();
    const RealGeom *yfn = grid->GetYfn();
    const RealGeom *zfn = grid->GetZfn();
    const RealGeom *xcc = grid->GetXcc();
    const RealGeom *ycc = grid->GetYcc();
    const RealGeom *zcc = grid->GetZcc();
    BCRecord **bcr = grid->Getbcr();

    IntType count = 0;
    for (IntType i = 0; i < nBFace; ++i) {
        const IntType type = bcr[i]->GetType();
        const IntType c1 = f2c[count++];
        const IntType c2 = f2c[count++];

        RealFlow tw = -1.0;
        RealFlow dta[3], dnn[3], dnnn;
        switch (type) {
            case INTERFACE:
                break;
            case WALL:
                dnnn = 
                    dtdx[c1] * xfn[i] + dtdy[c1] * yfn[i] + dtdz[c1] * zfn[i];
                dnn[0] = dnnn * xfn[i];
                dnn[1] = dnnn * yfn[i];
                dnn[2] = dnnn * zfn[i];
                dta[0] = dtdx[c1] - dnn[0];
                dta[1] = dtdy[c1] - dnn[1];
                dta[2] = dtdz[c1] - dnn[2];
                dtdx[c2] = dta[0] - dnn[0];
                dtdy[c2] = dta[1] - dnn[1];
                dtdz[c2] = dta[2] - dnn[2];
                
                bcr[i]->GetBCVar(&tw, REAL_FLOW, "tw", 0);
                if (tw > 0.0) {
                    dtdx[c2] = -dta[0] + dnn[0];
                    dtdy[c2] = -dta[1] + dnn[1];
                    dtdz[c2] = -dta[2] + dnn[2];
                }
                break;
            case SYMM:
                dnnn = 
                    dtdx[c1] * xfn[i] + dtdy[c1] * yfn[i] + dtdz[c1] * zfn[i];
                dnn[0] = dnnn * xfn[i];
                dnn[1] = dnnn * yfn[i];
                dnn[2] = dnnn * zfn[i];
                dta[0] = dtdx[c1] - dnn[0];
                dta[1] = dtdy[c1] - dnn[1];
                dta[2] = dtdz[c1] - dnn[2];
                dtdx[c2] = dta[0] - dnn[0];
                dtdy[c2] = dta[1] - dnn[1];
                dtdz[c2] = dta[2] - dnn[2];
                break;          
            case FAR_FIELD:
                dtdx[c2] = 0.0;
                dtdy[c2] = 0.0;
                dtdz[c2] = 0.0;
                break;
            default:
                dtdx[c2] = 0.0;
                dtdy[c2] = 0.0;
                dtdz[c2] = 0.0;
                break;
        }
    }
}

/*******************************************************************************\
           Solve equation for symmetry boundary ghost cell's gradient 
\*******************************************************************************/
void SolveEquationforGradSYMM(RealFlow gv1[3][3], RealFlow gv2[3][3], RealGeom xfn, RealGeom yfn, RealGeom zfn)
{
    RealGeom dtmp;
    RealGeom xft1,yft1,zft1,xft2,yft2,zft2;
    RealFlow gradvn1[3],gradvt11[3],gradvt21[3],gradvn2[3],gradvt12[3],gradvt22[3]; 
    
    
    // Get first tangential vector on the face
    if(xfn != 0.) {
        xft1 =  yfn;
        yft1 = -xfn;
        zft1 =  0.;
    } else if(yfn != 0.) {
        xft1 = -yfn;
        yft1 =  xfn;
        zft1 =  0.;
    } else if(zfn != 0.) {
        xft1 =  0.;
        yft1 = -zfn;
        zft1 =  yfn;
    } else {
        printf("Warning: Face is singular\n");
    }
    // normalize the tangential vector
    dtmp = sqrt(xft1*xft1 + yft1*yft1 + zft1*zft1);
    xft1 /= dtmp;
    yft1 /= dtmp;
    zft1 /= dtmp;
    
    // Get second tangential vector by cross dot t1 to normal
    xft2 = yfn*zft1 - zfn*yft1;
    yft2 = zfn*xft1 - xfn*zft1;
    zft2 = xfn*yft1 - yfn*xft1;
    
    gradvn1[0]  = gv1[0][0]*xfn+gv1[1][0]*yfn+gv1[2][0]*zfn;
    gradvn1[1]  = gv1[0][1]*xfn+gv1[1][1]*yfn+gv1[2][1]*zfn;
    gradvn1[2]  = gv1[0][2]*xfn+gv1[1][2]*yfn+gv1[2][2]*zfn;
    gradvt11[0] = gv1[0][0]*xft1+gv1[1][0]*yft1+gv1[2][0]*zft1;
    gradvt11[1] = gv1[0][1]*xft1+gv1[1][1]*yft1+gv1[2][1]*zft1;
    gradvt11[2] = gv1[0][2]*xft1+gv1[1][2]*yft1+gv1[2][2]*zft1;
    gradvt21[0] = gv1[0][0]*xft2+gv1[1][0]*yft2+gv1[2][0]*zft2;
    gradvt21[1] = gv1[0][1]*xft2+gv1[1][1]*yft2+gv1[2][1]*zft2;
    gradvt21[2] = gv1[0][2]*xft2+gv1[1][2]*yft2+gv1[2][2]*zft2;
    dtmp = gradvn1[0]*xfn+gradvn1[1]*yfn+gradvn1[2]*zfn;
    gradvn2[0]  = 2.0*dtmp*xfn-gradvn1[0];
    gradvn2[1]  = 2.0*dtmp*yfn-gradvn1[1];
    gradvn2[2]  = 2.0*dtmp*zfn-gradvn1[2];
    dtmp = gradvt11[0]*xfn+gradvt11[1]*yfn+gradvt11[2]*zfn;
    gradvt12[0]  = gradvt11[0]-2.0*dtmp*xfn;
    gradvt12[1]  = gradvt11[1]-2.0*dtmp*yfn;
    gradvt12[2]  = gradvt11[2]-2.0*dtmp*zfn;
    dtmp = gradvt21[0]*xfn+gradvt21[1]*yfn+gradvt21[2]*zfn;
    gradvt22[0]  = gradvt21[0]-2.0*dtmp*xfn;
    gradvt22[1]  = gradvt21[1]-2.0*dtmp*yfn;
    gradvt22[2]  = gradvt21[2]-2.0*dtmp*zfn;
    
    gv2[0][0] = xfn*gradvn2[0]+xft1*gradvt12[0]+xft2*gradvt22[0];
    gv2[1][0] = yfn*gradvn2[0]+yft1*gradvt12[0]+yft2*gradvt22[0];
    gv2[2][0] = zfn*gradvn2[0]+zft1*gradvt12[0]+zft2*gradvt22[0];
    gv2[0][1] = xfn*gradvn2[1]+xft1*gradvt12[1]+xft2*gradvt22[1];
    gv2[1][1] = yfn*gradvn2[1]+yft1*gradvt12[1]+yft2*gradvt22[1];
    gv2[2][1] = zfn*gradvn2[1]+zft1*gradvt12[1]+zft2*gradvt22[1];
    gv2[0][2] = xfn*gradvn2[2]+xft1*gradvt12[2]+xft2*gradvt22[2];
    gv2[1][2] = yfn*gradvn2[2]+yft1*gradvt12[2]+yft2*gradvt22[2];
    gv2[2][2] = zfn*gradvn2[2]+zft1*gradvt12[2]+zft2*gradvt22[2];

}

/// \brief  在已经计算梯度的情况下获取网格的速度梯度
/// \par    Update records:
/// <pre>
/// Date        Author      Description
/// 2021-09-28  王新建      编写函数
/// </pre>
void GetVelocityGradient(PolyGrid *grid, RealFlow *dvdxout[3], RealFlow *dvdyout[3], RealFlow *dvdzout[3]) {
    const IntType kNVar = 5;
    const IntType n = grid->GetNTCell() + grid->GetNBFace();

    RealFlow *dqdx = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdx"));
    RealFlow *dqdy = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdy"));
    RealFlow *dqdz = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdz"));

    if (dqdx == NULL || dqdy == NULL || dqdz == NULL) {
        mflog::log.set_each_grid_out();
        mflog::log << "Error! No gradient has been calculated!\n";
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }

    for (IntType i = 0; i < 3; ++i) {
        dvdxout[i] = &dqdx[(i + 1) * n];
        dvdyout[i] = &dqdy[(i + 1) * n];
        dvdzout[i] = &dqdz[(i + 1) * n];
    }
}


/*******************************************************************************\
       Compute fluxes in 3D according to its type for preconditioning
\*******************************************************************************/
void CompInvFlux(PolyGrid *grid, RealFlow *ql[5],   RealFlow *qr[5], RealFlow *flux[5],
                 RealGeom *xfn,  RealGeom *yfn,     RealGeom *zfn,   RealGeom *area, 
                 RealGeom *vgn,  IntType *face_act, RealFlow gam,    RealFlow p_bar,  
                 RealFlow alf_l, RealFlow alf_n,    IntType type_flux, 
#ifdef DC
                 RealFlow gascon, IntType EntropyCorType,
                 IntType steady, IntType *IsShockFace, 
                 IntType *IsNormalFace,
#endif
                 IntType ns,     IntType ne)
{
#ifdef TIMECOST//dingxin
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemRoe, endtimeTemRoe;
    double timeuseTemRoe;
    gettimeofday(&starttimeTemRoe, 0); 
#endif
#endif
    RoeFlux_noprec(grid, ql, qr, flux, xfn, yfn, zfn,
                   area, face_act, gam, p_bar, alf_l, alf_n, 
#ifdef DC
                   gascon, EntropyCorType, steady, IsShockFace, IsNormalFace,
#endif
                   ns, ne);
#ifdef TIMECOST//dingxin
#ifdef MPICH
#else
    gettimeofday(&endtimeTemRoe, 0); 
    timeuseTemRoe = (RealGeom) 1000000*(endtimeTemRoe.tv_sec - starttimeTemRoe.tv_sec) + endtimeTemRoe.tv_usec - starttimeTemRoe.tv_usec;
    timeuseTemRoe /= 1000000.0;
    time_roe += timeuseTemRoe;
#endif
#endif
}


/*******************************************************************************\
       set ql and qr the values of q
\*******************************************************************************/
void SetQlQrWithQ(PolyGrid *grid, RealFlow *q[], RealFlow *ql[], RealFlow *qr[], IntType ns, IntType ne) 
{
    IntType  *f2c   = grid->Getf2c();
    IntType  nvar, i, c1, c2, count, n, face;

    nvar = 5; 
#if defined(FS_OPENMP) && !defined(DC) 
#pragma omp parallel for private(face,count,i,c1,c2,n)
    for (face = ns; face < ne; face++) {
        count = 2 * face;
        i = face - ns;
        c1 = f2c[count++];
        c2 = f2c[count];
        for (n = 0; n < nvar; n++) {
            ql[n][i] = q[n][c1];
            qr[n][i] = q[n][c2];
        }

    }
#else
    for (face = ns; face < ne; face++) {
        count = 2 * face;
        i = face - ns;
        c1 = f2c[count++];
        c2 = f2c[count];
        for (n = 0; n < nvar; n++) {
            ql[n][i] = q[n][c1];
            qr[n][i] = q[n][c2];
        }

    }
#endif
}
/*******************************************************************************\
            calculate ql and qr
\*******************************************************************************/
void ModQlQrBou(PolyGrid *grid, RealFlow *ql[], RealFlow *qr[], 
#ifdef DC
    RealFlow *q[],
    IntType steady,
#endif
    IntType ns, IntType ne)
{
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    RealGeom *xfn  = grid->GetXfn();
    RealGeom *yfn  = grid->GetYfn();
    RealGeom *zfn  = grid->GetZfn();
    IntType  *f2c  = grid->Getf2c();
    BCRecord **bcr = grid->Getbcr();
    RealGeom  *vgn = grid->GetFaceNormalVelocity();

#ifndef DC
    IntType steady = 1;
    grid->GetData(&steady, INT, 1, "steady");
    
    // Get flow variables
    RealFlow *q[5];
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
#endif

    IntType  i, j, face, nMid, type, c1, c2;
    RealFlow tem, vn;
      
    // Check if there are boundary faces. If no, return.
    if(ns >= nBFace) return;
    if(ne <= nBFace) {
        // If they are all boundary faces
        nMid = ne;
    }else{
        // Part of them are boundary faces
        nMid = nBFace;
    }
#if defined(FS_OPENMP) && !defined(DC) 
#pragma omp parallel for
#endif
    for(IntType face=ns; face<nMid; face++) {
        IntType  i, j, type, c1, c2;
        RealFlow vn, tem;
        type = bcr[face]->GetType();
        if (type == INTERFACE) continue;
        c1 = f2c[face+face];
        c2 = f2c[face+face+1];     
        i  = face - ns;
        if(type == SYMM){
            //rho
            qr[0][i] = ql[0][i];
            //u,v,w
            vn = ql[1][i]*xfn[face]+ql[2][i]*yfn[face]+ql[3][i]*zfn[face];
            if(!steady){         //zhyb:对称面vgn为0，此处本来可以不考虑。但是在粘性计算时，有时可能会采用对称边界条件表示无粘的物面，
                vn -= vgn[face]; //因此在此需要加上非定常的情况
            }
            qr[1][i] = ql[1][i]-2.0*vn*xfn[face];
            qr[2][i] = ql[2][i]-2.0*vn*yfn[face];
            qr[3][i] = ql[3][i]-2.0*vn*zfn[face];
            //p
            qr[4][i] = ql[4][i];
        }
        else{
            for (j = 0; j < 5; j++) {
                tem = 0.5 * (q[j][c1] + q[j][c2]);
                ql[j][i] = tem;
                qr[j][i] = tem;
            }
        }
    }
/*
#ifdef FS_OPENMP
#pragma omp parallel for
#endif
    for (IntType face = ns; face < nMid; face++) {
        IntType  i, j, type, c1, c2;
        RealFlow tem; 
        type = bcr[face]->GetType();
        if (type == INTERFACE) continue;
        c1 = f2c[face + face];
        c2 = f2c[face + face + 1];
        i = face - ns;
        if (type != SYMM) {           
            for (j = 0; j < 5; j++) {
                tem = 0.5 * (q[j][c1] + q[j][c2]);
                ql[j][i] = tem;
                qr[j][i] = tem;
            }
        }
    }
*/   
}


/*******************************************************************************\
     Calculate limiter in 3D
        Only calculate limiter based on rho and p. If you need limiter for
        other variable, add it as necessary.
\*******************************************************************************/
RealFlow **GetLimiters_resp(PolyGrid *grid)
{
    IntType nTCell = grid->GetNTCell();
    IntType n      = nTCell + grid->GetNBFace();
    IntType i,j;
    
    // Allocate memories and initialize limiters with value one
    RealFlow **limit = NULL;
    mfmem::snew_array_2D(limit, 5,n,dmrfl,true);
    for(i=0; i<5; i++) {
        for(j=0; j<n; j++) limit[i][j] = 1.0;
    }
    
    const IntType kNVar = 5;
    RealFlow **dqdx = NULL, **dqdy = NULL, **dqdz = NULL;
    mfmem::snew_array_1D(dqdx, kNVar, dmrfl);
    mfmem::snew_array_1D(dqdy, kNVar, dmrfl);
    mfmem::snew_array_1D(dqdz, kNVar, dmrfl);
    dqdx[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdx"));
    dqdy[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdy"));
    dqdz[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdz"));
    for (IntType i = 1; i < kNVar; ++i) {
        dqdx[i] = &dqdx[i - 1][n];
        dqdy[i] = &dqdy[i - 1][n];
        dqdz[i] = &dqdz[i - 1][n];
    }

    // Get flow variables
    RealFlow *rho = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    RealFlow *u   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    RealFlow *v   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    RealFlow *w   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    RealFlow *p   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    
    IntType order;
    grid->GetData(&order, INT, 1, "order"); 

    switch(order) {
        case LIMITED_VENCAT:
             VencatLimiter(grid, &limit[0][0], rho, dqdx[0], dqdy[0], dqdz[0], 0);
             VencatLimiter(grid, &limit[1][0], u,   dqdx[1], dqdy[1], dqdz[1], 1);
             VencatLimiter(grid, &limit[2][0], v,   dqdx[2], dqdy[2], dqdz[2], 2);
             VencatLimiter(grid, &limit[3][0], w,   dqdx[3], dqdy[3], dqdz[3], 3);
             VencatLimiter(grid, &limit[4][0], p,   dqdx[4], dqdy[4], dqdz[4], 4);
             break;

        case SECOND_ORDER:
             // Doing nothing, because limiters have been set to one
             break;

        // Other cases?
        default:
             printf("Warning:\n");
             printf("Limiter for order %d has not implemented yet\n", (int)order);
             printf("Error in calling limiters\n");
             printf("Set values of limiter to one everywhere except boundary\n");
             break;
    } 

    RealFlow limitaver = 0.;
    IntType  count=0;
    for(i=0; i<5; i++) {
        for(j=0; j<nTCell; j++) {
            limitaver += limit[i][j];
            count++;
        }
    }
#ifdef MPICH
    RealFlow limitaver_total;
    IntType count_total;
    MPI_Allreduce(&limitaver, &limitaver_total, 1, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&count, &count_total, 1, MPIIntType, MPI_SUM, MPI_COMM_WORLD);
    limitaver = limitaver_total;
    count     = count_total;
#endif  
    limitaver /= (RealFlow)count;
    grid->UpdateData(&limitaver, REAL_FLOW, 1, "limitaver"); 
    
 
#ifdef DEBUG
    //输出限制器统计信息
    RealFlow **limitermin=NULL;
    mfmem::snew_array_2D(limitermin, 5, 10, dmrfl, true);

    for(i=0;i<5;i++){
        for(j=0;j<10;j++){
            limitermin[i][j] = 0.0;
        }
    }
    for(i=0;i<5;i++){
        for(j=0;j<nTCell;j++){
            if(limit[i][j]<0.1) limitermin[i][0]+=1.0;
            else if(limit[i][j]<0.2) limitermin[i][1]+=1.0;
            else if(limit[i][j]<0.3) limitermin[i][2]+=1.0;
            else if(limit[i][j]<0.4) limitermin[i][3]+=1.0;
            else if(limit[i][j]<0.5) limitermin[i][4]+=1.0;
            else if(limit[i][j]<0.6) limitermin[i][5]+=1.0;
            else if(limit[i][j]<0.7) limitermin[i][6]+=1.0;
            else if(limit[i][j]<0.8) limitermin[i][7]+=1.0;
            else if(limit[i][j]<0.9) limitermin[i][8]+=1.0;
            else limitermin[i][9]+=1.0;
        }
    }

    IntType nTCell_total = nTCell;
#ifdef MPICH
    Parallel::parallel_sum(nTCell_total, MPI_COMM_WORLD);
    Parallel::parallel_sum(limitermin[0], 50, MPI_COMM_WORLD);
#endif    
    for(i=0;i<5;i++){
        for(j=0;j<10;j++){            
            limitermin[i][j] *= (100.0/nTCell_total);
        }
    }
    /*
    mflog::log.set_one_processor_out();    
    mflog::log<<endl<<SEP_LINE<<endl;
    mflog::log<<"limiter value statistic(rho,u,v,w,p)" << endl;
    mflog::log << IOS_FP(2);
    mflog::log<<"limiter<0.1: "<<limitermin[0][0]<<"  "<<limitermin[1][0]<<"  "
              <<limitermin[2][0]<<"  "<<limitermin[3][0]<<"  "<<limitermin[4][0]<<endl;
    mflog::log<<"limiter<0.2: "<<limitermin[0][1]<<"  "<<limitermin[1][1]<<"  "
              <<limitermin[2][1]<<"  "<<limitermin[3][1]<<"  "<<limitermin[4][1]<<endl;
    mflog::log<<"limiter<0.3: "<<limitermin[0][2]<<"  "<<limitermin[1][2]<<"  "
              <<limitermin[2][2]<<"  "<<limitermin[3][2]<<"  "<<limitermin[4][2]<<endl;
    mflog::log<<"limiter<0.4: "<<limitermin[0][3]<<"  "<<limitermin[1][3]<<"  "
              <<limitermin[2][3]<<"  "<<limitermin[3][3]<<"  "<<limitermin[4][3]<<endl;
    mflog::log<<"limiter<0.5: "<<limitermin[0][4]<<"  "<<limitermin[1][4]<<"  "
              <<limitermin[2][4]<<"  "<<limitermin[3][4]<<"  "<<limitermin[4][0]<<endl;
    mflog::log<<"limiter<0.6: "<<limitermin[0][5]<<"  "<<limitermin[1][5]<<"  "
              <<limitermin[2][5]<<"  "<<limitermin[3][5]<<"  "<<limitermin[4][5]<<endl;
    mflog::log<<"limiter<0.7: "<<limitermin[0][6]<<"  "<<limitermin[1][6]<<"  "
              <<limitermin[2][6]<<"  "<<limitermin[3][6]<<"  "<<limitermin[4][6]<<endl;
    mflog::log<<"limiter<0.8: "<<limitermin[0][7]<<"  "<<limitermin[1][7]<<"  "
              <<limitermin[2][7]<<"  "<<limitermin[3][7]<<"  "<<limitermin[4][7]<<endl;
    mflog::log<<"limiter<0.9: "<<limitermin[0][8]<<"  "<<limitermin[1][8]<<"  "
              <<limitermin[2][8]<<"  "<<limitermin[3][8]<<"  "<<limitermin[4][8]<<endl;
    mflog::log<<"limiter>0.9: "<<limitermin[0][9]<<"  "<<limitermin[1][9]<<"  "
              <<limitermin[2][9]<<"  "<<limitermin[3][9]<<"  "<<limitermin[4][9]<<endl;
    mflog::log<<endl<<SEP_LINE<<endl;     
    */
    mfmem::sdel_array_2D(limitermin);
#endif

#ifdef MPICH
    grid->RecvSendVarNeighbor_Togeth(kNVar, limit);
    /*
    for (IntType i = 0; i < kNVar; ++i) {
        grid->CommInterfaceDataMPI(limit[i]);
    }
    */
#endif

    mfmem::sdel_array_1D(dqdx);
    mfmem::sdel_array_1D(dqdy);
    mfmem::sdel_array_1D(dqdz);

    return limit;
}


/*******************************************************************************\
     DC task-parallelization
        InviscidFlus Function Kernel.
\*******************************************************************************/
#ifdef DC
void InviscidFlux_kernel(char **userArgs, uTaskTreeArgs *treeArgs)
{	
	IntType ns = treeArgs->firstFace;
	IntType ne = treeArgs->lastFace + 1;

    if (ns >= ne) return;

	PolyGrid *grid 			= (PolyGrid *)userArgs[0];
	RealFlow **limit 		= (RealFlow **)userArgs[1];
	RealFlow **dqdx			= (RealFlow **)userArgs[2];
	RealFlow **dqdy 		= (RealFlow **)userArgs[3];
	RealFlow **dqdz 		= (RealFlow **)userArgs[4];
	IntType *IsNormalFace 	= (IntType *)userArgs[5];
	RealFlow **q			= (RealFlow **)userArgs[6];
	RealFlow **res			= (RealFlow **)userArgs[7];
    IntType *face_act		= (IntType *)userArgs[8];
	RealFlow gam			= *((RealFlow *)userArgs[9]);
	RealFlow p_bar			= *((RealFlow *)userArgs[10]);
	RealFlow alf_l			= *((RealFlow *)userArgs[11]);
	RealFlow alf_n			= *((RealFlow *)userArgs[12]);
	RealFlow gascon			= *((RealFlow *)userArgs[13]);
	IntType EntropyCorType 	= *((IntType *)userArgs[14]);
    IntType steady			= *((IntType *)userArgs[15]);
    IntType *IsShockFace	= (IntType *)userArgs[16];
    
	
	IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    
    // Get metrics
    RealGeom *xfn  = grid->GetXfn();
    RealGeom *yfn  = grid->GetYfn();
    RealGeom *zfn  = grid->GetZfn();
    RealGeom *area = grid->GetFaceArea();
    RealGeom *vgn  = grid->GetFaceNormalVelocity();
	RealFlow *flux[5], *ql[5], *qr[5];
    
    IntType len = ne - ns;
	ql[0] = new RealFlow[5*len];
    qr[0] = new RealFlow[5*len];
    flux[0] = new RealFlow[5*len];

    for(IntType i=1; i<5; i++){
        ql[i]   = &ql[i-1][len];
        qr[i]   = &qr[i-1][len];
        flux[i] = &flux[i-1][len];
    }

    SetQlQrWithQ(grid, q, ql, qr, ns, ne);
    
    if (limit != NULL) {		
        CalcuQlQr(grid, ql, qr, limit, dqdx, dqdy, dqdz, p_bar, ns, ne);
    }
    ModQlQrBou(grid, ql, qr, q, steady, ns, ne);
        
    CompInvFlux(grid, ql, qr, flux, &xfn[ns], &yfn[ns], &zfn[ns], &area[ns], &vgn[ns],
        			&face_act[ns], gam, p_bar, alf_l, alf_n, 0, gascon, EntropyCorType, 
                    steady, IsShockFace, IsNormalFace, ns, ne);
        
    // Load the fluxes to residuals
    LoadFlux_DC(grid, flux, ns, ne, res);
    
   
	delete[] ql[0];
    delete[] qr[0];
    delete[] flux[0];
}
#endif


/*******************************************************************************\
    Drive individual functions to calculate inviscid fluxes in 3D
\*******************************************************************************/
void InviscidFlux(PolyGrid *grid, RealFlow **limit, IntType level)
{
#ifdef TIMECOST//dingxin
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemInvis, endtimeTemInvis;
    double timeuseTemInvis;
    gettimeofday(&starttimeTemInvis, 0); 
#endif
#endif
    IntType i, ns, ne, len; 
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nTFace = grid->GetNTFace();
    IntType n      = nTCell + nBFace;

    // Get parameters
    IntType  steady;
    RealFlow gam, p_bar, alf_l, alf_n, disFact=1.;
    grid->GetData(&steady, INT, 1, "steady");
    grid->GetData(&gam,    REAL_FLOW, 1, "gam");
    grid->GetData(&p_bar,  REAL_FLOW, 1, "p_bar");
    grid->GetData(&alf_l,  REAL_FLOW, 1, "alf_l");
    grid->GetData(&alf_n,  REAL_FLOW, 1, "alf_n");
    grid->GetData(&disFact,  REAL_FLOW, 1, "disFact",0);
    
    // for overlap
    IntType *face_act = NULL;
  
    // Allocate temporary memories for ql, qr and flux
    RealFlow *q[5];

    // Get flow variables
    q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    
    const IntType kNVar = 5;
    RealFlow **dqdx = NULL, **dqdy = NULL, **dqdz = NULL;
    mfmem::snew_array_1D(dqdx, kNVar, dmrfl);
    mfmem::snew_array_1D(dqdy, kNVar, dmrfl);
    mfmem::snew_array_1D(dqdz, kNVar, dmrfl);
    dqdx[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdx"));
    dqdy[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdy"));
    dqdz[0] = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdz"));
    for (IntType i = 1; i < kNVar; ++i) {
        dqdx[i] = &dqdx[i - 1][n];
        dqdy[i] = &dqdy[i - 1][n];
        dqdz[i] = &dqdz[i - 1][n];
    }

    
#ifdef DC

    RealFlow gascon;
    grid->GetData(&gascon, REAL_FLOW, 1, "gascon");
    IntType EntropyCorType = 4;
    grid->GetData(&EntropyCorType, INT, 1, "EntropyCorType");
	
    RealFlow *res[5];
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");
    res[1] = &res[0][nTCell];
    res[2] = &res[1][nTCell];
    res[3] = &res[2][nTCell];
    res[4] = &res[3][nTCell];

    IntType* IsNormalFace = 0;
    IntType* IsShockFace = 0;
    if (EntropyCorType == 4) {
        IsNormalFace = (IntType*)grid->GetDataPtr(INT, nTFace, "IsNormalFace");
        if (!IsNormalFace) {
            grid->FindNormalFace();
            IsNormalFace = (IntType*)grid->GetDataPtr(INT, nTFace, "IsNormalFace");
        }

        // shock face or not
        IsShockFace = NULL;
        mfmem::snew_array_1D(IsShockFace, nTFace, dmrfl);
        CalIsShockFace(grid, IsShockFace, 0, nTFace);
    }

    uTaskTree *uTaskTreeRoot = grid->GetuTaskTree();

    char* userArgs[17] = { (char*)grid, (char *)limit, (char *)dqdx, (char *)dqdy, (char *)dqdz, (char *)IsNormalFace,
							(char *)q, (char *)res, (char *)face_act, (char *)&gam, (char *)&p_bar, (char *)&alf_l, (char *)&alf_n, 
							(char *)&gascon, (char *)&EntropyCorType, (char *)&steady, (char *)IsShockFace};

    uTaskTreeRoot->task_traversal(InviscidFlux_kernel, NULL, userArgs, Forward, grid->Getf2c(), nBFace);
	
    if (EntropyCorType == 4) { mfmem::sdel_array_1D(IsShockFace); }

#else   
    RealFlow *ql[5], *qr[5], *flux[5];
    //len = SEG_LEN;
    len = nTFace;
    ql[0]   = NULL;
    qr[0]   = NULL;
    flux[0] = NULL;
    mfmem::snew_array_1D(ql[0],   5*len,dmrfl);
    mfmem::snew_array_1D(qr[0],   5*len,dmrfl);
    mfmem::snew_array_1D(flux[0], 5*len,dmrfl);
    assert(ql[0] != 0);
    assert(qr[0] != 0);
    assert(flux[0] != 0);

    for(i=1; i<5; i++){
        ql[i]   = &ql[i-1][len];
        qr[i]   = &qr[i-1][len];
        flux[i] = &flux[i-1][len];
    }

    // Get metrics
    RealGeom *xfn  = grid->GetXfn();
    RealGeom *yfn  = grid->GetYfn();
    RealGeom *zfn  = grid->GetZfn();
    RealGeom *area = grid->GetFaceArea();
    RealGeom *vgn  = grid->GetFaceNormalVelocity();

    ns = 0;
    do {
       //ne   = ns + SEG_LEN;   
       //if(ne > nTFace) ne = nTFace;
       ne = nTFace;

       // Get left variables and right variables	 	
       SetQlQrWithQ(grid, q, ql, qr, ns, ne);

       if (limit != NULL) {		   
           CalcuQlQr(grid, ql, qr, limit, dqdx, dqdy, dqdz, ns, ne);   
       }
       ModQlQrBou(grid, ql, qr, ns, ne);
       
       CompInvFlux(grid, ql, qr, flux, &xfn[ns], &yfn[ns], &zfn[ns], &area[ns], &vgn[ns],
                   &face_act[ns], gam, p_bar, alf_l, alf_n, 0, ns, ne);       	
       LoadFlux(grid, flux, ns, ne);
       ns  = ne;
    } while (ns < nTFace);
    
    mfmem::sdel_array_1D(ql[0]);
    mfmem::sdel_array_1D(qr[0]);
    mfmem::sdel_array_1D(flux[0]);
#endif
    mfmem::sdel_array_1D(dqdx);
    mfmem::sdel_array_1D(dqdy);
    mfmem::sdel_array_1D(dqdz);

#ifdef TIMECOST//dingxin
#ifdef MPICH
    timecost[1] = timecost[1] + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemInvis, 0); 
    timeuseTemInvis = (RealGeom) 1000000*(endtimeTemInvis.tv_sec - starttimeTemInvis.tv_sec) + endtimeTemInvis.tv_usec - starttimeTemInvis.tv_usec;
    timecost[1] += timeuseTemInvis;
    timeuseTemInvis /= 1000000.0;
    time_invis += timeuseTemInvis;
#endif
#endif
}
/*******************************************************************************\
            calculate ql and qr in 3D using dqdx, dqdy, dqdz
 在CalcuQlQr_new函数基础上将UMUSCL、IfOneOrder等目前不再使用的功能删除，提高计算效率
 zhyb, 2019.03.11
\*******************************************************************************/
void CalcuQlQr(PolyGrid *grid, RealFlow *ql[5], RealFlow *qr[5], RealFlow **limit,
               RealFlow *dqdx[5], RealFlow *dqdy[5], RealFlow *dqdz[5],
#ifdef DC
               RealFlow p_bar,
#endif
               IntType ns, IntType ne)
{
    IntType  nBFace = grid->GetNBFace();
    IntType  *f2c   = grid->Getf2c();
    RealGeom *xfc   = grid->GetXfc();
    RealGeom *yfc   = grid->GetYfc();
    RealGeom *zfc   = grid->GetZfc();
    RealGeom *xcc   = grid->GetXcc();
    RealGeom *ycc   = grid->GetYcc();
    RealGeom *zcc   = grid->GetZcc();
    BCRecord **bcr  = grid->Getbcr();
 
#ifndef DC
    RealFlow p_bar;
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
#endif
    
    
    
    // Determine if there are boundary faces.
    IntType nMid  = ns;
    if(ne <= nBFace){  // If all boundary faces
        nMid = ne;
    }else if(ns < nBFace){  // Part of them are boundary faces
        nMid = nBFace;
    }
#if defined(FS_OPENMP) && !defined(DC) 
#pragma omp parallel for
#endif    
    for(IntType face=ns; face<nMid; face++){
        IntType  i, c1, c2, count, type;
        RealGeom dx, dy, dz;
        RealFlow trho, tpre;
        type  = bcr[face]->GetType();
        if(type!=INTERFACE && type!=SYMM) continue;
        
        count = 2*face;
        i     = face-ns;
        c1    = f2c[count];
        c2    = f2c[count+1];
      
        // Left one
        dx     = xfc[face] - xcc[c1];
        dy     = yfc[face] - ycc[c1];
        dz     = zfc[face] - zcc[c1];
        
        trho   = ql[0][i] + limit[0][c1]*(dqdx[0][c1]*dx + dqdy[0][c1]*dy + dqdz[0][c1]*dz);
        tpre   = ql[4][i] + limit[4][c1]*(dqdx[4][c1]*dx + dqdy[4][c1]*dy + dqdz[4][c1]*dz);
        if(trho > 0 && tpre > -p_bar){
            ql[0][i]  = trho;
            ql[1][i] += limit[1][c1]*(dqdx[1][c1]*dx + dqdy[1][c1]*dy + dqdz[1][c1]*dz);
            ql[2][i] += limit[2][c1]*(dqdx[2][c1]*dx + dqdy[2][c1]*dy + dqdz[2][c1]*dz);
            ql[3][i] += limit[3][c1]*(dqdx[3][c1]*dx + dqdy[3][c1]*dy + dqdz[3][c1]*dz);
            ql[4][i]  = tpre;
        }
        
        if (type == INTERFACE){
            // Right one
            dx     = xfc[face] - xcc[c2];
            dy     = yfc[face] - ycc[c2];
            dz     = zfc[face] - zcc[c2];
    
            trho   = qr[0][i] + limit[0][c2]*(dqdx[0][c2]*dx + dqdy[0][c2]*dy + dqdz[0][c2]*dz);
            tpre   = qr[4][i] + limit[4][c2]*(dqdx[4][c2]*dx + dqdy[4][c2]*dy + dqdz[4][c2]*dz);
            if(trho > 0 && tpre > -p_bar){
                qr[0][i]  = trho;
                qr[1][i] += limit[1][c2]*(dqdx[1][c2]*dx + dqdy[1][c2]*dy + dqdz[1][c2]*dz);
                qr[2][i] += limit[2][c2]*(dqdx[2][c2]*dx + dqdy[2][c2]*dy + dqdz[2][c2]*dz);
                qr[3][i] += limit[3][c2]*(dqdx[3][c2]*dx + dqdy[3][c2]*dy + dqdz[3][c2]*dz);
                qr[4][i]  = tpre;
            }
        }
    }

    
#ifdef FS_OPENMP 
#pragma omp parallel for
#endif    
    for(IntType face=nMid; face<ne; face++) {
        IntType  i, c1, c2, count, type;
        RealGeom dx, dy, dz;
        RealFlow trho, tpre;
        i     = face-ns;
        count = 2*face;
        c1     = f2c[count];
        c2     = f2c[count+1];
        
        // Left one
        dx     = xfc[face] - xcc[c1];
        dy     = yfc[face] - ycc[c1];
        dz     = zfc[face] - zcc[c1];
      
        trho   = ql[0][i] + limit[0][c1]*(dqdx[0][c1]*dx + dqdy[0][c1]*dy + dqdz[0][c1]*dz);
        tpre   = ql[4][i] + limit[4][c1]*(dqdx[4][c1]*dx + dqdy[4][c1]*dy + dqdz[4][c1]*dz);
        if(trho > 0 && tpre > -p_bar){
            ql[0][i]  = trho;
            ql[1][i] += limit[1][c1]*(dqdx[1][c1]*dx + dqdy[1][c1]*dy + dqdz[1][c1]*dz);
            ql[2][i] += limit[2][c1]*(dqdx[2][c1]*dx + dqdy[2][c1]*dy + dqdz[2][c1]*dz);
            ql[3][i] += limit[3][c1]*(dqdx[3][c1]*dx + dqdy[3][c1]*dy + dqdz[3][c1]*dz);
            ql[4][i]  = tpre;
        }
        
        // Right one
        dx     = xfc[face] - xcc[c2];
        dy     = yfc[face] - ycc[c2];
        dz     = zfc[face] - zcc[c2];
        
        trho   = qr[0][i] + limit[0][c2]*(dqdx[0][c2]*dx + dqdy[0][c2]*dy + dqdz[0][c2]*dz);
        tpre   = qr[4][i] + limit[4][c2]*(dqdx[4][c2]*dx + dqdy[4][c2]*dy + dqdz[4][c2]*dz);
        if(trho > 0 && tpre > -p_bar){
            qr[0][i]  = trho;
            qr[1][i] += limit[1][c2]*(dqdx[1][c2]*dx + dqdy[1][c2]*dy + dqdz[1][c2]*dz);
            qr[2][i] += limit[2][c2]*(dqdx[2][c2]*dx + dqdy[2][c2]*dy + dqdz[2][c2]*dz);
            qr[3][i] += limit[3][c2]*(dqdx[3][c2]*dx + dqdy[3][c2]*dy + dqdz[3][c2]*dz);
            qr[4][i]  = tpre;
        }      
    }
}


/*******************************************************************************\
  Compute inviscid fluxes Using Roe scheme
\*******************************************************************************/
void RoeFlux_noprec(PolyGrid* grid, RealFlow* ql[5], RealFlow* qr[5], RealFlow* flux[5],
    RealGeom* xfn, RealGeom* yfn, RealGeom* zfn, RealGeom* area, IntType* face_act,
    RealFlow gam, RealFlow p_bar, RealFlow alf_l, RealFlow alf_n,
#ifdef DC
    RealFlow gascon, IntType EntropyCorType,
    IntType steady, IntType *IsShockFace, 
    IntType *IsNormalFace,
#endif
    IntType ns, IntType ne)
{
    
    IntType  len;
    RealFlow gamm1;
    RealFlow tmp0, tmp1, tmp2, alpha1, alpha2, alpha3, eigv1, eigv2, eigv3;
    RealFlow drho, du, dv, dw, dp, dvn, dq2;
    RealGeom areax, areay, areaz;
    RealFlow spectral, epsaa, epsbb, epscc, epsa_r;
    RealFlow u_vgn, v_vgn, w_vgn;

    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  nTFace = grid->GetNTFace();
    IntType  n = nTCell + nBFace;
    //IntType* f2c = grid->Getf2c();

    RealGeom* vgn = grid->GetFaceNormalVelocity();

#ifndef DC
    IntType steady = 1;
    grid->GetData(&steady, INT, 1, "steady");
    RealFlow gascon;
    grid->GetData(&gascon, REAL_FLOW, 1, "gascon");
    IntType EntropyCorType = 4;
    grid->GetData(&EntropyCorType, INT, 1, "EntropyCorType");

    IntType* IsNormalFace = 0;
    IntType* IsShockFace = 0;
    if (EntropyCorType == 4) {
        IsNormalFace = (IntType*)grid->GetDataPtr(INT, nTFace, "IsNormalFace");
        if (!IsNormalFace) {
            grid->FindNormalFace();
            IsNormalFace = (IntType*)grid->GetDataPtr(INT, nTFace, "IsNormalFace");
        }

        // shock face or not
        IsShockFace = NULL;
        mfmem::snew_array_1D(IsShockFace, ne - ns, dmrfl);
        CalIsShockFace(grid, IsShockFace, ns, ne);
    }
#endif

    gamm1 = gam - 1.0;
    len = ne - ns;
#if (defined FS_SIMD) && (!defined FS_SIMD_AVX) && (!defined Tile)
//containing SIMD
    //const IntType    Vec = 8;
    IntType i, ni;
    IntType k;
#ifdef FS_OPENMP //OpenMP && SIMD
#pragma omp parallel for private(ni)
#endif   
    for(i = 0; i < len - Vec; i += Vec) {
        ni = ns + i;
        //IntType ni_v[Vec];//areax_v[Vec], areay_v[Vec], areaz_v[Vec];
        RealFlow et_l_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow et_r_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow ht_l_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow ht_r_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow vn_l_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow vn_r_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow tmp0_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow tmp1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow tmp2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow rho_a_v[Vec]            __attribute__((aligned(ALIGN)));
        RealFlow u_a_v[Vec]              __attribute__((aligned(ALIGN)));
        RealFlow v_a_v[Vec]              __attribute__((aligned(ALIGN)));
        RealFlow w_a_v[Vec]              __attribute__((aligned(ALIGN)));
        RealFlow vn_a_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow h_a_v[Vec]              __attribute__((aligned(ALIGN)));
        RealFlow q2_v[Vec]               __attribute__((aligned(ALIGN)));
        RealFlow c2_a_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow c_a_v[Vec]              __attribute__((aligned(ALIGN)));
        RealFlow eigv1_v[Vec]            __attribute__((aligned(ALIGN)));
        RealFlow eigv2_v[Vec]            __attribute__((aligned(ALIGN)));
        RealFlow eigv3_v[Vec]            __attribute__((aligned(ALIGN)));
        RealFlow epsa_r_v[Vec]           __attribute__((aligned(ALIGN)));
        RealFlow spectral_v[Vec]         __attribute__((aligned(ALIGN)));
        RealFlow u_vgn_v[Vec]            __attribute__((aligned(ALIGN)));
        RealFlow v_vgn_v[Vec]            __attribute__((aligned(ALIGN)));
        RealFlow w_vgn_v[Vec]            __attribute__((aligned(ALIGN))); 
        RealFlow epsaa_v[Vec]            __attribute__((aligned(ALIGN)));
        RealFlow epsbb_v[Vec]            __attribute__((aligned(ALIGN)));
        RealFlow epscc_v[Vec]            __attribute__((aligned(ALIGN)));
        RealFlow drho_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow du_v[Vec]               __attribute__((aligned(ALIGN)));
        RealFlow dv_v[Vec]               __attribute__((aligned(ALIGN)));
        RealFlow dw_v[Vec]               __attribute__((aligned(ALIGN)));
        RealFlow dp_v[Vec]               __attribute__((aligned(ALIGN)));
        RealFlow dvn_v[Vec]              __attribute__((aligned(ALIGN)));
        RealFlow dq2_v[Vec]              __attribute__((aligned(ALIGN)));
        RealFlow alpha1_v[Vec]           __attribute__((aligned(ALIGN)));
        RealFlow alpha2_v[Vec]           __attribute__((aligned(ALIGN)));
        RealFlow alpha3_v[Vec]           __attribute__((aligned(ALIGN)));
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            // Total energy
            et_l_v[iv] = (ql[4][i + iv] + p_bar) / gamm1 + 0.5 * ql[0][i + iv] *
                (ql[1][i + iv] * ql[1][i + iv] + ql[2][i + iv] * ql[2][i + iv] + ql[3][i + iv] * ql[3][i + iv]);
            et_r_v[iv] = (qr[4][i + iv] + p_bar)/gamm1 + 0.5*qr[0][i + iv]*
                (qr[1][i + iv]*qr[1][i + iv] + qr[2][i + iv]*qr[2][i + iv] + qr[3][i + iv]*qr[3][i + iv]);
            ht_l_v[iv] = et_l_v[iv] + ql[4][i + iv] + p_bar;
            ht_r_v[iv] = et_r_v[iv] + qr[4][i + iv] + p_bar;
            // Full flux
            vn_l_v[iv] = xfn[i + iv] * ql[1][i + iv] + yfn[i + iv] * ql[2][i + iv] + zfn[i + iv] * ql[3][i + iv];
            vn_r_v[iv] = xfn[i + iv] * qr[1][i + iv] + yfn[i + iv] * qr[2][i + iv] + zfn[i + iv] * qr[3][i + iv];
            if (!steady) {   //unsteady
                vn_l_v[iv] -= vgn[ni + iv];
                vn_r_v[iv] -= vgn[ni + iv];
            }
            tmp0_v[iv] = vn_l_v[iv] * ql[0][i + iv];
            tmp1_v[iv] = vn_r_v[iv] * qr[0][i + iv];

            flux[0][i + iv] = tmp0_v[iv] + tmp1_v[iv];
            flux[1][i + iv] = tmp0_v[iv] * ql[1][i + iv] + xfn[i + iv] * ql[4][i + iv]
                + tmp1_v[iv] * qr[1][i + iv] + xfn[i + iv] * qr[4][i + iv];
            flux[2][i + iv] = tmp0_v[iv] * ql[2][i + iv] + yfn[i + iv] * ql[4][i + iv]
                + tmp1_v[iv] * qr[2][i + iv] + yfn[i + iv] * qr[4][i + iv];
            flux[3][i + iv] = tmp0_v[iv] * ql[3][i + iv] + zfn[i + iv] * ql[4][i + iv]
                + tmp1_v[iv] * qr[3][i + iv] + zfn[i + iv] * qr[4][i + iv];
            flux[4][i + iv] = ht_l_v[iv] * vn_l_v[iv] + ht_r_v[iv] * vn_r_v[iv];
            if (!steady) flux[4][i + iv] += (ql[4][i + iv] + qr[4][i + iv] + 2.0 * p_bar) * vgn[ni + iv];   //unsteady, 0.5?ú×?oó3????yμ?μ?·?

            //2éó?roe???ù????μ￥?a??é?μ???àíá?
            tmp0_v[iv] = sqrt(qr[0][i + iv] / ql[0][i + iv]);
            tmp1_v[iv] = 1.0 / (1.0 + tmp0_v[iv]);
            rho_a_v[iv] = sqrt(qr[0][i + iv] * ql[0][i + iv]);
            u_a_v[iv] = (ql[1][i + iv] + qr[1][i + iv] * tmp0_v[iv]) * tmp1_v[iv];
            v_a_v[iv] = (ql[2][i + iv] + qr[2][i + iv] * tmp0_v[iv]) * tmp1_v[iv];
            w_a_v[iv] = (ql[3][i + iv] + qr[3][i + iv] * tmp0_v[iv]) * tmp1_v[iv];
            vn_a_v[iv] = u_a_v[iv] * xfn[i + iv] + v_a_v[iv] * yfn[i + iv] + w_a_v[iv] * zfn[i + iv];
            h_a_v[iv] = (ht_l_v[iv] / ql[0][i + iv] + ht_r_v[iv] / qr[0][i + iv] * tmp0_v[iv]) * tmp1_v[iv];
            q2_v[iv] = 0.5 * (u_a_v[iv] * u_a_v[iv] + v_a_v[iv] * v_a_v[iv] + w_a_v[iv] * w_a_v[iv]);
            c2_a_v[iv] = gamm1 * (h_a_v[iv] - q2_v[iv]);
            c2_a_v[iv] = fabs(c2_a_v[iv]);
            c_a_v[iv] = sqrt(c2_a_v[iv]);

            if (steady) {
                eigv1_v[iv] = fabs(vn_a_v[iv]);
                eigv2_v[iv] = fabs(vn_a_v[iv] + c_a_v[iv]);
                eigv3_v[iv] = fabs(vn_a_v[iv] - c_a_v[iv]);
            }
            else {   //unsteady
                eigv1_v[iv] = fabs(vn_a_v[iv] - vgn[ni + iv]);
                eigv2_v[iv] = fabs(vn_a_v[iv] - vgn[ni + iv] + c_a_v[iv]);
                eigv3_v[iv] = fabs(vn_a_v[iv] - vgn[ni + iv] - c_a_v[iv]);
            }
            //Entropy fix          
            if (EntropyCorType == 3) {
                epsa_r_v[iv] = alf_l;
            }
            else if (EntropyCorType == 4) {
                if (IsNormalFace[ni + iv] && IsShockFace[i + iv] == 0) {
                    epsa_r_v[iv] = 0.01 * alf_l;
                    //epsa_r = 0.0002;
                }
                else {
                    epsa_r_v[iv] = alf_l;
                }
            }
            else {
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
            //cfl3d form
            if (steady) {
                spectral_v[iv] = fabs(u_a_v[iv]) + fabs(v_a_v[iv]) + fabs(w_a_v[iv]) + c_a_v[iv];
            }
            else {
                u_vgn_v[iv] = vgn[ni + iv] * xfn[i + iv];
                v_vgn_v[iv] = vgn[ni + iv] * yfn[i + iv];
                w_vgn_v[iv] = vgn[ni + iv] * zfn[i + iv];
                spectral_v[iv] = fabs(u_a_v[iv] - u_vgn_v[iv]) + fabs(v_a_v[iv] - v_vgn_v[iv]) + fabs(w_a_v[iv] - w_vgn_v[iv]) + c_a_v[iv];
            }
            epsaa_v[iv] = epsa_r_v[iv] * spectral_v[iv];
            epsbb_v[iv] = 0.25 / std::max(epsaa_v[iv], TINY);
            epscc_v[iv] = 2.0 * epsaa_v[iv];
            if (eigv1_v[iv] < epscc_v[iv]) eigv1_v[iv] = eigv1_v[iv] * eigv1_v[iv] * epsbb_v[iv] + epsaa_v[iv];
            if (eigv2_v[iv] < epscc_v[iv]) eigv2_v[iv] = eigv2_v[iv] * eigv2_v[iv] * epsbb_v[iv] + epsaa_v[iv];
            if (eigv3_v[iv] < epscc_v[iv]) eigv3_v[iv] = eigv3_v[iv] * eigv3_v[iv] * epsbb_v[iv] + epsaa_v[iv];

            drho_v[iv] = qr[0][i + iv] - ql[0][i + iv];
            du_v[iv] = qr[1][i + iv] - ql[1][i + iv];
            dv_v[iv] = qr[2][i + iv] - ql[2][i + iv];
            dw_v[iv] = qr[3][i + iv] - ql[3][i + iv];
            dp_v[iv] = qr[4][i + iv] - ql[4][i + iv];
            dvn_v[iv] = vn_r_v[iv] - vn_l_v[iv];

            dq2_v[iv] = u_a_v[iv] * du_v[iv] + v_a_v[iv] * dv_v[iv] + w_a_v[iv] * dw_v[iv];

            tmp0_v[iv] = dp_v[iv] / c2_a_v[iv];
            tmp1_v[iv] = rho_a_v[iv] * dvn_v[iv] / c_a_v[iv];
            alpha1_v[iv] = (drho_v[iv] - tmp0_v[iv]) * eigv1_v[iv];
            alpha2_v[iv] = 0.5 * (tmp0_v[iv] + tmp1_v[iv]) * eigv2_v[iv];
            alpha3_v[iv] = 0.5 * (tmp0_v[iv] - tmp1_v[iv]) * eigv3_v[iv];

            tmp0_v[iv] = alpha1_v[iv] + alpha2_v[iv] + alpha3_v[iv];
            tmp1_v[iv] = eigv1_v[iv] * rho_a_v[iv];
            tmp2_v[iv] = -tmp1_v[iv] * dvn_v[iv] + (alpha2_v[iv] - alpha3_v[iv]) * c_a_v[iv];
            flux[0][i + iv] -= tmp0_v[iv];
            flux[1][i + iv] -= tmp0_v[iv] * u_a_v[iv] + tmp1_v[iv] * du_v[iv] + tmp2_v[iv] * xfn[i + iv];
            flux[2][i + iv] -= tmp0_v[iv] * v_a_v[iv] + tmp1_v[iv] * dv_v[iv] + tmp2_v[iv] * yfn[i + iv];
            flux[3][i + iv] -= tmp0_v[iv] * w_a_v[iv] + tmp1_v[iv] * dw_v[iv] + tmp2_v[iv] * zfn[i + iv];
            flux[4][i + iv] -= alpha1_v[iv] * q2_v[iv] + (alpha2_v[iv] + alpha3_v[iv]) * h_a_v[iv] + tmp1_v[iv] * dq2_v[iv] + tmp2_v[iv] * vn_a_v[iv];

            tmp0_v[iv] = 0.5 * area[i + iv];
            flux[0][i + iv] *= tmp0_v[iv];
            flux[1][i + iv] *= tmp0_v[iv];
            flux[2][i + iv] *= tmp0_v[iv];
            flux[3][i + iv] *= tmp0_v[iv];
            flux[4][i + iv] *= tmp0_v[iv];
        }                                                                                                     
    }
    k = i;
    for (i = k; i < len; i++) {
        IntType  ni;
        RealFlow rho_a, u_a, v_a, w_a, h_a, c_a, c2_a, vn_a, q2;
        RealFlow vn_l, et_l, ht_l, vn_r, et_r, ht_r;
        RealFlow tmp0, tmp1, tmp2, alpha1, alpha2, alpha3, eigv1, eigv2, eigv3;
        RealFlow drho, du, dv, dw, dp, dvn, dq2;
        RealGeom areax, areay, areaz;
        RealFlow spectral, epsaa, epsbb, epscc, epsa_r;
        RealFlow u_vgn, v_vgn, w_vgn;
        ni = ns + i;
        areax = xfn[i];
        areay = yfn[i];
        areaz = zfn[i];

        // Total energy
        et_l  = (ql[4][i] + p_bar)/gamm1 + 0.5*ql[0][i]*
                (ql[1][i]*ql[1][i] + ql[2][i]*ql[2][i] + ql[3][i]*ql[3][i]);
        et_r  = (qr[4][i] + p_bar)/gamm1 + 0.5*qr[0][i]*
                (qr[1][i]*qr[1][i] + qr[2][i]*qr[2][i] + qr[3][i]*qr[3][i]);
        ht_l  = et_l + ql[4][i] + p_bar;
        ht_r  = et_r + qr[4][i] + p_bar;

        // Full flux
        vn_l       = areax*ql[1][i] + areay*ql[2][i] + areaz*ql[3][i];
        vn_r       = areax*qr[1][i] + areay*qr[2][i] + areaz*qr[3][i];
        if(!steady){   //unsteady
            vn_l  -= vgn[ni];
            vn_r  -= vgn[ni];
        }

        tmp0       = vn_l*ql[0][i];
        tmp1       = vn_r*qr[0][i];
        flux[0][i] = tmp0 + tmp1;
        flux[1][i] = tmp0*ql[1][i] + areax*ql[4][i]
                   + tmp1*qr[1][i] + areax*qr[4][i];
        flux[2][i] = tmp0*ql[2][i] + areay*ql[4][i]
                   + tmp1*qr[2][i] + areay*qr[4][i];
        flux[3][i] = tmp0*ql[3][i] + areaz*ql[4][i]
                   + tmp1*qr[3][i] + areaz*qr[4][i];
        flux[4][i] = ht_l*vn_l + ht_r*vn_r;
        if(!steady) flux[4][i] += (ql[4][i]+qr[4][i]+2.0*p_bar)*vgn[ni];   //unsteady, 0.5在最后乘面积的地方

        //采用roe平均计算单元面上的物理量
        tmp0  = sqrt(qr[0][i]/ql[0][i]);
        tmp1  = 1.0/(1.0 + tmp0);
        rho_a = sqrt(qr[0][i]*ql[0][i]);
        u_a   = (ql[1][i] + qr[1][i]*tmp0)*tmp1;
        v_a   = (ql[2][i] + qr[2][i]*tmp0)*tmp1;
        w_a   = (ql[3][i] + qr[3][i]*tmp0)*tmp1;
        vn_a  = u_a*areax + v_a*areay + w_a*areaz;
        h_a   = (ht_l/ql[0][i] + ht_r/qr[0][i]*tmp0)*tmp1;

        q2    = 0.5*(u_a*u_a + v_a*v_a + w_a*w_a);
        c2_a  = gamm1*(h_a - q2);
        c2_a  = fabs(c2_a);
        c_a   = sqrt(c2_a);

        if(steady){
            eigv1 = fabs(vn_a);
            eigv2 = fabs(vn_a + c_a);
            eigv3 = fabs(vn_a - c_a);
        }else{   //unsteady
            eigv1 = fabs(vn_a - vgn[ns+i]);
            eigv2 = fabs(vn_a - vgn[ns+i] + c_a);
            eigv3 = fabs(vn_a - vgn[ns+i] - c_a);
        }

        //Entropy fix          
        if(EntropyCorType == 3){
            epsa_r= alf_l;
        }else if(EntropyCorType == 4){
            if(IsNormalFace[ni] && IsShockFace[i]==0){
                epsa_r = 0.01*alf_l;
                //epsa_r = 0.0002;
            }else{
                epsa_r = alf_l;
            }                
        }else{
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
        
        //cfl3d form
        if(steady){
            spectral = fabs(u_a) + fabs(v_a) + fabs(w_a) + c_a;
        }else{
            u_vgn = vgn[ni]*xfn[i];
            v_vgn = vgn[ni]*yfn[i];
            w_vgn = vgn[ni]*zfn[i];
            spectral = fabs(u_a-u_vgn)+fabs(v_a-v_vgn)+fabs(w_a-w_vgn)+c_a;
        }
        epsaa = epsa_r*spectral;
        epsbb = 0.25/std::max(epsaa,TINY);
        epscc = 2.0*epsaa;
        if(eigv1<epscc) eigv1 = eigv1*eigv1*epsbb+epsaa;
        if(eigv2<epscc) eigv2 = eigv2*eigv2*epsbb+epsaa;
        if(eigv3<epscc) eigv3 = eigv3*eigv3*epsbb+epsaa;
        
        drho     = qr[0][i] - ql[0][i];
        du       = qr[1][i] - ql[1][i];
        dv       = qr[2][i] - ql[2][i];
        dw       = qr[3][i] - ql[3][i];
        dp       = qr[4][i] - ql[4][i];
        dvn      = vn_r     - vn_l;

        dq2      = u_a*du + v_a*dv + w_a*dw;

        tmp0     = dp/c2_a;
        tmp1     = rho_a*dvn/c_a;
        alpha1   =     (drho - tmp0)*eigv1;
        alpha2   = 0.5*(tmp0 + tmp1)*eigv2;
        alpha3   = 0.5*(tmp0 - tmp1)*eigv3;

        tmp0        =  alpha1 + alpha2 + alpha3;
        tmp1        =  eigv1*rho_a;
        tmp2        = -tmp1*dvn + (alpha2 - alpha3)*c_a;
        flux[0][i] -=  tmp0;
        flux[1][i] -=  tmp0*u_a  + tmp1*du + tmp2*areax;
        flux[2][i] -=  tmp0*v_a  + tmp1*dv + tmp2*areay;
        flux[3][i] -=  tmp0*w_a  + tmp1*dw + tmp2*areaz;
        flux[4][i] -=  alpha1*q2 + (alpha2 + alpha3)*h_a + tmp1*dq2 + tmp2*vn_a;
            
        tmp0        = 0.5*area[i];
        flux[0][i] *= tmp0;
        flux[1][i] *= tmp0;
        flux[2][i] *= tmp0;
        flux[3][i] *= tmp0;
        flux[4][i] *= tmp0;
    }
#else
//not containing SIMD
#ifdef defined(FS_OPENMP) && !defined(DC) //only OpenMP
#pragma omp parallel for
#endif
//else: serial code  
    for (IntType i = 0; i < len; i++) {
        IntType  ni;
        RealFlow rho_a, u_a, v_a, w_a, h_a, c_a, c2_a, vn_a, q2;
        RealFlow vn_l, et_l, ht_l, vn_r, et_r, ht_r;
        RealFlow tmp0, tmp1, tmp2, alpha1, alpha2, alpha3, eigv1, eigv2, eigv3;
        RealFlow drho, du, dv, dw, dp, dvn, dq2;
        RealGeom areax, areay, areaz;
        RealFlow spectral, epsaa, epsbb, epscc, epsa_r;
        RealFlow u_vgn, v_vgn, w_vgn;
        ni = ns + i;
        areax = xfn[i];
        areay = yfn[i];
        areaz = zfn[i];

        // Total energy
        et_l = (ql[4][i] + p_bar) / gamm1 + 0.5 * ql[0][i] *
            (ql[1][i] * ql[1][i] + ql[2][i] * ql[2][i] + ql[3][i] * ql[3][i]);
        et_r = (qr[4][i] + p_bar) / gamm1 + 0.5 * qr[0][i] *
            (qr[1][i] * qr[1][i] + qr[2][i] * qr[2][i] + qr[3][i] * qr[3][i]);
        ht_l = et_l + ql[4][i] + p_bar;
        ht_r = et_r + qr[4][i] + p_bar;

        // Full flux
        vn_l = areax * ql[1][i] + areay * ql[2][i] + areaz * ql[3][i];
        vn_r = areax * qr[1][i] + areay * qr[2][i] + areaz * qr[3][i];
        if (!steady) {   //unsteady
            vn_l -= vgn[ni];
            vn_r -= vgn[ni];
        }

        tmp0 = vn_l * ql[0][i];
        tmp1 = vn_r * qr[0][i];
        flux[0][i] = tmp0 + tmp1;
        flux[1][i] = tmp0 * ql[1][i] + areax * ql[4][i]
            + tmp1 * qr[1][i] + areax * qr[4][i];
        flux[2][i] = tmp0 * ql[2][i] + areay * ql[4][i]
            + tmp1 * qr[2][i] + areay * qr[4][i];
        flux[3][i] = tmp0 * ql[3][i] + areaz * ql[4][i]
            + tmp1 * qr[3][i] + areaz * qr[4][i];
        flux[4][i] = ht_l * vn_l + ht_r * vn_r;
        if (!steady) flux[4][i] += (ql[4][i] + qr[4][i] + 2.0 * p_bar) * vgn[ni];   //unsteady, 0.5在最后乘面积的地方

        //采用roe平均计算单元面上的物理量
        tmp0 = sqrt(qr[0][i] / ql[0][i]);
        tmp1 = 1.0 / (1.0 + tmp0);
        rho_a = sqrt(qr[0][i] * ql[0][i]);
        u_a = (ql[1][i] + qr[1][i] * tmp0) * tmp1;
        v_a = (ql[2][i] + qr[2][i] * tmp0) * tmp1;
        w_a = (ql[3][i] + qr[3][i] * tmp0) * tmp1;
        vn_a = u_a * areax + v_a * areay + w_a * areaz;
        h_a = (ht_l / ql[0][i] + ht_r / qr[0][i] * tmp0) * tmp1;

        q2 = 0.5 * (u_a * u_a + v_a * v_a + w_a * w_a);
        c2_a = gamm1 * (h_a - q2);
        c2_a = fabs(c2_a);
        c_a = sqrt(c2_a);

        if (steady) {
            eigv1 = fabs(vn_a);
            eigv2 = fabs(vn_a + c_a);
            eigv3 = fabs(vn_a - c_a);
        }
        else {   //unsteady
            eigv1 = fabs(vn_a - vgn[ns + i]);
            eigv2 = fabs(vn_a - vgn[ns + i] + c_a);
            eigv3 = fabs(vn_a - vgn[ns + i] - c_a);
        }

        //Entropy fix          
        if (EntropyCorType == 3) {
            epsa_r = alf_l;
        }
        else if (EntropyCorType == 4) {
            if (IsNormalFace[ni] && IsShockFace[i] == 0) {
                epsa_r = 0.01 * alf_l;
                //epsa_r = 0.0002;
            }
            else {
                epsa_r = alf_l;
            }
        }
        else {
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }

        //cfl3d form
        if (steady) {
            spectral = fabs(u_a) + fabs(v_a) + fabs(w_a) + c_a;
        }
        else {
            u_vgn = vgn[ni] * xfn[i];
            v_vgn = vgn[ni] * yfn[i];
            w_vgn = vgn[ni] * zfn[i];
            spectral = fabs(u_a - u_vgn) + fabs(v_a - v_vgn) + fabs(w_a - w_vgn) + c_a;
        }
        epsaa = epsa_r * spectral;
        epsbb = 0.25 / std::max(epsaa, TINY);
        epscc = 2.0 * epsaa;
        if (eigv1 < epscc) eigv1 = eigv1 * eigv1 * epsbb + epsaa;
        if (eigv2 < epscc) eigv2 = eigv2 * eigv2 * epsbb + epsaa;
        if (eigv3 < epscc) eigv3 = eigv3 * eigv3 * epsbb + epsaa;

        drho = qr[0][i] - ql[0][i];
        du = qr[1][i] - ql[1][i];
        dv = qr[2][i] - ql[2][i];
        dw = qr[3][i] - ql[3][i];
        dp = qr[4][i] - ql[4][i];
        dvn = vn_r - vn_l;

        dq2 = u_a * du + v_a * dv + w_a * dw;

        tmp0 = dp / c2_a;
        tmp1 = rho_a * dvn / c_a;
        alpha1 = (drho - tmp0) * eigv1;
        alpha2 = 0.5 * (tmp0 + tmp1) * eigv2;
        alpha3 = 0.5 * (tmp0 - tmp1) * eigv3;

        tmp0 = alpha1 + alpha2 + alpha3;
        tmp1 = eigv1 * rho_a;
        tmp2 = -tmp1 * dvn + (alpha2 - alpha3) * c_a;
        flux[0][i] -= tmp0;
        flux[1][i] -= tmp0 * u_a + tmp1 * du + tmp2 * areax;
        flux[2][i] -= tmp0 * v_a + tmp1 * dv + tmp2 * areay;
        flux[3][i] -= tmp0 * w_a + tmp1 * dw + tmp2 * areaz;
        flux[4][i] -= alpha1 * q2 + (alpha2 + alpha3) * h_a + tmp1 * dq2 + tmp2 * vn_a;

        tmp0 = 0.5 * area[i];
        flux[0][i] *= tmp0;
        flux[1][i] *= tmp0;
        flux[2][i] *= tmp0;
        flux[3][i] *= tmp0;
        flux[4][i] *= tmp0;
    }
#endif

#ifndef DC
    if (EntropyCorType == 4) { mfmem::sdel_array_1D(IsShockFace); }
#endif
}

/*******************************************************************************\
      Judge one face is shock face or not for entropy correct 4
NOTE: The face is shock face if the relative pressure increase is up to threshold 
      along the direction of average velocity.
\*******************************************************************************/
void CalIsShockFace(PolyGrid *grid, IntType *IsShockFace, IntType ns, IntType ne)
{
    IntType  i, c1, c2;
    IntType  *f2c   = grid->Getf2c();
    RealGeom *xfn   = grid->GetXfn();
    RealGeom *yfn   = grid->GetYfn();
    RealGeom *zfn   = grid->GetZfn();

    RealFlow mach00, gam, p_bar;
    grid->GetData(&mach00, REAL_FLOW, 1, "mach");
    grid->GetData(&gam, REAL_FLOW, 1, "gam");
    grid->GetData(&p_bar,   REAL_FLOW, 1, "p_bar");
    RealFlow pref = p_bar*(1.0 + 0.5*(gam - 1.0)*mach00*mach00);

    IntType   n = grid->GetNTCell() + grid->GetNBFace();
    RealFlow *u = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    RealFlow *v = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    RealFlow *w = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    RealFlow *p = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    assert(u != NULL);

    IntType len   = ne  - ns;
    IntType count = 2*ns;
    IntType face;
    RealFlow dp, t, vx, vy, vz;
    RealFlow ThdShock = 0.5;    // threshold for shock face
    for(i=0; i<len; ++i) IsShockFace[i] = 0;
    for(i=0; i<len; ++i){
        c1 = f2c[count++];
        c2 = f2c[count++];

        // average velocity
        vx = u[c1] + u[c2];
        vy = v[c1] + v[c2];
        vz = w[c1] + w[c2];
        face = i + ns;
        t = vx*xfn[face] + vy*yfn[face] + vz*zfn[face];
        if(t>0) dp = p[c2] - p[c1];
        else    dp = p[c1] - p[c2];
        dp /= pref;
        if(dp>ThdShock) IsShockFace[i] = 1; 
    }
}


/*******************************************************************************\
     Update residuals in cell with the fluxes at cell faces in 3D
\*******************************************************************************/
void LoadFlux_DC(PolyGrid *grid, RealFlow *flux[], IntType ns, IntType ne
#ifdef DC
                ,RealFlow **res
#endif
            ) 
{
    IntType  face, i, c1, c2, count, nMid;
    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  *f2c   = grid->Getf2c();

    // Determine if there are boundary faces.
    nMid  = ns; 
    if(ne  <= nBFace) {
        // If all boundary faces
        nMid = ne;
    } else if(ns < nBFace) {
        // Part of them are boundary faces
        nMid = nBFace;
    }
    //cout<<"nMid: "<<nMid<<endl;
    //cout<<"nBFace: "<<nBFace<<endl;
    // Get Residual
#ifndef DC
    RealFlow *res[5];
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");
    res[1] = &res[0][nTCell];
    res[2] = &res[1][nTCell];
    res[3] = &res[2][nTCell];
    res[4] = &res[3][nTCell];
#endif

    //Group color openmp
#if (defined FS_OPENMP) && (defined GroupColor)
    if (grid->GroupColorSuccess) {
        IntType nIFace = grid->GetNIFace();
        IntType groupSize = grid->groupSize;
        IntType n_bcolor, n_icolor;
        n_bcolor = grid->bfacegroup.size();
        n_icolor = grid->ifacegroup.size();

        // physical face
        for (i = 0; i < n_bcolor; i++) {
            if (!i)
                ns = 0;
            else
                ns = grid->bfacegroup[i - 1];
            nMid = grid->bfacegroup[i];
#pragma omp parallel for private(face,c1) schedule(static,groupSize)
            for (face = ns; face < nMid; face++) {
                c1 = f2c[2 * face];

                res[0][c1] -= flux[0][face];
                res[1][c1] -= flux[1][face];
                res[2][c1] -= flux[2][face];
                res[3][c1] -= flux[3][face];
                res[4][c1] -= flux[4][face];
            }
        }

        // zone boundary face
        count = 2 * (nBFace - nIFace);
        for (face = nBFace - nIFace; face < nBFace; face++) {
            c1 = f2c[count++];
            count++;

            res[0][c1] -= flux[0][face];
            res[1][c1] -= flux[1][face];
            res[2][c1] -= flux[2][face];
            res[3][c1] -= flux[3][face];
            res[4][c1] -= flux[4][face];
        }

        // Interior faces
        for (i = 0; i < n_icolor; i++) {
            if (!i)
                nMid = nBFace;
            else
                nMid = grid->ifacegroup[i - 1];
            ne = grid->ifacegroup[i];
#pragma omp parallel for private(face,c1,c2) schedule(static,groupSize)
            for (face = nMid; face < ne; face++) {
                c1 = f2c[2 * face];
                c2 = f2c[2 * face + 1];

                res[0][c1] -= flux[0][face];
                res[1][c1] -= flux[1][face];
                res[2][c1] -= flux[2][face];
                res[3][c1] -= flux[3][face];
                res[4][c1] -= flux[4][face];

                res[0][c2] += flux[0][face];
                res[1][c2] += flux[1][face];
                res[2][c2] += flux[2][face];
                res[3][c2] += flux[3][face];
                res[4][c2] += flux[4][face];
            }
        }
    }
    else {
        // For boundary faces, remember c2 is ghost cell
        // Determine if there are boundary faces.
        count = 2 * ns;
        i = 0;
        for (face = ns; face < nMid; face++) {
            c1 = f2c[count++];
            count++;

            res[0][c1] -= flux[0][i];
            res[1][c1] -= flux[1][i];
            res[2][c1] -= flux[2][i];
            res[3][c1] -= flux[3][i];
            res[4][c1] -= flux[4][i];
            i++;
        }

        // Interior faces
        for (face = nMid; face < ne; face++) {
            c1 = f2c[count++];
            c2 = f2c[count++];

            res[0][c1] -= flux[0][i];
            res[1][c1] -= flux[1][i];
            res[2][c1] -= flux[2][i];
            res[3][c1] -= flux[3][i];
            res[4][c1] -= flux[4][i];

            res[0][c2] += flux[0][i];
            res[1][c2] += flux[1][i];
            res[2][c2] += flux[2][i];
            res[3][c2] += flux[3][i];
            res[4][c2] += flux[4][i];
            i++;
        }
    }
#elif (defined FS_OPENMP) && (defined FaceColoring)
    //lrt
    IntType    nIFace = grid->GetNIFace();
    IntType  nTFace = grid->GetNTFace();
    IntType     ifacenum = nTFace - nBFace;
    IntType     pfacenum = nBFace - nIFace;
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
            startFace = 0; //for ns>0 && ns<grid_bfacegroup[0]
        }
        else {
            startFace = grid_bfacegroup[fcolor - 1];
        }
        endFace = grid_bfacegroup[fcolor];
#pragma omp parallel for
        for (IntType face = startFace; face < endFace; face++) {
            IntType  c1, c2, count;
            IntType  i;
            count = 2*face;
            c1  = f2c[count];
            i = face - ns;
            //c1  = f2c[count++];
            //count++;
        
            res[0][c1] -= flux[0][i];
            res[1][c1] -= flux[1][i];
            res[2][c1] -= flux[2][i];
            res[3][c1] -= flux[3][i];
            res[4][c1] -= flux[4][i];
            //i++;
        
        }
    }
#ifdef MPICH    
    for (IntType face = pfacenum; face < nBFace; face++) {
        IntType count = 2*face;
        IntType c1 = f2c[count];
        IntType i = face - ns;
        res[0][c1] -= flux[0][i];
        res[1][c1] -= flux[1][i];
        res[2][c1] -= flux[2][i];
        res[3][c1] -= flux[3][i];
        res[4][c1] -= flux[4][i];

    }
#endif    
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
            IntType  c1, c2, count;
            IntType  i;
            count = 2*face;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            i = face - ns;
            //c1 = f2c[count++];
            //c2 = f2c[count++];

            res[0][c1] -= flux[0][i];
            res[1][c1] -= flux[1][i];
            res[2][c1] -= flux[2][i];
            res[3][c1] -= flux[3][i];
            res[4][c1] -= flux[4][i];
        
            res[0][c2] += flux[0][i];
            res[1][c2] += flux[1][i];
            res[2][c2] += flux[2][i];
            res[3][c2] += flux[3][i];
            res[4][c2] += flux[4][i];
            //i++;
            
        }
    }
    mfmem::sdel_array_1D(grid_bfacegroup);
    mfmem::sdel_array_1D(grid_ifacegroup);
#elif (defined FS_OPENMP) && (defined Reduction)//Manual reduction
    IntType* nFPC = CalnFPC(grid);
    IntType** C2F = CalC2F(grid);
    IntType j;
#pragma omp parallel for private(i,j,count,c1,c2,face)
    for (i = 0; i < nTCell; i++) {
        for (j = 0; j < nFPC[i]; j++) {
            face = C2F[i][j];
            count = 2 * face;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            if (i == c1) {
                res[0][i] -= flux[0][face];
                res[1][i] -= flux[1][face];
                res[2][i] -= flux[2][face];
                res[3][i] -= flux[3][face];
                res[4][i] -= flux[4][face];
            }
            else if (i == c2) {
                res[0][i] += flux[0][face];
                res[1][i] += flux[1][face];
                res[2][i] += flux[2][face];
                res[3][i] += flux[3][face];
                res[4][i] += flux[4][face];
            }
            else {
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
    }
#elif (defined FS_OPENMP) && (defined DIVREP)//Division & replication
    IntType threads = grid->threads;
    IntType nTFace = grid->GetNTFace();
    IntType startFace, endFace, t, k;
    if (grid->DivRepSuccess) {
    #pragma omp parallel for private(t,i,k,startFace,endFace,c1,c2,face)
        for (t = 0; t < threads; t++) {
            //Boundary faces
            startFace = grid->idx_pthreads_bface[t];
            endFace = grid->idx_pthreads_bface[t + 1];
            for (i = startFace; i < endFace; i++) {
                face = grid->id_division_bface[i];
                c1 = f2c[2 * face];

                res[0][c1] -= flux[0][face];
                res[1][c1] -= flux[1][face];
                res[2][c1] -= flux[2][face];
                res[3][c1] -= flux[3][face];
                res[4][c1] -= flux[4][face];
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
                    res[0][c1] -= flux[0][face];
                    res[1][c1] -= flux[1][face];
                    res[2][c1] -= flux[2][face];
                    res[3][c1] -= flux[3][face];
                    res[4][c1] -= flux[4][face];

                    res[0][c2] += flux[0][face];
                    res[1][c2] += flux[1][face];
                    res[2][c2] += flux[2][face];
                    res[3][c2] += flux[3][face];
                    res[4][c2] += flux[4][face];
                }
                else {
                    if (k > 0) {
                        res[0][c1] -= flux[0][face];
                        res[1][c1] -= flux[1][face];
                        res[2][c1] -= flux[2][face];
                        res[3][c1] -= flux[3][face];
                        res[4][c1] -= flux[4][face];
                    }
                    else {
                        res[0][c2] += flux[0][face];
                        res[1][c2] += flux[1][face];
                        res[2][c2] += flux[2][face];
                        res[3][c2] += flux[3][face];
                        res[4][c2] += flux[4][face];
                    }
                }
            }
        }
    }
#elif (defined FS_OPENMP) && (defined DIVCON) //D&C TREE
#pragma omp parallel
    {
    #pragma omp single nowait
        tree_traversal(grid->treeHead, res, flux, f2c);
    }
#else
    // For boundary faces, remember c2 is ghost cell
    // Determine if there are boundary faces.
    count = 2 * ns;
    i = 0;
    for (face = ns; face < nMid; face++) {
        c1 = f2c[count++];
        count++;

        res[0][c1] -= flux[0][i];
        res[1][c1] -= flux[1][i];
        res[2][c1] -= flux[2][i];
        res[3][c1] -= flux[3][i];
        res[4][c1] -= flux[4][i];
        i++;
    }

    // Interior faces
    for (face = nMid; face < ne; face++) {
        c1 = f2c[count++];
        c2 = f2c[count++];

        res[0][c1] -= flux[0][i];
        res[1][c1] -= flux[1][i];
        res[2][c1] -= flux[2][i];
        res[3][c1] -= flux[3][i];
        res[4][c1] -= flux[4][i];

        res[0][c2] += flux[0][i];
        res[1][c2] += flux[1][i];
        res[2][c2] += flux[2][i];
        res[3][c2] += flux[3][i];
        res[4][c2] += flux[4][i];
        i++;
    }
#endif
    
}



/*******************************************************************************\
     Update residuals in cell with the fluxes at cell faces in 3D
\*******************************************************************************/
void LoadFlux(PolyGrid *grid, RealFlow *flux[], IntType ns, IntType ne) 
{
    IntType  face, i, c1, c2, count, nMid;
    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  *f2c   = grid->Getf2c();

    // Determine if there are boundary faces.
    nMid  = ns; 
    if(ne  <= nBFace) {
        // If all boundary faces
        nMid = ne;
    } else if(ns < nBFace) {
        // Part of them are boundary faces
        nMid = nBFace;
    }
    //cout<<"nMid: "<<nMid<<endl;
    //cout<<"nBFace: "<<nBFace<<endl;
    // Get Residual
    RealFlow *res[5];
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");
    res[1] = &res[0][nTCell];
    res[2] = &res[1][nTCell];
    res[3] = &res[2][nTCell];
    res[4] = &res[3][nTCell];

    //Group color openmp
#if (defined FS_OPENMP) && (defined GroupColor)
    if (grid->GroupColorSuccess) {
        IntType nIFace = grid->GetNIFace();
        IntType groupSize = grid->groupSize;
        IntType n_bcolor, n_icolor;
        n_bcolor = grid->bfacegroup.size();
        n_icolor = grid->ifacegroup.size();

        // physical face
        for (i = 0; i < n_bcolor; i++) {
            if (!i)
                ns = 0;
            else
                ns = grid->bfacegroup[i - 1];
            nMid = grid->bfacegroup[i];
#pragma omp parallel for private(face,c1) schedule(static,groupSize)
            for (face = ns; face < nMid; face++) {
                c1 = f2c[2 * face];

                res[0][c1] -= flux[0][face];
                res[1][c1] -= flux[1][face];
                res[2][c1] -= flux[2][face];
                res[3][c1] -= flux[3][face];
                res[4][c1] -= flux[4][face];
            }
        }

        // zone boundary face
        count = 2 * (nBFace - nIFace);
        for (face = nBFace - nIFace; face < nBFace; face++) {
            c1 = f2c[count++];
            count++;

            res[0][c1] -= flux[0][face];
            res[1][c1] -= flux[1][face];
            res[2][c1] -= flux[2][face];
            res[3][c1] -= flux[3][face];
            res[4][c1] -= flux[4][face];
        }

        // Interior faces
        for (i = 0; i < n_icolor; i++) {
            if (!i)
                nMid = nBFace;
            else
                nMid = grid->ifacegroup[i - 1];
            ne = grid->ifacegroup[i];
#pragma omp parallel for private(face,c1,c2) schedule(static,groupSize)
            for (face = nMid; face < ne; face++) {
                c1 = f2c[2 * face];
                c2 = f2c[2 * face + 1];

                res[0][c1] -= flux[0][face];
                res[1][c1] -= flux[1][face];
                res[2][c1] -= flux[2][face];
                res[3][c1] -= flux[3][face];
                res[4][c1] -= flux[4][face];

                res[0][c2] += flux[0][face];
                res[1][c2] += flux[1][face];
                res[2][c2] += flux[2][face];
                res[3][c2] += flux[3][face];
                res[4][c2] += flux[4][face];
            }
        }
    }
    else {
        // For boundary faces, remember c2 is ghost cell
        // Determine if there are boundary faces.
        count = 2 * ns;
        i = 0;
        for (face = ns; face < nMid; face++) {
            c1 = f2c[count++];
            count++;

            res[0][c1] -= flux[0][i];
            res[1][c1] -= flux[1][i];
            res[2][c1] -= flux[2][i];
            res[3][c1] -= flux[3][i];
            res[4][c1] -= flux[4][i];
            i++;
        }

        // Interior faces
        for (face = nMid; face < ne; face++) {
            c1 = f2c[count++];
            c2 = f2c[count++];

            res[0][c1] -= flux[0][i];
            res[1][c1] -= flux[1][i];
            res[2][c1] -= flux[2][i];
            res[3][c1] -= flux[3][i];
            res[4][c1] -= flux[4][i];

            res[0][c2] += flux[0][i];
            res[1][c2] += flux[1][i];
            res[2][c2] += flux[2][i];
            res[3][c2] += flux[3][i];
            res[4][c2] += flux[4][i];
            i++;
        }
    }
#elif (defined FS_OPENMP) && (defined FaceColoring)
    //lrt
    IntType    nIFace = grid->GetNIFace();
    IntType  nTFace = grid->GetNTFace();
    IntType     ifacenum = nTFace - nBFace;
    IntType     pfacenum = nBFace - nIFace;
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
            startFace = 0; //for ns>0 && ns<grid_bfacegroup[0]
        }
        else {
            startFace = grid_bfacegroup[fcolor - 1];
        }
        endFace = grid_bfacegroup[fcolor];
#pragma omp parallel for
        for (IntType face = startFace; face < endFace; face++) {
            IntType  c1, c2, count;
            IntType  i;
            count = 2*face;
            c1  = f2c[count];
            i = face - ns;
            //c1  = f2c[count++];
            //count++;
        
            res[0][c1] -= flux[0][i];
            res[1][c1] -= flux[1][i];
            res[2][c1] -= flux[2][i];
            res[3][c1] -= flux[3][i];
            res[4][c1] -= flux[4][i];
            //i++;
        
        }
    }
#ifdef MPICH    
    for (IntType face = pfacenum; face < nBFace; face++) {
        IntType count = 2*face;
        IntType c1 = f2c[count];
        IntType i = face - ns;
        res[0][c1] -= flux[0][i];
        res[1][c1] -= flux[1][i];
        res[2][c1] -= flux[2][i];
        res[3][c1] -= flux[3][i];
        res[4][c1] -= flux[4][i];

    }
#endif    
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
            IntType  c1, c2, count;
            IntType  i;
            count = 2*face;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            i = face - ns;
            //c1 = f2c[count++];
            //c2 = f2c[count++];

            res[0][c1] -= flux[0][i];
            res[1][c1] -= flux[1][i];
            res[2][c1] -= flux[2][i];
            res[3][c1] -= flux[3][i];
            res[4][c1] -= flux[4][i];
        
            res[0][c2] += flux[0][i];
            res[1][c2] += flux[1][i];
            res[2][c2] += flux[2][i];
            res[3][c2] += flux[3][i];
            res[4][c2] += flux[4][i];
            //i++;
            
        }
    }
    mfmem::sdel_array_1D(grid_bfacegroup);
    mfmem::sdel_array_1D(grid_ifacegroup);
#elif (defined FS_OPENMP) && (defined Reduction)//Manual reduction
    IntType* nFPC = CalnFPC(grid);
    IntType** C2F = CalC2F(grid);
    IntType j;
#pragma omp parallel for private(i,j,count,c1,c2,face)
    for (i = 0; i < nTCell; i++) {
        for (j = 0; j < nFPC[i]; j++) {
            face = C2F[i][j];
            count = 2 * face;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            if (i == c1) {
                res[0][i] -= flux[0][face];
                res[1][i] -= flux[1][face];
                res[2][i] -= flux[2][face];
                res[3][i] -= flux[3][face];
                res[4][i] -= flux[4][face];
            }
            else if (i == c2) {
                res[0][i] += flux[0][face];
                res[1][i] += flux[1][face];
                res[2][i] += flux[2][face];
                res[3][i] += flux[3][face];
                res[4][i] += flux[4][face];
            }
            else {
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
    }
#elif (defined FS_OPENMP) && (defined DIVREP)//Division & replication
    IntType threads = grid->threads;
    IntType nTFace = grid->GetNTFace();
    IntType startFace, endFace, t, k;
    if (grid->DivRepSuccess) {
    #pragma omp parallel for private(t,i,k,startFace,endFace,c1,c2,face)
        for (t = 0; t < threads; t++) {
            //Boundary faces
            startFace = grid->idx_pthreads_bface[t];
            endFace = grid->idx_pthreads_bface[t + 1];
            for (i = startFace; i < endFace; i++) {
                face = grid->id_division_bface[i];
                c1 = f2c[2 * face];

                res[0][c1] -= flux[0][face];
                res[1][c1] -= flux[1][face];
                res[2][c1] -= flux[2][face];
                res[3][c1] -= flux[3][face];
                res[4][c1] -= flux[4][face];
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
                    res[0][c1] -= flux[0][face];
                    res[1][c1] -= flux[1][face];
                    res[2][c1] -= flux[2][face];
                    res[3][c1] -= flux[3][face];
                    res[4][c1] -= flux[4][face];

                    res[0][c2] += flux[0][face];
                    res[1][c2] += flux[1][face];
                    res[2][c2] += flux[2][face];
                    res[3][c2] += flux[3][face];
                    res[4][c2] += flux[4][face];
                }
                else {
                    if (k > 0) {
                        res[0][c1] -= flux[0][face];
                        res[1][c1] -= flux[1][face];
                        res[2][c1] -= flux[2][face];
                        res[3][c1] -= flux[3][face];
                        res[4][c1] -= flux[4][face];
                    }
                    else {
                        res[0][c2] += flux[0][face];
                        res[1][c2] += flux[1][face];
                        res[2][c2] += flux[2][face];
                        res[3][c2] += flux[3][face];
                        res[4][c2] += flux[4][face];
                    }
                }
            }
        }
    }
#elif (defined FS_OPENMP) && (defined DIVCON) //D&C TREE
#pragma omp parallel
    {
    #pragma omp single nowait
        tree_traversal(grid->treeHead, res, flux, f2c);
    }
#else
    // For boundary faces, remember c2 is ghost cell
    // Determine if there are boundary faces.
    count = 2 * ns;
    i = 0;
    for (face = ns; face < nMid; face++) {
        c1 = f2c[count++];
        count++;

        res[0][c1] -= flux[0][i];
        res[1][c1] -= flux[1][i];
        res[2][c1] -= flux[2][i];
        res[3][c1] -= flux[3][i];
        res[4][c1] -= flux[4][i];
        i++;
    }

    // Interior faces
    for (face = nMid; face < ne; face++) {
        c1 = f2c[count++];
        c2 = f2c[count++];

        res[0][c1] -= flux[0][i];
        res[1][c1] -= flux[1][i];
        res[2][c1] -= flux[2][i];
        res[3][c1] -= flux[3][i];
        res[4][c1] -= flux[4][i];

        res[0][c2] += flux[0][i];
        res[1][c2] += flux[1][i];
        res[2][c2] += flux[2][i];
        res[3][c2] += flux[3][i];
        res[4][c2] += flux[4][i];
        i++;
    }
#endif
    
}



/****************************************************************************************\
 Update residuals in cell with the fluxes at cell faces in 3D for the Central Scheme
\****************************************************************************************/
void LoadFlux(PolyGrid *grid, RealFlow *fluxl[], RealFlow *fluxr[], IntType ns, IntType ne) 
{
    IntType  face, i, c1, c2, count, nMid;
    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  *f2c   = grid->Getf2c();

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
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");
    res[1] = &res[0][nTCell];
    res[2] = &res[1][nTCell];
    res[3] = &res[2][nTCell];
    res[4] = &res[3][nTCell];
 
    // For boundary faces, remember c2 is ghost cell
    count = 2*ns;
    i     = 0;
    for(face=ns; face<nMid; face++) {
        c1  = f2c[count++];
        count++;
 
        res[0][c1] -= fluxl[0][i];
        res[1][c1] -= fluxl[1][i];
        res[2][c1] -= fluxl[2][i];
        res[3][c1] -= fluxl[3][i];
        res[4][c1] -= fluxl[4][i];
 
        i++;
    }
 
    // Interior faces
    for(face=nMid; face<ne; face++) {
        c1 = f2c[count++];
        c2 = f2c[count++];

        res[0][c1] -= fluxl[0][i];
        res[1][c1] -= fluxl[1][i];
        res[2][c1] -= fluxl[2][i];
        res[3][c1] -= fluxl[3][i];
        res[4][c1] -= fluxl[4][i];
 
        res[0][c2] += fluxr[0][i];
        res[1][c2] += fluxr[1][i];
        res[2][c2] += fluxr[2][i];
        res[3][c2] += fluxr[3][i];
        res[4][c2] += fluxr[4][i];
 
        i++;
    }
}


#ifdef DC
void ViscousFlux_Kernel(char **userArgs, uTaskTreeArgs *treeArgs)
{
	IntType ns = treeArgs->firstFace;
	IntType ne = treeArgs->lastFace + 1;
    if (ns >= ne) return; 

	PolyGrid *grid 		= (PolyGrid *)userArgs[0];
	RealFlow *t 		= (RealFlow *)userArgs[1];
	RealFlow **dvdx	 	= (RealFlow **)userArgs[2];
	RealFlow **dvdy	 	= (RealFlow **)userArgs[3];
	RealFlow **dvdz	 	= (RealFlow **)userArgs[4];
	RealFlow *dtdx 		= (RealFlow *)userArgs[5];
	RealFlow *dtdy 		= (RealFlow *)userArgs[6];
	RealFlow *dtdz 		= (RealFlow *)userArgs[7];
    RealFlow **res      = (RealFlow **)userArgs[8];
    RealFlow **vel      = (RealFlow **)userArgs[9];
	RealFlow *vis_l 	= (RealFlow *)userArgs[10];
    IntType vis_mode    = (IntType)(*(IntType *)userArgs[11]); 
    IntType cond_comp   = (IntType)(*(IntType *)userArgs[12]); 
    RealFlow gam        = (RealFlow)(*(RealFlow *)userArgs[13]);
    RealFlow gascon     = (RealFlow)(*(RealFlow *)userArgs[14]);
    RealFlow cp         = (RealFlow)(*(RealFlow *)userArgs[15]);
    RealFlow prl        = (RealFlow)(*(RealFlow *)userArgs[16]);
    RealGeom BadFaceAngle = (RealGeom)(*(RealGeom *)userArgs[17]);
	
	IntType len = ne - ns;
	IntType  nBFace = grid->GetNBFace();
    IntType  nTCell = grid->GetNTCell();
    IntType  n      = nTCell + nBFace;

    // Allocate temporary memories for Vel_f
    RealFlow *vel_f[3];
    vel_f[0]   = new RealFlow[3*len];
    vel_f[1]   = &vel_f[0][len];
    vel_f[2]   = &vel_f[1][len];
    
    // Allocate temporary memories for t_f
    RealFlow *t_f = new RealFlow[len];
    // Allocate temporary memories for visc_f and heat_f
    RealFlow *visc_f = new RealFlow[len];
    RealFlow *heat_f = new RealFlow[len];
    // Allocate temporary memories for the weights used to average dqdx, dqdy, dqdz
    RealGeom *deltl = new RealGeom[len];
    RealGeom *deltr = new RealGeom[len];
     
    // Allocate temporary memories for fluxes
    RealFlow  *flux[5];
    flux[0] = new RealFlow[5*len];
    for(IntType i=1; i<5; i++) flux[i] = &flux[i-1][len];
	
	CalDeriWeight(grid, deltl, deltr, ns, ne, 1);
	
	//average of value in cell centroid
	CalVisHeatFace_average(grid, vis_l, visc_f, heat_f, vis_mode, cond_comp, gam, gascon, cp, prl, ns, ne);
    
    CalVeloandTFace_average(grid, vel_f, vel, t_f, t, ns, ne);     
	
	CalVisFluxTest(grid, vel, t, vel_f, visc_f, heat_f, t_f,
				   dvdx, dvdy, dvdz, dtdx, dtdy, dtdz, deltl, deltr, flux, vis_mode, BadFaceAngle, ns, ne);
	
	LoadFlux_DC(grid, flux, ns, ne, res);
	
	delete[] vel_f[0];
	delete[] t_f;
	delete[] visc_f;
	delete[] heat_f;
	delete[] deltl;
	delete[] deltr;
	delete[] flux[0];
}
#endif


/*******************************************************************************\
                Drive actual function to compute viscous fluxes in 3D
\*******************************************************************************/
void ViscousFlux(PolyGrid *grid, IntType level)
{
    IntType  i, ns, ne, len;
    IntType  nBFace = grid->GetNBFace();
    IntType  nTCell = grid->GetNTCell();
    IntType  n      = nTCell + nBFace;
    IntType  nTFace = grid->GetNTFace(); 
    const IntType kNVar = 5;

    // Get velocities
    RealFlow *vel[3];
    vel[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    vel[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    vel[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    // 获取速度梯度
    RealFlow *dqdx = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdx"));
    RealFlow *dqdy = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdy"));
    RealFlow *dqdz = static_cast<RealFlow *>(
        grid->GetDataPtr(REAL_FLOW, kNVar * n, "dqdz"));
    RealFlow *dvdx[3], *dvdy[3], *dvdz[3];
    for (IntType i = 0; i < 3; ++i) {
        dvdx[i] = &dqdx[(i + 1) * n];
        dvdy[i] = &dqdy[(i + 1) * n];
        dvdz[i] = &dqdz[(i + 1) * n];
    }

    // Get temperature
    //未修改overlap
    RealFlow *t = GetTemperature(grid);
    RealFlow *dtdx = NULL;
    RealFlow *dtdy = NULL;
    RealFlow *dtdz = NULL;
    mfmem::snew_array_1D(dtdx, n,dmrfl);
    mfmem::snew_array_1D(dtdy, n,dmrfl);
    mfmem::snew_array_1D(dtdz, n,dmrfl);
	
	CompGradientQ(grid, t, dtdx, dtdy, dtdz, 0, NULL, NULL, NULL);

#ifdef MPICH
	/*
	IntType nvar = 3;
    RealFlow *q_mpi[3];
    q_mpi[0] = dtdx;
    q_mpi[1] = dtdy;
    q_mpi[2] = dtdz;
    grid->RecvSendVarNeighbor_Togeth(nvar, q_mpi);
	*/
    grid->CommInterfaceDataMPI(dtdx);
    grid->CommInterfaceDataMPI(dtdy);
    grid->CommInterfaceDataMPI(dtdz);
	
#endif
    SetGhostTemperatureGradient(grid, dtdx, dtdy, dtdz);
	
	
    //Get viscosity coefficients in each control volume
    RealFlow *vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
    if (vis_l == 0){
        printf("Should not come here! ViscousFlux!\n");
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    


#ifdef DC
    uTaskTree *uTaskTreeRoot = grid->GetuTaskTree();

    RealFlow *res[5];
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");
    res[1] = &res[0][nTCell];
    res[2] = &res[1][nTCell];
    res[3] = &res[2][nTCell];
    res[4] = &res[3][nTCell];
	
	
    IntType  vis_mode, cond_comp=1;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    grid->GetData(&cond_comp, INT, 1, "comp",0);

    RealFlow gam, gascon, cp;
    grid->GetData(&gam, REAL_FLOW, 1, "gam");
    grid->GetData(&gascon, REAL_FLOW, 1, "gascon");
    if(cond_comp == 0)grid->GetData(&cp, REAL_FLOW, 1, "cp");

    RealFlow prl;
    grid->GetData(&prl, REAL_FLOW, 1, "prl");
    
    RealGeom BadFaceAngle;
    
    grid->GetData(&BadFaceAngle, REAL_GEOM, 1, "BadFaceAngle"); 
	
	char* userArgs[18] = {(char *)grid, (char *)t, (char *)dvdx,(char *)dvdy, (char *)dvdz, (char *)dtdx, (char *)dtdy, 
                            (char *)dtdz, (char *)res, (char *)vel, (char *)vis_l, (char *)&vis_mode, (char *)&cond_comp,
                            (char *)&gam, (char *)&gascon, (char *)&cp, (char *)&prl, (char *)&BadFaceAngle};
	
#ifdef TIMECOST//dingxin
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemVis, endtimeTemVis;
    double timeuseTemVis;
    gettimeofday(&starttimeTemVis, 0); 
#endif
#endif

	uTaskTreeRoot->task_traversal(ViscousFlux_Kernel, NULL, userArgs, Forward);
	

#ifdef TIMECOST//dingxin
#ifdef MPICH
    timecost[1] = timecost[1] + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemVis, 0); 
    timeuseTemVis = (RealGeom) 1000000*(endtimeTemVis.tv_sec - starttimeTemVis.tv_sec) + endtimeTemVis.tv_usec - starttimeTemVis.tv_usec;
    timecost[1] += timeuseTemVis;
    timeuseTemVis /= 1000000.0;
    time_vis += timeuseTemVis;
#endif
#endif

#else
    len = nTFace;
    //len =SEG_LEN;
    // Allocate temporary memories for Vel_f
    RealFlow *vel_f[3];
    vel_f[0]   = NULL;
    mfmem::snew_array_1D(vel_f[0], 3*len,dmrfl);
    assert(vel_f[0] != 0);
    vel_f[1]   = &vel_f[0][len];
    vel_f[2]   = &vel_f[1][len];
    
    // Allocate temporary memories for t_f
    RealFlow *t_f = NULL;
    mfmem::snew_array_1D(t_f, len,dmrfl);
    // Allocate temporary memories for visc_f and heat_f
    RealFlow *visc_f = NULL;
    RealFlow *heat_f = NULL;  
    mfmem::snew_array_1D(visc_f, len,dmrfl);
    mfmem::snew_array_1D(heat_f, len,dmrfl);
    // Allocate temporary memories for the weights used to average dqdx, dqdy, dqdz
    RealGeom *deltl = NULL;
    RealGeom *deltr = NULL;
    mfmem::snew_array_1D(deltl, len,dmrfl);
    mfmem::snew_array_1D(deltr, len,dmrfl);
    
    // Allocate temporary memories for fluxes
    RealFlow  *flux[5];
    flux[0] = NULL;
    mfmem::snew_array_1D(flux[0], 5*len,dmrfl);
    assert(flux[0] != 0);
    for(i=1; i<5; i++) flux[i] = &flux[i-1][len];

#ifdef TIMECOST//dingxin
#ifdef MPICH
    double time_tmp;
    time_tmp = -MPI_Wtime();
#else
    struct timeval starttimeTemVis, endtimeTemVis;
    double timeuseTemVis;
    gettimeofday(&starttimeTemVis, 0); 
#endif
#endif
    ns = 0;
    do {
        //ne   = ns + SEG_LEN;
        //if(ne > nTFace) ne = nTFace;
        ne = nTFace;
        len = ne - ns;	
	
        CalDeriWeight(grid, deltl, deltr, ns, ne, 1);
		//average of value in cell centroid
		CalVisHeatFace_average(grid, vis_l, visc_f, heat_f, ns, ne);
		CalVeloandTFace_average(grid, vel_f, vel, t_f, t, ns, ne);   
		CalVisFluxTest(grid, vel, t, vel_f, visc_f, heat_f, t_f,
                       dvdx, dvdy, dvdz, dtdx, dtdy, dtdz, deltl, deltr, flux, ns, ne);
		LoadFlux(grid, flux, ns, ne);
                                                              
        ns  = ne;

    } while (ns < nTFace);

#ifdef TIMECOST//dingxin
#ifdef MPICH
    timecost[1] = timecost[1] + time_tmp + MPI_Wtime();
#else
    gettimeofday(&endtimeTemVis, 0); 
    timeuseTemVis = (RealGeom) 1000000*(endtimeTemVis.tv_sec - starttimeTemVis.tv_sec) + endtimeTemVis.tv_usec - starttimeTemVis.tv_usec;
    timecost[1] += timeuseTemVis;
    timeuseTemVis /= 1000000.0;
    time_vis += timeuseTemVis;
#endif
#endif
    
    mfmem::sdel_array_1D(vel_f[0]);
    mfmem::sdel_array_1D(t_f);
    mfmem::sdel_array_1D(flux[0]);
    mfmem::sdel_array_1D(visc_f);
    mfmem::sdel_array_1D(heat_f);
    mfmem::sdel_array_1D(deltl);
    mfmem::sdel_array_1D(deltr);
#endif

    mfmem::sdel_array_1D(t);
    mfmem::sdel_array_1D(dtdx);
    mfmem::sdel_array_1D(dtdy);
    mfmem::sdel_array_1D(dtdz);
    
}
/*******************************************************************************\
            Calculate the temperature
\*******************************************************************************/
RealFlow *GetTemperature(PolyGrid *grid)
{
    IntType i; 
    IntType nBFace = grid->GetNBFace();
    IntType n      = grid->GetNTCell() + nBFace;
    
    RealFlow p_bar, gascon;
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&gascon, REAL_FLOW, 1, "gascon"); 
    
    RealFlow *rho = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho"); 
    RealFlow *p   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p"); 
    
    RealFlow *t = NULL;
    mfmem::snew_array_1D(t,n,dmrfl);
    assert(t != 0);
    
    for(i=0; i<n; i++) t[i] = (p[i] + p_bar)/(rho[i]*gascon);
    
    return t;
}


/*******************************************************************************\
            Calculate the vis_l
\*******************************************************************************/
void ComputeVis_l(PolyGrid *grid)
{
    IntType n = grid->GetNTCell() + grid->GetNBFace();
    IntType nBFace = grid->GetNBFace();
    RealFlow *vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
    if (vis_l == 0){
        mfmem::snew_array_1D(vis_l,n,dmrfl);
        grid->UpdateDataPtr(vis_l, REAL_FLOW, n,"vis_l");
    }
    assert(vis_l != 0);
    IntType count = 0;
    IntType type, c1, c2;
    IntType *f2c   = grid->Getf2c();
    BCRecord   **bcr = grid->Getbcr(); 
    RealFlow vis_wall;

    IntType vis_mode, i;
    grid->GetData(&vis_mode,INT,1,"vis_mode");
    if( vis_mode==INVISCID ) {
        for(i=0; i<n; i++) vis_l[i] = 0.;
    }else{
        //RealFlow tref=273.0, sref=110.4, amuref=1.71e-5, temp;
        RealFlow tref=288.15, sref=110.4, amuref=1.78938e-5, temp;
        // viscosity function -- Sutherland's Law
        RealFlow *t = GetTemperature(grid);
        for(i=0; i<n; i++) {
            temp     = t[i]/tref;
            vis_l[i] = amuref*(temp*sqrt(temp)*(tref+sref)/(t[i]+sref));
        }
        for(i=0; i<nBFace; i++) {
            type  = bcr[i]->GetType(); 
            c1    = f2c[count++];
            c2    = f2c[count++];
            if(type == WALL){
                RealFlow tw = -1.0;
                bcr[i]->GetBCVar(&tw, REAL_FLOW, "tw",0);
                if(tw>0){
                    temp     = tw/tref;
                    vis_wall = amuref*(temp*sqrt(temp)*(tref+sref)/(tw+sref));
                    vis_l[c2]=2.0*vis_wall-vis_l[c1];
                }
            }
        }
        mfmem::sdel_array_1D(t);
    }
}


/*******************************************************************************\
                           Sutherland law
\*******************************************************************************/
RealFlow Sutherland_classic(RealFlow T)
{
    RealFlow amu, Trat; 
    RealFlow sref = 110.4, Tref = 288.15, amuref = 1.78938e-5;
    
    Trat = T/Tref;  
    amu = amuref*Trat*sqrt(Trat)*(Tref+sref)/(T+sref);
    
    return amu;    
}


/*******************************************************************************\
  初始化vis_t
  在euler、层流以及多重网格计算时必须使用，因为在以上情况下第一步计算时湍流模型还未做初始化，
  vis_t没有值
\*******************************************************************************/
void InitVis_t(PolyGrid *grid)
{
    IntType       n = grid->GetNTCell() + grid->GetNBFace();    
    RealFlow *vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
    if (vis_t == 0){
        mfmem::snew_array_1D(vis_t,n,dmrfl);
        grid->UpdateDataPtr(vis_t, REAL_FLOW, n,"vis_t");
    }
    assert(vis_t != 0);
    for(IntType i=0; i<n; i++) vis_t[i] = 0.;
    
}


/*******************************************************************************\
            Calculate the weights used to average dqdx,dqdy, etc
\*******************************************************************************/
void CalDeriWeight(PolyGrid *grid, RealGeom *deltl, RealGeom *deltr, IntType ns, IntType ne, IntType key)
{
    IntType  *f2c   = grid->Getf2c();
    RealGeom *xfc  = grid->GetXfc();
    RealGeom *yfc  = grid->GetYfc();
    RealGeom *zfc  = grid->GetZfc();
    RealGeom *xcc  = grid->GetXcc();
    RealGeom *ycc  = grid->GetYcc();
    RealGeom *zcc  = grid->GetZcc();
    RealGeom *xfn  = grid->GetXfn();
    RealGeom *yfn  = grid->GetYfn();
    RealGeom *zfn  = grid->GetZfn();
    RealGeom *vol  = grid->GetCellVol();
#ifdef FS_OPENMP
#pragma omp parallel for
#endif   
    for(IntType face=ns; face<ne; face++) {
        IntType i, c1, c2, count;
        RealGeom delt1, delt2, delta;
        count = 2*face;
        c1 = f2c[count];
        c2 = f2c[count+1];
        i  = face - ns;
 
        //Left
        if(key == 1){  // distance weight
            delt1 = sqrt((xcc[c1] - xfc[face])*(xcc[c1] - xfc[face])
                  +      (ycc[c1] - yfc[face])*(ycc[c1] - yfc[face])
                  +      (zcc[c1] - zfc[face])*(zcc[c1] - zfc[face]));        
        }else if(key == 2){  //normal distance weight
            delt1 = fabs((xcc[c1] - xfc[face])*xfn[face]
                  +      (ycc[c1] - yfc[face])*yfn[face]
                  +      (zcc[c1] - zfc[face])*zfn[face]);
        }else if(key == 3){  //volume weight
            delt1 = vol[c1];
        }
 
        // Right
        if(key == 1){   // distance weight
            delt2 = sqrt((xcc[c2] - xfc[face])*(xcc[c2] - xfc[face])
                  +      (ycc[c2] - yfc[face])*(ycc[c2] - yfc[face])
                  +      (zcc[c2] - zfc[face])*(zcc[c2] - zfc[face]));
        }else if(key == 2){   //normal distance weight
            delt2 = fabs((xcc[c2] - xfc[face])*xfn[face]
                  +      (ycc[c2] - yfc[face])*yfn[face]
                  +      (zcc[c2] - zfc[face])*zfn[face]);
        }else if(key == 3){  //volume weight
            delt2 = vol[c2];
        }
 
        delta    = 1./(delt1 + delt2 + TINY);
        deltl[i] = delt2*delta;
        deltr[i] = delt1*delta;
    }  
} 


/*******************************************************************************\
         Calculate velocity and temperature at cell face
                        use simple average
\*******************************************************************************/
void CalVeloandTFace_average(PolyGrid *grid, RealFlow *vel_f[3], RealFlow *vel[3],
                             RealFlow *t_f, RealFlow *t, IntType ns, IntType ne)
{
    
    IntType *f2c   = grid->Getf2c();
#ifndef DC
#ifdef FS_OPENMP
#pragma omp parallel for
#endif  
#endif  
    for(IntType face=ns; face<ne; face++) {
        IntType i, c1, c2, count;
        count = 2*face;
        c1          = f2c[count];
        c2          = f2c[count+1];
        i           = face - ns;
        
        vel_f[0][i] = 0.5*(vel[0][c1] + vel[0][c2]);
        vel_f[1][i] = 0.5*(vel[1][c1] + vel[1][c2]);
        vel_f[2][i] = 0.5*(vel[2][c1] + vel[2][c2]);
        
        t_f[i] = 0.5*(t[c1]+t[c2]);
    }    
}
/*******************************************************************************\
     Calculate viscosity and heat coefficients at cell face
\*******************************************************************************/
void CalVisHeatFace_average(PolyGrid *grid, RealFlow *vis_l, RealFlow *visc_f, RealFlow *heat_f, 
#ifdef DC
                                IntType vis_mode, IntType cond_comp, RealFlow gam, RealFlow gascon, 
                                RealFlow cp, RealFlow prl, 
#endif
                                IntType ns, IntType ne)
{

#ifndef DC
    // Get parameters
    IntType  vis_mode, cond_comp=1;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    grid->GetData(&cond_comp, INT, 1, "comp",0);

    // Get specific heat ratio, gas constant, cp
    RealFlow gam, gascon, cp;
    grid->GetData(&gam, REAL_FLOW, 1, "gam");
    grid->GetData(&gascon, REAL_FLOW, 1, "gascon");
    cp = gascon*gam/(gam - 1.);
    if(cond_comp == 0)grid->GetData(&cp, REAL_FLOW, 1, "cp");

    // Get viscosity, Prandtl number
    RealFlow prl;
    grid->GetData(&prl, REAL_FLOW, 1, "prl");
#endif

    RealFlow heat = cp/prl;
    
    IntType *f2c = grid->Getf2c();
    // Laminar Flows
#ifndef DC
#ifdef FS_OPENMP
#pragma omp parallel for
#endif
#endif
    for(IntType face=ns; face<ne; face++) {
        IntType      count=2*face;
        IntType      c1, c2, i;
        i         = face - ns;
        c1        = f2c[count];
        c2        = f2c[count+1];
        visc_f[i] = 0.5*(vis_l[c1]+vis_l[c2]);
        heat_f[i] = heat*visc_f[i];
    }
 
    //Turbulent viscosity (Eddy viscosity?)
    if(vis_mode == S_A_MODEL) {
       IntType      n      = grid->GetNTCell() + grid->GetNBFace();
       
       // Note: the size of vis_t
       RealFlow *vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");
       RealFlow prt;
       
       grid->GetData(&prt, REAL_FLOW, 1, "prt");
       heat  = cp/prt;
#ifndef DC
#ifdef FS_OPENMP
#pragma omp parallel for
#endif
#endif  
       for(IntType face=ns; face<ne; face++) {
           IntType      count = 2*face;
           IntType      c1, c2, i;
           RealFlow     tmp;
           i          = face - ns;
           c1         = f2c[count];
           c2         = f2c[count+1];
 
           tmp        = 0.5*(vis_t[c1] + vis_t[c2]);
           visc_f[i] += tmp;
           heat_f[i] += heat*tmp;
      }  
    } 
}
/*******************************************************************************\
 Actually calculate viscous fluxes in 3D ~~ the most recent version ~~~
\*******************************************************************************/
void CalVisFluxTest(PolyGrid *grid, RealFlow *vel[3], RealFlow *t, RealFlow *vel_f[3],
                    RealFlow *visc_f, RealFlow *heat_f, RealFlow *t_f,
                    RealFlow *dqdx[3], RealFlow *dqdy[3], RealFlow *dqdz[3],
                    RealFlow *dtdx, RealFlow *dtdy, RealFlow *dtdz,
                    RealGeom *deltl, RealGeom *deltr, RealFlow *flux[5],
#ifdef DC
                    IntType vis_mode, RealGeom BadFaceAngle,
#endif
                    IntType ns, IntType ne)
{
    IntType nTFace = grid->GetNTFace();
    IntType nBFace = grid->GetNBFace();
    IntType *f2c   = grid->Getf2c();
    IntType level  = grid->GetLevel();
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
    BCRecord **bcr = grid->Getbcr();    
    
#ifndef DC
    RealGeom BadFaceAngle=-1.0;
    grid->GetData(&BadFaceAngle, REAL_GEOM, 1, "BadFaceAngle");  
    
    IntType vis_mode;
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
#endif
    
    RealGeom *facecentroidskewness = grid->GetGridQualityFaceCentroidSkewness();
    
    IntType    n = grid->GetNTCell() + grid->GetNBFace();
           
    IntType face, i, count, c1, c2, type;
    RealGeom two3, areax, areay, areaz;
    RealFlow umid, vmid, wmid, tmid, d_vis, heat_con, tw;
    RealFlow t1x, t1y, t1z, t2x, t2y, t2z;
    RealFlow dtmp, d1, d2, u1, u2, v1, v2, w1, w2, t1, t2, x1, x2, y1, y2, z1, z2;
    RealFlow dud1, dud2, dvd1, dvd2, dwd1, dwd2, dtd1, dtd2;
    RealFlow dudn, dvdn, dwdn, dtdn;
    RealFlow dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz;
    RealFlow txx, tyy, tzz, txy, txz, tyz;
    RealFlow dudt1, dvdt1, dwdt1, dudt2, dvdt2, dwdt2;
    RealFlow angle1, angle2;
    RealFlow delta;
    static IntType warn = 1;
    two3 = 2.0/3.0;
    
    count = 2*ns;
    
#if (defined FS_SIMD) && (!defined FS_SIMD_AVX) && (!defined Tile)
//containing SIMD
    //const IntType    Vec = 8;
    //IntType i, ni;
    //IntType k;
    
#ifdef FS_OPENMP //OpenMP && SIMD
#pragma omp parallel for
#endif
    for (face = ns; face < ne - Vec; face += Vec) {
        IntType   i_v[Vec]             __attribute__((aligned(ALIGN)));
        IntType   count_v[Vec]             __attribute__((aligned(ALIGN)));
        IntType   c1_v[Vec]             __attribute__((aligned(ALIGN)));
        IntType   c2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  xfn_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  yfn_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  zfn_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  t1x_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  t1y_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  t1z_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  t2x_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  t2y_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  t2z_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dtmp_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  x1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  x2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  y1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  y2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  z1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  z2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  d1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  d2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  angle1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  angle2_v[Vec]             __attribute__((aligned(ALIGN)));

        //Load:
#pragma omp simd safelen(Vec)        
        for (IntType iv = 0; iv < Vec; iv++) {
            i_v[iv] = face + iv - ns;
            count_v[iv] = 2 * ns + 2 * i_v[iv];
            c1_v[iv] = f2c[count_v[iv]];
            c2_v[iv] = f2c[count_v[iv] + 1];
        }
        //Load:
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            xfn_v[iv] = xfn[face + iv];
            yfn_v[iv] = yfn[face + iv];
            zfn_v[iv] = zfn[face + iv];
        }
//#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            // Get first tangential vector on the face
            if (xfn_v[iv] != 0.) {
                t1x_v[iv] = yfn_v[iv];
                t1y_v[iv] = -xfn_v[iv];
                t1z_v[iv] = 0.;
            }
            else if (areay != 0.) {
                t1x_v[iv] = -yfn_v[iv];
                t1y_v[iv] = xfn_v[iv];
                t1z_v[iv] = 0.;
            }
            else if (areaz != 0.) {
                t1x_v[iv] = 0.;
                t1y_v[iv] = -zfn_v[iv];
                t1z_v[iv] = yfn_v[iv];
            }
            else {
                if (warn) printf("Warninng: %ldth Face is singular\n", (long)face);
                flux[0][i_v[iv]] = 0.0;
                flux[1][i_v[iv]] = 0.0;
                flux[2][i_v[iv]] = 0.0;
                flux[3][i_v[iv]] = 0.0;
                flux[4][i_v[iv]] = 0.0;
                if (warn != 0 && ne == nTFace) warn = 0;
                continue;
            }
        }
#pragma omp simd safelen(Vec)        
        for (IntType iv = 0; iv < Vec; iv++) {
            // normalize the tangential vector
            dtmp_v[iv] = sqrt(t1x_v[iv] * t1x_v[iv] + t1y_v[iv] * t1y_v[iv] + t1z_v[iv] * t1z_v[iv]);
            t1x_v[iv] /= dtmp_v[iv];
            t1y_v[iv] /= dtmp_v[iv];
            t1z_v[iv] /= dtmp_v[iv];
        }
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            // Get second tangential vector by cross dot t1 to normal
            t2x_v[iv] = yfn_v[iv] * t1z_v[iv] - zfn_v[iv] * t1y_v[iv];
            t2y_v[iv] = zfn_v[iv] * t1x_v[iv] - xfn_v[iv] * t1z_v[iv];
            t2z_v[iv] = xfn_v[iv] * t1y_v[iv] - yfn_v[iv] * t1x_v[iv];
        }
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            // positions
            x1_v[iv] = xcc[c1_v[iv]] - xfc[face + iv];
            y1_v[iv] = ycc[c1_v[iv]] - yfc[face + iv];
            z1_v[iv] = zcc[c1_v[iv]] - zfc[face + iv];
            x2_v[iv] = xcc[c2_v[iv]] - xfc[face + iv];
            y2_v[iv] = ycc[c2_v[iv]] - yfc[face + iv];
            z2_v[iv] = zcc[c2_v[iv]] - zfc[face + iv];
        }
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            d1_v[iv] = x1_v[iv] * xfn_v[iv] + y1_v[iv] * yfn_v[iv] + z1_v[iv] * zfn_v[iv];
            d2_v[iv] = x2_v[iv] * xfn_v[iv] + y2_v[iv] * yfn_v[iv] + z2_v[iv] * zfn_v[iv];
        }
#pragma omp simd safelen(Vec)        
        for (IntType iv = 0; iv < Vec; iv++) {
            dtmp_v[iv] = -d1_v[iv] / (sqrt(x1_v[iv] * x1_v[iv] + y1_v[iv] * y1_v[iv] + z1_v[iv] * z1_v[iv]) + TINY);
            if (dtmp_v[iv] > 1.0) dtmp_v[iv] = 1.0;
            if (dtmp_v[iv] < -1.0) dtmp_v[iv] = -1.0;
            angle1_v[iv] = asin(dtmp_v[iv]) * 180.0 / PI;
        }
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            dtmp_v[iv] = d2_v[iv] / (sqrt(x2_v[iv] * x2_v[iv] + y2_v[iv] * y2_v[iv] + z2_v[iv] * z2_v[iv]) + TINY);
            if (dtmp_v[iv] > 1.0) dtmp_v[iv] = 1.0;
            if (dtmp_v[iv] < -1.0) dtmp_v[iv] = -1.0;
            angle2_v[iv] = asin(dtmp_v[iv]) * 180.0 / PI;
        }
                
        RealFlow  u1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  v1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  w1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  t1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  u2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  v2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  w2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  t2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  umid_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  vmid_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  wmid_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  tmid_v[Vec]             __attribute__((aligned(ALIGN)));
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            // quentities at points 1 and 2
            u1_v[iv] = vel[0][c1_v[iv]];
            v1_v[iv] = vel[1][c1_v[iv]];
            w1_v[iv] = vel[2][c1_v[iv]];
            t1_v[iv] = t[c1_v[iv]];
            u2_v[iv] = vel[0][c2_v[iv]];
            v2_v[iv] = vel[1][c2_v[iv]];
            w2_v[iv] = vel[2][c2_v[iv]];
            t2_v[iv] = t[c2_v[iv]];
        }
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            umid_v[iv] = 0.5 * (u1_v[iv] + u2_v[iv]);
            vmid_v[iv] = 0.5 * (v1_v[iv] + v2_v[iv]);
            wmid_v[iv] = 0.5 * (w1_v[iv] + w2_v[iv]);
            tmid_v[iv] = 0.5 * (t1_v[iv] + t2_v[iv]);
        }
#pragma omp simd safelen(Vec)        
        for (IntType iv = 0; iv < Vec; iv++) {
            if (angle1_v[iv] > 10.0 && angle2_v[iv] > 10.0) {
                u1_v[iv] += dqdx[0][c1_v[iv]] * (d1_v[iv] * xfn_v[iv] - x1_v[iv]) + dqdy[0][c1_v[iv]] * (d1_v[iv] * yfn_v[iv] - y1_v[iv]) + dqdz[0][c1_v[iv]] * (d1_v[iv] * zfn_v[iv] - z1_v[iv]);
                v1_v[iv] += dqdx[1][c1_v[iv]] * (d1_v[iv] * xfn_v[iv] - x1_v[iv]) + dqdy[1][c1_v[iv]] * (d1_v[iv] * yfn_v[iv] - y1_v[iv]) + dqdz[1][c1_v[iv]] * (d1_v[iv] * zfn_v[iv] - z1_v[iv]);
                w1_v[iv] += dqdx[2][c1_v[iv]] * (d1_v[iv] * xfn_v[iv] - x1_v[iv]) + dqdy[2][c1_v[iv]] * (d1_v[iv] * yfn_v[iv] - y1_v[iv]) + dqdz[2][c1_v[iv]] * (d1_v[iv] * zfn_v[iv] - z1_v[iv]);

                u2_v[iv] += dqdx[0][c2_v[iv]] * (d2_v[iv] * xfn_v[iv] - x2_v[iv]) + dqdy[0][c2_v[iv]] * (d2_v[iv] * yfn_v[iv] - y2_v[iv]) + dqdz[0][c2_v[iv]] * (d2_v[iv] * zfn_v[iv] - z2_v[iv]);
                v2_v[iv] += dqdx[1][c2_v[iv]] * (d2_v[iv] * xfn_v[iv] - x2_v[iv]) + dqdy[1][c2_v[iv]] * (d2_v[iv] * yfn_v[iv] - y2_v[iv]) + dqdz[1][c2_v[iv]] * (d2_v[iv] * zfn_v[iv] - z2_v[iv]);
                w2_v[iv] += dqdx[2][c2_v[iv]] * (d2_v[iv] * xfn_v[iv] - x2_v[iv]) + dqdy[2][c2_v[iv]] * (d2_v[iv] * yfn_v[iv] - y2_v[iv]) + dqdz[2][c2_v[iv]] * (d2_v[iv] * zfn_v[iv] - z2_v[iv]);

                t1_v[iv] += dtdx[c1_v[iv]] * (d1_v[iv] * xfn_v[iv] - x1_v[iv]) + dtdy[c1_v[iv]] * (d1_v[iv] * yfn_v[iv] - y1_v[iv]) + dtdz[c1_v[iv]] * (d1_v[iv] * zfn_v[iv] - z1_v[iv]);
                t2_v[iv] += dtdx[c2_v[iv]] * (d2_v[iv] * xfn_v[iv] - x2_v[iv]) + dtdy[c2_v[iv]] * (d2_v[iv] * yfn_v[iv] - y2_v[iv]) + dtdz[c2_v[iv]] * (d2_v[iv] * zfn_v[iv] - z2_v[iv]);
                if (t1_v[iv] < TINY) t1_v[iv] = t[c1_v[iv]];
                if (t2_v[iv] < TINY) t2_v[iv] = t[c2_v[iv]];

                // quantities at the face
                umid_v[iv] = vel_f[0][i_v[iv]];
                vmid_v[iv] = vel_f[1][i_v[iv]];
                wmid_v[iv] = vel_f[2][i_v[iv]];
                tmid_v[iv] = t_f[i_v[iv]];
            }
        }
        
        
        RealFlow  dudx_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dudy_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dudz_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dvdx_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dvdy_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dvdz_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dwdx_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dwdy_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dwdz_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dudn_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dvdn_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dwdn_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dtdn_v[Vec]             __attribute__((aligned(ALIGN)));
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            dudx_v[iv] = dqdx[0][c1_v[iv]] * deltl[i_v[iv]] + dqdx[0][c2_v[iv]] * deltr[i_v[iv]];
            dudy_v[iv] = dqdy[0][c1_v[iv]] * deltl[i_v[iv]] + dqdy[0][c2_v[iv]] * deltr[i_v[iv]];
            dudz_v[iv] = dqdz[0][c1_v[iv]] * deltl[i_v[iv]] + dqdz[0][c2_v[iv]] * deltr[i_v[iv]];
            dvdx_v[iv] = dqdx[1][c1_v[iv]] * deltl[i_v[iv]] + dqdx[1][c2_v[iv]] * deltr[i_v[iv]];
            dvdy_v[iv] = dqdy[1][c1_v[iv]] * deltl[i_v[iv]] + dqdy[1][c2_v[iv]] * deltr[i_v[iv]];
            dvdz_v[iv] = dqdz[1][c1_v[iv]] * deltl[i_v[iv]] + dqdz[1][c2_v[iv]] * deltr[i_v[iv]];
            dwdx_v[iv] = dqdx[2][c1_v[iv]] * deltl[i_v[iv]] + dqdx[2][c2_v[iv]] * deltr[i_v[iv]];
            dwdy_v[iv] = dqdy[2][c1_v[iv]] * deltl[i_v[iv]] + dqdy[2][c2_v[iv]] * deltr[i_v[iv]];
            dwdz_v[iv] = dqdz[2][c1_v[iv]] * deltl[i_v[iv]] + dqdz[2][c2_v[iv]] * deltr[i_v[iv]];
        }
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            dudn_v[iv] = 0.0;
            dvdn_v[iv] = 0.0;
            dwdn_v[iv] = 0.0;
            dtdn_v[iv] = 0.0;
        }
        
        RealFlow  dud1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dvd1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dwd1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dtd1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dud2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dvd2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dwd2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  dtd2_v[Vec]             __attribute__((aligned(ALIGN)));
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            if (angle1_v[iv] > 0.0 && angle2_v[iv] > 0.0 && fabs(d1_v[iv]) > TINY && fabs(d2_v[iv]) > TINY) {
                dud1_v[iv] = (u1_v[iv] - umid_v[iv]) / d1_v[iv];
                dvd1_v[iv] = (v1_v[iv] - vmid_v[iv]) / d1_v[iv];
                dwd1_v[iv] = (w1_v[iv] - wmid_v[iv]) / d1_v[iv];
                dtd1_v[iv] = (t1_v[iv] - tmid_v[iv]) / d1_v[iv];
                dud2_v[iv] = (u2_v[iv] - umid_v[iv]) / d2_v[iv];
                dvd2_v[iv] = (v2_v[iv] - vmid_v[iv]) / d2_v[iv];
                dwd2_v[iv] = (w2_v[iv] - wmid_v[iv]) / d2_v[iv];
                dtd2_v[iv] = (t2_v[iv] - tmid_v[iv]) / d2_v[iv];
                dtmp_v[iv] = d1_v[iv] * d1_v[iv] + d2_v[iv] * d2_v[iv];
                d1_v[iv] = d1_v[iv] * d1_v[iv] / dtmp_v[iv];
                d2_v[iv] = d2_v[iv] * d2_v[iv] / dtmp_v[iv];
                dudn_v[iv] = dud1_v[iv] * d1_v[iv] + dud2_v[iv] * d2_v[iv];
                dvdn_v[iv] = dvd1_v[iv] * d1_v[iv] + dvd2_v[iv] * d2_v[iv];
                dwdn_v[iv] = dwd1_v[iv] * d1_v[iv] + dwd2_v[iv] * d2_v[iv];
                dtdn_v[iv] = dtd1_v[iv] * d1_v[iv] + dtd2_v[iv] * d2_v[iv];
            }
        }


        RealFlow dudt1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow dvdt1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow dwdt1_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow dudt2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow dvdt2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow dwdt2_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow delta_v[Vec]             __attribute__((aligned(ALIGN)));
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            // dqdt, does not matter too much
            dudt1_v[iv] = dudx_v[iv] * t1x_v[iv] + dudy_v[iv] * t1y_v[iv] + dudz_v[iv] * t1z_v[iv];
            dvdt1_v[iv] = dvdx_v[iv] * t1x_v[iv] + dvdy_v[iv] * t1y_v[iv] + dvdz_v[iv] * t1z_v[iv];
            dwdt1_v[iv] = dwdx_v[iv] * t1x_v[iv] + dwdy_v[iv] * t1y_v[iv] + dwdz_v[iv] * t1z_v[iv];
            dudt2_v[iv] = dudx_v[iv] * t2x_v[iv] + dudy_v[iv] * t2y_v[iv] + dudz_v[iv] * t2z_v[iv];
            dvdt2_v[iv] = dvdx_v[iv] * t2x_v[iv] + dvdy_v[iv] * t2y_v[iv] + dvdz_v[iv] * t2z_v[iv];
            dwdt2_v[iv] = dwdx_v[iv] * t2x_v[iv] + dwdy_v[iv] * t2y_v[iv] + dwdz_v[iv] * t2z_v[iv];
        }
#pragma omp simd safelen(Vec)        
        for (IntType iv = 0; iv < Vec; iv++) {
            // now true gradients
            dudx_v[iv] = dudn_v[iv] * xfn_v[iv] + dudt1_v[iv] * t1x_v[iv] + dudt2_v[iv] * t2x_v[iv];
            dudy_v[iv] = dudn_v[iv] * yfn_v[iv] + dudt1_v[iv] * t1y_v[iv] + dudt2_v[iv] * t2y_v[iv];
            dudz_v[iv] = dudn_v[iv] * zfn_v[iv] + dudt1_v[iv] * t1z_v[iv] + dudt2_v[iv] * t2z_v[iv];
            dvdx_v[iv] = dvdn_v[iv] * xfn_v[iv] + dvdt1_v[iv] * t1x_v[iv] + dvdt2_v[iv] * t2x_v[iv];
            dvdy_v[iv] = dvdn_v[iv] * yfn_v[iv] + dvdt1_v[iv] * t1y_v[iv] + dvdt2_v[iv] * t2y_v[iv];
            dvdz_v[iv] = dvdn_v[iv] * zfn_v[iv] + dvdt1_v[iv] * t1z_v[iv] + dvdt2_v[iv] * t2z_v[iv];
            dwdx_v[iv] = dwdn_v[iv] * xfn_v[iv] + dwdt1_v[iv] * t1x_v[iv] + dwdt2_v[iv] * t2x_v[iv];
            dwdy_v[iv] = dwdn_v[iv] * yfn_v[iv] + dwdt1_v[iv] * t1y_v[iv] + dwdt2_v[iv] * t2y_v[iv];
            dwdz_v[iv] = dwdn_v[iv] * zfn_v[iv] + dwdt1_v[iv] * t1z_v[iv] + dwdt2_v[iv] * t2z_v[iv];
        }
#pragma omp simd safelen(Vec)       
        for (IntType iv = 0; iv < Vec; iv++) {
            if (level == 0 && BadFaceAngle > 0.0 && facecentroidskewness[face + iv] < BadFaceAngle) {
                dudx_v[iv] = dudn_v[iv] * xfn_v[iv];
                dudy_v[iv] = dudn_v[iv] * yfn_v[iv];
                dudz_v[iv] = dudn_v[iv] * zfn_v[iv];
                dvdx_v[iv] = dvdn_v[iv] * xfn_v[iv];
                dvdy_v[iv] = dvdn_v[iv] * yfn_v[iv];
                dvdz_v[iv] = dvdn_v[iv] * zfn_v[iv];
                dwdx_v[iv] = dwdn_v[iv] * xfn_v[iv];
                dwdy_v[iv] = dwdn_v[iv] * yfn_v[iv];
                dwdz_v[iv] = dwdn_v[iv] * zfn_v[iv];
            }
        }
        IntType type_v[Vec];//             __attribute__((aligned(ALIGN)));
        RealFlow  tw_v[Vec]             __attribute__((aligned(ALIGN)));
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            if ((face + iv) < nBFace) {
                type_v[iv] = bcr[face + iv]->GetType();
                if (type_v[iv] != WALL && type_v[iv] != SYMM && type_v[iv] != FAR_FIELD && type_v[iv] != INTERFACE) {
                    delta_v[iv] = sqrt((xcc[c1_v[iv]] - xcc[c2_v[iv]]) * (xcc[c1_v[iv]] - xcc[c2_v[iv]]) +
                        (ycc[c1_v[iv]] - ycc[c2_v[iv]]) * (ycc[c1_v[iv]] - ycc[c2_v[iv]]) +
                        (zcc[c1_v[iv]] - zcc[c2_v[iv]]) * (zcc[c1_v[iv]] - zcc[c2_v[iv]]));

                    dvdn_v[iv] = (vel[0][c2_v[iv]] - vel[0][c1_v[iv]]) / delta_v[iv];
                    dudx_v[iv] = dvdn_v[iv] * xfn_v[iv];
                    dudy_v[iv] = dvdn_v[iv] * yfn_v[iv];
                    dudz_v[iv] = dvdn_v[iv] * zfn_v[iv];

                    dvdn_v[iv] = (vel[1][c2_v[iv]] - vel[1][c1_v[iv]]) / delta_v[iv];
                    dvdx_v[iv] = dvdn_v[iv] * xfn_v[iv];
                    dvdy_v[iv] = dvdn_v[iv] * yfn_v[iv];
                    dvdz_v[iv] = dvdn_v[iv] * zfn_v[iv];

                    dvdn_v[iv] = (vel[2][c2_v[iv]] - vel[2][c1_v[iv]]) / delta_v[iv];
                    dwdx_v[iv] = dvdn_v[iv] * xfn_v[iv];
                    dwdy_v[iv] = dvdn_v[iv] * yfn_v[iv];
                    dwdz_v[iv] = dvdn_v[iv] * zfn_v[iv];

                    dtdn_v[iv] = (t[c2_v[iv]] - t[c1_v[iv]]) / delta_v[iv];
                }

                //for aerodynamic heating!
                if (type_v[iv] == WALL) {
                    tw_v[iv] = -1.0;
                    bcr[face + iv]->GetBCVar(&tw_v[iv], REAL_FLOW, "tw", 0);
                    if (tw_v[iv] > 0.0) {
                        delta_v[iv] = (xfc[face] - xcc[c1_v[iv]]) * xfn_v[iv] +
                            (yfc[face] - ycc[c1_v[iv]]) * yfn_v[iv] +
                            (zfc[face] - zcc[c1_v[iv]]) * zfn_v[iv];
                        dtdn_v[iv] = (tw_v[iv] - t[c1_v[iv]]) / delta_v[iv];
                    }
                }
            }
        }
        
        RealFlow  d_vis_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow  heat_con_v[Vec]             __attribute__((aligned(ALIGN)));
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            // Get velocity at the face
            d_vis_v[iv] = visc_f[i_v[iv]];
            heat_con_v[iv] = heat_f[i_v[iv]];
        }

        RealFlow txx_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow tyy_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow tzz_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow txy_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow txz_v[Vec]             __attribute__((aligned(ALIGN)));
        RealFlow tyz_v[Vec]             __attribute__((aligned(ALIGN)));
#pragma omp simd safelen(Vec)
        for (IntType iv = 0; iv < Vec; iv++) {
            txx_v[iv] = (2. * dudx_v[iv] - dvdy_v[iv] - dwdz_v[iv]) * two3;
            tyy_v[iv] = (2. * dvdy_v[iv] - dudx_v[iv] - dwdz_v[iv]) * two3;
            tzz_v[iv] = (2. * dwdz_v[iv] - dudx_v[iv] - dvdy_v[iv]) * two3;
            txy_v[iv] = dudy_v[iv] + dvdx_v[iv];
            txz_v[iv] = dudz_v[iv] + dwdx_v[iv];
            tyz_v[iv] = dwdy_v[iv] + dvdz_v[iv];
        }
#pragma omp simd safelen(Vec)        
        for (IntType iv = 0; iv < Vec; iv++) {
            flux[0][i_v[iv]] = 0.;
            flux[1][i_v[iv]] = -d_vis_v[iv] * (txx_v[iv] * xfn_v[iv] + txy_v[iv] * yfn_v[iv] + txz_v[iv] * zfn_v[iv]) * area[face + iv];
            flux[2][i_v[iv]] = -d_vis_v[iv] * (txy_v[iv] * xfn_v[iv] + tyy_v[iv] * yfn_v[iv] + tyz_v[iv] * zfn_v[iv]) * area[face + iv];
            flux[3][i_v[iv]] = -d_vis_v[iv] * (txz_v[iv] * xfn_v[iv] + tyz_v[iv] * yfn_v[iv] + tzz_v[iv] * zfn_v[iv]) * area[face + iv];
            flux[4][i_v[iv]] = umid_v[iv] * flux[1][i_v[iv]] + vmid_v[iv] * flux[2][i_v[iv]] + wmid_v[iv] * flux[3][i_v[iv]]
                - dtdn_v[iv] * heat_con_v[iv] * area[face + iv];
        }

    }
    
    IntType k = face;
    for (face = k; face < ne; face++) {
        i     = face - ns;
        count = 2*ns+2*i;
        c1    = f2c[count];
        c2    = f2c[count+1];
        
        areax = xfn[face];
        areay = yfn[face];
        areaz = zfn[face];
        
        // Get first tangential vector on the face
        if(areax != 0.) {
            t1x =  areay;
            t1y = -areax;
            t1z =  0.;
        } else if(areay != 0.) {
            t1x = -areay;
            t1y =  areax;
            t1z =  0.;
        } else if(areaz != 0.) {
            t1x =  0.;
            t1y = -areaz;
            t1z =  areay;
        } else {
            if(warn) printf("Warninng: %ldth Face is singular\n", (long)face);
            flux[0][i] = 0.0;
            flux[1][i] = 0.0;
            flux[2][i] = 0.0;
            flux[3][i] = 0.0;
            flux[4][i] = 0.0;
            if(warn !=0 && ne == nTFace) warn = 0;
            continue;
        }
        
        // normalize the tangential vector
        dtmp = sqrt(t1x*t1x + t1y*t1y + t1z*t1z);
        t1x /= dtmp;
        t1y /= dtmp;
        t1z /= dtmp;
        
        // Get second tangential vector by cross dot t1 to normal
        t2x = areay*t1z - areaz*t1y;
        t2y = areaz*t1x - areax*t1z;
        t2z = areax*t1y - areay*t1x;
        
        // positions
        x1 = xcc[c1]  - xfc[face];
        y1 = ycc[c1]  - yfc[face];
        z1 = zcc[c1]  - zfc[face];
        x2 = xcc[c2]  - xfc[face];
        y2 = ycc[c2]  - yfc[face];
        z2 = zcc[c2]  - zfc[face];
        d1 = x1*areax + y1*areay + z1*areaz;
        d2 = x2*areax + y2*areay + z2*areaz;
        
        dtmp = -d1/(sqrt(x1*x1+ y1*y1 + z1*z1) + TINY);
        if(dtmp >  1.0) dtmp =  1.0;
        if(dtmp < -1.0) dtmp = -1.0;
        angle1 = asin(dtmp)*180.0/PI;
        
        dtmp = d2/(sqrt(x2*x2+ y2*y2 + z2*z2) + TINY);
        if(dtmp >  1.0) dtmp =  1.0;
        if(dtmp < -1.0) dtmp = -1.0;
        angle2 = asin(dtmp)*180.0/PI;
        
        // quentities at points 1 and 2
        u1   = vel[0][c1];
        v1   = vel[1][c1];
        w1   = vel[2][c1];
        t1   = t[c1];
        u2   = vel[0][c2];
        v2   = vel[1][c2];
        w2   = vel[2][c2];
        t2   = t[c2];
        umid = 0.5*(u1 + u2);
        vmid = 0.5*(v1 + v2);
        wmid = 0.5*(w1 + w2);
        tmid = 0.5*(t1 + t2);
        
        // Theroretically, more accurate to include the following terms
        if(angle1 > 10.0 && angle2 > 10.0) {
            u1 += dqdx[0][c1]*(d1*areax - x1) + dqdy[0][c1]*(d1*areay - y1) + dqdz[0][c1]*(d1*areaz - z1);
            v1 += dqdx[1][c1]*(d1*areax - x1) + dqdy[1][c1]*(d1*areay - y1) + dqdz[1][c1]*(d1*areaz - z1);
            w1 += dqdx[2][c1]*(d1*areax - x1) + dqdy[2][c1]*(d1*areay - y1) + dqdz[2][c1]*(d1*areaz - z1);
            
            u2 += dqdx[0][c2]*(d2*areax - x2) + dqdy[0][c2]*(d2*areay - y2) + dqdz[0][c2]*(d2*areaz - z2);
            v2 += dqdx[1][c2]*(d2*areax - x2) + dqdy[1][c2]*(d2*areay - y2) + dqdz[1][c2]*(d2*areaz - z2);
            w2 += dqdx[2][c2]*(d2*areax - x2) + dqdy[2][c2]*(d2*areay - y2) + dqdz[2][c2]*(d2*areaz - z2);
            
            t1 += dtdx[c1]*(d1*areax - x1)    + dtdy[c1]*(d1*areay - y1)    + dtdz[c1]*(d1*areaz - z1);
            t2 += dtdx[c2]*(d2*areax - x2)    + dtdy[c2]*(d2*areay - y2)    + dtdz[c2]*(d2*areaz - z2);
            if(t1 < TINY) t1  = t[c1];
            if(t2 < TINY) t2  = t[c2];
            
            // quantities at the face
            umid = vel_f[0][i];
            vmid = vel_f[1][i];
            wmid = vel_f[2][i];
            tmid = t_f[i];
        }
        
        dudx  = dqdx[0][c1]*deltl[i] + dqdx[0][c2]*deltr[i];
        dudy  = dqdy[0][c1]*deltl[i] + dqdy[0][c2]*deltr[i];
        dudz  = dqdz[0][c1]*deltl[i] + dqdz[0][c2]*deltr[i];
        dvdx  = dqdx[1][c1]*deltl[i] + dqdx[1][c2]*deltr[i];
        dvdy  = dqdy[1][c1]*deltl[i] + dqdy[1][c2]*deltr[i];
        dvdz  = dqdz[1][c1]*deltl[i] + dqdz[1][c2]*deltr[i];
        dwdx  = dqdx[2][c1]*deltl[i] + dqdx[2][c2]*deltr[i];
        dwdy  = dqdy[2][c1]*deltl[i] + dqdy[2][c2]*deltr[i];
        dwdz  = dqdz[2][c1]*deltl[i] + dqdz[2][c2]*deltr[i];
        
        dudn  = 0.0;
        dvdn  = 0.0;
        dwdn  = 0.0;
        dtdn  = 0.0;
        
        if(angle1 > 0.0 && angle2 > 0.0 && fabs(d1) > TINY && fabs(d2) > TINY) {
            dud1 = (u1 - umid)/d1;
            dvd1 = (v1 - vmid)/d1;
            dwd1 = (w1 - wmid)/d1;
            dtd1 = (t1 - tmid)/d1;
            dud2 = (u2 - umid)/d2;
            dvd2 = (v2 - vmid)/d2;
            dwd2 = (w2 - wmid)/d2;
            dtd2 = (t2 - tmid)/d2;
            dtmp = d1*d1 + d2*d2;
            d1   = d1*d1/dtmp;
            d2   = d2*d2/dtmp;
            dudn = dud1*d1 + dud2*d2;
            dvdn = dvd1*d1 + dvd2*d2;
            dwdn = dwd1*d1 + dwd2*d2;
            dtdn = dtd1*d1 + dtd2*d2;
        }
        
        // dqdt, does not matter too much
        dudt1 = dudx*t1x + dudy*t1y + dudz*t1z;
        dvdt1 = dvdx*t1x + dvdy*t1y + dvdz*t1z;
        dwdt1 = dwdx*t1x + dwdy*t1y + dwdz*t1z;
        dudt2 = dudx*t2x + dudy*t2y + dudz*t2z;
        dvdt2 = dvdx*t2x + dvdy*t2y + dvdz*t2z;
        dwdt2 = dwdx*t2x + dwdy*t2y + dwdz*t2z;
        
        // now true gradients
        dudx  = dudn*areax + dudt1*t1x + dudt2*t2x;
        dudy  = dudn*areay + dudt1*t1y + dudt2*t2y;
        dudz  = dudn*areaz + dudt1*t1z + dudt2*t2z;
        dvdx  = dvdn*areax + dvdt1*t1x + dvdt2*t2x;
        dvdy  = dvdn*areay + dvdt1*t1y + dvdt2*t2y;
        dvdz  = dvdn*areaz + dvdt1*t1z + dvdt2*t2z;
        dwdx  = dwdn*areax + dwdt1*t1x + dwdt2*t2x;
        dwdy  = dwdn*areay + dwdt1*t1y + dwdt2*t2y;
        dwdz  = dwdn*areaz + dwdt1*t1z + dwdt2*t2z;
        if(level==0 && BadFaceAngle>0.0 && facecentroidskewness[face]<BadFaceAngle){
            dudx  = dudn*areax;
            dudy  = dudn*areay;
            dudz  = dudn*areaz;
            dvdx  = dvdn*areax;
            dvdy  = dvdn*areay;
            dvdz  = dvdn*areaz;
            dwdx  = dwdn*areax;
            dwdy  = dwdn*areay;
            dwdz  = dwdn*areaz;
        }
         
        if(face<nBFace){
            type = bcr[face]->GetType();
            if(type!=WALL && type!=SYMM && type!=FAR_FIELD && type!=INTERFACE){  
                delta = sqrt((xcc[c1]-xcc[c2])*(xcc[c1]-xcc[c2]) +
                             (ycc[c1]-ycc[c2])*(ycc[c1]-ycc[c2]) +
                             (zcc[c1]-zcc[c2])*(zcc[c1]-zcc[c2]));
                
                dvdn  = (vel[0][c2]-vel[0][c1])/delta;
                dudx  = dvdn*areax;
                dudy  = dvdn*areay;
                dudz  = dvdn*areaz;
                
                dvdn  = (vel[1][c2]-vel[1][c1])/delta;
                dvdx  = dvdn*areax;
                dvdy  = dvdn*areay;
                dvdz  = dvdn*areaz;
                
                dvdn  = (vel[2][c2]-vel[2][c1])/delta;
                dwdx  = dvdn*areax;
                dwdy  = dvdn*areay;
                dwdz  = dvdn*areaz;
                
                dtdn  = (t[c2]-t[c1])/delta;
            }
            
            //for aerodynamic heating!
            if(type == WALL){
                tw = -1.0;
                bcr[face]->GetBCVar(&tw, REAL_FLOW, "tw",0);
                if(tw>0.0){
                    delta =(xfc[face]-xcc[c1])*areax+
                           (yfc[face]-ycc[c1])*areay+
                           (zfc[face]-zcc[c1])*areaz;
                    dtdn = (tw-t[c1])/delta;
                }
            }
        }  
        
        // Get velocity at the face
        d_vis    = visc_f[i];
        heat_con = heat_f[i];
        
        txx = (2.*dudx - dvdy - dwdz)*two3;
        tyy = (2.*dvdy - dudx - dwdz)*two3;
        tzz = (2.*dwdz - dudx - dvdy)*two3;
        txy = dudy + dvdx;
        txz = dudz + dwdx;
        tyz = dwdy + dvdz;
        
        flux[0][i] =  0.;
        flux[1][i] = -d_vis*(txx*areax + txy*areay + txz*areaz)*area[face];
        flux[2][i] = -d_vis*(txy*areax + tyy*areay + tyz*areaz)*area[face];
        flux[3][i] = -d_vis*(txz*areax + tyz*areay + tzz*areaz)*area[face];
        flux[4][i] =  umid*flux[1][i] + vmid*flux[2][i] + wmid*flux[3][i]
            -  dtdn*heat_con*area[face];
    }
#else 
//not containing SIMD      
#ifdef FS_OPENMP//only OpenMP
#pragma omp parallel for private(count,i,c1,c2,type,delta,areax,areay,areaz,t1x,t1y,t1z,dtmp,\
      t2x,t2y,t2z,d1, d2, angle1, angle2, x1, x2, y1, y2, z1, z2,\
      u1, u2, v1, v2, w1, w2, t1, t2, umid, vmid, wmid, tmid, d_vis, heat_con, tw,\
      dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz,dudn, dvdn, dwdn, dtdn,\
      dud1, dud2, dvd1, dvd2, dwd1, dwd2, dtd1, dtd2,\
      dudt1, dvdt1, dwdt1, dudt2, dvdt2, dwdt2,\
      txx, tyy, tzz, txy, txz, tyz) schedule(static)
#endif    
    for(face=ns; face<ne; face++) {
        i     = face - ns;
        count = 2*ns+2*i;
        c1    = f2c[count];
        c2    = f2c[count+1];
        
        areax = xfn[face];
        areay = yfn[face];
        areaz = zfn[face];
        
        // Get first tangential vector on the face
        if(areax != 0.) {
            t1x =  areay;
            t1y = -areax;
            t1z =  0.;
        } else if(areay != 0.) {
            t1x = -areay;
            t1y =  areax;
            t1z =  0.;
        } else if(areaz != 0.) {
            t1x =  0.;
            t1y = -areaz;
            t1z =  areay;
        } else {
            if(warn) printf("Warninng: %ldth Face is singular\n", (long)face);
            flux[0][i] = 0.0;
            flux[1][i] = 0.0;
            flux[2][i] = 0.0;
            flux[3][i] = 0.0;
            flux[4][i] = 0.0;
            if(warn !=0 && ne == nTFace) warn = 0;
            continue;
        }
        
        // normalize the tangential vector
        dtmp = sqrt(t1x*t1x + t1y*t1y + t1z*t1z);
        t1x /= dtmp;
        t1y /= dtmp;
        t1z /= dtmp;
        
        // Get second tangential vector by cross dot t1 to normal
        t2x = areay*t1z - areaz*t1y;
        t2y = areaz*t1x - areax*t1z;
        t2z = areax*t1y - areay*t1x;
        
        // positions
        x1 = xcc[c1]  - xfc[face];
        y1 = ycc[c1]  - yfc[face];
        z1 = zcc[c1]  - zfc[face];
        x2 = xcc[c2]  - xfc[face];
        y2 = ycc[c2]  - yfc[face];
        z2 = zcc[c2]  - zfc[face];
        d1 = x1*areax + y1*areay + z1*areaz;
        d2 = x2*areax + y2*areay + z2*areaz;
        
        dtmp = -d1/(sqrt(x1*x1+ y1*y1 + z1*z1) + TINY);
        if(dtmp >  1.0) dtmp =  1.0;
        if(dtmp < -1.0) dtmp = -1.0;
        angle1 = asin(dtmp)*180.0/PI;
        
        dtmp = d2/(sqrt(x2*x2+ y2*y2 + z2*z2) + TINY);
        if(dtmp >  1.0) dtmp =  1.0;
        if(dtmp < -1.0) dtmp = -1.0;
        angle2 = asin(dtmp)*180.0/PI;
        
        // quentities at points 1 and 2
        u1   = vel[0][c1];
        v1   = vel[1][c1];
        w1   = vel[2][c1];
        t1   = t[c1];
        u2   = vel[0][c2];
        v2   = vel[1][c2];
        w2   = vel[2][c2];
        t2   = t[c2];
        umid = 0.5*(u1 + u2);
        vmid = 0.5*(v1 + v2);
        wmid = 0.5*(w1 + w2);
        tmid = 0.5*(t1 + t2);
        
        // Theroretically, more accurate to include the following terms
        if(angle1 > 10.0 && angle2 > 10.0) {
            u1 += dqdx[0][c1]*(d1*areax - x1) + dqdy[0][c1]*(d1*areay - y1) + dqdz[0][c1]*(d1*areaz - z1);
            v1 += dqdx[1][c1]*(d1*areax - x1) + dqdy[1][c1]*(d1*areay - y1) + dqdz[1][c1]*(d1*areaz - z1);
            w1 += dqdx[2][c1]*(d1*areax - x1) + dqdy[2][c1]*(d1*areay - y1) + dqdz[2][c1]*(d1*areaz - z1);
            
            u2 += dqdx[0][c2]*(d2*areax - x2) + dqdy[0][c2]*(d2*areay - y2) + dqdz[0][c2]*(d2*areaz - z2);
            v2 += dqdx[1][c2]*(d2*areax - x2) + dqdy[1][c2]*(d2*areay - y2) + dqdz[1][c2]*(d2*areaz - z2);
            w2 += dqdx[2][c2]*(d2*areax - x2) + dqdy[2][c2]*(d2*areay - y2) + dqdz[2][c2]*(d2*areaz - z2);
            
            t1 += dtdx[c1]*(d1*areax - x1)    + dtdy[c1]*(d1*areay - y1)    + dtdz[c1]*(d1*areaz - z1);
            t2 += dtdx[c2]*(d2*areax - x2)    + dtdy[c2]*(d2*areay - y2)    + dtdz[c2]*(d2*areaz - z2);
            if(t1 < TINY) t1  = t[c1];
            if(t2 < TINY) t2  = t[c2];
            
            // quantities at the face
            umid = vel_f[0][i];
            vmid = vel_f[1][i];
            wmid = vel_f[2][i];
            tmid = t_f[i];
        }
        
        dudx  = dqdx[0][c1]*deltl[i] + dqdx[0][c2]*deltr[i];
        dudy  = dqdy[0][c1]*deltl[i] + dqdy[0][c2]*deltr[i];
        dudz  = dqdz[0][c1]*deltl[i] + dqdz[0][c2]*deltr[i];
        dvdx  = dqdx[1][c1]*deltl[i] + dqdx[1][c2]*deltr[i];
        dvdy  = dqdy[1][c1]*deltl[i] + dqdy[1][c2]*deltr[i];
        dvdz  = dqdz[1][c1]*deltl[i] + dqdz[1][c2]*deltr[i];
        dwdx  = dqdx[2][c1]*deltl[i] + dqdx[2][c2]*deltr[i];
        dwdy  = dqdy[2][c1]*deltl[i] + dqdy[2][c2]*deltr[i];
        dwdz  = dqdz[2][c1]*deltl[i] + dqdz[2][c2]*deltr[i];
        
        dudn  = 0.0;
        dvdn  = 0.0;
        dwdn  = 0.0;
        dtdn  = 0.0;
        
        if(angle1 > 0.0 && angle2 > 0.0 && fabs(d1) > TINY && fabs(d2) > TINY) {
            dud1 = (u1 - umid)/d1;
            dvd1 = (v1 - vmid)/d1;
            dwd1 = (w1 - wmid)/d1;
            dtd1 = (t1 - tmid)/d1;
            dud2 = (u2 - umid)/d2;
            dvd2 = (v2 - vmid)/d2;
            dwd2 = (w2 - wmid)/d2;
            dtd2 = (t2 - tmid)/d2;
            dtmp = d1*d1 + d2*d2;
            d1   = d1*d1/dtmp;
            d2   = d2*d2/dtmp;
            dudn = dud1*d1 + dud2*d2;
            dvdn = dvd1*d1 + dvd2*d2;
            dwdn = dwd1*d1 + dwd2*d2;
            dtdn = dtd1*d1 + dtd2*d2;
        }
        
        // dqdt, does not matter too much
        dudt1 = dudx*t1x + dudy*t1y + dudz*t1z;
        dvdt1 = dvdx*t1x + dvdy*t1y + dvdz*t1z;
        dwdt1 = dwdx*t1x + dwdy*t1y + dwdz*t1z;
        dudt2 = dudx*t2x + dudy*t2y + dudz*t2z;
        dvdt2 = dvdx*t2x + dvdy*t2y + dvdz*t2z;
        dwdt2 = dwdx*t2x + dwdy*t2y + dwdz*t2z;
        
        // now true gradients
        dudx  = dudn*areax + dudt1*t1x + dudt2*t2x;
        dudy  = dudn*areay + dudt1*t1y + dudt2*t2y;
        dudz  = dudn*areaz + dudt1*t1z + dudt2*t2z;
        dvdx  = dvdn*areax + dvdt1*t1x + dvdt2*t2x;
        dvdy  = dvdn*areay + dvdt1*t1y + dvdt2*t2y;
        dvdz  = dvdn*areaz + dvdt1*t1z + dvdt2*t2z;
        dwdx  = dwdn*areax + dwdt1*t1x + dwdt2*t2x;
        dwdy  = dwdn*areay + dwdt1*t1y + dwdt2*t2y;
        dwdz  = dwdn*areaz + dwdt1*t1z + dwdt2*t2z;
        if(level==0 && BadFaceAngle>0.0 && facecentroidskewness[face]<BadFaceAngle){
            dudx  = dudn*areax;
            dudy  = dudn*areay;
            dudz  = dudn*areaz;
            dvdx  = dvdn*areax;
            dvdy  = dvdn*areay;
            dvdz  = dvdn*areaz;
            dwdx  = dwdn*areax;
            dwdy  = dwdn*areay;
            dwdz  = dwdn*areaz;
        }
         
        if(face<nBFace){
            type = bcr[face]->GetType();
            if(type!=WALL && type!=SYMM && type!=FAR_FIELD && type!=INTERFACE){  
                delta = sqrt((xcc[c1]-xcc[c2])*(xcc[c1]-xcc[c2]) +
                             (ycc[c1]-ycc[c2])*(ycc[c1]-ycc[c2]) +
                             (zcc[c1]-zcc[c2])*(zcc[c1]-zcc[c2]));
                
                dvdn  = (vel[0][c2]-vel[0][c1])/delta;
                dudx  = dvdn*areax;
                dudy  = dvdn*areay;
                dudz  = dvdn*areaz;
                
                dvdn  = (vel[1][c2]-vel[1][c1])/delta;
                dvdx  = dvdn*areax;
                dvdy  = dvdn*areay;
                dvdz  = dvdn*areaz;
                
                dvdn  = (vel[2][c2]-vel[2][c1])/delta;
                dwdx  = dvdn*areax;
                dwdy  = dvdn*areay;
                dwdz  = dvdn*areaz;
                
                dtdn  = (t[c2]-t[c1])/delta;
            }
            
            //for aerodynamic heating!
            if(type == WALL){
                tw = -1.0;
                bcr[face]->GetBCVar(&tw, REAL_FLOW, "tw",0);
                if(tw>0.0){
                    delta =(xfc[face]-xcc[c1])*areax+
                           (yfc[face]-ycc[c1])*areay+
                           (zfc[face]-zcc[c1])*areaz;
                    dtdn = (tw-t[c1])/delta;
                }
            }
        }  
        
        // Get velocity at the face
        d_vis    = visc_f[i];
        heat_con = heat_f[i];
        
        txx = (2.*dudx - dvdy - dwdz)*two3;
        tyy = (2.*dvdy - dudx - dwdz)*two3;
        tzz = (2.*dwdz - dudx - dvdy)*two3;
        txy = dudy + dvdx;
        txz = dudz + dwdx;
        tyz = dwdy + dvdz;
        
        flux[0][i] =  0.;
        flux[1][i] = -d_vis*(txx*areax + txy*areay + txz*areaz)*area[face];
        flux[2][i] = -d_vis*(txy*areax + tyy*areay + tyz*areaz)*area[face];
        flux[3][i] = -d_vis*(txz*areax + tyz*areay + tzz*areaz)*area[face];
        flux[4][i] =  umid*flux[1][i] + vmid*flux[2][i] + wmid*flux[3][i]
            -  dtdn*heat_con*area[face];
    }
#endif
    // Turn off warning
    if(warn !=0 && ne == nTFace) warn = 0;
}


/*******************************************************************************\
       Calculate and print out L2 norms of  the residuals 
\*******************************************************************************/
void DumpNormResi(PolyGrid *grid, IntType iter, IntType zn, RealFlow t_now)
{
    FILE    *fp;
    ShortString    resid, deltaQ;
    String char_tmp;
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType n      = nTCell + nBFace;
    IntType i,j;
    static IntType here=0;
    
    // Open the file for norms of  the residuals
#ifdef MPICH
    if(myZone == 1) {
        sprintf(resid, "resid_glb.out");
        if(iter == 1) fp = fopen(resid, "w");
        else fp = fopen(resid, "a");
        if(!fp) {
            printf("Can't open file %s \n", resid);
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
#else
    sprintf(resid, "resid_glb.out");
    if(iter == 1) fp = fopen(resid, "w");
    else fp = fopen(resid, "a");
    
    if(!fp) {
        printf("Can't open file %s \n", resid);
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
#endif
    
    // Get the residuals
    RealFlow   *res[5];
    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");
    for(i=1; i<5; i++) res[i] = &res[i-1][nTCell];
    
    // Get the maximum and minimum of time steps
    RealFlow dt_max, dt_min;
    grid->GetData(&dt_max, REAL_FLOW, 1, "dt_max");
    grid->GetData(&dt_min, REAL_FLOW, 1, "dt_min");
    IntType order;
    grid->GetData(&order, INT, 1, "order");
    RealFlow limitaver = 0.0;
    if(order>1) grid->GetData(&limitaver, REAL_FLOW, 1, "limitaver"); 
    
    RealFlow norm[5] = {0.,0.,0.,0.,0.};
    
    for(j=0; j<5; j++){
        for(i=0; i<nTCell; i++) {
            norm[j] += res[j][i]*res[j][i];
        }
    }
#ifdef MPICH
    RealFlow total[5];
    MPI_Allreduce(norm, total, 5, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    for(j=0; j<5; j++) norm[j] = total[j]; 
#endif
    
    for(j=0; j<5; j++) norm[j] = sqrt(norm[j]);
    
    grid->UpdateData(&(norm[0]), REAL_FLOW, 1, "res_rho");   //用于判断是否中断子迭代的循环
    
#ifdef MPICH
    if(myZone == 1) {
        if(!here)          
            printf("#iter   rho_res      mx_res      my_res      mz_res     et_res     dt_max    dt_min    ns_cpu     aver_limit\n");    
        sprintf(char_tmp,"%5d %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e\n",
               (int)iter, norm[0], norm[1], norm[2], norm[3], norm[4], dt_max, dt_min, t_now, limitaver); 
        printf(char_tmp);
        
        if(iter == 1) 
            fprintf(fp, "#iter   rho_res      mx_res      my_res      mz_res     et_res     dt_max    dt_min    ns_cpu     aver_limit\n");
        fprintf(fp,char_tmp);
        fclose(fp);
    }
#else
    if(!here)          
        printf("#iter   rho_res      mx_res      my_res      mz_res     et_res     dt_max    dt_min    ns_cpu     aver_limit\n");    
    sprintf(char_tmp,"%5d %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e\n",
            (int)iter, norm[0], norm[1], norm[2], norm[3], norm[4], dt_max, dt_min, t_now, limitaver); 
    printf(char_tmp);
    
    if(iter == 1) 
        fprintf(fp, "#iter   rho_res      mx_res      my_res      mz_res     et_res     dt_max    dt_min    ns_cpu     aver_limit\n");
    fprintf(fp,char_tmp);
    fclose(fp);
#endif

    IfNAN(char_tmp);  //如果NAN，退出程序
    TellDivergence(grid,iter,norm[0]);  //判断是否发散
    
/* #ifdef MPICH
    //#####################
    //To output the relative conservation varialbe
    if(myZone==1){
        sprintf(deltaQ, "deltaQ_glb.out");
        if(iter == 1) fp = fopen(deltaQ, "w");
        else fp = fopen(deltaQ, "a");
        if(!fp) {
            printf("Can't open file %s \n", resid);
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
#else
    //#####################
    //To output the relative conservation varialbe
    sprintf(deltaQ, "deltaQ_glb.out");
    if(iter == 1) fp = fopen(deltaQ, "w");
    else fp = fopen(deltaQ, "a");
    if(!fp) {
        printf("Can't open file %s \n", resid);
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
#endif
    
    for(i=0; i<5; i++) norm[i] = 0.;
    
 
    RealFlow *DQ[5];
    DQ[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*n, "DQ");
    assert(DQ[0] != 0);
    for(i=1; i<5; i++) DQ[i] = &DQ[i-1][n];
    for(j=0; j<5; j++){
        for(i=0; i<nTCell; i++) {
            norm[j] += DQ[j][i]*DQ[j][i];
        }
    }
    
#ifdef MPICH
    for(i=0; i<5; i++) total[i] = 0.;
    MPI_Allreduce(norm, total, 5, MPIReal, MPI_SUM, MPI_COMM_WORLD);
    for(j=0; j<5; j++) norm[j] = total[j];  
#endif           
    
#ifdef MPICH
    if(myZone==1){
        for(j=0; j<5; j++) norm[j] = sqrt(norm[j]); 
        if(iter == 1) 
            fprintf(fp, "#iter   rho_res      rhou_res      rhov_res      rhow_res     et_res\n");  
        sprintf(char_tmp, "%5d  %.5e %.5e %.5e %.5e %.5e\n ", (int)iter, norm[0], norm[1], norm[2], norm[3], norm[4]);
        fprintf(fp,char_tmp);
        fclose(fp);
    }
#else
    for(j=0; j<5; j++) norm[j] = sqrt(norm[j]); 
    if(iter == 1) 
        fprintf(fp, "#iter   rho_res      rhou_res      rhov_res      rhow_res     et_res\n");  
    sprintf(char_tmp, "%5d  %.5e %.5e %.5e %.5e %.5e\n ", (int)iter, norm[0], norm[1], norm[2], norm[3], norm[4]);
    fprintf(fp,char_tmp);
    fclose(fp);
#endif
	//cout << "solver_ns: 5331. " << endl;
    IfNAN(char_tmp);  //如果NAN，退出程序
    //cout << "solver_ns: 5333. " << endl; */
    here = 1;

}
/*******************************************************************************\
       Calculate and dump out pressure force
\*******************************************************************************/
void NSSolver::DumpPressureForce(PolyGrid *grid, IntType iter, IntType zn, 
                                 RealFlow &pfx, RealFlow &pfy, RealFlow &pfz, RealFlow &total,
                                 RealFlow &pmx, RealFlow &pmy, RealFlow &pmz)
{
    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  n      = nTCell + nBFace;
    IntType  *f2c   = grid->Getf2c();
    BCRecord **bcr  = grid->Getbcr();
    RealGeom *area  = grid->GetFaceArea();
    RealGeom *xfn   = grid->GetXfn();
    RealGeom *yfn   = grid->GetYfn();
    RealGeom *zfn   = grid->GetZfn();
    RealGeom *xfc   = grid->GetXfc();
    RealGeom *yfc   = grid->GetYfc();
    RealGeom *zfc   = grid->GetZfc();
    
    
    
   
    
    
    
    RealFlow u00, v00, w00, rho00;
    grid->GetData(&u00,   REAL_FLOW, 1, "u");
    grid->GetData(&v00,   REAL_FLOW, 1, "v");
    grid->GetData(&w00,   REAL_FLOW, 1, "w");
    grid->GetData(&rho00, REAL_FLOW, 1, "rho");
    
    RealGeom xg=0.0, yg=0.0, zg=0.0; //zhyb: reference coordinates
    RealGeom length_ref=1.0, span_ref=1.0, area_ref=1.0; //zhyb: reference length, reference span, area
    RealGeom dx,dy,dz;
    grid->GetData(&xg, REAL_GEOM, 1, "xg", 0);
    grid->GetData(&yg, REAL_GEOM, 1, "yg", 0);
    grid->GetData(&zg, REAL_GEOM, 1, "zg", 0);
    grid->GetData(&length_ref, REAL_GEOM, 1, "length_ref", 0);
    span_ref = length_ref;
    grid->GetData(&span_ref, REAL_GEOM, 1, "span_ref", 0);
    grid->GetData(&area_ref, REAL_GEOM, 1, "area_ref", 0);
    
    RealFlow *p = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    IntType  i, c1, type;
    RealFlow tmpfx, tmpfy, tmpfz;
  
    pfx = 0.0;
    pfy = 0.0;
    pfz = 0.0;
    total = 0.0;
    pmx = 0.0;
    pmy = 0.0;
    pmz = 0.0;
    tmpfx = 0.0;
    tmpfy = 0.0;
    tmpfz = 0.0;
      
    IntType patch;
    bool comp=true;
    IntType n_patchout=0;
    IntType *patchout=0;
    if(n_patchout){
        patchout = NULL;
        mfmem::snew_array_1D(patchout, n_patchout,dmrfl);
        grid->GetData(patchout, INT, n_patchout, "patchpart");
    }

    RealFlow pfc;
    for(i=0; i<nBFace; i++)
    {
        type = bcr[i]->GetType();
        if(type != WALL) continue;
        patch = bcr[i]->GetPatchID();
        if(n_patchout) {
            comp = false;
            for(IntType j=0; j<n_patchout; j++)
                if( patch==patchout[j]) comp = true;
            if(!comp) continue;
        }

        c1 = f2c[i+i];
        
        pfc = p[c1];
        
        tmpfx = pfc*xfn[i]*area[i];
        tmpfy = pfc*yfn[i]*area[i];
        tmpfz = pfc*zfn[i]*area[i];
           
        dx = xfc[i]-xg;
        dy = yfc[i]-yg;
        dz = zfc[i]-zg;
        
        pfx += tmpfx;
        pfy += tmpfy;
        pfz += tmpfz;
        
        pmx += tmpfz*dy-tmpfy*dz;
        pmy += tmpfx*dz-tmpfz*dx;
        pmz += tmpfy*dx-tmpfx*dy;         
    }
   
    RealFlow coef=rho00*(u00*u00+v00*v00+w00*w00)*area_ref/2.0;
    pfx/=coef;
    pfy/=coef;
    pfz/=coef;
    pmx=pmx/coef/span_ref;
    pmy=pmy/coef/length_ref;
    pmz=pmz/coef/span_ref;

#ifdef MPICH
    RealFlow force_sum[6] = {pfx, pfy, pfz, pmx, pmy, pmz};
    Parallel::parallel_sum(force_sum, 6, GridComm);
    pfx = force_sum[0];  pfy = force_sum[1];  pfz = force_sum[2];
    pmx = force_sum[3];  pmy = force_sum[4];  pmz = force_sum[5];          
#endif
    total = sqrt(pfx*pfx + pfy*pfy + pfz*pfz);

    mfmem::sdel_array_1D(patchout);
    return;
}


/*******************************************************************************\
       Calculate and dump out viscous force
\*******************************************************************************/
void NSSolver::DumpViscousForce(PolyGrid *grid, IntType iter, IntType zn, 
                                RealFlow &vfx, RealFlow &vfy, RealFlow &vfz, RealFlow &total,
                                RealFlow &vmx, RealFlow &vmy, RealFlow &vmz)
{
    IntType  nTCell = grid->GetNTCell();
    IntType  nBFace = grid->GetNBFace();
    IntType  n      = nTCell + nBFace;
    IntType  *f2c   = grid->Getf2c();
    BCRecord **bcr  = grid->Getbcr();
    RealGeom *area  = grid->GetFaceArea();
    RealGeom *xfn   = grid->GetXfn();
    RealGeom *yfn   = grid->GetYfn();
    RealGeom *zfn   = grid->GetZfn();
    RealGeom *xcc   = grid->GetXcc();
    RealGeom *ycc   = grid->GetYcc();
    RealGeom *zcc   = grid->GetZcc();
    RealGeom *xfc   = grid->GetXfc();
    RealGeom *yfc   = grid->GetYfc();
    RealGeom *zfc   = grid->GetZfc();
    
    RealFlow u00, v00, w00, rho00;
    grid->GetData(&u00,   REAL_FLOW, 1, "u");
    grid->GetData(&v00,   REAL_FLOW, 1, "v");
    grid->GetData(&w00,   REAL_FLOW, 1, "w");
    grid->GetData(&rho00, REAL_FLOW, 1, "rho");
    
    IntType steady;
    grid->GetData(&steady, INT, 1, "steady");
    RealGeom *BFacevgx = grid->GetBoundaryFaceVelocityX();
    RealGeom *BFacevgy = grid->GetBoundaryFaceVelocityY();
    RealGeom *BFacevgz = grid->GetBoundaryFaceVelocityZ();
    
    RealGeom xg=0.0, yg=0.0, zg=0.0; //zhyb: reference coordinates
    RealGeom length_ref=1.0, span_ref=1.0, area_ref=1.0; //zhyb: reference length, reference span, area
    RealGeom dx,dy,dz;
    grid->GetData(&xg, REAL_GEOM, 1, "xg", 0);
    grid->GetData(&yg, REAL_GEOM, 1, "yg", 0);
    grid->GetData(&zg, REAL_GEOM, 1, "zg", 0);
    grid->GetData(&length_ref, REAL_GEOM, 1, "length_ref", 0);
    span_ref = length_ref;
    grid->GetData(&span_ref, REAL_GEOM, 1, "span_ref", 0);
    grid->GetData(&area_ref, REAL_GEOM, 1, "area_ref", 0);

    IntType  i, j, c1, c2, type;
    RealFlow tmpfx, tmpfy, tmpfz;
    RealFlow uu,vv,ww,tw;
    RealGeom dn;
    
    vfx = 0.0;
    vfy = 0.0;
    vfz = 0.0;
    total = 0.0;
    vmx = 0.0;
    vmy = 0.0;
    vmz = 0.0;
    
    IntType vis_mode=0;
    grid->GetData(&vis_mode, INT, 1, "vis_mode"); 
    RealFlow *vis_l,sum_mu,lamda;

    if(vis_mode == INVISCID) {
        return;
    }else{
        vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
    }
   
    RealFlow *vx  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    RealFlow *vy  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    RealFlow *vz  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");

   // compute the gradients
   RealFlow *dvdxout[3], *dvdyout[3], *dvdzout[3];
   GetVelocityGradient(grid, dvdxout, dvdyout, dvdzout);

   RealFlow *dvxdx = dvdxout[0];
   RealFlow *dvxdy = dvdyout[0];
   RealFlow *dvxdz = dvdzout[0];
   RealFlow *dvydx = dvdxout[1];
   RealFlow *dvydy = dvdyout[1];
   RealFlow *dvydz = dvdzout[1];
   RealFlow *dvzdx = dvdxout[2];
   RealFlow *dvzdy = dvdyout[2];
   RealFlow *dvzdz = dvdzout[2];  

    IntType patch;
    bool comp=true;
    IntType n_patchout=0;
    IntType *patchout=0;
    if(n_patchout){
        patchout = NULL;
        mfmem::snew_array_1D(patchout, n_patchout,dmrfl);
        grid->GetData(patchout, INT, n_patchout, "patchpart");
    }

   RealFlow tau_xx, tau_yy, tau_zz, tau_xy, tau_yz, tau_xz;
   for(i=0; i<nBFace; i++) {
        type = bcr[i]->GetType();
        if(type != WALL) continue;

        patch = bcr[i]->GetPatchID();
        if(n_patchout) {
            comp = false;
            for(j=0; j<n_patchout; j++)
                if( patch==patchout[j]) comp = true;
            if(!comp) continue;
        }

        c1  = f2c[i+i];
        c2  = f2c[i+i+1];
     
        tw = -1.0;
        bcr[i]->GetBCVar(&tw, REAL_FLOW, "tw",0);
        if(tw>0.0){
            sum_mu = Sutherland_classic(tw);
        }else{
            sum_mu = vis_l[c1];
        }
       
        lamda=-(2.0/3.0)*sum_mu;
          
        RealFlow dvdx[3],dvdy[3],dvdz[3],deltv[3];
        RealGeom d1,d2,d3,d4;
     
        uu = vx[c1];
        vv = vy[c1];
        ww = vz[c1];

        d1 = xfc[i]-xcc[c1];
        d2 = yfc[i]-ycc[c1];
        d3 = zfc[i]-zcc[c1];
        d4 = d1*xfn[i]+d2*yfn[i]+d3*zfn[i];
        d1 -= d4*xfn[i];
        d2 -= d4*yfn[i];
        d3 -= d4*zfn[i];
     
        uu += dvxdx[c1]*d1+dvxdy[c1]*d2+dvxdz[c1]*d3;
        vv += dvydx[c1]*d1+dvydy[c1]*d2+dvydz[c1]*d3;
        ww += dvzdx[c1]*d1+dvzdy[c1]*d2+dvzdz[c1]*d3;
     
        dn = (xfc[i]-xcc[c1])*xfn[i]+
            (yfc[i]-ycc[c1])*yfn[i]+
            (zfc[i]-zcc[c1])*zfn[i]+TINY;
        dn = 1.0/dn;
        deltv[0] = uu*dn;
        deltv[1] = vv*dn;
        deltv[2] = ww*dn;
        if(!steady){
            deltv[0] -= BFacevgx[i]*dn;
            deltv[1] -= BFacevgy[i]*dn;
            deltv[2] -= BFacevgz[i]*dn;
        }
     
        for(j=0; j<3; j++){
            dvdx[j] = -deltv[j]*xfn[i];
            dvdy[j] = -deltv[j]*yfn[i];
            dvdz[j] = -deltv[j]*zfn[i];
        } 
        tau_xx = lamda*(dvdy[1]+dvdz[2]-2*dvdx[0]);
        tau_yy = lamda*(dvdx[0]+dvdz[2]-2*dvdy[1]);
        tau_zz = lamda*(dvdx[0]+dvdy[1]-2*dvdz[2]);
        tau_xy = sum_mu*(dvdy[0]+dvdx[1]);
        tau_xz = sum_mu*(dvdz[0]+dvdx[2]);
        tau_yz = sum_mu*(dvdz[1]+dvdy[2]);
     
        tmpfx = -(tau_xx*xfn[i]+tau_xy*yfn[i]+tau_xz*zfn[i])*area[i];
        tmpfy = -(tau_xy*xfn[i]+tau_yy*yfn[i]+tau_yz*zfn[i])*area[i];
        tmpfz = -(tau_xz*xfn[i]+tau_yz*yfn[i]+tau_zz*zfn[i])*area[i];

        dx = xfc[i]-xg;
        dy = yfc[i]-yg;
        dz = zfc[i]-zg;
     
        vfx += tmpfx;
        vfy += tmpfy;
        vfz += tmpfz;
        vmx += tmpfz*dy-tmpfy*dz;
        vmy += tmpfx*dz-tmpfz*dx;
        vmz += tmpfy*dx-tmpfx*dy;   
    }

    RealFlow coef = rho00*(u00*u00+v00*v00+w00*w00)*area_ref/2.0;
    vfx /= coef;
    vfy /= coef;
    vfz /= coef;
    vmx = vmx/coef/span_ref;
    vmy = vmy/coef/length_ref;
    vmz = vmz/coef/span_ref;

#ifdef MPICH
    RealFlow force_sum[6] = {vfx, vfy, vfz, vmx, vmy, vmz};
    Parallel::parallel_sum(force_sum, 6, GridComm);
    vfx = force_sum[0];  vfy = force_sum[1];  vfz = force_sum[2];
    vmx = force_sum[3];  vmy = force_sum[4];  vmz = force_sum[5];          
#endif
    total = sqrt(vfx*vfx + vfy*vfy + vfz*vfz);

    mfmem::sdel_array_1D(patchout);
    return;
}

/*******************************************************************************\
       Calculate and dump out force
\*******************************************************************************/
void NSSolver::DumpForce(PolyGrid *grid, IntType iter, IntType zn) 
{    
    RealFlow pfx, pfy, pfz, totalp, pmx, pmy, pmz;
    RealFlow vfx, vfy, vfz, totalv, vmx, vmy, vmz;
    RealFlow fx, fy, fz, mx, my, mz, cl, cd, cdp, cdv; 
    
    RealGeom alpha,beita;
    grid->GetData(&alpha,  REAL_FLOW, 1, "alpha");
    grid->GetData(&beita,  REAL_FLOW, 1, "beita");
    alpha = alpha*PI/180.;
    beita = beita*PI/180.;
    
    IntType n_unst = 0;
    zone->simu->GetData(&n_unst, INT, 1, "n_unst");
    
    DumpPressureForce(grid, iter, zn, pfx, pfy, pfz, totalp, pmx, pmy, pmz);
    DumpViscousForce(grid, iter, zn, vfx, vfy, vfz, totalv, vmx, vmy, vmz);
    
    fx = pfx+vfx;
    fy = pfy+vfy;
    fz = pfz+vfz;
    mx = pmx+vmx;
    my = pmy+vmy;
    mz = pmz+vmz;

    cl = -fx*sin(alpha)+fz*cos(alpha);
    cd = fx*cos(alpha)*cos(beita)-fy*sin(beita)+fz*sin(alpha)*cos(beita);
    cdp= pfx*cos(alpha)*cos(beita)-pfy*sin(beita)+pfz*sin(alpha)*cos(beita);
    cdv= vfx*cos(alpha)*cos(beita)-vfy*sin(beita)+vfz*sin(alpha)*cos(beita);
    
    // Write the force to file
    bool out_flag = true;
    std::string file_name = "force_glb";
#ifdef MPICH         
    out_flag = myZone == 1 ? true : false;
#endif
    file_name += ".out";

    if (out_flag)
    {
         FILE *fp;
        if(iter == 1) fp = fopen(file_name.c_str(), "w");
        else fp = fopen(file_name.c_str(), "a");
        if(!fp) {
            std::cerr << "Can't open file " << file_name << std::endl;
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
        if(iter == 1) fprintf(fp,"#step       iter       CL           CD           CDp          CDv          Cx           Cy          Cz           Cmx           Cmy           Cmz\n");
        fprintf(fp,"%10d  %10d  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e\n",
            (int)n_unst, iter, cl, cd, cdp, cdv, fx, fy, fz, mx, my, mz);

        fclose(fp);
    }

    //储存所有的输出的力分量
    RealFlow forceout[10];
    forceout[0] = cl; forceout[1] = cd; forceout[2] = cdp;  forceout[3] = cdv; forceout[4] = fx; 
    forceout[5] = fy; forceout[6] = fz; forceout[7] = mx;   forceout[8] = my;  forceout[9] = mz;
    UpdateData(forceout, REAL_FLOW, 10, "ForceOut");

    return;
}
/******************************************************************************\
         Print out restart file 
\******************************************************************************/
void NSSolver::DumpRestart(PolyGrid *grid, IntType iter, IntType zn, RealFlow t_now)
{
    FILE    *fp;
    IntType nTCell = grid->GetNTCell();
    IntType n      = nTCell + grid->GetNBFace();
    IntType vis_mode, nvar, steady, n_unst;
    grid->GetData(&steady, INT, 1, "steady");
    zone->simu->GetData(&n_unst, INT, 1, "n_unst");

    IntType file_id = 0;
#ifdef MPICH
    file_id = myZone;
#else
    file_id = zn + 1;
#endif

    // create folder
    std::string folder = FieldIO::RESTART_FOLDER;
    CreatFolder_OneProcessor(folder);

    std::string rest_file = FieldIO::restart_file_with_id(file_id);

    fp = fopen(rest_file.c_str(),"wb");
    if(!fp) {
        std::cerr << "Failed to open file " << rest_file << " in function DumpRestart" << std::endl;
        return;
    }
    
    RealFlow *rho  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    RealFlow *u    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    RealFlow *v    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    RealFlow *w    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    RealFlow *p    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    
    RealFlow *rho_cur,*u_cur,*v_cur,*w_cur,*p_cur;
    RealFlow *rho_old,*u_old,*v_old,*w_old,*p_old;
    if(!steady){
        rho_cur  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "rho_cur");
        u_cur    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "u_cur");
        v_cur    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "v_cur");
        w_cur    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "w_cur");
        p_cur    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "p_cur");

        rho_old  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "rho_old");
        u_old    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "u_old");
        v_old    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "v_old");
        w_old    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "w_old");
        p_old    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "p_old");
    }
       
    grid->GetData(&vis_mode, INT, 1, "vis_mode");
    if(vis_mode == INVISCID)
      nvar = 5;
    else if(vis_mode == LAMINAR)
      nvar = 6;
    else if(vis_mode == S_A_MODEL)
      nvar = 8; 
    
    fwrite(&nTCell, sizeof(IntType), 1, fp);
    fwrite(&n_unst, sizeof(IntType), 1, fp);
    fwrite(&iter,   sizeof(IntType), 1, fp);
    fwrite(&nvar,   sizeof(IntType), 1, fp);
    fwrite(&t_now,  sizeof(RealFlow),1, fp);

    //add by dingxin
#ifdef REORDER
    RealFlow* rho_tmp = NULL, * u_tmp = NULL, * v_tmp = NULL, * w_tmp = NULL, * p_tmp = NULL;
    IntType* neworder = grid->GetNewOrder();
    mfmem::snew_array_1D(rho_tmp, nTCell, dmrfl);
    mfmem::snew_array_1D(u_tmp, nTCell, dmrfl);
    mfmem::snew_array_1D(v_tmp, nTCell, dmrfl);
    mfmem::snew_array_1D(w_tmp, nTCell, dmrfl);
    mfmem::snew_array_1D(p_tmp, nTCell, dmrfl);
    for (IntType i = 0; i < nTCell; i++) {
        rho_tmp[i] = rho[neworder[i]];
        u_tmp[i] = u[neworder[i]];
        v_tmp[i] = v[neworder[i]];
        w_tmp[i] = w[neworder[i]];
        p_tmp[i] = p[neworder[i]];
    }
    fwrite(rho_tmp, sizeof(RealFlow), nTCell, fp);
    fwrite(u_tmp, sizeof(RealFlow), nTCell, fp);
    fwrite(v_tmp, sizeof(RealFlow), nTCell, fp);
    fwrite(w_tmp, sizeof(RealFlow), nTCell, fp);
    fwrite(p_tmp, sizeof(RealFlow), nTCell, fp);
    if (!steady) {
        for (IntType i = 0; i < nTCell; i++) {
            rho_tmp[i] = rho_cur[neworder[i]];
            u_tmp[i] = u_cur[neworder[i]];
            v_tmp[i] = v_cur[neworder[i]];
            w_tmp[i] = w_cur[neworder[i]];
            p_tmp[i] = p_cur[neworder[i]];
        }
        fwrite(rho_tmp, sizeof(RealFlow), nTCell, fp);
        fwrite(u_tmp, sizeof(RealFlow), nTCell, fp);
        fwrite(v_tmp, sizeof(RealFlow), nTCell, fp);
        fwrite(w_tmp, sizeof(RealFlow), nTCell, fp);
        fwrite(p_tmp, sizeof(RealFlow), nTCell, fp);

        for (IntType i = 0; i < nTCell; i++) {
            rho_tmp[i] = rho_old[neworder[i]];
            u_tmp[i] = u_old[neworder[i]];
            v_tmp[i] = v_old[neworder[i]];
            w_tmp[i] = w_old[neworder[i]];
            p_tmp[i] = p_old[neworder[i]];
        }
        fwrite(rho_tmp, sizeof(RealFlow), nTCell, fp);
        fwrite(u_tmp, sizeof(RealFlow), nTCell, fp);
        fwrite(v_tmp, sizeof(RealFlow), nTCell, fp);
        fwrite(w_tmp, sizeof(RealFlow), nTCell, fp);
        fwrite(p_tmp, sizeof(RealFlow), nTCell, fp);
    }
    if (vis_mode == LAMINAR) {
        RealFlow* vis_l = (RealFlow*)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
        RealFlow* vis_l_tmp = NULL;
        mfmem::snew_array_1D(vis_l_tmp, nTCell, dmrfl);
        for (IntType i = 0; i < nTCell; i++) {
            vis_l_tmp[i] = vis_l[neworder[i]];
        }
        fwrite(vis_l_tmp, sizeof(RealFlow), nTCell, fp);
        mfmem::sdel_array_1D(vis_l_tmp);
    }
    neworder = NULL;
    mfmem::sdel_array_1D(rho_tmp);
    mfmem::sdel_array_1D(u_tmp);
    mfmem::sdel_array_1D(v_tmp);
    mfmem::sdel_array_1D(w_tmp);
    mfmem::sdel_array_1D(p_tmp);

#else
    fwrite(rho, sizeof(RealFlow), nTCell, fp);
    fwrite(u, sizeof(RealFlow), nTCell, fp);
    fwrite(v, sizeof(RealFlow), nTCell, fp);
    fwrite(w, sizeof(RealFlow), nTCell, fp);
    fwrite(p, sizeof(RealFlow), nTCell, fp);

    if (!steady) {
        fwrite(rho_cur, sizeof(RealFlow), nTCell, fp);
        fwrite(u_cur, sizeof(RealFlow), nTCell, fp);
        fwrite(v_cur, sizeof(RealFlow), nTCell, fp);
        fwrite(w_cur, sizeof(RealFlow), nTCell, fp);
        fwrite(p_cur, sizeof(RealFlow), nTCell, fp);

        fwrite(rho_old, sizeof(RealFlow), nTCell, fp);
        fwrite(u_old, sizeof(RealFlow), nTCell, fp);
        fwrite(v_old, sizeof(RealFlow), nTCell, fp);
        fwrite(w_old, sizeof(RealFlow), nTCell, fp);
        fwrite(p_old, sizeof(RealFlow), nTCell, fp);
    }

    if (vis_mode == LAMINAR) {
        RealFlow* vis_l = (RealFlow*)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
        fwrite(vis_l, sizeof(RealFlow), nTCell, fp);
    }
#endif
    
    fclose(fp);
}


/************************************************************************
  read step information from RESTART file
  tangj, 2019-11-05
************************************************************************/
void NSSolver::ReadStepInfoFromFile(PolyGrid *grid)
{
    FILE       *fp; 
    IntType    ntmp, iter, nvar;
    RealFlow   start_time;
    IntType    zn = grid->GetZone();
    IntType    steady, Unst_steps_Curt;
    grid->GetData(&steady, INT, 1, "steady");

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
    fread(&Unst_steps_Curt, sizeof(IntType), 1, fp);
    fread(&iter, sizeof(IntType), 1, fp);
    fread(&nvar, sizeof(IntType), 1, fp);
    fread(&start_time, sizeof(RealFlow), 1, fp);
    
    zone->simu->UpdateData(&Unst_steps_Curt, INT, 1, "Unst_steps_Curt");
    zone->simu->UpdateData(&iter, INT, 1, "iter_done");

    fclose(fp);

    UpdateData(&iter, INT, 1, "iter_done");
    UpdateData(&start_time, REAL_FLOW, 1, "start_time");
}


/************************************************************************
  allocate memory  for flow field
  tangj, 2019-11-05
************************************************************************/
void NSSolver::AllocateFlowfieldMemory(PolyGrid *grid)
{
    RealFlow *rho, *u, *v, *w, *p;
    IntType nTCell = grid->GetNTCell();
    IntType n = nTCell + grid->GetNBFace();
    IntType steady = 1;
    grid->GetData(&steady, INT, 1, "steady");

    // Allocate memories for flow variables
    rho = NULL; u = NULL;  v = NULL;  w = NULL;  p = NULL;
    mfmem::snew_array_1D(rho , n, dmrfl);
    mfmem::snew_array_1D(u   , n, dmrfl);
    mfmem::snew_array_1D(v   , n, dmrfl);                
    mfmem::snew_array_1D(w   , n, dmrfl);
    mfmem::snew_array_1D(p   , n, dmrfl);

    // attach the data to the grid now
    grid->UpdateDataPtr(rho, REAL_FLOW, n, "rho");
    grid->UpdateDataPtr(u  , REAL_FLOW, n, "u");
    grid->UpdateDataPtr(v  , REAL_FLOW, n, "v");
    grid->UpdateDataPtr(w  , REAL_FLOW, n, "w");
    grid->UpdateDataPtr(p  , REAL_FLOW, n, "p");

    if(!steady){
        RealFlow *rho_cur = NULL;  
        RealFlow *u_cur = NULL;  
        RealFlow *v_cur = NULL;  
        RealFlow *w_cur = NULL;   
        RealFlow *p_cur = NULL;
        RealFlow *rho_old = NULL;  
        RealFlow *u_old = NULL;  
        RealFlow *v_old = NULL;  
        RealFlow *w_old = NULL;   
        RealFlow *p_old = NULL;                  
        mfmem::snew_array_1D(rho_cur, nTCell, dmrfl);                
        mfmem::snew_array_1D(u_cur  , nTCell, dmrfl);
        mfmem::snew_array_1D(v_cur  , nTCell, dmrfl);
        mfmem::snew_array_1D(w_cur  , nTCell, dmrfl);                
        mfmem::snew_array_1D(p_cur  , nTCell, dmrfl);
        mfmem::snew_array_1D(rho_old, nTCell, dmrfl);                
        mfmem::snew_array_1D(u_old  , nTCell, dmrfl);
        mfmem::snew_array_1D(v_old  , nTCell, dmrfl);
        mfmem::snew_array_1D(w_old  , nTCell, dmrfl);                
        mfmem::snew_array_1D(p_old  , nTCell, dmrfl);

        grid->UpdateDataPtr(rho_cur, REAL_FLOW, nTCell, "rho_cur");
        grid->UpdateDataPtr(u_cur  , REAL_FLOW, nTCell, "u_cur");
        grid->UpdateDataPtr(v_cur  , REAL_FLOW, nTCell, "v_cur");
        grid->UpdateDataPtr(w_cur  , REAL_FLOW, nTCell, "w_cur");
        grid->UpdateDataPtr(p_cur  , REAL_FLOW, nTCell, "p_cur");

        grid->UpdateDataPtr(rho_old, REAL_FLOW, nTCell, "rho_old");
        grid->UpdateDataPtr(u_old  , REAL_FLOW, nTCell, "u_old");
        grid->UpdateDataPtr(v_old  , REAL_FLOW, nTCell, "v_old");
        grid->UpdateDataPtr(w_old  , REAL_FLOW, nTCell, "w_old");
        grid->UpdateDataPtr(p_old  , REAL_FLOW, nTCell, "p_old");        
    }
}


/************************************************************************
          Read flow variables from restart file
************************************************************************/
void NSSolver::ReadRestartFromFile(PolyGrid *grid)
{
    FILE       *fp; 
    IntType    ntmp, iter, nvar;
    RealFlow   *rho, *u, *v, *w, *p, start_time;
    RealFlow   *rho_cur, *u_cur, *v_cur, *w_cur, *p_cur;
    RealFlow   *rho_old, *u_old, *v_old, *w_old, *p_old;
    IntType    nTCell = grid->GetNTCell(), n = nTCell + grid->GetNBFace();
    IntType    zn = grid->GetZone();
    IntType    steady, Unst_steps_Curt;
    grid->GetData(&steady, INT, 1, "steady");

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
    fread(&Unst_steps_Curt, sizeof(IntType), 1, fp);
    fread(&iter, sizeof(IntType), 1, fp);
    fread(&nvar, sizeof(IntType), 1, fp);
    fread(&start_time, sizeof(RealFlow), 1, fp);    

    if(ntmp != nTCell) {
       printf("Total numbers of cells in the grid = %ld\n", (long)nTCell);
       printf("Total numbers of cells in the file = %ld\n", (long)ntmp);
       mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
  
    rho  = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    u    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    v    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    w    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    p    = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    assert(rho!=NULL);

    if(!steady){
        rho_cur = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "rho_cur");
        u_cur   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "u_cur");
        v_cur   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "v_cur");
        w_cur   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "w_cur");
        p_cur   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "p_cur");
        rho_old = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "rho_old");
        u_old   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "u_old");
        v_old   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "v_old");
        w_old   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "w_old");
        p_old   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "p_old");
        assert(rho_cur!=NULL);
    }
    
    fread(rho, sizeof(RealFlow), nTCell, fp);
    fread(u,   sizeof(RealFlow), nTCell, fp);
    fread(v,   sizeof(RealFlow), nTCell, fp);
    fread(w,   sizeof(RealFlow), nTCell, fp);
    fread(p,   sizeof(RealFlow), nTCell, fp);

    if(!steady){
        //cur stand for the current , n level time
        fread(rho_cur, sizeof(RealFlow), nTCell, fp);
        fread(u_cur,   sizeof(RealFlow), nTCell, fp);
        fread(v_cur,   sizeof(RealFlow), nTCell, fp);
        fread(w_cur,   sizeof(RealFlow), nTCell, fp);
        fread(p_cur,   sizeof(RealFlow), nTCell, fp);

        //old stand for the n-1 level time
        fread(rho_old, sizeof(RealFlow), nTCell, fp);
        fread(u_old,   sizeof(RealFlow), nTCell, fp);
        fread(v_old,   sizeof(RealFlow), nTCell, fp);
        fread(w_old,   sizeof(RealFlow), nTCell, fp);
        fread(p_old,   sizeof(RealFlow), nTCell, fp);
    }
    fclose(fp);    
}


/************************************************************************
                        Intialize flow field 
NOTE:   1、首先为全流场赋来流值
        2、对应发动机内部或喷流喷口前端，使用fluent格式的网格可以为某部分
        体单元特殊处理：
            喷流赋喷流边界值；
            发动机使用正激波关系式求得的波后值；
Update: 2012-5-5 13:26:24
        特殊处理发动机或喷流初值
************************************************************************/
void NSSolver::InitGridVar(PolyGrid *grid)
{
    IntType     n, i;
    RealFlow    *rho=0,*u,*v,*w,*p;
    RealFlow    rhoP,uP,vP,wP,pP;  // for the parameters

    IntType nTCell = grid->GetNTCell();
    n = nTCell + grid->GetNBFace();

    rho = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    u   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    v   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    w   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    p   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");

    if(!rho){
        u   = NULL;  v   = NULL;  w   = NULL;  p   = NULL;
        mfmem::snew_array_1D(rho , n,dmrfl);
        mfmem::snew_array_1D(u   , n,dmrfl);
        mfmem::snew_array_1D(v   , n,dmrfl);                
        mfmem::snew_array_1D(w   , n,dmrfl);
        mfmem::snew_array_1D(p   , n,dmrfl);
        // attach the data to the grid now
        grid->UpdateDataPtr(rho, REAL_FLOW, n, "rho");
        grid->UpdateDataPtr(u  , REAL_FLOW, n, "u");
        grid->UpdateDataPtr(v  , REAL_FLOW, n, "v");
        grid->UpdateDataPtr(w  , REAL_FLOW, n, "w");
        grid->UpdateDataPtr(p  , REAL_FLOW, n, "p");
    }
    
    GetData(&rhoP,REAL_FLOW,1,"rho");
    GetData(&uP  ,REAL_FLOW,1,"u");
    GetData(&vP  ,REAL_FLOW,1,"v");
    GetData(&wP  ,REAL_FLOW,1,"w");
    GetData(&pP  ,REAL_FLOW,1,"p");
    for(i=0; i<n; i++) {
        rho[i] = rhoP;
        u[i]   = uP;
        v[i]   = vP;
        w[i]   = wP;
        p[i]   = pP;
    }
}


/************************************************************************
          Initialize flow field as a uniform flow
************************************************************************/
void NSSolver::InitGridVarUnst(PolyGrid *grid)
{
    IntType     n,i;
    RealFlow    *rho_old,*u_old,*v_old,*w_old,*p_old;
    RealFlow    *rho_cur,*u_cur,*v_cur,*w_cur,*p_cur;
    RealFlow    rhoP,uP,vP,wP,pP;  // for the parameters
    IntType     nTCell = grid->GetNTCell();

    n = grid->GetNTCell()+grid->GetNBFace();    

    rho_cur = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "rho_cur");
    u_cur   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "u_cur");
    v_cur   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "v_cur");
    w_cur   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "w_cur");
    p_cur   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "p_cur");
    rho_old = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "rho_old");
    u_old   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "u_old");
    v_old   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "v_old");
    w_old   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "w_old");
    p_old   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, nTCell, "p_old");
    assert(rho_cur!=NULL);
        
    GetData(&rhoP,REAL_FLOW,1,"rho");
    GetData(&uP  ,REAL_FLOW,1,"u");
    GetData(&vP  ,REAL_FLOW,1,"v");
    GetData(&wP  ,REAL_FLOW,1,"w");
    GetData(&pP  ,REAL_FLOW,1,"p");
    for(i=0; i<nTCell; i++) {
        rho_cur[i] = rhoP;
        u_cur[i]   = uP;
        v_cur[i]   = vP;
        w_cur[i]   = wP;
        p_cur[i]   = pP;

        rho_old[i] = rhoP;
        u_old[i]   = uP;
        v_old[i]   = vP;
        w_old[i]   = wP;
        p_old[i]   = pP;
    }     

    // modified for danxun
    RealFlow *rho = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
    RealFlow *u   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
    RealFlow *v   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
    RealFlow *w   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
    RealFlow *p   = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");
    for(i=0; i<nTCell; i++) {
        rho_cur[i] = rho[i];
        u_cur[i]   = u[i];
        v_cur[i]   = v[i];
        w_cur[i]   = w[i];
        p_cur[i]   = p[i];

        rho_old[i] = rho[i];
        u_old[i]   = u[i];
        v_old[i]   = v[i];
        w_old[i]   = w[i];
        p_old[i]   = p[i];
    } 
}


/*******************************************************************************\
              modify input file for restart
\*******************************************************************************/
void ModifyRestartInputFile(PolyGrid *grid)
{
#ifdef MPICH
    if(myZone != 1) return;
#endif

    // The keys and values of parameters to be modified
    std::map<std::string, std::string> params;
    params.insert(std::make_pair("restart", "1"));

    // guess which type of parameter file exist.
    std::string file("input.para");
    if (CheckFileReadable(file))
    {
#ifdef MPICH
        if(myZone == 1)
            modify_parameter_file(file, params);
#else
        modify_parameter_file(file, params);
#endif        
    }
    else
    {
        file = "input";
        file += ".par";

        modify_parameter_file(file, params);
    }
}  


/*******************************************************************************\
              compute the unsteady source in the res
\*******************************************************************************/
void AddUnstSource(PolyGrid *grid)
{
    IntType  i, j, n;
    IntType  nTCell = grid->GetNTCell(), nBFace = grid->GetNBFace();
    RealFlow *rho, *u, *v, *w, *p;
    RealFlow *rho_old, *u_old, *v_old, *w_old, *p_old;
    RealFlow *rho_cur, *u_cur, *v_cur, *w_cur, *p_cur;
    RealFlow *res[5], Q[5], Q_cur[5], Q_old[5];
    RealGeom *vol = grid->GetCellVol();
    RealFlow time_accuracy, real_dt, p00, gam, gam1;

    n = nTCell+nBFace;

    rho = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "rho");  
    u   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "u");  
    v   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "v");  
    w   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "w");  
    p   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, n, "p");  

    rho_cur = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "rho_cur");  
    u_cur   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "u_cur");  
    v_cur   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "v_cur");  
    w_cur   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "w_cur");  
    p_cur   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "p_cur");  

    rho_old = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "rho_old");  
    u_old   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "u_old");  
    v_old   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "v_old");  
    w_old   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "w_old");  
    p_old   = (RealFlow *) grid->GetDataPtr(REAL_FLOW, nTCell, "p_old"); 

    grid->GetData(&time_accuracy, REAL_FLOW, 1, "time_accuracy");
    grid->GetData(&real_dt, REAL_FLOW, 1, "real_dt");
    grid->GetData(&p00, REAL_FLOW, 1, "p_bar");
    grid->GetData(&gam,    REAL_FLOW, 1, "gam");
    gam1 = gam - 1.0;

    res[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, 5*nTCell, "res");
    for(i=1; i<5; i++) res[i] = &res[i-1][nTCell];

    for(i=0; i<nTCell; i++){
        Q[0] = rho[i];
        Q[1] = rho[i]*u[i];
        Q[2] = rho[i]*v[i];
        Q[3] = rho[i]*w[i];
        Q[4] = (p[i]+p00)/gam1 + 0.5*rho[i]*(u[i]*u[i] + v[i]*v[i] + w[i]*w[i]);

        Q_cur[0] = rho_cur[i];
        Q_cur[1] = rho_cur[i]*u_cur[i];
        Q_cur[2] = rho_cur[i]*v_cur[i];
        Q_cur[3] = rho_cur[i]*w_cur[i];
        Q_cur[4] = (p_cur[i]+p00)/gam1 + 0.5*rho_cur[i]*(u_cur[i]*u_cur[i] + v_cur[i]*v_cur[i] + w_cur[i]*w_cur[i]);

        Q_old[0] = rho_old[i];
        Q_old[1] = rho_old[i]*u_old[i];
        Q_old[2] = rho_old[i]*v_old[i];
        Q_old[3] = rho_old[i]*w_old[i];
        Q_old[4] = (p_old[i]+p00)/gam1 + 0.5*rho_old[i]*(u_old[i]*u_old[i] + v_old[i]*v_old[i] + w_old[i]*w_old[i]);

        for(j=0; j<5; j++){
            res[j][i] += (-(1.0+time_accuracy)*Q[j]*vol[i] + (1.0+time_accuracy)*Q_cur[j]*vol[i]
                         + time_accuracy*(Q_cur[j]*vol[i] - Q_old[j]*vol[i]))/real_dt;
        }
    }
}


/******************************************************************************\
  判断是否主控方程残差是否发散
  目的是解决超声速（Ma 2~5）时偶尔出现的过鲁棒导致计算结果错误的问题，但是可能有误杀的情况，
  需要根据后续的实践情况进行改进！目前仅限于定常计算。
\******************************************************************************/
void TellDivergence(PolyGrid *grid, IntType iter, RealFlow norm)
{
    IntType steady;
    grid->GetData(&steady, INT, 1, "steady");
    if(!steady) return; //目前只应用到定常计算    
    
    if(iter == 1){
        if(norm < 1.0e-6){ //if the magnitude of norm is too small, then order it with 1.0. Such as in flat plate computation
            norm = 1.0;
        }
        grid->UpdateData(&norm, REAL_FLOW, 1, "res_rho_step1"); //zhyb:保存第一步的密度残差
        return;
    }else{
        RealFlow norm_step1 = -1.0;
        grid->GetData(&norm_step1, REAL_FLOW, 1, "res_rho_step1", 0);
        if(norm_step1<0.0){ //可能是续算导致获取不到第一步的值，因此需要在此从文件中读取
            string filename = "resid_glb.out";
            ifstream fin(filename.c_str(),ifstream::in);
            if(!fin){
                std::cerr<<"In function DumpNormResi, Error: can not open file "<<filename<<endl;
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
            string tmp;
            getline(fin,tmp);  //标题行
            getline(fin,tmp);  //第一步数据
            stringstream ss;
            ss<<tmp;
            int step1;
            ss>>step1;
            assert(step1 == 1);
            ss>>norm_step1;
            assert(norm_step1 > 0.0);
            
            if(norm_step1 < 1.0e-10){ //if the magnitude of norm is too small, then order it with 1.0. Such as in flat plate computation
                norm = 1.0;
            }
            grid->UpdateData(&norm_step1, REAL_FLOW, 1, "res_rho_step1");
            fin.close();
        }
        
        //zhyb:与第一步密度残差相比较，若当前步残差大于第一步的两个量级，则报错，提醒减小cfl数，退出计算
        //zhyb:误杀现象太严重，特别是做多重计算时，因此把它放大为三个量级，20190417
        IntType mark = 0;
        if(norm/norm_step1 > 1000.0) mark++;
        if(mark){
            mflog::log.set_one_processor_out();
            mflog::log<<endl<<"Error! Now residual is bigger than step1 for 3 order! "
                      <<"Maybe you need reduce CFL number!"<<endl;   
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
}

#undef CPP_FILD_ID  // clear out file id
} //~namespace mflow
