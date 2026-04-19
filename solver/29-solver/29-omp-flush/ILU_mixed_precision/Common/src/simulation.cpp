//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   simulation.cpp
/// \brief  A class for simulation
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
#include "simulation.h"

// C/C++ head files
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
using namespace std;

// other user defined head files
#include "zone.h"
#include "utility_functions.h"
#include "algm.h"
#include "solver_ns.h"
#include "io_grid.h"
#include "io_log.h"
#include "io_base_format.h"
#include "io_field.h"
#include "parallel_base_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"
#include "parameter_reader.h"
#include "solver_turb_sa.h"

// head files relying on condition-compiling
#ifdef MPICH
#include <mpi.h>
#endif

#ifdef FS_SIMD
#include <omp.h>
#endif

#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif

//dingxin
#ifdef TIMECOST
double* timecost;
int num_timecost;
double  time_flux, time_invis, time_roe, time_vis, time_calvis;
double  time_limiter;
double  time_gradient;
double  time_lusgs;
double  time_SA;
extern double ILUbuild, ILUexe, Matrixbuild, GMRESexe, MPIexe, GMRES_Schmidt;
extern int ite;
#endif
//TIMECOST

#if (defined FS_CUDA)||(defined FS_CUDA_DEBUG)
//gpuruitian, 2021.3.25
#include "cuData.cuh"
#include "cuDeviceControl.cuh"
#include <cuTurbulenceFlux.cuh>
#include <cuViscidFlux.cuh>
using namespace gpuData;
#endif

namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 11901  // define file id

#ifdef MPICH
extern int myZone;
extern int numprocs;
extern MPI_Comm GridComm;  //for each grid, tangj
#endif


Simulation::Simulation() : 
    steady(1), dynamic(0), nZones(0), zones(NULL)
{
}


/************************************************************************
*               delete all these zones                                  *
************************************************************************/
Simulation::~Simulation()
{
    IntType z;

    for(z=0; z<nZones; z++) {
        mfmem::sdel_object(zones[z]);
    }
    mfmem::sdel_array_1D(zones);
}


/************************************************************************
*          Initialize object as the constructor do                      *
* Note:  
* Author: tangj 2020-06-13
************************************************************************/ 
void Simulation::Construct(const ParameterReader *params)
{
    this->SetSteadyFlag(params->GetSteadyFlag());
    this->SetDynamicFlag(params->GetDynamicFlag());

    // copy the simulation parameters
    this->CopyParameters(params->get_simulation_parameters());

    // creat zone for simulation
    Zone *zone = NULL;
    mfmem::snew_object(zone, dmrfl);

    // figure out which grid locate in this simulation
    IntType grid_number = 1;

    // set parameter for zone
    zone->CopyParameters(params->get_zone_parameters(grid_number-1));

    // set boundary condition patches for zone
    zone->CopyZoneBCs(params->get_zone_bcond(grid_number-1));

    // insert zone to this simulation
    this->AddZone(zone);
}


/************************************************************************
*               add a new zone to the simulation                        *
************************************************************************/
void Simulation::AddZone(Zone *zone)
{
    if(nZones == 0) {
        mfmem::snew_array_1D(zones, nZones+1,dmrfl);
        zones[nZones++] = zone;
    } else {
        Zone **zonet;
        
        zonet = zones;
        zones = NULL;
        mfmem::snew_array_1D(zones, nZones+1,dmrfl);
        for(IntType i=0; i<nZones; i++) {
            zones[i] = zonet[i];
        }
        mfmem::sdel_array_1D(zonet);
        zones[nZones++] = zone;
    }
}


/************************************************************************
*               Simulation                                              *
************************************************************************/
void Simulation::Start(int argc, char *argv[])
{
    // pass simulation object to static variable of class Zone
    Zone::simu = this;    

    // Fix, deduce and check parameters for all zones
    UpdateParameter();
    
    // Load original grid
    LoadGridDataForZones();

    Init();
	
	// gpuruitian, 2022.3.30
    // gpu data initialization:
#if (defined FS_CUDA)||(defined FS_CUDA_DEBUG)
	//set gpu device:
	IntType mpirank = 0; //when mpirank = 0, mpi was off. 
	
#ifdef MPICH        
    MPI_Comm_rank(MPI_COMM_WORLD, & mpirank);	
#endif
	GetGPUNum(GPUNum);
	MultiGPUDevice(GPUNum, mpirank, GPUProp);
	Zone* cz;
	PolyGrid* grid;
	IntType nGrids, nTFace, nTCell, nBFace, nIFace, n, nTNode;
	IntType i, j, k;
	IntType* f2c;
	RealGeom* xfc, * yfc, * zfc, * xcc, * ycc, * zcc;
	RealGeom* xfn, * yfn, * zfn, * area, * vgn, * vol;
	BCRecord** bcr;
    RealFlow * tw_bcr;
    IntType* type_bcr;
    IntType  steady = 1;
    RealFlow p_bar;
    IntType *IndexC2F;
    IntType **C2F;
    IntType* nFPC;
    RealGeom *facecentroidskewness;
    RealGeom *angle_h;
    IntType* nNPF;
    IntType* IndexF2N;
    IntType** F2N;
    IntType GaussLayer = 1;
    IntType *cellwallnumber;
    IntType *CellLayerNo;
    // node to cell connection:
    IntType* nCPN;
    IntType *IndexN2C;
    IntType** N2C;
    IntType* Nmark;
    IntType *nodesymm = 0;
    RealGeom *xfn_n_symm, *yfn_n_symm, *zfn_n_symm;   
    RealFlow lhs_omga;
    RealFlow gam;
    IntType  **C2C, *IndexC2C, *nCPC; // cell to cell connect

    RealGeom** WeightNodeBFace2C;
    // IntType *IndexWeightNodeBFace2C;
    RealGeom** WeightNodeN2C;
    // IntType *IndexWeightNodeN2C;
    RealGeom* WeightNode;
    RealGeom *norm_dist_c2c;

    IntType *luorder; //为了对角占优，网格排序后的序号
    IntType *layer; //LUSGS迭代的层号，层号小为下三角，层号大为上三角
    IntType *cellsPerlayer;

    IntType *fcptr;
    RealGeom *dist2wall;
	
	NSSolver* solver;
	for (i = 0; i < nZones; ++i) {
		cz = zones[i];
		nGrids = cz->GetNoOfGrids();
		for (j = 0; j < nGrids; j++) {
			grid = (PolyGrid*)cz->GetGrid(j);
            type_bcr = NULL;
            tw_bcr = NULL;
            IndexC2F = NULL;
            IndexC2C = NULL;
            IndexF2N = NULL;
            IndexN2C = NULL;
            //angle_h = NULL;
			nTFace = grid->GetNTFace();
			nTCell = grid->GetNTCell();
			nBFace = grid->GetNBFace();
			nIFace = grid->GetNIFace();
            nTNode = grid->GetNTNode();
            grid->GetData(&steady, INT, 1, "steady");
            grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
			RealGeom eps_tmp;        
			RealFlow vol_avg = grid->GetVolAvg();
			assert(vol_avg > 0.0); //volumn average must exist
			RealFlow eps_vencat=1.0;
			grid->GetData(&eps_vencat, REAL_FLOW, 1, "eps_vencat",0);
			eps_tmp = eps_vencat*eps_vencat*eps_vencat/vol_avg; 
			
			n = nTCell + nBFace;
			mfmem::snew_array_1D(type_bcr, nBFace, dmrfl);
            mfmem::snew_array_1D(tw_bcr, nBFace, dmrfl);
            mfmem::snew_array_1D(IndexC2F, nTCell + 1, dmrfl);
            mfmem::snew_array_1D(IndexC2C, nTCell + 1, dmrfl);
            mfmem::snew_array_1D(IndexF2N, nTFace + 1, dmrfl);
            mfmem::snew_array_1D(IndexN2C, nTNode + 1, dmrfl);
			f2c = grid->Getf2c();
			xfc = grid->GetXfc();
			yfc = grid->GetYfc();
			zfc = grid->GetZfc();
			xcc = grid->GetXcc();
			ycc = grid->GetYcc();
			zcc = grid->GetZcc();
			xfn = grid->GetXfn();
			yfn = grid->GetYfn();
			zfn = grid->GetZfn();
			bcr = grid->Getbcr();
            vol = grid->GetCellVol();
			area = grid->GetFaceArea();
			vgn = grid->GetFaceNormalVelocity();           
                       
            facecentroidskewness = grid->GetGridQualityFaceCentroidSkewness(); 
                        
            grid->GetData(&GaussLayer, INT, 1, "GaussLayer");
            cellwallnumber = grid->GetGridQualityCellWallNumber();
            CellLayerNo = (IntType *)grid->GetDataPtr(INT, n, "CellLayerNo");

			for (k = 0; k < nBFace; k++){
				type_bcr[k] = bcr[k]->GetType();
                tw_bcr[k] = -1.0;
                bcr[k]->GetBCVar(&tw_bcr[k], REAL_FLOW, "tw", 0);
            }
            
            nFPC = CalnFPC(grid);
            C2F = grid->GetC2F(); 
            IndexC2F[0] = 0;
            for(IntType i = 1; i < nTCell + 1; i++){
                IndexC2F[i] = IndexC2F[i - 1] + nFPC[i - 1];
                // IndexC2F[nTCell] was the length of **C2F, gIndexC2F only copy from IndexC2F[0] to IndexC2F[nTCell-1]
            }

            nNPF = grid->GetnNPF();
            F2N = CalF2N(grid);
            IndexF2N[0] = 0;
            for(IntType i = 1; i < nTFace + 1; i++){
                IndexF2N[i] = IndexF2N[i - 1] + nNPF[i - 1];
                // IndexF2N[nTFace] was the length of **F2N, gIndexF2N only copy from IndexF2N[0] to IndexF2N[nTFace-1]
            }

            nCPN = CalnCPN(grid);
            N2C = CalN2C(grid);           
            IndexN2C[0] = 0;
            for(IntType i = 1; i < nTNode + 1; i++){
                IndexN2C[i] = IndexN2C[i - 1] + nCPN[i - 1];
                // IndexN2C[nTNode] was the length of **N2C, gIndexN2C only copy from IndexN2C[0] to IndexN2C[nTNode-1]
            }
            // N2C seems to be no 
            IntType* oneDimN2C = NULL;
            mfmem::snew_array_1D(oneDimN2C, IndexN2C[nTNode], dmrfl);
            IntType cdex = 0;
            for (IntType i = 0; i < nTNode; i++) {
                for (IntType j = 0; j < nCPN[i]; j++) {
                    oneDimN2C[cdex] = N2C[i][j];
                    cdex++;
                }
            }
            
            nCPC = CalnCPC(grid);
            C2C = CalC2C(grid); 
            IndexC2C[0] = 0;
            for(IntType i = 1; i < nTCell + 1; i++){
                IndexC2C[i] = IndexC2C[i - 1] + nCPC[i - 1];
                // IndexC2C[nTCell] was the length of **C2C, gIndexC2C only copy from IndexC2C[0] to IndexC2C[nTCell-1]
            }
            
            if (grid->GetWeightNodeDist() == NULL) {
                ComputeWeight3D_Node(grid);  //距离分之一权
            }
			WeightNodeBFace2C = grid->GetWeightNodeBFace2C();
            WeightNodeN2C = grid->GetWeightNodeN2C();   
            Nmark = grid->GetNodeType();

            WeightNode = grid->GetWeightNodeDist();

            nodesymm = (IntType *)grid->GetDataPtr(INT, nTNode, "nodesymm");
            if(!nodesymm){
                FindNodeSYMM(grid);
                nodesymm = (IntType *)grid->GetDataPtr(INT, nTNode, "nodesymm");
            }

            xfn_n_symm = (RealFlow *)grid->GetDataPtr(REAL_GEOM, nTNode, "xfn_n_symm");
            yfn_n_symm = (RealFlow *)grid->GetDataPtr(REAL_GEOM, nTNode, "yfn_n_symm");
            zfn_n_symm = (RealFlow *)grid->GetDataPtr(REAL_GEOM, nTNode, "zfn_n_symm");

            grid->GetData(&lhs_omga,  REAL_FLOW, 1, "lhs_omga");
            grid->GetData(&gam,    REAL_FLOW, 1, "gam");

            norm_dist_c2c = NULL;
            norm_dist_c2c = (RealGeom *)grid->GetDataPtr(REAL_GEOM, nTFace, "norm_dist_c2c");

            luorder = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGSCellOrder"); //为了对角占优，网格排序后的序号
            layer = (IntType *)grid->GetDataPtr(INT, n, "LUSGSLayer"); //LUSGS迭代的层号，层号小为下三角，层号大为上三角
            cellsPerlayer = (IntType *)grid->GetDataPtr(INT, nTCell, "LUSGScellsPerlayer");

            fcptr = (IntType*)grid->GetDataPtr(INT, 2*nTFace, "fcptr");
            if(!fcptr){  // Calculate fcptr
                CalCNNCF(grid);
                fcptr = (IntType*)grid->GetDataPtr(INT, 2*nTFace, "fcptr");
            }

            dist2wall = (RealGeom *) grid->GetDataPtr(REAL_GEOM, nTCell, "dist2wall_cell");
            
            angle_h = CalAngle(grid);

            IntType* IsNormalFace = 0;
            IntType EntropyCorType = 4;
            grid->GetData(&EntropyCorType, INT, 1, "EntropyCorType");
            if (EntropyCorType == 4) {
                IsNormalFace = (IntType*)grid->GetDataPtr(INT, nTFace, "IsNormalFace");
                if (!IsNormalFace) {
                    grid->FindNormalFace();
                    IsNormalFace = (IntType*)grid->GetDataPtr(INT, nTFace, "IsNormalFace");
                }
            }           
            
            GPUFlowCondition(steady, p_bar, GaussLayer, lhs_omga, gam, eps_tmp);
            // Index for bqs and bqr for nvar=5 (MPI transfer q):
			IntType *IndexMPIbqs = NULL;
			IntType *IndexMPIbqsSA = NULL;
			IntType *IndexMPIbqsGrad = NULL;
			IntType *IndexMPIbqsGradSA = NULL;
			
			IntType *IndexMPIbqr = NULL;
			IntType *IndexMPIbqrSA = NULL;
			IntType *IndexMPIbqrGradSA = NULL;

            // GMRES:
            IntType gmres = 0;
            IntType kspan = 10;
            grid->GetData(&gmres, INT, 1, "GMRES", 0);
            if(gmres == 1){                
                grid->GetData(&kspan, INT, 1, "kspan");
            }
			IntType sweeps = 1;
			grid->GetData(&sweeps, INT, 1, "sweeps");

#ifdef MPICH
            grid->cuGetLength_RecvSend();
            
            mfmem::snew_array_1D(IndexMPIbqs, glenbqsr, dmrfl);
            grid->cuGetIndex_RecvSend(IndexMPIbqs, 5);
           
            mfmem::snew_array_1D(IndexMPIbqsSA, glenbqsrSA, dmrfl);
            grid->cuGetIndex_RecvSend(IndexMPIbqsSA, 1);
            
            mfmem::snew_array_1D(IndexMPIbqsGrad, glenbqsrGrad, dmrfl);
            grid->cuGetIndex_RecvSend(IndexMPIbqsGrad, 15);          
			
			mfmem::snew_array_1D(IndexMPIbqr, glenbqsr, dmrfl);
            grid->cuGetIndex_RecvSend2(IndexMPIbqr, 5);
			
			mfmem::snew_array_1D(IndexMPIbqrSA, glenbqsrSA, dmrfl);
            grid->cuGetIndex_RecvSend2(IndexMPIbqrSA, 1);
			
#endif 			
			GPUFaceNumberTrans(nTFace, nTCell, nBFace, nIFace, nTNode);
			GPUFaceDataTrans(xfc, yfc, zfc, xfn, yfn, zfn, xcc, ycc, zcc,
                            area, vgn, f2c, type_bcr, tw_bcr);
                           
            GPUFaceDataTrans2(C2F, IndexC2F, nFPC, vol, facecentroidskewness, 
                            angle_h, nNPF, F2N, IndexF2N, cellwallnumber, CellLayerNo);
			
            GPUFaceDataTrans3(oneDimN2C, IndexN2C, nCPN, WeightNodeBFace2C, WeightNodeN2C, 
                            IndexF2N, Nmark, WeightNode, nodesymm, xfn_n_symm, yfn_n_symm, zfn_n_symm,
                            norm_dist_c2c);
            
			GPUFaceDataTrans4(luorder, layer, cellsPerlayer, C2C, IndexC2C, nCPC, fcptr, dist2wall, 
                            IsNormalFace, EntropyCorType, IndexMPIbqs, IndexMPIbqsSA, IndexMPIbqsGrad, IndexMPIbqr, IndexMPIbqrSA);
			mfmem::sdel_array_1D(type_bcr);
            mfmem::sdel_array_1D(tw_bcr);
            mfmem::sdel_array_1D(IndexC2F);
            mfmem::sdel_array_1D(IndexC2C);
            mfmem::sdel_array_1D(IndexF2N);
            mfmem::sdel_array_1D(IndexN2C);
			mfmem::sdel_array_1D(oneDimN2C);
			mfmem::sdel_array_1D(angle_h);
#if (defined GroupColor)
			IntType groupSize = grid->groupSize;
			IntType length_b_SMc2c, length_i_SMc2c;
			IntType length_b_f2SMc, length_i_f2SMc;
			IntType n_bcolor, n_icolor;
			n_bcolor = grid->bfacegroup.size();
			n_icolor = grid->ifacegroup.size();
			// SM cell to cell length:
			IntType num_b_group = grid->group_b_SM_color_index[n_bcolor - 1];
			length_b_SMc2c = grid->group_b_SM_index[num_b_group];
			IntType num_i_group = grid->group_i_SM_color_index[n_icolor - 1];
			length_i_SMc2c = grid->group_i_SM_index[num_i_group];
			cout << "rankid: " << mpirank << "-" << "length_b_SMc2c: " << length_b_SMc2c << "; length_i_SMc2c: " << length_i_SMc2c << endl;
			// global face to SM cell length:
			length_b_f2SMc = nBFace - nIFace;
			length_i_f2SMc = nTFace*2;
			GPUFaceDataTransGroupColor(length_b_SMc2c, length_i_SMc2c, length_b_f2SMc, length_i_f2SMc, num_b_group, num_i_group, 
									grid->group_b_SMc2c, grid->group_b_f2SMc, grid->group_i_SMc2c, grid->group_i_f2SMc,
									grid->group_b_SM_index, grid->group_i_SM_index);
#endif		

            GPUFlowMemoryAlloc();
#if defined MultiStream		
			GPUGrad_Limit_Init();
#endif
            GPUGMRESMemoryAlloc(kspan, gmres, sweeps);

            // flow variables init.
            RealFlow *q[5], *sa_nu;
            q[0] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "rho");
            q[1] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "u");
            q[2] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "v");
            q[3] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "w");
            q[4] = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "p");

			sa_nu = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "sa_nu");

            RealFlow *vis_l = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_l");
            RealFlow *vis_t = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, "vis_t");

            GPUFlowMemoryInit(q, sa_nu, vis_l, vis_t);

            RealFlow *t = cuGetTemperature(grid);
            mfmem::sdel_array_1D(t);
			/* cout << "Success! " << endl;
			exit(0); */
#ifdef MPICH
            mfmem::sdel_array_1D(IndexMPIbqs);
            mfmem::sdel_array_1D(IndexMPIbqsSA);
            mfmem::sdel_array_1D(IndexMPIbqsGrad);
            mfmem::sdel_array_1D(IndexMPIbqsGradSA);
			mfmem::sdel_array_1D(IndexMPIbqr);
			mfmem::sdel_array_1D(IndexMPIbqrSA);
			mfmem::sdel_array_1D(IndexMPIbqrGradSA);
#endif
		}		
		grid = (PolyGrid*)cz->GetGrid(0);		
		solver = (NSSolver*)cz->GetNSSolver();
		solver->QuantityGradient_Init(grid);

        

	}
#endif

    RunSimu();
	
	//gpuruitian, 2022.3.30
/* #if (defined FS_CUDA)
	for (i = 0; i < nZones; ++i) {
		cz = zones[i];
		nGrids = cz->GetNoOfGrids();
		for (j = 0; j < nGrids; j++) {
			grid = (PolyGrid*)cz->GetGrid(j);
			//type_bcr = grid->GetTB();
			f2c = grid->Getf2c();
			xfc = grid->GetXfc();
			yfc = grid->GetYfc();
			zfc = grid->GetZfc();
			xcc = grid->GetXcc();
			ycc = grid->GetYcc();
			zcc = grid->GetZcc();
			xfn = grid->GetXfn();
			yfn = grid->GetYfn();
			zfn = grid->GetZfn();
			//bcr = grid->Getbcr();
			area = grid->GetFaceArea();
			vgn = grid->GetFaceNormalVelocity();
			nTFace = grid->GetNTFace();
			nTCell = grid->GetNTCell();
			nBFace = grid->GetNBFace();
			n = nTCell + nBFace;

			
		}
	}
#endif */
    PostSimu();
}


/************************************************************************
*                                                                       *
************************************************************************/
void Simulation::RunSimu()
{    
#ifdef MPICH
    double  t1, t2;
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
#else
	struct timeval starttimeTemInvis, endtimeTemInvis;
    double timetotal;
    gettimeofday(&starttimeTemInvis, 0); 
#endif
    
    // 若为定常的气动力计算,Unst_steps = 1
    // 若为非定常计算,包括非定常流动和多体分离问题,Unst_steps为总的时间步数
    // n_steps为定常（或第一个物理时间步）的子迭代步数
    // n_steps_unst为第二个及以后物理时间步的子迭代步数
    IntType Unst_steps = 1, n_steps, n_steps_unst;
    GetData(&Unst_steps,  INT, 1, "Unst_steps");
    GetData(&n_steps,     INT, 1, "n_steps");
    GetData(&n_steps_unst,INT, 1, "n_steps_unst");
    
    // time steps have been computed, used to continue computing
    IntType Unst_steps_Curt = 0;
    GetData(&Unst_steps_Curt, INT, 1, "Unst_steps_Curt");    

    // Give a correct iteration steps used to continue computing
    IntType n_steps_Curt = AssignInitalIterationSteps();

    // initialize grid directory for each zone
    vector<string> grid_dir_of_zone(nZones);
    this->GetInitialGridDir(grid_dir_of_zone);
    
    // Print out restart file if necessary
    IntType n_wrest = 500;
    this->GetZone(0)->GetData(&n_wrest, INT, 1, "n_wrest");
    
    //zhyb:开始执行计算
    for(IntType n_unst = Unst_steps_Curt; n_unst < Unst_steps; )  //zhyb: n_unst的增加在循环内进行
    {
        UpdateData(&n_unst, INT, 1, "n_unst");

        mflog::log.set_one_processor_out();
        mflog::log << "This is the " << n_unst << "th step!" << endl;

        //修改非定常中计算的time_accuracy的参数
        if(!steady) UpdateTimeAccuracy(n_unst);

        // initialize steps of inner iteration
        if(n_unst != 0) n_steps = n_steps_unst;   //zhyb: 非定常子迭代数
        IntType n_steps_inner = n_steps;        
        
        //zhyb:更新非定常计算时的限制条件，即rho_min,rho_max,p_min,p_max,e_stag_max;
        if(!steady)
        {
            for(IntType zn = 0; zn < nZones; ++zn) zones[zn]->UpdateVariableLimit();
        }             

        //zhyb: 执行子迭代
        RealFlow res_rho_max = 0.0;
        for(IntType inner_iter = n_steps_Curt; inner_iter < n_steps_inner; ++inner_iter) 
        {
            bool exit_inner_iter = DoInnerIteration(inner_iter, res_rho_max);

            if (exit_inner_iter) break;
        } //n_steps_inner            

        // set starting step to zero for a new grid iteration.
        n_steps_Curt = 0;

        //zhyb: 下一个迭代步恢复为0,用于续算
        n_steps_Curt = 0;  

        // communicate interface/overlap data
        for(IntType zn = 0; zn < nZones; ++zn) 
        {
            zones[zn]->UpdateInterfaceData(); 
        }
        
        // Check whether the user wants to stop the simulation, for Unst_steps
        IntType run = zones[0]->BreakZone(); // At least one zone exists, tangj
        if(!run) 
        {
            mflog::log.set_one_processor_out();
            mflog::log << "User wants to stop the simulation!" << std::endl;
            mflog::log << "Please wait a moment." << std::endl;
            break;  
        }

        if(steady)
        {
            n_unst++;
            //UpdateData(&n_unst, INT, 1, "n_unst");
        }
        else
        {
            for(IntType zn = 0; zn < nZones; ++zn)
            {
                zones[zn]->UpdataUnstVolData();
            }
            n_unst++;
            UpdateData(&n_unst, INT, 1, "n_unst");
        }
    } // ~Unsteady steps

#ifdef MPICH
    t2 = MPI_Wtime();
    mflog::log.set_one_processor_out();
    mflog::log << "time = " << IOS_EP(6) << t2-t1 << std::endl;
#else
	gettimeofday(&endtimeTemInvis, 0); 
    timetotal = (RealGeom) 1000000*(endtimeTemInvis.tv_sec - starttimeTemInvis.tv_sec) + endtimeTemInvis.tv_usec - starttimeTemInvis.tv_usec;
    timetotal /= 1000000.0;
	mflog::log << "time = " << IOS_EP(6) << timetotal << std::endl;
#endif

    //dingxin
#ifdef TIMECOST
    mflog::log << "time statistics:(CompGradientQ | Flux | LUSGS | Limiters | SA | other)" << std::endl;
#ifdef MPICH

	IntType mpirank = 0; //when mpirank = 0, mpi was off.       
    MPI_Comm_rank(MPI_COMM_WORLD, & mpirank);
    //printf("mpirank:%d  implicit method time:%lf\n", mpirank, timecost[2]);

    mflog::log.set_one_processor_out();
	timecost[0] = timecost[0] - timecost[5]; 
	timecost[5] = t2 - t1 - (timecost[0] + timecost[1] + timecost[2] + timecost[3] + timecost[4]);
	IntType cout_num = 6;
    for (IntType i = 0; i < cout_num; i++) {
        mflog::log << IOS_EP(6) << timecost[i] << " ";
    }	
	mflog::log << std::endl;
	for (IntType i = 0; i < cout_num; i++) {
        mflog::log << IOS_EP(6) << timecost[i]/(t2-t1)*100 << " ";
    }

    if(!mpirank){
        printf("Matrix setup time: %.8f\n", Matrixbuild);
        printf("GMRES_ILU decomp time: %.8f\n", ILUbuild);
        printf("GMRES_sptrsv or LU-SGS time: %.8f with sptrsv iteration:%d\n", ILUexe, ite);
        printf("GMRES_SPMV time: %.8f\n", MPIexe);
        printf("GMRES_Schmidt time: %.8f\n",GMRES_Schmidt);
        printf("GMRES_ILU total exe: %.8f\n", GMRESexe);
    }
#else
    /*
	for (IntType i = 0; i < num_timecost; i++) {
        //timecost[i] = (double)timecost[i] / CLOCKS_PER_SEC;
		timecost[i] = (double)timecost[i] / 1000000.0;
    }	
	timecost[0] = timecost[0] - timecost[5]; 
	timecost[5] = timetotal - (timecost[0] + timecost[1] + timecost[2] + timecost[3] + timecost[4]);
    for (IntType i = 0; i < num_timecost; i++) {
        mflog::log << IOS_EP(6) << timecost[i] << " ";
    }	
	mflog::log << std::endl;
	for (IntType i = 0; i < num_timecost; i++) {
        mflog::log << IOS_EP(6) << timecost[i]/timetotal*100 << " ";
    }
    */
	time_gradient -= time_calvis;
	time_flux = time_invis + time_vis;
	time_calvis = timetotal - (time_gradient + time_flux + time_lusgs + time_limiter + time_SA);
	mflog::log << IOS_EP(6) << time_gradient << " ";
	mflog::log << IOS_EP(6) << time_flux << " ";
	mflog::log << IOS_EP(6) << time_lusgs << " ";
	mflog::log << IOS_EP(6) << time_limiter << " ";
	mflog::log << IOS_EP(6) << time_SA << " ";
	mflog::log << IOS_EP(6) << time_calvis << " ";
	
	mflog::log << std::endl;
	
	mflog::log << IOS_EP(6) << time_gradient/timetotal*100 << " ";
	mflog::log << IOS_EP(6) << time_flux/timetotal*100 << " ";
	mflog::log << IOS_EP(6) << time_lusgs/timetotal*100 << " ";
	mflog::log << IOS_EP(6) << time_limiter/timetotal*100 << " ";
	mflog::log << IOS_EP(6) << time_SA/timetotal*100 << " ";
	mflog::log << IOS_EP(6) << time_calvis/timetotal*100 << " ";
	
    cout << endl;
	
	/* cout << "invis_nomerge: " << timecost[6] << endl;
	cout << "vis_nomerge: " << timecost[7] << endl;
	cout << "invis_merge: " << timecost[8] << endl;
	cout << "vis_merge: " << timecost[9] << endl; */

#endif
    mflog::log << std::endl;
    mfmem::sdel_array_1D(timecost);
#endif
    //
}


//*****************************************************************************\
/// \brief Do inner iteration
/// 
/// \Note Execute one step of inner iteration. Return true if residual satisfies
///       convergence criteria or user wants to stop this simulation
///
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 2019-03-07  tangj      Create.
/// 2020-08-07  tangj      Update the parameter 'iter_step_physic_time' for NS/Turb
///                        convergence log file.
/// </pre>
//*****************************************************************************/
bool Simulation::DoInnerIteration(const IntType inner_step, RealFlow &res_rho_max)
{
    //n_order_shut：当子迭代连续方程的残差小于最大值n_order_shut个量级时，中断子
    //迭代，进入下一个时间步，但是在程序中限制至少需要进行5个子迭代
    IntType n_order_shut = 15;

    for(IntType zn = 0; zn < nZones; ++zn) 
    {
        Zone *current_zone = GetZone(zn);
        
        // for convergence log file to output the first step of each physical time step
        IntType iter_step_physic_time = inner_step;
        current_zone->UpdateData(&iter_step_physic_time, INT, 1, "iter_step_physic_time");
        
        current_zone->SolveZone();

#ifndef MPICH
        current_zone->UpdateInterfaceData();
#endif
        
    }

    // Check whether the user wants to stop the simulation, for n_steps
    IntType run = zones[0]->BreakZone(); // At least one zone exists, tangj
    if(!run) 
    {
        mflog::log.set_one_processor_out();
        mflog::log << "User wants to stop the simulation for sub-iteration!" << std::endl;
        return true;      
    }

    //判断是否中断子迭代的循环，当连续方程的残差下降超过输入参数的量级时，中断循环
    PolyGrid *grid = (PolyGrid *)zones[0]->GetGrid(0);
    RealFlow res_rho = -1.0;
    grid->GetData(&res_rho, REAL_FLOW, 1, "res_rho",0);
    if(res_rho > 0.0)  //有残差输出，由于残差输出不是每个时间步都有输出，因此需要判断
    {
        res_rho_max = MAX(res_rho_max, res_rho);
        if(inner_step >= 5)  //zhyb：至少迭代5步
        {
            if(log10(res_rho_max/res_rho) > n_order_shut)  //zhyb:子迭代下降的量级够了
            {
                return true;
            }
        }
    }

    return false;
}


void Simulation::PostSimu()
{
    IntType i;
    
    for(i=0; i<nZones; i++) {
        zones[i]->PostZone();
    }
    
#ifdef MPICH
    if(myZone==1){
        FILE *fp = 0;
        fp = fopen("status.run", "r");
        if(fp) {
            fclose(fp);
            fp = fopen("status.run", "w");
            fprintf(fp, "%d\n", 1);
            fclose(fp);
        }
    }
#else
    FILE *fp = 0;
    fp = fopen("status.run", "r");
    if(fp) {
        fclose(fp);
        fp = fopen("status.run", "w");
        fprintf(fp, "%d\n", 1);
        fclose(fp);
    }
#endif
}


void Simulation::Init()
{
    IntType i;
    
    CommGraph();
    CommGraph_node();

    for (IntType i = 0; i < nZones; ++i) {
        zones[i]->InitZone();
        zones[i]->TestReconstruction();
    }

    for(i=0; i<nZones; i++) {
        CreateSolversForZone(zones[i]);
        zones[i]->InitSolvers();
    }

    for(i=0; i<nZones; i++) {
        zones[i]->UpdateInterfaceData();
    }
}


/************************************************************************
*  generate the communication graph, i.e., for each zone, no of nb and 
*  the nbs
************************************************************************/
void Simulation::CommGraph()
{
    IntType i,j,nNeighbor,*nb;
    PolyGrid *grid;
    IntType *nbZ,nIFace;
    IntType maxz=0;
    IntType *count;

    // get the zone maximum, in case of MPICH
    
    for(i=0; i<nZones; i++) {

        grid = (PolyGrid *)zones[i]->GetGrid(0);      // get the fine grid
        if(grid == 0) return;
        nbZ = grid->GetnbZ();
        nIFace = grid->GetNIFace();
        for(j=0; j<nIFace; j++) {
            maxz = MAX(maxz,nbZ[j]);
        }
    }
    maxz++;
    
#ifndef MPICH
    assert(maxz == nZones);
#endif

    count = NULL;
    mfmem::snew_array_1D(count, maxz,dmrfl);

    for(i=0; i<nZones; i++) {
        for(j=0;j<maxz; j++) count[j]=0;

        grid = (PolyGrid *)zones[i]->GetGrid(0);        // get the fine grid
        nbZ = grid->GetnbZ();
        nIFace = grid->GetNIFace();
        for(j=0; j<nIFace; j++) {
            count[nbZ[j]] = 1;
        }

        nNeighbor=0;
        for(j=0; j<maxz; j++) {
            if(count[j]) nNeighbor++;
        }

        if(nNeighbor > 0) {
            nb = NULL;
            mfmem::snew_array_1D(nb, maxz,dmrfl);
            nNeighbor=0;
            for(j=0; j<maxz; j++) {
                if(count[j]) nb[nNeighbor++] = j;
            }
            // RFC added: reorder the nb so that nb[j] in the increasing order
            IntType nb_tmp, k;
            for(j=0; j<nNeighbor; j++) {
              for(k=j+1; k<nNeighbor; k++) {
                if(nb[j] > nb[k]) {
                  nb_tmp = nb[j];
                  nb[j]  = nb[k];
                  nb[k]  = nb_tmp;
                }
              }
            }
            zones[i]->SetnNeighbor(nNeighbor);
            zones[i]->Setnb(nb);
        }
    }
  
    mfmem::sdel_array_1D(count);
}


/************************************************************************
*  generate the communication graph, i.e., for each zone, no of nb and 
*  the nbs for the nodes of the interface
************************************************************************/
void Simulation::CommGraph_node()
{
    IntType i,j,nNeighborN,*nbN;
    PolyGrid *grid;
    IntType *nbZN,nINode;
    IntType maxz=0;
    IntType *count;

    // get the zone maximum, in case of MPICH
    
    for(i=0; i<nZones; i++) {
        grid = (PolyGrid *)zones[i]->GetGrid(0);      // get the fine grid
        if(grid == 0) return;
        nbZN = grid->GetnbZN();
        nINode = grid->GetNINode();
        for(j=0; j<nINode; j++) {
            maxz = MAX(maxz,nbZN[j]);
        }
    }
    maxz++;
    
#ifndef MPICH
    assert(maxz == nZones);
#endif

    count = NULL;
    mfmem::snew_array_1D(count, maxz,dmrfl);

    for(i=0; i<nZones; i++) {
        for(j=0;j<maxz; j++) count[j]=0;

        grid = (PolyGrid *)zones[i]->GetGrid(0);        // get the fine grid
        nbZN = grid->GetnbZN();
        nINode = grid->GetNINode();
        for(j=0; j<nINode; j++) {
            count[nbZN[j]] = 1;
        }

        nNeighborN=0;
        for(j=0; j<maxz; j++) {
            if(count[j]) nNeighborN++;
        }

        if(nNeighborN > 0) {
            nbN = NULL;
            mfmem::snew_array_1D(nbN, nNeighborN,dmrfl);
            nNeighborN=0;
            for(j=0; j<maxz; j++) {
                if(count[j]) nbN[nNeighborN++] = j;
            }
            // RFC added: reorder the nb so that nbN[j] in the increasing order
            IntType nbN_tmp, k;
            for(j=0; j<nNeighborN; j++) {
              for(k=j+1; k<nNeighborN; k++) {
                if(nbN[j] > nbN[k]) {
                  nbN_tmp = nbN[j];
                  nbN[j]  = nbN[k];
                  nbN[k]  = nbN_tmp;
                }
              }
            }
            zones[i]->SetnNeighborN(nNeighborN);
            zones[i]->SetnbN(nbN);
        }

#ifdef DEBUG
        mflog::log.set_all_processors_out();
#ifdef MPICH
        mflog::log << "Parallel node linked neighbor number for zone " << myZone << " is " << nNeighborN << std::endl;
#else
        mflog::log << "Node linked neighbor number for zone " << grid->GetZone() + 1 << " is " << nNeighborN << std::endl;
#endif
#endif
    }
  
    mfmem::sdel_array_1D(count);
}


/************************************************************************
*  Update the Zone parameters based on the input ones 
*  
************************************************************************/
void Simulation::CreateSolversForZone( Zone *zone )
{
    IntType nGrids = zone->GetNoOfGrids();
    if(nGrids <= 0) {
        printf("No grids available for solver !!!\n");
        return;
    }
    PolyGrid ** grids = (PolyGrid ** )zone->GetGrids();
    DataStore ** fields = zone->GetFields();
    DataSafe * zPara = zone->GetZonePara();
    BCond *bc = zone->GetZoneBc();
    // default to flow solver
    Solver *solver = new NSSolver(nGrids, grids, fields, zPara, bc, zone);
    zone->AddSolver(solver);

    // see if turbulence model
    IntType vis_mode=0;
    zone->GetData(&vis_mode, INT, 1, "vis_mode");

    if((vis_mode == INVISCID) || (vis_mode == LAMINAR))
    {
        return;
    }
    else if(vis_mode == S_A_MODEL)
    {
        Solver *solver;
        solver = new SASolver(nGrids, grids, fields, zPara, bc, zone);
        zone->AddSolver(solver);
    }
    else
    {
        printf("Other type solvers are needed to be developed !!!\n");
        return;
    }
}


/************************************************************************
*  Update the Zone parameters based on the input ones 
*  
************************************************************************/
void Simulation::UpdateParameter()
{
    for(IntType i=0; i<nZones; i++) 
    {
        Zone *zone = GetZone(i);
        zone->FixParameter();
        zone->UpdateParameter();
        zone->CheckParameter();
    }
    return;
}


/*******************************************************************************
*         Load grid information for all zones                                  *
* Note: ONLY mflow format are read for solvers
* For computation with a single core, read 
* For computation with multi-cores, read mmgrid*.in
* Update: 2020-01-10 tangj
*******************************************************************************/
void Simulation::LoadGridDataForZones()
{
    // determine the directory where grids locate
    vector<string> grid_dir_of_zone(nZones);
    GetInitialGridDir(grid_dir_of_zone);

    // Load grid
    ReLoadGridDataForZones(grid_dir_of_zone);
}


//
void Simulation::ReLoadGridDataForZones(vector<string> &grid_dir_of_zone)
{
    // Clear grid-related data in every zone before loading new grids
    ClearGridRelatedDataForAllZones();

    // Load grid
    LoadGrid(grid_dir_of_zone);
}


/*******************************************************************************
*         Load Grid for all zones                                              *
* Note: ONLY mflow format are read for solvers
* For computation with a single core, read 
* For computation with multi-cores, read mmgrid*.in
* Update: 2020-01-10 tangj
*******************************************************************************/
void Simulation::LoadGrid(vector<string> &grid_dir_of_zone)
{
    assert(grid_dir_of_zone.size() == nZones);

    // grids container for every zone
    vector<vector<PolyGrid *> > grids_container(nZones);

    // extra grid data container for every zone
    vector<GridIO::ExtraGridData> extra_grid_data_containor(nZones);

    // Read grid files
    for(IntType zn = 0; zn < nZones; ++zn) 
    {
        IntType mg_levels = 0;

        // User gets used to set the parameter 'mgrid' to 0 if they do not want
        // to use multi-grid method. But we need the level is 1 at least to read
        // a grid. 
        mg_levels = std::max(mg_levels, 1);

        grids_container[zn].resize(mg_levels);

        // allocate grid objects
        for (IntType g = 0; g < mg_levels; ++g)
        {
            PolyGrid *grid = NULL;
            mfmem::snew_object(grid, dmrfl);

            grid->SetLevel(g);

            grids_container[zn][g] = grid;
        }

        string &grid_dir = grid_dir_of_zone[zn];

        // Read parallel grid
        string grid_file;
       
#ifdef MPICH
        grid_file = grid_dir + "mmgrid" + int2str(myZone) + ".in";
#else
        grid_file = grid_dir + "serial_grid.mfl";
#endif
        GridIO::ReadMFlowGrid_binary(&(grids_container[zn][0]), mg_levels, grid_file, extra_grid_data_containor[zn]);
    }

    // Bind grids to zone and pass patch names into zone
    for(IntType zn = 0; zn < nZones; ++zn) 
    {
        Zone *current_zone = GetZone(zn);

        IntType mg_levels = 0;
        // User gets used to set the parameter 'mgrid' to 0 if they do not want
        // to use multi-grid method. But we need the level is 1 at least to read
        // a grid. 
        mg_levels = std::max(mg_levels, 1);

        for (IntType g = 0; g < mg_levels; ++g)
        {
            PolyGrid *current_grid = grids_container[zn][g];
            current_grid->SetZone(zn);

            // pass in grid to zone
            current_zone->AddGridAndField(current_grid);
        }

        // Set the patch name for zone
        IntType n_patchs = current_zone->GetNoBCRecord();
        GridIO::ExtraGridData::BCNamesType &bc_names = extra_grid_data_containor[zn].bc_patch_names;        

        // When one zone has been given grids and needs to re-read new grids in parallel mode,
        // the number of boundary patch(n_patchs) may be one larger than bc_names.size(). That
        // one patch is the parallel interface added by function SpecifyBC().
        // assert(n_patchs == bc_names.size() || n_patchs-1 == bc_names.size());
        if (!(n_patchs == bc_names.size() || n_patchs-1 == bc_names.size()))
        {
            mflog::log.set_each_grid_out();
            mflog::log << "The boundary conditions in input file don't match these in grid file."
                       << std::endl << "Please check input file!" << std::endl;
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }  

        for (IntType bc = 0; bc < bc_names.size(); ++bc)
        {
            BCRecord *bcr = current_zone->GetBCRecord(bc);
            bcr->SetPatchName(const_cast<char*>(bc_names[bc].c_str()));
        }
    }

}


/*******************************************************************************
*         Clear grids and fields in every zone                                 *
* Delete all grid related data in zone, and we can load new grids for zone
* 2020-01-10 tangj
*******************************************************************************/
void Simulation::ClearGridRelatedDataForAllZones()
{
    for(IntType zn = 0; zn < nZones; ++zn) 
    {
        Zone *zone = GetZone(zn);
        
        // Delete the solver, the old grid and the flow filed
        // The operations are similar with constructor of Zone here.
        zone->ClearAllSolvers(); 

        zone->ClearAllGridsAndFields();

        zone->ClearPartGridsToSingleGridInformation();

        zone->ClearOverlapZonesInformation();

        zone->ClearNeighborZonesInformation();
    }
}


// Update time accuracy for unsteady iteration
void Simulation::GetInitialGridDir(vector<string> &grid_dir_of_zone)
{
    // determine the directory where grids locate
    grid_dir_of_zone.resize(nZones);
    for (IntType zn = 0; zn < nZones; ++zn)
    {
        Zone *current_zone = GetZone(zn);

        String grid_dir_char;
        current_zone->GetData(&grid_dir_char, STRING, 1, "griddir");

        string grid_dir(grid_dir_char);

        grid_dir_of_zone[zn] = grid_dir;
    }
}


/*******************************************************************************
*         Update time accuracy for unsteady iteration                          *
* Set value of 'time_accuracy'
*******************************************************************************/
void Simulation::UpdateTimeAccuracy(const IntType time_step)
{
    for(IntType zn = 0; zn < nZones; ++zn) 
    {
        RealFlow time_accuracy = 0.0;
        if(time_step == 0) time_accuracy = -1.0;
        else if(time_step == 1) time_accuracy = 0.0;
        else if(time_step >= 2) time_accuracy = 0.5;
        zones[zn]->UpdateData(&time_accuracy, REAL_FLOW, 1, "time_accuracy");
    }
}

/*******************************************************************************
*        determine iteration steps for initial time step                       *
*******************************************************************************/
IntType Simulation::AssignInitalIterationSteps()
{
    // n_steps为定常（或第一个物理时间步）的子迭代步数
    // n_steps_unst为第二个及以后物理时间步的子迭代步数
    IntType n_steps, n_steps_unst;
    GetData(&n_steps,     INT, 1, "n_steps");
    GetData(&n_steps_unst,INT, 1, "n_steps_unst");

    IntType Unst_steps_Curt = 0;
    GetData(&Unst_steps_Curt, INT, 1, "Unst_steps_Curt");

    //zhyb: 用于续算
    IntType n_steps_coarse = 0, iter_done = 0, n_steps_Curt = 0;
    GetData(&n_steps_coarse, INT, 1, "n_steps_coarse");
    GetData(&iter_done,      INT, 1, "iter_done");
    if(iter_done)
    {
        n_steps_Curt = iter_done - n_steps_coarse;
        if(Unst_steps_Curt)
        {
            n_steps_Curt -= ((Unst_steps_Curt-1)*n_steps_unst+n_steps);
            if(n_steps_Curt<0 || n_steps_Curt>=n_steps_unst)
            {
                n_steps_Curt = 0;  //zhyb: 解决续算时增加或减少子迭代数出现的n_step_Curt为负或巨大，                          
            }                      //从而该步迭代数异常大或小的问题
        }
    }   

    return n_steps_Curt;
}


#undef CPP_FILD_ID  // clear out file id
} //~namespace mflow
