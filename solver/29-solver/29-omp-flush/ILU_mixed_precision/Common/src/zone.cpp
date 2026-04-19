//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   zone.cpp
/// \brief  A class of Zone
/// \author 
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 2020-07-21  tangj      Move functions about flowfield output to class Translation
/// </pre>

// direct head file
#include "zone.h"

// C++ build-in head files
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <numeric>
using namespace std;

// user defined head files
#include "constant.h"
#include "utility_functions.h"
#include "solver_ns.h"
#include "solver_turb_sa.h"
#include "algm.h"
#include "io_log.h"
#include "io_base_format.h"
#include "io_log.h"
#include "parallel_base_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"

#if (defined FS_SIMD) && (defined Tile)
#include "csr_loader.h"
#include "ds.h"
#include "tiling.h"
#include <load_tile_from_file.h>
#include <vector>
#endif

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
extern double  time_SA;
#endif
//TIMECOST

namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 12601  // define file id


#ifdef MPICH
extern int myZone;
extern int numprocs;
extern MPI_Comm GridComm;  //for each grid, tangj
#endif

#define ADF_NAME_LENGTH  32


Zone::Zone(IntType znin)
{
    zn              = znin;
    nGrids          = 0;
    grids           = NULL;
    fields          = NULL;
    nSolvers        = 0;
    solvers         = NULL;
    nNeighbor       = 0;
    nNeighborN      = 0;
    nb              = NULL;
    nbN             = NULL;
    nOverZone       = 0;
    nCh             = NULL; 
    OverZoneMark    = NULL; 

    NumZone         = 0;
    NumCell         = NULL;
    partnerZone     = NULL;
    partnerCell     = NULL;
}


Zone::Zone()
{
    zn              = 0;
    nGrids          = 0;
    grids           = NULL;
    fields          = NULL;
    nSolvers        = 0;
    solvers         = NULL;
    nNeighbor       = 0;
    nNeighborN      = 0;
    nb              = NULL;
    nbN             = NULL;
    nOverZone       = 0;
    nCh             = NULL; 
    OverZoneMark    = NULL; 

    NumZone         = 0;
    NumCell         = NULL;
    partnerZone     = NULL;
    partnerCell     = NULL;    
}


// delete all solvers in the zone
void Zone::ClearAllSolvers(void)
{
    if (nSolvers == 0)
    {
        return;
    }

    for(IntType s = 0; s < nSolvers; ++s) 
    {
        //mfmesh::sdel_object(solvers[g]);
        delete solvers[s];
    }
    mfmem::sdel_array_1D(solvers);

    // now no solver exists in the zone
    nSolvers = 0; 
}


// delete all grids in the zone
void Zone::ClearAllGridsAndFields(void)
{
    if (nGrids == 0)
    {
        return;
    }

    for(IntType g = 0; g < nGrids; ++g) 
    {       
        mfmem::sdel_object(grids[g]);
        mfmem::sdel_object(fields[g]);
    }
    mfmem::sdel_array_1D(grids);
    mfmem::sdel_array_1D(fields);

    // now no grid exists in the zone
    nGrids = 0;
}


// delete the mapping information from parallel multi-parts grids to a single grid
void Zone::ClearPartGridsToSingleGridInformation(void)
{
    if (NumZone == 0)
    {
        return;
    }

    // c2c information
    mfmem::sdel_array_1D(NumCell);
    mfmem::sdel_array_1D(partnerZone);
    mfmem::sdel_array_1D(partnerCell);

    NumZone = 0;
}


// Delete neighbors information of zones 
void Zone::ClearNeighborZonesInformation(void)
{
    // The ID of Zones which share grid face with this zone
    if (nNeighbor != 0)
    {
        mfmem::sdel_array_1D(nb);
        nNeighbor = 0;
    }
    
    // The ID of Zones which share grid points with this zone
    if (nNeighborN != 0)
    {
        mfmem::sdel_array_1D(nbN);
        nNeighborN = 0;
    }    
}


// Delete overlap information of zones
void Zone::ClearOverlapZonesInformation(void)
{
    if (nOverZone == 0)
    {
        return;
    }
    // The ID of Zones/processor which hold other grids
    mfmem::sdel_array_1D(nCh);
    mfmem::sdel_array_1D(OverZoneMark);

    nOverZone = 0;
}


// deconstructor
Zone::~Zone()
{
    ClearAllSolvers();
    
    ClearAllGridsAndFields();
    
    ClearPartGridsToSingleGridInformation();

    ClearOverlapZonesInformation();

    ClearNeighborZonesInformation();
}


/************************************************************************
* add a new grid to a zone, mean while, add the field too               *
************************************************************************/
void Zone::AddGridAndField(Grid *grid)
{
    DataStore *field;
    
    if(nGrids == 0) {
        grids  = NULL;
        fields = NULL;
        mfmem::snew_array_1D(grids , nGrids+1,dmrfl);
        mfmem::snew_array_1D(fields, nGrids+1,dmrfl);
        
        grids[nGrids] = grid;
        
        field = NULL;
        mfmem::snew_object(field, dmrfl);
        fields[nGrids] = field;
        grid->CopyFieldFrom(field);
        grid->CopyDataFrom(&zPara);
        nGrids++;
        
    } else {
        Grid **gridt;
        DataStore **fieldt;
        
        gridt = grids;
        fieldt = fields;
        grids  = NULL;
        fields = NULL;
        mfmem::snew_array_1D(grids , nGrids+1,dmrfl);
        mfmem::snew_array_1D(fields, nGrids+1,dmrfl);
        
        for(IntType i=0; i<nGrids; i++) {
            grids[i] = gridt[i];
            fields[i] = fieldt[i];
        }
        mfmem::sdel_array_1D(gridt);
        mfmem::sdel_array_1D(fieldt);
        
        grids[nGrids] = grid;
        
        field = NULL;
        mfmem::snew_object(field,dmrfl);
        fields[nGrids] = field;
        grid->CopyFieldFrom(field);
        grid->CopyDataFrom(&zPara);
        nGrids++;
    }
}


/************************************************************************
* Assign boundary conditions                                            *
* Before this, c2 is negative, indicating the patch number              *
* After this, c2 = b_face_no + n_cells                                  *
************************************************************************/
void Zone::SpecifyBC()
{
    IntType  j, k, c2, n_patch;    
    IntType  *f2c, nBFace, nTCell;
    PolyGrid *grid;

    n_patch = GetNoBCRecord();
    UpdateData(&n_patch, INT, 1, "n_patch");
  
    if(nNeighbor > 0) { //zhyb: multi-zone
        IntType bcr_int;
        // there must be some interface bcs, find the BCRecord
        BCRecord *inter=0,*bcr;
        for(j=0; j<n_patch; j++) //zhyb: find the interface bcr
        {
            bcr = GetBCRecord(j);
            if(bcr->GetType() == INTERFACE) {
                inter = bcr;
                bcr_int = inter->GetPatchID();
                break;
            }
        }
        if(inter == 0) {    //zhyb: if not have ,then add it
            bcr_int = ++n_patch;
            mfmem::snew_object(inter,dmrfl);
            inter->SetType(INTERFACE);
            inter->SetTypeSymbol("Interface");
            inter->SetPatchID(bcr_int);
            AddBCRecord(inter);
        }
        IntType nIFace;
        for(j=0; j<nGrids; j++) 
        {
            grid = (PolyGrid *) GetGrid(j);
            f2c = grid->Getf2c();
            nBFace = grid->GetNBFace();
            nIFace = grid->GetNIFace();
            for(k=nBFace-nIFace; k<nBFace; k++) {
                f2c[k+k+1] = -bcr_int;
            }
        }  
    }

    for(j=0; j<nGrids; j++) {
        grid = (PolyGrid *) GetGrid(j);
        nBFace = grid->GetNBFace();
        nTCell = grid->GetNTCell();
        f2c = grid->Getf2c();

        BCRecord **bcrs = NULL;
        mfmem::snew_array_1D(bcrs,nBFace,dmrfl);
        grid->Setbcr(bcrs);

        for(k=0; k<nBFace; k++) {
            c2 = -f2c[k*2+1];
            if(c2 <= 0) {
                assert(0);
            }
            if(c2>n_patch) {
                assert(0);
                bcrs[k] = GetBCRecord(n_patch-1);
            } else {
                bcrs[k] = GetBCRecord(c2-1);          
            }
            f2c[k*2+1] = k + nTCell;
        }
    }
//  bc.ListBCRecord();
}


/************************************************************************
*  solver the Zone for the solver
************************************************************************/
void Zone::SolveZone()
{
    for(IntType j=0; j<nSolvers; j++) {
        solvers[j]->Solve();        
    }
} 


/************************************************************************
*  Initial the Zone for the solver
************************************************************************/
void Zone::InitZone()
{
#ifdef MPICH
    if(myZone == 1){ 
#endif
        zPara.ListAllData();
#ifdef MPICH
    }
#endif
    
    IntType i;
    PolyGrid *pgrid;
  
    // zone related initializations
    SpecifyBC();        // now, bcs are converted to grid-based
#ifdef DEBUG
    mflog::log.set_one_processor_out();
    mflog::log << "Exit SpecifyBC" << std::endl;
#endif

    SpecifyNeighbors(); // put neighbor information into each grid
#ifdef DEBUG
    mflog::log.set_one_processor_out();
    mflog::log << "Exit SpecifyNeighbors" << std::endl;
#endif
  
#ifdef MPICH
    for(i=0; i<nGrids; i++) {
        pgrid = (PolyGrid *)grids[i];
        pgrid->SetUpComm();
        pgrid->SetUpComm_Node();
    }
#endif

    // Initialize Grid metrics
#ifdef DEBUG
    mflog::log.set_one_processor_out();
    mflog::log << "nGrids = " << nGrids << std::endl;
#endif
  
    IntType steady=1, vis_mode=0;
    GetData(&steady, INT, 1, "steady");
    GetData(&vis_mode, INT, 1, "vis_mode");
    for(i=0; i<nGrids; i++) {
        pgrid = (PolyGrid *)grids[i];
        
        //计算网格的几何量：面心、面积、面单位法向矢量、体心、体积
        pgrid->ComputeMetrics();
        
        pgrid->AdditionalInfoForGeometry();
    
        //初始化非定场计算的面心速度
        if(!steady){
            pgrid->InitialVgn();
        }
 
        //Compute the distance to wall for turbulence model
        if(i>=0 && vis_mode==S_A_MODEL){ //i==0
            pgrid->ComputeCellDist();
        }
        
#ifdef MPICH
        MPI_Barrier(MPI_COMM_WORLD);
#endif    
    }
    
    //体心体积传值
    CommCellCenterData(); 

    for(i=0; i<nGrids; i++){
        ((PolyGrid *)grids[i])->FindCellLayerNo();
    }
    
    //find order for lusgs
    ReorderCellforLUSGS();
    //find cell color for lusgs
    //0 or 1 to be chosen in param where 0 is imbalance greedy algorithm,1 is balance algorithm
#if (defined FS_SIMD) && (defined FS_SIMD_AVX) && (defined FaceColoring)
    //for FaceColoring SIMD, based SSE, ruitian, 2021.12.27
    for (IntType k = 0; k < nGrids; k++) {
        pgrid = (PolyGrid*)grids[k];
        IntType* f2c = pgrid->Getf2c();
        RealGeom   *area  = pgrid->GetFaceArea();
        RealGeom   *xfn   = pgrid->GetXfn();
        RealGeom   *yfn   = pgrid->GetYfn();
        RealGeom   *zfn   = pgrid->GetZfn();
        IntType    nTFace = pgrid->GetNTFace();
        IntType    nBFace = pgrid->GetNBFace();
        IntType    nTCell = pgrid->GetNTCell();
        pgrid->xfntile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nTFace, 64);
        pgrid->yfntile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nTFace, 64);
        pgrid->zfntile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nTFace, 64);
        pgrid->areatile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nTFace, 64);
        pgrid->qsumtile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nTFace, 64);

        pgrid->dqdxtile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nTCell, 64);
        pgrid->dqdytile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nTCell, 64);
        pgrid->dqdztile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nTCell, 64);

        pgrid->f2c1 = (IntType*)_mm_malloc(sizeof(IntType) * nTFace, 64);
        pgrid->f2c2 = (IntType*)_mm_malloc(sizeof(IntType) * nTFace, 64);

        //transform f2c into f2c1 and f2c2:
        for (IntType i = 0; i < nTFace; i++) {
            IntType count = 2 * i;
            pgrid->f2c1[i] = f2c[count];
            pgrid->f2c2[i] = f2c[count + 1];
        }
        for (IntType i = 0; i < nTFace; i++) {
            pgrid->xfntile[i] = xfn[i];
            pgrid->yfntile[i] = yfn[i];
            pgrid->zfntile[i] = zfn[i];
            pgrid->areatile[i] = area[i];
        }
    }
#elif (defined FS_SIMD) && (defined Tile)
    for (IntType k = 0; k < nGrids; k++) {
        pgrid = (PolyGrid*)grids[k];
        IntType* f2c = pgrid->Getf2c();
        IntType    nTFace = pgrid->GetNTFace();
        IntType    nBFace = pgrid->GetNBFace();
        IntType    nTCell = pgrid->GetNTCell();
        //for tiling:
        //2.3 update f2c,nNPF,f2n for boundary faces
        IntType ifacenum = nTFace - nBFace;
        IntType tilenum = 2 * ifacenum;
        IntType* f2c_backup = NULL;
        mfmem::snew_array_1D(f2c_backup, tilenum, dmrfl);

        for (IntType j = nBFace; j < nTFace; j++)
        {
            f2c_backup[2 * (j - nBFace)] = f2c[2 * j];
            f2c_backup[2 * (j - nBFace) + 1] = f2c[2 * j + 1];
        }

        MyCsr<int>* inters = CsrLoader<int>::Load(f2c_backup, ifacenum, nTCell, nBFace);

        IntType tilesize = 1024;
        //////////////// Tile the matrix. ///////////////
        // First extract the dense tiles.
        vector<Coo<int> >* peel_dense = Tiling<int>::Peel(inters, tilesize, 0);

        cout << "Peel done." << endl;
        // Make sure that inter-thread (inter-tile) execution has no conflict. 
        vector<vector<Coo<int> > >* diagonal_pack_tiles =
            Tiling<int>::KillTilesConflicts(peel_dense);
        cout << "Kill tile conflict done." << endl;

        // Make sure that the inter-lane (inter-nnz) execution in each thread has no conflict. 
        vector<vector<vector<Coo<int> > > >* multi_thread_tiles =
            RemoveConflictAndPack(diagonal_pack_tiles);
        cout << "Remove SIMD conflict done." << endl;

        // Tile the remaining tiles, using a twice larger tile size.
        // TODO(): take care of the tiling size.
        // How to gurantee the inters to be totally tiled?
        vector<Coo<int> >* peel_sparse = Tiling<int>::Peel(inters, tilesize, 0);
        // Make sure that inter-thread (inter-tile) execution has no conflict. 
        vector<vector<Coo<int> > >* diagonal_pack_tiles_sparse =
            Tiling<int>::KillTilesConflicts(peel_sparse);
        // Make sure that the inter-lane (inter-nnz) execution in each thread has no
        // conflict. 
        vector<vector<vector<Coo<int> > > >* multi_thread_tiles_sparse =
            RemoveConflictAndPack(diagonal_pack_tiles_sparse);
        // Combine the dense and sparse tiles.
        for (int i = 0; i < multi_thread_tiles_sparse->size(); ++i) {
            multi_thread_tiles->push_back((*multi_thread_tiles_sparse)[i]);
        }
        CountSimdRate(*multi_thread_tiles);
        // Combine the tiles (before removing lane conflicts)
        for (int i = 0; i < diagonal_pack_tiles_sparse->size(); ++i) {
            diagonal_pack_tiles->push_back((*diagonal_pack_tiles_sparse)[i]);
        }
        cout << "Sync steps: " << diagonal_pack_tiles->size() << endl;
        cout << "Max parallelism: " << ComputeMaxParallelism(diagonal_pack_tiles) << endl;
        cout << "Min parallelism: " << ComputeMinParallelism(diagonal_pack_tiles) << endl;
        cout << "Average parallelism: " << ComputeAverageParallelism(diagonal_pack_tiles) << endl;

        // File containing delemiters for separating tiles/multi-thread packs.
#ifdef MPICH
        IntType mpirank;
        MPI_Comm_rank( MPI_COMM_WORLD, & mpirank);
        string rankid = to_string(mpirank);

        string offset_file_name = "m6.offset"+rankid;
#else
        string offset_file_name = "m6.offset";
#endif
        // Write to file.
        ofstream offset_output(offset_file_name.c_str());

        // Then, write offsets.
        WriteOffsets(*multi_thread_tiles, offset_output);

        offset_output.close();

        cout << "Peel Done." << endl << endl;
        /*
        // File containing non-zeros before inter-lane conflict removal (for seq
        // execution).
        string seq_nnz_file_name = "m6.seq.tiling";
        ofstream seq_nnz_seq_output(seq_nnz_file_name.c_str());
        // Then, write seq tiling file.
        seq_nnz_seq_output << *diagonal_pack_tiles;
        seq_nnz_seq_output.close();
        // serial tile load：
        string nnzfile = "m6.seq.tiling";
        // Load NNZs. 
        PaddedNnz<int>* nnzs = nullptr;
        LoadTileFromFile(nnzfile, nnzs);
        cout << "Total NNZ for iface: " << nnzs->nnz << endl;
        pgrid->ifacerow = (int*)_mm_malloc(sizeof(int) * nnzs->nnz, 64);
        pgrid->ifacecol = (int*)_mm_malloc(sizeof(int) * nnzs->nnz, 64);
        pgrid->ifaceval = (int*)_mm_malloc(sizeof(int) * nnzs->nnz, 64);
        for (IntType ii = 0; ii < nnzs->nnz; ii++) {
            pgrid->ifacerow[ii] = nnzs->rows[ii];
            pgrid->ifacecol[ii] = nnzs->cols[ii];
            pgrid->ifaceval[ii] = nnzs->vals[ii];
        }
        pgrid->ifacennz = nnzs->nnz;
        */
        // Load offsets. 
#ifdef MPICH
        string offsetsfile = "m6.offset"+rankid;
#else
        string offsetsfile = "m6.offset";
#endif
        IntType temnnz = 0;
        LoadTileFromFile(offsetsfile, pgrid->ioffsets, &temnnz);
        
        pgrid->iSIMDnnz = temnnz;
        pgrid->iSIMDrow = (int*)_mm_malloc(sizeof(int) * pgrid->iSIMDnnz, 64);
        pgrid->iSIMDcol = (int*)_mm_malloc(sizeof(int) * pgrid->iSIMDnnz, 64);
        pgrid->iSIMDval = (int*)_mm_malloc(sizeof(int) * pgrid->iSIMDnnz, 64);
        pgrid->ifacezero = (int*)_mm_malloc(sizeof(int) * pgrid->iSIMDnnz, 64);

        IntType icount = 0;
        for (int i = 0; i < (*multi_thread_tiles).size(); ++i) {
            for (int j = 0; j < (*multi_thread_tiles)[i].size(); ++j) {
                for (int k = 0; k < (*multi_thread_tiles)[i][j].size(); ++k) {
                    for (int m = 0; m < (*multi_thread_tiles)[i][j][k].nnz; ++m) {
                        pgrid->iSIMDrow[icount] = (*multi_thread_tiles)[i][j][k].rows[m];
                        pgrid->iSIMDcol[icount] = (*multi_thread_tiles)[i][j][k].cols[m];
                        pgrid->iSIMDval[icount] = (*multi_thread_tiles)[i][j][k].vals[m];
                        pgrid->ifacezero[icount] = (*multi_thread_tiles)[i][j][k].facevals[m];
                        ++icount;
                    }
                }
            }
        }
        //ruitianSIMD
        pgrid->qsumtile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nTFace, 64);

        pgrid->dqdxtile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nTCell, 64);
        pgrid->dqdytile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nTCell, 64);
        pgrid->dqdztile = (RealGeom*)_mm_malloc(sizeof(RealGeom) * nTCell, 64);

        pgrid->xfnt = (RealGeom*)_mm_malloc(sizeof(RealGeom) * pgrid->iSIMDnnz, 64);
        pgrid->yfnt = (RealGeom*)_mm_malloc(sizeof(RealGeom) * pgrid->iSIMDnnz, 64);
        pgrid->zfnt = (RealGeom*)_mm_malloc(sizeof(RealGeom) * pgrid->iSIMDnnz, 64);
        pgrid->areat = (RealGeom*)_mm_malloc(sizeof(RealGeom) * pgrid->iSIMDnnz, 64);
        pgrid->qsumt = (RealGeom*)_mm_malloc(sizeof(RealGeom) * pgrid->iSIMDnnz, 64);
        pgrid->nNPFt = (IntType*)_mm_malloc(sizeof(IntType) * pgrid->iSIMDnnz, 64);

        RealGeom* area = pgrid->GetFaceArea();
        RealGeom* xfn = pgrid->GetXfn();
        RealGeom* yfn = pgrid->GetYfn();
        RealGeom* zfn = pgrid->GetZfn();
        IntType* nNPF = pgrid->GetnNPF();

        for (IntType ii = 0; ii < pgrid->iSIMDnnz; ii++) {
            IntType i = pgrid->iSIMDval[ii];
            pgrid->xfnt[ii] = xfn[i];
            pgrid->yfnt[ii] = yfn[i];
            pgrid->zfnt[ii] = zfn[i];
            pgrid->areat[ii] = area[i];
            pgrid->nNPFt[ii] = nNPF[i];
        }

        FreeTileVector(peel_dense);
        FreeTileVector(peel_sparse);
        FreeTileVector(multi_thread_tiles);
        mfmem::sdel_array_1D(f2c_backup);

        delete peel_dense;
        delete diagonal_pack_tiles;
        delete multi_thread_tiles;
        delete peel_sparse;
        delete diagonal_pack_tiles_sparse;
        delete multi_thread_tiles_sparse;
    }
    //add by ruitian
#endif 
}


void Zone::InitZoneTranGrid()
{
    zPara.ListAllData();

    SpecifyBC();        // now, bcs are converted to grid-based

    PolyGrid *grid; 
    grid = (PolyGrid *) grids[0];
    grid->WriteInfoDist(); 

    for(IntType i=0; i<nGrids; i++) {
        grids[i]->ComputeMetrics();
    }
}

void Zone::InitSolvers()
{
    for(IntType i=0; i<nSolvers; i++)
    {
        solvers[i]->Init();
    }
}


/************************************************************************
*  Break the Zone programme
************************************************************************/
IntType Zone::BreakZone()
{
    IntType run = 1;
    FILE *fp = 0;

#ifdef MPICH
    if(myZone==1){
        fp = fopen("status.run", "r");
        if(fp) {
            fscanf(fp, "%d\n", &run);
            fclose(fp);
            fp = 0;
        } else {
            run = 1;
        }
    }
    MPI_Bcast(&run, 1, MPIIntType, 0, MPI_COMM_WORLD);
#else
    fp = fopen("status.run", "r");
    if(fp) {
        fscanf(fp, "%d\n", &run);
        fclose(fp);
        fp = 0;
    } else {
        run = 1;
    }
#endif

    return run;
}


void Zone::output_info_c2c( IntType *CellToZone, IntType npart, IntType mg )
{
    PolyGrid *grid = (PolyGrid *) grids[mg];
    IntType nTCell = grid->GetNTCell();
    IntType *Cell2Zone = grid->GetCell2Zone();

    FILE *fp = NULL;
    fp = fopen("c2c.dat","wb");


    fwrite(&nTCell,  sizeof(IntType), 1, fp);
    fwrite(&npart,    sizeof(IntType), 1, fp);

    IntType *NumCell     = NULL;
    IntType *partnerZone = NULL;   //zone number of each cell
    IntType *partnerCell = NULL;  //cell number in partitioned zone of cell
    mfmem::snew_array_1D(NumCell    ,npart,dmrfl);
    mfmem::snew_array_1D(partnerZone,nTCell,dmrfl);
    mfmem::snew_array_1D(partnerCell,nTCell,dmrfl);
    for(IntType i=0; i<npart; i++) NumCell[i] = 0;
    for(IntType i=0; i<nTCell; i++) {
        partnerZone[i] = CellToZone[i];
        partnerCell[i] = NumCell[CellToZone[i]]++;
    }
    fwrite(NumCell, sizeof(IntType), npart, fp);
    fwrite(partnerZone, sizeof(IntType), nTCell, fp);
    fwrite(partnerCell, sizeof(IntType), nTCell, fp);
    if(Cell2Zone) fwrite(Cell2Zone, sizeof(IntType), nTCell, fp);
    fclose(fp);
    mfmem::sdel_array_1D(NumCell);
    mfmem::sdel_array_1D(partnerZone);
    mfmem::sdel_array_1D(partnerCell);
}

void Zone::GetFineGrids(IntType *CellToZone, PolyGrid **z_Grids, IntType n_part, IntType mg)
{
    PolyGrid *grid = (PolyGrid *) grids[mg];

    IntType   nTNode = grid->GetNTNode();
    IntType   nTFace = grid->GetNTFace();
    IntType   nTCell = grid->GetNTCell();

    IntType *fc2pc  = NULL;     //原始细网格cell所对应的分区上cell序号
    IntType **pc2fc = NULL;      //分区上cell所对应的原始细网格cell序号
    mfmem::snew_array_1D(fc2pc,nTCell,dmrfl);
    mfmem::snew_array_1D(pc2fc,n_part,dmrfl);
    for(IntType i=0; i<nTCell; i++) fc2pc[i] = -1;
    GetFineGrids_fc2pc(CellToZone, z_Grids, n_part, fc2pc, pc2fc, mg);

    IntType **ff2pf = NULL;  //原始细网格face所对应的分区上face序号
    IntType **pf2ff = NULL;       //分区上face所对应的原始细网格face序号    
    mfmem::snew_array_2D(ff2pf,4,nTFace,dmrfl,true);
    mfmem::snew_array_1D(pf2ff,n_part,dmrfl);
    for(IntType i=1; i<4*nTFace; i++) ff2pf[0][i] = -1;
    GetFineGrids_ff2pf(CellToZone, z_Grids, n_part, ff2pf, pf2ff, mg);  

    IntType **pn2fn = NULL;   //分区上node所对应的原始细网格node序号
    mfmem::snew_array_1D(pn2fn,n_part,dmrfl);
    GetFineGrids_pn2fn(z_Grids, n_part, pf2ff, pn2fn, mg);

    // get the local grid infomation.
    GetFineGrids_coord(grid, z_Grids, n_part, pn2fn);
    GetFineGrids_f2c(grid, z_Grids, n_part, pf2ff, fc2pc, CellToZone);
    GetFineGrids_F2N(nTNode, z_Grids, n_part, pn2fn);

    // get the parallel infomation for local grid
    GetFineGrids_nbBF(CellToZone, z_Grids, n_part, ff2pf, mg);
    GetFineGrids_nbRN(z_Grids, n_part, pn2fn, mg);


    mfmem::sdel_array_1D(fc2pc);
    mfmem::sdel_array_2D(ff2pf);
    for(IntType i=0; i<n_part; i++) {
        mfmem::sdel_array_1D(pc2fc[i]);
        mfmem::sdel_array_1D(pf2ff[i]);
        mfmem::sdel_array_1D(pn2fn[i]);
    }
    mfmem::sdel_array_1D(pc2fc);
    mfmem::sdel_array_1D(pf2ff);
    mfmem::sdel_array_1D(pn2fn);

}
void Zone::GetFineGrids_ff2pf(IntType *CellToZone, PolyGrid **z_Grids, IntType n_part,
                              IntType **ff2pf, IntType **pf2ff, IntType mg)
{
    PolyGrid *grid  = (PolyGrid *) grids[mg];
    IntType *f2c    = grid->Getf2c();
    IntType  nTFace = grid->GetNTFace();
    IntType  nBFace = grid->GetNBFace();

    IntType *nTF = NULL;
    IntType *nBF = NULL;
    IntType *nIF = NULL;
    mfmem::snew_array_1D(nTF,n_part,dmrfl);
    mfmem::snew_array_1D(nBF,n_part,dmrfl);
    mfmem::snew_array_1D(nIF,n_part,dmrfl);
    for(IntType i=0; i<n_part; i++) {
        nTF[i] = 0;     //分区的总面数
        nBF[i] = 0;     //分区的边界面数
        nIF[i] = 0;     //分区的并行边界面数
    }

    IntType count = 0;
    for(IntType i=0; i<nBFace; i++) {
        IntType c1, p1;
        c1 = f2c[count++];
        count++;
        p1 = CellToZone[c1];
        ff2pf[0][i] = p1;
        ff2pf[1][i] = nBF[p1]++;
    }
    for(IntType i=nBFace; i<nTFace; i++) {
        IntType c1, c2, p1, p2;
        c1 = f2c[count++];
        c2 = f2c[count++];
        p1 = CellToZone[c1];
        p2 = CellToZone[c2];
        if(p2==p1) {
            ff2pf[0][i] = p1;
            ff2pf[1][i] = -101;         //分区内部face
            nTF[p1]++;
        } else {
            ff2pf[0][i] = p1;         //分区并行face
            ff2pf[2][i] = p2;
            ff2pf[1][i] = nBF[p1]++;      //设置该面在p1分区中的序号
            ff2pf[3][i] = nBF[p2]++;
            //ff2pf[1][i] = -102;         //分区并行face
            //ff2pf[3][i] = -102;
            nIF[p1]++;
            nIF[p2]++;
        }
    }

    for(IntType i=0; i<n_part; i++) {
        nTF[i] += nBF[i];     //面的总数=内部面+边界面
        z_Grids[i]->SetNIFace(nIF[i]);
        z_Grids[i]->SetNBFace(nBF[i]);
        z_Grids[i]->SetNTFace(nTF[i]);
    }

    for(IntType i=nBFace; i<nTFace; i++) {
        if(ff2pf[1][i] == -101) {
            IntType p = ff2pf[0][i];
            ff2pf[1][i] = nBF[p]++;     //设置内部面在p分区中的序号
        }
    }

    for(IntType i=0; i<n_part; i++){
        pf2ff[i] = NULL;
        mfmem::snew_array_1D(pf2ff[i],nTF[i],dmrfl);
    }

    for(IntType i=0; i<nTFace; i++) {
        IntType p = ff2pf[0][i];
        IntType j = ff2pf[1][i];
        pf2ff[p][j] = i;
        p = ff2pf[2][i];
        if(p<0) continue;
        j = ff2pf[3][i];
        pf2ff[p][j] = i;
    }

    mfmem::sdel_array_1D(nBF);
    mfmem::sdel_array_1D(nIF);
    mfmem::sdel_array_1D(nTF);
}

void Zone::GetFineGrids_fc2pc(IntType *CellToZone, PolyGrid **z_Grids, IntType n_part, IntType *fc2pc, IntType **pc2fc, IntType mg)
{
    PolyGrid *grid = (PolyGrid *) grids[mg];

    IntType nTCell = grid->GetNTCell();

    IntType *nTC = NULL;
    mfmem::snew_array_1D(nTC,n_part,dmrfl);
    for(IntType i=0; i<n_part; i++) {
        nTC[i] = 0;
    }

    for(IntType i=0; i<nTCell; i++) {
        IntType p = CellToZone[i];
        fc2pc[i] = nTC[p]++;
    }

    for(IntType p=0; p<n_part; p++){
        pc2fc[p] = NULL;
        mfmem::snew_array_1D(pc2fc[p],nTC[p],dmrfl);
    }
    for(IntType i=0; i<nTCell; i++) {
        IntType p = CellToZone[i];
        IntType j = fc2pc[i];
        pc2fc[p][j] = i;
    }

    for(IntType p=0; p<n_part; p++)
        z_Grids[p]->SetNTCell(nTC[p]);

    mfmem::sdel_array_1D(nTC);
}

void Zone::GetFineGrids_pn2fn(PolyGrid **z_Grids, IntType n_part, IntType **pf2ff, IntType **pn2fn, IntType mg)
{
    PolyGrid *grid = (PolyGrid *) grids[mg];
    IntType  nTNode = grid->GetNTNode();
    IntType *nNPF   = grid->GetnNPF();
    IntType **F2N   = CalF2N(grid);

    IntType *tmm = NULL;
    IntType *nTN = NULL;
    mfmem::snew_array_1D(tmm,nTNode,dmrfl);
    mfmem::snew_array_1D(nTN,n_part,dmrfl);    
    for(IntType i=0; i<n_part; i++) {
        for(IntType j=0; j<nTNode; j++) tmm[j] = -1;  // bottle neck for large 'n_part'

        PolyGrid *pgrid = z_Grids[i];
        IntType   pnTFace = pgrid->GetNTFace();
        IntType  *pnNPF   = NULL;
        IntType **pF2N    = NULL;
        mfmem::snew_array_1D(pnNPF,pnTFace,dmrfl);
        for(IntType j=0; j<pnTFace; j++) {
            IntType ff = pf2ff[i][j];
            pnNPF[j] = nNPF[ff];
        }

        mfmem::snew_array_2D(pF2N ,pnTFace, pnNPF, dmrfl, true);

        for(IntType j=0; j<pnTFace; j++) {
            IntType ff = pf2ff[i][j];
            for(IntType k=0; k<pnNPF[j]; k++) {
                IntType node = F2N[ff][k];
                pF2N[j][k] = node;
                tmm[node] = 1;
            }
        }
        pgrid->SetnNPF(pnNPF);
        pgrid->Setf2n(pF2N[0]); // F2N is saved as the reference to f2n, and deleted with
        pgrid->SetF2N(pF2N);    // 1D operator, so F2N[0] will not be deleted. Here we pass
        // F2N[0] to f2n and so F2N[0] can be deleted with f2n 
        // deletion operator. tangj

        nTN[i]=0;
        for(IntType j=0; j<nTNode; j++) 
            if(tmm[j]>-1) nTN[i]++;
        z_Grids[i]->SetNTNode(nTN[i]);

        pn2fn[i] = NULL;
        mfmem::snew_array_1D(pn2fn[i],nTN[i],dmrfl);
        nTN[i]=0;
        for(IntType j=0; j<nTNode; j++)
            if(tmm[j]>-1) pn2fn[i][nTN[i]++]=j;
    }

    mfmem::sdel_array_1D(nTN);
    mfmem::sdel_array_1D(tmm);
}


void Zone::GetFineGrids_coord(PolyGrid *grid, PolyGrid **z_Grids, IntType n_part, IntType **pn2fn)
{
    RealGeom *x = grid->GetX();
    RealGeom *y = grid->GetY();
    RealGeom *z = grid->GetZ();

    RealGeom **px = NULL;
    RealGeom **py = NULL;
    RealGeom **pz = NULL;
    mfmem::snew_array_1D(px,n_part,dmrfl);
    mfmem::snew_array_1D(py,n_part,dmrfl);
    mfmem::snew_array_1D(pz,n_part,dmrfl);
    for(IntType i=0; i<n_part; i++) {
        IntType pnTNode = z_Grids[i]->GetNTNode();
        px[i] = NULL;
        py[i] = NULL;
        pz[i] = NULL;
        mfmem::snew_array_1D(px[i],pnTNode,dmrfl);
        mfmem::snew_array_1D(py[i],pnTNode,dmrfl);
        mfmem::snew_array_1D(pz[i],pnTNode,dmrfl);
        for(IntType j=0; j<pnTNode; j++) {
            px[i][j] = x[pn2fn[i][j]];
            py[i][j] = y[pn2fn[i][j]];
            pz[i][j] = z[pn2fn[i][j]];
        }
        z_Grids[i]->SetX(px[i]);
        z_Grids[i]->SetY(py[i]);
        z_Grids[i]->SetZ(pz[i]);
    }
    mfmem::sdel_array_1D(px);
    mfmem::sdel_array_1D(py);
    mfmem::sdel_array_1D(pz);
}

void Zone::GetFineGrids_f2c(PolyGrid *grid, PolyGrid **z_Grids, IntType n_part, IntType **pf2ff, IntType *fc2pc, IntType *CellToZone)
{
    IntType Mark_para = -9;
    IntType *f2c = grid->Getf2c();

    BCRecord **bcr = grid->Getbcr();

    IntType pnTFace, pnBFace, pnIFace, *pf2c, type;
    for(IntType p=0; p<n_part; p++) {
        pnTFace = z_Grids[p]->GetNTFace();
        pnBFace = z_Grids[p]->GetNBFace();
        pnIFace = z_Grids[p]->GetNIFace();
        pf2c = NULL;
        mfmem::snew_array_1D(pf2c,pnTFace<<1,dmrfl);
        z_Grids[p]->Setf2c(pf2c);
        IntType count = 0;
        for(IntType i=0; i<pnBFace-pnIFace; i++) {
            IntType ff = pf2ff[p][i];
            IntType fc1, fc2, pc1, pc2;
            fc1 = f2c[ff<<1];
            fc2 = f2c[(ff<<1)+1];
            pc1 = fc2pc[fc1];
            pc2 = fc2;
            pf2c[count++] = pc1;
            type = bcr[ff]->GetPatchID();
            pf2c[count++] = -type;
        }
        for(IntType i=pnBFace-pnIFace; i<pnBFace; i++) {
            IntType ff = pf2ff[p][i];
            IntType fc1, fc2, pc1, pc2;
            fc1 = f2c[ff<<1];
            fc2 = f2c[(ff<<1)+1];
            pc1 = fc2pc[fc1];
            pc2 = fc2pc[fc2];
            if(CellToZone[fc1]==p) {
                pf2c[count++] = pc1;
                pf2c[count++] = Mark_para;
            } else {
                pf2c[count++] = -pc2-1;              //表示并行边界面的右单元在分区中，为分辨pc2=0的情况，再减1
                pf2c[count++] = Mark_para;
            }
        }
        for(IntType i=pnBFace; i<pnTFace; i++) {
            IntType ff = pf2ff[p][i];
            IntType fc1, fc2, pc1, pc2;
            fc1 = f2c[ff<<1];
            fc2 = f2c[(ff<<1)+1];
            pc1 = fc2pc[fc1];
            pc2 = fc2pc[fc2];
            pf2c[count++] = pc1;
            pf2c[count++] = pc2;
        }        
    }
}

void Zone::GetFineGrids_F2N(IntType nfTNode, PolyGrid **z_Grids, IntType n_part, IntType **pn2fn)
{
    IntType nTFace, nBFace, nIFace;
    IntType *nNPF, **F2N, *f2c;

    IntType *fn2pn = NULL; 
    mfmem::snew_array_1D(fn2pn,nfTNode,dmrfl);

    for(IntType p=0; p<n_part; p++) {
        for(IntType i=0; i<nfTNode; i++) fn2pn[i]=-1;

        IntType nTNode = z_Grids[p]->GetNTNode();
        for(IntType i=0; i<nTNode; i++) fn2pn[pn2fn[p][i]] = i;

        nTFace = z_Grids[p]->GetNTFace();
        nBFace = z_Grids[p]->GetNBFace();
        nIFace = z_Grids[p]->GetNIFace();
        nNPF   = z_Grids[p]->GetnNPF();
        F2N    = z_Grids[p]->GetF2N();

        f2c    = z_Grids[p]->Getf2c();
        for(IntType i=nBFace-nIFace; i<nBFace; i++) {
            if(f2c[i<<1]<0) {
                IntType *fn = NULL;
                mfmem::snew_array_1D(fn,nNPF[i],dmrfl);
                for(IntType j=0; j<nNPF[i]; j++) fn[j] = F2N[i][j];
                for(IntType j=0; j<nNPF[i]; j++) F2N[i][j] = fn[nNPF[i]-1-j];
                mfmem::sdel_array_1D(fn);
                f2c[i<<1] = -f2c[i<<1]-1;
            }
        }
        for(IntType i=0; i<nTFace; i++) {
            for(IntType j=0; j<nNPF[i]; j++) {
                IntType fn = F2N[i][j];
                F2N[i][j] = fn2pn[fn];
            }
        }
    }
    mfmem::sdel_array_1D(fn2pn);

}

void Zone::GetFineGrids_nbBF(IntType *CellToZone, PolyGrid **z_Grids, IntType n_part, IntType **ff2pf, IntType mg)
{
    PolyGrid *grid = (PolyGrid *) grids[mg];
    IntType  nTFace = grid->GetNTFace();
    IntType  nBFace = grid->GetNBFace();
    IntType *f2c    = grid->Getf2c();

    //B03: 计算各分区并行边界面所对应的对方分区序号和面序号
    IntType **pnbZ  = NULL;
    IntType **pnbBF = NULL;
    IntType *pnIFace = NULL;
    mfmem::snew_array_1D(pnbZ ,n_part,dmrfl);
    mfmem::snew_array_1D(pnbBF,n_part,dmrfl);
    mfmem::snew_array_1D(pnIFace ,n_part,dmrfl);   
    for(IntType i=0; i<n_part; i++) {
        pnIFace[i] = z_Grids[i]->GetNIFace();
        pnbZ[i] = NULL;
        pnbBF[i]= NULL;
        mfmem::snew_array_1D(pnbZ[i] ,pnIFace[i],dmrfl);
        mfmem::snew_array_1D(pnbBF[i],pnIFace[i],dmrfl);
    }

    for(IntType i=0; i<n_part; i++) 
        pnIFace[i] = 0;

    IntType count=nBFace<<1;
    for(IntType i=nBFace; i<nTFace; i++) {
        IntType c1,c2,p1,p2;
        c1 = f2c[count++];
        c2 = f2c[count++];
        p1 = CellToZone[c1];
        p2 = CellToZone[c2];
        if(p1 != p2) {
            IntType f1, f2;
            if(p1==ff2pf[0][i]) {
                f1 = ff2pf[1][i];
                f2 = ff2pf[3][i];
            } else {
                f1 = ff2pf[3][i];
                f2 = ff2pf[1][i];
            }
            pnbZ[p1][pnIFace[p1]] = p2;
            pnbBF[p1][pnIFace[p1]]= f2;
            pnbZ[p2][pnIFace[p2]] = p1;
            pnbBF[p2][pnIFace[p2]]= f1;
            pnIFace[p1]++;
            pnIFace[p2]++;
        }
    }

    for(IntType i=0; i<n_part; i++) {
        z_Grids[i]->SetnbZ(pnbZ[i]);
        z_Grids[i]->SetnbBF(pnbBF[i]);
    }
    mfmem::sdel_array_1D(pnIFace);
    mfmem::sdel_array_1D(pnbZ);
    mfmem::sdel_array_1D(pnbBF);
}

void Zone::GetFineGrids_nbRN(PolyGrid **z_Grids, IntType n_part, IntType **pn2fn, IntType mg)
{
    // 计算 nINode,  *nbSN, *nbZN, *nbRN;
    PolyGrid *grid = (PolyGrid *) grids[mg];
    IntType  nTNode = grid->GetNTNode();

    IntType *pnTNode = NULL;
    mfmem::snew_array_1D(pnTNode ,n_part,dmrfl);
    for(IntType p=0; p<n_part; p++) pnTNode[p] = z_Grids[p]->GetNTNode();

    //B03: 计算各分区并行边界点所对应的对方分区序号和点序号
    IntType *pnINode = NULL;
    mfmem::snew_array_1D(pnINode ,n_part,dmrfl);
    for(IntType p=0; p<n_part; p++) pnINode[p] = 0;

    // how many processors each node belongs to.
    // the value must be large than 1 if the node on the parallel interface
    IntType *nZPN = NULL;
    mfmem::snew_array_1D(nZPN ,nTNode,dmrfl);
    for(IntType i=0; i<nTNode; i++) nZPN[i] = 0;
    for(IntType p=0; p<n_part; p++) {
        for(IntType i=0; i<pnTNode[p]; i++) {
            nZPN[pn2fn[p][i]]++;
        }
    }

    IntType *count = NULL;
    IntType **N2Z  = NULL;
    IntType **N2PN = NULL;
    mfmem::snew_array_1D(count ,nTNode,dmrfl);
    mfmem::snew_array_1D(N2Z ,nTNode,dmrfl);
    mfmem::snew_array_1D(N2PN,nTNode,dmrfl);
    for(IntType i=0; i<nTNode; i++) {
        count[i] = 0;
        N2Z[i] = 0;
        N2PN[i] = 0;
        if(nZPN[i]<2) continue;
        N2Z[i]  = NULL;
        N2PN[i] = NULL;
        mfmem::snew_array_1D(N2Z[i] ,nZPN[i],dmrfl);
        mfmem::snew_array_1D(N2PN[i],nZPN[i],dmrfl);
    }

    for(IntType p=0; p<n_part; p++) {
        for(IntType i=0; i<pnTNode[p]; i++) {
            IntType j=pn2fn[p][i];
            if(nZPN[j]<2) continue;
            pnINode[p] += nZPN[j]-1;
            N2Z[j][count[j]]    = p;
            N2PN[j][count[j]++] = i;
        }
    }

    IntType **pnbSN = NULL;
    IntType **pnbZN = NULL;
    IntType **pnbRN = NULL;
    mfmem::snew_array_1D(pnbSN ,n_part,dmrfl);
    mfmem::snew_array_1D(pnbZN ,n_part,dmrfl);
    mfmem::snew_array_1D(pnbRN ,n_part,dmrfl);
    for(IntType p=0; p<n_part; p++) {
        pnbSN[p] = NULL;
        pnbZN[p] = NULL;
        pnbRN[p] = NULL;
        mfmem::snew_array_1D(pnbSN[p],pnINode[p],dmrfl);
        mfmem::snew_array_1D(pnbZN[p],pnINode[p],dmrfl);
        mfmem::snew_array_1D(pnbRN[p],pnINode[p],dmrfl);
        IntType count = 0;
        for(IntType i=0; i<pnTNode[p]; i++) {
            IntType fn=pn2fn[p][i];
            if(nZPN[fn]>1) {
                for(IntType j=0; j<nZPN[fn]; j++) {
                    if(N2Z[fn][j]==p) continue;
                    pnbSN[p][count  ] = i;
                    pnbZN[p][count  ] = N2Z[fn][j];
                    pnbRN[p][count++] = N2PN[fn][j];
                }
            }
        }
        z_Grids[p]->SetNINode(pnINode[p]);
        z_Grids[p]->SetnbSN(pnbSN[p]);
        z_Grids[p]->SetnbZN(pnbZN[p]);
        z_Grids[p]->SetnbRN(pnbRN[p]);
    }

    mfmem::sdel_array_1D(pnTNode);
    mfmem::sdel_array_1D(pnbSN);
    mfmem::sdel_array_1D(pnbZN);
    mfmem::sdel_array_1D(pnbRN);

    mfmem::sdel_array_1D(pnINode);
    mfmem::sdel_array_1D(count);
    for(IntType i=0; i<nTNode; i++) {
        mfmem::sdel_array_1D(N2Z[i]);
        mfmem::sdel_array_1D(N2PN[i]);
    }
    mfmem::sdel_array_1D(N2Z );
    mfmem::sdel_array_1D(N2PN);
    mfmem::sdel_array_1D(nZPN);
}

/************************************************************************
*  send data from current zone to its neighbors
************************************************************************/
void Zone::CommInterfaceData(char *name)
{
    IntType g;
    PolyGrid *grid;

#ifndef MPICH
    IntType i,nbz;

    for(i=0; i<nNeighbor; i++) {
        nbz = nb[i];
        Zone *nz = ((Simulation *)simu)->GetZone(nbz);
        for(g=0; g<nGrids; g++) {
            grid = (PolyGrid *) grids[g];
            grid->CommInterfaceData(nbz, (PolyGrid*)nz->GetGrid(g), name);
        }
    }
#else
    for(g=0; g<nGrids; g++) {
        grid = (PolyGrid *) grids[g];
        IntType n = grid->GetNTCell()+grid->GetNBFace();
        RealFlow *q = (RealFlow *)grid->GetDataPtr(REAL_FLOW, n, name);
        grid->CommInterfaceDataMPI(q);
    }

#endif
}


void Zone::UpdateInterfaceData()
{
    for(IntType i=0; i<nSolvers; i++) 
        solvers[i]->UpdateInterfaceData();
}


void Zone::UpdataUnstVolData(){
    for(IntType i=0; i<nSolvers; i++) 
        solvers[i]->UpdataUnstVolData();
}


void Zone::CommCellCenterData()
{    
    IntType g;
    PolyGrid *grid;

#ifndef MPICH
    IntType i,nbz;

    for(i=0; i<nNeighbor; i++) {
        nbz = nb[i];
        Zone *nz = simu->GetZone(nbz);
        for(g=0; g<nGrids; g++) {
            grid = (PolyGrid *) grids[g];
            grid->CommCellCenterData(nbz, (PolyGrid*)nz->GetGrid(g));
        }
    }
#else
    for(g=0; g<nGrids; g++) {
        grid = (PolyGrid *) grids[g];
        RealGeom *xcc = grid->GetXcc();
        RealGeom *ycc = grid->GetYcc();
        RealGeom *zcc = grid->GetZcc();
        RealGeom *vol = grid->GetCellVol();
        grid->CommInterfaceDataMPI(xcc);
        grid->CommInterfaceDataMPI(ycc);
        grid->CommInterfaceDataMPI(zcc);
        grid->CommInterfaceDataMPI(vol);
    }
#endif
}


/************************************************************************
*                 test grid quality and gradient                        *
************************************************************************/
void Zone::TestReconstruction()
{
    mflog::log.set_one_processor_out();
#ifdef DEBUG
    mflog::log << "Entering TestReconstruction" << std::endl;
#endif

    for(IntType i=0; i<nGrids; i++){
        ((PolyGrid*)grids[i])->CheckGridQuality();
    }

#ifdef DEBUG
    mflog::log << "Exiting TestReconstruction" << std::endl;
#endif
}


/************************************************************************
*               add a new grid to a zone                                *
************************************************************************/
void Zone::AddSolver(Solver *solver)
{
    if(nSolvers == 0) {
        solvers = NULL;
        mfmem::snew_array_1D(solvers,nSolvers+1,dmrfl);
        solvers[nSolvers++] = solver;
    } else {
        Solver **solvert;
        
        solvert = solvers;
        solvers = NULL;
        mfmem::snew_array_1D(solvers,nSolvers+1,dmrfl);
        for(IntType i=0; i<nSolvers; i++) {
            solvers[i] = solvert[i];
        }
        mfmem::sdel_array_1D(solvert);
        solvers[nSolvers++] = solver;
    }
}


/************************************************************************
*  Assign neighbors for each grid                                       *
* Neighbor relationship(nNeighbor, nNeighborN) of all grids is same to 
* that of zone, but data memory(nb and nbN) is allocated for each grid.
*
* If parallel relationship data (nbZ, nbBF, nbSN, nbZN, nbRN) is read 
* from mmgrid*.in, the memory of each these data is allocated by each 
* grid, although these data are same for all grids in the same zone, such
* as finest grid, coarser grid, much coarser grid.... 
* However, if data of multi-grids is calculated by solver not by
* preprocessing(transgrid), the data memory of all coarse grids is not
* allocated but use that of the finest grid directly(only hold a pointer
* of finest grid). So deallocation of coarse grid needs much more cares.
************************************************************************/
void Zone::SpecifyNeighbors()
{
    PolyGrid *pgrid, *fgrid;
    IntType   i, j;

    fgrid = (PolyGrid *)  grids[0];
    IntType *fnbZ   = fgrid->GetnbZ();
    IntType *fnbBF  = fgrid->GetnbBF();

    IntType *fnbSN  = fgrid->GetnbSN();
    IntType *fnbZN  = fgrid->GetnbZN();
    IntType *fnbRN  = fgrid->GetnbRN();

    IntType nIFace = fgrid->GetNIFace();
    IntType nINode = fgrid->GetNINode();

    for(i=0; i<nGrids; i++) {
        pgrid = (PolyGrid *)  grids[i];
        pgrid->SetNumberOfFaceNeighbors(nNeighbor);
        pgrid->SetNumberOfNodeNeighbors(nNeighborN);
        if(nNeighbor > 0) {
            IntType *nb = NULL;
            mfmem::snew_array_1D(nb, nNeighbor, dmrfl);
            pgrid->SetFaceNeighborZones(nb);
        }
        if(nNeighborN > 0) {
            IntType *nbN = NULL;
            mfmem::snew_array_1D(nbN, nNeighborN, dmrfl);
            pgrid->SetNodeNeighborZones(nbN);
        }

        IntType *face_neighbor_zones = pgrid->GetFaceNeighborZones();
        for(j=0; j<nNeighbor; j++) {
            face_neighbor_zones[j] = nb[j];
        }

        IntType *node_neighbor_zones = pgrid->GetNodeNeighborZones();
        for(j=0; j<nNeighborN; j++) {
            node_neighbor_zones[j] = nbN[j];
        }

#ifndef MPICH
        IntType nbz;
        PolyGrid **neighbor_grids = NULL;
        if(nNeighbor > 0) {            
            mfmem::snew_array_1D(neighbor_grids, nNeighbor, dmrfl);
            pgrid->SetNeighborGrids(neighbor_grids);
        }
        for(j=0; j<nNeighbor; j++) {
            nbz = nb[j];
            Zone *nz = ((Simulation *)simu)->GetZone(nbz);
            neighbor_grids[j] = (PolyGrid *)nz->GetGrid(i);
        }
#endif
    } 
  //是否删除 nbN;lihuan-2018-11-21（zone定义的nbN赋值给grid后，已经不用了），nb太难搜索了，不确定
  //mfmesh::sdel_array_1D(nbN);
}


/************************************************************************
*                                                                       *
************************************************************************/
void Zone::PostZone()
{
    for(IntType i=0; i<nSolvers; i++)
        solvers[i]->Post();  
}


/******************************************************************************\
                          check parameter
\******************************************************************************/
void Zone::CheckParameter()
{
    IntType RunExit = 0;
    
    //ensure cfl_nstep>0
    IntType cfl_nstep;
    GetData(&cfl_nstep, INT, 1, "cfl_nstep");
    if(cfl_nstep <= 0){
        mflog::log.set_one_processor_out();
        mflog::log << std::endl << "Error! cfl_nstep = " << cfl_nstep << " . It must > 0!" << std::endl;
 
        RunExit = 1;
    }
    
    if(RunExit) mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
}


/******************************************************************************\
                          update parameter
\******************************************************************************/
void Zone::UpdateParameter()
{
    RealFlow mach, alpha, beita;
    RealFlow T, p_bar, p, re, rho;
    RealFlow gam, gamm1, gascon; 
    RealFlow amu, uqq, ainf, u, v, w, ss, amu120;
    RealFlow p_stag,rho_stag,e_stag,t_stag;
    RealFlow trat,temp;
    
    GetData(&mach,   REAL_FLOW, 1, "mach");
    GetData(&alpha,  REAL_FLOW, 1, "alpha");
    GetData(&beita,  REAL_FLOW, 1, "beita");
    GetData(&gam,    REAL_FLOW, 1, "gam");
    GetData(&gascon, REAL_FLOW, 1, "gascon");
    gamm1 = gam-1.0;
    
    IntType IncomingType;
    GetData(&IncomingType, INT, 1, "IncomingType");
    
    if((IncomingType == 1) || (IncomingType == 2)){  //有输入值
        GetData(&T,      REAL_FLOW, 1, "T");
    }else if(IncomingType == 3){  // Altitude，需要计算获取
        RealGeom Altitude = 0.0;
        GetData(&Altitude,   REAL_GEOM, 1, "Altitude");
        CalpandTfromAltitude(Altitude, p_bar, T);
        UpdateData(&T,    REAL_FLOW, 1, "T");
    }
    
    //viscousity function -- Sutherland's Law 
    RealFlow tref=288.15, sref=110.4, amuref=1.78938e-5;
    trat  = T/tref;
    amu   = amuref*trat*sqrt(trat)*(tref+sref)/(T+sref);
    trat  = 120.0/tref;
    amu120= amuref*trat*sqrt(trat)*(tref+sref)/(120.0+sref);
    ainf  = sqrt(gam*gascon*T);
    
    alpha = alpha*PI/180.;
    beita = beita*PI/180.;
    uqq   = ainf*mach;
    u     = uqq*cos(alpha)*cos(-1.*beita);
    v     = uqq*sin(-1.*beita);
    w     = uqq*sin(alpha)*cos(-1.*beita);
    p     = 0.0;  
    
    if(IncomingType == 1){  //Re and T
        GetData(&re, REAL_FLOW, 1, "re");  
        rho   = re*amu/uqq;
        p_bar = rho*gascon*T;
    }else if(IncomingType == 2){  //p and T
        GetData(&p_bar, REAL_FLOW, 1, "p_bar");
        rho = p_bar/gascon/T;
        re  = rho*uqq/amu;
    }else if(IncomingType == 3){  //Altitude, p and T has been obtained before.
        rho = p_bar/gascon/T;
        re  = rho*uqq/amu;
    }else{
        std::cerr<<"IncomingType is wrong!  Need new code!"<<endl;
    }
    
    temp   = 1.0+0.5*gamm1*mach*mach;
    p_stag = pow(temp, gam/gamm1);
    p_stag = (p_stag-1.0) * p_bar;  //note: p_stag-p_bar !!!
    rho_stag = pow(temp, 1.0/gamm1);
    rho_stag *= rho;   
    e_stag = p_bar/gamm1+0.5*rho*(u*u+v*v+w*w);
    t_stag = T*temp;
    ss = p_bar/pow(rho,gam);
 
    UpdateData(&u,    REAL_FLOW, 1, "u");
    UpdateData(&v,    REAL_FLOW, 1, "v");
    UpdateData(&w,    REAL_FLOW, 1, "w");
    UpdateData(&rho,  REAL_FLOW, 1, "rho");
    UpdateData(&re,   REAL_FLOW, 1, "re");
    UpdateData(&p,    REAL_FLOW, 1, "p");
    UpdateData(&p_bar,REAL_FLOW, 1, "p_bar");
    UpdateData(&amu,  REAL_FLOW, 1, "amu");
    UpdateData(&ainf, REAL_FLOW, 1, "ainf");
    UpdateData(&p_stag,   REAL_FLOW, 1, "p_stag");
    UpdateData(&rho_stag, REAL_FLOW, 1, "rho_stag");
    UpdateData(&e_stag, REAL_FLOW, 1, "e_stag");
    UpdateData(&t_stag, REAL_FLOW, 1, "t_stag");
    UpdateData(&ss, REAL_FLOW, 1, "entropy_ss");
    UpdateData(&amu120,  REAL_FLOW, 1, "amu120");
    
    IntType steady = simu->IsSteady();
    UpdateData(&steady,    INT, 1, "steady");

    //To updata the coarse grid's steps
    IntType n_steps_coarse = 0;
    UpdateData(&n_steps_coarse, INT, 1, "n_steps_coarse");
    simu->UpdateData(&n_steps_coarse, INT, 1, "n_steps_coarse");

    mflog::log.set_one_processor_out();
    mflog::log << endl << endl << IOS_AUTO;
    mflog::log << "The incoming flowing condition: "<<endl;
    mflog::log << SEP_LINE << endl;
    mflog::log << "      uinf     = " << u        << endl;
    mflog::log << "      vinf     = " << v        << endl;
    mflog::log << "      winf     = " << w        << endl;
    mflog::log << "      rhoinf   = " << rho      << endl;
    mflog::log << "      ainf     = " << ainf     << endl;
    mflog::log << "      p_bar    = " << p_bar    << endl;
    mflog::log << "      p        = " << p        << endl;
    mflog::log << "      T        = " << T        << endl;
    mflog::log << "      re       = " << re       << endl;
    mflog::log << "      amu      = " << amu      << endl;
    mflog::log << "      amu120   = " << amu120   << endl;
    mflog::log << "      p_stag   = " << p_stag   << endl;
    mflog::log << "      rho_stag = " << rho_stag << endl;
    mflog::log << "      e_stag   = " << e_stag   << endl;
    mflog::log << "      t_stag   = " << t_stag   << endl;
    mflog::log << "      n_steps_coarse   = " << n_steps_coarse   << endl;
    mflog::log << SEP_LINE << endl;
    mflog::log << endl << endl;


    //To update the boundary parameters
    IntType i;

    ComputeMachineZero();
//    ReadJetFile();
    
    //jet或NacOutNum不为零时需要计算，否则等于来流的总温和总压
    RealFlow rho_stag_jet = rho_stag;
    RealFlow p_stag_jet = p_stag;
    RealFlow rho_jet_min = rho;
    RealFlow p_jet_min = p;
    RealFlow e_stag_jet = e_stag;
    
#ifdef MPICH
    Parallel::parallel_max(p_stag_jet,  MPI_COMM_WORLD);
    Parallel::parallel_max(rho_stag_jet, MPI_COMM_WORLD);
#endif
    UpdateData(&p_stag_jet,   REAL_FLOW, 1, "p_stag_jet");
    UpdateData(&rho_stag_jet, REAL_FLOW, 1, "rho_stag_jet");
    
    //Compute limit value, p_min,p_max,rho_min,rho_max,e_stag_max
    RealFlow ratio_rhop_min = 1.0e-4, ratio_rhop_max = 10.0, ratio_estag_max = 1.0e5, ratio_p_break = 0.01;
    GetData(&ratio_rhop_min,  REAL_FLOW, 1, "ratio_rhop_min");
    GetData(&ratio_rhop_max,  REAL_FLOW, 1, "ratio_rhop_max");
    GetData(&ratio_estag_max, REAL_FLOW, 1, "ratio_estag_max");
    GetData(&ratio_p_break,   REAL_FLOW, 1, "ratio_p_break");
 
    RealFlow p_min, p_max, rho_min, rho_max, e_stag_max, p_break;
    p_min   = ratio_rhop_min*p_bar-p_bar;
    p_max   = ratio_rhop_max*(p_stag+p_bar)-p_bar;
    rho_min = ratio_rhop_min*rho;
    rho_max = ratio_rhop_max*rho_stag;
    e_stag_max = ratio_estag_max*e_stag;
    p_break = ratio_p_break*p_bar-p_bar;
    
    mflog::log.set_one_processor_out();
    mflog::log<<endl<<SEP_LINE
              <<endl<<"      rho_min    = "<< IOS_AUTO << rho_min
              <<endl<<"      rho_max    = "<< IOS_AUTO << rho_max
              <<endl<<"      p_min      = "<< IOS_AUTO << p_min
              <<endl<<"      p_max      = "<< IOS_AUTO << p_max
              <<endl<<"      e_stag_max = "<< IOS_AUTO << e_stag_max
              <<endl<<"      p_break    = "<< IOS_AUTO << p_break
              <<endl<<SEP_LINE
              <<endl;

    UpdateData(&p_min,      REAL_FLOW, 1, "p_min");
    UpdateData(&p_max,      REAL_FLOW, 1, "p_max");
    UpdateData(&rho_min,    REAL_FLOW, 1, "rho_min");
    UpdateData(&rho_max,    REAL_FLOW, 1, "rho_max");
    UpdateData(&e_stag_max, REAL_FLOW, 1, "e_stag_max");
    UpdateData(&p_break,    REAL_FLOW, 1, "p_break");
    
    //vv_max:存储运动物体的最大速度平方值，当大于来流值时用于更新限制量，此处用于初始化为0。
    RealFlow vv_max = 0.0;
    UpdateData(&vv_max, REAL_FLOW, 1, "vv_max");
    
    //# n_wconverg   -- 写残差和force文件的周期，默认为5步写一次
    //如果n_wconverg没有输入，给它赋默认值20
    IntType n_wconverg = 0;
    GetData(&n_wconverg, INT, 1, "n_wconverg", 0);
    if(n_wconverg == 0){
        n_wconverg = 20;
        UpdateData(&n_wconverg, INT, 1, "n_wconverg");
    }
    
    //将发动机入流参数NacPb的值改为相对量
    IntType NacInNum = 0;
    GetData(&NacInNum, INT, 1, "NacInNum", 0);
    if(NacInNum){
        RealFlow *NacPb = NULL; 
        mfmem::snew_array_1D(NacPb, NacInNum,dmrfl);
        GetData(NacPb,REAL_FLOW, NacInNum, "NacPb");
        
        for(i=0;i<NacInNum;i++) NacPb[i] -= p_bar;
      
        UpdateData(NacPb, REAL_FLOW, NacInNum, "NacPb");
        
        mfmem::sdel_array_1D(NacPb);
    }

    // get grid directory and update to parameter pool, tangj 2020-01-11
    this->GetGridDir();
}


/******************************************************************************\
                          fix some parameters
Note：    fix some parameters not change usually
Update:   2012-2-6  first implementation
Update:   2012-7-24 重叠网格时，多重插值使用体心到体心的对应插值 tangj
\******************************************************************************/
void Zone::FixParameter()
{
    //# boundary condition input from file or not (0 -- not; 1 -- input from file)
    IntType bcInput = 0;
    UpdateData(&bcInput,INT, 1, "bcInput"); 
    
    //# restart from step 1 or continue (0 -- restart; 1 -- continue)
    //# restart or continue for turbulence computation (0 -- restart; 1 -- continue)
    IntType restart=0, turbRst=0;
    GetData(&restart, INT, 1, "restart");
    turbRst = restart;
    UpdateData(&turbRst, INT, 1, "turbRst");
    
    //limit min and max of rho and p, limit max of e stag
    RealFlow ratio_rhop_min = 1.0e-4;
    RealFlow ratio_rhop_max = 10.0;
    RealFlow ratio_estag_max = 1.0e5;
    UpdateData(&ratio_rhop_min, REAL_FLOW, 1, "ratio_rhop_min");
    UpdateData(&ratio_rhop_max, REAL_FLOW, 1, "ratio_rhop_max");
    UpdateData(&ratio_estag_max, REAL_FLOW, 1, "ratio_estag_max");

    //##################################################
    //# Some parameters for the Invisflux computations #
    //##################################################
    //# EntropyCorType=3 entropy correction is original harten's, =1、2 is modified(1 is suggested)
    //IntType EntropyCorType = 1;
    //UpdateData(&EntropyCorType, INT, 1, "EntropyCorType");
    //# entropy correction constants for Roe flux
    //# (0.3 is suggested for alf_l, as 0.3 is suggested for alf_n, 0.0 is no entropy correction)
    RealFlow epsa_r;
    GetData(&epsa_r, REAL_FLOW, 1, "epsa_r");
    IntType EntropyCorType;
    GetData(&EntropyCorType, INT, 1, "EntropyCorType");
    if(EntropyCorType == 4){
        RealFlow mach;
        GetData(&mach,   REAL_FLOW, 1, "mach");
        /*
        if(mach<0.8){
            epsa_r = 0.0+(mach-0.0)/(0.8-0.0)*(0.1-0.0);
        }else if(mach<2.0){
            epsa_r = 0.1+(mach-0.8)/(2.0-0.8)*(0.2-0.1);
        }else if(mach<5.0){
            epsa_r = 0.2+(mach-2.0)/(5.0-2.0)*(0.3-0.2);
        }else{
            epsa_r = 0.3;
        }
        */
        
        /*
        if(mach<3.0){
            epsa_r = 0.2;
        }else{
            epsa_r = 0.3;
        }
        */
        
        if(mach<0.8){
            epsa_r = 0.025;
        }else if(mach<2.0){
            epsa_r = 0.025+(mach-0.8)/(2.0-0.8)*(0.2-0.025);
        }else if(mach<5.0){
            epsa_r = 0.2+(mach-2.0)/(5.0-2.0)*(0.3-0.2);
        }else{
            epsa_r = 0.3;
        }
    }
    
    RealFlow alf_l     = epsa_r;
    RealFlow alf_n     = epsa_r;
    RealFlow Re_target = 500.0;
    UpdateData(&alf_l,  REAL_FLOW, 1, "alf_l");
    UpdateData(&alf_n,  REAL_FLOW, 1, "alf_n");
    UpdateData(&Re_target,  REAL_FLOW, 1, "Re_target");
    

    //# the parameter for Rusanov Flux and AUSMPW+ (sigam=0.25 is suggested)
    IntType AUSM_Sound = 1;
    UpdateData(&AUSM_Sound, INT, 1, "AUSM_Sound");

    RealFlow sigam = 0.25;
    UpdateData(&sigam,  REAL_FLOW, 1, "sigam");

    //# the paremeters for Central Scheme with Artificial Dissipation
    RealFlow Kzero, Ksecond, Kfourth;
    Kzero   = 0.15;
    Ksecond = 0.5;
    Kfourth = 0.0078125;
    UpdateData(&Kzero,    REAL_FLOW, 1, "Kzero"  );
    UpdateData(&Ksecond,  REAL_FLOW, 1, "Ksecond");
    UpdateData(&Kfourth,  REAL_FLOW, 1, "Kfourth");
    
    //# EPS for Vencat's limiter(1.0~5.0 is suggested, 50 is almost equal to no limit)
    //RealFlow eps_vencat = 1.0;
    //UpdateData(&eps_vencat,   REAL_FLOW, 1, "eps_vencat"  );
    

    //# the parameters for U-MUSCL
    RealFlow kai_umuscl = 0.0;
    UpdateData(&kai_umuscl, REAL_FLOW, 1, "kai_umuscl"  );
    
    //#################################################
    //#   Some parameters for MUSCL reconstruction    #
    //#    for program developing, not used now!      #
    //#################################################
    //# Dis_Weight: 0 -- distance weight not used, 1 -- use
    //# Grd_Ratio: grid of aspect ratio little than this value will not use MUSCL
    //$I-Dis_Weight     $R-Grd_Ratio
    //1                 100.0
    //# MUSCL reconstruction type
    //# 0 -- no limiter second order MUSCL
    //# 1 -- minmod limiter
    //# 2 -- Van Albada limiter
    //# 3 -- WENO limiter
    //# 4 -- TAU's gradient method
    //$I-limiter_MUSCL
    //1
    IntType Dis_Weight;
    Dis_Weight = 1;
    UpdateData(&Dis_Weight, INT, 1, "Dis_Weight");
    RealFlow Grd_Ratio = 100.0;
    UpdateData(&Grd_Ratio,  REAL_FLOW, 1, "Grd_Ratio");
    IntType limiter_MUSCL = 1;
    UpdateData(&limiter_MUSCL, INT, 1, "limiter_MUSCL");

    //#################################################
    //# Some parameters for the Gradient computations #
    //#################################################
    //#if GradQ or GradQTurb = 6,then in boundary layer, the grid layer number of GaussLayer will use Gauss method
    //#if GaussLayer<=0,then all use Gauss node. 
    IntType GaussLayer = 5;
    UpdateData(&GaussLayer, INT, 1, "GaussLayer");

    //#For grid quality, 0.0 is the best, 1.0 is the worst
    //#Notice: set <0.0 or >=1.0, no degeneration, 
    //#set locally 1st order differencing on cell of bad quality
    //#neglected locally viscous fluxes in face of bad quality 
    //#0.9~1.0 is suggested
    //
    //$$$$$$$$需要直接输入$$$$$$$$
    //
    
    RealFlow BadCellAngle, BadFaceAngle;
    BadCellAngle = -1.0;
    BadFaceAngle = -1.0;
    UpdateData(&BadCellAngle, REAL_FLOW, 1, "BadCellAngle");
    UpdateData(&BadFaceAngle, REAL_FLOW, 1, "BadFaceAngle");
    

    //#weight for lsq(2 is suggested)
    //#     1 -- no weight
    //#     2 -- inverse distance weight
    //#     3 -- area weight
    //#     4 -- normal distance weight
    IntType weight_lsq = 2;
    UpdateData(&weight_lsq, INT, 1, "weight_lsq");
    
    //######################################################
    //# Some parameters for the turbulence computation     #
    //######################################################   
    //# some parameters for turbulence model
    
    IntType turb_substeps;
    turb_substeps = 1;
    UpdateData(&turb_substeps, INT, 1, "turb_substeps");
    
    //#Coefficient to multiply the limit of the turbulent production on SST turbulence model           
    //#plim = prolim*rho*k*sqrt(prod/mutc) ----(<0.0, no limit, >=0.31 is suggested)
    //RealFlow prolim = 1.0;
    //UpdateData(&prolim, REAL_FLOW, 1, "prolim");

    //#source_vorticity=1,vorticity source, 0 is tauij(1 is suggested)
    //#prod_part_neglect=1,neglect turbulent production term's r23*rho*sst_k*divg, 0 is no neglect(1 is suggested) 
    //#turb_order: turbulence advection term order(1 or 2, 1 is suggested)
    //IntType source_vorticity, prod_part_neglect;
    //source_vorticity  = 1;
    //prod_part_neglect = 1;
    IntType turb_order;
    turb_order         = 1;
    //UpdateData(&source_vorticity,  INT, 1, "source_vorticity" );
    //UpdateData(&prod_part_neglect, INT, 1, "prod_part_neglect");
    UpdateData(&turb_order,   INT, 1, "turb_order"  );

    //#Control the maximum allowable change in turbulence equation unknown at every iteration.
    //#For example, dqmax = 0.50 implies that the allowable change in turbulence variable
    //#would be no more than 50% of its value at every iteration.(0.5 is suggested)
    //#max ratio of muet to free stream muel(20000.0 is suggested)
    RealFlow dqmax_turb, max_muet;
    dqmax_turb = 0.5;
    max_muet   = 1.0e10;
    UpdateData(&dqmax_turb, REAL_FLOW, 1, "dqmax_turb");
    UpdateData(&max_muet,   REAL_FLOW, 1, "max_muet"  );

    //#######################################################
    //# Some parameters for the time iteration computations #
    //#######################################################
    //# number of sweep in sub-iteration
    //IntType sweeps = 1;
    //UpdateData(&sweeps,      INT, 1, "sweeps");

    //#limit for DQ in lu-sgs(2 is suggested for non-precondition, as 1 is used for precondition)
    IntType DQ_limit = 2;
    UpdateData(&DQ_limit,  INT, 1, "DQ_limit");

    //# convergence tolerance in sub-iteration
    //RealFlow epsilon = 1.0e-01;
    //UpdateData(&epsilon, REAL_FLOW, 1, "epsilon");

    //# ratio for lhs of the lusgs iteration(1.0 is suggested)
    RealFlow lhs_omga = 1.0;
    UpdateData(&lhs_omga, REAL_FLOW, 1, "lhs_omga");
    
    
    // restart=3或4时，从物理时间步数第n_read_unst步续算
    IntType n_read_unst = 2;
    UpdateData(&n_read_unst,  INT, 1, "n_read_unst");
    // n_wconverg   -- 写残差和force文件的周期，默认为20步写一次
    IntType  n_wrest_unst = 5;
    UpdateData(&n_wrest_unst,  INT, 1, "n_wrest_unst");
    IntType n_wconverg = 0;
    GetData(&n_wconverg, INT, 1, "n_wconverg", 0);
    if(n_wconverg == 0){
        n_wconverg = 20;
        UpdateData(&n_wconverg, INT, 1, "n_wconverg");
    }

    //#limit max_dt for max_dt/min_dt<=ratio_dtmax
    RealFlow ratio_dtmax = 1.0e80;  //approach to no limit
    UpdateData(&ratio_dtmax, REAL_FLOW, 1, "ratio_dtmax");

    //#######################################################
    //# Some parameters for the GMRES method                #
    //#######################################################
    IntType ADU; //, GMRES, kspan, GMRESLUSGS, gmresweeps;
    //GMRES    = 0;
    //kspan      = 10;
    //GMRESLUSGS = 0;
    //gmresweeps = 5;
    ADU        = 1;
    /*
    UpdateData(&GMRES,       INT, 1, "GMRES");
    UpdateData(&kspan,       INT, 1, "kspan");
    UpdateData(&GMRESLUSGS,  INT, 1, "GMRESLUSGS");
    UpdateData(&gmresweeps,  INT, 1, "gmresweeps");
    */
    UpdateData(&ADU,         INT, 1, "ADU");

    //RealFlow gmresepsilon = 1.0e-1;
    //UpdateData(&gmresepsilon, REAL_FLOW, 1, "gmresepsilon");

    //#################################################
    //# Some input conditions for the computation     #
    //#################################################
    //# CFL number and relatives
    //#the global cfl number is compute using cfl_start, cfl_end and cfl_nstep
    //# cfl_min used for minish cfl number where p is mini. cfl_min=0.5*cfl_start is suggested.
    //#when the cell's p<=p_min or quality is bad, cfl_min is used for this cell
    //#when the cell's p_min<p<p_break, the cfl number is between cfl_min and global cfl
    //#when the cell's p>=p_break, the global cfl number is used
    //#0.001 is suggested for p_min, p_break is suggested between 0.01 and 0.05  
    //#cfl_coeff：粗网格cfl系数，与细网格的cfl数相乘得到粗网格的cfl数，建议取0.5
    //注意：p_min通过参数ratio_rhop_min求出，p_break通过参数ratio_p_break求出
    //zhyb20200609: 修改cfl_min为当前步cfl数乘以0.5，不再固定为cfl_start乘以0.5，这样避免在cfl_start很小时，
    //造成局部流场收敛过慢甚至不收敛。具体执行在函数LimitTimeStep中。
    RealFlow ratio_p_break, cfl_coeff;
    ratio_p_break   = 0.01;
    cfl_coeff = 0.5;
    UpdateData(&ratio_p_break,   REAL_FLOW, 1, "ratio_p_break");
    UpdateData(&cfl_coeff, REAL_FLOW, 1, "cfl_coeff");
    
    RealFlow cfl_start, cfl_end, cfl_min, cfl_ratio;
    GetData(&cfl_start, REAL_FLOW, 1, "cfl_start");
    GetData(&cfl_end,   REAL_FLOW, 1, "cfl_end");
    cfl_min = 0.5*cfl_start;
    UpdateData(&cfl_min,   REAL_FLOW, 1, "cfl_min");
    cfl_ratio = cfl_end/cfl_start;
    UpdateData(&cfl_ratio, REAL_FLOW, 1, "cfl_ratio");
    
    //######################################################
    //# Some parameters for the unsteady computation       #
    //######################################################
    //# time_accuracy (0 is the first order , 0.5 is the second order)
    RealFlow time_accuracy = 0.5;
    UpdateData(&time_accuracy, REAL_FLOW, 1, "time_accuracy");

    //######################################################
    //# Some parameters for the output                     #
    //######################################################
    //# some parameters for output
    IntType output_Force;
    output_Force = 1;
    UpdateData(&output_Force, INT, 1, "output_Force");

    //######################################################
    //# Some parameters for the air                        #
    //######################################################
    //# some constants for air(Do Not modify)
    RealFlow gam, gascon, prl, prt, cp;
    gam    = 1.4;
    gascon = 287.053;
    prl    = 0.72;
    prt    = 0.9;
    cp     = 1003.0;
    UpdateData(&gam,    REAL_FLOW, 1, "gam");
    UpdateData(&gascon, REAL_FLOW, 1, "gascon");
    UpdateData(&prl,    REAL_FLOW, 1, "prl");
    UpdateData(&prt,    REAL_FLOW, 1, "prt");
    UpdateData(&cp,     REAL_FLOW, 1, "cp");

    //######################################################
    //# Some parameters unusing in the mflow00             #
    //######################################################
    //# number of multi-stage for Runge-Kutta method
    IntType n_stage = 3;
    UpdateData(&n_stage, INT, 1, "n_stage");

    //# parameters for N_stage Runge-Kutta method
    RealFlow lamda[3];
    lamda[0] = 0.33;
    lamda[1] = 0.5;
    lamda[2] = 1.0;
    UpdateData(lamda, REAL_FLOW, 3, "lamda");

    //# CEM or not (0 -- not; 1 -- CEM )
    IntType CEM = 0;
    UpdateData(&CEM, INT, 1, "CEM");

    //# moveing grid or not (0 -- stationary grid; 1 -- moving grid)
    IntType move_grid = 0;
    UpdateData(&move_grid, INT, 1, "move_grid");

    //# dual-time or not (0 -- not; 1 -- dual-time step method)
    IntType dual_time = 0;
    UpdateData(&dual_time, INT, 1, "dual_time");

    //# compressible flow or uncompressible flow (0 -- uncompressible; 1 -- compressible)
    IntType comp = 1;
    UpdateData(&comp, INT, 1, "comp");

    //# Discrete Numerical Method or not for LHS matrices (0 -- not; 1 -- DNM)
    IntType dnm = 0;
    UpdateData(&dnm, INT, 1, "dnm");

    RealFlow delta_dnm = 1.0e-6;
    UpdateData(&delta_dnm, REAL_FLOW, 1, "delta_dnm");

    //# Low Speed Preconditioned method (1-- PEPB; 2 -- PEPB1; 3 -- PESB)
    IntType iprecmethod = 1;
    UpdateData(&iprecmethod, INT, 1, "iprecmethod");

    //# some parameters for output
    IntType post_form = 0;
    UpdateData(&post_form, INT, 1, "post_form");

    RealFlow c = -1.0;
    UpdateData(&c, REAL_FLOW, 1, "c");
    
    //分区初始化参数，0表示不分区初始化，用于郑鸣编写的fluent格式的分区初始化方法
    IntType init_zone = 0;
    UpdateData(&init_zone,INT, 1, "init_zone"); 
    //# ZoneInitNum: initial live number
    //# ZoneInitNo: live no.
    //# ZoneInitBoud: partner boundary number for the initial live
    //# using for ZoneInit=1
    IntType ZoneInitNum = 1;
    UpdateData(&ZoneInitNum,INT, 1, "ZoneInitNum");
    IntType *ZoneInitNo=NULL, *ZoneInitBoud=NULL;
    mfmem::snew_array_1D(ZoneInitNo,ZoneInitNum,dmrfl);
    mfmem::snew_array_1D(ZoneInitBoud,ZoneInitNum,dmrfl);
    for(IntType i=0;i<ZoneInitNum;i++){
        ZoneInitNo[i] = 9;    //zhyb: 此处为随意给的值
        ZoneInitBoud[i] = 301;
    }
    UpdateData(ZoneInitNo,   INT, ZoneInitNum, "ZoneInitNo");
    UpdateData(ZoneInitBoud, INT, ZoneInitNum, "ZoneInitBoud");
    mfmem::sdel_array_1D(ZoneInitNo);
    mfmem::sdel_array_1D(ZoneInitBoud);
//-----------------------------------------------------------------------------
}


/******************************************************************************\
    update variable limit when unsteady flow compute at each time step
    zhyb：主要用于解决非定常计算时当物体运动速度过大，造成利用自由来流生成的默认限制值过窄的问题
\******************************************************************************/
void Zone::UpdateVariableLimit()
{
    IntType i,type;
    RealFlow vv_tmp, vv00;
    
    PolyGrid *grid = (PolyGrid *)GetGrid(0);
    
    IntType nBFace = grid->GetNBFace();
    
    RealFlow mach, ainf, u00, v00, w00;
    grid->GetData(&mach, REAL_FLOW, 1, "mach");
    grid->GetData(&ainf, REAL_FLOW, 1, "ainf");
    grid->GetData(&u00, REAL_FLOW, 1, "u");
    grid->GetData(&v00, REAL_FLOW, 1, "v");
    grid->GetData(&w00, REAL_FLOW, 1, "w");
    vv00 = mach*ainf*mach*ainf;
    
    RealFlow vv_max_old=0.0;
    grid->GetData(&vv_max_old, REAL_FLOW, 1, "vv_max"); 
    
    BCRecord **bcr       = grid->Getbcr();
    RealGeom *BFacevgx   = grid->GetBoundaryFaceVelocityX();
    RealGeom *BFacevgy   = grid->GetBoundaryFaceVelocityY();
    RealGeom *BFacevgz   = grid->GetBoundaryFaceVelocityZ();
  
    //首先寻找物面的最大运动速度,只需要在最密网格上寻找即可
    RealFlow vv_max_new = 0.0;
    RealFlow ur,vr,wr;
    for(i=0;i<nBFace;i++){
        type  = bcr[i]->GetType();
        if(type != WALL) continue;
        
        ur = BFacevgx[i];
        vr = BFacevgy[i];
        wr = BFacevgz[i];

        if(ur*u00 < 0.0) ur -= u00;  //zhyb: 物体运动速度与来流速度反号，求相对值
        if(vr*v00 < 0.0) vr -= v00;
        if(wr*w00 < 0.0) wr -= w00;
        vv_tmp = ur*ur+vr*vr+wr*wr;
        //vv_tmp = BFacevgx[i]*BFacevgx[i]+BFacevgy[i]*BFacevgy[i]+BFacevgz[i]*BFacevgz[i];
        vv_max_new = MAX(vv_max_new, vv_tmp);
    }
#ifdef MPICH
    RealFlow tmp_glb;
    MPI_Allreduce(&vv_max_new, &tmp_glb, 1, MPIReal, MPI_MAX, MPI_COMM_WORLD);
    vv_max_new = tmp_glb;
#endif
    if(vv_max_new < vv00) return;  //不需要更新
    if(vv_max_new < vv_max_old) return;  //不需要更新
    
    grid->UpdateData(&vv_max_new, REAL_FLOW, 1, "vv_max");
      
    RealFlow rho_min, rho_max, p_min, p_max, e_stag_max, p_break;
    grid->GetData(&rho_min,    REAL_FLOW, 1, "rho_min");
    grid->GetData(&rho_max,    REAL_FLOW, 1, "rho_max");
    grid->GetData(&p_min,      REAL_FLOW, 1, "p_min");
    grid->GetData(&p_max,      REAL_FLOW, 1, "p_max");
    grid->GetData(&e_stag_max, REAL_FLOW, 1, "e_stag_max");
    grid->GetData(&p_break,    REAL_FLOW, 1, "p_break");
    
    //简单更新，利用自由来流的密度和压力、最大速度来计算限制量
    RealFlow rho,p,p_bar,gam,gamm1;
    grid->GetData(&rho,   REAL_FLOW, 1, "rho");
    grid->GetData(&p,     REAL_FLOW, 1, "p");
    grid->GetData(&p_bar, REAL_FLOW, 1, "p_bar");
    grid->GetData(&gam,   REAL_FLOW, 1, "gam");
    gamm1 = gam-1.0;
    
    RealFlow mach_t, rho_t, p_t, p_stag_t, rho_stag_t, e_stag_t, temp, temp2;
    mach_t     = sqrt(vv_max_new)/ainf;
    temp       = 1.0+0.5*gamm1*mach_t*mach_t;
    p_stag_t   = pow(temp, RealFlow(gam/gamm1))*(p+p_bar);  //zhyb: 此处没有-p_bar !!!
    rho_stag_t = pow(temp, RealFlow(1.0/gamm1))*rho;
    e_stag_t   = p_bar/gamm1+0.5*rho*vv_max_new; 
    temp2      = 1.0+0.5*gamm1*mach*mach;
    rho_t      = rho*pow(temp2/temp,1.0/gamm1);
    p_t        = (p+p_bar)*pow(temp2/temp,gam/gamm1);  //zhyb: 此处没有-p_bar !!!
   
    RealFlow ratio_rhop_min = 1.0e-4, ratio_rhop_max = 10.0, ratio_estag_max = 1.0e5, ratio_p_break = 0.01;
    GetData(&ratio_rhop_min,  REAL_FLOW, 1, "ratio_rhop_min");
    GetData(&ratio_rhop_max,  REAL_FLOW, 1, "ratio_rhop_max");
    GetData(&ratio_estag_max, REAL_FLOW, 1, "ratio_estag_max");
    GetData(&ratio_p_break, REAL_FLOW,   1, "ratio_p_break");
    
    rho_max = MAX(rho_max, ratio_rhop_max*rho_stag_t);
    p_max   = MAX(p_max,   ratio_rhop_max*p_stag_t-p_bar);
    rho_min = MIN(rho_min, ratio_rhop_min*rho_t);
    p_min   = MIN(p_min,   ratio_rhop_min*p_t-p_bar);
    e_stag_max = MAX(e_stag_max, ratio_estag_max*e_stag_t);
    p_break = MIN(p_break, ratio_p_break*p_t-p_bar);
 
    UpdateData(&rho_max,    REAL_FLOW, 1, "rho_max");
    UpdateData(&p_max,      REAL_FLOW, 1, "p_max");
    UpdateData(&e_stag_max, REAL_FLOW, 1, "e_stag_max");
    UpdateData(&rho_min,    REAL_FLOW, 1, "rho_min");
    UpdateData(&p_min,      REAL_FLOW, 1, "p_min");
    UpdateData(&p_break,    REAL_FLOW, 1, "p_break");
    
    mflog::log.set_one_processor_out();
    mflog::log<<SEP_LINE<<endl;
    mflog::log<<"rho_min    = " << IOS_EP(6) << rho_min << endl;
    mflog::log<<"rho_max    = " << IOS_EP(6) << rho_max << endl;
    mflog::log<<"p_min      = " << IOS_EP(6) << p_min << endl;
    mflog::log<<"p_max      = " << IOS_EP(6) << p_max << endl;
    mflog::log<<"e_stag_max = " << IOS_EP(6) << e_stag_max << endl;
    mflog::log<<"p_break    = " << IOS_EP(6) << p_break << endl;
    mflog::log<<SEP_LINE<<endl;
}


/******************************************************************************\
      determine machine zero for use in setting tolerances
      (10.**(-iexp) is machine zero) 
\******************************************************************************/
void Zone::ComputeMachineZero()
{
    IntType i,n,iexp=15;
    RealFlow add,x11,compare; 

    compare = 1.0;
    for(i=0;i<50;i++){
        add = 1.0;
        for(n=0;n<i+1;n++)
            add *= 0.1; 
        x11 = compare +add;
        if(x11 == compare){
            iexp = i+1;
            break;
        }
    }
  
    UpdateData(&iexp, INT, 1, "iexp");

    mflog::log.set_one_processor_out();
    mflog::log<<endl<<"the machine zero = "<<-iexp<<endl; 
}


/******************************************************************************\
      
\******************************************************************************/
void Zone::GetGridDir()
{
    String filename;
    GetData(&filename, STRING, 1, "gridname");
    IntType filecharnum = static_cast<IntType> (strlen(filename));
    IntType fn = 0;
    for(IntType jf=filecharnum-1; jf>=0; jf--)
    {
        if(filename[jf]=='/')
        {
            fn = jf;
            fn++;
            break;
        }
    }
    String filename_sel;
    strcpy (filename_sel,""); 
    strncat(filename_sel,filename,fn);
    UpdateData(filename_sel, STRING, 1, "griddir");
}


/************************************************************************\
                        针对LUSGS，进行网格排序
\************************************************************************/
void Zone::ReorderCellforLUSGS()
{
    IntType i;
    
    IntType nGrids = GetNoOfGrids();
    Grid **grids = GetGrids();

    for(i=0;i<nGrids;i++){
#if (defined CellColoring)  
    /**************************Add by DZ 2021-8-20***************************/
    //((PolyGrid *)grids[i])->ReorderCellforLUSGS_1();  
    //((PolyGrid *)grids[i])->ReorderCellforLUSGS_2();
    //((PolyGrid *)grids[i])->ReorderCellforLUSGS_3();
    ((PolyGrid *)grids[i])->LUSGSGridColor( 1 );      //cell coloring in LUSGS
    //find cell color for lusgs
    //0 or 1 to be chosen in param where 0 is imbalance greedy algorithm,1 is balance algorithm
    //((PolyGrid *)grids[i])->SparseRecurrence_globalSyn(); 

    //((PolyGrid *)grids[i])->exportLayers();
    //((PolyGrid *)grids[i])->ComputeBSRindex( );
#else
        ((PolyGrid *)grids[i])->ReorderCellforLUSGS_0();  //no reorder
#endif
    /**************************Add by DZ 2021-8-20**************************/
    }
}

void Zone::PartitionGrids( IntType *CellToZone,IntType n_zone )
{
    PolyGrid *grid = (PolyGrid *) grids[nGrids-1];

    grid->PartitionGrids(CellToZone,n_zone);
}

#undef CPP_FILD_ID  // clear out file id
} // ~namespace mflow
