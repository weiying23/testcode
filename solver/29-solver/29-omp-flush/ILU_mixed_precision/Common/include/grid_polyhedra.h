//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   grid_polyhedra.h
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

#ifndef MFL_GRID_POLYHEDRA_H
#define MFL_GRID_POLYHEDRA_H

#include <vector>
#include <iostream>
#include <cstdlib>
using namespace std;

#include "number_type.h"
#include "constant.h"
#include "data_pool.h"
#include "boundary_condition.h"
#include "grid_base.h"
#include "memory_util.h"
#include "metis.h"

#ifdef MPICH
#include <mpi.h>
#endif

#ifdef DC0
#include "uTaskTree.h"
#endif

namespace mflow
{

//=============================================================================
//                          Class PolyGrid
//=============================================================================
class PolyGrid : public Grid
{
public:
    typedef   vector<IntType> VecInt;
    //xchf: for face colouring
    std::vector<IntType> bfacegroup;
    std::vector<IntType> ifacegroup;

    //dingxin: for group coloring
    const static IntType groupSize = 512;
    bool GroupColorSuccess;
	//dingxin: for face reorder
    IntType** C2F_ori;
    IntType* index_fr; // face index after face reorder, old -> new
    //dingxin: for divide and replicate
    IntType* idx_pthreads_bface;
    IntType* id_division_bface;
    IntType* idx_pthreads_iface;
    IntType* id_division_iface;
    IntType* endIndex_bFace_vec;
    IntType* endIndex_iFace_vec;
    IntType threads;
    bool DivRepSuccess;
	//dingxin: for divide and conquer approach
    IntType* cellPerm;
    tree_t* treeHead;
	
	// to load cell values into share memory:
	IntType *group_b_SMc2c;
	IntType *group_i_SMc2c;
	// to store the share memory unit index:
	IntType *group_b_SM_index;
	IntType *group_i_SM_index;
	// to store the share memory index between colors:
	IntType *group_b_SM_color_index;
	IntType *group_i_SM_color_index;
	// to load face values into share memory cell values:
	IntType *group_b_f2SMc;
	IntType *group_i_f2SMc;
	
#if (defined FS_SIMD) && (defined FS_SIMD_AVX) && (defined FaceColoring)
    //for FaceColoring SIMD, based SSE, ruitian, 2021.12.27
    RealGeom* xfntile, * yfntile, * zfntile;
    RealGeom* areatile, * qsumtile;
    RealGeom* dqdxtile, * dqdytile, * dqdztile;
    IntType* f2c1, *f2c2;
#elif (defined FS_SIMD) && (defined Tile)
    //for Tile SIMD, based SSE, ruitian, 2021.12.27
    //for boundary faces, not including nIFace
    IntType* bfacerow;
    IntType* bfacecol;
    IntType* bfaceval;
    IntType bfacennz;
    //for interior faces
    IntType* ifacerow;
    IntType* ifacecol;
    IntType* ifaceval;
    IntType ifacennz;
    //tile for SIMD, ruitian, 2021.12.19, Multi-thread and Multi-lane tiling
    //for boundary faces, not including nIFace, Multi-thread and Multi-lane execution
    IntType* bSIMDrow;
    IntType* bSIMDcol;
    IntType* bSIMDval;
    IntType bSIMDnnz;
    IntType* bfacezero;
    vector<vector<int> >* boffsets;
    //for interior faces, nTface-nBFace
    IntType* iSIMDrow;
    IntType* iSIMDcol;
    IntType* iSIMDval;
    IntType iSIMDnnz;
    IntType* ifacezero;
    std::vector<vector<int> >* ioffsets;

    RealGeom* qsumtile;
    RealGeom* dqdxtile, * dqdytile, * dqdztile;   
    RealGeom* xfnt, * yfnt, * zfnt;
    RealGeom* areat, * qsumt;
    IntType* nNPFt;
    //add by ruitian
#endif 
private:

    // 
    // kernel grid information for face based FVM method
    // 
#ifdef DC0
	uTaskTree *uTaskTreeRoot = NULL;
#endif

    IntType   level;          // multi-grid level, level 0 being the finest
    IntType   nTFace, nTCell; // no. of total faces, no of total cells
    IntType   nBFace;         // boundary faces which include interfaces
    IntType   nIFace;         // parallel interface faces(zero if serial program)
    IntType   nINode;         // parallel interface nodes(zero if serial program)

    IntType   *nNPF,*f2n,**F2N; // no. of node per face and face->node connectivity
    IntType   *f2c;             // face->cell connectivity

    RealGeom  *xcc, *ycc, *zcc;  // cell center data, including ghosts
    RealGeom  *xfc, *yfc, *zfc;  // face center data;
    RealGeom  *xfn, *yfn, *zfn;  // face unit normal;
    RealGeom  *area, *vol;       // face area, cell volume excluding ghosts;

    BCRecord  **bcr;          // associate each boundary. face with a record

    // Auxiliary connectivities
    IntType   *nCPC,**c2c;      // no of neighbor cells per cell, and cell->cell
    IntType   *nNPC,**C2N;
    IntType   *nFPC,**C2F;
    IntType   *nCPN,**N2C;
    
    // Neighbor information for zones
    IntType   nNeighbor;      // no of zonal neighbors linked by face 
    IntType   nNeighborN;     // no of zonal neighbors linked by node 
    
    // neighbor information of Zones
    IntType   *nb;            // the neighbor zone/rank number for face
    IntType   *nbN;           // the neighbor zone/rank number for node
    PolyGrid  **nbg;          // ??? not used

    // parallel information for each interface (saved in grid file of 'mmgrid' type)
    IntType   *nbZ, *nbBF;    // parallel information for interfaces: zone No. and face No. in that zone. Size is nIFace.
    IntType   *nbSN, *nbZN, *nbRN;  // parallel information for inter-nodes: nodes' id on current zone, zone No. and nodes' id on other zone. Size is nINode.    
    // Actual parallel information used to communication through MPI.
    IntType   *nZIFace;       // No. of interfaces for each neighbor    
    IntType   **bCNo;         // cell number in the current zone for each neighbor, each interface, ordered
                              // according to the current zone. Used to send data on cell center to other processor.
    IntType   **bFNo;         // boundary face number in the current zone for each neighbor, each interface, ordered
                              // according to the neighbor zone. Used to receive data on cell center from other processor.
    IntType   *nZINode;       // No. of inter-nodes for each neighbor
    IntType   **bNSNo;        // node number for sending data on nodes for each neighbor, each node.
    IntType   **bNRNo;        // node number for receiving data on nodes for each neighbor, each node.

    
    // Active and interpolation information for overlap grid
    IntType   *node_act, *edge_act; // the active mark of nodes, cells, faces and edges.
    IntType   *nChCellS;      // No. of cells sending to other processors whose grid is different.
    IntType   *nChCellR;      // No. of cells receiving from other processors whose grid is different.
    IntType   **nChSNo, **nChRNo;  // The cells id to send and receive 
    RealGeom  **nChxcc, **nChycc, **nChzcc; // Centroid of cells to interpolation on other processors
    IntType   *nChRNoComp;    // Cells id which have no donor-cells on current processor
    
    // Zone id for all cells used to initialize with different value for different zone.
    IntType   *Cell2Zone;

    // grid quality
    RealGeom  *facecentroidskewness;
    RealGeom  *faceangleskewness;
    RealGeom  *cellcentroidskewness;
    RealGeom  *cellvolsmoothness;
    RealGeom  *faceangle;
    IntType   *cellwallnumber;

    // flow reconstruction from cell to node
    IntType   *Nmark;                           // Node type
    RealGeom  *WeightNodeDist, **WeightNodeC2N, **WeightNodeN2C; // weight of node-distance
    RealGeom** WeightNodeBFace2C; //weight of node-distance for BFace to related Cells

    // weight for prolongation from coarse to fine grid  
    RealGeom  *WeightNodeProl;

    // Central Scheme using the face order, 0 for the first cell,
    // 1 for the next cell
    RealGeom  **Seta_Center;

    // moving grid
    RealGeom  *vgn;  // normal velocity for all faces
    RealGeom  *BFacevgx,*BFacevgy,*BFacevgz;  // 3 components of velocity for boundary faces.
    RealGeom  *vccx, *vccy, *vccz;  //velocity at the cell center, for preconditioning of unsteady flow. 
    
    //additional grid information for efficiency
    RealGeom  VolAvg;  // cell's average volume, using for venkatakrishnan's limiter
    IntType* order_cell_oTon;		//dingxin:the order of cell,old->new
    IntType* order_cell_nToo;		//dingxin:the order of cell,new->old
    IntType * ghost2global;

public:    

#ifdef DC0
	void       SetuTaskTree(uTaskTree *treeRoot);
	uTaskTree  *GetuTaskTree() const;
#endif

    void       SetNTFace(const IntType in);
    void       SetNTCell(const IntType in);
    void       SetNBFace(const IntType in);
    void       SetNIFace(const IntType in);
    void       SetNINode(const IntType in);

    IntType    GetNTFace() const;
    IntType    GetNTCell() const;
    IntType    GetNBFace() const;
    IntType    GetNIFace() const;
    IntType    GetNINode() const;

    void       SetnNPF(IntType *in);
    void       SetnFPC(IntType *in);
    void       SetnNPC(IntType *in);
    void       Setf2n(IntType *in);
    void       Setf2c(IntType *in);

    // Get and Set the cell center information
    RealGeom  *GetXcc() const;
    RealGeom  *GetYcc() const;
    RealGeom  *GetZcc() const;
    
    // Get and set cell volume     
    RealGeom  *GetCellVol() const;

    // Get and Set the face center information
    RealGeom  *GetXfc() const;
    RealGeom  *GetYfc() const;
    RealGeom  *GetZfc() const;

    // Get and Set the face normal information
    RealGeom  *GetXfn() const;
    RealGeom  *GetYfn() const;
    RealGeom  *GetZfn() const;
    
    // Get and set face area     
    RealGeom  *GetFaceArea() const;


    void      SetnCPC(IntType *in);
    void      SetnCPN(IntType* in);


    void      Setc2c(IntType **in);
    void      SetC2N(IntType **in);
    void      SetC2F(IntType **in);
    void      SetN2C(IntType** in);
    // F2N is special and just a reference to f2n, so use 1D array operator, tangj 
    void        SetF2N(IntType **in);

    IntType     *GetnNPF() const;
    IntType     *GetnFPC() const;
    IntType     *GetnNPC() const;
    IntType     *Getf2n() const;
    IntType     *Getf2c() const;

    IntType     *GetnCPC() const;
    IntType     *GetnCPN() const;
    IntType     **Getc2c() const;
    IntType     **GetC2N() const;
    IntType     **GetC2F() const;
    IntType     **GetF2N() const;
    IntType     **GetN2C() const;

    void        Setc2cc(IntType *in);

    void        SetNumberOfFaceNeighbors(const IntType n);
    IntType     GetNumberOfFaceNeighbors() const;
    void        SetNumberOfNodeNeighbors(const IntType n);
    IntType     GetNumberOfNodeNeighbors() const;

    void        SetFaceNeighborZones(IntType *fnz);
    IntType    *GetFaceNeighborZones() const;
    void        SetNodeNeighborZones(IntType *nnz);
    IntType    *GetNodeNeighborZones() const;
    void        SetNeighborGrids(PolyGrid **grids);   

    void        SetnbZ(IntType *in);
    IntType     *GetnbZ() const;
    void        SetnbBF(IntType *in);
    IntType     *GetnbBF() const;
    void        SetnbSN(IntType *in);
    IntType     *GetnbSN() const;
    void        SetnbZN(IntType *in);
    IntType     *GetnbZN() const;
    void        SetnbRN(IntType *in);
    IntType     *GetnbRN() const;
    IntType     *Getface_act()  const;

    void        Setcoe_trans(RealGeom *in);
    RealGeom    *Getcoe_trans() const;
  
    IntType     GetLevel() const;
    void        SetLevel(IntType lin);

    void        Setbcr(BCRecord **bcrs);
    BCRecord    **Getbcr() const;
  
    void        SetVolAvg(RealGeom in);
    RealGeom    GetVolAvg() const;
    
    RealGeom   *GetGridQualityFaceCentroidSkewness(void) const;

    IntType    *GetGridQualityCellWallNumber(void) const;
    // flow reconstruction from cell to node
    void        SetNodeType(IntType *node_type);
    IntType    *GetNodeType(void) const;
    void        SetWeightNodeDist(RealGeom *weight);
    RealGeom   *GetWeightNodeDist(void) const;
    void        SetWeightNodeC2N(RealGeom **weight_c2n);
    RealGeom  **GetWeightNodeC2N(void) const;
    void        SetWeightNodeN2C(RealGeom** weight_n2c);
    RealGeom** GetWeightNodeN2C(void) const;
    void        SetWeightNodeBFace2C(RealGeom** WeightNodebFace2c);
    RealGeom** GetWeightNodeBFace2C(void) const;

    // moving grid
    RealGeom   *GetFaceNormalVelocity(void) const;
    RealGeom   *GetBoundaryFaceVelocityX(void) const;
    RealGeom   *GetBoundaryFaceVelocityY(void) const;
    RealGeom   *GetBoundaryFaceVelocityZ(void) const;

    void ComputeMetrics();
    void InitialVgn();
    void ComputeDist2WallTriang(RealGeom *dist2wall_cell, IntType mark);
    void WriteInfoDist();
    void ComputeCellDist();

    void SetUpComm();
    void SetUpComm_Node();
    void Set_RecvSend(RealFlow ***qs, RealFlow ***qr,IntType nvar);
    void Add_RecvSend(RealFlow ***qs, RealFlow *q, IntType num_var);
    void Read_RecvSend(RealFlow ***qr, RealFlow *q, IntType num_var);
    void Add_MatrixRecvSend(RealFlow ***bqs, RealFlow *q, IntType nvar);
    void Read_MatrixRecvSend(RealFlow ***bqr, RealFlow *q, IntType nvar);

    void Set_MatrixRecvSend(MATRIXTYPE **bqs, MATRIXTYPE **bqr, IntType nvar);
    void Add_MatrixRecvSend2(MATRIXTYPE **bqs, MATRIXTYPE *q, IntType nvar);
    void Read_MatrixRecvSend2(MATRIXTYPE **bqr, MATRIXTYPE *q, IntType nvar);
    void CommInterfaceData(IntType zone, PolyGrid *grid, const char *name);
    void CommCellCenterData(IntType zone, PolyGrid *grid);

    void FindCellLayerNo();
    void FindNormalFace();

    void quick_sort(IntType *a, IntType is, IntType ie);
    void swap(IntType *a, IntType i, IntType j);

    // Reorder cell for LUSGS
    void ReorderCellforLUSGS_0();

    //check grid quality
    void CheckGridQuality();
    void SkewnessSummary();
    void EquiangleSkewnessSummary();
    void SmoothnessSummary();
    void FindIllWallCell();
    void CheckSymmetryFace();
    void CheckGridScale();
    void FaceAngleSummary();
    
    // deal bad grid for robust
    void DealBadGrid();
    
    //compute some additional info for geomety for efficiency 
    void AdditionalInfoForGeometry();
    RealGeom  CalculateVolumnAverage();
    RealGeom *CalNormalDistanceOfC2C();

    void PartitionGrids(IntType *CellToZone, IntType n_zone);
    void Getxadjadjncy(idx_t *xadj, idx_t *adjncy, idx_t *adjwgt);
    void SerialMetis(idx_t *xadj, idx_t *adjncy, idx_t *adjwgt, IntType n_zone, IntType *CellToZone);
    IntType     *GetCell2Zone() const
    {
        return Cell2Zone;
    }
    
    /***********************Add by DZ 2021-8-20***************************/
    void ReorderCellforLUSGS_1( );
    void ReorderCellforLUSGS_2( );
    void ReorderCellforLUSGS_3( );
    void LUSGSGridColor(IntType colort);
    void SparseRecurrence_globalSyn( );
    void SparseRecurrence_NonGlobalSyn( );
    void printcsrMatrix(IntType *row_ptr, IntType *col_ind, IntType nnz, IntType n, IntType nVar );
    void exportLayers();
    void ComputeBSRindex( );
    IntType * CalGhost2Global(IntType Bstart);
    /**********************ReferenceÃ¯Â¼Å¡AIAA94-0645************************/

    /*add by dingxin*/
#ifdef REORDER
    void Update_f2c();
    void CellReordering_CMK();
    void CellReordering_morton();
    void CellReordering_metis();
    void CellReordering_scotch();
    void FaceReordering();
    IntType* GetNewOrder(void) const {
        return order_cell_oTon;
    }
#endif // REORDER

#ifdef MPICH
    void CommInterfaceDataMPI(IntType *q);
    void CommInterfaceDataMPI(RealFlow *q);
    void CommInternodeDataMPI(IntType *q);  // MPI synchronize node overlap attribute
    void CommInternodeDataMPI2(IntType *q); // MPI
    void CommInternodeDataMPISUM(IntType *q); // MPI Sum node value
    void CommInternodeDataMPI(RealFlow *q);
    void CommInternodeDataMPI(RealFlow *q, IntType key);
    void RecvSendVarNeighbor(RealFlow *q);
    
    void RecvSendVarNeighbor(IntType *q);

    // communicate active attribute of nodes
    void RecvSendVarNeighbor_Node(IntType *q);

    // communicate node variable and synchronize to the minimum value with some constrains
    void RecvSendVarNeighbor_Node2(IntType *q);

    // communicate and accumulate node variable
    void RecvSendVarNeighbor_NodeSUM(IntType *q);

    void RecvSendVarNeighbor_Node(RealFlow *q);
    void RecvSendVarNeighbor_Node(RealFlow *q, IntType key);
    void RecvSendVarNeighbor_Over(RealFlow ***bqs, RealFlow ***bqr, MPI_Request *req_send, MPI_Request *req_recv, MPI_Status *status_array, IntType nvar);
    void RecvSendVarNeighbor_Over2(MATRIXTYPE **bqs, MATRIXTYPE **bqr, MPI_Request *req_send, MPI_Request *req_recv, MPI_Status *status_array, IntType nvar);

    void RecvSendVarNeighbor_Togeth(IntType nvar, RealFlow **q);
    void RecvSendVarMatrixNeighbor_Togeth(IntType nvar, RealFlow *q);
    void RecvSendVarMatrixNeighbor_Togeth2(IntType nvar, MATRIXTYPE *q);
    /// communicate and update the vector's ghost data, the original working vector store the inner cells variables by nTCell*nVar layout
    void UpdateVectorGhostVar(const RealFlow * vec, RealFlow * ghosts, IntType nVar);

	
#if (defined FS_CUDA)||(defined FS_CUDA_DEBUG)
	void cuGetLength_RecvSend();
	void cuGetIndex_RecvSend(IntType *Indexbqsr, IntType nvar);
	void cuGetIndex_RecvSend2(IntType *Indexbqsr, IntType nvar);
	void cuAdd_RecvSend(RealFlow ***bqs, RealFlow *q, IntType i, IntType k);
	void cuSet_RecvSend(RealFlow ***bqs, RealFlow ***bqr, IntType nvar);
	void cuRecvSendVarNeighbor_Togeth(IntType nvar, RealFlow **q, IntType type);
	void cuRecvSendVarNeighbor_Togeth_q5(IntType nvar);
	void cuRecvSendVarNeighbor_Togeth_SA(IntType nvar);	
	
	void RecvSendVarNeighbor_Over_Gradient(RealFlow *hostbqs, RealFlow *hostbqr, RealFlow ***bqs, RealFlow ***bqr, MPI_Request *req_send, MPI_Request *req_recv, MPI_Status *status_array, IntType nvar);
#if defined MultiStream	
	// to cover InterfaceData MPI Trans by Initial Function, including of Grad, Limit Init.
	void cuRecvSendVarNeighbor_Togeth_q5ForInterfaceData_unfold(IntType nvar);
	// to cover GradientData MPI Trans by First Part of Limit Function.
	void cuRecvSendVarNeighbor_TogethForGradient_unfold(IntType nvar, RealFlow **dqdx, RealFlow **dqdy, RealFlow **dqdz);
	// to cover LimitData MPI Trans by dt.
	void cuRecvSendVarNeighbor_Togeth_q5ForLimit_unfold(IntType nvar);
	void cuRecvSendVarNeighbor_TogethForGradient_unfold_MergedLimit(IntType nvar, RealFlow **dqdx, RealFlow **dqdy, RealFlow **dqdz);
	void cuRecvSendVarNeighbor_TogethForGradient_T_InVis(IntType nvar);
	void cuRecvSendVarNeighbor_Togeth_SAForInterfaceData_unfold(IntType nvar);
#endif	
	void cuRecvSendVarNeighbor_TogethForGradient_T(IntType nvar);
	void cuRecvSendVarNeighbor_TogethForGradient_SA(IntType nvar);
	void cuRecvSendVarNeighbor_TogethForGradient_SA_MultiStream(IntType nvar);
	void cuGetLength_RecvSend_Node();
	void cuGetIndex_RecvSend_Node(IntType *Indexbqsr, IntType nvar);
	void cuGetIndex_RecvSend2_Node(IntType *Indexbqsr, IntType nvar);
#endif

#endif

    explicit PolyGrid(IntType zin);
    PolyGrid(IntType zin,IntType lin);    
    PolyGrid();
    ~PolyGrid();
};

// inline functions of class PolyGrid
#include "grid_polyhedra.inl"


//=============================================================================
//                       Auxiliary Functions for PolyGrid 
//=============================================================================

// Compute face and cell centroid by simple average all node value
void FaceCellCenterbyAverage(PolyGrid *grid,RealGeom *xfc,RealGeom *yfc,RealGeom *zfc, RealGeom *xcc,RealGeom *ycc,RealGeom *zcc);

// Compute cell center and cell volume
void CellVolCentroid(PolyGrid *grid,RealGeom *vol,RealGeom *xcc,RealGeom *ycc,RealGeom *zcc);

// Compute the normal vector, the area and face center on each cell face in 3D
void FaceAreaNormalCentroid_cycle(PolyGrid *grid,RealGeom *area,RealGeom *xfn,RealGeom *yfn,RealGeom *zfn, RealGeom *xfc,RealGeom *yfc,RealGeom *zfc);

// Correct face normal vector for face of TINY AREA
void CorrectFaceNormal(PolyGrid *grid,RealGeom *xfn,RealGeom *yfn,RealGeom *zfn);

// Correct cell centroid if it is too close to wall
void CorrectCellCentroid(PolyGrid *grid, RealGeom *xcc, RealGeom *ycc, RealGeom *zcc,
                         RealGeom *xfc,  RealGeom *yfc, RealGeom *zfc,
                         RealGeom *xfn,  RealGeom *yfn, RealGeom *zfn);

// Return an real number whose value is the max absolute value of two real number and
// whose sign is equal to the first param 'a'
RealGeom AbsMaxSignFirst(RealGeom a, RealGeom b);

// Compute the weight for grid nodes based on distance
void ComputeWeight3D_Node( PolyGrid *grid);

// Check the closure of grid
void ClosureCheck(PolyGrid *grid, RealGeom *xfn, RealGeom *area);

// Calculate face to nodes connectivity
// NOTE: F2N is a reference to f2n so as to reduce memory use. 
IntType **CalF2N(PolyGrid *grid);

// Calculate node number of each cell
IntType *CalnNPC(PolyGrid *grid);

// Calculate node to cells connectivity
IntType** CalN2C(PolyGrid* grid);

//Calculate cell number of each node
IntType* CalnCPN(PolyGrid* grid);

// Calculate cell to nodes connectivity
IntType **CalC2N(PolyGrid *grid);

// Calculate face number of each cell
IntType *CalnFPC(PolyGrid *grid);

// Calculate cell to faces connectivity
IntType **CalC2F(PolyGrid *grid);

// Calculate number of neighbor cells sharing face
IntType *CalnCPC(PolyGrid *grid);

// Calculate cell to cells connectivity
IntType **CalC2C(PolyGrid *grid);

// 
void CalCNNCF(PolyGrid *grid);

// Calculate the information of nodes on symmetry plane, 
// including type flag(1) and normal vector.
void FindNodeSYMM(PolyGrid *grid);

// Sort the sub-array of size npt
void ReorderPnts(IntType *pnt,IntType npt);

void BreakFaceLoop(PolyGrid *grid);

//added by xchf
void quicksort_int(IntType s, IntType e, IntType* indx, IntType* v);
void quicksort_vecint(IntType s, IntType e, std::vector<IntType> &order, std::vector<IntType> &vecint);
void FaceColouring(PolyGrid* grid, IntType bgroupsize, IntType igroupsize);

void FaceColouringBalancing(PolyGrid* grid);//lrt

//add by dingxin
void UpdateIface(PolyGrid* grid, IntType* index_iface);
bool GroupColoring(PolyGrid* grid, bool balanceColors);

void GroupColoringGPU(PolyGrid* grid);

void BoundedColoring(IntType* f2c, IntType* bface, IntType* iface, IntType* order_b, IntType* order_i,
    IntType n_bface, IntType n_iface, IntType& bfacenum_vec, IntType& ifacenum_vec);
bool DivRep(PolyGrid* grid);//DivideReplicate
void colorAfterDivRep(PolyGrid* grid);
void updateOrder(IntType* a, IntType* newOrder, IntType start, IntType len);
#ifdef DIVCON
void DC_create_tree(PolyGrid* grid);
void DC_permute_int_2d_array(IntType* tab, IntType* perm, IntType nbItem, IntType* dimItem, IntType offset);
void merge_permutations(IntType* elemPerm, IntType* localElemPerm, IntType globalNbElem, IntType localNbElem,
    IntType firstElem, IntType lastElem);
void DC_create_permutation(IntType* perm, idx_t* part, IntType size, IntType nbPart);
void init_dc_tree(tree_t*& tree, PolyGrid* grid, IntType* cface_parent, IntType n_cface, IntType n_cface_compute,
    IntType firstElem, IntType lastElem, IntType nbSepElem, bool isSep, bool isLeaf);
void create_elem_part(idx_t* elemPart, idx_t* nodePart, IntType* elemToNode, IntType nbElem,
    IntType* dimElem, IntType separator, IntType offset, IntType* nbLeftElem, IntType* nbSepElem);
void tree_creation(tree_t*& tree, PolyGrid* grid, IntType* elemToNode, IntType* sepToNode, idx_t* nodePart,
    IntType globalNbElem, IntType* dimElem, IntType* cface_parent, IntType n_cface, IntType n_cface_compute,
    IntType firstPart, IntType lastPart, IntType firstElem, IntType lastElem, IntType sepOffset, bool isSep);
void mesh_to_nodal(idx_t*& graphIndex, idx_t*& graphValue, IntType* elemToNode, IntType nbElem,
    IntType* dimElem, IntType nbNodes);
IntType create_sepToNode(IntType*& sepToNode, IntType* index_sepToNode, IntType* elemToNode,
    IntType firstSepElem, IntType lastSepElem, IntType* dimElem);
void sep_partitioning(tree_t*& tree, PolyGrid* grid, IntType* elemToNode, IntType globalNbElem, IntType* dimElem,
    IntType* cface_parent, IntType n_cface, IntType n_cface_compute, IntType firstSepElem, IntType lastSepElem, bool isSep);
void partitioning(IntType* elemToNode, IntType nbElem, IntType* dimElem, IntType nbNodes,
    PolyGrid* grid);
void tree_free(tree_t*& tree);
void tree_traversal(tree_t*& tree, RealFlow* dqdx, RealFlow* dqdy, RealFlow* dqdz, RealGeom* tmpxyz,
    IntType* f2c, BCRecord** bcr, IntType* nNPF, IntType** F2N, RealFlow* q_n, RealFlow* q,
    RealGeom* area, RealGeom* xfn, RealGeom* yfn, RealGeom* zfn, IntType nBFace);
void tree_traversal(tree_t*& tree, RealFlow** res, RealGeom** flux, IntType* f2c);//NS
void tree_traversal(tree_t*& tree, RealFlow** res, RealGeom** flux, IntType nVar, IntType* f2c);//SA
void tree_traversal(tree_t*& tree, RealFlow* dmax, RealFlow* dmin, RealFlow* q, IntType* f2c, BCRecord** bcr);
void tree_traversal(tree_t*& tree, RealFlow* limit, RealFlow* tmp_limit, IntType* f2c, RealGeom* xfc, RealGeom* yfc, RealGeom* zfc,
    RealGeom* xcc, RealGeom* ycc, RealGeom* zcc, RealFlow* dqdx, RealFlow* dqdy, RealFlow* dqdz, RealFlow* espcell,
    RealFlow* dmax, RealFlow* dmin, IntType nBFace);
void tree_traversal(tree_t*& tree, RealFlow* dt, RealFlow* tmp, IntType* f2c, IntType nBFace, IntType vis_run,
    RealFlow* p, RealGeom* xfn, RealGeom* yfn, RealGeom* zfn, RealGeom* xfc, RealGeom* yfc, RealGeom* zfc,
    RealGeom* xcc, RealGeom* ycc, RealGeom* zcc, RealFlow* rho, RealFlow* u, RealFlow* v, RealFlow* w,
    RealGeom* vgn, RealGeom* area, RealGeom* vol, RealFlow gam, RealFlow p_bar, IntType steady, RealFlow C,
    RealFlow* vis_l, RealFlow* vis_t, RealFlow prl, RealFlow prt);
void tree_traversal(tree_t*& tree, RealFlow** lhsmat, RealFlow* dqdl, RealFlow* dqdr, IntType* f2c, IntType* fcptr);
void tree_traversal(tree_t*& tree, RealFlow* res, RealFlow* flux, RealFlow* tem, RealFlow* tem_c2, IntType* f2c);
#endif // DIVCON

#ifdef DC0
void CreateTree(PolyGrid *grid);
void uTaskTree_CellReordering(PolyGrid *grid);
void uTaskTree_FaceReordering(PolyGrid *grid);
void uTaskTree_NodeReordering(PolyGrid *grid);
#endif // DC

} // ~namespace mflow

#endif //~MFL_GRID_POLYHEDRA_H
