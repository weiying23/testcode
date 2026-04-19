//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   io_grid.cpp
/// \brief  functions for grid output
/// \author tangj, zhangyb, leebin, zhengming
/// \date   2020-01-12
/// \copyright  C.All rights reserved. 2020-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

// direct head file
#include "io_grid.h"

// C++ build-in head files
#include <assert.h>  // assert()
#include <stdlib.h>  // exit()
#include <stdio.h>   // printf()
#include <iostream>  // std::cout
#include <algorithm> // std::swap(), std::reverse()
#include <cmath>     // std::pow()

// Head files of third-part libraries
#include "cgnslib.h"

// Other head files of MFlow
#include "algm.h"
#include "constant.h"
#include "memory_util.h"
#include "io_base_format.h"
#include "io_log.h"
#include "system_base_functions.h"

// head files relying on condition-compiling
#ifdef MPICH
#include <mpi.h>
#endif

#ifdef FS_OPENMP
#include <omp.h>
#endif

namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 10902  // define file id

#ifdef MPICH
    extern int myZone;
    extern int numprocs;
    extern MPI_Comm GridComm;  //for each grid
#endif

namespace GridIO
{
using std::cout;
using std::endl;

/*******************************************************************************
*  Read grid in CGNS format: version 3.                                        *
*  Note:  After reading, the boundary face normals all point outward           *
*******************************************************************************/
#define ADF_NAME_LENGTH  32
void ReadCGNSGrid(PolyGrid *grid, const string &filename, ExtraGridData &extra_grid_data)
{
    // make sure grid is a real object
    assert(grid != NULL);

    mflog::log.set_one_processor_out();

    int ier, cg_in;
    ier = cg_open(filename.c_str(), CG_MODE_READ, &cg_in);
    if(ier != 0) 
    {
        std::cerr << "Cannot open CGNS file: " << filename << endl;
        cg_error_exit();
    } 
    else 
    {
        mflog::log << "Open CGNS file: " << filename << endl;
    }
    
    float version;
    ier = cg_version(cg_in, &version);
    printf("ier = %d  version = %5.2f\n", ier, version);
    
    //CGNS库信息，节点：CGNSBase_t
    int nBase = 0;
    ier = cg_nbases(cg_in, &nBase);
    if (nBase > 1) 
    {
        printf("\n ier = %d  nBase = %d\n",ier,nBase);
        printf("Need new code !!\n");
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    
    char basename[ADF_NAME_LENGTH];
    int cell_dim, phys_dim;
    ier = cg_base_read(cg_in, nBase, basename, &cell_dim, &phys_dim);
    
    //族，节点：Family_t
    int nfamilies, *nFamBC = NULL, *nGeo = NULL;
    ier = cg_nfamilies(cg_in, nBase, &nfamilies);
    if(nfamilies > 0) 
    {
        mfmem::snew_array_1D(nFamBC, nfamilies, dmrfl);
        mfmem::snew_array_1D(nGeo, nfamilies, dmrfl);
        BCType_t *famBC = NULL;
        mfmem::snew_array_1D(famBC, nfamilies, dmrfl);
        char **famname = NULL;
        mfmem::snew_array_2D(famname,nfamilies, ADF_NAME_LENGTH, dmrfl, true);
        char fambc_name[ADF_NAME_LENGTH];
        for(int fam = 0; fam < nfamilies; ++fam) 
        {
            int jfam = fam+1;
            ier = cg_family_read(cg_in, nBase, jfam, famname[fam], &nFamBC[fam], &nGeo[fam]);
            if(nFamBC[fam] == 1) 
            {
                ier = cg_goto(cg_in, nBase, "Family_t", jfam, "end");
                ier = cg_famname_read(famname[fam]);
                ier = cg_fambc_read(cg_in, nBase, jfam, 1, fambc_name, &famBC[fam]);
            }
        }
        mfmem::sdel_array_2D(famname);
        mfmem::sdel_array_1D(nFamBC);
        mfmem::sdel_array_1D(nGeo);
        mfmem::sdel_array_1D(famBC);
    }
    
    //模拟类型，节点：SimulationType_t
    SimulationType_t SimuType;
    ier = cg_simulation_type_read(cg_in, nBase, &SimuType);
    
    //块信息，节点：Zone_t
    int nZone;
    ier = cg_nzones(cg_in, nBase, &nZone);
    if(nZone != 1)
    {
        std::cerr <<"nZone = "<<nZone<<endl;
        std::cerr <<"Maybe you should merging this mesh to one zone!"<<endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }

    ZoneType_t zonetype;
    ier = cg_zone_type(cg_in, nBase, nZone, &zonetype);
    printf("\n zonetype = %i (%s) \n", zonetype, ZoneTypeName[zonetype]);
    if(zonetype != Unstructured)
    {
        std::cerr <<"This is not a unstructured zone!"<<endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }

    char zonename[ADF_NAME_LENGTH];
    cgsize_t size[3];
    ier = cg_zone_read(cg_in, nBase, nZone, zonename, size);
    printf("\n zonename=%s  size of pnts=%ld vols=%ld %ld \n", zonename, size[0], size[1], size[2]);
    
    //块Zone的网格坐标，节点：GridCoordinates_t
    int ngrids,ncoords;
    ier = cg_ngrids(cg_in, nBase, nZone, &ngrids);
    if(ngrids > 1)
    {
        printf("\n Need new code for ngrids=%d \n", ngrids);
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    ier = cg_ncoords(cg_in, nBase, nZone, &ncoords);
    if(ncoords != 3)
    {
        printf("\n ncoords=%d \n", ncoords);
    }

    char GridCoordName[ADF_NAME_LENGTH];
    ier = cg_grid_read(cg_in, nBase, nZone, ngrids, GridCoordName);
    if(ier != 0) printf("\n GridCoordName = %s \n", GridCoordName);
    
    DataType_t datatype;  // Realsingle or RealDouble
    char coordname[ADF_NAME_LENGTH];
    ier = cg_coord_info(cg_in, nBase, nZone, ncoords, &datatype, coordname);
    if (datatype !=4) 
    {
        printf("\n datatype = %d coordname = %s \n", datatype, coordname);
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    
    cgsize_t rmin, rmax;
    rmin = 1;
    rmax = size[0];
    RealGeom **xyz = NULL;
    mfmem::snew_array_2D(xyz, ncoords, static_cast<size_t>(rmax), dmrfl, true);
    for (IntType C = 0; C < ncoords; C++) 
    {
        ier = cg_coord_info(cg_in, nBase, nZone, C+1, &datatype, coordname);
        if(ier!=0) printf("\n datatype = %d coordname = %s \n", datatype, coordname);
        
        ier = cg_coord_read(cg_in, nBase, nZone, coordname, datatype, &rmin, &rmax, xyz[C]);
        if(ier!=0) printf("\n rang = %ld - %ld coord = %le \n", rmin, rmax, xyz[C][0]);
    }
    
    //单元数据，节点：Elements_t
    int nsections;
    ier = cg_nsections(cg_in, nBase, nZone, &nsections);
    if(ier!=0) printf("\n nsections = %d \n", nsections);
    
    cgsize_t *start = NULL;  // id range, base 1
    cgsize_t *end   = NULL;
    mfmem::snew_array_1D(start, nsections, dmrfl);
    mfmem::snew_array_1D(end, nsections, dmrfl);
    char **sectionname = NULL;
    mfmem::snew_array_2D(sectionname, nsections, ADF_NAME_LENGTH, dmrfl, true);
    ElementType_t *elemtype = NULL;
    cgsize_t **cg_elements  = NULL;
    cgsize_t *elementdatasize = NULL;
    cgsize_t **parentdata     = NULL;
    mfmem::snew_array_1D(elemtype, nsections, dmrfl);
    mfmem::snew_array_1D(cg_elements, nsections, dmrfl);
    mfmem::snew_array_1D(elementdatasize, nsections, dmrfl);
    mfmem::snew_array_1D(parentdata, nsections, dmrfl);
    for(int S=0; S < nsections; S++) 
    {
        int nbndry, parent_flag;
        ier = cg_section_read(cg_in, nBase, nZone, S+1, sectionname[S], &(elemtype[S]), &(start[S]), &(end[S]), &nbndry, &parent_flag);
        if(ier != 0)
        {
            printf("\n sectionname = %s elemtype = %d start = %d end = %d parent_flag = %d \n",sectionname[S], elemtype[S], start[S], end[S], parent_flag);
        }

        ier = cg_ElementDataSize(cg_in, nBase, nZone, S+1, &(elementdatasize[S]));
        cg_elements[S] = NULL;
        parentdata[S]  = NULL;
        mfmem::snew_array_1D(cg_elements[S], static_cast<size_t>(elementdatasize[S]), dmrfl);
        mfmem::snew_array_1D(parentdata[S], static_cast<size_t>(4*elementdatasize[S]), dmrfl);
        ier = cg_elements_read(cg_in, nBase, nZone, S+1, cg_elements[S], parentdata[S]);
        mfmem::sdel_array_1D(parentdata[S]);
    }
    mfmem::sdel_array_1D(parentdata);

    //边界条件类型和位置，节点：BC_t
    int nbocos;
    ier = cg_nbocos(cg_in, nBase, nZone, &nbocos);
    
    DataType_t normaldatatype;
    int normalindex, ndataset;
    cgsize_t normallistflag;

    char **boconame = NULL;   // boundary name(from user) for each BC
    char **BCfamname = NULL;  // ??
    mfmem::snew_array_2D(boconame, nbocos, ADF_NAME_LENGTH, dmrfl, true);
    mfmem::snew_array_2D(BCfamname, nbocos, ADF_NAME_LENGTH, dmrfl, true);
    BCType_t *bocotype = NULL;  // Boundary type for each BC, such as WALL, SYMM
    PointSetType_t *BC_ptset_type = NULL;  // PointRange or PointList
    cgsize_t *npnts = NULL;    // numbers for each BC, 2 for PointRange, variable for PointList
    mfmem::snew_array_1D(bocotype, nbocos, dmrfl);
    mfmem::snew_array_1D(BC_ptset_type, nbocos, dmrfl);
    mfmem::snew_array_1D(npnts, nbocos, dmrfl);
    cgsize_t **pnts = NULL;   // elements range or list of ids for each BC
    mfmem::snew_array_1D(pnts, nbocos, dmrfl);
    for(int BC = 0; BC < nbocos; BC++)
    {
        ier = cg_boco_info(cg_in, nBase, nZone, BC+1, boconame[BC],
                           &(bocotype[BC]), &BC_ptset_type[BC], &npnts[BC],
                           &normalindex, &normallistflag, &normaldatatype, &ndataset);
        pnts[BC] = NULL;
        mfmem::snew_array_1D(pnts[BC],static_cast<size_t>(npnts[BC]),dmrfl);
        ier = cg_boco_read(cg_in, nBase, nZone, BC+1, pnts[BC], NULL);
        ier = cg_goto(cg_in, nBase, "Zone_t", nZone, "ZoneBC_t", 1, "BC_t", BC+1, "end");
        ier = cg_famname_read(BCfamname[BC]);
    }

    GridLocation_t GridLocation;
    String *VpatchName = NULL;
    mfmem::snew_array_1D(VpatchName,nbocos,dmrfl);    

    for (IntType BC = 0; BC < nbocos; BC++)
    {
        strcpy(VpatchName[BC], boconame[BC]);

        ier = cg_goto(cg_in, nBase, "Zone_t", nZone, "ZoneBC_t", 1, "BC_t", BC+1, "end"); // NULL) 
        ier = cg_gridlocation_read(&GridLocation);
        if(ier==0) 
        {
            // Vertex ,FaceCenter and CellCenter are the three candidate types for
            // 3D unstructured mesh.
            // Others, such as IFaceCenter and KFaceCenter are for structured mesh.            

            // When PointRange or PointList is used, the choice between vertex or face
            // indices is determined by the value of GridLocation_t. When ElementRange
            // or ElementList is used, GridLocation_t is ignored. (from p6 of Standard
            // Interface Data Structure, Document version 3.1.7, CGNS version 3.1.3).

            bool regular_type = false;
            if (BC_ptset_type[BC]==ElementRange || BC_ptset_type[BC]==ElementList)
            {
                regular_type = true;
            }
            else
            {
                // Here we use ElementRange or ElementList for FaceCenter BC, while
                // PointRange or PointList for Vertex BC.
                if (GridLocation==FaceCenter && BC_ptset_type[BC]==PointRange ) 
                {
                    BC_ptset_type[BC] = ElementRange;
                    regular_type = true;
                }
                if (GridLocation==FaceCenter && BC_ptset_type[BC]==PointList ) 
                {
                    BC_ptset_type[BC] = ElementList;
                    regular_type = true;
                }
                if (GridLocation==Vertex && BC_ptset_type[BC]==PointRange ) 
                {
                    regular_type = true;
                }
                if (GridLocation==Vertex && BC_ptset_type[BC]==PointList ) 
                {
                    regular_type = true;
                }
            }        
            // we can not deal with other cases now. 
            if (!regular_type)
            {
                std::cerr << "Unknown type for BC " << BC << " in CGNS file, need new code" << std::endl;
                std::cerr << "GridLocation is " <<  GridLocation << ", points set type is " << BC_ptset_type[BC] << std::endl;
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
    }
    
    int n1to1;
    ier = cg_n1to1(cg_in, nBase, nZone, &n1to1);
    if (n1to1>0) printf("\n n1to1=%d \n", n1to1);
    
    int nconns;
    ier = cg_nconns(cg_in, nBase, nZone, &nconns);
    if (nconns>0)
    {
        printf("\n Need new code for nconns=%d \n", nconns);
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    
    //读取原始网格完毕
    ier = cg_close(cg_in);   
    
    mfmem::sdel_array_2D(boconame);
    
    cgsize_t MaxCell=0;  // maximum id for all elements
    for (int S = 0; S < nsections; S++) 
    {
        if (elemtype[S]<TRI_3) continue;
        if (MaxCell<end[S]) MaxCell=end[S];
    }
    IntType *nPFC = NULL;  // number of points for each element
    mfmem::snew_array_1D(nPFC, static_cast<size_t>(MaxCell+1),dmrfl);

    IntType nT_tet = 0, nT_pris = 0, nT_pyr = 0, nT_hex = 0; // NO. of 3D cells
    IntType nT_tri = 0, nT_quad = 0; // NO. of 2D faces
    IntType nT_lin = 0, nT_pnt = 0;  // 

    for (IntType S=0; S<nsections; S++) 
    {
        if (elemtype[S]<TRI_3) 
        {
            continue;
        } 
        else if (elemtype[S]==MIXED) 
        {
            int iele_nod;
            cgsize_t count=0;
            ElementType_t tmp_elemtype;
            for(cgsize_t i=start[S]; i<=end[S]; i++) 
            {
                tmp_elemtype = (ElementType_t) cg_elements[S][count++];
                if (tmp_elemtype==HEXA_8) 
                {
                    nT_hex ++;
                } 
                else if (tmp_elemtype==PENTA_6) 
                {
                    nT_pris ++;
                } 
                else if (tmp_elemtype==PYRA_5) 
                {
                    nT_pyr ++;
                } 
                else if (tmp_elemtype==TETRA_4) 
                {
                    nT_tet ++;
                } 
                else if (tmp_elemtype==QUAD_4) 
                {
                    nT_quad ++;
                } 
                else if (tmp_elemtype==TRI_3) 
                {
                    nT_tri ++;
                } 
                else 
                {
                    std::cerr << "Need new code ! tmp_elemtype="<<tmp_elemtype<<endl;
                    mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
                }
                ier = cg_npe(tmp_elemtype, &iele_nod);
                nPFC[i] = iele_nod;
                count += iele_nod;
            }
        } 
        else if (elemtype[S]==HEXA_8) 
        {
            nT_hex += static_cast<IntType> ((end[S]-start[S]+1));
            for(cgsize_t i=start[S]; i<=end[S]; i++) nPFC[i] = 8;
        } 
        else if (elemtype[S]==PENTA_6) 
        {
            nT_pris += static_cast<IntType> ((end[S]-start[S]+1));
            for(cgsize_t i=start[S]; i<=end[S]; i++) nPFC[i] = 6;
        } 
        else if (elemtype[S]==PYRA_5) 
        {
            nT_pyr += static_cast<IntType> ((end[S]-start[S]+1));
            for(cgsize_t i=start[S]; i<=end[S]; i++) nPFC[i] = 5;
        } 
        else if (elemtype[S]==TETRA_4) 
        {
            nT_tet += static_cast<IntType> ((end[S]-start[S]+1));
            for(cgsize_t i=start[S]; i<=end[S]; i++) nPFC[i] = 4;
        } 
        else if (elemtype[S]==QUAD_4) 
        {
            nT_quad += static_cast<IntType> ((end[S]-start[S]+1));
            for(cgsize_t i=start[S]; i<=end[S]; i++) nPFC[i] = 4;
        } 
        else if (elemtype[S]==TRI_3) 
        {
            nT_tri += static_cast<IntType> ((end[S]-start[S]+1));
            for(cgsize_t i=start[S]; i<=end[S]; i++) nPFC[i] = 3;
        } 
        else 
        {
            std::cerr << "Need new code ! elemtype[S]="<<elemtype[S]<<endl;
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }       
        
    } // Finished: for (IntType S = 1; S <= nsections; S++)
    
    
    nPFC[0] = 0;
    for(cgsize_t i=1; i<=MaxCell; i++) 
    {
        nPFC[i] += nPFC[i-1];
    }

    cgsize_t *elements = NULL;  // connectivities (Cell to Nodes) for all elements
    mfmem::snew_array_1D(elements, nPFC[MaxCell],dmrfl);
    for( IntType S = 0; S < nsections; S++) 
    {
        if (elemtype[S]<TRI_3) 
        {
            continue;
        } 
        else if (elemtype[S]==MIXED) 
        {
            int iele_nod;
            cgsize_t count=0;
            ElementType_t tmp_elemtype;
            for(cgsize_t i=start[S]-1; i<end[S]; i++) 
            {
                tmp_elemtype = (ElementType_t) cg_elements[S][count++];
                ier = cg_npe(tmp_elemtype, &iele_nod);
                for (IntType j=0; j<iele_nod; j++) 
                {
                    elements[nPFC[i]+j] = cg_elements[S][count++];
                }
            }
        } 
        else 
        {
            for(cgsize_t i=0; i<elementdatasize[S]; i++) 
            {
                elements[nPFC[start[S]-1]+i] = cg_elements[S][i];
            }
        }
    }
    mfmem::sdel_array_1D(elemtype);
    mfmem::sdel_array_1D(elementdatasize);
    
    for(int S=0; S<nsections; S++) mfmem::sdel_array_1D(cg_elements[S]);
    mfmem::sdel_array_1D(cg_elements);
    IntType nTCell_3d = static_cast<IntType> (size[1]);
    IntType nTCell_2d = static_cast<IntType> (MaxCell-size[1]);
    
    //边界条件类型和位置，节点：BC_t
    IntType *bc_pnt = NULL;
    mfmem::snew_array_1D(bc_pnt, static_cast<size_t>(rmax+1), dmrfl);
    IntType *bct = NULL;
    mfmem::snew_array_1D(bct, nTCell_2d, dmrfl);  // BC id for each face element
    IntType nbct=0;
    for (IntType BC = 0; BC < nbocos; BC++)
    {       
        switch( BC_ptset_type[BC] ) 
        {
            case ElementRange:  //ElementRange , npnts=2
                if ( pnts[BC][0]>nTCell_3d+nTCell_2d ) continue;
                nbct += static_cast<IntType> (pnts[BC][1]-pnts[BC][0]+1);
                for (cgsize_t j=pnts[BC][0]; j<=pnts[BC][1]; j++)
                    bct[j - nTCell_3d - 1] = BC;
                break;

            case ElementList:  //ElementList
                if ( pnts[BC][0]>nTCell_3d+nTCell_2d ) continue;
                nbct += static_cast<IntType> (npnts[BC]);
                for (cgsize_t j=0; j<npnts[BC]; j++)
                    bct[pnts[BC][j] - nTCell_3d - 1] = BC;
                
                break;
            case PointList:  //PointList
                for (IntType j=0; j<=rmax; j++) bc_pnt[j] = 0;
                for (IntType j=0; j<npnts[BC]; j++)
                    bc_pnt[ pnts[BC][j] ] = 1;      //bc_cgns[bocotype];
                for (IntType j=nTCell_3d; j<nTCell_3d+nTCell_2d; j++)
                {
                    IntType nptbc=0;
                    for (IntType jj=nPFC[j]; jj<nPFC[j+1]; jj++)
                        nptbc += bc_pnt[elements[jj]];
                    if (nptbc == nPFC[j+1]-nPFC[j]) {
                        bct[j - nTCell_3d] = BC;
                         nbct++;
                    }
                }
                break;
            case PointRange:  //PointRange
                for (IntType j=0; j<=rmax; j++) bc_pnt[j] = 0;
                for (cgsize_t j=pnts[BC][0]; j<pnts[BC][1]; j++)
                    bc_pnt[ j ] = 1;      //bc_cgns[bocotype];
                for (IntType j=nTCell_3d; j<nTCell_3d+nTCell_2d; j++)
                {
                    IntType nptbc=0;
                    for (IntType jj=nPFC[j]; jj<nPFC[j+1]; jj++)
                        nptbc += bc_pnt[elements[jj]];
                    if (nptbc == nPFC[j+1]-nPFC[j]) {
                        bct[j - nTCell_3d] = BC;
                        nbct++;
                    }
                }
                break;
            default:
                printf("\n Need new code for new BC_ptset_type=%d \n",BC_ptset_type[BC]);
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }   
    }

    for (IntType BC = 0; BC < nbocos; BC++) 
    {
        mfmem::sdel_array_1D(pnts[BC]);
    }
    mfmem::sdel_array_1D(pnts);
    mfmem::sdel_array_2D(sectionname);
    mfmem::sdel_array_2D(BCfamname);
    mfmem::sdel_array_1D(bc_pnt);
    mfmem::sdel_array_1D(start);
    mfmem::sdel_array_1D(end);
    mfmem::sdel_array_1D(bocotype);
    mfmem::sdel_array_1D(BC_ptset_type);
    mfmem::sdel_array_1D(npnts);
    
    printf("\n nTCell_2d=%ld nbct=%ld \n",(long)nTCell_2d, (long)nbct);
    if ( nTCell_2d-nbct!=0 ) mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    
    if(ier!=0) printf("\n ier=%d\n",ier);

    mflog::log << "Successfully read cgns_file " << filename << endl;
    
    printf("\n 3D elements: %ld tetra %ld pyramid %ld prism %ld hex \n", (long)nT_tet, (long)nT_pyr, (long)nT_pris, (long)nT_hex);
    printf(" Others: %ld point %ld line %ld quad %ld tri \n", (long)nT_pnt, (long)nT_lin, (long)nT_quad, (long)nT_tri);
        
    IntType nBFace, nTNode,nTFace;
    nBFace = nT_quad + nT_tri;
    nTNode = static_cast<IntType> (rmax);
    nTFace = 4*nT_tet + 5*nT_pyr + 5*nT_pris + 6*nT_hex + nT_tri + nT_quad;
    nTFace = nTFace/2;
    printf("\n nBFace=%ld nTNode=%ld nTFace=%ld nTCell_3d=%ld \n", (long)nBFace, (long)nTNode, (long)nTFace, (long)nTCell_3d);
    
    //Now find the connectivity between elements
    IntType *vfnod   = NULL;
    IntType *vfacnod = NULL;
    IntType *facnod  = NULL;
    IntType *nNPF    = NULL;
    mfmem::snew_array_1D(vfnod  ,4*nT_tet+5*nT_pyr+5*nT_pris+6*nT_hex+1,dmrfl);
    mfmem::snew_array_1D(vfacnod,12*nT_tet+16*nT_pyr+18*nT_pris+24*nT_hex,dmrfl);
    mfmem::snew_array_1D(facnod ,nTFace+1,dmrfl);
    mfmem::snew_array_1D(nNPF   ,nTFace,dmrfl);
    
    for (IntType i=0; i<nBFace; i++)
    {
        nNPF[i] = nPFC[i+1+nTCell_3d]-nPFC[i+nTCell_3d];
    }
    
    facnod[0] = 0;
    for(IntType i=0; i<nBFace; i++) 
    {
        facnod[i+1] = facnod[i] + nNPF[i];
    }
    printf("\n facnod[%ld]=%ld \n",(long)nBFace,(long)facnod[nBFace]);
    
    IntType *fnod = NULL;
    mfmem::snew_array_1D(fnod,6*nT_tet+8*nT_pyr+9*nT_pris+12*nT_hex+facnod[nBFace]/2,dmrfl);
    for (IntType j = 0; j < facnod[nBFace]; j++)
    {        
        fnod[j] = static_cast<IntType> (elements[nPFC[nTCell_3d]+j]);
    }
    for (IntType i = 0; i < nBFace; i++)
    {
        ReorderPnts(&fnod[facnod[i]], nNPF[i]);
    }
    
    IntType *f2cr = NULL;
    IntType *f2cl = NULL;
    mfmem::snew_array_1D(f2cr, nTFace, dmrfl);
    mfmem::snew_array_1D(f2cl, nTFace, dmrfl);
    
    IntType kk ;
    for(kk=0;kk<nTFace;kk++){
        f2cl[kk] = -nTFace; //-i-1;
        f2cr[kk] = -nTFace;
    }
    
    for (IntType i = 0; i < nBFace; i++)
    {
        f2cl[i] =  -bct[i] - 1; 
    }
    
    IntType *rl_fac = NULL;
    IntType *celfac = NULL;
    mfmem::snew_array_1D(rl_fac, 4*nTFace, dmrfl);
    mfmem::snew_array_1D(celfac, nTCell_3d+1, dmrfl);
    celfac[0] = 0;  //face number of cell
    
    IntType nfc = 0;
    vfnod[0] = 0;
    
    printf("\n nTCell_3d = %ld \n", (long)nTCell_3d);
    for (IntType i = 0; i < nTCell_3d; i++)
    {
        switch(nPFC[i+1]-nPFC[i])
        {
            case 8:
                /*六面体   GridGen和ICEM输出的CGNS格式相同

                     7------6   
                    /|     /|    F0473        
                   / |    / |    F1265        
                  /  3---/--2    F0321  ->OUT 
                 4--/---5  /     F4567        
                 | /    | /      F0154        
                 |/     |/       F2376        
                 0------1                     

                */                                        
            {
                celfac[i+1]=celfac[i]+6;

                vfnod[nfc+1]=vfnod[nfc]+4;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+0 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+4 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+7 ]);
                vfacnod[ vfnod[nfc]+3 ] = static_cast<IntType> (elements[ nPFC[i]+3 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],4 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+4;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+1 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+2 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+6 ]);
                vfacnod[ vfnod[nfc]+3 ] = static_cast<IntType> (elements[ nPFC[i]+5 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],4 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+4;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+0 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+3 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+2 ]);
                vfacnod[ vfnod[nfc]+3 ] = static_cast<IntType> (elements[ nPFC[i]+1 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],4 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+4;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+4 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+5 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+6 ]);
                vfacnod[ vfnod[nfc]+3 ] = static_cast<IntType> (elements[ nPFC[i]+7 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],4 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+4;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+0 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+1 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+5 ]);
                vfacnod[ vfnod[nfc]+3 ] = static_cast<IntType> (elements[ nPFC[i]+4 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],4 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+4;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+2 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+3 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+7 ]);
                vfacnod[ vfnod[nfc]+3 ] = static_cast<IntType> (elements[ nPFC[i]+6 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],4 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;
                    
                    
                break;
            }
            case 6:       
                // 三棱柱    GridGen和ICEM输出的CGNS格式相同
                /*              2——————5                                     
                                │.         │\                                    
                                │ .        │ \             面021 指外            
                                │  .       │  \            面345 指外            
                                │    ......│...\           面0352指外            
                                │   .1     │   /4          面5412指外            
                                │  .       │  /            面0143指外            
                                │ .        │ /                                   
                                │.         │/                                    
                                0——————3                                     
                */
            {
                celfac[i+1]=celfac[i]+5;

                vfnod[nfc+1]=vfnod[nfc]+4;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+0 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+3 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+5 ]);
                vfacnod[ vfnod[nfc]+3 ] = static_cast<IntType> (elements[ nPFC[i]+2 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],4 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+4;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+5 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+4 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+1 ]);
                vfacnod[ vfnod[nfc]+3 ] = static_cast<IntType> (elements[ nPFC[i]+2 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],4 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+4;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+0 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+1 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+4 ]);
                vfacnod[ vfnod[nfc]+3 ] = static_cast<IntType> (elements[ nPFC[i]+3 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],4 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+3;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+0 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+2 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+1 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],3 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+3;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+3 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+4 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+5 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],3 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                break;
            }
            case 5:         // 金字塔
                /*五面体（金字塔）          GridGen和ICEM输出的CGNS格式相同 
                   4                  
                 / |.\       
                /  | .\                面0321指外
               0---|-3 \               面014 指外
                \  |  . \              面124 指外
                 \ |   . \             面234 指外
                  \|    . \            面304 指外
                   1------2
                */
            {
                celfac[i+1]=celfac[i]+5;

                vfnod[nfc+1]=vfnod[nfc]+4;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+0 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+3 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+2 ]);
                vfacnod[ vfnod[nfc]+3 ] = static_cast<IntType> (elements[ nPFC[i]+1 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],4 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+3;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+0 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+1 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+4 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],3 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+3;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+1 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+2 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+4 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],3 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+3;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+2 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+3 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+4 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],3 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+3;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+3 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+0 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+4 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],3 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                break;
            }
            case 4:
                /* 四面体
                     0                     面021 指外
                   / | \                   面123 指外
                  /  |  \                  面032 指外
                 3---|---1                 面013 指外
                  \  |   /               
                   \ |  /                
                    \| /        
                     2                   
                */                                        
            {
                celfac[i+1]=celfac[i]+4;

                vfnod[nfc+1]=vfnod[nfc]+3;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+0 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+2 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+1 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],3 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+3;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+1 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+2 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+3 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],3 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+3;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+0 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+3 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+2 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],3 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                vfnod[nfc+1]=vfnod[nfc]+3;
                vfacnod[ vfnod[nfc]+0 ] = static_cast<IntType> (elements[ nPFC[i]+0 ]);
                vfacnod[ vfnod[nfc]+1 ] = static_cast<IntType> (elements[ nPFC[i]+1 ]);
                vfacnod[ vfnod[nfc]+2 ] = static_cast<IntType> (elements[ nPFC[i]+3 ]);
                ReorderPnts( &vfacnod[vfnod[nfc]],3 );
                rl_fac[(nfc<<1)  ]= -nTFace;
                rl_fac[(nfc<<1)+1]= i;
                nfc=nfc+1;

                break;
            }
            default:
            {
                printf("i = %ld\n", (long)i);
                printf("need new code for the elements of no_tetra");
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
    }
    printf("\n nfc=%ld \n", (long)nfc);
    
    // 建立Face中最小序号点的索引，边界面不计入
    IntType *nod_vfac,*n_vfac;
    nod_vfac = NULL;
    mfmem::snew_array_1D(nod_vfac, nTNode+1,dmrfl);
    for (IntType i=0; i<nTNode+1; i++) 
    {
        nod_vfac[i] = 0;
    }
    for (IntType jfc=0; jfc<4*nT_tet+5*nT_pyr+5*nT_pris+6*nT_hex; jfc++)      //计算所有点被几个面的第一个点共用
    {
        nod_vfac[ vfacnod[vfnod[jfc]] ]++; 
    }
    for (IntType i=1; i<nTNode+1; i++) 
    {
        nod_vfac[i] += nod_vfac[i-1];
    }
    printf("\n nod_vfac[%ld]=%ld \n", (long)nTNode,(long)nod_vfac[nTNode]);
    
    //依照面的第一个点的序号,将面分组排列
    n_vfac = NULL;
    mfmem::snew_array_1D(n_vfac, 4*nT_tet+5*nT_pyr+5*nT_pris+6*nT_hex,dmrfl);
    for (IntType i=0; i<4*nT_tet+5*nT_pyr+5*nT_pris+6*nT_hex; i++)
    {
        n_vfac[i] = 0;
    }
    
    {
        IntType *itmp = NULL;
        mfmem::snew_array_1D(itmp, nTNode,dmrfl);
        for (IntType i=0; i<nTNode; i++)
        {
            itmp[i] = 0;
        }
        for (IntType jfc=0; jfc<4*nT_tet+5*nT_pyr+5*nT_pris+6*nT_hex; jfc++)
        {
            IntType pt0=vfacnod[vfnod[jfc]];
            IntType jj=nod_vfac[pt0-1]+itmp[pt0];
            if (n_vfac[jj]==0)
            {
                n_vfac[ jj ] = jfc;
            }
            else
            {
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
            itmp[ pt0 ] ++;
        }
        mfmem::sdel_array_1D(itmp);
    }
    
    IntType nowfac;
    
    // TODO: move snew_array_1D() out of the cycle using maximum size, tangj
    // 对Cell的Face定位
    for (nowfac=0; nowfac<nBFace; nowfac++)
    {
        IntType *ptw = NULL;
        mfmem::snew_array_1D(ptw, nNPF[nowfac],dmrfl);
        for ( IntType i=facnod[nowfac]; i<facnod[nowfac+1]; i++)
        {
            ptw[i-facnod[nowfac]] = fnod[i];
        }
        for ( IntType i=nod_vfac[ptw[0]-1]; i<nod_vfac[ptw[0]]; i++)
        {
            nfc=n_vfac[i];
            if ( nNPF[nowfac] != vfnod[nfc+1]-vfnod[nfc] || rl_fac[(nfc<<1)]!=-nTFace )
            {
                continue;
            }
            IntType *ndw = NULL;
            mfmem::snew_array_1D(ndw, vfnod[nfc+1]-vfnod[nfc],dmrfl);
            for (IntType j=vfnod[nfc]; j<vfnod[nfc+1]; j++)
            {
                ndw[j-vfnod[nfc]] = vfacnod[j];
            }
            
            IntType ie=nNPF[nowfac];
            IntType now_ok;
            for (IntType iw=1; iw<ie; iw++)
            {
                now_ok=0;
                for (IntType jw=1; jw<ie; jw++)
                {
                    if (ptw[iw]==ndw[jw])
                    {
                        now_ok=1;
                        break;
                    }
                }
                if (now_ok==0)
                    break;
            }
            mfmem::sdel_array_1D(ndw);
            if (now_ok==1)
            {
                rl_fac[(nfc<<1)] = -nowfac-1 ;
                break;
            }
        }
        mfmem::sdel_array_1D(ptw);        
    }
    
    IntType nr_err=0;
    for (nowfac=0; nowfac<4*nT_tet+5*nT_pyr+5*nT_pris+6*nT_hex; nowfac++)
    {
        if (rl_fac[2*nowfac]!=-nTFace) continue;
        
        IntType *ptw;
        ptw = NULL;
        mfmem::snew_array_1D(ptw,vfnod[nowfac+1]-vfnod[nowfac],dmrfl);
        for ( IntType j=vfnod[nowfac]; j<vfnod[nowfac+1]; j++)
        {
            ptw[j-vfnod[nowfac]] = vfacnod[j];
        }
        
        for ( IntType i=nod_vfac[ptw[0]-1]; i<nod_vfac[ptw[0]]; i++)
        {
            nfc=n_vfac[i];
            if ( vfnod[nowfac+1]-vfnod[nowfac] != vfnod[nfc+1]-vfnod[nfc]
                || rl_fac[(nfc<<1)]!=-nTFace || nowfac==nfc )
            {
                continue;
            }
                
            IntType *ndw = NULL;
            mfmem::snew_array_1D(ndw, vfnod[nfc+1]-vfnod[nfc],dmrfl);
            for (IntType j=vfnod[nfc]; j<vfnod[nfc+1]; j++)
            {
                ndw[j-vfnod[nfc]] = vfacnod[j];
            }
            
            IntType ie=vfnod[nfc+1]-vfnod[nfc];
            IntType nod_ok=1;
            for (IntType iw=1; iw<ie; iw++)
            {
                if (ptw[iw]==ndw[ie-iw])
                {
                    nod_ok = nod_ok + 1;
                }
                else
                {
                    break;
                }
            }
            mfmem::sdel_array_1D(ndw);
            if (nod_ok==ie)
            {
                rl_fac[(nfc<<1)   ] = rl_fac[2*nowfac+1];
                rl_fac[2*nowfac] = rl_fac[(nfc<<1)   +1];
                break;
            }           
        }
        
        if (rl_fac[2*nowfac]==-nTFace)
        {
            nr_err++;
            printf("Cannot find the right side %ld \n",(long)nowfac);
            for ( IntType j=vfnod[nowfac]; j<vfnod[nowfac+1]; j++) 
            {
                printf(" %ld %f %f %f \n ", (long)vfacnod[j],xyz[0][ptw[0]],xyz[1][ptw[0]],xyz[2][ptw[0]]);
            }
            printf("\n");
        }
        mfmem::sdel_array_1D(ptw);        
    }

    if(nr_err>0) 
    {
        printf("\n Waring! nr_err=%ld \n",(long)nr_err);
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    
    for (IntType i=0; i<4*nT_tet+5*nT_pyr+5*nT_pris+6*nT_hex; i++)
    {
        if (rl_fac[2*i]==-nTFace || rl_fac[2*i]==rl_fac[2*i+1] )
        {
            printf (" error rl_fac %ld  %ld \n",(long)i, (long)rl_fac[2*i] );
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
    
    for (nfc=0; nfc<4*nT_tet+5*nT_pyr+5*nT_pris+6*nT_hex; nfc++)
    {
        if (rl_fac[(nfc<<1)  ] < 0 )
        {
            nowfac = -rl_fac[(nfc<<1)]-1;
            for (IntType j=vfnod[nfc]; j<vfnod[nfc+1]; j++)
            {
                fnod[facnod[nowfac]+j-vfnod[nfc]] = vfacnod[j];
            }
            f2cl[nowfac] = rl_fac[(nfc<<1)  ];
            f2cr[nowfac] = rl_fac[(nfc<<1)+1];
            facnod[nowfac+1] = facnod[nowfac] + nNPF[nowfac]; 
        }
    }
            
    for (IntType i=0; i < nBFace; i++)
    {
        f2cl[i] =  -bct[i] - 1; 
    }

    mfmem::sdel_array_1D(bct);
    nowfac = nBFace;
    for (nfc=0; nfc<4*nT_tet+5*nT_pyr+5*nT_pris+6*nT_hex; nfc++)
    {
        if (rl_fac[(nfc<<1)  ] >= 0 && rl_fac[(nfc<<1)  ] < rl_fac[(nfc<<1)+1] )
        {
            for (IntType j=vfnod[nfc]; j<vfnod[nfc+1]; j++)
                fnod[facnod[nowfac]+j-vfnod[nfc]]=vfacnod[j];
            f2cl[nowfac] = rl_fac[(nfc<<1)  ];
            f2cr[nowfac] = rl_fac[(nfc<<1)+1];
            nNPF[nowfac] = vfnod[nfc+1] - vfnod[nfc];
            facnod[nowfac+1] = facnod[nowfac] + nNPF[nowfac]; 
            nowfac ++;
        }
    }
    printf("Max faces be %ld \n",(long)nowfac);

    for (IntType i = 0; i < nBFace; i++)
    {
        if (f2cr[i]==-nTFace)
        {
            printf("f2cr[%ld]=%ld \n ",(long)i,(long)f2cr[i]);
        }
    }
    for (IntType i = nBFace; i < nTFace; i++)
    {
        if (f2cr[i]==-nTFace || f2cl[i]==-nTFace)
        {
            printf("f2cr[%ld]=%ld ,f2cl[%ld]=%ld\n ",(long)i,(long)f2cr[i], (long)i,(long)f2cl[i]);
        }
    }
                
    for(IntType i = 0; i < nBFace; i++)  ++f2cr[i];
    for(IntType i = nBFace; i < nTFace; i++)
    {
        ++f2cr[i];
        ++f2cl[i];
    }
    mfmem::sdel_array_1D(vfnod);
    mfmem::sdel_array_1D(vfacnod);
    mfmem::sdel_array_1D(celfac);

    printf("It is all right !\n");

    //
    // Set information for grid                
                                        
    grid->SetNTNode(nTNode);
    grid->SetNTFace(nTFace);
    grid->SetNTCell(nTCell_3d);
    grid->SetNBFace(nBFace);                    
                    
    printf("Grid has %ld points, %ld faces and %ld cells\n", (long)nTNode, (long)nTFace, (long)nTCell_3d);
                                  
    RealGeom *x = NULL;
    RealGeom *y = NULL;  
    RealGeom *z = NULL;
    mfmem::snew_array_1D(x, nTNode, dmrfl);
    mfmem::snew_array_1D(y, nTNode, dmrfl);
    mfmem::snew_array_1D(z, nTNode, dmrfl);
    grid->SetX(x);
    grid->SetY(y);
    grid->SetZ(z);
                    
    for(IntType i=0; i<nTNode; i++) 
    {
        x[i]  = RealGeom(xyz[0][i]);
        y[i]  = RealGeom(xyz[1][i]);
        z[i]  = RealGeom(xyz[2][i]);
    }
    mfmem::sdel_array_2D(xyz);
                    
    //set nNPF and f2n
    grid->SetnNPF(nNPF);
                    
    IntType n = facnod[nTFace];
    IntType *f2n = NULL;
    mfmem::snew_array_1D(f2n, n, dmrfl);
    grid->Setf2n(f2n);
                    
    for(IntType i = 0; i < n; i++) 
    {
        f2n[i] = fnod[i];
        (f2n[i])--;     // now node index starts from 0,网格是从1开始的
    }
    mfmem::sdel_array_1D(facnod);
    mfmem::sdel_array_1D(fnod);
                    
    //set nNPC and C2N
    IntType *nNPC = NULL;
    mfmem::snew_array_1D(nNPC, nTCell_3d, dmrfl);
    grid->SetnNPC(nNPC);
    assert(nNPC != 0);
                    
    for(IntType i = 0; i < nTCell_3d; i++)
    {
        nNPC[i] = nPFC[i+1] - nPFC[i];
    }
                    
    // Allocate memories for cell to node connection
    IntType **C2N = NULL;
    mfmem::snew_array_2D(C2N, nTCell_3d, nNPC, dmrfl, true);
    grid->SetC2N(C2N);
    IntType count = 0;
    for(IntType i = 0; i < nTCell_3d; i++)
    {
        // now node index starts from 0,网格是从1开始的
        for(IntType j = 0; j < nNPC[i]; j++) C2N[i][j] = static_cast<IntType> (--elements[count++]); 
    }
    mfmem::sdel_array_1D(nPFC);
    mfmem::sdel_array_1D(elements);
                    
    //set f2c
    n = nTFace<<1;
    IntType * f2c = NULL;
    mfmem::snew_array_1D(f2c,n,dmrfl);
    grid->Setf2c(f2c);
                    
    IntType n_patch = 0;
    count = 0;
                    
    for(IntType i = 0; i < nBFace; i++) 
    {
        f2c[count++] = --f2cr[i];
        f2c[count++] = f2cl[i];
        n_patch = MAX(n_patch, -f2cl[i]);
    }

    for(IntType i = nBFace; i < nTFace; i++) 
    {
        f2c[count++] = --f2cr[i];
        f2c[count++] = --f2cl[i];
    }

    ExtraGridData::BCNamesType &bc_names = extra_grid_data.bc_patch_names;
    bc_names.resize(n_patch);
    for(IntType ix = 0; ix < n_patch; ++ix)
    {
        bc_names[ix] = VpatchName[ix];
    }

    mfmem::sdel_array_1D(f2cr);
    mfmem::sdel_array_1D(f2cl);
    mfmem::sdel_array_1D(nod_vfac);
    mfmem::sdel_array_1D(n_vfac);
    mfmem::sdel_array_1D(rl_fac);
    mfmem::sdel_array_1D(VpatchName);

    printf("Grid has %ld boundary faces\n", (long)nBFace);
                
    //网格重排序
    //IntType IReorder=0;
    //if(IReorder) {
    //    for(IntType g=0; g<ngrids; g++) {
    //        grid = (PolyGrid *)GetGrid(g);
    //        grid->ReorderCell();
    //        printf("Finished Reorder all cells\n");
    //    }
    //}
    
    // split loop face to two faces
    BreakFaceLoop(grid);
}
#undef ADF_NAME_LENGTH
/*******************************************************************************
*  Read grid of MFlow binary format generated by MFlow preprocessing           *
*  All coarse grids are also read if mg > 1.                                   *
*******************************************************************************/
void ReadMFlowGrid_binary(PolyGrid **grids, const IntType mg, const string &filename, ExtraGridData &extra_grid_data)
{
    // make sure grid is a real object
    assert(grids != NULL);
    for (IntType g = 0; g < mg; ++g) assert(grids[g] != NULL);

    IntType mgrid = 1;  // levels of multi-grid

    FILE *fp = NULL;
    if((fp = fopen(filename.c_str(), "rb")) == NULL)
    {
        std::cerr << "Could not open file " << filename << endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }

    mflog::log.set_one_processor_out();
    mflog::log << endl << "Reading binary grids of MFlow format" << endl;

    PolyGrid *grid = NULL;
    for(IntType g = 0; g < mgrid; g++) 
    {
        grid = grids[g];
        grid->SetLevel(g);

        IntType nTNode, nTFace, nTCell;
        fread(&nTNode, sizeof(IntType), 1, fp);
        fread(&nTFace, sizeof(IntType), 1, fp);
        fread(&nTCell, sizeof(IntType), 1, fp);
        grid->SetNTNode(nTNode);
        grid->SetNTFace(nTFace);
        grid->SetNTCell(nTCell);

#ifdef DEBUG
        printf("Grid%d has %ld points, %ld faces and %ld cells\n",
                (int)(g+1),(long)nTNode, (long)nTFace, (long)nTCell);
#endif

        RealGeom *x = NULL;
        RealGeom *y = NULL;
        RealGeom *z = NULL;
        mfmem::snew_array_1D(x, nTNode, dmrfl);
        mfmem::snew_array_1D(y, nTNode, dmrfl);
        mfmem::snew_array_1D(z, nTNode, dmrfl);
        grid->SetX(x);
        grid->SetY(y);
        grid->SetZ(z);

        fread(x, sizeof(RealGeom), nTNode, fp);
        fread(y, sizeof(RealGeom), nTNode, fp);
        fread(z, sizeof(RealGeom), nTNode, fp);

        IntType *facnod = NULL;
        IntType *nNPF   = NULL;
        mfmem::snew_array_1D(facnod, nTFace+1, dmrfl);
        mfmem::snew_array_1D(nNPF, nTFace, dmrfl);
        fread(nNPF, sizeof(IntType), nTFace, fp);
        grid->SetnNPF(nNPF);

        facnod[0] = 0;
        IntType n = 0;
        for(IntType i = 0; i < nTFace; ++i)
        {
            n += nNPF[i];
            facnod[i+1] = n;
        }
        IntType *f2n = NULL;
        mfmem::snew_array_1D(f2n, n, dmrfl);
        grid->Setf2n(f2n);
        fread(f2n, sizeof(IntType), n, fp);

        n = 2*nTFace;
        IntType *f2c = NULL;
        mfmem::snew_array_1D(f2c, n, dmrfl);
        grid->Setf2c(f2c);
        fread(f2c, sizeof(IntType), n, fp);
        
        IntType nBFace = 0;
        IntType count = 0;

        for(IntType i = 0; i < nTFace; ++i)
        {
            IntType c1 = count++;
            IntType c2 = count++;

            if(f2c[c1] < 0)  //面的左边是物理边界
            {
                // need to reverse the node ordering, 保证点的顺序为右手系
                // Note: the first point of face will not move.
                std::reverse(&(f2n[facnod[i]+1]), &(f2n[facnod[i+1]]));

                // exchange c1 and c2
                std::swap(f2c[c1], f2c[c2]);

                ++nBFace;
            }
            else if(f2c[c2] < 0)  //面的右边是物理边界
            {
                ++nBFace;
            }
        }
        mfmem::sdel_array_1D(facnod);

        grid->SetNBFace(nBFace);

        //see if any interfaces 
        if(!feof(fp))   //exist interfaces
        {
            IntType nIFace,nINode;
            IntType *nbz, *nbf, *nbsN, *nbzN, *nbrN;

            fread(&nIFace, sizeof(IntType), 1, fp);
            grid->SetNIFace(nIFace); 
            nbz = NULL; //zhyb: 对应分区的区号
            nbf = NULL; //zhyb: 对应的面号
            mfmem::snew_array_1D(nbz, nIFace,dmrfl);
            mfmem::snew_array_1D(nbf, nIFace,dmrfl);
            fread(nbz, sizeof(IntType), nIFace, fp);
            fread(nbf, sizeof(IntType), nIFace, fp); 

            grid->SetnbZ(nbz);
            grid->SetnbBF(nbf);

            fread(&nINode, sizeof(IntType), 1, fp);
            grid->SetNINode(nINode);
            nbsN = NULL; // 本块中的并行点序号
            nbzN = NULL; // 对应点分区的区号
            nbrN = NULL; // 对应点在对应分区中的序号
            mfmem::snew_array_1D(nbsN, nINode,dmrfl);
            mfmem::snew_array_1D(nbzN, nINode,dmrfl);
            mfmem::snew_array_1D(nbrN, nINode,dmrfl);
            fread(nbsN, sizeof(IntType), nINode, fp);
            fread(nbzN, sizeof(IntType), nINode, fp);
            fread(nbrN, sizeof(IntType), nINode, fp);

            grid->SetnbSN(nbsN);
            grid->SetnbZN(nbzN);
            grid->SetnbRN(nbrN);
        }
        
#ifdef REORDER
        IntType methodsReorder = 1;
        switch (methodsReorder) {
        case 1:
            grid->CellReordering_CMK();
            break;
        case 2:
            grid->CellReordering_morton();
            break;
        case 3:
            grid->CellReordering_metis();
            break;
        case 4:
            grid->CellReordering_scotch();
            break;
        default:
            grid->CellReordering_CMK();
        }
        grid->Update_f2c(); 
        grid->FaceReordering();
#endif


#ifdef DC0
		CreateTree(grid);
		uTaskTree_CellReordering(grid);
        uTaskTree_NodeReordering(grid);
		uTaskTree_FaceReordering(grid);
#endif

		/* create C2F by ori f2c*/
#ifndef DC0
#if (defined REORDER) || (defined FaceColoring) || (defined GroupColor)
		{
		IntType c1, c2, count;
		IntType nTCell = grid->GetNTCell();
		IntType nBFace = grid->GetNBFace();
		IntType nTFace = grid->GetNTFace();
		IntType* f2c = grid->Getf2c();
		IntType* nFPC = CalnFPC(grid);
		grid->C2F_ori = NULL;
		mfmem::snew_array_2D(grid->C2F_ori, nTCell, nFPC, dmrfl, true);
		// Need to reset nFPC to 0 and recover it later
		for (IntType i = 0; i < nTCell; i++) nFPC[i] = 0;

		// Boundary faces
		for (IntType i = 0; i < nBFace; i++)
		{
			c1 = f2c[2 * i];
			grid->C2F_ori[c1][nFPC[c1]++] = i;
		}
		// Interior faces
		count = 2 * nBFace;
		for (IntType i = nBFace; i < nTFace; i++)
		{
			c1 = f2c[count++];
			c2 = f2c[count++];
			grid->C2F_ori[c1][nFPC[c1]++] = i;
			grid->C2F_ori[c2][nFPC[c2]++] = i;
		}
		grid->index_fr = NULL;
		}
#endif
#endif

#ifdef FaceColoring
        
//#ifdef FaceColoringBalancing
        //FaceColouringBalancing(grid);
//#else
		IntType igroupsize, bgroupsize;
        igroupsize = 204800;
        bgroupsize = 10240;
        FaceColouring(grid, bgroupsize, igroupsize);
//#endif

#endif
        grid->GroupColorSuccess = false;
#if (defined GroupColor) //dingxin
        grid->GroupColorSuccess = GroupColoring(grid, true);
        if (!grid->GroupColorSuccess) {
#ifdef MPICH
            mflog::log << "Rank " << myZone << " skip group coloring ." << std::endl;
#else
            mflog::log << "Skip group coloring ." << std::endl;
#endif
        }
#if (defined FS_CUDA)||(defined FS_CUDA_DEBUG_NS_Flux)
		if (grid->GroupColorSuccess) {
			GroupColoringGPU(grid);
		}
		else{
			cout << "Fail group coloring ." << endl;
			exit(0);
		}
#endif	
#endif

#if (defined FS_OPENMP) && (defined DIVREP) //dingxin
        grid->threads = omp_get_max_threads();
        grid->DivRepSuccess = DivRep(grid);
        if(!grid->DivRepSuccess)
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    #if (defined FS_SIMD) && (defined BoundedColoring)
        grid->endIndex_bFace_vec = NULL;
        grid->endIndex_iFace_vec = NULL;
        mfmem::snew_array_1D(grid->endIndex_bFace_vec, grid->threads, dmrfl);
        mfmem::snew_array_1D(grid->endIndex_iFace_vec, grid->threads, dmrfl);
        colorAfterDivRep(grid);
    #endif
#endif

#if (defined FS_OPENMP) && (defined DIVCON)//dingxin
        DC_create_tree(grid);
#endif
    }

    fclose(fp);

    // figure out how many physical boundary patches in the grid
    // TODO: We can read it from the file in future
    // The boundary faces did not merge when we get coarse grid, so n_patch is same for all level of grids.
    IntType n_patch = 0;
    PolyGrid *fine_grid = grids[0];
    IntType nBFace = fine_grid->GetNBFace();
    IntType nIFace = fine_grid->GetNIFace();
    IntType *f2c   = fine_grid->Getf2c();
    for(IntType i = 0; i < nBFace - nIFace; ++i)
    {
        n_patch = std::max(n_patch, -f2c[i*2+1]);        
    }

#ifdef MPICH
    IntType n_patch_local = n_patch;
    MPI_Allreduce(&n_patch_local, &n_patch, 1, MPIIntType, MPI_MAX, GridComm);
#endif

    // There is no patch name in MFlow grid format, so we
    // just only attach an distinguishing name for each patch
    // TODO: patch names will be also write in the mmgrid file, so we can
    // read them in future.
    ExtraGridData::BCNamesType &bc_names = extra_grid_data.bc_patch_names;
    bc_names.resize(n_patch);
    for(IntType bc = 0; bc < n_patch; ++bc)
    {
        string bc_name = "BC";
        bc_name += int2str(bc+1);
        bc_names[bc] = bc_name;
    }
}

} // ~namespace GridIO



#undef CPP_FILD_ID  // clear out file id
} // ~namespace mflow
