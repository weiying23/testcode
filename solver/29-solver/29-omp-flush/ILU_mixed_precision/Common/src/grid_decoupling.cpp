//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   grid_decoupling.cpp
/// \brief
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
#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <queue>
#include <list>
#include <set>
#include <map>
#include <iomanip>
#include <cfloat>
#include <algorithm>
//#include<immintrin.h>

using namespace std;

// user defined head files
#include "algm.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"
#include "utility_functions.h"
#include "io_log.h"
//#include "zone.h"
//#include "solver_ns.h"
//#include "io_base_format.h"
//#include "parallel_base_functions.h"

// this header file is copied from cart

#ifdef MPICH
#include "mpi.h"
#endif

#ifdef FS_OPENMP
#include <omp.h>
#endif

#define BOUNDARYFACE 1
#define INTERIORFACE 2

namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 10714  // define file id

#ifdef MPICH
extern int myZone;
extern int numprocs;
extern MPI_Comm GridComm;  //for each grid, tangj
#endif

#ifdef GroupColor
/************************************************************************
                    Color contiguous groups of faces
                        Add by dingxin 2021-10-10
************************************************************************/
bool GroupColoring(PolyGrid* grid, bool balanceColors = false)
{
    IntType MaxColors = 320;
    IntType threads;
    IntType groupEnd, tmp, color, nColor;
    IntType groupSize = grid->groupSize;
    IntType* idxColor = NULL;

    IntType nTCell, nBFace, nTFace, nIFace;
    IntType* f2c, * f2n, * nNPF;
    IntType* index_bface = NULL;
    IntType* f2c_backup = NULL;
    IntType* nNPF_bface = NULL;
    IntType* f2n_bface = NULL;
    IntType** F2N_bface = NULL;

    IntType* index_iface = NULL;
    /*IntType* nNPF_iface = NULL;
    IntType* f2n_iface = NULL;
    IntType** F2N_iface = NULL;*/

    /*
    IntType* f2c_bface = NULL;
    IntType* index_bface_1 = NULL;
    IntType* f2c_iface = NULL;
    IntType* index_iface_1 = NULL;*/

    nTCell = grid->GetNTCell();
    nBFace = grid->GetNBFace();
    nIFace = grid->GetNIFace();
    nTFace = grid->GetNTFace();
    f2c = grid->Getf2c();
    f2n = grid->Getf2n();
    nNPF = grid->GetnNPF();

    IntType ifacenum = nTFace - nBFace;

    IntType    n = nTCell + nBFace;
    IntType pfacenum = nBFace - nIFace;//不着色并行分区边界面
#ifdef FS_OPENMP
    threads = omp_get_max_threads();
#else
    threads = 1;
#endif // FS_OPENMP
    if (pfacenum < groupSize * threads || ifacenum < groupSize * threads) {
#ifdef MPICH
        mflog::log << "Rank " << myZone << " skip group coloring ." << std::endl;
#endif
        mflog::log << "groupSize too big." << std::endl;
        return false;
    }
    //allocate memory
    mfmem::snew_array_1D(idxColor, nTFace, dmrfl);
    mfmem::snew_array_1D(index_bface, pfacenum, dmrfl);
    mfmem::snew_array_1D(f2c_backup, 2 * pfacenum, dmrfl);
    mfmem::snew_array_1D(nNPF_bface, pfacenum, dmrfl);
    mfmem::snew_array_1D(F2N_bface, pfacenum, dmrfl);

    mfmem::snew_array_1D(index_iface, ifacenum, dmrfl);

    vector<IntType> colorSize(1, 0);
    vector<IntType> colorSize_bak(MaxColors, 0);
    vector<set<IntType> > innerInColor(MaxColors);
    vector<IntType>searchOrder(MaxColors);
    /* pface color */
    nColor = 1;
    for (IntType i = 0; i < pfacenum; i += groupSize) {
        groupEnd = MIN(i + groupSize, pfacenum);
        searchOrder.resize(nColor);

        for (IntType j = 0; j < nColor; j++)
            searchOrder[j] = j;
        /*--- Balance sizes by looking for space in smaller colors first. ---*/
        if (balanceColors) {
            colorSize_bak.resize(nColor);
            colorSize_bak.assign(colorSize.begin(), colorSize.end());
            quicksort_vecint(1, nColor, searchOrder, colorSize_bak);
        }
        vector<IntType>::iterator iter_color = searchOrder.begin();
        for (; iter_color != searchOrder.end(); ++iter_color)
        {
            bool free = true;
            tmp = 2 * i;
            /*--- Traverse entire group as a large outer index. ---*/
            for (IntType j = i; j < groupEnd && free; ++j)
            {
                IntType c = f2c[tmp];
                tmp += 2;
                /*if (innerInColor[*iter_color].find(c) != innerInColor[*iter_color].end())*/
                if (innerInColor[*iter_color].count(c) != 0)
                    free = false;
            }
            /*--- If none of the inner indices in the group appears in
             *    this color yet, it is assigned to the group. ---*/
            if (free) break;
        }

        IntType color;
        if (iter_color != searchOrder.end())
        {
            /*--- Found a color conflict-free. ---*/
            color = *iter_color;
        }
        else {
            /*--- No color was free, make space for a new one. ---*/
            color = nColor++;
            if (nColor == MaxColors) {
#ifdef MPICH
                mflog::log << "Rank " << myZone << " skip group coloring ." << std::endl;
#endif
                mflog::log << "pface colors over limit." << std::endl;

                return false;
            }
            colorSize.push_back(0);
        }
        //test
        for (IntType j = i; j < groupEnd; ++j) {
            assert(innerInColor[color].count(f2c[2 * j]) == 0);
        }

        tmp = 2 * i;
        for (IntType j = i; j < groupEnd; ++j) {
            idxColor[j] = color;
            innerInColor[color].insert(f2c[tmp]);
            tmp += 2;
        }
        colorSize[color] += groupEnd - i;
    }//pface color

    tmp = 0;
    for (IntType i = 0; i < nColor; i++) {
        tmp += (IntType)colorSize[i];
        grid->bfacegroup.push_back(tmp);
    }
    tmp = 0;
    for (IntType i = 0; i < nColor; i++) {
        for (IntType j = 0; j < pfacenum; j++) {
            if (idxColor[j] == i) {
                index_bface[tmp++] = j;
            }
        }
    }
    assert(tmp == pfacenum);
    // update f2c for physical faces
    for (IntType i = 0; i < pfacenum; i++)
    {
        f2c_backup[2 * i] = f2c[2 * i];
        f2c_backup[2 * i + 1] = f2c[2 * i + 1];
    }
    for (IntType i = 0; i < pfacenum; i++)
    {
        f2c[2 * i] = f2c_backup[index_bface[i] * 2];
        f2c[2 * i + 1] = f2c_backup[index_bface[i] * 2 + 1];
    }
    //update nNPF, f2n for physical faces
    tmp = 0;
    for (IntType i = 0; i < pfacenum; i++)
    {
        tmp += nNPF[i];
    }
    mfmem::snew_array_1D(f2n_bface, tmp, dmrfl);
    tmp = 0;
    for (IntType i = 0; i < pfacenum; i++)
    {
        nNPF_bface[i] = nNPF[i];
        for (IntType j = 0; j < nNPF[i]; j++)
        {
            f2n_bface[tmp] = f2n[tmp];
            tmp++;
        }
    }
    F2N_bface[0] = f2n_bface;
    for (IntType i = 1; i < pfacenum; i++)
    {
        F2N_bface[i] = &(F2N_bface[i - 1][nNPF_bface[i - 1]]);
    }
    for (IntType i = 0; i < pfacenum; i++)
    {
        nNPF[i] = nNPF_bface[index_bface[i]];
    }
    tmp = 0;
    for (IntType i = 0; i < pfacenum; i++)
    {
        for (IntType j = 0; j < nNPF[i]; j++)
        {
            f2n[tmp] = F2N_bface[index_bface[i]][j];
            tmp++;
        }
    }

    //deallocation
    mfmem::sdel_array_1D(index_bface);
    mfmem::sdel_array_1D(nNPF_bface);
    mfmem::sdel_array_1D(f2n_bface);
    mfmem::sdel_array_1D(F2N_bface);

    /* iface color */
    nColor = 1;
    colorSize.resize(1);
    colorSize[0] = 0;
    for (vector<set<IntType> >::iterator iter_color = innerInColor.begin(); iter_color != innerInColor.end(); iter_color++)
        (*iter_color).clear();
    for (IntType i = 0; i < ifacenum; i += groupSize) {
        groupEnd = MIN(i + groupSize, ifacenum);
        searchOrder.resize(nColor);

        for (IntType j = 0; j < nColor; j++)
            searchOrder[j] = j;
        /*--- Balance sizes by looking for space in smaller colors first. ---*/
        if (balanceColors) {
            colorSize_bak.resize(nColor);
            colorSize_bak.assign(colorSize.begin(), colorSize.end());
            quicksort_vecint(1, nColor, searchOrder, colorSize_bak);
        }

        vector<IntType>::iterator iter_color = searchOrder.begin();
        for (; iter_color != searchOrder.end(); ++iter_color)
        {
            bool free = true;
            tmp = 2 * (i + nBFace);
            /*--- Traverse entire group as a large outer index. ---*/
            for (IntType j = i; j < groupEnd && free; ++j)
            {
                IntType c1 = f2c[tmp++];
                IntType c2 = f2c[tmp++];
                /*if (innerInColor[*iter_color].find(c) != innerInColor[*iter_color].end())*/
                if (innerInColor[*iter_color].count(c1) != 0 || innerInColor[*iter_color].count(c2) != 0)
                    free = false;
            }
            /*--- If none of the inner indices in the group appears in
             *    this color yet, it is assigned to the group. ---*/
            if (free) break;
        }

        IntType color;
        if (iter_color != searchOrder.end())
        {
            /*--- Found a color conflict-free. ---*/
            color = *iter_color;
        }
        else {
            /*--- No color was free, make space for a new one. ---*/
            color = nColor++;
            if (nColor == MaxColors) {
#ifdef MPICH
                mflog::log << "Rank " << myZone << " skip group coloring ." << std::endl;
#endif
                mflog::log << "iface colors over limit." << std::endl;
                return false;
            }
            colorSize.push_back(0);
        }
        //test
        for (IntType j = i; j < groupEnd; ++j) {
            assert(innerInColor[color].count(f2c[2 * (j + nBFace)]) == 0);
            assert(innerInColor[color].count(f2c[2 * (j + nBFace) + 1]) == 0);
        }

        tmp = 2 * (i + nBFace);
        for (IntType j = i; j < groupEnd; ++j) {
            idxColor[j + nBFace] = color;
            innerInColor[color].insert(f2c[tmp++]);
            innerInColor[color].insert(f2c[tmp++]);
        }
        colorSize[color] += groupEnd - i;
    }//iface color

    tmp = nBFace;
    for (IntType i = 0; i < nColor; i++) {
        tmp += (IntType)colorSize[i];
        grid->ifacegroup.push_back(tmp);
    }
    tmp = 0;
    for (IntType i = 0; i < nColor; i++) {
        for (IntType j = nBFace; j < nTFace; j++) {
            if (idxColor[j] == i)
                index_iface[tmp++] = j;
        }
    }
    assert(tmp == ifacenum);
    for (vector<set<IntType> >::iterator iter_color = innerInColor.begin(); iter_color != innerInColor.end(); iter_color++)
        (*iter_color).clear();
    mfmem::sdel_array_1D(idxColor);

    //update f2c nNPF, f2n for interior faces
    UpdateIface(grid, index_iface);

    //deallocation
    mfmem::sdel_array_1D(index_iface);
    mfmem::sdel_array_1D(f2c_backup);
    return true;
}

void GroupColoringGPU(PolyGrid* grid){
	
	IntType* f2c = grid->Getf2c();
	
	IntType nIFace = grid->GetNIFace();
	IntType nBFace = grid->GetNBFace();
	IntType nTFace = grid->GetNTFace();
	IntType groupSize = grid->groupSize;
	IntType n_bcolor, n_icolor;
	n_bcolor = grid->bfacegroup.size();
	n_icolor = grid->ifacegroup.size();
	
	std::set<IntType> cellset;
	cellset.clear();
	
	IntType bf2cnnum = 0;
	
	IntType if2cnnumc1 = 0;
	IntType if2cnnumc2 = 0;
	IntType i, ns, ne, nMid, face, c1, c2;	
	
#ifdef MPICH   
	IntType mpirank = 0; //when mpirank = 0, mpi was off. 
    MPI_Comm_rank(MPI_COMM_WORLD, & mpirank);	
#endif
	
	mfmem::snew_array_1D(grid->group_b_SM_color_index, n_bcolor, dmrfl);	
	// the number of total groups:
	IntType num_b_group = 0; 	
	// for num_b_group:
	for (i = 0; i < n_bcolor; i++) {
		if (!i)
			ns = 0;
		else
			ns = grid->bfacegroup[i - 1];
		nMid = grid->bfacegroup[i];
		//cout << "bface-i: " << i << ": " << nMid - ns << endl;
		
		IntType this_num_b_group = (nMid - ns)/groupSize;
		
		num_b_group += (nMid - ns)/groupSize;
		
		if ((nMid - ns)%groupSize != 0){
			num_b_group++;
			this_num_b_group++;
		}
		//grid->group_b_SM_color_index[i] = this_num_b_group;
		grid->group_b_SM_color_index[i] = num_b_group;
    }
	
	mfmem::snew_array_1D(grid->group_b_SM_index, num_b_group + 1, dmrfl);
	grid->group_b_SM_index[0] = 0;
	IntType b_SM_index = 1; 	
	
	// grid->group_b_SM_index was set to store the start location of cell index in SM
	// for grid->group_b_SM_index:
	IntType length_group_b_SMc2c = 0;
	for (i = 0; i < n_bcolor; i++) {
		if (!i)
			ns = 0;
		else
			ns = grid->bfacegroup[i - 1];
		nMid = grid->bfacegroup[i];
		//cout << "rankid: " << mpirank << "-" << "bface-i: " << i << ": " << nMid - ns << endl;
		
		face = ns;
		for (; (face + groupSize) <= nMid; face+=groupSize) {
			IntType tmpbf2cnnum = 0;
			for (IntType innerface = 0; innerface < groupSize; innerface++){
				c1 = f2c[2 * (face + innerface)];			
				if (cellset.count(c1) == 0){
					cellset.insert(c1);
					tmpbf2cnnum++;				
				}
			}
			//if(mpirank==0){
			//cout << "rankid: " << mpirank << "-" << "tmpbf2cnnum: " << tmpbf2cnnum << ""<< endl;
			//}
			//if(tmpbf2cnnum > bf2cnnum) bf2cnnum = tmpbf2cnnum;
			length_group_b_SMc2c += tmpbf2cnnum;
			grid->group_b_SM_index[b_SM_index] = length_group_b_SMc2c;
			b_SM_index++;
			cellset.clear();
		}
		if((nMid - ns) < groupSize ){ //该颜色内面总数 < groupSize
			face = ns;
		}
		if ((nMid - ns)%groupSize != 0){ //该颜色内最后一个group面数量不足groupSize
			IntType tmpbf2cnnum = 0;
			for (; face < nMid; face++){			
				c1 = f2c[2 * face];			
				if (cellset.count(c1) == 0){
					cellset.insert(c1);
					tmpbf2cnnum++;				
				}
			}
			//if(mpirank==0){
			//cout << "rankid: " << mpirank << "-" << "tmpbf2cnnum: " << tmpbf2cnnum << endl;
			//}
			cellset.clear();
			length_group_b_SMc2c += tmpbf2cnnum;
			if (tmpbf2cnnum != 0){
				grid->group_b_SM_index[b_SM_index] = length_group_b_SMc2c;
				b_SM_index++;
			}
		}
    }
	//cout << "rankid: " << mpirank << "-" << "length_group_b_SMc2c: " << length_group_b_SMc2c << "; " << nBFace - nIFace << endl;
	//cout << "rankid: " << mpirank << "-" << "b_SM_index: " << b_SM_index << "; num_b_group: " << num_b_group << endl;
	
	mfmem::snew_array_1D(grid->group_b_SMc2c, length_group_b_SMc2c, dmrfl);		
	// grid->group_b_SMc2c was set to store cell index in SM
	// for grid->group_b_SMc2c:
	IntType index_group_b_SMc2c = 0;
	for (i = 0; i < n_bcolor; i++) {
		if (!i)
			ns = 0;
		else
			ns = grid->bfacegroup[i - 1];
		nMid = grid->bfacegroup[i];
		//cout << "bface-i: " << i << ": " << nMid - ns << endl;
		
		for (face = ns; (face + groupSize) <= nMid; face += groupSize) {
			for (IntType innerface = 0; innerface < groupSize; innerface++){
				c1 = f2c[2 * (face + innerface)];			
				if (cellset.count(c1) == 0){
					cellset.insert(c1);
					grid->group_b_SMc2c[index_group_b_SMc2c] = c1;
					index_group_b_SMc2c++;
				}
			} 
			/* for (IntType innerface = 0; innerface < groupSize; innerface++){
				c1 = f2c[2 * (face + innerface)];			
				if (cellset.count(c1) == 0){
					cellset.insert(c1);
				}
			}
			for (auto &i : cellset){
				grid->group_b_SMc2c[index_group_b_SMc2c] = i;
				index_group_b_SMc2c++;
			}*/	
			cellset.clear();
		}
		
		if ((nMid - ns)%groupSize != 0){
			for (; face < nMid; face++){			
				c1 = f2c[2 * face];			
				if (cellset.count(c1) == 0){
					cellset.insert(c1);
					grid->group_b_SMc2c[index_group_b_SMc2c] = c1;
					index_group_b_SMc2c++;	
				}
			} 
			/* for (; face < nMid; face++){			
				c1 = f2c[2 * face];			
				if (cellset.count(c1) == 0){
					cellset.insert(c1);
				}
			}
			for (auto &i : cellset){
				grid->group_b_SMc2c[index_group_b_SMc2c] = i;
				index_group_b_SMc2c++;
			}	*/
			cellset.clear();
		}
    }
	//cout << "rankid: " << mpirank << "-" << "index_group_b_SMc2c: " << index_group_b_SMc2c << endl;
	//cout << endl;
	cellset.clear(); 
	
	// bulid the group_b_f2SMc relationship:
	// each boundary face has a c1 index stored in group_b_f2SMc:
	mfmem::snew_array_1D(grid->group_b_f2SMc, nBFace - nIFace, dmrfl);
	for (i = 0; i < n_bcolor; i++) {
		if (!i)
			ns = 0;
		else
			ns = grid->bfacegroup[i - 1];
		nMid = grid->bfacegroup[i];		
		
		for (face = ns; (face + groupSize) <= nMid; face+=groupSize) {
			for (IntType innerface = 0; innerface < groupSize; innerface++){
				c1 = f2c[2 * (face + innerface)];			
				if (cellset.count(c1) == 0){
					cellset.insert(c1);			
					grid->group_b_f2SMc[face + innerface] = cellset.size() - 1;
				}
				else{
					//find c1 location in cellset which already has put c1 into the cellset:
					grid->group_b_f2SMc[face + innerface] = distance(cellset.begin(), find(cellset.begin(), cellset.end(), c1));
				}
				
			}						
			cellset.clear();			
		}
		
		if ((nMid - ns)%groupSize != 0){
			for (; face < nMid; face++){			
				c1 = f2c[2 * face];			
				if (cellset.count(c1) == 0){
					cellset.insert(c1);							
					grid->group_b_f2SMc[face] = cellset.size() - 1;	
				}
				else{
					grid->group_b_f2SMc[face] = distance(cellset.begin(), find(cellset.begin(), cellset.end(), c1));
				}				
			}
			cellset.clear();			
		}
    }  
	cellset.clear();
	
	// Interior faces
	IntType num_i_group = 0; 
	mfmem::snew_array_1D(grid->group_i_SM_color_index, n_icolor, dmrfl);
	for (i = 0; i < n_icolor; i++) {
		if (!i)
			nMid = nBFace;
		else
			nMid = grid->ifacegroup[i - 1];
		ne = grid->ifacegroup[i];
		//cout << "iface-i: " << i << ": " << ne - nMid << endl;		
		IntType this_num_i_group = (ne - nMid)/groupSize;
		
		num_i_group += (ne - nMid)/groupSize;
		
		if ((ne - nMid)%groupSize != 0){
			num_i_group++;
			this_num_i_group++;
		}
		grid->group_i_SM_color_index[i] = num_i_group;
	} 
	//cout << "rankid: " << mpirank << "-" << "num_i_group: " << num_i_group << endl;
	
	mfmem::snew_array_1D(grid->group_i_SM_index, num_i_group + 1, dmrfl);
	grid->group_i_SM_index[0] = 0;
	IntType i_SM_index = 1; 	
	// grid->group_i_SM_index was set to store the start location of cell index in SM
	// for grid->group_i_SM_index:
	IntType length_group_i_SMc2c = 0;
	for (i = 0; i < n_icolor; i++) {
		if (!i)
			nMid = nBFace;
		else
			nMid = grid->ifacegroup[i - 1];
		ne = grid->ifacegroup[i];
		//cout << "iface-i: " << i << ": " << ne - nMid << endl;				
		
		for (face = nMid; (face + groupSize) <= ne; face += groupSize) {
			IntType tmpif2cnnumc1 = 0;
			for (IntType innerface = 0; innerface < groupSize; innerface++){
				c1 = f2c[2 * (face + innerface)];
				c2 = f2c[2 * (face + innerface) + 1];
				if (cellset.count(c1) == 0){
					cellset.insert(c1);
					tmpif2cnnumc1++;				
				}
				if (cellset.count(c2) == 0){
					cellset.insert(c2);
					tmpif2cnnumc1++;				
				}
			}
			length_group_i_SMc2c += tmpif2cnnumc1;
			grid->group_i_SM_index[i_SM_index] = length_group_i_SMc2c;
			/* cout << length_group_i_SMc2c << endl;
			exit(0); */
			i_SM_index++;
			cellset.clear();			
		}
		if((ne - nMid) < groupSize ){
			face = ns;
		}
		if ((ne - nMid)%groupSize != 0){
			IntType tmpif2cnnumc1 = 0;
			for (; face < ne; face++){			
				c1 = f2c[2 * face];
				c2 = f2c[2 * face + 1];
				if (cellset.count(c1) == 0){
					cellset.insert(c1);
					tmpif2cnnumc1++;				
				}
				if (cellset.count(c2) == 0){
					cellset.insert(c2);
					tmpif2cnnumc1++;				
				}
			}
			cellset.clear();
			length_group_i_SMc2c += tmpif2cnnumc1;
			grid->group_i_SM_index[i_SM_index] = length_group_i_SMc2c;
			i_SM_index++;	
		}
	}
	//cout << "rankid: " << mpirank << "-" << "length_group_i_SMc2c: " << length_group_i_SMc2c << "; " << nTFace - nBFace << endl;
	//cout << "rankid: " << mpirank << "-" << "i_SM_index: " << i_SM_index << "; num_i_group: " << num_i_group << endl;
	
	mfmem::snew_array_1D(grid->group_i_SMc2c, length_group_i_SMc2c, dmrfl);	
	
	// grid->group_i_SMc2c was set to store cell index in SM
	// for grid->group_i_SMc2c:
	IntType index_group_i_SMc2c = 0;
	for (i = 0; i < n_icolor; i++) {
		if (!i)
			nMid = nBFace;
		else
			nMid = grid->ifacegroup[i - 1];
		ne = grid->ifacegroup[i];
		
		for (face = nMid; (face + groupSize) <= ne; face += groupSize) {
			IntType numCell = 0;
			for (IntType innerface = 0; innerface < groupSize; innerface++){
				c1 = f2c[2 * (face + innerface)];
				c2 = f2c[2 * (face + innerface) + 1];
				if (cellset.count(c1) == 0){
					cellset.insert(c1);
					numCell++;						
				}
				if (cellset.count(c2) == 0){
					cellset.insert(c2);
					numCell++;							
				}
			}
			for (auto &i : cellset){
				grid->group_i_SMc2c[index_group_i_SMc2c] = i;
				index_group_i_SMc2c++;
			}
			/* for (IntType innerface = 0; innerface < groupSize; innerface++){
				c1 = f2c[2 * (face + innerface)];
				c2 = f2c[2 * (face + innerface) + 1];
				if (cellset.count(c1) == 0){
					cellset.insert(c1);
					grid->group_i_SMc2c[index_group_i_SMc2c] = c1;//cout << index_group_i_SMc2c << ": " << c1 << endl;
					index_group_i_SMc2c++;						
				}
				if (cellset.count(c2) == 0){
					cellset.insert(c2);
					grid->group_i_SMc2c[index_group_i_SMc2c] = c2;//cout << index_group_i_SMc2c << ": " << c2 << endl;
					index_group_i_SMc2c++;							
				}
			} */
			
			//exit(0);
			cellset.clear();		
		}
		if((ne - nMid) < groupSize ){
			face = ns;
		}
		if ((ne - nMid)%groupSize != 0){
			/* for (; face < ne; face++){			
				c1 = f2c[2 * face];
				c2 = f2c[2 * face + 1];
				if (cellset.count(c1) == 0){
					cellset.insert(c1);
					grid->group_i_SMc2c[index_group_i_SMc2c] = c1;
					index_group_i_SMc2c++;				
				}
				if (cellset.count(c2) == 0){
					cellset.insert(c2);
					grid->group_i_SMc2c[index_group_i_SMc2c] = c2;
					index_group_i_SMc2c++;	
				}
			} */
			for (; face < ne; face++){			
				c1 = f2c[2 * face];
				c2 = f2c[2 * face + 1];
				if (cellset.count(c1) == 0){
					cellset.insert(c1);
				}
				if (cellset.count(c2) == 0){
					cellset.insert(c2);
				}
			}
			for (auto &i : cellset){
				grid->group_i_SMc2c[index_group_i_SMc2c] = i;
				index_group_i_SMc2c++;
			}
			cellset.clear();
		}
    }
	//cout << "rankid: " << mpirank << "-" << "index_group_i_SMc2c: " << index_group_i_SMc2c << endl;
	cout << endl;
	cellset.clear(); 
	
	// bulid the group_i_f2SMc relationship:
	mfmem::snew_array_1D(grid->group_i_f2SMc, nTFace*2, dmrfl); // *2 includes c1 and c2
	for (i = 0; i < n_icolor; i++) {
		if (!i)
			nMid = nBFace;
		else
			nMid = grid->ifacegroup[i - 1];
		ne = grid->ifacegroup[i];
		
		for (face = nMid; (face + groupSize) <= ne; face += groupSize) {
			for (IntType innerface = 0; innerface < groupSize; innerface++){
				c1 = f2c[2 * (face + innerface)];
				c2 = f2c[2 * (face + innerface) + 1];
				if (cellset.count(c1) == 0){
					cellset.insert(c1);		
				}
				if (cellset.count(c2) == 0){
					cellset.insert(c2);		
				}
			}
			/* for (auto &i : cellset){
				cout << i << endl;
			}
			exit(0); */
			for (IntType innerface = 0; innerface < groupSize; innerface++){
				c1 = f2c[2 * (face + innerface)];
				c2 = f2c[2 * (face + innerface) + 1];
				//find c1 location in cellset which already has put c1 into the cellset:
				grid->group_i_f2SMc[2 * (face + innerface)] = distance(cellset.begin(), find(cellset.begin(), cellset.end(), c1));
				//find c2 location in cellset which already has put c1 into the cellset:
				grid->group_i_f2SMc[2 * (face + innerface) + 1] = distance(cellset.begin(), find(cellset.begin(), cellset.end(), c2));
				//cout << face + innerface << ": " << grid->group_i_f2SMc[2 * (face + innerface)] << ", " << grid->group_i_f2SMc[2 * (face + innerface) + 1] << endl;
			}
			//exit(0);
			cellset.clear();		
		}
		if((ne - nMid) < groupSize ){
			face = ns;
		}
		
		if ((ne - nMid)%groupSize != 0){
			IntType tmpface = face;
			for (; tmpface < ne; tmpface++){			
				c1 = f2c[2 * tmpface];
				c2 = f2c[2 * tmpface + 1];
				if (cellset.count(c1) == 0){
					cellset.insert(c1);								
				}
				if (cellset.count(c2) == 0){
					cellset.insert(c2);
				}
			}
			tmpface = face;
			for (; tmpface < ne; tmpface++){			
				c1 = f2c[2 * tmpface];
				c2 = f2c[2 * tmpface + 1];
				//find c1 location in cellset which already has put c1 into the cellset:
				grid->group_i_f2SMc[2 * tmpface] = distance(cellset.begin(), find(cellset.begin(), cellset.end(), c1));
				//find c2 location in cellset which already has put c1 into the cellset:
				grid->group_i_f2SMc[2 * tmpface + 1] = distance(cellset.begin(), find(cellset.begin(), cellset.end(), c2));
			}			
			cellset.clear();
		}
    }  
	cellset.clear();
}

#endif // GroupColor

/************************************************************************
                        Build local bface to bface
                        Add by dingxin 2021-11-30
************************************************************************/
void setlocalF2F_b(IntType*& f2f, IntType*& f2f_index, IntType* f2c, IntType* bface, IntType n_bface) {
    IntType i, j, k, tmp;
    IntType c1, count, face, localID_face;
    set<IntType> countCell;
    vector<vector<IntType> > neighborFace;
    IntType* partCell = NULL;
    IntType* f2c1_local = NULL;

    mfmem::snew_array_1D(f2c1_local, n_bface, dmrfl);

    for (i = 0; i < n_bface; i++) {
        face = bface[i];
        countCell.insert(f2c[2 * face]);
    }
    count = countCell.size();
    mfmem::snew_array_1D(partCell, count, dmrfl);
    neighborFace.resize(count);
    countCell.clear();

    //get the relationship between bface and cell
    count = 0;
    for (i = 0; i < n_bface; i++) {
        face = bface[i];
        c1 = f2c[2 * face];
        j = 0;
        while (j < count && partCell[j] != c1) j++;
        if (j < count) {
            neighborFace[j].push_back(i);//put the local face id
        }
        else {
            partCell[j] = c1;
            neighborFace[j].push_back(i);//put the local face id
            count++;
        }
        f2c1_local[i] = j;
    }
    mfmem::snew_array_1D(f2f_index, n_bface + 1, dmrfl);
    f2f_index[0] = 0;
    for (i = 0; i < n_bface; i++) {
        c1 = f2c1_local[i];
        f2f_index[i + 1] = neighborFace[c1].size() - 1;
    }
    for (i = 1; i < n_bface + 1; i++)
        f2f_index[i] += f2f_index[i - 1];
    mfmem::snew_array_1D(f2f, f2f_index[n_bface], dmrfl);
    tmp = 0;
    for (i = 0; i < n_bface; i++) {
        c1 = f2c1_local[i];
        vector<IntType>::iterator iter_index = neighborFace[c1].begin();
        for (; iter_index != neighborFace[c1].end(); ++iter_index) {
            localID_face = *iter_index;
            if (localID_face != i)
                f2f[tmp++] = localID_face;
        }
    }
    for (i = 0; i < count; i++)
        vector<IntType>().swap(neighborFace[i]);
    vector<vector<IntType> >().swap(neighborFace);
    mfmem::sdel_array_1D(partCell);
    mfmem::sdel_array_1D(f2c1_local);
    assert(f2f_index[n_bface] == tmp);
}

/************************************************************************
                        Build local iface to iface
                        Add by dingxin 2021-11-30
************************************************************************/
void setlocalF2F_i(IntType*& f2f, IntType*& f2f_index, IntType* f2c, IntType* iface, IntType n_iface) {
    IntType i, j, k, tmp, index_c1, index_c2;
    IntType c1, c2, count, face, localID_face;
    vector<set<IntType> > neighborFace;
    IntType minCell = 0;
    IntType maxCell = f2c[iface[0] * 2];

    for (i = 0; i < n_iface; i++) {
        face = iface[i];
        c1 = f2c[2 * face];
        c2 = f2c[2 * face + 1];
        maxCell = MAX(maxCell, MAX(c1, c2));
        minCell = MIN(minCell, MIN(c1, c2));
    }
    //neighborFace.resize(count);
    count = maxCell - minCell + 1;
    neighborFace.resize(count);

    //get the relationship between local iface and cell
    //count = 0;
    for (i = 0; i < n_iface; i++) {
        face = iface[i];
        c1 = f2c[2 * face];
        c2 = f2c[2 * face + 1];
        neighborFace[c1 - minCell].insert(i);
        neighborFace[c2 - minCell].insert(i);
    }

    mfmem::snew_array_1D(f2f_index, n_iface + 1, dmrfl);
    memset(f2f_index, 0, (n_iface + 1) * sizeof(IntType));
    for (i = 0; i < n_iface; i++) {
        face = iface[i];
        c1 = f2c[2 * face];
        c2 = f2c[2 * face + 1];
        index_c1 = c1 - minCell;
        f2f_index[i + 1] += neighborFace[index_c1].size() - 1;
        index_c2 = c2 - minCell;
        set<IntType>::iterator iter_index = neighborFace[index_c2].begin();
        for (; iter_index != neighborFace[index_c2].end(); ++iter_index) {
            if (neighborFace[index_c1].count(*iter_index) == 0)
                f2f_index[i + 1]++;
        }
    }
    for (i = 1; i < n_iface + 1; i++)
        f2f_index[i] += f2f_index[i - 1];
    mfmem::snew_array_1D(f2f, f2f_index[n_iface], dmrfl);

    tmp = 0;
    for (i = 0; i < n_iface; i++) {
        face = iface[i];
        c1 = f2c[2 * face];
        c2 = f2c[2 * face + 1];
        index_c1 = c1 - minCell;
        index_c2 = c2 - minCell;
        set<IntType>::iterator iter_index = neighborFace[index_c1].begin();
        for (; iter_index != neighborFace[index_c1].end(); ++iter_index) {
            localID_face = *iter_index;
            if (localID_face != i)
                f2f[tmp++] = localID_face;
        }
        iter_index = neighborFace[index_c2].begin();
        for (; iter_index != neighborFace[index_c2].end(); ++iter_index) {
            localID_face = *iter_index;
            if (neighborFace[index_c1].count(localID_face) == 0)
                f2f[tmp++] = localID_face;
        }
    }

    for (i = 0; i < count; i++)
        neighborFace[i].clear();
    vector<set<IntType> >().swap(neighborFace);
    assert(f2f_index[n_iface] == tmp);
}

/************************************************************************
                        Build local iface to iface for small block
                        Add by dingxin 2021-11-30
************************************************************************/
void setlocalF2F_i2(IntType*& f2f, IntType*& f2f_index, IntType* f2c, IntType* iface, IntType n_iface) {
    IntType i, j, k, tmp, index_c1, index_c2;
    IntType c1, c2, count, face, localID_face;
    set<IntType> countCell;
    vector<set<IntType> > neighborFace;
    IntType* partCell = NULL;
    IntType* f2c_local = NULL;

    mfmem::snew_array_1D(f2c_local, n_iface * 2, dmrfl);
    for (i = 0; i < n_iface; i++) {
        face = iface[i];
        countCell.insert(f2c[2 * face]);
        countCell.insert(f2c[2 * face + 1]);
    }
    count = countCell.size();
    mfmem::snew_array_1D(partCell, count, dmrfl);
    neighborFace.resize(count);
    countCell.clear();

    //get the relationship between local iface and cell
    count = 0;
    for (i = 0; i < n_iface; i++) {
        face = iface[i];
        c1 = f2c[2 * face];
        c2 = f2c[2 * face + 1];
        index_c1 = -1;
        index_c2 = -1;
        for (j = 0; j < count; j++) {
            tmp = partCell[j];
            if (index_c1 < 0 && tmp == c1) {
                neighborFace[j].insert(i);
                index_c1 = j;
                if (index_c2 >= 0)
                    break;
                continue;
            }
            if (index_c2 < 0 && tmp == c2) {
                neighborFace[j].insert(i);
                index_c2 = j;
                if (index_c1 >= 0)
                    break;
                continue;
            }
        }
        if (index_c1 < 0) {
            partCell[count] = c1;
            neighborFace[count].insert(i);
            index_c1 = count;
            count++;
        }
        if (index_c2 < 0) {
            partCell[count] = c2;
            neighborFace[count].insert(i);
            index_c2 = count;
            count++;
        }
        f2c_local[2 * i] = index_c1;
        f2c_local[2 * i + 1] = index_c2;
    }

    mfmem::snew_array_1D(f2f_index, n_iface + 1, dmrfl);
    memset(f2f_index, 0, (n_iface + 1) * sizeof(IntType));
    for (i = 0; i < n_iface; i++) {
        index_c1 = f2c_local[2 * i];
        index_c2 = f2c_local[2 * i + 1];
        f2f_index[i + 1] += neighborFace[index_c1].size() - 1;
        set<IntType>::iterator iter_index = neighborFace[index_c2].begin();
        for (; iter_index != neighborFace[index_c2].end(); ++iter_index) {
            if (neighborFace[index_c1].count(*iter_index) == 0)
                f2f_index[i + 1]++;
        }
    }
    for (i = 1; i < n_iface + 1; i++)
        f2f_index[i] += f2f_index[i - 1];
    mfmem::snew_array_1D(f2f, f2f_index[n_iface], dmrfl);
    tmp = 0;

    for (i = 0; i < n_iface; i++) {
        index_c1 = f2c_local[2 * i];
        index_c2 = f2c_local[2 * i + 1];
        set<IntType>::iterator iter_index = neighborFace[index_c1].begin();
        for (; iter_index != neighborFace[index_c1].end(); ++iter_index) {
            localID_face = *iter_index;
            if (localID_face != i)
                f2f[tmp++] = localID_face;
        }
        iter_index = neighborFace[index_c2].begin();
        for (; iter_index != neighborFace[index_c2].end(); ++iter_index) {
            localID_face = *iter_index;
            if (neighborFace[index_c1].count(localID_face) == 0)
                f2f[tmp++] = localID_face;
        }
    }

    for (i = 0; i < count; i++)
        neighborFace[i].clear();
    vector<set<IntType> >().swap(neighborFace);
    mfmem::sdel_array_1D(partCell);
    mfmem::sdel_array_1D(f2c_local);
    assert(f2f_index[n_iface] == tmp);
}

/************************************************************************
                   Color local face by bounded color algorithm
                        Add by dingxin 2021-11-30
************************************************************************/
IntType create_bounded_color_part(IntType* f2f, IntType* f2f_index, IntType* newOrder, IntType n_face) {
    IntType i, j, count;
    IntType facenum_vec;// facenum_vec = n_colors * VEC_SIZE. the colors, face num of each color is VEC_SIZE. 
    IntType color, block, start_id, neighborColor;
    IntType* colorPart = NULL;//the color per face
    IntType* colorCard = NULL;//the number per color
    IntType* index_colorCard = NULL;
    IntType* neighborsColor = NULL;
    IntType* order_color = NULL;
    IntType* neworder_color = NULL;
    IntType NB_BLOCKS;
    IntType BLOCK_SIZE = 32;
    IntType nbColors = 0;

    if (n_face < 32)
        NB_BLOCKS = 1;
    else
        NB_BLOCKS = ceil(n_face / 32);

    mfmem::snew_array_1D(colorPart, n_face, dmrfl);
    mfmem::snew_array_1D(colorCard, n_face, dmrfl);
    mfmem::snew_array_1D(neighborsColor, NB_BLOCKS, dmrfl);

    memset(colorPart, -1, n_face * sizeof(IntType));
    memset(colorCard, 0, n_face * sizeof(IntType));
    for (i = 0; i < n_face; i++) {
        memset(neighborsColor, 0, NB_BLOCKS * sizeof(IntType));
        color = 0;//color id starts from 0
        block = 0;
        // Get the color of all neigbor faces
        start_id = f2f_index[i];
        for (j = start_id; j < f2f_index[i + 1]; j++) {
            neighborColor = colorPart[f2f[j]];
            if (neighborColor != -1) {
                neighborsColor[neighborColor / BLOCK_SIZE] |=  1 << (neighborColor % BLOCK_SIZE);
            }
        }
        // Get the first free color (position of the first 0 bit)
        while (neighborsColor[block] & 1 || colorCard[color] >= VEC_SIZE) {
            neighborsColor[block] = neighborsColor[block] >> 1;
            color++;
            if (color % BLOCK_SIZE == 0) block++;
        }
        // Assign the first free color to current face
        colorPart[i] = color;
        colorCard[color]++;
        // Compute the total number of colors
        if (color > nbColors) nbColors = color;
    }
    nbColors++;

    mfmem::snew_array_1D(order_color, nbColors, dmrfl);
    mfmem::snew_array_1D(neworder_color, nbColors, dmrfl);
    //reorder color
    for (color = 0; color < nbColors; color++) {
        if (colorCard[color] < VEC_SIZE) {
            start_id = color;
            break;
        }
        else {
            order_color[color] = color;
            neworder_color[color] = color;
        }
    }
    if (color < nbColors) {
        count = start_id;
        for (color = start_id + 1; color < nbColors; color++) {
            if (colorCard[color] == VEC_SIZE) {
                order_color[count] = color;
                neworder_color[color] = count;
                count++;
            }
        }
        facenum_vec = count * VEC_SIZE;
        for (color = start_id; color < nbColors; color++) {
            if (colorCard[color] < VEC_SIZE) {
                order_color[count] = color;
                neworder_color[color] = count;
                count++;
            }
        }
        assert(count == nbColors);
    }
    else
        facenum_vec = nbColors * VEC_SIZE;
    //mflog::log << "all colors are VEC_SIZE. " << std::endl;

    //reorder newOrder
    count = 0;
    mfmem::snew_array_1D(index_colorCard, nbColors + 1, dmrfl);
    index_colorCard[0] = 0;
    for (i = 0; i < nbColors; i++) {
        color = order_color[i];
        count += colorCard[color];
        index_colorCard[i + 1] = count;
        colorCard[color] = 0;
    }
    for (i = 0; i < n_face; i++) {
        color = colorPart[i];
        newOrder[index_colorCard[neworder_color[color]] + colorCard[color]] = i;
        colorCard[color]++;
    }

    mfmem::sdel_array_1D(colorPart);
    mfmem::sdel_array_1D(colorCard);
    mfmem::sdel_array_1D(index_colorCard);
    mfmem::sdel_array_1D(neighborsColor);
    mfmem::sdel_array_1D(order_color);
    mfmem::sdel_array_1D(neworder_color);
    return facenum_vec;
}

/************************************************************************
                   Update the order of array by newOrder
                        Add by dingxin 2021-11-30
************************************************************************/
void updateOrder(IntType* a, IntType* newOrder, IntType start, IntType len) {
    /*
    * reorder a by newOrder
    * index of a from start to (start+len-1)
    * the newOrder, a array of local id, is from 0 to len-1, new to old
    */
    IntType i;
    IntType* a_bak = NULL;
    mfmem::snew_array_1D(a_bak, len, dmrfl);
    for (i = 0; i < len; i++)
        a_bak[i] = a[i + start];
    for (i = 0; i < len; i++)
        a[i + start] = a_bak[newOrder[i]];
    mfmem::sdel_array_1D(a_bak);
}

/************************************************************************
                        Bounded Coloring for simd
                        Add by dingxin 2021-11-25
************************************************************************/
void BoundedColoring(IntType* f2c, IntType* bface, IntType* iface, IntType* order_b, IntType* order_i,
        IntType n_bface, IntType n_iface, IntType& bfacenum_vec, IntType& ifacenum_vec) {
    /* bface and iface is the global face id
       order_b is the new order of bface
       order_i is the new order of iface */
    IntType i, j, k, tmp;
    IntType count, c1, c2, face, face_cc;
    IntType* f2f = NULL;
    IntType* f2f_index = NULL;
    /* color bface */
    {
        setlocalF2F_b(f2f, f2f_index, f2c, bface, n_bface);
        bfacenum_vec = create_bounded_color_part(f2f, f2f_index, order_b, n_bface);
        mfmem::sdel_array_1D(f2f);
        mfmem::sdel_array_1D(f2f_index);
    }
    /* color iface */
    {
        if (n_iface > 10000) {
            setlocalF2F_i(f2f, f2f_index, f2c, iface, n_iface);
        }
        else {
            setlocalF2F_i2(f2f, f2f_index, f2c, iface, n_iface);
        }
        ifacenum_vec = create_bounded_color_part(f2f, f2f_index, order_i, n_iface);
        mfmem::sdel_array_1D(f2f);
        mfmem::sdel_array_1D(f2f_index);
    }
}

#ifdef DEBUG
bool validation_decoupling(IntType* face, IntType start, IntType len, IntType* f2c, IntType type) {
    //
    IntType i, j, count, id, c1, c2;
    set<IntType> cell;
    if (type == BOUNDARYFACE) {
        for (i = start; i < len; i += VEC_SIZE) {
            if (cell.empty()) {
                for (j = 0; j < VEC_SIZE; j++) {
                    id = face[i + j];
                    c1 = f2c[2 * id];
                    cell.insert(c1);
                }
                if (cell.size() != VEC_SIZE)
                    return false;
                cell.clear();
            }
        }
    }
    else if (type == INTERIORFACE) {
        for (i = start; i < len; i += VEC_SIZE) {
            if (cell.empty()) {
                for (j = 0; j < VEC_SIZE; j++) {
                    id = face[i + j];
                    c1 = f2c[2 * id];
                    c2 = f2c[2 * id + 1];
                    cell.insert(c1);
                    cell.insert(c2);
                }
                if (cell.size() != 2 * VEC_SIZE)
                    return false;
                cell.clear();
            }
        }
    }
    else
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    return true;
}
#endif // DEBUG

#ifdef DIVREP
/************************************************************************
            Divide subzone by Metis, and replicate face
                        Add by dingxin 2021-11-1
************************************************************************/
bool DivRep(PolyGrid* grid) {
    IntType i, j;
    IntType nTCell = grid->GetNTCell();
    IntType nBFace = grid->GetNBFace();
    IntType nTFace = grid->GetNTFace();
    IntType nSubZone = grid->threads;
    IntType* nCPC_tmp = NULL;
    IntType** c2c_tmp = NULL;
    IntType* f2c;
    IntType count, c1, c2, subzone;

    idx_t* xadj = NULL;//csr,start and end index
    idx_t* adjncy = NULL;//csr,cell id
    idx_t nvtxs, ncon, nparts, objval;
    idx_t* part = NULL;
    IntType status;

    IntType* idx;
    IntType* nface_pthread;
    if (nSubZone < 2) {
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    /* Division */
    {
        //get cell to cell，exclude ghost
        mfmem::snew_array_1D(nCPC_tmp, nTCell, dmrfl);
        for (i = 0; i < nTCell; i++) nCPC_tmp[i] = 0;
        f2c = grid->Getf2c();
        count = 2 * nBFace;
        for (i = nBFace; i < nTFace; i++) {
            c1 = f2c[count++];
            c2 = f2c[count++];
            nCPC_tmp[c1]++;
            nCPC_tmp[c2]++;
        }
        mfmem::snew_array_2D(c2c_tmp, nTCell, nCPC_tmp, dmrfl, true);
        for (i = 0; i < nTCell; i++) nCPC_tmp[i] = 0;
        count = 2 * nBFace;
        for (i = nBFace; i < nTFace; i++) {
            c1 = f2c[count++];
            c2 = f2c[count++];
            c2c_tmp[c1][nCPC_tmp[c1]++] = c2;
            c2c_tmp[c2][nCPC_tmp[c2]++] = c1;
        }
        //csr
        mfmem::snew_array_1D(xadj, nTCell + 1, dmrfl);
        mfmem::snew_array_1D(adjncy, 2 * (nTFace - nBFace), dmrfl);
        xadj[0] = 0;
        count = 0;
        for (i = 0; i < nTCell; i++) {
            xadj[i + 1] = xadj[i] + nCPC_tmp[i];
            for (j = 0; j < nCPC_tmp[i]; j++) {
                adjncy[count++] = c2c_tmp[i][j];
            }
        }
        mfmem::sdel_array_1D(nCPC_tmp);
        mfmem::sdel_array_2D(c2c_tmp);
        //divide by Metis
        nvtxs = (idx_t)nTCell;
        ncon = 1;
        nparts = (idx_t)nSubZone;
        mfmem::snew_array_1D(part, static_cast<size_t>(nvtxs), dmrfl);
        for (i = 0; i < nvtxs; i++) {
            part[i] = 0;
        }
        status = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, NULL,
            &nparts, NULL, NULL, NULL, &objval, part);
        mfmem::sdel_array_1D(xadj);
        mfmem::sdel_array_1D(adjncy);
        xadj = NULL;
        adjncy = NULL;
        if (status != METIS_OK) {
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            mfmem::sdel_array_1D(part);
        }
    }

    /* Replication */
    {
        //Bface
        grid->idx_pthreads_bface = NULL;
        grid->id_division_bface = NULL;
        nface_pthread = NULL;
        mfmem::snew_array_1D(grid->idx_pthreads_bface, nSubZone + 1, dmrfl);
        mfmem::snew_array_1D(nface_pthread, nSubZone, dmrfl);
        for (i = 0; i < nSubZone + 1; i++)
            grid->idx_pthreads_bface[i] = 0;
        count = 0;
        for (i = 0; i < nBFace; i++) {
            c1 = f2c[count];
            count += 2;
            subzone = (IntType)part[c1];
            grid->idx_pthreads_bface[subzone + 1]++;
        }
        for (i = 1; i < nSubZone + 1; i++) {
            grid->idx_pthreads_bface[i] += grid->idx_pthreads_bface[i - 1];
            nface_pthread[i - 1] = 0;
        }
        mfmem::snew_array_1D(grid->id_division_bface, grid->idx_pthreads_bface[nSubZone], dmrfl);
        count = 0;
        for (i = 0; i < nBFace; i++) {
            c1 = f2c[count];
            count += 2;
            subzone = (IntType)part[c1];
            grid->id_division_bface[grid->idx_pthreads_bface[subzone] + nface_pthread[subzone]] = i;
            nface_pthread[subzone]++;
        }
        //Iface
        grid->idx_pthreads_iface = NULL;
        grid->id_division_iface = NULL;
        mfmem::snew_array_1D(grid->idx_pthreads_iface, nSubZone + 1, dmrfl);
        for (i = 0; i < nSubZone + 1; i++)
            grid->idx_pthreads_iface[i] = 0;
        count = 2 * nBFace;
        for (i = nBFace; i < nTFace; i++) {
            c1 = f2c[count++];
            c2 = f2c[count++];
            subzone = (IntType)part[c1];
            grid->idx_pthreads_iface[subzone + 1]++;
            if (part[c1] != part[c2]) { //subzone boundary face,replication
                subzone = (IntType)part[c2];
                grid->idx_pthreads_iface[subzone + 1]++;
            }
        }
        for (i = 1; i < nSubZone + 1; i++) {
            grid->idx_pthreads_iface[i] += grid->idx_pthreads_iface[i - 1];
            nface_pthread[i - 1] = 0;
        }
        assert(grid->idx_pthreads_iface[nSubZone] - nTFace + nBFace == objval);
        mfmem::snew_array_1D(grid->id_division_iface, grid->idx_pthreads_iface[nSubZone], dmrfl);
        count = 2 * nBFace;
        for (i = nBFace; i < nTFace; i++) {
            c1 = f2c[count++];
            c2 = f2c[count++];
            if (part[c1] != part[c2]) { //subzone boundary face,replication
                subzone = (IntType)part[c1];
                j = i + nTFace;
                grid->id_division_iface[grid->idx_pthreads_iface[subzone] + nface_pthread[subzone]] = j;
                nface_pthread[subzone]++;
                subzone = (IntType)part[c2];
                j = -1 * (i + nTFace);
                grid->id_division_iface[grid->idx_pthreads_iface[subzone] + nface_pthread[subzone]] = j;
                nface_pthread[subzone]++;
            }
            else {
                subzone = (IntType)part[c1];
                grid->id_division_iface[grid->idx_pthreads_iface[subzone] + nface_pthread[subzone]] = i;
                nface_pthread[subzone]++;
            }
        }
        mfmem::sdel_array_1D(nface_pthread);
    }

    mfmem::sdel_array_1D(part);
    return true;
}

#ifdef BoundedColoring
/************************************************************************
               Color faces of each subzone after Divide & Replicate
                        Add by dingxin 2021-11-30
************************************************************************/
void colorAfterDivRep(PolyGrid* grid) {
    IntType i, t, k, startFace, endFace;
    IntType n_bface, n_iface;
    IntType nTFace = grid->GetNTFace();
    IntType* f2c = grid->Getf2c();
    IntType bfacenum_vec, ifacenum_vec;
    
#pragma omp parallel for private(t,i,k,startFace,endFace,n_bface,n_iface,bfacenum_vec,ifacenum_vec)
    for (t = 0; t < grid->threads; t++) {
        IntType* bface = NULL;
        IntType* iface = NULL;
        IntType* order_b = NULL;
        IntType* order_i = NULL;
        //Boundary faces
        startFace = grid->idx_pthreads_bface[t];
        grid->endIndex_bFace_vec[t] = startFace;
        endFace = grid->idx_pthreads_bface[t + 1];
        n_bface = endFace - startFace;
        mfmem::snew_array_1D(bface, n_bface, dmrfl);
        mfmem::snew_array_1D(order_b, n_bface, dmrfl);
        for (i = startFace; i < endFace; i++)
            bface[i - startFace] = grid->id_division_bface[i];

        //Interior faces
        startFace = grid->idx_pthreads_iface[t];
        grid->endIndex_iFace_vec[t] = startFace;
        endFace = grid->idx_pthreads_iface[t + 1];
        n_iface = endFace - startFace;
        mfmem::snew_array_1D(iface, n_iface, dmrfl);
        mfmem::snew_array_1D(order_i, n_iface, dmrfl);
        for (i = startFace; i < endFace; i++) {
            k = grid->id_division_iface[i];
            if (abs(k) < nTFace)
                iface[i - startFace] = k;
            else
                iface[i - startFace] = abs(k) - nTFace;
        }
        BoundedColoring(f2c, bface, iface, order_b, order_i, n_bface, n_iface, bfacenum_vec, ifacenum_vec);
        grid->endIndex_bFace_vec[t] += bfacenum_vec;
        grid->endIndex_iFace_vec[t] += ifacenum_vec;
        updateOrder(grid->id_division_bface, order_b, grid->idx_pthreads_bface[t], n_bface);
        updateOrder(grid->id_division_iface, order_i, startFace, n_iface);
#ifdef DEBUG
        {
            updateOrder(bface, order_b, 0, n_bface);
            if (!validation_decoupling(bface, 0, bfacenum_vec, f2c, BOUNDARYFACE)) {
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
            updateOrder(iface, order_i, 0, n_iface);
            if (!validation_decoupling(iface, 0, ifacenum_vec, f2c, INTERIORFACE)) {
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }
        }
#endif // DEBUG
        mfmem::sdel_array_1D(bface);
        mfmem::sdel_array_1D(iface);
        mfmem::sdel_array_1D(order_b);
        mfmem::sdel_array_1D(order_i);
    }
}
#endif // BoundedColoring
#endif // DIVREP

#ifdef DIVCON
// Permute "tab" 2D array of int using "perm"
void DC_permute_int_2d_array(IntType* elemToNode, IntType* localPerm, IntType n_localPerm, IntType* dimItem, IntType offset) {
    IntType i, j, tmp;
    IntType* localElemToNode = NULL;
    IntType* localDimItem = NULL;
    IntType num, endIndex_global;
    num = dimItem[n_localPerm + offset] - dimItem[offset];
    mfmem::snew_array_1D(localElemToNode, num, dmrfl);
    mfmem::snew_array_1D(localDimItem, n_localPerm + 1, dmrfl);
    localDimItem[0] = 0;
    for (i = 0; i < n_localPerm; i++) {
        tmp = i + offset;//global id of elem
        localDimItem[localPerm[i] + 1] = dimItem[tmp + 1] - dimItem[tmp];
    }
    for (i = 1; i < n_localPerm + 1; i++) {
        localDimItem[i] += localDimItem[i - 1];
    }
    //generate localElemToNode from global elemToNode
    for (i = 0; i < n_localPerm; i++) {
        endIndex_global = dimItem[i + offset + 1];
        tmp = localDimItem[localPerm[i]];
        num = 0;
        for (j = dimItem[i + offset]; j < endIndex_global; j++) {
            localElemToNode[tmp + num] = elemToNode[j];
            num++;
        }
    }
    //merge local & global
    tmp = dimItem[offset];
    for (i = 0; i < localDimItem[n_localPerm]; i++) {
        elemToNode[tmp + i] = localElemToNode[i];
    }
    for (i = 1; i < n_localPerm + 1; i++) {
        dimItem[i + offset] = localDimItem[i] + tmp;
    }

    mfmem::sdel_array_1D(localElemToNode);
    mfmem::sdel_array_1D(localDimItem);
}

// Apply local element permutation to global element permutation
void merge_permutations(IntType* elemPerm, IntType* localElemPerm, IntType globalNbElem, IntType localNbElem,
    IntType firstElem, IntType lastElem)
{
    IntType i, dst, ctr = 0;
    for (i = 0; i < globalNbElem; i++) {
        dst = elemPerm[i];
        if (dst >= firstElem && dst <= lastElem) {
            elemPerm[i] = localElemPerm[dst - firstElem] + firstElem;
            ctr++;
        }
        if (ctr == localNbElem)	break;
    }
}

// Create permutation array from partition array
void DC_create_permutation(IntType* perm, idx_t* part, IntType size, IntType nbPart)
{
    IntType ptr = 0, i, j;
    for (i = 0; i < nbPart; i++) {
        for (j = 0; j < size; j++) {
            if (part[j] == i) {
                perm[j] = ptr;
                ptr++;
            }
        }
    }
}

// Initialize the content of D&C tree nodes
void init_dc_tree(tree_t*& tree, PolyGrid* grid, IntType* cface_parent, IntType n_cface, IntType n_cface_compute,
    IntType firstElem, IntType lastElem, IntType nbSepElem, bool isSep, bool isLeaf)
{
    /*
    ifaceType:1. compute, then write back to c1 and c2
              2. compute, and write back to c1
              3. compute, and write back to c2
              4. get data of face, and write back to c1
              5. get data of face, and write back to c2
     */
    tree->firstElem = firstElem;
    tree->lastElem = lastElem - nbSepElem;
    tree->lastSep = lastElem;
    tree->vecOffset = 0;
    tree->isSep = isSep;
    tree->left = NULL;
    tree->right = NULL;
    tree->sep = NULL;
    tree->bfaceID = NULL;
    tree->ifaceID = NULL;
    tree->ifaceType = NULL;

    if (isLeaf == false) {
        tree->n_bface = 0;
        tree->n_iface = 0;
        mfmem::snew_array_1D(tree->left, 1, dmrfl);
        mfmem::snew_array_1D(tree->right, 1, dmrfl);
        if (nbSepElem > 0) {
            mfmem::snew_array_1D(tree->sep, 1, dmrfl);
        }
    }
    else {
        if (!isSep && n_cface != n_cface_compute)
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        IntType i, j, c1, c2, count, tmp;
        IntType n_bface = 0, n_iface = 0;
        IntType* iface_type = NULL;
        IntType* elemPerm = grid->cellPerm;
        IntType* f2c = grid->Getf2c();
        IntType nBFace = grid->GetNBFace();
        IntType nTFace = grid->GetNTFace();
        //count number of bface and number of iface for current leaf node
        count = 0;
        for (i = 0; i < nBFace; i++) {
            c1 = f2c[count];
            count += 2;
            c1 = elemPerm[c1];
            if (c1 >= firstElem && c1 <= lastElem) {
                n_bface++;
            }
        }
        count = 2 * nBFace;
        for (i = nBFace; i < nTFace; i++) {
            c1 = f2c[count++];
            c2 = f2c[count++];
            c1 = elemPerm[c1];
            c2 = elemPerm[c2];
            if ((c1 >= firstElem && c1 <= lastElem) && (c2 >= firstElem && c2 <= lastElem)) {
                n_iface++;
            }
        }
        n_iface += n_cface;

        tree->n_bface = n_bface;
        tree->n_iface = n_iface;
        mfmem::snew_array_1D(tree->bfaceID, n_bface, dmrfl);
        mfmem::snew_array_1D(tree->ifaceID, n_iface, dmrfl);
        mfmem::snew_array_1D(tree->ifaceType, n_iface, dmrfl);
        n_bface = 0;
        n_iface = 0;
        count = 0;
        for (i = 0; i < nBFace; i++) {
            c1 = f2c[count];
            count += 2;
            c1 = elemPerm[c1];
            if (c1 >= firstElem && c1 <= lastElem) {
                tree->bfaceID[n_bface++] = i;
            }
        }

        mfmem::snew_array_1D(iface_type, nTFace-nBFace, dmrfl);
        memset(iface_type, 0, (nTFace - nBFace) * sizeof(IntType));
        for (i = 0; i < n_cface_compute; i++) {
            tmp = cface_parent[i];
            if (tmp > 0)
                iface_type[tmp - nBFace] = 2;
            else
                iface_type[-1 * tmp - nBFace] = 3;
        }
        for (i = n_cface_compute; i < n_cface; i++) {
            tmp = cface_parent[i];
            if (tmp > 0)
                iface_type[tmp - nBFace] = 4;
            else
                iface_type[-1 * tmp - nBFace] = 5;
        }

        count = 2 * nBFace;
        for (i = nBFace; i < nTFace; i++) {
            c1 = f2c[count++];
            c2 = f2c[count++];
            c1 = elemPerm[c1];
            c2 = elemPerm[c2];
            if ((c1 >= firstElem && c1 <= lastElem) && (c2 >= firstElem && c2 <= lastElem)) {
                tree->ifaceID[n_iface] = i;
                tree->ifaceType[n_iface] = 1;
                n_iface++;
                continue;
            }
            tmp = iface_type[i - nBFace];
            if (tmp == 0)
                continue;
            else {
                tree->ifaceID[n_iface] = i;
                tree->ifaceType[n_iface] = tmp;
                n_iface++;
            }
        }
        mfmem::sdel_array_1D(iface_type);
    }
}

// Create element partition & count left & separator elements
void create_elem_part(idx_t* elemPart, idx_t* nodePart, IntType* elemToNode, IntType nbElem,
        IntType* dimElem, IntType separator, IntType offset, IntType* nbLeftElem, IntType* nbSepElem) {
    IntType i, j, leftCtr, rightCtr, tmp, nodes;
    for (i = 0; i < nbElem; i++) {
        leftCtr = 0;
        rightCtr = 0;
        tmp = dimElem[i + offset + 1];
        nodes = tmp - dimElem[i + offset];
        for (j = dimElem[i + offset]; j < tmp; j++) {
            if (nodePart[elemToNode[j]] <= separator)
                leftCtr++;
            else
                rightCtr++;
        }
        if (leftCtr == nodes) {
            elemPart[i] = 0;
            (*nbLeftElem)++;
        }
        else if (rightCtr == nodes) {
            elemPart[i] = 1;
        }
        else {
            elemPart[i] = 2;
            (*nbSepElem)++;
        }
    }
}

void get_common_face(PolyGrid* grid, IntType*& cface_left, IntType*& cface_right, IntType*& cface_sep, IntType* cface_parent,
    IntType* nFace, IntType firstElem, IntType lastElem, IntType nbLeftElem, IntType nbSepElem)
{
    IntType i, c1, c2, count, tmp;
    IntType n_cl, n_cr, n_sep;
    IntType n_cface_compute, n_cface, index_sep, index_right;
    IntType* elemPerm = grid->cellPerm;
    IntType* f2c = grid->Getf2c();
    IntType nBFace = grid->GetNBFace();
    IntType nTFace = grid->GetNTFace();

    index_sep = lastElem - nbSepElem + 1;//start of seperator elems
    index_right = firstElem + nbLeftElem;//start of right elems
    n_cl = 0;
    n_cr = 0;
    n_sep = 0;
    n_cface = nFace[0];
    n_cface_compute = nFace[1];

    //count cface from parent
    for (i = 0; i < n_cface; i++) {
        tmp = 2 * abs(cface_parent[i]);
        c1 = f2c[tmp++];
        c2 = f2c[tmp];
        c1 = elemPerm[c1];
        c2 = elemPerm[c2];
        if ((c1 >= firstElem && c1 < index_right) || (c2 >= firstElem && c2 < index_right))
            n_cl++;
        else if ((c1 >= index_right && c1 < index_sep) || (c2 >= index_right && c2 < index_sep))
            n_cr++;
        else if ((c1 >= index_sep && c1 <= lastElem) || (c2 >= index_sep && c2 <= lastElem))
            n_sep++;
        else
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    tmp = n_cl + n_cr;
    //count cface in current node
    count = 2 * nBFace;
    for (i = nBFace; i < nTFace; i++) {
        c1 = f2c[count++];
        c2 = f2c[count++];
        c1 = elemPerm[c1];
        c2 = elemPerm[c2];
        if ((c1 >= firstElem && c1 <= lastElem) && (c2 >= firstElem && c2 <= lastElem)) {
            if (c1 >= index_sep && c2 < index_sep) {
                if (c2 < index_right) {//c2 in left, c1 in seperator
                    n_cl++;
                }
                else {//c2 in right, c1 in seperator
                    n_cr++;
                }
            }
            if (c2 >= index_sep && c1 < index_sep) {
                if (c1 < index_right) {//c1 in left, c2 in seperator
                    n_cl++;
                }
                else {//c1 in right, c2 in seperator
                    n_cr++;
                }
            }
        }
    }
    n_sep = n_sep + (n_cl + n_cr - tmp);
    mfmem::snew_array_1D(cface_left, n_cl, dmrfl);
    mfmem::snew_array_1D(cface_right, n_cr, dmrfl);
    mfmem::snew_array_1D(cface_sep, n_sep, dmrfl);
    nFace[2] = n_cl;
    nFace[4] = n_cr;
    nFace[6] = n_sep;
    n_cl = 0;
    n_cr = 0;
    n_sep = 0;
    for (i = 0; i < n_cface_compute; i++) {
        tmp = 2 * abs(cface_parent[i]);
        c1 = f2c[tmp++];
        c2 = f2c[tmp];
        c1 = elemPerm[c1];
        c2 = elemPerm[c2];
        if ((c1 >= firstElem && c1 < index_right) || (c2 >= firstElem && c2 < index_right)) {
            cface_left[n_cl] = cface_parent[i];
            n_cl++;
        }
        else if ((c1 >= index_right && c1 < index_sep) || (c2 >= index_right && c2 < index_sep)) {
            cface_right[n_cr] = cface_parent[i];
            n_cr++;
        }
        else {
            cface_sep[n_sep] = cface_parent[i];
            n_sep++;
        }
    }
    nFace[7] = n_sep;
    count = 2 * nBFace;
    for (i = nBFace; i < nTFace; i++) {
        c1 = f2c[count++];
        c2 = f2c[count++];
        c1 = elemPerm[c1];
        c2 = elemPerm[c2];
        if ((c1 >= firstElem && c1 <= lastElem) && (c2 >= firstElem && c2 <= lastElem)) {
            if (c1 >= index_sep && c2 < index_sep) {
                cface_sep[n_sep++] = i;
                if (c2 < index_right) {//c2 in left, c1 in seperator
                    cface_left[n_cl] = -1 * i;
                    n_cl++;
                }
                else {//c2 in right, c1 in seperator
                    cface_right[n_cr] = -1 * i;
                    n_cr++;
                }
            }
            if (c2 >= index_sep && c1 < index_sep) {
                cface_sep[n_sep++] = -1 * i;
                if (c1 < index_right) {//c1 in left, c2 in seperator
                    cface_left[n_cl] = i;
                    n_cl++;
                }
                else {//c1 in right, c2 in seperator
                    cface_right[n_cr] = i;
                    n_cr++;
                }
            }
        }
    }
    nFace[3] = n_cl;
    nFace[5] = n_cr;
    for (i = n_cface_compute; i < n_cface; i++) {
        tmp = 2 * abs(cface_parent[i]);
        c1 = f2c[tmp++];
        c2 = f2c[tmp];
        c1 = elemPerm[c1];
        c2 = elemPerm[c2];
        if ((c1 >= firstElem && c1 < index_right) || (c2 >= firstElem && c2 < index_right)) {
            cface_left[n_cl] = cface_parent[i];
            n_cl++;
        }
        else if ((c1 >= index_right && c1 < index_sep) || (c2 >= index_right && c2 < index_sep)) {
            cface_right[n_cr] = cface_parent[i];
            n_cr++;
        }
        else {
            cface_sep[n_sep] = cface_parent[i];
            n_sep++;
        }
    }
}

// Create the D&C tree and the element permutation, and compute the intervals of nodes
// and elements at each node of the tree
void tree_creation(tree_t*& tree, PolyGrid* grid, IntType* elemToNode, IntType* sepToNode, idx_t* nodePart,
        IntType globalNbElem, IntType* dimElem, IntType* cface_parent, IntType n_cface, IntType n_cface_compute,
        IntType firstPart, IntType lastPart, IntType firstElem, IntType lastElem, IntType sepOffset, bool isSep) {
    IntType nbPart = lastPart - firstPart + 1;
    IntType localNbElem = lastElem - firstElem + 1;

    // If current node is a leaf
    if (nbPart < 2 || localNbElem <= MAX_ELEM_PER_PART) {
        // Initialize the leaf
        init_dc_tree(tree, grid, cface_parent, n_cface, n_cface_compute, firstElem, lastElem, 0, isSep, true);
        return;
    }
    // Else, prepare next left, right & separator recursion
    IntType nbLeftElem = 0, nbSepElem = 0;
    idx_t* localElemPart = NULL;
    IntType* localElemPerm = NULL;
    IntType separator = firstPart + (lastPart - firstPart) / 2;

    // Create local element partition & count left & separator elements
    mfmem::snew_array_1D(localElemPart, localNbElem, dmrfl);
    if (isSep) {
        create_elem_part(localElemPart, nodePart, sepToNode, localNbElem, dimElem,
            separator, sepOffset, &nbLeftElem, &nbSepElem);
    }
    else {
        create_elem_part(localElemPart, nodePart, elemToNode, localNbElem, dimElem,
            separator, firstElem, &nbLeftElem, &nbSepElem);
    }
    // Create local element permutation
    mfmem::snew_array_1D(localElemPerm, localNbElem, dmrfl);
    DC_create_permutation(localElemPerm, localElemPart, localNbElem, 3);
    mfmem::sdel_array_1D(localElemPart);

    // Apply local element permutation to global element permutation
    merge_permutations(grid->cellPerm, localElemPerm, globalNbElem, localNbElem, firstElem, lastElem);//lock if omp!!!
    
    // Find common face between left and seperator ,or right and seperator, and merge cface
    IntType* cface_left = NULL;
    IntType* cface_right = NULL;
    IntType* cface_sep = NULL;
    IntType* nFace = NULL;//
    IntType n_cl, n_cl_compute, n_cr, n_cr_compute, n_sep, n_sep_compute;
    /*index of nFace from 2 ot 7->left common faces, left common faces required computing
                                  right common faces, right common faces required computing
                                  seperator common faces, seperator common faces required computing*/
    mfmem::snew_array_1D(nFace, 8, dmrfl);
    nFace[0] = n_cface;
    nFace[1] = n_cface_compute;
    get_common_face(grid, cface_left, cface_right, cface_sep, cface_parent,
        nFace, firstElem, lastElem, nbLeftElem, nbSepElem);
    n_cl = nFace[2];
    n_cl_compute= nFace[3];
    n_cr = nFace[4];
    n_cr_compute= nFace[5];
    n_sep = nFace[6];
    n_sep_compute = nFace[7];
    mfmem::sdel_array_1D(nFace);

    // Permute elemToNode, sepToNode, and dimElem with local element permutation
    if (isSep) {
        DC_permute_int_2d_array(sepToNode, localElemPerm, localNbElem, dimElem, sepOffset);
    }
    else {
        DC_permute_int_2d_array(elemToNode, localElemPerm, localNbElem, dimElem, firstElem);
    }
    mfmem::sdel_array_1D(localElemPerm);

    // Initialize current node
    init_dc_tree(tree, grid, cface_parent, n_cface, n_cface_compute, firstElem, lastElem, nbSepElem, isSep, false);

    // Left & right recursion
    tree_creation(tree->right, grid, elemToNode, sepToNode, nodePart,
        globalNbElem, dimElem, cface_right, n_cr, n_cr_compute, separator + 1, lastPart, firstElem + nbLeftElem, lastElem - nbSepElem,
        sepOffset + nbLeftElem, isSep);
    tree_creation(tree->left, grid, elemToNode, sepToNode, nodePart,
        globalNbElem, dimElem, cface_left, n_cl, n_cl_compute, firstPart, separator, firstElem, firstElem + nbLeftElem - 1,
        sepOffset, isSep);
    mfmem::sdel_array_1D(cface_left);
    mfmem::sdel_array_1D(cface_right);
    // Synchronization
    //#pragma omp taskwait
    // D&C partitioning of separator elements
    if (nbSepElem > 0) {
        sep_partitioning(tree->sep, grid, elemToNode, globalNbElem, dimElem, cface_sep, n_sep, n_sep_compute, lastElem -
            nbSepElem + 1, lastElem, isSep);
    }
    mfmem::sdel_array_1D(cface_sep);
}

// Create a nodal graph
void mesh_to_nodal(idx_t*& graphIndex, idx_t*& graphValue, IntType* elemToNode, IntType nbElem,
        IntType* dimElem, IntType nbNodes) {
    idx_t index;
    IntType i, j, k, cellId, nodeId, tmp;
    IntType* index_n2e = NULL, * nodeToElem = NULL, * marker = NULL;
    mfmem::snew_array_1D(index_n2e, nbNodes + 1, dmrfl);
    memset(index_n2e, 0, (nbNodes + 1) * sizeof(IntType));
    for (i = 0; i < dimElem[nbElem]; i++) {//count the number of elem per node
        index_n2e[elemToNode[i]]++;
    }
    for (i = 1; i < nbNodes; i++) {
        index_n2e[i] += index_n2e[i - 1];
    }
    for (i = nbNodes; i > 0; i--) {
        index_n2e[i] = index_n2e[i - 1];
    }
    index_n2e[0] = 0;
    mfmem::snew_array_1D(nodeToElem, index_n2e[nbNodes], dmrfl);
    for (i = 0; i < nbElem; i++) {
        tmp = dimElem[i + 1];
        for (j = dimElem[i]; j < tmp; j++) {
            nodeToElem[index_n2e[elemToNode[j]]++] = i;
        }
    }
    for (i = nbNodes; i > 0; i--) {
        index_n2e[i] = index_n2e[i - 1];
    }
    index_n2e[0] = 0;
    mfmem::snew_array_1D(marker, nbNodes, dmrfl);
    mfmem::snew_array_1D(graphIndex, nbNodes + 1, dmrfl);
    for (i = 0; i < nbNodes; i++)
        marker[i] = -1;
    graphIndex[0] = 0;
    index = 0;
    for (i = 0; i < nbNodes; i++) {
        marker[i] = i;
        for (j = index_n2e[i]; j < index_n2e[i + 1]; j++) {
            cellId = nodeToElem[j];
            tmp= dimElem[cellId + 1];
            for (k = dimElem[cellId]; k < tmp; k++) {
                nodeId = elemToNode[k];
                if (marker[nodeId] != i) {
                    marker[nodeId] = i;
                    index++;
                }
            }
        }
        graphIndex[i + 1] = index;
    }
    mfmem::snew_array_1D(graphValue, index, dmrfl);
    for (i = 0; i < nbNodes; i++)
        marker[i] = -1;
    index = 0;
    for (i = 0; i < nbNodes; i++) {
        marker[i] = i;
        for (j = index_n2e[i]; j < index_n2e[i + 1]; j++) {
            cellId = nodeToElem[j];
            tmp = dimElem[cellId + 1];
            for (k = dimElem[cellId]; k < tmp; k++) {
                nodeId = elemToNode[k];
                if (marker[nodeId] != i) {
                    marker[nodeId] = i;
                    graphValue[index++] = nodeId;
                }
            }
        }
    }
    mfmem::sdel_array_1D(index_n2e);
    mfmem::sdel_array_1D(nodeToElem);
    mfmem::sdel_array_1D(marker);
}

// Create local elemToNode array containing elements indexed contiguously from 0 to
// nbElem and return the number of nodes of separator elements
IntType create_sepToNode(IntType*& sepToNode, IntType* index_sepToNode, IntType* elemToNode,
        IntType firstSepElem, IntType lastSepElem, IntType* dimElem) {
    IntType i, j, start, end;
    IntType newNode, oldNode, nbNodes;
    bool isNew;
    IntType* tmp = NULL;

    //build index_sepToNode
    index_sepToNode[0] = 0;
    start = dimElem[firstSepElem];
    end = lastSepElem - firstSepElem + 2;
    for (i = 1; i < end; i++) {
        index_sepToNode[i] = dimElem[firstSepElem + i] - start;
    }

    //build sepToNode
    end = dimElem[lastSepElem + 1];
    mfmem::snew_array_1D(sepToNode, end - start, dmrfl);
    mfmem::snew_array_1D(tmp, end - start, dmrfl);
    for (i = 0; i < end - start; i++) {
        tmp[i] = -1;
    }
    nbNodes = 0;
    for (i = start, j = 0; i < end; i++, j++) {
        newNode = 0;
        oldNode = elemToNode[i];
        isNew = true;
        for (; newNode < nbNodes; newNode++) {
            if (oldNode == tmp[newNode]) {
                isNew = false;
                break;
            }
        }
        if (isNew) {
            tmp[nbNodes] = oldNode;
            nbNodes++;
        }
        sepToNode[j] = newNode;
    }
    
    mfmem::sdel_array_1D(tmp);
    return nbNodes;
}

// D&C partitioning of separators with more than MAX_ELEM_PER_PART elements
void sep_partitioning(tree_t*& tree, PolyGrid* grid, IntType* elemToNode, IntType globalNbElem, IntType* dimElem,
        IntType* cface_parent, IntType n_cface, IntType n_cface_compute, IntType firstSepElem, IntType lastSepElem, bool isSep) {
    IntType i, nbSepNodes;
    IntType* sepToNode = NULL;
    IntType* index_sepToNode = NULL;
    idx_t* xadj = NULL;//csr,start and end index
    idx_t* adjncy = NULL;//csr,node id
    idx_t nvtxs, ncon, nparts, objval;
    idx_t* nodePart = NULL;
    IntType status;
    // If there is not enough element in the separator
    int nbSepElem = lastSepElem - firstSepElem + 1;
    int nbSepPart = ceil(nbSepElem / (double)MAX_ELEM_PER_PART);
    if (nbSepPart < 2 || nbSepElem <= MAX_ELEM_PER_PART || isSep) {
        // Initialize the leaf
        init_dc_tree(tree, grid, cface_parent, n_cface, n_cface_compute, firstSepElem, lastSepElem, 0, true, true);
        return;
    }
    // Create temporal elemToNode containing the separator elements
    mfmem::snew_array_1D(index_sepToNode, lastSepElem - firstSepElem + 2, dmrfl);
    nbSepNodes = create_sepToNode(sepToNode, index_sepToNode, elemToNode, firstSepElem, lastSepElem, dimElem);
    /* Division by Metis*/
    {
        nparts = (idx_t)nbSepPart;
        nvtxs = (idx_t)nbSepNodes;
        ncon = 1;
        mfmem::snew_array_1D(nodePart, static_cast<size_t>(nvtxs), dmrfl);
        for (i = 0; i < nvtxs; i++) {
            nodePart[i] = 0;
        }
        mesh_to_nodal(xadj, adjncy, sepToNode, nbSepElem, index_sepToNode, nbSepNodes);
        status = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, NULL,
            &nparts, NULL, NULL, NULL, &objval, nodePart);
        mfmem::sdel_array_1D(xadj);
        mfmem::sdel_array_1D(adjncy);
    }
    // Create the separator D&C tree
    tree_creation(tree, grid, elemToNode, sepToNode, nodePart, globalNbElem, index_sepToNode, cface_parent,
        n_cface, n_cface_compute, 0, nbSepPart - 1, firstSepElem, lastSepElem, 0, true);
    mfmem::sdel_array_1D(sepToNode);
    mfmem::sdel_array_1D(nodePart);
    mfmem::sdel_array_1D(index_sepToNode);
}

// Divide & Conquer partitioning
void partitioning(IntType* elemToNode, IntType nbElem, IntType* dimElem, IntType nbNodes, PolyGrid* grid) {
    IntType i;

    idx_t* xadj = NULL;//csr,start and end index
    idx_t* adjncy = NULL;//csr,node id
    idx_t nvtxs, ncon, nparts, objval;
    idx_t* nodePart = NULL;
    IntType status;

    /* Division by Metis*/
    {
        nparts = (idx_t)ceil(nbElem / (double)MAX_ELEM_PER_PART);
        nvtxs = (idx_t)nbNodes;
        ncon = 1;
        mfmem::snew_array_1D(nodePart, static_cast<size_t>(nvtxs), dmrfl);
        for (i = 0; i < nvtxs; i++) {
            nodePart[i] = 0;
        }
        mesh_to_nodal(xadj, adjncy, elemToNode, nbElem, dimElem, nbNodes);
        status = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, NULL,
            &nparts, NULL, NULL, NULL, &objval, nodePart);
        mfmem::sdel_array_1D(xadj);
        mfmem::sdel_array_1D(adjncy); 
    }

    // Initialize the global element permutation
    for (i = 0; i < nbElem; i++) {
        grid->cellPerm[i] = i;
    }
    // Create D&C tree
    tree_creation(grid->treeHead, grid, elemToNode, NULL, nodePart, nbElem,
        dimElem, NULL, 0, 0, 0, nparts - 1, 0, nbElem - 1, 0, false);
    mfmem::sdel_array_1D(nodePart);
}

// Free memory
void tree_free(tree_t*& tree) {
    if (tree->left == NULL && tree->right == NULL) {//Leaf
        mfmem::sdel_array_1D(tree->bfaceID);
        mfmem::sdel_array_1D(tree->ifaceID);
        mfmem::sdel_array_1D(tree->ifaceType);
    }
    else {
        tree_free(tree->left);
        tree_free(tree->right);
        mfmem::sdel_array_1D(tree->left);
        mfmem::sdel_array_1D(tree->right);
        if (tree->sep != NULL) {
            tree_free(tree->sep);
            mfmem::sdel_array_1D(tree->sep);
        }
    }
}

void tree_test(tree_t*& tree, IntType* tag) {
    IntType i;
    if (tree->left == NULL && tree->right == NULL) {//Leaf
        for (i = 0; i < tree->n_bface; i++)
            tag[tree->bfaceID[i]]++;
        for (i = 0; i < tree->n_iface; i++) {
            if (tree->ifaceType[i] < 4)
                tag[tree->ifaceID[i]]++;
        }
    }
    else {
        tree_test(tree->left, tag);
        tree_test(tree->right, tag);
        if (tree->sep != NULL) {
            tree_test(tree->sep, tag);
        }
    }
}

/************************************************************************
					  Divide and conquer approach
					   Add by dingxin 2021-11-10
************************************************************************/
void DC_create_tree(PolyGrid* grid)
{
    // Allocate the D&C tree & the permutation functions
    IntType nTCell, nTNode;
    IntType* c2n = NULL;//elemToNode
    IntType* f2c = NULL;
    IntType* index_c2n = NULL;
    IntType** C2N = CalC2N(grid);
    IntType* nNPC = CalnNPC(grid);
    IntType nTFace = grid->GetNTFace();
    
    f2c = grid->Getf2c();
    nTCell = grid->GetNTCell();
    nTNode = grid->GetNTNode();
    grid->treeHead = NULL;
    grid->cellPerm = NULL;
    mfmem::snew_array_1D(grid->treeHead, 1, dmrfl);
    mfmem::snew_array_1D(grid->cellPerm, nTCell, dmrfl);

    mfmem::snew_array_1D(index_c2n, nTCell + 1, dmrfl);
    index_c2n[0] = 0;
    for (IntType i = 1; i < nTCell + 1; i++) {
        index_c2n[i] = index_c2n[i - 1] + nNPC[i - 1];
    }
    mfmem::snew_array_1D(c2n, index_c2n[nTCell], dmrfl);
    for (IntType i = 0; i < nTCell; i++) {
        IntType end = index_c2n[i + 1];
        IntType tmp = 0;
        for (IntType j = index_c2n[i]; j < end; j++) {
            c2n[j] = C2N[i][tmp++];
        }
    }
    // Create the D&C tree & the permutation functions
    partitioning(c2n, nTCell, index_c2n, nTNode, grid);

    // finish
    mfmem::sdel_array_1D(grid->cellPerm);
    mfmem::sdel_array_1D(index_c2n);
    mfmem::sdel_array_1D(c2n);

#ifdef DEBUG
    IntType* tag = NULL;
    mfmem::snew_array_1D(tag, nTFace, dmrfl);
    memset(tag, 0, nTFace * sizeof(IntType));
    tree_test(grid->treeHead, tag);
    for (IntType i = 0; i < nTFace; i++) {
        if (tag[i] != 1) {
            tree_free(grid->treeHead);
            mfmem::sdel_array_1D(tag);
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
    mfmem::sdel_array_1D(tag);
    cout << "D&C tree has been created. " << endl;
#endif
    
    // Vectorial version with coloring of the leaves of the D&C tree
#ifdef FS_SIMD
    coloring(elemToNode, nbElem, dimElem, nbNodes);//future
#endif
}

//traverse tree for CompGradientQ_Gauss_Node
void tree_traversal(tree_t*& tree, RealFlow* dqdx, RealFlow* dqdy, RealFlow* dqdz, RealGeom* tmpxyz,
        IntType* f2c, BCRecord** bcr, IntType* nNPF, IntType** F2N, RealFlow* q_n, RealFlow* q,
        RealGeom* area, RealGeom* xfn, RealGeom* yfn, RealGeom* zfn, IntType nBFace) {
    if (tree->left == NULL && tree->right == NULL) {//Leaf
        IntType i, j, c1, c2, type, count, id;
        RealFlow qsum;
        RealGeom tmpx, tmpy, tmpz;
        for (i = 0; i < tree->n_bface; i++) {
            id = tree->bfaceID[i];
            count = 2 * id;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            type = bcr[id]->GetType();
            qsum = 0.0;
            if (type == INTERFACE || type == SYMM) {
                for (j = 0; j < nNPF[id]; j++)
                    qsum += q_n[F2N[id][j]];
                qsum /= RealFlow(nNPF[id]);
            }
            else {
                qsum = 0.5 * (q[c1] + q[c2]);
            }

            qsum *= area[id];
            tmpx = qsum * xfn[id];
            tmpy = qsum * yfn[id];
            tmpz = qsum * zfn[id];
            dqdx[c1] += tmpx;
            dqdy[c1] += tmpy;
            dqdz[c1] += tmpz;
        }
        for (i = 0; i < tree->n_iface; i++) {
            id = tree->ifaceID[i];
            count = 2 * id;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            count = 3 * (id - nBFace);
            if (tree->ifaceType[i] < 4) { //compute face data, and storage
                qsum = 0.0;

                for (j = 0; j < nNPF[id]; j++)
                    qsum += q_n[F2N[id][j]];
                qsum /= RealFlow(nNPF[id]);

                qsum *= area[id];
                tmpx = qsum * xfn[id];
                tmpy = qsum * yfn[id];
                tmpz = qsum * zfn[id];
                tmpxyz[count++] = tmpx;
                tmpxyz[count++] = tmpy;
                tmpxyz[count] = tmpz;
            }
            else {
                tmpx = tmpxyz[count++];
                tmpy = tmpxyz[count++];
                tmpz = tmpxyz[count];
            }
            if (tree->ifaceType[i] == 1) { //write back to c1 and c2
                dqdx[c1] += tmpx;
                dqdy[c1] += tmpy;
                dqdz[c1] += tmpz;
                dqdx[c2] -= tmpx;
                dqdy[c2] -= tmpy;
                dqdz[c2] -= tmpz;
            }
            else if (tree->ifaceType[i] == 2 || tree->ifaceType[i] == 4) { //just write back to c1
                dqdx[c1] += tmpx;
                dqdy[c1] += tmpy;
                dqdz[c1] += tmpz;
            }
            else if (tree->ifaceType[i] == 3 || tree->ifaceType[i] == 5) { //just write back to c2
                dqdx[c2] -= tmpx;
                dqdy[c2] -= tmpy;
                dqdz[c2] -= tmpz;
            }
            else
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
    else {
#pragma omp task default(shared)
        tree_traversal(tree->left, dqdx, dqdy, dqdz, tmpxyz,
            f2c, bcr, nNPF, F2N, q_n, q, area, xfn, yfn, zfn, nBFace);
#pragma omp task default(shared)
        tree_traversal(tree->right, dqdx, dqdy, dqdz, tmpxyz,
             f2c, bcr, nNPF, F2N, q_n, q, area, xfn, yfn, zfn, nBFace);
#pragma omp taskwait
        if (tree->sep != NULL) {
            tree_traversal(tree->sep, dqdx, dqdy, dqdz, tmpxyz,
                 f2c, bcr, nNPF, F2N, q_n, q, area, xfn, yfn, zfn, nBFace);
        }
    }
}
//traverse tree for NS LoadFlux
void tree_traversal(tree_t*& tree, RealFlow** res, RealGeom** flux, IntType* f2c) {
    if (tree->left == NULL && tree->right == NULL) {//Leaf
        IntType i, c1, c2, count, id;
        //bFace
        for (i = 0; i < tree->n_bface; i++) {
            id = tree->bfaceID[i];
            c1 = f2c[2 * id];
            res[0][c1] -= flux[0][id];
            res[1][c1] -= flux[1][id];
            res[2][c1] -= flux[2][id];
            res[3][c1] -= flux[3][id];
            res[4][c1] -= flux[4][id];
        }
        //iFace
        for (i = 0; i < tree->n_iface; i++) {
            id = tree->ifaceID[i];
            count = 2 * id;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            if (tree->ifaceType[i] == 1) { //write back to c1 and c2
                res[0][c1] -= flux[0][id];
                res[1][c1] -= flux[1][id];
                res[2][c1] -= flux[2][id];
                res[3][c1] -= flux[3][id];
                res[4][c1] -= flux[4][id];

                res[0][c2] += flux[0][id];
                res[1][c2] += flux[1][id];
                res[2][c2] += flux[2][id];
                res[3][c2] += flux[3][id];
                res[4][c2] += flux[4][id];
            }
            else if (tree->ifaceType[i] == 2 || tree->ifaceType[i] == 4) { //just write back to c1
                res[0][c1] -= flux[0][id];
                res[1][c1] -= flux[1][id];
                res[2][c1] -= flux[2][id];
                res[3][c1] -= flux[3][id];
                res[4][c1] -= flux[4][id];
            }
            else if (tree->ifaceType[i] == 3 || tree->ifaceType[i] == 5) { //just write back to c2
                res[0][c2] += flux[0][id];
                res[1][c2] += flux[1][id];
                res[2][c2] += flux[2][id];
                res[3][c2] += flux[3][id];
                res[4][c2] += flux[4][id];
            }
            else
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
    else {
#pragma omp task default(shared)
        tree_traversal(tree->left, res, flux, f2c);
#pragma omp task default(shared)
        tree_traversal(tree->right, res, flux, f2c);
#pragma omp taskwait
        if (tree->sep != NULL) {
            tree_traversal(tree->sep, res, flux, f2c);
        }
    }
}

//traverse tree for SA LoadFlux
void tree_traversal(tree_t*& tree, RealFlow** res, RealGeom** flux, IntType nVar, IntType* f2c) {
    if (tree->left == NULL && tree->right == NULL) {//Leaf
        IntType i, j, c1, c2, count, id;
        //bFace
        for (i = 0; i < tree->n_bface; i++) {
            id = tree->bfaceID[i];
            c1 = f2c[2 * id];
            for (j = 0; j < nVar; j++) {
                res[j][c1] -= flux[j][id];
            }
        }
        //iFace
        for (i = 0; i < tree->n_iface; i++) {
            id = tree->ifaceID[i];
            count = 2 * id;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            if (tree->ifaceType[i] == 1) { //write back to c1 and c2
                for (j = 0; j < nVar; j++) {
                    res[j][c1] -= flux[j][id];
                    res[j][c2] += flux[j][id];
                }
            }
            else if (tree->ifaceType[i] == 2 || tree->ifaceType[i] == 4) { //just write back to c1
                for (j = 0; j < nVar; j++) {
                    res[j][c1] -= flux[j][id];
                }
            }
            else if (tree->ifaceType[i] == 3 || tree->ifaceType[i] == 5) { //just write back to c2
                for (j = 0; j < nVar; j++) {
                    res[j][c2] += flux[j][id];
                }
            }
            else
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
    else {
#pragma omp task default(shared)
        tree_traversal(tree->left, res, flux, nVar, f2c);
#pragma omp task default(shared)
        tree_traversal(tree->right, res, flux, nVar, f2c);
#pragma omp taskwait
        if (tree->sep != NULL) {
            tree_traversal(tree->sep, res, flux, nVar, f2c);
        }
    }
}

//traverse tree for MaxMinDiff
void tree_traversal(tree_t*& tree, RealFlow* dmax, RealFlow* dmin, RealFlow* q, IntType* f2c, BCRecord** bcr) {
    if (tree->left == NULL && tree->right == NULL) {//Leaf
        IntType i, c1, c2, count, id, type;
        //bFace
        for (i = 0; i < tree->n_bface; i++) {
            id = tree->bfaceID[i];
            count = 2 * id;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            type = bcr[id]->GetType();
            if (type != INTERFACE) continue;

            dmax[c1] = MAX(dmax[c1], q[c2]);
            dmin[c1] = MIN(dmin[c1], q[c2]);
        }
        //iFace
        for (i = 0; i < tree->n_iface; i++) {
            count = 2 * tree->ifaceID[i];
            c1 = f2c[count];
            c2 = f2c[count + 1];
            if (tree->ifaceType[i] == 1) { //write back to c1 and c2
                dmax[c1] = MAX(dmax[c1], q[c2]);
                dmin[c1] = MIN(dmin[c1], q[c2]);

                dmax[c2] = MAX(dmax[c2], q[c1]);
                dmin[c2] = MIN(dmin[c2], q[c1]);
            }
            else if (tree->ifaceType[i] == 2 || tree->ifaceType[i] == 4) { //just write back to c1
                dmax[c1] = MAX(dmax[c1], q[c2]);
                dmin[c1] = MIN(dmin[c1], q[c2]);
            }
            else if (tree->ifaceType[i] == 3 || tree->ifaceType[i] == 5) { //just write back to c2
                dmax[c2] = MAX(dmax[c2], q[c1]);
                dmin[c2] = MIN(dmin[c2], q[c1]);
            }
            else
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
    else {
#pragma omp task default(shared)
        tree_traversal(tree->left, dmax, dmin, q, f2c, bcr);
#pragma omp task default(shared)
        tree_traversal(tree->right, dmax, dmin, q, f2c, bcr);
#pragma omp taskwait
        if (tree->sep != NULL) {
            tree_traversal(tree->sep, dmax, dmin, q, f2c, bcr);
        }
    }
}

//traverse tree for VencatLimiter
void tree_traversal(tree_t*& tree, RealFlow* limit, RealFlow* tmp_limit, IntType* f2c, RealGeom* xfc, RealGeom* yfc, RealGeom* zfc,
        RealGeom* xcc, RealGeom* ycc, RealGeom* zcc, RealFlow* dqdx, RealFlow* dqdy, RealFlow* dqdz, RealFlow* espcell,
        RealFlow* dmax, RealFlow* dmin, IntType nBFace) {
    if (tree->left == NULL && tree->right == NULL) {//Leaf
        IntType i, j, c1, c2, count, id;
        RealGeom dx, dy, dz, eps;
        RealFlow dq_face, tmp1, tmp2;
        //bFace
        for (i = 0; i < tree->n_bface; i++) {
            id = tree->bfaceID[i];
            count = 2 * id;
            c1 = f2c[count];
            dx = xfc[id] - xcc[c1];
            dy = yfc[id] - ycc[c1];
            dz = zfc[id] - zcc[c1];
            dq_face = dqdx[c1] * dx + dqdy[c1] * dy + dqdz[c1] * dz;

            eps = espcell[c1];
            if (EqualZero(dq_face))
                tmp1 = 1.0;
            else {
                if (dq_face > 0.0) {
                    tmp1 = VenFun(dmax[c1], dq_face, eps);
                }
                else {
                    tmp1 = VenFun(dmin[c1], dq_face, eps);
                }
                tmp1 /= dq_face;
            }
            limit[c1] = MIN(limit[c1], tmp1);
        }
        //iFace
        for (i = 0; i < tree->n_iface; i++) {
            id = tree->ifaceID[i];
            count = 2 * id;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            count = 2 * (id - nBFace);
            if (tree->ifaceType[i] < 4) { //compute face data, and storage
                dx = xfc[id] - xcc[c1];
                dy = yfc[id] - ycc[c1];
                dz = zfc[id] - zcc[c1];
                dq_face = dqdx[c1] * dx + dqdy[c1] * dy + dqdz[c1] * dz;
                eps = espcell[c1];
                if (EqualZero(dq_face))
                    tmp1 = 1.0;
                else {
                    if (dq_face > 0.0) {
                        tmp1 = VenFun(dmax[c1], dq_face, eps);
                    }
                    else {
                        tmp1 = VenFun(dmin[c1], dq_face, eps);
                    }
                    tmp1 /= dq_face;
                }

                dx = xfc[id] - xcc[c2];
                dy = yfc[id] - ycc[c2];
                dz = zfc[id] - zcc[c2];
                dq_face = dqdx[c2] * dx + dqdy[c2] * dy + dqdz[c2] * dz;

                eps = espcell[c2];

                if (EqualZero(dq_face))
                    tmp2 = 1.0;
                else {
                    if (dq_face > 0.0) {
                        tmp2 = VenFun(dmax[c2], dq_face, eps);
                    }
                    else {
                        tmp2 = VenFun(dmin[c2], dq_face, eps);
                    }
                    tmp2 /= dq_face;
                }
                tmp_limit[count++] = tmp1;
                tmp_limit[count] = tmp2;
            }
            else {
                tmp1 = tmp_limit[count++];
                tmp2 = tmp_limit[count];
            }
            if (tree->ifaceType[i] == 1) { //write back to c1 and c2
                limit[c1] = MIN(limit[c1], tmp1);
                limit[c2] = MIN(limit[c2], tmp2);
            }
            else if (tree->ifaceType[i] == 2 || tree->ifaceType[i] == 4) { //just write back to c1
                limit[c1] = MIN(limit[c1], tmp1);
            }
            else if (tree->ifaceType[i] == 3 || tree->ifaceType[i] == 5) { //just write back to c2
                limit[c2] = MIN(limit[c2], tmp2);
            }
            else
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
    else {
#pragma omp task default(shared)
        tree_traversal(tree->left, limit, tmp_limit, f2c, xfc, yfc, zfc,
            xcc, ycc, zcc, dqdx, dqdy, dqdz, espcell, dmax, dmin, nBFace);
#pragma omp task default(shared)
        tree_traversal(tree->right, limit, tmp_limit, f2c, xfc, yfc, zfc,
            xcc, ycc, zcc, dqdx, dqdy, dqdz, espcell, dmax, dmin, nBFace);
#pragma omp taskwait
        if (tree->sep != NULL) {
            tree_traversal(tree->sep, limit, tmp_limit, f2c, xfc, yfc, zfc,
                xcc, ycc, zcc, dqdx, dqdy, dqdz, espcell, dmax, dmin, nBFace);
        }
    }
}

//traverse tree for TimeStepNormal_new
void tree_traversal(tree_t*& tree, RealFlow* dt, RealFlow* tmp, IntType* f2c, IntType nBFace, IntType vis_run,
        RealFlow* p, RealGeom* xfn, RealGeom* yfn, RealGeom* zfn, RealGeom* xfc, RealGeom* yfc, RealGeom* zfc,
        RealGeom* xcc, RealGeom* ycc, RealGeom* zcc, RealFlow* rho, RealFlow* u, RealFlow* v, RealFlow* w,
        RealGeom* vgn, RealGeom* area, RealGeom* vol, RealFlow gam, RealFlow p_bar, IntType steady, RealFlow C,
        RealFlow* vis_l, RealFlow* vis_t, RealFlow prl, RealFlow prt) {
    if (tree->left == NULL && tree->right == NULL) {//Leaf
        IntType i, j, c1, c2, count, id;
        RealFlow eigv, dn, vn, c2tmp, gam_tmp, muoopr;
        RealFlow tmp1, tmp2;
        //bFace
        for (i = 0; i < tree->n_bface; i++) {
            id = tree->bfaceID[i];
            c1 = f2c[2 * id];
            c2tmp = gam * (p[c1] + p_bar) / rho[c1];
            dn = fabs((xfc[id] - xcc[c1]) * xfn[id] + (yfc[id] - ycc[c1]) * yfn[id] + (zfc[id] - zcc[c1]) * zfn[id]);

            vn = u[c1] * xfn[id] + v[c1] * yfn[id] + w[c1] * zfn[id];
            if (!steady) vn -= vgn[id];
            vn = fabs(vn);
            eigv = vn + sqrt(c2tmp);

            if (vis_run) {
                muoopr = vis_l[c1] / prl + vis_t[c1] / prt;
                gam_tmp = gam;

                //eigv += C*gam_tmp/rho[c1]*muoopr/(dn+TINY);
                eigv += C * gam_tmp / rho[c1] * muoopr * area[id] / vol[c1];
            }
            dt[c1] = MIN(dt[c1], dn / eigv);
        }
        //iFace
        for (i = 0; i < tree->n_iface; i++) {
            id = tree->ifaceID[i];
            count = 2 * id;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            count = 2 * (id - nBFace);
            if (tree->ifaceType[i] < 4) { //compute face data, and storage
                c2tmp = gam * (p[c1] + p_bar) / rho[c1];
                dn = fabs((xfc[id] - xcc[c1]) * xfn[id] + (yfc[id] - ycc[c1]) * yfn[id] + (zfc[id] - zcc[c1]) * zfn[id]);

                vn = u[c1] * xfn[id] + v[c1] * yfn[id] + w[c1] * zfn[id];
                if (!steady) vn -= vgn[id];
                vn = fabs(vn);
                eigv = vn + sqrt(c2tmp);

                if (vis_run) {
                    muoopr = vis_l[c1] / prl + vis_t[c1] / prt;

                    gam_tmp = gam;

                    //eigv += C*gam_tmp/rho[c1]*muoopr/dn;
                    eigv += C * gam_tmp / rho[c1] * muoopr * area[id] / vol[c1];
                }
                tmp1 = dn / eigv;

                c2tmp = gam * (p[c2] + p_bar) / rho[c2];
                dn = fabs((xfc[id] - xcc[c2]) * xfn[id] + (yfc[id] - ycc[c2]) * yfn[id] + (zfc[id] - zcc[c2]) * zfn[id]);

                vn = u[c2] * xfn[id] + v[c2] * yfn[id] + w[c2] * zfn[id];
                if (!steady) vn -= vgn[id];
                vn = fabs(vn);
                eigv = vn + sqrt(c2tmp);

                if (vis_run) {
                    muoopr = vis_l[c2] / prl + vis_t[c2] / prt;
                    gam_tmp = gam;

                    //eigv += C*gam_tmp/rho[c2]*muoopr/dn;
                    eigv += C * gam_tmp / rho[c2] * muoopr * area[id] / vol[c2];
                }
                tmp2 = dn / eigv;
                tmp[count++] = tmp1;
                tmp[count] = tmp2;
            }
            else {
                tmp1 = tmp[count++];
                tmp2 = tmp[count];
            }
            if (tree->ifaceType[i] == 1) { //write back to c1 and c2
                dt[c1] = MIN(dt[c1], tmp1);
                dt[c2] = MIN(dt[c2], tmp2);
            }
            else if (tree->ifaceType[i] == 2 || tree->ifaceType[i] == 4) { //just write back to c1
                dt[c1] = MIN(dt[c1], tmp1);
            }
            else if (tree->ifaceType[i] == 3 || tree->ifaceType[i] == 5) { //just write back to c2
                dt[c2] = MIN(dt[c2], tmp2);
            }
            else
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
    else {
#pragma omp task default(shared)
        tree_traversal(tree->left, dt, tmp, f2c, nBFace, vis_run, p, xfn, yfn, zfn, xfc, yfc, zfc,
            xcc, ycc, zcc, rho, u, v, w, vgn, area, vol, gam, p_bar, steady, C, vis_l, vis_t, prl, prt);
#pragma omp task default(shared)
        tree_traversal(tree->right, dt, tmp, f2c, nBFace, vis_run, p, xfn, yfn, zfn, xfc, yfc, zfc,
            xcc, ycc, zcc, rho, u, v, w, vgn, area, vol, gam, p_bar, steady, C, vis_l, vis_t, prl, prt);
#pragma omp taskwait
        if (tree->sep != NULL) {
            tree_traversal(tree->sep, dt, tmp, f2c, nBFace, vis_run, p, xfn, yfn, zfn, xfc, yfc, zfc,
                xcc, ycc, zcc, rho, u, v, w, vgn, area, vol, gam, p_bar, steady, C, vis_l, vis_t, prl, prt);
        }
    }
}

//traverse tree for PutScalarDqToLhs
void tree_traversal(tree_t*& tree, RealFlow** lhsmat, RealFlow* dqdl, RealFlow* dqdr, IntType* f2c, IntType* fcptr) {
    if (tree->left == NULL && tree->right == NULL) {//Leaf
        IntType i, j, c1, c2, count, id;
        IntType nc1, nc2;
        //bFace
        for (i = 0; i < tree->n_bface; i++) {
            id = tree->bfaceID[i];
            count = 2 * id;
            c1 = f2c[count];
            nc1 = fcptr[count];
            lhsmat[c1][0] += dqdl[id];
            if (nc1 > 0) lhsmat[c1][nc1] += dqdr[id];
        }
        //iFace
        for (i = 0; i < tree->n_iface; i++) {
            id = tree->ifaceID[i];
            count = 2 * id;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            nc1 = fcptr[count];
            nc2 = fcptr[count + 1];
            if (tree->ifaceType[i] == 1) { //write back to c1 and c2
                lhsmat[c1][0] += dqdl[id];
                if (nc1 > 0) lhsmat[c1][nc1] += dqdr[id];
                lhsmat[c2][0] -= dqdr[id];
                if (nc2 > 0) lhsmat[c2][nc2] -= dqdl[id];
            }
            else if (tree->ifaceType[i] == 2 || tree->ifaceType[i] == 4) { //just write back to c1
                lhsmat[c1][0] += dqdl[id];
                if (nc1 > 0) lhsmat[c1][nc1] += dqdr[id];
            }
            else if (tree->ifaceType[i] == 3 || tree->ifaceType[i] == 5) { //just write back to c2
                lhsmat[c2][0] -= dqdr[id];
                if (nc2 > 0) lhsmat[c2][nc2] -= dqdl[id];
            }
            else
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
    else {
#pragma omp task default(shared)
        tree_traversal(tree->left, lhsmat, dqdl, dqdr, f2c, fcptr);
#pragma omp task default(shared)
        tree_traversal(tree->right, lhsmat, dqdl, dqdr, f2c, fcptr);
#pragma omp taskwait
        if (tree->sep != NULL) {
            tree_traversal(tree->sep, lhsmat, dqdl, dqdr, f2c, fcptr);
        }
    }
}

//traverse tree for ViscousFluxScalar3D_New3
void tree_traversal(tree_t*& tree, RealFlow* res, RealFlow* flux, RealFlow* tem, RealFlow* tem_c2, IntType* f2c) {
    if (tree->left == NULL && tree->right == NULL) {//Leaf
        IntType i, c1, c2, count, id;
        RealFlow factor_c1, factor_c2;
        //bFace
        for (i = 0; i < tree->n_bface; i++) {
            id = tree->bfaceID[i];
            c1 = f2c[2 * id];
            factor_c1 = flux[id] + tem[id];
            res[c1] += factor_c1;
        }
        //iFace
        for (i = 0; i < tree->n_iface; i++) {
            id = tree->ifaceID[i];
            count = 2 * id;
            c1 = f2c[count];
            c2 = f2c[count + 1];
            if (tree->ifaceType[i] == 1) { //write back to c1 and c2
                factor_c1 = flux[id] + tem[id];
                res[c1] += factor_c1;
                factor_c2 = flux[id] + tem_c2[id];
                res[c2] -= factor_c2;
            }
            else if (tree->ifaceType[i] == 2 || tree->ifaceType[i] == 4) { //just write back to c1
                factor_c1 = flux[id] + tem[id];
                res[c1] += factor_c1;
            }
            else if (tree->ifaceType[i] == 3 || tree->ifaceType[i] == 5) { //just write back to c2
                factor_c2 = flux[id] + tem_c2[id];
                res[c2] -= factor_c2;
            }
            else
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }
    }
    else {
#pragma omp task default(shared)
        tree_traversal(tree->left, res, flux, tem, tem_c2, f2c);
#pragma omp task default(shared)
        tree_traversal(tree->right, res, flux, tem, tem_c2, f2c);
#pragma omp taskwait
        if (tree->sep != NULL) {
            tree_traversal(tree->sep, res, flux, tem, tem_c2, f2c);
        }
    }
}
#endif // DIVCON
#undef CPP_FILD_ID  // clear out file id
} //~namespace mflow