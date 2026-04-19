//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   grid_reorder.cpp
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
#include <cfloat>
#include <algorithm>

//scotch
#include <stdint.h>
#ifdef REORDER
#include <scotch.h>
#endif
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
#include <omp.h>
#endif

namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 10713  // define file id

#ifdef MPICH
extern int myZone;
extern int numprocs;
extern MPI_Comm GridComm;  //for each grid, tangj
#endif

//added by xchf
/************************************************************************
Purpose:  To interchange v[i] and v[j] for IntType
************************************************************************/
void xswap_int(IntType *indx, IntType *v, IntType i, IntType j)
{
	IntType temp;
	temp = v[i];
	v[i] = v[j];
	v[j] = temp;
	IntType itmp;
	itmp = indx[i];
	indx[i] = indx[j];
	indx[j] = itmp;
}
/************************************************************************
Purpose:  To interchange v[i] and v[j] stored in vector<IntType>
************************************************************************/
void xswap_vecint(std::vector<IntType> &order, std::vector<IntType> &vecint, IntType i, IntType j)
{
	std::swap(vecint[i], vecint[j]);
	std::swap(order[i], order[j]);
}
/************************************************************************
Purpose:  To sort a list of points for IntType
************************************************************************/
void quicksort_int(IntType s, IntType e, IntType *indx, IntType *v)
{
	IntType i, last;

	if (e - s <= 0) // nothing to do
		return;

	last = s - 1;
	for (i = s; i<e; i++)
	{
		if (v[i]<v[s - 1])
			xswap_int(indx, v, ++last, i);
	}

	xswap_int(indx, v, s - 1, last);
	quicksort_int(s, last, indx, v);
	quicksort_int(last + 2, e, indx, v);
}
/************************************************************************
Purpose:  To sort a list of points for IntType stored in vector<IntType>
************************************************************************/
void quicksort_vecint(IntType s, IntType e, std::vector<IntType> &order, std::vector<IntType> &vecint)
{
	IntType i, last;

	if (e - s <= 0) // nothing to do
		return;

	last = s - 1;
	for (i = s; i<e; i++)
	{
		if (vecint[i] < vecint[s - 1]) //ascending
			xswap_vecint(order, vecint, ++last, i);
	}

	xswap_vecint(order, vecint, s - 1, last);
	quicksort_vecint(s, last, order, vecint);
	quicksort_vecint(last + 2, e, order, vecint);
}

/************************************************************************
				Update interface conectivities after reordering.
						Add by dingxin 2021-9-25
************************************************************************/
void UpdateIface(PolyGrid* grid, IntType* index_iface) {
	IntType nBFace, nTFace;
	IntType* f2c, * f2n, * nNPF;
	IntType tmp, nodecount_bface = 0;

	IntType* f2c_backup = NULL;
	IntType* nNPF_iface = NULL;
	IntType* f2n_iface = NULL;
	IntType** F2N_iface = NULL;

	nBFace = grid->GetNBFace();
	nTFace = grid->GetNTFace();
	f2c = grid->Getf2c();
	f2n = grid->Getf2n();
	nNPF = grid->GetnNPF();

	IntType ifacenum = nTFace - nBFace;

	mfmem::snew_array_1D(nNPF_iface, ifacenum, dmrfl);
	mfmem::snew_array_1D(F2N_iface, ifacenum, dmrfl);
	mfmem::snew_array_1D(f2c_backup, 2 * ifacenum, dmrfl);
	// update f2c for interior faces
	for (IntType i = nBFace; i < nTFace; i++)
	{
		f2c_backup[2 * (i - nBFace)] = f2c[2 * i];
		f2c_backup[2 * (i - nBFace) + 1] = f2c[2 * i + 1];
	}
	for (IntType i = nBFace; i < nTFace; i++)
	{
		tmp = (index_iface[i - nBFace] - nBFace) * 2;
		f2c[2 * i] = f2c_backup[tmp];
		f2c[2 * i + 1] = f2c_backup[tmp + 1];
	}
	//update nNPF, f2n for interior faces
	tmp = 0;
	for (IntType i = nBFace; i < nTFace; i++)
	{
		tmp += nNPF[i];
	}
	mfmem::snew_array_1D(f2n_iface, tmp, dmrfl);

	tmp = 0;
	for (IntType i = 0; i < nBFace; ++i)//不着色并行分区边界面
		nodecount_bface += nNPF[i];
	for (IntType i = nBFace; i < nTFace; i++)
	{
		nNPF_iface[i - nBFace] = nNPF[i];
		for (IntType j = 0; j < nNPF[i]; j++)
		{
			f2n_iface[tmp] = f2n[nodecount_bface + tmp];
			tmp++;
		}
	}

	F2N_iface[0] = f2n_iface;
	for (IntType i = 1; i < ifacenum; i++)//?
	{
		F2N_iface[i] = &(F2N_iface[i - 1][nNPF_iface[i - 1]]);
	}

	//update nNPF for interior faces
	for (IntType i = nBFace; i < nTFace; i++)
	{
		nNPF[i] = nNPF_iface[index_iface[i - nBFace] - nBFace];
	}

	//update f2n for interior faces
	tmp = 0;
	for (IntType i = nBFace; i < nTFace; i++)
	{
		for (IntType j = 0; j < nNPF[i]; j++)
		{
			f2n[nodecount_bface + tmp] = F2N_iface[index_iface[i - nBFace] - nBFace][j];
			++tmp;
		}
	}
	mfmem::sdel_array_1D(nNPF_iface);
	mfmem::sdel_array_1D(f2n_iface);
	mfmem::sdel_array_1D(F2N_iface);
	mfmem::sdel_array_1D(f2c_backup);
}

#ifdef REORDER
/* Sort an array "a" between its left bound "l" and its right bound "r" */
static void _scotch_sort_shell(SCOTCH_Num l, SCOTCH_Num r, SCOTCH_Num* cell_neighbors)
{
	IntType i, j, h;
	SCOTCH_Num v;
	/* Compute stride */
	for (h = 1; h <= (r - l) / 9; h = 3 * h + 1);

	/* Sort array */
	for (; h > 0; h /= 3) {
		for (i = l + h; i < r; i++) {
			v = cell_neighbors[i];
			j = i;
			while ((j >= l + h) && (v < cell_neighbors[j - h])) {
				cell_neighbors[j] = cell_neighbors[j - h];
				j -= h;
			}
			cell_neighbors[j] = v;
		} /* Loop on array elements */
	} /* End of loop on stride */

}

inline static bool _a_gt_b(fvm_morton_code_t  code_a, fvm_morton_code_t  code_b) {
	int i, a, b, a_diff, b_diff;
	int l = MAX(code_a.L, code_b.L);

	a_diff = l - code_a.L;
	b_diff = l - code_b.L;

	if (a_diff > 0) {
		code_a.L = l;
		code_a.X[0] = code_a.X[0] << a_diff;
		code_a.X[1] = code_a.X[1] << a_diff;
		code_a.X[2] = code_a.X[2] << a_diff;
	}

	if (b_diff > 0) {
		code_b.L = l;
		code_b.X[0] = code_b.X[0] << b_diff;
		code_b.X[1] = code_b.X[1] << b_diff;
		code_b.X[2] = code_b.X[2] << b_diff;
	}

	i = l - 1;
	while (i > 0) {
		if (code_a.X[0] >> i != code_b.X[0] >> i
			|| code_a.X[1] >> i != code_b.X[1] >> i
			|| code_a.X[2] >> i != code_b.X[2] >> i)
			break;
		i--;
	}

	a = ((code_a.X[0] >> i) % 2) * 4
		+ ((code_a.X[1] >> i) % 2) * 2
		+ ((code_a.X[2] >> i) % 2);
	b = ((code_b.X[0] >> i) % 2) * 4
		+ ((code_b.X[1] >> i) % 2) * 2
		+ ((code_b.X[2] >> i) % 2);

	return (a > b) ? true : false;
}

inline static bool _a_ge_b(fvm_morton_code_t  code_a, fvm_morton_code_t  code_b) {
	int i, a, b, a_diff, b_diff;
	int l = MAX(code_a.L, code_b.L);

	a_diff = l - code_a.L;
	b_diff = l - code_b.L;

	if (a_diff > 0) {
		code_a.L = l;
		code_a.X[0] = code_a.X[0] << a_diff;
		code_a.X[1] = code_a.X[1] << a_diff;
		code_a.X[2] = code_a.X[2] << a_diff;
	}

	if (b_diff > 0) {
		code_b.L = l;
		code_b.X[0] = code_b.X[0] << b_diff;
		code_b.X[1] = code_b.X[1] << b_diff;
		code_b.X[2] = code_b.X[2] << b_diff;
	}

	i = l - 1;
	while (i > 0) {
		if (code_a.X[0] >> i != code_b.X[0] >> i
			|| code_a.X[1] >> i != code_b.X[1] >> i
			|| code_a.X[2] >> i != code_b.X[2] >> i)
			break;
		i--;
	}

	a = ((code_a.X[0] >> i) % 2) * 4
		+ ((code_a.X[1] >> i) % 2) * 2
		+ ((code_a.X[2] >> i) % 2);
	b = ((code_b.X[0] >> i) % 2) * 4
		+ ((code_b.X[1] >> i) % 2) * 2
		+ ((code_b.X[2] >> i) % 2);

	return (a >= b) ? true : false;
}
/* Build a heap structure or order a heap structure with a working array to save the ordering */
static void _descend_morton_heap_with_order(fvm_morton_int_t parent, IntType n_codes, const fvm_morton_code_t *morton_codes, IntType* order) {
	IntType tmp;
	IntType child = 2 * parent + 1;
	while (child < n_codes) {

		if (child + 1 < n_codes) {
			if (_a_gt_b(morton_codes[order[child + 1]],
				morton_codes[order[child]]))
				child++;
		}

		if (_a_ge_b(morton_codes[order[parent]],
			morton_codes[order[child]]))
			return;

		tmp = order[parent];
		order[parent] = order[child];
		order[child] = tmp;
		parent = child;
		child = 2 * parent + 1;

	} /* End while */
}


void _compute_cell_center(PolyGrid *grid, RealGeom* cell_center) {
	IntType nBFace = grid->GetNBFace();
	IntType nTFace = grid->GetNTFace();
	IntType nTCell = grid->GetNTCell();
	IntType* f2c = grid->Getf2c();
	IntType* f2n = grid->Getf2n();
	IntType* nNPF = grid->GetnNPF();
	RealGeom* x = grid->GetX();
	RealGeom* y = grid->GetY();
	RealGeom* z = grid->GetZ();
	RealGeom* xfc = NULL;
	RealGeom* yfc = NULL;
	RealGeom* zfc = NULL;
	IntType* nface = NULL;
	IntType i, j, p1, node, cell, c1, c2;

	//face center
	mfmem::snew_array_1D(xfc, nTFace, dmrfl);
	mfmem::snew_array_1D(yfc, nTFace, dmrfl);
	mfmem::snew_array_1D(zfc, nTFace, dmrfl);
	node = 0;
	for (i = 0; i < nTFace; i++) {
		xfc[i] = 0.0;
		yfc[i] = 0.0;
		zfc[i] = 0.0;

		for (j = 0; j < nNPF[i]; j++) {
			p1 = f2n[node++];
			xfc[i] += x[p1];
			yfc[i] += y[p1];
			zfc[i] += z[p1];
		}
		xfc[i] /= nNPF[i];
		yfc[i] /= nNPF[i];
		zfc[i] /= nNPF[i];
	}

	//cell center, the average of face center
	for (i = 0; i < 3*nTCell; i++) {
		cell_center[i] = 0.0;
	}
	
	mfmem::snew_array_1D(nface, nTCell, dmrfl);
	for (i = 0; i < nTCell; i++) nface[i] = 0;
	cell = 0;
	for (i = 0; i < nBFace; i++) {
		c1 = f2c[cell++];
		cell++;
		cell_center[3 * c1] += xfc[i];
		cell_center[3 * c1+1] += yfc[i];
		cell_center[3 * c1+2] += zfc[i];
		nface[c1]++;
	}
	for (i = nBFace; i < nTFace; i++) {
		c1 = f2c[cell++];
		c2 = f2c[cell++];
		cell_center[3 * c1] += xfc[i];
		cell_center[3 * c1 + 1] += yfc[i];
		cell_center[3 * c1 + 2] += zfc[i];
		cell_center[3 * c2] += xfc[i];
		cell_center[3 * c2 + 1] += yfc[i];
		cell_center[3 * c2 + 2] += zfc[i];
		nface[c1]++;
		nface[c2]++;
	}
	for (i = 0; i < nTCell; i++) {
		cell_center[3 * i] /= nface[i];
		cell_center[3 * i + 1] /= nface[i];
		cell_center[3 * i + 2] /= nface[i];
	}
	mfmem::sdel_array_1D(nface);
	mfmem::sdel_array_1D(xfc);
	mfmem::sdel_array_1D(yfc);
	mfmem::sdel_array_1D(zfc);
}
/* Determine the local extents associated with a set of coordinates */
void _compute_coord_extents(IntType dim, IntType n_cells, RealGeom* cell_center, RealGeom* extents) {
	fvm_morton_int_t  i, j;

	/* Get local min/max coordinates */
	for (j = 0; j < (size_t)dim; j++) {
		extents[j] = DBL_MAX;
		extents[j + dim] = -DBL_MAX;
	}
	for (i = 0; i < n_cells; i++) {
		for (j = 0; j < (size_t)dim; j++) {
			if (cell_center[i * dim + j] < extents[j])
				extents[j] = cell_center[i * dim + j];
			if (cell_center[i * dim + j] > extents[j + dim])
				extents[j + dim] = cell_center[i * dim + j];
		}
	}
}
/* Encode an array of coordinates */
void fvm_morton_encode_coords(IntType dim, fvm_morton_int_t level, const RealGeom *extents, IntType  n_coords,
	const RealGeom  *coords, fvm_morton_code_t  *m_code) {
	fvm_morton_int_t i, j;
	RealGeom s[3], d[3], n[3];
	RealGeom d_max = 0.0;
	fvm_morton_int_t  refinement = 1u << level;

	for (i = 0; i < (fvm_morton_int_t)dim; i++) {
		s[i] = extents[i];
		d[i] = extents[i + dim] - extents[i];
		d_max = MAX(d_max, d[i]);
	}

	for (i = 0; i < (fvm_morton_int_t)dim; i++) { /* Reduce effective dimension */
		if (d[i] < d_max * 1e-10)
			d[i] = d_max * 1e-10;
	}
	for (i = 0; i < n_coords; i++) {
		m_code[i].L = level;
		for (j = 0; j < 3; j++) {
			n[j] = (coords[i * dim + j] - s[j]) / d[j];
			m_code[i].X[j] = MIN((fvm_morton_int_t)floor(n[j] * refinement), refinement - 1);
		}
	}
}

/* Locally order a list of Morton ids */
void fvm_morton_local_order(IntType n_codes, fvm_morton_code_t* morton_codes, IntType* order) {
	IntType i, tmp;

	assert(n_codes == 0 || morton_codes != NULL);
	for (i = 0; i < n_codes; i++)
		order[i] = i;

	/* Build heap */

	for (i = n_codes / 2 - 1; (int)i >= 0; i--)
		_descend_morton_heap_with_order(i, n_codes, morton_codes, order);

	/* Sort array */
	for (i = n_codes - 1; (int)i >= 0; i--) {
		tmp = order[0];
		order[0] = order[i];
		order[i] = tmp;

		_descend_morton_heap_with_order(0, i, morton_codes, order);
	}
	for (i = 1; i < n_codes; i++) {
		if (_a_gt_b(morton_codes[order[i - 1]], morton_codes[order[i]])) {
			mflog::log.set_one_processor_out();
			mflog::log << "Id: %u inconsistent: bad ordering of Morton codes." << std::endl;
		}
	}
}

/************************************************************************
	Build Metis graph (cell -> cell connectivity, ignoring ghosts)
						Add by dingxin 2021-9-25
************************************************************************/
void _metis_graph(const PolyGrid* grid, idx_t n_cells, idx_t *&cell_idx, idx_t *&cell_neighbors) {
	IntType i,tmp,c1,c2;
	IntType* n_neighbors;
	IntType n_b_face = grid->GetNBFace();
	IntType nTFace = grid->GetNTFace();
	IntType* f2c = grid->Getf2c();

	mfmem::snew_array_1D(n_neighbors, n_cells, dmrfl);
	mfmem::snew_array_1D(cell_idx, n_cells+1, dmrfl);
	for (i = 0; i < n_cells; i++)
		n_neighbors[i] = 0;
	tmp = 2 * n_b_face;
	for (i = n_b_face; i < nTFace; i++) {
		c1 = f2c[tmp++];
		c2 = f2c[tmp++];
		assert(c1 < n_cells&& c2 < n_cells);
		n_neighbors[c1]++;
		n_neighbors[c2]++;
	}
	cell_idx[0] = 0;
	for (i = 0; i < n_cells; i++)
		cell_idx[i + 1] = cell_idx[i] + n_neighbors[i];
	mfmem::snew_array_1D(cell_neighbors, cell_idx[n_cells], dmrfl);
	for (i = 0; i < n_cells; i++)
		n_neighbors[i] = 0;
	tmp = 2 * n_b_face;
	for (i = n_b_face; i < nTFace; i++) {
		c1 = f2c[tmp++];
		c2 = f2c[tmp++];
		cell_neighbors[cell_idx[c1] + n_neighbors[c1]] = c2;
		n_neighbors[c1]++;
		cell_neighbors[cell_idx[c2] + n_neighbors[c2]] = c1;
		n_neighbors[c2]++;
	}

	mfmem::sdel_array_1D(n_neighbors);
}

/************************************************************************
	Build SCOTCH graph (cell -> cell connectivity, ignoring ghosts)
						Add by dingxin 2021-9-25
************************************************************************/
static SCOTCH_Num _scotch_graph(const PolyGrid* grid, SCOTCH_Num n_cells, SCOTCH_Num*& cell_idx, SCOTCH_Num*& cell_neighbors) {
	SCOTCH_Num  i;
	SCOTCH_Num  start_id, end_id;
	SCOTCH_Num* n_neighbors = NULL;
	IntType* _mesh_to_graph = NULL;
	IntType* f2c = grid->Getf2c();
	IntType c1, c2, tmp;
	IntType n_b_face = grid->GetNBFace();
	IntType nTFace = grid->GetNTFace();
	const SCOTCH_Num n_faces = nTFace - n_b_face;
	SCOTCH_Num  n_graph_cells = 0;

	/* Count and allocate arrays */
	mfmem::snew_array_1D(n_neighbors, n_cells, dmrfl);
	mfmem::snew_array_1D(cell_idx, n_cells + 1, dmrfl);
	mfmem::snew_array_1D(_mesh_to_graph, n_cells , dmrfl);

	for (i = 0; i < n_cells; i++)
		n_neighbors[i] = 0;
	tmp = 2 * n_b_face;
	for (i = n_b_face; i < nTFace; i++) {
		c1 = f2c[tmp++];
		c2 = f2c[tmp++];
		assert(c1 < n_cells&& c2 < n_cells);
		n_neighbors[c1] += 1;
		n_neighbors[c2] += 1;
	}
	cell_idx[0] = 0;
	for (i = 0; i < n_cells; i++) {
		if (n_neighbors[i] > 0) {
			cell_idx[n_graph_cells + 1] = cell_idx[n_graph_cells] + n_neighbors[i];
			_mesh_to_graph[i] = n_graph_cells;
			n_graph_cells++;
		}
		else
			_mesh_to_graph[i] = -1;
	}
	mfmem::snew_array_1D(cell_neighbors, cell_idx[n_graph_cells], dmrfl);
	for (i = 0; i < n_graph_cells; i++)
		n_neighbors[i] = 0;
	tmp = 2 * n_b_face;
	for (i = n_b_face; i < nTFace; i++) {
		c1 = _mesh_to_graph[f2c[tmp++]];
		c2 = _mesh_to_graph[f2c[tmp++]];
		if (c1 > -1 && c2 > -1) {
			cell_neighbors[cell_idx[c1] + n_neighbors[c1]] = c2;
			n_neighbors[c1] += 1;
			cell_neighbors[cell_idx[c2] + n_neighbors[c2]] = c1;
			n_neighbors[c2] += 1;
		}
	}
	/* Clean graph */
	tmp = 0;
	start_id = cell_idx[0];
	end_id = 0;
	for (i = 0; i < n_graph_cells; i++) {
		SCOTCH_Num n_prev;
		end_id = cell_idx[i + 1];
		if (end_id > start_id) {
			_scotch_sort_shell(start_id, end_id, cell_neighbors);
			n_prev = cell_neighbors[start_id];
			cell_neighbors[tmp] = n_prev;
			tmp += 1;
			for (SCOTCH_Num j = start_id + 1; j < end_id; j++) {
				if (cell_neighbors[j] != n_prev) {
					n_prev = cell_neighbors[j];
					cell_neighbors[tmp] = n_prev;
					tmp += 1;
				}
			}
		}
		start_id = end_id;
		cell_idx[i + 1] = tmp;
	}
	if (tmp < end_id)
		mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
	mfmem::sdel_array_1D(n_neighbors);
	mfmem::sdel_array_1D(_mesh_to_graph);
	return n_graph_cells;
}

/* Update face to cell */
void PolyGrid::Update_f2c() {
	IntType* f2c = this->Getf2c();
	IntType* newOrder = this->order_cell_oTon;
	IntType* f2c_backup = NULL;
	mfmem::snew_array_1D(f2c_backup, 2 * this->nTFace, dmrfl);
	for (IntType i = 0; i < 2 * this->nTFace; i++)
	{
		f2c_backup[i] = f2c[i];
	}
	for (IntType i = 0; i < this->nBFace; i++)
	{
		f2c[2 * i] = newOrder[f2c_backup[2 * i]];
	}
	for (IntType i = this->nBFace; i < this->nTFace; i++)
	{
		f2c[2 * i] = newOrder[f2c_backup[2 * i]];
		f2c[2 * i + 1] = newOrder[f2c_backup[2 * i + 1]];
	}
	mfmem::sdel_array_1D(f2c_backup);
}

/************************************************************************
Purpose:  Renumber cells in a grid using the Cuthill-McKee algorithm
************************************************************************/
void PolyGrid::CellReordering_CMK()
{
    IntType count;
    IntType* nCPC_tmp = NULL;//must set to be NULL
    IntType** c2c_tmp = NULL;
    IntType* f2c;
    IntType c1, c2;
    std::set<IntType> visitedCells;
    IntType* newOrder = NULL;

    //bool* is_cell_interface = NULL;

    //mfmem::snew_array_1D(is_cell_interface, nTCell, dmrfl);
    mfmem::snew_array_1D(nCPC_tmp, nTCell, dmrfl);
    mfmem::snew_array_1D(newOrder, nTCell, dmrfl);//this is ultimately what we got!!!
    //1. preprocessing. note tha both nCPC_tmp and c2c_tmp do not count ghost cells
    for (IntType i = 0; i < nTCell; i++) nCPC_tmp[i] = 0;

    f2c = this->Getf2c();
    //for (IntType i = 0; i < nTCell; i++) is_cell_interface[i] = false;
    /*标记所有并行边界面的c1，并固定其编号*/
    /*for (IntType i = nBFace - nIFace; i < nBFace; i++) {
        c1 = f2c[i * 2];
        if (!is_cell_interface[c1]) {
            is_cell_interface[c1] = true;
            newOrder[c1] = c1;
        }
    }*/

    count = 2 * nBFace;
    for (IntType i = nBFace; i < nTFace; i++) {
        c1 = f2c[count++];
        c2 = f2c[count++];
        nCPC_tmp[c1]++;
        nCPC_tmp[c2]++;
    }
    mfmem::snew_array_2D(c2c_tmp, nTCell, nCPC_tmp, dmrfl, true);
    // Need to reset nCPC to 0 and recover it later
    for (IntType i = 0; i < nTCell; i++) nCPC_tmp[i] = 0;

    count = 2 * nBFace;
    for (IntType i = nBFace; i < nTFace; i++) {
        c1 = f2c[count++];
        c2 = f2c[count++];
        c2c_tmp[c1][nCPC_tmp[c1]++] = c2;
        c2c_tmp[c2][nCPC_tmp[c2]++] = c1;
    }

    //2. reordering cells
    std::vector<IntType> nextCell;
    std::vector<IntType> nbrs;
    std::vector<IntType> weights;
    IntType labelMax = 0;
    IntType minWeight;
    IntType cellInOrder = 0;

    for (IntType i = 0; i < nTCell; i++) 
        labelMax = MAX(labelMax, nCPC_tmp[i]);

    /*从非并行边界cell开始计数*/
    /*while (cellInOrder < nTCell && is_cell_interface[cellInOrder]) {
        cellInOrder++;
    }*/

    while (true)
    {
        IntType currentCell = -1;
        minWeight = labelMax;

        //2.1 find the lowest connected cell that has not been visited yet
        //for a disconnected region: unnecessary???
        for (IntType i = 0; i < nTCell; i++)
        {
            if (visitedCells.count(i) == 0)
            {
                if (nCPC_tmp[i] < minWeight)
                {
                    minWeight = nCPC_tmp[i];
                    currentCell = i;
                }
            }
        }
        if (currentCell == -1)
        {
            break;
        }

        //2.2 use this cell as a start and loop through the nextcell vector
        //consider all cells and their neighbours
        nextCell.push_back(currentCell);
        while (nextCell.size())
        {
            currentCell = nextCell.front();
            std::vector<IntType>::iterator k = nextCell.begin();
            nextCell.erase(k); //removehead
            if (visitedCells.count(currentCell) == 0)
            {
                visitedCells.insert(currentCell);
                /*if (!is_cell_interface[currentCell]) {
                    newOrder[currentCell] = cellInOrder;
                    //newOrder[cellInOrder] = currentCell;
                    cellInOrder++;
                    while (cellInOrder < nTCell && is_cell_interface[cellInOrder])//此处增加判断，判断cellInOrder的值是否是并行边界面的c1，是就再自加
                        cellInOrder++;
                }*/
                newOrder[currentCell] = cellInOrder;
                cellInOrder++;
                nbrs.clear();
                weights.clear();
                for (IntType j = 0; j < nCPC_tmp[currentCell]; j++)
                {
                    IntType nbr = c2c_tmp[currentCell][j];
                    if (visitedCells.count(nbr) == 0)
                    {
                        nbrs.push_back(nbr);
                        weights.push_back(nCPC_tmp[nbr]);
                    }
                }
                //sort in ascending order
                IntType* order, * weights_int;
                order = NULL;
                weights_int = NULL;
                mfmem::snew_array_1D(order, weights.size(), dmrfl);
                mfmem::snew_array_1D(weights_int, weights.size(), dmrfl);
                //std::vector<IntType> order(weights.size());
                for (IntType j = 0; j < weights.size(); j++)
                {
                    order[j] = j;
                    weights_int[j] = weights[j];
                }
                //quicksort_vecint(1, weights.size(), order, weights);
                quicksort_int(1, weights.size(), order, weights_int);
                //add in sorted order
                for (IntType j = 0; j < weights.size(); j++)
                {
                    nextCell.push_back(nbrs[order[j]]);
                }
                mfmem::sdel_array_1D(order);
                mfmem::sdel_array_1D(weights_int);
            }
        }
        /*if (cellInOrder != nTCell) {
            string errorinfo = "after once search,cellInOrder != nTCell !!";
            debugInfo(errorinfo);
        }*/
    }
    
    //3.update index
    this->order_cell_oTon = newOrder;
    this->order_cell_nToo = NULL;
    mfmem::snew_array_1D(this->order_cell_nToo, nTCell, dmrfl);
    for (IntType i = 0; i < nTCell; i++) {
        this->order_cell_nToo[newOrder[i]] = i;
    }

    //deallocation
    newOrder = NULL;
    f2c = NULL;
    mfmem::sdel_array_1D(nCPC_tmp);
    mfmem::sdel_array_2D(c2c_tmp);
}

/************************************************************************
				Renumber cells based on local Morton encoding.
						Add by dingxin 2021-9-25
************************************************************************/
void PolyGrid::CellReordering_morton() {
	RealGeom extents[6];
	fvm_morton_code_t* m_code = NULL;
	IntType n_cells = this->nTCell;
	IntType dim = 3;
	const int level = sizeof(fvm_morton_int_t) * 8 - 1;
	RealGeom *cell_center = NULL;
	IntType* new_to_old = NULL;

	/* Build Morton encoding and order it */
	mfmem::snew_array_1D(cell_center, n_cells*3, dmrfl);
	mfmem::snew_array_1D(m_code, n_cells, dmrfl);
	mfmem::snew_array_1D(new_to_old, n_cells, dmrfl);

	_compute_cell_center(this, cell_center);
	_compute_coord_extents(dim, n_cells, cell_center, extents);
	fvm_morton_encode_coords(dim, level, extents, n_cells,cell_center,m_code);
	fvm_morton_local_order(n_cells, m_code, new_to_old);

	/* Update order_cell_oTon */
	this->order_cell_nToo = new_to_old;
	this->order_cell_oTon = NULL;
	mfmem::snew_array_1D(this->order_cell_oTon, n_cells, dmrfl);
	for (IntType i = 0; i < n_cells; i++) {
		this->order_cell_oTon[this->order_cell_nToo[i]] = i;
	}
	mfmem::sdel_array_1D(cell_center);
	mfmem::sdel_array_1D(m_code);
}

/************************************************************************
					Compute local ordering using METIS
						Add by dingxin 2021-9-25
************************************************************************/
void PolyGrid::CellReordering_metis() {
	idx_t n_cells = (idx_t)this->nTCell;
	idx_t* perm = NULL, * iperm = NULL;
	idx_t* cell_idx = NULL, * cell_neighbors = NULL;
	int retcode = METIS_OK;
	IntType* new_to_old = NULL;
	IntType i;

	mfmem::snew_array_1D(new_to_old, n_cells, dmrfl);
	mfmem::snew_array_1D(iperm, n_cells, dmrfl);
	mfmem::snew_array_1D(perm, n_cells, dmrfl);
	_metis_graph(this, n_cells, cell_idx, cell_neighbors);
	retcode = METIS_NodeND(&n_cells, cell_idx, cell_neighbors, NULL, NULL, perm, iperm);

	/* Update order_cell_oTon */
	if (retcode != METIS_OK) {
		for (i = 0; i < n_cells; i++)
			new_to_old[i] = i;
#ifdef MPICH
		mflog::log.set_one_processor_out();
#endif
		mflog::log << "CellReordering_metis failed..." <<std::endl;
	}
	else {
		for (i = 0; i < n_cells; i++)
			new_to_old[i] = (IntType)perm[i];
	}
	this->order_cell_nToo = new_to_old;
	this->order_cell_oTon = NULL;
	mfmem::snew_array_1D(this->order_cell_oTon, n_cells, dmrfl);
	for (i = 0; i < n_cells; i++) {
		this->order_cell_oTon[new_to_old[i]] = i;
	}
	mfmem::sdel_array_1D(iperm);
	mfmem::sdel_array_1D(perm);
}

/************************************************************************
					Compute local ordering using SCOTCH
						Add by dingxin 2021-9-25
************************************************************************/
void PolyGrid::CellReordering_scotch() {
	SCOTCH_Num   n_cells = this->nTCell;
	SCOTCH_Num   n_graph_cells = 0;
	SCOTCH_Graph  grafdat;  /* Scotch graph object to interface with libScotch */
	SCOTCH_Strat  stradat;
	SCOTCH_Ordering  order;

	SCOTCH_Num* peritab = NULL, * listtab = NULL;
	SCOTCH_Num* cell_idx = NULL, * cell_neighbors = NULL;
	IntType* new_to_old = NULL;

	int  retval = 0;
	n_graph_cells = _scotch_graph(this,n_cells,cell_idx,cell_neighbors);
	mfmem::snew_array_1D(new_to_old, n_cells, dmrfl);
	mfmem::snew_array_1D(peritab, n_graph_cells, dmrfl);
	mfmem::snew_array_1D(listtab, n_graph_cells, dmrfl);
	for (SCOTCH_Num i = 0; i < n_cells; i++) {
		new_to_old[i] = i;
		listtab[i] = i;
	}

	for (SCOTCH_Num i = n_cells; i < n_graph_cells; i++)
		peritab[i] = i; /* simple precaution, probably not required */
	/* Order using libScotch */
	SCOTCH_graphInit(&grafdat);
	retval = SCOTCH_graphBuild(&grafdat, 0, n_graph_cells, cell_idx, NULL, NULL, NULL,
		cell_idx[n_graph_cells], cell_neighbors, NULL);
	if (retval == 0) {
		SCOTCH_stratInit(&stradat);
		if (SCOTCH_graphCheck(&grafdat) == 0) {
			retval = SCOTCH_graphOrderInit(&grafdat,
				&order,
				NULL,   /* permtab */
				peritab,
				NULL,   /* cblkprt */
				NULL,   /* rangtab */
				NULL);  /* treetab */
			if (retval == 0) {
				retval = SCOTCH_graphOrderComputeList(&grafdat,
					&order,
					n_graph_cells,
					listtab,
					&stradat);  /* treetab */
				if (retval != 0) {
					for (SCOTCH_Num i = 0; i < n_cells; i++)
						peritab[i] = i;
				}
				SCOTCH_graphOrderExit(&grafdat, &order);
			}
		}
	}
	SCOTCH_graphExit(&grafdat);

	/* Update order_cell_oTon */
	if (retval != 0) {
#ifdef MPICH
		mflog::log.set_one_processor_out();
#endif
		mflog::log << "CellReordering_scotch failed..." << std::endl;
	}
	else {
		SCOTCH_Num j = 0;
		for (SCOTCH_Num i = 0; i < n_graph_cells; i++) {
			if (peritab[i] < n_cells)
				new_to_old[j++] = peritab[i];
		}
	}

	this->order_cell_nToo = new_to_old;
	this->order_cell_oTon = NULL;
	mfmem::snew_array_1D(this->order_cell_oTon, n_cells, dmrfl);
	for (IntType i = 0; i < n_cells; i++) {
		this->order_cell_oTon[new_to_old[i]] = i;
	}
	mfmem::sdel_array_1D(listtab);
	mfmem::sdel_array_1D(peritab);
	mfmem::sdel_array_1D(cell_idx);
	mfmem::sdel_array_1D(cell_neighbors);
}

/************************************************************************
Purpose:  reorder faces according to cell lables (c1 and c2) ascendingly
************************************************************************/
void PolyGrid::FaceReordering()
{
	IntType *f2c;
	IntType *nNPF;
	IntType *f2n;
	
	f2c = this->Getf2c();
	f2n = this->Getf2n();
	nNPF = this->GetnNPF();
	IntType *f2c_pface = NULL;
	IntType *f2c_pbackup = NULL;
	IntType *index_pface = NULL;

	IntType *f2c_iface = NULL;
	IntType *index_iface = NULL;
	IntType *f2c_iface_c2 = NULL;

	IntType pfacenum = nBFace - nIFace;
	IntType ifacenum = nTFace - nBFace;
	mfmem::snew_array_1D(f2c_pface, pfacenum, dmrfl);
	mfmem::snew_array_1D(index_pface, pfacenum, dmrfl);
	mfmem::snew_array_1D(f2c_pbackup, 2 * pfacenum, dmrfl);
	mfmem::snew_array_1D(f2c_iface, ifacenum, dmrfl);
	mfmem::snew_array_1D(index_iface, ifacenum, dmrfl);

	mfmem::snew_array_1D(f2c_iface_c2, ifacenum, dmrfl);
	//1. reordering faces according to c1
	for (int i = 0; i < pfacenum; ++i)
		f2c_pface[i] = f2c[i * 2];
	for (IntType i = 0; i < pfacenum; ++i)
		index_pface[i] = i;
	for (int i = nBFace; i < nTFace; ++i)
	{
		f2c_iface[i-nBFace] = f2c[i * 2];
		//f2c_iface_c2[i] = f2c[i * 2 + 1];
	}
	for (IntType i = nBFace; i < nTFace; ++i)
		index_iface[i-nBFace] = i;

	//reorder physical faces (not include interface faces)
	quicksort_int(1, pfacenum, index_pface, f2c_pface);
	//reorder interior faces according to c1
	quicksort_int(1, ifacenum, index_iface, f2c_iface);	

	//2. update f2c for physical faces
	for (int i = 0; i < 2 * pfacenum; ++i)
		f2c_pbackup[i] = f2c[i];

	for (int i = 0; i < pfacenum; ++i)
	{
		f2c[i * 2] = f2c_pface[i];
		f2c[i * 2 + 1] = f2c_pbackup[index_pface[i] * 2 + 1];
	}

	//3 update f2n and nNPF for physical faces
	//3.1 prepare information and data structures
	IntType *nNPF_pface = NULL;
	mfmem::snew_array_1D(nNPF_pface, pfacenum, dmrfl);
	IntType *f2n_pface = NULL;

	IntType m = 0;
	for (IntType i = 0; i < pfacenum; ++i)
	{
		m += nNPF[i];
	}
	mfmem::snew_array_1D(f2n_pface, m, dmrfl);

	m = 0;
	for (IntType i = 0; i < pfacenum; ++i)
	{
		nNPF_pface[i] = nNPF[i];
		for (IntType j = 0; j < nNPF[i]; ++j)
		{
			f2n_pface[m] = f2n[m];
			++m;
		}
	}

	IntType    **F2N_pface = NULL;
	mfmem::snew_array_1D(F2N_pface, pfacenum, dmrfl);
	F2N_pface[0] = f2n_pface;
	for (IntType i = 1; i < pfacenum; ++i)
	{
		F2N_pface[i] = &(F2N_pface[i - 1][nNPF_pface[i - 1]]);
	}

	//3.2. update nNPF for physical faces
	for (IntType i = 0; i < pfacenum; ++i)
	{
		nNPF[i] = nNPF_pface[index_pface[i]];
	}
	//3.3. update f2n for physical faces
	m = 0;
	for (IntType i = 0; i < pfacenum; ++i)
	{
		for (IntType j = 0; j < nNPF[i]; ++j)
		{
			f2n[m] = F2N_pface[index_pface[i]][j];
			++m;
		}

	}

	//4 .update f2c f2n and nNPF for interior faces
	UpdateIface(this, index_iface);

	//6. further reorder interior faces according to c2
	for (int i = nBFace; i < nTFace; ++i)
	{
		f2c_iface_c2[i - nBFace] = f2c[i * 2 + 1];
	}
	for (IntType i = nBFace; i < nTFace; ++i)
		index_iface[i-nBFace] = i;
	map<IntType, IntType> index_c1;
	IntType count = 1;
	for (IntType i = nBFace; i < nTFace; ++i)
	{
		if (f2c[i * 2] == f2c[(i + 1) * 2])
		{
			++count;
		}
		if (f2c[i * 2] != f2c[(i + 1) * 2])
		//if (f2c[(i+1) * 2] != f2c[(i + 2) * 2])
		{
			if (count>1)
			{
				index_c1[i-count+1] = count;
				count = 1;
			}
		}
		//i = i + count-1;
	}
	std::map<IntType, IntType>::iterator it = index_c1.begin();
	while (it != index_c1.end())
	{
		
		quicksort_int(it->first+1-nBFace, (it->first+it->second-nBFace), index_iface, f2c_iface_c2);
		it++;
	}
	//update f2c f2n and nNPF for interior faces
	UpdateIface(this, index_iface);

	mfmem::sdel_array_1D(index_pface);
	mfmem::sdel_array_1D(f2c_pface);
	mfmem::sdel_array_1D(f2c_pbackup);
	mfmem::sdel_array_1D(nNPF_pface);
	mfmem::sdel_array_1D(F2N_pface);
	mfmem::sdel_array_1D(f2n_pface);
	mfmem::sdel_array_1D(f2c_iface);
	mfmem::sdel_array_1D(index_iface);
	mfmem::sdel_array_1D(f2c_iface_c2);
}
#endif // REORDER

/************************************************************************
Purpose:  colouring faces for fine-grained parallelization (i.e., openmp and simd)
************************************************************************/
void FaceColouring(PolyGrid* grid, IntType bgroupsize, IntType igroupsize)
{
	std::set<IntType> cellset;
	IntType nTCell, nBFace, nTFace, nIFace;
	IntType* f2c, * f2n, * nNPF;

	nTCell = grid->GetNTCell();
	nBFace = grid->GetNBFace();
	nIFace = grid->GetNIFace();
	nTFace = grid->GetNTFace();
	f2c = grid->Getf2c();
	f2n = grid->Getf2n();
	nNPF = grid->GetnNPF();

	IntType* index_bface = NULL;
	IntType* nNPF_bface = NULL;
	IntType* f2n_bface = NULL;
	IntType** F2N_bface = NULL;
	IntType* index_bface_1 = NULL;

	IntType* index_iface = NULL;
	IntType* index_iface_1 = NULL;

	IntType ifacenum = nTFace - nBFace;

	IntType    n = nTCell + nBFace;
	IntType pfacenum = nBFace - nIFace;

	//allocate memory
	//mfmem::snew_array_1D(f2c_pface, pfacenum, dmrfl);
	mfmem::snew_array_1D(index_bface, pfacenum, dmrfl);
	//mfmem::snew_array_1D(nNPF_bface, nBFace, dmrfl);
	//mfmem::snew_array_1D(f2n_bface, nBFace, dmrfl);
	//mfmem::snew_array_1D(F2N_bface, nBFace, dmrfl);
	mfmem::snew_array_1D(index_bface_1, pfacenum, dmrfl);

	mfmem::snew_array_1D(index_iface, ifacenum, dmrfl);
	mfmem::snew_array_1D(index_iface_1, ifacenum, dmrfl);

	//initialization
	for (IntType i = 0; i < pfacenum; i++)
		index_bface[i] = -1;

	for (IntType i = nBFace; i < nTFace; i++)
		index_iface[i - nBFace] = -1;

	IntType newindex = 0, bgroupnum = 0, igroupnum = 0;
	//std::vector<IntType> bfacegroup;
	//std::vector<IntType> ifacegroup;
	cellset.clear();
	//2.2 colouring boundary faces including interface faces
	IntType bgrouplength = 0;
	do {
		bgrouplength = 0;
		for (IntType i = 0; i < pfacenum; i++)
		{
			if (index_bface[i] == -1)//unmarked face
			{
				if (cellset.count(f2c[2 * i]) == 0)//unmarked c1 cell, cellset容器里还没有f2c[2*i], 
				 //该重循环以f2c[2*i]为reference cell
				{
					index_bface[i] = newindex; //将newindex存入index_bface, 表明该面已着色,注意i 仍有信息
					index_bface_1[newindex] = i; //index_bface[i]转置存储格式，用于重排序，将相同的颜色面放在一起
					newindex = newindex + 1;    //此面已着色，着色数量加一
					cellset.insert(f2c[2 * i]);  //mark c1 cell
					bgrouplength = bgrouplength + 1;
					if (bgrouplength >= bgroupsize)
						break;
				}
			}
		}
		std::cout << "newindex:" << newindex << endl;
		
		//edpas(ngroup)=newindex-1;
		(*grid).bfacegroup.push_back(newindex);//??attention :newindex is also ok, 表明该颜色group内c1 cell个数
	  //cellsets.push_back(cellset);//unnecessary???
		cellset.clear();
		bgroupnum = bgroupnum + 1;  //最终得到bgroupnum个颜色，对于边界面

	} while (newindex < pfacenum);//until all boundary faces marked, 每重do循环新建一个color group

	//2.3 update f2c,nNPF,f2n for boundary faces
	IntType* f2c_backup = NULL;
	mfmem::snew_array_1D(f2c_backup, 2 * pfacenum, dmrfl);
	/*
	for (IntType i = 0; i < 2 * nTFace; i++)
	{
		f2c_backup[i] = f2c[i];
	}
	*/
	for (IntType i = 0; i < pfacenum; i++)
	{
		f2c_backup[2 * i] = f2c[2 * i];
		f2c_backup[2 * i + 1] = f2c[2 * i + 1];
	}
	//2.3.1 update f2c for boundary faces
	for (IntType i = 0; i < pfacenum; i++)
	{
		f2c[2 * i] = f2c_backup[index_bface_1[i] * 2];
		f2c[2 * i + 1] = f2c_backup[index_bface_1[i] * 2 + 1];
	}

	IntType bfirst = (*grid).bfacegroup[0];

	//update nNPF, f2n for boundary faces
	//prepare information and data structures
	//IntType *nNPF_bface = NULL;
	mfmem::snew_array_1D(nNPF_bface, pfacenum, dmrfl);
	//IntType *f2n_bface = NULL;

	IntType m = 0;
	for (IntType i = 0; i < pfacenum; i++)
	{
		m += nNPF[i];
	}
	mfmem::snew_array_1D(f2n_bface, m, dmrfl);

	m = 0;
	for (IntType i = 0; i < pfacenum; i++)
	{
		nNPF_bface[i] = nNPF[i];
		for (IntType j = 0; j < nNPF[i]; j++)
		{
			f2n_bface[m] = f2n[m];
			m++;
		}
	}

	//IntType    **F2N_bface = NULL;
	mfmem::snew_array_1D(F2N_bface, pfacenum, dmrfl);
	F2N_bface[0] = f2n_bface;
	for (IntType i = 1; i < pfacenum; i++)
	{
		F2N_bface[i] = &(F2N_bface[i - 1][nNPF_bface[i - 1]]);
	}

	//2.3.2. update nNPF for physical faces
	for (IntType i = 0; i < pfacenum; i++)
	{
		nNPF[i] = nNPF_bface[index_bface_1[i]];
	}
	//2.3.3. update f2n for physical faces
	m = 0;
	for (IntType i = 0; i < pfacenum; i++)
	{
		for (IntType j = 0; j < nNPF[i]; j++)
		{
			f2n[m] = F2N_bface[index_bface_1[i]][j];
			m++;
		}

	}

	//2.4 colouring interior faces
	newindex = nBFace;
	cellset.clear();
	IntType igrouplength = 0;
	do {
		igrouplength = 0;
		for (IntType i = nBFace; i < nTFace; i++)
		{
			if (index_iface[i - nBFace] == -1)//unmarked face
			{
				if (cellset.count(f2c[2 * i]) == 0)//unmarked c1 cell
				{
					if (cellset.count(f2c[2 * i + 1]) == 0)//unmarked c2 cel
					{
						index_iface[i - nBFace] = newindex;//?
						//newindex = newindex + 1;
						//index_iface[newindex - nBFace] = i;//?
						index_iface_1[newindex - nBFace] = i;
						newindex = newindex + 1;
						cellset.insert(f2c[2 * i]);  //mark c1 cell
						cellset.insert(f2c[2 * i + 1]);
						igrouplength = igrouplength + 1;
						if (igrouplength >= igroupsize)
							break;
					}
				}
			}
		}
		//edpas(ngroup)=newindex-1;
		(*grid).ifacegroup.push_back(newindex);
		//	cellsets.push_back(cellset);
		cellset.clear();
		igroupnum = igroupnum + 1;

	} while (newindex < nTFace);//until all interior faces marked

	//2.5 update f2c,nNPF,f2n for interior faces
	UpdateIface(grid, index_iface_1);

	//deallocation
	mfmem::sdel_array_1D(index_bface);
	mfmem::sdel_array_1D(nNPF_bface);
	mfmem::sdel_array_1D(f2n_bface);
	mfmem::sdel_array_1D(F2N_bface);
	mfmem::sdel_array_1D(index_bface_1);

	mfmem::sdel_array_1D(index_iface);
	mfmem::sdel_array_1D(f2c_backup);
	mfmem::sdel_array_1D(index_iface_1);
}
//end added by xchf

void FaceColouringBalancing(PolyGrid* grid)
{
    std::set<IntType> cellset;

    IntType nTCell, nBFace, nTFace, nIFace;
    IntType* f2c, * f2n, * nNPF;

    nTCell = grid->GetNTCell();
    nBFace = grid->GetNBFace();
    nTFace = grid->GetNTFace();
    nIFace = grid->GetNIFace();
    f2c = grid->Getf2c();
    f2n = grid->Getf2n();
    nNPF = grid->GetnNPF();

    IntType ifacenum = nTFace - nBFace;
    IntType pfacenum = nBFace - nIFace;
    // Get number of faces for each cell
    IntType* nFPC = CalnFPC(grid);
    // Get cell to face conections
    IntType** C2F = CalC2F(grid);
    IntType maxFace = 0;
    for (IntType i = 0; i < nTCell; i++) {
        if (maxFace < nFPC[i]) {
            maxFace = nFPC[i];
        }
    }
    //std::cout << "maxnumFace: " << maxFace << endl;
    // Get number of faces for each cell
    IntType BMaxFace = 0;
    IntType IMaxFace = 0;
    //find the maxnumber of boundary faces per cell:BMaxFace
    for (IntType i = 0; i < nTCell; i++) {
        IntType localMaxFace = 0;
        for (IntType j = 0; j < nFPC[i]; j++) {
            IntType f1 = C2F[i][j];
            if (f1 < pfacenum) {
                localMaxFace++;
            }
        }
        if (BMaxFace < localMaxFace) {
            BMaxFace = localMaxFace;
        }
    }
    //std::cout << "maxnumFace for BFace: " << BMaxFace << endl;
    //find the maxnumber of interior faces per cell:IMaxFace
    for (IntType i = 0; i < nTCell; i++) {
        IntType localMaxFace = 0;
        for (IntType j = 0; j < nFPC[i]; j++) {
            IntType f1 = C2F[i][j];
            if (f1 >= pfacenum) {
                localMaxFace++;
            }
        }
        if (IMaxFace < localMaxFace) {
            IMaxFace = localMaxFace;
        }
    }
    //std::cout << "maxnumFace for IFace: " << IMaxFace << endl;
    //mfmem::sdel_array_2D(C2F);
    C2F = NULL;
    grid->SetC2F(C2F);
    //mfmem::sdel_array_1D(nFPC);
    IntType BMaxColors = BMaxFace + 3;  //this case equals to 2+2
    IntType IMaxColors = IMaxFace + 10;  //this case equals to 6+3
    vector< set<IntType> > BFaceColor(BMaxColors);
    vector< set<IntType> > IFaceColor(IMaxColors);

    IntType* nNPF_bface = NULL;
    IntType* f2n_bface = NULL;
    IntType** F2N_bface = NULL;
    IntType* f2c_bface = NULL;
    IntType* index_bface_1 = NULL;

    IntType* index_iface = NULL;
    IntType* nNPF_iface = NULL;
    IntType* f2n_iface = NULL;
    IntType** F2N_iface = NULL;
    IntType* f2c_iface = NULL;
    IntType* index_iface_1 = NULL;


    IntType    n = nTCell + nBFace;
    /*
    IntType* conflict_bface, * conflict_iface;
    conflict_bface = NULL;
    conflict_iface = NULL;
    mfmem::snew_array_1D(conflict_bface, n, dmrfl);
    mfmem::snew_array_1D(conflict_iface, n, dmrfl);
    for (IntType i = 0; i < n; i++) {
        conflict_bface[i] = 0;
        conflict_iface[i] = 0;
    }
    */
    IntType* index_bface = NULL;
    mfmem::snew_array_1D(index_bface, pfacenum, dmrfl);
    for (IntType i = 0; i < pfacenum; i++) {
        index_bface[i] = -1;
    }
    mfmem::snew_array_1D(index_bface_1, pfacenum, dmrfl);//记录着色面更新后需要存的新位置

    IntType* index_bcolor = NULL;
    mfmem::snew_array_1D(index_bcolor, BMaxColors, dmrfl);//记录每种颜色的面数量
    //每种颜色的面数量赋零
    for (IntType i = 0; i < BMaxColors; i++) {
        index_bcolor[i] = 0;
    }
    IntType facecolor = 0; //已着色面总数
    //New Method:
    //初始可用颜色数量：
    IntType initialcolor = BMaxFace;
    //当前颜色数量：
    IntType currentcolor = initialcolor;
    do {
        //加色：currentcolor++;
        //颜色容器初始化：
        //容器颜色清空：
        for (IntType j = 0; j < currentcolor; j++) {
            BFaceColor[j].clear();
        }
        facecolor = 0;
        for (IntType j = 0; j < BMaxColors; j++) {
            index_bcolor[j] = 0;
        }
        for (IntType i = 0; i < pfacenum; i++) {
            //找到临近c1 cell:
            IntType c1 = f2c[2 * i];
            //从各个颜色set容器里统计该面剩余可用颜色数量：
            IntType avialcolor = 0;
            for (IntType j = 0; j < currentcolor; j++) {
                if (BFaceColor[j].count(c1) == 0) { //此类颜色对于该面是nice的^_^ nice!
                    avialcolor++;
                }
            }//得到该面剩余的nice色数量
            if (avialcolor == 0) { //对于该面，无色可用………………
                currentcolor++;
                break; //需要加色！重启do循环！
            }
            //根据nice色数量avialcolor, 随机抽取一种颜色：
            IntType randcolor = rand() % avialcolor; //randcolor: 0 to (avialcolor-1)
            IntType samecolor = 0;
            for (IntType j = 0; j < currentcolor; j++) {
                if (BFaceColor[j].count(c1) == 0) { //avialcolor次if判断成立
                    if (samecolor == randcolor) {
                        BFaceColor[j].insert(c1);
                        facecolor++;//着色总面数
                        index_bcolor[j]++;//每种颜色的着色面数量
                        index_bface[i] = j;//每个面的所着颜色
                        break;
                    }
                    samecolor++;
                }
            }
        }
    } while (facecolor < pfacenum);

    //currentcolor代表最终着色数量
    //eg. 颜色0：200面，颜色1:500面，颜色2:100面；currentcolor=3
    IntType sumfacecolor = 0;
    for (IntType j = 0; j < currentcolor; j++) {
        sumfacecolor += index_bcolor[j];
        (*grid).bfacegroup.push_back(sumfacecolor);
        if (j > 0) {
            index_bcolor[j] = sumfacecolor - index_bcolor[j];
        }
        else {
            index_bcolor[j] = 0;
        }
    }

    for (IntType i = 0; i < pfacenum; i++) {
        index_bface_1[index_bcolor[index_bface[i]]] = i;
        index_bcolor[index_bface[i]]++;
    }

    mfmem::snew_array_1D(index_iface, ifacenum, dmrfl);
    mfmem::snew_array_1D(nNPF_iface, ifacenum, dmrfl);
    mfmem::snew_array_1D(index_iface_1, ifacenum, dmrfl);

    mfmem::snew_array_1D(F2N_iface, ifacenum, dmrfl);

    //initialization
    for (IntType i = nBFace; i < nTFace; i++)
        index_iface[i - nBFace] = -1;

    IntType newindex = 0, bgroupnum = 0, igroupnum = 0;


    //2.3 update f2c,nNPF,f2n for boundary faces
    IntType* f2c_backup = NULL;
    mfmem::snew_array_1D(f2c_backup, 2 * nTFace, dmrfl);

    for (IntType i = 0; i < nTFace; i++)
    {
        f2c_backup[2 * i] = f2c[2 * i];
        f2c_backup[2 * i + 1] = f2c[2 * i + 1];
    }
    //2.3.1 update f2c for boundary faces
    for (IntType i = 0; i < pfacenum; i++)
    {
        f2c[2 * i] = f2c_backup[index_bface_1[i] * 2];
        f2c[2 * i + 1] = f2c_backup[index_bface_1[i] * 2 + 1];
    }

    //update nNPF, f2n for boundary faces
    //prepare information and data structures
    //IntType *nNPF_bface = NULL;
    mfmem::snew_array_1D(nNPF_bface, pfacenum, dmrfl);
    //IntType *f2n_bface = NULL;

    IntType m = 0;
    for (IntType i = 0; i < pfacenum; i++)
    {
        m += nNPF[i];
    }
    mfmem::snew_array_1D(f2n_bface, m, dmrfl);

    m = 0;
    for (IntType i = 0; i < pfacenum; i++)
    {
        nNPF_bface[i] = nNPF[i];
        for (IntType j = 0; j < nNPF[i]; j++)
        {
            f2n_bface[m] = f2n[m];
            m++;
        }
    }

    //IntType    **F2N_bface = NULL;
    mfmem::snew_array_1D(F2N_bface, pfacenum, dmrfl);
    F2N_bface[0] = f2n_bface;
    for (IntType i = 1; i < pfacenum; i++)
    {
        F2N_bface[i] = &(F2N_bface[i - 1][nNPF_bface[i - 1]]);
    }

    //2.3.2. update nNPF for physical faces
    for (IntType i = 0; i < pfacenum; i++)
    {
        nNPF[i] = nNPF_bface[index_bface_1[i]];
    }
    //2.3.3. update f2n for physical faces
    m = 0;
    for (IntType i = 0; i < pfacenum; i++)
    {
        for (IntType j = 0; j < nNPF[i]; j++)
        {
            f2n[m] = F2N_bface[index_bface_1[i]][j];
            m++;
        }

    }
    IntType nodecount_pface = m;
    /*
    for (IntType i = 0; i < n; i++) {
        conflict_bface[i] = 0;
    }
    IntType bfirst = (*grid).bfacegroup[0];
    for (IntType i = 0; i < bfirst; i++) {
        IntType c1 = f2c[2 * i];
        conflict_bface[c1] += 1;
    }
    for (IntType i = 0; i < n; i++) {
        if (conflict_bface[i] > 1) {
            std::cout << "conflict coloring: bface!!!!!!!" << endl;
            cout << i << " " << conflict_bface[i] << endl;
            break;
        }
    }
    */
    //2.4 colouring interior faces

    IntType* index_icolor = NULL;
    mfmem::snew_array_1D(index_icolor, IMaxColors, dmrfl);//记录每种颜色的面数量
    //每种颜色的面数量赋零
    for (IntType i = 0; i < IMaxColors; i++) {
        index_icolor[i] = 0;
    }
    facecolor = nBFace; //已着色面总数
    //New Method:
    //初始可用颜色数量：
    initialcolor = IMaxFace + 3;
    //当前颜色数量：
    currentcolor = initialcolor;
    do {
        //加色：currentcolor++;
        //颜色容器初始化：
        //容器颜色清空：
        for (IntType j = 0; j < currentcolor; j++) {
            IFaceColor[j].clear();
        }
        facecolor = nBFace;
        for (IntType j = 0; j < IMaxColors; j++) {
            index_icolor[j] = 0;
        }
        for (IntType i = nBFace; i < nTFace; i++) {
            //找到临近c1,c2 cell:
            IntType c1 = f2c[2 * i];
            IntType c2 = f2c[2 * i + 1];
            //从各个颜色set容器里统计该面剩余可用颜色数量：
            IntType avialcolor = 0;
            for (IntType j = 0; j < currentcolor; j++) {
                if (IFaceColor[j].count(c1) == 0) { //此类颜色对于该面是nice的^_^ nice!
                    if (IFaceColor[j].count(c2) == 0) {//此类颜色对于该面是nice的^_^ nice again!
                        avialcolor++;
                    }
                }
            }//得到该面剩余的nice色数量
            if (avialcolor == 0) { //对于该面，无色可用………………
                currentcolor++;
                break; //需要加色！重启do循环！
            }
            //根据nice色数量avialcolor, 随机抽取一种颜色：
            IntType randcolor = rand() % avialcolor; //randcolor: 0 to (avialcolor-1)
            IntType samecolor = 0;
            for (IntType j = 0; j < currentcolor; j++) {
                if (IFaceColor[j].count(c1) == 0) { //avialcolor次if判断成立
                    if (IFaceColor[j].count(c2) == 0) {
                        if (samecolor == randcolor) {
                            IFaceColor[j].insert(c1);
                            IFaceColor[j].insert(c2);
                            facecolor++;//着色总面数
                            index_icolor[j]++;//每种颜色的着色面数量
                            index_iface[i - nBFace] = j;//每个面的所着颜色
                            break;
                        }
                        samecolor++;
                    }
                }
            }
        }
        IntType samecolor = 0;
    } while (facecolor < nTFace);
    //currentcolor代表最终着色数量
    //eg. 颜色0：200面，颜色1:500面，颜色2:100面；currentcolor=3
    /*
    for (IntType j = 0; j < currentcolor; j++) {
        cout << "index_icolor[" << j << "]: " << index_icolor[j] << endl;
    }
    */
    sumfacecolor = nBFace;
    for (IntType j = 0; j < currentcolor; j++) {
        sumfacecolor += index_icolor[j];
        (*grid).ifacegroup.push_back(sumfacecolor);
        if (j > 0) {
            index_icolor[j] = sumfacecolor - index_icolor[j];
        }
        else {
            index_icolor[j] = nBFace;
        }
    }
    /*
    for (IntType j = 0; j < currentcolor; j++) {
        cout << "index_icolor[" << j << "]: " << index_icolor[j] << endl;
    }
    */
    for (IntType i = nBFace; i < nTFace; i++) {
        //index_iface_1[index_icolor[index_iface[i - nBFace]] - nBFace] = i; //着色后需移动到的新面位置
        IntType jj = index_iface[i - nBFace];
        IntType ii = index_icolor[jj] - nBFace;
        index_iface_1[ii] = i; //着色后需移动到的新面位置
        index_icolor[jj]++;
    }

    /*
    //simple test:
    for (IntType j = 0; j < currentcolor; j++) {
        cout << "color " << j << ":" << index_icolor[j] << endl;
    }
    //test:
    for (IntType i = nBFace; i < nTFace; i++) {
        if (index_iface[i - nBFace] == 1) {
            IntType c1 = f2c[2 * i];
            IntType c2 = f2c[2 * i + 1];
            conflict_iface[c1] += 1;
            conflict_iface[c2] += 1;
        }
    }
    for (IntType i = 0; i < n; i++) {
        if (conflict_iface[i] > 1) {
            std::cout << "conflict coloring: bface!!!!!!!" << endl;
            cout << i << " " << conflict_iface[i] << endl;
            break;
        }
    }
    */

    //2.5 update f2c,nNPF,f2n for interior faces
	UpdateIface(grid, index_iface_1);
    /*
    IntType ifirst = (*grid).ifacegroup[0];
    for (IntType i = nBFace; i < ifirst; i++) {
        IntType c1 = f2c[2 * i];
        IntType c2 = f2c[2 * i + 1];
        conflict_iface[c1] += 1;
        conflict_iface[c2] += 1;
    }
    for (IntType i = 0; i < n; i++) {
        if (conflict_iface[i] > 1) {
            std::cout << "conflict coloring: bface!!!!!!!" << endl;
            cout << i << " " << conflict_iface[i] << endl;
            break;
        }
    }
    */
    //deallocation
    mfmem::sdel_array_1D(index_bface);
    mfmem::sdel_array_1D(nNPF_bface);
    mfmem::sdel_array_1D(f2n_bface);
    mfmem::sdel_array_1D(F2N_bface);
    mfmem::sdel_array_1D(index_bface_1);
    
    mfmem::sdel_array_1D(index_bcolor);
    
    //mfmem::sdel_array_1D(f2c_bface);

    mfmem::sdel_array_1D(index_iface);
    mfmem::sdel_array_1D(nNPF_iface);
    mfmem::sdel_array_1D(f2n_iface);
    mfmem::sdel_array_1D(F2N_iface);
    //mfmem::sdel_array_1D(f2c_iface);
    mfmem::sdel_array_1D(f2c_backup);
    mfmem::sdel_array_1D(index_iface_1);
    mfmem::sdel_array_1D(index_icolor);

    //mfmem::sdel_array_1D(conflict_bface);
    //mfmem::sdel_array_1D(conflict_iface);

}//add by ruitian

#undef CPP_FILD_ID  // clear out file id
} //~namespace mflow
