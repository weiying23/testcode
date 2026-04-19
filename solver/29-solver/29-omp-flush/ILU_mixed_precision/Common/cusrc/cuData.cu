#include <stdio.h>
#include <iostream>

#include <number_type.h>
#include <solver_ns.h>

#include <cuData.cuh>
#include <cuErrorReturn.cuh>
#include <cuLUSGS.cuh>
#include "cuLimit.cuh"
#include "cuGradientQ_Gauss.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

namespace gpuData{
	
	cudaStream_t flowstream[5];

	//GPU Device Parameter:
	IntType   threadsPerBlock = 512;	//thread size per block 
	
	//Grid Info:
	IntType   gnTFace, gnTCell; 		// no. of total faces, no of total cells
    IntType   gnBFace;         			// boundary faces which include interfaces
    IntType   gnIFace;         			// parallel interface faces(zero if serial program)
    IntType   gnTNode;					// no. of total nodes
    IntType   glenC2C;

	IntType  *gf2c;						// face->cell connectivity
	RealGeom *gxcc, *gycc, *gzcc;  		// cell center data, including ghosts
    RealGeom *gxfc, *gyfc, *gzfc;  		// face center data;
	RealGeom *gxfn, *gyfn, *gzfn;		// face unit normal;
	RealGeom *garea, *gvol;       		// face area, cell volume excluding ghosts;
	RealGeom *gvgn;  					// normal velocity for all faces
	IntType  *gtype_bcr;				// associate each boundary. face with a record
	RealGeom *gtw_bcr;  				// viscous iso-thermal wall
	IntType  *gC2F, *gIndexC2F, *gnFPC;	// cell to face connect
	RealGeom *gfacecentroidskewness;
	RealFlow *gangle_h;		// for Vis asin() function on GPU
	IntType  *gnNPF, *gF2N, *gIndexF2N;	// face->node connectivity
	IntType  *gN2C, *gIndexN2C, *gnCPN;	// node to cell connection
	RealGeom *gWeightNodeN2C, *gWeightNodeBFace2C;
	IntType  *gNmark;
	RealGeom *gWeightNode;
	IntType  *gnodesymm;
	RealGeom *gxfn_n_symm, *gyfn_n_symm, *gzfn_n_symm; 
	RealGeom *gnorm_dist_c2c;
	IntType  *gfcptr;

	IntType  *gcellwallnumber, *gCellLayerNo;

	IntType  *glayer, *gluorder, *gcellsPerlayer;
	IntType  *gC2C, *gIndexC2C, *gnCPC;	// cell to cell connect
	RealGeom *gdist2wall;
	
#if (defined GroupColor)
	// to load cell values into share memory:
	IntType *g_b_SMc2c;
	IntType *g_i_SMc2c;
	// to store the share memory unit index:
	IntType *g_b_SM_index;
	IntType *g_i_SM_index;
	// to store the share memory index between colors:
	//IntType *g_b_SM_color_index;
	//IntType *g_i_SM_color_index;
	// to load face values into share memory cell values:
	IntType *g_b_f2SMc;
	IntType *g_i_f2SMc;
#endif
  	
  	//Flow Info:
  	IntType	  gsteady;					//steady(1) or unsteady(0) flow
  	RealFlow  gp_bar;
  	IntType   gGaussLayer;				// for Gasuu Gradient Comput.
  	RealFlow  glhs_omga;
  	RealFlow  ggam, geps_tmp;

  	// Flow Memory Alloc:
  	// for reduction.
  	RealFlow *gtmpvar;
  	// Flux Comput.
  	IntType  *gIsNormalFace, *gIsShockFace;
  	RealFlow *gql, *gqr, *gq;
  	RealFlow *gsa_nu;
  	RealFlow *gflux;
  	RealFlow *gdqdx, *gdqdy, *gdqdz;
  	RealFlow *gdtdx, *gdtdy, *gdtdz;
  	RealFlow *glimit;
  	RealFlow *gres;
  	RealFlow *gvis_l, *gvis_t, *gvisc_f, *gheat_f;
  	RealFlow *gvel, *gvel_f, *gt, *gt_f;
  	RealGeom *gdeltl, *gdeltr;
  	RealFlow *gdt;
  	// Limiter Comput.
  	RealFlow *gdmax, *gdmin, *gespcell, *gdmax_MStream, *gdmin_MStream, *gespcell_MStream;
  	RealFlow *gtmp_limit;
  	// Gradient Comput.
  	RealFlow *gtmpxyz;
  	RealFlow *gq_n, *gfacq;
  	RealFlow *gdnutdx, *gdnutdy, *gdnutdz;
  	RealFlow *gfacu, *gfacv, *gfacw;
  	RealFlow *gu_n, *gv_n, *gw_n, *gvn;
#ifdef MultiStream
	RealFlow *gmsq_n;
	RealFlow *hostmsq_n;
	RealFlow *gtmpxyz_u, *gtmpxyz_v, *gtmpxyz_w, *gtmpxyz_p, *gtmpxyz_T, *gtmpxyz_sa;
	RealFlow *hostu_n, *hostv_n, *hostw_n;
#endif	
  	// LUSGS Comput.
  	RealFlow *gDiag, *gDiag_v;
  	RealFlow *gDQ;
	RealFlow *gdqo, *gnorm;
	// RK COmput.
	RealFlow *goldq;
  	// SA Comput.
  	RealFlow *glhsmat;
  	RealFlow *gomaga;
	RealFlow *ggradnue2;
	RealFlow *gdqdl, *gdqdr;
	RealFlow *gtem, *gtem_c2;
	RealFlow *gdkdn, *gdkdnc1, *gdkdnc2;
	// TimeStep: dt
	IntType  *gdet;
	// MPI:
	RealFlow *gMPI;
	RealFlow *gbqs, *gbqr;				// the length of bqs/bqr selects the maximum length, that is Grad bqs/bqr, 3*5*(nTCell+nBFace)
	RealFlow *hostbqs, *hostbqr;		// the length of bqs/bqr selects the maximum length, that is Grad bqs/bqr, 3*5*(nTCell+nBFace)

	IntType  *gIndexbqsr;				// the Index of bqs/bqr for q, dq, limit, of which the length equals to 5*(nTCell+nBFace)
	IntType  *gIndexbqsrSA;				// the length of bqs/bqr for sa_nu, vis_l, vis_t, dt, of which the length equals to nTCell+nBFace
	IntType  *gIndexbqsrGrad;			// the length of bqs/bqr for dqdx, dqdy, dqdz, of which the length equals to 3*5*(nTCell+nBFace)
	IntType  *gIndexbqsr2;				// the Index of bqs/bqr for q, dq, limit, of which the length equals to 5*(nTCell+nBFace)
	IntType  *gIndexbqsrGradSA;			// the length of bqs/bqr for dqdx, dqdy, dqdz, of which the length equals to 3*(nTCell+nBFace)
	IntType  *gIndexbqsr2SA;				// the length of bqs/bqr for sa_nu, vis_l, vis_t, dt, of which the length equals to nTCell+nBFace
	IntType  *gIndexbqsr2GradSA;			// the length of bqs/bqr for dqdx, dqdy, dqdz, of which the length equals to 3*(nTCell+nBFace)

	IntType   glenbqsr;					// the length of bqs/bqr for q, dq, limit, of which the length equals to 5*(nTCell+nBFace)
    IntType   glenbqsrSA;				// the length of bqs/bqr for sa_nu, vis_l?, vis_t, dt?, of which the length equals to nTCell+nBFace
	IntType   glenbqsrGrad;				// the length of bqs/bqr for dqdx, dqdy, dqdz, of which the length equals to 3*5*(nTCell+nBFace)
	IntType   glenbqsrGradSA;			// the length of bqs/bqr for dnutdx, dnutdy, dnutdz, dtdx, dtdy, dtdz, of which the length equals to 3*(nTCell+nBFace)
	
	IntType   glenbqsr_Node;
	
	// GMRES:
	RealFlow *greso, *gdq;
	RealFlow *gDQo, *gv, *gw;
	RealFlow *gturburhs, *gdqadu;
	RealFlow *gsumv, *godata;
	IntType   gnsum, gnodata;
	RealFlow *gsumv2, *godata2;
	IntType   gnsum2, gnodata2;
	RealFlow *gSAsumv2, *gSAodata2;
	IntType   gSAnsum2, gSAnodata2;
	
	// dt max and min:
	RealFlow *gdtmaxsumv2, *gdtmaxodata2;
	IntType   gdtmaxnsum2, gdtmaxnodata2;
	RealFlow *gdtminsumv2, *gdtminodata2;
	IntType   gdtminnsum2, gdtminnodata2;
	
	// Unified Memory for Reduction:
	RealFlow *val_Reduction;

	void GPUFaceNumberTrans(IntType nTFace, IntType nTCell, IntType nBFace, IntType nIFace, IntType nTNode){
		gnTFace = nTFace;
		gnBFace = nBFace;
		gnTCell = nTCell;
		gnIFace = nIFace;
		gnTNode = nTNode;
	}

	void GPUFaceDataTrans(RealGeom *xfc, RealGeom *yfc, RealGeom *zfc, RealGeom *xfn, RealGeom *yfn, RealGeom *zfn,
						 RealGeom *xcc, RealGeom *ycc, RealGeom *zcc, RealGeom *area, RealGeom *vgn, IntType *f2c,
						 IntType *type_bcr, RealGeom *tw_bcr){
		size_t sizeface = gnTFace*sizeof(RealGeom);
		size_t sizecell = (gnTCell + gnBFace)*sizeof(RealGeom);		// including ghosts

		// face center data:
		//cudaMalloc((void **)&gxfc, sizeface)
		HANDLE_API_ERR(cudaMalloc((void **)&gxfc, sizeface));		
		HANDLE_API_ERR(cudaMalloc((void **)&gyfc, sizeface));		
		HANDLE_API_ERR(cudaMalloc((void **)&gzfc, sizeface));

		// face unit normal:
		HANDLE_API_ERR(cudaMalloc((void **)&gxfn, sizeface));		
		HANDLE_API_ERR(cudaMalloc((void **)&gyfn, sizeface));		
		HANDLE_API_ERR(cudaMalloc((void **)&gzfn, sizeface));

		HANDLE_API_ERR(cudaMalloc((void **)&garea, sizeface));		

		if(!gsteady){
			HANDLE_API_ERR(cudaMalloc((void **)&gvgn, sizeface));
		}

		// face->cell connectivity
		HANDLE_API_ERR(cudaMalloc((void **)&gf2c, 2*gnTFace*sizeof(IntType)));

		// associate each boundary. face with a record
		HANDLE_API_ERR(cudaMalloc((void **)&gtype_bcr, gnBFace*sizeof(IntType)));
		// viscous iso-thermal wall
		HANDLE_API_ERR(cudaMalloc((void **)&gtw_bcr, gnBFace*sizeof(RealGeom)));
		
		// cell center data, including ghosts
		HANDLE_API_ERR(cudaMalloc((void **)&gxcc, sizecell));
		HANDLE_API_ERR(cudaMalloc((void **)&gycc, sizecell));
		HANDLE_API_ERR(cudaMalloc((void **)&gzcc, sizecell));
		
		
		// Transfer host data into device *xfc, *yfc, *zfc:
		HANDLE_API_ERR(cudaMemcpy(gxfc, xfc, sizeface, cudaMemcpyHostToDevice));		
		HANDLE_API_ERR(cudaMemcpy(gyfc, yfc, sizeface, cudaMemcpyHostToDevice));		
		HANDLE_API_ERR(cudaMemcpy(gzfc, zfc, sizeface, cudaMemcpyHostToDevice));

		// Transfer host data into device *xfn, *yfn, *zfn:
		HANDLE_API_ERR(cudaMemcpy(gxfn, xfn, sizeface, cudaMemcpyHostToDevice));		
		HANDLE_API_ERR(cudaMemcpy(gyfn, yfn, sizeface, cudaMemcpyHostToDevice));		
		HANDLE_API_ERR(cudaMemcpy(gzfn, zfn, sizeface, cudaMemcpyHostToDevice));

		// Transfer host data into device *xcc, *ycc, *zcc:
		HANDLE_API_ERR(cudaMemcpy(gxcc, xcc, sizecell, cudaMemcpyHostToDevice));		
		HANDLE_API_ERR(cudaMemcpy(gycc, ycc, sizecell, cudaMemcpyHostToDevice));		
		HANDLE_API_ERR(cudaMemcpy(gzcc, zcc, sizecell, cudaMemcpyHostToDevice));

		// Transfer host data into device *type_bcr, *tw_bcr:
		HANDLE_API_ERR(cudaMemcpy(gtype_bcr, type_bcr, gnBFace*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gtw_bcr, tw_bcr, gnBFace*sizeof(RealGeom), cudaMemcpyHostToDevice));

		HANDLE_API_ERR(cudaMemcpy(garea, area, sizeface, cudaMemcpyHostToDevice));

		if(!gsteady){
			HANDLE_API_ERR(cudaMemcpy(gvgn, vgn, sizeface, cudaMemcpyHostToDevice));
		}

		HANDLE_API_ERR(cudaMemcpy(gf2c, f2c, 2*gnTFace*sizeof(IntType), cudaMemcpyHostToDevice));
		
	}

	void GPUFaceDataTrans2(IntType **C2F, IntType *IndexC2F, IntType *nFPC, RealGeom *vol, RealGeom *facecentroidskewness,
						RealFlow *angle_h, IntType *nNPF, IntType **F2N, IntType *IndexF2N,
						IntType *cellwallnumber, IntType *CellLayerNo){

		size_t sizeface = gnTFace*sizeof(RealGeom);
		size_t sizecell = (gnTCell + gnBFace)*sizeof(RealGeom);		// including ghosts
		// cell to face connect
		HANDLE_API_ERR(cudaMalloc((void **)&gC2F, IndexC2F[gnTCell]*sizeof(IntType)));
		HANDLE_API_ERR(cudaMalloc((void **)&gIndexC2F, gnTCell*sizeof(IntType)));

		HANDLE_API_ERR(cudaMalloc((void **)&gnFPC, gnTCell*sizeof(IntType)));

		HANDLE_API_ERR(cudaMalloc((void **)&gvol, sizecell));

		HANDLE_API_ERR(cudaMalloc((void **)&gfacecentroidskewness, gnTFace*sizeof(RealGeom)));

		// Transfer host data into device **C2F, *IndexC2F:
		HANDLE_API_ERR(cudaMemcpy(gC2F, C2F[0], IndexC2F[gnTCell]*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gIndexC2F, IndexC2F, gnTCell*sizeof(IntType), cudaMemcpyHostToDevice));

		HANDLE_API_ERR(cudaMemcpy(gnFPC, nFPC, gnTCell*sizeof(IntType), cudaMemcpyHostToDevice));

		HANDLE_API_ERR(cudaMemcpy(gvol, vol, sizecell, cudaMemcpyHostToDevice));

		HANDLE_API_ERR(cudaMemcpy(gfacecentroidskewness, facecentroidskewness, gnTFace*sizeof(RealGeom), cudaMemcpyHostToDevice));

		// for face angel:
		HANDLE_API_ERR(cudaMalloc((void **)&gangle_h, 2*sizeface));
		//HANDLE_API_ERR(cudaMalloc((void **)&gangle2, sizeface));

		HANDLE_API_ERR(cudaMemcpy(gangle_h, angle_h, 2*sizeface, cudaMemcpyHostToDevice));
		//HANDLE_API_ERR(cudaMemcpy(gangle2, angle2, sizeface, cudaMemcpyHostToDevice));

		// for cellwallnumber, CellLayerNo
		HANDLE_API_ERR(cudaMalloc((void **)&gcellwallnumber, gnTCell*sizeof(IntType)));
		HANDLE_API_ERR(cudaMalloc((void **)&gCellLayerNo, gnTCell*sizeof(IntType)));

		HANDLE_API_ERR(cudaMemcpy(gcellwallnumber, cellwallnumber, gnTCell*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gCellLayerNo, CellLayerNo, gnTCell*sizeof(IntType), cudaMemcpyHostToDevice));

		// face->node connectivity:
		HANDLE_API_ERR(cudaMalloc((void **)&gnNPF, gnTFace*sizeof(IntType)));
		HANDLE_API_ERR(cudaMalloc((void **)&gF2N, IndexF2N[gnTFace]*sizeof(IntType)));
		HANDLE_API_ERR(cudaMalloc((void **)&gIndexF2N, gnTFace*sizeof(IntType)));

		HANDLE_API_ERR(cudaMemcpy(gnNPF, nNPF, gnTFace*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gF2N, F2N[0], IndexF2N[gnTFace]*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gIndexF2N, IndexF2N, gnTFace*sizeof(IntType), cudaMemcpyHostToDevice));

	}

	void GPUFaceDataTrans3(IntType *N2C, IntType *IndexN2C, IntType *nCPN, RealGeom **WeightNodeBFace2C, 
						RealGeom **WeightNodeN2C, IntType *IndexF2N, IntType *Nmark, RealGeom *WeightNode, 
						IntType *nodesymm, RealGeom *xfn_n_symm, RealGeom *yfn_n_symm, RealGeom *zfn_n_symm,
						RealGeom *norm_dist_c2c){
		// node to cell connection:
		HANDLE_API_ERR(cudaMalloc((void **)&gnCPN, gnTNode*sizeof(IntType)));
		HANDLE_API_ERR(cudaMalloc((void **)&gN2C, IndexN2C[gnTNode]*sizeof(IntType)));
		HANDLE_API_ERR(cudaMalloc((void **)&gIndexN2C, gnTNode*sizeof(IntType)));

		HANDLE_API_ERR(cudaMemcpy(gnCPN, nCPN, gnTNode*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gN2C, N2C, IndexN2C[gnTNode]*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gIndexN2C, IndexN2C, gnTNode*sizeof(IntType), cudaMemcpyHostToDevice));

		// WeightNodeN2C:
		HANDLE_API_ERR(cudaMalloc((void **)&gWeightNodeN2C, IndexN2C[gnTNode]*sizeof(RealGeom)));
		HANDLE_API_ERR(cudaMalloc((void **)&gWeightNodeBFace2C, IndexF2N[gnBFace]*sizeof(RealGeom)));
		
		HANDLE_API_ERR(cudaMemcpy(gWeightNodeN2C, WeightNodeN2C[0], IndexN2C[gnTNode]*sizeof(RealGeom), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gWeightNodeBFace2C, WeightNodeBFace2C[0], IndexF2N[gnBFace]*sizeof(RealGeom), cudaMemcpyHostToDevice));
		
		// Nmark:
		HANDLE_API_ERR(cudaMalloc((void **)&gNmark, gnTNode*sizeof(IntType)));
		HANDLE_API_ERR(cudaMemcpy(gNmark, Nmark, gnTNode*sizeof(IntType), cudaMemcpyHostToDevice));

		// WeightNode:
		HANDLE_API_ERR(cudaMalloc((void **)&gWeightNode, gnTNode*sizeof(RealGeom)));
		HANDLE_API_ERR(cudaMemcpy(gWeightNode, WeightNode, gnTNode*sizeof(RealGeom), cudaMemcpyHostToDevice));

		// nodesymm:
		HANDLE_API_ERR(cudaMalloc((void **)&gnodesymm, gnTNode*sizeof(IntType)));
		HANDLE_API_ERR(cudaMemcpy(gnodesymm, nodesymm, gnTNode*sizeof(IntType), cudaMemcpyHostToDevice));

		// xfn_n_symm:
		HANDLE_API_ERR(cudaMalloc((void **)&gxfn_n_symm, gnTNode*sizeof(RealGeom)));
		HANDLE_API_ERR(cudaMalloc((void **)&gyfn_n_symm, gnTNode*sizeof(RealGeom)));
		HANDLE_API_ERR(cudaMalloc((void **)&gzfn_n_symm, gnTNode*sizeof(RealGeom)));

		HANDLE_API_ERR(cudaMemcpy(gxfn_n_symm, xfn_n_symm, gnTNode*sizeof(RealGeom), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gyfn_n_symm, yfn_n_symm, gnTNode*sizeof(RealGeom), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gzfn_n_symm, zfn_n_symm, gnTNode*sizeof(RealGeom), cudaMemcpyHostToDevice));

		// norm_dist_c2c:
		HANDLE_API_ERR(cudaMalloc((void **)&gnorm_dist_c2c, gnTFace*sizeof(RealGeom)));
		HANDLE_API_ERR(cudaMemcpy(gnorm_dist_c2c, norm_dist_c2c, gnTFace*sizeof(RealGeom), cudaMemcpyHostToDevice));

	}

	void GPUFaceDataTrans4(IntType *luorder, IntType *layer, IntType *cellsPerlayer, IntType **C2C, IntType *IndexC2C, IntType *nCPC, 
						IntType *fcptr, RealGeom *dist2wall, IntType *IsNormalFace, IntType EntropyCorType, IntType *IndexMPIbqs, 
						IntType *IndexMPIbqsSA, IntType *IndexMPIbqsGrad, IntType *IndexMPIbqr, IntType *IndexMPIbqrSA){

		// *gluorder, *glayer, *gcellsPerlayer:
		HANDLE_API_ERR(cudaMalloc((void **)&gluorder, gnTCell*sizeof(IntType)));
		HANDLE_API_ERR(cudaMalloc((void **)&glayer, (gnTCell + gnBFace)*sizeof(IntType)));
		HANDLE_API_ERR(cudaMalloc((void **)&gcellsPerlayer, gnTCell*sizeof(IntType)));
#ifdef CellColoring
		HANDLE_API_ERR(cudaMemcpy(gluorder, luorder, gnTCell*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(glayer, layer, (gnTCell + gnBFace)*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gcellsPerlayer, cellsPerlayer, gnTCell*sizeof(IntType), cudaMemcpyHostToDevice));
#endif
		// cell to cell connect
		HANDLE_API_ERR(cudaMalloc((void **)&gC2C, IndexC2C[gnTCell]*sizeof(IntType)));
		HANDLE_API_ERR(cudaMalloc((void **)&gIndexC2C, gnTCell*sizeof(IntType)));
		HANDLE_API_ERR(cudaMalloc((void **)&gnCPC, gnTCell*sizeof(IntType)));

		// Transfer host data into device **C2C, *IndexC2C:
		HANDLE_API_ERR(cudaMemcpy(gC2C, C2C[0], IndexC2C[gnTCell]*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gIndexC2C, IndexC2C, gnTCell*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gnCPC, nCPC, gnTCell*sizeof(IntType), cudaMemcpyHostToDevice));
		glenC2C = IndexC2C[gnTCell];
		// SA:
		HANDLE_API_ERR(cudaMalloc((void **)&glhsmat, (gnTCell + IndexC2C[gnTCell])*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gfcptr, 2*gnTFace*sizeof(IntType)));
		HANDLE_API_ERR(cudaMemcpy(gfcptr, fcptr, 2*gnTFace*sizeof(IntType), cudaMemcpyHostToDevice));

		// dist2wall:
		HANDLE_API_ERR(cudaMalloc((void **)&gdist2wall, gnTCell*sizeof(RealGeom)));
		HANDLE_API_ERR(cudaMemcpy(gdist2wall, dist2wall, gnTCell*sizeof(RealGeom), cudaMemcpyHostToDevice));

		// IsNormalFace:
		if (EntropyCorType == 4) {
			HANDLE_API_ERR(cudaMalloc((void **)&gIsNormalFace, gnTFace*sizeof(IntType)));
			HANDLE_API_ERR(cudaMemcpy(gIsNormalFace, IsNormalFace, gnTFace*sizeof(IntType), cudaMemcpyHostToDevice));
		}

		
#ifdef MPICH		
		HANDLE_API_ERR(cudaMalloc((void **)&gIndexbqsr, glenbqsr*sizeof(IntType)));
		HANDLE_API_ERR(cudaMemcpy(gIndexbqsr, IndexMPIbqs, glenbqsr*sizeof(IntType), cudaMemcpyHostToDevice));

		HANDLE_API_ERR(cudaMalloc((void **)&gIndexbqsrSA, glenbqsrSA*sizeof(IntType)));
		HANDLE_API_ERR(cudaMemcpy(gIndexbqsrSA, IndexMPIbqsSA, glenbqsrSA*sizeof(IntType), cudaMemcpyHostToDevice));

		HANDLE_API_ERR(cudaMalloc((void **)&gIndexbqsrGrad, glenbqsrGrad*sizeof(IntType)));
		HANDLE_API_ERR(cudaMemcpy(gIndexbqsrGrad, IndexMPIbqsGrad, glenbqsrGrad*sizeof(IntType), cudaMemcpyHostToDevice));	

		HANDLE_API_ERR(cudaMalloc((void **)&gIndexbqsr2, glenbqsr*sizeof(IntType)));
		HANDLE_API_ERR(cudaMemcpy(gIndexbqsr2, IndexMPIbqr, glenbqsr*sizeof(IntType), cudaMemcpyHostToDevice));		
		
		HANDLE_API_ERR(cudaMalloc((void **)&gIndexbqsr2SA, glenbqsrSA*sizeof(IntType)));
		HANDLE_API_ERR(cudaMemcpy(gIndexbqsr2SA, IndexMPIbqrSA, glenbqsrSA*sizeof(IntType), cudaMemcpyHostToDevice));
#endif
	}
#if (defined GroupColor)
	void GPUFaceDataTransGroupColor(IntType length_b_SMc2c, IntType length_i_SMc2c, IntType length_b_f2SMc, IntType length_i_f2SMc,
									IntType num_b_group, IntType num_i_group, 
									IntType *group_b_SMc2c, IntType *group_b_f2SMc, IntType *group_i_SMc2c, IntType *group_i_f2SMc,
									IntType *group_b_SM_index, IntType *group_i_SM_index){
		
		HANDLE_API_ERR(cudaMalloc((void **)&g_b_SMc2c, length_b_SMc2c*sizeof(IntType)));	
		HANDLE_API_ERR(cudaMalloc((void **)&g_i_SMc2c, length_i_SMc2c*sizeof(IntType)));									
		HANDLE_API_ERR(cudaMalloc((void **)&g_b_f2SMc, length_b_f2SMc*sizeof(IntType)));									
		HANDLE_API_ERR(cudaMalloc((void **)&g_i_f2SMc, length_i_f2SMc*sizeof(IntType)));
		HANDLE_API_ERR(cudaMalloc((void **)&g_b_SM_index, (num_b_group + 1)*sizeof(IntType)));	
		HANDLE_API_ERR(cudaMalloc((void **)&g_i_SM_index, (num_i_group + 1)*sizeof(IntType)));	
		
		HANDLE_API_ERR(cudaMemcpy(g_b_SMc2c, group_b_SMc2c, length_b_SMc2c*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(g_i_SMc2c, group_i_SMc2c, length_i_SMc2c*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(g_b_f2SMc, group_b_f2SMc, length_b_f2SMc*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(g_i_f2SMc, group_i_f2SMc, length_i_f2SMc*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(g_b_SM_index, group_b_SM_index, (num_b_group + 1)*sizeof(IntType), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(g_i_SM_index, group_i_SM_index, (num_i_group + 1)*sizeof(IntType), cudaMemcpyHostToDevice));
	}
#endif	
	void GPUFlowCondition(IntType steady, RealFlow p_bar, IntType GaussLayer, RealFlow lhs_omga, RealFlow gam, RealFlow eps_tmp){
		gsteady = steady;
		gp_bar = p_bar;
		gGaussLayer = GaussLayer;
		glhs_omga = lhs_omga;
		ggam = gam;
		geps_tmp = eps_tmp;
	}

	void GPUFlowMemoryAlloc(){
		size_t sizefaceint = gnTFace*sizeof(IntType);
		size_t sizeface = gnTFace*sizeof(RealFlow);
		size_t sizeflux = 5*gnTFace*sizeof(RealFlow);
		size_t sizecell = (gnTCell + gnBFace)*sizeof(RealFlow);		// including ghosts

		// tem cudaMalloc
		// for reduction:
		HANDLE_API_ERR(cudaMalloc((void **)&gtmpvar, 2*sizeface));
		//.................................................................................//
		
		HANDLE_API_ERR(cudaMalloc((void **)&gIsShockFace, sizefaceint));

		HANDLE_API_ERR(cudaMalloc((void **)&gql, sizeflux));	
		HANDLE_API_ERR(cudaMalloc((void **)&gqr, sizeflux));

		HANDLE_API_ERR(cudaMalloc((void **)&gq, sizecell*5));

		HANDLE_API_ERR(cudaMalloc((void **)&gsa_nu, sizecell));

		HANDLE_API_ERR(cudaMalloc((void **)&gflux, sizeflux));

		HANDLE_API_ERR(cudaMalloc((void **)&gres, gnTCell*sizeof(RealFlow)*5));

		// Vis Flux:
		HANDLE_API_ERR(cudaMalloc((void **)&gdeltl, sizeface));
		HANDLE_API_ERR(cudaMalloc((void **)&gdeltr, sizeface));
		//.................................................................................//

		HANDLE_API_ERR(cudaMalloc((void **)&gvis_l, sizecell));
		HANDLE_API_ERR(cudaMalloc((void **)&gvis_t, sizecell));
		HANDLE_API_ERR(cudaMalloc((void **)&gt, sizecell));

		HANDLE_API_ERR(cudaMalloc((void **)&gvisc_f, sizeface));
		HANDLE_API_ERR(cudaMalloc((void **)&gheat_f, sizeface));

		HANDLE_API_ERR(cudaMalloc((void **)&gvel_f, 3*sizeface));
		HANDLE_API_ERR(cudaMalloc((void **)&gt_f, sizeface));

		// gradient:
		HANDLE_API_ERR(cudaMalloc((void **)&gdqdx, 5*sizecell));
		HANDLE_API_ERR(cudaMalloc((void **)&gdqdy, 5*sizecell));
		HANDLE_API_ERR(cudaMalloc((void **)&gdqdz, 5*sizecell));
		// temperature gradient:
		HANDLE_API_ERR(cudaMalloc((void **)&gdtdx, sizecell));
		HANDLE_API_ERR(cudaMalloc((void **)&gdtdy, sizecell));
		HANDLE_API_ERR(cudaMalloc((void **)&gdtdz, sizecell));

		// limit:
		HANDLE_API_ERR(cudaMalloc((void **)&glimit, 5*sizecell));
		HANDLE_API_ERR(cudaMalloc((void **)&gdmax, gnTCell*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gdmin, gnTCell*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gespcell, gnTCell*sizeof(RealFlow)));	
		HANDLE_API_ERR(cudaMalloc((void **)&gtmp_limit, 2*sizeface));

		// Gradient:
		HANDLE_API_ERR(cudaMalloc((void **)&gtmpxyz, 3*sizeface));
		HANDLE_API_ERR(cudaMalloc((void **)&gq_n, gnTNode*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gfacq, gnBFace*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gfacu, gnBFace*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gfacv, gnBFace*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gfacw, gnBFace*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gu_n, gnTNode*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gv_n, gnTNode*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gw_n, gnTNode*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gvn, gnTNode*sizeof(RealFlow)));
		
#ifdef MultiStream
		HANDLE_API_ERR(cudaMalloc((void **)&gmsq_n, 5*gnTNode*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaHostAlloc((void **)&hostmsq_n, 5*gnTNode*sizeof(RealFlow), cudaHostAllocDefault));
		HANDLE_API_ERR(cudaMalloc((void **)&gtmpxyz_u, 3*sizeface));
		HANDLE_API_ERR(cudaMalloc((void **)&gtmpxyz_v, 3*sizeface));
		HANDLE_API_ERR(cudaMalloc((void **)&gtmpxyz_w, 3*sizeface));
		HANDLE_API_ERR(cudaMalloc((void **)&gtmpxyz_p, 3*sizeface));
		HANDLE_API_ERR(cudaMalloc((void **)&gtmpxyz_T, 3*sizeface));
		HANDLE_API_ERR(cudaMalloc((void **)&gtmpxyz_sa, 3*sizeface));
		HANDLE_API_ERR(cudaHostAlloc((void **)&hostu_n, gnTNode*sizeof(RealFlow), cudaHostAllocDefault));
		HANDLE_API_ERR(cudaHostAlloc((void **)&hostv_n, gnTNode*sizeof(RealFlow), cudaHostAllocDefault));
		HANDLE_API_ERR(cudaHostAlloc((void **)&hostw_n, gnTNode*sizeof(RealFlow), cudaHostAllocDefault));
		
		HANDLE_API_ERR(cudaMalloc((void **)&gdmax_MStream, 5*gnTCell*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gdmin_MStream, 5*gnTCell*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gespcell_MStream, 5*gnTCell*sizeof(RealFlow)));	
#endif		
		// MPI:
		HANDLE_API_ERR(cudaHostAlloc((void **)&hostbqs, 3*glenbqsr*sizeof(RealFlow), cudaHostAllocDefault));
		HANDLE_API_ERR(cudaHostAlloc((void **)&hostbqr, 3*glenbqsr*sizeof(RealFlow), cudaHostAllocDefault));
		

		// SA gradient:
		HANDLE_API_ERR(cudaMalloc((void **)&gdnutdx, sizecell));
		HANDLE_API_ERR(cudaMalloc((void **)&gdnutdy, sizecell));
		HANDLE_API_ERR(cudaMalloc((void **)&gdnutdz, sizecell));

		// dt:
		HANDLE_API_ERR(cudaMalloc((void **)&gdt, gnTCell*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gdet, gnTCell*sizeof(IntType)));

		// LUSGS: 
		HANDLE_API_ERR(cudaMalloc((void **)&gDiag, gnTCell*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gDiag_v, gnTCell*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gDQ, 5*sizecell));
		HANDLE_API_ERR(cudaMalloc((void **)&gdqo, 5*sizecell));
		HANDLE_API_ERR(cudaMalloc((void **)&gnorm, gnTCell*sizeof(RealFlow)));
		// RK:
		HANDLE_API_ERR(cudaMalloc((void **)&goldq, 5*gnTCell*sizeof(RealFlow)));	
		
		// SA: 
		HANDLE_API_ERR(cudaMalloc((void **)&gomaga, gnTCell*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&ggradnue2, gnTCell*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gdqdl, gnTFace*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gdqdr, gnTFace*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gtem, gnTFace*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gtem_c2, gnTFace*sizeof(RealFlow)));

		HANDLE_API_ERR(cudaMalloc((void **)&gdkdn, gnTFace*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gdkdnc1, gnTFace*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gdkdnc2, gnTFace*sizeof(RealFlow)));

		// MPI:
		HANDLE_API_ERR(cudaMalloc((void **)&gbqs, 3*glenbqsr*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gbqr, 3*glenbqsr*sizeof(RealFlow)));

		// GMRES:
		HANDLE_API_ERR(cudaMalloc((void **)&greso, 5*gnTCell*sizeof(RealFlow)));
		HANDLE_API_ERR(cudaMalloc((void **)&gdq, 5*gnTCell*sizeof(RealFlow)));
	}

	__global__ void gpuDQInit_target(RealFlow *DQ, IntType n, RealFlow target){
		
		IntType i = blockDim.x*blockIdx.x + threadIdx.x;
		if(i < n){
			DQ[i] = target;
		}
		
	}

	void GPUGMRESMemoryAlloc(IntType kspan, IntType gmres, IntType sweep){
		
		IntType blocksPerGrid;
		if(gmres == 1){
			HANDLE_API_ERR(cudaMalloc((void **)&gv, (kspan + 1)*5*gnTCell*sizeof(RealFlow)));
			HANDLE_API_ERR(cudaMalloc((void **)&gw, 5*gnTCell*sizeof(RealFlow)));
			HANDLE_API_ERR(cudaMalloc((void **)&gDQo, 5*gnTCell*sizeof(RealFlow)));
			HANDLE_API_ERR(cudaMalloc((void **)&gturburhs, gnTCell*sizeof(RealFlow)));
			HANDLE_API_ERR(cudaMalloc((void **)&gdqadu, (gnTCell + gnBFace)*sizeof(RealFlow)));
			//gnsum = (5*gnTCell + 2*threadsPerBlock - 1) / (2*threadsPerBlock);
			//gnodata = gnsum;
			//gnsum *= 2*threadsPerBlock;

			gnsum = (5*gnTCell + threadsPerBlock - 1) / (threadsPerBlock);
			gnodata = gnsum;
			gnsum *= threadsPerBlock;
			HANDLE_API_ERR(cudaMalloc((void **)&gsumv, gnsum*sizeof(RealFlow)));
			blocksPerGrid = (gnsum + threadsPerBlock - 1) / threadsPerBlock;	
			gpuDQInit <<< blocksPerGrid, threadsPerBlock >>> (gsumv, gnsum);
			HANDLE_API_ERR(cudaMalloc((void **)&godata, blocksPerGrid*sizeof(RealFlow)));

			gnsum2= (5*gnTCell + 2*threadsPerBlock - 1) / (2*threadsPerBlock);
			gnodata2 = gnsum2;
			gnsum2 *= 2*threadsPerBlock;			
			HANDLE_API_ERR(cudaMalloc((void **)&gsumv2, gnsum2*sizeof(RealFlow)));
			blocksPerGrid = gnodata2;	
			gpuDQInit <<< blocksPerGrid, threadsPerBlock >>> (gsumv2, gnsum2);
			HANDLE_API_ERR(cudaMalloc((void **)&godata2, blocksPerGrid*sizeof(RealFlow)));
		}
		gSAnsum2= (gnTCell + 2*threadsPerBlock - 1) / (2*threadsPerBlock);
		gSAnodata2 = gSAnsum2;
		gSAnsum2 *= 2*threadsPerBlock;			
		HANDLE_API_ERR(cudaMalloc((void **)&gSAsumv2, gSAnsum2*sizeof(RealFlow)));
		blocksPerGrid = gSAnodata2;	
		gpuDQInit <<< blocksPerGrid, 2*threadsPerBlock >>> (gSAsumv2, gSAnsum2);
		HANDLE_API_ERR(cudaMalloc((void **)&gSAodata2, blocksPerGrid*sizeof(RealFlow)));
		
		gdtmaxnsum2= (gnTCell + 2*threadsPerBlock - 1) / (2*threadsPerBlock);
		gdtmaxnodata2 = gdtmaxnsum2;
		gdtmaxnsum2 *= 2*threadsPerBlock;			
		HANDLE_API_ERR(cudaMalloc((void **)&gdtmaxsumv2, gdtmaxnsum2*sizeof(RealFlow)));
		blocksPerGrid = gdtmaxnodata2;	
		gpuDQInit_target <<< blocksPerGrid, 2*threadsPerBlock >>> (gdtmaxsumv2, gdtmaxnsum2, 0.0);
		HANDLE_API_ERR(cudaMalloc((void **)&gdtmaxodata2, blocksPerGrid*sizeof(RealFlow)));
		
		gdtminnsum2= (gnTCell + 2*threadsPerBlock - 1) / (2*threadsPerBlock);
		gdtminnodata2 = gdtminnsum2;
		gdtminnsum2 *= 2*threadsPerBlock;			
		HANDLE_API_ERR(cudaMalloc((void **)&gdtminsumv2, gdtminnsum2*sizeof(RealFlow)));
		blocksPerGrid = gdtminnodata2;	
		gpuDQInit_target <<< blocksPerGrid, 2*threadsPerBlock >>> (gdtminsumv2, gdtminnsum2, (RealFlow)BIG);
		HANDLE_API_ERR(cudaMalloc((void **)&gdtminodata2, blocksPerGrid*sizeof(RealFlow)));
		
		// Unified Memory for Reduction:
		HANDLE_API_ERR(cudaMallocManaged((void **)&val_Reduction, 10*sizeof(RealFlow)));
	}

	void GPUFlowMemoryInit(RealFlow *q[5], RealFlow *sa_nu, RealFlow *vis_l, RealFlow *vis_t){
		HANDLE_API_ERR(cudaMemcpy(gq, q[0], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(&gq[(gnTCell + gnBFace)], q[1], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(&gq[2*(gnTCell + gnBFace)], q[2], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(&gq[3*(gnTCell + gnBFace)], q[3], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(&gq[4*(gnTCell + gnBFace)], q[4], (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));	
		HANDLE_API_ERR(cudaMemcpy(gsa_nu, sa_nu, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));

		HANDLE_API_ERR(cudaMemcpy(gvis_l, vis_l, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
		HANDLE_API_ERR(cudaMemcpy(gvis_t, vis_t, (gnTCell + gnBFace)*sizeof(RealFlow), cudaMemcpyHostToDevice));
		
		for (IntType i = 0; i < 5; i++){
			cudaStreamCreate(&flowstream[i]);
		}
	}
#if defined MultiStream
void GPUGrad_Limit_Init(){
		
		IntType blocksPerGrid = (gnTNode + threadsPerBlock - 1) / threadsPerBlock;
		gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gu_n, gnTNode);
		gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gv_n, gnTNode);
		gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gw_n, gnTNode);
		
		blocksPerGrid = (5*(gnTCell + gnBFace) + threadsPerBlock - 1) / threadsPerBlock;
		gpuGradientInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdqdx, gdqdy, gdqdz, 5*(gnTCell + gnBFace));
		
		blocksPerGrid = ((gnTCell + gnBFace) + threadsPerBlock - 1) / threadsPerBlock;
		gpuGradientInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gdtdx, gdtdy, gdtdz, (gnTCell + gnBFace));
		
		blocksPerGrid = (5*gnTNode + threadsPerBlock - 1) / threadsPerBlock;
		gpuCompNodeInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (gmsq_n, 5*gnTNode);		
		
		blocksPerGrid = (gnTCell + gnBFace + threadsPerBlock - 1) / threadsPerBlock;
		gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (glimit, gnTCell, gnBFace, 0);
		gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (glimit, gnTCell, gnBFace, 1);
		gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (glimit, gnTCell, gnBFace, 2);
		gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (glimit, gnTCell, gnBFace, 3);
		gpuLimitInit <<< blocksPerGrid, threadsPerBlock, 0, flowstream[0] >>> (glimit, gnTCell, gnBFace, 4);
		
		cudaStreamSynchronize(flowstream[0]);
	}
#endif	
}

void NSSolver::QuantityGradient_Init(PolyGrid* grid) {
	if (grad_method_) {
		AllocateQuantityGradientMemory(grid);
		CalculateQuantityGradient(grid);
	}
}
