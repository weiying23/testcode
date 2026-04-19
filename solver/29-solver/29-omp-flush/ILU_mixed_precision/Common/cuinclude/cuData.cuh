#include <stdio.h>
#include <iostream>

#include <number_type.h>
#include <solver_ns.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mflow;

namespace gpuData{

	extern cudaStream_t flowstream[5];

	// GPU Device Parameter:
	extern IntType   threadsPerBlock;			//thread size per block 
	
	// Grid Info:
	extern IntType   gnTFace, gnTCell; 			// no. of total faces, no of total cells
    extern IntType   gnBFace;         			// boundary faces which include interfaces
    extern IntType   gnIFace;         			// parallel interface faces(zero if serial program)
    extern IntType   gnTNode;					// no. of total nodes
    extern IntType   glenC2C;
	
	extern IntType  *gf2c;						// face->cell connectivity
	extern RealGeom *gxcc, *gycc, *gzcc;  		// cell center data, including ghosts
    extern RealGeom *gxfc, *gyfc, *gzfc;  		// face center data;
	extern RealGeom *gxfn, *gyfn, *gzfn;		// face unit normal;
	extern RealGeom *garea, *gvol;       		// face area, cell volume excluding ghosts;
	extern RealGeom *gvgn;  					// normal velocity for all faces
	extern IntType  *gtype_bcr;					// associate each boundary. face with a record
	extern RealGeom *gtw_bcr;  					// viscous iso-thermal wall
	extern IntType  *gC2F, *gIndexC2F, *gnFPC;	// cell to face connect
	extern RealGeom *gfacecentroidskewness;		//
	extern RealFlow *gangle_h;		// for Vis asin() function on GPU
	extern IntType  *gnNPF, *gF2N, *gIndexF2N;	// face->node connectivity
	extern IntType  *gN2C, *gIndexN2C, *gnCPN;	// node to cell connection
	extern RealGeom *gWeightNodeN2C, *gWeightNodeBFace2C;
	extern IntType  *gNmark;
	extern RealGeom *gWeightNode;
	extern IntType  *gnodesymm;
	extern RealGeom *gxfn_n_symm, *gyfn_n_symm, *gzfn_n_symm;  
	extern RealGeom *gnorm_dist_c2c; 
	extern IntType 	*gfcptr;

	extern IntType  *gcellwallnumber, *gCellLayerNo;

	extern IntType  *glayer, *gluorder, *gcellsPerlayer;	// LUSGS layers
	extern IntType  *gC2C, *gIndexC2C, *gnCPC;	// cell to cell connect
	extern RealGeom *gdist2wall;
	
#if (defined GroupColor)
	// to load cell values into share memory:
	extern IntType *g_b_SMc2c;
	extern IntType *g_i_SMc2c;
	// to store the share memory unit index:
	extern IntType *g_b_SM_index;
	extern IntType *g_i_SM_index;
	// to store the share memory index between colors:
	//IntType *g_b_SM_color_index;
	//IntType *g_i_SM_color_index;
	// to load face values into share memory cell values:
	extern IntType *g_b_f2SMc;
	extern IntType *g_i_f2SMc;
#endif	

	// Flow Info:
  	extern IntType	 gsteady;					//steady(1) or unsteady(0) flow
  	extern RealFlow  gp_bar;
  	extern IntType   gGaussLayer;				// for Gasuu Gradient Comput.
  	extern RealFlow  glhs_omga;
  	extern RealFlow  ggam, geps_tmp;

  	// Flow Memory Alloc:
  	// for reduction.
  	extern RealFlow *gtmpvar;
  	// Flux Comput.
  	extern IntType  *gIsNormalFace, *gIsShockFace;
  	extern RealFlow *gql, *gqr, *gq;
  	extern RealFlow *gsa_nu;
  	extern RealFlow *gflux;
  	extern RealFlow *gdqdx, *gdqdy, *gdqdz;
  	extern RealFlow *gdtdx, *gdtdy, *gdtdz;
  	extern RealFlow *glimit;
  	extern RealFlow *gres;
  	extern RealFlow *gvis_l, *gvis_t, *gvisc_f, *gheat_f;
  	extern RealFlow *gvel, *gvel_f, *gt, *gt_f;
  	extern RealGeom *gdeltl, *gdeltr;
  	extern RealFlow *gdt;
	// Limiter Comput.
  	extern RealFlow *gdmax, *gdmin, *gespcell, *gdmax_MStream, *gdmin_MStream, *gespcell_MStream;
  	extern RealFlow *gtmp_limit;
  	// Gradient Comput.
  	extern RealFlow *gtmpxyz;
  	extern RealFlow *gq_n, *gfacq;
  	extern RealFlow *gdnutdx, *gdnutdy, *gdnutdz;
  	extern RealFlow *gfacu, *gfacv, *gfacw;
  	extern RealFlow *gu_n, *gv_n, *gw_n, *gvn;
#ifdef MultiStream
	extern RealFlow *gmsq_n;
	extern RealFlow *hostmsq_n;
	extern RealFlow *gtmpxyz_u, *gtmpxyz_v, *gtmpxyz_w, *gtmpxyz_p, *gtmpxyz_T, *gtmpxyz_sa;
	extern RealFlow *hostu_n, *hostv_n, *hostw_n;
#endif	
  	// LUSGS Comput.
  	extern RealFlow *gDiag, *gDiag_v;
  	extern RealFlow *gDQ;
	extern RealFlow *gdqo, *gnorm;
	// RK COmput.
	extern RealFlow *goldq;
  	// SA Comput.
  	extern RealFlow *glhsmat;
  	extern RealFlow *gomaga;
  	extern RealFlow *ggradnue2;
  	extern RealFlow *gdqdl, *gdqdr;
  	extern RealFlow *gtem, *gtem_c2;
  	extern RealFlow *gdkdn, *gdkdnc1, *gdkdnc2;
  	// TimeStep: dt
  	extern IntType  *gdet;
  	// MPI:
	extern RealFlow *gMPI;
	extern RealFlow *gbqs, *gbqr;				// the length of bqs/bqr selects the maximum length, that is Grad bqs/bqr, 3*5*(nTCell+nBFace)
	extern RealFlow *hostbqs, *hostbqr;			// the length of bqs/bqr selects the maximum length, that is Grad bqs/bqr, 3*5*(nTCell+nBFace)

	extern IntType  *gIndexbqsr;				// the Index of bqs/bqr for q, dq, limit, of which the length equals to 5*(nTCell+nBFace)
	extern IntType  *gIndexbqsrSA;				// the length of bqs/bqr for sa_nu, vis_l, vis_t, dt, of which the length equals to nTCell+nBFace
	extern IntType  *gIndexbqsrGrad;			// the length of bqs/bqr for dqdx, dqdy, dqdz, of which the length equals to 3*5*(nTCell+nBFace)
	extern IntType  *gIndexbqsr2;				// the Index of bqs/bqr for q, dq, limit, of which the length equals to 5*(nTCell+nBFace)
	extern IntType  *gIndexbqsrGradSA;			// the length of bqs/bqr for dqdx, dqdy, dqdz, of which the length equals to 3*(nTCell+nBFace)
	extern IntType  *gIndexbqsr2SA;				// the length of bqs/bqr for sa_nu, vis_l, vis_t, dt, of which the length equals to nTCell+nBFace
	extern IntType  *gIndexbqsr2GradSA;			// the length of bqs/bqr for dqdx, dqdy, dqdz, of which the length equals to 3*(nTCell+nBFace)
	
	extern IntType   glenbqsr;					// the length of bqs/bqr for q, dq, limit, of which the length equals to 5*(nTCell+nBFace)
    extern IntType   glenbqsrSA;				// the length of bqs/bqr for sa_nu, vis_l, vis_t, dt, of which the length equals to nTCell+nBFace
	extern IntType   glenbqsrGrad;				// the length of bqs/bqr for dqdx, dqdy, dqdz, of which the length equals to 3*5*(nTCell+nBFace)
	extern IntType   glenbqsrGradSA;			// the length of bqs/bqr for dqdx, dqdy, dqdz, of which the length equals to 3*(nTCell+nBFace)
	
	extern IntType   glenbqsr_Node;
	
	// GMRES:
	extern RealFlow *greso, *gdq;
	extern RealFlow *gDQo, *gv, *gw;
	extern RealFlow *gturburhs, *gdqadu;
	extern RealFlow *gsumv, *godata;
	extern IntType   gnsum, gnodata;
	extern RealFlow *gsumv2, *godata2;
	extern IntType   gnsum2, gnodata2;
	extern RealFlow *gSAsumv2, *gSAodata2;
	extern IntType   gSAnsum2, gSAnodata2;
	
	// dt max and min:
	extern RealFlow *gdtmaxsumv2, *gdtmaxodata2;
	extern IntType   gdtmaxnsum2, gdtmaxnodata2;
	extern RealFlow *gdtminsumv2, *gdtminodata2;
	extern IntType   gdtminnsum2, gdtminnodata2;
	
	// Unified Memory for Reduction:
	extern RealFlow *val_Reduction;
	
	void GPUFaceNumberTrans(IntType nTFace, IntType nTCell, IntType nBFace, IntType nIFace, IntType nTNode);
	void GPUFaceDataTrans(RealGeom *xfc, RealGeom *yfc, RealGeom *zfc, RealGeom *xfn, RealGeom *yfn, RealGeom *zfn, 
						RealGeom *xcc, RealGeom *ycc, RealGeom *zcc, RealGeom *area, RealGeom *vgn, IntType *f2c,
						IntType *type_bcr, RealGeom *tw_bcr);
	void GPUFaceDataTrans2(IntType **C2F, IntType *IndexC2F, IntType *nFPC, RealGeom *vol, RealGeom *facecentroidskewness,
						RealFlow *angle_h, IntType *nNPF, IntType **F2N, IntType *IndexF2N,
						IntType *cellwallnumber, IntType *CellLayerNo);
	void GPUFaceDataTrans3(IntType *N2C, IntType *IndexN2C, IntType *nCPN, RealGeom **WeightNodeBFace2C, 
						RealGeom **WeightNodeN2C, IntType *IndexF2N, IntType *Nmark, RealGeom *WeightNode, 
						IntType *nodesymm, RealGeom *xfn_n_symm, RealGeom *yfn_n_symm, RealGeom *zfn_n_symm,
						RealGeom *norm_dist_c2c);
	void GPUFaceDataTrans4(IntType *luorder, IntType *layer, IntType *cellsPerlayer, IntType **C2C, IntType *IndexC2C, IntType *nCPC, 
						IntType *fcptr, RealGeom *dist2wall, IntType *IsNormalFace, IntType EntropyCorType, IntType *IndexMPIbqs, 
						IntType *IndexMPIbqsSA, IntType *IndexMPIbqsGrad, IntType *IndexMPIbqr, IntType *IndexMPIbqrSA);
#if (defined GroupColor)
	void GPUFaceDataTransGroupColor(IntType length_b_SMc2c, IntType length_i_SMc2c, IntType length_b_f2SMc, IntType length_i_f2SMc,
									IntType num_b_group, IntType num_i_group, 
									IntType *group_b_SMc2c, IntType *group_b_f2SMc, IntType *group_i_SMc2c, IntType *group_i_f2SMc,
									IntType *group_b_SM_index, IntType *group_i_SM_index);
#endif	
	void GPUFlowCondition(IntType steady, RealFlow p_bar, IntType GaussLayer, RealFlow lhs_omga, RealFlow gam, RealFlow eps_tmp);
	void GPUFlowMemoryAlloc();
	void GPUGMRESMemoryAlloc(IntType kspan, IntType gmres, IntType sweep);
	void GPUFlowMemoryInit(RealFlow *q[5], RealFlow *sa_nu, RealFlow *vis_l, RealFlow *vis_t);

#if defined MultiStream	
	void GPUGrad_Limit_Init();
#endif
}