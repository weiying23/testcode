//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   parallel_mpi.cpp
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

// user defined head files
#include "grid_polyhedra.h"
#include "io_log.h"
#include "utility_functions.h"
#include "system_base_functions.h"
#include "grid_patch_type.h"

// build-in head files
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <limits>
using namespace std;

// head files relying on condition-compiling
#ifdef MPICH

#include <mpi.h>

namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 11603  // define file id

#ifdef MPICH
    extern int myZone;
    extern int numprocs;
    extern MPI_Comm GridComm;  //for each grid, tangj
#endif

static RealFlow  t1=0, t2=0;
RealFlow mpi_time=0., send_time=0., recv_time=0.;


/********************************************\
   �����߽���Ĵ�ֵ����
\********************************************/
void PolyGrid::SetUpComm()
{
    IntType n,i,ni,g,np;
    BCRecord **bcr=Getbcr();
    IntType nbZone;
    MPI_Status status;
    MPI_Request *req_send,*req_recv;
    MPI_Status *status_array ;

    IntType nNeighbor = this->GetNumberOfFaceNeighbors();

    if(nNeighbor == 0) return;
    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    nZIFace      = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv, nNeighbor,dmrfl);
    mfmem::snew_array_1D(nZIFace, nNeighbor,dmrfl);
   
    bCNo = NULL;
    mfmem::snew_array_1D(bCNo, nNeighbor,dmrfl);
    IntType **bqs, **bqr;
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr, nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs, nNeighbor,dmrfl);
    for(g=0; g<nNeighbor; g++) {
        nbZone = nb[g];

        //�����������ڵĸ���ֱ��ж��ٸ����б߽���
        n=0;
        ni=0;
        for(i=0; i<nBFace; i++) {
            if(bcr[i]->GetType() == INTERFACE) {
                if(nbZ[ni] == nbZone) {
                    n++;
                }
                ni++;
            }
        }
        nZIFace[g] = n;
        //����ǰ���е���Ҫ���ݵ�����Ŵ�����bCNo[g][n]��:g������Ҫ�����ڼ���,n��ʾ���к�
        bCNo[g] = NULL;
        bqs[g]  = NULL;
        mfmem::snew_array_1D(bCNo[g], n,dmrfl);
        mfmem::snew_array_1D(bqs[g], n,dmrfl);
        n=0;
        ni=0;
        for(i=0; i<nBFace; i++) {
            if(bcr[i]->GetType() == INTERFACE) {
                if(nbZ[ni] == nbZone) {
                    bCNo[g][n] = f2c[i*2];
                    bqs[g][n++] = nbBF[ni];
                }
                ni++;
            }
        }
    }

    // now receiving
    for(g=0; g<nNeighbor; g++) {
        nbZone = nb[g];
        MPI_Send(&nZIFace[g], 1, MPIIntType, nbZone, level, MPI_COMM_WORLD);
    }
    for(g=0; g<nNeighbor; g++) {
        nbZone = nb[g];
        MPI_Recv(&n, 1, MPIIntType, nbZone, level, MPI_COMM_WORLD, &status);
        assert(n == nZIFace[g]);
        }

    for(g=0; g<nNeighbor; g++) {
        bqr[g]= NULL;
        mfmem::snew_array_1D( bqr[g], nZIFace[g],dmrfl);
    }

  for(np=1;np<=numprocs;np++){
     if(myZone==np)
     {   
//       send   to other    
       for(g=0; g<nNeighbor; g++) {
           nbZone = nb[g];
           MPI_Isend(bqs[g], nZIFace[g], MPIIntType, nbZone, level, MPI_COMM_WORLD,&req_send[g]);
       }
     } else{   
//       receive   from np  
       for(g=0; g<nNeighbor; g++) {
         nbZone = nb[g];
         if(nbZone ==(np-1)){
           MPI_Irecv(bqr[g], nZIFace[g], MPIIntType, nbZone, level, MPI_COMM_WORLD, &req_recv[g]);
         }
       } 
     } 
  }   // np
     
   MPI_Waitall(nNeighbor,req_recv,status_array);
   bFNo = NULL;
   mfmem::snew_array_2D(bFNo, nNeighbor, nZIFace, dmrfl,false);
   for(g=0; g<nNeighbor; g++) {
        for(i=0; i<nZIFace[g]; i++) {
           bFNo[g][i] = bqr[g][i];
        }
    }
   MPI_Waitall(nNeighbor,req_send,status_array);


  for(g=0; g<nNeighbor; g++) {
     mfmem::sdel_array_1D(bqr[g]);
     mfmem::sdel_array_1D(bqs[g]);
   }
  mfmem::sdel_array_1D(bqr);
  mfmem::sdel_array_1D(bqs);
  mfmem::sdel_array_1D(req_send);
  mfmem::sdel_array_1D(req_recv);
  mfmem::sdel_array_1D(status_array);


    //��Ⲣ�з����Ƿ���ڷ�һһ��Ӧ������
    IntType *count = NULL;
    mfmem::snew_array_1D(count, nBFace,dmrfl);
    IntType mark = 1;
    for(i=0; i<nBFace; i++) count[i]=0;
    for(g=0; g<nNeighbor; g++) {
        for(i=0; i<nZIFace[g]; i++) {
            count[bFNo[g][i]]++;
        }
    }
    mflog::log.set_all_processors_out();
    for(i=0; i<nBFace; i++) {
        if(count[i]>1){
            mflog::log << "myZone = " << myZone << ", and the face number is" << i << endl;
            mflog::log << "The count is " << count[i] << endl;
            mark = 0;
        }
    }
    mfmem::sdel_array_1D(count);
    if(!mark) mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
}


/********************************************\
   �����߽��Ĵ�ֵ����
\********************************************/
void PolyGrid::SetUpComm_Node()
{
    IntType n,i,g, np;
    IntType nbZone;
    MPI_Status  status;
    MPI_Request *req_send,*req_recv;
    MPI_Status *status_array ;

    IntType nNeighborN = this->GetNumberOfNodeNeighbors();

    if(nNeighborN == 0) return;
    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    nZINode      = NULL;
    bNSNo        = NULL;
    mfmem::snew_array_1D(status_array, nNeighborN,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighborN,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighborN,dmrfl);
    mfmem::snew_array_1D(nZINode,      nNeighborN,dmrfl);
    mfmem::snew_array_1D(bNSNo,        nNeighborN,dmrfl);
    
    IntType **bqr = NULL;
    IntType **bqs = NULL;
    mfmem::snew_array_1D(bqr, nNeighborN,dmrfl);
    mfmem::snew_array_1D(bqs, nNeighborN,dmrfl);
    for(g=0; g<nNeighborN; g++) {
        nbZone = nbN[g];

        //�����������ڵĸ���ֱ��ж��ٸ����бߵ�
        n=0;
        for(i=0; i<nINode; i++) {
            if(nbZN[i] == nbZone) {
                n++;
            }
        }
        nZINode[g] = n;

        //����ǰ���е���Ҫ���ݵĵ�Ŵ�����bNSNo[g][n]��:g������Ҫ�����ڼ���,n��ʾ���к�
        bNSNo[g] = NULL;
        bqs[g]   = NULL;
        mfmem::snew_array_1D(bNSNo[g],n,dmrfl);
        mfmem::snew_array_1D(bqs[g],n,dmrfl);
        n=0;
        for(i=0; i<nINode; i++) {
            if(nbZN[i] == nbZone) {
                    bNSNo[g][n] = nbSN[i];
                    bqs[g][n++]  = nbRN[i];
            }
        }
    }

    // now receiving
    for(g=0; g<nNeighborN; g++) {
        nbZone = nbN[g];
        MPI_Send(&nZINode[g], 1, MPIIntType, nbZone, level, MPI_COMM_WORLD);
    }
    for(g=0; g<nNeighborN; g++) {
        nbZone = nbN[g];
        MPI_Recv(&n, 1, MPIIntType, nbZone, level, MPI_COMM_WORLD,&status);
        assert(n == nZINode[g]);
    }

    for(g=0; g<nNeighborN; g++) {
        bqr[g] = NULL;
        mfmem::snew_array_1D(bqr[g],nZINode[g],dmrfl);
    }
    for(np=1;np<=numprocs;np++){
        if(myZone==np)
        {   
            //       send   to other    
            for(g=0; g<nNeighborN; g++) {
                nbZone = nbN[g];
                MPI_Isend(bqs[g], nZINode[g], MPIIntType, nbZone, level, MPI_COMM_WORLD,&req_send[g]);
            }
        } else{   
            //       receive   from np  
            for(g=0; g<nNeighborN; g++) {
                nbZone = nbN[g];
                if(nbZone ==(np-1)){
                    MPI_Irecv(bqr[g], nZINode[g], MPIIntType, nbZone, level, MPI_COMM_WORLD, &req_recv[g]);
                }
            } 
        } 
    }   // np

    MPI_Waitall(nNeighborN,req_recv,status_array);
    bNRNo = NULL;
    mfmem::snew_array_1D(bNRNo, nNeighborN,dmrfl);
    for(g=0; g<nNeighborN; g++) {
        bNRNo[g] = NULL;
        mfmem::snew_array_1D(bNRNo[g], nZINode[g],dmrfl);
        for(i=0; i<nZINode[g]; i++) {
            bNRNo[g][i] = bqr[g][i];
        }
    }
    MPI_Waitall(nNeighborN,req_send,status_array);

    for(g=0; g<nNeighborN; g++) {
        mfmem::sdel_array_1D(bqr[g]);
        mfmem::sdel_array_1D(bqs[g]);
    }
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs);
    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
}


void PolyGrid::CommInterfaceDataMPI(IntType *rho)
{
    if(nNeighbor == 0) return;
    if(rho) RecvSendVarNeighbor(rho);
}


void PolyGrid::CommInterfaceDataMPI(RealFlow *rho)
{
    if(nNeighbor == 0) return;
    if(rho) RecvSendVarNeighbor(rho);
}


void PolyGrid::CommInternodeDataMPI(IntType *rho)
{
    if(nNeighborN == 0) return;
    if(rho) RecvSendVarNeighbor_Node(rho);
}


void PolyGrid::CommInternodeDataMPI2(IntType *rho)
{
    if(nNeighborN == 0) return;
    if(rho) RecvSendVarNeighbor_Node2(rho);
}


void PolyGrid::CommInternodeDataMPISUM(IntType *rho)
{
    if(nNeighborN == 0) return;
    if(rho) RecvSendVarNeighbor_NodeSUM(rho);
}


void PolyGrid::CommInternodeDataMPI(RealFlow *rho)
{
    if(nNeighborN == 0) return;
    if(rho) RecvSendVarNeighbor_Node(rho);
}


/****************************************************************************\
  zhyb: ���ݵ�Ĳ���ֵ��key=1��ȡ���ֵ,  key=2��ȡ��Сֵ��
                     key=-1��ȡ���ֵ, key=-2��ȡ��Сֵ���������ֵ��Сֵ������������ֵ
\****************************************************************************/
void PolyGrid::CommInternodeDataMPI(RealFlow *rho, IntType key)
{
    if(nNeighborN == 0) return;
    if(rho) RecvSendVarNeighbor_Node(rho, key);
}


// ���������ͣ�����������
void PolyGrid::RecvSendVarNeighbor(RealFlow *q)
{
    IntType i, j,  nbZone, ghost;
    IntType np ;
    RealFlow **bqs;
    RealFlow **bqr;

    MPI_Request *req_send,*req_recv;
    MPI_Status *status_array ;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_2D(bqr,nNeighbor,nZIFace,dmrfl,false);
    mfmem::snew_array_2D(bqs,nNeighbor,nZIFace,dmrfl,false);

    t1 = MPI_Wtime();
 
    for(np=1;np<=numprocs;np++)
    {
        if(myZone==np)
        {   
            // send to other    
            for(i=0; i<nNeighbor; i++) {
                nbZone = nb[i];
                for(j=0; j<nZIFace[i]; j++) {
                    bqs[i][j] = q[bCNo[i][j]];
                }
                MPI_Isend(bqs[i], nZIFace[i], MPIReal, nbZone, level, MPI_COMM_WORLD,&req_send[i]);
            }
        } else{   
            // receive from np  
            for(i=0; i<nNeighbor; i++) {
                nbZone = nb[i];
                if(nbZone ==(np-1)){
                    MPI_Irecv(bqr[i], nZIFace[i], MPIReal, nbZone, level, MPI_COMM_WORLD, &req_recv[i]);
                }
            } 
        } 
    }   // np
     
    MPI_Waitall(nNeighbor,req_recv,status_array);
    for(i=0; i<nNeighbor; i++) {
        for(j=0; j<nZIFace[i]; j++) {
            ghost    = nTCell + bFNo[i][j];
            q[ghost] = bqr[i][j];
        }
    } 
    MPI_Waitall(nNeighbor,req_send,status_array);

    t2         = MPI_Wtime();
    mpi_time  += t2 - t1;
    mfmem::sdel_array_2D(bqr,nNeighbor,false);
    mfmem::sdel_array_2D(bqs,nNeighbor,false);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(status_array);
}


void PolyGrid::RecvSendVarNeighbor(IntType *q)
{
    IntType i, j,  nbZone, ghost;
    IntType np ;
    IntType **bqs;
    IntType **bqr;

    MPI_Request *req_send,*req_recv;
    MPI_Status *status_array ;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_2D(bqr,nNeighbor,nZIFace,dmrfl,false);
    mfmem::snew_array_2D(bqs,nNeighbor,nZIFace,dmrfl,false);

    t1 = MPI_Wtime();
 
    for(np=1;np<=numprocs;np++)
    {
        if(myZone==np)
        {   
            // send to other    
            for(i=0; i<nNeighbor; i++) {
                nbZone = nb[i];
                for(j=0; j<nZIFace[i]; j++) {
                    bqs[i][j] = q[bCNo[i][j]];
                }
                MPI_Isend(bqs[i], nZIFace[i], MPIIntType, nbZone, level, MPI_COMM_WORLD,&req_send[i]);
            }
        } 
        else
        {   
            // receive from np  
            for(i=0; i<nNeighbor; i++) {
                nbZone = nb[i];
                if(nbZone ==(np-1)){
                    MPI_Irecv(bqr[i], nZIFace[i], MPIIntType, nbZone, level, MPI_COMM_WORLD, &req_recv[i]);
                }
            } 
        } 
    }   // np
     
    MPI_Waitall(nNeighbor,req_recv,status_array);
    for(i=0; i<nNeighbor; i++) {
        for(j=0; j<nZIFace[i]; j++) {
            ghost    = nTCell + bFNo[i][j];
            q[ghost] = bqr[i][j];
        }
    } 
    MPI_Waitall(nNeighbor,req_send,status_array);

    t2         = MPI_Wtime();
    mpi_time  += t2 - t1;

    mfmem::sdel_array_2D(bqr,nNeighbor,false);
    mfmem::sdel_array_2D(bqs,nNeighbor,false);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(status_array);
}


// test6 ���������ͣ�����������
void PolyGrid::RecvSendVarNeighbor_Node(RealFlow *q)
{
    IntType i, j, nbZone;
    IntType np ;
    RealFlow  **bqs;
    RealFlow  **bqr;

    MPI_Request *req_send,*req_recv;
    MPI_Status *status_array ;
    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighborN,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighborN,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighborN,dmrfl);
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_2D(bqr,nNeighborN,nZINode,dmrfl,false);
    mfmem::snew_array_2D(bqs,nNeighborN,nZINode,dmrfl,false);
    t1 = MPI_Wtime();
 
    for(np=1;np<=numprocs;np++)
    {
        if(myZone==np){
            // send to other   
            for(i=0; i<nNeighborN; i++) {
                nbZone = nbN[i];
                for(j=0; j<nZINode[i]; j++) {
                    bqs[i][j] = q[bNSNo[i][j]];
                }
                MPI_Isend(bqs[i], nZINode[i], MPIReal, nbZone, level, MPI_COMM_WORLD,&req_send[i]);
            }
        } else{
            //  receive from np     
            for(i=0; i<nNeighborN; i++) {
                nbZone = nbN[i];
                if(nbZone ==(np-1)){
                    MPI_Irecv(bqr[i], nZINode[i], MPIReal, nbZone, level, MPI_COMM_WORLD, &req_recv[i]);
                }
            } 
        } 
    }   // np

    MPI_Waitall(nNeighborN,req_recv,status_array);
    for(i=0; i<nNeighborN; i++) {
        for(j=0; j<nZINode[i]; j++) {
            q[bNRNo[i][j]] += bqr[i][j];
        }
    } 
    MPI_Waitall(nNeighborN,req_send,status_array);

    t2         = MPI_Wtime();
    mpi_time  += t2 - t1;

    mfmem::sdel_array_2D(bqr,nNeighborN,false);
    mfmem::sdel_array_2D(bqs,nNeighborN,false);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(status_array);
}


/*****************************************************************************
  zhyb: ���ݵ�Ĳ���ֵ��key=1��ȡ���ֵ,  key=2��ȡ��Сֵ��
                     key=-1��ȡ���ֵ, key=-2��ȡ��Сֵ���������ֵ��Сֵ������������ֵ
******************************************************************************/
void PolyGrid::RecvSendVarNeighbor_Node(RealFlow *q, IntType key)
{
    IntType i, j, nbZone;
    IntType np;
    RealFlow **bqs;
    RealFlow **bqr;

    MPI_Request *req_send,*req_recv;
    MPI_Status *status_array ;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighborN,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighborN,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighborN,dmrfl);

    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_2D(bqr,nNeighborN,nZINode,dmrfl,false);
    mfmem::snew_array_2D(bqs,nNeighborN,nZINode,dmrfl,false);
    t1 = MPI_Wtime();

    for(np=1;np<=numprocs;np++)
    {
        if(myZone==np){
            //send to other   
            for(i=0; i<nNeighborN; i++) {
                nbZone = nbN[i];
                for(j=0; j<nZINode[i]; j++) {
                    bqs[i][j] = q[bNSNo[i][j]];
                }
                MPI_Isend(bqs[i], nZINode[i], MPIReal, nbZone, level, MPI_COMM_WORLD,&req_send[i]);
            }
        } else{
            //receive from np     
            for(i=0; i<nNeighborN; i++) {
                nbZone = nbN[i];
                if(nbZone ==(np-1)){
                    MPI_Irecv(bqr[i], nZINode[i], MPIReal, nbZone, level, MPI_COMM_WORLD, &req_recv[i]);
                }
            } 
        } 
    }   // np

    MPI_Waitall(nNeighborN,req_recv,status_array);
    for(i=0; i<nNeighborN; i++){
        for(j=0; j<nZINode[i]; j++){
            //zhyb:���¸���ֵ�����ڲ���������ֵ�����ֵ��Сֵ��ȡ
            if(key == -1){
                q[bNRNo[i][j]] = -BIG;
            }else if(key == -2){
                q[bNRNo[i][j]] = BIG;
            }
        }
    }
    for(i=0; i<nNeighborN; i++){
        for(j=0; j<nZINode[i]; j++){
            if(key==1 || key==-1){ //�����ֵ
                q[bNRNo[i][j]] = MAX(bqr[i][j], q[bNRNo[i][j]]);
            }else if(key==2 || key==-2){  //����Сֵ
                q[bNRNo[i][j]] = MIN(bqr[i][j], q[bNRNo[i][j]]);
            }else{
                mflog::log.set_all_processors_out();
                mflog::log<<endl<<"Error in function RecvSendVarNeighbor_Node(RealFlow *q, IntType key), key=0!"<<endl;
            }
        }
    } 

    MPI_Waitall(nNeighborN,req_send,status_array);

    t2         = MPI_Wtime();
    mpi_time  += t2 - t1;

    mfmem::sdel_array_2D(bqr,nNeighborN,false);
    mfmem::sdel_array_2D(bqs,nNeighborN,false);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(status_array);
}


void PolyGrid::RecvSendVarNeighbor_Node(IntType *q)
{
    IntType i, j, nbZone;
    IntType np ;
    IntType **bqs;
    IntType **bqr;

    MPI_Request *req_send,*req_recv;
    MPI_Status *status_array ;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighborN,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighborN,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighborN,dmrfl);
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_2D(bqr,nNeighborN,nZINode,dmrfl,false);
    mfmem::snew_array_2D(bqs,nNeighborN,nZINode,dmrfl,false);
    t1 = MPI_Wtime();
 
    for(np=1;np<=numprocs;np++)
    {
        if(myZone==np){
            // send to other   
            for(i=0; i<nNeighborN; i++) {
                nbZone = nbN[i];
                for(j=0; j<nZINode[i]; j++) {
                    bqs[i][j] = q[bNSNo[i][j]];
                }
                MPI_Isend(bqs[i], nZINode[i], MPIIntType, nbZone, level, MPI_COMM_WORLD,&req_send[i]);
            }
        } else{
            // receive from np     
            for(i=0; i<nNeighborN; i++) {
                nbZone = nbN[i];
                if(nbZone ==(np-1)){
                    MPI_Irecv(bqr[i], nZINode[i], MPIIntType, nbZone, level, MPI_COMM_WORLD, &req_recv[i]);
                }
            } 
        } 
    }   // np

    MPI_Waitall(nNeighborN,req_recv,status_array);
    for(i=0; i<nNeighborN; i++) {
        for(j=0; j<nZINode[i]; j++) {
            if(q[bNRNo[i][j]]==CHANGACTIVE && bqr[i][j]==INACTIVE)
                q[bNRNo[i][j]] = INACTIVE;
        }
    } 
    MPI_Waitall(nNeighborN,req_send,status_array);

    t2         = MPI_Wtime();
    mpi_time  += t2 - t1;

    mfmem::sdel_array_2D(bqr,nNeighborN,false);
    mfmem::sdel_array_2D(bqs,nNeighborN,false);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(status_array);
}

// node variable communication and synchronize to the minimum value
// of multi-nodes among processors
void PolyGrid::RecvSendVarNeighbor_Node2(IntType *q)
{
    IntType i, j, nbZone;
    IntType np ;
    IntType **bqs;
    IntType **bqr;

    MPI_Request *req_send,*req_recv;
    MPI_Status *status_array ;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighborN,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighborN,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighborN,dmrfl);
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_2D(bqr,nNeighborN,nZINode,dmrfl,false);
    mfmem::snew_array_2D(bqs,nNeighborN,nZINode,dmrfl,false);

    t1 = MPI_Wtime();
 
    for(np=1;np<=numprocs;np++)
    {
        if(myZone==np){
            // send to other   
            for(i=0; i<nNeighborN; i++) {
                nbZone = nbN[i];
                for(j=0; j<nZINode[i]; j++) {
                    bqs[i][j] = q[bNSNo[i][j]];
                }
                MPI_Isend(bqs[i], nZINode[i], MPIIntType, nbZone, level, MPI_COMM_WORLD,&req_send[i]);
            }
        } else{
            // receive from np     
            for(i=0; i<nNeighborN; i++) {
                nbZone = nbN[i];
                if(nbZone ==(np-1)){
                    MPI_Irecv(bqr[i], nZINode[i], MPIIntType, nbZone, level, MPI_COMM_WORLD, &req_recv[i]);
                }
            } 
        } 
    }   // np

    MPI_Waitall(nNeighborN,req_recv,status_array);
    for(i=0; i<nNeighborN; i++) {
        for(j=0; j<nZINode[i]; j++) {
            if(bqr[i][j]==0) continue;
            if(q[bNRNo[i][j]]==0 ){
                q[bNRNo[i][j]] = bqr[i][j];
            }else if(q[bNRNo[i][j]] > bqr[i][j]){
                q[bNRNo[i][j]] = bqr[i][j];
            }
        }
    } 
    MPI_Waitall(nNeighborN,req_send,status_array);

    t2         = MPI_Wtime();
    mpi_time  += t2 - t1;

    mfmem::sdel_array_2D(bqr,nNeighborN,false);
    mfmem::sdel_array_2D(bqs,nNeighborN,false);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(status_array);
}

// communicate and accumulate node variable
void PolyGrid::RecvSendVarNeighbor_NodeSUM(IntType *q)
{
    IntType i, j, nbZone;
    IntType np ;
    IntType **bqs;
    IntType **bqr;

    MPI_Request *req_send,*req_recv;
    MPI_Status *status_array ;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighborN,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighborN,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighborN,dmrfl);
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_2D(bqr,nNeighborN,nZINode,dmrfl,false);
    mfmem::snew_array_2D(bqs,nNeighborN,nZINode,dmrfl,false);
    t1 = MPI_Wtime();
 
    for(np=1;np<=numprocs;np++)
    {
        if(myZone==np){
            // send to other   
            for(i=0; i<nNeighborN; i++) {
                nbZone = nbN[i];
                for(j=0; j<nZINode[i]; j++) {
                    bqs[i][j] = q[bNSNo[i][j]];
                }
                MPI_Isend(bqs[i], nZINode[i], MPIIntType, nbZone, level, MPI_COMM_WORLD,&req_send[i]);
            }
        } else{
            // receive from np     
            for(i=0; i<nNeighborN; i++) {
                nbZone = nbN[i];
                if(nbZone ==(np-1)){
                    MPI_Irecv(bqr[i], nZINode[i], MPIIntType, nbZone, level, MPI_COMM_WORLD, &req_recv[i]);
                }
            } 
        } 
    }   // np

    MPI_Waitall(nNeighborN,req_recv,status_array);
    for(i=0; i<nNeighborN; i++) {
        for(j=0; j<nZINode[i]; j++) {
            q[bNRNo[i][j]] += bqr[i][j];
        }
    } 
    MPI_Waitall(nNeighborN,req_send,status_array);

    t2         = MPI_Wtime();
    mpi_time  += t2 - t1;

    mfmem::sdel_array_2D(bqr,nNeighborN,false);
    mfmem::sdel_array_2D(bqs,nNeighborN,false);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(status_array);
}

#ifdef FS_CUDA

//for the overlap of the MPI and compute 
void PolyGrid::RecvSendVarNeighbor_Over_Gradient(RealFlow *hostbqs, RealFlow *hostbqr, RealFlow ***bqs, RealFlow ***bqr, MPI_Request *req_send, 
                                        MPI_Request *req_recv, MPI_Status *status_array, IntType nvar){
											
    IntType i, nbZone;
    IntType np ;

    for(np=1;np<=numprocs;np++)
    {
        if(myZone==np)
        {   
            // send to other    
            for(i=0; i<nNeighbor; i++) {
                nbZone = nb[i];
                MPI_Isend(&hostbqs[&bqs[i][0][0] - &bqs[0][0][0]], nZIFace[i]*nvar, MPIReal, nbZone, level, MPI_COMM_WORLD,&req_send[i]);
            }
        } else {   
            // receive from np  
            for(i=0; i<nNeighbor; i++) {
                nbZone = nb[i];
                if(nbZone ==(np-1)){
                    MPI_Irecv(&hostbqr[&bqr[i][0][0] - &bqr[0][0][0]], nZIFace[i]*nvar, MPIReal, nbZone, level, MPI_COMM_WORLD, &req_recv[i]);
                }
            } 
        } 
    }   // np     
}

#endif

//for the overlap of the MPI and compute 
void PolyGrid::RecvSendVarNeighbor_Over(RealFlow ***bqs, RealFlow ***bqr, MPI_Request *req_send, 
                                        MPI_Request *req_recv, MPI_Status *status_array, IntType nvar)
{
    IntType i, nbZone;
    IntType np ;

    for(np=1;np<=numprocs;np++)
    {
        if(myZone==np)
        {   
            // send to other    
            for(i=0; i<nNeighbor; i++) {
                nbZone = nb[i];
                MPI_Isend(bqs[i][0], nZIFace[i]*nvar, MPIReal, nbZone, level, MPI_COMM_WORLD,&req_send[i]);
            }
        } else {   
            // receive from np  
            for(i=0; i<nNeighbor; i++) {
                nbZone = nb[i];
                if(nbZone ==(np-1)){
                    MPI_Irecv(bqr[i][0], nZIFace[i]*nvar, MPIReal, nbZone, level, MPI_COMM_WORLD, &req_recv[i]);
                }
            } 
        } 
    }   // np     
}

void PolyGrid::RecvSendVarNeighbor_Over2(MATRIXTYPE **bqs, MATRIXTYPE **bqr, MPI_Request *req_send, 
                                        MPI_Request *req_recv, MPI_Status *status_array, IntType nvar)
{
    IntType i, nbZone;
    IntType np ;

    for(np=1;np<=numprocs;np++)
    {
        if(myZone==np)
        {   
            // send to other    
            for(i=0; i<nNeighbor; i++) {
                nbZone = nb[i];
                MPI_Isend(bqs[i], nZIFace[i]*nvar, MATRIXMPITYPE, nbZone, level, MPI_COMM_WORLD, &req_send[i]);
            }
        } else {   
            // receive from np  
            for(i=0; i<nNeighbor; i++) {
                nbZone = nb[i];
                if(nbZone ==(np-1)){
                    MPI_Irecv(bqr[i], nZIFace[i]*nvar, MATRIXMPITYPE, nbZone, level, MPI_COMM_WORLD, &req_recv[i]);
                }
            } 
        } 
    }   // np     
}

// test6.2 ���������ͣ�����������
void PolyGrid::RecvSendVarNeighbor_Togeth(IntType nvar, RealFlow **q)
{
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;
    IntType i;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
    Set_RecvSend(bqs, bqr, nvar);
    for(i=0; i<nvar; i++)
        Add_RecvSend(bqs, q[i], i);

    RecvSendVarNeighbor_Over(bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);

    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
    for(i=0; i<nvar; i++)
        Read_RecvSend(bqr, q[i], i);

    mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);

}

void PolyGrid::RecvSendVarMatrixNeighbor_Togeth(IntType nvar, RealFlow *q)
{
    if(nNeighbor == 0) return;

    RealFlow ***bqs=0, ***bqr=0;
    IntType i;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
    Set_RecvSend(bqs, bqr, nvar);

    // for(i=0; i<nvar; i++)
    //     Add_RecvSend(bqs, q[i], i);

    Add_MatrixRecvSend(bqs, q, nvar);

    RecvSendVarNeighbor_Over(bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);

    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);
    // for(i=0; i<nvar; i++)
    //     Read_RecvSend(bqr, q[i], i);

    Read_MatrixRecvSend(bqr, q, nvar);
    
    mfmem::sdel_array_1D(bqr[0][0]);
    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0][0]);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);

}

void PolyGrid::RecvSendVarMatrixNeighbor_Togeth2(IntType nvar, MATRIXTYPE *q)
{
    if(nNeighbor == 0) return;

    MATRIXTYPE **bqs=0, **bqr=0;
    IntType i;

    MPI_Request *req_send=0, *req_recv=0;
    MPI_Status *status_array=0;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
    bqr = NULL;
    bqs = NULL;
    mfmem::snew_array_1D(bqr,nNeighbor,dmrfl);
    mfmem::snew_array_1D(bqs,nNeighbor,dmrfl);
    
    Set_MatrixRecvSend(bqs, bqr, nvar);

    Add_MatrixRecvSend2(bqs, q, nvar);

    RecvSendVarNeighbor_Over2(bqs, bqr, req_send, req_recv, status_array, nvar);

    MPI_Waitall(nNeighbor,req_recv,status_array);
    MPI_Waitall(nNeighbor,req_send,status_array);

    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(status_array);

    Read_MatrixRecvSend2(bqr, q, nvar);

    mfmem::sdel_array_1D(bqr[0]);
    mfmem::sdel_array_1D(bqr);
    mfmem::sdel_array_1D(bqs[0]);
    mfmem::sdel_array_1D(bqs);

}

void PolyGrid::UpdateVectorGhostVar( const RealFlow * vec, RealFlow * ghosts, IntType nVar )
{
    RealFlow **bqs;
    RealFlow **bqr;

    MPI_Request *req_send,*req_recv;
    MPI_Status *status_array ;

    status_array = NULL;
    req_send     = NULL;
    req_recv     = NULL;
    mfmem::snew_array_1D(status_array, nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_send,     nNeighbor,dmrfl);
    mfmem::snew_array_1D(req_recv,     nNeighbor,dmrfl);
    bqr = NULL;
    bqs = NULL;
    IntType * bufSize = NULL;
    mfmem::snew_array_1D(bufSize, nNeighbor, dmrfl);
    for(IntType i = 0; i < nNeighbor; i++)
    {
        bufSize[i] = nZIFace[i]*nVar;
    }
    mfmem::snew_array_2D(bqr,nNeighbor,bufSize,dmrfl,false);
    mfmem::snew_array_2D(bqs,nNeighbor,bufSize,dmrfl,false);

    //// fill the send buf
    for(IntType i = 0; i < nNeighbor; i++)
    {
        for(IntType j = 0; j < nZIFace[i]; j++){
            IntType idx = bCNo[i][j]*nVar;
            IntType bufOffset = j * nVar;
            for(IntType iVar = 0; iVar < nVar; iVar++)
            {
                bqs[i][bufOffset+iVar] = vec[idx+iVar];
            }
        }
    }

    for(IntType np = 1; np <=numprocs; np++)
    {
        if(myZone==np)
        {   
            // send to other    
            for(IntType i = 0; i < nNeighbor; i++) {
                MPI_Isend(bqs[i], bufSize[i], MPIReal, nb[i], level, MPI_COMM_WORLD,&req_send[i]);
            }
        } 
        else
        {   
            // receive from np  
            for(IntType i = 0; i < nNeighbor; i++) {
                IntType nbZone = nb[i];
                if(nbZone ==(np-1)){
                    MPI_Irecv(bqr[i], bufSize[i], MPIReal, nbZone, level, MPI_COMM_WORLD, &req_recv[i]);
                }
            } 
        } 
    }   // np

    MPI_Waitall(nNeighbor,req_recv,status_array);

    /// fill the ghosts from recv buf
    IntType interFaceOffset = nBFace-nIFace;
    for(IntType i = 0; i < nNeighbor; i++) {
        for(IntType j = 0; j < nZIFace[i]; j++) {
            IntType ghostIdx    = (bFNo[i][j]-interFaceOffset)*nVar;
            IntType bufOffset = j * nVar;
            for(IntType iVar = 0 ;iVar < nVar; iVar++)
            {
                ghosts[ghostIdx+iVar] = bqr[i][bufOffset+iVar];
            }
            
        }
    } 
    MPI_Waitall(nNeighbor,req_send,status_array);

    MPI_Barrier(MPI_COMM_WORLD);
    mfmem::sdel_array_2D(bqr,nNeighbor,false);
    mfmem::sdel_array_2D(bqs,nNeighbor,false);
    mfmem::sdel_array_1D(req_recv);
    mfmem::sdel_array_1D(req_send);
    mfmem::sdel_array_1D(status_array);
    mfmem::sdel_array_1D(bufSize);
}
#undef CPP_FILD_ID  // clear out file id
} //~namespace mflow

#endif
