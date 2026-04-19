#include <cstdlib>
#include <string>
#include "cuErrorReturn.cuh"
#include "cuDeviceControl.cuh"
#include "stdio.h"

#include <number_type.h>

#ifdef MPICH
#include <mpi.h>
#endif

using namespace mflow;
//using namespace std;

// define the external variables of GPUManage namespace
IntType GPUNum = 0;
cudaDeviceProp GPUProp = cudaDeviceProp();

void MultiGPUDevice(IntType& GPUNum, IntType& rank, cudaDeviceProp &GPUProp)
{
	GetGPUNum(GPUNum);
	//check GPU numbers
	if (rank == 0) {
		cout<<"The computing platform owns "<<GPUNum<<" GPUs"<<endl;
	}
	/* if ((GPUNum < 2)&&(rank > 0)) {
		cout<<"The cuda code must run on platform with at least 2 GPUs"<<endl;
		exit(0);
	} */
	IntType np = 0;
#if (defined MPICH)
	MPI_Comm_size(MPI_COMM_WORLD, &np);
#endif
	if (np == 0){
		//no mpi run:
		IntType GPUId= 7;//GPUNum - 1; //just set GPU 0 for running CUDA code!
		GPUInfoQuery(GPUId, GPUProp);
		// device initialize
		cudaSetDevice(GPUId);
	}
	else{
		//mpi run:
		//set each rank refers to one GPU card:
		//cudaSetDevice(rank + 2);
		cudaSetDevice(rank);
	}
#if (defined MPICH)
	if (rank == 0){
		cout << endl << "Open MPI and Using: " << np << " GPUs. " << endl;
	}
#else
	cout << endl << "Close MPI and Using Single GPU. " << endl;
#endif
#if (defined ShareMemory)	
	if (rank == 0){
		cudaSharedMemConfig *pConfig = NULL;
		cudaDeviceGetSharedMemConfig(pConfig);
		cout << endl << "Open ShateMemory Opt on GPU, and Shared Mem Config Info: " << pConfig << endl;
	}
#endif
	
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    
}	
/* get GPU count on the node */
void GetGPUNum(IntType& GPUNum)
{
	HANDLE_API_ERR(cudaGetDeviceCount(&GPUNum));
	if(GPUNum == 0)	{
		cout << "No GPU Card Detected on This Compute Node." << endl
		<< "Please Change the Platform or Turn off CUDA Compile Mode" << endl
		<< "Exit!" << endl;
		exit(0);	
    }
} 
/* get detailed info about GPU card */
void GPUInfoQuery(const IntType GPUId, cudaDeviceProp &GPUProp)
{
	HANDLE_API_ERR(cudaGetDeviceProperties(&GPUProp, GPUId));
}
