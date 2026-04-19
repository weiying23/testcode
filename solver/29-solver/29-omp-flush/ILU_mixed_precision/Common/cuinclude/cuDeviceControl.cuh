#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <number_type.h>

using namespace mflow;
using namespace std;

extern IntType GPUNum;
extern cudaDeviceProp GPUProp;
       	
void MultiGPUDevice(IntType& GPUNum, IntType& rank, cudaDeviceProp &GPUProp);
void GetGPUNum(IntType& GPUNum);
void GPUInfoQuery(const IntType GPUId, cudaDeviceProp &GPUProp);