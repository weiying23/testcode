#include <stdio.h>
#include <iostream>
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuErrorReturn.cuh"

// Return CUDA API error
void handleAPIErr(cudaError_t err, char const *file, const int line)
{
    if(err!= cudaSuccess)
     {
        cout<<"Error: ##"<<cudaGetErrorString(err)<<"## In codes file: ##"<<file
            <<"## At line: ##"<<line<<"##"<<endl;
        exit(EXIT_FAILURE);
     }
}
// Return CUDA kernel error 
void handleKernelErr(char const *file, const int line)
{
    cudaError_t err;
    int tline= line;  // the handleKernelErr is called next to Kernel function
	cudaDeviceSynchronize();
	err= cudaPeekAtLastError();
     if(err != cudaSuccess)
       {
            cout<<"Error: Fail to Launch kernel in file "<<file<<" at line "<<tline<<endl
                <<"Error Description: "<<cudaGetErrorString(err)<<endl;
            exit(EXIT_FAILURE);
       }
}