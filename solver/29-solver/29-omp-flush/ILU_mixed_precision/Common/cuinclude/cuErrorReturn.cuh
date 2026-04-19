#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define HANDLE_API_ERR(err) (handleAPIErr(err, __FILE__, __LINE__))

using namespace std;

// Return CUDA API error
void handleAPIErr(cudaError_t err,  char const *file, const int line);
// Return CUDA kernel error 
void handleKernelErr(char const *file, const int line);