#!/bin/sh

rm -f testblas
rm -f testpthread
clang++ -O3 -mcpu=apple-m4 -g -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp  -pthread -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -framework Accelerate matrix.cpp    -o testblas

clang++ -O3  -g -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp  -pthread -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -framework Accelerate matrix_pthreadpool.cpp    -o testpthread

chmod 777 testblas 
chmod 777 testpthread
##./testblas
./testpthread
