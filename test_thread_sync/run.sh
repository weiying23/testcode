#!/bin/sh


#clang++ -O3 -Xpreprocessor	 -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp  test_hello_world_openmp.cpp -o testomp

clang -O3 -Xpreprocessor -pthread test_omp.c -o testomp

mpirun -np 1 --allow-run-as-root  -x OMP_NUM_THREADS=2 -x OMP_PROC_BIND=close  ./testomp
