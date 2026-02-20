#!/bin/sh

rm test

mpicc -O3  -mcpu=apple-m4   -Xpreprocessor   -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -lSystem  gmres_ilu.c -o test


chmod 777 test

mpirun -np 1 --allow-run-as-root  -x OMP_NUM_THREADS=2 -x OMP_PROC_BIND=close ./test


