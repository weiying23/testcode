#!/bin/sh


#clang++ -O3 -Xpreprocessor	 -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp  test_hello_world_openmp.cpp -o testomp

rm -rf testomp

#clang -O3 -Xpreprocessor -pthread -DBUSY_WAIT_BARRIER test_omp.c -o testomp
#打开忙等待

clang -O3 -Xpreprocessor -pthread test_omp.c -o testomp

echo 'compile completed'

mpirun -np 1 --allow-run-as-root -x  OMP_PROC_BIND=close --cpu-set 0-3  ./testomp
