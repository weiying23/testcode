#!/bin/sh

rm -f testalloc


#clang++ -O3 -Xpreprocessor	 -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp  test_hello_world_openmp.cpp -o testomp

clang -O0 -Xpreprocessor -pthread -L/opt/homebrew/Cellar/gperftools/2.17.2/lib/ -ltcmalloc_minimal test.c -o testalloc

##clang++ -O3 -Xpreprocessor -pthread  test_hello_world_pthread.cpp -o testpthread


mpirun -np 1 --allow-run-as-root   ./testalloc
