#!/bin/sh

rm -f test

#export OMPI_MCA_btl=self,vader
#export OMPI_MCA_btl_tcp_if_include=lo0
#export OMPI_MCA_btl=^openib


#clang++ -O3 -Xpreprocessor	 -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp  test_hello_world_openmp.cpp -o testomp

clang++ -O3 -Xpreprocessor -pthread graph.cpp -o testgraph

##clang++ -O3 -Xpreprocessor -pthread  test_hello_world_pthread.cpp -o testpthread

##mpicxx -O3 test_hello_world_mpi.cpp -o testmpi

##clang -mcpu=apple-m4 -c  sme_sve_test.s -o sme_sve_test


##chmod 777 test*
##chmod 777 sme*

#mpirun -np 1 ./testpthread
mpirun -np 1 --allow-run-as-root   ./testgraph
