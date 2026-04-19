clang++ -std=c++11 -fopenmp -O3 test_lusgs.cpp -o test_lusgs

export OMP_NUM_THREADS=4

./test_lusgs
