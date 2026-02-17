#!/bin/sh

rm -rf test

#clang++ -O3 -std=c++20  -arch arm64 -march=armv9-a+sme -lSystem  -c matrix.cpp -o matrix.o
#clang++ -O3 -std=c++20  -arch arm64 -march=armv9-a+sme -lSystem  -c matrix_multiply.cpp -o matrix_multiply.o
#clang++ -O3 -std=c++20  -arch arm64 -march=armv9-a+sme -lSystem matrix.o matrix_multiply.o -o test

clang++ -O0 -g -std=c++20 -arch arm64 -mcpu=apple-m4 -lSystem matrix.cpp matrix_multiply.cpp -o test
chmod 777 test
#dsymutil test -o test.dSYM

./test
#./smeop
