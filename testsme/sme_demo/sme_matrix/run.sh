#!/bin/sh

rm -rf test

clang++ -O0 -g -arch arm64 -march=armv9-a+sve2+sme-f64f64 -lSystem  -c matrix.cpp -o matrix.o
clang++ -O0 -g -arch arm64 -march=armv9-a+sve2+sme-f64f64 -lSystem  -c matrix_multiply.cpp -o matrix_multiply.o
clang++ -O0 -g -arch arm64 -march=armv9-a+sve2+sme-f64f64 -lSystem matrix.o matrix_multiply.o -o test
chmod 777 test
dsymutil test -o test.dSYM

./test
#./smeop
