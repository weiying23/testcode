#!/bin/sh

#clang++ -O3 -mcpu=apple-m4+sve+sme  sme_matrix.cpp -o sme_test
#clang -arch arm64 -march=armv9-a+sme -lSystem  smeop.s -o smeop
#clang -O3 -arch arm64 -march=armv9-a+sme -lSystem  print_smeop.c smeop.s -o test
#clang -O3  -march=armv9-a+sme --rtlib=compiler-rt  sme_test.cpp -o sme_test
clang -O3 -arch arm64 -march=armv9-a+sme -lSystem  print_smeop.c smeop.s -o test
chmod 777 sme*

#./sme_test
#./smeop
