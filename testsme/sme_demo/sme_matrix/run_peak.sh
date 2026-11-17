#!/bin/sh
set -e
/usr/bin/clang++ -O3 peak_fma.cpp -o peak_fma
./peak_fma 0.5
/usr/bin/clang++ -O3 peak_asm_mt.cpp -o peak_asm_mt
./peak_asm_mt 0.5
/usr/bin/clang++ -O0 -march=armv9-a+sve+sve2+sme+sme-f64f64 sme_peak.cpp -o sme_peak
./sme_peak 20000000
/usr/bin/clang++ -c -O2 -march=armv9-a packing.cpp -o packing.o
/usr/bin/clang++ -c -O1 -march=armv9-a+sve+sve2+sme+sme-f64f64 matrix_multiply_single.cpp -o matrix_multiply_single.o
/usr/bin/clang++ -O0 -march=armv9-a+sve+sve2+sme+sme-f64f64 -DACCELERATE_NEW_LAPACK -Wno-deprecated-declarations single_bench.cpp matrix_multiply_single.o packing.o -framework Accelerate -o single_bench
./single_bench
