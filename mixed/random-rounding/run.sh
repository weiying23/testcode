#!/bin/sh

rm test32
rm test64

#clang -O0 -g -mcpu=apple-m4  -lSystem  rr32_16.c -o test32
clang -O0  -mcpu=apple-m4  -lSystem  rr64_32_new.c -o test64


#chmod 777 test32
chmod 777 test64

#./test32
./test64


