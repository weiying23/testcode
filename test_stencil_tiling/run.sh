#!/bin/sh

rm -f test_tiling


clang -O3 -Xpreprocessor -pthread  test_tiling.c -o test_tiling

mpirun -np 1 --allow-run-as-root   ./test_tiling


echo "测试完成！"
