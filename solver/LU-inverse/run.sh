#!/bin/sh

# 线程数通过 OMP_NUM_THREADS 环境变量设置
# 用法：OMP_NUM_THREADS=8 ./run.sh

# 1. 编译并运行三种方法对比 (LAPACK vs Eigen vs Blockwise)
clang++ -O3 -mcpu=apple-m4 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -framework Accelerate -lm -I/opt/homebrew/include/eigen3 final_compare.cpp -o final_compare && ./final_compare

# 2. 编译并运行 LAPACK vs Blockwise 对比
#clang -O3 -mcpu=apple-m4 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -framework Accelerate -lm invertible_perf_test.c -o invertible_perf && ./invertible_perf

# 3. 编译并运行矩阵可逆性检测
#clang -O3 -mcpu=apple-m4 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -framework Accelerate -lm check_invertible.c -o check_invertible && ./check_invertible
