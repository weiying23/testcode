#!/bin/sh

rm -f threadalloc

echo "编译多线程内存分配性能测试程序..."
#clang++ -O3 -Xpreprocessor	 -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp  test_hello_world_openmp.cpp -o testomp

clang -O3 -Xpreprocessor -pthread -DUSE_TCMALLOC -lm -I/opt/homebrew/Cellar/gperftools/2.17.2/include/gperftools -L/opt/homebrew/Cellar/gperftools/2.17.2/lib/ -ltcmalloc_minimal test_multithread_alloc.c -o threadalloc

##clang++ -O3 -Xpreprocessor -pthread  test_hello_world_pthread.cpp -o testpthread

echo "开始多线程内存分配性能测试..."
echo ""

# 运行基本测试
echo "=== 基础性能测试 ==="
mpirun -np 1 --allow-run-as-root   ./threadalloc


echo "测试完成！"
