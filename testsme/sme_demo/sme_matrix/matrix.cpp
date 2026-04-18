/*
 * matrix.cpp
 *
 * 主程序：初始化矩阵数据，调用 SME GEMM kernel，测量执行时间。
 */

#include <cstdlib>
#include <cstdio>
#include <random>
#include <sys/time.h>

#include "matrix_methods.h"

/*
 * init — 用 [1.0, 5.0] 范围内的随机双精度浮点数填充矩阵。
 *
 * 参数：
 *   data — 目标数组指针
 *   size — 元素总数
 */
void init(double *data, double size)
{
    double randomleft  = 1.0;
    double randomright = 5.0;
    std::random_device seed;
    std::mt19937 engine(seed());
    std::uniform_real_distribution<double> distrib(randomleft, randomright);
    for (int i = 0; i < size; i++) {
        data[i] = distrib(engine);
    }
}

/*
 * cleararraryO — 将矩阵 C 全部置零。
 *
 * 参数：
 *   des  — 目标数组指针
 *   size — 元素总数
 */
void cleararraryO(double *des, int size) {
    for (int i = 0; i < size; i++) {
        des[i] = 0.0;
    }
}

/*
 * timeinterval — 计算两个 timeval 之间的秒数差。
 *
 * 参数：
 *   begin — 起始时间戳
 *   end   — 结束时间戳
 * 返回值：时间差（秒，含微秒精度）
 */
double timeinterval(struct timeval begin, struct timeval end) {
    double interval = static_cast<double>(end.tv_sec  - begin.tv_sec);
    interval += static_cast<double>(end.tv_usec - begin.tv_usec) * 1e-6;
    return interval;
}

int main() {
    /* 启动时全面验证 SME 硬件状态 */
    verify_sme();

    /* 矩阵维度：A(M×K) × B(K×N) → C(M×N) */
    const int mlength = 1024;
    const int nlength = 1024;
    const int klength = 1024;

    /* 分配矩阵内存 */
    double *matrixa = static_cast<double*>(malloc(mlength * klength * sizeof(double)));
    double *matrixb = static_cast<double*>(malloc(klength * nlength * sizeof(double)));
    double *matrixc = static_cast<double*>(malloc(mlength * nlength * sizeof(double)));

    /* 随机初始化 A、B，C 置零 */
    init(matrixa, mlength * klength);
    init(matrixb, klength * nlength);
    cleararraryO(matrixc, mlength * nlength);

    struct timeval start, end;

    /* 计时并调用 SME GEMM kernel */
    gettimeofday(&start, nullptr);
    gemmkernel(matrixa, matrixb, matrixc, mlength, nlength, klength, 1);
    gettimeofday(&end, nullptr);

    double smetime = timeinterval(start, end);
    printf("time is \t %10.6f sec\n", smetime);

    free(matrixa);
    free(matrixb);
    free(matrixc);

    return 0;
}
