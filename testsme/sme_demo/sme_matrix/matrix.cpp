#include <cstdlib>
#include <cstdio>
#include <random>
#include <sys/time.h>

#include "matrix_methods.h"

void init(double *data, double size)
{
    double randomleft=1.0;
    double randomright=5.0;
    std::random_device seed;
    std::mt19937 engine(seed());
    std::uniform_real_distribution<double> distrib(randomleft, randomright);
    for (int i=0; i<size; i++){
        data[i] = distrib(engine);
    }
}
void cleararraryO(double *des, int size){
    for(int i=0;i<size;i++){
        des[i]=0.0;
    }
}

double timeinterval(struct timeval begin, struct timeval end){
    double interval = static_cast<double>(end.tv_sec - begin.tv_sec);
    interval += static_cast<double>(end.tv_sec - begin.tv_sec) * 1e-6;
    return interval;
}

int main(){
    const int mlength = 1024;
    const int nlength = 1024;
    const int klength = 1024;
    double *matrixa = static_cast<double*>(malloc(mlength*klength*sizeof(double)));
    double *matrixb = static_cast<double*>(malloc(klength*nlength*sizeof(double)));
    double *matrixc = static_cast<double*>(malloc(mlength*nlength*sizeof(double)));
    init(matrixa,mlength*klength);
    init(matrixb,klength*nlength);
    struct timeval start;
    struct timeval end;

    cleararraryO(matrixc,mlength*nlength);

    gettimeofday(&start,nullptr);
    test_kernel();
    //gemmkernel(matrixa,matrixb,matrixc,mlength,nlength,klength,1) ;
    gettimeofday(&end,nullptr);
    double smetime=timeinterval(start, end);
    printf("time is \t %10.6f sec\n", smetime);

    return 0;
}

