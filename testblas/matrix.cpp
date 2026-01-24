#include <iostream>
#include <vector>
#include <time.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>
#include <Accelerate/Accelerate.h> 

#define m  512
#define k  512
#define n  512
clock_t start1, end1, start2, end2, start3, end3;
clock_t start4, end4, start5, end5, start6, end6;

double matA[m][k];
double matB[k][n];
double result[m][n];

// 线程参数结构体
typedef struct {
    int row;     // 计算的行号
} ThreadParam;

void* compute_row(void* arg) {
    ThreadParam* param = (ThreadParam*)arg;
    int i = param->row;
    for (int j = 0; j < n; j++) {
        for (int l = 0; l < k; l++) {
            result[i][j] += matA[i][l] * matB[l][j];
        }
    }
    pthread_exit(NULL);
}


int main() {
    
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            matA[i][j] = 1;
     

    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            matB[i][j] = 1;


    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            result[i][j] = 0;

    // 3 loop 
    start1 = clock();
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            for (int l = 0; l < k; l++){
                result[i][j] += matA[i][l] * matB[l][j];
            }
        }
    }
    end1 =clock();

    start2 = clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, (double *)matA, m, (double *)matB, k, 1.0, (double *)result, m);
    end2 =clock();

    //pthread
    pthread_t threads[m];
    ThreadParam params[m];
    start3 = clock();
    for (int i = 0; i < m; i++) {
        params[i].row = i;
        pthread_create(&threads[i], NULL, compute_row, &params[i]);
    }

    for (int i = 0; i < m; i++) {
        pthread_join(threads[i], NULL);
    }
    end3 = clock();

    omp_set_num_threads(4);

    // loop tiling 
    start4 = clock();
    int BLOCK_SIZE = 8;
 #pragma omp_parallel_for private(i)schedule(dynamic, 1)  
    for (int i = 0; i < m; i+=BLOCK_SIZE){
        for (int j = 0; j < n; j+=BLOCK_SIZE){
            for (int l = 0; l < k; l+=BLOCK_SIZE){
            
            int i1 = std::min(i + BLOCK_SIZE, m);
            int j1 = std::min(j + BLOCK_SIZE, n);
            int l1 = std::min(l + BLOCK_SIZE, k);
            for (int i0 = i; i0 < i1; i0++)
            {
                for (int j0 = j; j0 < j1; j0++)
                {
                    for (int l0 = l; l0 < l1; l0++)
                    {
                        result[i0][j0] += matA[i0][l0] * matB[l0][j0];
                    }
                    
                }
                
            }
            }
        }
    }
    end4 =clock();

    /*for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << result[i][j] << " ";
        }
        std::cout << std::endl;
    }*/

    std::cout << "overall computing time orgn = " << double(end1-start1) << " ms " << std::endl;
    std::cout << "overall computing time blas = " << double(end2-start2) << " ms " << std::endl;
    std::cout << "overall computing time pthd = " << double(end3-start3) << " ms " << std::endl;
    std::cout << "overall computing time tiling = " << double(end4-start4) << " ms " << std::endl;

    return 0;
}

