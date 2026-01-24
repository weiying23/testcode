#include <iostream>
#include <vector>
#include <time.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>
#include </Users/yingwei/Documents/code/testblas/threadpool.h>
#include <Accelerate/Accelerate.h> 
#include <arm_sme.h>


#define m  512
#define k  512
#define n  512
clock_t start1, end1, start2, end2, start3, end3;
clock_t start4, end4, start5, end5, start6, end6;

double matA[m][k];
double matB[k][n];
double ref[m][n];
double result[m][n];    
int BLOCK_SIZE = 32;

void compute_block(const BlockTask& task) {
    const int block_size = task.block_size;
    const int i_start = task.block_i * block_size;
    const int j_start = task.block_j * block_size;
    const int i_end = std::min(i_start + block_size, task.mm);
    const int j_end = std::min(j_start + block_size, task.nn);

    svbool_t pg = svptrue_b32();
    svzero_za();  // 初始化ZA寄存器为0

    //计算核心

    for (int i = i_start; i < i_end; ++i){
        for (int j = j_start; j < j_end; ++j){
            for (int l = 0; l < task.kk; ++l){
            // 计算当前块 
                result[i][j] += matA[i][l] * matB[l][j];
            }
        }
    }
    
}




int main() {
    
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            matA[i][j] = 1.1;
     

    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            matB[i][j] = 1.1;

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            result[i][j] = 0;
    
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            ref[i][j] = 0;


    //loop apple acc lib
    start2 = clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, (double *)matA, k, (double *)matB, n, 1.0, (double *)ref, n);
    end2 =clock();
    //

    //transpose2D((double *)matB, m, n);

    // loop tiling 
    const int num_blocks1 = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_blocks2 = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 创建线程池
    int num_threads = 4;
    ThreadPool pool(num_threads);

    // 提交分块任务到线程池
    start4 = clock();

    for (int bi = 0; bi < num_blocks1; ++bi) {
        for (int bj = 0; bj < num_blocks2; ++bj) {
            BlockTask task{bi, bj, BLOCK_SIZE, (double *)matA, (double *)matB, (double *)result, m, n, k};
            pool.enqueue([task] { compute_block(task); }); // 提交任务
        }
    }

    end4 = clock();

    pool.shutdown(); // 销毁线程池，否则会出现计算结果出错

    // VALIDATE RESULTS
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            bool diff = (std::abs(ref[i][j] - result[i][j]) > 1e-8);
            if (diff) {
                std::cout << "error!!!; diff = " << ref[i][j] - result[i][j] << " " << std::endl;
                break;
            }
        }
    }

    std::cout << std::endl;
    /*for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << result[i][j] << " ";
        }
        std::cout << std::endl;
    }*/

    std::cout << "overall computing time blas = " << double(end2-start2) << " ms " << std::endl;
    std::cout << "overall computing time tiling = " << double(end4-start4) << " ms " << std::endl;

    return 0;
}

