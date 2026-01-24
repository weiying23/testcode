//测试不同omp策略&pthread策略对性能影响
//#include <omp.h>
#include <pthread.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include <algorithm>
#include <time.h>
using namespace std;

/* template for taskloop
template <typename T>
void process_array(T* arr, size_t size, size_t i) {
    #pragma omp taskloop private(i) shared(arr) grainsize(25000)
    for (i = 0; i < size; i++) {
        arr[i] = arr[i] + arr[i] * i / 3.14;
    }
}
*/

// struct for pthread parallel
typedef struct {
    int thread_id;
    int start_idx;
    int end_idx;
    double* a;
    double* b;
    double* c;
    double* d;
    double* e;
    double* f;
} ThreadData;

// 线程处理函数
void* process_arrays(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    for (int i = data->start_idx; i < data->end_idx; i++) {
        double factor = i / 3.14;
        data->a[i] = data->a[i] + data->a[i] * factor;
        data->b[i] = data->b[i] + data->b[i] * factor;
        data->c[i] = data->c[i] + data->c[i] * factor;
        data->d[i] = data->d[i] + data->d[i] * factor;
        data->e[i] = data->e[i] + data->e[i] * factor;
        data->f[i] = data->f[i] + data->f[i] * factor;
    }
    
    return NULL;
}
//for pthread onlt


int main() {

    int i;
    double a[100000];
    double b[100000];
    double c[100000];  
    double d[100000];
    double e[100000];
    double f[100000];  
    double s = 0;  

    struct timespec start, end;

    std::random_device rd;  // 真随机数种子
    std::mt19937 gen(rd()); // Mersenne Twister 引擎
    std::uniform_real_distribution<> dis(0, 100); // 0-100 均匀分布
    for (i = 0; i < 100000; ++i) {
        a[i] = i/10000;
        b[i] = i/10000;
        c[i] = i/10000;
    }
    
    //for pthread only
    int NUM_THREADS = 2;
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    int chunk_size = 100000 / NUM_THREADS;

    //for omp only
    //omp_set_num_threads(2);


    /* test openmp fundmental function
    #pragma omp parallel for private(i)schedule(dynamic, 4) 
    for (i=0; i<16; i++)
    {
        //获取当前线程的IDs
        int thread_id = omp_get_thread_num();
        // 获取当前正在使用的线程总数
        int num_threads = omp_get_num_threads();
        // 打印当前线程的ID和总的线程数
        printf("iter %d Thread %d out of %d threads\n", i, thread_id, num_threads);
    }
    */

    //test performance
    //omp_set_num_threads(2);
/*
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (i=0; i<100000; i++){
        a[i] = a[i]+a[i]*i/3.14;
        b[i] = b[i]+b[i]*i/3.14;
        c[i] = c[i]+c[i]*i/3.14;

        d[i] = d[i]+d[i]*i/3.14;
        e[i] = e[i]+e[i]*i/3.14;
        f[i] = f[i]+f[i]*i/3.14;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsedTime = (end.tv_sec - start.tv_sec)*1000;
    elapsedTime += (end.tv_nsec - start.tv_nsec) / 1000000.0;
    s = 0;
    for (i=0; i<100000; i++){
        s += a[i];
        s += b[i];
        s += c[i];
        s += d[i];
        s += e[i];
        s += f[i];
    } // s=2.93801e+10
    std::cout << "base耗时: " << elapsedTime << "        " << s << endl;
return 0;
*/

    clock_gettime(CLOCK_MONOTONIC, &start);
//#pragma omp parallel for schedule(dynamic, 500) Method1 直接并行
/* METHOD2 TASKLOOP
#pragma omp parallel 
{
    #pragma omp single nowait
    { 
        #pragma omp taskloop private(i) shared(a) grainsize(5000)
        {
            for (i=0; i<100000; i++){
                a[i] = a[i]+a[i]*i/3.14;
            }
        }
        #pragma omp taskloop private(i) shared(b) grainsize(5000)
        {
            for (i=0; i<100000; i++){
                b[i] = b[i]+b[i]*i/3.14;
            }
        }
        #pragma omp taskloop private(i) shared(c) grainsize(5000)
        {
            for (i=0; i<100000; i++){
                c[i] = c[i]+c[i]*i/3.14;
            }
        }
        #pragma omp taskloop private(i) shared(d) grainsize(5000)
        {
            for (i=0; i<100000; i++){
                d[i] = d[i]+d[i]*i/3.14;
            }
        }
        #pragma omp taskloop private(i) shared(e) grainsize(5000)
        {
            for (i=0; i<100000; i++){
                e[i] = e[i]+e[i]*i/3.14;
            }
        }
        #pragma omp taskloop private(i) shared(f) grainsize(5000)
        {
            for (i=0; i<100000; i++){
                f[i] = f[i]+f[i]*i/3.14;
            }
        }
    }
}
*/

/*
// METHOD3 TEMPLATE TASKLOOP
#pragma omp parallel
{
    #pragma omp single nowait
    {
        {
            // 设置任务调度策略
            omp_set_schedule(omp_sched_dynamic, 0);
            
            // 创建处理任务
            process_array(a, 100000, i);
            process_array(b, 100000, i);
            process_array(c, 100000, i);
            process_array(d, 100000, i);
            process_array(e, 100000, i);
            process_array(f, 100000, i);
        } // 隐式等待所有任务
    }
    //#pragma omp taskwait
}
*/

/*
// METHOD3 direct TASKLOOP
#pragma omp parallel
{
    #pragma omp single
    {
        #pragma omp taskloop grainsize(10000)
        {
            for (i=0; i<100000; i++){
                a[i] = a[i]+a[i]*i/3.14;
                b[i] = b[i]+b[i]*i/3.14;
                c[i] = c[i]+c[i]*i/3.14;

                d[i] = d[i]+d[i]*i/3.14;
                e[i] = e[i]+e[i]*i/3.14;
                f[i] = f[i]+f[i]*i/3.14;
            }
        }
    }
}
*/

// METHOD4 OMP SECTION __ no profit at all
/*
#pragma omp parallel 
{
    #pragma omp sections nowait
    { 
        #pragma omp section
        {
            for (i=0; i<100000; i++){
                a[i] = a[i]+a[i]*i/3.14;

                b[i] = b[i]+b[i]*i/3.14;

                c[i] = c[i]+c[i]*i/3.14;
            }
        }
        #pragma omp section
        {
            for (i=0; i<100000; i++){
                d[i] = d[i]+d[i]*i/3.14;

                e[i] = e[i]+e[i]*i/3.14;

                f[i] = f[i]+f[i]*i/3.14;
            }
        }
    }
}
*/

    // METHOD 5 FOR PTHREAD CREATION AND START
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].start_idx = i * chunk_size;
        thread_data[i].end_idx = (i == NUM_THREADS - 1) ? 100000 : (i + 1) * chunk_size;
        thread_data[i].a = a;
        thread_data[i].b = b;
        thread_data[i].c = c;
        thread_data[i].d = d;
        thread_data[i].e = e;
        thread_data[i].f = f;
        
        pthread_create(&threads[i], NULL, process_arrays, &thread_data[i]);
    }
    
    // 等待所有线程完成
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }








    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsedTime = (end.tv_sec - start.tv_sec)*1000;
    elapsedTime += (end.tv_nsec - start.tv_nsec) / 1000000.0;

    s = 0;
    for (i=0; i<100000; i++){
        s += a[i];
        s += b[i];
        s += c[i];
        s += d[i];
        s += e[i];
        s += f[i];
    }
    
    std::cout << "执行耗时: " << elapsedTime << "        " << s << endl;
    return 0;
    
}
