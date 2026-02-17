// 线程pthread实现一等多同步算法
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <stdatomic.h>

#define N 1000           // 网格大小 N x N
#define T 4             // 线程数
#define ITERATIONS 10000    // 迭代次数


// 全局变量
double **grid;           // 当前网格
double **new_grid;       // 新网格

//========================= 线程参数
// 同步屏障结构体
typedef struct {
    pthread_mutex_t mutex; // 互斥锁，用于保护对屏障结构的访问
    pthread_cond_t cond; // 条件变量，用于线程等待和唤醒，只是线程等待和唤醒的机制
    int count;           // 已到达屏障的线程数
    int total;           // 总线程数
    int generation;      // 屏障的代数，屏障轮次（防止上一轮的唤醒影响下一轮）
#if BUSY_WAIT_BARRIER
    volatile int ready;    // 忙等待标志
#endif
} Barrier;

Barrier barrier;         // 全局屏障

// 线程参数结构体
typedef struct {
    int tid;            // 线程ID
    int start_row;      // 起始行
    int end_row;        // 结束行（不包含）
} ThreadArgs;

// 初始化屏障
void barrier_init(Barrier *b, int total) {
    pthread_mutex_init(&b->mutex, NULL);
    pthread_cond_init(&b->cond, NULL);
    b->count = 0;
    b->total = total;
    b->generation = 0;
#if BUSY_WAIT_BARRIER
    b->ready = 0;
#endif
}
//=========================

//========================= 网格初始化和释放
// 初始化网格 中心点100度 
void init_grid() {
    grid = (double**)malloc(N * sizeof(double*));
    new_grid = (double**)malloc(N * sizeof(double*));
    
    for (int i = 0; i < N; i++) {
        grid[i] = (double*)malloc(N * sizeof(double));
        new_grid[i] = (double*)malloc(N * sizeof(double));
        for (int j = 0; j < N; j++) {
            if (i == N/2 && j == N/2) {
                grid[i][j] = 100.0;  // 中心点热源
            } else {
                grid[i][j] = 10.0;
            }
            new_grid[i][j] = 0.0;
        }
    }
}

// 释放网格内存
void free_grid() {
    for (int i = 0; i < N; i++) {
        free(grid[i]);
        free(new_grid[i]);
    }
    free(grid);
    free(new_grid);
}
//========================= 

//========================= 关键函数 --> 一对多、多对一同步
// 屏障同步函数
void barrier_wait(Barrier *b) {
#ifdef BUSY_WAIT_BARRIER
    // 使用忙等待的屏障实现
    pthread_mutex_lock(&b->mutex);
    
    int my_generation = b->generation;  // 记录当前的屏障代次
    b->count++;  // 增加到达屏障的线程数
    
    if (b->count == b->total) {
        // 最后一个线程到达屏障
        b->count = 0;  // 重置计数器
        b->generation++;  // 增加代次
        b->ready = 1;  // 设置就绪标志
        pthread_mutex_unlock(&b->mutex);
        // 保存当前状态后释放锁
        pthread_cond_broadcast(&b->cond);
        return;
    }
    
    pthread_mutex_unlock(&b->mutex);
    
    // 忙等待循环
    while (1) {
        pthread_mutex_lock(&b->mutex);
        
        // 关键修复：只检查generation是否增加
        // 不要检查ready标志，因为它可能在检查后被重置
        if (b->generation > my_generation) {
            // 所有线程都已到达，可以继续执行
            pthread_mutex_unlock(&b->mutex);
            break;
        }
        
        pthread_mutex_unlock(&b->mutex);
        
        // 避免完全空转，稍微让出CPU时间
        // 这样可以防止线程饥饿，同时保持忙等待的特性
        sched_yield();  // 让出CPU给其他线程
    }
#else    
    pthread_mutex_lock(&b->mutex); // 互斥锁，确保同一时间只有一个线程可以访问共享资源
    
    int gen = b->generation;
    b->count++;
    
    if (b->count == b->total) { // --> 已经同步的线程数 = 总线程数
        // 最后一个线程到达，唤醒所有等待的线程
        b->count = 0; // 重置计数器，为下一轮做准备
        b->generation++; // 增加"代"号，表示进入新的轮次
        pthread_cond_broadcast(&b->cond); // 用于唤醒所有正在等待的线程，让他们继续执行
        // pthread_cond_signal函数只唤醒至少一个等待的线程
        // pthread_cond_broadcast会唤醒所有等待该条件变量的线程。
    } else {
        // 等待其他线程
        while (gen == b->generation) {
            pthread_cond_wait(&b->cond, &b->mutex); // 释放互斥锁并等待条件变量
        }
    }
    
    pthread_mutex_unlock(&b->mutex); // 互斥锁解锁
#endif
}

// 销毁屏障
void barrier_destroy(Barrier *b) {
    pthread_mutex_destroy(&b->mutex);
    pthread_cond_destroy(&b->cond);
}
//========================= 关键函数

//========================= 计算部分，每线程计算start_row ~ int end_row部分
// stencil计算：5点拉普拉斯算子
void stencil_computation(int start_row, int end_row) {
    // 确保不越界访问
    int actual_start = (start_row == 0) ? 1 : start_row;
    int actual_end = (end_row == N) ? N-1 : end_row;
    
    for (int i = actual_start; i < actual_end; i++) {
        for (int j = 1; j < N-1; j++) {
            // 5点拉普拉斯算子
            new_grid[i][j] = (grid[i-1][j] + grid[i+1][j] + 
                             grid[i][j-1] + grid[i][j+1]) / 4.0;
        }
    }
    
    // 边界处理
    if (start_row == 0) {
        for (int j = 0; j < N; j++) {
            new_grid[0][j] = grid[0][j];
        }
    }
    if (end_row == N) {
        for (int j = 0; j < N; j++) {
            new_grid[N-1][j] = grid[N-1][j];
        }
    }
}
//=========================

// 线程函数
void* thread_func(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int tid = args->tid;
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // 第一次stencil计算
        stencil_computation(args->start_row, args->end_row);
        
        // 第一次同步
        barrier_wait(&barrier);
        
        // 交换网格指针
        if (tid == 0) {
            double** temp = grid;
            grid = new_grid;
            new_grid = temp;
        }
        
        // 确保所有线程看到交换后的网格
        barrier_wait(&barrier);
        
        // 第二次stencil计算
        stencil_computation(args->start_row, args->end_row);
        
        // 第二次同步
        barrier_wait(&barrier);
        
        // 再次交换网格指针
        if (tid == 0) {
            double** temp = grid;
            grid = new_grid;
            new_grid = temp;
        }
        
        // 确保所有线程看到交换后的网格
        barrier_wait(&barrier);
    }
    
    return NULL;
}

int main() {
    pthread_t threads[T];
    ThreadArgs args[T];
    
    printf("开始初始化...\n");
    
    // 初始化网格
    init_grid();
    
    // 初始化屏障
    barrier_init(&barrier, T); // T是线程数
    
    // 计算每个线程处理的行范围
    int rows_per_thread = N / T;
    for (int i = 0; i < T; i++) {
        args[i].tid = i;
        args[i].start_row = i * rows_per_thread;
        args[i].end_row = (i == T-1) ? N : (i+1) * rows_per_thread;
        printf("线程 %d: 处理行 %d 到 %d\n", i, args[i].start_row, args[i].end_row-1);
    }
    
    // 记录开始时间
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    printf("开始计算...\n");
    
    // 创建线程
    for (int i = 0; i < T; i++) {
        if (pthread_create(&threads[i], NULL, thread_func, &args[i]) != 0) {
            perror("pthread_create 失败");
            exit(1);
        }
    }
    
    // 等待所有线程完成
    for (int i = 0; i < T; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // 记录结束时间
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // 输出结果和性能信息
    printf("\n计算完成！\n");
    printf("网格大小: %d x %d\n", N, N);
    printf("线程数: %d\n", T);
    printf("迭代次数: %d (每次迭代2次stencil计算)\n", ITERATIONS);
    printf("总耗时: %.3f秒\n", elapsed);
    printf("每秒迭代次数: %.2f\n", ITERATIONS / elapsed);
    
    // 计算并输出网格中心区域的值
    printf("\n中心区域温度值:\n");
    for (int i = N/2 - 2; i <= N/2 + 2; i++) {
        for (int j = N/2 - 2; j <= N/2 + 2; j++) {
            printf("%6.2f ", grid[i][j]);
        }
        printf("\n");
    }
    
    // 清理资源
    barrier_destroy(&barrier);
    free_grid();
    
    return 0;
}
