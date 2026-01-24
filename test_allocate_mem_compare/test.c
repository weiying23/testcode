#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS 4
#define ALLOCS_PER_THREAD 100000
#define ITERATIONS 1000000

//性能影响因素排序
//分配频率：频繁小分配 > 少量大分配
//碎片程度：高碎片 > 低碎片
//线程竞争：多线程竞争 > 单线程
//缓存友好性：随机访问 > 连续访问
//分配器选择：不同分配器性能差异可达2-5倍

// 从最重要到最不重要的优化
//1. 减少不必要的分配（重用对象）
//2. 使用栈分配小对象
//3. 批量分配代替逐个分配
//4. 选择合适的内存池策略
//5. 使用高效的分配器（tcmalloc/jemalloc）
//6. 优化数据结构布局
//7. 减少多线程竞争
//8. 监控和调优分配参数

void benchmark() {
    struct timeval start, end;
    
    // malloc性能测试
    gettimeofday(&start, NULL);
    for (int i = 0; i < ITERATIONS; i++) {
        int* p = malloc(sizeof(int));
        free(p);
    }
    gettimeofday(&end, NULL);
    printf("malloc/free: %.6f ns per pair\n", 
           ((end.tv_sec - start.tv_sec)*1e9 + 
            (end.tv_usec - start.tv_usec)*1000) / ITERATIONS);
}

// malloc的碎片化影响
void fragmentation_impact() {
    // 分配不同大小的块，模拟碎片
    struct timeval start, end;
    void* blocks[100];
    
    // 交错分配和释放不同大小的块
    gettimeofday(&start, NULL);
    for (int i = 0; i < 100; i++) {
        size_t size = (i % 10 + 1) * 16;  // 16-160字节
        blocks[i] = malloc(size);
    }
    gettimeofday(&end, NULL);
    printf("stride malloc: %.6f ns \n", 
           ((end.tv_sec - start.tv_sec)*1e9 + 
            (end.tv_usec - start.tv_usec)*1000) / 100);
    
    // 释放奇数索引的块
    gettimeofday(&start, NULL);
    for (int i = 1; i < 100; i += 2) {
        free(blocks[i]);
        blocks[i] = NULL;
    }
    gettimeofday(&end, NULL);
    printf("free odd blocks: %.6f ns \n", 
           ((end.tv_sec - start.tv_sec)*1e9 + 
            (end.tv_usec - start.tv_usec)*1000) / 100);
    
    // 现在尝试分配一个中等大小的块 - 可能变慢
    gettimeofday(&start, NULL);
    void* p = malloc(12800);  // 分配器需要搜索合适的空闲块
    free(p);
    gettimeofday(&end, NULL);
    printf("malloc a pre-difined block - 128B: %.6f ns \n", 
           ((end.tv_sec - start.tv_sec)*1e9 + 
            (end.tv_usec - start.tv_usec)*1000) / 100);
}

//calloc vs malloc性能对比
void calloc_vs_malloc_perf() {
    const size_t SIZE = 100 * sizeof(int);
    //小内存可能malloc快
    //大内存可能calloc快
    
    // malloc + 手动清零
    clock_t start = clock();
    int* p1 = malloc(SIZE);
    if (p1) memset(p1, 0, SIZE);  // 额外开销
    clock_t end = clock();
    printf("malloc+memset: %.3f ms\n", 
           (double)(end-start)*1000/CLOCKS_PER_SEC);
    
    // calloc（单步完成）
    start = clock();
    int* p2 = calloc(1000000, sizeof(int));
    end = clock();
    printf("calloc: %.3f ms\n", 
           (double)(end-start)*1000/CLOCKS_PER_SEC);
    
    free(p1);
    free(p2);
}

//realloc的性能考虑
void realloc_performance_issue() {
    int* arr = NULL;
    size_t capacity = 10;
    
    // 低效方式：频繁realloc
    for (int i = 0; i < 1000; i++) {
        arr = realloc(arr, (i + 1) * sizeof(int));  // 每次增长1
        arr[i] = i;
    }
    
    // 高效方式：几何增长策略
    int* arr2 = NULL;
    size_t cap = 0;
    size_t len = 0;
    
    for (int i = 0; i < 1000; i++) {
        if (len >= cap) {
            cap = cap ? cap * 2 : 16;  // 几何增长
            arr2 = realloc(arr2, cap * sizeof(int));
        }
        arr2[len++] = i;
    }
    
    free(arr);
    free(arr2);
}
//realloc可能导致内存复制（当需要移动时）
//频繁小幅度增长效率极低
//几何增长（2倍）可分摊复制开销

//测试锁竞争问题
void* thread_func(void* arg) {
    for (int i = 0; i < ALLOCS_PER_THREAD; i++) {
        // 每个线程都在竞争全局分配器的锁
        void* p = malloc(64);
        // 模拟工作
        *(int*)p = i;
        free(p);
    }
    return NULL;
}
void lock_contention_test() {
    pthread_t threads[THREADS];
    
    clock_t start = clock();
    
    for (int i = 0; i < THREADS; i++) {
        pthread_create(&threads[i], NULL, thread_func, NULL);
    }
    
    for (int i = 0; i < THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_t end = clock();
    printf("多线程分配: %.3f ms\n", 
           (double)(end-start)*1000/CLOCKS_PER_SEC);
}
//使用线程本地缓存（tcmalloc, jemalloc）
//批量分配
//避免频繁小分配

//测试锁竞争问题


//  缓存友好性分析，测试分配模式对缓存的影响
void cache_locality_test() {
    const int N = 10000000;
    int** ptrs = malloc(N * sizeof(int*));
    
    // 情况1：连续分配，连续访问
    for (int i = 0; i < N; i++) {
        ptrs[i] = malloc(sizeof(int)); // 分配N个独立的int
    }
    //内存块在堆中的位置是随机的，可能散布在堆的不同位置
    //指针本身是连续存储
    //导致大量缓存未命中
    
    clock_t start = clock();
    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += *ptrs[i];  // 通过指针间接访问,可能缓存不友好
    }
    clock_t end = clock();
    printf("随机指针访问: %.3f ms\n", 
           (double)(end-start)*1000/CLOCKS_PER_SEC);
    
    // 情况2：数组连续访问
    int* arr = malloc(N * sizeof(int)); // 一次性分配连续内存
    start = clock();
    sum = 0;
    for (int i = 0; i < N; i++) {
        sum += arr[i];  // 直接访问连续内存,缓存友好
    }
    end = clock();
    printf("连续数组访问: %.3f ms\n", 
           (double)(end-start)*1000/CLOCKS_PER_SEC);
    
    // 清理
    for (int i = 0; i < N; i++) free(ptrs[i]);
    free(ptrs);
    free(arr);
}
// TLB缓存虚拟地址到物理地址的映射
// 连续内存访问TLB命中率高

// 情况1：大量随机内存块
// 每个块可能在不同内存页
// 导致大量TLB未命中，需要查询页表

// 情况2：连续数组
// 大部分访问在同一内存页内
// TLB命中率高，性能好


// 简单内存池实现
typedef struct MemoryPool {
    char* block;
    size_t block_size;
    size_t used;
    struct MemoryPool* next;
} MemoryPool;
MemoryPool* pool_create(size_t block_size) {
    MemoryPool* pool = malloc(sizeof(MemoryPool));
    pool->block = malloc(block_size);
    pool->block_size = block_size;
    pool->used = 0;
    pool->next = NULL;
    return pool;
}
void* pool_alloc(MemoryPool* pool, size_t size) {
    // 简单实现：从当前块分配
    if (pool->used + size <= pool->block_size) {
        void* ptr = pool->block + pool->used;
        pool->used += size;
        return ptr;
    }
    return malloc(size);  // 回退到普通malloc
}
// 简单内存池实现




int main() {
    benchmark();
    fragmentation_impact();
    calloc_vs_malloc_perf();
    lock_contention_test();
    cache_locality_test();

    return 0;
}