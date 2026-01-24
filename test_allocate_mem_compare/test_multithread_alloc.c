// thread_local_pool_replacement.c
// 文件名：线程本地池替换测试程序

#include <stdio.h>      // 标准输入输出，用于打印结果
#include <stdlib.h>     // 标准库，提供malloc/free等
#include <pthread.h>    // POSIX线程库，用于多线程编程
#include <time.h>       // 时间函数，用于获取时间
#include <string.h>     // 字符串操作函数
#include <sys/time.h>   // 系统时间，用于高精度计时
#include <stdint.h>     // 标准整数类型定义，如uintptr_t

// ============ 配置 ============
#define NUM_THREADS     8      // 测试使用的线程数
#define ALLOCS_PER_THREAD 500000  // 每个线程执行内存分配/释放操作的次数
#define OBJECT_SIZE     64     // 每次分配的内存对象大小（字节）
#define LOCAL_POOL_SIZE 20000  // 每个线程本地内存池的容量（对象个数）

// ============ 工具函数 ============
// 获取当前时间（毫秒）
double get_time_ms() {
    struct timeval tv;           // 时间结构体（秒+微秒）
    gettimeofday(&tv, NULL);     // 获取当前时间
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;  // 转换为毫秒
}

// ============ 1. 标准malloc ============
// 使用系统默认的malloc分配内存
void* std_malloc(size_t size) {
    return malloc(size);         // 调用标准库malloc
}

// 使用系统默认的free释放内存
void std_free(void* ptr) {
    free(ptr);                   // 调用标准库free
}

// ============ 2. 线程本地缓存(TLS)分配器 ============
// 注释：让每个线程拥有变量的独立副本的机制，而不是所有线程共享同一个变量

// 定义线程本地缓存结构体
typedef struct TLSCache {
    void* cache[100];  // 固定大小的简单缓存数组，存储空闲对象指针
    int count;         // 当前缓存中的对象数量
} TLSCache;

// 声明线程特定数据键（用于存储每个线程的缓存）
static pthread_key_t tls_key;
// 一次性初始化控制变量（确保键只创建一次）
static pthread_once_t tls_once = PTHREAD_ONCE_INIT;

// 创建线程特定数据键的函数（只会执行一次）
static void create_tls_key() {
    pthread_key_create(&tls_key, NULL);  // 创建键，不设置析构函数
}

// 获取当前线程的TLSCache指针
static TLSCache* get_tls_cache() {
    // 确保键只创建一次
    pthread_once(&tls_once, create_tls_key); //确保全局资源只初始化一次
    // 获取当前线程的TLSCache指针
    TLSCache* cache = pthread_getspecific(tls_key);
    // 如果当前线程还没有缓存，则创建一个
    if (!cache) {
        cache = malloc(sizeof(TLSCache));  // 分配TLSCache结构体
        cache->count = 0;                  // 初始化计数为0
        pthread_setspecific(tls_key, cache);  // 将缓存绑定到当前线程
    }
    return cache;  // 返回缓存指针
}

// TLS分配器的分配函数实现
static void* tls_malloc_impl(size_t size) {
    TLSCache* cache = get_tls_cache();  // 获取当前线程的缓存
    
    // 如果缓存中有空闲对象，直接返回最后一个
    if (cache->count > 0) {
        cache->count--;  // 减少缓存计数
        return cache->cache[cache->count];  // 返回缓存中的对象
    }
    
    // 缓存为空，批量分配100个对象
    void* batch = malloc(100 * size);  // 分配一个大块内存
    if (!batch) return NULL;           // 分配失败返回NULL
    
    // 将批量分配的内存分割成独立对象，放入缓存（除了第一个对象直接返回）
    // 注意：第一个对象（i=0）将作为本次分配的返回值
    for (int i = 1; i < 100; i++) {
        // 计算第i个对象的地址，存入缓存
        cache->cache[cache->count] = (char*)batch + i * size;
        cache->count++;  // 缓存计数增加
    }
    
    // 返回批量分配内存的起始地址（第一个对象）
    return batch;
}

// TLS分配器的释放函数实现
static void tls_free_impl(void* ptr) {
    TLSCache* cache = get_tls_cache();  // 获取当前线程的缓存
    
    // 如果缓存未满，将对象放回缓存
    if (cache->count < 100) {
        // 将对象指针放入缓存数组
        cache->cache[cache->count] = ptr;
        cache->count++;  // 缓存计数增加
    } else {
        // 缓存已满，直接释放内存
        free(ptr);
    }
}

// ============ 3. 每个线程独立的内存池 ============
// 线程本地内存池结构体定义
typedef struct ThreadLocalPool {
    void** free_list;           // 空闲对象链表（指针数组）
    char* memory_block;         // 预先分配的大内存块
    int free_count;             // 当前空闲对象数量
    int capacity;               // 内存池容量（对象个数）
    int total_allocated;        // 总分配计数（用于调试统计）
} ThreadLocalPool;

// 线程本地池指针（使用__thread关键字，每个线程有独立副本）
static __thread ThreadLocalPool* thread_pool = NULL;

// 初始化线程本地池的函数
static void init_thread_local_pool() {
    if (thread_pool) return;  // 如果已经初始化，直接返回
    
    // 分配ThreadLocalPool结构体内存
    thread_pool = malloc(sizeof(ThreadLocalPool));
    if (!thread_pool) {
        fprintf(stderr, "线程本地池分配失败\n");
        exit(1);  // 分配失败，退出程序
    }
    
    // 初始化结构体字段
    thread_pool->capacity = LOCAL_POOL_SIZE;
    thread_pool->free_count = 0;
    thread_pool->total_allocated = 0;
    thread_pool->free_list = NULL;
    
    // 分配大内存块（足够容纳所有对象）
    // 每个对象需要OBJECT_SIZE + sizeof(void*)字节（额外的指针用于链表连接）
    size_t total_size = LOCAL_POOL_SIZE * (OBJECT_SIZE + sizeof(void*));
    thread_pool->memory_block = malloc(total_size);
    
    // 检查内存块是否分配成功
    if (!thread_pool->memory_block) {
        free(thread_pool);      // 失败时释放结构体
        thread_pool = NULL;     // 将指针设为NULL
        fprintf(stderr, "线程本地池内存分配失败\n");
        return;
    }
    
    // 初始化空闲链表（这里采用懒初始化，首次分配时再填充）
    thread_pool->free_list = NULL;
}

// 线程本地池分配函数
static void* thread_local_pool_malloc(size_t size) {
    (void)size;  // 显式忽略参数（避免编译器警告），我们使用固定的OBJECT_SIZE
    
    // 如果线程本地池未初始化，先初始化
    if (!thread_pool) {
        init_thread_local_pool();
        // 初始化失败，回退到标准malloc
        if (!thread_pool) {
            return malloc(OBJECT_SIZE);
        }
    }
    
    // 如果本地池为空且空闲链表为空，进行懒初始化填充
    if (thread_pool->free_count == 0 && thread_pool->free_list == NULL) {
        // 计算批量填充的大小
        int batch_size = LOCAL_POOL_SIZE;
        char* batch = thread_pool->memory_block;  // 内存块起始地址
        
        // 构建空闲链表：将每个对象链接起来
        for (int i = 0; i < batch_size; i++) {
            // 计算第i个对象的地址
            void* obj = batch + i * (OBJECT_SIZE + sizeof(void*));
            // 将对象插入空闲链表头部
            *(void**)obj = thread_pool->free_list;  // 新对象的next指向当前链表头
            thread_pool->free_list = obj;           // 链表头更新为新对象
            thread_pool->free_count++;              // 空闲计数增加
        }
    }
    
    // 从空闲链表分配对象（如果有）
    if (thread_pool->free_count > 0) {
        void* ptr = thread_pool->free_list;           // 获取链表头（要分配的对象）
        thread_pool->free_list = *(void**)ptr;        // 链表头指向下一个对象
        thread_pool->free_count--;                    // 空闲计数减少
        thread_pool->total_allocated++;               // 总分配计数增加
        return ptr;                                   // 返回分配的对象
    }
    
    // 池为空（不应该发生，因为容量足够），回退到malloc
    thread_pool->total_allocated++;
    return malloc(OBJECT_SIZE);
}

// 线程本地池释放函数
static void thread_local_pool_free(void* ptr) {
    // 如果线程池未初始化或指针为空，直接调用标准free
    if (!thread_pool || !ptr) {
        free(ptr);
        return;
    }
    
    // 检查指针是否在本地池的内存块范围内
    char* pool_start = thread_pool->memory_block;
    char* pool_end = pool_start + LOCAL_POOL_SIZE * (OBJECT_SIZE + sizeof(void*));
    
    if ((char*)ptr >= pool_start && (char*)ptr < pool_end) {
        // 指针在内存池范围内，将其放回空闲链表
        *(void**)ptr = thread_pool->free_list;  // 将释放的对象指向当前链表头
        thread_pool->free_list = ptr;           // 链表头更新为刚释放的对象
        thread_pool->free_count++;              // 空闲计数增加
    } else {
        // 指针不在内存池中（可能是回退分配的对象），直接调用标准free
        free(ptr);
    }
}

// 清理线程本地池（线程结束时调用）
static void cleanup_thread_local_pool() {
    if (thread_pool) {
        free(thread_pool->memory_block);  // 释放内存块
        free(thread_pool);                // 释放池结构体
        thread_pool = NULL;               // 指针设为NULL
    }
}

// ============ 4. TCMalloc包装器 ============
// 如果定义了USE_TCMALLOC，则包含TCMalloc头文件并包装其函数
#ifdef USE_TCMALLOC
#include <tcmalloc.h>

// TCMalloc分配函数包装
static void* tc_malloc_wrap(size_t size) {
    return tc_malloc(size);  // 调用TCMalloc的分配函数
}

// TCMalloc释放函数包装
static void tc_free_wrap(void* ptr) {
    tc_free(ptr);  // 调用TCMalloc的释放函数
}
#endif

// ============ 测试线程函数 ============
// 线程参数结构体
typedef struct {
    const char* name;            // 分配器名称
    void* (*alloc_func)(size_t); // 分配函数指针
    void (*free_func)(void*);    // 释放函数指针
    int thread_id;               // 线程ID
    int allocs_per_thread;       // 每个线程的分配次数
    double* time_ms;             // 指向存储线程执行时间的指针
    double* throughput;          // 指向存储线程吞吐量的指针
} ThreadArgs;

// 获取线程特定的随机数种子
static unsigned int get_thread_seed(int thread_id) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    // 组合时间、线程ID和线程指针生成种子
    return (unsigned int)(tv.tv_sec ^ tv.tv_usec ^ (uintptr_t)pthread_self() ^ thread_id);
}

// 测试线程的入口函数
static void* test_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;      // 获取线程参数
    unsigned int seed = get_thread_seed(args->thread_id);  // 获取种子（未使用）
    struct timeval start, end;                // 计时变量
    
    gettimeofday(&start, NULL);               // 记录开始时间
    
    // 预热阶段：执行少量分配/释放，避免冷启动影响
    for (int i = 0; i < 100; i++) {
        void* p = args->alloc_func(OBJECT_SIZE);  // 分配
        if (p) args->free_func(p);                // 释放
    }
    
    // 正式测试阶段：执行指定次数的分配/释放
    for (int i = 0; i < args->allocs_per_thread; i++) {
        // 分配内存
        void* ptr = args->alloc_func(OBJECT_SIZE);
        if (!ptr) {
            // 分配失败，打印错误信息
            fprintf(stderr, "%s (线程%d): 分配失败\n", args->name, args->thread_id);
            continue;
        }
        
        // 简单访问内存：写入和读取，防止编译器优化掉分配/释放操作
        *(int*)ptr = i;                 // 写入数据
        volatile int sink = *(int*)ptr; // 读取数据（volatile防止优化）
        (void)sink;                     // 避免未使用变量警告
        
        // 释放内存
        args->free_func(ptr);
    }
    
    gettimeofday(&end, NULL);  // 记录结束时间
    
    // 计算线程执行时间（毫秒）
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                       (end.tv_usec - start.tv_usec) / 1000.0;
    
    // 将结果存入线程参数指向的位置
    *args->time_ms = elapsed_ms;
    // 计算吞吐量（每秒操作数）
    *args->throughput = (args->allocs_per_thread * 1000.0) / elapsed_ms;
    
    return NULL;  // 线程退出
}

// ============ 分配器定义 ============
// 分配器结构体：包含名称和函数指针
typedef struct {
    const char* name;            // 分配器名称
    void* (*alloc_func)(size_t); // 分配函数
    void (*free_func)(void*);    // 释放函数
} Allocator;

// ============ 主测试函数 ============
// 对指定的分配器进行性能测试
static void benchmark_allocator(Allocator* alloc, int num_threads) {
    pthread_t threads[num_threads];         // 线程ID数组
    ThreadArgs args[num_threads];           // 线程参数数组
    double thread_times[num_threads];       // 存储每个线程的执行时间
    double thread_throughputs[num_threads]; // 存储每个线程的吞吐量
    
    printf("测试 %s:\n", alloc->name);
    
    double start_total = get_time_ms();  // 记录测试开始的总时间
    
    // 创建并启动所有测试线程
    for (int i = 0; i < num_threads; i++) {
        // 设置线程参数
        args[i].name = alloc->name;
        args[i].alloc_func = alloc->alloc_func;
        args[i].free_func = alloc->free_func;
        args[i].thread_id = i;
        args[i].allocs_per_thread = ALLOCS_PER_THREAD;
        args[i].time_ms = &thread_times[i];
        args[i].throughput = &thread_throughputs[i];
        
        // 创建线程，执行test_thread函数
        pthread_create(&threads[i], NULL, test_thread, &args[i]);
    }
    
    // 等待所有线程完成
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    double end_total = get_time_ms();  // 记录测试结束的总时间
    
    // 计算总体统计信息
    double total_time = end_total - start_total;  // 总耗时
    double avg_thread_time = 0;                   // 平均线程时间
    double total_throughput = 0;                  // 总吞吐量
    
    // 累加所有线程的时间和吞吐量
    for (int i = 0; i < num_threads; i++) {
        avg_thread_time += thread_times[i];
        total_throughput += thread_throughputs[i];
    }
    
    avg_thread_time /= num_threads;  // 计算平均线程时间
    
    // 计算总分配次数
    long long total_allocs = (long long)num_threads * ALLOCS_PER_THREAD;
    // 计算整体吞吐量（每秒操作数）
    double overall_throughput = total_allocs / (total_time / 1000.0);
    
    // 打印测试结果
    printf("  总时间: %.2f ms, 总吞吐量: %.0f ops/s\n", 
           total_time, overall_throughput);
    printf("  平均线程时间: %.2f ms, 平均线程吞吐量: %.0f ops/s\n\n",
           avg_thread_time, total_throughput / num_threads);
}

// ============ 主函数 ============
int main() {
    // 打印测试配置信息
    printf("========== 多线程内存分配器性能对比测试 ==========\n");
    printf("配置: %d线程, 每线程%d次分配, 对象大小%d字节\n\n",
           NUM_THREADS, ALLOCS_PER_THREAD, OBJECT_SIZE);
    printf("线程本地池大小: %d 个对象\n\n", LOCAL_POOL_SIZE);
    
    // 1. 测试标准malloc
    Allocator std_alloc = {
        .name = "标准malloc",
        .alloc_func = std_malloc,
        .free_func = std_free
    };
    benchmark_allocator(&std_alloc, NUM_THREADS);
    
    // 2. 测试线程本地缓存(TLS)分配器
    Allocator tls_alloc = {
        .name = "线程本地缓存(TLS)",
        .alloc_func = tls_malloc_impl,
        .free_func = tls_free_impl
    };
    benchmark_allocator(&tls_alloc, NUM_THREADS);
    
    // 3. 测试线程独立内存池
    Allocator thread_pool_alloc = {
        .name = "线程独立内存池",
        .alloc_func = thread_local_pool_malloc,
        .free_func = thread_local_pool_free
    };
    benchmark_allocator(&thread_pool_alloc, NUM_THREADS);
    
    // 4. 测试TCMalloc（如果编译时定义了USE_TCMALLOC）
#ifdef USE_TCMALLOC
    Allocator tc_alloc = {
        .name = "TCMalloc",
        .alloc_func = tc_malloc_wrap,
        .free_func = tc_free_wrap
    };
    benchmark_allocator(&tc_alloc, NUM_THREADS);
#else
    // 如果没有定义USE_TCMALLOC，提示如何启用
    printf("TCMalloc: 未启用 (编译时使用 -DUSE_TCMALLOC 和 -ltcmalloc)\n\n");
#endif
    
    printf("测试完成！\n");
    return 0;  // 程序正常退出
}