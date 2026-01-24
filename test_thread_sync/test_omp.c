#include <stdio.h>
#include <pthread.h>

// 共享资源和互斥锁
int shared_data = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// 线程函数：增加共享数据
void* thread_func(void* arg) {
    int thread_id = *(int*)arg;
    
    // 获取锁
    pthread_mutex_lock(&mutex);
    
    // 临界区开始 - 同一时间只有一个线程可执行
    printf("线程 %d 进入临界区\n", thread_id);
    
    // 保存当前值并模拟一些处理
    int temp = shared_data;
    printf("线程 %d 读取值: %d\n", thread_id, temp);
    
    // 增加共享数据
    temp += 1;
    shared_data = temp;
    
    printf("线程 %d 写入值: %d\n", thread_id, temp);
    printf("线程 %d 离开临界区\n\n", thread_id);
    // 临界区结束
    
    // 释放锁
    pthread_mutex_unlock(&mutex);
    
    return NULL;
}

int main() {
    pthread_t threads[5];
    int thread_ids[5];
    
    printf("程序开始 - 初始共享数据值: %d\n\n", shared_data);
    
    // 创建5个线程
    for (int i = 0; i < 5; i++) {
        thread_ids[i] = i + 1;
        if (pthread_create(&threads[i], NULL, thread_func, &thread_ids[i]) != 0) {
            perror("创建线程失败");
            return 1;
        }
    }
    
    // 等待所有线程完成
    for (int i = 0; i < 5; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("等待线程失败");
            return 1;
        }
    }
    
    // 销毁互斥锁
    pthread_mutex_destroy(&mutex);
    
    printf("\n所有线程完成 - 最终共享数据值: %d\n", shared_data);
    
    return 0;
}
