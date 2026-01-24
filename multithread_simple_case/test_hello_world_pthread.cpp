// pthread实现小demo
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

clock_t start1, end1, start2, end2, start3, end3;
clock_t start4, end4, start5, end5, start6, end6;
 
int counter = 0;
int index = 0;
pthread_mutex_t mutex;
 
void* increase_counter(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        //pthread_mutex_lock(&mutex); // 加锁
        counter += i-100;
        //pthread_mutex_unlock(&mutex); // 解锁
    }
    return NULL;
}

void* calculate_sum(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        //pthread_mutex_lock(&mutex); // 加锁
        counter += i-100;
        //pthread_mutex_unlock(&mutex); // 解锁
    }
    return NULL;
}
 
int main() {
    pthread_t thread[10];

    //pthread_mutex_init(&mutex, NULL); // 初始化互斥锁
 
    start1 = clock();
    for (int loop = 0; loop < 10; loop++){
        pthread_create(&thread[loop], NULL, increase_counter, NULL);
    }
    for (int loop = 0; loop < 10; loop++){
        pthread_join(thread[loop], NULL);
    }
    end1 = clock();

    printf("Counter: %d\n", counter); // 
    //pthread_mutex_destroy(&mutex); // 销毁互斥锁



    start2 = clock();
    for (int loop = 0; loop < 10; loop++){
        for (int i = 0; i < 1000000; i++) {
            index += i-100;
        }
    }
    printf("Index: %d\n", index); // 
    end2 = clock();

    printf("pthread takes: %lf ; while loop takes %lf\n", double(end1-start1), double(end2-start2)); //

    return 0;
}

