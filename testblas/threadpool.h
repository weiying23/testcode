#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

class ThreadPool {
public:
    ThreadPool(size_t num_threads) : stop(false) { // 构造函数，创建工作线程
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] { // 创建线程并绑定执行函数
                while (true) { // 保持线程持续运行
                    std::function<void()> task; // 任务容器
                    {   //unique_lock自动管理锁的状态，配合条件变量实现阻塞等待
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        //条件变量等待：如果stop，或者任务队列非空则唤醒
                        this->condition.wait(lock, [this] { // 阻塞等待
                            return this->stop || !this->tasks.empty();
                        });
                        // 终止条件检查
                        if (this->stop && this->tasks.empty()) return;
                        //从队列获取任务（移动语义避免拷贝）
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    } // 自动锁释放

                    task(); // 执行任务
                }
            });
        }
    }

    // 任务入队方法（模板函数支持任意可调用对象）
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex); //加锁，保护队列线程安全
            tasks.emplace(std::forward<F>(f)); // 完美转发保持参数类型
        }
        condition.notify_one(); // 通知一个等待线程
    }

    // 析构函数：安全关闭线程池
    ~ThreadPool() {
        shutdown();
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true; // 设置停止标志
        }
        condition.notify_all(); // 唤醒所有线程
        
        // 等待所有线程结束
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    std::vector<std::thread> workers; // 工作线程容器
    std::queue<std::function<void()>> tasks; // 任务队列
    //同步原语
    std::mutex queue_mutex; // 队列互斥锁
    std::condition_variable condition; // 条件变量
    bool stop;
};

struct BlockTask {
    int block_i;        // 块的行索引
    int block_j;        // 块的列索引
    int block_size;     // 分块大小
    double* matA;           // 矩阵A指针
    double* matB;           // 矩阵B指针
    double* result;           // 结果矩阵C指针
    int mm;              // 矩阵维度
    int nn;              // 矩阵维度
    int kk;              // 矩阵维度
};

void transpose2D(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = i + 1; j < cols; ++j) {
            std::swap(matrix[i * cols + j], matrix[j * rows + i]);
        }
    }
}