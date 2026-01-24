//测试图调度算法
#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <chrono>
#include <unordered_map>
#include <random>
#include <iomanip>

using namespace std;

// 线程池类，用于管理一组工作线程
class ThreadPool {
public:
    // 构造函数，创建指定数量的工作线程
    ThreadPool(size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            // 创建工作线程
            workers.emplace_back([this] {
                while (true) {
                    function<void()> task;
                    {
                        // 使用互斥锁保护任务队列
                        unique_lock<mutex> lock(queue_mutex);
                        // 等待条件：线程池停止或任务队列非空
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });
                        
                        // 如果线程池已停止且任务队列为空，则线程退出
                        if (stop && tasks.empty()) {
                            return;
                        }
                        
                        // 从队列中取出一个任务
                        task = move(tasks.front());
                        tasks.pop();
                    }
                    // 执行任务
                    task();
                }
            });
        }
    }
    
    // 将任务添加到任务队列中
    template<class F>
    void enqueue(F&& f) {
        {
            // 使用互斥锁保护任务队列
            unique_lock<mutex> lock(queue_mutex);
            // 将任务添加到队列
            tasks.emplace(forward<F>(f));
        }
        // 通知一个等待的工作线程
        condition.notify_one();
    }
    
    // 析构函数：停止线程池并等待所有工作线程结束
    ~ThreadPool() {
        {
            unique_lock<mutex> lock(queue_mutex);
            stop = true; // 设置停止标志
        }
        condition.notify_all(); // 通知所有工作线程
        for (thread& worker : workers) {
            worker.join(); // 等待所有工作线程结束
        }
    }

private:
    vector<thread> workers; // 工作线程集合
    queue<function<void()>> tasks; // 任务队列
    mutex queue_mutex; // 保护任务队列的互斥锁
    condition_variable condition; // 条件变量，用于线程间通信
    bool stop; // 停止标志
};

// 图调度器类，用于管理有依赖关系的任务执行
class GraphScheduler {
public:
    // 构造函数：初始化任务数量、依赖关系和线程池大小
    GraphScheduler(int num_tasks, const vector<pair<int, int>>& dependencies, int num_workers)
        : n(num_tasks), indegree(num_tasks + 1, 0), completed(0) {
        
        // 构建任务依赖图（邻接表）和入度表
        for (const auto& dep : dependencies) {
            int u = dep.first, v = dep.second;
            // u -> v 表示任务u依赖于任务v（v必须在u之前完成）
            graph[u].push_back(v);
            indegree[v]++; // 增加任务v的入度
        }
        
        // 创建线程池，指定工作线程数量
        pool = make_unique<ThreadPool>(num_workers);
    }
    
    // 启动调度器
    void run() {
        // 将初始入度为0的任务（没有前置依赖的任务）加入线程池
        for (int i = 1; i <= n; ++i) {
            if (indegree[i] == 0) {
                enqueue_task(i);
            }
        }
        
        // 等待所有任务完成
        unique_lock<mutex> lock(mtx);
        // 当所有任务完成时，条件变量会通知
        cv_finished.wait(lock, [this] { 
            return completed == n; 
        });
    }
    
private:
    // 将任务加入线程池队列
    void enqueue_task(int task_id) {
        pool->enqueue([this, task_id] {
            execute_task(task_id);
        });
    }
    
    // 执行任务
    void execute_task(int task_id) {
        // 模拟任务执行时间 (100-500ms)
        this_thread::sleep_for(chrono::milliseconds(100 + rand() % 400));
        
        // 输出任务执行信息（使用互斥锁保护输出）
        {
            lock_guard<mutex> lock(output_mutex);
            cout << "任务 " << task_id << " 由线程 " 
                 << this_thread::get_id() << " 执行完成" << endl;
        }
        
        // 处理后续任务
        vector<int> next_tasks;
        {
            // 使用互斥锁保护共享数据结构（入度表和完成计数）
            lock_guard<mutex> lock(mtx);
            // 遍历当前任务的所有后继任务
            for (int neighbor : graph[task_id]) {
                // 减少后继任务的入度
                if (--indegree[neighbor] == 0) {
                    // 如果入度变为0，表示该任务的所有前置已完成
                    next_tasks.push_back(neighbor);
                }
            }
            // 增加已完成任务计数
            completed++;
        }
        
        // 将新就绪的任务加入线程池
        for (int next : next_tasks) {
            enqueue_task(next);
        }
        
        // 检查是否所有任务都已完成
        if (completed == n) {
            // 通知主线程所有任务完成
            cv_finished.notify_one();
        }
    }
    
    // 成员变量
    int n; // 任务总数
    unordered_map<int, vector<int>> graph; // 邻接表，存储任务依赖关系
    vector<int> indegree; // 每个任务的入度（前置任务数量）
    atomic<int> completed; // 已完成的任务数（原子操作）
    unique_ptr<ThreadPool> pool; // 线程池指针
    
    mutex mtx; // 保护共享数据（indegree和completed）的互斥锁
    mutex output_mutex; // 保护输出流的互斥锁
    condition_variable cv_finished; // 用于通知所有任务完成的条件变量
};

int main() {
    // 设置随机种子
    srand(static_cast<unsigned>(time(nullptr)));
    
    cout << "=== 多线程图调度系统 ===" << endl;
    
    // 定义任务依赖关系
    int num_tasks = 12;
    // 依赖关系对 (u, v) 表示任务u依赖于任务v（即v必须在u之前完成）
    vector<pair<int, int>> dependencies = {
        {1, 2}, {1, 3},     // 任务1依赖于任务2和3
        {2, 4}, {2, 5},     // 任务2依赖于任务4和5
        {3, 6}, {3, 7},     // 任务3依赖于任务6和7
        {4, 8},             // 任务4依赖于任务8
        {5, 8}, {5, 9},     // 任务5依赖于任务8和9
        {6, 9}, {6, 10},    // 任务6依赖于任务9和10
        {7, 10}, {7, 11},   // 任务7依赖于任务10和11
        {8, 12},            // 任务8依赖于任务12
        {9, 12},            // 任务9依赖于任务12
        {10, 12},           // 任务10依赖于任务12
        {11, 12}            // 任务11依赖于任务12
    };
    
    // 创建调度器，指定任务数量、依赖关系和线程池大小（4个工作线程）
    GraphScheduler scheduler(num_tasks, dependencies, 8);
    
    // 记录开始时间
    auto start = chrono::high_resolution_clock::now();
    
    // 运行调度器
    scheduler.run();
    
    // 记录结束时间
    auto end = chrono::high_resolution_clock::now();
    // 计算总耗时
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "\n所有任务完成! 总耗时: " << duration.count() << " ms" << endl;
    
    return 0;
}