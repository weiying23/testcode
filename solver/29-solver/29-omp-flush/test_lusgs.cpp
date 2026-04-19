#include <iostream>                     // 标准输入输出流库
#include <vector>                       // 动态数组容器
#include <cmath>                        // 数学函数库（abs等）
#include <cstdlib>                      // C标准库（atoi、getenv等）
#include <cstring>                      // C字符串操作（memset等）
#include <omp.h>                         // OpenMP并行编程库

// 类型定义：使用别名便于统一修改类型
typedef int IntType;                    // 整型索引类型
typedef double MATRIXTYPE;              // 矩阵元素类型（浮点数）

// 函数声明：前向 LUSGS 求解（下三角部分）
void forward_LUSGS(
    const IntType* __restrict row_ptr,  // CSR格式行指针数组
    const IntType* __restrict col_ind,  // CSR格式列索引数组
    const IntType* __restrict dia_ptr,  // 每行对角块在col_ind中的位置
    const MATRIXTYPE* __restrict matrix,// 矩阵非零元素值（块存储）
    const MATRIXTYPE* __restrict vec,   // 右侧向量（求解Ax=b中的b）
    MATRIXTYPE *&prod,                  // 解向量（输出）
    IntType n,                          // 矩阵块行数
    IntType nvar );                     // 每块的变量数（块大小nvar×nvar）

// 函数声明：后向 LUSGS 求解（上三角部分）
void backward_LUSGS(
    const IntType* __restrict row_ptr,  // CSR格式行指针数组
    const IntType* __restrict col_ind,  // CSR格式列索引数组
    const IntType* __restrict dia_ptr,  // 每行对角块在col_ind中的位置
    const MATRIXTYPE* __restrict matrix,// 矩阵非零元素值（块存储）
    const MATRIXTYPE* __restrict vec,   // 右侧向量
    MATRIXTYPE *&prod,                  // 解向量（输出）
    IntType n,                          // 矩阵块行数
    IntType nTCell,                     // 总单元数（用于同步数组大小）
    IntType nvar );                     // 每块的变量数

// 函数定义：稀疏块矩阵向量乘法 y = A * x（用于验证结果）
void spmv_block(
    const IntType* row_ptr,             // CSR行指针
    const IntType* col_ind,             // CSR列索引
    const MATRIXTYPE* matrix,           // 矩阵值数组
    const MATRIXTYPE* x,                // 输入向量
    MATRIXTYPE* y,                      // 输出向量
    IntType n,                          // 块行数
    IntType nvar)                       // 每块变量数
{
    // 遍历每一块行
    for (IntType i = 0; i < n; ++i) {
        // 将输出向量y的第i块初始化为0
        for (IntType jvar = 0; jvar < nvar; ++jvar)
            y[i*nvar + jvar] = 0.0;     // y[i块的第jvar个分量]=0

        // 遍历第i行的所有非零块（从row_ptr[i]到row_ptr[i+1])
        for (IntType idx = row_ptr[i]; idx < row_ptr[i+1]; ++idx) {
            IntType col = col_ind[idx]; // 当前块的列索引（对应x的哪个块）
            const MATRIXTYPE* block = &matrix[idx * nvar * nvar]; // 当前块的起始地址

            // 块矩阵乘法：y[i] += A[idx] * x[col]
            for (IntType jvar = 0; jvar < nvar; ++jvar) {         // y的每个分量
                for (IntType kvar = 0; kvar < nvar; ++kvar) {     // x的每个分量
                    // 累加：y[jvar] += block[jvar,kvar] * x[kvar]
                    y[i*nvar + jvar] += block[jvar*nvar + kvar] * x[col*nvar + kvar];
                }
            }
        }
    }
}

// 主函数：测试前向和后向LUSGS的正确性
int main() {
    // 从环境变量OMP_NUM_THREADS读取线程数，若未设置则使用最大可用线程数
    const char* env_threads = std::getenv("OMP_NUM_THREADS"); // 获取环境变量
    int num_threads = (env_threads != nullptr) ? std::atoi(env_threads) : omp_get_max_threads(); // 解析或默认
    omp_set_num_threads(num_threads);                        // 设置OpenMP线程数
    std::cout << "Using " << num_threads << " OpenMP threads.\n" << std::endl; // 打印线程数信息

    // ========== 定义问题规模参数 ==========
    const IntType n = 20;          // 块行数（矩阵有20个块行）
    const IntType nTCell = n;      // 单元数（与块行数相同，用于backward函数）
    const IntType nvar = 3;        // 每块大小（3x3的小块矩阵）

    // ========== 构造测试矩阵（块三对角矩阵）==========
    // 矩阵结构说明：
    // - 对角块：2I（单位矩阵乘以2，保证可逆）
    // - 下三角块：第i行有(i,i-1)位置的块，元素全为0.1
    // - 上三角块：第i行有(i,i+1)位置的块，元素全为0.1
    // 目的：使前向求解依赖左侧邻居，后向求解依赖右侧邻居

    // CSR格式数据结构
    std::vector<IntType> row_ptr(n+1);    // 行指针数组（n+1个元素，标记每行起始）
    std::vector<IntType> dia_ptr(n);       // 对角块位置数组（每行对角块的索引）
    std::vector<IntType> col_ind;          // 列索引数组（动态增长）
    std::vector<MATRIXTYPE> matrix_vals;   // 矩阵值数组（动态增长）

    // 预分配空间，提高效率
    col_ind.reserve(3 * n);                        // 每行最多3个块，预留3n空间
    matrix_vals.reserve(3 * n * nvar * nvar);      // 每个块有nvar*nvar个元素

    // 遍历每一行，构造CSR格式数据
    IntType nnz = 0;                               // 非零块计数器（也作为col_ind的索引）
    for (IntType i = 0; i < n; ++i) {
        row_ptr[i] = nnz;                          // 记录第i行的起始位置

        // ===== 1. 添加下三角块 (i, i-1) =====
        // 第0行没有下三角块（没有i-1），跳过
        if (i > 0) {
            col_ind.push_back(i - 1);              // 列索引为i-1
            // 添加nvar*nvar个元素值，全部设为0.1
            for (IntType j = 0; j < nvar * nvar; ++j)
                matrix_vals.push_back(0.1);        // 下三角块元素=0.1
            nnz++;                                 // 非零块计数加1
        }

        // ===== 2. 添加对角块 (i, i) =====
        dia_ptr[i] = nnz;                          // 记录对角块在col_ind中的位置
        col_ind.push_back(i);                      // 列索引为i（自身）
        // 先将所有元素设为0.0
        for (IntType j = 0; j < nvar * nvar; ++j)
            matrix_vals.push_back(0.0);            // 对角块初始为0
        // 再将对角元素设为2.0（构造2I）
        for (IntType d = 0; d < nvar; ++d)
            matrix_vals[nnz * nvar * nvar + d * nvar + d] = 2.0; // 对角线元素=2.0
        nnz++;                                     // 非零块计数加1

        // ===== 3. 添加上三角块 (i, i+1) =====
        // 第n-1行没有上三角块（没有i+1），跳过
        if (i + 1 < n) {
            col_ind.push_back(i + 1);              // 列索引为i+1
            // 添加nvar*nvar个元素值，全部设为0.1
            for (IntType j = 0; j < nvar * nvar; ++j)
                matrix_vals.push_back(0.1);        // 上三角块元素=0.1
            nnz++;                                 // 非零块计数加1
        }
    }
    row_ptr[n] = nnz;                              // 最后一行的结束位置（总非零块数）

    // 获取原始指针（用于函数调用，避免vector开销）
    const IntType* row_ptr_data = row_ptr.data();         // 行指针数组指针
    const IntType* col_ind_data = col_ind.data();         // 列索引数组指针
    const IntType* dia_ptr_data = dia_ptr.data();         // 对角块位置数组指针
    const MATRIXTYPE* matrix_data = matrix_vals.data();   // 矩阵值数组指针

    // ========== 构造右侧向量 vec ==========
    std::vector<MATRIXTYPE> vec(n * nvar, 1.0);   // 右侧向量，所有分量设为1.0

    // ========== 构造解向量容器 ==========
    std::vector<MATRIXTYPE> prod_serial(n * nvar, 0.0);   // 单线程结果向量
    std::vector<MATRIXTYPE> prod_parallel(n * nvar, 0.0); // 多线程结果向量

    // ========== 测试前向 LUSGS（下三角求解）==========
    std::cout << "=== Testing forward_LUSGS ===" << std::endl; // 打印测试标题

    // 单线程运行（用于对比验证）
    omp_set_num_threads(1);                       // 设置为单线程模式
    MATRIXTYPE* prod_ptr = prod_serial.data();    // 获取单线程解向量指针
    forward_LUSGS(row_ptr_data, col_ind_data, dia_ptr_data, matrix_data,
                  vec.data(), prod_ptr, n, nvar); // 调用前向求解
    std::cout << "Serial forward run finished." << std::endl; // 打印完成信息

    // 多线程运行
    omp_set_num_threads(num_threads);             // 设置为多线程模式
    prod_ptr = prod_parallel.data();              // 获取多线程解向量指针
    forward_LUSGS(row_ptr_data, col_ind_data, dia_ptr_data, matrix_data,
                  vec.data(), prod_ptr, n, nvar); // 调用前向求解
    std::cout << "Parallel forward run finished with " << num_threads << " threads." << std::endl;

    // ========== 对比单线程和多线程结果 ==========
    double max_diff = 0.0;                        // 最大差异值
    for (IntType i = 0; i < n * nvar; ++i) {
        double diff = std::abs(prod_serial[i] - prod_parallel[i]); // 计算绝对差异
        if (diff > max_diff) max_diff = diff;     // 更新最大差异
    }
    std::cout << "Max diff (serial vs parallel forward): " << max_diff << std::endl;

    // ========== 计算残差验证结果正确性 ==========
    std::vector<MATRIXTYPE> Ax(n * nvar);         // 存放矩阵向量乘积A*x
    spmv_block(row_ptr_data, col_ind_data, matrix_data, prod_parallel.data(), Ax.data(), n, nvar); // 计算A*x
    double residual = 0.0;                        // 残差平方和
    for (IntType i = 0; i < n * nvar; ++i) {
        double diff = Ax[i] - vec[i];             // 残差分量 = A*x - b
        residual += diff * diff;                  // 累加平方
    }
    std::cout << "Residual norm squared (forward): " << residual << std::endl;

    // 打印第一块解向量（用于人工检查）
    std::cout << "First block solution (parallel forward): ";
    for (IntType i = 0; i < nvar; ++i)
        std::cout << prod_parallel[i] << " ";     // 输出第0块的nvar个分量
    std::cout << "\n" << std::endl;

    // ========== 测试后向 LUSGS（上三角求解）==========
    std::cout << "=== Testing backward_LUSGS ===" << std::endl;

    // 重置解向量为0（后向求解前需要初始化）
    std::fill(prod_serial.begin(), prod_serial.end(), 0.0);   // 单线程向量清零
    std::fill(prod_parallel.begin(), prod_parallel.end(), 0.0); // 多线程向量清零

    // 单线程运行
    omp_set_num_threads(1);                       // 设置为单线程模式
    prod_ptr = prod_serial.data();                // 获取单线程解向量指针
    backward_LUSGS(row_ptr_data, col_ind_data, dia_ptr_data, matrix_data,
                   vec.data(), prod_ptr, n, nTCell, nvar); // 调用后向求解
    std::cout << "Serial backward run finished." << std::endl;

    // 多线程运行
    omp_set_num_threads(num_threads);             // 设置为多线程模式
    prod_ptr = prod_parallel.data();              // 获取多线程解向量指针
    backward_LUSGS(row_ptr_data, col_ind_data, dia_ptr_data, matrix_data,
                   vec.data(), prod_ptr, n, nTCell, nvar); // 调用后向求解
    std::cout << "Parallel backward run finished with " << num_threads << " threads." << std::endl;

    // ========== 对比单线程和多线程结果 ==========
    max_diff = 0.0;                               // 重置最大差异值
    for (IntType i = 0; i < n * nvar; ++i) {
        double diff = std::abs(prod_serial[i] - prod_parallel[i]); // 计算绝对差异
        if (diff > max_diff) max_diff = diff;     // 更新最大差异
    }
    std::cout << "Max diff (serial vs parallel backward): " << max_diff << std::endl;

    // ========== 计算残差验证结果正确性 ==========
    spmv_block(row_ptr_data, col_ind_data, matrix_data, prod_parallel.data(), Ax.data(), n, nvar); // 计算A*x
    residual = 0.0;                               // 重置残差平方和
    for (IntType i = 0; i < n * nvar; ++i) {
        double diff = Ax[i] - vec[i];             // 残差分量 = A*x - b
        residual += diff * diff;                  // 累加平方
    }
    std::cout << "Residual norm squared (backward): " << residual << std::endl;

    // 打印第一块解向量
    std::cout << "First block solution (parallel backward): ";
    for (IntType i = 0; i < nvar; ++i)
        std::cout << prod_parallel[i] << " ";     // 输出第0块的nvar个分量
    std::cout << std::endl;

    return 0;                                     // 程序正常退出
}

// ========== forward_LUSGS 函数实现（前向求解，下三角部分）==========
// 算法说明：求解 (D + L) * x = b，其中L是下三角，D是对角
// 公式：x[i] = D^{-1} * (b[i] - L[i] * x[已求解的邻居])
void forward_LUSGS(
    const IntType* __restrict row_ptr,  // CSR行指针
    const IntType* __restrict col_ind,  // CSR列索引
    const IntType* __restrict dia_ptr,  // 对角块位置
    const MATRIXTYPE* __restrict matrix,// 矩阵值
    const MATRIXTYPE* __restrict vec,   // 右侧向量b
    MATRIXTYPE *&prod,                  // 解向量x（输出）
    IntType n,                          // 块行数
    IntType nvar )                      // 每块变量数
{
    // ========== 定义循环范围和局部变量 ==========
    IntType start = 0;                          // 起始行索引
    IntType end = n;                            // 结束行索引
    const int NNUMBER = 5;                      // 块最大大小（用于静态数组）

    // ========== 创建同步标志数组 ==========
    bool *isReady = (bool *)malloc(n * sizeof(bool)); // 分配n个bool标记每行是否完成
    memset(isReady, false, n * sizeof(bool));         // 初始化全部为false（未完成）

    // ========== OpenMP 并行循环 ==========
    // schedule(dynamic)：动态调度，线程完成一个任务后自动获取下一个
    // shared(isReady)：isReady数组在所有线程间共享
    #pragma omp parallel for schedule(dynamic) shared(isReady)
    for (IntType iPoint = start; iPoint < end; iPoint++) {
        // ========== 局部变量声明 ==========
        IntType idx = iPoint * nvar;             // 当前块的解向量起始索引
        IntType iVar, jVar, kVar, col_j;         // 循环变量和列索引
        MATRIXTYPE low_prod[NNUMBER], block[NNUMBER*NNUMBER], weight; // 局部数组和工作变量

        // ========== 第一步：计算下三角贡献 L * x* ==========
        // low_prod 用于累加下三角部分对当前行的贡献
        memset(low_prod, 0, NNUMBER * sizeof(MATRIXTYPE)); // 初始化low_prod为0

        // 遍历当前行的下三角部分（row_ptr到dia_ptr之间的非零块）
        for (iVar = row_ptr[iPoint]; iVar < dia_ptr[iPoint]; iVar++) {
            col_j = col_ind[iVar];               // 获取依赖的邻居行索引

            // ========== 等待依赖行完成（同步点）==========
            // 自旋等待：直到邻居行col_j完成计算
            while (!isReady[col_j]) {
                #pragma omp flush(isReady)       // 强制从内存重新读取isReady
            }
            #pragma omp flush(prod)              // 强制从内存读取最新的prod值

            // ========== 累加下三角块贡献 ==========
            // low_prod += matrix[iVar] * prod[col_j]
            for (jVar = 0ul; jVar < nvar; jVar++) {           // 遍历输出分量
                for (kVar = 0ul; kVar < nvar; kVar++)         // 遍历输入分量
                    // 累加：low_prod[jVar] += A[jVar,kVar] * x[col_j,kVar]
                    low_prod[jVar] += matrix[iVar*nvar*nvar + jVar*nvar + kVar] * prod[col_j*nvar + kVar];
            }
        }

        // ========== 第二步：计算右侧修正 y = b - L * x* ==========
        for (iVar = 0; iVar < nvar; iVar++)
            low_prod[iVar] = vec[idx + iVar] - low_prod[iVar]; // y = b - L*x*

        // ========== 第三步：求解对角块方程 D * x = y ==========
        // 复制对角块到局部数组block（用于Gauss消元）
        for (iVar = 0ul; iVar < nvar * nvar; ++iVar)
            block[iVar] = matrix[dia_ptr[iPoint] * nvar * nvar + iVar]; // 复制D块

        // 定义宏简化二维索引访问：A(I,J) 表示 block[I行,J列]
        #define A(I,J) block[(I)*nvar + (J)]

        // ========== Gauss消元：将D变换为上三角矩阵 ==========
        // 外层循环：消去第iVar行以下的对角线左侧元素
        for (iVar = 1ul; iVar < nvar; iVar++) {
            // 内层循环：用第jVar行消去第iVar行的第jVar列元素
            for (jVar = 0ul; jVar < iVar; jVar++) {
                weight = A(iVar, jVar) / A(jVar, jVar); // 计算消元系数
                // 更新第iVar行：A[iVar,k] -= weight * A[jVar,k]
                for (kVar = jVar; kVar < nvar; kVar++)
                    A(iVar, kVar) -= weight * A(jVar, kVar);
                // 同时更新右侧向量：y[iVar] -= weight * y[jVar]
                low_prod[iVar] -= weight * low_prod[jVar];
            }
        }

        // ========== 回代求解：从最后一行向上求解x ==========
        for (iVar = nvar; iVar > 0ul;) {
            iVar--;                               // 递减索引（从nvar-1到0）
            // 减去已求解的上三角贡献
            for (jVar = iVar + 1; jVar < nvar; jVar++)
                low_prod[iVar] -= A(iVar, jVar) * low_prod[jVar];
            // 除以对角元素得到解
            low_prod[iVar] /= A(iVar, iVar);
        }
        #undef A                                   // 取消宏定义

        // ========== 第四步：将解写入全局数组 ==========
        for (iVar = 0; iVar < nvar; iVar++)
            prod[idx + iVar] = low_prod[iVar];    // 将局部解写入全局prod

        // ========== 第五步：标记当前行完成并通知其他线程 ==========
        isReady[iPoint] = true;                   // 设置完成标志为true
        #pragma omp flush(isReady)                // 强制写入内存，通知其他线程
        // 注意：这里缺少 #pragma omp flush(prod)，可能导致可见性问题
    }

    // ========== 释放同步数组 ==========
    free(isReady);                                // 释放malloc分配的内存
}

// ========== backward_LUSGS 函数实现（后向求解，上三角部分）==========
// 算法说明：求解 (D + U) * x = b，其中U是上三角，D是对角
// 公式：x[i] = D^{-1} * (D*x_old - U * x[已求解的邻居])
// 注意：从最后一行向前求解（逆序）
void backward_LUSGS(
    const IntType* __restrict row_ptr,  // CSR行指针
    const IntType* __restrict col_ind,  // CSR列索引
    const IntType* __restrict dia_ptr,  // 对角块位置
    const MATRIXTYPE* __restrict matrix,// 矩阵值
    const MATRIXTYPE* __restrict vec,   // 右侧向量（此函数未直接使用vec）
    MATRIXTYPE *&prod,                  // 解向量x（既是输入也是输出）
    IntType n,                          // 块行数
    IntType nTCell,                     // 总单元数（同步数组大小）
    IntType nvar )                      // 每块变量数
{
    // ========== 定义循环范围 ==========
    IntType begin = 0;                          // 起始行索引
    IntType end = n;                            // 结束行索引

    // ========== 创建同步标志数组 ==========
    bool *isReady = (bool *)malloc(nTCell * sizeof(bool)); // 分配同步数组
    memset(isReady, false, nTCell * sizeof(bool));         // 初始化为false

    const int NNUMBER = 5;                      // 块最大大小（用于静态数组）

    // ========== OpenMP 并行循环（逆序遍历）==========
    // 从end-1到begin（即从最后一行向前求解）
    #pragma omp parallel for schedule(dynamic) shared(isReady)
    for (IntType iPoint = end - 1; iPoint >= begin; iPoint--) {
        // ========== 局部变量声明 ==========
        MATRIXTYPE up_prod[nvar], dia_prod[nvar]; // 上三角贡献和对角贡献（VLA，非标准）
        MATRIXTYPE block[nvar * nvar];             // 对角块副本
        IntType idx = iPoint * nvar;               // 当前块起始索引
        IntType iVar, jVar, kVar, col_j;           // 循环变量

        // ========== 第一步：计算对角块贡献 D * x_old ==========
        // 注意：这里用的是当前prod的值（前向求解后的值）
        for (iVar = 0ul; iVar < nvar; iVar++) {
            dia_prod[iVar] = 0.0;                  // 初始化为0
            for (jVar = 0ul; jVar < nvar; jVar++) {
                // 累加：dia_prod[iVar] += D[iVar,jVar] * x[jVar]
                dia_prod[iVar] += matrix[dia_ptr[iPoint] * nvar * nvar + iVar * nvar + jVar] * prod[idx + jVar];
            }
        }

        // ========== 第二步：计算上三角贡献 U * x_new ==========
        // 初始化up_prod为0
        for (iVar = 0ul; iVar < nvar; iVar++)
            up_prod[iVar] = 0.0;

        // 遍历当前行的上三角部分（dia_ptr+1到row_ptr[i+1]之间的非零块）
        for (iVar = dia_ptr[iPoint] + 1; iVar < row_ptr[iPoint + 1]; iVar++) {
            col_j = col_ind[iVar];                 // 获取依赖的邻居行索引（右侧邻居）

            // ========== 等待依赖行完成（同步点）==========
            // 自旋等待：直到右侧邻居行完成计算
            while (!isReady[col_j]) {
                #pragma omp flush(isReady)         // 强制读取最新的isReady
            }
            #pragma omp flush(prod)                // 强制读取最新的prod

            // ========== 累加上三角块贡献 ==========
            for (jVar = 0ul; jVar < nvar; jVar++) {
                for (kVar = 0ul; kVar < nvar; kVar++)
                    // 累加：up_prod[jVar] += U[jVar,kVar] * x[col_j,kVar]
                    up_prod[jVar] += matrix[iVar * nvar * nvar + jVar * nvar + kVar] * prod[col_j * nvar + kVar];
            }
        }

        // ========== 第三步：计算修正后的右侧 y = D*x_old - U*x_new ==========
        for (iVar = 0; iVar < nvar; iVar++)
            up_prod[iVar] = dia_prod[iVar] - up_prod[iVar]; // y = D*x_old - U*x_new

        // ========== 第四步：求解对角块方程 D * x = y ==========
        // 复制对角块到局部数组
        for (iVar = 0ul; iVar < nvar * nvar; ++iVar)
            block[iVar] = matrix[dia_ptr[iPoint] * nvar * nvar + iVar];

        #define A(I,J) block[(I)*nvar + (J)]       // 定义索引宏

        // ========== Gauss消元 ==========
        for (iVar = 1ul; iVar < nvar; iVar++) {
            for (jVar = 0ul; jVar < iVar; jVar++) {
                MATRIXTYPE weight = A(iVar, jVar) / A(jVar, jVar); // 消元系数
                for (kVar = jVar; kVar < nvar; kVar++)
                    A(iVar, kVar) -= weight * A(jVar, kVar);       // 更新矩阵行
                up_prod[iVar] -= weight * up_prod[jVar];          // 更新右侧向量
            }
        }

        // ========== 回代求解 ==========
        for (iVar = nvar; iVar > 0ul;) {
            iVar--;                                 // 递减索引
            for (jVar = iVar + 1; jVar < nvar; jVar++)
                up_prod[iVar] -= A(iVar, jVar) * up_prod[jVar]; // 减去上三角贡献
            up_prod[iVar] /= A(iVar, iVar);         // 除以对角元素
        }
        #undef A                                     // 取消宏定义

        // ========== 第五步：将解写入全局数组 ==========
        for (iVar = 0; iVar < nvar; iVar++)
            prod[idx + iVar] = up_prod[iVar];       // 写入解

        // ========== 第六步：标记完成并通知其他线程 ==========
        isReady[iPoint] = true;                     // 设置完成标志
        #pragma omp flush(isReady)                  // 强制写入内存
        // 注意：这里缺少 #pragma omp flush(prod)
    }

    // ========== 释放同步数组 ==========
    free(isReady);                                  // 释放内存
}