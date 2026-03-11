/**
 * @file hotspot.c
 * @brief ILU (Incomplete LU) 预条件子的前向和后向代入求解
 *
 * 用于迭代求解大型稀疏线性方程组 Ax = b
 * 典型应用于计算流体力学 (CFD) 中的压力 - 速度耦合求解
 *
 * 算法说明:
 * - forward_compute:  前向代入，求解 L * y = b (从 0 到 n-1)
 * - backward_compute: 后向代入，求解 U * x = y (从 n-1 到 0)
 * 两者结合构成对称 Gauss-Seidel 预条件子
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cassert>
#include <cstring>      // memset
#include <algorithm>
#include "omp.h"        // OpenMP 并行库

// 用于 C++ 调用时的 extern "C"
#ifdef __cplusplus
extern "C" {
#endif

#define MATRIXTYPE float  // 矩阵元素数据类型
#define IntType int       // 索引数据类型

/**
 * @brief 前向代入求解 (Forward Substitution)
 *
 * 求解下三角系统：L * prod = vec
 *
 * @param row_ptr   [in]  CSR 格式行指针，长度 n+1
 * @param col_ind   [in]  CSR 格式列索引，长度为非零块数
 * @param dia_ptr   [in]  对角块起始位置数组，长度 n
 * @param matrix    [in]  稀疏矩阵数据 (块状存储，每块 nvar x nvar)
 * @param vec       [in]  右端向量
 * @param prod      [out] 输出结果向量 (解)
 * @param n         [in]  网格点数/块数
 * @param nvar      [in]  每个网格点的变量数 (如 3 个速度分量)
 *
 * @note 前向代入必须按顺序执行 (从 0 到 n-1)，因为每个点依赖于前面的点
 * @note 这是一个串行实现，并行版本需要更复杂的拓扑排序和调度策略
 */
void forward_compute(const IntType *row_ptr, const IntType *col_ind, const IntType *dia_ptr,
    const MATRIXTYPE *matrix, const MATRIXTYPE *vec, MATRIXTYPE *prod, IntType n, IntType nvar)
{
    const int NNUMBER = 5;  // 最大变量数 (用于静态数组分配)

    // 串行遍历所有网格点 (必须保持顺序以保证依赖关系)
    for (IntType iPoint = 0; iPoint < n; iPoint++) {

        IntType idx = iPoint * nvar;  // 当前点在 prod 中的起始索引
        IntType iVar, jVar, kVar, col_j;

        // low_prod: 存储下三角部分的累积结果
        // block:    存储当前点的对角块矩阵 (用于 LU 分解)
        MATRIXTYPE low_prod[NNUMBER], block[NNUMBER * NNUMBER], weight;

        // --- 步骤 1: 初始化低阶乘积项 ---
        for (iVar = 0; iVar < nvar; iVar++)
            low_prod[iVar] = 0.0;

        // --- 步骤 2: 累积下三角部分贡献 (L * prod) ---
        // 遍历当前行之前的所有非零块 (row_ptr 到 dia_ptr 之间)
        for (iVar = row_ptr[iPoint]; iVar < dia_ptr[iPoint]; iVar++) {
            col_j = col_ind[iVar];

            // 跳过无效列索引 (边界检查)
            if (col_j < 0 || col_j >= iPoint) continue;

            // 块矩阵乘法：low_prod += matrix[iVar] * prod[col_j]
            for (jVar = 0; jVar < nvar; jVar++) {
                for (kVar = 0; kVar < nvar; kVar++)
                    low_prod[jVar] += matrix[iVar * nvar * nvar + jVar * nvar + kVar]
                                    * prod[col_j * nvar + kVar];
            }
        }

        // --- 步骤 3: 计算右端项 (vec - L*prod) ---
        for (iVar = 0; iVar < nvar; iVar++)
            low_prod[iVar] = vec[idx + iVar] - low_prod[iVar];

        // --- 步骤 4: 提取对角块矩阵 ---
        for (iVar = 0; iVar < nvar * nvar; ++iVar)
            block[iVar] = matrix[dia_ptr[iPoint] * nvar * nvar + iVar];

        // --- 步骤 5: 对角块 LU 分解 (高斯消元转为上三角) ---
        #define A(I,J) block[(I)*nvar+(J)]  // 宏定义简化二维访问

        // 将 block 转换为上三角矩阵 (原地 LU 分解)
        for (iVar = 1; iVar < nvar; iVar++) {
            for (jVar = 0; jVar < iVar; jVar++) {
                weight = A(iVar, jVar) / A(jVar, jVar);  // 消元系数
                // 更新当前行
                for (kVar = jVar; kVar < nvar; kVar++) {
                    A(iVar, kVar) -= weight * A(jVar, kVar);
                }
                // 同步更新右端项
                low_prod[iVar] -= weight * low_prod[jVar];
            }
        }

        // --- 步骤 6: 回代求解 (Backwards Substitution) ---
        // 从最后一行开始反向求解上三角系统
        for (iVar = nvar; iVar > 0;) {
            iVar--;  // 无符号类型递减
            // 减去已知变量的贡献
            for (jVar = iVar + 1; jVar < nvar; jVar++) {
                low_prod[iVar] -= A(iVar, jVar) * low_prod[jVar];
            }
            // 除以对角元得到解
            low_prod[iVar] /= A(iVar, iVar);
        }
        #undef A

        // --- 步骤 7: 写入结果 ---
        for (iVar = 0; iVar < nvar; iVar++)
            prod[idx + iVar] = low_prod[iVar];
    }
}

/**
 * @brief 后向代入求解 (Backward Substitution)
 *
 * 求解上三角系统：U * prod = vec
 *
 * @param row_ptr   [in]  CSR 格式行指针，长度 n+1
 * @param col_ind   [in]  CSR 格式列索引，长度为非零块数
 * @param dia_ptr   [in]  对角块起始位置数组，长度 n
 * @param matrix    [in]  稀疏矩阵数据 (块状存储，每块 nvar x nvar)
 * @param vec       [in]  右端向量
 * @param prod      [out] 输出结果向量 (解)
 * @param n         [in]  网格点数/块数
 * @param nTCell    [in]  总单元数 (当前未使用，保留接口)
 * @param nvar      [in]  每个网格点的变量数
 *
 * @note 后向代入必须按逆序执行 (从 n-1 到 0)，因为每个点依赖于后面的点
 * @note 这是一个串行实现，并行版本需要更复杂的拓扑排序和调度策略
 */
void backward_compute(IntType *row_ptr, IntType *col_ind, IntType *dia_ptr,
    MATRIXTYPE *matrix, MATRIXTYPE *vec, MATRIXTYPE *prod,
    IntType n, IntType nTCell, IntType nvar)
{
    const int NNUMBER = 5;  // 最大变量数 (用于静态数组分配)

    // 串行逆序遍历所有网格点 (必须保持顺序以保证依赖关系)
    for (IntType iPoint = n - 1; iPoint >= 0; iPoint--) {

        // up_prod:   存储上三角部分的累积结果
        // dia_prod:  存储对角块与当前解的乘积
        // block:     存储当前点的对角块矩阵 (用于 LU 分解)
        MATRIXTYPE up_prod[NNUMBER], dia_prod[NNUMBER];
        MATRIXTYPE block[NNUMBER * NNUMBER];

        IntType idx = iPoint * nvar;  // 当前点在 prod 中的起始索引
        IntType iVar, jVar, kVar, col_j;

        // --- 步骤 1: 计算对角块贡献 (D * prod) ---
        for (iVar = 0; iVar < nvar; iVar++) {
            dia_prod[iVar] = 0.0;
            for (jVar = 0; jVar < nvar; jVar++) {
                dia_prod[iVar] += matrix[dia_ptr[iPoint] * nvar * nvar + iVar * nvar + jVar]
                                * prod[idx + jVar];
            }
        }

        // --- 步骤 2: 初始化上三角乘积项 ---
        for (iVar = 0; iVar < nvar; iVar++)
            up_prod[iVar] = 0.0;

        // --- 步骤 3: 累积上三角部分贡献 (U * prod) ---
        // 遍历当前点之后的所有非零块 (dia_ptr+1 到 row_ptr[i+1])
        for (iVar = dia_ptr[iPoint] + 1; iVar < row_ptr[iPoint + 1]; iVar++) {
            col_j = col_ind[iVar];

            // 跳过无效列索引 (边界检查)
            if (col_j < 0 || col_j <= iPoint) continue;

            // 块矩阵乘法：up_prod += matrix[iVar] * prod[col_j]
            for (jVar = 0; jVar < nvar; jVar++) {
                for (kVar = 0; kVar < nvar; kVar++)
                    up_prod[jVar] += matrix[iVar * nvar * nvar + jVar * nvar + kVar]
                                   * prod[col_j * nvar + kVar];
            }
        }

        // --- 步骤 4: 计算右端项 (D*prod - U*prod) ---
        for (iVar = 0; iVar < nvar; iVar++)
            up_prod[iVar] = dia_prod[iVar] - up_prod[iVar];

        // --- 步骤 5: 提取对角块矩阵 ---
        for (iVar = 0; iVar < nvar * nvar; ++iVar)
            block[iVar] = matrix[dia_ptr[iPoint] * nvar * nvar + iVar];

        // --- 步骤 6: 对角块 LU 分解 (高斯消元转为上三角) ---
        #define A(I,J) block[(I)*nvar+(J)]  // 宏定义简化二维访问

        // 将 block 转换为上三角矩阵 (原地 LU 分解)
        for (iVar = 1; iVar < nvar; iVar++) {
            for (jVar = 0; jVar < iVar; jVar++) {
                MATRIXTYPE weight = A(iVar, jVar) / A(jVar, jVar);  // 消元系数
                // 更新当前行
                for (kVar = jVar; kVar < nvar; kVar++) {
                    A(iVar, kVar) -= weight * A(jVar, kVar);
                }
                // 同步更新右端项
                up_prod[iVar] -= weight * up_prod[jVar];
            }
        }

        // --- 步骤 7: 回代求解 (Backwards Substitution) ---
        // 从最后一行开始反向求解上三角系统
        for (iVar = nvar; iVar > 0;) {
            iVar--;  // 无符号类型递减
            // 减去已知变量的贡献
            for (jVar = iVar + 1; jVar < nvar; jVar++) {
                up_prod[iVar] -= A(iVar, jVar) * up_prod[jVar];
            }
            // 除以对角元得到解
            up_prod[iVar] /= A(iVar, iVar);
        }
        #undef A

        // --- 步骤 8: 写入结果 ---
        for (iVar = 0; iVar < nvar; iVar++)
            prod[idx + iVar] = up_prod[iVar];
    }
}

#ifdef __cplusplus
}
#endif
