#define __ARM_FEATURE_SVE 1
#define __ARM_FEATURE_SME 1
#include <cstdint>
#include <arm_sme.h>
#include <arm_sve.h>
#include <iostream>
#include <cstdlib>

#define START_SMSTREAM_ZA() asm volatile("smstart za")
#define STOP_SMSTREAM_ZA() asm volatile("smstop za")
// Transpose the A matrix
// -- Only needed if transpose is not already available from a previous calculation
// –- SVL-agnostic; operates properly for any allowed SVL
#define M 16
#define N 16
//float32_t A[M*P]; float32_t AT[P*M]; svfloat32_t ZrowBuffer; // M rows x P columns

// A, transposed: P rows x M columns
// Z register. svfloat32_t is equivalent to
// float32x16_t on Apple silicon, but varies by SVL
// on other platforms

__attribute__((target_version("sme2"))) void transpose(float32_t* aPtr, float32_t* atPtr) __arm_streaming 
{
	uint64_t tile_size = svcntsb() / sizeof(float); // SME图块大小（元素数）
    uint64_t vec_size = svcntw();                   // SVE向量大小（32位元素数）
	svbool_t pg = svptrue_b32();

    __arm_new_za();

	// 分块处理矩阵
    for (uint64_t tile_row = 0; tile_row < M; tile_row += tile_size) {
        uint64_t rows_in_tile = tile_size;
        if (tile_row + tile_size > M) {
            rows_in_tile = M - tile_row;
        }
        
        for (uint64_t tile_col = 0; tile_col < N; tile_col += tile_size) {
            uint64_t cols_in_tile = tile_size;
            if (tile_col + tile_size > N) {
                cols_in_tile = N - tile_col;
            }
            
            // 加载当前图块到ZA寄存器
            for (uint64_t r = 0; r < rows_in_tile; r++) {
                // 创建行谓词
                uint64_t base_idx = tile_col;
                svbool_t row_pg = svwhilelt_b32(base_idx, base_idx + cols_in_tile);
                
                // 加载行数据
                svfloat32_t row = svld1(row_pg, &aPtr[(tile_row + r) * N + base_idx]);
                
                // 写入ZA水平方向（原始行）
                svwrite_hor_za32_m(0, r, row_pg, row);
            }
            
            // 从垂直方向读取（实现转置）
            for (uint64_t c = 0; c < cols_in_tile; c++) {
                // 创建列谓词
                uint64_t base_idx = tile_row;
                svbool_t col_pg = svwhilelt_b32(base_idx, base_idx + rows_in_tile);
                
                // 从ZA垂直方向读取（转置后的行）
                svfloat32_t transposed_row = svread_ver_za32(0, c, col_pg);
                
                // 存储到转置矩阵
                svst1(col_pg, &atPtr[(tile_col + c) * M + base_idx], transposed_row);
            }
        }
    }
	
}

int main(){
	float32_t A[M*N]; float32_t AT[N*M];
	for (int i = 0; i < M; i++) {
        for (int  j = 0; j < N; j++) {
            A[i * N + j] = i * 10.0f + j;  // 唯一值
        }
    } 
	transpose(A, AT);
	return 0;
}
