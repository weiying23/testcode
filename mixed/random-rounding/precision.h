#ifndef PRECISION_SIMULATION_H
#define PRECISION_SIMULATION_H

#include <stdint.h>  // for uint32_t

// ============================================================================
// 精度格式定义 (文档 Table 4.1)
// ============================================================================

#define PRECISION_BINARY32  24   // 23 存储 + 1 隐含 (binary32/single)
#define PRECISION_BINARY16  11   // 10 存储 + 1 隐含 (binary16/half)
#define PRECISION_BFLOAT16  8    // 7 存储 + 1 隐含 (bfloat16/brain)

// 格式名称 (用于输出)
extern const char* format_names[];

// ============================================================================
// 【修复】函数前向声明 (Function Prototypes)
// 确保编译器在遇到函数调用前已知晓函数签名，解决 implicit declaration 错误
// ============================================================================

// 舍入函数
float stochastic_round(float x, int precision_bits);
float round_to_lowprec_rn(float x, int precision_bits);
float round_to_float32_rn(float x);

// PDE 1: 1D 热方程
float heat1d_initial(float x);
double heat1d_exact(double x, double t);
double compute_l2_error_1d(float* u, int nx, int ny, double t);
void solve_heat1d(float* u, int nx, int ny, int nt, float dt, int prec_bits, int use_sr, float* u_ref);

// PDE 2: Burgers 方程
float burgers_initial(float x);
double compute_l2_error_burgers(float* u, int nx, int ny, double t);
void solve_burgers(float* u, int nx, int ny, int nt, float dt, int prec_bits, int use_sr, float* u_ref);
int init_burgers_test(int nx, int nt, float dt);
void cleanup_burgers_test(void);
void run_burgers_comparison(const char* pde_name, int nx, int nt, float dt);

// PDE 3: 2D 热方程
float heat2d_initial(float x, float y);
double heat2d_exact(double x, double y, double t);
double compute_l2_error_2d(float* u, int nx, int ny, double t);
void solve_heat2d(float* u, int nx, int ny, int nt, float dt, int prec_bits, int use_sr, float* u_ref);

// 验证与测试
int verify_float32_simulation(void);
void test_stagnation(void);

// 通用运行器 (必须在 run_burgers_comparison 之前声明)
void run_pde_comparison(const char* pde_name, int nx, int ny, int nt, float dt, double final_t,
                        double (*compute_error)(float*, int, int, double),
                        void (*solve_func)(float*, int, int, int, float, int, int, float*));

#endif // PRECISION_SIMULATION_H
