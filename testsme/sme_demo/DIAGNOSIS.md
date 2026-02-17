# Illegal Instruction Error - Root Cause Analysis

## Problem Summary

The code in `sme_matrix/matrix_multiply.cpp` fails with **"Illegal instruction: 4"** when executed on Apple M4.

## Root Cause

**Apple M4 does NOT support ARM's Scalable Matrix Extension (SME).**

Your code uses SME-specific instructions:
```cpp
__asm__ volatile("SMSTART ZA");  // Illegal on Apple M4
__asm__ volatile("SMSTOP ZA");   // Illegal on Apple M4
```

And SME intrinsics:
- `svld1_hor_za64()` - Load into ZA tile matrix
- `svread_ver_za64_f64_m()` - Read from ZA tile
- `svmopa_za64_f64_m()` - Matrix outer product accumulate
- `svst1_hor_za64()` - Store from ZA tile

## Why This Happens

### What is SME?

ARM's **Scalable Matrix Extension (SME)** is a hardware feature for matrix operations, introduced in ARMv9-A architecture. It provides:
- ZA tile matrix register (up to 2048 bits wide)
- Streaming SVE mode
- Matrix multiply-accumulate instructions
- Designed for AI/ML workloads

### Apple's Approach

Apple Silicon (M1/M2/M3/M4) uses a **different architecture**:
- **AMX (Apple Matrix coprocessor)** - Proprietary matrix accelerator
- **NEON** - Standard ARM SIMD (supported)
- **SVE/SME** - NOT supported (ARM optional extensions)

Apple chose NOT to implement SME, instead using their proprietary AMX which is accessed via the **Accelerate framework**.

## Evidence

### Binary Analysis

```bash
$ otool -tV test | grep -A 5 test_kernel
```

Shows SME control instructions:
```assembly
msr S0_3_C4_C7_3, xzr  # SMSTART - Sets streaming mode
msr S0_3_C4_C6_3, xzr  # SMSTOP - Exits streaming mode
```

These system register writes are **illegal** on Apple M4 because the hardware doesn't implement the FEAT_SME feature.

### Compilation

Your build command:
```bash
clang++ -march=armv9-a+sve+sme-f64f64 ...
```

This tells the compiler to generate SME instructions, which the CPU cannot execute.

## Solutions

### Option 1: Use Apple Accelerate Framework (RECOMMENDED)

Replace SME code with Apple's optimized BLAS:

```cpp
#include <Accelerate/Accelerate.h>

void gemm_accelerate(double *A, double *B, double *C, 
                     int M, int N, int K, double alpha) {
    double beta = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
}
```

**Performance**: Accelerate uses AMX hardware and achieves 200+ GFLOPS on M4.

### Option 2: Use NEON Intrinsics

Write portable ARM SIMD code:

```cpp
#include <arm_neon.h>

// 2x2 kernel with NEON float64x2_t vectors
// See: ../dgemm_m4/dgemm.c for complete example
```

**Performance**: 4-5 GFLOPS (slower but portable)

### Option 3: Run on Real SME Hardware

This code requires ARM Neoverse V2 or newer CPUs:
- AWS Graviton4 instances
- NVIDIA Grace CPUs
- Ampere Altra Max (SVE only, not SME)

Apple M-series chips will NEVER support SME.

### Option 4: Remove SME, Keep Portable Code

If you just want it to compile and run (slowly):

```cpp
// Comment out SME-specific code
void gemmkernel(double *mata, double *matb, double *matc, 
                int M, int N, int K, double alpha) {
    // Remove: __arm_streaming attribute
    // Remove: SMSTART/SMSTOP
    // Remove: All sv* intrinsics
    
    // Add simple triple loop
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            double sum = 0.0;
            for(int k = 0; k < K; k++) {
                sum += mata[i*K + k] * matb[k*N + j];
            }
            matc[i*N + j] = alpha * sum;
        }
    }
}
```

## Quick Fix to Test

If you just want to verify the build system works:

```cpp
// In matrix_multiply.cpp
void test_kernel() {
    // Remove SME code, just print
    printf("SME not available on Apple M4\n");
    printf("In streaming_mode: %d\n", 0);
}

void gemmkernel(double *mata, double *matb, double *matc, 
                int M, int N, int K, double alpha) {
    printf("Using fallback implementation\n");
    // Add naive implementation above
}
```

Then recompile WITHOUT SME flags:
```bash
clang++ -O3 -std=c++20 matrix.cpp matrix_multiply.cpp -o test
```

## Recommended Next Steps

1. **For production on Apple M4**: Use the Accelerate framework example in `dgemm_m4/`
2. **For learning SME**: Get access to AWS Graviton4 or other SME-capable hardware
3. **For portable code**: Write version that detects CPU features at runtime

## Summary

| Feature | Apple M4 | ARM Neoverse V2+ |
|---------|----------|------------------|
| SME     | ❌ NO    | ✅ YES           |
| SVE     | ❌ NO    | ✅ YES           |
| AMX     | ✅ YES   | ❌ NO            |
| NEON    | ✅ YES   | ✅ YES           |

Your code is written for ARM Neoverse V2+ architecture, not Apple Silicon.
