// blas_utils.h - Basic BLAS-like operations for s-step GMRES
#ifndef BLAS_UTILS_H
#define BLAS_UTILS_H

#include <cmath>
#include <cstring>

// Dot product: return x^T * y
inline double vdot(int n, const double* x, const double* y) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i] * y[i];
    return sum;
}

// AXPY: y = a*x + y
inline void vaxpy(int n, double a, const double* x, double* y) {
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}

// Scale: x = a*x
inline void vscal(int n, double a, double* x) {
    for (int i = 0; i < n; i++) x[i] *= a;
}

// Copy: y = x
inline void vcopy(int n, const double* x, double* y) {
    std::memcpy(y, x, n * sizeof(double));
}

// Norm: return ||x||_2
inline double vnorm(int n, const double* x) {
    return std::sqrt(vdot(n, x, x));
}

// Zero: x = 0
inline void vzero(int n, double* x) {
    std::memset(x, 0, n * sizeof(double));
}

// Initialize: x[i] = val for all i
inline void vinit(int n, double val, double* x) {
    for (int i = 0; i < n; i++) x[i] = val;
}

#endif // BLAS_UTILS_H