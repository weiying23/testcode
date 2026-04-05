---
title: s-step GMRES Implementation Guide
date: 2026-04-05
source: arXiv:2001.04886v2 - "s-Step Orthomin and GMRES implemented on parallel computers"
---

# s-step GMRES Step-by-Step Implementation Guide

## Overview

s-step GMRES improves data locality and parallelism by grouping s consecutive GMRES iterations into one "super iteration". Instead of generating one orthogonal vector per iteration, we generate s vectors simultaneously using BLAS3 operations.

**Key Benefits**:
- Memory references reduced to 1/s of standard GMRES
- 2s inner products computed simultaneously (less global synchronization)
- Better cache utilization through block operations

---

## Part 1: Mathematical Foundation

### 1.1 Standard GMRES Recap

Given initial residual r0, GMRES(m) builds orthonormal basis {q1, q2, ..., qm} of Krylov subspace:
```
Km = span{r0, Ar0, A²r0, ..., A^(m-1)r0}
```

Each iteration:
- hi,j = qi^T * A*qj (i ≤ j)
- qj+1 = A*qj - Σ hi,j * qi (orthogonalization)
- Normalize qj+1

### 1.2 s-step Modification

Instead of one vector per iteration, generate s vectors:
```
V̄k = [v_k^1, v_k^2, ..., v_k^s]
where: v_k^i = A^(i-1) * v_k^1  (same Krylov sequence)
```

**Important**: Vectors within V̄k are NOT orthogonal to each other, but V̄1, V̄2, ..., V̄k are mutually orthogonal blocks.

---

## Part 2: Algorithm Structure

### 2.1 s-step Arnoldi (Algorithm 4.1)

```
Input: Initial vector v1^1, parameter s, total iterations m
Output: {V̄1, V̄2, ..., V̄m/s} and Hessenberg matrix Gk

1. Initialize:
   V̄1 = [v1^1, v1^2=Av1^1, v1^3=A²v1^1, ..., v1^s=A^(s-1)v1^1]

2. For k = 1 to m/s:
   (a) Scalar1: Decompose Wi = V̄i^T * V̄i for each previous block
       Solve Wi * h^q_ik = b^q_ik for q = 1,...,s

   (b) Compute v_{k+1}^1 = A*v_k^s - Σ_{i=1}^k V̄i * [h^1_i1, ..., h^s_ik]^T

   (c) Generate new block:
       V̄_{k+1} = [v_{k+1}^1, Av_{k+1}^1, A²v_{k+1}^1, ..., A^(s-1)v_{k+1}^1]

   (d) Compute inner products:
       - (A^i * v_k^1, v_l^j) for 1 ≤ i,j ≤ s, 1 ≤ l ≤ k-1
       - (A^i * v_k^1, A^j * v_k^1) for 0 ≤ i ≤ s-1, i ≤ j ≤ s

   (e) Scalar2: Solve Wi * t^q_ik = c^q_ik for q = 1,...,s-1

   (f) Orthogonalize:
       V̄_{k+1} = V̄_{k+1} - Σ_{i=1}^k V̄i * [0, t^1_ik, ..., t^(s-1)_ik]^T
```

### 2.2 s-step GMRES (Algorithm 4.2)

```
Input: Linear system Ax = b, initial guess x0, parameters s, m, tolerance ε
Output: Approximate solution xm

RESTART CYCLE:
1. Compute r0 = b - A*x0
2. Set v1^1 = r0 / ||r0||, β = ||r0||
3. Run s-step Arnoldi for k = 1,...,m/s:
   Generate {V̄1, V̄2, ..., V̄m/s} and matrix Gm
4. Solve least squares:
   ym = argmin ||β*e1 - L*Gm*y||
   where D = diag(V̄1^T V̄1, ..., V̄m^T V̄m) = L^T * L (Cholesky)
5. Compute solution:
   xm = x0 + Vm * ym
   where Vm = [V̄1, V̄2, ..., V̄m/s]
6. Check convergence:
   rm = b - A*xm
   If ||rm|| < ε: STOP
   Else: x0 = xm, r0 = rm, goto RESTART CYCLE
```

---

## Part 3: Implementation Steps

### Step 1: Basic Data Structures

```cpp
// Core structures
struct sstepGMRES {
    int n;          // Problem dimension
    int s;          // s-step parameter (typically 2-5)
    int m;          // Restart parameter (m/s blocks)
    int max_blocks; // m/s

    // Vectors
    double *x;      // Current solution (n)
    double *r;      // Current residual (n)
    double *work;   // Work vector (n)

    // Basis vectors: V̄1, V̄2, ..., V̄m/s
    // Each V̄k has s vectors, total storage: (m/s)*s*n = m*n
    double **Vbar;  // Vbar[k][j][i] or flattened: V[k*s + j][i]

    // Hessenberg matrix G: dimension (m+1) × m
    // But stored in block form
    double *G;      // Upper Hessenberg, column-major

    // Wi matrices: s×s for each block
    double **W;     // W[k] is s×s matrix

    // Least squares work arrays
    double *D_diag; // Diagonal blocks of D
    double *L;      // Cholesky factor
    double *y;      // Solution of least squares
};
```

### Step 2: Initialize First Block V̄1

```cpp
void init_first_block(sstepGMRES *solver, double *r0) {
    int n = solver->n;
    int s = solver->s;

    // v1^1 = r0 / ||r0||
    double beta = norm(r0, n);
    scale(solver->Vbar[0], r0, 1.0/beta, n);

    // Generate Krylov sequence: v1^j = A^(j-1) * v1^1
    for (int j = 1; j < s; j++) {
        matvec(solver->Vbar[j], solver->A, solver->Vbar[j-1], n);
    }

    // Compute W1 = V̄1^T * V̄1 (s×s matrix)
    compute_Wi(solver->W[0], solver->Vbar, 0, s, n);
}
```

### Step 3: Scalar1 - Orthogonalize v_{k+1}^1

```cpp
void scalar1(sstepGMRES *solver, int k) {
    int s = solver->s;
    int n = solver->n;

    // For each previous block i = 1,...,k
    for (int i = 0; i < k; i++) {
        // b^q_ik = V̄i^T * (A^q * v_{k+1}^1)
        for (int q = 0; q < s; q++) {
            // Compute A^q * v_{k+1}^1 (or use from step 3c)
            double *Aq_v = solver->temp[q];
            dot_products(solver->b[q], solver->Vbar[i], Aq_v, s, n);

            // Solve Wi * h^q_ik = b^q_ik
            solve_linear_system(solver->h[q], solver->W[i], solver->b[q], s);
        }

        // Accumulate orthogonalization
        // v_{k+1}^1 -= Σ_q Σ_j h^q_ij[k] * v_i^j
        for (int q = 0; q < s; q++) {
            for (int j = 0; j < s; j++) {
                axpy(v_new, -solver->h[q][j], solver->Vbar[i*s + j], n);
            }
        }
    }
}
```

### Step 4: Generate New Block V̄_{k+1}

```cpp
void generate_block(sstepGMRES *solver, int k) {
    int n = solver->n;
    int s = solver->s;
    int block_idx = k+1;

    // After Scalar1, we have v_{k+1}^1 (orthogonalized)
    // Generate: v_{k+1}^j = A^(j-1) * v_{k+1}^1
    for (int j = 1; j < s; j++) {
        matvec(solver->Vbar[block_idx*s + j],
               solver->A,
               solver->Vbar[block_idx*s + j-1],
               n);
    }
}
```

### Step 5: Compute Inner Products (Key for Scalar2)

```cpp
void compute_inner_products(sstepGMRES *solver, int k) {
    int s = solver->s;
    int n = solver->n;

    // Two sets of inner products:
    // 1. (A^i * v_k^1, v_l^j) for l < k (cross-block)
    // 2. (A^i * v_k^1, A^j * v_k^1) for i ≤ j (within-block)

    // Within-block: forms the Gram matrix for W_{k+1}
    for (int i = 0; i < s; i++) {
        for (int j = i; j < s; j++) {
            solver->inner_within[i][j] = dot(Vbar[k][i], Vbar[k][j], n);
            solver->inner_within[j][i] = solver->inner_within[i][j]; // symmetric
        }
    }

    // Cross-block: needed for Scalar2
    // These give coefficients for orthogonalizing V̄_{k+1} against V̄i
    for (int i = 0; i < k; i++) {
        for (int q = 0; q < s-1; q++) {
            // c^q_ik[j] = (A^{q+1} * v_{k+1}^1, v_i^j) or related
            compute_cross_inner_products(solver, i, k, q);
        }
    }
}
```

### Step 6: Scalar2 - Complete Block Orthogonalization

```cpp
void scalar2(sstepGMRES *solver, int k) {
    int s = solver->s;

    // For each previous block i and each q = 1,...,s-1
    // Solve: Wi * t^q_ik = c^q_ik

    for (int i = 0; i < k; i++) {
        for (int q = 0; q < s-1; q++) {
            solve_linear_system(solver->t[q], solver->W[i], solver->c[q], s);
        }
    }

    // Orthogonalize V̄_{k+1} against previous blocks
    // V̄_{k+1} -= Σ_i V̄i * T_i
    // where T_i = [0, t^1_i, t^2_i, ..., t^{s-1}_i]^T
}
```

### Step 7: Least Squares Solution

```cpp
void solve_least_squares(sstepGMRES *solver, int num_blocks) {
    int s = solver->s;
    int m = num_blocks * s;

    // Build D = diag(W1, W2, ..., Wk, ||v_{k+1}^1||²)
    // D is (m+1)×(m+1) block diagonal

    // Cholesky decomposition: D = L^T * L
    // Since D is block diagonal, L is also block diagonal
    // Each Wi needs its own Cholesky decomposition

    for (int i = 0; i < num_blocks; i++) {
        cholesky_decompose(solver->L_block[i], solver->W[i], s);
    }

    // Transform least squares:
    // min ||β*e1 - L*G*y||
    // This is a standard QR least squares problem on small system

    // Build transformed residual: β*e1 (only first element)
    // Build transformed matrix: L*G (apply Cholesky factor)

    // Solve using QR decomposition (Givens rotations recommended)
    solve_qr_least_squares(solver->y, solver->G_transformed, solver->resid, m+1, m);
}
```

### Step 8: Compute Solution

```cpp
void compute_solution(sstepGMRES *solver, int num_blocks) {
    int n = solver->n;
    int m = num_blocks * s;

    // xm = x0 + Vm * ym
    // Vm = [V̄1, V̄2, ..., V̄k] concatenated

    for (int j = 0; j < m; j++) {
        axpy(solver->x, solver->y[j], solver->Vbar[j], n);
    }
}
```

---

## Part 4: Practical Implementation Notes

### 4.1 Choosing s

Paper recommends: **s ≤ 5** for numerical stability.

Larger s causes loss of orthogonality within blocks. Solutions:
1. Keep s small (2-5)
2. Use Modified Gram-Schmidt within each block
3. Use polynomial basis (Chebyshev) instead of monomials A^i*r

### 4.2 Storage Requirements

| Method | Vector Storage |
|--------|----------------|
| GMRES(m) | A + (m+1) vectors |
| s-GMRES(m/s) | A + (m/s)*s + extra = A + m + ... |

Actually: need V̄1,...,V̄k plus v_{k+1}^1 = m/s * s + 1 = m+1 vectors (same as standard)

### 4.3 Operation Counts (per cycle)

| Operation | GMRES(ms) | s-GMRES(m) |
|-----------|-----------|------------|
| Dot products | ms + ms(ms+1)/2 | m(m-1)s²/2 + s(s+1)/2 + s |
| Mat-vec | ms+1 | s(m+1) |
| Vector updates | m²s²/2 + ms | m(m+1)s² |

**Key insight**: s-GMRES has fewer mat-vec operations but more dot products per block.

### 4.4 Stability Considerations

The paper notes that vectors v_k^1, ..., v_k^s within a block are not orthogonal, making the Wi matrices potentially ill-conditioned.

**Remedies**:
1. Use Chebyshev polynomial basis instead of {r, Ar, A²r, ...}
2. Apply MGS orthogonalization within each V̄k after generation
3. Use reorthogonalization if needed

### 4.5 MPI Parallel Implementation

Key parallel operations:
1. **Mat-vec**: Standard sparse matrix-vector product (already parallel)
2. **Inner products**: Group s² inner products together, compute in one MPI_Allreduce
3. **Scalar1/2**: Small s×s systems solved locally (no communication)
4. **Vector updates**: Local operations (AXPY)

Communication pattern:
- Standard GMRES: 1 global sync per iteration (for norm/inner products)
- s-GMRES: 1 global sync per s iterations (grouped inner products)

---

## Part 5: Implementation Checklist

### Phase 1: Basic Implementation

1. [ ] Implement standard GMRES(m) as reference
2. [ ] Implement data structures for s-step version
3. [ ] Implement first block generation (Krylov sequence)
4. [ ] Implement Wi computation (s×s Gram matrix)
5. [ ] Implement Scalar1 (orthogonalize new starting vector)
6. [ ] Implement block generation after Scalar1
7. [ ] Implement Scalar2 (complete orthogonalization)
8. [ ] Implement least squares solver
9. [ ] Implement restart mechanism
10. [ ] Test against standard GMRES for correctness

### Phase 2: Optimization

1. [ ] Use polynomial basis (Chebyshev) instead of monomials
2. [ ] Implement MGS within-block orthogonalization
3. [ ] Optimize memory layout for cache efficiency
4. [ ] Implement fused operations (reduce temporary vectors)

### Phase 3: Parallelization

1. [ ] Implement MPI parallel mat-vec
2. [ ] Group inner products for single Allreduce
3. [ ] Implement parallel vector norms
4. [ ] Benchmark and tune communication

---

## Part 6: Code Skeleton

```cpp
// Main s-step GMRES driver
int sstep_gmres(Matrix *A, double *b, double *x,
                int s, int m, double tol, int max_iter) {

    int num_blocks = m / s;
    sstepGMRES solver;
    init_solver(&solver, A, n, s, m);

    for (int restart = 0; restart < max_iter; restart++) {
        // Step 1: Initial residual
        compute_residual(r, A, b, x, n);
        double beta = norm(r, n);
        if (beta < tol) return SUCCESS;

        // Step 2: Initialize first block
        init_first_block(&solver, r);

        // Step 3: s-step Arnoldi loop
        for (int k = 0; k < num_blocks; k++) {
            // Generate new block
            generate_new_block(&solver, k);

            // Orthogonalize (Scalar1 + Scalar2)
            orthogonalize_block(&solver, k);

            // Update Hessenberg matrix
            update_hessenberg(&solver, k);
        }

        // Step 4: Solve least squares
        solve_least_squares(&solver, num_blocks);

        // Step 5: Update solution
        update_solution(x, &solver);

        // Check convergence
        compute_residual(r, A, b, x, n);
        if (norm(r, n) < tol) return SUCCESS;
    }

    return MAX_ITER_EXCEEDED;
}
```

---

## References

- Paper: arXiv:2001.04886v2 - "s-Step Orthomin and GMRES implemented on parallel computers"
- Original GMRES: Saad & Schultz, SIAM J. Sci. Stat. Comput., Vol. 7, 1986
- s-step methods: Chronopoulos & Gear, J. Comp. Appl. Math. 25, 1989