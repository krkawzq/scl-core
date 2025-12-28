# Sparse Optimization

Sparse matrix optimization operations for linear algebra problems.

## Overview

Sparse optimization kernels provide:

- **Least Squares Solver** - Solve sparse least squares problems
- **Iterative Methods** - Convergence-based solvers
- **SIMD Optimization** - Vectorized operations
- **Parallel Processing** - Efficient for large systems

## Least Squares Solver

### sparse_least_squares

Solve sparse least squares problem: `min ||Ax - b||^2`

```cpp
#include "scl/kernel/sparse_opt.hpp"

Sparse<Real, true> A = /* ... */;      // Sparse matrix [n_rows x n_cols]
Array<const Real> b = /* ... */;        // Right-hand side [n_rows]
Array<Real> x(A.cols());                // Solution vector [n_cols]

// Standard solver
scl::kernel::sparse_opt::sparse_least_squares(A, b.ptr, A.rows(), A.cols(),
                                              x, max_iter = 100, tol = 1e-6);

// With custom tolerance
scl::kernel::sparse_opt::sparse_least_squares(A, b.ptr, A.rows(), A.cols(),
                                              x, max_iter = 200, tol = 1e-8);
```

**Parameters:**
- `A`: Sparse matrix (CSR format), size = n_rows × n_cols
- `b`: Right-hand side vector, size = n_rows
- `n_rows`: Number of rows
- `n_cols`: Number of columns
- `x`: Output solution vector, must be pre-allocated, size = n_cols
- `max_iter`: Maximum iterations (default: 100)
- `tol`: Convergence tolerance (default: 1e-6)

**Postconditions:**
- `x` contains approximate solution to `Ax ≈ b`
- Solution minimizes `||Ax - b||^2`
- Matrix A and vector b unchanged

**Algorithm:**
Iterative solver (typically conjugate gradient or LSQR):
1. Initialize solution vector
2. Iterate until convergence:
   - Compute residual: `r = b - Ax`
   - Update solution: `x = x + alpha * direction`
   - Check convergence: `||r|| < tol`
3. Return when converged or max_iter reached

**Complexity:**
- Time: O(max_iter * nnz) per iteration
- Space: O(n_cols) auxiliary for solution and workspace

**Thread Safety:**
- Safe - uses parallelized SpMV operations internally

**Use cases:**
- Linear regression with sparse design matrix
- Overdetermined systems (more equations than unknowns)
- Regularized least squares
- Matrix factorization problems

## Configuration

### Default Parameters

```cpp
namespace config {
    constexpr Real EPSILON = Real(1e-15);
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size SIMD_THRESHOLD = 16;
}
```

**Convergence:**
- Algorithm stops when residual norm < `tol`
- Numerical stability threshold: `EPSILON`

**Performance:**
- Parallel processing for matrices with > `PARALLEL_THRESHOLD` rows
- SIMD optimization for dense operations with > `SIMD_THRESHOLD` elements

## Examples

### Linear Regression

```cpp
#include "scl/kernel/sparse_opt.hpp"

// Design matrix (sparse features)
Sparse<Real, true> X = /* ... */;  // [n_samples x n_features]
Array<Real> y = /* ... */;          // Target values [n_samples]

// Solve: X * beta = y
Array<Real> beta(X.cols());
scl::kernel::sparse_opt::sparse_least_squares(X, y.ptr, X.rows(), X.cols(),
                                              beta, max_iter = 100, tol = 1e-6);

// beta contains regression coefficients
```

### Overdetermined System

```cpp
// More equations than unknowns: Ax = b where A is [m x n], m > n
Sparse<Real, true> A = /* ... */;  // [1000 x 100]
Array<Real> b = /* ... */;          // [1000]
Array<Real> x(100);                 // Solution [100]

scl::kernel::sparse_opt::sparse_least_squares(A, b.ptr, 1000, 100,
                                              x, max_iter = 200, tol = 1e-8);
```

## Performance Considerations

### Convergence

- Typically converges in 10-100 iterations for well-conditioned problems
- Poorly conditioned matrices may require more iterations or preconditioning
- Use `tol` to balance accuracy vs computation time

### Matrix Structure

- Best performance for sparse matrices (low density)
- CSR format required (row-major access pattern)
- Matrix-vector products are the dominant operation

### Parallelization

- SpMV operations are parallelized
- Multiple iterations are sequential (each depends on previous)
- Optimal for large, sparse systems

---

::: tip Convergence
For ill-conditioned systems, consider preprocessing (centering, scaling) or using a preconditioner. The default tolerance (1e-6) is usually sufficient for most applications.
:::

