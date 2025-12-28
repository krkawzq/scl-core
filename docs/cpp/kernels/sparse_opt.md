# sparse_opt.hpp

> scl/kernel/sparse_opt.hpp · Sparse matrix optimization operations

## Overview

This file provides sparse matrix optimization operations for linear algebra problems, specifically sparse least squares solvers using iterative methods.

This file provides:
- Sparse least squares solver using iterative methods
- Convergence-based optimization
- SIMD-optimized sparse matrix-vector operations
- Parallel processing support

**Header**: `#include "scl/kernel/sparse_opt.hpp"`

---

## Main APIs

### sparse_least_squares

::: source_code file="scl/kernel/sparse_opt.hpp" symbol="sparse_least_squares" collapsed
:::

**Algorithm Description**

Solve sparse least squares problem: minimize ||Ax - b||^2 using iterative methods:

1. **Initialization**: Initialize solution vector x (typically to zero)

2. **Iterative refinement**: For each iteration up to max_iter:
   - Compute residual: r = b - A*x (sparse matrix-vector multiplication)
   - Compute step direction (typically gradient or conjugate gradient)
   - Update solution: x = x + alpha * step_direction
   - Check convergence: if ||r|| < tol, exit early

3. **Convergence check**: Terminate if:
   - Residual norm < tolerance (tol)
   - Maximum iterations reached

4. **Optimization**: Uses parallelized SpMV (sparse matrix-vector multiply) for efficiency

**Edge Cases**

- **Empty matrix**: If A has no rows, returns zero solution
- **Zero right-hand side**: If b = 0, returns zero solution
- **Singular matrix**: May not converge if A is singular (ill-conditioned)
- **No convergence**: Returns best solution found after max_iter iterations

**Data Guarantees (Preconditions)**

- `A` must be valid CSR sparse matrix
- `x` must have capacity >= n_cols (pre-allocated)
- `b` must have length >= n_rows
- `max_iter > 0`
- `tol > 0`

**Complexity Analysis**

- **Time**: O(max_iter * nnz)
  - O(nnz) per iteration for SpMV
  - max_iter iterations (typically 100)
- **Space**: O(n_cols) auxiliary space for solution vector
  - Additional O(n_rows) for residual vector

**Example**

```cpp
#include "scl/kernel/sparse_opt.hpp"

// Sparse matrix A: n_rows x n_cols
Sparse<Real, true> A = /* ... */;
Index n_rows = A.rows();
Index n_cols = A.cols();

// Right-hand side vector
Array<Real> b(n_rows);
// ... fill b ...

// Pre-allocate solution vector
Array<Real> x(n_cols);

// Solve least squares problem
scl::kernel::sparse_opt::sparse_least_squares(
    A,
    b.data(),
    n_rows,
    n_cols,
    x,
    100,      // max_iter
    1e-6      // tol
);

// x now contains approximate solution to min ||Ax - b||^2
```

---

## Notes

**Convergence**:
- Typical convergence requires 10-100 iterations depending on condition number
- Smaller tolerance requires more iterations
- Ill-conditioned matrices may not converge

**Performance**:
- Uses parallelized SpMV for efficiency
- SIMD-optimized where applicable
- Memory-efficient sparse storage

**Typical Usage**:
- Solve overdetermined systems (n_rows > n_cols)
- Regularization problems
- Linear regression with sparse features

## See Also

- [Sparse Matrix Operations](/cpp/kernels/sparse) - General sparse matrix utilities
- [Linear Algebra](/cpp/math) - Additional linear algebra operations
