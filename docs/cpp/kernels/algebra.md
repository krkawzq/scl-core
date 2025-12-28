# algebra.hpp

> scl/kernel/algebra.hpp · High-performance sparse linear algebra kernels

## Overview

This file provides high-performance sparse matrix-vector and matrix-matrix multiplication kernels with SIMD optimization and adaptive strategies for different matrix structures.

This file provides:
- Sparse matrix-vector multiplication (SpMV) with alpha/beta scaling
- Sparse matrix-matrix multiplication (SpMM)
- Transpose operations
- Row-wise operations (norms, sums, scaling)
- Adaptive optimization based on row/column length

**Header**: `#include "scl/kernel/algebra.hpp"`

---

## Main APIs

### spmv

::: source_code file="scl/kernel/algebra.hpp" symbol="spmv" collapsed
:::

**Algorithm Description**

Computes sparse matrix-vector multiplication with alpha and beta scaling: `y = alpha * A * x + beta * y`

1. **Beta Scaling Phase**: Scale output vector y by beta using SIMD-optimized operations:
   - If beta == 0: Zero-fill y using SIMD (fast path)
   - If beta == 1: Skip scaling (no-op)
   - Otherwise: Scale y by beta using 4-way unrolled SIMD with prefetching

2. **Early Exit**: If alpha == 0, return immediately (no matrix multiplication needed)

3. **Parallel Loop**: Process each row (CSR) or column (CSC) in parallel:
   - Extract row/column values and indices
   - Compute dot product using adaptive strategy based on non-zero count:
     - **Short rows** (nnz < 8): Scalar loop (minimal overhead)
     - **Medium rows** (8 <= nnz < 64): 4-way unrolled loop
     - **Long rows** (64 <= nnz < 256): 8-way unrolled loop with prefetching
     - **Very long rows** (nnz >= 256): 8-way unroll with aggressive prefetching
   - **Consecutive Index Optimization**: If indices are consecutive, use dense SIMD path (much faster)
   - Accumulate result: `y[i] += alpha * dot_product`

**Edge Cases**

- **Empty matrix**: Returns immediately after beta scaling
- **Zero alpha**: Returns after beta scaling only
- **Zero beta**: Output vector is zero-filled before multiplication
- **Unit alpha/beta**: Optimized paths for common cases (alpha=1, beta=0 or beta=1)
- **NaN/Inf in input**: Propagates through standard floating-point arithmetic
- **Very sparse rows**: Handled efficiently with scalar loop

**Data Guarantees (Preconditions)**

- `y.size() >= primary_dim(A)` (rows for CSR, columns for CSC)
- `x.size() >= secondary_dim(A)` (columns for CSR, rows for CSC)
- Matrix A must be valid sparse format (sorted indices, no duplicates)
- Output vector y must be pre-allocated
- Input vector x must be valid and accessible

**Complexity Analysis**

- **Time**: O(nnz) sequential work, parallelized across primary_dim(A)
  - Beta scaling: O(primary_dim) with SIMD
  - Dot products: O(nnz) total
  - Parallel overhead: O(primary_dim) for thread scheduling
- **Space**: O(1) auxiliary space per thread
  - No temporary allocations
  - Stack-allocated accumulators only

**Example**

```cpp
#include "scl/kernel/algebra.hpp"

// Create sparse matrix (CSR format)
scl::Sparse<Real, true> A = /* ... */;  // [n_rows x n_cols]
scl::Array<const Real> x = /* ... */;    // Input vector [n_cols]
scl::Array<Real> y(n_rows);              // Output vector [n_rows], pre-allocated

// Compute: y = 2.0 * A * x + 0.5 * y
scl::kernel::algebra::spmv(A, x, y, Real(2.0), Real(0.5));

// Simple multiplication: y = A * x
scl::kernel::algebra::spmv(A, x, y, Real(1), Real(0));

// Accumulate: y += A * x
scl::kernel::algebra::spmv(A, x, y, Real(1), Real(1));
```

---

### spmv_transpose

::: source_code file="scl/kernel/algebra.hpp" symbol="spmv_transpose" collapsed
:::

**Algorithm Description**

Computes transpose sparse matrix-vector multiplication: `y = alpha * A^T * x + beta * y`

1. **Beta Scaling**: Same as spmv (SIMD-optimized)

2. **Transpose Multiplication**: For CSR matrix A, computes A^T * x:
   - Iterate over non-zero elements of A
   - For each element A[i][j], accumulate: `y[j] += alpha * A[i][j] * x[i]`
   - Parallelized over non-zero elements using atomic operations or reduction

3. **Adaptive Strategy**: Uses similar optimization as spmv based on column length

**Edge Cases**

- **Empty matrix**: Returns after beta scaling
- **Zero alpha**: Returns after beta scaling only
- **Sparse columns**: Handled efficiently with atomic accumulation

**Data Guarantees (Preconditions)**

- `y.size() >= secondary_dim(A)` (columns for CSR)
- `x.size() >= primary_dim(A)` (rows for CSR)
- Matrix A must be valid sparse format
- Output vector y must be pre-allocated

**Complexity Analysis**

- **Time**: O(nnz) parallelized over non-zeros
- **Space**: O(1) auxiliary per thread

**Example**

```cpp
// Compute transpose: y = A^T * x
scl::Array<Real> y(n_cols);  // Output size is columns
scl::kernel::algebra::spmv_transpose(A, x, y, Real(1), Real(0));
```

---

### spmm

::: source_code file="scl/kernel/algebra.hpp" symbol="spmm" collapsed
:::

**Algorithm Description**

Sparse matrix-matrix multiplication: `C = alpha * A * B + beta * C`

1. **Beta Scaling**: Scale output matrix C by beta (SIMD-optimized)

2. **Blocked Multiplication**: Process in blocks for cache efficiency:
   - Block size: config::SPMM_BLOCK_ROWS x config::SPMM_BLOCK_COLS
   - For each block, compute sparse-dense matrix product
   - Parallelize over output blocks

3. **Row-wise Processing**: For each row of A:
   - Extract non-zero elements
   - Multiply with corresponding columns of B
   - Accumulate into output row of C

**Edge Cases**

- **Empty matrices**: Returns after beta scaling
- **Dimension mismatch**: Checked via assertions
- **Sparse output**: Efficiently handles sparse result patterns

**Data Guarantees (Preconditions)**

- `A.cols() == B.rows()` (dimension compatibility)
- `C.rows() == A.rows()` and `C.cols() == B.cols()`
- Output matrix C must be pre-allocated

**Complexity Analysis**

- **Time**: O(nnz_A * B.cols()) parallelized
- **Space**: O(block_size) auxiliary per thread

**Example**

```cpp
scl::Sparse<Real, true> A = /* ... */;  // [m x k]
scl::Sparse<Real, true> B = /* ... */;  // [k x n]
scl::Sparse<Real, true> C = /* ... */;  // [m x n], pre-allocated

// Compute: C = A * B
scl::kernel::algebra::spmm(A, B, C, Real(1), Real(0));
```

---

## Utility Functions

### spmv_simple

Simplified sparse matrix-vector multiplication: `y = A * x`

Equivalent to `spmv(A, x, y, T(1), T(0))`

::: source_code file="scl/kernel/algebra.hpp" symbol="spmv_simple" collapsed
:::

**Complexity**

- Time: O(nnz) parallelized
- Space: O(1) auxiliary

---

### spmv_add

Accumulate sparse matrix-vector product: `y += A * x`

Equivalent to `spmv(A, x, y, T(1), T(1))`

::: source_code file="scl/kernel/algebra.hpp" symbol="spmv_add" collapsed
:::

**Complexity**

- Time: O(nnz) parallelized
- Space: O(1) auxiliary

---

### spmv_scaled

Scaled sparse matrix-vector multiplication: `y = alpha * A * x`

Equivalent to `spmv(A, x, y, alpha, T(0))`

::: source_code file="scl/kernel/algebra.hpp" symbol="spmv_scaled" collapsed
:::

**Complexity**

- Time: O(nnz) parallelized
- Space: O(1) auxiliary

---

### spmv_transpose_simple

Simplified transpose multiplication: `y = A^T * x`

Equivalent to `spmv_transpose(A, x, y, T(1), T(0))`

::: source_code file="scl/kernel/algebra.hpp" symbol="spmv_transpose_simple" collapsed
:::

**Complexity**

- Time: O(nnz) parallelized
- Space: O(1) auxiliary

---

### row_norms

Compute L2 norm of each row in sparse matrix.

::: source_code file="scl/kernel/algebra.hpp" symbol="row_norms" collapsed
:::

**Complexity**

- Time: O(nnz) parallelized
- Space: O(1) auxiliary

---

### row_sums

Compute sum of each row in sparse matrix.

::: source_code file="scl/kernel/algebra.hpp" symbol="row_sums" collapsed
:::

**Complexity**

- Time: O(nnz) parallelized
- Space: O(1) auxiliary

---

### extract_diagonal

Extract diagonal elements from sparse matrix.

::: source_code file="scl/kernel/algebra.hpp" symbol="extract_diagonal" collapsed
:::

**Complexity**

- Time: O(min(rows, cols))
- Space: O(1) auxiliary

---

### scale_rows

Scale each row of sparse matrix by given factors.

::: source_code file="scl/kernel/algebra.hpp" symbol="scale_rows" collapsed
:::

**Complexity**

- Time: O(nnz) parallelized
- Space: O(1) auxiliary

---

## Configuration

Performance tuning constants in `scl::kernel::algebra::config`:

- `PREFETCH_DISTANCE = 64`: Cache line prefetch distance
- `SHORT_ROW_THRESHOLD = 8`: Threshold for scalar loop
- `MEDIUM_ROW_THRESHOLD = 64`: Threshold for 4-way unroll
- `LONG_ROW_THRESHOLD = 256`: Threshold for aggressive prefetch
- `CONSECUTIVE_CHECK_THRESHOLD = 16`: Minimum length to check for consecutive indices
- `SPMM_BLOCK_ROWS = 32`: Block size for SpMM
- `SPMM_BLOCK_COLS = 64`: Block size for SpMM

---

## Performance Notes

### Adaptive Optimization

The implementation automatically selects optimal strategy based on row/column length:
- Short rows use minimal overhead scalar loops
- Medium rows use 4-way unrolling for instruction-level parallelism
- Long rows use 8-way unrolling with prefetching to hide memory latency
- Consecutive indices trigger dense SIMD path for maximum throughput

### SIMD Acceleration

- Beta scaling uses SIMD for 4x-8x speedup
- Dense dot products use SIMD when indices are consecutive
- Prefetching hides memory latency for long rows

### Parallelization

- All operations parallelize over primary dimension (rows for CSR)
- No synchronization needed (distinct output elements)
- Scales with hardware concurrency

---

## See Also

- [Sparse Matrix Types](../core/sparse)
- [SIMD Operations](../core/simd)
- [Memory Management](../core/memory)
