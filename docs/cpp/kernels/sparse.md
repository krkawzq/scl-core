# sparse.hpp

> scl/kernel/sparse.hpp · Sparse matrix statistics kernels

## Overview

This file provides high-performance kernels for computing statistics on sparse matrices, format conversion, data cleanup, and validation. It includes functions for computing sums, means, variances along primary dimensions, exporting to contiguous arrays (CSR/CSC) and COO format, eliminating zeros, pruning, and memory analysis.

**Header**: `#include "scl/kernel/sparse.hpp"`

---

## Main APIs

### primary_sums

::: source_code file="scl/kernel/sparse.hpp" symbol="primary_sums" collapsed
:::

**Algorithm Description**

Computes the sum of values along each primary dimension (row for CSR, column for CSC):

1. For each primary index in parallel:
   - Get values span for the primary slice using sparse matrix accessors
   - Use `scl::vectorize::sum` for SIMD-optimized reduction
   - Accumulate sum of all non-zero values in the slice
2. Empty slices produce sum = 0
3. Parallelized over primary dimension for efficiency

**Edge Cases**

- **Empty slices**: Return sum = 0 for slices with no non-zeros
- **All zeros**: Returns zero sums
- **Single non-zero**: Returns that value
- **Very sparse**: Efficiently handles slices with few non-zeros

**Data Guarantees (Preconditions)**

- `output.len == matrix.primary_dim()`
- Matrix is valid sparse format (CSR or CSC)
- Output array is writable

**Complexity Analysis**

- **Time**: O(nnz) - processes each non-zero once
- **Space**: O(1) auxiliary per thread - only accumulator

**Example**

```cpp
#include "scl/kernel/sparse.hpp"

Sparse<Real, true> matrix = /* sparse matrix, CSR */;
Array<Real> row_sums(matrix.rows());

scl::kernel::sparse::primary_sums(matrix, row_sums);

// row_sums[i] = sum of all non-zeros in row i
```

---

### primary_means

::: source_code file="scl/kernel/sparse.hpp" symbol="primary_means" collapsed
:::

**Algorithm Description**

Computes the mean of values along each primary dimension, accounting for implicit zeros:

1. For each primary index in parallel:
   - Compute sum using `primary_sums`
   - Divide by secondary_dim (not nnz): `mean = sum / secondary_dim`
2. This accounts for implicit zeros in sparse representation
3. Empty slices produce mean = 0

**Edge Cases**

- **Empty slices**: Return mean = 0
- **Zero secondary_dim**: Division by zero avoided, returns 0
- **Constant slices**: Returns constant value
- **Sparse slices**: Mean reflects implicit zeros

**Data Guarantees (Preconditions)**

- `output.len == matrix.primary_dim()`
- `matrix.secondary_dim() > 0` (for meaningful means)
- Matrix is valid sparse format

**Complexity Analysis**

- **Time**: O(nnz) - dominated by sum computation
- **Space**: O(1) auxiliary per thread

**Example**

```cpp
Array<Real> row_means(matrix.rows());

scl::kernel::sparse::primary_means(matrix, row_means);

// row_means[i] = sum(row_i) / n_cols (includes implicit zeros)
```

---

### primary_variances

::: source_code file="scl/kernel/sparse.hpp" symbol="primary_variances" collapsed
:::

**Algorithm Description**

Computes variance along each primary dimension using fused sum and sum-of-squares:

1. For each primary index in parallel:
   - Use SIMD fused sum+sumsq helper (4-way unroll with FMA)
   - Compute mean: `mean = sum / secondary_dim`
   - Compute variance: `var = (sumsq - sum*mean) / (secondary_dim - ddof)`
   - Clamp negative values to zero for numerical stability
2. Uses compensated summation pattern for improved accuracy
3. ddof (delta degrees of freedom) defaults to 1 for sample variance

**Edge Cases**

- **Zero variance**: Returns 0 for constant slices
- **Negative variance**: Clamped to 0 (numerical stability)
- **ddof >= secondary_dim**: Division by zero avoided
- **Empty slices**: Returns variance = 0

**Data Guarantees (Preconditions)**

- `output.len == matrix.primary_dim()`
- `ddof >= 0 and ddof < secondary_dim`
- Matrix is valid sparse format

**Complexity Analysis**

- **Time**: O(nnz) - single pass with fused computation
- **Space**: O(1) auxiliary per thread

**Example**

```cpp
Array<Real> row_variances(matrix.rows());

scl::kernel::sparse::primary_variances(matrix, row_variances, 1);

// row_variances[i] = sample variance of row i (ddof=1)
```

---

### to_contiguous_arrays

::: source_code file="scl/kernel/sparse.hpp" symbol="to_contiguous_arrays" collapsed
:::

**Algorithm Description**

Exports sparse matrix to contiguous CSR/CSC format with registry-registered arrays:

1. Allocate data, indices, indptr arrays via registry
2. Build indptr by cumulative sum of row/column lengths
3. Copy values and indices sequentially to contiguous arrays
4. Register arrays with HandlerRegistry for memory management
5. Returns structure with registered pointers

**Edge Cases**

- **Empty matrix (nnz=0)**: Returns all-null pointers, indptr allocated
- **Already contiguous**: Still creates new arrays (no zero-copy)
- **Allocation failure**: Returns all-null pointers
- **Registry full**: Returns all-null pointers

**Data Guarantees (Preconditions)**

- Matrix is valid sparse format
- Sufficient memory available for allocation

**Complexity Analysis**

- **Time**: O(nnz) - sequential copy of all elements
- **Space**: O(nnz + primary_dim) for output arrays

**Example**

```cpp
auto arrs = scl::kernel::sparse::to_contiguous_arrays(matrix);

// arrs.data, arrs.indices, arrs.indptr are registry-registered
// arrs.nnz = total non-zeros
// arrs.primary_dim = rows (CSR) or cols (CSC)

// Transfer ownership to Python before unregistering
```

---

### to_coo_arrays

::: source_code file="scl/kernel/sparse.hpp" symbol="to_coo_arrays" collapsed
:::

**Algorithm Description**

Exports sparse matrix to COO (Coordinate) format with registry-registered arrays:

1. Compute offsets for each primary slice
2. Parallel conversion: each thread handles one or more primary slices
3. Write COO triplets (row, col, value) to pre-computed positions
4. Entries in row-major order (CSR) or column-major order (CSC)
5. Register arrays with HandlerRegistry

**Edge Cases**

- **Empty matrix**: Returns all-null pointers
- **Allocation failure**: Returns all-null pointers
- **Parallel overhead**: Small matrices may be slower than sequential

**Data Guarantees (Preconditions)**

- Matrix is valid sparse format
- Sufficient memory available

**Complexity Analysis**

- **Time**: O(nnz / n_threads + primary_dim) - parallel conversion
- **Space**: O(nnz + primary_dim) for output arrays and offsets

**Example**

```cpp
auto coo = scl::kernel::sparse::to_coo_arrays(matrix);

// coo.row_indices[i], coo.col_indices[i], coo.values[i] form one entry
// coo.nnz = total non-zeros
```

---

### eliminate_zeros

::: source_code file="scl/kernel/sparse.hpp" symbol="eliminate_zeros" collapsed
:::

**Algorithm Description**

Removes zero-valued elements from sparse matrix:

1. Parallel count non-zeros per row/column after filtering by tolerance
2. Allocate new matrix with reduced nnz
3. Parallel copy non-zero elements only (where |value| > tolerance)
4. Indices remain sorted
5. Creates new matrix, original unchanged

**Edge Cases**

- **No zeros**: Returns copy of original matrix
- **All zeros**: Returns empty matrix
- **Tolerance > 0**: Removes near-zeros as well
- **Very sparse**: Efficiently handles matrices with many zeros

**Data Guarantees (Preconditions)**

- Matrix is valid sparse format
- `tolerance >= 0`

**Complexity Analysis**

- **Time**: O(nnz / n_threads) - parallel filtering and copying
- **Space**: O(nnz_output + primary_dim) for result and counts

**Example**

```cpp
Sparse<Real, true> cleaned = scl::kernel::sparse::eliminate_zeros(
    matrix,
    1e-10  // tolerance
);

// cleaned has all |value| <= 1e-10 removed
```

---

### prune

::: source_code file="scl/kernel/sparse.hpp" symbol="prune" collapsed
:::

**Algorithm Description**

Removes small values from sparse matrix, optionally preserving structure:

1. If `keep_structure = true`:
   - Set values with |value| < threshold to zero
   - Keep matrix structure (indices, pointers) unchanged
2. If `keep_structure = false`:
   - Remove elements with |value| < threshold entirely
   - Compact structure (reduce nnz)
3. Creates new matrix, original unchanged

**Edge Cases**

- **All values pruned**: Returns empty matrix (if !keep_structure)
- **No values pruned**: Returns copy of original
- **keep_structure = true**: Matrix size unchanged, some values zero
- **keep_structure = false**: Matrix size reduced

**Data Guarantees (Preconditions)**

- Matrix is valid sparse format
- `threshold >= 0`

**Complexity Analysis**

- **Time**: O(nnz) - single pass filtering
- **Space**: O(nnz) for result (keep_structure) or O(nnz_output) (remove)

**Example**

```cpp
// Prune and remove structure
Sparse<Real, true> pruned = scl::kernel::sparse::prune(
    matrix,
    0.01,      // threshold
    false      // remove structure
);

// Prune but keep structure
Sparse<Real, true> pruned_keep = scl::kernel::sparse::prune(
    matrix,
    0.01,
    true       // keep structure
);
```

---

## Utility Functions

### primary_nnz

Gets the number of non-zero elements in each primary slice.

::: source_code file="scl/kernel/sparse.hpp" symbol="primary_nnz" collapsed
:::

**Complexity**

- Time: O(primary_dim)
- Space: O(1) auxiliary

---

### from_contiguous_arrays

Creates sparse matrix from contiguous CSR/CSC format arrays.

::: source_code file="scl/kernel/sparse.hpp" symbol="from_contiguous_arrays" collapsed
:::

**Complexity**

- Time: O(primary_dim) for metadata setup
- Space: O(primary_dim) for metadata arrays

---

### validate

Validates sparse matrix structure integrity.

::: source_code file="scl/kernel/sparse.hpp" symbol="validate" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(1)

---

### memory_info

Gets detailed memory usage information for sparse matrix.

::: source_code file="scl/kernel/sparse.hpp" symbol="memory_info" collapsed
:::

**Complexity**

- Time: O(primary_dim) for block counting
- Space: O(1)

---

### make_contiguous

Converts sparse matrix to contiguous storage layout if not already.

::: source_code file="scl/kernel/sparse.hpp" symbol="make_contiguous" collapsed
:::

**Complexity**

- Time: O(nnz) if conversion needed, O(primary_dim) if already contiguous
- Space: O(nnz) if conversion needed

---

### resize_secondary

Resizes secondary dimension of sparse matrix (metadata only).

::: source_code file="scl/kernel/sparse.hpp" symbol="resize_secondary" collapsed
:::

**Complexity**

- Time: O(1) in release mode, O(nnz) in debug mode when shrinking
- Space: O(1)

---

## See Also

- [Sparse Matrix Core](../core/sparse) - Core sparse matrix types
- [Memory Module](../core/memory) - Memory management

