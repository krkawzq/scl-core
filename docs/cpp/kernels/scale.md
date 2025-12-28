# scale.hpp

> scl/kernel/scale.hpp · Scaling operations for sparse matrices

## Overview

This file provides scaling operations for sparse matrices, including standardization (z-score normalization), row scaling, and row shifting. These operations are essential for data preprocessing and normalization.

Key features:
- Standardization with mean and std
- Row-wise scaling and shifting
- SIMD-optimized implementations
- In-place operations
- Optional clipping for standardization

**Header**: `#include "scl/kernel/scale.hpp"`

---

## Main APIs

### standardize

::: source_code file="scl/kernel/scale.hpp" symbol="standardize" collapsed
:::

**Algorithm Description**

Standardize sparse matrix values in-place: (x - mean) / std:

1. **For each row i in parallel**:
   - If `std[i] == 0`: Skip row (unchanged)
   - Otherwise:
     - Compute `inv_sigma = 1.0 / std[i]` (replace division with multiplication)
     - For each non-zero element in row:
       - If `zero_center`: `value = (value - mean[i]) * inv_sigma`
       - Else: `value = value * inv_sigma`
       - If `max_value > 0`: Clip to [-max_value, max_value]
     - Use adaptive SIMD strategy:
       - Short rows (< 16): scalar loop
       - Medium (16-128): 4-way SIMD unroll
       - Long (>= 128): 8-way SIMD unroll with prefetch

Standardization transforms data to have zero mean and unit variance, making features comparable across different scales.

**Edge Cases**

- **Zero std rows**: Skipped (unchanged)
- **max_value = 0**: No clipping applied
- **zero_center = false**: Only scaling, no mean subtraction
- **Negative values**: Handled correctly (clipped if max_value > 0)

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR or CSC format with mutable values
- `means.len == matrix.primary_dim()`
- `stds.len == matrix.primary_dim()`
- `stds[i] > 0` for rows to be processed (zero std rows are skipped)

**Complexity Analysis**

- **Time**: O(nnz) - single pass through all non-zero elements
  - Parallelized over primary dimension (rows for CSR)
  - SIMD acceleration for longer rows
- **Space**: O(1) auxiliary per thread

**Example**

```cpp
#include "scl/kernel/scale.hpp"

Sparse<Real, true> matrix = /* ... */;
Array<Real> means(n_rows);
Array<Real> stds(n_rows);

// ... compute means and stds ...

Real max_value = 10.0;  // Clip to [-10, 10]
bool zero_center = true;  // Subtract mean before scaling

scl::kernel::scale::standardize(
    matrix,
    means,
    stds,
    max_value,
    zero_center
);

// Each value v transformed to: (v - mean) / std
// Results clipped to [-max_value, max_value]
```

---

### scale_rows

::: source_code file="scl/kernel/scale.hpp" symbol="scale_rows" collapsed
:::

**Algorithm Description**

Multiply each primary dimension by a corresponding scale factor:

1. **For each row i in parallel**:
   - If `scales[i] == 1.0`: Skip row (early exit optimization)
   - Otherwise:
     - For each non-zero element in row:
       - `value = value * scales[i]`
     - Use SIMD 4-way unroll with prefetch for efficiency

Row scaling multiplies all values in a row by a constant factor, useful for normalization or feature weighting.

**Edge Cases**

- **scale = 1.0**: Row unchanged (early exit)
- **scale = 0.0**: All values become 0
- **scale < 0**: Values become negative (if originally positive)

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR or CSC format with mutable values
- `scales.len == matrix.primary_dim()`

**Complexity Analysis**

- **Time**: O(nnz) - single pass through all non-zero elements
  - Parallelized over primary dimension
  - Early exit for scale = 1.0 rows
- **Space**: O(1) auxiliary per thread

**Example**

```cpp
Array<Real> scales(n_rows);
// ... set scales, e.g., based on row sums ...

scl::kernel::scale::scale_rows(
    matrix,
    scales
);

// Each value in row i multiplied by scales[i]
```

---

### shift_rows

::: source_code file="scl/kernel/scale.hpp" symbol="shift_rows" collapsed
:::

**Algorithm Description**

Add a constant offset to each primary dimension:

1. **For each row i in parallel**:
   - If `offsets[i] == 0.0`: Skip row (early exit optimization)
   - Otherwise:
     - For each non-zero element in row:
       - `value = value + offsets[i]`
     - Use SIMD 4-way unroll with prefetch for efficiency

Row shifting adds a constant to all values in a row, useful for centering or offsetting data.

**Edge Cases**

- **offset = 0.0**: Row unchanged (early exit)
- **Implicit zeros**: Only stored (non-zero) values are shifted
  - Implicit zeros remain zero (sparse matrix property)
  - For true shift of all values, matrix must be densified first

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR or CSC format with mutable values
- `offsets.len == matrix.primary_dim()`

**Complexity Analysis**

- **Time**: O(nnz) - single pass through all non-zero elements
  - Parallelized over primary dimension
  - Early exit for offset = 0.0 rows
- **Space**: O(1) auxiliary per thread

**Example**

```cpp
Array<Real> offsets(n_rows);
// ... set offsets, e.g., negative means for centering ...

scl::kernel::scale::shift_rows(
    matrix,
    offsets
);

// Each value in row i increased by offsets[i]
// Note: Only stored (non-zero) values are shifted
```

---

## Notes

**SIMD Optimization**

All functions use adaptive SIMD strategies:
- Short rows: scalar loop (no SIMD overhead)
- Medium rows: 4-way SIMD unroll
- Long rows: 8-way SIMD unroll with prefetch

**In-Place Operations**

All functions modify the matrix in-place:
- Matrix structure (indices, indptr) unchanged
- Only values are modified
- Memory efficient (no copies)

**Numerical Considerations**

- **Division avoidance**: Uses `inv_sigma = 1/std` to replace division with multiplication
- **Early exits**: Skips rows with scale=1 or offset=0 for efficiency
- **Zero std handling**: Rows with std=0 are skipped in standardization

**Use Cases**

- **Standardization**: Normalize features to zero mean, unit variance
- **Scaling**: Normalize by row sums or other factors
- **Centering**: Shift data to zero mean
- **Feature weighting**: Apply different scales to different features

**Thread Safety**

All functions are thread-safe and parallelized over primary dimension (rows for CSR, columns for CSC).

## See Also

- [Normalize](/cpp/kernels/normalize) - Normalization operations
- [Statistics](/cpp/kernels/statistics) - Statistical analysis
