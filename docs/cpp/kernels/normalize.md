---
title: Normalization
description: Normalization operations for sparse matrices
---

# Normalization

The `normalize` kernel provides efficient normalization operations for sparse matrices, including total count normalization, L1/L2 normalization, and masked operations.

## Overview

Normalization is a fundamental operation in single-cell analysis, typically performed to:
- Normalize counts per cell to a target sum
- Scale rows/columns to unit norm (L1 or L2)
- Handle masked or filtered data

All operations are optimized with SIMD and parallelization.

## Functions

### Row/Column Sums

#### `compute_row_sums`

Compute the sum of each row in a sparse matrix.

```cpp
template <typename T, bool IsCSR>
void compute_row_sums(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output
);
```

**Parameters**:
- `matrix` [in]: Input sparse matrix
- `output` [out]: Output array of row sums (length must be ≥ matrix.rows())

**Example**:
```cpp
#include "scl/kernel/normalize.hpp"

auto matrix = CSR::create(1000, 2000, 10000);
auto row_sums = memory::aligned_alloc<Real>(matrix.rows());
Array<Real> sums_view = {row_sums.get(), static_cast<Size>(matrix.rows())};

kernel::normalize::compute_row_sums(matrix, sums_view);
```

**Complexity**: O(nnz) time, O(rows) space

#### `compute_col_sums`

Compute the sum of each column (for CSC matrices).

```cpp
template <typename T, bool IsCSR>
void compute_col_sums(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output
);
```

### Scaling Operations

#### `scale_primary`

Scale each row (CSR) or column (CSC) by a given factor.

```cpp
template <typename T, bool IsCSR>
void scale_primary(
    Sparse<T, IsCSR>& matrix,
    Array<const Real> scales
);
```

**Parameters**:
- `matrix` [in,out]: Matrix to scale (modified in-place)
- `scales` [in]: Scaling factors (length must be ≥ primary dimension)

**Example**:
```cpp
// Scale each row by its inverse sum
auto row_sums = memory::aligned_alloc<Real>(matrix.rows());
Array<Real> sums_view = {row_sums.get(), static_cast<Size>(matrix.rows())};
kernel::normalize::compute_row_sums(matrix, sums_view);

// Compute inverse scales
auto scales = memory::aligned_alloc<Real>(matrix.rows());
Array<Real> scales_view = {scales.get(), static_cast<Size>(matrix.rows())};
for (Index i = 0; i < matrix.rows(); ++i) {
    scales_view[i] = (sums_view[i] > 0) ? Real(1) / sums_view[i] : Real(0);
}

// Scale matrix
kernel::normalize::scale_primary(matrix, scales_view);
```

### Masked Operations

#### `primary_sums_masked`

Compute sums with a mask to exclude certain elements.

```cpp
template <typename T, bool IsCSR>
void primary_sums_masked(
    const Sparse<T, IsCSR>& matrix,
    Array<const Byte> mask,
    Array<Real> output
);
```

**Parameters**:
- `matrix` [in]: Input sparse matrix
- `mask` [in]: Mask array (1 = exclude, 0 = include)
- `output` [out]: Output sums

**Example**:
```cpp
// Exclude highly expressed genes from sum
Array<Byte> mask = {mask_ptr, matrix.cols()};
Array<Real> masked_sums = {sums_ptr, matrix.rows()};
kernel::normalize::primary_sums_masked(matrix, mask, masked_sums);
```

### Highly Expressed Detection

#### `detect_highly_expressed`

Detect elements that exceed a fraction of the row/column sum.

```cpp
template <typename T, bool IsCSR>
void detect_highly_expressed(
    const Sparse<T, IsCSR>& matrix,
    Array<const Real> row_sums,
    Real max_fraction,
    Array<Byte> out_mask
);
```

**Parameters**:
- `matrix` [in]: Input sparse matrix
- `row_sums` [in]: Precomputed row/column sums
- `max_fraction` [in]: Maximum fraction threshold (e.g., 0.05 for 5%)
- `out_mask` [out]: Output mask (1 = highly expressed, 0 = normal)

**Example**:
```cpp
// Detect genes expressed in >5% of total counts per cell
auto row_sums = memory::aligned_alloc<Real>(matrix.rows());
Array<Real> sums_view = {row_sums.get(), static_cast<Size>(matrix.rows())};
kernel::normalize::compute_row_sums(matrix, sums_view);

auto mask = memory::aligned_alloc<Byte>(matrix.cols());
Array<Byte> mask_view = {mask.get(), static_cast<Size>(matrix.cols())};
kernel::normalize::detect_highly_expressed(
    matrix, sums_view, Real(0.05), mask_view
);
```

## Common Patterns

### Total Count Normalization

Normalize each row to a target sum:

```cpp
void normalize_total(CSR& matrix, Real target_sum) {
    // Compute row sums
    auto row_sums = memory::aligned_alloc<Real>(matrix.rows());
    Array<Real> sums_view = {row_sums.get(), static_cast<Size>(matrix.rows())};
    kernel::normalize::compute_row_sums(matrix, sums_view);
    
    // Compute scaling factors
    auto scales = memory::aligned_alloc<Real>(matrix.rows());
    Array<Real> scales_view = {scales.get(), static_cast<Size>(matrix.rows())};
    for (Index i = 0; i < matrix.rows(); ++i) {
        Real sum = sums_view[i];
        scales_view[i] = (sum > Real(1e-10)) ? target_sum / sum : Real(0);
    }
    
    // Scale matrix
    kernel::normalize::scale_primary(matrix, scales_view);
}
```

### L2 Normalization

Normalize each row to unit L2 norm:

```cpp
void normalize_l2(CSR& matrix) {
    // Compute L2 norms
    auto norms = memory::aligned_alloc<Real>(matrix.rows());
    Array<Real> norms_view = {norms.get(), static_cast<Size>(matrix.rows())};
    
    // Compute squared norms
    for (Index i = 0; i < matrix.rows(); ++i) {
        auto values = matrix.row_values(i);
        Real sum_sq = Real(0);
        for (Index k = 0; k < matrix.row_length(i); ++k) {
            Real val = static_cast<Real>(values[k]);
            sum_sq += val * val;
        }
        norms_view[i] = std::sqrt(sum_sq);
    }
    
    // Compute inverse norms
    auto scales = memory::aligned_alloc<Real>(matrix.rows());
    Array<Real> scales_view = {scales.get(), static_cast<Size>(matrix.rows())};
    for (Index i = 0; i < matrix.rows(); ++i) {
        Real norm = norms_view[i];
        scales_view[i] = (norm > Real(1e-10)) ? Real(1) / norm : Real(0);
    }
    
    // Scale matrix
    kernel::normalize::scale_primary(matrix, scales_view);
}
```

## Performance Considerations

### Parallelization

All functions automatically parallelize when data size exceeds thresholds:

```cpp
namespace config {
    constexpr Size PREFETCH_DISTANCE = 64;
}
```

### SIMD Optimization

Scaling operations use SIMD for vectorized multiplication:

```cpp
// SIMD-accelerated scaling
detail::scale_simd(values.ptr, len, scale);
```

### Memory Efficiency

- In-place operations avoid extra allocations
- Masked operations use atomic writes for thread safety
- Prefetching reduces memory latency

## Configuration

```cpp
namespace scl::kernel::normalize::config {
    constexpr Size PREFETCH_DISTANCE = 64;
}
```

## Related Documentation

- [Kernels Overview](./overview.md) - General kernel usage
- [Sparse Matrices](../core/sparse.md) - Sparse matrix operations
- [Threading](../threading.md) - Parallel execution

