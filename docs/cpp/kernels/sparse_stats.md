---
title: Sparse Matrix Statistics
description: Statistical operations on sparse matrices
---

# Sparse Matrix Statistics

The `sparse` kernel provides efficient statistical operations for sparse matrices, optimized with SIMD and parallelization.

## Overview

Sparse matrix statistics are used for:
- Computing row/column statistics
- Variance and standard deviation
- Summary statistics
- Data quality assessment

## Functions

### `primary_sums`

Compute sums for each row (CSR) or column (CSC).

```cpp
template <typename T, bool IsCSR>
void primary_sums(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output
);
```

### `primary_means`

Compute means for each row/column.

```cpp
template <typename T, bool IsCSR>
void primary_means(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output
);
```

### `primary_variances`

Compute variances for each row/column.

```cpp
template <typename T, bool IsCSR>
void primary_variances(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output
);
```

**Example**:
```cpp
#include "scl/kernel/sparse.hpp"

// Compute row statistics
auto row_sums = memory::aligned_alloc<Real>(matrix.rows());
Array<Real> sums_view = {row_sums.get(), static_cast<Size>(matrix.rows())};
kernel::sparse::primary_sums(matrix, sums_view);

auto row_means = memory::aligned_alloc<Real>(matrix.rows());
Array<Real> means_view = {row_means.get(), static_cast<Size>(matrix.rows())};
kernel::sparse::primary_means(matrix, means_view);

auto row_vars = memory::aligned_alloc<Real>(matrix.rows());
Array<Real> vars_view = {row_vars.get(), static_cast<Size>(matrix.rows())};
kernel::sparse::primary_variances(matrix, vars_view);
```

## Configuration

```cpp
namespace scl::kernel::sparse::config {
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Size PARALLEL_THRESHOLD = 1024;
    constexpr Size BATCH_SIZE = 64;
}
```

## Related Documentation

- [Normalization](./normalize.md) - Normalization operations
- [Kernels Overview](./overview.md) - General kernel usage

