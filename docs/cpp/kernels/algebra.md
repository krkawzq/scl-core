---
title: Matrix Algebra
description: High-performance sparse linear algebra operations
---

# Matrix Algebra

The `algebra` kernel provides efficient sparse matrix algebra operations.

## Overview

Matrix algebra operations include:
- Sparse matrix multiplication (SpMM)
- Sparse matrix-vector multiplication (SpMV)
- Matrix addition and scaling
- Transposition

## Functions

### `spmm`

Sparse matrix-matrix multiplication.

```cpp
template <typename T, bool IsCSR1, bool IsCSR2>
CSR spmm(
    const Sparse<T, IsCSR1>& A,
    const Sparse<T, IsCSR2>& B
);
```

**Returns**: Result matrix C = A * B

**Example**:
```cpp
#include "scl/kernel/algebra.hpp"

CSR A = ...;
CSR B = ...;
CSR C = kernel::algebra::spmm(A, B);
```

### `spmv`

Sparse matrix-vector multiplication.

```cpp
template <typename T, bool IsCSR>
void spmv(
    const Sparse<T, IsCSR>& matrix,
    Array<const Real> x,
    Array<Real> y
);
```

**Parameters**:
- `matrix` [in]: Sparse matrix
- `x` [in]: Input vector
- `y` [out]: Output vector (y = matrix * x)

**Example**:
```cpp
Array<Real> x = {x_ptr, matrix.cols()};
auto y = memory::aligned_alloc<Real>(matrix.rows());
Array<Real> y_view = {y.get(), static_cast<Size>(matrix.rows())};

kernel::algebra::spmv(matrix, x, y_view);
```

## Configuration

```cpp
namespace scl::kernel::algebra::config {
    constexpr Size PREFETCH_DISTANCE = 64;
    constexpr Size SPMM_BLOCK_COLS = 64;
    constexpr Size SPMM_BLOCK_ROWS = 32;
    constexpr Size PARALLEL_THRESHOLD = 128;
}
```

## Related Documentation

- [Gram Matrix](./gram.md) - Gram matrix computation
- [Sparse Matrices](../core/sparse.md) - Sparse matrix operations
