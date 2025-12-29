---
title: Matrix Slicing
description: Efficient sparse matrix slicing operations
---

# Matrix Slicing

The `slice` kernel provides efficient slicing operations for sparse matrices with optimized parallel processing.

## Overview

Matrix slicing is used for:
- Extracting submatrices
- Filtering rows/columns
- Data subsetting
- Batch processing

## Functions

### `slice_rows`

Extract rows from sparse matrix.

```cpp
template <typename T, bool IsCSR>
Sparse<T, IsCSR> slice_rows(
    const Sparse<T, IsCSR>& matrix,
    Array<const Index> row_indices
);
```

**Parameters**:
- `matrix` [in]: Input matrix
- `row_indices` [in]: Indices of rows to extract

**Returns**: Sliced submatrix

**Example**:
```cpp
#include "scl/kernel/slice.hpp"

Array<Index> selected_rows = {indices_ptr, n_selected};
CSR submatrix = kernel::slice::slice_rows(matrix, selected_rows);
```

## Configuration

```cpp
constexpr Size PARALLEL_THRESHOLD_ROWS = 512;
constexpr Size PARALLEL_THRESHOLD_NNZ = 10000;
```

## Related Documentation

- [Sparse Matrices](../core/sparse.md) - Sparse matrix operations
- [Kernels Overview](./overview.md) - General kernel usage
