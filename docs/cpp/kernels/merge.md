---
title: Matrix Merging
description: Matrix concatenation and merging operations
---

# Matrix Merging

The `merge` kernel provides efficient matrix merging operations with SIMD optimization.

## Overview

Matrix merging is used for:
- Combining datasets
- Vertical/horizontal stacking
- Batch integration
- Data aggregation

## Functions

### `vstack`

Vertically stack two matrices (row-wise concatenation).

```cpp
template <typename T, bool IsCSR>
Sparse<T, IsCSR> vstack(
    const Sparse<T, IsCSR>& matrix1,
    const Sparse<T, IsCSR>& matrix2,
    BlockStrategy strategy = BlockStrategy::adaptive()
);
```

**Example**:
```cpp
#include "scl/kernel/merge.hpp"

CSR merged = kernel::merge::vstack(matrix1, matrix2);
```

### `hstack`

Horizontally stack two matrices (column-wise concatenation).

```cpp
template <typename T, bool IsCSR>
Sparse<T, IsCSR> hstack(
    const Sparse<T, IsCSR>& matrix1,
    const Sparse<T, IsCSR>& matrix2
);
```

## Related Documentation

- [Sparse Matrices](../core/sparse.md) - Sparse matrix operations
- [Kernels Overview](./overview.md) - General kernel usage
