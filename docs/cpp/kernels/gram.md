---
title: Gram Matrix
description: Gram matrix computation with adaptive algorithms
---

# Gram Matrix

The `gram` kernel provides efficient Gram matrix computation for sparse matrices.

## Overview

Gram matrices are used for:
- Kernel methods
- Similarity computation
- Feature transformations
- Machine learning pipelines

## Functions

### `gram_matrix`

Compute Gram matrix (X * X^T or X^T * X).

```cpp
template <typename T, bool IsCSR>
CSR gram_matrix(
    const Sparse<T, IsCSR>& matrix,
    bool transpose = false
);
```

**Parameters**:
- `matrix` [in]: Input matrix
- `transpose` [in]: Compute X^T * X if true, X * X^T if false

**Returns**: Gram matrix

**Example**:
```cpp
#include "scl/kernel/gram.hpp"

// Compute X * X^T (cell-cell similarity)
CSR cell_gram = kernel::gram::gram_matrix(matrix, false);

// Compute X^T * X (gene-gene similarity)
CSR gene_gram = kernel::gram::gram_matrix(matrix, true);
```

## Configuration

```cpp
namespace scl::kernel::gram::config {
    constexpr Size PREFETCH_DISTANCE = 32;
    constexpr Size RATIO_THRESHOLD = 32;
    constexpr Size GALLOP_THRESHOLD = 256;
}
```

## Related Documentation

- [Algebra](./algebra.md) - Matrix algebra
- [Neighbors](./neighbors.md) - Neighbor search
