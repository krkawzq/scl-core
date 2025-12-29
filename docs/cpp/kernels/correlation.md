---
title: Correlation
description: Pearson correlation computation for sparse matrices
---

# Correlation

The `correlation` kernel provides efficient Pearson correlation computation for sparse matrices.

## Overview

Correlation analysis is used for:
- Gene co-expression analysis
- Feature similarity
- Network construction
- Quality assessment

## Functions

### `pearson_correlation`

Compute Pearson correlation matrix.

```cpp
template <typename T, bool IsCSR>
CSR pearson_correlation(
    const Sparse<T, IsCSR>& matrix
);
```

**Returns**: Correlation matrix (symmetric)

**Example**:
```cpp
#include "scl/kernel/correlation.hpp"

CSR corr_matrix = kernel::correlation::pearson_correlation(matrix);
```

## Configuration

```cpp
namespace scl::kernel::correlation::config {
    constexpr Size CHUNK_SIZE = 64;
    constexpr Size STAT_CHUNK = 256;
    constexpr Size PREFETCH_DISTANCE = 32;
}
```

## Related Documentation

- [Statistics](./statistics.md) - Statistical tests
- [Kernels Overview](./overview.md) - General kernel usage
