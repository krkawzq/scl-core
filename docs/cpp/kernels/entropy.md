---
title: Entropy
description: Information theory measures for sparse data
---

# Entropy

The `entropy` kernel provides efficient entropy and information theory measures for sparse data analysis.

## Overview

Entropy measures are used for:
- Feature selection
- Data complexity assessment
- Information content analysis
- Quality metrics

## Functions

### `compute_entropy`

Compute Shannon entropy.

```cpp
template <typename T, bool IsCSR>
void compute_entropy(
    const Sparse<T, IsCSR>& matrix,
    Array<Real> entropy,
    bool use_log2 = false
);
```

**Parameters**:
- `matrix` [in]: Input matrix
- `entropy` [out]: Entropy values
- `use_log2` [in]: Use log base 2 (default: false, uses natural log)

**Example**:
```cpp
#include "scl/kernel/entropy.hpp"

auto entropy = memory::aligned_alloc<Real>(matrix.rows());
Array<Real> ent_view = {entropy.get(), static_cast<Size>(matrix.rows())};

kernel::entropy::compute_entropy(matrix, ent_view, use_log2=true);
```

## Configuration

```cpp
namespace scl::kernel::entropy::config {
    constexpr Real EPSILON = 1e-15;
    constexpr Size PARALLEL_THRESHOLD = 128;
    constexpr Size PREFETCH_DISTANCE = 64;
}
```

## Related Documentation

- [HVG](./hvg.md) - Highly variable genes
- [Kernels Overview](./overview.md) - General kernel usage
