---
title: Softmax
description: Softmax activation with SIMD optimization
---

# Softmax

The `softmax` kernel provides efficient softmax operations with adaptive SIMD optimization.

## Overview

Softmax is used for:
- Probability normalization
- Multi-class classification
- Attention mechanisms
- Activation functions

## Functions

### `softmax_inplace`

Apply softmax transformation in-place.

```cpp
template <typename T, bool IsCSR>
void softmax_inplace(Sparse<T, IsCSR>& matrix);
```

**Mathematical Operation**: `softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)`

**Example**:
```cpp
#include "scl/kernel/softmax.hpp"

kernel::softmax::softmax_inplace(matrix);
```

## Configuration

```cpp
namespace scl::kernel::softmax::config {
    constexpr Size SHORT_THRESHOLD = 16;
    constexpr Size MEDIUM_THRESHOLD = 128;
    constexpr Size PREFETCH_DISTANCE = 16;
}
```

## Related Documentation

- [Normalization](./normalize.md) - Normalization operations
- [Kernels Overview](./overview.md) - General kernel usage
