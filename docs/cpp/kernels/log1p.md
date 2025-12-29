---
title: Logarithmic Transforms
description: Log(1+x) and related transformations
---

# Logarithmic Transforms

The `log1p` kernel provides efficient logarithmic transformations for sparse matrices, including `log1p`, `log2p1`, and `expm1` operations.

## Overview

Logarithmic transforms are essential in single-cell analysis for:
- Stabilizing variance in count data
- Preparing data for downstream analysis
- Reversing log transformations

All operations are SIMD-optimized and parallelized.

## Functions

### `log1p_inplace`

Apply log(1+x) transformation in-place.

```cpp
template <typename T, bool IsCSR>
void log1p_inplace(Sparse<T, IsCSR>& matrix);
```

**Parameters**:
- `matrix` [in,out]: Matrix to transform (modified in-place)

**Mathematical Operation**: `x → log(1 + x)`

**Example**:
```cpp
#include "scl/kernel/log1p.hpp"

auto matrix = CSR::create(1000, 2000, 10000);
// ... populate matrix ...

// Apply log1p transformation
kernel::log1p::log1p_inplace(matrix);
```

**Complexity**: O(nnz) time, O(1) space

**Thread Safety**: Thread-safe (each row processed independently)

### `log2p1_inplace`

Apply log₂(1+x) transformation in-place.

```cpp
template <typename T, bool IsCSR>
void log2p1_inplace(Sparse<T, IsCSR>& matrix);
```

**Parameters**:
- `matrix` [in,out]: Matrix to transform (modified in-place)

**Mathematical Operation**: `x → log₂(1 + x) = log(1 + x) / log(2)`

**Example**:
```cpp
// Apply log2p1 transformation
kernel::log1p::log2p1_inplace(matrix);
```

**Note**: Computed as `log1p(x) * inv_ln2` for efficiency.

### `expm1_inplace`

Apply exp(x) - 1 transformation in-place (inverse of log1p).

```cpp
template <typename T, bool IsCSR>
void expm1_inplace(Sparse<T, IsCSR>& matrix);
```

**Parameters**:
- `matrix` [in,out]: Matrix to transform (modified in-place)

**Mathematical Operation**: `x → exp(x) - 1`

**Example**:
```cpp
// Reverse log1p transformation
kernel::log1p::expm1_inplace(matrix);
```

**Note**: `expm1` is the inverse of `log1p` for small values, providing better numerical stability than `exp(x) - 1`.

## Common Patterns

### Standard Log-Normalization Pipeline

```cpp
void log_normalize(CSR& matrix, Real target_sum) {
    // 1. Total count normalization
    normalize_total(matrix, target_sum);
    
    // 2. Log transform
    kernel::log1p::log1p_inplace(matrix);
}
```

### Log2 Normalization

```cpp
void log2_normalize(CSR& matrix, Real target_sum) {
    // Normalize to target sum
    normalize_total(matrix, target_sum);
    
    // Apply log2 transform
    kernel::log1p::log2p1_inplace(matrix);
}
```

### Reversing Log Transform

```cpp
void reverse_log1p(CSR& matrix) {
    // Convert back from log space
    kernel::log1p::expm1_inplace(matrix);
    
    // Optionally renormalize
    normalize_total(matrix, target_sum);
}
```

## Performance Considerations

### SIMD Optimization

All operations use SIMD for vectorized computation:

```cpp
// SIMD-accelerated log1p
namespace s = scl::simd;
auto v = s::Load(d, vals + k);
s::Store(s::Log1p(d, v), d, vals + k);
```

### Parallelization

Operations are automatically parallelized:

```cpp
// Each row processed in parallel
threading::parallel_for(0, matrix.rows(), [&](size_t i) {
    // Process row i
});
```

### Prefetching

Prefetching reduces memory latency:

```cpp
if (k + PREFETCH_DISTANCE < len) {
    SCL_PREFETCH_READ(vals + k + PREFETCH_DISTANCE, 0);
}
```

## Numerical Stability

### Why log1p?

For small values of x, `log(1 + x)` is more accurate than `log(1.0 + x)`:

```cpp
// Good: Accurate for small x
Real result = std::log1p(x);

// Avoid: May lose precision for small x
Real result = std::log(1.0 + x);  // 1.0 + x may round to 1.0
```

### Why expm1?

Similarly, `expm1(x)` is more accurate than `exp(x) - 1`:

```cpp
// Good: Accurate for small x
Real result = std::expm1(x);

// Avoid: May lose precision for small x
Real result = std::exp(x) - 1.0;  // exp(x) ≈ 1.0, subtraction loses precision
```

## Configuration

```cpp
namespace scl::kernel::log1p::config {
    constexpr Size PREFETCH_DISTANCE = 64;
    constexpr double INV_LN2 = 1.44269504088896340736;  // 1 / ln(2)
    constexpr double LN2 = 0.6931471805599453;
}
```

## Related Documentation

- [Normalization](./normalize.md) - Normalization operations
- [Kernels Overview](./overview.md) - General kernel usage
- [SIMD](../core/simd.md) - SIMD operations
