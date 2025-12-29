---
title: Scaling and Standardization
description: Scaling operations with SIMD optimization
---

# Scaling and Standardization

The `scale` kernel provides efficient scaling and standardization operations for sparse matrices, including z-score normalization and clipping.

## Overview

Scaling operations are used for:
- Standardizing features (z-score normalization)
- Clipping extreme values
- Preparing data for machine learning algorithms

All operations are SIMD-optimized with adaptive algorithms for different data sizes.

## Functions

### Standardization

#### `standardize_primary`

Standardize each row (CSR) or column (CSC) to zero mean and unit variance.

```cpp
template <typename T, bool IsCSR>
void standardize_primary(
    Sparse<T, IsCSR>& matrix,
    Array<const Real> means,
    Array<const Real> std_devs,
    bool zero_center = true,
    Real max_value = Real(10.0)
);
```

**Parameters**:
- `matrix` [in,out]: Matrix to standardize (modified in-place)
- `means` [in]: Mean values for each row/column
- `std_devs` [in]: Standard deviations for each row/column
- `zero_center` [in]: Whether to subtract mean (default: true)
- `max_value` [in]: Maximum absolute value for clipping (default: 10.0)

**Mathematical Operation**: `x → clip((x - μ) / σ, -max_val, max_val)`

**Example**:
```cpp
#include "scl/kernel/scale.hpp"

// Compute means and standard deviations
auto means = memory::aligned_alloc<Real>(matrix.rows());
auto std_devs = memory::aligned_alloc<Real>(matrix.rows());
Array<Real> means_view = {means.get(), static_cast<Size>(matrix.rows())};
Array<Real> std_devs_view = {std_devs.get(), static_cast<Size>(matrix.rows())};

compute_statistics(matrix, means_view, std_devs_view);

// Standardize matrix
kernel::scale::standardize_primary(
    matrix, means_view, std_devs_view,
    true,  // zero_center
    Real(10.0)  // max_value for clipping
);
```

**Complexity**: O(nnz) time, O(1) space

### Scaling

#### `scale_primary`

Scale each row/column by a factor.

```cpp
template <typename T, bool IsCSR>
void scale_primary(
    Sparse<T, IsCSR>& matrix,
    Array<const Real> scales
);
```

**Parameters**:
- `matrix` [in,out]: Matrix to scale (modified in-place)
- `scales` [in]: Scaling factors for each row/column

**Example**:
```cpp
// Scale each row by its standard deviation
auto scales = memory::aligned_alloc<Real>(matrix.rows());
Array<Real> scales_view = {scales.get(), static_cast<Size>(matrix.rows())};

for (Index i = 0; i < matrix.rows(); ++i) {
    Real std_dev = compute_std_dev(matrix, i);
    scales_view[i] = (std_dev > Real(1e-10)) ? Real(1) / std_dev : Real(0);
}

kernel::scale::scale_primary(matrix, scales_view);
```

## Adaptive Algorithms

The scale kernel uses adaptive algorithms based on data size:

### Short Arrays (≤ 16 elements)

Uses scalar implementation for small arrays:

```cpp
// Scalar fallback for short arrays
void standardize_short(
    T* vals, Size len,
    T mu, T inv_sigma,
    T max_val,
    bool zero_center, bool do_clip
);
```

### Medium Arrays (17-128 elements)

Uses SIMD with 4-way unrolling:

```cpp
// SIMD with 4-way unrolling
void standardize_medium(
    T* vals, Size len,
    T mu, T inv_sigma,
    T max_val,
    bool zero_center, bool do_clip
);
```

### Long Arrays (> 128 elements)

Uses SIMD with 8-way unrolling and prefetching:

```cpp
// SIMD with 8-way unrolling and prefetching
void standardize_long(
    T* vals, Size len,
    T mu, T inv_sigma,
    T max_val,
    bool zero_center, bool do_clip
);
```

## Common Patterns

### Z-Score Normalization

```cpp
void z_score_normalize(CSR& matrix) {
    Index n_rows = matrix.rows();
    
    // Compute means
    auto means = memory::aligned_alloc<Real>(n_rows);
    Array<Real> means_view = {means.get(), static_cast<Size>(n_rows)};
    kernel::normalize::compute_row_sums(matrix, means_view);
    
    // Compute row lengths for mean
    for (Index i = 0; i < n_rows; ++i) {
        Index len = matrix.row_length(i);
        if (len > 0) {
            means_view[i] /= static_cast<Real>(len);
        }
    }
    
    // Compute standard deviations
    auto std_devs = memory::aligned_alloc<Real>(n_rows);
    Array<Real> std_devs_view = {std_devs.get(), static_cast<Size>(n_rows)};
    
    for (Index i = 0; i < n_rows; ++i) {
        Real mean = means_view[i];
        Real sum_sq = Real(0);
        auto values = matrix.row_values(i);
        Index len = matrix.row_length(i);
        
        for (Index k = 0; k < len; ++k) {
            Real diff = static_cast<Real>(values[k]) - mean;
            sum_sq += diff * diff;
        }
        
        Real var = (len > 1) ? sum_sq / static_cast<Real>(len - 1) : Real(0);
        std_devs_view[i] = std::sqrt(var);
    }
    
    // Standardize
    kernel::scale::standardize_primary(
        matrix, means_view, std_devs_view,
        true,   // zero_center
        Real(10.0)  // clip at ±10
    );
}
```

### Min-Max Scaling

```cpp
void min_max_scale(CSR& matrix, Real min_val, Real max_val) {
    Index n_rows = matrix.rows();
    
    // Find min and max per row
    auto mins = memory::aligned_alloc<Real>(n_rows);
    auto maxs = memory::aligned_alloc<Real>(n_rows);
    Array<Real> mins_view = {mins.get(), static_cast<Size>(n_rows)};
    Array<Real> maxs_view = {maxs.get(), static_cast<Size>(n_rows)};
    
    for (Index i = 0; i < n_rows; ++i) {
        auto values = matrix.row_values(i);
        Index len = matrix.row_length(i);
        
        if (len == 0) {
            mins_view[i] = Real(0);
            maxs_view[i] = Real(0);
            continue;
        }
        
        Real row_min = static_cast<Real>(values[0]);
        Real row_max = static_cast<Real>(values[0]);
        
        for (Index k = 1; k < len; ++k) {
            Real val = static_cast<Real>(values[k]);
            row_min = std::min(row_min, val);
            row_max = std::max(row_max, val);
        }
        
        mins_view[i] = row_min;
        maxs_view[i] = row_max;
    }
    
    // Scale to [min_val, max_val]
    auto scales = memory::aligned_alloc<Real>(n_rows);
    auto offsets = memory::aligned_alloc<Real>(n_rows);
    Array<Real> scales_view = {scales.get(), static_cast<Size>(n_rows)};
    Array<Real> offsets_view = {offsets.get(), static_cast<Size>(n_rows)};
    
    for (Index i = 0; i < n_rows; ++i) {
        Real range = maxs_view[i] - mins_view[i];
        if (range > Real(1e-10)) {
            scales_view[i] = (max_val - min_val) / range;
            offsets_view[i] = min_val - mins_view[i] * scales_view[i];
        } else {
            scales_view[i] = Real(1);
            offsets_view[i] = Real(0);
        }
    }
    
    // Apply scaling
    kernel::scale::scale_primary(matrix, scales_view);
    // Note: Adding offsets requires a separate operation
}
```

## Performance Considerations

### SIMD Optimization

All operations use SIMD for vectorized computation:

```cpp
// SIMD-accelerated standardization
namespace s = scl::simd;
auto v = s::Load(d, vals + k);
if (zero_center) v = s::Sub(v, v_mu);
v = s::Mul(v, v_inv_sigma);
if (do_clip) v = s::Min(s::Max(v, v_min), v_max);
s::Store(v, d, vals + k);
```

### Adaptive Thresholds

Algorithm selection based on array length:

```cpp
namespace config {
    constexpr Size SHORT_THRESHOLD = 16;   // Use scalar
    constexpr Size MEDIUM_THRESHOLD = 128;  // Use 4-way SIMD
    // > 128: Use 8-way SIMD with prefetching
}
```

### Prefetching

Prefetching reduces memory latency for long arrays:

```cpp
if (k + PREFETCH_DISTANCE < len) {
    SCL_PREFETCH_READ(vals + k + PREFETCH_DISTANCE, 0);
}
```

## Configuration

```cpp
namespace scl::kernel::scale::config {
    constexpr Size SHORT_THRESHOLD = 16;
    constexpr Size MEDIUM_THRESHOLD = 128;
    constexpr Size PREFETCH_DISTANCE = 16;
}
```

## Related Documentation

- [Normalization](./normalize.md) - Normalization operations
- [Kernels Overview](./overview.md) - General kernel usage
- [SIMD](../core/simd.md) - SIMD operations
