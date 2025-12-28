# Scaling Operations

In-place scaling, shifting, and standardization of sparse matrices with SIMD optimization.

## Overview

Scaling kernels provide:

- **Standardization** - (x - mean) / std with optional clipping
- **Row Scaling** - Multiply each row by a scale factor
- **Row Shifting** - Add offset to each row
- **Adaptive SIMD** - Optimized for different row lengths
- **Zero-Centering Control** - Flexible standardization modes

## Standardization

### standardize

Standardize sparse matrix values in-place: `(x - mean) / std`, with optional clipping and zero-centering control:

```cpp
#include "scl/kernel/scale.hpp"

Sparse<Real, true> matrix = /* ... */;
Array<const Real> means = /* ... */;   // Per-row means [primary_dim]
Array<const Real> stds = /* ... */;    // Per-row standard deviations [primary_dim]

// Standardize with zero-centering and clipping
Real max_value = 10.0;  // Clip to [-10, 10]
bool zero_center = true;
scl::kernel::scale::standardize(matrix, means, stds, max_value, zero_center);

// Standardize without clipping
scl::kernel::scale::standardize(matrix, means, stds, Real(0), true);

// Scale only (no zero-centering)
scl::kernel::scale::standardize(matrix, means, stds, Real(0), false);
```

**Parameters:**
- `matrix`: Sparse matrix to standardize (modified in-place)
- `means`: Per-primary-dimension means
- `stds`: Per-primary-dimension standard deviations
- `max_value`: Clip threshold (0 disables clipping)
- `zero_center`: Whether to subtract mean before scaling

**Postconditions:**
- Each value transformed to: `(v - mean) / std` (if zero_center) or `v / std` (if not)
- Results clipped to `[-max_value, max_value]` if `max_value > 0`
- Rows with `std = 0` are unchanged (skipped)

**Algorithm:**
Uses 3-tier adaptive strategy based on row length:
- **Short (< 16)**: scalar loop
- **Medium (16-128)**: 4-way SIMD unroll
- **Long (>= 128)**: 8-way SIMD unroll with prefetch

Branch conditions (zero_center, do_clip) lifted outside inner loops for efficiency.

**Use cases:**
- Z-score normalization
- Feature scaling for machine learning
- Data preprocessing pipelines
- Standardization with outlier clipping

## Row Scaling

### scale_rows

Multiply each primary dimension by a corresponding scale factor:

```cpp
Array<const Real> scales = /* ... */;  // Per-row scale factors [primary_dim]

scl::kernel::scale::scale_rows(matrix, scales);
```

**Parameters:**
- `matrix`: Sparse matrix to scale (modified in-place)
- `scales`: Per-row scale factors, size = `primary_dim`

**Postconditions:**
- Each value in row `i` multiplied by `scales[i]`
- Rows with `scales[i] == 1` are unchanged (early exit optimization)

**Algorithm:**
- For each primary index in parallel:
  1. Skip if scale == 1 (optimization)
  2. Use SIMD 4-way unroll with prefetch for scaling

**Use cases:**
- Unit norm scaling
- Per-cell normalization factors
- Batch correction

## Row Shifting

### shift_rows

Add a constant offset to each primary dimension:

```cpp
Array<const Real> offsets = /* ... */;  // Per-row offsets [primary_dim]

scl::kernel::scale::shift_rows(matrix, offsets);
```

**Parameters:**
- `matrix`: Sparse matrix to shift (modified in-place)
- `offsets`: Per-row offsets to add, size = `primary_dim`

**Postconditions:**
- Each value in row `i` increased by `offsets[i]`
- Rows with `offsets[i] == 0` are unchanged (early exit optimization)

**Algorithm:**
- For each primary index in parallel:
  1. Skip if offset == 0 (optimization)
  2. Use SIMD 4-way unroll with prefetch for addition

**Numerical notes:**
- Only modifies stored (non-zero) values
- Implicit zeros remain zero
- For true shift of all values including zeros, matrix must be densified

**Use cases:**
- Mean centering (after computing means)
- Baseline correction
- Offset adjustments

## Examples

### Z-Score Normalization Pipeline

```cpp
#include "scl/kernel/scale.hpp"
#include "scl/kernel/feature.hpp"

Sparse<Real, true> counts = /* ... */;

// Step 1: Compute statistics
Array<Real> means(counts.rows());
Array<Real> vars(counts.rows());
scl::kernel::feature::standard_moments(counts, means, vars, 1);

// Step 2: Compute standard deviations
Array<Real> stds(counts.rows());
for (Index i = 0; i < counts.rows(); ++i) {
    stds[i] = std::sqrt(vars[i]);
}

// Step 3: Standardize (zero-center and scale)
scl::kernel::scale::standardize(counts, means, stds, Real(0), true);
```

### Standardization with Clipping

```cpp
// Standardize with outlier clipping
Real clip_threshold = 10.0;  // Clip to 10 standard deviations
scl::kernel::scale::standardize(counts, means, stds, clip_threshold, true);
```

### Per-Cell Total Count Normalization

```cpp
// Compute total counts per cell
Array<Real> totals(counts.rows());
scl::kernel::sparse::primary_sums(counts, totals);

// Compute scale factors (normalize to 10,000)
Real target_sum = 10000.0;
Array<Real> scales(counts.rows());
for (Index i = 0; i < counts.rows(); ++i) {
    scales[i] = (totals[i] > Real(0)) ? (target_sum / totals[i]) : Real(1);
}

// Scale rows
scl::kernel::scale::scale_rows(counts, scales);
```

### Mean Centering

```cpp
// Compute means
Array<Real> means(counts.rows());
Array<Real> vars(counts.rows());
scl::kernel::feature::standard_moments(counts, means, vars, 1);

// Negate means for shifting
Array<Real> offsets(counts.rows());
for (Index i = 0; i < counts.rows(); ++i) {
    offsets[i] = -means[i];
}

// Shift rows (center around zero)
scl::kernel::scale::shift_rows(counts, offsets);
```

## Performance Considerations

### Adaptive Strategy

The standardization function uses different algorithms based on row length:

- **Short rows**: Minimal overhead with scalar loops
- **Medium rows**: Balance between overhead and parallelism (4-way SIMD)
- **Long rows**: Maximum parallelism with aggressive prefetching (8-way SIMD)

### Early Exit Optimizations

- `scale_rows`: Skips rows with scale == 1
- `shift_rows`: Skips rows with offset == 0
- `standardize`: Skips rows with std == 0

### SIMD Operations

- Uses inverse of standard deviation (`inv_sigma = 1/std`) to replace division with multiplication
- Vectorized min/max for clipping operations
- 4-way and 8-way unrolling for better ILP (Instruction-Level Parallelism)

### Parallelization

- All operations parallelized over primary dimension
- Each thread processes independent rows
- No synchronization needed

---

::: tip Performance Tip
For standardization, pre-compute means and standard deviations once and reuse them for multiple matrices with the same structure.
:::

