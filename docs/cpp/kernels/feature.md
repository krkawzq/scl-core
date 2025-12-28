# feature.hpp

> scl/kernel/feature.hpp · Feature statistics with SIMD optimization

## Overview

This file provides efficient computation of feature-level statistics for sparse matrices:

- **Mean and Variance**: Compute moments accounting for implicit zeros
- **Clipped Moments**: Statistics with per-row clipping for robust analysis
- **Detection Rate**: Fraction of non-zero entries per dimension
- **Dispersion**: Variance-to-mean ratio for feature selection

All operations use SIMD-optimized fused sum and sum-of-squares computation, and are parallelized over the primary dimension.

**Header**: `#include "scl/kernel/feature.hpp"`

---

## Main APIs

### standard_moments

::: source_code file="scl/kernel/feature.hpp" symbol="standard_moments" collapsed
:::

**Algorithm Description**

Compute mean and variance for each primary dimension (row/column) of a sparse matrix, accounting for implicit zeros:

1. For each primary index in parallel:
   - Iterate over non-zero elements in the primary dimension
   - Use SIMD-optimized fused computation: accumulate sum and sum-of-squares together in a single pass
   - Apply 4-way unroll with multiple accumulators to hide latency
2. Calculate mean: `mean = sum / N` where `N = secondary_dim` (total columns/rows)
3. Calculate variance: `variance = (sumsq - sum * mean) / (N - ddof)`
4. Clamp variance to non-negative to handle numerical errors

The SIMD implementation uses 4-way unrolling with prefetching for optimal cache performance.

**Edge Cases**

- **Empty rows/columns**: Mean = 0, variance = 0
- **Single element**: Variance = 0 (if ddof=0) or undefined (if ddof=1 and N=1)
- **All zeros**: Mean = 0, variance = 0
- **Very large values**: Standard floating-point arithmetic, no overflow protection

**Data Guarantees (Preconditions)**

- `out_means.len >= matrix.primary_dim()`
- `out_vars.len >= matrix.primary_dim()`
- Matrix must be valid sparse format (CSR or CSC)
- `ddof` must be non-negative and less than `secondary_dim`

**Complexity Analysis**

- **Time**: O(nnz) where nnz is the number of non-zeros, parallelized over primary dimension
- **Space**: O(1) auxiliary space per thread (accumulators only)

**Example**

```cpp
#include "scl/kernel/feature.hpp"

// Create sparse matrix (cells x genes)
scl::Sparse<Real, true> counts = /* ... */;

// Pre-allocate output buffers
scl::Array<Real> means(counts.rows());
scl::Array<Real> vars(counts.rows());

// Compute with sample variance (ddof=1)
scl::kernel::feature::standard_moments(counts, means, vars, 1);

// Use for quality control
for (scl::Index i = 0; i < counts.rows(); ++i) {
    if (means[i] < min_mean_threshold) {
        // Filter out low-quality cell
    }
}
```

---

### clipped_moments

::: source_code file="scl/kernel/feature.hpp" symbol="clipped_moments" collapsed
:::

**Algorithm Description**

Compute mean and variance with per-row clipping of maximum values for robust statistics:

1. For each primary index in parallel:
   - Load values and clip to threshold using SIMD min operations: `value = min(value, clip_vals[i])`
   - Compute fused sum and sum-of-squares on clipped values
   - Calculate mean and variance with ddof=1 (sample variance)

This approach provides outlier-resistant statistics by capping extreme values before computation.

**Edge Cases**

- **Empty rows/columns**: Mean = 0, variance = 0
- **All values below threshold**: No clipping occurs, result same as standard_moments
- **All values above threshold**: All values clipped to threshold, variance = 0
- **Zero clip threshold**: All values become zero, mean = 0, variance = 0

**Data Guarantees (Preconditions)**

- All buffer sizes >= `matrix.primary_dim()`
- `clip_vals` contains positive values (negative thresholds not supported)
- Matrix must be valid sparse format

**Complexity Analysis**

- **Time**: O(nnz) parallelized over primary dimension
- **Space**: O(1) auxiliary space per thread

**Example**

```cpp
#include "scl/kernel/feature.hpp"

scl::Sparse<Real, true> counts = /* ... */;

// Define clip thresholds (e.g., 99th percentile per cell)
scl::Array<Real> clip_vals(counts.rows());
// ... compute clip_vals from percentiles ...

scl::Array<Real> robust_means(counts.rows());
scl::Array<Real> robust_vars(counts.rows());

// Compute robust statistics with clipping
scl::kernel::feature::clipped_moments(
    counts, clip_vals, robust_means, robust_vars
);

// Use for outlier-resistant analysis
```

---

### detection_rate

::: source_code file="scl/kernel/feature.hpp" symbol="detection_rate" collapsed
:::

**Algorithm Description**

Compute the fraction of non-zero entries per primary dimension:

1. For each primary index in parallel:
   - Count non-zero elements in the primary dimension
   - Calculate rate: `rate = nnz_count / secondary_dim`
   - Result is in range [0, 1]

This is a simple O(primary_dim) operation that reads the sparse matrix structure.

**Edge Cases**

- **Empty rows/columns**: Rate = 0
- **Fully dense rows/columns**: Rate = 1.0
- **Zero secondary dimension**: Division by zero avoided, rate = 0

**Data Guarantees (Preconditions)**

- `out_rates.len >= matrix.primary_dim()`
- Matrix must be valid sparse format

**Complexity Analysis**

- **Time**: O(primary_dim) - only reads structure, not values
- **Space**: O(1) auxiliary space

**Example**

```cpp
#include "scl/kernel/feature.hpp"

scl::Sparse<Real, true> counts = /* ... */;
scl::Array<Real> rates(counts.rows());

scl::kernel::feature::detection_rate(counts, rates);

// Filter cells by detection rate
for (scl::Index i = 0; i < counts.rows(); ++i) {
    if (rates[i] < min_detection_rate) {
        // Filter out low-detection cell
    }
}
```

---

### dispersion

::: source_code file="scl/kernel/feature.hpp" symbol="dispersion" collapsed
:::

**Algorithm Description**

Compute dispersion index (variance / mean) for each feature:

1. For each element using 4-way SIMD unroll:
   - Load mean and variance vectors
   - Create mask for `mean > epsilon` (epsilon = 1e-12)
   - Compute division with masked select: `dispersion = (mean > epsilon) ? variance / mean : 0`

The dispersion index measures variability relative to the mean, useful for identifying highly variable features.

**Edge Cases**

- **Zero mean**: Dispersion = 0 (avoid division by zero)
- **Negative variance**: Should not occur, but handled by clamping in variance computation
- **Very small mean**: Uses epsilon threshold to avoid numerical instability

**Data Guarantees (Preconditions)**

- All arrays have same length
- `means` and `vars` computed from same data
- Variances should be non-negative (from standard_moments or clipped_moments)

**Complexity Analysis**

- **Time**: O(n) with SIMD 4-way unroll
- **Space**: O(1) auxiliary space

**Example**

```cpp
#include "scl/kernel/feature.hpp"

// Pre-computed statistics
scl::Array<Real> means = /* ... */;
scl::Array<Real> vars = /* ... */;
scl::Array<Real> dispersion(means.len());

scl::kernel::feature::dispersion(means, vars, dispersion);

// Select highly variable genes
std::vector<scl::Index> hvg_indices;
for (scl::Index i = 0; i < dispersion.len(); ++i) {
    if (dispersion[i] > threshold && means[i] > min_mean) {
        hvg_indices.push_back(i);
    }
}
```

---

## Utility Functions

### detail::compute_sum_sq_simd

SIMD-optimized fused sum and sum-of-squares computation.

::: source_code file="scl/kernel/feature.hpp" symbol="compute_sum_sq_simd" collapsed
:::

**Complexity**

- Time: O(n) with SIMD acceleration
- Space: O(1)

---

### detail::compute_clipped_sum_sq_simd

SIMD-optimized clipped sum and sum-of-squares computation.

::: source_code file="scl/kernel/feature.hpp" symbol="compute_clipped_sum_sq_simd" collapsed
:::

**Complexity**

- Time: O(n) with SIMD acceleration
- Space: O(1)

---

## Notes

- All operations are parallelized by default using `scl::threading::parallel_for`
- SIMD operations use Google Highway library for optimal performance
- Variance computation uses the numerically stable formula: `var = (sumsq - sum * mean) / (N - ddof)`
- For gene-level statistics, transpose the matrix or use CSC format (genes as primary dimension)

## See Also

- [Sparse Matrix Operations](../core/sparse)
- [SIMD Operations](../core/simd)
- [Vectorized Operations](../core/vectorize)
