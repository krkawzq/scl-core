# correlation.hpp

> scl/kernel/correlation.hpp · Pearson correlation kernels with SIMD optimization

## Overview

This file provides efficient computation of Pearson correlation matrices for sparse feature matrices. It includes statistics computation (mean and inverse standard deviation) and full pairwise correlation matrix computation. All operations are SIMD-accelerated and parallelized with cache-blocking optimizations.

**Header**: `#include "scl/kernel/correlation.hpp"`

---

## Main APIs

### compute_stats

::: source_code file="scl/kernel/correlation.hpp" symbol="compute_stats" collapsed
:::

**Algorithm Description**

Computes mean and inverse standard deviation for each row (feature) in parallel:

1. For each row in parallel:
   - Fused sum and sum-of-squares computation using 4-way SIMD unrolling
   - Load 4 values per iteration, accumulate in dual accumulators (v_sum0, v_sum1, v_sq0, v_sq1)
   - Use FMA (fused multiply-add) for sum-of-squares: `v_sq = v_sq + v * v`
   - Reduce to scalar using SumOfLanes
2. Compute mean: `mean = sum / n_samples`
3. Compute variance: `var = sum_sq / n_samples - mean^2`
4. Compute inverse std: `inv_std = 1 / sqrt(var)` if var > 0, else 0
5. Variance clamped to >= 0 for numerical stability

**Edge Cases**

- **Zero-variance rows**: Rows with zero variance get `inv_std = 0` (prevents division by zero)
- **Empty rows**: Rows with no non-zero values have mean = 0, inv_std = 0
- **Constant rows**: Rows with all same value have variance = 0, inv_std = 0
- **Numerical precision**: Variance clamped to >= 0 to handle floating-point errors

**Data Guarantees (Preconditions)**

- `out_means.len >= matrix.primary_dim()`
- `out_inv_stds.len >= matrix.primary_dim()`
- Matrix is valid sparse format (CSR or CSC)
- Matrix shape is (n_features, n_samples)

**Complexity Analysis**

- **Time**: O(nnz / n_threads) - parallelized over rows with SIMD acceleration
- **Space**: O(1) auxiliary - only accumulators needed

**Example**

```cpp
#include "scl/kernel/correlation.hpp"

Sparse<Real, true> matrix = /* sparse matrix (n_features, n_samples) */;
Array<Real> means(matrix.primary_dim());
Array<Real> inv_stds(matrix.primary_dim());

scl::kernel::correlation::compute_stats(matrix, means, inv_stds);

// Use precomputed stats for correlation computation
Array<Real> corr_matrix(n_features * n_features);
scl::kernel::correlation::pearson(matrix, means, inv_stds, corr_matrix);
```

---

### pearson

::: source_code file="scl/kernel/correlation.hpp" symbol="pearson" collapsed
:::

**Algorithm Description**

Computes full pairwise Pearson correlation matrix with multiple optimizations:

1. **Symmetric computation**: Only compute upper triangle, copy to lower triangle
2. **Sparse centered dot product**: Use algebraic identity to avoid materializing dense vectors:
   - `cov(a,b) = sum(a*b) - mean_a*sum(b) - mean_b*sum(a) + n*mean_a*mean_b`
   - Where sum(a*b) computed via sparse merge of non-zero indices
3. **8/4-way skip optimization**: In sparse merge, skip 8 elements at a time when indices are far apart
4. **Chunk-based parallelization**: Process rows in chunks for cache locality
5. **Early skip**: Skip zero-variance features (correlation = 0 with all others)
6. **Correlation formula**: `corr(a,b) = cov(a,b) / (std_a * std_b) = cov(a,b) * inv_std_a * inv_std_b`
7. **Clamping**: Correlation values clamped to [-1, 1] for numerical stability

**Edge Cases**

- **Zero-variance features**: Features with zero variance have correlation 0 with all others
- **Perfect correlation**: Returns exactly 1.0 or -1.0 for perfectly correlated features
- **Sparse features**: Features with very few non-zeros handled efficiently via sparse merge
- **Numerical overflow**: Clamping prevents values outside [-1, 1] range

**Data Guarantees (Preconditions)**

- `output.len >= n_features^2`
- If using overload with means/inv_stds: `means.len >= n_features` and `inv_stds.len >= n_features`
- Matrix is valid sparse format
- Matrix shape is (n_features, n_samples)
- If stats provided, they must match matrix dimensions

**Complexity Analysis**

- **Time**: O(n_features^2 * avg_nnz_per_row / n_threads) - quadratic in features, linear in sparsity
- **Space**: O(1) beyond output - no temporary matrices needed

**Example**

```cpp
// Option 1: Compute stats internally
Array<Real> corr_matrix(n_features * n_features);
scl::kernel::correlation::pearson(matrix, corr_matrix);

// Option 2: Use precomputed stats (faster if stats already computed)
Array<Real> means(n_features);
Array<Real> inv_stds(n_features);
scl::kernel::correlation::compute_stats(matrix, means, inv_stds);
scl::kernel::correlation::pearson(matrix, means, inv_stds, corr_matrix);

// Access correlation between feature i and j
Real corr_ij = corr_matrix[i * n_features + j];  // Symmetric: corr_ij == corr_ji
```

---

## Configuration

Internal configuration constants (not exposed in API):

- `CHUNK_SIZE = 64`: Row chunk size for cache blocking
- `STAT_CHUNK = 256`: Statistics computation chunk size
- `PREFETCH_DISTANCE = 32`: Elements to prefetch ahead for cache optimization

---

## Performance Notes

### SIMD Acceleration

- 4-way unrolled operations for maximum throughput
- FMA (fused multiply-add) instructions for sum-of-squares
- Dual accumulators reduce dependency chains

### Cache Optimization

- Chunk-based processing improves cache locality
- Prefetching reduces memory latency
- Sparse merge avoids materializing dense vectors

### Parallelization

- Scales linearly with CPU cores
- Chunk-based scheduling for load balancing
- No shared state conflicts

---

## See Also

- [Vectorize Module](../core/vectorize) - SIMD-optimized operations
- [Sparse Matrix](../core/sparse) - Sparse matrix operations
