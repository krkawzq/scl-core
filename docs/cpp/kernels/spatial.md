# spatial.hpp

> scl/kernel/spatial.hpp · Spatial autocorrelation statistics kernels

## Overview

This file provides high-performance spatial autocorrelation statistics for spatial transcriptomics and spatial data analysis. It implements Moran's I and Geary's C statistics with SIMD optimization and nested parallelism for efficient computation on large datasets.

This file provides:
- Moran's I spatial autocorrelation statistic
- Geary's C spatial autocorrelation statistic
- Weight sum computation for graph matrices
- SIMD-optimized computation with 8-way unrolling
- Nested parallelism for large-scale analysis

**Header**: `#include "scl/kernel/spatial.hpp"`

---

## Main APIs

### morans_i

::: source_code file="scl/kernel/spatial.hpp" symbol="morans_i" collapsed
:::

**Algorithm Description**

Compute Moran's I spatial autocorrelation statistic for each feature:

1. **Centering**: For each feature f:
   - Compute mean: mean = sum(x[f, :]) / n_cells
   - Center values: z[i] = x[f, i] - mean

2. **Numerator computation**: For each feature f:
   - Compute weighted neighbor sum: lag[i] = sum_j(w_ij * z[j])
   - Uses 8-way unrolled loop with prefetch for efficiency
   - Multi-accumulator pattern for latency hiding
   - Accumulate: numerator += sum_i(z[i] * lag[i])

3. **Denominator computation**: For each feature f:
   - denominator = sum_i(z[i]^2)

4. **Moran's I computation**:
   - I = (N / W) * (numerator / denominator)
   - Where N = n_cells, W = sum of all weights

5. **Optimization strategies**:
   - 8-way unrolled weighted neighbor sum with prefetch
   - Multi-accumulator pattern for FP latency hiding
   - Nested parallelism for single-feature large-cell case
   - Block-based parallel reduction

**Edge Cases**

- **Zero variance**: Features with zero variance return I = 0
- **Zero total weight**: If W = 0, returns I = 0
- **Empty graph**: Graph with no edges returns I = 0
- **Constant features**: Features with constant values return I = 0

**Data Guarantees (Preconditions)**

- `graph` must be square: graph.rows() == graph.cols()
- `features.secondary_dim() == graph.primary_dim()`
- `output.len == features.primary_dim()`
- Graph weights should be non-negative (typically row-normalized)
- Graph must be valid sparse matrix format

**Complexity Analysis**

- **Time**: O(n_features * nnz_graph)
  - O(nnz_graph) per feature for neighbor sum computation
  - n_features features processed
- **Space**: O(n_cells * n_threads) for thread-local z buffers
  - WorkspacePool allocates buffers per thread

**Example**

```cpp
#include "scl/kernel/spatial.hpp"

// Spatial weights matrix: cells x cells
Sparse<Real, true> graph = /* ... */;
Index n_cells = graph.rows();

// Feature matrix: features x cells
Sparse<Real, true> features = /* ... */;
Index n_features = features.rows();

// Pre-allocate output
Array<Real> morans_i_scores(n_features);

// Compute Moran's I for all features
scl::kernel::spatial::morans_i(
    graph,
    features,
    morans_i_scores
);

// Interpret results
for (Index f = 0; f < n_features; ++f) {
    Real I = morans_i_scores[f];
    if (I > 0) {
        // Positive spatial autocorrelation (clustering)
    } else if (I < 0) {
        // Negative spatial autocorrelation (dispersion)
    } else {
        // Random spatial pattern
    }
}
```

---

### gearys_c

::: source_code file="scl/kernel/spatial.hpp" symbol="gearys_c" collapsed
:::

**Algorithm Description**

Compute Geary's C spatial autocorrelation statistic for each feature:

1. **Centering**: For each feature f:
   - Compute mean and center values: z[i] = x[f, i] - mean

2. **Numerator computation**: For each feature f:
   - Compute sum of weighted squared differences:
     numerator = sum_ij(w_ij * (z[i] - z[j])^2)
   - Uses 8-way unrolled loop with multi-accumulator pattern
   - Aggressive prefetching for indirect z access
   - 4-way cleanup loop for remainder

3. **Denominator computation**: For each feature f:
   - denominator = 2 * W * sum_i(z[i]^2)

4. **Geary's C computation**:
   - C = (N-1) * numerator / denominator

5. **Optimization strategies**:
   - 8-way unrolled difference squared with multi-accumulator
   - Aggressive prefetching for indirect z access
   - Nested parallelism for single-feature large-cell case

**Edge Cases**

- **Zero variance**: Features with zero variance return C = 0
- **Zero total weight**: If W = 0, returns C = 0
- **Empty graph**: Graph with no edges returns C = 0
- **Constant features**: Features with constant values return C = 0

**Data Guarantees (Preconditions)**

- `graph` must be square: graph.rows() == graph.cols()
- `features.secondary_dim() == graph.primary_dim()`
- `output.len == features.primary_dim()`
- Graph weights should be non-negative
- Graph must be valid sparse matrix format

**Complexity Analysis**

- **Time**: O(n_features * nnz_graph)
  - O(nnz_graph) per feature for difference computation
  - n_features features processed
- **Space**: O(n_cells * n_threads) for thread-local z buffers

**Example**

```cpp
// Compute Geary's C for all features
Array<Real> gearys_c_scores(n_features);

scl::kernel::spatial::gearys_c(
    graph,
    features,
    gearys_c_scores
);

// Interpret results (inverse of Moran's I)
for (Index f = 0; f < n_features; ++f) {
    Real C = gearys_c_scores[f];
    if (C < 1) {
        // Positive spatial autocorrelation (clustering)
    } else if (C > 1) {
        // Negative spatial autocorrelation (dispersion)
    } else {
        // Random spatial pattern (C ≈ 1)
    }
}
```

---

### weight_sum

::: source_code file="scl/kernel/spatial.hpp" symbol="weight_sum" collapsed
:::

**Algorithm Description**

Compute sum of all edge weights in a sparse graph:

1. **Parallel partitioning**: Partition rows across threads
2. **Partial sum computation**: Each thread computes partial sum using SIMD vectorize::sum
3. **Reduction**: Reduce partial sums to final result

**Edge Cases**

- **Empty graph**: Returns 0 if graph has no edges
- **Zero weights**: Handles zero-weighted edges correctly

**Data Guarantees (Preconditions)**

- `graph` must be valid sparse matrix

**Complexity Analysis**

- **Time**: O(nnz / n_threads) - parallel reduction
- **Space**: O(n_threads) for partial sums

**Example**

```cpp
Real total_weight;
scl::kernel::spatial::weight_sum(graph, total_weight);
```

---

## Utility Functions

### detail::compute_weighted_neighbor_sum

Internal helper function that computes weighted sum of neighbor values with adaptive optimization.

::: source_code file="scl/kernel/spatial.hpp" symbol="detail::compute_weighted_neighbor_sum" collapsed
:::

**Complexity**

- Time: O(len) where len is number of neighbors
- Space: O(1)

---

### detail::compute_moran_numer_block

Internal helper function that computes Moran's I numerator for a block of cells.

::: source_code file="scl/kernel/spatial.hpp" symbol="detail::compute_moran_numer_block" collapsed
:::

**Complexity**

- Time: O(sum of neighbor counts in block)
- Space: O(1)

---

### detail::compute_geary_numer_block

Internal helper function that computes Geary's C numerator for a block of cells.

::: source_code file="scl/kernel/spatial.hpp" symbol="detail::compute_geary_numer_block" collapsed
:::

**Complexity**

- Time: O(sum of neighbor counts in block)
- Space: O(1)

---

## Notes

**Moran's I vs Geary's C**:
- Moran's I: Range [-1, 1], positive indicates clustering
- Geary's C: Range [0, 2], C < 1 indicates clustering
- Geary's C is inversely related to Moran's I
- Both measure spatial autocorrelation but with different formulations

**Performance**:
- SIMD-optimized with 8-way unrolling
- Multi-accumulator pattern for FP latency hiding
- Nested parallelism for large-scale analysis
- WorkspacePool for efficient thread-local storage

**Typical Usage**:
- Spatial transcriptomics analysis
- Spatial pattern detection
- Spatial autocorrelation testing
- Feature selection based on spatial structure

## See Also

- [Hotspot Detection](/cpp/kernels/hotspot) - Local spatial statistics and hotspot detection
- [Spatial Pattern](/cpp/kernels/spatial_pattern) - Spatial pattern detection methods
