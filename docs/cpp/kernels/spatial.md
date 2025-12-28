# Spatial Statistics

Spatial autocorrelation statistics (Moran's I, Geary's C) for spatial transcriptomics.

## Overview

Spatial statistics kernels provide:

- **Moran's I** - Measure spatial autocorrelation
- **Geary's C** - Alternative spatial autocorrelation measure
- **Weight Sum** - Compute total graph weight
- **High Performance** - SIMD-optimized with nested parallelism

## Moran's I

### morans_i

Compute Moran's I spatial autocorrelation statistic for each feature:

```cpp
#include "scl/kernel/spatial.hpp"

Sparse<Real, true> graph = /* ... */;        // Spatial weights matrix [n_cells x n_cells]
Sparse<Real, true> features = /* ... */;      // Feature matrix [n_features x n_cells]
Index n_features = features.rows();
Index n_cells = features.cols();

Array<Real> output(n_features);              // Pre-allocated output

scl::kernel::spatial::morans_i(graph, features, output);

// output[f] contains Moran's I for feature f
```

**Parameters:**
- `graph`: Spatial weights matrix, shape (n_cells, n_cells), must be square
- `features`: Feature matrix, shape (n_features, n_cells)
- `output`: Output array, must be pre-allocated, size = n_features

**Postconditions:**
- `output[f]` contains Moran's I for feature f
- Values typically in range [-1, 1]
- Positive values indicate spatial clustering
- Negative values indicate spatial dispersion
- Zero indicates random spatial pattern

**Algorithm:**
For each feature f:
1. Compute mean of feature values
2. Compute z = x - mean (centered values)
3. Compute numerator: sum_i(z_i * sum_j(w_ij * z_j))
4. Compute denominator: sum_i(z_i^2)
5. Moran's I = (N / W) * (numerator / denominator)

Optimizations:
- 8-way unrolled weighted neighbor sum with prefetch
- Multi-accumulator pattern for latency hiding
- Nested parallelism for single-feature large-cell case
- Block-based parallel reduction

**Complexity:**
- Time: O(n_features * nnz_graph)
- Space: O(n_cells * n_threads) for z buffers

**Thread Safety:**
- Safe - uses WorkspacePool for thread-local z arrays

**Use cases:**
- Spatial transcriptomics analysis
- Identify spatially variable genes
- Spatial pattern detection
- Spatial clustering analysis

## Geary's C

### gearys_c

Compute Geary's C spatial autocorrelation statistic:

```cpp
Sparse<Real, true> graph = /* ... */;
Sparse<Real, true> features = /* ... */;
Index n_features = features.rows();

Array<Real> output(n_features);              // Pre-allocated output

scl::kernel::spatial::gearys_c(graph, features, output);

// output[f] contains Geary's C for feature f
```

**Parameters:**
- `graph`: Spatial weights matrix, shape (n_cells, n_cells)
- `features`: Feature matrix, shape (n_features, n_cells)
- `output`: Output array, must be pre-allocated, size = n_features

**Postconditions:**
- `output[f]` contains Geary's C for feature f
- Values typically in range [0, 2]
- Values < 1 indicate positive autocorrelation
- Values > 1 indicate negative autocorrelation
- Value = 1 indicates no autocorrelation
- Geary's C is inversely related to Moran's I

**Complexity:**
- Time: O(n_features * nnz_graph)
- Space: O(n_cells * n_threads) for z buffers

**Thread Safety:**
- Safe - uses WorkspacePool for thread-local z arrays

**Numerical Notes:**
- Returns 0 for features with zero variance
- Returns 0 if total graph weight is zero
- Geary's C is inversely related to Moran's I

## Weight Sum

### weight_sum

Compute sum of all edge weights in a sparse graph:

```cpp
Sparse<Real, true> graph = /* ... */;
Real total_weight;

scl::kernel::spatial::weight_sum(graph, total_weight);

// total_weight contains sum of all non-zero values
```

**Parameters:**
- `graph`: Sparse graph matrix (CSR or CSC format)
- `out_sum`: Output scalar receiving total weight sum

**Postconditions:**
- `out_sum` contains sum of all non-zero values in graph
- If graph has no edges, out_sum = 0

**Algorithm:**
1. Partition rows across threads
2. Each thread computes partial sum using SIMD vectorize::sum
3. Reduce partial sums to final result

**Complexity:**
- Time: O(nnz / n_threads)
- Space: O(n_threads) for partial sums

**Thread Safety:**
- Safe - uses WorkspacePool for thread-local storage

---

::: tip Moran's I vs. Geary's C
Moran's I is more sensitive to global patterns, while Geary's C emphasizes local differences. Use both for comprehensive spatial analysis.
:::

