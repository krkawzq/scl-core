---
title: Kernels Index
description: Complete index of computational kernels
---

# Kernels Index

Complete reference for all computational kernels in SCL-Core.

## Data Processing

### Normalization
- **[normalize](./normalize.md)** - Normalization operations (total count, L1, L2)
- **[scale](./scale.md)** - Scaling and standardization (z-score)
- **[log1p](./log1p.md)** - Logarithmic transforms (log1p, log2p1, expm1)
- **[softmax](./softmax.md)** - Softmax activation

### Transformation
- **[impute](./impute.md)** - Missing value imputation
- **[slice](./slice.md)** - Matrix slicing utilities
- **[reorder](./reorder.md)** - Reordering operations
- **[merge](./merge.md)** - Matrix merging

## Neighbor Search

- **[neighbors](./neighbors.md)** - K-nearest neighbors search
- **[bbknn](./bbknn.md)** - Batch-balanced KNN
- **[spatial](./spatial.md)** - Spatial neighbor search and statistics

## Graph Algorithms

- **[louvain](./louvain.md)** - Louvain community detection
- **[leiden](./leiden.md)** - Leiden algorithm (improved Louvain)
- **[components](./components.md)** - Connected components
- **[centrality](./centrality.md)** - Centrality measures
- **[gnn](./gnn.md)** - Graph neural network operations

## Feature Selection

- **[hvg](./hvg.md)** - Highly variable gene selection
- **[markers](./markers.md)** - Marker gene detection
- **[feature](./feature.md)** - Feature selection utilities

## Statistical Analysis

- **[statistics](./statistics.md)** - Statistical tests (t-test, MWU, ANOVA, etc.)
- **[mwu](./mwu.md)** - Mann-Whitney U test
- **[correlation](./correlation.md)** - Correlation computation
- **[metrics](./metrics.md)** - Distance metrics and similarity measures

## Dimensionality Reduction

- **[projection](./projection.md)** - Random projection
- **[diffusion](./diffusion.md)** - Diffusion maps
- **[pseudotime](./pseudotime.md)** - Pseudotime inference

## Clustering and Classification

- **[subpopulation](./subpopulation.md)** - Subpopulation identification
- **[doublet](./doublet.md)** - Doublet detection
- **[outlier](./outlier.md)** - Outlier detection

## Spatial Analysis

- **[spatial](./spatial.md)** - Spatial autocorrelation (Moran's I)
- **[spatial_pattern](./spatial_pattern.md)** - Spatial pattern detection
- **[hotspot](./hotspot.md)** - Hotspot detection

## Other Operations

- **[algebra](./algebra.md)** - Matrix algebra operations
- **[gram](./gram.md)** - Gram matrix computation
- **[sampling](./sampling.md)** - Sampling strategies
- **[qc](./qc.md)** - Quality control metrics

## Quick Reference

### Most Common Operations

```cpp
// Normalization
kernel::normalize::normalize_rows_inplace(matrix, target_sum);
kernel::log1p::log1p_inplace(matrix);

// Neighbor search
CSR knn = kernel::neighbors::knn_sparse(matrix, k=15);

// Clustering
kernel::louvain::louvain(graph, communities);

// Feature selection
kernel::hvg::select_hvg(matrix, dispersions, n_top, hvg_indices);

// Statistical tests
Real t_stat = kernel::stat::ttest::t_test(group1, group2);
```

## Performance Tips

1. **Use in-place operations** when possible to avoid extra allocations
2. **Choose appropriate sparse format** (CSR for row ops, CSC for column ops)
3. **Leverage automatic parallelization** - kernels parallelize automatically
4. **Pre-allocate output buffers** for better performance
5. **Use SIMD-optimized kernels** - all kernels use SIMD where applicable

## Related Documentation

- [Kernels Overview](./overview.md) - General kernel usage patterns
- [Core Types](../core/types.md) - Type system
- [Sparse Matrices](../core/sparse.md) - Sparse matrix operations
- [Threading](../threading.md) - Parallel execution

