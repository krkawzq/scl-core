---
title: Spatial Analysis
description: Spatial statistics and neighbor search
---

# Spatial Analysis

The `spatial` kernel provides spatial statistics and neighbor search for spatial transcriptomics data, including Moran's I, spatial autocorrelation, and spatial neighbor graphs.

## Overview

Spatial analysis is essential for:
- Spatial transcriptomics data analysis
- Detecting spatial patterns and gradients
- Computing spatial autocorrelation
- Building spatial neighborhood graphs

All operations are optimized with SIMD and parallelization.

## Functions

### Weight Sum

#### `weight_sum`

Compute the sum of all weights in a spatial graph.

```cpp
template <typename T, bool GraphCSR>
void weight_sum(
    const Sparse<T, GraphCSR>& graph,
    T& out_sum
);
```

**Parameters**:
- `graph` [in]: Spatial neighbor graph (weights)
- `out_sum` [out]: Sum of all weights

**Example**:
```cpp
#include "scl/kernel/spatial.hpp"

CSR spatial_graph = build_spatial_graph(coordinates, radius);
Real total_weight = Real(0);
kernel::spatial::weight_sum(spatial_graph, total_weight);
```

### Moran's I

#### `morans_i`

Compute Moran's I statistic for spatial autocorrelation.

```cpp
template <typename T, bool GraphCSR>
Real morans_i(
    const Sparse<T, GraphCSR>& graph,
    Array<const Real> z,
    Real z_mean,
    Real z_var
);
```

**Parameters**:
- `graph` [in]: Spatial neighbor graph (weights)
- `z` [in]: Values to test (e.g., gene expression)
- `z_mean` [in]: Mean of z values
- `z_var` [in]: Variance of z values

**Returns**: Moran's I statistic (range: -1 to 1)

**Mathematical Definition**:
```
I = (n/W) * Σᵢ Σⱼ wᵢⱼ(zᵢ - z̄)(zⱼ - z̄) / Σᵢ(zᵢ - z̄)²
```

where:
- `n`: Number of cells
- `W`: Sum of all weights
- `wᵢⱼ`: Weight between cells i and j
- `z̄`: Mean of z values

**Example**:
```cpp
// Compute Moran's I for a gene
Array<Real> gene_expression = {expr_ptr, n_cells};
Real mean = compute_mean(gene_expression);
Real var = compute_variance(gene_expression, mean);

Real morans_i = kernel::spatial::morans_i(
    spatial_graph, gene_expression, mean, var
);

// Interpret: I > 0 = positive autocorrelation (clustering)
//            I < 0 = negative autocorrelation (dispersion)
//            I ≈ 0 = no spatial pattern
```

### Spatial Neighbors

#### `spatial_neighbors`

Build spatial neighbor graph from coordinates.

```cpp
CSR spatial_neighbors(
    Array<const Real> coordinates,  // x, y coordinates
    Index n_cells,
    Real radius
);
```

**Parameters**:
- `coordinates` [in]: Array of coordinates (length = n_cells * 2, [x0, y0, x1, y1, ...])
- `n_cells` [in]: Number of cells
- `radius` [in]: Neighbor search radius

**Returns**: CSR matrix of spatial neighbors (binary or distance-weighted)

**Example**:
```cpp
// Coordinates: [x0, y0, x1, y1, ...]
Array<Real> coords = {coords_ptr, n_cells * 2};
Real radius = 50.0;  // pixels or units

CSR neighbors = kernel::spatial::spatial_neighbors(
    coords, n_cells, radius
);
```

## Weighted Neighbor Sum

### `compute_weighted_neighbor_sum`

Compute weighted sum of neighbor values (used internally for Moran's I).

```cpp
template <typename T>
T compute_weighted_neighbor_sum(
    const T* weights,
    const Index* indices,
    Size len,
    const T* z
);
```

**Optimizations**:
- SIMD for long arrays (≥ 16 elements)
- Scalar fallback for short arrays
- 8-way unrolled loop with prefetching
- Multi-accumulator pattern

## Common Patterns

### Computing Moran's I for Multiple Genes

```cpp
void compute_morans_i_batch(
    const CSR& spatial_graph,
    const CSR& expression_matrix,
    Array<Real> morans_i_results
) {
    Index n_genes = expression_matrix.cols();
    
    // Precompute graph statistics
    Real total_weight = Real(0);
    kernel::spatial::weight_sum(spatial_graph, total_weight);
    
    // For each gene
    for (Index g = 0; g < n_genes; ++g) {
        // Extract gene expression
        auto gene_expr = expression_matrix.col_values(g);
        Array<Real> expr_view = {gene_expr.ptr, gene_expr.size};
        
        // Compute statistics
        Real mean = compute_mean(expr_view);
        Real var = compute_variance(expr_view, mean);
        
        // Compute Moran's I
        morans_i_results[g] = kernel::spatial::morans_i(
            spatial_graph, expr_view, mean, var
        );
    }
}
```

### Building Spatial Graph with Distance Weights

```cpp
CSR build_weighted_spatial_graph(
    Array<const Real> coordinates,
    Index n_cells,
    Real radius
) {
    // Build binary graph first
    CSR binary_graph = kernel::spatial::spatial_neighbors(
        coordinates, n_cells, radius
    );
    
    // Convert to distance-weighted
    CSR weighted_graph = CSR::create(n_cells, n_cells, binary_graph.nnz());
    
    for (Index i = 0; i < n_cells; ++i) {
        auto neighbors = binary_graph.row_indices(i);
        auto n_neighbors = binary_graph.row_length(i);
        
        Real x_i = coordinates[i * 2];
        Real y_i = coordinates[i * 2 + 1];
        
        for (Index k = 0; k < n_neighbors; ++k) {
            Index j = neighbors[k];
            Real x_j = coordinates[j * 2];
            Real y_j = coordinates[j * 2 + 1];
            
            // Compute distance
            Real dx = x_i - x_j;
            Real dy = y_i - y_j;
            Real dist = std::sqrt(dx * dx + dy * dy);
            
            // Inverse distance weight
            Real weight = Real(1) / (dist + Real(1e-10));
            weighted_graph.set(i, j, weight);
        }
    }
    
    return weighted_graph;
}
```

## Performance Considerations

### Parallelization

Spatial operations are automatically parallelized:

```cpp
// Block-wise parallel computation
constexpr Size CELL_BLOCK_SIZE = 256;
threading::parallel_for(0, n_cells, [&](size_t i) {
    // Process cell i
});
```

### SIMD Optimization

Weighted neighbor sums use SIMD for long arrays:

```cpp
// SIMD for arrays ≥ 16 elements
if (len >= SIMD_GATHER_THRESHOLD) {
    return compute_weighted_neighbor_sum_simd(...);
} else {
    return compute_weighted_neighbor_sum_scalar(...);
}
```

### Prefetching

Aggressive prefetching for indirect memory access:

```cpp
// Prefetch ahead for z[indices[k + PREFETCH_DISTANCE]]
if (k + PREFETCH_DISTANCE < len) {
    SCL_PREFETCH_READ(&z[indices[k + PREFETCH_DISTANCE]], 0);
}
```

## Configuration

```cpp
namespace scl::kernel::spatial::config {
    constexpr Size PREFETCH_DISTANCE = 8;
    constexpr Size SIMD_GATHER_THRESHOLD = 16;
    constexpr Size PARALLEL_CELL_THRESHOLD = 1024;
    constexpr Size CELL_BLOCK_SIZE = 256;
}
```

## Related Documentation

- [Neighbor Search](./neighbors.md) - General neighbor search
- [Kernels Overview](./overview.md) - General kernel usage
- [Sparse Matrices](../core/sparse.md) - Sparse matrix operations
