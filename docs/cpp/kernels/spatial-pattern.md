# Spatial Pattern Detection

Spatial variation analysis and gradient computation for spatial transcriptomics (SpatialDE-style).

## Overview

Spatial pattern detection kernels provide:

- **Spatially Variable Genes** - Identify genes with spatial expression patterns
- **Spatial Gradients** - Compute expression gradients in space
- **Pattern Recognition** - Detect spatial expression domains
- **SpatialDE Analysis** - Statistical spatial variation testing

## Spatially Variable Genes

### spatially_variable_genes

Identify spatially variable genes using spatial autocorrelation:

```cpp
#include "scl/kernel/spatial_pattern.hpp"

Sparse<Real, true> expression = /* ... */;     // Expression matrix [n_cells x n_genes]
const Real* coordinates = /* ... */;            // Spatial coordinates [n_cells * n_dims]
Index n_cells = expression.rows();
Index n_genes = expression.cols();
Size n_dims = 2;                                // 2D or 3D

Array<Real> sv_scores(n_genes);                // Pre-allocated output

// Standard analysis
scl::kernel::spatial_pattern::spatially_variable_genes(
    expression, coordinates,
    n_cells, n_genes, n_dims,
    sv_scores,
    bandwidth = 100.0                           // Spatial bandwidth
);

// sv_scores[g] contains spatial variation score for gene g
```

**Parameters:**
- `expression`: Expression matrix (cells × genes, CSR format)
- `coordinates`: Spatial coordinates, size = n_cells × n_dims
- `n_cells`: Number of cells
- `n_genes`: Number of genes
- `n_dims`: Number of spatial dimensions (typically 2 or 3)
- `sv_scores`: Output spatial variation scores, must be pre-allocated, size = n_genes
- `bandwidth`: Spatial bandwidth for kernel (controls neighborhood size)

**Postconditions:**
- `sv_scores[g]` contains spatial variation score for gene g
- Higher scores indicate stronger spatial patterns
- Scores can be used for ranking and filtering

**Algorithm:**
For each gene in parallel:
1. Compute spatial autocorrelation using kernel function
2. Measure deviation from random spatial distribution
3. Compute variation score (e.g., Moran's I, SpatialDE statistic)

**Complexity:**
- Time: O(n_genes * n_cells^2) - quadratic in cells (pairwise comparisons)
- Space: O(n_cells) auxiliary per gene

**Thread Safety:**
- Safe - parallelized over genes
- Each gene processed independently

**Use cases:**
- Spatial transcriptomics analysis
- Identify domain-specific genes
- SpatialDE-style analysis
- Pattern discovery in tissue

## Spatial Gradients

### spatial_gradient

Compute spatial gradient of gene expression:

```cpp
Sparse<Real, true> expression = /* ... */;
const Real* coordinates = /* ... */;
Index gene_index = 5;                           // Gene to analyze
Index n_cells = expression.rows();
Size n_dims = 2;

Array<Real> gradients(n_cells * n_dims);       // Pre-allocated output

scl::kernel::spatial_pattern::spatial_gradient(
    expression, coordinates,
    gene_index, n_cells, n_dims,
    gradients.ptr);

// gradients[i * n_dims + d] contains gradient component d for cell i
```

**Parameters:**
- `expression`: Expression matrix (cells × genes, CSR format)
- `coordinates`: Spatial coordinates, size = n_cells × n_dims
- `gene_index`: Index of gene to analyze
- `n_cells`: Number of cells
- `n_dims`: Number of spatial dimensions
- `gradients`: Output gradient vectors, must be pre-allocated, size = n_cells × n_dims

**Postconditions:**
- `gradients[i * n_dims + d]` contains gradient component d for cell i
- Gradient magnitude indicates rate of change
- Gradient direction indicates direction of increasing expression

**Algorithm:**
For each cell in parallel:
1. Identify spatial neighbors
2. Compute expression differences
3. Estimate gradient using local linear regression or finite differences

**Complexity:**
- Time: O(n_cells * n_neighbors) where n_neighbors = average neighbors per cell
- Space: O(n_cells) auxiliary for neighbor indices

**Thread Safety:**
- Safe - parallelized over cells
- Each cell processed independently

**Use cases:**
- Gradient-based pattern analysis
- Directional expression changes
- Morphogen gradient detection
- Spatial differentiation patterns

## Configuration

### Default Parameters

```cpp
namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_NEIGHBORS = 3;
    constexpr Size DEFAULT_N_NEIGHBORS = 15;
    constexpr Size MAX_ITERATIONS = 100;
    constexpr Real BANDWIDTH_SCALE = Real(0.3);
}
```

**Bandwidth:**
- Controls spatial neighborhood size
- Smaller = local patterns, larger = global patterns
- Should match spatial resolution of data

**Neighbor Count:**
- Number of neighbors for gradient computation
- Default 15 is usually sufficient
- Increase for smoother gradients

## Examples

### Identify Spatially Variable Genes

```cpp
#include "scl/kernel/spatial_pattern.hpp"

Sparse<Real, true> expression = /* ... */;
const Real* coords = /* ... */;  // [n_cells * 2] for 2D
Index n_cells = expression.rows();
Index n_genes = expression.cols();

Array<Real> sv_scores(n_genes);
Real bandwidth = 50.0;  // Adjust based on tissue scale

scl::kernel::spatial_pattern::spatially_variable_genes(
    expression, coords,
    n_cells, n_genes, 2,  // 2D coordinates
    sv_scores,
    bandwidth
);

// Rank genes by spatial variation
std::vector<std::pair<Real, Index>> ranked;
for (Index g = 0; g < n_genes; ++g) {
    ranked.push_back({sv_scores[g], g});
}
std::sort(ranked.rbegin(), ranked.rend());  // Sort descending

// Top spatially variable genes
Index top_n = 100;
for (Index i = 0; i < top_n && i < ranked.size(); ++i) {
    std::cout << "Gene " << ranked[i].second
              << " (SV score: " << ranked[i].first << ")\n";
}
```

### Compute Expression Gradients

```cpp
Index gene_index = 10;  // Gene of interest
Size n_dims = 2;

Array<Real> gradients(n_cells * n_dims);
scl::kernel::spatial_pattern::spatial_gradient(
    expression, coords,
    gene_index, n_cells, n_dims,
    gradients.ptr);

// Compute gradient magnitudes
Array<Real> magnitudes(n_cells);
for (Index i = 0; i < n_cells; ++i) {
    Real gx = gradients[i * n_dims + 0];
    Real gy = gradients[i * n_dims + 1];
    magnitudes[i] = std::sqrt(gx * gx + gy * gy);
}

// Find cells with strong gradients (expression boundaries)
Real threshold = std::percentile(magnitudes.begin(), magnitudes.end(), 90);
for (Index i = 0; i < n_cells; ++i) {
    if (magnitudes[i] > threshold) {
        // Cell i is in a gradient/boundary region
    }
}
```

### Gradient Direction Analysis

```cpp
// Analyze gradient directions to identify expression sources/sinks
Array<Real> gradients(n_cells * n_dims);
scl::kernel::spatial_pattern::spatial_gradient(
    expression, coords,
    gene_index, n_cells, n_dims,
    gradients.ptr);

// Cluster gradient directions to identify domains
// (implementation depends on your clustering library)
// Or visualize gradient vectors directly
```

---

::: tip Bandwidth Selection
Choose bandwidth based on expected pattern scale: small for fine-grained patterns, large for broad domains. Consider using multiple bandwidths for multi-scale analysis.
:::

