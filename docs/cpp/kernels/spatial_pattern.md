# spatial_pattern.hpp

> scl/kernel/spatial_pattern.hpp · Spatial pattern detection kernels (SpatialDE-style)

## Overview

This file provides spatial pattern detection methods for identifying spatially variable genes and computing spatial gradients in spatial transcriptomics data. These methods are inspired by SpatialDE and related approaches for analyzing spatial gene expression patterns.

This file provides:
- Spatially variable gene identification
- Spatial gradient computation
- Spatial pattern analysis
- Distance-based spatial statistics

**Header**: `#include "scl/kernel/spatial_pattern.hpp"`

---

## Main APIs

### spatially_variable_genes

::: source_code file="scl/kernel/spatial_pattern.hpp" symbol="spatially_variable_genes" collapsed
:::

**Algorithm Description**

Identify spatially variable genes using spatial autocorrelation:

1. **Spatial kernel computation**: For each gene g:
   - Compute spatial kernel weights based on coordinates and bandwidth
   - Uses distance-based kernel (e.g., Gaussian) to weight spatial neighbors

2. **Spatial variation score**: For each gene g:
   - Compute weighted spatial autocorrelation or variance
   - Score reflects how much gene expression varies across spatial locations
   - Higher scores indicate stronger spatial patterns

3. **Parallel processing**: Process genes in parallel for efficiency

**Edge Cases**

- **Zero bandwidth**: If bandwidth = 0, returns zero scores (no spatial smoothing)
- **Identical coordinates**: If all cells have same coordinates, returns zero scores
- **Constant expression**: Genes with constant expression return zero scores
- **Empty matrix**: Returns zero scores if expression matrix is empty

**Data Guarantees (Preconditions)**

- `sv_scores` must have capacity >= n_genes (pre-allocated)
- `coordinates` must have length >= n_cells * n_dims
- Expression matrix must be valid CSR format
- `bandwidth > 0` for meaningful results
- Coordinates should be normalized for consistent bandwidth interpretation

**Complexity Analysis**

- **Time**: O(n_genes * n_cells^2)
  - O(n_cells^2) per gene for pairwise distance computation
  - n_genes genes processed in parallel
- **Space**: O(n_cells) auxiliary space per gene
  - Kernel weights and intermediate computations

**Example**

```cpp
#include "scl/kernel/spatial_pattern.hpp"

// Expression matrix: cells x genes
Sparse<Real, true> expression = /* ... */;
Index n_cells = expression.rows();
Index n_genes = expression.cols();
Size n_dims = 2;  // 2D spatial coordinates

// Spatial coordinates (x, y) for each cell
Array<Real> coordinates(n_cells * n_dims);
// ... fill coordinates ...

// Pre-allocate output
Array<Real> sv_scores(n_genes);

// Compute spatial variation scores
Real bandwidth = 0.3;  // Spatial bandwidth parameter
scl::kernel::spatial_pattern::spatially_variable_genes(
    expression,
    coordinates.data(),
    n_cells,
    n_genes,
    n_dims,
    sv_scores,
    bandwidth
);

// Identify top spatially variable genes
// Sort by sv_scores and select top genes
```

---

### spatial_gradient

::: source_code file="scl/kernel/spatial_pattern.hpp" symbol="spatial_gradient" collapsed
:::

**Algorithm Description**

Compute spatial gradient of gene expression:

1. **Neighbor identification**: For each cell i:
   - Find spatial neighbors within bandwidth
   - Compute distances to neighbors

2. **Gradient computation**: For each cell i and dimension d:
   - Compute weighted gradient: grad[i, d] = sum_j(w_ij * (expr[j] - expr[i]) * (coord[j, d] - coord[i, d]) / dist_ij^2)
   - Weights w_ij based on distance (closer neighbors weighted more)
   - Gradient indicates direction and magnitude of expression change

3. **Parallel processing**: Process cells in parallel

**Edge Cases**

- **No neighbors**: Cells with no neighbors within bandwidth have zero gradient
- **Zero bandwidth**: If bandwidth = 0, returns zero gradients
- **Constant expression**: Genes with constant expression return zero gradients
- **Isolated cells**: Cells far from others have undefined gradients (set to zero)

**Data Guarantees (Preconditions)**

- `gradients` must have capacity >= n_cells * n_dims (pre-allocated)
- `coordinates` must have length >= n_cells * n_dims
- `gene_index` must be in [0, n_genes)
- Expression matrix must be valid CSR format

**Complexity Analysis**

- **Time**: O(n_cells * n_neighbors)
  - O(n_neighbors) per cell for neighbor search and gradient computation
  - n_cells cells processed in parallel
- **Space**: O(n_cells) auxiliary space
  - Neighbor lists and intermediate computations

**Example**

```cpp
// Compute spatial gradient for a specific gene
Index gene_index = 42;
Array<Real> gradients(n_cells * n_dims);

scl::kernel::spatial_pattern::spatial_gradient(
    expression,
    coordinates.data(),
    gene_index,
    n_cells,
    n_dims,
    gradients.data()
);

// Access gradient for cell i in dimension d
// Real grad_x = gradients[i * n_dims + 0];
// Real grad_y = gradients[i * n_dims + 1];
```

---

## Notes

**Spatial Bandwidth**:
- Controls spatial scale of analysis
- Smaller bandwidth: local patterns
- Larger bandwidth: global patterns
- Typical values: 0.1 - 0.5 (normalized coordinates)

**Performance**:
- Parallelized over genes (for spatially_variable_genes) and cells (for spatial_gradient)
- Distance computation is O(n_cells^2) per gene, can be expensive for large datasets
- Consider spatial indexing (e.g., KD-tree) for large-scale analysis

**Typical Usage**:
- Identify genes with spatial expression patterns
- Analyze spatial gene expression gradients
- Detect spatial domains or boundaries
- Feature selection for spatial analysis

## See Also

- [Spatial Statistics](/cpp/kernels/spatial) - Moran's I and Geary's C statistics
- [Hotspot Detection](/cpp/kernels/hotspot) - Local spatial statistics

