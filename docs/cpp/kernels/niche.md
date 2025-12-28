# niche.hpp

> scl/kernel/niche.hpp · Cellular niche analysis for spatial neighborhoods

## Overview

This file provides kernels for analyzing cellular niches in spatial contexts, computing cell type composition in spatial neighborhoods defined by spatial neighbor graphs.

This file provides:
- Niche composition computation
- Spatial neighborhood analysis
- Cell type fraction calculation

**Header**: `#include "scl/kernel/niche.hpp"`

---

## Main APIs

### niche_composition

::: source_code file="scl/kernel/niche.hpp" symbol="niche_composition" collapsed
:::

**Algorithm Description**

Compute cell type composition in spatial neighborhoods for each cell:

1. **Neighbor Extraction**: For each cell i:
   - Extract neighbors from spatial graph (CSR format)
   - Get neighbor indices from graph row i

2. **Type Counting**: For each cell i:
   - Count occurrences of each cell type among neighbors
   - Include cell i itself in the count (self-inclusive neighborhood)
   - Store counts per type

3. **Normalization**: For each cell i:
   - Sum total neighbor count (including self)
   - Divide type counts by total to get fractions
   - Handle cells with no neighbors (uniform distribution: 1/n_types)

4. **Output**: Store composition matrix (cells x types):
   - `composition[i * n_types + t]` = fraction of type t in niche of cell i

**Edge Cases**

- **No neighbors**: Cells with no neighbors get uniform composition (1/n_types for each type)
- **Self-only**: Cells with only self as neighbor get composition = 1.0 for own type
- **Empty graph**: All cells get uniform composition
- **Invalid type labels**: Type labels outside [0, n_types) are ignored
- **Isolated cells**: Single cells with no connections get uniform composition

**Data Guarantees (Preconditions)**

- `composition` has capacity >= n_cells * n_types
- `cell_types` has length >= n_cells
- Cell type labels must be valid indices [0, n_types)
- Spatial graph must be valid CSR format
- Graph must be square (n_cells x n_cells)

**Complexity Analysis**

- **Time**: O(nnz + n_cells * n_types) where nnz is number of edges in spatial graph
  - Neighbor extraction: O(nnz) total
  - Type counting: O(nnz) total
  - Normalization: O(n_cells * n_types)
  - Parallelized over cells
- **Space**: O(n_types) auxiliary per cell for type counting

**Example**

```cpp
#include "scl/kernel/niche.hpp"

scl::Sparse<Real, true> spatial_graph = /* ... */;  // [n_cells x n_cells]
scl::Array<const Index> cell_types = /* ... */;     // [n_cells]
Index n_cells = spatial_graph.rows();
Index n_types = /* number of unique cell types */;

Real* composition = /* allocate n_cells * n_types */;

scl::kernel::niche::niche_composition(
    spatial_graph,
    cell_types,
    n_cells,
    n_types,
    composition
);

// Access composition: composition[i * n_types + t]
for (Index i = 0; i < n_cells; ++i) {
    for (Index t = 0; t < n_types; ++t) {
        Real fraction = composition[i * n_types + t];
        // fraction is proportion of type t in niche of cell i
    }
}
```

---

## Configuration

Default parameters in `scl::kernel::niche::config`:

- `EPSILON = 1e-10`: Small constant for numerical stability
- `DEFAULT_K = 15`: Default number of neighbors (if used in future functions)
- `PARALLEL_THRESHOLD = 500`: Minimum size for parallel processing

---

## Performance Notes

### Parallelization

- Parallelized over cells
- Each thread processes independent cells
- No synchronization needed (distinct output elements)

### Memory Efficiency

- Pre-allocated output buffer
- Minimal temporary allocations
- Efficient sparse graph access

---

## Use Cases

### Spatial Tissue Analysis

```cpp
// Compute niche composition for spatial transcriptomics
scl::kernel::niche::niche_composition(
    spatial_graph,  // Spatial neighbor graph from coordinates
    cell_types,     // Annotated cell types
    n_cells,
    n_types,
    composition
);

// Analyze tissue regions with specific cell type compositions
// e.g., identify immune niches, stromal niches, etc.
```

### Neighborhood Characterization

```cpp
// Characterize local cellular environment around each cell
// composition[i * n_types + t] tells us what types are nearby cell i

// Find cells in mixed niches (multiple types present)
for (Index i = 0; i < n_cells; ++i) {
    Index n_types_present = 0;
    for (Index t = 0; t < n_types; ++t) {
        if (composition[i * n_types + t] > 0.1) {
            n_types_present++;
        }
    }
    if (n_types_present > 2) {
        // Cell i is in a mixed niche
    }
}
```

---

## See Also

- [Spatial Analysis](../spatial)
- [Neighbors](../neighbors)
- [Sparse Matrices](../core/sparse)
