# lineage.hpp

> scl/kernel/lineage.hpp · Lineage tracing and fate mapping

## Overview

This file provides functions for lineage tracing and fate mapping analysis. It computes coupling matrices between clones and cell types, and quantifies fate bias of clones toward different cell types.

**Header**: `#include "scl/kernel/lineage.hpp"`

---

## Main APIs

### lineage_coupling

::: source_code file="scl/kernel/lineage.hpp" symbol="lineage_coupling" collapsed
:::

**Algorithm Description**

Compute coupling matrix between clones and cell types, representing the fraction of each clone that belongs to each cell type:

1. Initialize coupling matrix to zeros
2. For each cell:
   - Get clone ID and cell type
   - Increment coupling[clone_id * n_types + cell_type] atomically
3. Normalize each row (clone) to sum to 1.0
4. Result: coupling[c * n_types + t] = P(type t | clone c)

The coupling matrix represents conditional probability distribution of cell types given a clone.

**Edge Cases**

- **Empty input**: Returns zero coupling matrix
- **Single clone**: All cells in one clone, coupling sums to 1.0 for that clone
- **Single cell type**: All cells of same type, coupling is 1.0 for all clones
- **Clone with no cells**: Row sums to 0.0 (should not occur in practice)
- **Invalid clone IDs**: Negative or out-of-range IDs cause undefined behavior

**Data Guarantees (Preconditions)**

- `clone_ids` must have length == n_cells (implicit from array size)
- `cell_types` must have length == n_cells
- `coupling_matrix` must have capacity >= n_clones * n_types
- Clone IDs should be in range [0, n_clones)
- Cell type IDs should be in range [0, n_types)
- Arrays must be valid (non-null pointers)

**Complexity Analysis**

- **Time**: O(n_cells) - single pass through all cells
- **Space**: O(n_clones * n_types) auxiliary - coupling matrix storage

**Example**

```cpp
#include "scl/kernel/lineage.hpp"

// Prepare data
Array<Index> clone_ids = /* clone IDs [n_cells] */;
Array<Index> cell_types = /* cell type labels [n_cells] */;
Size n_clones = 10;  // Number of unique clones
Size n_types = 5;    // Number of unique cell types

// Pre-allocate coupling matrix
Array<Real> coupling(n_clones * n_types);

// Compute coupling
scl::kernel::lineage::lineage_coupling(
    clone_ids, cell_types,
    coupling.ptr, n_clones, n_types
);

// Analyze clone-to-type distribution
for (Index c = 0; c < n_clones; ++c) {
    std::cout << "Clone " << c << ":\n";
    for (Index t = 0; t < n_types; ++t) {
        Real frac = coupling[c * n_types + t];
        if (frac > 0.1) {  // At least 10%
            std::cout << "  Type " << t << ": " << frac * 100 << "%\n";
        }
    }
}
```

---

### fate_bias

::: source_code file="scl/kernel/lineage.hpp" symbol="fate_bias" collapsed
:::

**Algorithm Description**

Compute fate bias of clones toward different cell types, measuring enrichment relative to background:

1. Compute coupling matrix (as in lineage_coupling)
2. Compute background type distribution: P(type) = count(type) / n_cells
3. For each clone c and type t:
   - Compute bias = coupling[c, t] / P(type t)
   - Higher values indicate stronger preference
4. Result: fate_bias[c * n_types + t] = enrichment of clone c toward type t

Fate bias quantifies how much a clone prefers a cell type compared to the overall population.

**Edge Cases**

- **Empty input**: Returns zero bias matrix
- **Uniform distribution**: All clones have bias = 1.0 (no preference)
- **Clone absent from type**: Bias = 0.0
- **Rare type in background**: Can produce very high bias values
- **Zero background probability**: Returns 0.0 or NaN depending on implementation

**Data Guarantees (Preconditions)**

- `clone_ids` must have length == n_cells
- `cell_types` must have length == n_cells
- `fate_bias` must have capacity >= n_clones * n_types
- Clone IDs should be in range [0, n_clones)
- Cell type IDs should be in range [0, n_types)
- Arrays must be valid (non-null pointers)

**Complexity Analysis**

- **Time**: O(n_cells + n_clones * n_types) - pass through cells, then compute bias for all clone-type pairs
- **Space**: O(n_clones * n_types) auxiliary - coupling matrix and bias matrix

**Example**

```cpp
#include "scl/kernel/lineage.hpp"

// Prepare data
Array<Index> clone_ids = /* ... */;
Array<Index> cell_types = /* ... */;
Size n_clones = 10;
Size n_types = 5;

// Pre-allocate bias matrix
Array<Real> bias(n_clones * n_types);

// Compute fate bias
scl::kernel::lineage::fate_bias(
    clone_ids, cell_types,
    n_clones, n_types,
    bias.ptr
);

// Find clones with strong bias toward specific type
Index target_type = 3;  // e.g., neuron
Real bias_threshold = 2.0;  // 2-fold enrichment

std::vector<Index> biased_clones;
for (Index c = 0; c < n_clones; ++c) {
    Real b = bias[c * n_types + target_type];
    if (b > bias_threshold) {
        biased_clones.push_back(c);
    }
}

std::cout << "Found " << biased_clones.size()
          << " clones biased toward type " << target_type << "\n";

// Identify multipotent clones (present in multiple types)
Real multipotency_threshold = 0.1;
std::vector<Index> multipotent_clones;
for (Index c = 0; c < n_clones; ++c) {
    Size n_types_present = 0;
    for (Index t = 0; t < n_types; ++t) {
        Real coupling_val = /* get from coupling matrix */;
        if (coupling_val > multipotency_threshold) {
            n_types_present++;
        }
    }
    if (n_types_present >= 3) {
        multipotent_clones.push_back(c);
    }
}
```

---

## Configuration

### Default Parameters

```cpp
namespace scl::kernel::lineage::config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Index NO_PARENT = -1;
    constexpr Size MIN_CLONE_SIZE = 2;
}
```

**NO_PARENT Constant**: Special value for cells without parent (root of lineage tree), used in lineage tree construction.

**Minimum Clone Size**: Used for filtering small clones to ensure statistical reliability.

---

## Notes

**Coupling vs. Bias**: Use coupling for raw clone-to-type distributions. Use fate bias for enrichment analysis relative to background. Bias is better for identifying fate preferences and comparing clones.

**Multipotency**: A clone is multipotent if it gives rise to multiple cell types. Use coupling matrix to identify clones present in >= 3 types with significant fractions (>10%).

**Statistical Considerations**: Small clones may have unreliable coupling estimates. Consider filtering clones with size < MIN_CLONE_SIZE before analysis.

**Thread Safety**: Uses atomic operations for parallel accumulation, safe for concurrent execution.

---

## See Also

- [Subpopulation](/cpp/kernels/subpopulation) - Subpopulation analysis
- [Clustering](/cpp/kernels/clustering) - Cell type clustering
