# clonotype.hpp

> scl/kernel/clonotype.hpp · TCR/BCR clonal diversity and expansion analysis for immune repertoire studies

## Overview

This file provides kernels for analyzing T-cell receptor (TCR) and B-cell receptor (BCR) clonal diversity and expansion. Clonotype analysis is essential for understanding immune repertoire composition, tracking clonal expansion, and linking clones to cell phenotypes.

Key features:
- Clonal diversity metrics (Shannon entropy, Simpson index, Gini coefficient)
- Clone expansion identification
- Clone-phenotype association analysis
- Memory-efficient implementations

**Header**: `#include "scl/kernel/clonotype.hpp"`

---

## Main APIs

### clonal_diversity

::: source_code file="scl/kernel/clonotype.hpp" symbol="clonal_diversity" collapsed
:::

**Algorithm Description**

Compute clonal diversity metrics for immune repertoire:

1. **Frequency counting**: Count occurrences of each clone ID
2. **Probability computation**: For each clone i, compute `p_i = count_i / n_cells`
3. **Shannon entropy**: `H = -sum(p_i * log(p_i))` - measures overall diversity, sensitive to rare clones
4. **Simpson index**: `D = 1 - sum(p_i^2)` - measures diversity with emphasis on common clones, ranges [0, 1)
5. **Gini coefficient**: Compute inequality measure using sorted clone sizes

**Edge Cases**

- **No clones** (all NO_CLONE): All metrics return 0
- **Single clone**: Shannon = 0, Simpson = 0, Gini = 0 (perfect equality)
- **All unique clones**: Maximum diversity (Shannon = log(n_clones), Simpson approaches 1)
- **Empty input**: Returns 0 for all metrics

**Data Guarantees (Preconditions)**

- `clone_ids.len >= n_cells` (or n_cells <= clone_ids.len)
- Clone IDs should be non-negative (NO_CLONE = -1 is handled)
- Valid clone IDs in range [0, max_clone_id]

**Complexity Analysis**

- **Time**: O(n_cells) - single pass through data for counting
  - Additional O(n_clones) for metric computation
- **Space**: O(n_clones) auxiliary for frequency counting

**Example**

```cpp
#include "scl/kernel/clonotype.hpp"

Array<Index> clone_ids = /* ... */;  // Clone IDs for each cell
Size n_cells = clone_ids.len;

Real shannon_entropy, simpson_index, gini_coeff;

scl::kernel::clonotype::clonal_diversity(
    clone_ids,
    n_cells,
    shannon_entropy,
    simpson_index,
    gini_coeff
);

// shannon_entropy: H = -sum(p_i * log(p_i))
// simpson_index: 1 - sum(p_i^2), ranges [0, 1)
// gini_coeff: inequality measure, ranges [0, 1]
```

---

### clone_expansion

::: source_code file="scl/kernel/clonotype.hpp" symbol="clone_expansion" collapsed
:::

**Algorithm Description**

Identify expanded clones based on size threshold:

1. **Frequency counting**: Count occurrences of each clone ID
2. **Filtering**: Select clones with size >= min_size
3. **Sorting**: Sort selected clones by size (largest first)
4. **Output**: Return top max_results clones with their sizes

**Edge Cases**

- **No expanded clones**: Returns 0, output arrays unchanged
- **More than max_results expanded clones**: Returns max_results, largest clones selected
- **All clones below threshold**: Returns 0
- **Empty input**: Returns 0

**Data Guarantees (Preconditions)**

- `expanded_clones` has capacity >= max_results
- `clone_sizes` has capacity >= max_results
- `max_results > 0`
- `min_size >= 0`

**Complexity Analysis**

- **Time**: O(n_cells + n_clones * log(n_clones))
  - O(n_cells) for counting
  - O(n_clones * log(n_clones)) for sorting expanded clones
- **Space**: O(n_clones) auxiliary for frequency counting

**Example**

```cpp
Array<Index> clone_ids = /* ... */;
Size n_cells = clone_ids.len;

Index max_results = 100;
Array<Index> expanded_clones(max_results);
Array<Size> clone_sizes(max_results);

Index n_expanded = scl::kernel::clonotype::clone_expansion(
    clone_ids,
    n_cells,
    expanded_clones.ptr,
    clone_sizes.ptr,
    min_size = 5,        // Minimum clone size
    max_results
);

// expanded_clones[0..n_expanded-1] contains clone IDs
// clone_sizes[0..n_expanded-1] contains corresponding sizes
// Results sorted by size (largest first)
```

---

### clone_phenotype_association

::: source_code file="scl/kernel/clonotype.hpp" symbol="clone_phenotype_association" collapsed
:::

**Algorithm Description**

Test association between clones and phenotypes:

1. **Co-occurrence counting**: Count cells for each (clone, phenotype) pair
2. **Association computation**: For each clone c and phenotype p:
   - Compute association strength (e.g., conditional probability or enrichment)
   - `association[c * n_phenotypes + p] = P(phenotype=p | clone=c)`
3. **Normalization**: Normalize across phenotypes per clone (optional, implementation-dependent)

**Edge Cases**

- **Clones with no cells**: Association values are 0
- **Phenotypes with no cells**: Association values are 0
- **Empty input**: All associations are 0
- **Invalid clone/phenotype IDs**: Handled gracefully (ignored or set to 0)

**Data Guarantees (Preconditions)**

- `clone_ids.len == phenotypes.len == n_cells`
- `association_matrix` has capacity >= n_clones * n_phenotypes
- Clone IDs in range [0, n_clones) or NO_CLONE
- Phenotype IDs in range [0, n_phenotypes)

**Complexity Analysis**

- **Time**: O(n_cells + n_clones * n_phenotypes)
  - O(n_cells) for counting co-occurrences
  - O(n_clones * n_phenotypes) for association computation
- **Space**: O(n_clones * n_phenotypes) auxiliary for counting matrix

**Example**

```cpp
Array<Index> clone_ids = /* ... */;      // Clone IDs [n_cells]
Array<Index> phenotypes = /* ... */;      // Phenotype labels [n_cells]
Size n_clones = /* ... */;
Size n_phenotypes = /* ... */;

Array<Real> association(n_clones * n_phenotypes);

scl::kernel::clonotype::clone_phenotype_association(
    clone_ids,
    phenotypes,
    n_clones,
    n_phenotypes,
    association.ptr
);

// association[c * n_phenotypes + p] contains association strength
// Higher values indicate stronger clone-phenotype association
```

---

## Notes

**Configuration Constants**

Default parameters in `scl::kernel::clonotype::config`:
- `EPSILON = 1e-10`
- `NO_CLONE = -1` - Special value indicating no clone assignment
- `MIN_CLONE_SIZE = 2` - Default minimum size for expansion

**Diversity Metrics Interpretation**

- **Shannon Entropy**: Higher = more diverse, sensitive to rare clones
- **Simpson Index**: Higher = more diverse, weights common clones more
- **Gini Coefficient**: Higher = more unequal (few large clones), lower = more even distribution

**Thread Safety**

All functions are thread-safe and parallelized where applicable.

## See Also

- [Statistics](/cpp/kernels/statistics) - Statistical analysis
- [Association](/cpp/kernels/association) - General association analysis
