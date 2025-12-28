# communication.hpp

> scl/kernel/communication.hpp · Cell-cell communication analysis (CellChat/CellPhoneDB-style)

## Overview

This file provides high-performance kernels for analyzing ligand-receptor (L-R) interactions between cell types. It supports multiple scoring methods, permutation-based statistical testing, batch processing, and spatial context-aware communication analysis.

**Header**: `#include "scl/kernel/communication.hpp"`

---

## Main APIs

### lr_score_matrix

::: source_code file="scl/kernel/communication.hpp" symbol="lr_score_matrix" collapsed
:::

**Algorithm Description**

Computes ligand-receptor interaction scores between all cell type pairs:

1. Extract expression values for ligand and receptor genes across all cells
2. Group cells by cell type
3. For each sender-receiver type pair:
   - Compute mean ligand expression in sender type
   - Compute mean receptor expression in receiver type
   - Apply scoring method (MeanProduct, GeometricMean, MinMean, Product, or Natmi)
4. Store results in score matrix where `score_matrix[s * n_types + r]` contains score for sender s and receiver r

**Edge Cases**

- **Zero expression**: If a gene has zero expression in a cell type, the score contribution is zero
- **Missing cell types**: Empty cell types are handled gracefully with zero scores
- **Invalid gene indices**: Must be valid indices within expression matrix

**Data Guarantees (Preconditions)**

- `score_matrix` has capacity >= `n_types * n_types`
- All cell type labels are valid (0 <= label < n_types)
- Expression matrix is valid CSR format
- Ligand and receptor gene indices are within valid range

**Complexity Analysis**

- **Time**: O(n_cells + n_types^2) - linear scan of cells plus pairwise scoring
- **Space**: O(n_cells + n_types) auxiliary - temporary arrays for type grouping

**Example**

```cpp
#include "scl/kernel/communication.hpp"

Sparse<Real, true> expression = /* cells x genes, CSR format */;
Array<const Index> cell_type_labels = /* cell type for each cell */;
Real* score_matrix = new Real[n_types * n_types];

scl::kernel::communication::lr_score_matrix(
    expression,
    cell_type_labels,
    ligand_gene,      // Index of ligand gene
    receptor_gene,    // Index of receptor gene
    n_cells,
    n_types,
    score_matrix,
    scl::kernel::communication::ScoreMethod::MeanProduct
);

// Access score for sender type s and receiver type r:
Real score = score_matrix[s * n_types + r];
```

---

### lr_score_with_permutation

::: source_code file="scl/kernel/communication.hpp" symbol="lr_score_with_permutation" collapsed
:::

**Algorithm Description**

Computes L-R interaction score with permutation-based p-value for statistical significance:

1. Compute observed score for specified sender-receiver pair using selected scoring method
2. Perform n_permutations random shuffles of cell type labels
3. For each permutation:
   - Compute L-R score with shuffled labels
   - Track how many permuted scores >= observed score
4. P-value = (count + 1) / (n_permutations + 1) using standard permutation test formula
5. Uses thread-local random number generators for parallel execution

**Edge Cases**

- **Zero observed score**: P-value is 1.0 if all permuted scores are also zero
- **Perfect score**: P-value approaches 0 if observed score is much higher than permuted scores
- **Small n_permutations**: Minimum recommended is 1000 for reliable p-values

**Data Guarantees (Preconditions)**

- Sender and receiver type indices are valid (0 <= index < n_types)
- Expression matrix is valid CSR format
- Cell type labels array has length == n_cells
- All type labels are valid

**Complexity Analysis**

- **Time**: O(n_permutations * n_cells) - each permutation requires full cell scan
- **Space**: O(n_cells) auxiliary - temporary arrays for shuffled labels per thread

**Example**

```cpp
Real observed_score;
Real p_value;

scl::kernel::communication::lr_score_with_permutation(
    expression,
    cell_type_labels,
    ligand_gene,
    receptor_gene,
    sender_type,      // Sender cell type index
    receiver_type,    // Receiver cell type index
    n_cells,
    observed_score,   // Output: observed interaction strength
    p_value,          // Output: permutation p-value
    1000,             // Number of permutations
    scl::kernel::communication::ScoreMethod::MeanProduct,
    42                // Random seed
);

if (p_value < 0.05) {
    // Significant interaction detected
}
```

---

### batch_lr_scoring

::: source_code file="scl/kernel/communication.hpp" symbol="batch_lr_scoring" collapsed
:::

**Algorithm Description**

Efficiently computes L-R scores for multiple ligand-receptor pairs in parallel:

1. Process each L-R pair in parallel
2. For each pair:
   - Extract ligand and receptor expression vectors
   - Group cells by type
   - Compute type-pair scores using selected method
   - Store in output array at offset `p * n_types^2 + s * n_types + r`
3. Uses parallel_for with dynamic scheduling for load balancing

**Edge Cases**

- **Empty pairs list**: Returns immediately with no computation
- **Invalid gene indices**: Pairs with invalid indices produce zero scores
- **Memory limits**: Very large n_pairs may require chunked processing

**Data Guarantees (Preconditions)**

- `scores` array has capacity >= `n_pairs * n_types * n_types`
- `ligand_genes` and `receptor_genes` arrays have length == n_pairs
- All gene indices are valid (0 <= index < n_genes)
- Expression matrix is valid CSR format

**Complexity Analysis**

- **Time**: O(n_pairs * (n_cells + n_types^2)) - linear in number of pairs
- **Space**: O(n_cells * max_gene) auxiliary - per-thread workspace for expression extraction

**Example**

```cpp
const Index* ligand_genes = /* array of ligand gene indices [n_pairs] */;
const Index* receptor_genes = /* array of receptor gene indices [n_pairs] */;
Real* scores = new Real[n_pairs * n_types * n_types];

scl::kernel::communication::batch_lr_scoring(
    expression,
    cell_type_labels,
    ligand_genes,
    receptor_genes,
    n_pairs,
    n_cells,
    n_types,
    scores,
    scl::kernel::communication::ScoreMethod::MeanProduct
);

// Access score for pair p, sender s, receiver r:
Real score = scores[p * n_types * n_types + s * n_types + r];
```

---

### batch_lr_permutation_test

::: source_code file="scl/kernel/communication.hpp" symbol="batch_lr_permutation_test" collapsed
:::

**Algorithm Description**

Computes permutation p-values for multiple L-R pairs simultaneously:

1. Process each L-R pair in parallel
2. For each pair:
   - Compute observed score for all type combinations
   - Perform n_permutations random shuffles
   - Compute p-value for each type combination
3. Results stored in flat arrays: `scores[p * n_types^2 + s * n_types + r]` and `p_values[p * n_types^2 + s * n_types + r]`
4. Uses early stopping optimization when possible

**Edge Cases**

- **Zero scores**: P-values are 1.0 for zero observed scores
- **Large n_pairs**: Memory usage scales with n_pairs * n_types^2
- **Thread contention**: Parallel execution with thread-local RNG avoids contention

**Data Guarantees (Preconditions)**

- `scores` and `p_values` arrays have capacity >= `n_pairs * n_types^2`
- All input arrays have matching lengths
- Expression matrix is valid CSR format

**Complexity Analysis**

- **Time**: O(n_pairs * n_permutations * n_cells) - quadratic in pairs and permutations
- **Space**: O(n_cells) auxiliary per thread - shuffled label arrays

**Example**

```cpp
Real* scores = new Real[n_pairs * n_types * n_types];
Real* p_values = new Real[n_pairs * n_types * n_types];

scl::kernel::communication::batch_lr_permutation_test(
    expression,
    cell_type_labels,
    ligand_genes,
    receptor_genes,
    n_pairs,
    n_cells,
    n_types,
    scores,          // Output: observed scores
    p_values,       // Output: p-values
    1000,           // Number of permutations
    scl::kernel::communication::ScoreMethod::MeanProduct,
    42              // Random seed
);

// Filter significant interactions
for (Index p = 0; p < n_pairs; ++p) {
    for (Index s = 0; s < n_types; ++s) {
        for (Index r = 0; r < n_types; ++r) {
            Index idx = p * n_types * n_types + s * n_types + r;
            if (p_values[idx] < 0.05) {
                // Significant interaction
            }
        }
    }
}
```

---

### spatial_communication_score

::: source_code file="scl/kernel/communication.hpp" symbol="spatial_communication_score" collapsed
:::

**Algorithm Description**

Computes spatial context-aware communication scores using spatial neighbor graph:

1. For each cell in parallel:
   - Extract ligand expression for current cell
   - Iterate over spatial neighbors (from spatial_graph)
   - For each neighbor, extract receptor expression
   - Accumulate weighted L-R interaction: ligand_i * receptor_j * weight_ij
2. Result is per-cell communication score reflecting local spatial context
3. Uses sparse matrix iteration for efficient neighbor access

**Edge Cases**

- **Isolated cells**: Cells with no neighbors get zero score
- **Self-loops**: If spatial graph includes self-loops, they contribute to score
- **Disconnected graph**: Each connected component computed independently

**Data Guarantees (Preconditions)**

- `cell_scores` has capacity >= n_cells
- Spatial graph is valid CSR format with n_cells x n_cells dimensions
- Expression matrix rows == n_cells
- Spatial graph represents valid neighborhood structure

**Complexity Analysis**

- **Time**: O(n_cells * avg_neighbors) - linear in cells and average degree
- **Space**: O(n_cells) auxiliary - output array only

**Example**

```cpp
Sparse<Index, true> spatial_graph = /* spatial neighbor graph, CSR */;
Real* cell_scores = new Real[n_cells];

scl::kernel::communication::spatial_communication_score(
    expression,
    spatial_graph,
    ligand_gene,
    receptor_gene,
    n_cells,
    cell_scores
);

// cell_scores[i] contains spatial communication score for cell i
```

---

### expression_specificity

::: source_code file="scl/kernel/communication.hpp" symbol="expression_specificity" collapsed
:::

**Algorithm Description**

Computes expression specificity of a gene across cell types:

1. Extract expression values for specified gene across all cells
2. Group cells by type and compute mean expression per type
3. Compute specificity score for each type using formula:
   - specificity[t] = mean_t / (sum of means across all types + epsilon)
4. Normalize to ensure sum of specificities is meaningful
5. Higher specificity indicates gene is more specific to that cell type

**Edge Cases**

- **Zero expression**: Types with zero expression get zero specificity
- **Uniform expression**: If gene expressed equally in all types, specificities are uniform
- **Single type expression**: Gene expressed in only one type gets specificity 1.0 for that type

**Data Guarantees (Preconditions)**

- `specificity` array has capacity >= n_types
- Gene index is valid (0 <= gene < n_genes)
- Expression matrix is valid CSR format
- Cell type labels are valid

**Complexity Analysis**

- **Time**: O(n_cells) - single pass through cells
- **Space**: O(n_cells + n_types) auxiliary - expression extraction and type grouping

**Example**

```cpp
Real* specificity = new Real[n_types];

scl::kernel::communication::expression_specificity(
    expression,
    cell_type_labels,
    gene,            // Gene index
    n_cells,
    n_types,
    specificity
);

// Find most specific type
Index max_type = 0;
for (Index t = 1; t < n_types; ++t) {
    if (specificity[t] > specificity[max_type]) {
        max_type = t;
    }
}
```

---

## Utility Functions

### filter_significant_interactions

Filters significant L-R interactions by p-value threshold.

::: source_code file="scl/kernel/communication.hpp" symbol="filter_significant_interactions" collapsed
:::

**Complexity**

- Time: O(n_pairs * n_types^2)
- Space: O(1) auxiliary

---

### aggregate_to_network

Aggregates L-R scores into cell type communication network.

::: source_code file="scl/kernel/communication.hpp" symbol="aggregate_to_network" collapsed
:::

**Complexity**

- Time: O(n_pairs * n_types^2)
- Space: O(1) auxiliary

---

## Configuration

Default parameters in `scl::kernel::communication::config`:

- `DEFAULT_N_PERM = 1000`: Default number of permutations
- `DEFAULT_PVAL_THRESHOLD = 0.05`: Default p-value threshold
- `EPSILON = 1e-15`: Numerical stability constant
- `MIN_EXPRESSION = 0.1`: Minimum expression threshold
- `MIN_PERCENT_EXPRESSED = 0.1`: Minimum percent cells expressing

---

## Scoring Methods

The `ScoreMethod` enum provides different scoring approaches:

- `MeanProduct`: mean(ligand) * mean(receptor) - standard method
- `GeometricMean`: sqrt(mean(ligand) * mean(receptor)) - balanced scoring
- `MinMean`: min(mean(ligand), mean(receptor)) - conservative scoring
- `Product`: Direct product - simple multiplication
- `Natmi`: NATMI-style scoring - compatibility with NATMI tool

---

## See Also

- [Comparison Module](./comparison) - Statistical testing utilities
- [Sparse Matrix](../core/sparse) - Sparse matrix operations
