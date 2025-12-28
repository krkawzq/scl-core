# niche.hpp

> scl/kernel/niche.hpp · Cellular neighborhood and microenvironment analysis

## Overview

This file provides high-performance kernels for analyzing cellular niches in spatial contexts. It computes cell type composition in spatial neighborhoods, co-localization scores, contact matrices, and niche diversity metrics using SIMD optimization and parallelization.

This file provides:
- Neighborhood cell type composition computation
- Neighborhood enrichment analysis with permutation testing
- Cell-cell contact frequency matrices
- Co-localization scoring for cell type pairs
- Niche similarity computation using Jensen-Shannon divergence
- Niche diversity (Shannon entropy) calculation
- Niche boundary detection

**Header**: `#include "scl/kernel/niche.hpp"`

---

## Main APIs

### neighborhood_composition

**SUMMARY:**
Compute cell type composition fractions in spatial neighborhoods for each cell.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void neighborhood_composition(
    const Sparse<T, IsCSR>& spatial_neighbors,  // Spatial neighbor graph [n_cells x n_cells]
    Array<const Index> cell_type_labels,         // Cell type labels [n_cells]
    Index n_cell_types,                          // Number of distinct cell types
    Array<Real> composition_output               // Output [n_cells x n_cell_types]
);
```

**PARAMETERS:**
- spatial_neighbors [in]     Sparse neighbor graph, shape (n_cells, n_cells)
- cell_type_labels  [in]     Cell type label for each cell, range [0, n_cell_types)
- n_cell_types      [in]     Total number of distinct cell types
- composition_output [out]   Pre-allocated buffer, size = n_cells * n_cell_types

**PRECONDITIONS:**
- spatial_neighbors must be valid sparse matrix format
- cell_type_labels.len >= n_cells
- composition_output.len >= n_cells * n_cell_types
- n_cell_types > 0
- Cell type labels in range [0, n_cell_types)

**POSTCONDITIONS:**
- composition_output[i * n_cell_types + t] contains fraction of type t in neighborhood of cell i
- All fractions sum to 1.0 for each cell (if neighbors exist)
- Cells with no neighbors get zero composition vector

**MUTABILITY:**
Writes to composition_output, does not modify inputs

**ALGORITHM:**
For each cell i in parallel:
1. Extract neighbor indices from spatial_neighbors row i
2. Count occurrences of each cell type among neighbors (8-way unrolled with prefetch)
3. Normalize counts to fractions by dividing by total neighbor count

**COMPLEXITY:**
- Time:  O(nnz / n_threads) where nnz is total edges in spatial_neighbors
- Space: O(n_cell_types * n_threads) for thread-local count buffers

**THREAD SAFETY:**
Safe - parallelized over cells with per-thread workspace

---

### neighborhood_enrichment

**SUMMARY:**
Compute enrichment/depletion z-scores for cell type pairs in neighborhoods.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void neighborhood_enrichment(
    const Sparse<T, IsCSR>& spatial_neighbors,  // Spatial neighbor graph
    Array<const Index> cell_type_labels,         // Cell type labels
    Index n_cell_types,                          // Number of cell types
    Array<Real> enrichment_scores,               // Output z-scores [n_types x n_types]
    Array<Real> p_values,                        // Output p-values [n_types x n_types]
    Index n_permutations = 1000                  // Number of permutations for p-value
);
```

**PARAMETERS:**
- spatial_neighbors   [in]  Sparse neighbor graph, shape (n_cells, n_cells)
- cell_type_labels    [in]  Cell type labels, size n_cells
- n_cell_types        [in]  Number of distinct cell types
- enrichment_scores   [out] Z-scores, size n_types * n_types
- p_values            [out] P-values, size n_types * n_types
- n_permutations      [in]  Number of permutations for significance testing

**PRECONDITIONS:**
- enrichment_scores.len >= n_cell_types * n_cell_types
- p_values.len >= n_cell_types * n_cell_types
- n_permutations > 0

**POSTCONDITIONS:**
- enrichment_scores[a * n_types + b] = z-score for type_a having type_b neighbors
- p_values[a * n_types + b] = two-tailed p-value
- Positive z-score indicates enrichment, negative indicates depletion

**ALGORITHM:**
1. Compute observed contact frequencies between all type pairs
2. Compute expected frequencies from global type distributions
3. Compute z-scores as (observed - expected) / std_dev
4. Estimate p-values from standard normal distribution

**COMPLEXITY:**
- Time:  O(nnz + n_permutations * nnz / n_threads)
- Space: O(n_types^2 * n_threads) for accumulators

**THREAD SAFETY:**
Safe - uses parallel reduction with thread-local accumulators

---

### cell_cell_contact

**SUMMARY:**
Compute normalized cell-cell contact frequency matrix between cell types.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void cell_cell_contact(
    const Sparse<T, IsCSR>& spatial_neighbors,  // Spatial neighbor graph
    Array<const Index> cell_type_labels,         // Cell type labels
    Index n_cell_types,                          // Number of cell types
    Array<Real> contact_matrix                   // Output [n_types x n_types]
);
```

**PARAMETERS:**
- spatial_neighbors [in]  Sparse neighbor graph
- cell_type_labels  [in]  Cell type labels
- n_cell_types      [in]  Number of cell types
- contact_matrix    [out] Normalized contact frequencies, size n_types * n_types

**PRECONDITIONS:**
- contact_matrix.len >= n_cell_types * n_cell_types

**POSTCONDITIONS:**
- contact_matrix[a * n_types + b] = fraction of all contacts that are (type_a, type_b)
- Sum of all entries equals 1.0

**ALGORITHM:**
1. Count contacts between each type pair using 4-way unrolled parallel loop
2. Reduce thread-local counts
3. Normalize by total contact count

**COMPLEXITY:**
- Time:  O(nnz / n_threads)
- Space: O(n_types^2 * n_threads) for thread-local accumulators

**THREAD SAFETY:**
Safe - parallel reduction pattern

---

### colocalization_score

**SUMMARY:**
Compute co-localization score and p-value for a specific cell type pair.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void colocalization_score(
    const Sparse<T, IsCSR>& spatial_neighbors,  // Spatial neighbor graph
    Array<const Index> cell_type_labels,         // Cell type labels
    Index n_cell_types,                          // Number of cell types
    Index type_a,                                 // First cell type
    Index type_b,                                 // Second cell type
    Real& colocalization,                         // Output: log2 fold enrichment
    Real& p_value,                                // Output: permutation p-value
    Index n_permutations = 1000                   // Number of permutations
);
```

**PARAMETERS:**
- spatial_neighbors [in]  Sparse neighbor graph
- cell_type_labels  [in]  Cell type labels
- n_cell_types      [in]  Number of cell types
- type_a            [in]  Index of first cell type
- type_b            [in]  Index of second cell type
- colocalization    [out] Log2 fold enrichment score
- p_value           [out] Two-tailed permutation p-value
- n_permutations    [in]  Number of label permutations

**PRECONDITIONS:**
- type_a in range [0, n_cell_types)
- type_b in range [0, n_cell_types)
- n_permutations > 0

**POSTCONDITIONS:**
- colocalization = log2(observed / expected)
- Positive value indicates enrichment, negative indicates depletion
- p_value in range (0, 1]

**ALGORITHM:**
1. Count observed contacts where type_a neighbors type_b
2. Compute expected count from global type_b frequency
3. Compute log2 fold enrichment
4. Permute labels n_permutations times to estimate p-value

**COMPLEXITY:**
- Time:  O(n_permutations * nnz)
- Space: O(n_cells) for permuted labels

**THREAD SAFETY:**
Safe - sequential but can be called in parallel for different type pairs

---

### colocalization_matrix

**SUMMARY:**
Compute co-localization matrix (log2 fold enrichment) for all cell type pairs.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void colocalization_matrix(
    const Sparse<T, IsCSR>& spatial_neighbors,  // Spatial neighbor graph
    Array<const Index> cell_type_labels,         // Cell type labels
    Index n_cell_types,                          // Number of cell types
    Array<Real> coloc_matrix                     // Output [n_types x n_types]
);
```

**PARAMETERS:**
- spatial_neighbors [in]  Sparse neighbor graph
- cell_type_labels  [in]  Cell type labels
- n_cell_types      [in]  Number of cell types
- coloc_matrix      [out] Log2 fold enrichment matrix, size n_types * n_types

**PRECONDITIONS:**
- coloc_matrix.len >= n_cell_types * n_cell_types

**POSTCONDITIONS:**
- coloc_matrix[a * n_types + b] = log2(observed_ab / expected_ab)
- Values clamped to [-10, 10] for extreme cases

**COMPLEXITY:**
- Time:  O(nnz / n_threads)
- Space: O(n_types^2 * n_threads + n_types) for accumulators

**THREAD SAFETY:**
Safe - parallel reduction

---

### niche_similarity

**SUMMARY:**
Compute pairwise similarity between cells based on neighborhood composition.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void niche_similarity(
    const Sparse<T, IsCSR>& spatial_neighbors,  // Spatial neighbor graph
    Array<const Index> cell_type_labels,         // Cell type labels
    Index n_cell_types,                          // Number of cell types
    const Index* query_cells,                    // Cell indices to compare
    Size n_query,                                // Number of query cells
    Array<Real> similarity_output                // Output [n_query x n_query]
);
```

**PARAMETERS:**
- spatial_neighbors  [in]  Sparse neighbor graph
- cell_type_labels   [in]  Cell type labels
- n_cell_types       [in]  Number of cell types
- query_cells        [in]  Array of cell indices to compare
- n_query            [in]  Number of query cells
- similarity_output  [out] Pairwise similarity matrix, size n_query * n_query

**PRECONDITIONS:**
- query_cells contains valid cell indices
- similarity_output.len >= n_query * n_query

**POSTCONDITIONS:**
- similarity_output[i * n_query + j] = 1 - sqrt(JSD(comp_i, comp_j))
- Values in range [0, 1], where 1 = identical composition

**ALGORITHM:**
1. Compute composition vector for each query cell
2. Compute pairwise Jensen-Shannon divergence
3. Convert to similarity: sim = 1 - sqrt(JSD)

**COMPLEXITY:**
- Time:  O(n_query * (avg_neighbors + n_query * n_types))
- Space: O(n_query * n_types + n_types * n_threads)

**THREAD SAFETY:**
Safe - parallel over query cells

---

### niche_diversity

**SUMMARY:**
Compute Shannon diversity (entropy) of neighborhood composition for each cell.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void niche_diversity(
    const Sparse<T, IsCSR>& spatial_neighbors,  // Spatial neighbor graph
    Array<const Index> cell_type_labels,         // Cell type labels
    Index n_cell_types,                          // Number of cell types
    Array<Real> diversity_output                 // Output [n_cells]
);
```

**PARAMETERS:**
- spatial_neighbors [in]  Sparse neighbor graph
- cell_type_labels  [in]  Cell type labels
- n_cell_types      [in]  Number of cell types
- diversity_output  [out] Shannon entropy for each cell, size n_cells

**PRECONDITIONS:**
- diversity_output.len >= n_cells

**POSTCONDITIONS:**
- diversity_output[i] = -sum(p * log2(p)) for composition of cell i
- Range [0, log2(n_cell_types)]
- Higher values indicate more diverse neighborhoods

**ALGORITHM:**
For each cell in parallel:
1. Compute neighborhood composition
2. Compute Shannon entropy: H = -sum(p * log2(p))

**COMPLEXITY:**
- Time:  O(nnz / n_threads)
- Space: O(n_types * n_threads) for workspace

**THREAD SAFETY:**
Safe - parallelized over cells

---

### niche_boundary_score

**SUMMARY:**
Identify cells at niche boundaries based on neighbor type heterogeneity.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void niche_boundary_score(
    const Sparse<T, IsCSR>& spatial_neighbors,  // Spatial neighbor graph
    Array<const Index> cell_type_labels,         // Cell type labels
    Index n_cell_types,                          // Number of cell types
    Array<Real> boundary_scores                  // Output [n_cells]
);
```

**PARAMETERS:**
- spatial_neighbors [in]  Sparse neighbor graph
- cell_type_labels  [in]  Cell type labels
- n_cell_types      [in]  Number of cell types
- boundary_scores   [out] Boundary score for each cell, size n_cells

**PRECONDITIONS:**
- boundary_scores.len >= n_cells

**POSTCONDITIONS:**
- boundary_scores[i] = 1 - (fraction of same-type neighbors)
- Range [0, 1]
- Higher values indicate cells at tissue/niche boundaries

**ALGORITHM:**
For each cell in parallel:
1. Count neighbor types using 8-way unrolled loop
2. Compute fraction of neighbors with same type as cell
3. Return 1 - same_type_fraction

**COMPLEXITY:**
- Time:  O(nnz / n_threads)
- Space: O(n_types * n_threads) for count buffers

**THREAD SAFETY:**
Safe - parallelized over cells

---

## Configuration

Default parameters in `scl::kernel::niche::config`:

- `DEFAULT_N_NEIGHBORS = 10`: Default number of neighbors
- `DEFAULT_RADIUS = 50.0`: Default spatial radius
- `PREFETCH_DISTANCE = 8`: Prefetch ahead distance for cache optimization
- `SIMD_THRESHOLD = 16`: Minimum size for SIMD optimization path
- `PARALLEL_THRESHOLD = 256`: Minimum size for parallel processing
- `BLOCK_SIZE = 64`: Block size for blocked algorithms
- `UNROLL_FACTOR = 8`: Loop unrolling factor
- `EPS = 1e-12`: Small constant for numerical stability

---

## Performance Notes

### SIMD Optimization

- 8-way unrolled loops for neighbor type counting
- Multi-accumulator pattern to hide memory latency
- Aggressive prefetching for indirect memory access

### Parallelization

- Parallelized over cells using parallel_for
- Thread-local workspace pools to avoid allocation overhead
- Parallel reduction for global statistics

### Memory Efficiency

- WorkspacePool for per-thread buffers
- Pre-allocated output buffers
- Efficient sparse graph access

---

## Use Cases

### Spatial Tissue Analysis

```cpp
// Compute niche composition for spatial transcriptomics
scl::kernel::niche::neighborhood_composition(
    spatial_graph,
    cell_types,
    n_types,
    composition
);

// Identify cells at tissue boundaries
scl::kernel::niche::niche_boundary_score(
    spatial_graph,
    cell_types,
    n_types,
    boundary_scores
);
```

### Tumor Microenvironment Analysis

```cpp
// Compute co-localization between tumor and immune cells
Real coloc_score, p_val;
scl::kernel::niche::colocalization_score(
    spatial_graph,
    cell_types,
    n_types,
    TUMOR_TYPE_IDX,
    IMMUNE_TYPE_IDX,
    coloc_score,
    p_val,
    1000  // permutations
);

if (coloc_score > 0 && p_val < 0.05) {
    // Significant co-localization of tumor and immune cells
}
```

### Niche Diversity Mapping

```cpp
// Map neighborhood diversity across tissue
scl::kernel::niche::niche_diversity(
    spatial_graph,
    cell_types,
    n_types,
    diversity
);

// High diversity regions may indicate transition zones
```

---

## See Also

- [Spatial Analysis](spatial.md) - Spatial autocorrelation statistics
- [Neighbors](neighbors.md) - Neighbor graph construction
- [Sparse Matrices](../core/sparse.md) - Sparse matrix types
