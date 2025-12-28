# annotation.hpp

> scl/kernel/annotation.hpp · Cell type annotation from reference datasets

## Overview

This file provides kernels for annotating query cells using reference datasets with multiple annotation methods including KNN voting, correlation matching, and marker gene scoring.

This file provides:
- Reference mapping using KNN voting
- Correlation-based assignment (SingleR-style)
- Marker gene scoring (scType-style)
- Consensus annotation from multiple methods
- Novel cell type detection
- Reference profile building

**Header**: `#include "scl/kernel/annotation.hpp"`

---

## Main APIs

### reference_mapping

::: source_code file="scl/kernel/annotation.hpp" symbol="reference_mapping" collapsed
:::

**Algorithm Description**

Annotate query cells using K-nearest neighbor voting from reference dataset:

1. **KNN Collection**: For each query cell:
   - Use pre-computed KNN graph (query_to_ref_neighbors) to get k nearest reference neighbors
   - Extract reference cell indices and optionally distances

2. **Vote Counting**: For each query cell:
   - Count votes for each cell type among k neighbors
   - Weight votes by distance (optional, inverse distance weighting)
   - Store vote counts per type

3. **Assignment**: Assign cell type with most votes:
   - Select type with maximum vote count
   - Break ties by selecting type with highest average confidence
   - Handle cases where multiple types have equal votes

4. **Confidence Computation**: Compute confidence score:
   - Confidence = (votes_for_assigned_type) / (total_votes)
   - Higher confidence indicates stronger agreement among neighbors
   - Low confidence (< threshold) may indicate novel cell type

**Edge Cases**

- **No neighbors**: Query cell with no neighbors in KNN graph gets unassigned (label = -1)
- **Tied votes**: Selects first type with maximum votes (deterministic)
- **Empty reference**: Returns unassigned labels for all query cells
- **Invalid KNN graph**: Skips cells with invalid neighbor indices
- **Low confidence**: Cells with confidence < threshold may be flagged as uncertain

**Data Guarantees (Preconditions)**

- `query_labels` has capacity >= n_query
- `confidence_scores` has capacity >= n_query
- KNN graph must connect query cells to reference cells
- Reference labels must be valid indices [0, n_types)
- Both expression matrices must have same number of genes

**Complexity Analysis**

- **Time**: O(n_query * k * n_genes) for KNN voting
  - Vote counting: O(n_query * k)
  - Type assignment: O(n_query * n_types)
  - Confidence computation: O(n_query)
- **Space**: O(k * n_types) auxiliary per thread for vote storage

**Example**

```cpp
#include "scl/kernel/annotation.hpp"

scl::Sparse<Real, true> query_expression = /* ... */;      // [n_query x n_genes]
scl::Sparse<Real, true> reference_expression = /* ... */; // [n_ref x n_genes]
scl::Array<const Index> reference_labels = /* ... */;     // [n_ref]
scl::Sparse<Index, true> query_to_ref_neighbors = /* ... */; // KNN graph

scl::Array<Index> query_labels(n_query);
scl::Array<Real> confidence_scores(n_query);

scl::kernel::annotation::reference_mapping(
    query_expression,
    reference_expression,
    reference_labels,
    query_to_ref_neighbors,
    n_query,
    n_ref,
    n_types,
    query_labels,
    confidence_scores
);

// Process results
for (Index i = 0; i < n_query; ++i) {
    Index assigned_type = query_labels[i];
    Real confidence = confidence_scores[i];
    if (confidence > 0.5) {
        // High confidence assignment
    }
}
```

---

### correlation_assignment

::: source_code file="scl/kernel/annotation.hpp" symbol="correlation_assignment" collapsed
:::

**Algorithm Description**

Assign cell types using correlation with reference profiles (SingleR-style):

1. **Profile Preparation**: Use pre-computed reference profiles (types x genes):
   - Profiles are average expression per type (from build_reference_profiles)
   - Normalize profiles if needed

2. **Correlation Computation**: For each query cell:
   - Compute correlation between query cell expression and each type profile
   - Use Pearson correlation or cosine similarity
   - Handle sparse expression efficiently

3. **Assignment**: Select type with highest correlation:
   - Find maximum correlation across all types
   - Assign corresponding type label
   - Store best correlation as confidence score

4. **Optional Full Correlations**: If all_correlations provided:
   - Store correlations to all types for downstream analysis
   - Useful for uncertainty quantification

**Edge Cases**

- **Zero expression**: Cells with all-zero expression get correlation = 0 with all types
- **Perfect match**: Correlation = 1.0 indicates perfect match to type profile
- **Negative correlation**: Negative correlations indicate poor match (may be novel type)
- **Tied correlations**: Selects first type with maximum correlation
- **Empty profiles**: Returns unassigned if no valid profiles

**Data Guarantees (Preconditions)**

- `assigned_labels` has capacity >= n_query
- `correlation_scores` has capacity >= n_query
- Reference profiles must have same number of genes as query expression
- Profiles must be valid (non-empty)

**Complexity Analysis**

- **Time**: O(n_query * n_types * n_genes) for correlation computation
  - Correlation per cell-type pair: O(n_genes)
  - Total: O(n_query * n_types * n_genes)
- **Space**: O(n_types) auxiliary per query cell for correlation storage

**Example**

```cpp
scl::Sparse<Real, true> query_expression = /* ... */;  // [n_query x n_genes]
scl::Sparse<Real, true> reference_profiles = /* ... */; // [n_types x n_genes]

scl::Array<Index> assigned_labels(n_query);
scl::Array<Real> correlation_scores(n_query);
scl::Array<Real> all_correlations(n_query * n_types);  // Optional

scl::kernel::annotation::correlation_assignment(
    query_expression,
    reference_profiles,
    n_query,
    n_types,
    n_genes,
    assigned_labels,
    correlation_scores,
    all_correlations  // Optional
);

// Check assignment quality
for (Index i = 0; i < n_query; ++i) {
    Real best_corr = correlation_scores[i];
    if (best_corr > 0.7) {
        // Strong correlation match
    }
}
```

---

### build_reference_profiles

::: source_code file="scl/kernel/annotation.hpp" symbol="build_reference_profiles" collapsed
:::

**Algorithm Description**

Build average expression profiles for each cell type in reference dataset:

1. **Type Grouping**: Group reference cells by type label
   - Create type-to-cells mapping
   - Count cells per type

2. **Profile Computation**: For each cell type:
   - Sum expression across all cells of that type
   - Divide by cell count to get mean expression
   - Handle sparse expression efficiently

3. **Normalization** (optional): Normalize profiles:
   - L2 normalization for cosine similarity
   - Or keep raw means for correlation

4. **Output**: Store profiles in dense matrix (types x genes)

**Edge Cases**

- **Empty types**: Types with no cells get zero profiles
- **Single cell per type**: Profile equals that cell's expression
- **Sparse expression**: Efficiently handles sparse matrices
- **Missing types**: Types not in labels get zero profiles

**Data Guarantees (Preconditions)**

- `profiles` has capacity >= n_types * n_genes
- Reference labels must be valid indices [0, n_types)
- Expression matrix must be valid

**Complexity Analysis**

- **Time**: O(nnz_ref) for summing expression
  - Iterate over all non-zeros in reference matrix
  - Accumulate per type
- **Space**: O(n_types) auxiliary for cell counts per type

**Example**

```cpp
scl::Sparse<Real, true> reference_expression = /* ... */; // [n_ref x n_genes]
scl::Array<const Index> reference_labels = /* ... */;     // [n_ref]

Real* profiles = /* allocate n_types * n_genes */;

scl::kernel::annotation::build_reference_profiles(
    reference_expression,
    reference_labels,
    n_ref,
    n_types,
    n_genes,
    profiles
);

// profiles[t * n_genes + g] contains mean expression of type t for gene g
```

---

### marker_gene_score

::: source_code file="scl/kernel/annotation.hpp" symbol="marker_gene_score" collapsed
:::

**Algorithm Description**

Score cells using marker gene expression (scType-style):

1. **Marker Extraction**: For each cell type:
   - Use pre-defined marker genes (positive markers)
   - Optionally use negative markers (genes that should be low)

2. **Score Computation**: For each cell:
   - Sum expression of positive markers for each type
   - Subtract expression of negative markers (if used)
   - Normalize by number of markers (optional)

3. **Normalization**: If normalize=true:
   - Normalize scores per cell (sum to 1)
   - Convert to probabilities
   - Otherwise keep raw scores

4. **Output**: Store scores in matrix (cells x types)

**Edge Cases**

- **No markers**: Types with no markers get zero scores
- **Missing genes**: Genes not in expression matrix are ignored
- **Zero expression**: Cells with zero marker expression get zero scores
- **All zeros**: Cells with all-zero scores get uniform distribution if normalized

**Data Guarantees (Preconditions)**

- `scores` has capacity >= n_cells * n_types
- Marker gene indices must be valid [0, n_genes)
- Expression matrix must be valid

**Complexity Analysis**

- **Time**: O(n_cells * sum(marker_counts)) for scoring
  - For each cell, iterate over all markers of all types
  - Sparse access for expression values
- **Space**: O(n_genes) auxiliary per thread for marker lookup

**Example**

```cpp
scl::Sparse<Real, true> expression = /* ... */;  // [n_cells x n_genes]

// Define markers per type
const Index* marker_genes[] = {
    /* type 0 markers */,
    /* type 1 markers */,
    /* ... */
};
const Index marker_counts[] = { /* counts per type */ };

Real* scores = /* allocate n_cells * n_types */;

scl::kernel::annotation::marker_gene_score(
    expression,
    marker_genes,
    marker_counts,
    n_cells,
    n_genes,
    n_types,
    scores,
    true  // normalize
);

// scores[i * n_types + t] contains marker score for cell i and type t
```

---

### consensus_annotation

::: source_code file="scl/kernel/annotation.hpp" symbol="consensus_annotation" collapsed
:::

**Algorithm Description**

Combine predictions from multiple annotation methods:

1. **Vote Collection**: For each cell:
   - Collect predictions from all methods
   - Optionally weight by confidence scores

2. **Consensus Computation**:
   - **Majority Vote**: Select type predicted by most methods
   - **Weighted Vote**: Weight predictions by confidence, select highest weighted type
   - **Unanimous**: Require all methods to agree (strict consensus)

3. **Confidence**: Compute consensus confidence:
   - Agreement measure: fraction of methods agreeing
   - Weighted average of individual confidences
   - Lower confidence indicates disagreement

4. **Output**: Store consensus labels and confidence

**Edge Cases**

- **No agreement**: If no majority, selects first method's prediction
- **Tied votes**: Selects first type with maximum votes
- **All methods disagree**: Low confidence consensus
- **Single method**: Returns that method's prediction directly

**Data Guarantees (Preconditions)**

- All prediction arrays have length n_cells
- Confidence arrays (if provided) have length n_cells
- All predictions use same type indices [0, n_types)

**Complexity Analysis**

- **Time**: O(n_cells * n_methods * n_types) for vote counting
  - For each cell, count votes per type across methods
  - Select consensus type
- **Space**: O(n_types) auxiliary per cell for vote storage

**Example**

```cpp
const Index* predictions[] = {
    /* method 0 predictions */,
    /* method 1 predictions */,
    /* method 2 predictions */
};
const Real* confidences[] = {
    /* method 0 confidences */,
    /* method 1 confidences */,
    /* method 2 confidences */
};

scl::Array<Index> consensus_labels(n_cells);
scl::Array<Real> consensus_confidence(n_cells);

scl::kernel::annotation::consensus_annotation(
    predictions,
    confidences,  // Optional
    n_methods,
    n_cells,
    n_types,
    consensus_labels,
    consensus_confidence
);
```

---

### detect_novel_cell_types

::: source_code file="scl/kernel/annotation.hpp" symbol="detect_novel_cell_types" collapsed
:::

**Algorithm Description**

Detect cells that do not match any reference type well:

1. **Distance Computation**: For each query cell:
   - Compute distance to assigned type profile
   - Use cosine distance or Euclidean distance
   - Compare to distance threshold

2. **Novelty Detection**: Mark cells as novel if:
   - Distance to assigned type > threshold
   - Or correlation < minimum threshold
   - Or confidence < minimum threshold

3. **Optional Distance Output**: If distance_to_assigned provided:
   - Store distance to assigned type for each cell
   - Useful for downstream analysis

**Edge Cases**

- **All cells novel**: If threshold too strict, all cells marked novel
- **No novel cells**: If threshold too loose, no cells marked novel
- **Unassigned cells**: Cells with label = -1 are automatically novel
- **Perfect matches**: Cells with distance = 0 are never novel

**Data Guarantees (Preconditions)**

- `is_novel` has capacity >= n_query
- Assigned labels must be valid or -1 (unassigned)
- Reference profiles must be valid

**Complexity Analysis**

- **Time**: O(n_query * n_genes) for distance computation
  - For each query cell, compute distance to assigned type profile
- **Space**: O(n_genes) auxiliary per query cell for distance computation

**Example**

```cpp
scl::Sparse<Real, true> query_expression = /* ... */;
const Real* reference_profiles = /* ... */;
scl::Array<const Index> assigned_labels = /* ... */;

scl::Array<Byte> is_novel(n_query);
scl::Array<Real> distance_to_assigned(n_query);  // Optional

scl::kernel::annotation::detect_novel_cell_types(
    query_expression,
    reference_profiles,
    assigned_labels,
    n_query,
    n_types,
    n_genes,
    is_novel,
    config::DEFAULT_NOVELTY_THRESHOLD,  // 0.3
    distance_to_assigned
);

// Count novel cells
Index n_novel = 0;
for (Index i = 0; i < n_query; ++i) {
    if (is_novel[i]) n_novel++;
}
```

---

## Utility Functions

### count_cell_types

Count number of distinct cell types in label array.

::: source_code file="scl/kernel/annotation.hpp" symbol="count_cell_types" collapsed
:::

**Complexity**

- Time: O(n)
- Space: O(1) auxiliary

---

### assign_from_scores

Assign cell types from score matrix by selecting maximum.

::: source_code file="scl/kernel/annotation.hpp" symbol="assign_from_scores" collapsed
:::

**Complexity**

- Time: O(n_cells * n_types)
- Space: O(1) auxiliary

---

## Configuration

Default parameters in `scl::kernel::annotation::config`:

- `DEFAULT_CONFIDENCE_THRESHOLD = 0.5`: Minimum confidence for assignment
- `EPSILON = 1e-15`: Small constant for numerical stability
- `DEFAULT_K = 15`: Default number of neighbors for KNN voting
- `DEFAULT_NOVELTY_THRESHOLD = 0.3`: Distance threshold for novelty detection
- `PARALLEL_THRESHOLD = 500`: Minimum size for parallel processing

---

## Performance Notes

### Parallelization

- All main functions parallelize over cells
- Efficient sparse matrix access
- Minimal synchronization overhead

### Memory Efficiency

- Pre-allocated output buffers
- Efficient vote counting with minimal allocations
- Sparse expression handling

---

## See Also

- [Alignment](../alignment)
- [Neighbors](../neighbors)
- [Markers](../markers)
