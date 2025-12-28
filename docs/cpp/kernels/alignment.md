# alignment.hpp

> scl/kernel/alignment.hpp · Multi-modal data alignment and batch integration

## Overview

This file provides kernels for aligning and integrating multi-modal datasets, particularly for batch correction and reference mapping in single-cell analysis.

This file provides:
- Anchor finding between datasets (Seurat-style integration)
- Mutual nearest neighbors (MNN) batch correction
- Integration quality scoring
- Cross-dataset KNN computation

**Header**: `#include "scl/kernel/alignment.hpp"`

---

## Main APIs

### find_anchors

::: source_code file="scl/kernel/alignment.hpp" symbol="find_anchors" collapsed
:::

**Algorithm Description**

Find anchor pairs between two datasets using Seurat-style integration approach:

1. **KNN Computation**: For each query cell, find k nearest neighbors in reference dataset:
   - Compute distances between query cell and all reference cells
   - Use cosine distance or Euclidean distance (configurable)
   - Select top k neighbors by distance

2. **Mutual Nearest Neighbor Check**: For each query-reference pair:
   - Check if query cell is in reference cell's k nearest neighbors
   - If mutual (both are in each other's k-NN), mark as potential anchor

3. **Anchor Scoring**: Score each potential anchor pair:
   - Base score: Inverse of distance between cells
   - Mutual neighbor bonus: Higher score if both are mutual neighbors
   - Filter by score threshold (config::ANCHOR_SCORE_THRESHOLD)

4. **Anchor Selection**: Select top anchors up to max_anchors:
   - Sort by score (descending)
   - Limit to max_anchors_per_cell per query cell
   - Store pairs and scores in output arrays

**Edge Cases**

- **Empty datasets**: Returns 0 anchors
- **No mutual neighbors**: Returns 0 anchors if no MNN pairs found
- **All scores below threshold**: Returns 0 anchors
- **More anchors than max_anchors**: Truncates to top max_anchors
- **Dimension mismatch**: Checked via assertions (must have same number of genes)

**Data Guarantees (Preconditions)**

- Both datasets must have same number of genes (n_genes)
- `anchor_pairs` has capacity >= max_anchors * 2
- `anchor_scores` has capacity >= max_anchors
- Query and reference matrices must be valid CSR format
- k must be positive and reasonable (< n_ref)

**Complexity Analysis**

- **Time**: O(n_query * n_ref * log(n_ref)) for distance computation and sorting
  - KNN search: O(n_query * n_ref * n_genes) for distance computation
  - Sorting: O(n_query * k * log(k)) for neighbor sorting
  - Anchor scoring: O(n_query * k)
- **Space**: O(n_query * k) auxiliary for storing neighbor indices and distances

**Example**

```cpp
#include "scl/kernel/alignment.hpp"

scl::Sparse<Real, true> query_data = /* ... */;      // [n_query x n_genes]
scl::Sparse<Real, true> reference_data = /* ... */;   // [n_ref x n_genes]

Index* anchor_pairs = /* allocate max_anchors * 2 */;
Real* anchor_scores = /* allocate max_anchors */;

Index n_anchors = scl::kernel::alignment::find_anchors(
    query_data,
    reference_data,
    n_query,
    n_ref,
    config::DEFAULT_K,  // k = 30
    anchor_pairs,
    anchor_scores,
    max_anchors
);

// Process anchors
for (Index i = 0; i < n_anchors; ++i) {
    Index query_idx = anchor_pairs[i * 2];
    Index ref_idx = anchor_pairs[i * 2 + 1];
    Real score = anchor_scores[i];
    // Use anchor pair for integration
}
```

---

### mnn_correction

::: source_code file="scl/kernel/alignment.hpp" symbol="mnn_correction" collapsed
:::

**Algorithm Description**

Apply mutual nearest neighbors (MNN) batch correction to align query dataset with reference:

1. **Correction Vector Computation**: For each MNN pair:
   - Compute difference vector: `diff = reference[ref_idx] - query[query_idx]`
   - Store correction vector for each pair

2. **Smoothing**: Smooth correction vectors across nearby cells:
   - Use KNN graph to propagate corrections
   - Weight by distance to MNN pairs
   - Avoid over-correction for distant cells

3. **Apply Correction**: For each query cell:
   - Find nearest MNN pairs
   - Interpolate correction vector from nearby MNN pairs
   - Apply correction: `query_corrected = query_original + correction`

4. **In-place Update**: Modify query_data.values() directly

**Edge Cases**

- **No MNN pairs**: Query data unchanged
- **Empty query dataset**: Returns immediately
- **Invalid MNN pairs**: Pairs with indices out of bounds are skipped
- **Zero correction**: Cells far from MNN pairs receive minimal correction

**Data Guarantees (Preconditions)**

- `query_data` values must be mutable (non-const)
- `mnn_pairs` contains valid indices [0, n_query) and [0, n_ref)
- `n_mnn` must match actual number of pairs in mnn_pairs
- Both datasets must have same number of genes

**MUTABILITY**

INPLACE - modifies `query_data.values()` directly

**Complexity Analysis**

- **Time**: O(n_mnn * n_genes) for correction computation
  - Correction vectors: O(n_mnn * n_genes)
  - Smoothing: O(n_query * k * n_genes) if using KNN smoothing
  - Application: O(n_query * n_genes)
- **Space**: O(n_genes) auxiliary per thread for correction vectors

**Example**

```cpp
// Find MNN pairs first
Index* mnn_pairs = /* ... */;
Index n_mnn = /* ... */;

// Apply MNN correction
scl::kernel::alignment::mnn_correction(
    query_data,        // Modified in-place
    reference_data,
    mnn_pairs,
    n_mnn,
    n_genes
);

// query_data is now corrected toward reference
```

---

### integration_score

::: source_code file="scl/kernel/alignment.hpp" symbol="integration_score" collapsed
:::

**Algorithm Description**

Compute integration quality score between query and reference datasets:

1. **KNN Computation**: For each cell in both datasets:
   - Find k nearest neighbors within same dataset
   - Find k nearest neighbors in other dataset

2. **Mixing Score**: Compute mixing quality:
   - For each cell, count neighbors from other dataset
   - Higher mixing indicates better integration
   - Score = fraction of cross-dataset neighbors

3. **Distance Score**: Compute average distance to cross-dataset neighbors:
   - Lower distance indicates better alignment
   - Normalize by within-dataset distances

4. **Combined Score**: Combine mixing and distance scores:
   - Weighted average: `score = w_mix * mixing_score + w_dist * distance_score`
   - Higher score indicates better integration

**Edge Cases**

- **Empty datasets**: Returns score = 0
- **No overlap**: Returns low score if datasets are completely separated
- **Perfect mixing**: Returns score = 1.0 if datasets are perfectly integrated
- **Small k**: May give noisy scores if k is too small

**Data Guarantees (Preconditions)**

- Both datasets must have same number of genes
- k must be positive and < min(n_query, n_ref)
- Output score pointer must be valid

**Complexity Analysis**

- **Time**: O((n_query + n_ref) * k * log(n_ref))
  - KNN search: O((n_query + n_ref) * n_ref * log(n_ref))
  - Score computation: O((n_query + n_ref) * k)
- **Space**: O(k) auxiliary per cell for neighbor storage

**Example**

```cpp
Real score;
scl::kernel::alignment::integration_score(
    query_data,
    reference_data,
    n_query,
    n_ref,
    config::DEFAULT_K,  // k = 30
    score
);

// score ranges from 0 (poor integration) to 1 (perfect integration)
if (score > 0.7) {
    // Good integration quality
}
```

---

## Configuration

Default parameters in `scl::kernel::alignment::config`:

- `EPSILON = 1e-10`: Small constant for numerical stability
- `DEFAULT_K = 30`: Default number of neighbors for anchor finding
- `ANCHOR_SCORE_THRESHOLD = 0.5`: Minimum score for anchor acceptance
- `MAX_ANCHORS_PER_CELL = 10`: Maximum anchors per query cell
- `PARALLEL_THRESHOLD = 32`: Minimum size for parallel processing

---

## Performance Notes

### Parallelization

- `find_anchors`: Parallelized over query cells
- `mnn_correction`: Parallelized over MNN pairs
- `integration_score`: Parallelized over cells

### Memory Efficiency

- Pre-allocated output buffers
- Minimal temporary allocations
- Efficient sparse matrix access

---

## See Also

- [Neighbor Finding](../neighbors)
- [Annotation](../annotation)
- [Sparse Matrices](../core/sparse)
