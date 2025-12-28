# doublet.hpp

> scl/kernel/doublet.hpp · Doublet detection kernels for single-cell data

## Overview

This file provides high-performance doublet detection algorithms for single-cell RNA-seq data. It implements Scrublet-style and DoubletFinder-style methods using k-nearest neighbor analysis on simulated doublets.

**Header**: `#include "scl/kernel/doublet.hpp"`

Key features:
- Synthetic doublet simulation by averaging cell pairs
- k-NN based doublet scoring
- Multiple detection methods (Scrublet, DoubletFinder, Hybrid)
- Automatic threshold estimation
- Cluster-aware doublet classification

---

## Main APIs

### simulate_doublets

::: source_code file="scl/kernel/doublet.hpp" symbol="simulate_doublets" collapsed
:::

**Algorithm Description**

Simulate synthetic doublets by averaging random cell pairs:

1. For each doublet d in parallel:
   - Randomly select two distinct cells (cell1, cell2)
   - For each gene expressed in either cell:
     - Set `profile[gene] = 0.5 * value_cell1 + 0.5 * value_cell2`
   - Store profile in doublet_profiles[d * n_genes : (d+1) * n_genes]
2. Each doublet profile is the average of two randomly selected cells
3. Uses parallel processing over doublets for efficiency

**Edge Cases**

- **n_doublets = 0**: Returns immediately, no profiles generated
- **n_cells < 2**: Cannot generate meaningful doublets
- **Empty matrix**: All profiles remain zero
- **Same cell pair selected**: Still valid (self-doublet)

**Data Guarantees (Preconditions)**

- X must be CSR format (cells x genes)
- `doublet_profiles` must be pre-allocated with `n_doublets * n_genes` elements
- `n_cells >= 2` for meaningful simulation
- Random seed ensures reproducibility

**Complexity Analysis**

- **Time**: O(n_doublets * avg_nnz_per_cell) where avg_nnz is average non-zeros per cell
- **Space**: O(n_doublets * n_genes) for output profiles

**Example**

```cpp
#include "scl/kernel/doublet.hpp"

scl::Sparse<Real, true> X = /* expression matrix [n_cells x n_genes] */;
Index n_doublets = 2 * n_cells;  // Auto: 2x number of cells

scl::Array<Real> doublet_profiles(n_doublets * n_genes);

scl::kernel::doublet::simulate_doublets(
    X, n_cells, n_genes, n_doublets, 
    doublet_profiles.data(), 42  // seed
);

// doublet_profiles now contains n_doublets synthetic profiles
```

---

### compute_knn_doublet_scores

::: source_code file="scl/kernel/doublet.hpp" symbol="compute_knn_doublet_scores" collapsed
:::

**Algorithm Description**

Compute doublet scores by k-NN against observed and simulated cells:

1. For each cell i in parallel:
   - Convert cell i to dense vector
   - Compute squared Euclidean distances to:
     - All observed cells (excluding self)
     - All simulated doublet profiles
   - Find k nearest neighbors using heap-based selection (O(n log k))
   - Count fraction of neighbors that are simulated doublets
   - Score = count / k (ranges from 0 to 1)
2. Higher scores indicate cell is more similar to simulated doublets

**Edge Cases**

- **k_neighbors > total_cells**: Uses all available neighbors
- **No simulated doublets**: All scores are 0
- **k_neighbors = 0**: Undefined behavior (should be > 0)
- **Identical cells**: May have perfect doublet neighbors

**Data Guarantees (Preconditions)**

- X must be CSR format
- `doublet_profiles` contains n_doublets profiles
- `doublet_scores.len >= n_cells`
- `k_neighbors > 0` and `k_neighbors <= n_cells + n_doublets`

**Complexity Analysis**

- **Time**: O(n_cells * (n_cells + n_doublets) * n_genes)
- **Space**: O(n_threads * (n_genes + n_total + k_neighbors)) workspace

**Example**

```cpp
Index k_neighbors = 30;
scl::Array<Real> doublet_scores(n_cells);

scl::kernel::doublet::compute_knn_doublet_scores(
    X, n_cells, n_genes,
    doublet_profiles.data(), n_doublets,
    k_neighbors, doublet_scores
);

// doublet_scores[i] = fraction of k nearest neighbors
// that are simulated doublets (0 = singlet, 1 = doublet)
```

---

### scrublet_scores

::: source_code file="scl/kernel/doublet.hpp" symbol="scrublet_scores" collapsed
:::

**Algorithm Description**

Full Scrublet-style doublet detection pipeline:

1. Simulate synthetic doublets (if n_simulated = 0, auto: 2x n_cells)
2. Compute k-NN doublet scores for all cells
3. Returns scores ready for thresholding
4. Combines simulation and scoring in one call

**Edge Cases**

- **n_simulated = 0**: Automatically uses 2 * n_cells
- **Small n_cells**: May have insufficient neighbors
- **Empty matrix**: All scores are 0

**Data Guarantees (Preconditions)**

- X must be CSR format
- `scores.len >= n_cells`
- Random seed ensures reproducibility

**Complexity Analysis**

- **Time**: O(n_cells * (n_cells + n_simulated) * n_genes)
- **Space**: O(n_simulated * n_genes) for doublet profiles

**Example**

```cpp
scl::Array<Real> scores(n_cells);

scl::kernel::doublet::scrublet_scores(
    X, n_cells, n_genes, scores,
    0,      // n_simulated: 0 = auto (2x n_cells)
    30,     // k_neighbors
    42      // seed
);

// scores[i] contains Scrublet doublet score for cell i
```

---

### detect_doublets

::: source_code file="scl/kernel/doublet.hpp" symbol="detect_doublets" collapsed
:::

**Algorithm Description**

Full doublet detection pipeline (simulate, score, threshold, call):

1. Simulate synthetic doublets based on method
2. Compute doublet scores using selected method
3. Estimate threshold from expected doublet rate
4. Call doublets: `is_doublet[i] = (scores[i] > threshold)`
5. Returns total number of detected doublets

**Edge Cases**

- **expected_rate = 0**: No doublets called
- **expected_rate = 1**: All cells called as doublets
- **Method = Hybrid**: Combines multiple scoring signals

**Data Guarantees (Preconditions)**

- X must be CSR format
- `scores.len >= n_cells`
- `is_doublet.len >= n_cells`
- `expected_rate` in (0, 1)

**Complexity Analysis**

- **Time**: O(n_cells * (n_cells + n_simulated) * n_genes)
- **Space**: O(n_simulated * n_genes) for profiles

**Example**

```cpp
scl::Array<Real> scores(n_cells);
scl::Array<bool> is_doublet(n_cells);

Index n_detected = scl::kernel::doublet::detect_doublets(
    X, n_cells, n_genes,
    scores, is_doublet,
    scl::kernel::doublet::DoubletMethod::Scrublet,
    0.06,   // expected_doublet_rate
    30,     // k_neighbors
    42      // seed
);

// is_doublet[i] = true if cell i is a doublet
// n_detected = total number of doublets found
```

---

### estimate_threshold

::: source_code file="scl/kernel/doublet.hpp" symbol="estimate_threshold" collapsed
:::

**Algorithm Description**

Estimate score threshold from expected doublet rate:

1. Copy scores to temporary buffer
2. Sort scores using SIMD-optimized sort (O(n log n))
3. Find percentile: `index = (1 - expected_rate) * n`
4. Return `scores[index]` as threshold

**Edge Cases**

- **expected_rate = 0**: Returns maximum score (no doublets)
- **expected_rate = 1**: Returns minimum score (all doublets)
- **Empty scores**: Undefined behavior

**Data Guarantees (Preconditions)**

- `scores.len > 0`
- `expected_rate` in (0, 1)

**Complexity Analysis**

- **Time**: O(n log n) for sorting
- **Space**: O(n) for sorted copy

**Example**

```cpp
Real threshold = scl::kernel::doublet::estimate_threshold(
    scores, 0.06  // 6% expected doublet rate
);

// threshold is score at 94th percentile
// Approximately 6% of cells will have score > threshold
```

---

### call_doublets

::: source_code file="scl/kernel/doublet.hpp" symbol="call_doublets" collapsed
:::

**Algorithm Description**

Call doublets based on score threshold:

1. For each cell i:
   - `is_doublet[i] = (scores[i] > threshold)`
2. Count total number of doublets
3. Returns count

**Edge Cases**

- **threshold = infinity**: No doublets called
- **threshold = -infinity**: All cells called as doublets
- **Empty scores**: Returns 0

**Data Guarantees (Preconditions)**

- `is_doublet.len >= scores.len`

**Complexity Analysis**

- **Time**: O(n)
- **Space**: O(1)

**Example**

```cpp
scl::Array<bool> is_doublet(n_cells);

Index n_doublets = scl::kernel::doublet::call_doublets(
    scores, threshold, is_doublet
);

// is_doublet[i] = true if scores[i] > threshold
// n_doublets = count of true values
```

---

## Utility Functions

### detect_bimodal_threshold

Detect threshold using bimodal distribution (histogram valley detection).

::: source_code file="scl/kernel/doublet.hpp" symbol="detect_bimodal_threshold" collapsed
:::

**Complexity**

- Time: O(n + n_bins)
- Space: O(n_bins)

---

### doublet_score_stats

Compute statistics (mean, std_dev, median) of doublet scores.

::: source_code file="scl/kernel/doublet.hpp" symbol="doublet_score_stats" collapsed
:::

**Complexity**

- Time: O(n log n) for median
- Space: O(n) for sorted copy

---

### expected_doublets

Calculate expected number of doublets given cell count and rate.

::: source_code file="scl/kernel/doublet.hpp" symbol="expected_doublets" collapsed
:::

**Complexity**

- Time: O(1)
- Space: O(1)

---

## Notes

- Doublet simulation uses random sampling - results vary with seed
- k-NN computation is the most expensive step - consider PCA reduction for large datasets
- Threshold estimation assumes scores follow expected distribution
- Scrublet method is most commonly used in practice

## See Also

- [Neighbors Module](./neighbors) - For k-NN graph construction
- [PCA/Projection](./projection) - For dimensionality reduction before scoring
