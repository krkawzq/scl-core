# coexpression.hpp

> scl/kernel/coexpression.hpp · High-performance co-expression module detection (WGCNA-style) for gene network analysis

## Overview

This file provides WGCNA-style co-expression analysis for detecting gene modules. The pipeline includes correlation computation, adjacency matrix construction, topological overlap, hierarchical clustering, and module eigengene computation.

Key features:
- Multiple correlation types (Pearson, Spearman, Bicor)
- Soft power adjacency transformation
- Topological Overlap Matrix (TOM) computation
- Hierarchical clustering for module detection
- Module eigengene computation
- Module-trait correlation analysis

**Header**: `#include "scl/kernel/coexpression.hpp"`

---

## Main APIs

### correlation_matrix

::: source_code file="scl/kernel/coexpression.hpp" symbol="correlation_matrix" collapsed
:::

**Algorithm Description**

Compute pairwise correlation matrix for genes:

1. **For each gene pair (i, j)** (parallelized):
   - Extract expression vectors for genes i and j across all cells
   - Compute correlation based on `corr_type`:
     - **Pearson**: Standard Pearson correlation coefficient
     - **Spearman**: Rank-based correlation (robust to outliers)
     - **Bicor**: Biweight midcorrelation (robust, handles outliers)
2. **Symmetric storage**: Store upper triangular matrix (symmetric)

**Edge Cases**

- **Zero variance genes**: Correlation is undefined, set to 0 or NaN (implementation-dependent)
- **Constant genes**: Correlation with constant is 0
- **Empty expression matrix**: All correlations are 0 or NaN
- **Sparse expression**: Handled efficiently via sparse matrix operations

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR format (cells x genes)
- `corr_matrix` has capacity >= n_genes * n_genes
- Expression values should be normalized (recommended)

**Complexity Analysis**

- **Time**: O(n_genes^2 * n_cells)
  - For each gene pair: O(n_cells) for correlation computation
  - Parallelized over gene pairs
- **Space**: O(n_genes * n_cells) auxiliary for storing gene expression vectors

**Example**

```cpp
#include "scl/kernel/coexpression.hpp"

Sparse<Real, true> expression = /* ... */;  // cells x genes
Real* corr_matrix = /* allocate n_genes * n_genes */;

scl::kernel::coexpression::correlation_matrix(
    expression,
    n_cells,
    n_genes,
    corr_matrix,
    CorrelationType::Pearson  // or Spearman, Bicor
);

// corr_matrix[i * n_genes + j] contains correlation between gene i and j
// Matrix is symmetric: corr_matrix[i][j] == corr_matrix[j][i]
```

---

### detect_modules

::: source_code file="scl/kernel/coexpression.hpp" symbol="detect_modules" collapsed
:::

**Algorithm Description**

Detect co-expression modules from dissimilarity matrix:

1. **Hierarchical clustering**: Perform hierarchical clustering on dissimilarity matrix
   - Use average linkage or other linkage method
   - Build dendrogram structure
2. **Dynamic tree cutting**: Cut dendrogram to form modules
   - Apply minimum module size constraint
   - Merge modules with similarity > merge_cut_height
3. **Module assignment**: Assign each gene to a module label

**Edge Cases**

- **All genes identical**: Single module containing all genes
- **No similarity**: Each gene becomes its own module (if min_module_size = 1)
- **Very small modules**: Merged or discarded based on min_module_size
- **Empty dissimilarity**: Returns 0 modules

**Data Guarantees (Preconditions)**

- `dissim` is valid dissimilarity matrix (symmetric, non-negative)
- `module_labels` has capacity >= n_genes
- `min_module_size >= 1`
- `merge_cut_height` in range [0, 1]

**Complexity Analysis**

- **Time**: O(n_genes^2 * log(n_genes))
  - Hierarchical clustering: O(n_genes^2 * log(n_genes))
  - Tree cutting: O(n_genes)
- **Space**: O(n_genes^2) auxiliary for clustering

**Example**

```cpp
Real* dissim = /* ... */;  // Dissimilarity matrix from TOM
Index* module_labels = /* allocate n_genes */;

Index n_modules = scl::kernel::coexpression::detect_modules(
    dissim,
    n_genes,
    module_labels,
    config::DEFAULT_MIN_MODULE_SIZE,  // min_module_size = 30
    config::DEFAULT_MERGE_CUT_HEIGHT  // merge_cut_height = 0.25
);

// module_labels[i] contains module ID for gene i
// Returns number of modules detected
```

---

### adjacency_matrix

::: source_code file="scl/kernel/coexpression.hpp" symbol="adjacency_matrix" collapsed
:::

**Algorithm Description**

Convert correlation matrix to adjacency matrix using soft power:

1. **For each correlation value** (parallelized):
   - Apply soft power transformation based on `adj_type`:
     - **Unsigned**: `adj = |corr|^power`
     - **Signed**: `adj = (0.5 + 0.5 * corr)^power`
     - **SignedHybrid**: `adj = corr^power` if corr > 0, else 0
2. **Clamping**: Ensure values are in valid range [0, 1] or [-1, 1]

Soft power transformation enhances strong correlations and suppresses weak ones, creating a scale-free network topology.

**Edge Cases**

- **Negative correlations** (Unsigned): Absolute value used
- **Zero correlations**: Result is 0
- **Perfect correlations** (|corr| = 1): Result is 1
- **Very small correlations**: Suppressed by power transformation

**Data Guarantees (Preconditions)**

- `corr_matrix` contains valid correlations in range [-1, 1]
- `adjacency` has capacity >= n_genes * n_genes
- `power > 0` (typically in range [4, 20])

**Complexity Analysis**

- **Time**: O(n_genes^2) - parallelized over matrix elements
- **Space**: O(1) auxiliary

**Example**

```cpp
Real* corr_matrix = /* ... */;
Real* adjacency = /* allocate n_genes * n_genes */;

scl::kernel::coexpression::adjacency_matrix(
    corr_matrix,
    n_genes,
    adjacency,
    config::DEFAULT_SOFT_POWER,  // power = 6
    AdjacencyType::Unsigned      // or Signed, SignedHybrid
);

// adjacency[i * n_genes + j] contains adjacency value
// Values in [0, 1] for unsigned, [-1, 1] for signed
```

---

### topological_overlap_matrix

::: source_code file="scl/kernel/coexpression.hpp" symbol="topological_overlap_matrix" collapsed
:::

**Algorithm Description**

Compute Topological Overlap Matrix (TOM) from adjacency matrix:

1. **For each gene pair (i, j)** (parallelized):
   - Compute shared neighbors: `sum_k(min(adj[i,k], adj[j,k]))`
   - Compute degrees: `degree[i] = sum_k(adj[i,k])`
   - TOM formula: `TOM[i,j] = (shared_neighbors + adj[i,j]) / (min(degree[i], degree[j]) + 1 - adj[i,j])`

TOM measures the similarity between genes based on shared neighbors, providing a smoothed version of the adjacency matrix.

**Edge Cases**

- **Isolated genes** (degree = 0): TOM with other genes is 0
- **Perfectly connected genes**: TOM = 1
- **No shared neighbors**: TOM depends only on direct adjacency

**Data Guarantees (Preconditions)**

- `adjacency` is valid adjacency matrix (symmetric, non-negative)
- `tom` has capacity >= n_genes * n_genes

**Complexity Analysis**

- **Time**: O(n_genes^3)
  - For each gene pair: O(n_genes) to compute shared neighbors
  - Parallelized over gene pairs
- **Space**: O(n_genes) auxiliary for storing degrees

**Example**

```cpp
Real* adjacency = /* ... */;
Real* tom = /* allocate n_genes * n_genes */;

scl::kernel::coexpression::topological_overlap_matrix(
    adjacency,
    n_genes,
    tom
);

// tom[i * n_genes + j] contains TOM value
// TOM measures shared neighbors between genes
```

---

### module_eigengene

::: source_code file="scl/kernel/coexpression.hpp" symbol="module_eigengene" collapsed
:::

**Algorithm Description**

Compute module eigengene (first principal component) for a module:

1. **Extract module genes**: Select genes belonging to specified module
2. **PCA computation**: Compute first principal component of module expression
   - Center expression matrix (subtract mean per gene)
   - Compute covariance matrix or use iterative method
   - Extract dominant eigenvector
3. **Projection**: Project cell expression onto eigengene

The eigengene represents the "summary" expression pattern of a module across cells.

**Edge Cases**

- **Empty module**: Returns zero vector
- **Single gene module**: Eigengene equals normalized gene expression
- **Constant module**: Eigengene is constant

**Data Guarantees (Preconditions)**

- `module_labels` contains valid module assignments
- `module_id` is valid module ID (exists in module_labels)
- `eigengene` has capacity >= n_cells

**Complexity Analysis**

- **Time**: O(n_cells * n_module_genes)
  - O(n_module_genes^2) for covariance computation
  - O(n_cells * n_module_genes) for projection
- **Space**: O(n_cells) auxiliary

**Example**

```cpp
Sparse<Real, true> expression = /* ... */;
const Index* module_labels = /* ... */;
Index module_id = 5;  // Module to compute eigengene for
Array<Real> eigengene(n_cells);

scl::kernel::coexpression::module_eigengene(
    expression,
    module_labels,
    module_id,
    n_cells,
    n_genes,
    eigengene
);

// eigengene[i] contains first PC of module expression in cell i
```

---

## Utility Functions

### tom_dissimilarity

Convert TOM matrix to dissimilarity matrix for clustering.

::: source_code file="scl/kernel/coexpression.hpp" symbol="tom_dissimilarity" collapsed
:::

**Complexity**

- Time: O(n_genes^2)
- Space: O(1) auxiliary

---

### hierarchical_clustering

Perform hierarchical clustering on dissimilarity matrix.

::: source_code file="scl/kernel/coexpression.hpp" symbol="hierarchical_clustering" collapsed
:::

**Complexity**

- Time: O(n_genes^2 * log(n_genes))
- Space: O(n_genes^2) auxiliary

---

### cut_tree

Cut hierarchical clustering tree at specified height.

::: source_code file="scl/kernel/coexpression.hpp" symbol="cut_tree" collapsed
:::

**Complexity**

- Time: O(n_genes)
- Space: O(n_genes) auxiliary

---

### all_module_eigengenes

Compute eigengenes for all modules.

::: source_code file="scl/kernel/coexpression.hpp" symbol="all_module_eigengenes" collapsed
:::

**Complexity**

- Time: O(n_modules * n_cells * avg_module_size)
- Space: O(n_cells) auxiliary per module

---

### module_trait_correlation

Compute correlation between module eigengenes and traits.

::: source_code file="scl/kernel/coexpression.hpp" symbol="module_trait_correlation" collapsed
:::

**Complexity**

- Time: O(n_modules * n_traits * n_samples)
- Space: O(n_samples) auxiliary

---

## Notes

**Configuration Constants**

Default parameters in `scl::kernel::coexpression::config`:
- `DEFAULT_SOFT_POWER = 6`
- `DEFAULT_MIN_MODULE_SIZE = 30`
- `DEFAULT_MERGE_CUT_HEIGHT = 0.25`

**WGCNA Pipeline**

Complete pipeline:
1. `correlation_matrix` - Compute correlations
2. `adjacency_matrix` - Build adjacency
3. `topological_overlap_matrix` - Compute TOM
4. `tom_dissimilarity` - Convert to dissimilarity
5. `detect_modules` - Detect modules
6. `all_module_eigengenes` - Compute eigengenes
7. `module_trait_correlation` - Correlate with traits

**Performance Considerations**

- TOM computation is O(n_genes^3) - use for moderate-sized gene sets (n_genes < 10000)
- Most operations are parallelized for better performance
- Pre-allocate output buffers to avoid repeated allocations

## See Also

- [Correlation](/cpp/kernels/correlation) - General correlation analysis
- [Statistics](/cpp/kernels/statistics) - Statistical analysis
