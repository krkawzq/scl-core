# gram.hpp

> scl/kernel/gram.hpp · Efficient Gram matrix computation with adaptive dot product algorithms

## Overview

This file provides efficient computation of Gram matrices (inner product matrices) for sparse matrices:

- **Gram Matrix**: Compute G[i,j] = dot(row_i, row_j) for all row pairs
- **Adaptive Algorithms**: Automatically selects optimal dot product algorithm based on sparsity pattern
- **Symmetric Computation**: Only computes upper triangle, writes symmetrically
- **High Performance**: SIMD-optimized diagonal computation and parallel processing

**Header**: `#include "scl/kernel/gram.hpp"`

---

## Main APIs

### gram

::: source_code file="scl/kernel/gram.hpp" symbol="gram" collapsed
:::

**Algorithm Description**

Compute Gram matrix (inner product matrix) for sparse matrix rows with adaptive algorithm selection:

1. **Diagonal computation**: For each row i, compute G[i,i] = ||row_i||^2
   - Uses `vectorize::sum_squared` for SIMD-optimized L2 norm computation
   - Diagonal elements computed independently in parallel
2. **Off-diagonal computation**: For each pair (i, j) where i < j (upper triangle)
   - Compute G[i,j] = dot(row_i, row_j) using adaptive sparse dot product
   - Algorithm selection based on size ratio `ratio = max(n_i, n_j) / min(n_i, n_j)`:
     - **ratio < 32**: Linear merge with 8/4-way skip optimization
     - **32 <= ratio < 256**: Binary search with range narrowing
     - **ratio >= 256**: Galloping (exponential) search
   - Early exit if ranges are disjoint (O(1) check)
3. **Symmetric write**: G[j,i] = G[i,j] (copy from upper triangle)

The adaptive algorithm ensures optimal performance across different sparsity patterns.

**Edge Cases**

- **Empty rows**: Diagonal = 0, all off-diagonal elements with empty row = 0
- **Disjoint rows**: Early exit optimization detects disjoint ranges in O(1)
- **Identical rows**: G[i,j] = G[i,i] = G[j,j] (full correlation)
- **Orthogonal rows**: G[i,j] = 0 (no overlap)

**Data Guarantees (Preconditions)**

- `output.len >= n_rows^2`
- Matrix must be valid CSR format
- Row indices must be sorted within each row (standard CSR requirement)

**Complexity Analysis**

- **Time**: O(n_rows^2 * avg_nnz_per_row / n_threads)
  - Diagonal: O(n_rows * avg_nnz_per_row) parallelized
  - Off-diagonal: O(n_rows^2 * avg_nnz_per_row) with adaptive algorithm
  - Actual performance depends on sparsity pattern and algorithm selection
- **Space**: O(1) beyond output (no auxiliary allocations)

**Example**

```cpp
#include "scl/kernel/gram.hpp"

// Input sparse matrix
scl::Sparse<Real, true> matrix = /* ... */;  // [n_rows x n_cols]
scl::Index n_rows = matrix.rows();

// Pre-allocate output Gram matrix
scl::Array<Real> output(n_rows * n_rows);

// Compute Gram matrix
scl::kernel::gram::gram(matrix, output);

// Access results: output[i * n_rows + j] = dot(row_i, row_j)
// Matrix is symmetric: output[i * n_rows + j] == output[j * n_rows + i]
// Diagonal: output[i * n_rows + i] = ||row_i||^2
```

---

## Internal Algorithms

The implementation uses several internal algorithms for sparse dot products, automatically selected based on vector size ratios.

### detail::sparse_dot_adaptive

Adaptive sparse dot product that selects optimal algorithm.

**Algorithm Selection**:
- Let `n_small = min(n1, n2)`, `n_large = max(n1, n2)`, `ratio = n_large / n_small`
- **ratio < 32**: `dot_linear_branchless` - Linear merge with skip optimization
- **32 <= ratio < 256**: `dot_binary` - Binary search in large array
- **ratio >= 256**: `dot_gallop` - Galloping (exponential) search

**Early Exit**:
- Empty vectors: return 0
- Range disjoint: `max(idx1) < min(idx2)` or vice versa, return 0

### detail::dot_linear_branchless

Linear merge with skip optimization for similar-sized vectors.

**Algorithm**:
1. 8-way skip: If `idx1[i+7] < idx2[j]`, skip 8 elements
2. 4-way skip: Similar for 4 elements
3. Linear merge with prefetch for remaining elements

**Complexity**: O(n1 + n2), often much faster with skip optimization

### detail::dot_binary

Binary search for moderately imbalanced vectors.

**Algorithm**:
1. Narrow search range using `lower_bound`/`upper_bound`
2. For each element in small vector:
   - Binary search in remaining large vector range
   - If found, accumulate product
   - Advance search base to found position

**Complexity**: O(n_small * log(n_large))

### detail::dot_gallop

Galloping (exponential) search for highly imbalanced vectors.

**Algorithm**:
1. Narrow range using gallop + binary search for boundaries
2. For each element in small vector:
   - Exponential search: check positions 1, 2, 4, 8, 16...
   - Binary search within found bounds
   - Advance search base to found position

**Complexity**: O(n_small * log(ratio)) amortized

**Note**: Galloping is optimal when consecutive matches are nearby.

---

## Use Cases

### Kernel Methods

```cpp
// Compute RBF kernel from Gram matrix
scl::Array<Real> gram(n_rows * n_rows);
scl::kernel::gram::gram(matrix, gram);

Real gamma = 1.0;
for (scl::Index i = 0; i < n_rows; ++i) {
    for (scl::Index j = 0; j < n_rows; ++j) {
        Real dist_sq = gram[i * n_rows + i] + gram[j * n_rows + j] 
                      - 2 * gram[i * n_rows + j];
        Real kernel = std::exp(-gamma * dist_sq);
        // Use kernel for SVM, GP, etc.
    }
}
```

### Similarity Matrix

```cpp
// Gram matrix is similarity matrix (inner products)
scl::Array<Real> similarity(n_rows * n_rows);
scl::kernel::gram::gram(matrix, similarity);

// Normalize to cosine similarity
for (scl::Index i = 0; i < n_rows; ++i) {
    Real norm_i = std::sqrt(similarity[i * n_rows + i]);
    for (scl::Index j = 0; j < n_rows; ++j) {
        Real norm_j = std::sqrt(similarity[j * n_rows + j]);
        similarity[i * n_rows + j] /= (norm_i * norm_j);
    }
}
```

---

## Notes

- The adaptive algorithm automatically selects the best dot product method based on sparsity patterns
- Symmetric computation reduces work by ~50% (only upper triangle computed)
- Early exit optimization for disjoint ranges provides significant speedup for sparse data
- Diagonal computation uses SIMD-optimized `vectorize::sum_squared` for maximum performance
- All operations are parallelized and thread-safe

## See Also

- [Sparse Matrix Operations](../core/sparse)
- [Vectorized Operations](../core/vectorize)
- [SIMD Operations](../core/simd)
