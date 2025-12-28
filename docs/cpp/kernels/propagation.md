# propagation.hpp

> scl/kernel/propagation.hpp · Label propagation kernels for semi-supervised learning

## Overview

This file provides kernels for label propagation and spreading in graph-based semi-supervised learning, including hard label voting, soft label spreading, and inductive transfer.

This file provides:
- Label propagation with hard label majority voting
- Label spreading with soft probabilities
- Inductive transfer from reference to query
- Confidence-weighted propagation
- Harmonic function for regression
- Utility functions for label conversion

**Header**: `#include "scl/kernel/propagation.hpp"`

---

## Main APIs

### label_propagation

::: source_code file="scl/kernel/propagation.hpp" symbol="label_propagation" collapsed
:::

**Algorithm Description**

Perform label propagation for semi-supervised classification using hard label majority voting:

1. **Initialization**: 
   - Identify labeled nodes (labels >= 0)
   - Unlabeled nodes have labels = UNLABELED (-1)

2. **Iteration Loop**: For each iteration until convergence:
   - **Shuffle Node Order**: Randomize node processing order using Fisher-Yates
   - **Vote Collection**: For each node in shuffled order:
     - Collect weighted votes from neighbors
     - Vote weight = edge weight (adjacency value)
     - Count votes per class
   - **Label Assignment**: Assign majority class:
     - Select class with maximum weighted votes
     - Break ties deterministically (first class with max votes)
   - **Convergence Check**: Stop if no labels changed

3. **Output**: Modified labels array:
   - Unlabeled nodes assigned to majority neighbor class
   - Labeled nodes remain unchanged

**Edge Cases**

- **No labeled nodes**: Returns without changes (all remain unlabeled)
- **Isolated nodes**: Nodes with no neighbors keep original label
- **Tied votes**: Selects first class with maximum votes
- **Disconnected graph**: Each component propagates independently
- **All neighbors unlabeled**: Node remains unlabeled until neighbors get labels

**Data Guarantees (Preconditions)**

- `labels.len >= adjacency.primary_dim()`
- At least one node must have valid label (>= 0)
- Adjacency edge weights should be non-negative
- Labels array is modified in-place

**MUTABILITY**

INPLACE - modifies `labels` array directly

**Complexity Analysis**

- **Time**: O(max_iter * edges) expected
  - Each iteration: O(edges) for vote collection
  - Typically converges in few iterations
  - Parallelized with WorkspacePool
- **Space**: O(n + n_classes) auxiliary per thread for vote buffers

**Example**

```cpp
#include "scl/kernel/propagation.hpp"

scl::Sparse<Real, true> adjacency = /* ... */;  // Graph adjacency matrix
scl::Array<Index> labels(n_nodes);

// Initialize: some nodes labeled, others unlabeled (-1)
for (Index i = 0; i < n_labeled; ++i) {
    labels[labeled_indices[i]] = class_labels[i];
}
for (Index i = n_labeled; i < n_nodes; ++i) {
    labels[unlabeled_indices[i]] = config::UNLABELED;
}

// Propagate labels
scl::kernel::propagation::label_propagation(
    adjacency,
    labels,
    config::DEFAULT_MAX_ITER,  // max_iter = 100
    42                          // seed
);

// labels now contains propagated assignments
```

---

### label_spreading

::: source_code file="scl/kernel/propagation.hpp" symbol="label_spreading" collapsed
:::

**Algorithm Description**

Perform regularized label spreading with soft probability labels:

1. **Normalized Laplacian**: Compute S = D^(-1/2) * W * D^(-1/2):
   - D = diagonal degree matrix
   - W = adjacency matrix
   - Normalized for stability

2. **Iteration**: For each iteration until convergence:
   - **Propagation**: Y_new = alpha * S * Y + (1-alpha) * Y0
     - Y = current soft label probabilities
     - Y0 = initial labels (clamped for labeled nodes)
     - alpha = propagation strength (0 to 1)
   - **Normalization**: Normalize each row to sum to 1
   - **Clamping**: Reset labeled nodes to initial probabilities
   - **Convergence**: Check L1 norm change < tolerance

3. **Output**: Converged soft label probabilities:
   - Each row sums to 1
   - Labeled nodes retain (1-alpha) fraction of initial labels

**Edge Cases**

- **No labeled nodes**: All nodes get uniform probabilities
- **Alpha = 0**: No propagation, labels remain initial
- **Alpha = 1**: Full propagation, no clamping to initial
- **Disconnected graph**: Each component propagates independently
- **Zero degree nodes**: Handled by normalization

**Data Guarantees (Preconditions)**

- `label_probs.len >= n_nodes * n_classes` (row-major layout)
- `is_labeled` has length n_nodes
- `0 < alpha < 1` for stable propagation
- Initial probs for labeled nodes should sum to 1

**MUTABILITY**

INPLACE - modifies `label_probs` array directly

**Complexity Analysis**

- **Time**: O(max_iter * edges * n_classes) for propagation
  - Each iteration: O(edges * n_classes) for matrix multiplication
  - Normalization: O(n_nodes * n_classes)
- **Space**: O(n_nodes * n_classes) auxiliary for temporary storage

**Example**

```cpp
scl::Sparse<Real, true> adjacency = /* ... */;
scl::Array<Real> label_probs(n_nodes * n_classes);
bool* is_labeled = /* ... */;  // Boolean mask

// Initialize soft labels
scl::kernel::propagation::init_soft_labels(
    hard_labels, n_classes, label_probs, 1.0, 0.0
);

// Spread labels
scl::kernel::propagation::label_spreading(
    adjacency,
    label_probs,
    n_classes,
    is_labeled,
    config::DEFAULT_ALPHA,      // alpha = 0.99
    config::DEFAULT_MAX_ITER,   // max_iter = 100
    config::DEFAULT_TOLERANCE   // tol = 1e-6
);

// Convert to hard labels
scl::Array<Index> hard_labels(n_nodes);
scl::kernel::propagation::get_hard_labels(
    label_probs, n_nodes, n_classes, hard_labels
);
```

---

### inductive_transfer

::: source_code file="scl/kernel/propagation.hpp" symbol="inductive_transfer" collapsed
:::

**Algorithm Description**

Transfer labels from reference dataset to query dataset using weighted k-NN voting:

1. **Vote Collection**: For each query node in parallel:
   - Extract neighbors from ref_to_query similarity matrix
   - For each reference neighbor:
     - Get reference label
     - Weight vote by similarity (edge weight)
     - Accumulate votes per class

2. **Label Assignment**: For each query node:
   - Find class with maximum weighted votes
   - Compute confidence = best_votes / total_votes
   - If confidence >= threshold: assign class
   - Otherwise: assign UNLABELED

3. **Output**: Store predicted labels in query_labels:
   - High confidence assignments get class labels
   - Low confidence assignments get UNLABELED

**Edge Cases**

- **No neighbors**: Query nodes with no neighbors get UNLABELED
- **Tied votes**: Selects first class with maximum votes
- **Low confidence**: Nodes with confidence < threshold get UNLABELED
- **Empty reference**: All query nodes get UNLABELED

**Data Guarantees (Preconditions)**

- `ref_to_query.rows() == number of query nodes`
- `reference_labels.len >= max column index in ref_to_query`
- `query_labels.len >= ref_to_query.rows()`
- Similarity matrix must be valid

**Complexity Analysis**

- **Time**: O(nnz_ref_to_query) for vote collection
  - Each non-zero contributes to vote counting
  - Parallelized over query nodes
- **Space**: O(n_classes) per thread for vote storage

**Example**

```cpp
scl::Sparse<Real, true> ref_to_query = /* ... */;  // Query-to-reference similarities
scl::Array<const Index> reference_labels = /* ... */;  // Reference labels
scl::Array<Index> query_labels(n_query);

scl::kernel::propagation::inductive_transfer(
    ref_to_query,
    reference_labels,
    query_labels,
    n_classes,
    Real(0.5)  // confidence_threshold
);

// query_labels contains transferred labels
```

---

### confidence_propagation

::: source_code file="scl/kernel/propagation.hpp" symbol="confidence_propagation" collapsed
:::

**Algorithm Description**

Label propagation with confidence scores that modulate vote weights:

1. **Confidence-Weighted Voting**: For each node:
   - Collect votes from neighbors
   - Weight each vote by neighbor's confidence
   - Add self-vote with weight alpha * own_confidence

2. **Label Assignment**: 
   - Assign majority class from weighted votes
   - Compute new confidence = best_votes / total_votes

3. **Iteration**: Repeat until convergence:
   - Update labels and confidence scores
   - Stop when no labels change

4. **Output**: 
   - Labels propagated using confidence-weighted voting
   - Confidence scores updated to reflect certainty

**Edge Cases**

- **Low confidence neighbors**: Contribute less to voting
- **High confidence nodes**: Influence neighbors more strongly
- **Confidence = 0**: Node has no influence on neighbors
- **Confidence = 1**: Node has maximum influence

**Data Guarantees (Preconditions)**

- `labels.len >= adjacency.primary_dim()`
- `confidence.len >= adjacency.primary_dim()`
- Initial confidence scores in [0, 1]

**MUTABILITY**

INPLACE - modifies both `labels` and `confidence` arrays

**Complexity Analysis**

- **Time**: O(max_iter * edges) for propagation
  - Each iteration: O(edges) for vote collection
- **Space**: O(n + n_classes) auxiliary per thread

**Example**

```cpp
scl::Sparse<Real, true> adjacency = /* ... */;
scl::Array<Index> labels(n_nodes);
scl::Array<Real> confidence(n_nodes);

// Initialize confidence (e.g., 1.0 for labeled, 0.0 for unlabeled)
// ...

scl::kernel::propagation::confidence_propagation(
    adjacency,
    labels,
    confidence,
    n_classes,
    config::DEFAULT_ALPHA,      // alpha = 0.99
    config::DEFAULT_MAX_ITER    // max_iter = 100
);
```

---

### harmonic_function

::: source_code file="scl/kernel/propagation.hpp" symbol="harmonic_function" collapsed
:::

**Algorithm Description**

Solve the harmonic function for semi-supervised regression:

1. **Initialization**: 
   - Known values fixed (from is_known mask)
   - Unknown values initialized (e.g., to mean of known values)

2. **Iteration**: For each iteration until convergence:
   - **Jacobi Update**: For each unknown node:
     - value = weighted_avg(neighbor_values)
     - weight = edge weight from adjacency
   - **Convergence Check**: Track maximum absolute change
   - **Stop**: When max_change < tolerance

3. **Output**: 
   - Unknown values converged to harmonic solution
   - Known values unchanged
   - Solution minimizes Dirichlet energy on graph

**Edge Cases**

- **No known values**: Returns without changes
- **Isolated nodes**: Unknown isolated nodes keep initial value
- **Disconnected graph**: Each component solved independently
- **All neighbors unknown**: Node value remains initial until neighbors converge

**Data Guarantees (Preconditions)**

- `values.len >= adjacency.primary_dim()`
- `is_known` has length adjacency.primary_dim()
- At least one node must have is_known[i] = true
- Graph should be connected for well-defined solution

**MUTABILITY**

INPLACE - modifies `values` array for unknown nodes only

**Complexity Analysis**

- **Time**: O(max_iter * edges) for iteration
  - Each iteration: O(edges) for value updates
  - Typically converges in few iterations
- **Space**: O(n) auxiliary for temporary values

**Example**

```cpp
scl::Sparse<Real, true> adjacency = /* ... */;
scl::Array<Real> values(n_nodes);
bool* is_known = /* ... */;  // Boolean mask

// Initialize: some values known, others unknown
// ...

scl::kernel::propagation::harmonic_function(
    adjacency,
    values,
    is_known,
    config::DEFAULT_MAX_ITER,   // max_iter = 100
    config::DEFAULT_TOLERANCE   // tol = 1e-6
);

// values now contains interpolated values for unknown nodes
```

---

## Utility Functions

### get_hard_labels

Convert soft probability labels to hard class assignments by argmax.

::: source_code file="scl/kernel/propagation.hpp" symbol="get_hard_labels" collapsed
:::

**Complexity**

- Time: O(n_nodes * n_classes)
- Space: O(1) auxiliary

---

### init_soft_labels

Initialize soft label probability matrix from hard labels.

::: source_code file="scl/kernel/propagation.hpp" symbol="init_soft_labels" collapsed
:::

**Complexity**

- Time: O(n_nodes * n_classes)
- Space: O(1) auxiliary

---

## Configuration

Default parameters in `scl::kernel::propagation::config`:

- `DEFAULT_ALPHA = 0.99`: Default propagation parameter
- `DEFAULT_MAX_ITER = 100`: Maximum iterations
- `DEFAULT_TOLERANCE = 1e-6`: Convergence tolerance
- `UNLABELED = -1`: Marker for unlabeled nodes
- `PARALLEL_THRESHOLD = 500`: Minimum size for parallel processing
- `SIMD_THRESHOLD = 32`: Minimum size for SIMD operations
- `PREFETCH_DISTANCE = 16`: Cache line prefetch distance

---

## Performance Notes

### Parallelization

- All main functions parallelize over nodes
- Uses WorkspacePool for thread-local buffers
- Efficient sparse matrix access

### Convergence

- Label propagation typically converges in 5-20 iterations
- Label spreading may need more iterations for tight tolerance
- Harmonic function converges quickly for well-connected graphs

---

## See Also

- [Neighbors](../neighbors)
- [Graph Algorithms](../components)
- [Sparse Matrices](../core/sparse)
