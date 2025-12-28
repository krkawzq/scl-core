# Propagation

Label propagation kernels for semi-supervised learning and graph-based classification.

## Overview

Propagation provides:

- **Label Propagation** - Hard label majority voting
- **Label Spreading** - Soft probability labels with regularization
- **Inductive Transfer** - Transfer labels from reference to query
- **Confidence Propagation** - Confidence-weighted label propagation
- **Harmonic Function** - Semi-supervised regression
- **Utility Functions** - Label conversion and initialization

## Label Propagation

### label_propagation

Perform label propagation for semi-supervised classification using hard label majority voting:

```cpp
#include "scl/kernel/propagation.hpp"

Sparse<Real, true> adjacency = /* ... */;  // Graph adjacency matrix
Array<Index> labels(n_nodes);
// Initialize: labels[i] = class_id for labeled, -1 for unlabeled

scl::kernel::propagation::label_propagation(
    adjacency,
    labels,
    config::DEFAULT_MAX_ITER,  // max_iter = 100
    42                         // seed
);
```

**Parameters:**
- `adjacency`: Graph adjacency matrix (weights as edge similarities)
- `labels`: Node labels (UNLABELED=-1 for unlabeled nodes), modified in-place
- `max_iter`: Maximum number of iterations
- `seed`: Random seed for node ordering

**Postconditions:**
- Unlabeled nodes assigned to majority neighbor class
- Converged when no labels change in an iteration
- Labels remain unchanged for originally labeled nodes

**Algorithm:**
For each iteration:
1. Shuffle node order using Fisher-Yates
2. For each node in shuffled order:
   - Compute weighted votes from neighbors
   - Assign majority class label
3. Stop if no labels changed

**Complexity:**
- Time: O(max_iter * edges) expected
- Space: O(n + n_classes) auxiliary

**Use cases:**
- Semi-supervised classification
- Graph-based learning
- When only few labels available

## Label Spreading

### label_spreading

Perform regularized label spreading with soft probability labels:

```cpp
Array<Real> label_probs(n_nodes * n_classes);  // Soft probabilities
const bool* is_labeled = /* ... */;  // Labeled node mask

scl::kernel::propagation::label_spreading(
    adjacency,
    label_probs,
    n_classes,
    is_labeled,
    config::DEFAULT_ALPHA,      // alpha = 0.99
    config::DEFAULT_MAX_ITER,
    config::DEFAULT_TOLERANCE  // tol = 1e-6
);
```

**Parameters:**
- `adjacency`: Graph adjacency matrix
- `label_probs`: Soft label probabilities [n_nodes * n_classes], modified in-place
- `n_classes`: Number of distinct classes
- `is_labeled`: Boolean mask for labeled nodes
- `alpha`: Propagation parameter (0 to 1)
- `max_iter`: Maximum iterations
- `tol`: Convergence tolerance (L1 norm)

**Postconditions:**
- Soft labels converged or max_iter reached
- Each row of label_probs sums to 1 (normalized)
- Labeled nodes retain (1-alpha) fraction of initial labels

**Algorithm:**
Uses normalized graph Laplacian S = D^(-1/2) * W * D^(-1/2):
1. Compute row sums and D^(-1/2)
2. Iterate: Y_new = alpha * S * Y + (1-alpha) * Y0
3. Normalize each row to sum to 1
4. Check L1 convergence

**Complexity:**
- Time: O(max_iter * edges * n_classes)
- Space: O(n * n_classes) auxiliary

**Use cases:**
- Soft classification
- Probability estimation
- When confidence scores needed

## Inductive Transfer

### inductive_transfer

Transfer labels from reference dataset to query dataset using weighted k-NN voting:

```cpp
Sparse<Real, true> ref_to_query = /* ... */;  // Similarity matrix
Array<const Index> reference_labels = /* ... */;
Array<Index> query_labels(n_query);

scl::kernel::propagation::inductive_transfer(
    ref_to_query,
    reference_labels,
    query_labels,
    n_classes,
    Real(0.5)  // confidence_threshold
);
```

**Parameters:**
- `ref_to_query`: Similarity matrix (rows=query, cols=reference)
- `reference_labels`: Labels of reference nodes
- `query_labels`: Predicted labels for query nodes
- `n_classes`: Number of distinct classes
- `confidence_threshold`: Minimum confidence to assign label

**Postconditions:**
- `query_labels[i]` = predicted class or UNLABELED if confidence < threshold
- Confidence = best_votes / total_votes

**Algorithm:**
For each query node in parallel:
1. Accumulate weighted votes from reference neighbors
2. Find class with maximum votes
3. Assign if confidence >= threshold, else UNLABELED

**Complexity:**
- Time: O(nnz_ref_to_query)
- Space: O(n_classes) per thread

**Use cases:**
- Transfer learning
- Reference-based annotation
- Cross-dataset labeling

## Confidence Propagation

### confidence_propagation

Label propagation with confidence scores that modulate vote weights:

```cpp
Array<Index> labels(n_nodes);
Array<Real> confidence(n_nodes);  // Confidence scores [0, 1]

scl::kernel::propagation::confidence_propagation(
    adjacency,
    labels,
    confidence,
    n_classes,
    config::DEFAULT_ALPHA,  // Self-vote weight multiplier
    config::DEFAULT_MAX_ITER
);
```

**Parameters:**
- `adjacency`: Graph adjacency matrix
- `labels`: Node labels, modified in-place
- `confidence`: Node confidence scores [0, 1], modified in-place
- `n_classes`: Number of classes
- `alpha`: Self-vote weight multiplier

**Postconditions:**
- Labels propagated using confidence-weighted voting
- Confidence updated to reflect voting certainty
- Converged when no labels change

**Algorithm:**
For each iteration:
1. For each node: accumulate confidence-weighted neighbor votes
2. Add self-vote with weight alpha * own_confidence
3. Assign majority class
4. Update confidence = best_votes / total_votes

**Complexity:**
- Time: O(max_iter * edges)
- Space: O(n + n_classes) auxiliary

**Use cases:**
- Confidence-aware propagation
- Quality control
- Uncertainty quantification

## Harmonic Function

### harmonic_function

Solve the harmonic function for semi-supervised regression:

```cpp
Array<Real> values(n_nodes);
const bool* is_known = /* ... */;  // Known value mask

scl::kernel::propagation::harmonic_function(
    adjacency,
    values,
    is_known,
    config::DEFAULT_MAX_ITER,
    config::DEFAULT_TOLERANCE
);
```

**Parameters:**
- `adjacency`: Graph adjacency matrix
- `values`: Node values (known values fixed, unknown interpolated), modified in-place
- `is_known`: Boolean mask for nodes with known values
- `max_iter`: Maximum iterations
- `tol`: Convergence tolerance (max absolute change)

**Postconditions:**
- Unknown values converged to harmonic solution
- Known values unchanged
- Unknown value[i] = weighted_avg(neighbors[i])

**Algorithm:**
Gauss-Seidel / Jacobi-style iteration:
1. For each unknown node: value = sum(w_ij * value_j) / sum(w_ij)
2. Track maximum change
3. Stop when max_change < tol

**Complexity:**
- Time: O(max_iter * edges)
- Space: O(n) auxiliary

**Use cases:**
- Semi-supervised regression
- Value interpolation
- Missing data imputation

## Utility Functions

### get_hard_labels

Convert soft probability labels to hard class assignments by argmax:

```cpp
Array<const Real> probs = /* ... */;  // [n_nodes * n_classes]
Array<Index> labels(n_nodes);
Array<Real> max_probs(n_nodes);  // Optional

scl::kernel::propagation::get_hard_labels(
    probs,
    n_nodes,
    n_classes,
    labels,
    max_probs  // Optional
);
```

**Parameters:**
- `probs`: Soft label probabilities [n_nodes * n_classes]
- `n_nodes`: Number of nodes
- `n_classes`: Number of classes
- `labels`: Hard label assignments
- `max_probs`: Optional maximum probability for each node

**Postconditions:**
- `labels[i]` = argmax_c(probs[i * n_classes + c])
- `max_probs[i]` = max_c(probs[i * n_classes + c]) if provided

**Complexity:**
- Time: O(n_nodes * n_classes)
- Space: O(1) auxiliary

### init_soft_labels

Initialize soft label probability matrix from hard labels:

```cpp
Array<const Index> hard_labels = /* ... */;  // -1 for unlabeled
Array<Real> soft_labels(n_nodes * n_classes);

scl::kernel::propagation::init_soft_labels(
    hard_labels,
    n_classes,
    soft_labels,
    Real(1.0),   // labeled_confidence
    Real(0.0)    // unlabeled_prior (0=uniform)
);
```

**Parameters:**
- `hard_labels`: Hard label assignments (UNLABELED=-1 for unknown)
- `n_classes`: Number of classes
- `soft_labels`: Output probability matrix [n * n_classes]
- `labeled_confidence`: Probability mass on labeled class
- `unlabeled_prior`: Prior probability for unlabeled nodes (0=uniform)

**Postconditions:**
- Labeled nodes: prob[label] = confidence, others = (1-conf)/(n-1)
- Unlabeled nodes: uniform 1/n_classes or specified prior
- Each row sums to 1

**Complexity:**
- Time: O(n_nodes * n_classes)
- Space: O(1) auxiliary

## Configuration

Default parameters in `scl::kernel::propagation::config`:

```cpp
namespace config {
    constexpr Real DEFAULT_ALPHA = 0.99;
    constexpr Index DEFAULT_MAX_ITER = 100;
    constexpr Real DEFAULT_TOLERANCE = 1e-6;
    constexpr Index UNLABELED = -1;
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 16;
}
```

## Performance Considerations

### Parallelization

All propagation functions are parallelized:
- `label_propagation`: Parallel with WorkspacePool
- `label_spreading`: Parallel over nodes with SIMD
- `inductive_transfer`: Parallel over query nodes
- `confidence_propagation`: Parallel with WorkspacePool

### Memory Efficiency

- WorkspacePool for thread-local buffers
- In-place modifications when possible
- Minimal temporary allocations

## Best Practices

### 1. Initialize Labels Properly

```cpp
// For label propagation
Array<Index> labels(n_nodes, config::UNLABELED);
labels[seed_nodes] = /* class assignments */;

// For label spreading
Array<Real> probs(n_nodes * n_classes);
scl::kernel::propagation::init_soft_labels(
    hard_labels, n_classes, probs
);
```

### 2. Choose Appropriate Method

```cpp
// For hard classification
scl::kernel::propagation::label_propagation(adjacency, labels);

// For soft probabilities
scl::kernel::propagation::label_spreading(
    adjacency, probs, n_classes, is_labeled
);

// For regression/interpolation
scl::kernel::propagation::harmonic_function(
    adjacency, values, is_known
);
```

### 3. Tune Alpha Parameter

```cpp
// Higher alpha: spread labels further
scl::kernel::propagation::label_spreading(
    adjacency, probs, n_classes, is_labeled, 0.99
);

// Lower alpha: trust initial labels more
scl::kernel::propagation::label_spreading(
    adjacency, probs, n_classes, is_labeled, 0.5
);
```

## Examples

### Complete Semi-Supervised Classification

```cpp
// 1. Initialize labels
Array<Index> labels(n_nodes, config::UNLABELED);
labels[labeled_indices] = /* class assignments */;

// 2. Propagate labels
scl::kernel::propagation::label_propagation(
    adjacency, labels, 100, 42
);

// 3. Get results
for (Index i = 0; i < n_nodes; ++i) {
    if (labels[i] != config::UNLABELED) {
        // Node i classified as class labels[i]
    }
}
```

### Label Spreading with Soft Probabilities

```cpp
// 1. Initialize soft labels
Array<Index> hard_labels(n_nodes, config::UNLABELED);
hard_labels[labeled_indices] = /* classes */;
Array<Real> probs(n_nodes * n_classes);
scl::kernel::propagation::init_soft_labels(
    hard_labels, n_classes, probs
);

// 2. Spread labels
const bool* is_labeled = /* ... */;
scl::kernel::propagation::label_spreading(
    adjacency, probs, n_classes, is_labeled
);

// 3. Convert to hard labels
Array<Index> final_labels(n_nodes);
Array<Real> confidences(n_nodes);
scl::kernel::propagation::get_hard_labels(
    probs, n_nodes, n_classes, final_labels, confidences
);
```

---

::: tip Alpha Parameter
Higher alpha (closer to 1) spreads labels further in the graph. Lower alpha trusts initial labels more.
:::

::: warning Convergence
Label propagation may not converge for some graphs. Check iteration count and consider using label spreading with tolerance.
:::

