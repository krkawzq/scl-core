// =============================================================================
// FILE: scl/kernel/propagation.h
// BRIEF: API reference for label propagation kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::propagation {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_ALPHA = Real(0.99);
    constexpr Index DEFAULT_MAX_ITER = 100;
    constexpr Real DEFAULT_TOLERANCE = Real(1e-6);
    constexpr Index UNLABELED = -1;
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 16;
}

/* -----------------------------------------------------------------------------
 * FUNCTION: label_propagation
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Perform label propagation for semi-supervised classification using
 *     hard label majority voting with random node order.
 *
 * PARAMETERS:
 *     adjacency [in]     Graph adjacency matrix (weights as edge similarities)
 *     labels    [in,out] Node labels (UNLABELED=-1 for unlabeled nodes)
 *     max_iter  [in]     Maximum number of iterations
 *     seed      [in]     Random seed for node ordering
 *
 * PRECONDITIONS:
 *     - labels.len >= adjacency.primary_dim()
 *     - At least one node must have a valid label (>= 0)
 *     - Adjacency edge weights should be non-negative
 *
 * POSTCONDITIONS:
 *     - Unlabeled nodes assigned to majority neighbor class
 *     - Converged when no labels change in an iteration
 *     - Labels remain unchanged for originally labeled nodes
 *
 * MUTABILITY:
 *     INPLACE - modifies labels array directly
 *
 * ALGORITHM:
 *     For each iteration:
 *         1. Shuffle node order using Fisher-Yates
 *         2. For each node in shuffled order:
 *            a. Compute weighted votes from neighbors
 *            b. Assign majority class label
 *         3. Stop if no labels changed
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * edges) expected
 *     Space: O(n + n_classes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool for thread-local vote buffers
 *
 * NUMERICAL NOTES:
 *     Uses weighted voting where edge weight increases vote strength
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void label_propagation(
    const Sparse<T, IsCSR>& adjacency,   // Graph adjacency matrix
    Array<Index> labels,                  // Node labels, -1 for unlabeled
    Index max_iter = config::DEFAULT_MAX_ITER,  // Maximum iterations
    uint64_t seed = 42                    // Random seed for shuffling
);

/* -----------------------------------------------------------------------------
 * FUNCTION: label_spreading
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Perform regularized label spreading with soft probability labels and
 *     clamping to initial labels for labeled nodes.
 *
 * PARAMETERS:
 *     adjacency   [in]     Graph adjacency matrix
 *     label_probs [in,out] Soft label probabilities [n_nodes * n_classes]
 *     n_classes   [in]     Number of distinct classes
 *     is_labeled  [in]     Boolean mask for labeled nodes
 *     alpha       [in]     Propagation parameter (0 to 1)
 *     max_iter    [in]     Maximum number of iterations
 *     tol         [in]     Convergence tolerance (L1 norm)
 *
 * PRECONDITIONS:
 *     - label_probs.len >= n_nodes * n_classes (row-major layout)
 *     - is_labeled has length n_nodes
 *     - 0 < alpha < 1 for stable propagation
 *     - Initial probs for labeled nodes should sum to 1
 *
 * POSTCONDITIONS:
 *     - Soft labels converged or max_iter reached
 *     - Each row of label_probs sums to 1 (normalized)
 *     - Labeled nodes retain (1-alpha) fraction of initial labels
 *
 * MUTABILITY:
 *     INPLACE - modifies label_probs array directly
 *
 * ALGORITHM:
 *     Uses normalized graph Laplacian S = D^(-1/2) * W * D^(-1/2):
 *         1. Compute row sums and D^(-1/2)
 *         2. Iterate: Y_new = alpha * S * Y + (1-alpha) * Y0
 *         3. Normalize each row to sum to 1
 *         4. Check L1 convergence
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * edges * n_classes)
 *     Space: O(n * n_classes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over nodes with SIMD accumulation
 *
 * NUMERICAL NOTES:
 *     Higher alpha spreads labels further; lower alpha trusts initial labels more
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void label_spreading(
    const Sparse<T, IsCSR>& adjacency,   // Graph adjacency matrix
    Array<Real> label_probs,              // Soft label probs [n * n_classes]
    Index n_classes,                      // Number of classes
    const bool* is_labeled,               // Labeled node mask [n]
    Real alpha = config::DEFAULT_ALPHA,   // Propagation parameter
    Index max_iter = config::DEFAULT_MAX_ITER,  // Maximum iterations
    Real tol = config::DEFAULT_TOLERANCE  // Convergence tolerance
);

/* -----------------------------------------------------------------------------
 * FUNCTION: inductive_transfer
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Transfer labels from reference dataset to query dataset using
 *     weighted k-NN voting through a precomputed similarity graph.
 *
 * PARAMETERS:
 *     ref_to_query          [in]  Similarity matrix (rows=query, cols=reference)
 *     reference_labels      [in]  Labels of reference nodes
 *     query_labels          [out] Predicted labels for query nodes
 *     n_classes             [in]  Number of distinct classes
 *     confidence_threshold  [in]  Minimum confidence to assign label
 *
 * PRECONDITIONS:
 *     - ref_to_query.rows() == number of query nodes
 *     - reference_labels.len >= max column index in ref_to_query
 *     - query_labels.len >= ref_to_query.rows()
 *
 * POSTCONDITIONS:
 *     - query_labels[i] = predicted class or UNLABELED if confidence < threshold
 *     - Confidence = best_votes / total_votes
 *
 * ALGORITHM:
 *     For each query node in parallel:
 *         1. Accumulate weighted votes from reference neighbors
 *         2. Find class with maximum votes
 *         3. Assign if confidence >= threshold, else UNLABELED
 *
 * COMPLEXITY:
 *     Time:  O(nnz_ref_to_query)
 *     Space: O(n_classes) per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized with WorkspacePool for vote buffers
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void inductive_transfer(
    const Sparse<T, IsCSR>& ref_to_query,       // Query-to-reference similarities
    Array<const Index> reference_labels,         // Reference node labels
    Array<Index> query_labels,                   // Output query labels
    Index n_classes,                             // Number of classes
    Real confidence_threshold = Real(0.5)        // Minimum confidence
);

/* -----------------------------------------------------------------------------
 * FUNCTION: confidence_propagation
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Label propagation with confidence scores that modulate vote weights
 *     and are updated during propagation.
 *
 * PARAMETERS:
 *     adjacency  [in]     Graph adjacency matrix
 *     labels     [in,out] Node labels
 *     confidence [in,out] Node confidence scores [0, 1]
 *     n_classes  [in]     Number of classes
 *     alpha      [in]     Self-vote weight multiplier
 *     max_iter   [in]     Maximum iterations
 *
 * PRECONDITIONS:
 *     - labels.len >= adjacency.primary_dim()
 *     - confidence.len >= adjacency.primary_dim()
 *     - Initial confidence scores in [0, 1]
 *
 * POSTCONDITIONS:
 *     - Labels propagated using confidence-weighted voting
 *     - Confidence updated to reflect voting certainty
 *     - Converged when no labels change
 *
 * MUTABILITY:
 *     INPLACE - modifies both labels and confidence arrays
 *
 * ALGORITHM:
 *     For each iteration:
 *         1. For each node: accumulate confidence-weighted neighbor votes
 *         2. Add self-vote with weight alpha * own_confidence
 *         3. Assign majority class
 *         4. Update confidence = best_votes / total_votes
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * edges)
 *     Space: O(n + n_classes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with WorkspacePool
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void confidence_propagation(
    const Sparse<T, IsCSR>& adjacency,   // Graph adjacency matrix
    Array<Index> labels,                  // Node labels
    Array<Real> confidence,               // Node confidence scores
    Index n_classes,                      // Number of classes
    Real alpha = config::DEFAULT_ALPHA,   // Self-vote weight
    Index max_iter = config::DEFAULT_MAX_ITER  // Maximum iterations
);

/* -----------------------------------------------------------------------------
 * FUNCTION: harmonic_function
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Solve the harmonic function for semi-supervised regression, where
 *     unknown values are the weighted average of neighbors.
 *
 * PARAMETERS:
 *     adjacency [in]     Graph adjacency matrix
 *     values    [in,out] Node values (known values fixed, unknown interpolated)
 *     is_known  [in]     Boolean mask for nodes with known values
 *     max_iter  [in]     Maximum iterations
 *     tol       [in]     Convergence tolerance (max absolute change)
 *
 * PRECONDITIONS:
 *     - values.len >= adjacency.primary_dim()
 *     - is_known has length adjacency.primary_dim()
 *     - At least one node must have is_known[i] = true
 *     - Graph should be connected for well-defined solution
 *
 * POSTCONDITIONS:
 *     - Unknown values converged to harmonic solution
 *     - Known values unchanged
 *     - Unknown value[i] = weighted_avg(neighbors[i])
 *
 * MUTABILITY:
 *     INPLACE - modifies values array for unknown nodes only
 *
 * ALGORITHM:
 *     Gauss-Seidel / Jacobi-style iteration:
 *         1. For each unknown node: value = sum(w_ij * value_j) / sum(w_ij)
 *         2. Track maximum change
 *         3. Stop when max_change < tol
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * edges)
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses Jacobi-style updates with atomic max tracking
 *
 * NUMERICAL NOTES:
 *     Solution minimizes Dirichlet energy on the graph
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void harmonic_function(
    const Sparse<T, IsCSR>& adjacency,   // Graph adjacency matrix
    Array<Real> values,                   // Node values
    const bool* is_known,                 // Known value mask [n]
    Index max_iter = config::DEFAULT_MAX_ITER,  // Maximum iterations
    Real tol = config::DEFAULT_TOLERANCE  // Convergence tolerance
);

/* -----------------------------------------------------------------------------
 * FUNCTION: get_hard_labels
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Convert soft probability labels to hard class assignments by argmax.
 *
 * PARAMETERS:
 *     probs     [in]  Soft label probabilities [n_nodes * n_classes]
 *     n_nodes   [in]  Number of nodes
 *     n_classes [in]  Number of classes
 *     labels    [out] Hard label assignments
 *     max_probs [out] Optional maximum probability for each node
 *
 * PRECONDITIONS:
 *     - probs.len >= n_nodes * n_classes
 *     - labels.len >= n_nodes
 *     - max_probs.len >= n_nodes if provided
 *
 * POSTCONDITIONS:
 *     - labels[i] = argmax_c(probs[i * n_classes + c])
 *     - max_probs[i] = max_c(probs[i * n_classes + c]) if provided
 *
 * COMPLEXITY:
 *     Time:  O(n_nodes * n_classes)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over nodes
 * -------------------------------------------------------------------------- */
void get_hard_labels(
    Array<const Real> probs,              // Soft probs [n * n_classes]
    Index n_nodes,                        // Number of nodes
    Index n_classes,                      // Number of classes
    Array<Index> labels,                  // Output hard labels
    Array<Real> max_probs = Array<Real>(nullptr, 0)  // Optional max probs
);

/* -----------------------------------------------------------------------------
 * FUNCTION: init_soft_labels
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Initialize soft label probability matrix from hard labels.
 *
 * PARAMETERS:
 *     hard_labels        [in]  Hard label assignments (UNLABELED=-1 for unknown)
 *     n_classes          [in]  Number of classes
 *     soft_labels        [out] Output probability matrix [n * n_classes]
 *     labeled_confidence [in]  Probability mass on labeled class
 *     unlabeled_prior    [in]  Prior probability for unlabeled nodes (0=uniform)
 *
 * PRECONDITIONS:
 *     - soft_labels.len >= hard_labels.len * n_classes
 *     - labeled_confidence in (0, 1]
 *     - n_classes > 0
 *
 * POSTCONDITIONS:
 *     - Labeled nodes: prob[label] = confidence, others = (1-conf)/(n-1)
 *     - Unlabeled nodes: uniform 1/n_classes or specified prior
 *     - Each row sums to 1
 *
 * COMPLEXITY:
 *     Time:  O(n_nodes * n_classes)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with SIMD fill operations
 * -------------------------------------------------------------------------- */
void init_soft_labels(
    Array<const Index> hard_labels,       // Hard labels, -1 for unlabeled
    Index n_classes,                      // Number of classes
    Array<Real> soft_labels,              // Output soft probs [n * n_classes]
    Real labeled_confidence = Real(1.0),  // Confidence for labeled nodes
    Real unlabeled_prior = Real(0.0)      // Prior for unlabeled (0=uniform)
);

} // namespace scl::kernel::propagation
