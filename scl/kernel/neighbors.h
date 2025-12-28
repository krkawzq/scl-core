// =============================================================================
// FILE: scl/kernel/neighbors.h
// BRIEF: API reference for K-nearest neighbors computation
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::neighbors {

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_norms
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute squared L2 norms for each row/column of a sparse matrix.
 *
 * PARAMETERS:
 *     matrix   [in]  Sparse matrix (CSR or CSC)
 *     norms_sq [out] Pre-allocated buffer for squared norms
 *
 * PRECONDITIONS:
 *     - norms_sq.len >= matrix.primary_dim()
 *
 * POSTCONDITIONS:
 *     - norms_sq[i] = sum of squared values in row/column i
 *
 * ALGORITHM:
 *     For each row in parallel:
 *         Use SIMD-optimized scl::vectorize::sum_squared
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows with no shared mutable state
 *
 * THROWS:
 *     SCL_CHECK_DIM - if norms_sq size is insufficient
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_norms(
    const Sparse<T, IsCSR>& matrix, // Input sparse matrix
    Array<T> norms_sq               // Output squared norms [primary_dim]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: knn
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Find K nearest neighbors for each row in a sparse matrix using
 *     Euclidean distance.
 *
 * PARAMETERS:
 *     matrix        [in]  Sparse matrix (n_samples x n_features)
 *     norms_sq      [in]  Pre-computed squared norms from compute_norms()
 *     k             [in]  Number of neighbors to find
 *     out_indices   [out] Neighbor indices, shape (n_samples * k)
 *     out_distances [out] Neighbor distances, shape (n_samples * k)
 *
 * PRECONDITIONS:
 *     - norms_sq.len >= matrix.primary_dim()
 *     - norms_sq contains valid squared norms from compute_norms()
 *     - out_indices.len >= matrix.primary_dim() * k
 *     - out_distances.len >= matrix.primary_dim() * k
 *     - k > 0
 *
 * POSTCONDITIONS:
 *     - For each sample i:
 *       - out_indices[i*k : i*k+k] contains indices of k nearest neighbors
 *       - out_distances[i*k : i*k+k] contains Euclidean distances to neighbors
 *       - Neighbors are sorted by distance (ascending)
 *       - Self (i) is excluded from neighbors
 *     - If fewer than k neighbors exist: remaining slots filled with
 *       index=-1 and distance=infinity
 *
 * ALGORITHM:
 *     For each sample i in parallel:
 *     1. Maintain max-heap of size k for nearest neighbors
 *     2. For each candidate j != i:
 *        a. Cauchy-Schwarz pruning: skip if |norm_i - norm_j| >= current_max
 *        b. Compute sparse dot product using adaptive strategy:
 *           - Linear merge: for similar-size vectors
 *           - Binary search: for ratio >= 32
 *           - Galloping: for ratio >= 256
 *        c. Compute distance: sqrt(norm_i + norm_j - 2*dot)
 *        d. Update heap if distance < current max
 *     3. Sort final heap to get ascending order
 *
 *     Sparse dot optimizations:
 *     - 8-way/4-way skip for non-overlapping index ranges
 *     - Prefetch in merge loop
 *     - Early exit on disjoint ranges (O(1) check)
 *
 * COMPLEXITY:
 *     Time:  O(n^2 * avg_nnz) worst case, often much better with pruning
 *     Space: O(k) per thread for heap storage
 *
 * THREAD SAFETY:
 *     Safe - parallelized over samples with thread-local workspace
 *
 * NUMERICAL NOTES:
 *     - Distance computed as sqrt(norm_i + norm_j - 2*dot)
 *     - Negative values from numerical error clamped to 0
 *     - Cauchy-Schwarz lower bound enables significant pruning
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void knn(
    const Sparse<T, IsCSR>& matrix, // Input sparse matrix [n_samples x n_features]
    Array<const T> norms_sq,        // Pre-computed squared norms [n_samples]
    Size k,                         // Number of neighbors to find
    Array<Index> out_indices,       // Output neighbor indices [n_samples * k]
    Array<T> out_distances          // Output distances [n_samples * k]
);

} // namespace scl::kernel::neighbors
