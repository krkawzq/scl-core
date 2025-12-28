// =============================================================================
// FILE: scl/kernel/bbknn.h
// BRIEF: API reference for Batch Balanced KNN kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::bbknn {

/* -----------------------------------------------------------------------------
 * CONFIGURATION CONSTANTS
 * -----------------------------------------------------------------------------
 * CHUNK_SIZE         - Processing chunk size for parallelization (64)
 * PREFETCH_DISTANCE  - Elements to prefetch ahead (8)
 * MIN_SAMPLES_PARALLEL - Minimum samples for parallel processing (128)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * STRUCT: BatchGroups
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Memory-efficient storage for batch-grouped sample indices.
 *
 * FIELDS:
 *     indices    - All sample indices concatenated contiguously
 *     offsets    - Start offset for each batch, size = n_batches + 1
 *     n_batches  - Number of batches
 *     total_size - Total number of valid samples
 *
 * METHODS:
 *     batch_size(b)  - Returns number of samples in batch b
 *     batch_data(b)  - Returns pointer to indices for batch b
 *
 * MEMORY LAYOUT:
 *     indices: [batch0_samples... | batch1_samples... | ...]
 *     offsets: [0, len0, len0+len1, ...]
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: build_batch_groups
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Build batch-grouped index structure from batch labels.
 *
 * PARAMETERS:
 *     batch_labels [in]  Batch label for each sample, size = n_samples
 *     n_batches    [in]  Number of distinct batches
 *     out          [out] Output BatchGroups structure
 *
 * PRECONDITIONS:
 *     - batch_labels[i] in range [0, n_batches) or negative (ignored)
 *     - n_batches > 0
 *
 * POSTCONDITIONS:
 *     - out.indices contains sample indices grouped by batch
 *     - out.offsets[b] is start offset for batch b
 *     - out.offsets[n_batches] == total valid samples
 *     - Memory allocated via scl::memory::aligned_alloc
 *
 * ALGORITHM:
 *     1. Count samples per batch (single pass)
 *     2. Prefix sum to compute offsets
 *     3. Second pass to fill indices array
 *
 * COMPLEXITY:
 *     Time:  O(n_samples)
 *     Space: O(n_samples + n_batches)
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
void build_batch_groups(
    Array<const int32_t> batch_labels,  // Batch label per sample
    Size n_batches,                      // Number of batches
    BatchGroups& out                     // Output structure
);

/* -----------------------------------------------------------------------------
 * FUNCTION: free_batch_groups
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Free memory allocated by build_batch_groups.
 *
 * PARAMETERS:
 *     groups [in,out] BatchGroups to free
 *
 * POSTCONDITIONS:
 *     - groups.indices and groups.offsets are freed
 *     - Pointers set to nullptr
 *
 * MUTABILITY:
 *     INPLACE - modifies groups structure
 * -------------------------------------------------------------------------- */
void free_batch_groups(
    BatchGroups& groups                  // Structure to free
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_norms
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Precompute squared L2 norms for all rows of a sparse matrix.
 *
 * PARAMETERS:
 *     matrix   [in]  Sparse matrix, shape (n_samples, n_features)
 *     norms_sq [out] Output array, size = n_samples
 *
 * PRECONDITIONS:
 *     - norms_sq.len >= matrix.primary_dim()
 *
 * POSTCONDITIONS:
 *     - norms_sq[i] = sum(matrix[i,:]^2)
 *
 * ALGORITHM:
 *     Parallel over rows using scl::vectorize::sum_squared
 *
 * COMPLEXITY:
 *     Time:  O(nnz / n_threads)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - parallel over independent rows
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_norms(
    const Sparse<T, IsCSR>& matrix,      // Input sparse matrix
    Array<T> norms_sq                     // Output squared norms [n_samples]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: bbknn
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Batch Balanced K-Nearest Neighbors search.
 *     Finds k nearest neighbors from EACH batch for every sample.
 *
 * PARAMETERS:
 *     matrix        [in]  Sparse matrix, shape (n_samples, n_features)
 *     batch_labels  [in]  Batch label per sample, size = n_samples
 *     n_batches     [in]  Number of distinct batches
 *     k             [in]  Neighbors per batch
 *     out_indices   [out] Neighbor indices, size = n_samples * n_batches * k
 *     out_distances [out] Neighbor distances, size = n_samples * n_batches * k
 *     norms_sq      [in]  Precomputed squared norms (optional overload)
 *
 * PRECONDITIONS:
 *     - batch_labels.len == matrix.primary_dim()
 *     - out_indices.len >= n_samples * n_batches * k
 *     - out_distances.len >= n_samples * n_batches * k
 *     - norms_sq.len >= n_samples (if provided)
 *
 * POSTCONDITIONS:
 *     - For sample i, batch b:
 *       - out_indices[i * n_batches * k + b * k + j] = j-th nearest neighbor from batch b
 *       - out_distances[...] = corresponding Euclidean distance
 *     - Neighbors sorted by distance (ascending) within each batch
 *     - If fewer than k neighbors in batch: index = -1, distance = infinity
 *
 * OUTPUT LAYOUT:
 *     For sample i, batch b, neighbor j:
 *         offset = i * (n_batches * k) + b * k + j
 *
 * ALGORITHM:
 *     Key optimizations:
 *     1. Batch-grouped processing for memory locality
 *     2. Fixed-size max-heap with manual sift operations
 *     3. Cauchy-Schwarz lower bound pruning:
 *        min_dist_sq = norm_a^2 + norm_b^2 - 2*sqrt(norm_a^2 * norm_b^2)
 *        Skip candidate if min_dist_sq >= current k-th distance
 *     4. Sparse dot product with 8/4-way skip optimization
 *     5. Euclidean distance: dist^2 = norm_a^2 + norm_b^2 - 2*dot(a,b)
 *
 * COMPLEXITY:
 *     Time:  O(n_samples * avg_batch_size * (nnz_per_row + k*log(k)))
 *     Space: O(n_threads * n_batches * k) for heaps
 *
 * THREAD SAFETY:
 *     Safe - uses thread-local heap storage
 *
 * NUMERICAL NOTES:
 *     - Distance clamped to >= 0 for numerical stability
 *     - Output is Euclidean distance (not squared)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void bbknn(
    const Sparse<T, IsCSR>& matrix,      // Input sparse matrix [n_samples x n_features]
    Array<const int32_t> batch_labels,   // Batch label per sample [n_samples]
    Size n_batches,                       // Number of batches
    Size k,                               // Neighbors per batch
    Array<Index> out_indices,             // Output indices [n_samples * n_batches * k]
    Array<T> out_distances,               // Output distances [n_samples * n_batches * k]
    Array<const T> norms_sq               // Precomputed squared norms [n_samples]
);

template <typename T, bool IsCSR>
void bbknn(
    const Sparse<T, IsCSR>& matrix,      // Input sparse matrix
    Array<const int32_t> batch_labels,   // Batch labels
    Size n_batches,                       // Number of batches
    Size k,                               // Neighbors per batch
    Array<Index> out_indices,             // Output indices
    Array<T> out_distances                // Output distances
);

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::KHeap
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Fixed-capacity max-heap for k-nearest neighbor tracking.
 *
 * FIELDS:
 *     data   - Array of (distance_squared, index) entries
 *     k      - Maximum capacity
 *     count  - Current number of entries
 *
 * METHODS:
 *     init(storage, capacity) - Initialize with external storage
 *     clear()                 - Reset to empty
 *     max_dist_sq()           - Return largest distance in heap
 *     try_insert(dist, idx)   - Insert if better than worst
 *     extract_sorted(...)     - Extract k neighbors sorted by distance
 *
 * ALGORITHM:
 *     Max-heap property: root has largest distance.
 *     try_insert:
 *         - If not full: insert and sift up
 *         - If full and dist < root: replace root and sift down
 *     extract_sorted: heapsort to ascending order
 *
 * COMPLEXITY:
 *     try_insert:     O(log k)
 *     extract_sorted: O(k log k)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::sparse_dot
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute dot product of two sparse vectors with skip optimization.
 *
 * ALGORITHM:
 *     1. O(1) range disjointness check
 *     2. 8-way skip: if block of 8 elements in A all < first of B, skip block
 *     3. 4-way skip: similar for blocks of 4
 *     4. Linear merge with prefetch for overlapping region
 *
 * COMPLEXITY:
 *     Time:  O(len_a + len_b) worst case, often much better with skip
 *     Space: O(1)
 * -------------------------------------------------------------------------- */

} // namespace scl::kernel::bbknn
