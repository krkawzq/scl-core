// =============================================================================
// FILE: scl/kernel/slice.h
// BRIEF: API reference for sparse matrix slicing kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::slice {

/* -----------------------------------------------------------------------------
 * CONFIGURATION CONSTANTS
 * -----------------------------------------------------------------------------
 * PARALLEL_THRESHOLD_ROWS - Min rows for parallel processing (512)
 * PARALLEL_THRESHOLD_NNZ  - Min nnz for parallel processing (10000)
 * MEMCPY_THRESHOLD        - Min elements for memcpy vs loop (8)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: inspect_slice_primary
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Count total non-zeros in selected primary dimension slices.
 *
 * PARAMETERS:
 *     matrix       [in] Sparse matrix to slice
 *     keep_indices [in] Indices of primary dimension elements to keep
 *
 * PRECONDITIONS:
 *     - All indices in keep_indices in range [0, primary_dim)
 *
 * POSTCONDITIONS:
 *     - Returns sum of row lengths for selected indices
 *
 * RETURNS:
 *     Total number of non-zeros in selected slices
 *
 * ALGORITHM:
 *     Parallel reduction over keep_indices using parallel_reduce_nnz
 *
 * COMPLEXITY:
 *     Time:  O(n_keep / n_threads)
 *     Space: O(n_threads) for partial sums
 *
 * THREAD SAFETY:
 *     Safe - read-only parallel reduction
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Index inspect_slice_primary(
    const Sparse<T, IsCSR>& matrix,  // Input sparse matrix
    Array<const Index> keep_indices   // Indices to keep
);

/* -----------------------------------------------------------------------------
 * FUNCTION: materialize_slice_primary
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Copy selected primary slices to pre-allocated output arrays.
 *
 * PARAMETERS:
 *     matrix       [in]  Source sparse matrix
 *     keep_indices [in]  Indices of rows/cols to keep
 *     out_data     [out] Output values array
 *     out_indices  [out] Output column/row indices array
 *     out_indptr   [out] Output row/col pointer array
 *
 * PRECONDITIONS:
 *     - out_data.len >= inspect_slice_primary result
 *     - out_indices.len >= inspect_slice_primary result
 *     - out_indptr.len >= keep_indices.len + 1
 *
 * POSTCONDITIONS:
 *     - out_data contains copied values in order
 *     - out_indices contains copied indices (unchanged)
 *     - out_indptr[i] = start of i-th selected row
 *
 * ALGORITHM:
 *     1. Sequential scan to build out_indptr
 *     2. Parallel copy of data and indices using fast_copy_with_prefetch
 *
 * COMPLEXITY:
 *     Time:  O(nnz_output / n_threads + n_keep)
 *     Space: O(1) beyond output
 *
 * THREAD SAFETY:
 *     Safe - parallel copy to disjoint output regions
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void materialize_slice_primary(
    const Sparse<T, IsCSR>& matrix,  // Source matrix
    Array<const Index> keep_indices,  // Indices to keep
    Array<T> out_data,                // Output values
    Array<Index> out_indices,         // Output indices
    Array<Index> out_indptr           // Output pointers
);

/* -----------------------------------------------------------------------------
 * FUNCTION: slice_primary
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Create new sparse matrix containing selected primary slices.
 *
 * PARAMETERS:
 *     matrix       [in] Source sparse matrix
 *     keep_indices [in] Indices of rows (CSR) or cols (CSC) to keep
 *
 * PRECONDITIONS:
 *     - All indices in range [0, primary_dim)
 *
 * POSTCONDITIONS:
 *     - Result contains only selected rows/cols
 *     - Column/row indices unchanged (secondary dim preserved)
 *     - Order matches keep_indices order
 *
 * RETURNS:
 *     New sparse matrix with selected slices
 *
 * ALGORITHM:
 *     1. inspect_slice_primary to count output nnz
 *     2. Allocate output arrays
 *     3. materialize_slice_primary to copy data
 *     4. Wrap as new Sparse matrix
 *
 * COMPLEXITY:
 *     Time:  O(nnz_output / n_threads + n_keep)
 *     Space: O(nnz_output) for result
 *
 * THREAD SAFETY:
 *     Safe - uses parallel materialize
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Sparse<T, IsCSR> slice_primary(
    const Sparse<T, IsCSR>& matrix,  // Source matrix
    Array<const Index> keep_indices   // Indices to keep
);

/* -----------------------------------------------------------------------------
 * FUNCTION: inspect_filter_secondary
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Count non-zeros after filtering by secondary dimension mask.
 *
 * PARAMETERS:
 *     matrix [in] Sparse matrix to filter
 *     mask   [in] Boolean mask for secondary dimension (1 = keep)
 *
 * PRECONDITIONS:
 *     - mask.len >= secondary_dim
 *     - mask values are 0 or 1
 *
 * POSTCONDITIONS:
 *     - Returns count of elements where mask[index] == 1
 *
 * RETURNS:
 *     Total non-zeros after filtering
 *
 * ALGORITHM:
 *     Parallel reduction using count_masked_fast (8-way unrolled)
 *
 * COMPLEXITY:
 *     Time:  O(nnz / n_threads)
 *     Space: O(n_threads) for partial sums
 *
 * THREAD SAFETY:
 *     Safe - read-only parallel reduction
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Index inspect_filter_secondary(
    const Sparse<T, IsCSR>& matrix,  // Input sparse matrix
    Array<const uint8_t> mask         // Keep mask for secondary dim
);

/* -----------------------------------------------------------------------------
 * FUNCTION: materialize_filter_secondary
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Copy elements passing secondary mask to pre-allocated output.
 *
 * PARAMETERS:
 *     matrix      [in]  Source sparse matrix
 *     mask        [in]  Boolean mask for secondary dimension
 *     new_indices [in]  Mapping from old to new secondary indices
 *     out_data    [out] Output values
 *     out_indices [out] Output indices (remapped)
 *     out_indptr  [out] Output row pointers
 *
 * PRECONDITIONS:
 *     - new_indices built via build_index_mapping
 *     - Output arrays sized per inspect_filter_secondary
 *
 * POSTCONDITIONS:
 *     - out_data contains values where mask[old_index] == 1
 *     - out_indices contains remapped indices via new_indices
 *     - out_indptr contains cumulative counts
 *
 * COMPLEXITY:
 *     Time:  O(nnz / n_threads + primary_dim)
 *     Space: O(1) beyond output
 *
 * THREAD SAFETY:
 *     Safe - parallel over primary dimension
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void materialize_filter_secondary(
    const Sparse<T, IsCSR>& matrix,       // Source matrix
    Array<const uint8_t> mask,             // Keep mask
    Array<const Index> new_indices,        // Old to new index mapping
    Array<T> out_data,                     // Output values
    Array<Index> out_indices,              // Output indices
    Array<Index> out_indptr                // Output pointers
);

/* -----------------------------------------------------------------------------
 * FUNCTION: filter_secondary
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Create new sparse matrix filtering by secondary dimension mask.
 *
 * PARAMETERS:
 *     matrix [in] Source sparse matrix
 *     mask   [in] Boolean mask for columns (CSR) or rows (CSC)
 *
 * PRECONDITIONS:
 *     - mask.len >= secondary_dim
 *     - mask values are 0 or 1
 *
 * POSTCONDITIONS:
 *     - Result secondary_dim = count of 1s in mask
 *     - Only elements with mask[index] == 1 retained
 *     - Indices remapped to compact range [0, new_secondary_dim)
 *
 * RETURNS:
 *     New sparse matrix with filtered secondary dimension
 *
 * ALGORITHM:
 *     1. Build index mapping (old -> new indices)
 *     2. inspect_filter_secondary to count output nnz
 *     3. Allocate output arrays
 *     4. materialize_filter_secondary to copy and remap
 *
 * COMPLEXITY:
 *     Time:  O(nnz / n_threads + secondary_dim)
 *     Space: O(nnz_output + secondary_dim)
 *
 * THREAD SAFETY:
 *     Safe - uses parallel materialize
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Sparse<T, IsCSR> filter_secondary(
    const Sparse<T, IsCSR>& matrix,  // Source matrix
    Array<const uint8_t> mask         // Keep mask for secondary dim
);

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::count_masked_fast
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Count elements passing mask with 8-way unrolled loop.
 *
 * ALGORITHM:
 *     8-way scalar unroll:
 *         count += mask[indices[k+0]] + ... + mask[indices[k+7]]
 *
 * NOTE:
 *     Indirect access mask[indices[k]] prevents SIMD gather.
 *     8-way scalar unroll provides best ILP for this pattern.
 *
 * COMPLEXITY:
 *     Time:  O(len)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::build_index_mapping
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Build old-to-new index mapping from boolean mask.
 *
 * PARAMETERS:
 *     mask        [in]  Boolean mask (0 or 1)
 *     new_indices [out] Mapping array (same size as mask)
 *     size        [in]  Size of mask
 *
 * POSTCONDITIONS:
 *     - new_indices[i] = new compact index if mask[i] == 1
 *     - new_indices[i] = -1 if mask[i] == 0
 *     - Returns count of 1s in mask
 *
 * COMPLEXITY:
 *     Time:  O(size)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */

} // namespace scl::kernel::slice
