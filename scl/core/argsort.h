// =============================================================================
// FILE: scl/core/argsort.h
// BRIEF: API reference for argument sorting (returns sorted indices)
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>

namespace scl::sort {

// =============================================================================
// MODULE OVERVIEW
// =============================================================================

/* -----------------------------------------------------------------------------
 * MODULE: Argument Sorting
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Returns permutation indices that would sort an array, rather than
 *     sorting the array itself.
 *
 * PURPOSE:
 *     Argsort is fundamental for:
 *     - Top-K selection (e.g., highly variable genes)
 *     - Ranking and percentile calculations
 *     - Indirect sorting of multiple related arrays
 *     - Preserving original data while obtaining sorted order
 *
 * VARIANTS:
 *     1. In-Place: Modifies keys array, fast for temporary arrays
 *     2. Buffered: Preserves keys array, requires external buffer
 *     3. Indirect: Never modifies keys, uses lambda comparisons (slowest)
 *
 * PERFORMANCE:
 *     - SIMD-optimized index initialization
 *     - Leverages Highway VQSort for key-value sorting
 *     - Typical speedup: 5-10x vs std::sort with lambda
 * -------------------------------------------------------------------------- */

// =============================================================================
// IN-PLACE ARGSORT
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: argsort_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sort keys and return corresponding indices (ascending order).
 *
 * SIGNATURE:
 *     template <typename T>
 *     void argsort_inplace(Array<T> keys, Array<Index> indices)
 *
 * PARAMETERS:
 *     T       [template] - Key type (must be sortable)
 *     keys    [in,out]   - Array to sort (modified)
 *     indices [out]      - Output permutation indices
 *
 * PRECONDITIONS:
 *     - keys.len == indices.len
 *     - indices buffer is allocated
 *
 * POSTCONDITIONS:
 *     - keys is sorted in ascending order
 *     - indices[i] contains original position of keys[i]
 *     - Applying indices to original data yields sorted order
 *
 * MUTABILITY:
 *     INPLACE - modifies keys array
 *
 * ALGORITHM:
 *     1. Initialize indices to [0, 1, 2, ..., n-1] using SIMD
 *     2. Sort (keys, indices) pairs by keys
 *     3. Result: keys sorted, indices contain original positions
 *
 * COMPLEXITY:
 *     Time:  O(n log n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - concurrent access causes race conditions
 *
 * WHEN TO USE:
 *     - keys array is temporary and can be modified
 *     - Need maximum performance
 *     - keys won't be needed in original order
 * -------------------------------------------------------------------------- */
template <typename T>
void argsort_inplace(
    Array<T> keys,             // Keys to sort (modified)
    Array<Index> indices       // Output indices
);

/* -----------------------------------------------------------------------------
 * FUNCTION: argsort_inplace_descending
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sort keys and return corresponding indices (descending order).
 *
 * SIGNATURE:
 *     template <typename T>
 *     void argsort_inplace_descending(Array<T> keys, Array<Index> indices)
 *
 * PARAMETERS:
 *     T       [template] - Key type (must be sortable)
 *     keys    [in,out]   - Array to sort (modified)
 *     indices [out]      - Output permutation indices
 *
 * PRECONDITIONS:
 *     - keys.len == indices.len
 *
 * POSTCONDITIONS:
 *     - keys is sorted in descending order
 *     - indices[i] contains original position of keys[i]
 *
 * MUTABILITY:
 *     INPLACE - modifies keys array
 *
 * COMPLEXITY:
 *     Time:  O(n log n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - concurrent access causes race conditions
 *
 * USE CASE:
 *     Finding top-K largest elements (e.g., most variable genes)
 * -------------------------------------------------------------------------- */
template <typename T>
void argsort_inplace_descending(
    Array<T> keys,             // Keys to sort (modified)
    Array<Index> indices       // Output indices
);

// =============================================================================
// BUFFERED ARGSORT
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: argsort_buffered
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sort without modifying original keys (uses external buffer).
 *
 * SIGNATURE:
 *     template <typename T>
 *     void argsort_buffered(
 *         Array<const T> keys,
 *         Array<Index> indices,
 *         Array<Byte> buffer
 *     )
 *
 * PARAMETERS:
 *     T       [template] - Key type
 *     keys    [in]       - Original keys (preserved)
 *     indices [out]      - Output permutation indices
 *     buffer  [in]       - Scratch space for key copy
 *
 * PRECONDITIONS:
 *     - keys.len == indices.len
 *     - buffer.len >= keys.len * sizeof(T)
 *     - buffer is properly aligned for type T
 *
 * POSTCONDITIONS:
 *     - keys is unchanged
 *     - indices contains permutation to sort keys
 *     - buffer contains sorted copy of keys (unspecified state)
 *
 * MUTABILITY:
 *     CONST on keys, writes to buffer and indices
 *
 * ALGORITHM:
 *     1. Copy keys to buffer using fast memcpy
 *     2. Call argsort_inplace on buffer copy
 *     3. Original keys remain unchanged
 *
 * COMPLEXITY:
 *     Time:  O(n log n + n) - sort + copy
 *     Space: O(n) - buffer space
 *
 * THREAD SAFETY:
 *     Unsafe - buffer and indices must not be accessed concurrently
 *
 * WHEN TO USE:
 *     - Need to preserve original keys
 *     - Have pre-allocated buffer available
 *     - Avoiding heap allocation
 *
 * BUFFER MANAGEMENT:
 *     Use scl::memory::aligned_alloc() or AlignedBuffer for buffer allocation
 * -------------------------------------------------------------------------- */
template <typename T>
void argsort_buffered(
    Array<const T> keys,       // Original keys (preserved)
    Array<Index> indices,      // Output indices
    Array<Byte> buffer         // Scratch buffer
);

/* -----------------------------------------------------------------------------
 * FUNCTION: argsort_buffered_descending
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sort descending without modifying original keys.
 *
 * SIGNATURE:
 *     template <typename T>
 *     void argsort_buffered_descending(
 *         Array<const T> keys,
 *         Array<Index> indices,
 *         Array<Byte> buffer
 *     )
 *
 * PARAMETERS:
 *     T       [template] - Key type
 *     keys    [in]       - Original keys (preserved)
 *     indices [out]      - Output permutation indices
 *     buffer  [in]       - Scratch space for key copy
 *
 * PRECONDITIONS:
 *     - keys.len == indices.len
 *     - buffer.len >= keys.len * sizeof(T)
 *
 * POSTCONDITIONS:
 *     - keys is unchanged
 *     - indices contains permutation for descending sort
 *
 * MUTABILITY:
 *     CONST on keys
 *
 * COMPLEXITY:
 *     Time:  O(n log n + n)
 *     Space: O(n)
 *
 * THREAD SAFETY:
 *     Unsafe
 * -------------------------------------------------------------------------- */
template <typename T>
void argsort_buffered_descending(
    Array<const T> keys,       // Original keys (preserved)
    Array<Index> indices,      // Output indices
    Array<Byte> buffer         // Scratch buffer
);

// =============================================================================
// INDIRECT ARGSORT
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: argsort_indirect
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Pure index-based sorting using lambda comparisons.
 *
 * SIGNATURE:
 *     template <typename T>
 *     void argsort_indirect(Array<const T> keys, Array<Index> indices)
 *
 * PARAMETERS:
 *     T       [template] - Key type
 *     keys    [in]       - Original keys (never modified)
 *     indices [out]      - Output permutation indices
 *
 * PRECONDITIONS:
 *     - keys.len == indices.len
 *
 * POSTCONDITIONS:
 *     - keys is unchanged
 *     - indices sorted such that keys[indices[i]] is ascending
 *
 * MUTABILITY:
 *     CONST - never modifies keys
 *
 * ALGORITHM:
 *     1. Initialize indices to [0, 1, ..., n-1] using SIMD
 *     2. std::sort(indices) with lambda: keys[a] < keys[b]
 *     3. All comparisons indirect through indices
 *
 * COMPLEXITY:
 *     Time:  O(n log n) - with higher constant factor
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - if keys is not modified concurrently
 *
 * PERFORMANCE:
 *     - Slower than buffered variants (2-3x overhead)
 *     - Every comparison is indirect (cache-unfriendly)
 *     - No SIMD acceleration possible
 *
 * WHEN TO USE:
 *     - Cannot allocate buffer
 *     - keys must never be modified
 *     - Sorting small arrays (< 1000 elements)
 *     - Simplicity over performance
 *
 * WHEN NOT TO USE:
 *     - Large arrays (prefer buffered variant)
 *     - Performance-critical code
 *     - keys can be temporary
 * -------------------------------------------------------------------------- */
template <typename T>
void argsort_indirect(
    Array<const T> keys,       // Original keys (never modified)
    Array<Index> indices       // Output indices
);

} // namespace scl::sort
