// =============================================================================
// FILE: scl/core/sort.h
// BRIEF: API reference for high-performance sorting via Google Highway VQSort
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>

namespace scl::sort {

// =============================================================================
// MODULE OVERVIEW
// =============================================================================

/* -----------------------------------------------------------------------------
 * MODULE: High-Performance Sorting
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     SIMD-accelerated sorting using Google Highway VQSort backend.
 *
 * PURPOSE:
 *     Provides sorting primitives optimized for numerical computing:
 *     - Single-array sorting (ascending/descending)
 *     - Key-value pair sorting (maintains correspondence)
 *     - Architecture-agnostic SIMD acceleration
 *
 * PERFORMANCE:
 *     - Automatically selects best SIMD instructions (AVX2/AVX-512/NEON)
 *     - Typical speedup: 2-5x vs std::sort for numerical types
 *     - Near-optimal cache utilization
 *     - Stable performance across data distributions
 *
 * SUPPORTED TYPES:
 *     - Integers: int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
 *     - Floats: float, double
 *     - Custom types with < operator (may not use SIMD path)
 *
 * THREAD SAFETY:
 *     All operations are unsafe - concurrent access causes race conditions
 * -------------------------------------------------------------------------- */

// =============================================================================
// BASIC SORTING
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: sort
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sort array in ascending order using SIMD-optimized VQSort.
 *
 * SIGNATURE:
 *     template <typename T>
 *     void sort(Array<T> data)
 *
 * PARAMETERS:
 *     T    [template] - Element type (must be sortable)
 *     data [in,out]   - Array to sort
 *
 * PRECONDITIONS:
 *     - data.ptr must be valid or nullptr (if data.len == 0)
 *
 * POSTCONDITIONS:
 *     - data is sorted in ascending order
 *     - data[i] <= data[i+1] for all valid i
 *
 * MUTABILITY:
 *     INPLACE - modifies data array
 *
 * ALGORITHM:
 *     Google Highway VQSort (vectorized quicksort variant)
 *
 * COMPLEXITY:
 *     Time:  O(n log n) average, O(n log n) worst-case
 *     Space: O(log n) stack for recursion
 *
 * THREAD SAFETY:
 *     Unsafe - concurrent access causes race conditions
 *
 * PERFORMANCE:
 *     - Uses SIMD partitioning and comparison
 *     - Optimized for modern CPU cache hierarchies
 *     - Best for arrays > 100 elements
 * -------------------------------------------------------------------------- */
template <typename T>
void sort(
    Array<T> data              // Array to sort
);

/* -----------------------------------------------------------------------------
 * FUNCTION: sort_descending
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sort array in descending order using SIMD-optimized VQSort.
 *
 * SIGNATURE:
 *     template <typename T>
 *     void sort_descending(Array<T> data)
 *
 * PARAMETERS:
 *     T    [template] - Element type (must be sortable)
 *     data [in,out]   - Array to sort
 *
 * PRECONDITIONS:
 *     - data.ptr must be valid or nullptr (if data.len == 0)
 *
 * POSTCONDITIONS:
 *     - data is sorted in descending order
 *     - data[i] >= data[i+1] for all valid i
 *
 * MUTABILITY:
 *     INPLACE - modifies data array
 *
 * ALGORITHM:
 *     Google Highway VQSort with descending comparator
 *
 * COMPLEXITY:
 *     Time:  O(n log n) average, O(n log n) worst-case
 *     Space: O(log n) stack for recursion
 *
 * THREAD SAFETY:
 *     Unsafe - concurrent access causes race conditions
 * -------------------------------------------------------------------------- */
template <typename T>
void sort_descending(
    Array<T> data              // Array to sort
);

// =============================================================================
// KEY-VALUE PAIR SORTING
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: sort_pairs
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sort key-value pairs by keys in ascending order.
 *
 * SIGNATURE:
 *     template <typename Key, typename Value>
 *     void sort_pairs(Array<Key> keys, Array<Value> values)
 *
 * PARAMETERS:
 *     Key    [template] - Key type (must be sortable)
 *     Value  [template] - Value type (arbitrary)
 *     keys   [in,out]   - Key array (modified)
 *     values [in,out]   - Value array (permuted to match keys)
 *
 * PRECONDITIONS:
 *     - keys.len == values.len
 *
 * POSTCONDITIONS:
 *     - keys is sorted in ascending order
 *     - values[i] corresponds to original keys[i]
 *     - (keys[i], values[i]) pairs maintain original association
 *
 * MUTABILITY:
 *     INPLACE - modifies both keys and values
 *
 * ALGORITHM:
 *     1. Allocate temporary buffer for (Key, Value) pairs
 *     2. Pack arrays into interleaved buffer using SIMD (if sizeof(Key) == sizeof(Value))
 *     3. Sort pairs using introsort with inline comparison
 *     4. Unpack sorted pairs back to separate arrays using SIMD
 *     5. Free temporary buffer
 *
 * BUFFER MANAGEMENT:
 *     - Small arrays (<= 8KB): stack allocation via alloca
 *     - Large arrays (> 8KB): heap allocation with aligned_alloc
 *
 * COMPLEXITY:
 *     Time:  O(n log n + n) - sort + pack/unpack overhead
 *     Space: O(n) temporary buffer
 *
 * THREAD SAFETY:
 *     Unsafe - concurrent access causes race conditions
 *
 * PERFORMANCE:
 *     - Uses SIMD for pack/unpack when sizeof(Key) == sizeof(Value)
 *     - Introsort prevents O(n^2) worst-case
 *     - 4-way unrolled SIMD loops for packing
 *     - Prefetching for large arrays
 *
 * WHEN TO USE:
 *     - Need to maintain correspondence between keys and values
 *     - Implementing argsort (keys = indices, values = data)
 *     - Sorting multiple related arrays by single key
 * -------------------------------------------------------------------------- */
template <typename Key, typename Value>
void sort_pairs(
    Array<Key> keys,           // Key array (modified)
    Array<Value> values        // Value array (permuted)
);

/* -----------------------------------------------------------------------------
 * FUNCTION: sort_pairs_descending
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sort key-value pairs by keys in descending order.
 *
 * SIGNATURE:
 *     template <typename Key, typename Value>
 *     void sort_pairs_descending(Array<Key> keys, Array<Value> values)
 *
 * PARAMETERS:
 *     Key    [template] - Key type (must be sortable)
 *     Value  [template] - Value type (arbitrary)
 *     keys   [in,out]   - Key array (modified)
 *     values [in,out]   - Value array (permuted to match keys)
 *
 * PRECONDITIONS:
 *     - keys.len == values.len
 *
 * POSTCONDITIONS:
 *     - keys is sorted in descending order
 *     - values[i] corresponds to original keys[i]
 *
 * MUTABILITY:
 *     INPLACE - modifies both keys and values
 *
 * ALGORITHM:
 *     Same as sort_pairs but with descending comparator (a.k > b.k)
 *
 * COMPLEXITY:
 *     Time:  O(n log n + n)
 *     Space: O(n) temporary buffer
 *
 * THREAD SAFETY:
 *     Unsafe - concurrent access causes race conditions
 *
 * WHEN TO USE:
 *     - Finding top-K elements with associated data
 *     - Reverse sorting with value tracking
 * -------------------------------------------------------------------------- */
template <typename Key, typename Value>
void sort_pairs_descending(
    Array<Key> keys,           // Key array (modified)
    Array<Value> values        // Value array (permuted)
);

// =============================================================================
// CONVENIENCE ALIASES
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: sort_real
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sort Real array in ascending order.
 *
 * SIGNATURE:
 *     void sort_real(Array<Real> data)
 *
 * PARAMETERS:
 *     data [in,out] - Real array to sort
 *
 * POSTCONDITIONS:
 *     - data is sorted in ascending order
 *
 * NOTES:
 *     Convenience wrapper around sort<Real>
 * -------------------------------------------------------------------------- */
void sort_real(
    Array<Real> data           // Real array
);

/* -----------------------------------------------------------------------------
 * FUNCTION: sort_index
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sort Index array in ascending order.
 *
 * SIGNATURE:
 *     void sort_index(Array<Index> data)
 *
 * PARAMETERS:
 *     data [in,out] - Index array to sort
 *
 * POSTCONDITIONS:
 *     - data is sorted in ascending order
 *
 * NOTES:
 *     Convenience wrapper around sort<Index>
 * -------------------------------------------------------------------------- */
void sort_index(
    Array<Index> data          // Index array
);

// =============================================================================
// IMPLEMENTATION DETAILS
// =============================================================================

namespace detail {

/* -----------------------------------------------------------------------------
 * CONSTANT: STACK_BUFFER_THRESHOLD
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Maximum buffer size for stack allocation in sort_pairs.
 *
 * VALUE:
 *     8192 bytes (8 KB)
 *
 * RATIONALE:
 *     - Avoids stack overflow for large arrays
 *     - Keeps hot data in L1 cache for small arrays
 *     - Typical stack size is 1-8 MB
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * STRUCT: RawType
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Maps sizeof(T) to unsigned integer type for bitwise operations.
 *
 * SPECIALIZATIONS:
 *     RawType<1>::type = uint8_t
 *     RawType<2>::type = uint16_t
 *     RawType<4>::type = uint32_t
 *     RawType<8>::type = uint64_t
 *     RawType<16>::type = __uint128_t (if available)
 *
 * PURPOSE:
 *     Enables SIMD pack/unpack via type punning
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: pack_interleaved
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Pack separate key/value arrays into interleaved buffer using SIMD.
 *
 * ALGORITHM:
 *     1. Reinterpret keys/values as RawT* (uint32_t, uint64_t, etc.)
 *     2. 4-way unrolled SIMD loop: Load keys[i], values[i] -> Store interleaved
 *     3. Prefetch ahead by 2*N elements
 *     4. Scalar loop for tail elements
 *
 * PERFORMANCE:
 *     - Uses StoreInterleaved2 Highway intrinsic
 *     - 4-way unrolling improves instruction throughput
 *     - Prefetching hides memory latency
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: unpack_interleaved
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Unpack interleaved buffer back to separate key/value arrays using SIMD.
 *
 * ALGORITHM:
 *     1. Reinterpret source as RawT*
 *     2. SIMD loop: LoadInterleaved2 -> Store keys[i], values[i]
 *     3. Prefetch ahead by 2*N elements
 *     4. Scalar loop for tail elements
 *
 * PERFORMANCE:
 *     - Uses LoadInterleaved2 Highway intrinsic
 *     - Symmetric to pack_interleaved
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: insertion_sort
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Classic insertion sort for small arrays.
 *
 * THRESHOLD:
 *     Used for arrays <= 16 elements
 *
 * COMPLEXITY:
 *     Time:  O(n^2) worst-case, O(n) best-case (nearly sorted)
 *     Space: O(1)
 *
 * RATIONALE:
 *     - Faster than quicksort for small n due to low overhead
 *     - Good cache locality
 *     - Adaptive to presorted data
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: partition
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Partition step for quicksort using median-of-three pivot selection.
 *
 * ALGORITHM:
 *     1. Choose pivot as median of low, mid, high
 *     2. Two-pointer partition: move elements < pivot left, >= pivot right
 *     3. Return final pivot position
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: introsort_impl
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Introspective sort: quicksort with heapsort fallback.
 *
 * ALGORITHM:
 *     1. If size <= 16: return (final insertion_sort handles it)
 *     2. If recursion depth limit reached: switch to heapsort
 *     3. Otherwise: partition and recurse on smaller half, iterate on larger
 *
 * DEPTH LIMIT:
 *     2 * log2(n) - prevents O(n^2) worst-case
 *
 * COMPLEXITY:
 *     Time:  O(n log n) guaranteed
 *     Space: O(log n) stack
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: sort_pairs_impl
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Top-level dispatcher for pair sorting.
 *
 * ALGORITHM:
 *     1. If n <= 1: return
 *     2. If n <= 16: use insertion_sort
 *     3. Otherwise: introsort_impl + final insertion_sort
 *
 * NOTES:
 *     Final insertion_sort handles small unsorted sublists left by introsort
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * CLASS: BufferManager
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     RAII wrapper for automatic stack/heap buffer management.
 *
 * MEMBERS:
 *     ptr     [T*]   - Pointer to buffer
 *     on_heap [bool] - True if heap-allocated
 *
 * CONSTRUCTOR:
 *     BufferManager(size_t n)
 *         If n * sizeof(T) <= 8KB: allocate on stack via alloca
 *         Otherwise: allocate on heap via aligned_alloc
 *
 * DESTRUCTOR:
 *     ~BufferManager()
 *         If on_heap: call free(ptr)
 *         Otherwise: no-op (stack memory auto-freed)
 *
 * THREAD SAFETY:
 *     Safe - each instance owns independent memory
 *
 * RATIONALE:
 *     - Avoids heap allocation overhead for small arrays
 *     - Prevents stack overflow for large arrays
 *     - Exception-safe RAII cleanup
 * -------------------------------------------------------------------------- */

} // namespace detail

} // namespace scl::sort
