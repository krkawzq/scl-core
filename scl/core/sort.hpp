#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/simd.hpp" // Setup HWY environment

// Highway Sort Contrib
// Note: This header defines VQSortStatic inside hwy::HWY_NAMESPACE
#include <hwy/contrib/sort/vqsort-inl.h>

// =============================================================================
/// @file sort.hpp
/// @brief SCL High-Performance Sorting (via Google Highway VQSort)
///
/// Wraps the "Static" version of VQSort to provide vectorized QuickSort.
///
/// @section Why Static?
/// We use `VQSortStatic` instead of `hwy::VQSort` to avoid runtime dispatch overhead.
/// The instruction set is already determined by the compiler flags (-march=native)
/// consistent with the rest of SCL.
///
/// @section Performance
/// VQSort is typically 10x-20x faster than std::sort for primitive types
/// by utilizing SIMD instructions (AVX2/AVX-512/NEON) for partitioning.
///
// =============================================================================

namespace scl::sort {

// =============================================================================
// 1. Core Sorting wrappers
// =============================================================================

/// @brief Sort a generic span in ascending order (SIMD Optimized).
///
/// @tparam T Data type (must be supported by Highway, e.g., Real, Index, int, etc.)
/// @param data Mutable view of the array to sort.
template <typename T>
SCL_FORCE_INLINE void sort(MutableSpan<T> data) {
    // 2. Call Static VQSort
    // SortAscending() is a stateless functor provided by Highway
    hwy::HWY_NAMESPACE::VQSortStatic(data.ptr, data.size, hwy::SortAscending());
}

/// @brief Sort a generic span in descending order (SIMD Optimized).
template <typename T>
SCL_FORCE_INLINE void sort_descending(MutableSpan<T> data) {
    hwy::HWY_NAMESPACE::VQSortStatic(data.ptr, data.size, hwy::SortDescending());
}

// =============================================================================
// 2. Key-Value Sorting (Pairs)
// =============================================================================

/// @brief Sort keys and move values accordingly (Ascending).
///
/// Useful for keeping track of indices (Values) while sorting data (Keys).
/// Requires sizeof(Key) == sizeof(Value) for optimal SIMD performance.
template <typename Key, typename Value>
SCL_FORCE_INLINE void sort_pairs(MutableSpan<Key> keys, MutableSpan<Value> values) {
    SCL_ASSERT(keys.size == values.size, "Sort keys and values must have same size");

    const hwy::HWY_NAMESPACE::ScalableTag<Key> d;
    
    // VQSortStatic for pairs
    hwy::HWY_NAMESPACE::VQSortStatic(
        d, 
        keys.ptr, 
        values.ptr, 
        keys.size, 
        hwy::SortAscending()
    );
}

/// @brief Sort keys and move values accordingly (Descending).
///
/// Used by argsort_descending (e.g. finding Top-K highest expression genes).
template <typename Key, typename Value>
SCL_FORCE_INLINE void sort_pairs_descending(MutableSpan<Key> keys, MutableSpan<Value> values) {
    SCL_ASSERT(keys.size == values.size, "Sort keys and values must have same size");

    const hwy::HWY_NAMESPACE::ScalableTag<Key> d;
    
    // VQSortStatic for pairs with Descending comparator
    hwy::HWY_NAMESPACE::VQSortStatic(
        d, 
        keys.ptr, 
        values.ptr, 
        keys.size, 
        hwy::SortDescending()
    );
}

// =============================================================================
// 2. Convenience Aliases
// =============================================================================

/// @brief Sort Real numbers (float32/64) ascending.
SCL_FORCE_INLINE void sort_real(MutableRealSpan data) {
    sort<Real>(data);
}

/// @brief Sort Indices (int64) ascending.
SCL_FORCE_INLINE void sort_index(MutableIndexSpan data) {
    sort<Index>(data);
}

} // namespace scl::sort
