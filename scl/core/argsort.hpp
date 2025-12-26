#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/memory.hpp" // New memory component
#include <algorithm>

namespace scl::sort {

namespace detail {

    /// @brief Serial Iota with Aggressive SIMD Unrolling.
    /// Fills [0, 1, 2, ... N].
    /// Optimized for throughput using Instruction Level Parallelism (ILP).
    SCL_FORCE_INLINE void iota_serial(Array<Index> indices) {
        namespace s = scl::simd;
        
        const s::IndexTag d;
        const size_t N = indices.len;
        const size_t lanes = s::lanes();
        
        // Prepare steps for 4-way unrolling
        // Step = [L, L, ...]
        // BlockStep = [4L, 4L, ...]
        const auto v_step_1 = s::Set(d, static_cast<scl::Index>(lanes));
        const auto v_step_4 = s::Set(d, static_cast<scl::Index>(4 * lanes));
        
        // Initial vectors:
        // v0: [0, 1, ...]
        // v1: [L, L+1, ...]
        // v2: [2L, 2L+1, ...]
        // v3: [3L, 3L+1, ...]
        auto v0 = s::Iota(d, 0);
        auto v1 = s::Add(v0, v_step_1);
        auto v2 = s::Add(v1, v_step_1);
        auto v3 = s::Add(v2, v_step_1);

        size_t i = 0;

        // 1. Unrolled SIMD Loop (4x)
        // Processes 4 * lanes elements per iteration.
        // This keeps the CPU execution ports saturated.
        for (; i + 4 * lanes <= N; i += 4 * lanes) {
            s::Store(v0, d, indices.ptr + i);
            s::Store(v1, d, indices.ptr + i + lanes);
            s::Store(v2, d, indices.ptr + i + 2 * lanes);
            s::Store(v3, d, indices.ptr + i + 3 * lanes);

            // Advance all vectors by 4*lanes
            v0 = s::Add(v0, v_step_4);
            v1 = s::Add(v1, v_step_4);
            v2 = s::Add(v2, v_step_4);
            v3 = s::Add(v3, v_step_4);
        }

        // 2. Single Block Loop (Handle remaining chunks < 4*lanes)
        // Reset v0 to the current sequence head for safety/simplicity
        // (Though v0 is technically already correct from the loop above)
        // Re-sync v_curr for the standard loop approach to save register pressure
        auto v_curr = s::Iota(d, static_cast<scl::Index>(i));

        for (; i + lanes <= N; i += lanes) {
            s::Store(v_curr, d, indices.ptr + i);
            v_curr = s::Add(v_curr, v_step_1);
        }

        // 3. Scalar Loop (Tail)
        for (; i < N; ++i) {
            indices[i] = static_cast<scl::Index>(i);
        }
    }
}

// =============================================================================
// 1. In-Place Argsort
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void argsort_inplace(Array<T> keys, Array<Index> indices) {
    SCL_ASSERT(keys.len == indices.len, "Argsort: Size mismatch");
    detail::iota_serial(indices);
    scl::sort::sort_pairs(keys, indices);
}

template <typename T>
SCL_FORCE_INLINE void argsort_inplace_descending(Array<T> keys, Array<Index> indices) {
    SCL_ASSERT(keys.len == indices.len, "Argsort: Size mismatch");
    detail::iota_serial(indices);
    scl::sort::sort_pairs_descending(keys, indices);
}

// =============================================================================
// 2. Buffered Argsort
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void argsort_buffered(
    Array<const T> keys, 
    Array<Index> indices, 
    Array<Byte> buffer
) {
    SCL_ASSERT(keys.len == indices.len, "Argsort: Indices size mismatch");
    SCL_ASSERT(buffer.len >= keys.byte_size(), "Argsort: Buffer too small");

    T* buffer_ptr = reinterpret_cast<T*>(buffer.ptr);
    Array<T> key_copy(buffer_ptr, keys.len);

    // Use unified memory component
    scl::memory::copy_fast(keys, key_copy);

    argsort_inplace(key_copy, indices);
}

template <typename T>
SCL_FORCE_INLINE void argsort_buffered_descending(
    Array<const T> keys, 
    Array<Index> indices, 
    Array<Byte> buffer
) {
    SCL_ASSERT(keys.len == indices.len, "Argsort: Indices size mismatch");
    SCL_ASSERT(buffer.len >= keys.byte_size(), "Argsort: Buffer too small");

    T* buffer_ptr = reinterpret_cast<T*>(buffer.ptr);
    Array<T> key_copy(buffer_ptr, keys.len);

    // Use unified memory component
    scl::memory::copy_fast(keys, key_copy);

    argsort_inplace_descending(key_copy, indices);
}

// =============================================================================
// 3. Indirect Argsort
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void argsort_indirect(Array<const T> keys, Array<Index> indices) {
    SCL_ASSERT(keys.len == indices.len, "Argsort: Size mismatch");
    detail::iota_serial(indices); // SIMD Optimized initialization
    std::sort(indices.ptr, indices.ptr + indices.len, 
        [&](Index a, Index b) {
            return keys[a] < keys[b];
        }
    );
}

} // namespace scl::sort
