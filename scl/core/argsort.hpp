#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/memory.hpp"
#include <algorithm>
#include <memory>

// =============================================================================
// FILE: scl/core/argsort.hpp
// BRIEF: Argument sorting (returns sorted indices)
// =============================================================================

namespace scl::sort {

namespace detail {

    SCL_FORCE_INLINE void iota_simd(Array<Index> indices) {
        namespace s = scl::simd;

        const s::IndexTag d;
        const size_t N = indices.len;
        const size_t lanes = s::lanes();

        const auto v_step_1 = s::Set(d, static_cast<scl::Index>(lanes));
        const auto v_step_4 = s::Set(d, static_cast<scl::Index>(4 * lanes));

        auto v0 = s::Iota(d, 0);
        auto v1 = s::Add(v0, v_step_1);
        auto v2 = s::Add(v1, v_step_1);
        auto v3 = s::Add(v2, v_step_1);

        size_t i = 0;

        for (; i + 4 * lanes <= N; i += 4 * lanes) {
            s::Store(v0, d, indices.ptr + i);
            s::Store(v1, d, indices.ptr + i + lanes);
            s::Store(v2, d, indices.ptr + i + 2 * lanes);
            s::Store(v3, d, indices.ptr + i + 3 * lanes);

            v0 = s::Add(v0, v_step_4);
            v1 = s::Add(v1, v_step_4);
            v2 = s::Add(v2, v_step_4);
            v3 = s::Add(v3, v_step_4);
        }

        auto v_curr = s::Iota(d, static_cast<scl::Index>(i));

        for (; i + lanes <= N; i += lanes) {
            s::Store(v_curr, d, indices.ptr + i);
            v_curr = s::Add(v_curr, v_step_1);
        }

        for (; i < N; ++i) {
            indices[i] = static_cast<scl::Index>(i);
        }
    }
}

// =============================================================================
// In-Place Argsort
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void argsort_inplace(Array<T> keys, Array<Index> indices) {
    SCL_ASSERT(keys.len == indices.len, "Argsort: Size mismatch");
    if (keys.len == 0) return;
    detail::iota_simd(indices);
    scl::sort::sort_pairs(keys, indices);
}

template <typename T>
SCL_FORCE_INLINE void argsort_inplace_descending(Array<T> keys, Array<Index> indices) {
    SCL_ASSERT(keys.len == indices.len, "Argsort: Size mismatch");
    if (keys.len == 0) return;
    detail::iota_simd(indices);
    scl::sort::sort_pairs_descending(keys, indices);
}

// =============================================================================
// Buffered Argsort
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void argsort_buffered(
    Array<const T> keys,
    Array<Index> indices,
    Array<Byte> buffer
) {
    SCL_ASSERT(keys.len == indices.len, "Argsort: Indices size mismatch");
    if (keys.len == 0) return;
    
    constexpr std::size_t alignment = alignof(T);
    const std::size_t required_size = keys.len * sizeof(T) + alignment - 1;
    SCL_ASSERT(buffer.len >= required_size, "Argsort: Buffer too small for alignment");
    
    void* raw_ptr = buffer.ptr;
    std::size_t space = buffer.len;
    void* aligned_ptr = std::align(alignment, keys.len * sizeof(T), raw_ptr, space);
    SCL_ASSERT(aligned_ptr != nullptr, "Argsort: Buffer alignment failed");
    
    T* buffer_ptr = static_cast<T*>(aligned_ptr);
    Array<T> key_copy(buffer_ptr, keys.len);
    
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
    if (keys.len == 0) return;
    
    constexpr std::size_t alignment = alignof(T);
    const std::size_t required_size = keys.len * sizeof(T) + alignment - 1;
    SCL_ASSERT(buffer.len >= required_size, "Argsort: Buffer too small for alignment");
    
    void* raw_ptr = buffer.ptr;
    std::size_t space = buffer.len;
    void* aligned_ptr = std::align(alignment, keys.len * sizeof(T), raw_ptr, space);
    SCL_ASSERT(aligned_ptr != nullptr, "Argsort: Buffer alignment failed");
    
    T* buffer_ptr = static_cast<T*>(aligned_ptr);
    Array<T> key_copy(buffer_ptr, keys.len);
    
    scl::memory::copy_fast(keys, key_copy);
    argsort_inplace_descending(key_copy, indices);
}

// =============================================================================
// Indirect Argsort
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void argsort_indirect(Array<const T> keys, Array<Index> indices) {
    SCL_ASSERT(keys.len == indices.len, "Argsort: Size mismatch");
    if (keys.len == 0) return;
    detail::iota_simd(indices);
    std::sort(indices.ptr, indices.ptr + indices.len,
        [&](Index a, Index b) {
            return keys[a] < keys[b];
        }
    );
}

template <typename T>
SCL_FORCE_INLINE void argsort_indirect_descending(Array<const T> keys, Array<Index> indices) {
    SCL_ASSERT(keys.len == indices.len, "Argsort: Size mismatch");
    if (keys.len == 0) return;
    detail::iota_simd(indices);
    std::sort(indices.ptr, indices.ptr + indices.len,
        [&](Index a, Index b) {
            return keys[a] > keys[b];
        }
    );
}

} // namespace scl::sort
