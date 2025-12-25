#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/simd.hpp"
#include <cstring> // For std::memcpy, std::memmove, std::memset
#include <atomic> // For std::atomic_thread_fence

// =============================================================================
/// @file memory.hpp
/// @brief SCL Low-Level Memory Primitives
///
/// Provides primitives for data initialization and movement.
///
/// @section Safety Levels
/// - **Safe**: Handles overlaps (`memmove`), checks bounds in Debug.
/// - **Fast**: Assumes NO overlap (`memcpy`), Undefined Behavior if violated.
/// - **Stream**: Bypasses cache (Non-temporal), Assumes NO overlap.
// =============================================================================

namespace scl::memory {

// =============================================================================
// 1. Initialization (Fill / Zero)
// =============================================================================

/// @brief Fill memory with a value using aggressive SIMD unrolling.
template <typename T>
SCL_FORCE_INLINE void fill(MutableSpan<T> span, T value) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t N = span.size;
    const size_t lanes = s::lanes();
    
    // Broadcast value
    const auto v_val = s::Set(d, value);

    size_t i = 0;
    
    // 4-way Unrolled SIMD Loop
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(v_val, d, span.ptr + i);
        s::Store(v_val, d, span.ptr + i + lanes);
        s::Store(v_val, d, span.ptr + i + 2 * lanes);
        s::Store(v_val, d, span.ptr + i + 3 * lanes);
    }

    // Handle remaining vectors
    for (; i + lanes <= N; i += lanes) {
        s::Store(v_val, d, span.ptr + i);
    }

    // Handle scalar tail
    for (; i < N; ++i) {
        span[i] = value;
    }
}

/// @brief Zero out memory.
template <typename T>
SCL_FORCE_INLINE void zero(MutableSpan<T> span) {
    if constexpr (std::is_trivial_v<T>) {
        std::memset(span.ptr, 0, span.byte_size());
    } else {
        fill(span, T(0));
    }
}

// =============================================================================
// 2. Data Movement
// =============================================================================

/// @brief **Unsafe** Copy: Assumes NO overlap (memcpy).
///
/// @warning
/// - **Undefined Behavior** if `src` and `dst` memory ranges overlap.
/// - **Undefined Behavior** if sizes do not match (in Release mode).
///
/// Use this when you are absolutely certain inputs are distinct buffers.
/// Compilers can optimize this better than `memmove`.
template <typename T>
SCL_FORCE_INLINE void copy_fast(Span<const T> src, MutableSpan<T> dst) {
    // Debug-only checks. In Release, this executes 0 instructions overhead.
    SCL_ASSERT(src.size == dst.size, "copy_fast: Size mismatch");
    SCL_ASSERT(src.end() <= dst.begin() || dst.end() <= src.begin(), 
               "copy_fast: Overlap detected! Use scl::memory::copy instead.");

    if constexpr (std::is_trivially_copyable_v<T>) {
        // std::memcpy implies __restrict__ semantics in standard C++
        std::memcpy(dst.ptr, src.ptr, src.byte_size());
    } else {
        // Fallback for non-trivial types
        for (size_t i = 0; i < src.size; ++i) {
            dst[i] = src[i];
        }
    }
}

/// @brief **Safe** Copy: Handles overlap correctly (memmove).
///
/// Safe to use even if `src` and `dst` overlap (e.g., sliding a window).
/// Slightly slower than `copy_fast`.
template <typename T>
SCL_FORCE_INLINE void copy(Span<const T> src, MutableSpan<T> dst) {
    SCL_ASSERT(src.size == dst.size, "copy: Size mismatch");

    if constexpr (std::is_trivially_copyable_v<T>) {
        // std::memmove handles overlapping regions safely
        std::memmove(dst.ptr, src.ptr, src.byte_size());
    } else {
        // Manual overlap handling for non-trivial types
        if (dst.ptr < src.ptr) {
            // Copy forward
            for (size_t i = 0; i < src.size; ++i) dst[i] = src[i];
        } else if (dst.ptr > src.ptr) {
            // Copy backward
            for (size_t i = src.size; i > 0; --i) dst[i-1] = src[i-1];
        }
    }
}

/// @brief **Stream** Copy: Non-temporal / Cache-bypassing copy.
///
/// Writes directly to RAM, bypassing L1/L2 caches.
/// @warning Assumes **NO overlap**.
///
/// Best for: Large buffers (> 1MB) that will NOT be read immediately.
/// Avoid for: Small buffers (cache bypass overhead is too high).
template <typename T>
SCL_FORCE_INLINE void stream_copy(Span<const T> src, MutableSpan<T> dst) {
    SCL_ASSERT(src.size == dst.size, "stream_copy: Size mismatch");
    // Overlap check skipped for speed, implies UB if violated.

    namespace s = scl::simd;
    const s::Tag d;
    const size_t N = src.size;
    const size_t lanes = s::lanes();

    size_t i = 0;
    
    // 2-way Unrolled Stream Loop
    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        auto v0 = s::Load(d, src.ptr + i);
        auto v1 = s::Load(d, src.ptr + i + lanes);
        
        // Non-temporal Store
        s::Stream(v0, d, dst.ptr + i);
        s::Stream(v1, d, dst.ptr + i + lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        auto v = s::Load(d, src.ptr + i);
        s::Stream(v, d, dst.ptr + i);
    }

    // Scalar tail (Regular store)
    for (; i < N; ++i) {
        dst[i] = src[i];
    }
    
    // Memory fence ensures NT stores are visible
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

} // namespace scl::memory
