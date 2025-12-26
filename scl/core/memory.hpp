#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/simd.hpp"
#include <cstring>  // For std::memcpy, std::memmove, std::memset
#include <atomic>   // For std::atomic_thread_fence
#include <cstdlib>  // For aligned_alloc, free
#include <new>      // For std::align_val_t, operator new/delete

// =============================================================================
/// @file memory.hpp
/// @brief SCL Low-Level Memory Primitives
///
/// Provides primitives for memory allocation, initialization, and movement.
///
/// Module Sections:
/// 1. Aligned Allocation: Cache-aligned memory allocation for SIMD
/// 2. Initialization: Fill and zero operations
/// 3. Data Movement: Copy operations with various safety/performance tradeoffs
///
/// Safety Levels:
/// - Safe: Handles overlaps (memmove), checks bounds in Debug
/// - Fast: Assumes NO overlap (memcpy), Undefined Behavior if violated
/// - Stream: Bypasses cache (Non-temporal), Assumes NO overlap
///
/// Alignment Requirements:
/// For optimal SIMD performance, allocate buffers with 64-byte alignment
/// (matches AVX-512 cache line size).
// =============================================================================

namespace scl::memory {

// =============================================================================
// 1. Aligned Memory Allocation
// =============================================================================

/// @brief Allocate aligned memory for primitive types.
///
/// Use Cases:
/// - SIMD workspaces (require 16/32/64-byte alignment)
/// - Cache-line optimization (64-byte alignment)
/// - Large temporary buffers in kernels
///
/// Platform Support:
/// - C++17+: Uses aligned operator new
/// - Fallback: Uses posix_memalign (POSIX) or _aligned_malloc (Windows)
///
/// Important: Memory allocated with this function MUST be freed with
/// aligned_free, NOT regular free or delete.
///
/// @tparam T Element type (must be trivially constructible)
/// @param count Number of elements to allocate
/// @param alignment Alignment in bytes (must be power of 2, ≥ sizeof(void*))
/// @return Aligned pointer, or nullptr on failure
///
/// Example:
/// auto* buffer = scl::memory::aligned_alloc<double>(1024, 64);
/// // ... use buffer ...
/// scl::memory::aligned_free(buffer);
template <typename T>
SCL_FORCE_INLINE T* aligned_alloc(size_t count, size_t alignment = 64) {
    static_assert(std::is_trivially_constructible_v<T>, 
                  "aligned_alloc: Type must be trivially constructible");
    
    if (count == 0) return nullptr;
    
    const size_t byte_size = count * sizeof(T);
    
    // Use C++17 aligned new with zero-initialization for numeric types
    if constexpr (std::is_arithmetic_v<T>) {
        try {
            return new (std::align_val_t(alignment)) T[count]();
        } catch (...) {
            return nullptr;
        }
    } else {
        // For non-numeric types, use placement new
        void* ptr = nullptr;
        
#if defined(_WIN32) || defined(_WIN64)
        // Windows: _aligned_malloc
        ptr = _aligned_malloc(byte_size, alignment);
#else
        // POSIX: posix_memalign
        if (posix_memalign(&ptr, alignment, byte_size) != 0) {
            ptr = nullptr;
        }
#endif
        
        if (ptr) {
            // Zero-initialize memory
            std::memset(ptr, 0, byte_size);
        }
        
        return static_cast<T*>(ptr);
    }
}

/// @brief Free memory allocated with `aligned_alloc`.
///
/// Warning: Using regular free or delete on aligned memory may crash.
///
/// @tparam T Element type
/// @param ptr Pointer to free (nullptr is safe)
template <typename T>
SCL_FORCE_INLINE void aligned_free(T* ptr) {
    if (!ptr) return;
    
    if constexpr (std::is_arithmetic_v<T>) {
        // Memory allocated with aligned operator new
        operator delete[](ptr, std::align_val_t(alignof(T)));
    } else {
#if defined(_WIN32) || defined(_WIN64)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
}

/// @brief RAII wrapper for aligned memory.
///
/// Use Case: Automatic cleanup of temporary buffers in exception-safe code.
///
/// Example:
/// {
///     auto buffer = scl::memory::AlignedBuffer<Real>(1024, 64);
///     Real* data = buffer.get();
///     // ... use data ...
/// } // Automatically freed
///
/// @tparam T Element type
template <typename T>
class AlignedBuffer {
public:
    /// @brief Allocate aligned buffer
    AlignedBuffer(size_t count, size_t alignment = 64) 
        : ptr_(aligned_alloc<T>(count, alignment)), count_(count) {}
    
    /// @brief Destructor: automatic cleanup
    ~AlignedBuffer() {
        aligned_free(ptr_);
    }
    
    // Non-copyable
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    
    // Movable
    AlignedBuffer(AlignedBuffer&& other) noexcept 
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            aligned_free(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    /// @brief Get raw pointer
    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    
    /// @brief Get element count
    size_t size() const noexcept { return count_; }
    
    /// @brief Array access
    T& operator[](size_t i) noexcept { return ptr_[i]; }
    const T& operator[](size_t i) const noexcept { return ptr_[i]; }
    
    /// @brief Get as Span
    Array<T> span() noexcept { return Array<T>(ptr_, count_); }
    Array<const T> span() const noexcept { return Array<const T>(ptr_, count_); }
    
    /// @brief Check if buffer is valid
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
    
private:
    T* ptr_;
    size_t count_;
};

// =============================================================================
// 2. Initialization (Fill / Zero)
// =============================================================================

/// @brief Fill memory with a value using aggressive SIMD unrolling.
template <typename T>
SCL_FORCE_INLINE void fill(Array<T> span, T value) {
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
SCL_FORCE_INLINE void zero(Array<T> span) {
    if constexpr (std::is_trivial_v<T>) {
        std::memset(span.ptr, 0, span.byte_size());
    } else {
        fill(span, T(0));
    }
}

// =============================================================================
// 2. Data Movement
// =============================================================================

/// @brief Unsafe Copy: Assumes NO overlap (memcpy).
///
/// @warning
/// - Undefined Behavior if src and dst memory ranges overlap.
/// - Undefined Behavior if sizes do not match (in Release mode).
///
/// Use this when you are absolutely certain inputs are distinct buffers.
/// Compilers can optimize this better than `memmove`.
template <typename T>
SCL_FORCE_INLINE void copy_fast(Array<const T> src, Array<T> dst) {
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

/// @brief Safe Copy: Handles overlap correctly (memmove).
///
/// Safe to use even if `src` and `dst` overlap (e.g., sliding a window).
/// Slightly slower than `copy_fast`.
template <typename T>
SCL_FORCE_INLINE void copy(Array<const T> src, Array<T> dst) {
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

/// @brief Stream Copy: Non-temporal / Cache-bypassing copy.
///
/// Writes directly to RAM, bypassing L1/L2 caches.
/// @warning Assumes NO overlap.
///
/// Best for: Large buffers (> 1MB) that will NOT be read immediately.
/// Avoid for: Small buffers (cache bypass overhead is too high).
template <typename T>
SCL_FORCE_INLINE void stream_copy(Array<const T> src, Array<T> dst) {
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
