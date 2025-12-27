#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"
#include "scl/core/simd.hpp"
#include <cstring>
#include <atomic>
#include <cstdlib>
#include <new>
#include <utility>

// =============================================================================
// FILE: scl/core/memory.hpp
// BRIEF: SCL Low-Level Memory Primitives
// =============================================================================

namespace scl::memory {

// =============================================================================
// Aligned Memory Allocation
// =============================================================================

template <typename T>
SCL_FORCE_INLINE T* aligned_alloc(size_t count, size_t alignment = 64) {
    static_assert(std::is_trivially_constructible_v<T>,
                  "aligned_alloc: Type must be trivially constructible");

    if (SCL_UNLIKELY(count == 0)) { return nullptr; }

    const size_t byte_size = count * sizeof(T);

    if constexpr (std::is_arithmetic_v<T>) {
        try {
            return new (std::align_val_t(alignment)) T[count]();
        } catch (...) {
            return nullptr;
        }
    } else {
        void* ptr = nullptr;

#if defined(_WIN32) || defined(_WIN64)
        ptr = _aligned_malloc(byte_size, alignment);
#else
        if (SCL_UNLIKELY(posix_memalign(&ptr, alignment, byte_size) != 0)) {
            ptr = nullptr;
        }
#endif

        if (SCL_LIKELY(ptr)) {
            T* typed_ptr = static_cast<T*>(ptr);
            for (size_t i = 0; i < count; ++i) {
                new (typed_ptr + i) T();
            }
        }

        return static_cast<T*>(ptr);
    }
}

template <typename T>
SCL_FORCE_INLINE void aligned_free(T* ptr, size_t alignment = 64) {
    if (SCL_UNLIKELY(!ptr)) return;

    if constexpr (std::is_arithmetic_v<T>) {
        operator delete[](ptr, std::align_val_t(alignment));
    } else {
#if defined(_WIN32) || defined(_WIN64)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
}

template <typename T>
struct AlignedBuffer {
    AlignedBuffer(size_t count, size_t alignment = 64)
        : ptr_(aligned_alloc<T>(count, alignment)), count_(count), alignment_(alignment) {}

    ~AlignedBuffer() {
        aligned_free(ptr_, alignment_);
    }

    // Non-copyable
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    // Movable
    AlignedBuffer(AlignedBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_), alignment_(other.alignment_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.alignment_ = 64;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            aligned_free(ptr_, alignment_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            alignment_ = other.alignment_;
            other.ptr_ = nullptr;
            other.count_ = 0;
            other.alignment_ = 64;
        }
        return *this;
    }

    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }

    size_t size() const noexcept { return count_; }

    T& operator[](size_t i) noexcept { return ptr_[i]; }
    const T& operator[](size_t i) const noexcept { return ptr_[i]; }

    Array<T> span() noexcept { return Array<T>(ptr_, count_); }
    Array<const T> span() const noexcept { return Array<const T>(ptr_, count_); }

    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    T* ptr_;
    size_t count_;
    size_t alignment_;
};

// =============================================================================
// Initialization (Fill / Zero)
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void fill(Array<T> span, T value) {
    namespace s = scl::simd;

    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);

    const auto v_val = s::Set(d, value);

    size_t i = 0;

    // 4-way unrolled SIMD loop
    for (; SCL_LIKELY(i + 4 * lanes <= N); i += 4 * lanes) {
        s::Store(v_val, d, span.ptr + i);
        s::Store(v_val, d, span.ptr + i + lanes);
        s::Store(v_val, d, span.ptr + i + 2 * lanes);
        s::Store(v_val, d, span.ptr + i + 3 * lanes);
    }

    for (; SCL_LIKELY(i + lanes <= N); i += lanes) {
        s::Store(v_val, d, span.ptr + i);
    }

    // Scalar tail
    for (; i < N; ++i) {
        span[i] = value;
    }
}

template <typename T>
SCL_FORCE_INLINE void zero(Array<T> span) {
    if constexpr (std::is_trivial_v<T>) {
        std::memset(span.ptr, 0, span.len * sizeof(T));
    } else {
        fill(span, T(0));
    }
}

// =============================================================================
// Data Movement
// =============================================================================

// Unsafe copy: assumes NO overlap (memcpy)
template <typename T>
SCL_FORCE_INLINE void copy_fast(Array<const T> src, Array<T> dst) {
    SCL_ASSERT(src.len == dst.len, "copy_fast: Size mismatch");
    SCL_ASSERT(src.end() <= dst.begin() || dst.end() <= src.begin(),
               "copy_fast: Overlap detected! Use scl::memory::copy instead.");

    if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(dst.ptr, src.ptr, src.len * sizeof(T));
    } else {
        for (size_t i = 0; i < src.len; ++i) {
            dst[i] = src[i];
        }
    }
}

// Safe copy: handles overlap correctly (memmove)
template <typename T>
SCL_FORCE_INLINE void copy(Array<const T> src, Array<T> dst) {
    SCL_ASSERT(src.len == dst.len, "copy: Size mismatch");

    if (SCL_UNLIKELY(src.ptr == dst.ptr)) return;

    if constexpr (std::is_trivially_copyable_v<T>) {
        std::memmove(dst.ptr, src.ptr, src.len * sizeof(T));
    } else {
        if (dst.ptr < src.ptr) {
            // Copy forward
            for (size_t i = 0; i < src.len; ++i) dst[i] = src[i];
        } else {
            // Copy backward
            for (size_t i = src.len; i > 0; --i) dst[i-1] = src[i-1];
        }
    }
}

// Stream copy: non-temporal, cache-bypassing
template <typename T>
SCL_FORCE_INLINE void stream_copy(Array<const T> src, Array<T> dst) {
    SCL_ASSERT(src.len == dst.len, "stream_copy: Size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = src.len;
    const size_t lanes = s::Lanes(d);
    constexpr size_t STREAM_ALIGN = 64;

    // Check alignment: non-temporal stores require 64-byte alignment
    const uintptr_t src_align = reinterpret_cast<uintptr_t>(src.ptr) % STREAM_ALIGN;
    const uintptr_t dst_align = reinterpret_cast<uintptr_t>(dst.ptr) % STREAM_ALIGN;

    if (SCL_UNLIKELY(src_align != 0 || dst_align != 0)) {
        copy_fast(src, dst);
        return;
    }

    size_t i = 0;

    // 2-way unrolled stream loop
    for (; SCL_LIKELY(i + 2 * lanes <= N); i += 2 * lanes) {
        auto v0 = s::Load(d, src.ptr + i);
        auto v1 = s::Load(d, src.ptr + i + lanes);

        s::Stream(v0, d, dst.ptr + i);
        s::Stream(v1, d, dst.ptr + i + lanes);
    }

    for (; SCL_LIKELY(i + lanes <= N); i += lanes) {
        auto v = s::Load(d, src.ptr + i);
        s::Stream(v, d, dst.ptr + i);
    }

    // Scalar tail
    for (; i < N; ++i) {
        dst[i] = src[i];
    }

    std::atomic_thread_fence(std::memory_order_seq_cst);
}

// =============================================================================
// Prefetch Utilities
// =============================================================================

template <int Locality = 3, typename T>
SCL_FORCE_INLINE void prefetch_range_read(const T* ptr, size_t bytes, size_t max_prefetches = 16) {
    static_assert(Locality >= 0 && Locality <= 3, "Locality must be 0-3");
    constexpr size_t CACHE_LINE = 64;
    const char* p = reinterpret_cast<const char*>(ptr);
    const char* end = p + bytes;

    size_t count = 0;
    for (; p < end && count < max_prefetches; p += CACHE_LINE, ++count) {
        SCL_PREFETCH_READ(p, Locality);
    }
}

template <int Locality = 3, typename T>
SCL_FORCE_INLINE void prefetch_range_write(T* ptr, size_t bytes, size_t max_prefetches = 16) {
    static_assert(Locality >= 0 && Locality <= 3, "Locality must be 0-3");
    constexpr size_t CACHE_LINE = 64;
    char* p = reinterpret_cast<char*>(ptr);
    char* end = p + bytes;

    size_t count = 0;
    for (; p < end && count < max_prefetches; p += CACHE_LINE, ++count) {
        SCL_PREFETCH_WRITE(p, Locality);
    }
}

template <typename T, size_t PREFETCH_DISTANCE = 8>
SCL_FORCE_INLINE void prefetch_ahead(const T* data, size_t current_idx, size_t total_size) {
    const size_t ahead_idx = current_idx + PREFETCH_DISTANCE;
    if (SCL_LIKELY(ahead_idx < total_size)) {
        SCL_PREFETCH_READ(data + ahead_idx, 0);
    }
}

// =============================================================================
// Memory Comparison
// =============================================================================

template <typename T>
SCL_FORCE_INLINE bool equal(Array<const T> a, Array<const T> b) {
    if (SCL_UNLIKELY(a.len != b.len)) return false;
    if (SCL_UNLIKELY(a.ptr == b.ptr)) return true;
    if (SCL_UNLIKELY(a.len == 0)) return true;

    if constexpr (std::is_trivially_copyable_v<T>) {
        return std::memcmp(a.ptr, b.ptr, a.len * sizeof(T)) == 0;
    } else {
        for (size_t i = 0; i < a.len; ++i) {
            if (SCL_UNLIKELY(!(a[i] == b[i]))) return false;
        }
        return true;
    }
}

template <typename T>
SCL_FORCE_INLINE int compare(Array<const T> a, Array<const T> b) {
    if constexpr (std::is_trivially_copyable_v<T> && std::is_arithmetic_v<T>) {
        const size_t min_len = (a.len < b.len) ? a.len : b.len;
        if (min_len > 0) {
            const int cmp = std::memcmp(a.ptr, b.ptr, min_len * sizeof(T));
            if (cmp != 0) return (cmp < 0) ? -1 : 1;
        }
        if (a.len < b.len) return -1;
        if (a.len > b.len) return 1;
        return 0;
    } else {
        const size_t min_len = (a.len < b.len) ? a.len : b.len;

        for (size_t i = 0; i < min_len; ++i) {
            if (a[i] < b[i]) return -1;
            if (a[i] > b[i]) return 1;
        }

        if (a.len < b.len) return -1;
        if (a.len > b.len) return 1;
        return 0;
    }
}

// =============================================================================
// Swap Operations
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void swap(T& a, T& b) noexcept {
    T tmp = static_cast<T&&>(a);
    a = static_cast<T&&>(b);
    b = static_cast<T&&>(tmp);
}

template <typename T>
SCL_FORCE_INLINE void swap_ranges(Array<T> a, Array<T> b) {
    SCL_ASSERT(a.len == b.len, "swap_ranges: Size mismatch");

    if (SCL_UNLIKELY(a.ptr == b.ptr)) return;

    SCL_ASSERT(a.end() <= b.begin() || b.end() <= a.begin(),
               "swap_ranges: Overlap detected!");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = a.len;
    const size_t lanes = s::Lanes(d);

    size_t i = 0;

    // SIMD swap: load both, store swapped
    for (; SCL_LIKELY(i + lanes <= N); i += lanes) {
        auto va = s::Load(d, a.ptr + i);
        auto vb = s::Load(d, b.ptr + i);
        s::Store(vb, d, a.ptr + i);
        s::Store(va, d, b.ptr + i);
    }

    // Scalar tail
    for (; i < N; ++i) {
        swap(a[i], b[i]);
    }
}

// =============================================================================
// Reverse Operations
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void reverse(Array<T> span) {
    if (SCL_UNLIKELY(span.len <= 1)) return;

    T* SCL_RESTRICT left = span.ptr;
    T* SCL_RESTRICT right = span.ptr + span.len - 1;

    while (left < right) {
        swap(*left, *right);
        ++left;
        --right;
    }
}

template <typename T>
SCL_FORCE_INLINE void reverse_copy(Array<const T> src, Array<T> dst) {
    SCL_ASSERT(src.len == dst.len, "reverse_copy: Size mismatch");

    const size_t N = src.len;
    for (size_t i = 0; i < N; ++i) {
        dst[i] = src[N - 1 - i];
    }
}

} // namespace scl::memory
