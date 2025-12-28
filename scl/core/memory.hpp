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
#include <memory>

// =============================================================================
// FILE: scl/core/memory.hpp
// BRIEF: SCL Low-Level Memory Primitives - Modernized with Array<T>
// =============================================================================

namespace scl::memory {

// Constants are defined in scl/config.hpp

// =============================================================================
// Aligned Memory Allocation
// =============================================================================

// Custom deleter for aligned memory - zero-overhead stateless deleter
template <typename T>
struct AlignedDeleter {
    std::size_t alignment_;

    explicit AlignedDeleter(std::size_t alignment = DEFAULT_ALIGNMENT) noexcept
        : alignment_(alignment) {}

    void operator()(T* ptr) const noexcept {
        if (SCL_UNLIKELY(!ptr)) return;

        if constexpr (std::is_arithmetic_v<T>) {
            operator delete[](ptr, std::align_val_t(alignment_));
        } else {
#if defined(_WIN32) || defined(_WIN64)
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
    }
};

// Modernized: Returns unique_ptr with custom deleter
// Intentional: unique_ptr<T[]> is the standard way to manage dynamic arrays
template <typename T>
// unique_ptr<T[]> is standard library type for managing dynamic arrays, not C-style array
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
SCL_FORCE_INLINE auto aligned_alloc(Size count, std::size_t alignment = DEFAULT_ALIGNMENT) -> std::unique_ptr<T[], AlignedDeleter<T>> {
    static_assert(std::is_trivially_constructible_v<T>,
                  "aligned_alloc: Type must be trivially constructible");

    if (SCL_UNLIKELY(count == 0)) {
        // NOLINTNEXTLINE(modernize-avoid-c-arrays)
        return std::unique_ptr<T[], AlignedDeleter<T>>(nullptr, AlignedDeleter<T>(alignment));
    }

    const std::size_t byte_size = static_cast<std::size_t>(count) * sizeof(T);
    T* raw_ptr = nullptr;

    if constexpr (std::is_arithmetic_v<T>) {
        try {
            raw_ptr = new (std::align_val_t(alignment)) T[count]();
        } catch (...) {
            // NOLINTNEXTLINE(modernize-avoid-c-arrays)
            return std::unique_ptr<T[], AlignedDeleter<T>>(nullptr, AlignedDeleter<T>(alignment));
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
            raw_ptr = static_cast<T*>(ptr);
            for (Size i = 0; i < count; ++i) {
                new (raw_ptr + i) T();
            }
        } else {
            // NOLINTNEXTLINE(modernize-avoid-c-arrays)
            return std::unique_ptr<T[], AlignedDeleter<T>>(nullptr, AlignedDeleter<T>(alignment));
        }
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    return std::unique_ptr<T[], AlignedDeleter<T>>(raw_ptr, AlignedDeleter<T>(alignment));
}

// Legacy function for backward compatibility - deprecated, use unique_ptr version
template <typename T>
[[deprecated("Use aligned_alloc returning unique_ptr instead")]]
SCL_FORCE_INLINE auto aligned_alloc_raw(Size count, std::size_t alignment = DEFAULT_ALIGNMENT) -> T* {
    auto ptr = aligned_alloc<T>(count, alignment);
    return ptr.release();
}

// Legacy function for backward compatibility
template <typename T>
SCL_FORCE_INLINE void aligned_free(T* ptr, std::size_t alignment = DEFAULT_ALIGNMENT) {
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

// Modernized RAII wrapper using unique_ptr internally
// Intentional: unique_ptr<T[]> is the standard way to manage dynamic arrays
template <typename T>
struct AlignedBuffer {
    AlignedBuffer(Size count, std::size_t alignment = DEFAULT_ALIGNMENT)
        : ptr_(aligned_alloc<T>(count, alignment)), count_(count) {}

    // Destructor: unique_ptr automatically handles cleanup
    ~AlignedBuffer() = default;

    // Non-copyable
    AlignedBuffer(const AlignedBuffer&) = delete;
    auto operator=(const AlignedBuffer&) -> AlignedBuffer& = delete;

    // Movable - default move semantics work with unique_ptr
    AlignedBuffer(AlignedBuffer&&) noexcept = default;
    auto operator=(AlignedBuffer&&) noexcept -> AlignedBuffer& = default;

    // Modernized: Return Array view instead of raw pointer
    [[nodiscard]] auto array() noexcept -> Array<T> { 
        return Array<T>(ptr_.get(), count_); 
    }
    
    [[nodiscard]] auto array() const noexcept -> Array<const T> { 
        return Array<const T>(ptr_.get(), count_); 
    }

    // Legacy accessors for backward compatibility
    auto get() noexcept -> T* { return ptr_.get(); }
    auto get() const noexcept -> const T* { return ptr_.get(); }

    [[nodiscard]] auto size() const noexcept -> Size { return count_; }

    auto operator[](Size i) noexcept -> T& { 
        return ptr_[i]; 
    }
    
    auto operator[](Size i) const noexcept -> const T& { 
        return ptr_[i]; 
    }

    // Legacy span() method - deprecated, use array() instead
    [[deprecated("Use array() instead")]]
    auto span() noexcept -> Array<T> { return array(); }
    
    [[deprecated("Use array() instead")]]
    auto span() const noexcept -> Array<const T> { return array(); }

    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    // unique_ptr<T[]> is standard library type for managing dynamic arrays, not C-style array
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    std::unique_ptr<T[], AlignedDeleter<T>> ptr_;
    Size count_;
};

// =============================================================================
// Initialization (Fill / Zero)
// =============================================================================

// Modernized: Accept Array<T> instead of raw pointer
template <typename T>
SCL_FORCE_INLINE void fill(Array<T> arr, T value) {
    namespace s = scl::simd;

    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = arr.size();
    const Size lanes = static_cast<Size>(s::Lanes(d));

    const auto v_val = s::Set(d, value);

    Size i = 0;

    // 4-way unrolled SIMD loop
    for (; SCL_LIKELY(i + 4 * lanes <= N); i += 4 * lanes) {
        s::Store(v_val, d, arr.data() + i);
        s::Store(v_val, d, arr.data() + i + lanes);
        s::Store(v_val, d, arr.data() + i + 2 * lanes);
        s::Store(v_val, d, arr.data() + i + 3 * lanes);
    }

    for (; SCL_LIKELY(i + lanes <= N); i += lanes) {
        s::Store(v_val, d, arr.data() + i);
    }

    // Scalar tail
    for (; i < N; ++i) {
        arr[i] = value;
    }
}

// Modernized: Accept Array<T> instead of raw pointer
template <typename T>
SCL_FORCE_INLINE void zero(Array<T> arr) {
    if constexpr (std::is_trivial_v<T>) {
        std::memset(arr.data(), 0, arr.size() * sizeof(T));
    } else {
        fill(arr, T(0));
    }
}

// =============================================================================
// Data Movement
// =============================================================================

// Modernized: Accept Array<const T> and Array<T> instead of raw pointers
template <typename T>
SCL_FORCE_INLINE void copy_fast(Array<const T> src, Array<T> dst) {
    SCL_ASSERT(src.size() == dst.size(), "copy_fast: Size mismatch");
    SCL_ASSERT(src.end() <= dst.begin() || dst.end() <= src.begin(),
               "copy_fast: Overlap detected! Use scl::memory::copy instead.");

    if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(dst.data(), src.data(), src.size() * sizeof(T));
    } else {
        for (Size i = 0; i < src.size(); ++i) {
            dst[i] = src[i];
        }
    }
}

// Modernized: Accept Array<const T> and Array<T> instead of raw pointers
template <typename T>
SCL_FORCE_INLINE void copy(Array<const T> src, Array<T> dst) {
    SCL_ASSERT(src.size() == dst.size(), "copy: Size mismatch");

    if (SCL_UNLIKELY(src.data() == dst.data())) return;

    if constexpr (std::is_trivially_copyable_v<T>) {
        std::memmove(dst.data(), src.data(), src.size() * sizeof(T));
    } else {
        if (dst.data() < src.data()) {
            // Copy forward
            for (Size i = 0; i < src.size(); ++i) {
                dst[i] = src[i];
            }
        } else {
            // Copy backward
            for (Size i = src.size(); i > 0; --i) {
                dst[i - 1] = src[i - 1];
            }
        }
    }
}

// Modernized: Accept Array<const T> and Array<T> instead of raw pointers
template <typename T>
SCL_FORCE_INLINE void stream_copy(Array<const T> src, Array<T> dst) {
    SCL_ASSERT(src.size() == dst.size(), "stream_copy: Size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = src.size();
    const Size lanes = static_cast<Size>(s::Lanes(d));

    // Check alignment: non-temporal stores require 64-byte alignment
    const std::uintptr_t src_align = reinterpret_cast<std::uintptr_t>(src.data()) % STREAM_ALIGNMENT;
    const std::uintptr_t dst_align = reinterpret_cast<std::uintptr_t>(dst.data()) % STREAM_ALIGNMENT;

    if (SCL_UNLIKELY(src_align != 0 || dst_align != 0)) {
        copy_fast(src, dst);
        return;
    }

    Size i = 0;

    // 2-way unrolled stream loop
    for (; SCL_LIKELY(i + 2 * lanes <= N); i += 2 * lanes) {
        auto v0 = s::Load(d, src.data() + i);
        auto v1 = s::Load(d, src.data() + i + lanes);

        s::Stream(v0, d, dst.data() + i);
        s::Stream(v1, d, dst.data() + i + lanes);
    }

    for (; SCL_LIKELY(i + lanes <= N); i += lanes) {
        auto v = s::Load(d, src.data() + i);
        s::Stream(v, d, dst.data() + i);
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

// Modernized: Accept Array<const T> instead of raw pointer
template <int Locality = 3, typename T>
SCL_FORCE_INLINE void prefetch_range_read(
    Array<const T> arr,
    Size max_prefetches = DEFAULT_MAX_PREFETCHES) {
    static_assert(Locality >= 0 && Locality <= 3, "Locality must be 0-3");
    
    const char* p = reinterpret_cast<const char*>(arr.data());
    const Size bytes = arr.size() * sizeof(T);
    const char* end = p + bytes;

    Size count = 0;
    for (; p < end && count < max_prefetches; p += CACHE_LINE_SIZE, ++count) {
        SCL_PREFETCH_READ(p, Locality);
    }
}

// Modernized: Accept Array<T> instead of raw pointer
template <int Locality = 3, typename T>
SCL_FORCE_INLINE void prefetch_range_write(
    Array<T> arr,
    Size max_prefetches = DEFAULT_MAX_PREFETCHES) {
    static_assert(Locality >= 0 && Locality <= 3, "Locality must be 0-3");
    
    char* p = reinterpret_cast<char*>(arr.data());
    const Size bytes = arr.size() * sizeof(T);
    char* end = p + bytes;

    Size count = 0;
    for (; p < end && count < max_prefetches; p += CACHE_LINE_SIZE, ++count) {
        SCL_PREFETCH_WRITE(p, Locality);
    }
}

// Modernized: Accept Array<const T> instead of raw pointer
template <typename T, Size PREFETCH_DISTANCE = DEFAULT_PREFETCH_DISTANCE>
SCL_FORCE_INLINE void prefetch_ahead(
    Array<const T> arr,
    Size current_idx) {
    const Size ahead_idx = current_idx + PREFETCH_DISTANCE;
    if (SCL_LIKELY(ahead_idx < arr.size())) {
        SCL_PREFETCH_READ(arr.data() + ahead_idx, 0);
    }
}

// =============================================================================
// Memory Comparison
// =============================================================================

// Modernized: Already uses Array<const T>
template <typename T>
SCL_FORCE_INLINE auto equal(Array<const T> a, Array<const T> b) -> bool {
    if (SCL_UNLIKELY(a.size() != b.size())) return false;
    if (SCL_UNLIKELY(a.data() == b.data())) return true;
    if (SCL_UNLIKELY(a.size() == 0)) return true;

    if constexpr (std::is_trivially_copyable_v<T>) {
        return std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
    } else {
        for (Size i = 0; i < a.size(); ++i) {
            if (SCL_UNLIKELY(!(a[i] == b[i]))) return false;
        }
        return true;
    }
}

// Modernized: Already uses Array<const T>
template <typename T>
SCL_FORCE_INLINE auto compare(Array<const T> a, Array<const T> b) -> int {
    if constexpr (std::is_trivially_copyable_v<T> && std::is_arithmetic_v<T>) {
        const Size min_len = (a.size() < b.size()) ? a.size() : b.size();
        if (min_len > 0) {
            const int cmp = std::memcmp(a.data(), b.data(), min_len * sizeof(T));
            if (cmp != 0) return (cmp < 0) ? -1 : 1;
        }
        if (a.size() < b.size()) return -1;
        if (a.size() > b.size()) return 1;
        return 0;
    } else {
        const Size min_len = (a.size() < b.size()) ? a.size() : b.size();

        for (Size i = 0; i < min_len; ++i) {
            if (a[i] < b[i]) return -1;
            if (a[i] > b[i]) return 1;
        }

        if (a.size() < b.size()) return -1;
        if (a.size() > b.size()) return 1;
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

// Modernized: Accept Array<T> instead of raw pointers
template <typename T>
SCL_FORCE_INLINE void swap_ranges(Array<T> a, Array<T> b) {
    SCL_ASSERT(a.size() == b.size(), "swap_ranges: Size mismatch");

    if (SCL_UNLIKELY(a.data() == b.data())) return;

    SCL_ASSERT(a.end() <= b.begin() || b.end() <= a.begin(),
               "swap_ranges: Overlap detected!");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = a.size();
    const Size lanes = static_cast<Size>(s::Lanes(d));

    Size i = 0;

    // SIMD swap: load both, store swapped
    for (; SCL_LIKELY(i + lanes <= N); i += lanes) {
        auto va = s::Load(d, a.data() + i);
        auto vb = s::Load(d, b.data() + i);
        s::Store(vb, d, a.data() + i);
        s::Store(va, d, b.data() + i);
    }

    // Scalar tail
    for (; i < N; ++i) {
        swap(a[i], b[i]);
    }
}

// =============================================================================
// Reverse Operations
// =============================================================================

// Modernized: Accept Array<T> instead of raw pointer
template <typename T>
SCL_FORCE_INLINE void reverse(Array<T> arr) {
    if (SCL_UNLIKELY(arr.size() <= 1)) return;

    T* SCL_RESTRICT left = arr.data();
    T* SCL_RESTRICT right = arr.data() + arr.size() - 1;

    while (left < right) {
        swap(*left, *right);
        ++left;
        --right;
    }
}

// Modernized: Accept Array<const T> and Array<T> instead of raw pointers
template <typename T>
SCL_FORCE_INLINE void reverse_copy(Array<const T> src, Array<T> dst) {
    SCL_ASSERT(src.size() == dst.size(), "reverse_copy: Size mismatch");

    const Size N = src.size();
    for (Size i = 0; i < N; ++i) {
        dst[i] = src[N - 1 - i];
    }
}

} // namespace scl::memory
