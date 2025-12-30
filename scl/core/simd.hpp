#pragma once

/// @file scl/core/simd.hpp
/// @brief SCL SIMD Wrapper and High-Performance Algorithms (Google Highway)
///
/// This header provides:
///   - Highway SIMD namespace injection and type tags
///   - High-performance sorting (VQSort-based)
///   - Argument sorting (argsort)
///   - SIMD utility functions (iota, etc.)
///
/// All sorting functions are optimized using Highway's vectorized algorithms
/// and fall back to optimized scalar implementations when needed.

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"
#include "scl/core/error.hpp"
#include "scl/core/bits.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

// =============================================================================
// Highway Configuration
// =============================================================================

#if defined(SCL_ONLY_SCALAR) && !defined(HWY_COMPILE_ONLY_SCALAR)
    #define HWY_COMPILE_ONLY_SCALAR
#endif

#define HWY_DISABLED_TARGETS_LOG

#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>
#include <hwy/contrib/sort/vqsort-inl.h>

namespace scl::simd {

// =============================================================================
// SECTION 1: Core Namespace Injection
// =============================================================================

// Import Highway functions into scl::simd namespace
using namespace hwy::HWY_NAMESPACE;

// =============================================================================
// SECTION 2: Smart Tags (Type Inference)
// =============================================================================

using RealTag = ScalableTag<scl::Real>;
using IndexTag = ScalableTag<scl::Index>;
using ReinterpretTag = RebindToUnsigned<RealTag>;

/// @brief Type-based SIMD tag selection
template<typename T>
using SimdTagFor = std::conditional_t<
    std::is_same_v<T, Real>, RealTag,
    std::conditional_t<std::is_same_v<T, Index>, IndexTag,
        ScalableTag<T>>>;

// =============================================================================
// SECTION 3: Sorting Configuration
// =============================================================================

namespace config {

/// @brief Threshold for switching to insertion sort
inline constexpr Size INSERTION_THRESHOLD = 24;

/// @brief Stack buffer threshold (bytes) for temporary allocations
inline constexpr Size STACK_BUFFER_THRESHOLD = 8192;

/// @brief Default memory alignment for sort buffers
inline constexpr Size SORT_ALIGNMENT = 64;

}  // namespace config

// =============================================================================
// SECTION 4: Sorting Concepts
// =============================================================================

/// @brief Concept for totally ordered types
template<typename T>
concept TotallyOrdered = std::totally_ordered<T>;

/// @brief Concept for copyable types
template<typename T>
concept Copyable = std::copyable<T>;

/// @brief Concept for movable types
template<typename T>
concept Movable = std::movable<T>;

/// @brief Concept for comparator functions
template<typename Cmp, typename T>
concept Comparator = std::predicate<Cmp, T, T>;

// =============================================================================
// SECTION 5: Sorting Detail Implementations
// =============================================================================

namespace detail {

// -------------------------------------------------------------------------
// Raw Type Mapping (for bit-level interleaving)
// -------------------------------------------------------------------------

template<std::size_t Size>
struct RawType;

template<> struct RawType<1> { using type = std::uint8_t; };
template<> struct RawType<2> { using type = std::uint16_t; };
template<> struct RawType<4> { using type = std::uint32_t; };
template<> struct RawType<8> { using type = std::uint64_t; };

#if SCL_HAS_INT128
template<> struct RawType<16> { using type = __uint128_t; };
#endif

template<std::size_t Size>
using RawTypeT = typename RawType<Size>::type;

// -------------------------------------------------------------------------
// SIMD Interleaved Pack/Unpack
// -------------------------------------------------------------------------

/// @brief Pack keys and values into interleaved format using SIMD
template<typename Key, typename Value, typename RawT = RawTypeT<sizeof(Key)>>
SCL_FORCE_INLINE 
void pack_interleaved(
    const Key* SCL_RESTRICT keys,
    const Value* SCL_RESTRICT values,
    void* SCL_RESTRICT dest,
    Size size
) {
    const ScalableTag<RawT> d;
    const auto* k_ptr = reinterpret_cast<const RawT*>(keys);
    const auto* v_ptr = reinterpret_cast<const RawT*>(values);
    auto* d_ptr = reinterpret_cast<RawT*>(dest);

    const Size N = Lanes(d);
    Size i = 0;

    if (size >= 2 * N) [[likely]] {
        SCL_PREFETCH(k_ptr + N, 0, 2);
        SCL_PREFETCH(v_ptr + N, 0, 2);
    }

    for (; i + N <= size; i += N) {
        if (i + 2 * N <= size) [[likely]] {
            SCL_PREFETCH(k_ptr + i + 2 * N, 0, 2);
            SCL_PREFETCH(v_ptr + i + 2 * N, 0, 2);
        }

        auto vk = LoadU(d, k_ptr + i);
        auto vv = LoadU(d, v_ptr + i);
        StoreInterleaved2(vk, vv, d, d_ptr + 2 * i);
    }

    // Scalar remainder
    for (; i < size; ++i) {
        d_ptr[2 * i] = k_ptr[i];
        d_ptr[2 * i + 1] = v_ptr[i];
    }
}

/// @brief Unpack interleaved format back to separate keys and values using SIMD
template<typename Key, typename Value, typename RawT = RawTypeT<sizeof(Key)>>
SCL_FORCE_INLINE 
void unpack_interleaved(
    const void* SCL_RESTRICT src,
    Key* SCL_RESTRICT keys,
    Value* SCL_RESTRICT values,
    Size size
) {
    const ScalableTag<RawT> d;
    const auto* s_ptr = reinterpret_cast<const RawT*>(src);
    auto* k_ptr = reinterpret_cast<RawT*>(keys);
    auto* v_ptr = reinterpret_cast<RawT*>(values);

    const Size N = Lanes(d);
    Size i = 0;

    if (size >= 2 * N) [[likely]] {
        SCL_PREFETCH(s_ptr + 2 * N, 0, 2);
    }

    for (; i + N <= size; i += N) {
        if (i + 2 * N <= size) [[likely]] {
            SCL_PREFETCH(s_ptr + 2 * (i + 2 * N), 0, 2);
        }

        auto vk = Undefined(d);
        auto vv = Undefined(d);
        LoadInterleaved2(d, s_ptr + 2 * i, vk, vv);
        StoreU(vk, d, k_ptr + i);
        StoreU(vv, d, v_ptr + i);
    }

    // Scalar remainder
    for (; i < size; ++i) {
        k_ptr[i] = s_ptr[2 * i];
        v_ptr[i] = s_ptr[2 * i + 1];
    }
}

// -------------------------------------------------------------------------
// Insertion Sort (for small arrays)
// -------------------------------------------------------------------------

template<typename Pair, Comparator<Pair> Comp>
SCL_FORCE_INLINE void insertion_sort(Pair* data, Size n, Comp comp) {
    for (Size i = 1; i < n; ++i) {
        Pair tmp = data[i];
        Size j = i;
        while (j > 0 && comp(tmp, data[j - 1])) {
            data[j] = data[j - 1];
            --j;
        }
        data[j] = tmp;
    }
}

// -------------------------------------------------------------------------
// Introsort Partition
// -------------------------------------------------------------------------

template<typename Pair, Comparator<Pair> Comp>
SCL_FORCE_INLINE auto partition(Pair* data, Size low, Size high, Comp comp) -> Size {
    // Median-of-three pivot selection
    Size mid = low + (high - low) / 2;
    if (comp(data[mid], data[low])) std::swap(data[mid], data[low]);
    if (comp(data[high], data[low])) std::swap(data[high], data[low]);
    if (comp(data[high], data[mid])) std::swap(data[high], data[mid]);

    Pair pivot = data[mid];
    std::swap(data[mid], data[high - 1]);

    Size i = low;
    Size j = high - 1;

    while (true) {
        while (comp(data[++i], pivot)) {}
        while (comp(pivot, data[--j])) {}
        if (i >= j) break;
        std::swap(data[i], data[j]);
    }

    std::swap(data[i], data[high - 1]);
    return i;
}

// -------------------------------------------------------------------------
// Introsort Implementation
// -------------------------------------------------------------------------

template<typename Pair, Comparator<Pair> Comp>
void introsort_impl(Pair* data, Size low, Size high, int depth_limit, Comp comp) {
    while (high - low > config::INSERTION_THRESHOLD) {
        if (depth_limit == 0) {
            // Fallback to heapsort
            std::make_heap(data + low, data + high + 1, comp);
            std::sort_heap(data + low, data + high + 1, comp);
            return;
        }

        --depth_limit;
        Size pivot = partition(data, low, high, comp);

        // Tail recursion optimization
        if (pivot - low < high - pivot) {
            introsort_impl(data, low, pivot - 1, depth_limit, comp);
            low = pivot + 1;
        } else {
            introsort_impl(data, pivot + 1, high, depth_limit, comp);
            high = pivot - 1;
        }
    }
}

// -------------------------------------------------------------------------
// Pair Sorting Entry Point
// -------------------------------------------------------------------------

template<typename Pair, Comparator<Pair> Comp>
SCL_FORCE_INLINE void sort_pairs_impl(Pair* data, Size n, Comp comp) {
    if (n <= 1) [[unlikely]] return;

    if (n <= config::INSERTION_THRESHOLD) {
        insertion_sort(data, n, comp);
        return;
    }

    // Calculate depth limit: 2 * log2(n)
    const auto log2n = sizeof(Size) * 8 - static_cast<Size>(scl::bits::clz(static_cast<std::uint64_t>(n)));
    auto depth_limit = static_cast<int>(2 * log2n);
    introsort_impl(data, 0, n - 1, depth_limit, comp);

    // Final insertion sort pass
    insertion_sort(data, n, comp);
}

// -------------------------------------------------------------------------
// Buffer Manager (Stack/Heap allocation)
// -------------------------------------------------------------------------

template<typename T>
class SortBuffer {
public:
    explicit SortBuffer(Size n) : size_(n) {
        const Size bytes = n * sizeof(T);
        if (bytes <= config::STACK_BUFFER_THRESHOLD) {
            // Use stack allocation (alloca)
            ptr_ = static_cast<T*>(SCL_ALLOCA(bytes));
            on_heap_ = false;
        } else {
            // Use heap allocation with alignment
            ptr_ = static_cast<T*>(std::aligned_alloc(config::SORT_ALIGNMENT, bytes));
            on_heap_ = true;
        }
    }
    
    ~SortBuffer() {
        if (on_heap_ && ptr_) {
            std::free(ptr_);
        }
    }
    
    SortBuffer(const SortBuffer&) = delete;
    auto operator=(const SortBuffer&) -> SortBuffer& = delete;
    SortBuffer(SortBuffer&&) = delete;
    auto operator=(SortBuffer&&) -> SortBuffer& = delete;
    
    [[nodiscard]] auto data() noexcept -> T* { return ptr_; }
    [[nodiscard]] auto data() const noexcept -> const T* { return ptr_; }
    [[nodiscard]] auto size() const noexcept -> Size { return size_; }

private:
    T* ptr_ = nullptr;
    Size size_ = 0;
    bool on_heap_ = false;
};

// -------------------------------------------------------------------------
// SIMD Iota (fill with 0, 1, 2, ...)
// -------------------------------------------------------------------------

SCL_FORCE_INLINE void iota_simd(Index* indices, Size n) {
    const IndexTag d;
    const Size lanes = Lanes(d);

    const auto v_step_1 = Set(d, static_cast<Index>(lanes));
    const auto v_step_4 = Set(d, static_cast<Index>(4 * lanes));

    auto v0 = Iota(d, 0);
    auto v1 = Add(v0, v_step_1);
    auto v2 = Add(v1, v_step_1);
    auto v3 = Add(v2, v_step_1);

    Size i = 0;

    // 4x unrolled loop
    for (; i + 4 * lanes <= n; i += 4 * lanes) {
        Store(v0, d, indices + i);
        Store(v1, d, indices + i + lanes);
        Store(v2, d, indices + i + 2 * lanes);
        Store(v3, d, indices + i + 3 * lanes);

        v0 = Add(v0, v_step_4);
        v1 = Add(v1, v_step_4);
        v2 = Add(v2, v_step_4);
        v3 = Add(v3, v_step_4);
    }

    // 1x loop
    auto v_curr = Iota(d, static_cast<Index>(i));
    for (; i + lanes <= n; i += lanes) {
        Store(v_curr, d, indices + i);
        v_curr = Add(v_curr, v_step_1);
    }

    // Scalar remainder
    for (; i < n; ++i) {
        indices[i] = static_cast<Index>(i);
    }
}

}  // namespace detail

// =============================================================================
// SECTION 6: Public Sorting API
// =============================================================================

/// @brief Sort array in ascending order using SIMD-optimized VQSort
/// @tparam T Element type (must be totally ordered)
/// @param data Pointer to data
/// @param n Number of elements
///
/// Uses Highway's VQSort for optimal performance on supported types.
template<TotallyOrdered T>
SCL_FORCE_INLINE void sort(T* data, Size n) {
    VQSortStatic(data, n, hwy::SortAscending());
}

/// @brief Sort array in descending order using SIMD-optimized VQSort
/// @tparam T Element type (must be totally ordered)
/// @param data Pointer to data
/// @param n Number of elements
template<TotallyOrdered T>
SCL_FORCE_INLINE void sort_descending(T* data, Size n) {
    VQSortStatic(data, n, hwy::SortDescending());
}

/// @brief Sort key-value pairs by keys in ascending order
/// @tparam Key Key type (must be copyable)
/// @tparam Value Value type (must be copyable)
/// @param keys Pointer to keys
/// @param values Pointer to values
/// @param n Number of elements
///
/// @note Keys and values are reordered together
template<Copyable Key, Copyable Value>
SCL_FORCE_INLINE void sort_pairs(Key* keys, Value* values, Size n) {
    SCL_CHECK_ARG(keys != nullptr, "sort_pairs: keys is null");
    SCL_CHECK_ARG(values != nullptr, "sort_pairs: values is null");
    
    if (n <= 1) [[unlikely]] return;

    struct alignas(config::SORT_ALIGNMENT) Pair { Key k; Value v; };

    detail::SortBuffer<Pair> buffer(n);

    if constexpr (sizeof(Key) == sizeof(Value)) {
        detail::pack_interleaved(keys, values, buffer.data(), n);
    } else {
        for (Size i = 0; i < n; ++i) {
            buffer.data()[i] = {keys[i], values[i]};
        }
    }

    detail::sort_pairs_impl(buffer.data(), n, [](const Pair& a, const Pair& b) {
        return a.k < b.k;
    });

    if constexpr (sizeof(Key) == sizeof(Value)) {
        detail::unpack_interleaved(buffer.data(), keys, values, n);
    } else {
        for (Size i = 0; i < n; ++i) {
            keys[i] = buffer.data()[i].k;
            values[i] = buffer.data()[i].v;
        }
    }
}

/// @brief Sort key-value pairs by keys in descending order
template<Copyable Key, Copyable Value>
SCL_FORCE_INLINE void sort_pairs_descending(Key* keys, Value* values, Size n) {
    SCL_CHECK_ARG(keys != nullptr, "sort_pairs_descending: keys is null");
    SCL_CHECK_ARG(values != nullptr, "sort_pairs_descending: values is null");
    
    if (n <= 1) [[unlikely]] return;

    struct alignas(config::SORT_ALIGNMENT) Pair { Key k; Value v; };

    detail::SortBuffer<Pair> buffer(n);

    if constexpr (sizeof(Key) == sizeof(Value)) {
        detail::pack_interleaved(keys, values, buffer.data(), n);
    } else {
        for (Size i = 0; i < n; ++i) {
            buffer.data()[i] = {keys[i], values[i]};
        }
    }

    detail::sort_pairs_impl(buffer.data(), n, [](const Pair& a, const Pair& b) {
        return a.k > b.k;
    });

    if constexpr (sizeof(Key) == sizeof(Value)) {
        detail::unpack_interleaved(buffer.data(), keys, values, n);
    } else {
        for (Size i = 0; i < n; ++i) {
            keys[i] = buffer.data()[i].k;
            values[i] = buffer.data()[i].v;
        }
    }
}

// =============================================================================
// SECTION 7: Argsort API
// =============================================================================

/// @brief Compute sorted indices in-place (modifies keys)
/// @tparam T Element type
/// @param keys Pointer to keys (will be sorted)
/// @param indices Pointer to output indices (will be filled with sorted order)
/// @param n Number of elements
///
/// @note This modifies the keys array!
template<TotallyOrdered T>
SCL_FORCE_INLINE void argsort_inplace(T* keys, Index* indices, Size n) {
    SCL_CHECK_ARG(keys != nullptr, "argsort_inplace: keys is null");
    SCL_CHECK_ARG(indices != nullptr, "argsort_inplace: indices is null");
    
    if (n == 0) return;
    
    detail::iota_simd(indices, n);
    sort_pairs(keys, indices, n);
}

/// @brief Compute sorted indices in-place descending (modifies keys)
template<TotallyOrdered T>
SCL_FORCE_INLINE void argsort_inplace_descending(T* keys, Index* indices, Size n) {
    SCL_CHECK_ARG(keys != nullptr, "argsort_inplace_descending: keys is null");
    SCL_CHECK_ARG(indices != nullptr, "argsort_inplace_descending: indices is null");
    
    if (n == 0) return;
    
    detail::iota_simd(indices, n);
    sort_pairs_descending(keys, indices, n);
}

/// @brief Compute sorted indices using external buffer (preserves keys)
/// @tparam T Element type
/// @param keys Pointer to keys (const, not modified)
/// @param indices Pointer to output indices
/// @param buffer External buffer for key copy (must be at least n * sizeof(T) bytes)
/// @param n Number of elements
template<TotallyOrdered T>
SCL_FORCE_INLINE void argsort_buffered(
    const T* keys,
    Index* indices,
    void* buffer,
    Size n
) {
    SCL_CHECK_ARG(keys != nullptr, "argsort_buffered: keys is null");
    SCL_CHECK_ARG(indices != nullptr, "argsort_buffered: indices is null");
    SCL_CHECK_ARG(buffer != nullptr, "argsort_buffered: buffer is null");
    
    if (n == 0) return;
    
    // Copy keys to buffer
    auto* key_copy = static_cast<T*>(buffer);
    std::memcpy(key_copy, keys, n * sizeof(T));
    
    argsort_inplace(key_copy, indices, n);
}

/// @brief Compute sorted indices using external buffer descending
template<TotallyOrdered T>
SCL_FORCE_INLINE void argsort_buffered_descending(
    const T* keys,
    Index* indices,
    void* buffer,
    Size n
) {
    SCL_CHECK_ARG(keys != nullptr, "argsort_buffered_descending: keys is null");
    SCL_CHECK_ARG(indices != nullptr, "argsort_buffered_descending: indices is null");
    SCL_CHECK_ARG(buffer != nullptr, "argsort_buffered_descending: buffer is null");
    
    if (n == 0) return;
    
    auto* key_copy = static_cast<T*>(buffer);
    std::memcpy(key_copy, keys, n * sizeof(T));
    
    argsort_inplace_descending(key_copy, indices, n);
}

/// @brief Compute sorted indices using indirect comparison (preserves keys, no buffer needed)
/// @tparam T Element type
/// @param keys Pointer to keys (const, not modified)
/// @param indices Pointer to output indices
/// @param n Number of elements
///
/// @note Uses std::sort with indirect comparison. Slower than buffered version
///       for large arrays, but doesn't require extra memory.
template<TotallyOrdered T>
SCL_FORCE_INLINE void argsort_indirect(const T* keys, Index* indices, Size n) {
    SCL_CHECK_ARG(keys != nullptr, "argsort_indirect: keys is null");
    SCL_CHECK_ARG(indices != nullptr, "argsort_indirect: indices is null");
    
    if (n == 0) return;
    
    detail::iota_simd(indices, n);
    std::sort(indices, indices + n,
        [keys](Index a, Index b) {
            return keys[a] < keys[b];
        }
    );
}

/// @brief Compute sorted indices using indirect comparison descending
template<TotallyOrdered T>
SCL_FORCE_INLINE void argsort_indirect_descending(const T* keys, Index* indices, Size n) {
    SCL_CHECK_ARG(keys != nullptr, "argsort_indirect_descending: keys is null");
    SCL_CHECK_ARG(indices != nullptr, "argsort_indirect_descending: indices is null");
    
    if (n == 0) return;
    
    detail::iota_simd(indices, n);
    std::sort(indices, indices + n,
        [keys](Index a, Index b) {
            return keys[a] > keys[b];
        }
    );
}

// =============================================================================
// SECTION 8: Convenience Aliases
// =============================================================================

/// @brief Sort array of Real values
SCL_FORCE_INLINE void sort_real(Real* data, Size n) {
    sort(data, n);
}

/// @brief Sort array of Index values
SCL_FORCE_INLINE void sort_index(Index* data, Size n) {
    sort(data, n);
}

/// @brief Fill array with SIMD-optimized iota (0, 1, 2, ...)
/// @param data Output array
/// @param n Number of elements
SCL_FORCE_INLINE void iota(Index* data, Size n) {
    detail::iota_simd(data, n);
}

}  // namespace scl::simd
