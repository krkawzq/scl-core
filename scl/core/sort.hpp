#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/simd.hpp"

#include <hwy/contrib/sort/vqsort-inl.h>

#include <algorithm>
#include <memory>
#include <type_traits>
#include <cstring>
#include <concepts>

// =============================================================================
// FILE: scl/core/sort.hpp
// BRIEF: SCL High-Performance Sorting (via Google Highway VQSort)
// =============================================================================

namespace scl::sort {

using namespace scl::sort::config;

// =============================================================================
// Concepts
// =============================================================================

template <typename T>
concept TotallyOrdered = std::totally_ordered<T>;

template <typename T>
concept Copyable = std::copyable<T>;

template <typename T>
concept Movable = std::movable<T>;

template <typename Cmp, typename T>
concept Comparator = std::predicate<Cmp, T, T>;

// =============================================================================
// Core Sorting Wrappers
// =============================================================================

template <TotallyOrdered T>
SCL_FORCE_INLINE void sort(Array<T> data) {
    hwy::HWY_NAMESPACE::VQSortStatic(data.ptr, data.len, hwy::SortAscending());
}

template <TotallyOrdered T>
SCL_FORCE_INLINE void sort_descending(Array<T> data) {
    hwy::HWY_NAMESPACE::VQSortStatic(data.ptr, data.len, hwy::SortDescending());
}

// =============================================================================
// Key-Value Sorting - Implementation Details
// =============================================================================

namespace detail {

    template <std::size_t Size> struct RawType;
    template <> struct RawType<1> { using type = std::uint8_t; };
    template <> struct RawType<2> { using type = std::uint16_t; };
    template <> struct RawType<4> { using type = std::uint32_t; };
    template <> struct RawType<8> { using type = std::uint64_t; };
#if defined(__SIZEOF_INT128__)
    template <> struct RawType<16> { using type = __uint128_t; };
#endif

    template <typename Key, typename Value, typename RawT = typename RawType<sizeof(Key)>::type>
    SCL_FORCE_INLINE void pack_interleaved(
        const Key* SCL_RESTRICT keys,
        const Value* SCL_RESTRICT values,
        void* SCL_RESTRICT dest,
        Size size
    ) {
        const hwy::HWY_NAMESPACE::ScalableTag<RawT> d;
        auto* k_ptr = reinterpret_cast<const RawT*>(keys);
        auto* v_ptr = reinterpret_cast<const RawT*>(values);
        auto* d_ptr = reinterpret_cast<RawT*>(dest);

        const Size N = hwy::HWY_NAMESPACE::Lanes(d);
        Size i = 0;

        if (SCL_LIKELY(size >= 2 * N)) {
            SCL_PREFETCH_READ(k_ptr + N, 2);
            SCL_PREFETCH_READ(v_ptr + N, 2);
        }

        for (; i + N <= size; i += N) {
            if (SCL_LIKELY(i + 2 * N <= size)) {
                SCL_PREFETCH_READ(k_ptr + i + 2 * N, 2);
                SCL_PREFETCH_READ(v_ptr + i + 2 * N, 2);
            }

            auto vk = hwy::HWY_NAMESPACE::LoadU(d, k_ptr + i);
            auto vv = hwy::HWY_NAMESPACE::LoadU(d, v_ptr + i);
            hwy::HWY_NAMESPACE::StoreInterleaved2(vk, vv, d, d_ptr + 2 * i);
        }

        for (; i < size; ++i) {
            d_ptr[2 * i] = k_ptr[i];
            d_ptr[2 * i + 1] = v_ptr[i];
        }
    }

    template <typename Key, typename Value, typename RawT = typename RawType<sizeof(Key)>::type>
    SCL_FORCE_INLINE void unpack_interleaved(
        const void* SCL_RESTRICT src,
        Key* SCL_RESTRICT keys,
        Value* SCL_RESTRICT values,
        Size size
    ) {
        const hwy::HWY_NAMESPACE::ScalableTag<RawT> d;
        auto* s_ptr = reinterpret_cast<const RawT*>(src);
        auto* k_ptr = reinterpret_cast<RawT*>(keys);
        auto* v_ptr = reinterpret_cast<RawT*>(values);

        const Size N = hwy::HWY_NAMESPACE::Lanes(d);
        Size i = 0;

        if (SCL_LIKELY(size >= 2 * N)) {
            SCL_PREFETCH_READ(s_ptr + 2 * N, 2);
        }

        for (; i + N <= size; i += N) {
            if (SCL_LIKELY(i + 2 * N <= size)) {
                SCL_PREFETCH_READ(s_ptr + 2 * (i + 2 * N), 2);
            }

            auto vk = hwy::HWY_NAMESPACE::Undefined(d);
            auto vv = hwy::HWY_NAMESPACE::Undefined(d);
            hwy::HWY_NAMESPACE::LoadInterleaved2(d, s_ptr + 2 * i, vk, vv);
            hwy::HWY_NAMESPACE::StoreU(vk, d, k_ptr + i);
            hwy::HWY_NAMESPACE::StoreU(vv, d, v_ptr + i);
        }

        for (; i < size; ++i) {
            k_ptr[i] = s_ptr[2 * i];
            v_ptr[i] = s_ptr[2 * i + 1];
        }
    }

    template <typename Pair, Comparator<Pair> Comp>
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

    template <typename Pair, Comparator<Pair> Comp>
    SCL_FORCE_INLINE Size partition(Pair* data, Size low, Size high, Comp comp) {
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

    template <typename Pair, Comparator<Pair> Comp>
    void introsort_impl(Pair* data, Size low, Size high, int depth_limit, Comp comp) {
        while (high - low > INSERTION_THRESHOLD) {
            if (depth_limit == 0) {
                std::make_heap(data + low, data + high + 1, comp);
                std::sort_heap(data + low, data + high + 1, comp);
                return;
            }

            --depth_limit;
            Size pivot = partition(data, low, high, comp);

            if (pivot - low < high - pivot) {
                introsort_impl(data, low, pivot - 1, depth_limit, comp);
                low = pivot + 1;
            } else {
                introsort_impl(data, pivot + 1, high, depth_limit, comp);
                high = pivot - 1;
            }
        }
    }

    template <typename Pair, Comparator<Pair> Comp>
    SCL_FORCE_INLINE void sort_pairs_impl(Pair* data, Size n, Comp comp) {
        if (SCL_UNLIKELY(n <= 1)) return;

        if (n <= INSERTION_THRESHOLD) {
            insertion_sort(data, n, comp);
            return;
        }

        auto depth_limit = static_cast<int>(2 * (sizeof(Size) * 8 - SCL_CLZ(n)));
        introsort_impl(data, 0, n - 1, depth_limit, comp);

        insertion_sort(data, n, comp);
    }

    template <typename T>
    struct BufferManager {
        T* ptr;
        bool on_heap;

        explicit BufferManager(Size n) {
            Size bytes = n * sizeof(T);
            if (bytes <= STACK_BUFFER_THRESHOLD) {
                ptr = static_cast<T*>(SCL_ALLOCA(bytes));
                on_heap = false;
            } else {
                ptr = static_cast<T*>(scl::aligned_alloc_impl(SCL_ALIGNMENT, bytes));
                on_heap = true;
            }
        }

        ~BufferManager() {
            if (on_heap) {
                scl::aligned_free_impl(ptr);
            }
        }

        T* data() { return ptr; }

        BufferManager(const BufferManager&) = delete;
        BufferManager& operator=(const BufferManager&) = delete;
        BufferManager(BufferManager&&) = delete;
        BufferManager& operator=(BufferManager&&) = delete;
    };

} // namespace detail

// =============================================================================
// Public Key-Value Sorting API
// =============================================================================

template <Copyable Key, Copyable Value>
SCL_FORCE_INLINE void sort_pairs(Array<Key> keys, Array<Value> values) {
#ifndef NDEBUG
    SCL_ASSERT(keys.len == values.len, "Sort keys and values must have same size");
#endif

    if (SCL_UNLIKELY(keys.len <= 1)) return;

    struct alignas(SCL_ALIGNMENT) Pair { Key k; Value v; };

    const Size n = keys.len;
    detail::BufferManager<Pair> buffer(n);

    if constexpr (sizeof(Key) == sizeof(Value)) {
        detail::pack_interleaved(keys.ptr, values.ptr, buffer.data(), n);
    } else {
        for (Size i = 0; i < n; ++i) {
            buffer.data()[i] = {keys.ptr[i], values.ptr[i]};
        }
    }

    detail::sort_pairs_impl(buffer.data(), n, [](const Pair& a, const Pair& b) {
        return a.k < b.k;
    });

    if constexpr (sizeof(Key) == sizeof(Value)) {
        detail::unpack_interleaved(buffer.data(), keys.ptr, values.ptr, n);
    } else {
        for (Size i = 0; i < n; ++i) {
            keys.ptr[i] = buffer.data()[i].k;
            values.ptr[i] = buffer.data()[i].v;
        }
    }
}

template <Copyable Key, Copyable Value>
SCL_FORCE_INLINE void sort_pairs_descending(Array<Key> keys, Array<Value> values) {
#ifndef NDEBUG
    SCL_ASSERT(keys.len == values.len, "Sort keys and values must have same size");
#endif

    if (SCL_UNLIKELY(keys.len <= 1)) return;

    struct alignas(SCL_ALIGNMENT) Pair { Key k; Value v; };

    const Size n = keys.len;
    detail::BufferManager<Pair> buffer(n);

    if constexpr (sizeof(Key) == sizeof(Value)) {
        detail::pack_interleaved(keys.ptr, values.ptr, buffer.data(), n);
    } else {
        for (Size i = 0; i < n; ++i) {
            buffer.data()[i] = {keys.ptr[i], values.ptr[i]};
        }
    }

    detail::sort_pairs_impl(buffer.data(), n, [](const Pair& a, const Pair& b) {
        return a.k > b.k;
    });

    if constexpr (sizeof(Key) == sizeof(Value)) {
        detail::unpack_interleaved(buffer.data(), keys.ptr, values.ptr, n);
    } else {
        for (Size i = 0; i < n; ++i) {
            keys.ptr[i] = buffer.data()[i].k;
            values.ptr[i] = buffer.data()[i].v;
        }
    }
}

// =============================================================================
// Convenience Aliases
// =============================================================================

SCL_FORCE_INLINE void sort_real(Array<Real> data) {
    sort<Real>(data);
}

SCL_FORCE_INLINE void sort_index(Array<Index> data) {
    sort<Index>(data);
}

} // namespace scl::sort
