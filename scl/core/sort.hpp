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

// =============================================================================
// FILE: scl/core/sort.hpp
// BRIEF: SCL High-Performance Sorting (via Google Highway VQSort)
// =============================================================================

namespace scl::sort {

// =============================================================================
// Core Sorting Wrappers
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void sort(Array<T> data) {
    hwy::HWY_NAMESPACE::VQSortStatic(data.ptr, data.len, hwy::SortAscending());
}

template <typename T>
SCL_FORCE_INLINE void sort_descending(Array<T> data) {
    hwy::HWY_NAMESPACE::VQSortStatic(data.ptr, data.len, hwy::SortDescending());
}

// =============================================================================
// Key-Value Sorting - Implementation Details
// =============================================================================

namespace detail {

    constexpr size_t STACK_BUFFER_THRESHOLD = 8192;

    template <size_t Size> struct RawType;
    template <> struct RawType<1> { using type = uint8_t; };
    template <> struct RawType<2> { using type = uint16_t; };
    template <> struct RawType<4> { using type = uint32_t; };
    template <> struct RawType<8> { using type = uint64_t; };
#if defined(__SIZEOF_INT128__)
    template <> struct RawType<16> { using type = __uint128_t; };
#endif

    // SIMD pack: separate arrays -> interleaved buffer
    template <typename Key, typename Value, typename RawT = typename RawType<sizeof(Key)>::type>
    SCL_FORCE_INLINE void pack_interleaved(const Key* SCL_RESTRICT keys,
                                            const Value* SCL_RESTRICT values,
                                            void* SCL_RESTRICT dest,
                                            size_t size) {
        const hwy::HWY_NAMESPACE::ScalableTag<RawT> d;
        auto* k_ptr = reinterpret_cast<const RawT*>(keys);
        auto* v_ptr = reinterpret_cast<const RawT*>(values);
        auto* d_ptr = reinterpret_cast<RawT*>(dest);

        const size_t N = hwy::HWY_NAMESPACE::Lanes(d);
        size_t i = 0;

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

    // SIMD unpack: interleaved buffer -> separate arrays
    template <typename Key, typename Value, typename RawT = typename RawType<sizeof(Key)>::type>
    SCL_FORCE_INLINE void unpack_interleaved(const void* SCL_RESTRICT src,
                                              Key* SCL_RESTRICT keys,
                                              Value* SCL_RESTRICT values,
                                              size_t size) {
        const hwy::HWY_NAMESPACE::ScalableTag<RawT> d;
        auto* s_ptr = reinterpret_cast<const RawT*>(src);
        auto* k_ptr = reinterpret_cast<RawT*>(keys);
        auto* v_ptr = reinterpret_cast<RawT*>(values);

        const size_t N = hwy::HWY_NAMESPACE::Lanes(d);
        size_t i = 0;

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

    // Insertion sort for small arrays
    template <typename Pair, typename Comp>
    SCL_FORCE_INLINE void insertion_sort(Pair* data, size_t n, Comp comp) {
        for (size_t i = 1; i < n; ++i) {
            Pair tmp = data[i];
            size_t j = i;
            while (j > 0 && comp(tmp, data[j - 1])) {
                data[j] = data[j - 1];
                --j;
            }
            data[j] = tmp;
        }
    }

    // Partition step for quicksort
    template <typename Pair, typename Comp>
    SCL_FORCE_INLINE size_t partition(Pair* data, size_t low, size_t high, Comp comp) {
        size_t mid = low + (high - low) / 2;
        if (comp(data[mid], data[low])) std::swap(data[mid], data[low]);
        if (comp(data[high], data[low])) std::swap(data[high], data[low]);
        if (comp(data[high], data[mid])) std::swap(data[high], data[mid]);

        Pair pivot = data[mid];
        std::swap(data[mid], data[high - 1]);

        size_t i = low;
        size_t j = high - 1;

        while (true) {
            while (comp(data[++i], pivot)) {}
            while (comp(pivot, data[--j])) {}
            if (i >= j) break;
            std::swap(data[i], data[j]);
        }

        std::swap(data[i], data[high - 1]);
        return i;
    }

    // Introsort implementation
    template <typename Pair, typename Comp>
    void introsort_impl(Pair* data, size_t low, size_t high, int depth_limit, Comp comp) {
        constexpr size_t INSERTION_THRESHOLD = 16;

        while (high - low > INSERTION_THRESHOLD) {
            if (depth_limit == 0) {
                std::make_heap(data + low, data + high + 1, comp);
                std::sort_heap(data + low, data + high + 1, comp);
                return;
            }

            --depth_limit;
            size_t pivot = partition(data, low, high, comp);

            if (pivot - low < high - pivot) {
                introsort_impl(data, low, pivot - 1, depth_limit, comp);
                low = pivot + 1;
            } else {
                introsort_impl(data, pivot + 1, high, depth_limit, comp);
                high = pivot - 1;
            }
        }
    }

    // Optimized introsort for pairs
    template <typename Pair, typename Comp>
    SCL_FORCE_INLINE void sort_pairs_impl(Pair* data, size_t n, Comp comp) {
        if (SCL_UNLIKELY(n <= 1)) return;

        constexpr size_t INSERTION_THRESHOLD = 16;

        if (n <= INSERTION_THRESHOLD) {
            insertion_sort(data, n, comp);
            return;
        }

        int depth_limit = 2 * (sizeof(size_t) * 8 - __builtin_clzl(n));
        introsort_impl(data, 0, n - 1, depth_limit, comp);

        insertion_sort(data, n, comp);
    }

    // Smart buffer manager: stack for small, heap for large
    template <typename T>
    struct BufferManager {
        T* ptr;
        bool on_heap;

        explicit BufferManager(size_t n) {
            size_t bytes = n * sizeof(T);
            if (bytes <= STACK_BUFFER_THRESHOLD) {
                ptr = static_cast<T*>(alloca(bytes));
                on_heap = false;
            } else {
                ptr = static_cast<T*>(aligned_alloc(SCL_ALIGNMENT, bytes));
                on_heap = true;
            }
        }

        ~BufferManager() {
            if (on_heap) {
                free(ptr);
            }
        }

        T* data() { return ptr; }

        BufferManager(const BufferManager&) = delete;
        BufferManager& operator=(const BufferManager&) = delete;
    };

} // namespace detail

// =============================================================================
// Public Key-Value Sorting API
// =============================================================================

template <typename Key, typename Value>
SCL_FORCE_INLINE void sort_pairs(Array<Key> keys, Array<Value> values) {
#ifndef NDEBUG
    SCL_ASSERT(keys.len == values.len, "Sort keys and values must have same size");
#endif

    if (SCL_UNLIKELY(keys.len <= 1)) return;

    struct alignas(SCL_ALIGNMENT) Pair { Key k; Value v; };

    const size_t n = keys.len;
    const size_t buffer_bytes = n * sizeof(Pair);

    if (buffer_bytes <= detail::STACK_BUFFER_THRESHOLD) {
        Pair* buffer = static_cast<Pair*>(alloca(buffer_bytes));

        // Pack
        if constexpr (sizeof(Key) == sizeof(Value)) {
            detail::pack_interleaved(keys.ptr, values.ptr, buffer, n);
        } else {
            for (size_t i = 0; i < n; ++i) {
                buffer[i] = {keys.ptr[i], values.ptr[i]};
            }
        }

        // Sort
        detail::sort_pairs_impl(buffer, n, [](const Pair& a, const Pair& b) {
            return a.k < b.k;
        });

        // Unpack
        if constexpr (sizeof(Key) == sizeof(Value)) {
            detail::unpack_interleaved(buffer, keys.ptr, values.ptr, n);
        } else {
            for (size_t i = 0; i < n; ++i) {
                keys.ptr[i] = buffer[i].k;
                values.ptr[i] = buffer[i].v;
            }
        }
    } else {
        Pair* buffer = static_cast<Pair*>(aligned_alloc(SCL_ALIGNMENT, buffer_bytes));

        // Pack
        if constexpr (sizeof(Key) == sizeof(Value)) {
            detail::pack_interleaved(keys.ptr, values.ptr, buffer, n);
        } else {
            for (size_t i = 0; i < n; ++i) {
                buffer[i] = {keys.ptr[i], values.ptr[i]};
            }
        }

        // Sort
        detail::sort_pairs_impl(buffer, n, [](const Pair& a, const Pair& b) {
            return a.k < b.k;
        });

        // Unpack
        if constexpr (sizeof(Key) == sizeof(Value)) {
            detail::unpack_interleaved(buffer, keys.ptr, values.ptr, n);
        } else {
            for (size_t i = 0; i < n; ++i) {
                keys.ptr[i] = buffer[i].k;
                values.ptr[i] = buffer[i].v;
            }
        }

        free(buffer);
    }
}

template <typename Key, typename Value>
SCL_FORCE_INLINE void sort_pairs_descending(Array<Key> keys, Array<Value> values) {
#ifndef NDEBUG
    SCL_ASSERT(keys.len == values.len, "Sort keys and values must have same size");
#endif

    if (SCL_UNLIKELY(keys.len <= 1)) return;

    struct alignas(SCL_ALIGNMENT) Pair { Key k; Value v; };

    const size_t n = keys.len;
    const size_t buffer_bytes = n * sizeof(Pair);

    if (buffer_bytes <= detail::STACK_BUFFER_THRESHOLD) {
        Pair* buffer = static_cast<Pair*>(alloca(buffer_bytes));

        // Pack
        if constexpr (sizeof(Key) == sizeof(Value)) {
            detail::pack_interleaved(keys.ptr, values.ptr, buffer, n);
        } else {
            for (size_t i = 0; i < n; ++i) {
                buffer[i] = {keys.ptr[i], values.ptr[i]};
            }
        }

        // Sort (Descending)
        detail::sort_pairs_impl(buffer, n, [](const Pair& a, const Pair& b) {
            return a.k > b.k;
        });

        // Unpack
        if constexpr (sizeof(Key) == sizeof(Value)) {
            detail::unpack_interleaved(buffer, keys.ptr, values.ptr, n);
        } else {
            for (size_t i = 0; i < n; ++i) {
                keys.ptr[i] = buffer[i].k;
                values.ptr[i] = buffer[i].v;
            }
        }
    } else {
        Pair* buffer = static_cast<Pair*>(aligned_alloc(SCL_ALIGNMENT, buffer_bytes));

        // Pack
        if constexpr (sizeof(Key) == sizeof(Value)) {
            detail::pack_interleaved(keys.ptr, values.ptr, buffer, n);
        } else {
            for (size_t i = 0; i < n; ++i) {
                buffer[i] = {keys.ptr[i], values.ptr[i]};
            }
        }

        // Sort (Descending)
        detail::sort_pairs_impl(buffer, n, [](const Pair& a, const Pair& b) {
            return a.k > b.k;
        });

        // Unpack
        if constexpr (sizeof(Key) == sizeof(Value)) {
            detail::unpack_interleaved(buffer, keys.ptr, values.ptr, n);
        } else {
            for (size_t i = 0; i < n; ++i) {
                keys.ptr[i] = buffer[i].k;
                values.ptr[i] = buffer[i].v;
            }
        }

        free(buffer);
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
