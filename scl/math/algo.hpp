#pragma once

/// @file scl/math/algo.hpp
/// @brief High-performance algorithms for mathematical operations
///
/// This header provides:
///   - Binary search operations (lower_bound, upper_bound)
///   - Partial sorting (nth_element, partial_sort)
///   - Heap operations (make_heap, heap_push, heap_pop, heap_sort)
///   - Sparse vector operations (sparse_dot, intersection)
///   - Range operations (is_sorted, unique, rotate, partition)
///   - Index operations (iota, reverse, gather, scatter)
///   - Utility operations (swap, min/max, clamp, argmin/argmax)
///
/// @note All functions assume valid inputs - caller must ensure preconditions
/// @note For vectorized operations, see scl/vectorize namespace

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"

#include <concepts>
#include <type_traits>
#include <utility>

namespace scl::math {

// =============================================================================
// SECTION 1: Concepts
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
// SECTION 2: Binary Search Operations
// =============================================================================

/// @brief Binary search for first element >= target
/// @tparam T Totally ordered element type
/// @tparam V Value type (comparable with T)
/// @param[in] first Pointer to start of sorted range
/// @param[in] last Pointer to end of sorted range
/// @param[in] target Target value to search for
/// @return Pointer to first element >= target, or last if not found
/// @pre [first, last) is sorted in ascending order
/// @pre first <= last
/// @note Complexity: O(log(last - first))
template<TotallyOrdered T, typename V>
    requires std::totally_ordered_with<T, V>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto lower_bound(const T* first, const T* last, const V& target) noexcept -> const T* {
    SCL_ASSUME(first <= last);

    Size len = static_cast<Size>(last - first);

    while (len > 0) {
        Size half = len >> 1;
        const T* mid = first + half;

        if (*mid < target) [[unlikely]] {
            first = mid + 1;
            len -= half + 1;
        } else [[likely]] {
            len = half;
        }
    }

    return first;
}

/// @brief Binary search for first element > target
/// @tparam T Totally ordered element type
/// @tparam V Value type (comparable with T)
/// @param[in] first Pointer to start of sorted range
/// @param[in] last Pointer to end of sorted range
/// @param[in] target Target value to search for
/// @return Pointer to first element > target, or last if not found
/// @pre [first, last) is sorted in ascending order
/// @pre first <= last
/// @note Complexity: O(log(last - first))
template<TotallyOrdered T, typename V>
    requires std::totally_ordered_with<T, V>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto upper_bound(const T* first, const T* last, const V& target) noexcept -> const T* {
    SCL_ASSUME(first <= last);

    Size len = static_cast<Size>(last - first);

    while (len > 0) {
        Size half = len >> 1;
        const T* mid = first + half;

        if (!(target < *mid)) [[likely]] {
            first = mid + 1;
            len -= half + 1;
        } else [[unlikely]] {
            len = half;
        }
    }

    return first;
}

/// @brief Binary search with custom comparator
/// @tparam T Element type
/// @tparam V Value type
/// @tparam Cmp Comparator type
/// @param[in] first Pointer to start of sorted range
/// @param[in] last Pointer to end of sorted range
/// @param[in] target Target value to search for
/// @param[in] cmp Comparison function
/// @return Pointer to first element for which cmp(element, target) is false
/// @pre [first, last) is sorted according to cmp
/// @pre first <= last
template<typename T, typename V, Comparator<T> Cmp>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto lower_bound(const T* first, const T* last, const V& target, Cmp cmp) noexcept -> const T* {
    SCL_ASSUME(first <= last);

    Size len = static_cast<Size>(last - first);

    while (len > 0) {
        Size half = len >> 1;
        const T* mid = first + half;

        if (cmp(*mid, target)) [[unlikely]] {
            first = mid + 1;
            len -= half + 1;
        } else [[likely]] {
            len = half;
        }
    }

    return first;
}

// =============================================================================
// SECTION 3: Partial Sorting (nth_element)
// =============================================================================

namespace detail {

/// @brief Insertion sort for small arrays
template<Movable T>
SCL_FORCE_INLINE
constexpr
auto insertion_sort(T* first, T* last) noexcept -> void {
    for (T* i = first + 1; i < last; ++i) {
        T key = static_cast<T&&>(*i);
        T* j = i;

        while (j > first && *(j - 1) > key) {
            *j = static_cast<T&&>(*(j - 1));
            --j;
        }

        *j = static_cast<T&&>(key);
    }
}

/// @brief Median of three for pivot selection
template<TotallyOrdered T>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto median_of_three(T* a, T* b, T* c) noexcept -> T* {
    if (*a < *b) {
        if (*b < *c) return b;
        if (*a < *c) return c;
        return a;
    }
    if (*a < *c) return a;
    if (*b < *c) return c;
    return b;
}

/// @brief Partition for quickselect
template<Movable T>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto partition(T* first, T* last, const T& pivot) noexcept -> T* {
    while (true) {
        while (*first < pivot) ++first;
        --last;
        while (pivot < *last) --last;

        if (first >= last) return first;

        T tmp = static_cast<T&&>(*first);
        *first = static_cast<T&&>(*last);
        *last = static_cast<T&&>(tmp);
        ++first;
    }
}

} // namespace detail

/// @brief Partition around nth element (quickselect)
/// @tparam T Movable and totally ordered element type
/// @param[in,out] first Pointer to start of range
/// @param[in,out] nth Pointer to nth element position
/// @param[in,out] last Pointer to end of range
/// @pre first <= nth < last
/// @post nth points to the element that would be in that position if sorted
/// @note Complexity: O(n) average, O(n²) worst case
template<Movable T>
    requires TotallyOrdered<T>
constexpr
auto nth_element(T* first, T* nth, T* last) noexcept -> void {
    SCL_ASSUME(first <= nth && nth < last);

    constexpr Size INSERTION_THRESHOLD = 16;

    while (last - first > static_cast<std::ptrdiff_t>(INSERTION_THRESHOLD)) {
        T* mid = first + (last - first) / 2;
        T* pivot_pos = detail::median_of_three(first, mid, last - 1);
        T pivot = *pivot_pos;

        T* cut = detail::partition(first, last, pivot);

        if (cut <= nth) [[likely]] {
            first = cut;
        } else [[unlikely]] {
            last = cut;
        }
    }

    detail::insertion_sort(first, last);
}

// =============================================================================
// SECTION 4: Heap Operations
// =============================================================================

/// @brief Max-heap sift down operation
/// @tparam T Movable and totally ordered element type
/// @param[in,out] data Pointer to heap array
/// @param[in] heap_size Size of the heap
/// @param[in] pos Position to sift down from
/// @pre pos < heap_size
/// @note Complexity: O(log heap_size)
template<Movable T>
    requires TotallyOrdered<T>
SCL_FORCE_INLINE
constexpr
auto heap_sift_down(T* data, Size heap_size, Size pos) noexcept -> void {
    while (true) {
        Size largest = pos;
        Size left = 2 * pos + 1;
        Size right = 2 * pos + 2;

        if (left < heap_size && data[left] > data[largest]) [[unlikely]] {
            largest = left;
        }
        if (right < heap_size && data[right] > data[largest]) [[unlikely]] {
            largest = right;
        }

        if (largest == pos) [[likely]] break;

        T tmp = static_cast<T&&>(data[pos]);
        data[pos] = static_cast<T&&>(data[largest]);
        data[largest] = static_cast<T&&>(tmp);
        pos = largest;
    }
}

/// @brief Max-heap sift up operation
/// @tparam T Movable and totally ordered element type
/// @param[in,out] data Pointer to heap array
/// @param[in] pos Position to sift up from
/// @note Complexity: O(log pos)
template<Movable T>
    requires TotallyOrdered<T>
SCL_FORCE_INLINE
constexpr
auto heap_sift_up(T* data, Size pos) noexcept -> void {
    while (pos > 0) {
        Size parent = (pos - 1) / 2;
        if (data[pos] <= data[parent]) [[likely]] break;

        T tmp = static_cast<T&&>(data[pos]);
        data[pos] = static_cast<T&&>(data[parent]);
        data[parent] = static_cast<T&&>(tmp);
        pos = parent;
    }
}

/// @brief Build max-heap from array
/// @tparam T Movable and totally ordered element type
/// @param[in,out] data Pointer to array
/// @param[in] n Size of array
/// @post data[0..n-1] forms a max-heap
/// @note Complexity: O(n)
template<Movable T>
    requires TotallyOrdered<T>
SCL_FORCE_INLINE
constexpr
auto make_heap(T* data, Size n) noexcept -> void {
    if (n < 2) return;

    for (Size i = n / 2; i > 0; --i) {
        heap_sift_down(data, n, i - 1);
    }
}

/// @brief Push element to max-heap
/// @tparam T Movable and totally ordered element type
/// @param[in,out] data Pointer to heap array
/// @param[in] n New size of heap (element already at position n-1)
/// @pre data[0..n-2] forms a valid max-heap
/// @post data[0..n-1] forms a valid max-heap
/// @note Complexity: O(log n)
template<Movable T>
    requires TotallyOrdered<T>
SCL_FORCE_INLINE
constexpr
auto heap_push(T* data, Size n) noexcept -> void {
    if (n > 1) {
        heap_sift_up(data, n - 1);
    }
}

/// @brief Pop max element from heap
/// @tparam T Movable and totally ordered element type
/// @param[in,out] data Pointer to heap array
/// @param[in] n Current size of heap
/// @pre data[0..n-1] forms a valid max-heap
/// @post data[0..n-2] forms a valid max-heap, max element moved to data[n-1]
/// @note Complexity: O(log n)
template<Movable T>
    requires TotallyOrdered<T>
SCL_FORCE_INLINE
constexpr
auto heap_pop(T* data, Size n) noexcept -> void {
    if (n > 1) {
        T tmp = static_cast<T&&>(data[0]);
        data[0] = static_cast<T&&>(data[n - 1]);
        data[n - 1] = static_cast<T&&>(tmp);
        heap_sift_down(data, n - 1, 0);
    }
}

// =============================================================================
// SECTION 5: Min-Heap Operations
// =============================================================================

/// @brief Min-heap sift down operation
/// @tparam T Movable and totally ordered element type
/// @param[in,out] data Pointer to heap array
/// @param[in] heap_size Size of the heap
/// @param[in] pos Position to sift down from
/// @pre pos < heap_size
/// @note Complexity: O(log heap_size)
template<Movable T>
    requires TotallyOrdered<T>
SCL_FORCE_INLINE
constexpr
auto min_heap_sift_down(T* data, Size heap_size, Size pos) noexcept -> void {
    while (true) {
        Size smallest = pos;
        Size left = 2 * pos + 1;
        Size right = 2 * pos + 2;

        if (left < heap_size && data[left] < data[smallest]) [[unlikely]] {
            smallest = left;
        }
        if (right < heap_size && data[right] < data[smallest]) [[unlikely]] {
            smallest = right;
        }

        if (smallest == pos) [[likely]] break;

        T tmp = static_cast<T&&>(data[pos]);
        data[pos] = static_cast<T&&>(data[smallest]);
        data[smallest] = static_cast<T&&>(tmp);
        pos = smallest;
    }
}

/// @brief Min-heap sift up operation
/// @tparam T Movable and totally ordered element type
/// @param[in,out] data Pointer to heap array
/// @param[in] pos Position to sift up from
/// @note Complexity: O(log pos)
template<Movable T>
    requires TotallyOrdered<T>
SCL_FORCE_INLINE
constexpr
auto min_heap_sift_up(T* data, Size pos) noexcept -> void {
    while (pos > 0) {
        Size parent = (pos - 1) / 2;
        if (data[pos] >= data[parent]) [[likely]] break;

        T tmp = static_cast<T&&>(data[pos]);
        data[pos] = static_cast<T&&>(data[parent]);
        data[parent] = static_cast<T&&>(tmp);
        pos = parent;
    }
}

/// @brief Build min-heap from array
/// @tparam T Movable and totally ordered element type
/// @param[in,out] data Pointer to array
/// @param[in] n Size of array
/// @post data[0..n-1] forms a min-heap
/// @note Complexity: O(n)
template<Movable T>
    requires TotallyOrdered<T>
SCL_FORCE_INLINE
constexpr
auto make_min_heap(T* data, Size n) noexcept -> void {
    if (n < 2) return;

    for (Size i = n / 2; i > 0; --i) {
        min_heap_sift_down(data, n, i - 1);
    }
}

// =============================================================================
// SECTION 6: Sorting Operations
// =============================================================================

/// @brief Heap sort (ascending order)
/// @tparam T Movable and totally ordered element type
/// @param[in,out] data Pointer to array
/// @param[in] n Size of array
/// @post data[0..n-1] is sorted in ascending order
/// @note Complexity: O(n log n)
template<Movable T>
    requires TotallyOrdered<T>
constexpr
auto heap_sort(T* data, Size n) noexcept -> void {
    if (n < 2) return;

    make_heap(data, n);

    for (Size i = n; i > 1; --i) {
        T tmp = static_cast<T&&>(data[0]);
        data[0] = static_cast<T&&>(data[i - 1]);
        data[i - 1] = static_cast<T&&>(tmp);
        heap_sift_down(data, i - 1, 0);
    }
}

/// @brief Partial sort: sort first k elements (smallest k)
/// @tparam T Movable and totally ordered element type
/// @param[in,out] data Pointer to array
/// @param[in] n Size of array
/// @param[in] k Number of elements to sort
/// @post data[0..k-1] contains the k smallest elements in sorted order
/// @note Complexity: O(n log k)
template<Movable T>
    requires TotallyOrdered<T>
constexpr
auto partial_sort(T* data, Size n, Size k) noexcept -> void {
    if (k == 0 || n == 0) return;
    if (k >= n) {
        heap_sort(data, n);
        return;
    }

    // Build max-heap of first k elements
    make_heap(data, k);

    // For remaining elements, if smaller than heap root, replace and sift
    for (Size i = k; i < n; ++i) {
        if (data[i] < data[0]) [[unlikely]] {
            T tmp = static_cast<T&&>(data[0]);
            data[0] = static_cast<T&&>(data[i]);
            data[i] = static_cast<T&&>(tmp);
            heap_sift_down(data, k, 0);
        }
    }

    // Sort the k-heap to get ascending order
    for (Size i = k; i > 1; --i) {
        T tmp = static_cast<T&&>(data[0]);
        data[0] = static_cast<T&&>(data[i - 1]);
        data[i - 1] = static_cast<T&&>(tmp);
        heap_sift_down(data, i - 1, 0);
    }
}

/// @brief Partial sort with custom comparator
/// @tparam T Movable element type
/// @tparam Cmp Comparator type
/// @param[in,out] data Pointer to array
/// @param[in] n Size of array
/// @param[in] k Number of elements to sort
/// @param[in] cmp Comparison function
/// @post data[0..k-1] contains the k smallest elements according to cmp
/// @note Complexity: O(n log k) average
template<Movable T, Comparator<T> Cmp>
constexpr
auto partial_sort(T* data, Size n, Size k, Cmp cmp) noexcept -> void {
    if (k == 0 || n == 0) return;
    if (k >= n) k = n;

    constexpr Size INSERTION_THRESHOLD = 16;

    T* first = data;
    T* last = data + n;
    T* kth = data + k;

    // Quickselect to partition around kth element
    while (last - first > static_cast<std::ptrdiff_t>(INSERTION_THRESHOLD)) {
        T* mid = first + (last - first) / 2;

        // Median of three
        if (cmp(*mid, *first)) {
            T tmp = static_cast<T&&>(*mid);
            *mid = static_cast<T&&>(*first);
            *first = static_cast<T&&>(tmp);
        }
        if (cmp(*(last - 1), *first)) {
            T tmp = static_cast<T&&>(*(last - 1));
            *(last - 1) = static_cast<T&&>(*first);
            *first = static_cast<T&&>(tmp);
        }
        if (cmp(*(last - 1), *mid)) {
            T tmp = static_cast<T&&>(*(last - 1));
            *(last - 1) = static_cast<T&&>(*mid);
            *mid = static_cast<T&&>(tmp);
        }

        T pivot = *mid;

        T* lo = first;
        T* hi = last - 1;

        while (true) {
            while (cmp(*lo, pivot)) ++lo;
            while (cmp(pivot, *hi)) --hi;
            if (lo >= hi) break;
            T tmp = static_cast<T&&>(*lo);
            *lo = static_cast<T&&>(*hi);
            *hi = static_cast<T&&>(tmp);
            ++lo; --hi;
        }

        if (lo <= kth) [[likely]] {
            first = lo;
        } else [[unlikely]] {
            last = lo;
        }
    }

    // Insertion sort the small range containing first k
    for (T* i = data + 1; i < data + k; ++i) {
        T key = static_cast<T&&>(*i);
        T* j = i;
        while (j > data && cmp(key, *(j - 1))) {
            *j = static_cast<T&&>(*(j - 1));
            --j;
        }
        *j = static_cast<T&&>(key);
    }
}

// =============================================================================
// SECTION 7: Sparse Vector Operations
// =============================================================================

/// @brief Sparse dot product (merge-based, for sorted index arrays)
/// @tparam T Value type
/// @param[in] idx1 First index array (sorted)
/// @param[in] val1 First value array
/// @param[in] n1 Size of first arrays
/// @param[in] idx2 Second index array (sorted)
/// @param[in] val2 Second value array
/// @param[in] n2 Size of second arrays
/// @return Sum of val1[i] * val2[j] where idx1[i] == idx2[j]
/// @pre idx1 and idx2 are sorted in ascending order
/// @note Complexity: O(n1 + n2)
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto sparse_dot(
    const Index* SCL_RESTRICT idx1,
    const T* SCL_RESTRICT val1,
    Size n1,
    const Index* SCL_RESTRICT idx2,
    const T* SCL_RESTRICT val2,
    Size n2
) noexcept -> T {
    if (n1 == 0 || n2 == 0) return T(0);

    // O(1) disjointness check
    if (idx1[n1 - 1] < idx2[0] || idx2[n2 - 1] < idx1[0]) [[unlikely]] {
        return T(0);
    }

    T sum = T(0);
    Size i = 0, j = 0;

    // 8-way skip for non-overlapping ranges
    while (i + 8 <= n1 && j + 8 <= n2) {
        if (idx1[i + 7] < idx2[j]) { i += 8; continue; }
        if (idx2[j + 7] < idx1[i]) { j += 8; continue; }
        break;
    }

    // 4-way skip
    while (i + 4 <= n1 && j + 4 <= n2) {
        if (idx1[i + 3] < idx2[j]) { i += 4; continue; }
        if (idx2[j + 3] < idx1[i]) { j += 4; continue; }
        break;
    }

    // Main merge
    while (i < n1 && j < n2) {
        const Index a = idx1[i];
        const Index b = idx2[j];

        if (a == b) [[unlikely]] {
            sum += val1[i] * val2[j];
            ++i; ++j;
        } else if (a < b) {
            ++i;
        } else {
            ++j;
        }
    }

    return sum;
}

/// @brief Sparse dot with galloping (for very different sparsity)
/// @tparam T Value type
/// @param[in] idx_small Smaller index array (sorted)
/// @param[in] val_small Smaller value array
/// @param[in] n_small Size of smaller arrays
/// @param[in] idx_large Larger index array (sorted)
/// @param[in] val_large Larger value array
/// @param[in] n_large Size of larger arrays
/// @return Sum of products where indices match
/// @pre idx_small and idx_large are sorted, n_small <= n_large
/// @note Complexity: O(n_small * log(n_large / n_small))
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto sparse_dot_gallop(
    const Index* SCL_RESTRICT idx_small,
    const T* SCL_RESTRICT val_small,
    Size n_small,
    const Index* SCL_RESTRICT idx_large,
    const T* SCL_RESTRICT val_large,
    Size n_large
) noexcept -> T {
    if (n_small == 0 || n_large == 0) return T(0);

    T sum = T(0);
    Size j = 0;

    for (Size i = 0; i < n_small && j < n_large; ++i) {
        const Index target = idx_small[i];

        // Galloping search
        Size step = 1;
        while (j + step < n_large && idx_large[j + step] < target) {
            step *= 2;
        }

        Size lo = j;
        Size hi = (j + step < n_large) ? (j + step) : n_large;

        while (lo < hi) {
            Size mid = lo + (hi - lo) / 2;
            if (idx_large[mid] < target) [[likely]] {
                lo = mid + 1;
            } else [[unlikely]] {
                hi = mid;
            }
        }

        j = lo;
        if (j < n_large && idx_large[j] == target) [[unlikely]] {
            sum += val_small[i] * val_large[j];
            ++j;
        }
    }

    return sum;
}

/// @brief Adaptive sparse dot (chooses best strategy)
/// @tparam T Value type
/// @param[in] idx1 First index array (sorted)
/// @param[in] val1 First value array
/// @param[in] n1 Size of first arrays
/// @param[in] idx2 Second index array (sorted)
/// @param[in] val2 Second value array
/// @param[in] n2 Size of second arrays
/// @return Sum of products where indices match
/// @pre idx1 and idx2 are sorted in ascending order
/// @note Automatically selects merge or galloping based on size ratio
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto sparse_dot_adaptive(
    const Index* idx1,
    const T* val1,
    Size n1,
    const Index* idx2,
    const T* val2,
    Size n2
) noexcept -> T {
    if (n1 == 0 || n2 == 0) return T(0);

    // Ensure n1 <= n2
    if (n1 > n2) {
        const Index* tmp_idx = idx1; idx1 = idx2; idx2 = tmp_idx;
        const T* tmp_val = val1; val1 = val2; val2 = tmp_val;
        Size tmp_n = n1; n1 = n2; n2 = tmp_n;
    }

    constexpr Size GALLOP_RATIO = 32;

    if (n2 / n1 >= GALLOP_RATIO) [[unlikely]] {
        return sparse_dot_gallop(idx1, val1, n1, idx2, val2, n2);
    } else [[likely]] {
        return sparse_dot(idx1, val1, n1, idx2, val2, n2);
    }
}

/// @brief Count intersection size (without computing values)
/// @tparam T Index type
/// @param[in] idx1 First index array (sorted)
/// @param[in] n1 Size of first array
/// @param[in] idx2 Second index array (sorted)
/// @param[in] n2 Size of second array
/// @return Number of common indices
/// @pre idx1 and idx2 are sorted in ascending order
/// @note Complexity: O(n1 + n2)
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto sparse_intersection_size(
    const T* SCL_RESTRICT idx1,
    Size n1,
    const T* SCL_RESTRICT idx2,
    Size n2
) noexcept -> Size {
    if (n1 == 0 || n2 == 0) return 0;
    if (idx1[n1 - 1] < idx2[0] || idx2[n2 - 1] < idx1[0]) return 0;

    Size count = 0;
    Size i = 0, j = 0;

    while (i < n1 && j < n2) {
        if (idx1[i] == idx2[j]) [[unlikely]] {
            ++count;
            ++i; ++j;
        } else if (idx1[i] < idx2[j]) {
            ++i;
        } else {
            ++j;
        }
    }

    return count;
}

// =============================================================================
// SECTION 8: Range Operations
// =============================================================================

/// @brief Check if array is sorted
/// @tparam T Totally ordered element type
/// @param[in] data Pointer to array
/// @param[in] n Size of array
/// @return true if sorted in ascending order
/// @note Complexity: O(n)
template<TotallyOrdered T>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto is_sorted(const T* data, Size n) noexcept -> bool {
    for (Size i = 1; i < n; ++i) {
        if (data[i] < data[i - 1]) [[unlikely]] return false;
    }
    return true;
}

/// @brief Remove consecutive duplicates
/// @tparam T Movable element type
/// @param[in,out] data Pointer to array
/// @param[in] n Size of array
/// @return New size after removing duplicates
/// @note Complexity: O(n)
template<Movable T>
[[nodiscard]]
constexpr
auto unique(T* data, Size n) noexcept -> Size {
    if (n <= 1) return n;

    Size write = 1;
    for (Size read = 1; read < n; ++read) {
        if (!(data[read] == data[write - 1])) [[likely]] {
            if (write != read) {
                data[write] = static_cast<T&&>(data[read]);
            }
            ++write;
        }
    }
    return write;
}

/// @brief Rotate left by k positions
/// @tparam T Movable element type
/// @param[in,out] data Pointer to array
/// @param[in] n Size of array
/// @param[in] k Number of positions to rotate
/// @note Complexity: O(n)
template<Movable T>
constexpr
auto rotate_left(T* data, Size n, Size k) noexcept -> void;

/// @brief Partition: move elements satisfying predicate to front
/// @tparam T Movable element type
/// @tparam Pred Predicate type
/// @param[in,out] data Pointer to array
/// @param[in] n Size of array
/// @param[in] pred Predicate function
/// @return Count of elements satisfying predicate
/// @note Complexity: O(n)
template<Movable T, typename Pred>
    requires std::predicate<Pred, T>
[[nodiscard]]
constexpr
auto partition(T* data, Size n, Pred pred) noexcept -> Size {
    Size write = 0;

    for (Size read = 0; read < n; ++read) {
        if (pred(data[read])) [[unlikely]] {
            if (write != read) {
                T tmp = static_cast<T&&>(data[write]);
                data[write] = static_cast<T&&>(data[read]);
                data[read] = static_cast<T&&>(tmp);
            }
            ++write;
        }
    }

    return write;
}

// =============================================================================
// SECTION 9: Index Operations
// =============================================================================

/// @brief Fill with consecutive values
/// @tparam T Arithmetic element type
/// @param[out] data Pointer to array
/// @param[in] n Size of array
/// @param[in] start Starting value (default 0)
/// @note Complexity: O(n)
template<typename T>
    requires std::integral<T> || std::floating_point<T>
SCL_FORCE_INLINE
constexpr
auto iota(T* data, Size n, T start = T(0)) noexcept -> void {
    for (Size i = 0; i < n; ++i) {
        data[i] = start + static_cast<T>(i);
    }
}

/// @brief Reverse array in place
/// @tparam T Movable element type
/// @param[in,out] data Pointer to array
/// @param[in] n Size of array
/// @note Complexity: O(n)
template<Movable T>
SCL_FORCE_INLINE
constexpr
auto reverse(T* data, Size n) noexcept -> void {
    T* left = data;
    T* right = data + n - 1;

    while (left < right) {
        T tmp = static_cast<T&&>(*left);
        *left = static_cast<T&&>(*right);
        *right = static_cast<T&&>(tmp);
        ++left;
        --right;
    }
}

/// @brief Gather: dst[i] = src[indices[i]]
/// @tparam T Element type
/// @tparam I Index type
/// @param[in] src Source array
/// @param[in] indices Index array
/// @param[out] dst Destination array
/// @param[in] n Number of elements to gather
/// @note Complexity: O(n)
template<typename T, typename I>
    requires std::integral<I>
SCL_FORCE_INLINE
constexpr
auto gather(
    const T* SCL_RESTRICT src,
    const I* SCL_RESTRICT indices,
    T* SCL_RESTRICT dst,
    Size n
) noexcept -> void {
    for (Size i = 0; i < n; ++i) {
        dst[i] = src[indices[i]];
    }
}

/// @brief Scatter: dst[indices[i]] = src[i]
/// @tparam T Element type
/// @tparam I Index type
/// @param[in] src Source array
/// @param[in] indices Index array
/// @param[out] dst Destination array
/// @param[in] n Number of elements to scatter
/// @note Complexity: O(n)
template<typename T, typename I>
    requires std::integral<I>
SCL_FORCE_INLINE
constexpr
auto scatter(
    const T* SCL_RESTRICT src,
    const I* SCL_RESTRICT indices,
    T* SCL_RESTRICT dst,
    Size n
) noexcept -> void {
    for (Size i = 0; i < n; ++i) {
        dst[indices[i]] = src[i];
    }
}

// =============================================================================
// SECTION 10: Utility Operations
// =============================================================================

/// @brief Swap two elements
/// @tparam T Movable element type
/// @param[in,out] a First element
/// @param[in,out] b Second element
template<Movable T>
SCL_FORCE_INLINE
constexpr
auto swap(T& a, T& b) noexcept -> void {
    T tmp = static_cast<T&&>(a);
    a = static_cast<T&&>(b);
    b = static_cast<T&&>(tmp);
}

/// @brief Min of two values
/// @tparam T Totally ordered element type
/// @param[in] a First value
/// @param[in] b Second value
/// @return Smaller of a and b
template<TotallyOrdered T>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto min2(const T& a, const T& b) noexcept -> T {
    return (a < b) ? a : b;
}

/// @brief Max of two values
/// @tparam T Totally ordered element type
/// @param[in] a First value
/// @param[in] b Second value
/// @return Larger of a and b
template<TotallyOrdered T>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto max2(const T& a, const T& b) noexcept -> T {
    return (a > b) ? a : b;
}

/// @brief Clamp value to range
/// @tparam T Totally ordered element type
/// @param[in] val Value to clamp
/// @param[in] lo Lower bound
/// @param[in] hi Upper bound
/// @return Clamped value
template<TotallyOrdered T>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto clamp(const T& val, const T& lo, const T& hi) noexcept -> T {
    return (val < lo) ? lo : ((val > hi) ? hi : val);
}

/// @brief Find index of maximum element
/// @tparam T Totally ordered element type
/// @param[in] data Pointer to array
/// @param[in] n Size of array (must be > 0)
/// @return Index of maximum element
/// @pre n > 0
/// @note Complexity: O(n)
template<TotallyOrdered T>
[[nodiscard]]
SCL_FORCE_INLINE
auto argmax(const T* data, Size n) noexcept -> Size {
    SCL_ASSUME(n > 0);

    Size best_idx = 0;
    T best_val = data[0];

    for (Size i = 1; i < n; ++i) {
        if (data[i] > best_val) [[unlikely]] {
            best_val = data[i];
            best_idx = i;
        }
    }

    return best_idx;
}

/// @brief Find index of minimum element
/// @tparam T Totally ordered element type
/// @param[in] data Pointer to array
/// @param[in] n Size of array (must be > 0)
/// @return Index of minimum element
/// @pre n > 0
/// @note Complexity: O(n)
template<TotallyOrdered T>
[[nodiscard]]
SCL_FORCE_INLINE
auto argmin(const T* data, Size n) noexcept -> Size {
    SCL_ASSUME(n > 0);

    Size best_idx = 0;
    T best_val = data[0];

    for (Size i = 1; i < n; ++i) {
        if (data[i] < best_val) [[unlikely]] {
            best_val = data[i];
            best_idx = i;
        }
    }

    return best_idx;
}

/// @brief Find both min and max in single pass
/// @tparam T Totally ordered element type
/// @param[in] data Pointer to array
/// @param[in] n Size of array (must be > 0)
/// @param[out] out_min Minimum value
/// @param[out] out_max Maximum value
/// @pre n > 0
/// @note Complexity: O(n) with ~1.5n comparisons
template<TotallyOrdered T>
SCL_FORCE_INLINE
auto minmax(const T* data, Size n, T& out_min, T& out_max) noexcept -> void {
    SCL_ASSUME(n > 0);

    out_min = data[0];
    out_max = data[0];

    Size i = 1;

    // Process pairs for efficiency
    for (; i + 1 < n; i += 2) {
        const T a = data[i];
        const T b = data[i + 1];

        if (a < b) [[likely]] {
            if (a < out_min) out_min = a;
            if (b > out_max) out_max = b;
        } else {
            if (b < out_min) out_min = b;
            if (a > out_max) out_max = a;
        }
    }

    // Handle odd element
    if (i < n) {
        if (data[i] < out_min) out_min = data[i];
        if (data[i] > out_max) out_max = data[i];
    }
}

// =============================================================================
// Implementation of rotate_left (forward declared)
// =============================================================================

template<Movable T>
constexpr
auto rotate_left(T* data, Size n, Size k) noexcept -> void {
    if (n == 0 || k == 0) return;
    k = k % n;
    if (k == 0) return;

    reverse(data, k);
    reverse(data + k, n - k);
    reverse(data, n);
}

} // namespace scl::math

