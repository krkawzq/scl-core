#pragma once

#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/simd.hpp"

#include <cstring>
#include <utility>

// =============================================================================
// FILE: scl/core/algo.hpp
// BRIEF: High-performance algorithms without boundary checks
// NOTE: All functions assume valid inputs - caller must ensure preconditions
// =============================================================================

namespace scl::algo {

// =============================================================================
// SECTION 1: Binary Search (unchecked)
// =============================================================================

// Binary search for first element >= target
// PRECONDITION: [first, last) is sorted, first <= last
template <typename T, typename V>
SCL_FORCE_INLINE SCL_HOT
const T* lower_bound(const T* first, const T* last, V target) noexcept {
    SCL_ASSUME(first <= last);

    size_t len = static_cast<size_t>(last - first);

    while (len > 0) {
        size_t half = len >> 1;
        const T* mid = first + half;

        if (*mid < target) {
            first = mid + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }

    return first;
}

// Binary search for first element > target
// PRECONDITION: [first, last) is sorted, first <= last
template <typename T, typename V>
SCL_FORCE_INLINE SCL_HOT
const T* upper_bound(const T* first, const T* last, V target) noexcept {
    SCL_ASSUME(first <= last);

    size_t len = static_cast<size_t>(last - first);

    while (len > 0) {
        size_t half = len >> 1;
        const T* mid = first + half;

        if (!(target < *mid)) {
            first = mid + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }

    return first;
}

// Binary search with custom comparator
template <typename T, typename V, typename Cmp>
SCL_FORCE_INLINE SCL_HOT
const T* lower_bound(const T* first, const T* last, V target, Cmp cmp) noexcept {
    SCL_ASSUME(first <= last);

    size_t len = static_cast<size_t>(last - first);

    while (len > 0) {
        size_t half = len >> 1;
        const T* mid = first + half;

        if (cmp(*mid, target)) {
            first = mid + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }

    return first;
}

// =============================================================================
// SECTION 2: Partial Sorting (nth_element)
// =============================================================================

namespace detail {

// Insertion sort for small arrays
template <typename T>
SCL_FORCE_INLINE void insertion_sort(T* first, T* last) noexcept {
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

// Median of three for pivot selection
template <typename T>
SCL_FORCE_INLINE T* median_of_three(T* a, T* b, T* c) noexcept {
    if (*a < *b) {
        if (*b < *c) return b;
        if (*a < *c) return c;
        return a;
    }
    if (*a < *c) return a;
    if (*b < *c) return c;
    return b;
}

// Partition for quickselect
template <typename T>
SCL_FORCE_INLINE T* partition(T* first, T* last, T pivot) noexcept {
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

// Partition around nth element (quickselect)
// PRECONDITION: first <= nth < last
template <typename T>
void nth_element(T* first, T* nth, T* last) noexcept {
    SCL_ASSUME(first <= nth && nth < last);

    constexpr size_t INSERTION_THRESHOLD = 16;

    while (last - first > static_cast<std::ptrdiff_t>(INSERTION_THRESHOLD)) {
        T* mid = first + (last - first) / 2;
        T* pivot_pos = detail::median_of_three(first, mid, last - 1);
        T pivot = *pivot_pos;

        T* cut = detail::partition(first, last, pivot);

        if (cut <= nth) {
            first = cut;
        } else {
            last = cut;
        }
    }

    detail::insertion_sort(first, last);
}

// =============================================================================
// SECTION 3: Memory Operations (unchecked)
// =============================================================================

// Fast copy without overlap check (uses memcpy)
template <typename T>
SCL_FORCE_INLINE void copy(const T* SCL_RESTRICT src, T* SCL_RESTRICT dst, size_t n) noexcept {
    if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(dst, src, n * sizeof(T));
    } else {
        for (size_t i = 0; i < n; ++i) {
            dst[i] = src[i];
        }
    }
}

// Fast fill using SIMD
template <typename T>
SCL_FORCE_INLINE void fill(T* dst, size_t n, T value) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    const auto v_val = s::Set(d, value);

    size_t i = 0;

    SCL_UNROLL(4)
    for (; i + 4 * lanes <= n; i += 4 * lanes) {
        s::Store(v_val, d, dst + i);
        s::Store(v_val, d, dst + i + lanes);
        s::Store(v_val, d, dst + i + 2 * lanes);
        s::Store(v_val, d, dst + i + 3 * lanes);
    }

    for (; i + lanes <= n; i += lanes) {
        s::Store(v_val, d, dst + i);
    }

    for (; i < n; ++i) {
        dst[i] = value;
    }
}

// Fast zero
template <typename T>
SCL_FORCE_INLINE void zero(T* dst, size_t n) noexcept {
    if constexpr (std::is_trivial_v<T>) {
        std::memset(dst, 0, n * sizeof(T));
    } else {
        fill(dst, n, T(0));
    }
}

// =============================================================================
// SECTION 5: Reduction Operations (unchecked)
// =============================================================================

// Sum with 4-way unroll
template <typename T>
SCL_FORCE_INLINE T sum(const T* data, size_t n) noexcept {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    size_t i = 0;

    for (; i + 4 * lanes <= n; i += 4 * lanes) {
        v_sum0 = s::Add(v_sum0, s::Load(d, data + i));
        v_sum1 = s::Add(v_sum1, s::Load(d, data + i + lanes));
        v_sum2 = s::Add(v_sum2, s::Load(d, data + i + 2 * lanes));
        v_sum3 = s::Add(v_sum3, s::Load(d, data + i + 3 * lanes));
    }

    auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

    for (; i + lanes <= n; i += lanes) {
        v_sum = s::Add(v_sum, s::Load(d, data + i));
    }

    T result = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; i < n; ++i) {
        result += data[i];
    }

    return result;
}

// Max element
template <typename T>
SCL_FORCE_INLINE T max(const T* data, size_t n) noexcept {
    SCL_ASSUME(n > 0);

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_max = s::Set(d, data[0]);

    size_t i = 0;

    for (; i + lanes <= n; i += lanes) {
        v_max = s::Max(v_max, s::Load(d, data + i));
    }

    T result = s::GetLane(s::MaxOfLanes(d, v_max));

    for (; i < n; ++i) {
        if (data[i] > result) result = data[i];
    }

    return result;
}

// Min element
template <typename T>
SCL_FORCE_INLINE T min(const T* data, size_t n) noexcept {
    SCL_ASSUME(n > 0);

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_min = s::Set(d, data[0]);

    size_t i = 0;

    for (; i + lanes <= n; i += lanes) {
        v_min = s::Min(v_min, s::Load(d, data + i));
    }

    T result = s::GetLane(s::MinOfLanes(d, v_min));

    for (; i < n; ++i) {
        if (data[i] < result) result = data[i];
    }

    return result;
}

// =============================================================================
// SECTION 6: Heap Operations (unchecked)
// =============================================================================

// Max-heap sift down
template <typename T>
SCL_FORCE_INLINE void heap_sift_down(T* data, size_t heap_size, size_t pos) noexcept {
    while (true) {
        size_t largest = pos;
        size_t left = 2 * pos + 1;
        size_t right = 2 * pos + 2;

        if (left < heap_size && data[left] > data[largest]) {
            largest = left;
        }
        if (right < heap_size && data[right] > data[largest]) {
            largest = right;
        }

        if (largest == pos) break;

        T tmp = static_cast<T&&>(data[pos]);
        data[pos] = static_cast<T&&>(data[largest]);
        data[largest] = static_cast<T&&>(tmp);
        pos = largest;
    }
}

// Max-heap sift up
template <typename T>
SCL_FORCE_INLINE void heap_sift_up(T* data, size_t pos) noexcept {
    while (pos > 0) {
        size_t parent = (pos - 1) / 2;
        if (data[pos] <= data[parent]) break;

        T tmp = static_cast<T&&>(data[pos]);
        data[pos] = static_cast<T&&>(data[parent]);
        data[parent] = static_cast<T&&>(tmp);
        pos = parent;
    }
}

// Build max-heap
template <typename T>
SCL_FORCE_INLINE void make_heap(T* data, size_t n) noexcept {
    if (n < 2) return;

    for (size_t i = n / 2; i > 0; --i) {
        heap_sift_down(data, n, i - 1);
    }
}

// Push element to max-heap (element already at position n-1)
template <typename T>
SCL_FORCE_INLINE void heap_push(T* data, size_t n) noexcept {
    if (n > 1) {
        heap_sift_up(data, n - 1);
    }
}

// Pop max element from heap
template <typename T>
SCL_FORCE_INLINE void heap_pop(T* data, size_t n) noexcept {
    if (n > 1) {
        T tmp = static_cast<T&&>(data[0]);
        data[0] = static_cast<T&&>(data[n - 1]);
        data[n - 1] = static_cast<T&&>(tmp);
        heap_sift_down(data, n - 1, 0);
    }
}

// =============================================================================
// SECTION 7: Counting and Searching
// =============================================================================

// Count elements equal to value
template <typename T, typename V>
SCL_FORCE_INLINE size_t count(const T* data, size_t n, V value) noexcept {
    size_t result = 0;

    size_t i = 0;

    SCL_UNROLL(4)
    for (; i + 4 <= n; i += 4) {
        result += (data[i + 0] == value);
        result += (data[i + 1] == value);
        result += (data[i + 2] == value);
        result += (data[i + 3] == value);
    }

    for (; i < n; ++i) {
        result += (data[i] == value);
    }

    return result;
}

// Find first element equal to value (returns n if not found)
template <typename T, typename V>
SCL_FORCE_INLINE size_t find(const T* data, size_t n, V value) noexcept {
    for (size_t i = 0; i < n; ++i) {
        if (data[i] == value) return i;
    }
    return n;
}

// =============================================================================
// SECTION 8: Index Operations
// =============================================================================

// Iota: fill with consecutive values
template <typename T>
SCL_FORCE_INLINE void iota(T* data, size_t n, T start = T(0)) noexcept {
    for (size_t i = 0; i < n; ++i) {
        data[i] = start + static_cast<T>(i);
    }
}

// Reverse array in place
template <typename T>
SCL_FORCE_INLINE void reverse(T* data, size_t n) noexcept {
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

// =============================================================================
// SECTION 9: Swap and Compare Utilities
// =============================================================================

// Swap two elements
template <typename T>
SCL_FORCE_INLINE void swap(T& a, T& b) noexcept {
    T tmp = static_cast<T&&>(a);
    a = static_cast<T&&>(b);
    b = static_cast<T&&>(tmp);
}

// Min of two values
template <typename T>
SCL_FORCE_INLINE constexpr T min2(T a, T b) noexcept {
    return (a < b) ? a : b;
}

// Max of two values
template <typename T>
SCL_FORCE_INLINE constexpr T max2(T a, T b) noexcept {
    return (a > b) ? a : b;
}

// Clamp value to range
template <typename T>
SCL_FORCE_INLINE constexpr T clamp(T val, T lo, T hi) noexcept {
    return (val < lo) ? lo : ((val > hi) ? hi : val);
}

// =============================================================================
// SECTION 10: ArgMax/ArgMin
// =============================================================================

// Find index of maximum element
template <typename T>
SCL_FORCE_INLINE size_t argmax(const T* data, size_t n) noexcept {
    SCL_ASSUME(n > 0);

    size_t best_idx = 0;
    T best_val = data[0];

    for (size_t i = 1; i < n; ++i) {
        if (data[i] > best_val) {
            best_val = data[i];
            best_idx = i;
        }
    }

    return best_idx;
}

// Find index of minimum element
template <typename T>
SCL_FORCE_INLINE size_t argmin(const T* data, size_t n) noexcept {
    SCL_ASSUME(n > 0);

    size_t best_idx = 0;
    T best_val = data[0];

    for (size_t i = 1; i < n; ++i) {
        if (data[i] < best_val) {
            best_val = data[i];
            best_idx = i;
        }
    }

    return best_idx;
}

// Find both min and max in single pass
template <typename T>
SCL_FORCE_INLINE void minmax(const T* data, size_t n, T& out_min, T& out_max) noexcept {
    SCL_ASSUME(n > 0);

    out_min = data[0];
    out_max = data[0];

    size_t i = 1;

    // Process pairs for efficiency
    for (; i + 1 < n; i += 2) {
        T a = data[i];
        T b = data[i + 1];

        if (a < b) {
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
// SECTION 11: Min-Heap Operations (for bottom-k selection)
// =============================================================================

// Min-heap sift down
template <typename T>
SCL_FORCE_INLINE void min_heap_sift_down(T* data, size_t heap_size, size_t pos) noexcept {
    while (true) {
        size_t smallest = pos;
        size_t left = 2 * pos + 1;
        size_t right = 2 * pos + 2;

        if (left < heap_size && data[left] < data[smallest]) {
            smallest = left;
        }
        if (right < heap_size && data[right] < data[smallest]) {
            smallest = right;
        }

        if (smallest == pos) break;

        T tmp = static_cast<T&&>(data[pos]);
        data[pos] = static_cast<T&&>(data[smallest]);
        data[smallest] = static_cast<T&&>(tmp);
        pos = smallest;
    }
}

// Min-heap sift up
template <typename T>
SCL_FORCE_INLINE void min_heap_sift_up(T* data, size_t pos) noexcept {
    while (pos > 0) {
        size_t parent = (pos - 1) / 2;
        if (data[pos] >= data[parent]) break;

        T tmp = static_cast<T&&>(data[pos]);
        data[pos] = static_cast<T&&>(data[parent]);
        data[parent] = static_cast<T&&>(tmp);
        pos = parent;
    }
}

// Build min-heap
template <typename T>
SCL_FORCE_INLINE void make_min_heap(T* data, size_t n) noexcept {
    if (n < 2) return;

    for (size_t i = n / 2; i > 0; --i) {
        min_heap_sift_down(data, n, i - 1);
    }
}

// =============================================================================
// SECTION 12: Partial Sort and Heap Sort
// =============================================================================

// Heap sort (ascending order)
template <typename T>
void heap_sort(T* data, size_t n) noexcept {
    if (n < 2) return;

    make_heap(data, n);

    for (size_t i = n; i > 1; --i) {
        T tmp = static_cast<T&&>(data[0]);
        data[0] = static_cast<T&&>(data[i - 1]);
        data[i - 1] = static_cast<T&&>(tmp);
        heap_sift_down(data, i - 1, 0);
    }
}

// Partial sort: sort first k elements (smallest k)
template <typename T>
void partial_sort(T* data, size_t n, size_t k) noexcept {
    if (k == 0 || n == 0) return;
    if (k >= n) {
        heap_sort(data, n);
        return;
    }

    // Build max-heap of first k elements
    make_heap(data, k);

    // For remaining elements, if smaller than heap root, replace and sift
    for (size_t i = k; i < n; ++i) {
        if (data[i] < data[0]) {
            T tmp = static_cast<T&&>(data[0]);
            data[0] = static_cast<T&&>(data[i]);
            data[i] = static_cast<T&&>(tmp);
            heap_sift_down(data, k, 0);
        }
    }

    // Sort the k-heap to get ascending order
    for (size_t i = k; i > 1; --i) {
        T tmp = static_cast<T&&>(data[0]);
        data[0] = static_cast<T&&>(data[i - 1]);
        data[i - 1] = static_cast<T&&>(tmp);
        heap_sift_down(data, i - 1, 0);
    }
}

// Partial sort with custom comparator (for indices by scores)
template <typename T, typename Cmp>
void partial_sort(T* data, size_t n, size_t k, Cmp cmp) noexcept {
    if (k == 0 || n == 0) return;
    if (k >= n) k = n;

    // Use quickselect + insertion sort for small k
    constexpr size_t INSERTION_THRESHOLD = 16;

    T* first = data;
    T* last = data + n;
    T* kth = data + k;

    // Quickselect to partition around kth element
    while (last - first > static_cast<std::ptrdiff_t>(INSERTION_THRESHOLD)) {
        T* mid = first + (last - first) / 2;

        // Median of three
        if (cmp(*mid, *first)) swap(*mid, *first);
        if (cmp(*(last-1), *first)) swap(*(last-1), *first);
        if (cmp(*(last-1), *mid)) swap(*(last-1), *mid);

        T pivot = *mid;

        T* lo = first;
        T* hi = last - 1;

        while (true) {
            while (cmp(*lo, pivot)) ++lo;
            while (cmp(pivot, *hi)) --hi;
            if (lo >= hi) break;
            swap(*lo, *hi);
            ++lo; --hi;
        }

        if (lo <= kth) {
            first = lo;
        } else {
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
// SECTION 13: Sparse Vector Operations
// =============================================================================

// Sparse dot product (merge-based, for sorted index arrays)
// Returns sum of val1[i] * val2[j] where idx1[i] == idx2[j]
template <typename T>
SCL_FORCE_INLINE T sparse_dot(
    const Index* SCL_RESTRICT idx1, const T* SCL_RESTRICT val1, size_t n1,
    const Index* SCL_RESTRICT idx2, const T* SCL_RESTRICT val2, size_t n2
) noexcept {
    if (n1 == 0 || n2 == 0) return T(0);

    // O(1) disjointness check
    if (idx1[n1-1] < idx2[0] || idx2[n2-1] < idx1[0]) {
        return T(0);
    }

    T sum = T(0);
    size_t i = 0, j = 0;

    // 8-way skip for non-overlapping ranges
    while (i + 8 <= n1 && j + 8 <= n2) {
        if (idx1[i+7] < idx2[j]) { i += 8; continue; }
        if (idx2[j+7] < idx1[i]) { j += 8; continue; }
        break;
    }

    // 4-way skip
    while (i + 4 <= n1 && j + 4 <= n2) {
        if (idx1[i+3] < idx2[j]) { i += 4; continue; }
        if (idx2[j+3] < idx1[i]) { j += 4; continue; }
        break;
    }

    // Main merge
    while (i < n1 && j < n2) {
        Index a = idx1[i];
        Index b = idx2[j];

        if (a == b) {
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

// Sparse dot with galloping (for very different sparsity)
template <typename T>
SCL_FORCE_INLINE T sparse_dot_gallop(
    const Index* SCL_RESTRICT idx_small, const T* SCL_RESTRICT val_small, size_t n_small,
    const Index* SCL_RESTRICT idx_large, const T* SCL_RESTRICT val_large, size_t n_large
) noexcept {
    if (n_small == 0 || n_large == 0) return T(0);

    T sum = T(0);
    size_t j = 0;

    for (size_t i = 0; i < n_small && j < n_large; ++i) {
        Index target = idx_small[i];

        // Galloping search
        size_t step = 1;
        while (j + step < n_large && idx_large[j + step] < target) {
            step *= 2;
        }

        size_t lo = j;
        size_t hi = (j + step < n_large) ? (j + step) : n_large;

        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (idx_large[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        j = lo;
        if (j < n_large && idx_large[j] == target) {
            sum += val_small[i] * val_large[j];
            ++j;
        }
    }

    return sum;
}

// Adaptive sparse dot (chooses best strategy)
template <typename T>
SCL_FORCE_INLINE T sparse_dot_adaptive(
    const Index* idx1, const T* val1, size_t n1,
    const Index* idx2, const T* val2, size_t n2
) noexcept {
    if (n1 == 0 || n2 == 0) return T(0);

    // Ensure n1 <= n2
    if (n1 > n2) {
        const Index* tmp_idx = idx1; idx1 = idx2; idx2 = tmp_idx;
        const T* tmp_val = val1; val1 = val2; val2 = tmp_val;
        size_t tmp_n = n1; n1 = n2; n2 = tmp_n;
    }

    constexpr size_t GALLOP_RATIO = 32;

    if (n2 / n1 >= GALLOP_RATIO) {
        return sparse_dot_gallop(idx1, val1, n1, idx2, val2, n2);
    } else {
        return sparse_dot(idx1, val1, n1, idx2, val2, n2);
    }
}

// Count intersection size (without computing values)
template <typename T>
SCL_FORCE_INLINE size_t sparse_intersection_size(
    const T* SCL_RESTRICT idx1, size_t n1,
    const T* SCL_RESTRICT idx2, size_t n2
) noexcept {
    if (n1 == 0 || n2 == 0) return 0;
    if (idx1[n1-1] < idx2[0] || idx2[n2-1] < idx1[0]) return 0;

    size_t count = 0;
    size_t i = 0, j = 0;

    while (i < n1 && j < n2) {
        if (idx1[i] == idx2[j]) {
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
// SECTION 14: Range Operations
// =============================================================================

// Check if array is sorted
template <typename T>
SCL_FORCE_INLINE bool is_sorted(const T* data, size_t n) noexcept {
    for (size_t i = 1; i < n; ++i) {
        if (data[i] < data[i - 1]) return false;
    }
    return true;
}

// Unique: remove consecutive duplicates, return new size
template <typename T>
size_t unique(T* data, size_t n) noexcept {
    if (n <= 1) return n;

    size_t write = 1;
    for (size_t read = 1; read < n; ++read) {
        if (!(data[read] == data[write - 1])) {
            if (write != read) {
                data[write] = static_cast<T&&>(data[read]);
            }
            ++write;
        }
    }
    return write;
}

// Rotate left by k positions
template <typename T>
void rotate_left(T* data, size_t n, size_t k) noexcept {
    if (n == 0 || k == 0) return;
    k = k % n;
    if (k == 0) return;

    reverse(data, k);
    reverse(data + k, n - k);
    reverse(data, n);
}

// Partition: move elements satisfying predicate to front, return count
template <typename T, typename Pred>
size_t partition(T* data, size_t n, Pred pred) noexcept {
    size_t write = 0;

    for (size_t read = 0; read < n; ++read) {
        if (pred(data[read])) {
            if (write != read) {
                swap(data[write], data[read]);
            }
            ++write;
        }
    }

    return write;
}

// Gather: dst[i] = src[indices[i]]
template <typename T, typename I>
SCL_FORCE_INLINE void gather(
    const T* SCL_RESTRICT src,
    const I* SCL_RESTRICT indices,
    T* SCL_RESTRICT dst,
    size_t n
) noexcept {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[indices[i]];
    }
}

// Scatter: dst[indices[i]] = src[i]
template <typename T, typename I>
SCL_FORCE_INLINE void scatter(
    const T* SCL_RESTRICT src,
    const I* SCL_RESTRICT indices,
    T* SCL_RESTRICT dst,
    size_t n
) noexcept {
    for (size_t i = 0; i < n; ++i) {
        dst[indices[i]] = src[i];
    }
}

} // namespace scl::algo
