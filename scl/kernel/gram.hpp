#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
// FILE: scl/kernel/gram.hpp
// BRIEF: Gram matrix computation with adaptive sparse dot product
// =============================================================================

namespace scl::kernel::gram {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 32;
    constexpr Size RATIO_THRESHOLD = 32;
    constexpr Size GALLOP_THRESHOLD = 256;
    constexpr Size CHUNK_SIZE = 64;
}

// =============================================================================
// SIMD Utilities
// =============================================================================

namespace detail {

template <typename T>
SCL_FORCE_INLINE T dot_linear_branchless(
    const Index* SCL_RESTRICT idx1, const T* SCL_RESTRICT val1, Size n1,
    const Index* SCL_RESTRICT idx2, const T* SCL_RESTRICT val2, Size n2
) {
    T sum = T(0);
    Size i = 0, j = 0;

    // 8-way skip optimization for large non-overlapping ranges
    while (i + 8 <= n1 && j + 8 <= n2) {
        Index i7 = idx1[i+7], j0 = idx2[j];
        Index i0 = idx1[i], j7 = idx2[j+7];

        if (i7 < j0) { i += 8; continue; }
        if (j7 < i0) { j += 8; continue; }
        break;
    }

    // 4-way skip optimization for non-overlapping ranges
    while (i + 4 <= n1 && j + 4 <= n2) {
        Index i3 = idx1[i+3], j0 = idx2[j];
        Index i0 = idx1[i], j3 = idx2[j+3];

        if (i3 < j0) { i += 4; continue; }
        if (j3 < i0) { j += 4; continue; }
        break;
    }

    // Main merge with prefetching
    while (i < n1 && j < n2) {
        if (SCL_LIKELY(i + config::PREFETCH_DISTANCE < n1)) {
            SCL_PREFETCH_READ(&idx1[i + config::PREFETCH_DISTANCE], 0);
            SCL_PREFETCH_READ(&val1[i + config::PREFETCH_DISTANCE], 0);
        }
        if (SCL_LIKELY(j + config::PREFETCH_DISTANCE < n2)) {
            SCL_PREFETCH_READ(&idx2[j + config::PREFETCH_DISTANCE], 0);
            SCL_PREFETCH_READ(&val2[j + config::PREFETCH_DISTANCE], 0);
        }

        Index r1 = idx1[i];
        Index r2 = idx2[j];

        if (r1 == r2) {
            sum += val1[i] * val2[j];
            ++i; ++j;
        } else if (r1 < r2) {
            ++i;
        } else {
            ++j;
        }
    }

    return sum;
}

template <typename T>
SCL_FORCE_INLINE T dot_binary(
    const Index* SCL_RESTRICT idx_small, const T* SCL_RESTRICT val_small, Size n_small,
    const Index* SCL_RESTRICT idx_large, const T* SCL_RESTRICT val_large, Size n_large
) {
    T sum = T(0);

    // Narrow search range using boundary information
    // Find first element in large that could match first in small
    Index first_target = idx_small[0];
    Index last_target = idx_small[n_small - 1];

    // Binary search for start position in large array
    const Index* start = scl::algo::lower_bound(idx_large, idx_large + n_large, first_target);
    if (start == idx_large + n_large) return T(0);

    // Binary search for end position in large array
    const Index* end = scl::algo::upper_bound(start, idx_large + n_large, last_target);

    const Index* base = start;
    Size len = static_cast<Size>(end - start);

    if (len == 0) return T(0);

    for (Size i = 0; i < n_small; ++i) {
        if (SCL_LIKELY(i + 4 < n_small)) {
            SCL_PREFETCH_READ(&idx_small[i + 4], 0);
            SCL_PREFETCH_READ(&val_small[i + 4], 0);
        }

        Index target = idx_small[i];

        // Early termination: target exceeds remaining range
        if (SCL_UNLIKELY(target > idx_large[n_large - 1])) break;

        auto it = scl::algo::lower_bound(base, base + len, target);

        if (it != base + len && *it == target) {
            Size offset = static_cast<Size>(it - idx_large);
            sum += val_small[i] * val_large[offset];

            Size step = static_cast<Size>(it - base) + 1;
            if (SCL_UNLIKELY(step >= len)) break;
            base += step;
            len -= step;
        } else {
            Size step = static_cast<Size>(it - base);
            if (SCL_UNLIKELY(step >= len)) break;
            base += step;
            len -= step;
        }
    }

    return sum;
}

template <typename T>
SCL_FORCE_INLINE T dot_gallop(
    const Index* SCL_RESTRICT idx_small, const T* SCL_RESTRICT val_small, Size n_small,
    const Index* SCL_RESTRICT idx_large, const T* SCL_RESTRICT val_large, Size n_large
) {
    T sum = T(0);

    // Narrow search range using boundary information
    Index first_target = idx_small[0];
    Index last_target = idx_small[n_small - 1];

    // Find start position via galloping
    Size j = 0;
    Size step = 1;
    while (j + step < n_large && idx_large[j + step] < first_target) {
        step *= 2;
    }
    Size lo = j;
    Size hi = scl::algo::min2(j + step, n_large);
    while (lo < hi) {
        Size mid = lo + (hi - lo) / 2;
        if (idx_large[mid] < first_target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    j = lo;

    // Early exit if start beyond large array
    if (j >= n_large) return T(0);

    // Find effective end via binary search
    Size n_large_effective = n_large;
    if (last_target < idx_large[n_large - 1]) {
        auto end_it = scl::algo::upper_bound(idx_large + j, idx_large + n_large, last_target);
        n_large_effective = static_cast<Size>(end_it - idx_large);
    }

    for (Size i = 0; i < n_small && j < n_large_effective; ++i) {
        if (SCL_LIKELY(i + 4 < n_small)) {
            SCL_PREFETCH_READ(&idx_small[i + 4], 0);
            SCL_PREFETCH_READ(&val_small[i + 4], 0);
        }

        Index target = idx_small[i];

        // Exponential search (galloping)
        step = 1;
        while (j + step < n_large_effective && idx_large[j + step] < target) {
            step *= 2;
        }

        lo = j;
        hi = scl::algo::min2(j + step, n_large_effective);

        // Binary search within bounds
        while (lo < hi) {
            Size mid = lo + (hi - lo) / 2;
            if (idx_large[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        j = lo;
        if (j < n_large_effective && idx_large[j] == target) {
            sum += val_small[i] * val_large[j];
            ++j;
        }
    }

    return sum;
}

template <typename T>
SCL_FORCE_INLINE T sparse_dot_adaptive(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
) {
    // Early exit for empty vectors
    if (SCL_UNLIKELY(n1 == 0 || n2 == 0)) {
        return T(0);
    }

    // O(1) range disjointness check - critical optimization
    // If index ranges don't overlap, dot product is zero
    if (SCL_UNLIKELY(idx1[n1-1] < idx2[0] || idx2[n2-1] < idx1[0])) {
        return T(0);
    }

    // Ensure n1 <= n2 for algorithm selection
    if (n1 > n2) {
        scl::algo::swap(idx1, idx2);
        scl::algo::swap(val1, val2);
        scl::algo::swap(n1, n2);
    }

    Size ratio = n2 / n1;

    if (ratio >= config::GALLOP_THRESHOLD) {
        return dot_gallop(idx1, val1, n1, idx2, val2, n2);
    } else if (ratio >= config::RATIO_THRESHOLD) {
        return dot_binary(idx1, val1, n1, idx2, val2, n2);
    } else {
        return dot_linear_branchless(idx1, val1, n1, idx2, val2, n2);
    }
}

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool IsCSR>
void gram(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index N = matrix.primary_dim();
    const Size N_size = static_cast<Size>(N);

    SCL_CHECK_DIM(output.len >= N_size * N_size, "Gram: Output size mismatch");

    scl::threading::parallel_for(Size(0), N_size, [&](size_t i) {
        const Index idx_i = static_cast<Index>(i);
        const Index len_i = matrix.primary_length(idx_i);
        const Size len_i_sz = static_cast<Size>(len_i);

        auto idx_i_arr = matrix.primary_indices(idx_i);
        auto val_i_arr = matrix.primary_values(idx_i);

        T* row_ptr = output.ptr + (i * N_size);

        row_ptr[i] = scl::vectorize::sum_squared(Array<const T>(val_i_arr.ptr, len_i_sz));

        for (Size j = i + 1; j < N_size; ++j) {
            const Index idx_j = static_cast<Index>(j);
            const Index len_j = matrix.primary_length(idx_j);
            const Size len_j_sz = static_cast<Size>(len_j);

            auto idx_j_arr = matrix.primary_indices(idx_j);
            auto val_j_arr = matrix.primary_values(idx_j);

            T dot = detail::sparse_dot_adaptive(
                idx_i_arr.ptr, val_i_arr.ptr, len_i_sz,
                idx_j_arr.ptr, val_j_arr.ptr, len_j_sz
            );

            row_ptr[j] = dot;
            output.ptr[j * N_size + i] = dot;
        }
    });
}

} // namespace scl::kernel::gram

