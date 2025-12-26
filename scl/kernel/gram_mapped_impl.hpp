#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <algorithm>

// =============================================================================
/// @file gram_mapped_impl.hpp
/// @brief Gram Matrix for Memory-Mapped Sparse Matrices
///
/// ## Key Optimizations
///
/// 1. Streaming Access Pattern
///    - Prefetch hints for OS page cache
///    - Sequential row access where possible
///
/// 2. Adaptive Dot Product
///    - Linear merge / binary search / galloping
///    - Auto-selected based on length ratio
///
/// 3. SIMD Self Dot Product
///    - 4-way unrolled for diagonal elements
///
/// Performance: Near-RAM performance for cached data
// =============================================================================

namespace scl::kernel::gram::mapped {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 32;
    constexpr Size RATIO_THRESHOLD = 32;
    constexpr Size GALLOP_THRESHOLD = 256;
}

// =============================================================================
// SECTION 2: Utilities (shared implementation)
// =============================================================================

namespace detail {

/// @brief SIMD self dot product
template <typename T>
SCL_FORCE_INLINE T self_dot_simd(const T* SCL_RESTRICT vals, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);

    Size k = 0;

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        v_sum0 = s::MulAdd(v0, v0, v_sum0);
        v_sum1 = s::MulAdd(v1, v1, v_sum1);
        v_sum0 = s::MulAdd(v2, v2, v_sum0);
        v_sum1 = s::MulAdd(v3, v3, v_sum1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::MulAdd(v, v, v_sum);
    }

    T result = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < len; ++k) {
        result += vals[k] * vals[k];
    }

    return result;
}

/// @brief Linear merge dot product
template <typename T>
SCL_FORCE_INLINE T dot_linear(
    const Index* SCL_RESTRICT idx1, const T* SCL_RESTRICT val1, Size n1,
    const Index* SCL_RESTRICT idx2, const T* SCL_RESTRICT val2, Size n2
) {
    T sum0 = T(0), sum1 = T(0), sum2 = T(0), sum3 = T(0);
    Size i = 0, j = 0;

    while (i < n1 && j < n2) {
        if (SCL_LIKELY(i + config::PREFETCH_DISTANCE < n1)) {
            SCL_PREFETCH_READ(&idx1[i + config::PREFETCH_DISTANCE], 0);
        }
        if (SCL_LIKELY(j + config::PREFETCH_DISTANCE < n2)) {
            SCL_PREFETCH_READ(&idx2[j + config::PREFETCH_DISTANCE], 0);
        }

        Index r1 = idx1[i];
        Index r2 = idx2[j];

        if (r1 == r2) {
            T prod = val1[i] * val2[j];
            sum0 += prod;
            T tmp = sum0; sum0 = sum1; sum1 = sum2; sum2 = sum3; sum3 = tmp;
            ++i; ++j;
        } else if (r1 < r2) {
            ++i;
        } else {
            ++j;
        }
    }

    return sum0 + sum1 + sum2 + sum3;
}

/// @brief Binary search dot product
template <typename T>
SCL_FORCE_INLINE T dot_binary(
    const Index* SCL_RESTRICT idx_small, const T* SCL_RESTRICT val_small, Size n_small,
    const Index* SCL_RESTRICT idx_large, const T* SCL_RESTRICT val_large, Size n_large
) {
    T sum = T(0);
    const Index* base = idx_large;
    Size len = n_large;

    for (Size i = 0; i < n_small; ++i) {
        Index target = idx_small[i];
        auto it = std::lower_bound(base, base + len, target);

        if (it != base + len && *it == target) {
            Size offset = static_cast<Size>(it - idx_large);
            sum += val_small[i] * val_large[offset];

            Size step = static_cast<Size>(it - base) + 1;
            if (step >= len) break;
            base += step;
            len -= step;
        } else {
            Size step = static_cast<Size>(it - base);
            if (step >= len) break;
            base += step;
            len -= step;
        }
    }

    return sum;
}

/// @brief Galloping search dot product
template <typename T>
SCL_FORCE_INLINE T dot_gallop(
    const Index* SCL_RESTRICT idx_small, const T* SCL_RESTRICT val_small, Size n_small,
    const Index* SCL_RESTRICT idx_large, const T* SCL_RESTRICT val_large, Size n_large
) {
    T sum = T(0);
    Size j = 0;

    for (Size i = 0; i < n_small && j < n_large; ++i) {
        Index target = idx_small[i];

        Size step = 1;
        while (j + step < n_large && idx_large[j + step] < target) {
            step *= 2;
        }

        Size lo = j;
        Size hi = std::min(j + step, n_large);

        while (lo < hi) {
            Size mid = lo + (hi - lo) / 2;
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

/// @brief Adaptive dot product dispatcher
template <typename T>
SCL_FORCE_INLINE T sparse_dot_adaptive(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
) {
    if (SCL_UNLIKELY(n1 == 0 || n2 == 0)) {
        return T(0);
    }

    if (n1 > n2) {
        std::swap(idx1, idx2);
        std::swap(val1, val2);
        std::swap(n1, n2);
    }

    Size ratio = n2 / n1;

    if (ratio >= config::GALLOP_THRESHOLD) {
        return dot_gallop(idx1, val1, n1, idx2, val2, n2);
    } else if (ratio >= config::RATIO_THRESHOLD) {
        return dot_binary(idx1, val1, n1, idx2, val2, n2);
    } else {
        return dot_linear(idx1, val1, n1, idx2, val2, n2);
    }
}

} // namespace detail

// =============================================================================
// SECTION 3: MappedCustomSparse Gram Matrix
// =============================================================================

/// @brief Gram matrix for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void gram_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);

    SCL_CHECK_DIM(output.len >= N_size * N_size, "Gram: Output size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), N_size, [&](size_t i) {
        auto values_i = scl::primary_values(matrix, static_cast<Index>(i));
        auto indices_i = scl::primary_indices(matrix, static_cast<Index>(i));

        const Index* SCL_RESTRICT idx_i = indices_i.ptr;
        const T* SCL_RESTRICT val_i = values_i.ptr;
        Size len_i = values_i.len;

        T* row_ptr = output.ptr + (i * N_size);

        // Diagonal
        row_ptr[i] = detail::self_dot_simd(val_i, len_i);

        // Upper triangle
        for (Size j = i + 1; j < N_size; ++j) {
            auto values_j = scl::primary_values(matrix, static_cast<Index>(j));
            auto indices_j = scl::primary_indices(matrix, static_cast<Index>(j));

            const Index* SCL_RESTRICT idx_j = indices_j.ptr;
            const T* SCL_RESTRICT val_j = values_j.ptr;
            Size len_j = values_j.len;

            T dot = detail::sparse_dot_adaptive(idx_i, val_i, len_i, idx_j, val_j, len_j);

            row_ptr[j] = dot;
            output.ptr[j * N_size + i] = dot;
        }
    });
}

// =============================================================================
// SECTION 4: MappedVirtualSparse Gram Matrix
// =============================================================================

/// @brief Gram matrix for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void gram_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);

    SCL_CHECK_DIM(output.len >= N_size * N_size, "Gram: Output size mismatch");

    scl::threading::parallel_for(Size(0), N_size, [&](size_t i) {
        auto values_i = scl::primary_values(matrix, static_cast<Index>(i));
        auto indices_i = scl::primary_indices(matrix, static_cast<Index>(i));

        const Index* SCL_RESTRICT idx_i = indices_i.ptr;
        const T* SCL_RESTRICT val_i = values_i.ptr;
        Size len_i = values_i.len;

        T* row_ptr = output.ptr + (i * N_size);

        row_ptr[i] = detail::self_dot_simd(val_i, len_i);

        for (Size j = i + 1; j < N_size; ++j) {
            auto values_j = scl::primary_values(matrix, static_cast<Index>(j));
            auto indices_j = scl::primary_indices(matrix, static_cast<Index>(j));

            const Index* SCL_RESTRICT idx_j = indices_j.ptr;
            const T* SCL_RESTRICT val_j = values_j.ptr;
            Size len_j = values_j.len;

            T dot = detail::sparse_dot_adaptive(idx_i, val_i, len_i, idx_j, val_j, len_j);

            row_ptr[j] = dot;
            output.ptr[j * N_size + i] = dot;
        }
    });
}

// =============================================================================
// SECTION 5: Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate mapped implementation
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void gram_mapped_dispatch(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    gram_mapped(matrix, output);
}

} // namespace scl::kernel::gram::mapped
