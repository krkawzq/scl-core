#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Backend implementations
#include "scl/kernel/gram_fast_impl.hpp"

#include <algorithm>

// =============================================================================
/// @file gram.hpp
/// @brief Gram Matrix Computation (G = A * A^T or A^T * A)
///
/// ## Algorithm
///
/// For CSR: G = A * A^T (sample similarity matrix)
/// For CSC: G = A^T * A (feature correlation matrix)
///
/// G[i,j] = dot(row_i, row_j) for CSR
/// G[i,j] = dot(col_i, col_j) for CSC
///
/// ## Optimizations
///
/// 1. Adaptive Sparse Dot Product
///    - Linear merge: O(n + m) for similar lengths
///    - Binary search: O(n log m) for skewed lengths
///    - Galloping: O(n log(m/n)) for extreme skew
///
/// 2. Symmetric Output
///    - Only compute upper triangle
///    - Mirror to lower triangle
///
/// 3. SIMD Self Dot
///    - 4-way unrolled FMA for diagonal
///
/// ## Backend Dispatch
///
/// - MappedSparseLike -> gram_mapped_impl.hpp
/// - CustomSparseLike -> gram_fast_impl.hpp
/// - VirtualSparseLike -> gram_fast_impl.hpp
/// - Generic -> This file (fallback)
///
/// ## Complexity
///
/// Time: O(N^2 * avg_nnz) for N rows/cols
/// Space: O(N^2) for output matrix
///
/// ## Performance
///
/// Target: ~500M dot products/sec per core
// =============================================================================

namespace scl::kernel::gram {

// =============================================================================
// SECTION 1: Generic Implementation (Fallback)
// =============================================================================

namespace detail {

/// @brief Linear merge dot product O(n + m)
template <typename T>
SCL_FORCE_INLINE T dot_linear(
    const Index* SCL_RESTRICT idx1, const T* SCL_RESTRICT val1, Size n1,
    const Index* SCL_RESTRICT idx2, const T* SCL_RESTRICT val2, Size n2
) {
    T sum = T(0);
    Size i = 0, j = 0;

    while (i < n1 && j < n2) {
        Index r1 = idx1[i];
        Index r2 = idx2[j];

        if (SCL_LIKELY(r1 == r2)) {
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

/// @brief Binary search dot product O(n log m), n << m
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

/// @brief Adaptive dot product dispatcher
template <typename T>
SCL_FORCE_INLINE T dot_product(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
) {
    if (SCL_UNLIKELY(n1 == 0 || n2 == 0)) {
        return T(0);
    }

    constexpr Size RATIO_THRESHOLD = 32;

    if (n1 > n2) {
        std::swap(idx1, idx2);
        std::swap(val1, val2);
        std::swap(n1, n2);
    }

    if (n2 > n1 * RATIO_THRESHOLD) {
        return dot_binary(idx1, val1, n1, idx2, val2, n2);
    } else {
        return dot_linear(idx1, val1, n1, idx2, val2, n2);
    }
}

/// @brief SIMD self dot product
template <typename T>
SCL_FORCE_INLINE T self_dot_simd(const T* SCL_RESTRICT vals, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum = s::Zero(d);
    Size k = 0;

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

/// @brief Generic Gram matrix implementation
template <typename MatrixT>
    requires AnySparse<MatrixT>
void gram_generic(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    using T = typename MatrixT::ValueType;
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);

    SCL_CHECK_DIM(output.size() >= N_size * N_size, "Gram: Output size mismatch");

    scl::threading::parallel_for(Size(0), N_size, [&](size_t i) {
        const Index idx = static_cast<Index>(i);

        auto idx_i = scl::primary_indices(matrix, idx);
        auto val_i = scl::primary_values(matrix, idx);

        T* row_ptr = output.ptr + (i * N_size);

        // Diagonal: SIMD self dot
        row_ptr[i] = self_dot_simd(val_i.ptr, val_i.len);

        // Upper triangle
        for (Size j = i + 1; j < N_size; ++j) {
            const Index jdx = static_cast<Index>(j);
            auto idx_j = scl::primary_indices(matrix, jdx);
            auto val_j = scl::primary_values(matrix, jdx);

            T dot = dot_product(
                idx_i.ptr, val_i.ptr, idx_i.len,
                idx_j.ptr, val_j.ptr, idx_j.len
            );

            row_ptr[j] = dot;
            output.ptr[j * N_size + i] = dot;  // Mirror
        }
    });
}

} // namespace detail

// =============================================================================
// SECTION 2: Public API
// =============================================================================

/// @brief Compute Gram matrix (G = A * A^T for CSR, G = A^T * A for CSC)
///
/// @param matrix Input sparse matrix (any backend)
/// @param output Output dense Gram matrix [N x N], row-major, PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void gram(const MatrixT& matrix, Array<typename MatrixT::ValueType> output) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        fast::gram_fast<MatrixT, IsCSR>(matrix, output);
    } else {
        detail::gram_generic(matrix, output);
    }
}

} // namespace scl::kernel::gram
