#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Backend implementations
#include "scl/kernel/correlation_fast_impl.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file correlation.hpp
/// @brief Pearson Correlation Computation
///
/// ## Algorithm
///
/// For sparse matrices, Pearson correlation is computed as:
///
///     corr(x, y) = cov(x, y) / (std(x) * std(y))
///
/// Where covariance accounts for zero elements:
///
///     cov(x, y) = E[(x - mu_x)(y - mu_y)]
///              = (1/N) * sum_i (x_i - mu_x)(y_i - mu_y)
///
/// ## Optimizations
///
/// 1. Direct Sparse Computation
///    - No Gram matrix intermediate step
///    - Sparse-sparse centered dot product
///
/// 2. Symmetric Matrix
///    - Only compute upper triangle
///    - Mirror to lower triangle
///
/// 3. Backend Dispatch
///    - MappedSparseLike -> correlation_mapped_impl.hpp
///    - CustomSparseLike -> correlation_fast_impl.hpp
///    - VirtualSparseLike -> correlation_fast_impl.hpp
///    - Generic -> This file (fallback)
///
/// ## Complexity
///
/// Time: O(N^2 * avg_nnz) for sparse data
/// Space: O(N) for statistics + O(N^2) for output
///
/// ## Performance
///
/// ~1000 features: target < 100ms (16 cores)
// =============================================================================

namespace scl::kernel::correlation {

// =============================================================================
// SECTION 1: Generic Implementation (Fallback)
// =============================================================================

namespace detail {

/// @brief Generic statistics computation
template <typename MatrixT>
    requires AnySparse<MatrixT>
void compute_stats_generic(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> out_means,
    Array<typename MatrixT::ValueType> out_inv_stds
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(matrix);
    const Size secondary_dim = static_cast<Size>(scl::secondary_size(matrix));
    const T inv_n = T(1) / static_cast<T>(secondary_dim);

    SCL_CHECK_DIM(out_means.size() >= static_cast<Size>(primary_dim), "Means size mismatch");
    SCL_CHECK_DIM(out_inv_stds.size() >= static_cast<Size>(primary_dim), "Inv_stds size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        auto vals = scl::primary_values(matrix, idx);

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        auto v_sum = s::Zero(d);
        auto v_sq_sum = s::Zero(d);
        Size k = 0;

        for (; k + lanes <= vals.len; k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v_sum = s::Add(v_sum, v);
            v_sq_sum = s::MulAdd(v, v, v_sq_sum);
        }

        T sum = s::GetLane(s::SumOfLanes(d, v_sum));
        T sq_sum = s::GetLane(s::SumOfLanes(d, v_sq_sum));

        for (; k < vals.len; ++k) {
            T v = vals[k];
            sum += v;
            sq_sum += v * v;
        }

        T mean = sum * inv_n;
        T var = (sq_sum * inv_n) - (mean * mean);
        if (var < T(0)) var = T(0);

        out_means[p] = mean;
        out_inv_stds[p] = (var > T(0)) ? (T(1) / std::sqrt(var)) : T(0);
    });
}

/// @brief Sparse-sparse centered dot product
template <typename T>
T sparse_centered_dot(
    const T* vals_a, const Index* inds_a, Size len_a, T mean_a,
    const T* vals_b, const Index* inds_b, Size len_b, T mean_b,
    Size total_dim
) {
    T dot = T(0);
    Size matched = 0;
    Size i = 0, j = 0;

    while (i < len_a && j < len_b) {
        Index ia = inds_a[i];
        Index ib = inds_b[j];

        if (ia == ib) {
            dot += (vals_a[i] - mean_a) * (vals_b[j] - mean_b);
            ++matched;
            ++i; ++j;
        } else if (ia < ib) {
            dot += (vals_a[i] - mean_a) * (-mean_b);
            ++i;
        } else {
            dot += (-mean_a) * (vals_b[j] - mean_b);
            ++j;
        }
    }

    while (i < len_a) {
        dot += (vals_a[i] - mean_a) * (-mean_b);
        ++i;
    }

    while (j < len_b) {
        dot += (-mean_a) * (vals_b[j] - mean_b);
        ++j;
    }

    Size zeros_both = total_dim - len_a - len_b + matched;
    dot += static_cast<T>(zeros_both) * mean_a * mean_b;

    return dot;
}

/// @brief Generic correlation matrix computation
template <typename MatrixT>
    requires AnySparse<MatrixT>
void pearson_generic(
    const MatrixT& matrix,
    Array<const typename MatrixT::ValueType> means,
    Array<const typename MatrixT::ValueType> inv_stds,
    Array<typename MatrixT::ValueType> output
) {
    using T = typename MatrixT::ValueType;
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);
    const Size M = static_cast<Size>(scl::secondary_size(matrix));
    const T inv_m = T(1) / static_cast<T>(M);

    SCL_CHECK_DIM(output.len >= N_size * N_size, "Output size mismatch");

    constexpr Size CHUNK_SIZE = 64;
    const Size n_chunks = (N_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

    scl::threading::parallel_for(Size(0), n_chunks, [&](size_t chunk_idx) {
        Size i_start = chunk_idx * CHUNK_SIZE;
        Size i_end = std::min(i_start + CHUNK_SIZE, N_size);

        for (Size i = i_start; i < i_end; ++i) {
            auto vals_i = scl::primary_values(matrix, static_cast<Index>(i));
            auto inds_i = scl::primary_indices(matrix, static_cast<Index>(i));
            T mean_i = means[i];
            T inv_std_i = inv_stds[i];

            T* row_ptr = output.ptr + i * N_size;

            // Diagonal
            row_ptr[i] = (inv_std_i > T(0)) ? T(1) : T(0);

            // Upper triangle
            for (Size j = i + 1; j < N_size; ++j) {
                auto vals_j = scl::primary_values(matrix, static_cast<Index>(j));
                auto inds_j = scl::primary_indices(matrix, static_cast<Index>(j));
                T mean_j = means[j];
                T inv_std_j = inv_stds[j];

                T cov = sparse_centered_dot(
                    vals_i.ptr, inds_i.ptr, vals_i.len, mean_i,
                    vals_j.ptr, inds_j.ptr, vals_j.len, mean_j,
                    M
                ) * inv_m;

                T corr = cov * inv_std_i * inv_std_j;

                if (corr > T(1)) corr = T(1);
                if (corr < T(-1)) corr = T(-1);
                if (inv_std_i == T(0) || inv_std_j == T(0)) corr = T(0);

                row_ptr[j] = corr;
                output.ptr[j * N_size + i] = corr;
            }
        }
    });
}

} // namespace detail

// =============================================================================
// SECTION 2: Public API
// =============================================================================

/// @brief Compute Pearson correlation matrix
///
/// Computes pairwise Pearson correlation coefficients between all rows
/// (for CSR) or columns (for CSC) of the sparse matrix.
///
/// @param matrix Input sparse matrix (any backend)
/// @param output Output dense correlation matrix [size = primary_dim^2], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void pearson(const MatrixT& matrix, Array<typename MatrixT::ValueType> output) {
    using T = typename MatrixT::ValueType;
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    const Index primary_dim = scl::primary_size(matrix);
    const Size N = static_cast<Size>(primary_dim);

    SCL_CHECK_DIM(output.size() >= N * N, "Pearson: Output size mismatch");

    // Dispatch to optimized backend
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        // Use fast implementation
        fast::pearson_fast<MatrixT, IsCSR>(matrix, output);
    } else {
        // Generic fallback
        std::vector<T> means(N);
        std::vector<T> inv_stds(N);
        detail::compute_stats_generic(matrix, Array<T>(means.data(), N), Array<T>(inv_stds.data(), N));
        detail::pearson_generic(matrix, Array<const T>(means.data(), N), Array<const T>(inv_stds.data(), N), output);
    }
}

/// @brief Compute statistics only (mean, inverse std)
///
/// Useful when you need statistics for other purposes or
/// want to compute correlation incrementally.
///
/// @param matrix Input sparse matrix
/// @param out_means Output means [size = primary_dim], PRE-ALLOCATED
/// @param out_inv_stds Output inverse stds [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void compute_stats(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> out_means,
    Array<typename MatrixT::ValueType> out_inv_stds
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        fast::compute_stats_fast<MatrixT, IsCSR>(matrix, out_means, out_inv_stds);
    } else {
        detail::compute_stats_generic(matrix, out_means, out_inv_stds);
    }
}

} // namespace scl::kernel::correlation
