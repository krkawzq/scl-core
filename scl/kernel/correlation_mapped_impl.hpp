#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>
#include <algorithm>

// =============================================================================
/// @file correlation_mapped_impl.hpp
/// @brief Pearson Correlation for Memory-Mapped Sparse Matrices
///
/// ## Key Optimizations
///
/// 1. Streaming Access Pattern
///    - Sequential row access for cache efficiency
///    - Prefetch hints for OS page cache
///
/// 2. Direct Correlation Computation
///    - Sparse-sparse centered dot product
///    - No Gram matrix intermediate step
///
/// 3. Symmetric Matrix Optimization
///    - Only compute upper triangle
///    - Mirror to lower triangle
///
/// Performance: Near-RAM performance for cached data
// =============================================================================

namespace scl::kernel::correlation::mapped {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size CHUNK_SIZE = 64;
    constexpr Size STAT_CHUNK = 256;
}

// =============================================================================
// SECTION 2: Utilities
// =============================================================================

namespace detail {

/// @brief SIMD fused sum + sum_sq
template <typename T>
SCL_FORCE_INLINE void compute_sum_sq_simd(
    const T* SCL_RESTRICT vals,
    Size len,
    T& out_sum,
    T& out_sq_sum
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum = s::Zero(d);
    auto v_sq = s::Zero(d);

    Size k = 0;
    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::Add(v_sum, v);
        v_sq = s::MulAdd(v, v, v_sq);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));
    T sq_sum = s::GetLane(s::SumOfLanes(d, v_sq));

    for (; k < len; ++k) {
        T v = vals[k];
        sum += v;
        sq_sum += v * v;
    }

    out_sum = sum;
    out_sq_sum = sq_sum;
}

/// @brief Sparse-sparse centered dot product
template <typename T>
SCL_FORCE_INLINE T sparse_centered_dot(
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

} // namespace detail

// =============================================================================
// SECTION 3: Statistics - MappedCustomSparse
// =============================================================================

/// @brief Compute statistics for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_stats_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> out_means,
    Array<T> out_inv_stds
) {
    const Index n_primary = scl::primary_size(matrix);
    const Size secondary_dim = static_cast<Size>(scl::secondary_size(matrix));
    const T inv_n = T(1) / static_cast<T>(secondary_dim);

    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(n_primary), "Means size mismatch");
    SCL_CHECK_DIM(out_inv_stds.len >= static_cast<Size>(n_primary), "Inv_stds size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        auto values = scl::primary_values(matrix, p);

        T sum, sq_sum;
        detail::compute_sum_sq_simd(values.ptr, values.len, sum, sq_sum);

        T mean = sum * inv_n;
        T var = (sq_sum * inv_n) - (mean * mean);
        if (var < T(0)) var = T(0);

        out_means[p] = mean;
        out_inv_stds[p] = (var > T(0)) ? (T(1) / std::sqrt(var)) : T(0);
    });
}

/// @brief Compute statistics for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_stats_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<T> out_means,
    Array<T> out_inv_stds
) {
    const Index n_primary = scl::primary_size(matrix);
    const Size secondary_dim = static_cast<Size>(scl::secondary_size(matrix));
    const T inv_n = T(1) / static_cast<T>(secondary_dim);

    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(n_primary), "Means size mismatch");
    SCL_CHECK_DIM(out_inv_stds.len >= static_cast<Size>(n_primary), "Inv_stds size mismatch");

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        auto values = scl::primary_values(matrix, p);

        T sum, sq_sum;
        detail::compute_sum_sq_simd(values.ptr, values.len, sum, sq_sum);

        T mean = sum * inv_n;
        T var = (sq_sum * inv_n) - (mean * mean);
        if (var < T(0)) var = T(0);

        out_means[p] = mean;
        out_inv_stds[p] = (var > T(0)) ? (T(1) / std::sqrt(var)) : T(0);
    });
}

// =============================================================================
// SECTION 4: Pearson Correlation - MappedCustomSparse
// =============================================================================

/// @brief Compute correlation matrix for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void pearson_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const T> means,
    Array<const T> inv_stds,
    Array<T> output
) {
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);
    const Size M = static_cast<Size>(scl::secondary_size(matrix));
    const T inv_m = T(1) / static_cast<T>(M);

    SCL_CHECK_DIM(output.len >= N_size * N_size, "Output size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    const Size n_chunks = (N_size + config::CHUNK_SIZE - 1) / config::CHUNK_SIZE;

    scl::threading::parallel_for(Size(0), n_chunks, [&](size_t chunk_idx) {
        Size i_start = chunk_idx * config::CHUNK_SIZE;
        Size i_end = std::min(i_start + config::CHUNK_SIZE, N_size);

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

                T cov = detail::sparse_centered_dot(
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

/// @brief Compute correlation matrix for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void pearson_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const T> means,
    Array<const T> inv_stds,
    Array<T> output
) {
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);
    const Size M = static_cast<Size>(scl::secondary_size(matrix));
    const T inv_m = T(1) / static_cast<T>(M);

    SCL_CHECK_DIM(output.len >= N_size * N_size, "Output size mismatch");

    const Size n_chunks = (N_size + config::CHUNK_SIZE - 1) / config::CHUNK_SIZE;

    scl::threading::parallel_for(Size(0), n_chunks, [&](size_t chunk_idx) {
        Size i_start = chunk_idx * config::CHUNK_SIZE;
        Size i_end = std::min(i_start + config::CHUNK_SIZE, N_size);

        for (Size i = i_start; i < i_end; ++i) {
            auto vals_i = scl::primary_values(matrix, static_cast<Index>(i));
            auto inds_i = scl::primary_indices(matrix, static_cast<Index>(i));
            T mean_i = means[i];
            T inv_std_i = inv_stds[i];

            T* row_ptr = output.ptr + i * N_size;

            row_ptr[i] = (inv_std_i > T(0)) ? T(1) : T(0);

            for (Size j = i + 1; j < N_size; ++j) {
                auto vals_j = scl::primary_values(matrix, static_cast<Index>(j));
                auto inds_j = scl::primary_indices(matrix, static_cast<Index>(j));
                T mean_j = means[j];
                T inv_std_j = inv_stds[j];

                T cov = detail::sparse_centered_dot(
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

// =============================================================================
// SECTION 5: Unified Dispatchers
// =============================================================================

/// @brief Statistics dispatcher
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void compute_stats_mapped_dispatch(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> out_means,
    Array<typename MatrixT::ValueType> out_inv_stds
) {
    compute_stats_mapped(matrix, out_means, out_inv_stds);
}

/// @brief Correlation dispatcher
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void pearson_mapped_dispatch(
    const MatrixT& matrix,
    Array<const typename MatrixT::ValueType> means,
    Array<const typename MatrixT::ValueType> inv_stds,
    Array<typename MatrixT::ValueType> output
) {
    pearson_mapped(matrix, means, inv_stds, output);
}

} // namespace scl::kernel::correlation::mapped
