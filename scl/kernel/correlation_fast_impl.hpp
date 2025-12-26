#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/sparse.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>
#include <algorithm>

// Mapped backend support
#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/correlation_mapped_impl.hpp"

// =============================================================================
/// @file correlation_fast_impl.hpp
/// @brief Extreme Performance Pearson Correlation
///
/// ## Key Optimizations
///
/// 1. Fused Statistics Computation
///    - Single-pass sum + sum_sq accumulation
///    - 4-way unrolled SIMD
///
/// 2. Direct Correlation Computation
///    - Sparse-sparse centered dot product
///    - Skip Gram matrix intermediate step
///
/// 3. Symmetric Matrix Optimization
///    - Only compute upper triangle
///    - Mirror to lower triangle
///
/// 4. Cache-Blocked Processing
///    - Process rows in chunks for L2 cache
///
/// Performance Target: 2-3x faster than Gram-based approach
// =============================================================================

namespace scl::kernel::correlation::fast {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size CHUNK_SIZE = 64;      // Rows per cache block
    constexpr Size STAT_CHUNK = 256;     // Rows per stats chunk
}

// =============================================================================
// SECTION 2: SIMD Statistics Computation
// =============================================================================

namespace detail {

/// @brief SIMD fused sum + sum_sq computation
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

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sq0 = s::Zero(d);
    auto v_sq1 = s::Zero(d);

    Size k = 0;

    // 4-way unrolled
    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        v_sum0 = s::Add(v_sum0, v0);
        v_sum1 = s::Add(v_sum1, v1);
        v_sum0 = s::Add(v_sum0, v2);
        v_sum1 = s::Add(v_sum1, v3);

        v_sq0 = s::MulAdd(v0, v0, v_sq0);
        v_sq1 = s::MulAdd(v1, v1, v_sq1);
        v_sq0 = s::MulAdd(v2, v2, v_sq0);
        v_sq1 = s::MulAdd(v3, v3, v_sq1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);
    auto v_sq = s::Add(v_sq0, v_sq1);

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
///
/// Computes: sum((x_i - mu_x) * (y_i - mu_y)) for matching indices
/// Note: For sparse data, zeros contribute: (-mu_x) * (-mu_y) = mu_x * mu_y
template <typename T>
SCL_FORCE_INLINE T sparse_centered_dot(
    const T* vals_a, const Index* inds_a, Size len_a, T mean_a,
    const T* vals_b, const Index* inds_b, Size len_b, T mean_b,
    Size total_dim
) {
    T dot = T(0);
    Size matched = 0;
    Size i = 0, j = 0;

    // Merge-based intersection
    while (i < len_a && j < len_b) {
        Index ia = inds_a[i];
        Index ib = inds_b[j];

        if (ia == ib) {
            T va = vals_a[i] - mean_a;
            T vb = vals_b[j] - mean_b;
            dot += va * vb;
            ++matched;
            ++i; ++j;
        } else if (ia < ib) {
            // a has value, b is zero
            T va = vals_a[i] - mean_a;
            T vb = -mean_b;
            dot += va * vb;
            ++i;
        } else {
            // b has value, a is zero
            T va = -mean_a;
            T vb = vals_b[j] - mean_b;
            dot += va * vb;
            ++j;
        }
    }

    // Remaining elements in a (b is zero)
    while (i < len_a) {
        dot += (vals_a[i] - mean_a) * (-mean_b);
        ++i;
    }

    // Remaining elements in b (a is zero)
    while (j < len_b) {
        dot += (-mean_a) * (vals_b[j] - mean_b);
        ++j;
    }

    // Contribution from positions where both are zero
    Size zeros_both = total_dim - len_a - len_b + matched;
    dot += static_cast<T>(zeros_both) * mean_a * mean_b;

    return dot;
}

} // namespace detail

// =============================================================================
// SECTION 3: Statistics Computation - CustomSparse
// =============================================================================

/// @brief Compute mean and inv_std for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void compute_stats_custom(
    const CustomSparse<T, IsCSR>& matrix,
    Array<T> out_means,
    Array<T> out_inv_stds
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size secondary_dim = static_cast<Size>(scl::secondary_size(matrix));
    const T inv_n = T(1) / static_cast<T>(secondary_dim);

    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(primary_dim), "Means size mismatch");
    SCL_CHECK_DIM(out_inv_stds.len >= static_cast<Size>(primary_dim), "Inv_stds size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        T sum, sq_sum;
        detail::compute_sum_sq_simd(matrix.data + start, len, sum, sq_sum);

        T mean = sum * inv_n;
        T var = (sq_sum * inv_n) - (mean * mean);
        if (var < T(0)) var = T(0);

        out_means[p] = mean;
        out_inv_stds[p] = (var > T(0)) ? (T(1) / std::sqrt(var)) : T(0);
    });
}

// =============================================================================
// SECTION 4: Statistics Computation - VirtualSparse
// =============================================================================

/// @brief Compute mean and inv_std for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void compute_stats_virtual(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<T> out_means,
    Array<T> out_inv_stds
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size secondary_dim = static_cast<Size>(scl::secondary_size(matrix));
    const T inv_n = T(1) / static_cast<T>(secondary_dim);

    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(primary_dim), "Means size mismatch");
    SCL_CHECK_DIM(out_inv_stds.len >= static_cast<Size>(primary_dim), "Inv_stds size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Size len = static_cast<Size>(matrix.lengths[p]);
        const T* vals = static_cast<const T*>(matrix.data_ptrs[p]);

        T sum, sq_sum;
        detail::compute_sum_sq_simd(vals, len, sum, sq_sum);

        T mean = sum * inv_n;
        T var = (sq_sum * inv_n) - (mean * mean);
        if (var < T(0)) var = T(0);

        out_means[p] = mean;
        out_inv_stds[p] = (var > T(0)) ? (T(1) / std::sqrt(var)) : T(0);
    });
}

// =============================================================================
// SECTION 5: Correlation Matrix - CustomSparse
// =============================================================================

/// @brief Compute full correlation matrix for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void pearson_custom(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const T> means,
    Array<const T> inv_stds,
    Array<T> output
) {
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);
    const Size M = static_cast<Size>(scl::secondary_size(matrix));
    const T inv_m = T(1) / static_cast<T>(M);

    SCL_CHECK_DIM(output.len >= N_size * N_size, "Output size mismatch");

    // Process upper triangle in chunks
    const Size n_chunks = (N_size + config::CHUNK_SIZE - 1) / config::CHUNK_SIZE;

    scl::threading::parallel_for(Size(0), n_chunks, [&](size_t chunk_idx) {
        Size i_start = chunk_idx * config::CHUNK_SIZE;
        Size i_end = std::min(i_start + config::CHUNK_SIZE, N_size);

        for (Size i = i_start; i < i_end; ++i) {
            Index idx_i = static_cast<Index>(i);
            Index start_i = matrix.indptr[idx_i];
            Index end_i = matrix.indptr[idx_i + 1];
            Size len_i = static_cast<Size>(end_i - start_i);

            const T* vals_i = matrix.data + start_i;
            const Index* inds_i = matrix.indices + start_i;
            T mean_i = means[i];
            T inv_std_i = inv_stds[i];

            T* row_ptr = output.ptr + i * N_size;

            // Diagonal: self-correlation = 1
            row_ptr[i] = (inv_std_i > T(0)) ? T(1) : T(0);

            // Upper triangle
            for (Size j = i + 1; j < N_size; ++j) {
                Index idx_j = static_cast<Index>(j);
                Index start_j = matrix.indptr[idx_j];
                Index end_j = matrix.indptr[idx_j + 1];
                Size len_j = static_cast<Size>(end_j - start_j);

                const T* vals_j = matrix.data + start_j;
                const Index* inds_j = matrix.indices + start_j;
                T mean_j = means[j];
                T inv_std_j = inv_stds[j];

                // Centered dot product
                T cov = detail::sparse_centered_dot(
                    vals_i, inds_i, len_i, mean_i,
                    vals_j, inds_j, len_j, mean_j,
                    M
                ) * inv_m;

                // Correlation
                T corr = cov * inv_std_i * inv_std_j;

                // Clamp to [-1, 1]
                if (corr > T(1)) corr = T(1);
                if (corr < T(-1)) corr = T(-1);
                if (inv_std_i == T(0) || inv_std_j == T(0)) corr = T(0);

                // Store symmetric
                row_ptr[j] = corr;
                output.ptr[j * N_size + i] = corr;
            }
        }
    });
}

// =============================================================================
// SECTION 6: Correlation Matrix - VirtualSparse
// =============================================================================

/// @brief Compute full correlation matrix for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void pearson_virtual(
    const VirtualSparse<T, IsCSR>& matrix,
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
            Size len_i = static_cast<Size>(matrix.lengths[i]);
            const T* vals_i = static_cast<const T*>(matrix.data_ptrs[i]);
            const Index* inds_i = static_cast<const Index*>(matrix.indices_ptrs[i]);
            T mean_i = means[i];
            T inv_std_i = inv_stds[i];

            T* row_ptr = output.ptr + i * N_size;

            row_ptr[i] = (inv_std_i > T(0)) ? T(1) : T(0);

            for (Size j = i + 1; j < N_size; ++j) {
                Size len_j = static_cast<Size>(matrix.lengths[j]);
                const T* vals_j = static_cast<const T*>(matrix.data_ptrs[j]);
                const Index* inds_j = static_cast<const Index*>(matrix.indices_ptrs[j]);
                T mean_j = means[j];
                T inv_std_j = inv_stds[j];

                T cov = detail::sparse_centered_dot(
                    vals_i, inds_i, len_i, mean_i,
                    vals_j, inds_j, len_j, mean_j,
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
// SECTION 7: Unified Dispatchers
// =============================================================================

/// @brief Statistics dispatcher
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void compute_stats_fast(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> out_means,
    Array<typename MatrixT::ValueType> out_inv_stds
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::correlation::mapped::compute_stats_mapped_dispatch<MatrixT, IsCSR>(
            matrix, out_means, out_inv_stds
        );
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        compute_stats_custom(matrix, out_means, out_inv_stds);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        compute_stats_virtual(matrix, out_means, out_inv_stds);
    }
}

/// @brief Full correlation matrix dispatcher
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void pearson_fast(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(matrix);
    const Size N = static_cast<Size>(primary_dim);

    // Compute statistics
    std::vector<T> means(N);
    std::vector<T> inv_stds(N);
    compute_stats_fast<MatrixT, IsCSR>(matrix, Array<T>(means.data(), N), Array<T>(inv_stds.data(), N));

    // Dispatch to correlation computation
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::correlation::mapped::pearson_mapped_dispatch<MatrixT, IsCSR>(
            matrix,
            Array<const T>(means.data(), N),
            Array<const T>(inv_stds.data(), N),
            output
        );
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        pearson_custom(matrix, Array<const T>(means.data(), N), Array<const T>(inv_stds.data(), N), output);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        pearson_virtual(matrix, Array<const T>(means.data(), N), Array<const T>(inv_stds.data(), N), output);
    }
}

} // namespace scl::kernel::correlation::fast
