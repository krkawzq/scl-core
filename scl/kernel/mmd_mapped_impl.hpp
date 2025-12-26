#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file mmd_mapped_impl.hpp
/// @brief Mapped Backend MMD Operations
///
/// MMD (Maximum Mean Discrepancy) operations can stream directly from
/// mapped data for read-only kernel evaluation.
///
/// Key optimizations:
/// - Chunk-based processing for cache efficiency
/// - Streaming reads from mapped memory
/// - SIMD exp computation
// =============================================================================

namespace scl::kernel::mmd::mapped {

// =============================================================================
// Helper Functions (from mmd_fast_impl.hpp)
// =============================================================================

namespace detail {

/// @brief Ultra-fast unary exp sum
template <typename T>
SCL_FORCE_INLINE void unary_exp_sum(
    const T* SCL_RESTRICT vals,
    Size nnz,
    T gamma,
    T* cache,
    T& out_sum
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_gamma = s::Set(d, gamma);

    auto v_sum = s::Zero(d);
    size_t k = 0;

    for (; k + lanes <= nnz; k += lanes) {
        auto v = s::Load(d, vals + k);
        auto v_sq = s::Mul(v, v);
        auto v_exp = s::Exp(d, s::Neg(s::Mul(v_sq, v_gamma)));
        s::Store(v_exp, d, cache + k);
        v_sum = s::Add(v_sum, v_exp);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < nnz; ++k) {
        T val = vals[k];
        T exp_term = std::exp(-gamma * val * val);
        cache[k] = exp_term;
        sum += exp_term;
    }

    out_sum = sum;
}

/// @brief Cross kernel sum computation
template <typename T>
SCL_FORCE_INLINE void cross_kernel_sum(
    const T* SCL_RESTRICT vals_x, Size nnz_x,
    const T* SCL_RESTRICT vals_y, Size nnz_y,
    Size N_x, Size N_y,
    T gamma,
    T sum_x_unary,
    T sum_y_unary,
    T& out_sum
) {
    const Size zeros_x = N_x - nnz_x;
    const Size zeros_y = N_y - nnz_y;

    T sum = static_cast<T>(0.0);

    // Zero-Zero
    sum += static_cast<T>(zeros_x * zeros_y);

    // Zero-Val interactions
    if (zeros_x > 0) {
        sum += static_cast<T>(zeros_x) * sum_y_unary;
    }
    if (zeros_y > 0) {
        sum += static_cast<T>(zeros_y) * sum_x_unary;
    }

    // Val-Val with SIMD
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_gamma = s::Set(d, gamma);

    T cross_sum = static_cast<T>(0.0);

    for (size_t i = 0; i < nnz_x; ++i) {
        const T xi = vals_x[i];
        const auto v_xi = s::Set(d, xi);

        auto v_row_sum = s::Zero(d);
        size_t j = 0;

        for (; j + lanes <= nnz_y; j += lanes) {
            auto v_yj = s::Load(d, vals_y + j);
            auto v_diff = s::Sub(v_xi, v_yj);
            auto v_sq = s::Mul(v_diff, v_diff);
            auto v_exp = s::Exp(d, s::Neg(s::Mul(v_sq, v_gamma)));
            v_row_sum = s::Add(v_row_sum, v_exp);
        }

        cross_sum += s::GetLane(s::SumOfLanes(d, v_row_sum));

        for (; j < nnz_y; ++j) {
            T diff = xi - vals_y[j];
            cross_sum += std::exp(-gamma * diff * diff);
        }
    }

    sum += cross_sum;
    out_sum = sum;
}

} // namespace detail

// =============================================================================
// MappedCustomSparse MMD
// =============================================================================

/// @brief Compute MMD row kernel sums for MappedCustomSparse
///
/// Streaming algorithm - reads row data once for unary kernel computation.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_row_unary_kernels_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    T gamma,
    Array<T> out_sums,
    std::vector<std::vector<T>>& cache
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_sums.len == static_cast<Size>(n_primary), "Output size mismatch");

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    // Resize cache
    cache.resize(n_primary);

    // Process in chunks for cache efficiency
    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (n_primary + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), n_primary);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto values = scl::primary_values(matrix, p);
            Size nnz = values.len;

            cache[p].resize(nnz);

            if (nnz == 0) {
                out_sums[p] = static_cast<T>(0.0);
                return;
            }

            T sum;
            detail::unary_exp_sum(values.ptr, nnz, gamma, cache[p].data(), sum);
            out_sums[p] = sum;
        });
    }
}

/// @brief Compute MMD between two rows for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
T compute_row_pair_kernel_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Index row_x,
    Index row_y,
    T gamma,
    T sum_x_unary,
    T sum_y_unary
) {
    auto values_x = scl::primary_values(matrix, row_x);
    auto values_y = scl::primary_values(matrix, row_y);

    Size N = static_cast<Size>(scl::secondary_size(matrix));

    T cross_sum;
    detail::cross_kernel_sum(
        values_x.ptr, values_x.len,
        values_y.ptr, values_y.len,
        N, N,
        gamma,
        sum_x_unary,
        sum_y_unary,
        cross_sum
    );

    return cross_sum;
}

// =============================================================================
// MappedVirtualSparse MMD
// =============================================================================

/// @brief Compute MMD row kernel sums for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_row_unary_kernels_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    T gamma,
    Array<T> out_sums,
    std::vector<std::vector<T>>& cache
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_sums.len == static_cast<Size>(n_primary), "Output size mismatch");

    cache.resize(n_primary);

    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (n_primary + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), n_primary);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto values = scl::primary_values(matrix, p);
            Size nnz = values.len;

            cache[p].resize(nnz);

            if (nnz == 0) {
                out_sums[p] = static_cast<T>(0.0);
                return;
            }

            T sum;
            detail::unary_exp_sum(values.ptr, nnz, gamma, cache[p].data(), sum);
            out_sums[p] = sum;
        });
    }
}

} // namespace scl::kernel::mmd::mapped
