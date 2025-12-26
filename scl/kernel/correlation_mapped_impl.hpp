#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file correlation_mapped_impl.hpp
/// @brief Mapped Backend Correlation Statistics
///
/// Correlation statistics are read-only operations, can stream directly from
/// mapped data. No materialization needed.
///
/// Key optimizations:
/// - Single-pass fused mean/variance computation
/// - Chunk-based processing for cache efficiency
// =============================================================================

namespace scl::kernel::correlation::mapped {

// =============================================================================
// MappedCustomSparse Correlation Statistics
// =============================================================================

/// @brief Compute correlation statistics for MappedCustomSparse
///
/// Single-pass streaming algorithm with fused mean + variance accumulation.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_stats_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_inv_stds
) {
    const Index n_primary = scl::primary_size(matrix);
    const Size secondary_dim = static_cast<Size>(scl::secondary_size(matrix));
    const Real inv_n = static_cast<Real>(1.0) / static_cast<Real>(secondary_dim);

    SCL_CHECK_DIM(out_means.len == static_cast<Size>(n_primary), "Means size mismatch");
    SCL_CHECK_DIM(out_inv_stds.len == static_cast<Size>(n_primary), "Inv_stds size mismatch");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    // Process in chunks for cache efficiency
    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (n_primary + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), n_primary);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto values = scl::primary_values(matrix, p);

            const T* SCL_RESTRICT vals = values.ptr;
            const Size len = values.len;

            // Fused SIMD accumulation
            auto v_sum = s::Zero(d);
            auto v_sq_sum = s::Zero(d);

            Size k = 0;

            // 4-way unrolled
            for (; k + 4 * lanes <= len; k += 4 * lanes) {
                auto v0 = s::Load(d, vals + k + 0 * lanes);
                auto v1 = s::Load(d, vals + k + 1 * lanes);
                auto v2 = s::Load(d, vals + k + 2 * lanes);
                auto v3 = s::Load(d, vals + k + 3 * lanes);

                v_sum = s::Add(v_sum, v0);
                v_sum = s::Add(v_sum, v1);
                v_sum = s::Add(v_sum, v2);
                v_sum = s::Add(v_sum, v3);

                v_sq_sum = s::MulAdd(v0, v0, v_sq_sum);
                v_sq_sum = s::MulAdd(v1, v1, v_sq_sum);
                v_sq_sum = s::MulAdd(v2, v2, v_sq_sum);
                v_sq_sum = s::MulAdd(v3, v3, v_sq_sum);
            }

            for (; k + lanes <= len; k += lanes) {
                auto v = s::Load(d, vals + k);
                v_sum = s::Add(v_sum, v);
                v_sq_sum = s::MulAdd(v, v, v_sq_sum);
            }

            Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
            Real sq_sum = s::GetLane(s::SumOfLanes(d, v_sq_sum));

            for (; k < len; ++k) {
                Real v = static_cast<Real>(vals[k]);
                sum += v;
                sq_sum += v * v;
            }

            Real mean = sum * inv_n;
            Real var = (sq_sum * inv_n) - (mean * mean);
            if (var < 0) var = 0;

            out_means[p] = mean;
            out_inv_stds[p] = (var > 0) ? (1.0 / std::sqrt(var)) : 0.0;
        });
    }
}

/// @brief Compute centered correlation matrix for MappedCustomSparse
///
/// Uses precomputed statistics for centering.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void correlation_matrix_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Real> means,
    Array<const Real> inv_stds,
    Array<Real> output
) {
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);
    const Size M = static_cast<Size>(scl::secondary_size(matrix));
    const Real inv_m = 1.0 / static_cast<Real>(M);

    SCL_CHECK_DIM(output.len == N_size * N_size, "Output size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    constexpr Size CHUNK_SIZE = 64;
    const Size n_chunks = (N_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Size chunk_start = chunk_id * CHUNK_SIZE;
        Size chunk_end = std::min(chunk_start + CHUNK_SIZE, N_size);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](size_t i) {
            auto values_i = scl::primary_values(matrix, static_cast<Index>(i));
            auto indices_i = scl::primary_indices(matrix, static_cast<Index>(i));

            Real mean_i = means[i];
            Real inv_std_i = inv_stds[i];

            Real* row_ptr = output.ptr + (i * N_size);

            // Diagonal is always 1 (self-correlation)
            row_ptr[i] = 1.0;

            // Upper triangle
            for (Size j = i + 1; j < N_size; ++j) {
                auto values_j = scl::primary_values(matrix, static_cast<Index>(j));
                auto indices_j = scl::primary_indices(matrix, static_cast<Index>(j));

                Real mean_j = means[j];
                Real inv_std_j = inv_stds[j];

                // Sparse dot product with centering
                Real sum = 0.0;
                Size pi = 0, pj = 0;

                while (pi < values_i.len && pj < values_j.len) {
                    Index idx_i = indices_i.ptr[pi];
                    Index idx_j = indices_j.ptr[pj];

                    if (idx_i == idx_j) {
                        Real vi = (static_cast<Real>(values_i.ptr[pi]) - mean_i) * inv_std_i;
                        Real vj = (static_cast<Real>(values_j.ptr[pj]) - mean_j) * inv_std_j;
                        sum += vi * vj;
                        ++pi; ++pj;
                    } else if (idx_i < idx_j) {
                        // Zero contribution from centered mean
                        sum += (-mean_i * inv_std_i) * (-mean_j * inv_std_j);
                        ++pi;
                    } else {
                        sum += (-mean_i * inv_std_i) * (-mean_j * inv_std_j);
                        ++pj;
                    }
                }

                // Account for remaining zeros
                Size remaining = M - std::max(values_i.len, values_j.len);
                sum += remaining * mean_i * inv_std_i * mean_j * inv_std_j;

                Real corr = sum * inv_m;
                row_ptr[j] = corr;
                output.ptr[j * N_size + i] = corr;  // Mirror
            }
        });
    }
}

// =============================================================================
// MappedVirtualSparse Correlation Statistics
// =============================================================================

/// @brief Compute correlation statistics for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_stats_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_inv_stds
) {
    const Index n_primary = scl::primary_size(matrix);
    const Size secondary_dim = static_cast<Size>(scl::secondary_size(matrix));
    const Real inv_n = static_cast<Real>(1.0) / static_cast<Real>(secondary_dim);

    SCL_CHECK_DIM(out_means.len == static_cast<Size>(n_primary), "Means size mismatch");
    SCL_CHECK_DIM(out_inv_stds.len == static_cast<Size>(n_primary), "Inv_stds size mismatch");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (n_primary + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), n_primary);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto values = scl::primary_values(matrix, p);

            const T* SCL_RESTRICT vals = values.ptr;
            const Size len = values.len;

            auto v_sum = s::Zero(d);
            auto v_sq_sum = s::Zero(d);

            Size k = 0;

            for (; k + lanes <= len; k += lanes) {
                auto v = s::Load(d, vals + k);
                v_sum = s::Add(v_sum, v);
                v_sq_sum = s::MulAdd(v, v, v_sq_sum);
            }

            Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
            Real sq_sum = s::GetLane(s::SumOfLanes(d, v_sq_sum));

            for (; k < len; ++k) {
                Real v = static_cast<Real>(vals[k]);
                sum += v;
                sq_sum += v * v;
            }

            Real mean = sum * inv_n;
            Real var = (sq_sum * inv_n) - (mean * mean);
            if (var < 0) var = 0;

            out_means[p] = mean;
            out_inv_stds[p] = (var > 0) ? (1.0 / std::sqrt(var)) : 0.0;
        });
    }
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate mapped fast path
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void compute_stats_mapped_dispatch(
    const MatrixT& matrix,
    Array<Real> out_means,
    Array<Real> out_inv_stds
) {
    compute_stats_mapped(matrix, out_means, out_inv_stds);
}

} // namespace scl::kernel::correlation::mapped
