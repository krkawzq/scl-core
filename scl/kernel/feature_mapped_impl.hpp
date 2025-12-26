#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file feature_mapped_impl.hpp
/// @brief Mapped Backend Feature Statistics
///
/// Feature statistics are read-only operations, can stream directly from
/// mapped data. No materialization needed.
///
/// Supported operations:
/// - Standard moments: mean and variance per row (single pass)
// =============================================================================

namespace scl::kernel::feature::mapped {

// =============================================================================
// MappedCustomSparse Feature Statistics
// =============================================================================

/// @brief Compute standard moments (mean, variance) for MappedCustomSparse
///
/// Single-pass streaming algorithm with fused sum + sum_sq accumulation.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void standard_moments_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    const Index n_primary = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real denom = N - static_cast<Real>(ddof);

    SCL_CHECK_DIM(out_means.len == static_cast<Size>(n_primary), "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len == static_cast<Size>(n_primary), "Vars size mismatch");

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

            // 4-way unrolled SIMD
            auto v_sum0 = s::Zero(d);
            auto v_sum1 = s::Zero(d);
            auto v_sum2 = s::Zero(d);
            auto v_sum3 = s::Zero(d);

            auto v_ssq0 = s::Zero(d);
            auto v_ssq1 = s::Zero(d);
            auto v_ssq2 = s::Zero(d);
            auto v_ssq3 = s::Zero(d);

            Size k = 0;
            for (; k + 4 * lanes <= len; k += 4 * lanes) {
                auto v0 = s::Load(d, vals + k + 0 * lanes);
                auto v1 = s::Load(d, vals + k + 1 * lanes);
                auto v2 = s::Load(d, vals + k + 2 * lanes);
                auto v3 = s::Load(d, vals + k + 3 * lanes);

                v_sum0 = s::Add(v_sum0, v0);
                v_sum1 = s::Add(v_sum1, v1);
                v_sum2 = s::Add(v_sum2, v2);
                v_sum3 = s::Add(v_sum3, v3);

                v_ssq0 = s::MulAdd(v0, v0, v_ssq0);
                v_ssq1 = s::MulAdd(v1, v1, v_ssq1);
                v_ssq2 = s::MulAdd(v2, v2, v_ssq2);
                v_ssq3 = s::MulAdd(v3, v3, v_ssq3);
            }

            auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
            auto v_ssq = s::Add(s::Add(v_ssq0, v_ssq1), s::Add(v_ssq2, v_ssq3));

            for (; k + lanes <= len; k += lanes) {
                auto v = s::Load(d, vals + k);
                v_sum = s::Add(v_sum, v);
                v_ssq = s::MulAdd(v, v, v_ssq);
            }

            Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
            Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));

            for (; k < len; ++k) {
                Real v = static_cast<Real>(vals[k]);
                sum += v;
                sum_sq += v * v;
            }

            Real mu = sum / N;
            Real var = (denom > 0) ? ((sum_sq - sum * mu) / denom) : 0.0;
            if (var < 0) var = 0.0;

            out_means[p] = mu;
            out_vars[p] = var;
        });
    }
}

/// @brief Compute only means for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_means_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<Real> out_means
) {
    const Index n_primary = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));

    SCL_CHECK_DIM(out_means.len == static_cast<Size>(n_primary), "Means size mismatch");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    kernel::mapped::hint_prefetch(matrix);

    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (n_primary + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), n_primary);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto values = scl::primary_values(matrix, p);

            if (values.len == 0) {
                out_means[p] = 0.0;
                return;
            }

            const T* SCL_RESTRICT vals = values.ptr;
            const Size len = values.len;

            auto v_sum = s::Zero(d);

            Size k = 0;
            for (; k + lanes <= len; k += lanes) {
                v_sum = s::Add(v_sum, s::Load(d, vals + k));
            }

            Real sum = s::GetLane(s::SumOfLanes(d, v_sum));

            for (; k < len; ++k) {
                sum += static_cast<Real>(vals[k]);
            }

            out_means[p] = sum / N;
        });
    }
}

// =============================================================================
// MappedVirtualSparse Feature Statistics
// =============================================================================

/// @brief Compute standard moments (mean, variance) for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void standard_moments_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    const Index n_primary = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real denom = N - static_cast<Real>(ddof);

    SCL_CHECK_DIM(out_means.len == static_cast<Size>(n_primary), "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len == static_cast<Size>(n_primary), "Vars size mismatch");

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
            auto v_ssq = s::Zero(d);

            Size k = 0;
            for (; k + lanes <= len; k += lanes) {
                auto v = s::Load(d, vals + k);
                v_sum = s::Add(v_sum, v);
                v_ssq = s::MulAdd(v, v, v_ssq);
            }

            Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
            Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));

            for (; k < len; ++k) {
                Real v = static_cast<Real>(vals[k]);
                sum += v;
                sum_sq += v * v;
            }

            Real mu = sum / N;
            Real var = (denom > 0) ? ((sum_sq - sum * mu) / denom) : 0.0;
            if (var < 0) var = 0.0;

            out_means[p] = mu;
            out_vars[p] = var;
        });
    }
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate mapped fast path
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void standard_moments_mapped_dispatch(
    const MatrixT& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    standard_moments_mapped(matrix, out_means, out_vars, ddof);
}

} // namespace scl::kernel::feature::mapped
