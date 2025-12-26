#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file normalize_mapped_impl.hpp
/// @brief Mapped Backend Normalization Operations
///
/// Key insight: Mapped data is READ-ONLY (memory-mapped files).
/// For in-place operations, we must:
/// 1. Materialize to OwnedSparse
/// 2. Apply operation on owned data
/// 3. Return OwnedSparse (caller takes ownership)
///
/// For read-only operations (computing scales), we use streaming.
// =============================================================================

namespace scl::kernel::normalize::mapped {

// =============================================================================
// Read-Only Operations (Direct Streaming)
// =============================================================================

/// @brief Compute row sums for normalization (streaming)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_row_sums_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len == static_cast<Size>(n_primary), "Output size mismatch");

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

            if (values.len == 0) {
                output[p] = static_cast<T>(0);
                return;
            }

            const T* SCL_RESTRICT vals = values.ptr;
            const Size len = values.len;

            // SIMD accumulation
            auto v_sum = s::Zero(d);

            Size k = 0;
            for (; k + lanes <= len; k += lanes) {
                v_sum = s::Add(v_sum, s::Load(d, vals + k));
            }

            T sum = s::GetLane(s::SumOfLanes(d, v_sum));

            for (; k < len; ++k) {
                sum += vals[k];
            }

            output[p] = sum;
        });
    }
}

/// @brief Compute normalization scales (1 / sum) for each row
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_normalization_scales_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> scales,
    T target_sum = 1.0
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(scales.len == static_cast<Size>(n_primary), "Scales size mismatch");

    // First compute sums
    compute_row_sums_mapped(matrix, scales);

    // Then convert to scales
    scl::threading::parallel_for(Size(0), scales.len, [&](size_t i) {
        T sum = scales[i];
        scales[i] = (sum != 0) ? (target_sum / sum) : 0;
    });
}

// =============================================================================
// Write Operations (Materialize + Apply)
// =============================================================================

/// @brief Scale rows of mapped matrix - returns materialized result
///
/// Since mapped data is read-only, we must materialize first.
/// Returns OwnedSparse that caller owns.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> scale_rows_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const T> scales
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(scales.len == static_cast<Size>(n_primary), "Scales dim mismatch");

    // Materialize to owned storage
    auto owned = matrix.materialize();

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    // Apply scaling in-place on owned data
    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        T scale = scales[p];
        if (scale == 1.0) return;

        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* SCL_RESTRICT vals = owned.data.data() + start;
        const auto v_scale = s::Set(d, scale);

        Index k = 0;
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Mul(v, v_scale), d, vals + k);
        }

        for (; k < len; ++k) {
            vals[k] *= scale;
        }
    });

    return owned;
}

/// @brief Normalize rows to target sum - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> normalize_rows_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    T target_sum = 1.0
) {
    const Index n_primary = scl::primary_size(matrix);

    // Compute scales
    std::vector<T> scales(n_primary);
    Array<T> scales_arr(scales.data(), scales.size());
    compute_normalization_scales_mapped(matrix, scales_arr, target_sum);

    // Apply scaling
    return scale_rows_mapped(matrix, Array<const T>(scales.data(), scales.size()));
}

// =============================================================================
// MappedVirtualSparse Support
// =============================================================================

/// @brief Compute row sums for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_row_sums_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len == static_cast<Size>(n_primary), "Output size mismatch");

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

            if (values.len == 0) {
                output[p] = static_cast<T>(0);
                return;
            }

            const T* SCL_RESTRICT vals = values.ptr;
            const Size len = values.len;

            auto v_sum = s::Zero(d);

            Size k = 0;
            for (; k + lanes <= len; k += lanes) {
                v_sum = s::Add(v_sum, s::Load(d, vals + k));
            }

            T sum = s::GetLane(s::SumOfLanes(d, v_sum));

            for (; k < len; ++k) {
                sum += vals[k];
            }

            output[p] = sum;
        });
    }
}

/// @brief Scale rows of MappedVirtualSparse - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> scale_rows_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const T> scales
) {
    // Materialize first
    auto owned = matrix.materialize();

    const Index n_primary = scl::primary_size(matrix);

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    // Apply scaling
    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        T scale = scales[p];
        if (scale == 1.0) return;

        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* SCL_RESTRICT vals = owned.data.data() + start;
        const auto v_scale = s::Set(d, scale);

        Index k = 0;
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Mul(v, v_scale), d, vals + k);
        }

        for (; k < len; ++k) {
            vals[k] *= scale;
        }
    });

    return owned;
}

} // namespace scl::kernel::normalize::mapped
