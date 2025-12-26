#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file neighbors_mapped_impl.hpp
/// @brief Mapped Backend KNN Operations
///
/// KNN norm computation is read-only, can stream directly from mapped data.
/// No materialization needed.
///
/// Key optimizations:
/// - Chunk-based processing for cache efficiency
/// - SIMD norm accumulation
// =============================================================================

namespace scl::kernel::neighbors::mapped {

// =============================================================================
// MappedCustomSparse Norm Computation
// =============================================================================

/// @brief Compute squared norms for MappedCustomSparse
///
/// Streaming algorithm - reads row data once for norm computation.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_norms_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> norms_sq
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(norms_sq.len == static_cast<Size>(n_primary), "Norms size mismatch");

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
                norms_sq[p] = static_cast<T>(0);
                return;
            }

            const T* SCL_RESTRICT vals = values.ptr;
            const Size len = values.len;

            // 4-way unrolled SIMD accumulation
            auto v_sum0 = s::Zero(d);
            auto v_sum1 = s::Zero(d);
            auto v_sum2 = s::Zero(d);
            auto v_sum3 = s::Zero(d);

            Size k = 0;
            for (; k + 4 * lanes <= len; k += 4 * lanes) {
                auto v0 = s::Load(d, vals + k + 0 * lanes);
                auto v1 = s::Load(d, vals + k + 1 * lanes);
                auto v2 = s::Load(d, vals + k + 2 * lanes);
                auto v3 = s::Load(d, vals + k + 3 * lanes);

                v_sum0 = s::MulAdd(v0, v0, v_sum0);
                v_sum1 = s::MulAdd(v1, v1, v_sum1);
                v_sum2 = s::MulAdd(v2, v2, v_sum2);
                v_sum3 = s::MulAdd(v3, v3, v_sum3);
            }

            auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

            for (; k + lanes <= len; k += lanes) {
                auto v = s::Load(d, vals + k);
                v_sum = s::MulAdd(v, v, v_sum);
            }

            T sum_sq = s::GetLane(s::SumOfLanes(d, v_sum));

            for (; k < len; ++k) {
                T val = vals[k];
                sum_sq += val * val;
            }

            norms_sq[p] = sum_sq;
        });
    }
}

/// @brief Compute Euclidean norms for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_euclidean_norms_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> norms
) {
    compute_norms_mapped(matrix, norms);

    // Take square root
    scl::threading::parallel_for(Size(0), norms.len, [&](size_t i) {
        norms[i] = std::sqrt(norms[i]);
    });
}

// =============================================================================
// MappedVirtualSparse Norm Computation
// =============================================================================

/// @brief Compute squared norms for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_norms_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<T> norms_sq
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(norms_sq.len == static_cast<Size>(n_primary), "Norms size mismatch");

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
                norms_sq[p] = static_cast<T>(0);
                return;
            }

            const T* SCL_RESTRICT vals = values.ptr;
            const Size len = values.len;

            auto v_sum = s::Zero(d);

            Size k = 0;
            for (; k + lanes <= len; k += lanes) {
                auto v = s::Load(d, vals + k);
                v_sum = s::MulAdd(v, v, v_sum);
            }

            T sum_sq = s::GetLane(s::SumOfLanes(d, v_sum));

            for (; k < len; ++k) {
                T val = vals[k];
                sum_sq += val * val;
            }

            norms_sq[p] = sum_sq;
        });
    }
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate mapped fast path
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void compute_norms_mapped_dispatch(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> norms_sq
) {
    compute_norms_mapped(matrix, norms_sq);
}

} // namespace scl::kernel::neighbors::mapped
