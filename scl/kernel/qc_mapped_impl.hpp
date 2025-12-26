#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file qc_mapped_impl.hpp
/// @brief Mapped Backend QC Metrics
///
/// QC operations are read-only statistics, can stream directly from mapped data.
/// No materialization needed.
///
/// Supported operations:
/// - Basic QC: n_genes (nnz per row), total_counts (row sums)
// =============================================================================

namespace scl::kernel::qc::mapped {

// =============================================================================
// MappedCustomSparse QC
// =============================================================================

/// @brief Compute basic QC metrics for MappedCustomSparse
///
/// Streaming algorithm - reads data once, computes n_genes and total_counts.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_basic_qc_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_n_genes.len == static_cast<Size>(n_primary), "n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.len == static_cast<Size>(n_primary), "total_counts size mismatch");

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
            Index len = static_cast<Index>(values.len);

            out_n_genes[p] = len;

            if (len == 0) {
                out_total_counts[p] = 0.0;
                return;
            }

            const T* SCL_RESTRICT vals = values.ptr;

            // 4-way unrolled SIMD accumulation
            auto v_sum0 = s::Zero(d);
            auto v_sum1 = s::Zero(d);
            auto v_sum2 = s::Zero(d);
            auto v_sum3 = s::Zero(d);

            Size k = 0;
            for (; k + 4 * lanes <= values.len; k += 4 * lanes) {
                v_sum0 = s::Add(v_sum0, s::Load(d, vals + k + 0 * lanes));
                v_sum1 = s::Add(v_sum1, s::Load(d, vals + k + 1 * lanes));
                v_sum2 = s::Add(v_sum2, s::Load(d, vals + k + 2 * lanes));
                v_sum3 = s::Add(v_sum3, s::Load(d, vals + k + 3 * lanes));
            }

            auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

            for (; k + lanes <= values.len; k += lanes) {
                v_sum = s::Add(v_sum, s::Load(d, vals + k));
            }

            Real sum = static_cast<Real>(s::GetLane(s::SumOfLanes(d, v_sum)));

            for (; k < values.len; ++k) {
                sum += static_cast<Real>(vals[k]);
            }

            out_total_counts[p] = sum;
        });
    }
}

/// @brief Compute extended QC metrics for MappedCustomSparse
///
/// Computes n_genes, total_counts, and pct_counts_in_top_genes in one pass.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_extended_qc_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts,
    Array<Real> out_pct_top,
    Index n_top
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_n_genes.len == static_cast<Size>(n_primary), "n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.len == static_cast<Size>(n_primary), "total_counts size mismatch");
    SCL_CHECK_DIM(out_pct_top.len == static_cast<Size>(n_primary), "pct_top size mismatch");

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    // Process in chunks
    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (n_primary + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), n_primary);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto values = scl::primary_values(matrix, p);
            Index len = static_cast<Index>(values.len);

            out_n_genes[p] = len;

            if (len == 0) {
                out_total_counts[p] = 0.0;
                out_pct_top[p] = 0.0;
                return;
            }

            const T* SCL_RESTRICT vals = values.ptr;

            // Compute sum
            Real sum = 0.0;
            for (Size k = 0; k < values.len; ++k) {
                sum += static_cast<Real>(vals[k]);
            }
            out_total_counts[p] = sum;

            // Find top n values and compute their sum
            // Use partial sort approach for efficiency
            if (static_cast<Index>(values.len) <= n_top) {
                out_pct_top[p] = 100.0;
            } else {
                // Copy values to temporary buffer for partial sorting
                std::vector<Real> temp(values.len);
                for (Size k = 0; k < values.len; ++k) {
                    temp[k] = static_cast<Real>(vals[k]);
                }

                // Partial sort to get top n_top values
                std::partial_sort(temp.begin(), temp.begin() + n_top, temp.end(),
                                  std::greater<Real>());

                Real top_sum = 0.0;
                for (Index i = 0; i < n_top; ++i) {
                    top_sum += temp[i];
                }

                out_pct_top[p] = (sum > 0.0) ? (top_sum / sum * 100.0) : 0.0;
            }
        });
    }
}

// =============================================================================
// MappedVirtualSparse QC
// =============================================================================

/// @brief Compute basic QC metrics for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_basic_qc_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_n_genes.len == static_cast<Size>(n_primary), "n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.len == static_cast<Size>(n_primary), "total_counts size mismatch");

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
            Index len = static_cast<Index>(values.len);

            out_n_genes[p] = len;

            if (len == 0) {
                out_total_counts[p] = 0.0;
                return;
            }

            const T* SCL_RESTRICT vals = values.ptr;

            auto v_sum = s::Zero(d);

            Size k = 0;
            for (; k + lanes <= values.len; k += lanes) {
                v_sum = s::Add(v_sum, s::Load(d, vals + k));
            }

            Real sum = static_cast<Real>(s::GetLane(s::SumOfLanes(d, v_sum)));

            for (; k < values.len; ++k) {
                sum += static_cast<Real>(vals[k]);
            }

            out_total_counts[p] = sum;
        });
    }
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate mapped fast path
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void compute_basic_qc_mapped_dispatch(
    const MatrixT& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts
) {
    compute_basic_qc_mapped(matrix, out_n_genes, out_total_counts);
}

} // namespace scl::kernel::qc::mapped
