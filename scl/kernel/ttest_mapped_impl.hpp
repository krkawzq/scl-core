#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <cmath>

// =============================================================================
/// @file ttest_mapped_impl.hpp
/// @brief T-Test for Mapped Sparse Matrices
///
/// T-test is primarily compute-bound (not memory-bound).
/// For Mapped matrices, we stream the data and compute group statistics.
///
/// Operations:
/// - compute_group_stats_mapped: Streaming group statistics computation
/// - ttest_mapped: Full t-test pipeline with streaming
// =============================================================================

namespace scl::kernel::ttest::mapped {

namespace detail {
constexpr Size CHUNK_SIZE = 256;
}

// =============================================================================
// Group Statistics - Streaming Implementation
// =============================================================================

/// @brief Compute group statistics from mapped matrix (streaming)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_group_stats_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Real> out_sums,
    Array<Real> out_sum_sqs,
    Array<Size> out_counts
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size total_size = static_cast<Size>(primary_dim) * n_groups;

    SCL_CHECK_DIM(out_sums.len >= total_size, "Sums size mismatch");
    SCL_CHECK_DIM(out_sum_sqs.len >= total_size, "SumSqs size mismatch");
    SCL_CHECK_DIM(out_counts.len >= total_size, "Counts size mismatch");

    // Zero initialize
    for (Size i = 0; i < total_size; ++i) {
        out_sums[i] = 0.0;
        out_sum_sqs[i] = 0.0;
        out_counts[i] = 0;
    }

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;

        Real* sum_ptr = out_sums.ptr + (p * n_groups);
        Real* sum_sq_ptr = out_sum_sqs.ptr + (p * n_groups);
        Size* count_ptr = out_counts.ptr + (p * n_groups);

        if (len == 0) return;

        const T* SCL_RESTRICT values = matrix.data + start;
        const Index* SCL_RESTRICT indices = matrix.indices + start;

        // Accumulate group statistics
        for (Index k = 0; k < len; ++k) {
            Index idx = indices[k];
            int32_t g = group_ids[idx];

            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                Real v = static_cast<Real>(values[k]);
                sum_ptr[g] += v;
                sum_sq_ptr[g] += v * v;
                count_ptr[g]++;
            }
        }
    });
}

/// @brief Compute group statistics from mapped virtual sparse (streaming)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_group_stats_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Real> out_sums,
    Array<Real> out_sum_sqs,
    Array<Size> out_counts
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size total_size = static_cast<Size>(primary_dim) * n_groups;

    SCL_CHECK_DIM(out_sums.len >= total_size, "Sums size mismatch");
    SCL_CHECK_DIM(out_sum_sqs.len >= total_size, "SumSqs size mismatch");
    SCL_CHECK_DIM(out_counts.len >= total_size, "Counts size mismatch");

    // Zero initialize
    for (Size i = 0; i < total_size; ++i) {
        out_sums[i] = 0.0;
        out_sum_sqs[i] = 0.0;
        out_counts[i] = 0;
    }

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];

        Real* sum_ptr = out_sums.ptr + (p * n_groups);
        Real* sum_sq_ptr = out_sum_sqs.ptr + (p * n_groups);
        Size* count_ptr = out_counts.ptr + (p * n_groups);

        if (len == 0) return;

        const T* SCL_RESTRICT values = static_cast<const T*>(matrix.data_ptrs[p]);
        const Index* SCL_RESTRICT indices = static_cast<const Index*>(matrix.indices_ptrs[p]);

        // Accumulate group statistics
        for (Index k = 0; k < len; ++k) {
            Index idx = indices[k];
            int32_t g = group_ids[idx];

            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                Real v = static_cast<Real>(values[k]);
                sum_ptr[g] += v;
                sum_sq_ptr[g] += v * v;
                count_ptr[g]++;
            }
        }
    });
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void compute_group_stats_mapped_dispatch(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Real> out_sums,
    Array<Real> out_sum_sqs,
    Array<Size> out_counts
) {
    compute_group_stats_mapped(matrix, group_ids, n_groups, out_sums, out_sum_sqs, out_counts);
}

} // namespace scl::kernel::ttest::mapped
