#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file sparse_mapped_impl.hpp
/// @brief Mapped Backend Sparse Statistics
///
/// ## Key Optimizations
///
/// 1. Prefetch Hints
///    - hint_prefetch() for OS page cache
///    - Helps with sequential scan patterns
///
/// 2. 4-Way Unrolled SIMD
///    - Same as fast_impl for hot data
///    - Maximizes throughput when data is cached
///
/// 3. Fused Sum + Sum_Sq
///    - Single IO scan for variance
///    - Uses FMA instructions
///
/// 4. Direct Parallel (No Chunking)
///    - Removed chunk-based outer loop
///    - Let parallel_for handle scheduling
///
/// Performance:
/// - Cached: ~15 GB/s per core
/// - Cold: Depends on storage speed
// =============================================================================

namespace scl::kernel::sparse::mapped {

// =============================================================================
// SECTION 1: SIMD Helpers
// =============================================================================

namespace detail {

/// @brief 4-way unrolled SIMD sum
template <typename T>
SCL_FORCE_INLINE T simd_sum_4way(const T* SCL_RESTRICT vals, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    Size k = 0;
    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        v_sum0 = s::Add(v_sum0, s::Load(d, vals + k + 0 * lanes));
        v_sum1 = s::Add(v_sum1, s::Load(d, vals + k + 1 * lanes));
        v_sum2 = s::Add(v_sum2, s::Load(d, vals + k + 2 * lanes));
        v_sum3 = s::Add(v_sum3, s::Load(d, vals + k + 3 * lanes));
    }

    auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

    for (; k + lanes <= len; k += lanes) {
        v_sum = s::Add(v_sum, s::Load(d, vals + k));
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < len; ++k) {
        sum += vals[k];
    }

    return sum;
}

/// @brief Fused sum + sum_sq with FMA
template <typename T>
SCL_FORCE_INLINE void simd_sum_sumsq_fused(
    const T* SCL_RESTRICT vals,
    Size len,
    T& out_sum,
    T& out_sumsq
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_ssq0 = s::Zero(d);
    auto v_ssq1 = s::Zero(d);

    Size k = 0;
    for (; k + 2 * lanes <= len; k += 2 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);

        v_sum0 = s::Add(v_sum0, v0);
        v_sum1 = s::Add(v_sum1, v1);
        v_ssq0 = s::MulAdd(v0, v0, v_ssq0);
        v_ssq1 = s::MulAdd(v1, v1, v_ssq1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);
    auto v_ssq = s::Add(v_ssq0, v_ssq1);

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::Add(v_sum, v);
        v_ssq = s::MulAdd(v, v, v_ssq);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));
    T sumsq = s::GetLane(s::SumOfLanes(d, v_ssq));

    for (; k < len; ++k) {
        T v = vals[k];
        sum += v;
        sumsq += v * v;
    }

    out_sum = sum;
    out_sumsq = sumsq;
}

/// @brief Compute variance from sum and sum_sq
template <typename T>
SCL_FORCE_INLINE T compute_variance(T sum, T sum_sq, T N, T denom) {
    if (denom <= T(0)) return T(0);

    T mu = sum / N;
    T var = (sum_sq - sum * mu) / denom;

    return (var < T(0)) ? T(0) : var;
}

} // namespace detail

// =============================================================================
// SECTION 2: MappedCustomSparse
// =============================================================================

/// @brief Primary sums (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires ::scl::kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void primary_sums_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len == static_cast<Size>(n_primary), "Output size mismatch");

    // Prefetch hint for OS
    ::scl::kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = matrix.indptr()[p];
        Index end = matrix.indptr()[p + 1];
        Size len = static_cast<Size>(end - start);

        if (len == 0) {
            output[p] = T(0);
            return;
        }

        output[p] = detail::simd_sum_4way(matrix.data() + start, len);
    });
}

/// @brief Primary means (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires scl::kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void primary_means_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index n_primary = scl::primary_size(matrix);
    const T inv_n = T(1) / static_cast<T>(scl::secondary_size(matrix));

    SCL_CHECK_DIM(output.len == static_cast<Size>(n_primary), "Output size mismatch");

    scl::kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = matrix.indptr()[p];
        Index end = matrix.indptr()[p + 1];
        Size len = static_cast<Size>(end - start);

        if (len == 0) {
            output[p] = T(0);
            return;
        }

        output[p] = detail::simd_sum_4way(matrix.data() + start, len) * inv_n;
    });
}

/// @brief Primary variances (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires scl::kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void primary_variances_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> output,
    int ddof = 1
) {
    const Index n_primary = scl::primary_size(matrix);
    const T N = static_cast<T>(scl::secondary_size(matrix));
    const T denom = N - static_cast<T>(ddof);

    SCL_CHECK_DIM(output.len == static_cast<Size>(n_primary), "Output size mismatch");

    scl::kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = matrix.indptr()[p];
        Index end = matrix.indptr()[p + 1];
        Size len = static_cast<Size>(end - start);

        T sum = T(0), sumsq = T(0);

        if (len > 0) {
            detail::simd_sum_sumsq_fused(matrix.data() + start, len, sum, sumsq);
        }

        output[p] = detail::compute_variance(sum, sumsq, N, denom);
    });
}

/// @brief Primary NNZ counts (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires scl::kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void primary_nnz_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<Index> output
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len == static_cast<Size>(n_primary), "Output size mismatch");

    // Can compute directly from indptr (no data access)
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        output[p] = matrix.indptr()[p + 1] - matrix.indptr()[p];
    });
}

// =============================================================================
// SECTION 3: MappedVirtualSparse
// =============================================================================

/// @brief Primary sums (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires scl::kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void primary_sums_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len == static_cast<Size>(n_primary), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Size len = static_cast<Size>(matrix.lengths[p]);

        if (len == 0) {
            output[p] = T(0);
            return;
        }

        const T* vals = static_cast<const T*>(matrix.data_ptrs[p]);
        output[p] = detail::simd_sum_4way(vals, len);
    });
}

/// @brief Primary means (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires scl::kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void primary_means_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index n_primary = scl::primary_size(matrix);
    const T inv_n = T(1) / static_cast<T>(scl::secondary_size(matrix));

    SCL_CHECK_DIM(output.len == static_cast<Size>(n_primary), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Size len = static_cast<Size>(matrix.lengths[p]);

        if (len == 0) {
            output[p] = T(0);
            return;
        }

        const T* vals = static_cast<const T*>(matrix.data_ptrs[p]);
        output[p] = detail::simd_sum_4way(vals, len) * inv_n;
    });
}

/// @brief Primary variances (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires scl::kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void primary_variances_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<T> output,
    int ddof = 1
) {
    const Index n_primary = scl::primary_size(matrix);
    const T N = static_cast<T>(scl::secondary_size(matrix));
    const T denom = N - static_cast<T>(ddof);

    SCL_CHECK_DIM(output.len == static_cast<Size>(n_primary), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Size len = static_cast<Size>(matrix.lengths[p]);

        T sum = T(0), sumsq = T(0);

        if (len > 0) {
            const T* vals = static_cast<const T*>(matrix.data_ptrs[p]);
            detail::simd_sum_sumsq_fused(vals, len, sum, sumsq);
        }

        output[p] = detail::compute_variance(sum, sumsq, N, denom);
    });
}

/// @brief Primary NNZ counts (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires scl::kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void primary_nnz_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<Index> output
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len == static_cast<Size>(n_primary), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        output[p] = matrix.lengths[p];
    });
}

// =============================================================================
// SECTION 4: Unified Dispatchers
// =============================================================================

/// @brief Primary sums dispatcher
template <typename MatrixT, bool IsCSR>
    requires scl::kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void primary_sums_mapped_dispatch(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    primary_sums_mapped(matrix, output);
}

/// @brief Primary means dispatcher
template <typename MatrixT, bool IsCSR>
    requires scl::kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void primary_means_mapped_dispatch(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    primary_means_mapped(matrix, output);
}

/// @brief Primary variances dispatcher
template <typename MatrixT, bool IsCSR>
    requires scl::kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void primary_variances_mapped_dispatch(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output,
    int ddof = 1
) {
    primary_variances_mapped(matrix, output, ddof);
}

/// @brief Primary NNZ dispatcher
template <typename MatrixT, bool IsCSR>
    requires scl::kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void primary_nnz_mapped_dispatch(
    const MatrixT& matrix,
    Array<Index> output
) {
    primary_nnz_mapped(matrix, output);
}

} // namespace scl::kernel::sparse::mapped
