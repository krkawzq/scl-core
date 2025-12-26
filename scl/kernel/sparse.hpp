#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Include optimized backends
#include "scl/kernel/sparse_fast_impl.hpp"

#include <cmath>

// =============================================================================
/// @file sparse.hpp
/// @brief Sparse Matrix Statistics and Aggregations
///
/// ## Operations
///
/// - primary_sums: Sum along primary dimension (rows for CSR, cols for CSC)
/// - primary_means: Mean along primary dimension
/// - primary_variances: Variance along primary dimension
/// - primary_nnz_counts: Non-zero counts per row/column
///
/// ## Performance Optimizations
///
/// 1. Automatic Backend Dispatch
///    - CustomSparse: Direct array SIMD
///    - VirtualSparse: Per-row SIMD
///    - MappedSparse: Prefetch + SIMD
///
/// 2. 4-Way Unrolled SIMD
///    - Hides latency, maximizes throughput
///
/// 3. Fused Sum + Sum_Sq
///    - Single pass for variance
///    - Uses FMA instructions
///
/// Performance: 15-20 GB/s per core
// =============================================================================

namespace scl::kernel::sparse {

// =============================================================================
// SECTION 1: SIMD Helpers (Generic Fallback)
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
// SECTION 2: Primary Sums
// =============================================================================

/// @brief Compute sums for primary dimension (unified for CSR/CSC)
///
/// Dispatches to optimized backend based on matrix type.
///
/// @tparam MatrixT Sparse matrix type (CustomSparse, VirtualSparse, or Mapped)
/// @param matrix Input sparse matrix
/// @param output Output sums [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void primary_sums(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    using T = typename MatrixT::ValueType;
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");

    // Dispatch to optimized backends
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        fast::primary_sums_custom(matrix, output);
        return;
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        fast::primary_sums_virtual(matrix, output);
        return;
    } else if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        mapped::primary_sums_mapped(matrix, output);
        return;
    }

    // Generic fallback
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));

        if (vals.len == 0) {
            output[p] = T(0);
            return;
        }

        output[p] = detail::simd_sum_4way(vals.ptr, vals.len);
    });
}

// =============================================================================
// SECTION 3: Primary Means
// =============================================================================

/// @brief Compute means for primary dimension
///
/// @tparam MatrixT Sparse matrix type
/// @param matrix Input sparse matrix
/// @param output Output means [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void primary_means(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    using T = typename MatrixT::ValueType;
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    const Index primary_dim = scl::primary_size(matrix);
    const T inv_n = T(1) / static_cast<T>(scl::secondary_size(matrix));

    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");

    // Dispatch to optimized backends
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        fast::primary_means_custom(matrix, output);
        return;
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        fast::primary_means_virtual(matrix, output);
        return;
    } else if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        mapped::primary_means_mapped(matrix, output);
        return;
    }

    // Generic fallback
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));

        if (vals.len == 0) {
            output[p] = T(0);
            return;
        }

        output[p] = detail::simd_sum_4way(vals.ptr, vals.len) * inv_n;
    });
}

// =============================================================================
// SECTION 4: Primary Variances
// =============================================================================

/// @brief Compute variances for primary dimension
///
/// Uses single-pass algorithm with fused sum + sum_sq.
///
/// @tparam MatrixT Sparse matrix type
/// @param matrix Input sparse matrix
/// @param output Output variances [size = primary_dim], PRE-ALLOCATED
/// @param ddof Delta degrees of freedom (default 1 for sample variance)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void primary_variances(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output,
    int ddof = 1
) {
    using T = typename MatrixT::ValueType;
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    const Index primary_dim = scl::primary_size(matrix);
    const T N = static_cast<T>(scl::secondary_size(matrix));
    const T denom = N - static_cast<T>(ddof);

    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");

    // Dispatch to optimized backends
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        fast::primary_variances_custom(matrix, output, ddof);
        return;
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        fast::primary_variances_virtual(matrix, output, ddof);
        return;
    } else if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        mapped::primary_variances_mapped(matrix, output, ddof);
        return;
    }

    // Generic fallback
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));

        T sum = T(0), sumsq = T(0);

        if (vals.len > 0) {
            detail::simd_sum_sumsq_fused(vals.ptr, vals.len, sum, sumsq);
        }

        output[p] = detail::compute_variance(sum, sumsq, N, denom);
    });
}

// =============================================================================
// SECTION 5: Primary NNZ Counts
// =============================================================================

/// @brief Count non-zeros for primary dimension
///
/// @tparam MatrixT Sparse matrix type
/// @param matrix Input sparse matrix
/// @param output Output counts [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void primary_nnz_counts(
    const MatrixT& matrix,
    Array<Index> output
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");

    // Dispatch to optimized backends
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        fast::primary_nnz_custom(matrix, output);
        return;
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        fast::primary_nnz_virtual(matrix, output);
        return;
    } else if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        mapped::primary_nnz_mapped(matrix, output);
        return;
    }

    // Generic fallback
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        output[p] = scl::primary_length(matrix, static_cast<Index>(p));
    });
}

// =============================================================================
// SECTION 6: Total Statistics
// =============================================================================

/// @brief Compute total sum of all elements
///
/// @tparam MatrixT Sparse matrix type
/// @param matrix Input sparse matrix
/// @return Total sum
template <typename MatrixT>
    requires AnySparse<MatrixT>
typename MatrixT::ValueType total_sum(const MatrixT& matrix) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(matrix);

    // Parallel reduction with thread-local accumulators
    std::atomic<double> total{0.0};

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));

        if (vals.len == 0) return;

        T sum = detail::simd_sum_4way(vals.ptr, vals.len);

        // Atomic add
        double current = total.load(std::memory_order_relaxed);
        while (!total.compare_exchange_weak(
            current, current + static_cast<double>(sum),
            std::memory_order_relaxed, std::memory_order_relaxed)) {}
    });

    return static_cast<T>(total.load());
}

/// @brief Compute total number of non-zeros
///
/// @tparam MatrixT Sparse matrix type
/// @param matrix Input sparse matrix
/// @return Total NNZ
template <typename MatrixT>
    requires AnySparse<MatrixT>
Size total_nnz(const MatrixT& matrix) {
    const Index primary_dim = scl::primary_size(matrix);

    std::atomic<Size> total{0};

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index len = scl::primary_length(matrix, static_cast<Index>(p));
        total.fetch_add(static_cast<Size>(len), std::memory_order_relaxed);
    });

    return total.load();
}

} // namespace scl::kernel::sparse
