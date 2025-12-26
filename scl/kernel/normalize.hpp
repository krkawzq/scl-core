#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/threading/parallel_for.hpp"

// Backend implementations
#include "scl/kernel/normalize_fast_impl.hpp"

#include <cmath>
#include <algorithm>

// =============================================================================
/// @file normalize.hpp
/// @brief Normalization Kernels for Sparse Matrices
///
/// ## Supported Operations
///
/// 1. Row/Column Scaling
///    - scale_primary: Scale each row/column by a factor
///
/// 2. Row Sum Computation
///    - compute_row_sums: Sum values in each row/column
///
/// 3. Highly Expressed Feature Detection
///    - detect_highly_expressed: Find features exceeding threshold
///
/// 4. Masked Reductions
///    - primary_sums_masked: Sum excluding masked features
///
/// ## Backend Dispatch
///
/// - CustomSparseLike -> normalize_fast_impl.hpp
/// - VirtualSparseLike -> normalize_fast_impl.hpp
/// - MappedSparseLike -> normalize_mapped_impl.hpp
/// - Generic -> This file (fallback)
///
/// ## Key Optimizations
///
/// 1. SIMD Sum/Scale (4-way unrolled)
/// 2. Fused Copy + Scale (for Mapped writes)
/// 3. Lock-Free Atomic Mask Updates
/// 4. Full Parallelization
///
/// Performance Target: 3-5x faster than naive
// =============================================================================

namespace scl::kernel::normalize {

// =============================================================================
// SECTION 1: Generic Implementations (Fallback)
// =============================================================================

namespace detail {

/// @brief Generic SIMD sum
template <typename T>
SCL_FORCE_INLINE T sum_simd_generic(const T* vals, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum = s::Zero(d);
    Size k = 0;

    for (; k + lanes <= len; k += lanes) {
        v_sum = s::Add(v_sum, s::Load(d, vals + k));
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < len; ++k) {
        sum += vals[k];
    }

    return sum;
}

/// @brief Generic row sums
template <typename MatrixT>
    requires AnySparse<MatrixT>
void compute_row_sums_generic(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len >= static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));
        output[p] = (vals.len > 0) ? sum_simd_generic(vals.ptr, vals.len) : T(0);
    });
}

/// @brief Generic scale primary
template <typename MatrixT>
    requires AnySparse<MatrixT>
void scale_primary_generic(
    MatrixT& matrix,
    Array<const Real> scales
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(scales.len >= static_cast<Size>(primary_dim), "Scales dim mismatch");

    namespace s = scl::simd;

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Real scale = scales[p];
        if (scale == Real(1)) return;

        auto vals = scl::primary_values(matrix, static_cast<Index>(p));

        const s::Tag d;
        const size_t lanes = s::lanes();
        const auto v_scale = s::Set(d, scale);

        size_t k = 0;
        for (; k + lanes <= vals.len; k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            s::Store(s::Mul(v, v_scale), d, vals.ptr + k);
        }

        for (; k < vals.len; ++k) {
            vals[k] *= scale;
        }
    });
}

/// @brief Generic masked sums
template <typename MatrixT>
    requires AnySparse<MatrixT>
void primary_sums_masked_generic(
    const MatrixT& matrix,
    Array<const Byte> mask,
    Array<Real> output
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len >= static_cast<Size>(primary_dim), "Output mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        auto indices = scl::primary_indices(matrix, static_cast<Index>(p));
        auto values = scl::primary_values(matrix, static_cast<Index>(p));

        Real sum = Real(0);
        for (Size k = 0; k < values.len; ++k) {
            if (mask[indices[k]] == 0) {
                sum += values[k];
            }
        }
        output[p] = sum;
    });
}

/// @brief Generic highly expressed detection
template <typename MatrixT>
    requires AnySparse<MatrixT>
void detect_highly_expressed_generic(
    const MatrixT& matrix,
    Array<const Real> row_sums,
    Real max_fraction,
    Array<Byte> out_mask
) {
    const Index primary_dim = scl::primary_size(matrix);

    scl::memory::zero(out_mask);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Real total = row_sums[p];
        if (total <= Real(0)) return;

        Real threshold = total * max_fraction;

        auto indices = scl::primary_indices(matrix, static_cast<Index>(p));
        auto values = scl::primary_values(matrix, static_cast<Index>(p));

        for (Size k = 0; k < values.len; ++k) {
            if (values[k] > threshold) {
                __atomic_store_n(&out_mask.ptr[indices[k]], 1, __ATOMIC_RELAXED);
            }
        }
    });
}

} // namespace detail

// =============================================================================
// SECTION 2: Public API
// =============================================================================

/// @brief Compute row/column sums
///
/// @param matrix Input sparse matrix
/// @param output Output sums [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void compute_row_sums(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        fast::compute_row_sums_fast<MatrixT, IsCSR>(matrix, output);
    } else {
        detail::compute_row_sums_generic(matrix, output);
    }
}

/// @brief Scale primary dimension (rows for CSR, columns for CSC)
///
/// @param matrix Input/output sparse matrix (modified in-place)
/// @param scales Scale factors [size = primary_dim]
template <typename MatrixT>
    requires AnySparse<MatrixT>
void scale_primary(
    MatrixT& matrix,
    Array<const Real> scales
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR> || VirtualSparseLike<MatrixT, IsCSR>) {
        fast::scale_primary_fast<MatrixT, IsCSR>(matrix, scales);
    } else {
        detail::scale_primary_generic(matrix, scales);
    }
}

/// @brief Detect highly expressed features
///
/// Marks features that exceed max_fraction of total in any row.
///
/// @param matrix Input sparse matrix
/// @param feature_sums Pre-computed row sums [size = primary_dim]
/// @param max_fraction Threshold fraction (e.g., 0.05 for 5%)
/// @param out_mask Output boolean mask [size = secondary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void detect_highly_expressed(
    const MatrixT& matrix,
    Array<const Real> feature_sums,
    Real max_fraction,
    Array<Byte> out_mask
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    const Index secondary_dim = scl::secondary_size(matrix);
    SCL_CHECK_DIM(out_mask.len >= static_cast<Size>(secondary_dim), "Output mask mismatch");

    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        fast::detect_highly_expressed_fast<MatrixT, IsCSR>(matrix, feature_sums, max_fraction, out_mask);
    } else {
        detail::detect_highly_expressed_generic(matrix, feature_sums, max_fraction, out_mask);
    }
}

/// @brief Compute primary sums excluding masked secondary elements
///
/// @param matrix Input sparse matrix
/// @param secondary_mask Byte mask [size = secondary_dim]
/// @param out_sums Output sums [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void primary_sums_masked(
    const MatrixT& matrix,
    Array<const Byte> secondary_mask,
    Array<Real> out_sums
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    const Index secondary_dim = scl::secondary_size(matrix);
    SCL_CHECK_DIM(secondary_mask.len >= static_cast<Size>(secondary_dim), "Mask mismatch");

    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        fast::primary_sums_masked_fast<MatrixT, IsCSR>(matrix, secondary_mask, out_sums);
    } else {
        detail::primary_sums_masked_generic(matrix, secondary_mask, out_sums);
    }
}

/// @brief Compute median of array
///
/// @param data Input data
/// @param workspace Temporary buffer (size >= data.size())
/// @param out_median Output median value
void median(
    Array<const Real> data,
    Array<Real> workspace,
    Real& out_median
) {
    if (data.len == 0) {
        out_median = Real(0);
        return;
    }
    SCL_CHECK_DIM(workspace.len >= data.len, "Workspace too small");

    scl::memory::copy(data, workspace);

    Array<Real> work_view(workspace.ptr, data.len);
    Size n = work_view.len;
    Size mid = n / 2;

    std::nth_element(work_view.ptr, work_view.ptr + mid, work_view.ptr + n);

    if (n % 2 == 1) {
        out_median = work_view[mid];
    } else {
        Real upper = work_view[mid];
        Real lower = *std::max_element(work_view.ptr, work_view.ptr + mid);
        out_median = (lower + upper) * Real(0.5);
    }
}

/// @brief Compute normalization scales (target_sum / row_sum)
///
/// @param matrix Input sparse matrix
/// @param scales Output scales [size = primary_dim], PRE-ALLOCATED
/// @param target_sum Target sum (default 1.0)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void compute_normalization_scales(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> scales,
    typename MatrixT::ValueType target_sum = typename MatrixT::ValueType(1)
) {
    using T = typename MatrixT::ValueType;

    // Compute sums
    compute_row_sums(matrix, scales);

    // Convert to scales
    scl::threading::parallel_for(Size(0), scales.len, [&](size_t i) {
        T sum = scales[i];
        scales[i] = (sum != T(0)) ? (target_sum / sum) : T(0);
    });
}

} // namespace scl::kernel::normalize
