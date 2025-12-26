#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/scale_fast_impl.hpp"
#include "scl/kernel/scale_mapped_impl.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <cmath>

// =============================================================================
/// @file scale.hpp
/// @brief Matrix Scaling and Standardization
///
/// Unified entry point for sparse matrix scaling with automatic backend dispatch:
///
/// - CustomSparse / VirtualSparse: Uses scale_fast_impl.hpp (in-place)
/// - MappedCustomSparse / MappedVirtualSparse: Uses scale_mapped_impl.hpp
///
/// ## Operations
///
/// 1. standardize: z-score transformation (X - mean) / std
///    - Optional clipping to [-max, max]
///    - Optional centering (subtract mean)
///
/// 2. scale_rows: Multiply each row by a scalar
///    - For CSR: scales samples
///    - For CSC: scales genes
///
/// 3. shift_rows: Add offset to each row
///    - For CSR: shifts samples
///    - For CSC: shifts genes
///
/// ## Performance Optimizations
///
/// 1. 8-Way Unrolled SIMD: Maximum instruction-level parallelism
/// 2. Fused Operations: Single pass for materialize + transform
/// 3. Adaptive Row Processing: Short/medium/long row strategies
/// 4. Skip Optimization: Skip rows with scale=1 or offset=0
///
/// ## Note on Sparsity
///
/// For sparse matrices, only NON-ZERO values are transformed.
/// This maintains sparsity but is mathematically approximate.
/// If exact centering is needed, consider dense conversion first.
// =============================================================================

namespace scl::kernel::scale {

// =============================================================================
// SECTION 1: Generic Implementation (Fallback)
// =============================================================================

namespace detail {

/// @brief Generic standardize for any sparse matrix
template <typename T>
SCL_FORCE_INLINE void standardize_generic(
    T* SCL_RESTRICT vals,
    Size len,
    T mu,
    T inv_sigma,
    T max_val,
    bool zero_center,
    bool do_clip
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_mu = s::Set(d, mu);
    const auto v_inv_sigma = s::Set(d, inv_sigma);
    const auto v_max = s::Set(d, max_val);
    const auto v_min = s::Set(d, -max_val);

    Size k = 0;

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        if (zero_center) v = s::Sub(v, v_mu);
        v = s::Mul(v, v_inv_sigma);
        if (do_clip) v = s::Min(s::Max(v, v_min), v_max);
        s::Store(v, d, vals + k);
    }

    for (; k < len; ++k) {
        T v = vals[k];
        if (zero_center) v -= mu;
        v *= inv_sigma;
        if (do_clip) {
            if (v > max_val) v = max_val;
            if (v < -max_val) v = -max_val;
        }
        vals[k] = v;
    }
}

} // namespace detail

// =============================================================================
// SECTION 2: In-Place Standardize (CustomSparse / VirtualSparse)
// =============================================================================

/// @brief Standardize matrix (z-score transformation)
///
/// Applies: X' = (X - mean) / std with optional clipping.
/// In-place modification for mutable sparse matrices.
///
/// @tparam MatrixT Sparse matrix type (must satisfy AnySparse)
/// @param matrix Input sparse matrix (modified in-place)
/// @param means Means for each primary dimension [size = primary_dim]
/// @param stds Standard deviations [size = primary_dim]
/// @param max_value Clipping threshold (0 = no clipping)
/// @param zero_center If true, subtract mean (default true)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void standardize(
    MatrixT& matrix,
    Array<const typename MatrixT::ValueType> means,
    Array<const typename MatrixT::ValueType> stds,
    typename MatrixT::ValueType max_value = typename MatrixT::ValueType(0),
    bool zero_center = true
) {
    using T = typename MatrixT::ValueType;
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        fast::standardize_custom_fast(matrix, means, stds, max_value, zero_center);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        fast::standardize_virtual_fast(matrix, means, stds, max_value, zero_center);
    } else {
        // Generic fallback
        const Index primary_dim = scl::primary_size(matrix);
        const bool do_clip = (max_value > T(0));

        SCL_CHECK_DIM(means.len == static_cast<Size>(primary_dim), "Means dim mismatch");
        SCL_CHECK_DIM(stds.len == static_cast<Size>(primary_dim), "Stds dim mismatch");

        scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
            T sigma = stds[p];
            if (sigma == T(0)) return;

            auto vals = scl::primary_values(matrix, p);
            if (vals.len == 0) return;

            T mu = means[p];
            T inv_sigma = T(1) / sigma;

            detail::standardize_generic(
                vals.ptr, vals.len,
                mu, inv_sigma, max_value,
                zero_center, do_clip
            );
        });
    }
}

/// @brief Scale rows by factors (in-place)
///
/// Multiplies each row by the corresponding scale factor.
///
/// @param matrix Input sparse matrix (modified in-place)
/// @param scales Scale factors [size = primary_dim]
template <typename MatrixT>
    requires AnySparse<MatrixT>
void scale_rows(
    MatrixT& matrix,
    Array<const typename MatrixT::ValueType> scales
) {
    using T = typename MatrixT::ValueType;
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        fast::scale_rows_custom_fast(matrix, scales);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        fast::scale_rows_virtual_fast(matrix, scales);
    } else {
        const Index primary_dim = scl::primary_size(matrix);

        SCL_CHECK_DIM(scales.len == static_cast<Size>(primary_dim), "Scales dim mismatch");

        scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
            T scale = scales[p];
            if (scale == T(1)) return;

            auto vals = scl::primary_values(matrix, p);
            for (Size k = 0; k < vals.len; ++k) {
                vals[k] *= scale;
            }
        });
    }
}

/// @brief Shift rows by offsets (in-place)
///
/// Adds offset to each element in the row.
///
/// @param matrix Input sparse matrix (modified in-place)
/// @param offsets Offset values [size = primary_dim]
template <typename MatrixT>
    requires AnySparse<MatrixT>
void shift_rows(
    MatrixT& matrix,
    Array<const typename MatrixT::ValueType> offsets
) {
    using T = typename MatrixT::ValueType;
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        fast::shift_rows_custom_fast(matrix, offsets);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        fast::shift_rows_virtual_fast(matrix, offsets);
    } else {
        const Index primary_dim = scl::primary_size(matrix);

        SCL_CHECK_DIM(offsets.len == static_cast<Size>(primary_dim), "Offsets dim mismatch");

        scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
            T offset = offsets[p];
            if (offset == T(0)) return;

            auto vals = scl::primary_values(matrix, p);
            for (Size k = 0; k < vals.len; ++k) {
                vals[k] += offset;
            }
        });
    }
}

// =============================================================================
// SECTION 3: Mapped Standardize (Returns OwnedSparse)
// =============================================================================

/// @brief Standardize mapped matrix
///
/// Materializes to OwnedSparse with standardization applied.
/// Uses fused materialize + transform for efficiency.
///
/// @param matrix Input mapped sparse matrix (read-only)
/// @param means Means for each primary dimension
/// @param stds Standard deviations
/// @param max_value Clipping threshold (0 = no clipping)
/// @param zero_center If true, subtract mean
/// @return OwnedSparse with standardized data
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> standardize(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const T> means,
    Array<const T> stds,
    T max_value = T(0),
    bool zero_center = true
) {
    return mapped::standardize_mapped_custom(matrix, means, stds, max_value, zero_center);
}

template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> standardize(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const T> means,
    Array<const T> stds,
    T max_value = T(0),
    bool zero_center = true
) {
    return mapped::standardize_mapped_virtual(matrix, means, stds, max_value, zero_center);
}

/// @brief Scale rows of mapped matrix
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> scale_rows(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const T> scales
) {
    return mapped::scale_rows_mapped_custom(matrix, scales);
}

template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> scale_rows(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const T> scales
) {
    return mapped::scale_rows_mapped_virtual(matrix, scales);
}

// =============================================================================
// SECTION 4: Convenience Functions
// =============================================================================

/// @brief Scale to unit variance (divide by std only)
///
/// Applies X' = X / std without centering.
template <typename MatrixT>
    requires AnySparse<MatrixT>
void scale_to_unit_variance(
    MatrixT& matrix,
    Array<const typename MatrixT::ValueType> stds
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(matrix);

    // Create dummy means (zeros)
    std::vector<T> zeros(static_cast<size_t>(primary_dim), T(0));
    Array<const T> means(zeros.data(), zeros.size());

    standardize(matrix, means, stds, T(0), false);  // zero_center = false
}

/// @brief Center to zero mean (subtract mean only)
///
/// Applies X' = X - mean without scaling.
template <typename MatrixT>
    requires AnySparse<MatrixT>
void center(
    MatrixT& matrix,
    Array<const typename MatrixT::ValueType> means
) {
    using T = typename MatrixT::ValueType;
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR> || VirtualSparseLike<MatrixT, IsCSR>) {
        // Use shift with negated means
        const Index primary_dim = scl::primary_size(matrix);
        std::vector<T> neg_means(static_cast<size_t>(primary_dim));
        for (Size i = 0; i < static_cast<Size>(primary_dim); ++i) {
            neg_means[i] = -means[i];
        }
        shift_rows(matrix, Array<const T>(neg_means.data(), neg_means.size()));
    } else {
        const Index primary_dim = scl::primary_size(matrix);

        scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
            T mu = means[p];
            if (mu == T(0)) return;

            auto vals = scl::primary_values(matrix, p);
            for (Size k = 0; k < vals.len; ++k) {
                vals[k] -= mu;
            }
        });
    }
}

/// @brief Clip values to range [-max, max]
template <typename MatrixT>
    requires AnySparse<MatrixT>
void clip(
    MatrixT& matrix,
    typename MatrixT::ValueType max_value
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(matrix);

    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        auto vals = scl::primary_values(matrix, p);

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        const auto v_max = s::Set(d, max_value);
        const auto v_min = s::Set(d, -max_value);

        Size k = 0;
        for (; k + lanes <= vals.len; k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v = s::Min(s::Max(v, v_min), v_max);
            s::Store(v, d, vals.ptr + k);
        }

        for (; k < vals.len; ++k) {
            T v = vals[k];
            if (v > max_value) vals[k] = max_value;
            else if (v < -max_value) vals[k] = -max_value;
        }
    });
}

} // namespace scl::kernel::scale
