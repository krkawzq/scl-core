#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file scale_mapped_impl.hpp
/// @brief Mapped Backend Standardization Operations
///
/// Key insight: Mapped data is READ-ONLY (memory-mapped files).
/// For in-place operations, we must:
/// 1. Materialize to OwnedSparse
/// 2. Apply standardization on owned data
/// 3. Return OwnedSparse (caller takes ownership)
///
/// Standardization: (X - mean) / std with optional clipping
// =============================================================================

namespace scl::kernel::scale::mapped {

// =============================================================================
// MappedCustomSparse Standardization
// =============================================================================

/// @brief Standardize mapped matrix - returns materialized result
///
/// Applies (X - mean) / std transformation with optional clipping.
/// Since mapped data is read-only, materializes first.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> standardize_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Real> means,
    Array<const Real> stds,
    Real max_value,
    bool zero_center
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(means.len == static_cast<Size>(n_primary), "Means dim mismatch");
    SCL_CHECK_DIM(stds.len == static_cast<Size>(n_primary), "Stds dim mismatch");

    // Materialize to owned storage
    auto owned = matrix.materialize();

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        Real mu = means[p];
        Real sigma = stds[p];
        Real inv_sigma = (sigma != 0.0) ? (1.0 / sigma) : 0.0;

        if (sigma == 0.0) return;

        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* SCL_RESTRICT vals = owned.data.data() + start;

        const auto v_mu = s::Set(d, mu);
        const auto v_inv_sigma = s::Set(d, inv_sigma);
        const auto v_max = s::Set(d, max_value);
        const auto v_min = s::Set(d, -max_value);
        const bool do_clip = (max_value > 0.0);

        Index k = 0;

        // 4-way unrolled SIMD loop with FMA
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);

            if (zero_center) {
                v0 = s::Sub(v0, v_mu);
                v1 = s::Sub(v1, v_mu);
                v2 = s::Sub(v2, v_mu);
                v3 = s::Sub(v3, v_mu);
            }

            v0 = s::Mul(v0, v_inv_sigma);
            v1 = s::Mul(v1, v_inv_sigma);
            v2 = s::Mul(v2, v_inv_sigma);
            v3 = s::Mul(v3, v_inv_sigma);

            if (do_clip) {
                v0 = s::Min(s::Max(v0, v_min), v_max);
                v1 = s::Min(s::Max(v1, v_min), v_max);
                v2 = s::Min(s::Max(v2, v_min), v_max);
                v3 = s::Min(s::Max(v3, v_min), v_max);
            }

            s::Store(v0, d, vals + k + 0 * lanes);
            s::Store(v1, d, vals + k + 1 * lanes);
            s::Store(v2, d, vals + k + 2 * lanes);
            s::Store(v3, d, vals + k + 3 * lanes);
        }

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);

            if (zero_center) {
                v = s::Sub(v, v_mu);
            }

            v = s::Mul(v, v_inv_sigma);

            if (do_clip) {
                v = s::Min(s::Max(v, v_min), v_max);
            }

            s::Store(v, d, vals + k);
        }

        for (; k < len; ++k) {
            Real v = static_cast<Real>(vals[k]);

            if (zero_center) {
                v -= mu;
            }

            v *= inv_sigma;

            if (do_clip) {
                if (v > max_value) v = max_value;
                if (v < -max_value) v = -max_value;
            }

            vals[k] = static_cast<T>(v);
        }
    });

    return owned;
}

/// @brief Scale rows by given factors - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> scale_rows_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const T> scales
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(scales.len == static_cast<Size>(n_primary), "Scales dim mismatch");

    auto owned = matrix.materialize();

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

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

// =============================================================================
// MappedVirtualSparse Standardization
// =============================================================================

/// @brief Standardize MappedVirtualSparse - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> standardize_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Real> means,
    Array<const Real> stds,
    Real max_value,
    bool zero_center
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(means.len == static_cast<Size>(n_primary), "Means dim mismatch");
    SCL_CHECK_DIM(stds.len == static_cast<Size>(n_primary), "Stds dim mismatch");

    auto owned = matrix.materialize();

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        Real mu = means[p];
        Real sigma = stds[p];
        Real inv_sigma = (sigma != 0.0) ? (1.0 / sigma) : 0.0;

        if (sigma == 0.0) return;

        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* SCL_RESTRICT vals = owned.data.data() + start;

        const auto v_mu = s::Set(d, mu);
        const auto v_inv_sigma = s::Set(d, inv_sigma);
        const auto v_max = s::Set(d, max_value);
        const auto v_min = s::Set(d, -max_value);
        const bool do_clip = (max_value > 0.0);

        Index k = 0;

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);

            if (zero_center) {
                v = s::Sub(v, v_mu);
            }

            v = s::Mul(v, v_inv_sigma);

            if (do_clip) {
                v = s::Min(s::Max(v, v_min), v_max);
            }

            s::Store(v, d, vals + k);
        }

        for (; k < len; ++k) {
            Real v = static_cast<Real>(vals[k]);

            if (zero_center) {
                v -= mu;
            }

            v *= inv_sigma;

            if (do_clip) {
                if (v > max_value) v = max_value;
                if (v < -max_value) v = -max_value;
            }

            vals[k] = static_cast<T>(v);
        }
    });

    return owned;
}

/// @brief Scale rows of MappedVirtualSparse - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> scale_rows_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const T> scales
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(scales.len == static_cast<Size>(n_primary), "Scales dim mismatch");

    auto owned = matrix.materialize();

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

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

} // namespace scl::kernel::scale::mapped
