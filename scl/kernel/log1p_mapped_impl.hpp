#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file log1p_mapped_impl.hpp
/// @brief Mapped Backend Logarithmic Transforms
///
/// Key insight: Mapped data is READ-ONLY (memory-mapped files).
/// For in-place operations, we must:
/// 1. Materialize to OwnedSparse
/// 2. Apply transform on owned data
/// 3. Return OwnedSparse (caller takes ownership)
///
/// Supported transforms:
/// - log1p: log(1 + x)
/// - log2p1: log2(1 + x)
/// - expm1: exp(x) - 1
// =============================================================================

namespace scl::kernel::log1p::mapped {

// =============================================================================
// MappedCustomSparse Transforms
// =============================================================================

/// @brief Apply log1p transform to mapped matrix - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> log1p_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix
) {
    // Materialize to owned storage
    auto owned = matrix.materialize();

    const Index n_primary = scl::primary_size(matrix);

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    // Apply log1p in-place on owned data
    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* SCL_RESTRICT vals = owned.data.data() + start;

        Index k = 0;

        // 4-way unrolled SIMD
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);

            s::Store(s::Log1p(d, v0), d, vals + k + 0 * lanes);
            s::Store(s::Log1p(d, v1), d, vals + k + 1 * lanes);
            s::Store(s::Log1p(d, v2), d, vals + k + 2 * lanes);
            s::Store(s::Log1p(d, v3), d, vals + k + 3 * lanes);
        }

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Log1p(d, v), d, vals + k);
        }

        for (; k < len; ++k) {
            vals[k] = std::log1p(vals[k]);
        }
    });

    return owned;
}

/// @brief Apply log2p1 transform to mapped matrix - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> log2p1_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix
) {
    constexpr double INV_LN2 = 1.44269504088896340736;

    auto owned = matrix.materialize();

    const Index n_primary = scl::primary_size(matrix);

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_inv_ln2 = s::Set(d, INV_LN2);

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* SCL_RESTRICT vals = owned.data.data() + start;

        Index k = 0;

        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);

            s::Store(s::Mul(s::Log1p(d, v0), v_inv_ln2), d, vals + k + 0 * lanes);
            s::Store(s::Mul(s::Log1p(d, v1), v_inv_ln2), d, vals + k + 1 * lanes);
            s::Store(s::Mul(s::Log1p(d, v2), v_inv_ln2), d, vals + k + 2 * lanes);
            s::Store(s::Mul(s::Log1p(d, v3), v_inv_ln2), d, vals + k + 3 * lanes);
        }

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Mul(s::Log1p(d, v), v_inv_ln2), d, vals + k);
        }

        for (; k < len; ++k) {
            vals[k] = std::log1p(vals[k]) * INV_LN2;
        }
    });

    return owned;
}

/// @brief Apply expm1 transform to mapped matrix - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> expm1_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix
) {
    auto owned = matrix.materialize();

    const Index n_primary = scl::primary_size(matrix);

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* SCL_RESTRICT vals = owned.data.data() + start;

        Index k = 0;

        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);

            s::Store(s::Expm1(d, v0), d, vals + k + 0 * lanes);
            s::Store(s::Expm1(d, v1), d, vals + k + 1 * lanes);
            s::Store(s::Expm1(d, v2), d, vals + k + 2 * lanes);
            s::Store(s::Expm1(d, v3), d, vals + k + 3 * lanes);
        }

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Expm1(d, v), d, vals + k);
        }

        for (; k < len; ++k) {
            vals[k] = std::expm1(vals[k]);
        }
    });

    return owned;
}

// =============================================================================
// MappedVirtualSparse Transforms
// =============================================================================

/// @brief Apply log1p transform to MappedVirtualSparse - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> log1p_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix
) {
    auto owned = matrix.materialize();

    const Index n_primary = scl::primary_size(matrix);

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* SCL_RESTRICT vals = owned.data.data() + start;

        Index k = 0;

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Log1p(d, v), d, vals + k);
        }

        for (; k < len; ++k) {
            vals[k] = std::log1p(vals[k]);
        }
    });

    return owned;
}

/// @brief Apply log2p1 transform to MappedVirtualSparse - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> log2p1_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix
) {
    constexpr double INV_LN2 = 1.44269504088896340736;

    auto owned = matrix.materialize();

    const Index n_primary = scl::primary_size(matrix);

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_inv_ln2 = s::Set(d, INV_LN2);

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* SCL_RESTRICT vals = owned.data.data() + start;

        Index k = 0;

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Mul(s::Log1p(d, v), v_inv_ln2), d, vals + k);
        }

        for (; k < len; ++k) {
            vals[k] = std::log1p(vals[k]) * INV_LN2;
        }
    });

    return owned;
}

/// @brief Apply expm1 transform to MappedVirtualSparse - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> expm1_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix
) {
    auto owned = matrix.materialize();

    const Index n_primary = scl::primary_size(matrix);

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* SCL_RESTRICT vals = owned.data.data() + start;

        Index k = 0;

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Expm1(d, v), d, vals + k);
        }

        for (; k < len; ++k) {
            vals[k] = std::expm1(vals[k]);
        }
    });

    return owned;
}

} // namespace scl::kernel::log1p::mapped
