#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file log1p_mapped_impl.hpp
/// @brief Logarithmic Transforms for Memory-Mapped Sparse Matrices
///
/// ## Design Note
///
/// Mapped matrices are READ-ONLY (memory-mapped files).
/// For "in-place" operations, we must:
/// 1. Materialize to OwnedSparse (copies data)
/// 2. Apply transform on owned data
/// 3. Return OwnedSparse (caller takes ownership)
///
/// This is semantically "out-of-place" but provides the same API.
///
/// ## Optimizations
///
/// 1. Fused Copy + Transform
///    - Read from mapped, transform, write to owned in one pass
///    - Avoids extra copy compared to materialize() + inplace()
///
/// 2. 4-Way SIMD Unrolling
///    - Process 4 vectors per iteration
///
/// 3. Prefetch for Sequential Access
///    - Hint OS page cache for mapped data
///
/// ## Supported Transforms
///
/// - log1p_mapped: ln(1 + x)
/// - log2p1_mapped: log2(1 + x)
/// - expm1_mapped: exp(x) - 1
// =============================================================================

namespace scl::kernel::log1p::mapped {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 64;
    constexpr double INV_LN2 = 1.44269504088896340736;
}

// =============================================================================
// SECTION 2: SIMD Transform Utilities (Fused Copy + Transform)
// =============================================================================

namespace detail {

/// @brief Fused copy + log1p (out-of-place)
template <typename T>
SCL_FORCE_INLINE void copy_log1p_simd(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Index len
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    Index k = 0;

    // 4-way unrolled
    for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
        if (k + config::PREFETCH_DISTANCE < static_cast<Size>(len)) {
            SCL_PREFETCH_READ(src + k + config::PREFETCH_DISTANCE, 0);
        }

        auto v0 = s::Load(d, src + k + 0 * lanes);
        auto v1 = s::Load(d, src + k + 1 * lanes);
        auto v2 = s::Load(d, src + k + 2 * lanes);
        auto v3 = s::Load(d, src + k + 3 * lanes);

        s::Store(s::Log1p(d, v0), d, dst + k + 0 * lanes);
        s::Store(s::Log1p(d, v1), d, dst + k + 1 * lanes);
        s::Store(s::Log1p(d, v2), d, dst + k + 2 * lanes);
        s::Store(s::Log1p(d, v3), d, dst + k + 3 * lanes);
    }

    for (; k + static_cast<Index>(lanes) <= len; k += static_cast<Index>(lanes)) {
        auto v = s::Load(d, src + k);
        s::Store(s::Log1p(d, v), d, dst + k);
    }

    for (; k < len; ++k) {
        dst[k] = std::log1p(src[k]);
    }
}

/// @brief Fused copy + log2p1
template <typename T>
SCL_FORCE_INLINE void copy_log2p1_simd(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Index len
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_inv_ln2 = s::Set(d, config::INV_LN2);

    Index k = 0;

    for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
        if (k + config::PREFETCH_DISTANCE < static_cast<Size>(len)) {
            SCL_PREFETCH_READ(src + k + config::PREFETCH_DISTANCE, 0);
        }

        auto v0 = s::Load(d, src + k + 0 * lanes);
        auto v1 = s::Load(d, src + k + 1 * lanes);
        auto v2 = s::Load(d, src + k + 2 * lanes);
        auto v3 = s::Load(d, src + k + 3 * lanes);

        s::Store(s::Mul(s::Log1p(d, v0), v_inv_ln2), d, dst + k + 0 * lanes);
        s::Store(s::Mul(s::Log1p(d, v1), v_inv_ln2), d, dst + k + 1 * lanes);
        s::Store(s::Mul(s::Log1p(d, v2), v_inv_ln2), d, dst + k + 2 * lanes);
        s::Store(s::Mul(s::Log1p(d, v3), v_inv_ln2), d, dst + k + 3 * lanes);
    }

    for (; k + static_cast<Index>(lanes) <= len; k += static_cast<Index>(lanes)) {
        auto v = s::Load(d, src + k);
        s::Store(s::Mul(s::Log1p(d, v), v_inv_ln2), d, dst + k);
    }

    for (; k < len; ++k) {
        dst[k] = std::log1p(src[k]) * config::INV_LN2;
    }
}

/// @brief Fused copy + expm1
template <typename T>
SCL_FORCE_INLINE void copy_expm1_simd(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Index len
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    Index k = 0;

    for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
        if (k + config::PREFETCH_DISTANCE < static_cast<Size>(len)) {
            SCL_PREFETCH_READ(src + k + config::PREFETCH_DISTANCE, 0);
        }

        auto v0 = s::Load(d, src + k + 0 * lanes);
        auto v1 = s::Load(d, src + k + 1 * lanes);
        auto v2 = s::Load(d, src + k + 2 * lanes);
        auto v3 = s::Load(d, src + k + 3 * lanes);

        s::Store(s::Expm1(d, v0), d, dst + k + 0 * lanes);
        s::Store(s::Expm1(d, v1), d, dst + k + 1 * lanes);
        s::Store(s::Expm1(d, v2), d, dst + k + 2 * lanes);
        s::Store(s::Expm1(d, v3), d, dst + k + 3 * lanes);
    }

    for (; k + static_cast<Index>(lanes) <= len; k += static_cast<Index>(lanes)) {
        auto v = s::Load(d, src + k);
        s::Store(s::Expm1(d, v), d, dst + k);
    }

    for (; k < len; ++k) {
        dst[k] = std::expm1(src[k]);
    }
}

} // namespace detail

// =============================================================================
// SECTION 3: MappedCustomSparse Transforms
// =============================================================================

/// @brief log1p for MappedCustomSparse - returns materialized result
///
/// Fused copy + transform for better performance than materialize() + inplace()
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> log1p_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix
) {
    const Index n_primary = scl::primary_size(matrix);
    const Index nnz = matrix.nnz();

    // Allocate owned storage
    scl::io::OwnedSparse<T, IsCSR> owned(
        matrix.rows, matrix.cols, nnz);

    // Copy structure (indptr, indices)
    std::copy(matrix.indptr(), matrix.indptr() + n_primary + 1, owned.indptr.begin());
    std::copy(matrix.indices(), matrix.indices() + nnz, owned.indices.begin());

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    // Fused copy + transform for data
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = matrix.indptr()[p];
        Index end = matrix.indptr()[p + 1];
        Index len = end - start;

        if (len > 0) {
            const T* src = matrix.data() + start;
            T* dst = owned.data.data() + start;
            detail::copy_log1p_simd(src, dst, len);
        }
    });

    return owned;
}

/// @brief log2p1 for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> log2p1_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix
) {
    const Index n_primary = scl::primary_size(matrix);
    const Index nnz = matrix.nnz();

    scl::io::OwnedSparse<T, IsCSR> owned(
        matrix.rows, matrix.cols, nnz);

    std::copy(matrix.indptr(), matrix.indptr() + n_primary + 1, owned.indptr.begin());
    std::copy(matrix.indices(), matrix.indices() + nnz, owned.indices.begin());

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = matrix.indptr()[p];
        Index end = matrix.indptr()[p + 1];
        Index len = end - start;

        if (len > 0) {
            const T* src = matrix.data() + start;
            T* dst = owned.data.data() + start;
            detail::copy_log2p1_simd(src, dst, len);
        }
    });

    return owned;
}

/// @brief expm1 for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> expm1_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix
) {
    const Index n_primary = scl::primary_size(matrix);
    const Index nnz = matrix.nnz();

    scl::io::OwnedSparse<T, IsCSR> owned(
        matrix.rows, matrix.cols, nnz);

    std::copy(matrix.indptr(), matrix.indptr() + n_primary + 1, owned.indptr.begin());
    std::copy(matrix.indices(), matrix.indices() + nnz, owned.indices.begin());

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = matrix.indptr()[p];
        Index end = matrix.indptr()[p + 1];
        Index len = end - start;

        if (len > 0) {
            const T* src = matrix.data() + start;
            T* dst = owned.data.data() + start;
            detail::copy_expm1_simd(src, dst, len);
        }
    });

    return owned;
}

// =============================================================================
// SECTION 4: MappedVirtualSparse Transforms
// =============================================================================

/// @brief log1p for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> log1p_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix
) {
    // Materialize first (VirtualSparse structure is complex)
    auto owned = matrix.materialize();

    const Index n_primary = scl::primary_size(matrix);

    // Apply transform in-place on owned data
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len > 0) {
            T* vals = owned.data.data() + start;

            namespace s = scl::simd;
            const s::Tag d;
            const size_t lanes = s::lanes();

            Index k = 0;

            for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
                auto v0 = s::Load(d, vals + k + 0 * lanes);
                auto v1 = s::Load(d, vals + k + 1 * lanes);
                auto v2 = s::Load(d, vals + k + 2 * lanes);
                auto v3 = s::Load(d, vals + k + 3 * lanes);

                s::Store(s::Log1p(d, v0), d, vals + k + 0 * lanes);
                s::Store(s::Log1p(d, v1), d, vals + k + 1 * lanes);
                s::Store(s::Log1p(d, v2), d, vals + k + 2 * lanes);
                s::Store(s::Log1p(d, v3), d, vals + k + 3 * lanes);
            }

            for (; k + static_cast<Index>(lanes) <= len; k += static_cast<Index>(lanes)) {
                auto v = s::Load(d, vals + k);
                s::Store(s::Log1p(d, v), d, vals + k);
            }

            for (; k < len; ++k) {
                vals[k] = std::log1p(vals[k]);
            }
        }
    });

    return owned;
}

/// @brief log2p1 for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> log2p1_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix
) {
    auto owned = matrix.materialize();

    const Index n_primary = scl::primary_size(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len > 0) {
            T* vals = owned.data.data() + start;

            namespace s = scl::simd;
            const s::Tag d;
            const size_t lanes = s::lanes();
            const auto v_inv_ln2 = s::Set(d, config::INV_LN2);

            Index k = 0;

            for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
                auto v0 = s::Load(d, vals + k + 0 * lanes);
                auto v1 = s::Load(d, vals + k + 1 * lanes);
                auto v2 = s::Load(d, vals + k + 2 * lanes);
                auto v3 = s::Load(d, vals + k + 3 * lanes);

                s::Store(s::Mul(s::Log1p(d, v0), v_inv_ln2), d, vals + k + 0 * lanes);
                s::Store(s::Mul(s::Log1p(d, v1), v_inv_ln2), d, vals + k + 1 * lanes);
                s::Store(s::Mul(s::Log1p(d, v2), v_inv_ln2), d, vals + k + 2 * lanes);
                s::Store(s::Mul(s::Log1p(d, v3), v_inv_ln2), d, vals + k + 3 * lanes);
            }

            for (; k + static_cast<Index>(lanes) <= len; k += static_cast<Index>(lanes)) {
                auto v = s::Load(d, vals + k);
                s::Store(s::Mul(s::Log1p(d, v), v_inv_ln2), d, vals + k);
            }

            for (; k < len; ++k) {
                vals[k] = std::log1p(vals[k]) * config::INV_LN2;
            }
        }
    });

    return owned;
}

/// @brief expm1 for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> expm1_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix
) {
    auto owned = matrix.materialize();

    const Index n_primary = scl::primary_size(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len > 0) {
            T* vals = owned.data.data() + start;

            namespace s = scl::simd;
            const s::Tag d;
            const size_t lanes = s::lanes();

            Index k = 0;

            for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
                auto v0 = s::Load(d, vals + k + 0 * lanes);
                auto v1 = s::Load(d, vals + k + 1 * lanes);
                auto v2 = s::Load(d, vals + k + 2 * lanes);
                auto v3 = s::Load(d, vals + k + 3 * lanes);

                s::Store(s::Expm1(d, v0), d, vals + k + 0 * lanes);
                s::Store(s::Expm1(d, v1), d, vals + k + 1 * lanes);
                s::Store(s::Expm1(d, v2), d, vals + k + 2 * lanes);
                s::Store(s::Expm1(d, v3), d, vals + k + 3 * lanes);
            }

            for (; k + static_cast<Index>(lanes) <= len; k += static_cast<Index>(lanes)) {
                auto v = s::Load(d, vals + k);
                s::Store(s::Expm1(d, v), d, vals + k);
            }

            for (; k < len; ++k) {
                vals[k] = std::expm1(vals[k]);
            }
        }
    });

    return owned;
}

} // namespace scl::kernel::log1p::mapped
