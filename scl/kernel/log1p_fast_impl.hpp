#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file log1p_fast_impl.hpp
/// @brief Extreme Performance Logarithmic Transforms
///
/// ## Supported Transforms
///
/// - log1p(x): ln(1 + x)    - Sparsity preserving (log1p(0) = 0)
/// - log2p1(x): log2(1 + x) - Base-2 variant
/// - expm1(x): exp(x) - 1   - Inverse of log1p
///
/// ## Key Optimizations
///
/// 1. 4-Way SIMD Unrolling
///    - Process 4 vectors per iteration
///    - Better instruction-level parallelism
///
/// 2. Parallel Over Primary Dimension
///    - Each row/column processed independently
///    - No synchronization needed
///
/// 3. Prefetch for Sequential Access
///    - Hint next cache line during processing
///
/// ## Numerical Properties
///
/// log1p(x) is numerically stable for small x:
/// - std::log(1 + 1e-15) = 0 (catastrophic cancellation)
/// - std::log1p(1e-15) = 1e-15 (correct)
///
/// Performance Target: 2-3x faster than generic (~1.5 GB/s per core)
// =============================================================================

namespace scl::kernel::log1p::fast {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 64;
    constexpr double INV_LN2 = 1.44269504088896340736;
    constexpr double LN2 = 0.6931471805599453;
}

// =============================================================================
// SECTION 2: Core SIMD Transform Templates
// =============================================================================

namespace detail {

/// @brief Apply log1p with 4-way SIMD unrolling
template <typename T>
SCL_FORCE_INLINE void apply_log1p_simd(T* SCL_RESTRICT vals, Index len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    Index k = 0;

    // 4-way unrolled SIMD
    for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
        // Prefetch
        if (k + config::PREFETCH_DISTANCE < static_cast<Size>(len)) {
            SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE, 0);
        }

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

/// @brief Apply log2p1 with 4-way SIMD unrolling
template <typename T>
SCL_FORCE_INLINE void apply_log2p1_simd(T* SCL_RESTRICT vals, Index len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_inv_ln2 = s::Set(d, config::INV_LN2);

    Index k = 0;

    for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
        if (k + config::PREFETCH_DISTANCE < static_cast<Size>(len)) {
            SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE, 0);
        }

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

/// @brief Apply expm1 with 4-way SIMD unrolling
template <typename T>
SCL_FORCE_INLINE void apply_expm1_simd(T* SCL_RESTRICT vals, Index len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    Index k = 0;

    for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
        if (k + config::PREFETCH_DISTANCE < static_cast<Size>(len)) {
            SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE, 0);
        }

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

/// @brief Out-of-place log1p (read from src, write to dst)
template <typename T>
SCL_FORCE_INLINE void apply_log1p_simd_copy(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Index len
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    Index k = 0;

    for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
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

/// @brief Out-of-place log2p1
template <typename T>
SCL_FORCE_INLINE void apply_log2p1_simd_copy(
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

/// @brief Out-of-place expm1
template <typename T>
SCL_FORCE_INLINE void apply_expm1_simd_copy(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Index len
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    Index k = 0;

    for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
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
// SECTION 3: CustomSparse Fast Path
// =============================================================================

/// @brief In-place log1p for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void log1p_inplace_custom(CustomSparse<T, IsCSR>& mat) {
    const Index primary_dim = scl::primary_size(mat);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index start = mat.indptr[p];
        Index end = mat.indptr[p + 1];
        Index len = end - start;

        if (len > 0) {
            detail::apply_log1p_simd(mat.data + start, len);
        }
    });
}

/// @brief In-place log2p1 for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void log2p1_inplace_custom(CustomSparse<T, IsCSR>& mat) {
    const Index primary_dim = scl::primary_size(mat);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index start = mat.indptr[p];
        Index end = mat.indptr[p + 1];
        Index len = end - start;

        if (len > 0) {
            detail::apply_log2p1_simd(mat.data + start, len);
        }
    });
}

/// @brief In-place expm1 for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void expm1_inplace_custom(CustomSparse<T, IsCSR>& mat) {
    const Index primary_dim = scl::primary_size(mat);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index start = mat.indptr[p];
        Index end = mat.indptr[p + 1];
        Index len = end - start;

        if (len > 0) {
            detail::apply_expm1_simd(mat.data + start, len);
        }
    });
}

// =============================================================================
// SECTION 4: VirtualSparse Fast Path
// =============================================================================

/// @brief In-place log1p for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void log1p_inplace_virtual(VirtualSparse<T, IsCSR>& mat) {
    const Index primary_dim = scl::primary_size(mat);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index len = mat.lengths[p];

        if (len > 0) {
            T* vals = static_cast<T*>(mat.data_ptrs[p]);
            detail::apply_log1p_simd(vals, len);
        }
    });
}

/// @brief In-place log2p1 for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void log2p1_inplace_virtual(VirtualSparse<T, IsCSR>& mat) {
    const Index primary_dim = scl::primary_size(mat);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index len = mat.lengths[p];

        if (len > 0) {
            T* vals = static_cast<T*>(mat.data_ptrs[p]);
            detail::apply_log2p1_simd(vals, len);
        }
    });
}

/// @brief In-place expm1 for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void expm1_inplace_virtual(VirtualSparse<T, IsCSR>& mat) {
    const Index primary_dim = scl::primary_size(mat);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index len = mat.lengths[p];

        if (len > 0) {
            T* vals = static_cast<T*>(mat.data_ptrs[p]);
            detail::apply_expm1_simd(vals, len);
        }
    });
}

// =============================================================================
// SECTION 5: Unified Dispatchers
// =============================================================================

/// @brief Auto-dispatch log1p to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void log1p_inplace_fast(MatrixT& mat) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        log1p_inplace_custom(mat);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        log1p_inplace_virtual(mat);
    }
}

/// @brief Auto-dispatch log2p1 to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void log2p1_inplace_fast(MatrixT& mat) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        log2p1_inplace_custom(mat);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        log2p1_inplace_virtual(mat);
    }
}

/// @brief Auto-dispatch expm1 to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void expm1_inplace_fast(MatrixT& mat) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        expm1_inplace_custom(mat);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        expm1_inplace_virtual(mat);
    }
}

} // namespace scl::kernel::log1p::fast
