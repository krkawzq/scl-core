#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Backend implementations
#include "scl/kernel/log1p_fast_impl.hpp"

#include <cmath>

// =============================================================================
/// @file log1p.hpp
/// @brief Logarithmic Transformations with SIMD
///
/// ## Supported Transforms
///
/// - log1p(x): ln(1 + x) - Natural logarithm
/// - log2p1(x): log2(1 + x) - Base-2 logarithm
/// - expm1(x): exp(x) - 1 - Inverse of log1p
///
/// ## Why Specialized Functions
///
/// 1. Numerical Stability
///    - std::log(1 + 1e-15) = 0 (catastrophic cancellation)
///    - std::log1p(1e-15) = 1e-15 (correct result)
///
/// 2. Sparsity Preservation
///    - log1p(0) = 0, maintains sparse structure
///    - Only non-zero elements need transformation
///
/// 3. SIMD Acceleration
///    - Highway provides vectorized log1p/expm1
///    - 4-8x faster than scalar std::log
///
/// ## Backend Dispatch
///
/// - CustomSparseLike -> log1p_fast_impl.hpp (direct array access)
/// - VirtualSparseLike -> log1p_fast_impl.hpp (pointer indirection)
/// - MappedSparseLike -> log1p_mapped_impl.hpp (returns OwnedSparse)
/// - Generic -> This file (fallback)
///
/// ## Performance
///
/// Target: ~1.5 GB/s per core (memory bandwidth limited)
// =============================================================================

namespace scl::kernel {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace log1p_config {
    constexpr double INV_LN2 = 1.44269504088896340736;
}

// =============================================================================
// SECTION 2: Array Operations
// =============================================================================

/// @brief Apply ln(1 + x) to array (SIMD, 4-way unrolled)
///
/// @param vals Input/output array (modified in-place)
SCL_FORCE_INLINE void log1p_inplace(Array<Real> vals) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    size_t k = 0;

    // 4-way unrolled
    for (; k + 4 * lanes <= vals.len; k += 4 * lanes) {
        auto v0 = s::Load(d, vals.ptr + k + 0 * lanes);
        auto v1 = s::Load(d, vals.ptr + k + 1 * lanes);
        auto v2 = s::Load(d, vals.ptr + k + 2 * lanes);
        auto v3 = s::Load(d, vals.ptr + k + 3 * lanes);

        s::Store(s::Log1p(d, v0), d, vals.ptr + k + 0 * lanes);
        s::Store(s::Log1p(d, v1), d, vals.ptr + k + 1 * lanes);
        s::Store(s::Log1p(d, v2), d, vals.ptr + k + 2 * lanes);
        s::Store(s::Log1p(d, v3), d, vals.ptr + k + 3 * lanes);
    }

    for (; k + lanes <= vals.len; k += lanes) {
        auto v = s::Load(d, vals.ptr + k);
        s::Store(s::Log1p(d, v), d, vals.ptr + k);
    }

    for (; k < vals.len; ++k) {
        vals[k] = std::log1p(vals[k]);
    }
}

/// @brief Apply log2(1 + x) to array (SIMD, 4-way unrolled)
SCL_FORCE_INLINE void log2p1_inplace(Array<Real> vals) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto inv_ln2 = s::Set(d, log1p_config::INV_LN2);

    size_t k = 0;

    for (; k + 4 * lanes <= vals.len; k += 4 * lanes) {
        auto v0 = s::Load(d, vals.ptr + k + 0 * lanes);
        auto v1 = s::Load(d, vals.ptr + k + 1 * lanes);
        auto v2 = s::Load(d, vals.ptr + k + 2 * lanes);
        auto v3 = s::Load(d, vals.ptr + k + 3 * lanes);

        s::Store(s::Mul(s::Log1p(d, v0), inv_ln2), d, vals.ptr + k + 0 * lanes);
        s::Store(s::Mul(s::Log1p(d, v1), inv_ln2), d, vals.ptr + k + 1 * lanes);
        s::Store(s::Mul(s::Log1p(d, v2), inv_ln2), d, vals.ptr + k + 2 * lanes);
        s::Store(s::Mul(s::Log1p(d, v3), inv_ln2), d, vals.ptr + k + 3 * lanes);
    }

    for (; k + lanes <= vals.len; k += lanes) {
        auto v = s::Load(d, vals.ptr + k);
        s::Store(s::Mul(s::Log1p(d, v), inv_ln2), d, vals.ptr + k);
    }

    for (; k < vals.len; ++k) {
        vals[k] = std::log1p(vals[k]) * log1p_config::INV_LN2;
    }
}

/// @brief Apply exp(x) - 1 to array (SIMD, 4-way unrolled)
SCL_FORCE_INLINE void expm1_inplace(Array<Real> vals) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    size_t k = 0;

    for (; k + 4 * lanes <= vals.len; k += 4 * lanes) {
        auto v0 = s::Load(d, vals.ptr + k + 0 * lanes);
        auto v1 = s::Load(d, vals.ptr + k + 1 * lanes);
        auto v2 = s::Load(d, vals.ptr + k + 2 * lanes);
        auto v3 = s::Load(d, vals.ptr + k + 3 * lanes);

        s::Store(s::Expm1(d, v0), d, vals.ptr + k + 0 * lanes);
        s::Store(s::Expm1(d, v1), d, vals.ptr + k + 1 * lanes);
        s::Store(s::Expm1(d, v2), d, vals.ptr + k + 2 * lanes);
        s::Store(s::Expm1(d, v3), d, vals.ptr + k + 3 * lanes);
    }

    for (; k + lanes <= vals.len; k += lanes) {
        auto v = s::Load(d, vals.ptr + k);
        s::Store(s::Expm1(d, v), d, vals.ptr + k);
    }

    for (; k < vals.len; ++k) {
        vals[k] = std::expm1(vals[k]);
    }
}

// =============================================================================
// SECTION 3: Sparse Matrix Operations (Unified API)
// =============================================================================

/// @brief Apply ln(1 + x) to sparse matrix values (in-place)
///
/// For CustomSparse/VirtualSparse: modifies data in-place
/// For MappedSparse: NOT SUPPORTED (use log1p::mapped::log1p_mapped instead)
///
/// @param mat Input/output sparse matrix
template <typename MatrixT>
    requires AnySparse<MatrixT>
void log1p_inplace(MatrixT& mat) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR> || VirtualSparseLike<MatrixT, IsCSR>) {
        log1p::fast::log1p_inplace_fast<MatrixT, IsCSR>(mat);
    } else {
        // Generic fallback
        const Index primary_dim = scl::primary_size(mat);

        scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
            Index idx = static_cast<Index>(p);
            auto vals = scl::primary_values(mat, idx);

            if (vals.len > 0) {
                using T = typename MatrixT::ValueType;
                Array<T> mutable_vals(const_cast<T*>(vals.ptr), vals.len);

                namespace s = scl::simd;
                const s::Tag d;
                const size_t lanes = s::lanes();

                size_t k = 0;
                for (; k + lanes <= mutable_vals.len; k += lanes) {
                    auto v = s::Load(d, mutable_vals.ptr + k);
                    s::Store(s::Log1p(d, v), d, mutable_vals.ptr + k);
                }
                for (; k < mutable_vals.len; ++k) {
                    mutable_vals[k] = std::log1p(mutable_vals[k]);
                }
            }
        });
    }
}

/// @brief Apply log2(1 + x) to sparse matrix values (in-place)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void log2p1_inplace(MatrixT& mat) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR> || VirtualSparseLike<MatrixT, IsCSR>) {
        log1p::fast::log2p1_inplace_fast<MatrixT, IsCSR>(mat);
    } else {
        const Index primary_dim = scl::primary_size(mat);

        scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
            Index idx = static_cast<Index>(p);
            auto vals = scl::primary_values(mat, idx);

            if (vals.len > 0) {
                using T = typename MatrixT::ValueType;
                Array<T> mutable_vals(const_cast<T*>(vals.ptr), vals.len);

                namespace s = scl::simd;
                const s::Tag d;
                const size_t lanes = s::lanes();
                const auto inv_ln2 = s::Set(d, log1p_config::INV_LN2);

                size_t k = 0;
                for (; k + lanes <= mutable_vals.len; k += lanes) {
                    auto v = s::Load(d, mutable_vals.ptr + k);
                    s::Store(s::Mul(s::Log1p(d, v), inv_ln2), d, mutable_vals.ptr + k);
                }
                for (; k < mutable_vals.len; ++k) {
                    mutable_vals[k] = std::log1p(mutable_vals[k]) * log1p_config::INV_LN2;
                }
            }
        });
    }
}

/// @brief Apply exp(x) - 1 to sparse matrix values (in-place)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void expm1_inplace(MatrixT& mat) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR> || VirtualSparseLike<MatrixT, IsCSR>) {
        log1p::fast::expm1_inplace_fast<MatrixT, IsCSR>(mat);
    } else {
        const Index primary_dim = scl::primary_size(mat);

        scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
            Index idx = static_cast<Index>(p);
            auto vals = scl::primary_values(mat, idx);

            if (vals.len > 0) {
                using T = typename MatrixT::ValueType;
                Array<T> mutable_vals(const_cast<T*>(vals.ptr), vals.len);

                namespace s = scl::simd;
                const s::Tag d;
                const size_t lanes = s::lanes();

                size_t k = 0;
                for (; k + lanes <= mutable_vals.len; k += lanes) {
                    auto v = s::Load(d, mutable_vals.ptr + k);
                    s::Store(s::Expm1(d, v), d, mutable_vals.ptr + k);
                }
                for (; k < mutable_vals.len; ++k) {
                    mutable_vals[k] = std::expm1(mutable_vals[k]);
                }
            }
        });
    }
}

} // namespace scl::kernel
