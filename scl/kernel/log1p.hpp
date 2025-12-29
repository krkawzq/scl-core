#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/log1p.hpp
// BRIEF: Logarithmic transforms with SIMD optimization
// =============================================================================

namespace scl::kernel::log1p {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 64;
    constexpr double INV_LN2 = 1.44269504088896340736;
    constexpr double LN2 = 0.6931471805599453;
}

// =============================================================================
// Core SIMD Transform Templates
// =============================================================================

namespace detail {

template <typename T>
SCL_FORCE_INLINE void apply_log1p_simd(T* SCL_RESTRICT vals, Index len) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const auto lanes = s::Lanes(d);

    Index k = 0;

    for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
        if (static_cast<Size>(k) + config::PREFETCH_DISTANCE < static_cast<Size>(len)) {
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

template <typename T>
SCL_FORCE_INLINE void apply_log2p1_simd(T* SCL_RESTRICT vals, Index len) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const auto lanes = s::Lanes(d);
    const auto v_inv_ln2 = s::Set(d, config::INV_LN2);

    Index k = 0;

    for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
        if (static_cast<Size>(k) + config::PREFETCH_DISTANCE < static_cast<Size>(len)) {
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

template <typename T>
SCL_FORCE_INLINE void apply_expm1_simd(T* SCL_RESTRICT vals, Index len) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const auto lanes = s::Lanes(d);

    Index k = 0;

    for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
        if (static_cast<Size>(k) + config::PREFETCH_DISTANCE < static_cast<Size>(len)) {
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

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool IsCSR>
void log1p_inplace(Sparse<T, IsCSR>& matrix) {
    const Index primary_dim = matrix.primary_dim();

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);

        if (len > 0) {
            auto values = matrix.primary_values_unsafe(idx);
            detail::apply_log1p_simd(values.ptr, len);
        }
    });
}

template <typename T, bool IsCSR>
void log2p1_inplace(Sparse<T, IsCSR>& matrix) {
    const Index primary_dim = matrix.primary_dim();

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);

        if (len > 0) {
            auto values = matrix.primary_values_unsafe(idx);
            detail::apply_log2p1_simd(values.ptr, len);
        }
    });
}

template <typename T, bool IsCSR>
void expm1_inplace(Sparse<T, IsCSR>& matrix) {
    const Index primary_dim = matrix.primary_dim();

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);

        if (len > 0) {
            auto values = matrix.primary_values_unsafe(idx);
            detail::apply_expm1_simd(values.ptr, len);
        }
    });
}

} // namespace scl::kernel::log1p
