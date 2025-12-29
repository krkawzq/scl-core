#pragma once

#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"

#include <cmath>

// =============================================================================
// Precise Statistical Distribution Functions
// Full precision via std::erfc, std::erf (~15 significant digits)
// For approximate version, see scl/math/approx/stats.hpp
// =============================================================================

namespace scl::math {

// =============================================================================
// Scalar Implementations
// =============================================================================

SCL_FORCE_INLINE double erfc(double x) {
    return std::erfc(x);
}

SCL_FORCE_INLINE double erf(double x) {
    return std::erf(x);
}

SCL_FORCE_INLINE double normal_cdf(double z) {
    return 0.5 * std::erfc(-z * 0.7071067811865475);
}

SCL_FORCE_INLINE double normal_sf(double z) {
    return 0.5 * std::erfc(z * 0.7071067811865475);
}

SCL_FORCE_INLINE double normal_pdf(double z) {
    constexpr double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * z * z);
}

SCL_FORCE_INLINE double normal_logcdf(double z) {
    if (z < -20.0) {
        // Asymptotic expansion for large negative z
        double z2 = z * z;
        return -0.5 * z2 - 0.9189385332046727 - std::log(-z);
    }
    return std::log(normal_cdf(z));
}

SCL_FORCE_INLINE double normal_logsf(double z) {
    if (z > 20.0) {
        // Asymptotic expansion for large positive z
        double z2 = z * z;
        return -0.5 * z2 - 0.9189385332046727 - std::log(z);
    }
    return std::log(normal_sf(z));
}

// =============================================================================
// SIMD Implementations
// =============================================================================

namespace simd {

namespace s = scl::simd;

// Lane-wise precise erfc via std::erfc
template <class D, class V>
SCL_FORCE_INLINE V erfc(D d, V x) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_in[s::Lanes(d)];
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_out[s::Lanes(d)];

    s::Store(x, d, buffer_in);

    for (size_t i = 0; i < s::Lanes(d); ++i) {
        buffer_out[i] = std::erfc(buffer_in[i]);
    }

    return s::Load(d, buffer_out);
}

template <class D, class V>
SCL_FORCE_INLINE V erf(D d, V x) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_in[s::Lanes(d)];
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_out[s::Lanes(d)];

    s::Store(x, d, buffer_in);

    for (size_t i = 0; i < s::Lanes(d); ++i) {
        buffer_out[i] = std::erf(buffer_in[i]);
    }

    return s::Load(d, buffer_out);
}

template <class D, class V>
SCL_FORCE_INLINE V normal_cdf(D d, V z) {
    const auto inv_sqrt2 = s::Set(d, 0.7071067811865475);
    const auto half = s::Set(d, 0.5);

    auto arg = s::Mul(s::Neg(z), inv_sqrt2);
    return s::Mul(half, erfc(d, arg));
}

template <class D, class V>
SCL_FORCE_INLINE V normal_sf(D d, V z) {
    const auto inv_sqrt2 = s::Set(d, 0.7071067811865475);
    const auto half = s::Set(d, 0.5);

    auto arg = s::Mul(z, inv_sqrt2);
    return s::Mul(half, erfc(d, arg));
}

template <class D, class V>
SCL_FORCE_INLINE V normal_pdf(D d, V z) {
    const auto inv_sqrt_2pi = s::Set(d, 0.3989422804014327);
    const auto half = s::Set(d, 0.5);

    auto z2 = s::Mul(z, z);
    auto exp_term = s::Exp(d, s::Neg(s::Mul(half, z2)));
    return s::Mul(inv_sqrt_2pi, exp_term);
}

template <class D, class V>
SCL_FORCE_INLINE V normal_logcdf(D d, V z) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_in[s::Lanes(d)];
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_out[s::Lanes(d)];

    s::Store(z, d, buffer_in);

    for (size_t i = 0; i < s::Lanes(d); ++i) {
        buffer_out[i] = scl::math::normal_logcdf(buffer_in[i]);
    }

    return s::Load(d, buffer_out);
}

template <class D, class V>
SCL_FORCE_INLINE V normal_logsf(D d, V z) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_in[s::Lanes(d)];
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_out[s::Lanes(d)];

    s::Store(z, d, buffer_in);

    for (size_t i = 0; i < s::Lanes(d); ++i) {
        buffer_out[i] = scl::math::normal_logsf(buffer_in[i]);
    }

    return s::Load(d, buffer_out);
}

} // namespace simd

} // namespace scl::math
