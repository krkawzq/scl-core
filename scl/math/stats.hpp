#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"

#include <cmath>

// =============================================================================
/// @file stats.hpp
/// @brief Precise Statistical Distribution Functions
///
/// Provides high-precision implementations of statistical distribution functions
/// using standard library calls. These are intended for applications where
/// numerical accuracy is paramount over raw performance.
///
/// Implemented Functions:
///
/// 1. Error Functions:
///    - erfc: Complementary error function
///    - erf: Error function
///
/// 2. Normal Distribution:
///    - normal_cdf: Cumulative distribution function
///    - normal_sf: Survival function (1 - CDF)
///    - normal_pdf: Probability density function
///    - normal_logcdf: Log of CDF (stable for extremes)
///    - normal_logsf: Log of SF (stable for extremes)
///
/// Implementation Notes:
///
/// - Scalar versions use std::erfc/std::erf for full precision
/// - SIMD versions perform lane-wise scalar calls (slower but precise)
/// - For fast approximations, see scl/math/fast/stats.hpp
///
/// Precision: Full IEEE 754 double precision (~15 significant digits)
// =============================================================================

namespace scl::math {

// =============================================================================
// SECTION 1: Scalar Implementations (Full Precision)
// =============================================================================

/// @brief Complementary error function erfc(x).
///
/// Computes erfc(x) = 1 - erf(x) = (2/sqrt(pi)) * integral from x to infinity
/// of exp(-t^2) dt.
///
/// @param x Input value
/// @return erfc(x) in range [0, 2]
SCL_FORCE_INLINE double erfc(double x) {
    return std::erfc(x);
}

/// @brief Error function erf(x).
///
/// Computes erf(x) = (2/sqrt(pi)) * integral from 0 to x of exp(-t^2) dt.
///
/// @param x Input value
/// @return erf(x) in range [-1, 1]
SCL_FORCE_INLINE double erf(double x) {
    return std::erf(x);
}

/// @brief Normal cumulative distribution function CDF(z).
///
/// Computes P(Z <= z) for standard normal distribution.
/// Formula: CDF(z) = 0.5 * erfc(-z / sqrt(2))
///
/// @param z Z-score (standard normal deviate)
/// @return Probability in range [0, 1]
SCL_FORCE_INLINE double normal_cdf(double z) {
    // 1/sqrt(2) = 0.7071067811865475
    return 0.5 * std::erfc(-z * 0.7071067811865475);
}

/// @brief Normal survival function SF(z) = 1 - CDF(z).
///
/// Computes P(Z > z) for standard normal distribution.
/// More numerically stable than 1 - CDF(z) for large z.
///
/// @param z Z-score (standard normal deviate)
/// @return Probability in range [0, 1]
SCL_FORCE_INLINE double normal_sf(double z) {
    return 0.5 * std::erfc(z * 0.7071067811865475);
}

/// @brief Normal probability density function PDF(z).
///
/// Computes f(z) = (1/sqrt(2*pi)) * exp(-z^2/2).
///
/// @param z Z-score (standard normal deviate)
/// @return Density value (always non-negative)
SCL_FORCE_INLINE double normal_pdf(double z) {
    // 1/sqrt(2*pi) = 0.3989422804014327
    constexpr double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * z * z);
}

/// @brief Natural logarithm of normal CDF.
///
/// Computes log(CDF(z)) with improved numerical stability for extreme values.
/// Uses asymptotic expansion for z < -20.
///
/// @param z Z-score (standard normal deviate)
/// @return log(P(Z <= z))
SCL_FORCE_INLINE double normal_logcdf(double z) {
    if (z < -20.0) {
        // Asymptotic expansion for large negative z:
        // log(Phi(z)) ~ -z^2/2 - log(sqrt(2*pi)) - log(-z)
        // -log(sqrt(2*pi)) = -0.9189385332046727
        double z2 = z * z;
        return -0.5 * z2 - 0.9189385332046727 - std::log(-z);
    }
    return std::log(normal_cdf(z));
}

/// @brief Natural logarithm of normal survival function.
///
/// Computes log(SF(z)) with improved numerical stability for extreme values.
/// Uses asymptotic expansion for z > 20.
///
/// @param z Z-score (standard normal deviate)
/// @return log(P(Z > z))
SCL_FORCE_INLINE double normal_logsf(double z) {
    if (z > 20.0) {
        // Asymptotic expansion for large positive z
        double z2 = z * z;
        return -0.5 * z2 - 0.9189385332046727 - std::log(z);
    }
    return std::log(normal_sf(z));
}

// =============================================================================
// SECTION 2: SIMD Implementations (Full Precision)
// =============================================================================

namespace simd {

namespace s = scl::simd;

/// @brief SIMD complementary error function (lane-wise precise).
///
/// Performs lane-wise calls to std::erfc for full precision.
/// Slower than fast approximation but maintains accuracy.
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param x Input vector
/// @return erfc(x) for each lane
template <class D, class V>
SCL_FORCE_INLINE V erfc(D d, V x) {
    alignas(64) double buffer_in[s::Lanes(d)];
    alignas(64) double buffer_out[s::Lanes(d)];

    s::Store(x, d, buffer_in);

    for (size_t i = 0; i < s::Lanes(d); ++i) {
        buffer_out[i] = std::erfc(buffer_in[i]);
    }

    return s::Load(d, buffer_out);
}

/// @brief SIMD error function (lane-wise precise).
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param x Input vector
/// @return erf(x) for each lane
template <class D, class V>
SCL_FORCE_INLINE V erf(D d, V x) {
    alignas(64) double buffer_in[s::Lanes(d)];
    alignas(64) double buffer_out[s::Lanes(d)];

    s::Store(x, d, buffer_in);

    for (size_t i = 0; i < s::Lanes(d); ++i) {
        buffer_out[i] = std::erf(buffer_in[i]);
    }

    return s::Load(d, buffer_out);
}

/// @brief SIMD normal CDF.
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param z Input Z-scores
/// @return CDF(z) for each lane
template <class D, class V>
SCL_FORCE_INLINE V normal_cdf(D d, V z) {
    const auto inv_sqrt2 = s::Set(d, 0.7071067811865475);
    const auto half = s::Set(d, 0.5);

    auto arg = s::Mul(s::Neg(z), inv_sqrt2);
    return s::Mul(half, erfc(d, arg));
}

/// @brief SIMD normal survival function.
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param z Input Z-scores
/// @return SF(z) for each lane
template <class D, class V>
SCL_FORCE_INLINE V normal_sf(D d, V z) {
    const auto inv_sqrt2 = s::Set(d, 0.7071067811865475);
    const auto half = s::Set(d, 0.5);

    auto arg = s::Mul(z, inv_sqrt2);
    return s::Mul(half, erfc(d, arg));
}

/// @brief SIMD normal PDF.
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param z Input Z-scores
/// @return PDF(z) for each lane
template <class D, class V>
SCL_FORCE_INLINE V normal_pdf(D d, V z) {
    const auto inv_sqrt_2pi = s::Set(d, 0.3989422804014327);
    const auto half = s::Set(d, 0.5);

    auto z2 = s::Mul(z, z);
    auto exp_term = s::Exp(d, s::Neg(s::Mul(half, z2)));
    return s::Mul(inv_sqrt_2pi, exp_term);
}

/// @brief SIMD normal log-CDF (lane-wise precise).
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param z Input Z-scores
/// @return log(CDF(z)) for each lane
template <class D, class V>
SCL_FORCE_INLINE V normal_logcdf(D d, V z) {
    alignas(64) double buffer_in[s::Lanes(d)];
    alignas(64) double buffer_out[s::Lanes(d)];

    s::Store(z, d, buffer_in);

    for (size_t i = 0; i < s::Lanes(d); ++i) {
        buffer_out[i] = scl::math::normal_logcdf(buffer_in[i]);
    }

    return s::Load(d, buffer_out);
}

/// @brief SIMD normal log-SF (lane-wise precise).
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param z Input Z-scores
/// @return log(SF(z)) for each lane
template <class D, class V>
SCL_FORCE_INLINE V normal_logsf(D d, V z) {
    alignas(64) double buffer_in[s::Lanes(d)];
    alignas(64) double buffer_out[s::Lanes(d)];

    s::Store(z, d, buffer_in);

    for (size_t i = 0; i < s::Lanes(d); ++i) {
        buffer_out[i] = scl::math::normal_logsf(buffer_in[i]);
    }

    return s::Load(d, buffer_out);
}

} // namespace simd

} // namespace scl::math
