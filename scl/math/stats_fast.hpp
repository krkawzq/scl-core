#pragma once

/// @file scl/math/stats_fast.hpp
/// @brief Fast approximate statistical distribution functions
///
/// This header provides:
///   - Approximate error functions using Abramowitz-Stegun (~1e-7 precision)
///   - Approximate normal distribution functions
///   - Both scalar and SIMD implementations
///
/// @note For full precision (~15 digits), see scl/math/stats.hpp
/// @note Trades accuracy for speed (~3-5x faster than precise versions)

#include "scl/core/simd.hpp"

#include <cmath>

namespace scl::math {

// =============================================================================
// SECTION 1: Approximate Error Functions (Scalar)
// =============================================================================

/// @brief Approximate complementary error function
/// @param[in] x Input value
/// @return Approximate erfc(x)
/// @note Uses Abramowitz-Stegun rational approximation (~1e-7 precision)
/// @note Significantly faster than std::erfc
[[nodiscard]]
SCL_FORCE_INLINE
auto erfc_fast(double x) -> double {
    const double ax = std::abs(x);
    const double t = 1.0 / (1.0 + 0.5 * ax);

    // Horner's method polynomial evaluation
    const double tau = t * std::exp(
        -ax * ax
        - 1.26551223
        + t * ( 1.00002368
        + t * ( 0.37409196
        + t * ( 0.09678418
        + t * (-0.18628806
        + t * ( 0.27886807
        + t * (-1.13520398
        + t * ( 1.48851587
        + t * (-0.82215223
        + t * ( 0.17087277 )))))))))
    );

    double r = (x >= 0.0) ? tau : 2.0 - tau;

    // Clamp to valid range
    if (r < 0.0) [[unlikely]] r = 0.0;
    if (r > 2.0) [[unlikely]] r = 2.0;

    return r;
}

// =============================================================================
// SECTION 2: Approximate Normal Distribution Functions (Scalar)
// =============================================================================

/// @brief Approximate normal survival function
/// @param[in] z Z-score (standardized value)
/// @return Approximate P(Z > z)
/// @note Uses fast erfc approximation (~1e-7 precision)
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_sf_fast(double z) -> double {
    return 0.5 * erfc_fast(z * 0.7071067811865475);
}

/// @brief Approximate normal CDF
/// @param[in] z Z-score (standardized value)
/// @return Approximate P(Z <= z)
/// @note Uses fast erfc approximation (~1e-7 precision)
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_cdf_fast(double z) -> double {
    return 0.5 * erfc_fast(-z * 0.7071067811865475);
}

// =============================================================================
// SECTION 3: SIMD Implementations
// =============================================================================

namespace simd {

namespace s = scl::simd;

/// @brief SIMD approximate erfc using Abramowitz-Stegun approximation
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] x Input vector
/// @return Approximate erfc(x) for each lane
/// @note ~1e-7 precision, significantly faster than precise version
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto erfc_fast(D d, V x) -> V {
    const auto zero = s::Zero(d);
    const auto half = s::Set(d, 0.5);
    const auto one = s::Set(d, 1.0);
    const auto two = s::Set(d, 2.0);

    auto ax = s::Abs(x);

    auto t = s::Div(one, s::Add(one, s::Mul(half, ax)));

    auto neg_ax2 = s::Neg(s::Mul(ax, ax));

    // Polynomial evaluation using Horner's method
    auto poly = s::Set(d, 0.17087277);
    poly = s::MulAdd(poly, t, s::Set(d, -0.82215223));
    poly = s::MulAdd(poly, t, s::Set(d,  1.48851587));
    poly = s::MulAdd(poly, t, s::Set(d, -1.13520398));
    poly = s::MulAdd(poly, t, s::Set(d,  0.27886807));
    poly = s::MulAdd(poly, t, s::Set(d, -0.18628806));
    poly = s::MulAdd(poly, t, s::Set(d,  0.09678418));
    poly = s::MulAdd(poly, t, s::Set(d,  0.37409196));
    poly = s::MulAdd(poly, t, s::Set(d,  1.00002368));

    auto exp_arg = s::Add(neg_ax2, s::Set(d, -1.26551223));
    exp_arg = s::MulAdd(poly, t, exp_arg);

    auto tau = s::Mul(t, s::Exp(d, exp_arg));

    // Result selection based on sign of x
    auto mask_pos = s::Ge(x, zero);
    auto r = s::IfThenElse(mask_pos, tau, s::Sub(two, tau));

    // Clamp to [0, 2]
    r = s::Min(s::Max(r, zero), two);

    return r;
}

/// @brief SIMD approximate normal survival function
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] z Z-score vector
/// @return Approximate SF(z) for each lane
/// @note Uses fast erfc approximation (~1e-7 precision)
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_sf_fast(D d, V z) -> V {
    const auto inv_sqrt2 = s::Set(d, 0.7071067811865475);
    const auto half = s::Set(d, 0.5);

    auto arg = s::Mul(z, inv_sqrt2);
    return s::Mul(half, erfc_fast(d, arg));
}

/// @brief SIMD approximate normal CDF
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] z Z-score vector
/// @return Approximate Φ(z) for each lane
/// @note Uses fast erfc approximation (~1e-7 precision)
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_cdf_fast(D d, V z) -> V {
    const auto inv_sqrt2 = s::Set(d, 0.7071067811865475);
    const auto half = s::Set(d, 0.5);

    auto arg = s::Mul(s::Neg(z), inv_sqrt2);
    return s::Mul(half, erfc_fast(d, arg));
}

} // namespace simd

} // namespace scl::math

