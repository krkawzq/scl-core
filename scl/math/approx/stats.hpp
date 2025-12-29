#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/math/approx/stats.hpp
// BRIEF: Approximate Statistical Distribution Functions
// =============================================================================

// Abramowitz-Stegun approximation for erfc (~1e-7 precision)
// For full precision, see scl/math/stats.hpp

namespace scl::math::approx {

// =============================================================================
// Scalar Implementations
// =============================================================================

// Approximate erfc using Abramowitz-Stegun rational approximation
SCL_FORCE_INLINE double erfc(double x) {
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
    if (r < 0.0) r = 0.0;
    if (r > 2.0) r = 2.0;

    return r;
}

SCL_FORCE_INLINE double normal_sf(double z) {
    return 0.5 * erfc(z * 0.7071067811865475);
}

SCL_FORCE_INLINE double normal_cdf(double z) {
    return 0.5 * erfc(-z * 0.7071067811865475);
}

// =============================================================================
// SIMD Implementations
// =============================================================================

namespace simd {

namespace s = scl::simd;

// SIMD approximate erfc using Abramowitz-Stegun approximation
template <class D, class V>
SCL_FORCE_INLINE V erfc(D d, V x) {
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

template <class D, class V>
SCL_FORCE_INLINE V normal_sf(D d, V z) {
    const auto inv_sqrt2 = s::Set(d, 0.7071067811865475);
    const auto half = s::Set(d, 0.5);

    auto arg = s::Mul(z, inv_sqrt2);
    return s::Mul(half, erfc(d, arg));
}

template <class D, class V>
SCL_FORCE_INLINE V normal_cdf(D d, V z) {
    const auto inv_sqrt2 = s::Set(d, 0.7071067811865475);
    const auto half = s::Set(d, 0.5);

    auto arg = s::Mul(s::Neg(z), inv_sqrt2);
    return s::Mul(half, erfc(d, arg));
}

} // namespace simd

} // namespace scl::math::approx
