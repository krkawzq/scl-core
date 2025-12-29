#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/math/stats.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/math/mwu.hpp
// BRIEF: Mann-Whitney U Test Statistics (Precise Implementation)
// =============================================================================

// Uses normal approximation with tie correction for p-value computation
// For approximate version, see scl/math/approx/mwu.hpp

namespace scl::math::mwu {

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

SCL_FORCE_INLINE void moments(
    double n1, double n2, double tie_sum,
    double& mu, double& sd
) {
    const double N = n1 + n2;
    mu = 0.5 * n1 * n2;

    const double denom = N * (N - 1.0);

    double var;
    if (denom > 0.0) {
        const double tie_correction = tie_sum / denom;
        var = (n1 * n2 / 12.0) * (N + 1.0 - tie_correction);
    } else {
        var = (n1 * n2 / 12.0) * (N + 1.0);
    }

    if (var < 0.0) var = 0.0;

    sd = std::sqrt(var);
}

SCL_FORCE_INLINE double compute_z(
    double U, double mu, double sd, double cc
) {
    if (sd <= 0.0) {
        return 0.0;
    }

    double diff = std::abs(U - mu) - cc;
    if (diff < 0.0) diff = 0.0;

    return diff / sd;
}

} // namespace detail

// =============================================================================
// Scalar P-Value Functions
// =============================================================================

SCL_FORCE_INLINE double p_value_two_sided(
    double U, double n1, double n2, double tie_sum = 0.0, double cc = 0.5
) {
    double mu, sd;
    detail::moments(n1, n2, tie_sum, mu, sd);

    if (sd <= 0.0) {
        return 1.0;
    }

    const double z = detail::compute_z(U, mu, sd, cc);

    return 2.0 * scl::math::normal_sf(z);
}

SCL_FORCE_INLINE double p_value_greater(
    double U, double n1, double n2, double tie_sum = 0.0, double cc = 0.5
) {
    double mu, sd;
    detail::moments(n1, n2, tie_sum, mu, sd);

    if (sd <= 0.0) {
        return (U > mu) ? 0.0 : 1.0;
    }

    double diff = U - mu - cc;
    const double z = diff / sd;

    return scl::math::normal_sf(z);
}

SCL_FORCE_INLINE double p_value_less(
    double U, double n1, double n2, double tie_sum = 0.0, double cc = 0.5
) {
    double mu, sd;
    detail::moments(n1, n2, tie_sum, mu, sd);

    if (sd <= 0.0) {
        return (U < mu) ? 0.0 : 1.0;
    }

    double diff = mu - U - cc;
    const double z = diff / sd;

    return scl::math::normal_sf(z);
}

// =============================================================================
// SIMD Implementations
// =============================================================================

namespace simd {

namespace s = scl::simd;

template <class D, class V>
SCL_FORCE_INLINE void moments(
    D d, V n1, V n2, V tie_sum,
    V& mu, V& sd
) {
    const auto zero = s::Zero(d);
    const auto one = s::Set(d, 1.0);
    const auto twelve = s::Set(d, 12.0);

    const auto N = s::Add(n1, n2);

    mu = s::Mul(s::Set(d, 0.5), s::Mul(n1, n2));

    const auto denom = s::Mul(N, s::Sub(N, one));
    const auto base = s::Div(s::Mul(n1, n2), twelve);

    const auto term_N_plus_1 = s::Add(N, one);

    const auto mask_denom = s::Gt(denom, zero);
    const auto correction = s::Div(tie_sum, denom);

    auto var_with_correction = s::Mul(base, s::Sub(term_N_plus_1, correction));
    auto var_without_correction = s::Mul(base, term_N_plus_1);

    auto var = s::IfThenElse(mask_denom, var_with_correction, var_without_correction);

    var = s::Max(var, zero);

    sd = s::Sqrt(var);
}

template <class D, class V>
SCL_FORCE_INLINE V compute_z(
    D d, V U, V mu, V sd, V cc
) {
    const auto zero = s::Zero(d);

    auto abs_diff = s::Abs(s::Sub(U, mu));
    auto diff = s::Sub(abs_diff, cc);
    diff = s::Max(diff, zero);

    auto mask_sd = s::Gt(sd, zero);
    auto z = s::IfThenElse(mask_sd, s::Div(diff, sd), zero);

    return z;
}

template <class D, class V>
SCL_FORCE_INLINE V p_value_two_sided(
    D d, V U, V n1, V n2, V tie_sum, V cc
) {
    V mu, sd;
    moments(d, n1, n2, tie_sum, mu, sd);

    const auto zero = s::Zero(d);
    const auto one = s::Set(d, 1.0);
    const auto two = s::Set(d, 2.0);

    auto z = compute_z(d, U, mu, sd, cc);

    auto sf = scl::math::simd::normal_sf(d, z);
    auto p = s::Mul(two, sf);

    auto mask_valid = s::Gt(sd, zero);
    return s::IfThenElse(mask_valid, p, one);
}

template <class D, class V>
SCL_FORCE_INLINE V p_value_greater(
    D d, V U, V n1, V n2, V tie_sum, V cc
) {
    V mu, sd;
    moments(d, n1, n2, tie_sum, mu, sd);

    const auto zero = s::Zero(d);
    const auto one = s::Set(d, 1.0);

    auto diff = s::Sub(s::Sub(U, mu), cc);

    auto mask_sd = s::Gt(sd, zero);
    auto z = s::IfThenElse(mask_sd, s::Div(diff, sd), zero);

    auto p = scl::math::simd::normal_sf(d, z);

    auto mask_greater = s::Gt(U, mu);
    auto p_degenerate = s::IfThenElse(mask_greater, zero, one);

    return s::IfThenElse(mask_sd, p, p_degenerate);
}

template <class D, class V>
SCL_FORCE_INLINE V p_value_less(
    D d, V U, V n1, V n2, V tie_sum, V cc
) {
    V mu, sd;
    moments(d, n1, n2, tie_sum, mu, sd);

    const auto zero = s::Zero(d);
    const auto one = s::Set(d, 1.0);

    auto diff = s::Sub(s::Sub(mu, U), cc);

    auto mask_sd = s::Gt(sd, zero);
    auto z = s::IfThenElse(mask_sd, s::Div(diff, sd), zero);

    auto p = scl::math::simd::normal_sf(d, z);

    auto mask_less = s::Lt(U, mu);
    auto p_degenerate = s::IfThenElse(mask_less, zero, one);

    return s::IfThenElse(mask_sd, p, p_degenerate);
}

} // namespace simd

} // namespace scl::math::mwu
