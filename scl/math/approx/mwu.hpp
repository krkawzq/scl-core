#pragma once

#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/math/approx/stats.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/math/approx/mwu.hpp
// BRIEF: Approximate Mann-Whitney U Test Statistics
// =============================================================================

// Uses approximate erfc approximation and returns inverse SD for faster computation
// For full precision, see scl/math/mwu.hpp

namespace scl::math::approx::mwu {

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Returns inv_sd instead of sd for faster computation
SCL_FORCE_INLINE void moments(
    double n1, double n2, double tie_sum,
    double& mu, double& inv_sd
) {
    const double N = n1 + n2;
    mu = 0.5 * n1 * n2;

    const double denom = N * (N - 1.0);
    const double base = n1 * n2 / 12.0;

    double var{};
    if (denom > 1e-9) {
        var = base * (N + 1.0 - tie_sum / denom);
    } else {
        var = base * (N + 1.0);
    }

    inv_sd = (var <= 1e-15) ? 0.0 : (1.0 / std::sqrt(var));
}

} // namespace detail

// =============================================================================
// Scalar P-Value Functions
// =============================================================================

SCL_FORCE_INLINE double p_value_two_sided(
    double U, double n1, double n2, double tie_sum, double cc = 0.5
) {
    double mu{}, inv_sd{};
    detail::moments(n1, n2, tie_sum, mu, inv_sd);

    if (inv_sd == 0.0) return 1.0;

    const double z = (std::abs(U - mu) - cc) * inv_sd;

    return 2.0 * scl::math::approx::normal_sf(z);
}

SCL_FORCE_INLINE double p_value_greater(
    double U, double n1, double n2, double tie_sum, double cc = 0.5
) {
    double mu{}, inv_sd{};
    detail::moments(n1, n2, tie_sum, mu, inv_sd);

    if (inv_sd == 0.0) return (U > mu) ? 0.0 : 1.0;

    const double z = (U - mu - cc) * inv_sd;
    return scl::math::approx::normal_sf(z);
}

SCL_FORCE_INLINE double p_value_less(
    double U, double n1, double n2, double tie_sum, double cc = 0.5
) {
    double mu{}, inv_sd{};
    detail::moments(n1, n2, tie_sum, mu, inv_sd);

    if (inv_sd == 0.0) return (U < mu) ? 0.0 : 1.0;

    const double z = (mu - U - cc) * inv_sd;
    return scl::math::approx::normal_sf(z);
}

// =============================================================================
// SIMD Implementations
// =============================================================================

namespace simd {

namespace s = scl::simd;

template <class D, class V>
SCL_FORCE_INLINE void moments(
    D d, V n1, V n2, V tie_sum,
    V& mu, V& inv_sd
) {
    const auto zero = s::Zero(d);
    const auto one = s::Set(d, 1.0);
    const auto twelve = s::Set(d, 12.0);

    const auto N = s::Add(n1, n2);

    mu = s::Mul(s::Set(d, 0.5), s::Mul(n1, n2));

    const auto denom = s::Mul(N, s::Sub(N, one));
    const auto base = s::Div(s::Mul(n1, n2), twelve);

    const auto term_N_plus_1 = s::Add(N, one);
    const auto correction = s::Div(tie_sum, denom);

    const auto mask_denom = s::Gt(denom, zero);

    auto var_normal = s::Mul(base, s::Sub(term_N_plus_1, correction));
    auto var_fallback = s::Mul(base, term_N_plus_1);

    auto var = s::IfThenElse(mask_denom, var_normal, var_fallback);

    const auto mask_var = s::Gt(var, zero);
    inv_sd = s::IfThenElse(mask_var,
        s::Div(one, s::Sqrt(var)),
        zero
    );
}

template <class D, class V>
SCL_FORCE_INLINE V p_value_two_sided(
    D d, V U, V n1, V n2, V tie_sum, V cc
) {
    V mu{}, inv_sd{};
    moments(d, n1, n2, tie_sum, mu, inv_sd);

    auto abs_diff = s::Abs(s::Sub(U, mu));
    auto z = s::Mul(s::Sub(abs_diff, cc), inv_sd);

    auto sf = scl::math::approx::simd::normal_sf(d, z);
    auto p = s::Mul(s::Set(d, 2.0), sf);

    return s::IfThenElse(s::Eq(inv_sd, s::Zero(d)), s::Set(d, 1.0), p);
}

template <class D, class V>
SCL_FORCE_INLINE V p_value_greater(
    D d, V U, V n1, V n2, V tie_sum, V cc
) {
    V mu{}, inv_sd{};
    moments(d, n1, n2, tie_sum, mu, inv_sd);

    const auto zero = s::Zero(d);
    const auto one = s::Set(d, 1.0);

    auto z = s::Mul(s::Sub(s::Sub(U, mu), cc), inv_sd);
    auto p = scl::math::approx::simd::normal_sf(d, z);

    auto mask_greater = s::Gt(U, mu);
    auto p_degenerate = s::IfThenElse(mask_greater, zero, one);

    return s::IfThenElse(s::Eq(inv_sd, zero), p_degenerate, p);
}

template <class D, class V>
SCL_FORCE_INLINE V p_value_less(
    D d, V U, V n1, V n2, V tie_sum, V cc
) {
    V mu{}, inv_sd{};
    moments(d, n1, n2, tie_sum, mu, inv_sd);

    const auto zero = s::Zero(d);
    const auto one = s::Set(d, 1.0);

    auto z = s::Mul(s::Sub(s::Sub(mu, U), cc), inv_sd);
    auto p = scl::math::approx::simd::normal_sf(d, z);

    auto mask_less = s::Lt(U, mu);
    auto p_degenerate = s::IfThenElse(mask_less, zero, one);

    return s::IfThenElse(s::Eq(inv_sd, zero), p_degenerate, p);
}

} // namespace simd

} // namespace scl::math::approx::mwu
