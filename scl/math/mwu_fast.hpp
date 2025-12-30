#pragma once

/// @file scl/math/mwu_fast.hpp
/// @brief Mann-Whitney U Test statistics (fast approximate implementation)
///
/// This header provides:
///   - Fast Mann-Whitney U test p-value computation
///   - Uses approximate erfc for ~3-5x speedup
///   - Returns inverse SD for faster computation
///   - Both scalar and SIMD implementations
///
/// @note Uses fast approximate normal distribution (~1e-7 precision)
/// @note For full precision, see scl/math/mwu.hpp

#include "scl/core/simd.hpp"
#include "scl/math/stats_fast.hpp"

#include <cmath>

namespace scl::math {

// =============================================================================
// SECTION 1: Internal Helpers
// =============================================================================

namespace detail {

/// @brief Compute mean and inverse SD for Mann-Whitney U statistic (fast)
/// @param[in] n1 Sample size of group 1
/// @param[in] n2 Sample size of group 2
/// @param[in] tie_sum Sum of tie correction terms
/// @param[out] mu Mean of U statistic under null hypothesis
/// @param[out] inv_sd Inverse standard deviation (1/sd) for faster computation
/// @note Returns inv_sd instead of sd to avoid division in z-score computation
SCL_FORCE_INLINE
auto mwu_moments_fast(
    double n1, double n2, double tie_sum,
    double& mu, double& inv_sd
) -> void {
    const double N = n1 + n2;
    mu = 0.5 * n1 * n2;

    const double denom = N * (N - 1.0);
    const double base = n1 * n2 / 12.0;

    double var{};
    if (denom > 1e-9) [[likely]] {
        var = base * (N + 1.0 - tie_sum / denom);
    } else [[unlikely]] {
        var = base * (N + 1.0);
    }

    inv_sd = (var <= 1e-15) ? 0.0 : (1.0 / std::sqrt(var));
}

} // namespace detail

// =============================================================================
// SECTION 2: Scalar P-Value Functions
// =============================================================================

/// @brief Fast Mann-Whitney U test p-value (two-sided)
/// @param[in] U U statistic
/// @param[in] n1 Sample size of group 1
/// @param[in] n2 Sample size of group 2
/// @param[in] tie_sum Sum of tie correction terms (default 0)
/// @param[in] cc Continuity correction (default 0.5)
/// @return Approximate two-sided p-value
/// @note Uses fast approximate normal distribution (~1e-7 precision)
/// @note ~3-5x faster than precise version
[[nodiscard]]
SCL_FORCE_INLINE
auto mwu_pvalue_two_sided_fast(
    double U, double n1, double n2, double tie_sum = 0.0, double cc = 0.5
) -> double {
    double mu{}, inv_sd{};
    detail::mwu_moments_fast(n1, n2, tie_sum, mu, inv_sd);

    if (inv_sd == 0.0) [[unlikely]] return 1.0;

    const double z = (std::abs(U - mu) - cc) * inv_sd;

    return 2.0 * scl::math::normal_sf_fast(z);
}

/// @brief Fast Mann-Whitney U test p-value (greater alternative)
/// @param[in] U U statistic
/// @param[in] n1 Sample size of group 1
/// @param[in] n2 Sample size of group 2
/// @param[in] tie_sum Sum of tie correction terms (default 0)
/// @param[in] cc Continuity correction (default 0.5)
/// @return Approximate one-sided p-value for H1: U > mu
/// @note Uses fast approximate normal distribution (~1e-7 precision)
[[nodiscard]]
SCL_FORCE_INLINE
auto mwu_pvalue_greater_fast(
    double U, double n1, double n2, double tie_sum = 0.0, double cc = 0.5
) -> double {
    double mu{}, inv_sd{};
    detail::mwu_moments_fast(n1, n2, tie_sum, mu, inv_sd);

    if (inv_sd == 0.0) [[unlikely]] return (U > mu) ? 0.0 : 1.0;

    const double z = (U - mu - cc) * inv_sd;
    return scl::math::normal_sf_fast(z);
}

/// @brief Fast Mann-Whitney U test p-value (less alternative)
/// @param[in] U U statistic
/// @param[in] n1 Sample size of group 1
/// @param[in] n2 Sample size of group 2
/// @param[in] tie_sum Sum of tie correction terms (default 0)
/// @param[in] cc Continuity correction (default 0.5)
/// @return Approximate one-sided p-value for H1: U < mu
/// @note Uses fast approximate normal distribution (~1e-7 precision)
[[nodiscard]]
SCL_FORCE_INLINE
auto mwu_pvalue_less_fast(
    double U, double n1, double n2, double tie_sum = 0.0, double cc = 0.5
) -> double {
    double mu{}, inv_sd{};
    detail::mwu_moments_fast(n1, n2, tie_sum, mu, inv_sd);

    if (inv_sd == 0.0) [[unlikely]] return (U < mu) ? 0.0 : 1.0;

    const double z = (mu - U - cc) * inv_sd;
    return scl::math::normal_sf_fast(z);
}

// =============================================================================
// SECTION 3: SIMD Implementations
// =============================================================================

namespace simd {

namespace s = scl::simd;

/// @brief SIMD computation of mean and inverse SD (fast)
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] n1 Sample sizes of group 1 (vector)
/// @param[in] n2 Sample sizes of group 2 (vector)
/// @param[in] tie_sum Tie correction sums (vector)
/// @param[out] mu Means (vector)
/// @param[out] inv_sd Inverse standard deviations (vector)
template<class D, class V>
SCL_FORCE_INLINE
auto mwu_moments_fast(
    D d, V n1, V n2, V tie_sum,
    V& mu, V& inv_sd
) -> void {
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

/// @brief SIMD fast Mann-Whitney U test p-value (two-sided)
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] U U statistics (vector)
/// @param[in] n1 Sample sizes of group 1 (vector)
/// @param[in] n2 Sample sizes of group 2 (vector)
/// @param[in] tie_sum Tie correction sums (vector)
/// @param[in] cc Continuity corrections (vector)
/// @return Approximate two-sided p-values (vector)
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto mwu_pvalue_two_sided_fast(
    D d, V U, V n1, V n2, V tie_sum, V cc
) -> V {
    V mu{}, inv_sd{};
    mwu_moments_fast(d, n1, n2, tie_sum, mu, inv_sd);

    auto abs_diff = s::Abs(s::Sub(U, mu));
    auto z = s::Mul(s::Sub(abs_diff, cc), inv_sd);

    auto sf = scl::math::simd::normal_sf_fast(d, z);
    auto p = s::Mul(s::Set(d, 2.0), sf);

    return s::IfThenElse(s::Eq(inv_sd, s::Zero(d)), s::Set(d, 1.0), p);
}

/// @brief SIMD fast Mann-Whitney U test p-value (greater alternative)
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] U U statistics (vector)
/// @param[in] n1 Sample sizes of group 1 (vector)
/// @param[in] n2 Sample sizes of group 2 (vector)
/// @param[in] tie_sum Tie correction sums (vector)
/// @param[in] cc Continuity corrections (vector)
/// @return Approximate one-sided p-values (vector)
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto mwu_pvalue_greater_fast(
    D d, V U, V n1, V n2, V tie_sum, V cc
) -> V {
    V mu{}, inv_sd{};
    mwu_moments_fast(d, n1, n2, tie_sum, mu, inv_sd);

    const auto zero = s::Zero(d);
    const auto one = s::Set(d, 1.0);

    auto z = s::Mul(s::Sub(s::Sub(U, mu), cc), inv_sd);
    auto p = scl::math::simd::normal_sf_fast(d, z);

    auto mask_greater = s::Gt(U, mu);
    auto p_degenerate = s::IfThenElse(mask_greater, zero, one);

    return s::IfThenElse(s::Eq(inv_sd, zero), p_degenerate, p);
}

/// @brief SIMD fast Mann-Whitney U test p-value (less alternative)
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] U U statistics (vector)
/// @param[in] n1 Sample sizes of group 1 (vector)
/// @param[in] n2 Sample sizes of group 2 (vector)
/// @param[in] tie_sum Tie correction sums (vector)
/// @param[in] cc Continuity corrections (vector)
/// @return Approximate one-sided p-values (vector)
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto mwu_pvalue_less_fast(
    D d, V U, V n1, V n2, V tie_sum, V cc
) -> V {
    V mu{}, inv_sd{};
    mwu_moments_fast(d, n1, n2, tie_sum, mu, inv_sd);

    const auto zero = s::Zero(d);
    const auto one = s::Set(d, 1.0);

    auto z = s::Mul(s::Sub(s::Sub(mu, U), cc), inv_sd);
    auto p = scl::math::simd::normal_sf_fast(d, z);

    auto mask_less = s::Lt(U, mu);
    auto p_degenerate = s::IfThenElse(mask_less, zero, one);

    return s::IfThenElse(s::Eq(inv_sd, zero), p_degenerate, p);
}

} // namespace simd

} // namespace scl::math

