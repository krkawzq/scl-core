#pragma once

/// @file scl/math/mwu.hpp
/// @brief Mann-Whitney U Test statistics (precise implementation)
///
/// This header provides:
///   - Mann-Whitney U test p-value computation
///   - Normal approximation with tie correction
///   - Two-sided, greater, and less alternative hypotheses
///   - Both scalar and SIMD implementations
///
/// @note Uses precise normal distribution functions
/// @note For fast approximate version, see scl/math/mwu_fast.hpp

#include "scl/core/simd.hpp"
#include "scl/math/stats.hpp"

#include <cmath>

namespace scl::math {

// =============================================================================
// SECTION 1: Internal Helpers
// =============================================================================

namespace detail {

/// @brief Compute mean and standard deviation for Mann-Whitney U statistic
/// @param[in] n1 Sample size of group 1
/// @param[in] n2 Sample size of group 2
/// @param[in] tie_sum Sum of tie correction terms
/// @param[out] mu Mean of U statistic under null hypothesis
/// @param[out] sd Standard deviation of U statistic
/// @note mu = n1 * n2 / 2
/// @note var = (n1 * n2 / 12) * (N + 1 - tie_correction)
SCL_FORCE_INLINE
auto mwu_moments(
    double n1, double n2, double tie_sum,
    double& mu, double& sd
) -> void {
    const double N = n1 + n2;
    mu = 0.5 * n1 * n2;

    const double denom = N * (N - 1.0);

    double var{};
    if (denom > 0.0) [[likely]] {
        const double tie_correction = tie_sum / denom;
        var = (n1 * n2 / 12.0) * (N + 1.0 - tie_correction);
    } else [[unlikely]] {
        var = (n1 * n2 / 12.0) * (N + 1.0);
    }

    if (var < 0.0) [[unlikely]] var = 0.0;

    sd = std::sqrt(var);
}

/// @brief Compute z-score with continuity correction
/// @param[in] U U statistic
/// @param[in] mu Mean of U under null
/// @param[in] sd Standard deviation of U
/// @param[in] cc Continuity correction (default 0.5)
/// @return Z-score
SCL_FORCE_INLINE
auto compute_z(
    double U, double mu, double sd, double cc
) -> double {
    if (sd <= 0.0) [[unlikely]] {
        return 0.0;
    }

    double diff = std::abs(U - mu) - cc;
    if (diff < 0.0) [[unlikely]] diff = 0.0;

    return diff / sd;
}

} // namespace detail

// =============================================================================
// SECTION 2: Scalar P-Value Functions
// =============================================================================

/// @brief Mann-Whitney U test p-value (two-sided)
/// @param[in] U U statistic
/// @param[in] n1 Sample size of group 1
/// @param[in] n2 Sample size of group 2
/// @param[in] tie_sum Sum of tie correction terms (default 0)
/// @param[in] cc Continuity correction (default 0.5)
/// @return Two-sided p-value
/// @note Uses normal approximation with tie correction
/// @note p = 2 * P(Z > |z|) where z is the standardized U statistic
[[nodiscard]]
SCL_FORCE_INLINE
auto mwu_pvalue_two_sided(
    double U, double n1, double n2, double tie_sum = 0.0, double cc = 0.5
) -> double {
    double mu{}, sd{};
    detail::mwu_moments(n1, n2, tie_sum, mu, sd);

    if (sd <= 0.0) [[unlikely]] {
        return 1.0;
    }

    const double z = detail::compute_z(U, mu, sd, cc);

    return 2.0 * scl::math::normal_sf(z);
}

/// @brief Mann-Whitney U test p-value (greater alternative)
/// @param[in] U U statistic
/// @param[in] n1 Sample size of group 1
/// @param[in] n2 Sample size of group 2
/// @param[in] tie_sum Sum of tie correction terms (default 0)
/// @param[in] cc Continuity correction (default 0.5)
/// @return One-sided p-value for H1: U > mu
/// @note p = P(Z > z) where z = (U - mu - cc) / sd
[[nodiscard]]
SCL_FORCE_INLINE
auto mwu_pvalue_greater(
    double U, double n1, double n2, double tie_sum = 0.0, double cc = 0.5
) -> double {
    double mu{}, sd{};
    detail::mwu_moments(n1, n2, tie_sum, mu, sd);

    if (sd <= 0.0) [[unlikely]] {
        return (U > mu) ? 0.0 : 1.0;
    }

    double diff = U - mu - cc;
    const double z = diff / sd;

    return scl::math::normal_sf(z);
}

/// @brief Mann-Whitney U test p-value (less alternative)
/// @param[in] U U statistic
/// @param[in] n1 Sample size of group 1
/// @param[in] n2 Sample size of group 2
/// @param[in] tie_sum Sum of tie correction terms (default 0)
/// @param[in] cc Continuity correction (default 0.5)
/// @return One-sided p-value for H1: U < mu
/// @note p = P(Z > z) where z = (mu - U - cc) / sd
[[nodiscard]]
SCL_FORCE_INLINE
auto mwu_pvalue_less(
    double U, double n1, double n2, double tie_sum = 0.0, double cc = 0.5
) -> double {
    double mu{}, sd{};
    detail::mwu_moments(n1, n2, tie_sum, mu, sd);

    if (sd <= 0.0) [[unlikely]] {
        return (U < mu) ? 0.0 : 1.0;
    }

    double diff = mu - U - cc;
    const double z = diff / sd;

    return scl::math::normal_sf(z);
}

// =============================================================================
// SECTION 3: SIMD Implementations
// =============================================================================

namespace simd {

namespace s = scl::simd;

/// @brief SIMD computation of mean and SD for Mann-Whitney U
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] n1 Sample sizes of group 1 (vector)
/// @param[in] n2 Sample sizes of group 2 (vector)
/// @param[in] tie_sum Tie correction sums (vector)
/// @param[out] mu Means (vector)
/// @param[out] sd Standard deviations (vector)
template<class D, class V>
SCL_FORCE_INLINE
auto mwu_moments(
    D d, V n1, V n2, V tie_sum,
    V& mu, V& sd
) -> void {
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

/// @brief SIMD computation of z-score with continuity correction
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] U U statistics (vector)
/// @param[in] mu Means (vector)
/// @param[in] sd Standard deviations (vector)
/// @param[in] cc Continuity corrections (vector)
/// @return Z-scores (vector)
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto compute_z(
    D d, V U, V mu, V sd, V cc
) -> V {
    const auto zero = s::Zero(d);

    auto abs_diff = s::Abs(s::Sub(U, mu));
    auto diff = s::Sub(abs_diff, cc);
    diff = s::Max(diff, zero);

    auto mask_sd = s::Gt(sd, zero);
    auto z = s::IfThenElse(mask_sd, s::Div(diff, sd), zero);

    return z;
}

/// @brief SIMD Mann-Whitney U test p-value (two-sided)
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] U U statistics (vector)
/// @param[in] n1 Sample sizes of group 1 (vector)
/// @param[in] n2 Sample sizes of group 2 (vector)
/// @param[in] tie_sum Tie correction sums (vector)
/// @param[in] cc Continuity corrections (vector)
/// @return Two-sided p-values (vector)
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto mwu_pvalue_two_sided(
    D d, V U, V n1, V n2, V tie_sum, V cc
) -> V {
    V mu{}, sd{};
    mwu_moments(d, n1, n2, tie_sum, mu, sd);

    const auto zero = s::Zero(d);
    const auto one = s::Set(d, 1.0);
    const auto two = s::Set(d, 2.0);

    auto z = compute_z(d, U, mu, sd, cc);

    auto sf = scl::math::simd::normal_sf(d, z);
    auto p = s::Mul(two, sf);

    auto mask_valid = s::Gt(sd, zero);
    return s::IfThenElse(mask_valid, p, one);
}

/// @brief SIMD Mann-Whitney U test p-value (greater alternative)
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] U U statistics (vector)
/// @param[in] n1 Sample sizes of group 1 (vector)
/// @param[in] n2 Sample sizes of group 2 (vector)
/// @param[in] tie_sum Tie correction sums (vector)
/// @param[in] cc Continuity corrections (vector)
/// @return One-sided p-values (vector)
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto mwu_pvalue_greater(
    D d, V U, V n1, V n2, V tie_sum, V cc
) -> V {
    V mu{}, sd{};
    mwu_moments(d, n1, n2, tie_sum, mu, sd);

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

/// @brief SIMD Mann-Whitney U test p-value (less alternative)
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] U U statistics (vector)
/// @param[in] n1 Sample sizes of group 1 (vector)
/// @param[in] n2 Sample sizes of group 2 (vector)
/// @param[in] tie_sum Tie correction sums (vector)
/// @param[in] cc Continuity corrections (vector)
/// @return One-sided p-values (vector)
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto mwu_pvalue_less(
    D d, V U, V n1, V n2, V tie_sum, V cc
) -> V {
    V mu{}, sd{};
    mwu_moments(d, n1, n2, tie_sum, mu, sd);

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

} // namespace scl::math

